"""Gradient-conflict diagnostics for SPNI DFL training.

This module reruns the SPNI DFL training update path while recording
per-scenario gradient alignment metrics. It reuses the existing graph and data
stages so cached datasets and interdiction tensors can still be reused, but it
skips the evaluation-heavy regret logging because the diagnostics depend only
on the training batches and the validation-loss learning-rate schedule.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyepo
import torch
from torch import nn

from dflintdpy.data.config import HP
from dflintdpy.predictors.hybrid_spop_loss import HybridSPOPLoss
from dflintdpy.simulation.spni.build import build_problem_bundle
from dflintdpy.simulation.spni.config import (
    CachePolicy,
    SPNIRunConfig,
    build_run_config,
    derive_seed_bundle,
    derive_seed_sweep,
    describe_run,
)
from dflintdpy.simulation.spni.data import assemble_dataset_bundle
from dflintdpy.utils.dfl_trainer import DFLTrainer

_METHOD_SPECS = {
    "dfl": {
        "train_loader_attr": "train_loader_baseline",
        "val_loader_attr": "val_loader_baseline",
        "dfl_variant": "a-dfl",
    },
    "rdfl": {
        "train_loader_attr": "train_loader_random",
        "val_loader_attr": "val_loader_random",
        "dfl_variant": "a-dfl",
    },
    "adfl": {
        "train_loader_attr": "train_loader_adversarial",
        "val_loader_attr": "val_loader_adversarial",
        "dfl_variant": "a-dfl",
    },
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read one config-style value from a mapping or legacy config object."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _set_cfg_value(cfg: Any, key: str, value: Any) -> None:
    """Write one override onto a mutable config-like object."""
    if isinstance(cfg, dict):
        cfg[key] = value
        return
    setter = getattr(cfg, "set", None)
    if callable(setter):
        setter(key, value)
        return
    setattr(cfg, key, value)


def _parse_override(raw_override: str) -> tuple[str, Any]:
    """Parse one ``key=value`` CLI override into a typed pair."""
    if "=" not in raw_override:
        raise ValueError(
            f"Override '{raw_override}' must use the form key=value."
        )
    key, raw_value = raw_override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Override keys must be non-empty.")
    raw_value = raw_value.strip()
    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        value = raw_value
    return key, value


def _normalize_scenarios(scenarios: str | Sequence[int]) -> list[int]:
    """Return validated integer scenario counts in caller-specified order."""
    if isinstance(scenarios, str):
        resolved = [
            int(part.strip())
            for part in scenarios.split(",")
            if part.strip()
        ]
    else:
        resolved = [int(scenario) for scenario in scenarios]
    if not resolved:
        raise ValueError("At least one scenario count is required.")
    return resolved


def _resolve_methods(method: str) -> list[str]:
    """Expand one CLI method selector into the concrete family names to run."""
    if method == "all":
        return ["rdfl", "adfl"]
    if method not in _METHOD_SPECS:
        supported = ", ".join(sorted(_METHOD_SPECS))
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: {supported}, all."
        )
    return [method]


def _set_seed(random_seed: int) -> None:
    """Match the predictor-initialization seeding used by the legacy helpers."""
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def _build_predictor_model(run_cfg: SPNIRunConfig, output_size: int) -> nn.Module:
    """Construct the predictor architecture used by SPNI DFL training."""
    input_size = int(run_cfg.num_features)
    if run_cfg.pred_model is None or run_cfg.pred_model == "nn":
        hidden_size = 64
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )
    return nn.Linear(input_size, output_size)


def _build_loss_fn(run_cfg: SPNIRunConfig, opt_model: Any) -> nn.Module:
    """Construct the DFL loss used by the current SPNI training path."""
    lam = float(_cfg_get(run_cfg.base_cfg, "lam", 0.0) or 0.0)
    if lam == 0:
        return pyepo.func.SPOPlus(opt_model, processes=1)
    anchor = _cfg_get(run_cfg.base_cfg, "anchor", "mse")
    return HybridSPOPLoss(opt_model, lam=lam, anchor=anchor)


def _fit_with_validation_schedule(
    trainer: DFLTrainer,
    train_loader: Any,
    val_loader: Any,
    *,
    epochs: int,
) -> None:
    """Run the DFL update path with validation-loss LR scheduling only."""
    trainer._prepare_loader_for_loss(train_loader)
    if val_loader is not None:
        trainer._prepare_loader_for_loss(val_loader)

    best_val_loss: float | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    epochs_since_best_val = 0

    if val_loader is not None:
        best_val_loss = trainer._evaluate_loss(val_loader)
        best_model_state = trainer._snapshot_model_state()

    for epoch in range(1, int(epochs) + 1):
        trainer._before_epoch(epoch - 1)
        trainer.train_epoch(train_loader)

        if val_loader is None:
            continue

        val_loss = trainer._evaluate_loss(val_loader)
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = trainer._snapshot_model_state()
            epochs_since_best_val = 0
        else:
            epochs_since_best_val += 1

        if (
            best_val_loss is not None
            and epochs_since_best_val >= trainer.LR_REDUCTION_PATIENCE
            and val_loss - best_val_loss
            > trainer.LOSS_INCREASE_THRESHOLD * best_val_loss
        ):
            trainer.optimizer.param_groups[0]["lr"] *= trainer.LR_REDUCTION_FACTOR
            epochs_since_best_val = 0

    if best_model_state is not None:
        trainer.pred_model.load_state_dict(best_model_state)


def _merge_run_metadata(
    run_cfg: SPNIRunConfig,
    *,
    method: str,
    log_every_n_steps: int,
    max_batches_per_epoch: int | None,
) -> dict[str, Any]:
    """Return the run-level metadata repeated across diagnostic rows."""
    return {
        "method": method,
        "grid_rows": int(run_cfg.grid_size[0]),
        "grid_cols": int(run_cfg.grid_size[1]),
        "configured_scenarios": int(run_cfg.num_scenarios),
        "sweep_seed": int(run_cfg.seed),
        "random_seed": int(run_cfg.random_seed),
        "intd_seed": int(run_cfg.intd_seed),
        "loader_seed": int(run_cfg.loader_seed),
        "dfl_epochs": int(run_cfg.dfl_epochs),
        "dfl_lr": float(run_cfg.dfl_lr),
        "batch_size_cfg": int(run_cfg.batch_size),
        "log_every_n_steps": int(log_every_n_steps),
        "max_batches_per_epoch": (
            int(max_batches_per_epoch)
            if max_batches_per_epoch is not None
            else None
        ),
    }


def _aggregate_epoch_rows(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average step-level diagnostic rows into one row per epoch."""
    if not step_rows:
        return []

    epoch_rows = []
    for epoch in sorted({int(row["epoch"]) for row in step_rows}):
        rows = [row for row in step_rows if int(row["epoch"]) == epoch]
        cosine_values = [
            float(row["mean_pairwise_cosine"])
            for row in rows
            if not math.isnan(float(row["mean_pairwise_cosine"]))
        ]
        cancellation_values = [
            float(row["cancellation_ratio"])
            for row in rows
            if not math.isnan(float(row["cancellation_ratio"]))
        ]
        summed_norms = [float(row["summed_gradient_norm"]) for row in rows]
        sum_individual_norms = [float(row["sum_individual_norms"]) for row in rows]
        epoch_row = dict(rows[0])
        epoch_row.update(
            {
                "epoch": int(epoch),
                "batches_measured": int(len(rows)),
                "global_step": int(rows[-1]["global_step"]),
                "batch_index": None,
                "mean_pairwise_cosine": (
                    float(np.mean(cosine_values))
                    if cosine_values
                    else float("nan")
                ),
                "cancellation_ratio": (
                    float(np.mean(cancellation_values))
                    if cancellation_values
                    else float("nan")
                ),
                "summed_gradient_norm": float(np.mean(summed_norms)),
                "sum_individual_norms": float(np.mean(sum_individual_norms)),
                "pair_count": int(sum(int(row["pair_count"]) for row in rows)),
                "batch_size": float(np.mean([float(row["batch_size"]) for row in rows])),
            }
        )
        epoch_rows.append(epoch_row)
    return epoch_rows


def _aggregate_phase_rows(epoch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize early, late, and full training windows from epoch rows."""
    if not epoch_rows:
        return []

    phase_width = max(1, len(epoch_rows) // 4)
    phase_buckets = {
        "full": epoch_rows,
        "early": epoch_rows[:phase_width],
        "late": epoch_rows[-phase_width:],
    }
    phase_rows = []
    for phase_name, rows in phase_buckets.items():
        cosine_values = [
            float(row["mean_pairwise_cosine"])
            for row in rows
            if not math.isnan(float(row["mean_pairwise_cosine"]))
        ]
        cancellation_values = [
            float(row["cancellation_ratio"])
            for row in rows
            if not math.isnan(float(row["cancellation_ratio"]))
        ]
        phase_row = dict(rows[0])
        phase_row.update(
            {
                "phase": phase_name,
                "epoch": None,
                "epoch_start": int(rows[0]["epoch"]),
                "epoch_end": int(rows[-1]["epoch"]),
                "batches_measured": int(
                    sum(int(row["batches_measured"]) for row in rows)
                ),
                "global_step": int(rows[-1]["global_step"]),
                "batch_index": None,
                "mean_pairwise_cosine": (
                    float(np.mean(cosine_values))
                    if cosine_values
                    else float("nan")
                ),
                "cancellation_ratio": (
                    float(np.mean(cancellation_values))
                    if cancellation_values
                    else float("nan")
                ),
                "summed_gradient_norm": float(
                    np.mean([float(row["summed_gradient_norm"]) for row in rows])
                ),
                "sum_individual_norms": float(
                    np.mean([float(row["sum_individual_norms"]) for row in rows])
                ),
                "pair_count": int(sum(int(row["pair_count"]) for row in rows)),
            }
        )
        phase_rows.append(phase_row)
    return phase_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist one row collection as CSV, preserving stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: float | int | None) -> str:
    """Render one scalar metric for compact console output."""
    if value is None:
        return "-"
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _summarize_phase_rows_for_console(
    phase_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate phase rows into console-friendly summaries and gaps."""
    if not phase_rows:
        return [], []

    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in phase_rows:
        key = (
            int(row["configured_scenarios"]),
            str(row["phase"]),
            str(row["method"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (scenario, phase, method), rows in sorted(grouped.items()):
        cosine_values = [
            float(row["mean_pairwise_cosine"])
            for row in rows
            if not math.isnan(float(row["mean_pairwise_cosine"]))
        ]
        cancellation_values = [
            float(row["cancellation_ratio"])
            for row in rows
            if not math.isnan(float(row["cancellation_ratio"]))
        ]
        summary_rows.append(
            {
                "configured_scenarios": scenario,
                "phase": phase,
                "method": method,
                "runs": int(len(rows)),
                "effective_scenarios": int(
                    round(
                        np.mean(
                            [float(row["effective_scenarios"]) for row in rows]
                        )
                    )
                ),
                "mean_pairwise_cosine": (
                    float(np.mean(cosine_values))
                    if cosine_values
                    else float("nan")
                ),
                "cancellation_ratio": (
                    float(np.mean(cancellation_values))
                    if cancellation_values
                    else float("nan")
                ),
            }
        )

    by_scenario_phase_method = {
        (
            int(row["configured_scenarios"]),
            str(row["phase"]),
            str(row["method"]),
        ): row
        for row in summary_rows
    }
    gap_rows: list[dict[str, Any]] = []
    for scenario in sorted({int(row["configured_scenarios"]) for row in summary_rows}):
        for phase in sorted({str(row["phase"]) for row in summary_rows}):
            rdfl_row = by_scenario_phase_method.get((scenario, phase, "rdfl"))
            adfl_row = by_scenario_phase_method.get((scenario, phase, "adfl"))
            if rdfl_row is None or adfl_row is None:
                continue
            gap_rows.append(
                {
                    "configured_scenarios": scenario,
                    "phase": phase,
                    "cosine_gap_adfl_minus_rdfl": (
                        float(adfl_row["mean_pairwise_cosine"])
                        - float(rdfl_row["mean_pairwise_cosine"])
                    ),
                    "cancellation_gap_adfl_minus_rdfl": (
                        float(adfl_row["cancellation_ratio"])
                        - float(rdfl_row["cancellation_ratio"])
                    ),
                }
            )
    return summary_rows, gap_rows


def _print_console_summary(result: dict[str, Any]) -> None:
    """Print a compact comparison summary for the completed diagnostics run."""
    summary_rows, gap_rows = _summarize_phase_rows_for_console(
        result["phase_rows"]
    )
    if not summary_rows:
        print("No gradient diagnostic rows were collected.")
        return

    print("\nGradient conflict summary")
    print(
        "scenario phase method runs effK mean_cosine cancellation_ratio"
    )
    for row in summary_rows:
        print(
            f"{int(row['configured_scenarios']):>8} "
            f"{str(row['phase']):>5} "
            f"{str(row['method']):>6} "
            f"{int(row['runs']):>4} "
            f"{int(row['effective_scenarios']):>4} "
            f"{_format_metric(row['mean_pairwise_cosine']):>11} "
            f"{_format_metric(row['cancellation_ratio']):>18}"
        )

    if gap_rows:
        print("\nA-DFL vs R-DFL gaps (A-DFL - R-DFL)")
        print("scenario phase cosine_gap cancellation_gap")
        for row in gap_rows:
            print(
                f"{int(row['configured_scenarios']):>8} "
                f"{str(row['phase']):>5} "
                f"{_format_metric(row['cosine_gap_adfl_minus_rdfl']):>10} "
                f"{_format_metric(row['cancellation_gap_adfl_minus_rdfl']):>16}"
            )


def _run_single_diagnostic(
    run_cfg: SPNIRunConfig,
    *,
    method: str,
    log_every_n_steps: int,
    max_batches_per_epoch: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one method/scenario/seed diagnostic session and return its rows."""
    spec = _METHOD_SPECS[method]
    graph_bundle = build_problem_bundle(run_cfg)
    dataset_bundle = assemble_dataset_bundle(run_cfg, graph_bundle)

    output_size = int(getattr(graph_bundle.graph, "num_cost"))
    _set_seed(int(run_cfg.random_seed))
    predictor = _build_predictor_model(run_cfg, output_size)
    loss_fn = _build_loss_fn(run_cfg, graph_bundle.opt_model)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=run_cfg.dfl_lr)

    step_rows: list[dict[str, Any]] = []

    def _record(row: dict[str, Any]) -> None:
        step_rows.append(row.copy())

    trainer = DFLTrainer(
        pred_model=predictor,
        opt_model=graph_bundle.opt_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dfl_variant=spec["dfl_variant"],
        diagnostics_callback=_record,
        diagnostics_log_every_n_steps=log_every_n_steps,
        diagnostics_max_batches_per_epoch=max_batches_per_epoch,
    )
    _fit_with_validation_schedule(
        trainer,
        getattr(dataset_bundle, spec["train_loader_attr"]),
        getattr(dataset_bundle, spec["val_loader_attr"]),
        epochs=run_cfg.dfl_epochs,
    )

    metadata = _merge_run_metadata(
        run_cfg,
        method=method,
        log_every_n_steps=log_every_n_steps,
        max_batches_per_epoch=max_batches_per_epoch,
    )
    for row in step_rows:
        row.update(metadata)
    epoch_rows = _aggregate_epoch_rows(step_rows)
    phase_rows = _aggregate_phase_rows(epoch_rows)
    return step_rows, epoch_rows, phase_rows


def run_gradient_conflict_diagnostics(
    *,
    cfg: Any | None = None,
    method: str = "all",
    scenarios: str | Sequence[int] = (1, 2, 3, 5, 8),
    num_seeds: int = 1,
    output_dir: str | Path = "results/gradient_diagnostics",
    log_every_n_steps: int = 1,
    max_batches_per_epoch: int | None = None,
    cache_policy: CachePolicy | None = None,
    **cfg_overrides,
) -> dict[str, Any]:
    """Run one diagnostic sweep and persist its outputs."""
    base_cfg = HP() if cfg is None else cfg
    for key, value in cfg_overrides.items():
        _set_cfg_value(base_cfg, key, value)

    resolved_cache_policy = cache_policy or CachePolicy()
    base_run_cfg = build_run_config(base_cfg, cache_policy=resolved_cache_policy)
    resolved_scenarios = _normalize_scenarios(scenarios)
    resolved_methods = _resolve_methods(method)
    if int(num_seeds) == 1:
        seed_bundles = [derive_seed_bundle(base_run_cfg)]
    else:
        seed_bundles = derive_seed_sweep(base_run_cfg, num_seeds=num_seeds)

    all_step_rows: list[dict[str, Any]] = []
    all_epoch_rows: list[dict[str, Any]] = []
    all_phase_rows: list[dict[str, Any]] = []

    for scenario_count in resolved_scenarios:
        scenario_run_cfg = replace(base_run_cfg, num_scenarios=int(scenario_count))
        for seed_bundle in seed_bundles:
            seeded_run_cfg = replace(
                scenario_run_cfg,
                seed=int(seed_bundle.sweep_seed),
                random_seed=int(seed_bundle.random_seed),
                intd_seed=int(seed_bundle.intd_seed),
                loader_seed=int(seed_bundle.loader_seed),
            )
            for resolved_method in resolved_methods:
                step_rows, epoch_rows, phase_rows = _run_single_diagnostic(
                    seeded_run_cfg,
                    method=resolved_method,
                    log_every_n_steps=log_every_n_steps,
                    max_batches_per_epoch=max_batches_per_epoch,
                )
                all_step_rows.extend(step_rows)
                all_epoch_rows.extend(epoch_rows)
                all_phase_rows.extend(phase_rows)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    step_csv = output_path / "gradient_conflict_steps.csv"
    epoch_csv = output_path / "gradient_conflict_epochs.csv"
    phase_csv = output_path / "gradient_conflict_phases.csv"
    metadata_json = output_path / "gradient_conflict_metadata.json"

    _write_csv(step_csv, all_step_rows)
    _write_csv(epoch_csv, all_epoch_rows)
    _write_csv(phase_csv, all_phase_rows)
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "methods": resolved_methods,
        "scenario_counts": resolved_scenarios,
        "num_seeds": int(num_seeds),
        "base_run_config": describe_run(base_run_cfg),
        "cache_policy": asdict(resolved_cache_policy),
        "output_files": {
            "steps": str(step_csv),
            "epochs": str(epoch_csv),
            "phases": str(phase_csv),
        },
    }
    metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "step_rows": all_step_rows,
        "epoch_rows": all_epoch_rows,
        "phase_rows": all_phase_rows,
        "output_dir": str(output_path),
        "output_files": {
            "steps": str(step_csv),
            "epochs": str(epoch_csv),
            "phases": str(phase_csv),
            "metadata": str(metadata_json),
        },
        "metadata": metadata,
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the terminal interface for the diagnostics runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run SPNI gradient-conflict diagnostics for A-DFL and R-DFL."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["dfl", "rdfl", "adfl", "all"],
        default="all",
        help="Which predictor family to diagnose. 'all' runs R-DFL and A-DFL.",
    )
    parser.add_argument(
        "--scenarios",
        default="1,2,3,5,8",
        help=(
            "Comma-separated scenario counts using the repository's current "
            "num_scenarios semantics."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of sweep seeds to run for each method/scenario setting.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/gradient_diagnostics",
        help="Directory where the diagnostic CSVs and metadata JSON are saved.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=1,
        help="Only measure gradient conflicts every N training batches.",
    )
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="Optional cap on measured batches per epoch to limit overhead.",
    )
    parser.add_argument(
        "--replace-data",
        action="store_true",
        help="Force regeneration of cached base data.",
    )
    parser.add_argument(
        "--replace-intd-adv",
        action="store_true",
        help="Force regeneration of cached adversarial interdictions.",
    )
    parser.add_argument(
        "--replace-intd-rnd",
        action="store_true",
        help="Force regeneration of cached random interdictions.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply one config override using Python literal syntax when possible.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for gradient-conflict diagnostics."""
    parser = _build_cli_parser()
    parsed = parser.parse_args(argv)
    overrides = dict(_parse_override(item) for item in parsed.overrides)
    cache_policy = CachePolicy(
        replace_data=bool(parsed.replace_data),
        replace_intd_adv=bool(parsed.replace_intd_adv),
        replace_intd_rnd=bool(parsed.replace_intd_rnd),
    )
    result = run_gradient_conflict_diagnostics(
        method=parsed.method,
        scenarios=parsed.scenarios,
        num_seeds=parsed.num_seeds,
        output_dir=parsed.output_dir,
        log_every_n_steps=parsed.log_every_n_steps,
        max_batches_per_epoch=parsed.max_batches_per_epoch,
        cache_policy=cache_policy,
        **overrides,
    )
    print(
        "Saved gradient diagnostics to "
        f"{result['output_dir']} "
        f"(steps={len(result['step_rows'])}, epochs={len(result['epoch_rows'])})."
    )
    _print_console_summary(result)
    return result


if __name__ == "__main__":
    main()
