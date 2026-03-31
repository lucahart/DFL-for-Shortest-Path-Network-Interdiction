"""Legacy sweep-script compatibility entrypoint for SPNI simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dflintdpy.data.config import HP
from dflintdpy.simulation.spni.pipeline import run_seed_sweep
from dflintdpy.simulation.spni.results import to_legacy_all_data
from dflintdpy.simulation.spni.types import SimulationResult, SweepResult
from dflintdpy.utils.analyse_results import analyze_results
from dflintdpy.utils.read_write import set_cache_replace_options
from dflintdpy.utils.read_write_results import save_results_to_csv


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read one legacy config value from either object or mapping style."""
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _default_output_path(cfg: Any, num_seeds: int) -> str:
    """Rebuild the legacy sweep-results filename format."""
    m_size, n_size = _cfg_get(cfg, "grid_size", (0, 0))
    return (
        "results_train_"
        f"{_cfg_get(cfg, 'num_train_samples', 0)}"
        "_valid_"
        f"{_cfg_get(cfg, 'num_val_samples', 0)}"
        "_test_"
        f"{_cfg_get(cfg, 'num_test_samples', 0)}"
        "_m_"
        f"{m_size}"
        "_n_"
        f"{n_size}"
        "_deg_"
        f"{_cfg_get(cfg, 'deg', 0)}"
        "_noise_"
        f"{_cfg_get(cfg, 'noise_width', 0)}"
        "_seeds_"
        f"{int(num_seeds)}.csv"
    )


def _legacy_result_dict(result: SimulationResult) -> dict[str, Any]:
    """Adapt one typed single-run result into the legacy export shape."""
    summary_bundle = result.summary_bundle
    return {
        "seed": int(result.seed_bundle.sweep_seed),
        "prediction_mean_std": summary_bundle.prediction_mean_std,
        "metrics": summary_bundle.metrics,
        "table_1": summary_bundle.table_1,
        "table_2": summary_bundle.table_2,
        "all_data": to_legacy_all_data(summary_bundle),
    }


def _legacy_results_payload(sweep_result: SweepResult) -> list[dict[str, Any]]:
    """Adapt one typed sweep result into the legacy persistence payload."""
    return [
        _legacy_result_dict(result)
        for result in sweep_result.results
    ]


def run_sweep(
    cfg: Any | None = None,
    *,
    num_seeds: int | None = None,
    compute_asym_intd_2: bool = True,
    compute_asym_intd: bool = True,
    persist_results: bool = False,
    analyze: bool = False,
    output_path: str | Path | None = None,
    apply_default_cache_policy: bool = False,
) -> SweepResult:
    """Run the legacy sweep entrypoint through the typed pipeline layer.

    This preserves the sweep script as a compatibility surface while moving
    the seed-loop business logic into
    `simulation.spni.pipeline.run_seed_sweep(...)`.
    """
    cfg = HP() if cfg is None else cfg

    if apply_default_cache_policy:
        set_cache_replace_options(
            replace_data=False,
            replace_intd=True,
            replace_pred=False,
            replace_eval=False,
        )

    resolved_num_seeds = int(
        _cfg_get(cfg, "num_seeds", 1) if num_seeds is None else num_seeds
    )
    sweep_result = run_seed_sweep(
        cfg,
        num_seeds=resolved_num_seeds,
        compute_asym_intd=compute_asym_intd,
        compute_wrong_asym_intd=compute_asym_intd_2,
    )

    legacy_results = _legacy_results_payload(sweep_result)
    resolved_output_path: str | None = None
    if persist_results:
        resolved_output_path = str(
            output_path or _default_output_path(cfg, resolved_num_seeds)
        )
        save_results_to_csv(legacy_results, output_path=resolved_output_path)

    if analyze:
        analyze_results()

    sweep_result.diagnostics.update(
        {
            "persist_results": bool(persist_results),
            "analysis_requested": bool(analyze),
            "legacy_output_path": resolved_output_path,
            "legacy_results_count": len(legacy_results),
        }
    )
    return sweep_result


def main() -> SweepResult:
    """Run the legacy sweep with explicit persistence and analysis enabled."""
    sweep_result = run_sweep(
        HP(),
        persist_results=True,
        analyze=True,
        apply_default_cache_policy=True,
    )
    print("Done with all simulations.")
    return sweep_result


if __name__ == "__main__":
    main()
