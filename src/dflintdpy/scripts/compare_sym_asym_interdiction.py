import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tabulate import tabulate

from dflintdpy.data.config import HP
from dflintdpy.models.grid import Grid
from dflintdpy.scripts.setup import (
    gen_data,
    gen_train_data,
    setup_dfl_predictor,
    setup_pfl_predictor,
)
from dflintdpy.solvers.asymmetric_interdictor import AsymmetricInterdictor
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor
from dflintdpy.utils.read_write import set_cache_replace_options
from dflintdpy.utils.versatile_utils import print_progress


VALID_PREDICTORS = {"oracle", "po", "dfl", "adfl", "rdfl"}


def _predict_cost(
    predictor: torch.nn.Module | None,
    feature: np.ndarray,
    true_cost: np.ndarray,
    normalization_constant: float,
) -> np.ndarray:
    if predictor is None:
        return true_cost
    with torch.no_grad():
        pred_cost = predictor(torch.tensor(feature, dtype=torch.float32)).detach().numpy()
    return pred_cost * normalization_constant


def compare_predictor_sym_vs_asym(
    cfg: HP,
    base_graph: Grid,
    test_data: Dict[str, np.ndarray],
    interdictions: Dict[str, np.ndarray],
    normalization_constant: float,
    predictor: torch.nn.Module | None = None,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    num_samples = cfg.get("num_test_samples")
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    evader_solver = ShortestPathGrb(base_graph)
    sym_objs: list[float] = []
    asym_objs: list[float] = []
    skipped_asym = 0

    for i in range(num_samples):
        true_cost = test_data["costs"][i] * normalization_constant
        interdiction_cost = interdictions["costs"][i] * normalization_constant
        feature = test_data["feats"][i]
        est_cost = _predict_cost(predictor, feature, true_cost, normalization_constant)

        sym_graph = Grid(*cfg.get("grid_size"))
        sym_graph.setObj(est_cost)
        sym_intd = SymmetricInterdictor(
            graph=sym_graph,
            k=cfg.get("budget"),
            interdiction_cost=interdiction_cost,
            max_cnt=cfg.get("benders_max_count"),
            eps=cfg.get("benders_eps"),
            output_flag=False,
        )
        x_sym, _, _ = sym_intd.solve(versatile=False)

        asym_intd = AsymmetricInterdictor(
            graph=base_graph,
            budget=cfg.get("budget"),
            true_costs=true_cost,
            true_delays=interdiction_cost,
            est_costs=est_cost,
            est_delays=interdiction_cost,
            lsd=cfg.get("lsd"),
        )
        x_asym, _ = asym_intd.solve()
        if x_asym is None:
            skipped_asym += 1
            print_progress(i, num_samples)
            continue
        x_asym_arr = np.asarray(x_asym, dtype=float)

        evader_solver.setObj(est_cost + x_sym * interdiction_cost)
        y_sym, _ = evader_solver.solve()
        sym_obj = float((true_cost + x_sym * interdiction_cost) @ y_sym)

        evader_solver.setObj(est_cost + x_asym_arr * interdiction_cost)
        y_asym, _ = evader_solver.solve()
        asym_obj = float((true_cost + x_asym_arr * interdiction_cost) @ y_asym)

        sym_objs.append(sym_obj)
        asym_objs.append(asym_obj)
        print_progress(i, num_samples)

    return np.asarray(sym_objs), np.asarray(asym_objs), skipped_asym


def load_requested_predictors(
    cfg: HP,
    graph: Grid,
    opt_model: ShortestPathGrb,
    requested: list[str],
) -> tuple[dict[str, torch.nn.Module | None], dict, float]:
    training_data_adv, test_data, normalization_constant, _ = gen_train_data(
        cfg,
        opt_model,
        interdiction_policy="adversarial",
    )

    predictors: dict[str, torch.nn.Module | None] = {"oracle": None}
    if "po" in requested:
        predictors["po"] = setup_pfl_predictor(
            cfg,
            graph,
            opt_model,
            training_data_adv,
            cache_tag="pfl",
            verbose=False,
        )

    if "dfl" in requested:
        nonadv_training_data = {
            "train_loader": training_data_adv["train_loader"].get_nonadverse_loader(),
            "val_loader": training_data_adv["val_loader"].get_nonadverse_loader(),
        }
        predictors["dfl"] = setup_dfl_predictor(
            cfg,
            graph,
            opt_model,
            nonadv_training_data,
            cache_tag="dfl",
            verbose=False,
        )

    if "adfl" in requested:
        predictors["adfl"] = setup_dfl_predictor(
            cfg,
            graph,
            opt_model,
            training_data_adv,
            cache_tag="adfl",
            verbose=False,
        )

    if "rdfl" in requested:
        training_data_rand, _, _, _ = gen_train_data(
            cfg,
            opt_model,
            interdiction_policy="random",
        )
        predictors["rdfl"] = setup_dfl_predictor(
            cfg,
            graph,
            opt_model,
            training_data_rand,
            cache_tag="rdfl",
            verbose=False,
        )

    for model in predictors.values():
        if model is not None:
            model.eval()

    return predictors, test_data, normalization_constant


def summarize_results(
    predictor_name: str,
    sym_objs: np.ndarray,
    asym_objs: np.ndarray,
    skipped_asym: int,
) -> dict:
    if len(sym_objs) == 0:
        raise RuntimeError(
            f"No valid samples for predictor '{predictor_name}'. "
            "Asymmetric solves may have all hit the time limit."
        )

    gap = asym_objs - sym_objs
    return {
        "predictor": predictor_name,
        "num_samples": int(len(sym_objs)),
        "skipped_asym": int(skipped_asym),
        "sym_mean": float(np.mean(sym_objs)),
        "sym_std": float(np.std(sym_objs)),
        "asym_mean": float(np.mean(asym_objs)),
        "asym_std": float(np.std(asym_objs)),
        "gap_mean": float(np.mean(gap)),
        "gap_std": float(np.std(gap)),
    }


def write_records_csv(output_path: Path, records: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "predictor",
        "sample_idx",
        "sym_objective",
        "asym_objective",
        "gap_asym_minus_sym",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare symmetric and asymmetric interdiction objectives."
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=["oracle", "po", "dfl"],
        help="Subset of {oracle, po, dfl, adfl, rdfl}.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of test samples.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output path for per-sample comparison CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = [p.lower() for p in args.predictors]
    invalid = sorted(set(requested) - VALID_PREDICTORS)
    if invalid:
        raise ValueError(f"Unknown predictor(s): {invalid}. Valid choices: {sorted(VALID_PREDICTORS)}")

    cfg = HP()
    set_cache_replace_options(
        replace_pred=False,
        replace_data=False,
        replace_intd_adv=False,
        replace_intd_rnd=False,
        replace_result=False,
        replace_fig=False,
        archive_replaced=True,
    )

    graph = Grid(*cfg.get("grid_size"))
    opt_model = ShortestPathGrb(graph)

    predictors, test_data, normalization_constant = load_requested_predictors(
        cfg,
        graph,
        opt_model,
        requested,
    )
    interdictions = gen_data(
        cfg,
        normalization_constant=normalization_constant,
        opt_model=opt_model,
        seed=cfg.get("intd_seed"),
    )

    summary_rows = []
    csv_rows: list[dict] = []

    for predictor_name in requested:
        print(f"\nComparing symmetric vs asymmetric for predictor: {predictor_name}")
        sym_objs, asym_objs, skipped_asym = compare_predictor_sym_vs_asym(
            cfg,
            graph,
            test_data,
            interdictions,
            normalization_constant,
            predictor=predictors[predictor_name],
            max_samples=args.max_samples,
        )
        summary = summarize_results(predictor_name, sym_objs, asym_objs, skipped_asym)
        summary_rows.append(summary)

        for i, (sym_obj, asym_obj) in enumerate(zip(sym_objs, asym_objs)):
            csv_rows.append(
                {
                    "predictor": predictor_name,
                    "sample_idx": i,
                    "sym_objective": f"{sym_obj:.10f}",
                    "asym_objective": f"{asym_obj:.10f}",
                    "gap_asym_minus_sym": f"{(asym_obj - sym_obj):.10f}",
                }
            )

    table = [
        [
            row["predictor"],
            row["num_samples"],
            row["skipped_asym"],
            f"{row['sym_mean']:.4f} +/- {row['sym_std']:.4f}",
            f"{row['asym_mean']:.4f} +/- {row['asym_std']:.4f}",
            f"{row['gap_mean']:+.4f} +/- {row['gap_std']:.4f}",
        ]
        for row in summary_rows
    ]
    headers = [
        "Predictor",
        "Samples",
        "Skipped Asym",
        "Symmetric Objective",
        "Asymmetric Objective",
        "Asym - Sym",
    ]
    print("\n" + tabulate(table, headers=headers, tablefmt="github"))

    if args.output_csv is not None:
        write_records_csv(args.output_csv, csv_rows)
        print(f"\nSaved per-sample comparison to: {args.output_csv}")


if __name__ == "__main__":
    main()
