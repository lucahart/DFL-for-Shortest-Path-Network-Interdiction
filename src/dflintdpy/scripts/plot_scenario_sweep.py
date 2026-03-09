import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dflintdpy.data.config import HP
from dflintdpy.utils.read_write import set_cache_replace_options
from dflintdpy.scripts.asym_spni_single_sim import single_sim


METHOD_ORDER = ["PO", "DFL", "R-DFL", "A-DFL"]
METHOD_SUFFIX = {
    "PO": "p",
    "DFL": "s",
    "R-DFL": "r",
    "A-DFL": "a",
}
METHOD_COLORS = {
    "PO": "#FF6B6B",
    "DFL": "#4ECDC4",
    "R-DFL": "#FFA552",
    "A-DFL": "#45B7D1",
}
CONDITION_PREFIX_ORACLE = {
    "unintd": ("o", "o_o"),
    "intd": ("s", "s_o"),
    "asym": ("a", "a_o"),
}
CONDITION_TITLES = {
    "unintd": "Uninterdicted",
    "intd": "Symmetric Interdiction",
    "asym": "Asymmetric Interdiction",
}
SCENARIOS_DEFAULT = "2,3,5"
set_cache_replace_options(
    replace_pred=False,
    replace_data=False,
    replace_intd_adv=False,
    replace_intd_rnd=False,
    replace_result=False,
    replace_fig=False,
    archive_replaced=True,
)


def _parse_scenarios(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No scenario values provided.")
    return vals


def _seed_triplet(seed_idx: int) -> tuple[int, int, int]:
    np.random.seed(seed_idx)
    return tuple(np.random.randint(0, 150, 3).tolist())


def _safe_pct(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (num - den) / den * 100.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_pct_sum(num: np.ndarray, den: np.ndarray) -> float:
    den_sum = float(np.sum(den))
    if den_sum == 0:
        return 0.0
    num_sum = float(np.sum(num - den))
    return float(np.nan_to_num(num_sum / den_sum * 100.0, nan=0.0, posinf=0.0, neginf=0.0))


def _init_storage(scenarios: list[int]) -> tuple[dict, dict]:
    conditions = list(CONDITION_PREFIX_ORACLE.keys())
    sim = {
        s: {c: {m: [] for m in METHOD_ORDER} for c in conditions}
        for s in scenarios
    }
    sample = {
        s: {c: {m: [] for m in METHOD_ORDER} for c in conditions}
        for s in scenarios
    }
    return sim, sample


def run_sweep(cfg: HP, scenarios: list[int], num_seeds: int) -> tuple[dict, dict]:
    sim_stats, sample_stats = _init_storage(scenarios)

    for scenario in scenarios:
        print("=" * 80)
        print(f"Running scenario sweep for num_scenarios={scenario}")
        print("=" * 80)
        cfg.set("num_scenarios", scenario)
        seed_0 = cfg.get("seed_sweep_offset")

        for seed_idx in range(num_seeds):
            seed1, seed2, seed3 = _seed_triplet(seed_idx + seed_0)
            cfg.set("seed", seed_idx + seed_0)
            cfg.set("random_seed", seed1)
            cfg.set("intd_seed", seed2)
            cfg.set("loader_seed", seed3)

            _, _, _, _, all_data = single_sim(
                cfg,
                visualize=False,
                compute_asym_intd=True,
                compute_asym_intd_2=False,
            )

            for method in METHOD_ORDER:
                suffix = METHOD_SUFFIX[method]
                for condition, (prefix, oracle_key) in CONDITION_PREFIX_ORACLE.items():
                    pred_key = f"{prefix}_{suffix}"
                    sim_val = _safe_pct_sum(all_data[pred_key], all_data[oracle_key])
                    sim_stats[scenario][condition][method].append(sim_val)

                    sample_val = _safe_pct(np.asarray(all_data[pred_key]), np.asarray(all_data[oracle_key]))
                    sample_stats[scenario][condition][method].extend(sample_val.tolist())

            print(f"  Seed {seed_idx + 1:02d}/{num_seeds} done. \n")

    return sim_stats, sample_stats


def _collect_mean_std(stats: dict, scenarios: list[int], condition: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for s in scenarios:
        arr = np.asarray(stats[s][condition][method], dtype=float)
        means.append(arr.mean())
        stds.append(arr.std())
    return np.asarray(means), np.asarray(stds)


def plot_stats(
    stats: dict,
    scenarios: list[int],
    output_path: Path,
    title: str,
    show: bool,
    conditions: list[str],
) -> None:
    x = np.asarray(scenarios, dtype=int)
    fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 5), sharex=True)
    if len(conditions) == 1:
        axes = [axes]

    for method in METHOD_ORDER:
        for idx, condition in enumerate(conditions):
            mean_c, std_c = _collect_mean_std(stats, scenarios, condition, method)
            axes[idx].errorbar(
                x,
                mean_c,
                yerr=std_c,
                marker="o",
                capsize=4,
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=method,
            )

    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        ax.set_title(CONDITION_TITLES[condition])
        ax.set_xlabel("Number of Adverse Scenarios")
        ax.set_ylabel("Percentage cost increase vs oracle (%)")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    root_dir = Path(__file__).parent.parent.parent.parent
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

    parser = argparse.ArgumentParser(
        description=(
            "Sweep num_scenarios and plot percentage cost increase vs oracle "
            "for PO/DFL/DFL+Rand/A-DFL."
        )
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=SCENARIOS_DEFAULT,
        help="Comma-separated scenario counts.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=cfg.get("num_seeds"),
        help="Number of seeds per scenario count.",
    )
    parser.add_argument(
        "--output-sim",
        type=str,
        default=str(root_dir / "figures" / "scenario_sweep_pct_by_simulation.png"),
        help="Output path for per-simulation figure.",
    )
    parser.add_argument(
        "--output-sample",
        type=str,
        default=str(root_dir / "figures" / "scenario_sweep_pct_by_sample.png"),
        help="Output path for per-sample figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures after saving.",
    )
    parser.add_argument(
        "--output-asym-sim",
        type=str,
        default=str(root_dir / "figures" / "scenario_sweep_asym_pct_by_simulation.png"),
        help="Output path for asymmetric-interdiction per-simulation figure.",
    )
    args = parser.parse_args()

    scenarios = _parse_scenarios(args.scenarios)
    sim_stats, sample_stats = run_sweep(cfg, scenarios, args.num_seeds)

    plot_stats(
        sim_stats,
        scenarios,
        output_path=Path(args.output_sim),
        title="Percentage Increase vs Oracle (Symmetric Simulations)",
        show=args.show,
        conditions=["unintd", "intd"],
    )
    plot_stats(
        sim_stats,
        scenarios,
        output_path=Path(args.output_asym_sim),
        title="Percentage Increase vs Oracle (Asymmetric Simulations)",
        show=args.show,
        conditions=["asym"],
    )
    # plot_stats(
    #     sample_stats,
    #     scenarios,
    #     output_path=Path(args.output_sample),
    #     title="Percentage Increase vs Oracle (Mean+Std over Samples)",
    #     show=args.show,
    #     conditions=["unintd", "intd"],
    # )
    # plot_stats(
    #     sim_stats,
    #     scenarios,
    #     output_path=Path(args.output_asym_sim),
    #     title="Asymmetric Interdiction: Percentage Increase vs Oracle (Mean+Std over Simulations)",
    #     show=args.show,
    #     conditions=["asym"],
    # )
    pass


if __name__ == "__main__":
    main()
