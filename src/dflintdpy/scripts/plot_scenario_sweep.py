import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dflintdpy.data.config import HP
from dflintdpy.utils.read_write import set_cache_replace_options
from dflintdpy.scripts.asym_spni_single_sim import single_sim


METHOD_ORDER = ["PO", "DFL", "DFL+Rand", "A-DFL"]
METHOD_KEYS = {
    "PO": ("o_p", "s_p"),
    "DFL": ("o_s", "s_s"),
    "DFL+Rand": ("o_r", "s_r"),
    "A-DFL": ("o_a", "s_a"),
}
METHOD_COLORS = {
    "PO": "#FF6B6B",
    "DFL": "#4ECDC4",
    "DFL+Rand": "#FFA552",
    "A-DFL": "#45B7D1",
}
SCENARIOS_DEFAULT = "2,3"
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
    sim = {
        s: {"unintd": {m: [] for m in METHOD_ORDER}, "intd": {m: [] for m in METHOD_ORDER}}
        for s in scenarios
    }
    sample = {
        s: {"unintd": {m: [] for m in METHOD_ORDER}, "intd": {m: [] for m in METHOD_ORDER}}
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
                compute_asym_intd=False,
                compute_asym_intd_2=False,
            )

            for method in METHOD_ORDER:
                no_key, sym_key = METHOD_KEYS[method]

                no_sim = _safe_pct_sum(all_data[no_key], all_data["o_o"])
                sym_sim = _safe_pct_sum(all_data[sym_key], all_data["s_o"])
                sim_stats[scenario]["unintd"][method].append(no_sim)
                sim_stats[scenario]["intd"][method].append(sym_sim)

                no_sample = _safe_pct(np.asarray(all_data[no_key]), np.asarray(all_data["o_o"]))
                sym_sample = _safe_pct(np.asarray(all_data[sym_key]), np.asarray(all_data["s_o"]))
                sample_stats[scenario]["unintd"][method].extend(no_sample.tolist())
                sample_stats[scenario]["intd"][method].extend(sym_sample.tolist())

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


def plot_stats(stats: dict, scenarios: list[int], output_path: Path, title: str, show: bool) -> None:
    x = np.asarray(scenarios, dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for method in METHOD_ORDER:
        mean_u, std_u = _collect_mean_std(stats, scenarios, "unintd", method)
        mean_i, std_i = _collect_mean_std(stats, scenarios, "intd", method)

        axes[0].errorbar(
            x,
            mean_u,
            yerr=std_u,
            marker="o",
            capsize=4,
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=method,
        )
        axes[1].errorbar(
            x,
            mean_i,
            yerr=std_i,
            marker="o",
            capsize=4,
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=method,
        )

    axes[0].set_title("Uninterdicted")
    axes[1].set_title("Symmetric Interdiction")
    for ax in axes:
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
        replace_pred=True,
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
    args = parser.parse_args()

    scenarios = _parse_scenarios(args.scenarios)
    sim_stats, sample_stats = run_sweep(cfg, scenarios, args.num_seeds)

    plot_stats(
        sim_stats,
        scenarios,
        output_path=Path(args.output_sim),
        title="Percentage Increase vs Oracle (Mean+Std over Simulations)",
        show=args.show,
    )
    plot_stats(
        sample_stats,
        scenarios,
        output_path=Path(args.output_sample),
        title="Percentage Increase vs Oracle (Mean+Std over Samples)",
        show=args.show,
    )
    pass


if __name__ == "__main__":
    main()
