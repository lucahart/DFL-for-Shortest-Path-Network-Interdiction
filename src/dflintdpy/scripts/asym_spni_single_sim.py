"""Legacy single-run compatibility entrypoint for SPNI simulations."""

from __future__ import annotations

from typing import Any

from dflintdpy.simulation.spni.pipeline import run_single_simulation
from dflintdpy.simulation.spni.results import to_legacy_all_data


def single_sim(
    cfg: Any,
    visualize: bool = False,
    compute_asym_intd_2: bool = True,
    compute_asym_intd: bool = True,
):
    """Run one SPNI simulation through the new pipeline and adapt its output.

    This preserves the legacy public entrypoint and its 5-tuple return shape
    while delegating the actual orchestration into
    `simulation.spni.pipeline.run_single_simulation(...)`.
    """
    del visualize

    result = run_single_simulation(
        cfg,
        compute_asym_intd=compute_asym_intd,
        compute_wrong_asym_intd=compute_asym_intd_2,
    )
    summary_bundle = result.summary_bundle
    return (
        summary_bundle.prediction_mean_std,
        summary_bundle.metrics,
        summary_bundle.table_1,
        summary_bundle.table_2,
        to_legacy_all_data(summary_bundle),
    )
