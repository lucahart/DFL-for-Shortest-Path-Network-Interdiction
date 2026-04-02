from pathlib import Path
from types import SimpleNamespace

from dflintdpy.simulation.spni.types import SweepResult

import dflintdpy.simulation.spni.pipeline as pipeline_module


############################
### Helper functionality ###
############################


def _percentage_increases() -> dict[str, dict[str, list[float]]]:
    """Return compact sweep statistics for the scenario smoke test."""
    return {
        "simulations": {
            "no_intd_p": [1.0],
            "no_intd_s": [2.0],
            "no_intd_r": [3.0],
            "no_intd_a": [4.0],
            "sym_intd_p": [5.0],
            "sym_intd_s": [6.0],
            "sym_intd_r": [7.0],
            "sym_intd_a": [8.0],
            "asym_intd_p": [9.0],
            "asym_intd_s": [10.0],
            "asym_intd_r": [11.0],
            "asym_intd_a": [12.0],
        },
        "samples": {
            "no_intd_p": [13.0],
            "no_intd_s": [14.0],
            "no_intd_r": [15.0],
            "no_intd_a": [16.0],
            "sym_intd_p": [17.0],
            "sym_intd_s": [18.0],
            "sym_intd_r": [19.0],
            "sym_intd_a": [20.0],
            "asym_intd_p": [21.0],
            "asym_intd_s": [22.0],
            "asym_intd_r": [23.0],
            "asym_intd_a": [24.0],
        },
    }


###############################
### test scenario-sweep smoke ###
###############################


def test_spni_pipeline_scenario_sweep_smoke_saves_result_figures(
    tmp_path,
    monkeypatch,
):
    """Verify that a tiny scenario sweep saves its result figures."""
    # Arrange a compact config, a temp figure directory, and a sweep stub.
    base_cfg = SimpleNamespace(num_seeds=4, num_scenarios=9, budget=1)
    figure_directory = tmp_path / "scenario-figures"
    calls: list[dict[str, object]] = []

    def _fake_run_seed_sweep(
        cfg,
        *,
        num_seeds,
        present_results,
        figure_directory=None,
        **options,
    ):
        calls.append(
            {
                "cfg": cfg,
                "num_seeds": num_seeds,
                "present_results": present_results,
                "figure_directory": figure_directory,
                "options": options,
            }
        )
        return SweepResult(
            run_config=cfg,
            results=[],
            aggregated_summary={
                "percentage_increases": _percentage_increases(),
            },
            diagnostics={"num_runs": num_seeds},
        )

    # Act by running the scenario sweep with presentation enabled.
    monkeypatch.setattr(
        pipeline_module,
        "run_seed_sweep",
        _fake_run_seed_sweep,
    )
    result = pipeline_module.run_scenario_sweep(
        base_cfg,
        scenarios=[2],
        num_seeds=3,
        present_results=True,
        figure_directory=figure_directory,
    )

    # Assert that the smoke run wrote the expected figure artefacts.
    assert calls[0]["present_results"] is False, \
        "Scenario sweeps should keep inner seed sweeps non-persistent."
    assert calls[0]["figure_directory"] is None, \
        "Scenario sweeps should not forward a figure directory to seed sweeps."
    assert result["diagnostics"]["present_results"] is True, \
        "Scenario sweeps should record when presentation is enabled."
    simulation_plot_path = Path(result["diagnostics"]["simulation_plot_path"])
    sample_plot_path = Path(result["diagnostics"]["sample_plot_path"])
    asym_simulation_plot_path = Path(
        result["diagnostics"]["asym_simulation_plot_path"]
    )
    asym_sample_plot_path = Path(result["diagnostics"]["asym_sample_plot_path"])
    assert simulation_plot_path.parent == figure_directory, \
        "Scenario sweeps should keep the symmetric simulation figure together."
    assert sample_plot_path.parent == figure_directory, \
        "Scenario sweeps should keep the sample figure together."
    assert asym_simulation_plot_path.parent == figure_directory, \
        "Scenario sweeps should keep the asymmetric simulation figure together."
    assert asym_sample_plot_path.parent == figure_directory, \
        "Scenario sweeps should keep the asymmetric sample figure together."
    assert simulation_plot_path.exists(), \
        "Scenario sweeps should write the symmetric simulation figure."
    assert sample_plot_path.exists(), \
        "Scenario sweeps should write the sample figure."
    assert asym_simulation_plot_path.exists(), \
        "Scenario sweeps should write the asymmetric simulation figure."
    assert asym_sample_plot_path.exists(), \
        "Scenario sweeps should write the asymmetric sample figure."
    assert result["sweep_results"][2].diagnostics["num_runs"] == 3, \
        "Scenario sweeps should preserve the inner sweep result payload."
    pass
