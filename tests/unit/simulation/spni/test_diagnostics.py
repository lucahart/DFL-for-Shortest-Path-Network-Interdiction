from __future__ import annotations

from pathlib import Path

import pytest

import dflintdpy.simulation.spni.diagnostics as diagnostics_module


pytestmark = [pytest.mark.unit, pytest.mark.torch, pytest.mark.pyepo]


def test_run_gradient_conflict_diagnostics_writes_sweep_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that the diagnostics sweep writes CSVs for every requested run."""
    calls: list[tuple[int, int, str, int, int | None]] = []

    def _fake_run_single_diagnostic(
        run_cfg,
        *,
        method: str,
        log_every_n_steps: int,
        max_batches_per_epoch: int | None,
    ):
        calls.append(
            (
                int(run_cfg.num_scenarios),
                int(run_cfg.seed),
                method,
                int(log_every_n_steps),
                max_batches_per_epoch,
            )
        )
        step_row = {
            "epoch": 1,
            "batch_index": 1,
            "global_step": 0,
            "requested_scenarios": int(run_cfg.num_scenarios),
            "effective_scenarios": max(1, int(run_cfg.num_scenarios) - 1),
            "parameter_count": 7,
            "mean_pairwise_cosine": -0.25,
            "cancellation_ratio": 0.4,
            "summed_gradient_norm": 1.2,
            "sum_individual_norms": 3.0,
            "pair_count": 1,
            "batch_size": 4,
            "method": method,
            "configured_scenarios": int(run_cfg.num_scenarios),
            "sweep_seed": int(run_cfg.seed),
        }
        epoch_row = dict(step_row)
        epoch_row["batches_measured"] = 1
        epoch_row["batch_index"] = None
        phase_row = dict(epoch_row)
        phase_row["phase"] = "full"
        phase_row["epoch"] = None
        phase_row["epoch_start"] = 1
        phase_row["epoch_end"] = 1
        return [step_row], [epoch_row], [phase_row]

    monkeypatch.setattr(
        diagnostics_module,
        "_run_single_diagnostic",
        _fake_run_single_diagnostic,
    )

    result = diagnostics_module.run_gradient_conflict_diagnostics(
        method="all",
        scenarios=[2, 3],
        num_seeds=2,
        output_dir=tmp_path,
        log_every_n_steps=5,
        max_batches_per_epoch=2,
        spo_epochs=1,
    )

    assert len(calls) == 8, (
        "The diagnostics sweep should run both methods across every scenario "
        "count and seed."
    )
    assert {call[2] for call in calls} == {"rdfl", "adfl"}, (
        "method='all' should expand to the R-DFL and A-DFL families."
    )
    assert all(call[3] == 5 for call in calls), (
        "The sweep did not forward the logging cadence into each run."
    )
    assert all(call[4] == 2 for call in calls), (
        "The sweep did not forward the batch cap into each run."
    )
    assert len(result["step_rows"]) == 8, (
        "One fake step row should be recorded for each sweep run."
    )
    assert len(result["epoch_rows"]) == 8, (
        "One fake epoch row should be recorded for each sweep run."
    )
    assert len(result["phase_rows"]) == 8, (
        "One fake phase row should be recorded for each sweep run."
    )
    assert Path(result["output_files"]["steps"]).exists(), (
        "The diagnostics sweep did not write the step-level CSV."
    )
    assert Path(result["output_files"]["epochs"]).exists(), (
        "The diagnostics sweep did not write the epoch-level CSV."
    )
    assert Path(result["output_files"]["phases"]).exists(), (
        "The diagnostics sweep did not write the phase-level CSV."
    )
    metadata = result["metadata"]
    assert metadata["methods"] == ["rdfl", "adfl"], (
        "The saved metadata did not record the expanded method list."
    )
    assert metadata["scenario_counts"] == [2, 3], (
        "The saved metadata did not record the requested scenario sweep."
    )
    pass


def test_diagnostics_main_prints_console_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Verify that the CLI prints the saved-path and comparison summary."""
    fake_result = {
        "step_rows": [{"global_step": 0}],
        "epoch_rows": [{"epoch": 1}],
        "phase_rows": [
            {
                "configured_scenarios": 2,
                "phase": "full",
                "method": "rdfl",
                "runs": 1,
                "effective_scenarios": 1,
                "mean_pairwise_cosine": 0.2,
                "cancellation_ratio": 0.7,
            },
            {
                "configured_scenarios": 2,
                "phase": "full",
                "method": "adfl",
                "runs": 1,
                "effective_scenarios": 1,
                "mean_pairwise_cosine": -0.1,
                "cancellation_ratio": 0.4,
            },
        ],
        "output_dir": "/tmp/gradient-diagnostics",
        "output_files": {
            "steps": "/tmp/gradient-diagnostics/gradient_conflict_steps.csv",
            "epochs": "/tmp/gradient-diagnostics/gradient_conflict_epochs.csv",
            "phases": "/tmp/gradient-diagnostics/gradient_conflict_phases.csv",
            "metadata": (
                "/tmp/gradient-diagnostics/gradient_conflict_metadata.json"
            ),
        },
        "metadata": {},
    }

    monkeypatch.setattr(
        diagnostics_module,
        "run_gradient_conflict_diagnostics",
        lambda **kwargs: fake_result,
    )

    diagnostics_module.main(["--method", "all", "--scenarios", "2"])
    captured = capsys.readouterr()

    assert "Saved gradient diagnostics to /tmp/gradient-diagnostics" in \
        captured.out, (
            "The CLI did not print the saved-output location."
        )
    assert "Gradient conflict summary" in captured.out, (
        "The CLI did not print the phase summary heading."
    )
    assert "A-DFL vs R-DFL gaps (A-DFL - R-DFL)" in captured.out, (
        "The CLI did not print the comparison gap section."
    )
    assert "2  full   adfl" in captured.out, (
        "The CLI summary did not include the A-DFL phase row."
    )
    pass
