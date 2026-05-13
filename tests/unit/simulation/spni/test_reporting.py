from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import dflintdpy.simulation.spni.reporting as reporting_module
from dflintdpy.simulation.spni.types import TrainingLogBundle


pytestmark = [pytest.mark.unit]


################
### Fixtures ###
################


@pytest.fixture
def simulation_result() -> SimpleNamespace:
    """Return a compact reporting result with metadata and summary payloads."""
    return SimpleNamespace(
        run_config=SimpleNamespace(
            grid_size=(4, 5),
            num_train_samples=10,
            num_test_samples=3,
            budget=2,
            pred_model="linear",
            num_scenarios=4,
            po_epochs=2,
            spo_epochs=3,
        ),
        seed_bundle=SimpleNamespace(
            sweep_seed=11,
            random_seed=13,
            intd_seed=17,
            loader_seed=19,
        ),
        graph_bundle=SimpleNamespace(
            graph_kind="synthetic",
            graph_source=None,
        ),
        predictor_bundle=SimpleNamespace(
            logs={
                "pfl": TrainingLogBundle(
                    train_loss=[4.0, 2.0],
                    train_regret=[3.0, 1.5],
                    val_loss=[5.0, 2.5],
                    val_regret=[6.0, 3.0],
                ),
                "rdfl": TrainingLogBundle(
                    train_loss=[8.0, 4.0],
                    train_regret=[7.0, 3.5],
                    val_loss=None,
                    val_regret=None,
                ),
            },
        ),
        summary_bundle=SimpleNamespace(
            prediction_mean_std={
                "po_mean": 1.25,
                "spo_mean": 2.5,
                "adfl_mean": 3.75,
            },
            metrics={
                "metric_1": 0.5,
                "asym_nan_rows_po": 1,
            },
            table_1={
                "t1_o_n_mean": 10.0,
                "t1_p_a_mean": 12.5,
            },
            table_2={
                "t2_p_s_mean": 7.5,
                "t2_p_a_mean": 8.5,
            },
        ),
        artifacts=SimpleNamespace(figure_paths={}),
    )


############################
### Helper functionality ###
############################


class _FakePyplot:
    """Capture matplotlib-style plotting calls without rendering figures."""

    def __init__(self) -> None:
        self.saved_paths: list[str] = []
        self.plotted_labels: list[str | None] = []
        self.closed_figures: list[object | None] = []

    def subplots(self, *args, **kwargs):
        """Return one fake figure and two fake axes."""
        fig = _FakeFigure(self)
        return fig, (_FakeAxis(self), _FakeAxis(self))

    def close(self, figure=None) -> None:
        self.closed_figures.append(figure)


class _FakeFigure:
    """Capture figure-level save calls."""

    def __init__(self, pyplot: _FakePyplot) -> None:
        self._pyplot = pyplot

    def suptitle(self, *args, **kwargs) -> None:
        pass

    def tight_layout(self, *args, **kwargs) -> None:
        pass

    def savefig(self, path, *args, **kwargs) -> None:
        self._pyplot.saved_paths.append(str(path))


class _FakeAxis:
    """Capture axis-level plotting calls."""

    def __init__(self, pyplot: _FakePyplot) -> None:
        self._pyplot = pyplot

    def plot(self, values, *args, label=None, **kwargs) -> None:
        self._pyplot.plotted_labels.append(label)

    def set_title(self, value: str) -> None:
        pass

    def set_xlabel(self, value: str) -> None:
        pass

    def set_ylabel(self, value: str) -> None:
        pass

    def grid(self, *args, **kwargs) -> None:
        pass

    def legend(self, *args, **kwargs) -> None:
        pass


def _joined_table_text(tables) -> str:
    """Return all formatted table strings as one searchable text block."""
    if isinstance(tables, dict):
        values = tables.values()
    elif isinstance(tables, (list, tuple)):
        values = tables
    else:
        values = [tables]
    return "\n".join(str(value) for value in values)


###############################################
### test format_simulation_summary_tables ###
###############################################


def test_spni_reporting_format_summary_tables_includes_metadata_and_labels(
    simulation_result: SimpleNamespace,
):
    """Verify that formatted summary tables expose metadata and labels."""
    # Act by formatting the planned reporting summary tables.
    tables = reporting_module.format_simulation_summary_tables(
        simulation_result
    )
    text = _joined_table_text(tables)

    # Assert that callers receive table strings with run metadata.
    assert isinstance(tables, (dict, list, tuple, str)), (
        "format_simulation_summary_tables should return table string content."
    )
    assert "Run metadata" in text, (
        "Formatted summary tables should include a run metadata heading."
    )
    assert "grid_size" in text and "(4, 5)" in text, (
        "Run metadata should include the configured grid size."
    )
    assert "sweep_seed" in text and "11" in text, (
        "Run metadata should include the sweep seed."
    )

    # Assert that all planned summary sections are represented.
    assert "Prediction summary" in text, (
        "Formatted tables should label the prediction summary section."
    )
    assert "Metrics" in text, (
        "Formatted tables should label the scalar metrics section."
    )
    assert "Table 1" in text, (
        "Formatted tables should label the primary summary table."
    )
    assert "Table 2" in text, (
        "Formatted tables should label the wrong-model summary table."
    )
    assert "PFL" in text and "12.5000" in text, (
        "Formatted tables should include summary table entries."
    )
    pass


########################################
### test print_simulation_summary ###
########################################


def test_spni_reporting_print_summary_emits_formatted_tables(
    simulation_result: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Verify that print_simulation_summary emits formatted table strings."""
    # Arrange a deterministic formatter result for the print wrapper.
    formatted_tables = {
        "metadata": "Run metadata\nseed: 11",
        "table_1": "Table 1\nvalue: 12.5",
        "table_2": "Table 2\nvalue: 8.5",
    }
    monkeypatch.setattr(
        reporting_module,
        "format_simulation_summary_tables",
        lambda result: formatted_tables,
    )

    # Act by printing the summary through the reporting helper.
    reporting_module.print_simulation_summary(simulation_result)
    captured = capsys.readouterr()

    # Assert that every formatted table string was printed.
    assert "Run metadata\nseed: 11" in captured.out, (
        "print_simulation_summary should emit the metadata table string."
    )
    assert "Table 1\nvalue: 12.5" in captured.out, (
        "print_simulation_summary should emit the Table 1 string."
    )
    assert "Table 2\nvalue: 8.5" in captured.out, (
        "print_simulation_summary should emit the Table 2 string."
    )
    pass


##########################################
### test save_learning_curve_plots ###
##########################################


def test_spni_reporting_save_learning_curve_plots_saves_expected_pngs(
    simulation_result: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that learning-curve plotting saves combined and per-log PNGs."""
    # Arrange a matplotlib stand-in on the reporting module.
    fake_pyplot = _FakePyplot()
    monkeypatch.setattr(reporting_module, "plt", fake_pyplot)

    # Act by saving learning-curve plots to an explicit figure directory.
    paths = reporting_module.save_learning_curve_plots(
        simulation_result,
        figure_directory=tmp_path,
    )

    # Assert that a combined plot and one plot per predictor log are saved.
    path_text = {str(path) for path in paths.values()}
    curve_dir = tmp_path / "learning_curves"
    stem = "learning_curves_seed_11_scenarios_4"
    expected_paths = {
        str(curve_dir / f"{stem}_combined.png"),
        str(curve_dir / f"{stem}_pfl.png"),
        str(curve_dir / f"{stem}_rdfl.png"),
    }
    assert expected_paths.issubset(path_text), (
        "save_learning_curve_plots should return combined and per-predictor "
        "PNG paths."
    )
    assert expected_paths.issubset(set(fake_pyplot.saved_paths)), (
        "save_learning_curve_plots should save every returned PNG path."
    )
    assert simulation_result.artifacts.figure_paths == paths, (
        "Saved learning-curve paths should be recorded on result artifacts."
    )
    assert "PFL train" in fake_pyplot.plotted_labels, (
        "The PFL train-loss curve should be plotted with a readable label."
    )
    assert "R-DFL train" in fake_pyplot.plotted_labels, (
        "The R-DFL train-loss curve should be plotted with a readable label."
    )
    pass
