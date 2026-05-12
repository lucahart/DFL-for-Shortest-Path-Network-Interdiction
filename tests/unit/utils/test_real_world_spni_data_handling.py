from pathlib import Path

import numpy as np
import pytest

from dflintdpy.utils.real_world_spni_data_handling import (
    csv_to_graph,
    real_world_graph_to_graph,
    tntp_to_graph,
)


#####################
### Test fixtures ###
#####################


def _write_csv_graph(path: Path) -> Path:
    """Write a compact real-world graph CSV fixture."""

    path.write_text(
        "From,To,Weight\n"
        "1,2,10.5\n"
        "2,4,11.5\n"
        "1,2,99.0\n"
        "4,3,12.5\n",
        encoding="utf-8",
    )
    return path


def _write_tntp_graph(path: Path) -> Path:
    """Write a compact TNTP graph fixture."""

    path.write_text(
        "<NUMBER OF NODES> 4\n"
        "<NUMBER OF LINKS> 3\n"
        "<END OF METADATA>\n"
        "\n"
        "~ init_node term_node capacity length free_flow_time b power speed "
        "toll link_type ;\n"
        "1 2 100 10 1.5 0.15 4 10 0 1 ;\n"
        "2 4 100 10 2.5 0.15 4 10 0 1 ;\n"
        "1 3 100 10 3.5 0.15 4 10 0 1 ;\n",
        encoding="utf-8",
    )
    return path


#########################
### test csv_to_graph ###
#########################


def test_real_world_spni_csv_to_graph_imports_topology_with_unit_costs(
        tmp_path: Path,
    ):
    """Verify CSV imports preserve ordered topology and use unit costs."""
    # Arrange a CSV with a duplicate arc and an ignored non-topology column.
    graph_path = _write_csv_graph(tmp_path / "sample.csv")

    # Act by importing with default topology-only behavior.
    graph = csv_to_graph(graph_path)

    # Assert only first-seen directed arcs define the graph.
    assert graph.arcs == [(1, 2), (2, 4), (4, 3)], \
        "CSV import should preserve first-seen directed arcs after de-duping."
    assert np.array_equal(graph.vertices, np.array([1, 2, 3, 4])), \
        "CSV import should preserve observed node labels."
    assert np.allclose(graph.cost, np.ones(3, dtype=float)), \
        "CSV import should use unit costs by default."
    assert graph.source == 1, \
        "CSV import should use the first sorted node as the default source."
    assert graph.target == 4, \
        "CSV import should use the largest node as the default target."
    pass


def test_real_world_spni_csv_to_graph_can_import_requested_cost_column(
        tmp_path: Path,
    ):
    """Verify CSV imports can load a numeric cost column for inspection."""
    # Arrange a CSV with a numeric Weight column.
    graph_path = _write_csv_graph(tmp_path / "sample.csv")

    # Act by asking the importer to use Weight as graph costs.
    graph = csv_to_graph(
        graph_path,
        topology_only=False,
        cost_column="Weight",
    )

    # Assert duplicate arcs keep the first-seen cost value.
    assert graph.arcs == [(1, 2), (2, 4), (4, 3)], \
        "CSV cost import should keep first-seen arc ordering."
    assert np.allclose(graph.cost, np.array([10.5, 11.5, 12.5])), \
        "CSV cost import should read the requested numeric cost column."
    pass


##########################
### test tntp_to_graph ###
##########################


def test_real_world_spni_tntp_to_graph_imports_topology_with_unit_costs(
        tmp_path: Path,
    ):
    """Verify TNTP imports preserve topology and use unit costs."""
    # Arrange a compact TNTP network fixture.
    graph_path = _write_tntp_graph(tmp_path / "sample.tntp")

    # Act by importing with default topology-only behavior.
    graph = tntp_to_graph(graph_path)

    # Assert metadata and rows define the graph structure.
    assert graph.arcs == [(1, 2), (2, 4), (1, 3)], \
        "TNTP import should preserve directed link ordering."
    assert np.array_equal(graph.vertices, np.array([1, 2, 3, 4])), \
        "TNTP import should include nodes implied by metadata."
    assert np.allclose(graph.cost, np.ones(3, dtype=float)), \
        "TNTP import should use unit costs by default."
    assert graph.real_world_metadata["format"] == "tntp", \
        "TNTP import should record file format metadata."
    pass


def test_real_world_spni_tntp_to_graph_can_import_free_flow_time_costs(
        tmp_path: Path,
    ):
    """Verify TNTP imports can load free-flow times as costs."""
    # Arrange a compact TNTP network fixture.
    graph_path = _write_tntp_graph(tmp_path / "sample.tntp")

    # Act by loading a real TNTP attribute as the graph cost vector.
    graph = tntp_to_graph(
        graph_path,
        topology_only=False,
        cost_column="free_flow_time",
    )

    # Assert the selected column is aligned with graph.arcs.
    assert np.allclose(graph.cost, np.array([1.5, 2.5, 3.5])), \
        "TNTP cost import should read free_flow_time in arc order."
    pass


##################################
### test real_world_graph_to_graph ###
##################################


def test_real_world_spni_graph_loader_dispatches_from_file_suffix(
        tmp_path: Path,
    ):
    """Verify the generic loader dispatches CSV and TNTP paths."""
    # Arrange one CSV and one TNTP fixture.
    csv_path = _write_csv_graph(tmp_path / "sample.csv")
    tntp_path = _write_tntp_graph(tmp_path / "sample.tntp")

    # Act by loading both through the format dispatcher.
    csv_graph = real_world_graph_to_graph(csv_path)
    tntp_graph = real_world_graph_to_graph(tntp_path)

    # Assert each suffix reaches the expected parser.
    assert csv_graph.real_world_metadata["format"] == "csv", \
        "Generic loader should dispatch .csv files to csv_to_graph."
    assert tntp_graph.real_world_metadata["format"] == "tntp", \
        "Generic loader should dispatch .tntp files to tntp_to_graph."
    pass


def test_real_world_spni_graph_loader_rejects_unknown_suffix(tmp_path: Path):
    """Verify the generic loader rejects unsupported graph file formats."""
    # Arrange an unsupported graph fixture path.
    graph_path = tmp_path / "sample.txt"
    graph_path.write_text("not a graph\n", encoding="utf-8")

    # Act and assert that the format error is explicit.
    with pytest.raises(ValueError, match="Unsupported real-world graph format"):
        real_world_graph_to_graph(graph_path)
    pass
