import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

pytest.importorskip("pyepo")
pytest.importorskip("gurobipy")

from dflintdpy.data.config import HP
from dflintdpy.simulation.spni.config import build_run_config
from dflintdpy.simulation.spni.types import GraphBundle

import dflintdpy.simulation.spni.build as build_module
import dflintdpy.simulation.spni.data as data_module


pytestmark = [pytest.mark.integration, pytest.mark.pyepo]


############################
### Helper functionality ###
############################


def _run_cfg_for_graph(graph_path: str):
    """Return a compact real-world graph run config for contract tests."""
    cfg = HP()
    cfg.set("num_features", 3)
    cfg.set("num_train_samples", 4)
    cfg.set("num_val_samples", 2)
    cfg.set("num_test_samples", 2)
    cfg.set("batch_size", 2)
    cfg.set("num_scenarios", 2)
    cfg.set("budget", 1)
    return build_run_config(cfg, load_real_world_graph=graph_path)


def _real_graph_path(project_root, filename: str) -> str:
    """Return one real-world graph data path as a string."""
    graph_path = project_root / "real_world_spni_data" / filename
    if not graph_path.exists():
        pytest.skip(f"Local real-world graph fixture is missing: {filename}.")

    return str(graph_path)


####################################
### test real-world graph import ###
####################################


def test_spni_real_world_graph_import_preserves_my_graph_topology(
    project_root,
):
    """Verify that the toy real-world CSV imports as the active topology."""
    # Arrange a topology-only real graph run config.
    graph_path = _real_graph_path(project_root, "my_graph.csv")
    run_cfg = _run_cfg_for_graph(graph_path)

    # Act by importing the graph through the SPNI build stage.
    graph = build_module.build_graph(run_cfg)
    solution, objective = graph.solve()

    # Assert that the imported topology, terminals, and solve are coherent.
    assert len(graph.arcs) == 25, \
        "my_graph.csv should import exactly 25 directed arcs."
    assert len(graph.vertices) == 19, \
        "my_graph.csv should import the 19 nodes implied by its endpoints."
    assert graph.arcs[:5] == [(0, 1), (1, 4), (4, 5), (0, 2), (0, 3)], \
        "The importer should preserve CSV arc ordering after de-duplication."
    assert graph.source == 0, \
        "The zero-indexed toy graph should use node 0 as the source."
    assert graph.target == 18, \
        "The zero-indexed toy graph should use node 18 as the target."
    assert objective == pytest.approx(5.0), \
        "Unit topology costs should make the shortest path five edges long."
    assert float(np.sum(solution)) == pytest.approx(5.0), \
        "The returned one-hot path should contain five selected arcs."
    pass


def test_spni_real_world_graph_import_uses_topology_only_for_county_csv(
    project_root,
):
    """Verify that non-topology CSV columns do not become base costs."""
    # Arrange the county graph, whose CSV contains Weight, Dist, Cost, IntProb.
    graph_path = _real_graph_path(project_root, "county_level_arcs.csv")
    run_cfg = _run_cfg_for_graph(graph_path)

    # Act by importing the graph and manually selecting valid terminals.
    graph = build_module.build_graph(run_cfg)
    graph.setObj(graph.cost, source=1, target=14)
    solution, objective = graph.solve()

    # Assert that only From/To define the graph in the current contract.
    assert len(graph.arcs) == 20, \
        "county_level_arcs.csv should import exactly 20 directed arcs."
    assert graph.arcs[:3] == [(1, 2), (1, 3), (1, 4)], \
        "The county importer should preserve the CSV arc ordering."
    assert np.allclose(graph.cost, np.ones(len(graph.arcs), dtype=float)), \
        "Topology-only import should leave base edge costs at one."
    assert objective == pytest.approx(6.0), \
        "County source 1 to target 14 should solve on unit topology costs."
    assert float(np.sum(solution)) == pytest.approx(6.0), \
        "The county shortest path should select six directed arcs."
    pass


def test_spni_real_world_graph_import_applies_configured_terminals(
    project_root,
):
    """Verify that real-world source and target config reaches the graph."""
    # Arrange the county graph with terminals that avoid the default node 0.
    graph_path = _real_graph_path(project_root, "county_level_arcs.csv")
    run_cfg = build_run_config(
        HP(),
        load_real_world_graph=graph_path,
        source_node=1,
        target_node=14,
    )

    # Act by importing the graph through the SPNI build stage.
    graph = build_module.build_graph(run_cfg)
    solution, objective = graph.solve()

    # Assert the configured terminals are applied before solver wrapping.
    assert graph.source == 1, \
        "build_graph should apply the configured real-world source node."
    assert graph.target == 14, \
        "build_graph should apply the configured real-world target node."
    assert objective == pytest.approx(6.0), \
        "Configured county terminals should produce a valid shortest path."
    assert float(np.sum(solution)) == pytest.approx(6.0), \
        "The configured county path should contain six selected arcs."
    pass


def test_spni_real_world_graph_import_town_csv_has_directed_terminal_pair(
    project_root,
):
    """Verify that directed real data needs a reachable terminal pair."""
    # Arrange a town graph whose default min/max terminals are not reachable.
    graph_path = _real_graph_path(project_root, "town_level_arcs.csv")
    run_cfg = _run_cfg_for_graph(graph_path)

    # Act by selecting a known directed terminal pair in the imported graph.
    graph = build_module.build_graph(run_cfg)
    graph.setObj(graph.cost, source=3, target=39)
    solution, objective = graph.solve()

    # Assert that the topology is usable once terminals match directed reach.
    assert len(graph.arcs) == 198, \
        "town_level_arcs.csv should import exactly 198 directed arcs."
    assert graph.source == 3, \
        "Graph.setObj should update the source for directed terminal tests."
    assert graph.target == 39, \
        "Graph.setObj should update the target for directed terminal tests."
    assert objective == pytest.approx(5.0), \
        "The selected town terminal pair should have a five-edge path."
    assert float(np.sum(solution)) == pytest.approx(5.0), \
        "The town shortest path should contain five selected arcs."
    pass


def test_spni_real_world_graph_import_anaheim_tntp_has_expected_topology(
    project_root,
):
    """Verify that the Anaheim TNTP network imports as real topology."""
    # Arrange the Anaheim transportation network with reachable terminals.
    graph_path = _real_graph_path(
        project_root,
        "transportation_networks/Anaheim_net.tntp",
    )
    run_cfg = build_run_config(
        HP(),
        load_real_world_graph=graph_path,
        source_node=1,
        target_node=416,
    )

    # Act by importing and solving on topology-only unit costs.
    graph = build_module.build_graph(run_cfg)
    solution, objective = graph.solve()

    # Assert the TNTP topology is preserved and usable by the SPNI graph.
    assert len(graph.vertices) == 416, \
        "Anaheim_net.tntp should import exactly 416 nodes."
    assert len(graph.arcs) == 914, \
        "Anaheim_net.tntp should import exactly 914 directed arcs."
    assert graph.arcs[:3] == [(1, 117), (2, 87), (3, 74)], \
        "Anaheim_net.tntp should preserve first-seen TNTP link ordering."
    assert np.allclose(graph.cost, np.ones(len(graph.arcs), dtype=float)), \
        "Anaheim TNTP import should use topology-only unit costs by default."
    assert graph.source == 1, \
        "build_graph should apply the Anaheim source node."
    assert graph.target == 416, \
        "build_graph should apply the Anaheim target node."
    assert objective == pytest.approx(16.0), \
        "Anaheim source 1 to target 416 should have a 16-edge unit path."
    assert float(np.sum(solution)) == pytest.approx(16.0), \
        "The Anaheim shortest path should select 16 directed arcs."
    pass


##############################################
### test real topology synthetic data width ###
##############################################


def test_spni_real_world_graph_synthetic_data_is_arc_aligned(
    monkeypatch,
    project_root,
):
    """Verify synthetic costs are generated against real topology width."""
    # Arrange a real graph and an opt-model stub exposing its arc count.
    graph_path = _real_graph_path(project_root, "my_graph.csv")
    run_cfg = _run_cfg_for_graph(graph_path)
    graph = build_module.build_graph(run_cfg)
    opt_model = SimpleNamespace(num_cost=len(graph.arcs), label="opt-model")
    graph_bundle = GraphBundle(
        graph=graph,
        opt_model=opt_model,
        graph_kind="real_world",
        graph_source=graph_path,
    )
    calls: list[dict] = []

    def _fake_gen_syn_data(cfg, opt_model=None, seed=None):
        calls.append(
            {
                "cfg": cfg,
                "opt_model": opt_model,
                "seed": seed,
            }
        )
        n_samples = (
            cfg.get("num_train_samples")
            + cfg.get("num_val_samples")
            + cfg.get("num_test_samples")
        )
        features = np.arange(
            n_samples * cfg.get("num_features"),
            dtype=float,
        ).reshape(n_samples, cfg.get("num_features"))
        costs = np.arange(
            n_samples * opt_model.num_cost,
            dtype=float,
        ).reshape(n_samples, opt_model.num_cost) + 1.0
        return features, costs

    monkeypatch.setattr(data_module, "gen_syn_data", _fake_gen_syn_data)

    # Act by generating base data through the SPNI data stage.
    base_data = data_module.generate_base_data(run_cfg, graph_bundle)

    # Assert that the real graph's arc count defines synthetic cost width.
    assert len(calls) == 1, \
        "Base data generation should call gen_syn_data exactly once."
    assert calls[0]["opt_model"] is opt_model, \
        "Base data generation should pass the real graph opt model through."
    assert calls[0]["seed"] is None, \
        "Base data generation should use the config random seed by default."
    assert base_data.costs.shape == (8, len(graph.arcs)), \
        "Synthetic costs should have one column per imported real arc."
    assert base_data.features.shape == (8, run_cfg.num_features), \
        "Synthetic features should preserve the configured feature width."
    assert base_data.diagnostics["num_costs"] == len(graph.arcs), \
        "Base-data diagnostics should record the real graph cost dimension."
    pass
