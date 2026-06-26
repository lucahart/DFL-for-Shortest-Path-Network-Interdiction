from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dflintdpy.simulation.spni.config import SPNIRunConfig

import dflintdpy.simulation.spni.build as build_module


################
### Fixtures ###
################


@pytest.fixture
def run_cfg() -> SPNIRunConfig:
    """Return a compact config for build-stage contract tests."""
    return SPNIRunConfig(
        base_cfg=SimpleNamespace(label="base"),
        grid_size=(3, 4),
        num_features=5,
        num_train_samples=10,
        num_val_samples=2,
        num_test_samples=3,
        batch_size=4,
        budget=1,
        num_scenarios=6,
        deg=2,
        noise_width=0.25,
        benders_max_count=7,
        benders_eps=1e-4,
        lsd=1e-5,
        seed=11,
        random_seed=13,
        intd_seed=17,
        loader_seed=19,
        pred_model="linear",
        pfl_epochs=8,
        dfl_epochs=9,
        pfl_lr=1e-3,
        dfl_lr=2e-3,
        compute_asym_intd=True,
        compute_wrong_asym_intd=False,
        load_real_world_graph=None,
        metadata={"source_type": "SimpleNamespace"},
    )


############################
### Helper functionality ###
############################


class _GridStub:
    """Minimal grid stub used to verify synthetic graph construction."""

    calls: list[tuple[int, int]] = []

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.calls.append((m, n))

    @classmethod
    def reset(cls) -> None:
        """Clear captured constructor calls."""
        cls.calls = []


class _OptModelStub:
    """Minimal solver stub used to verify shortest-path model wrapping."""

    calls: list[object] = []

    def __init__(self, graph):
        self.graph = graph
        type(self).calls.append(graph)

    @classmethod
    def reset(cls) -> None:
        """Clear captured constructor calls."""
        cls.calls = []


class _CsvGraphStub:
    """Minimal importer stub used to verify real-world delegation."""

    calls: list[str] = []

    @classmethod
    def reset(cls) -> None:
        """Clear captured import calls."""
        cls.calls = []

    def __call__(self, path: str):
        """Record the import path and return a graph-like stub."""
        type(self).calls.append(path)
        return SimpleNamespace(kind="real_world_graph", path=path)


######################################
### test build_graph(...) ###
######################################


def test_spni_build_build_graph_uses_grid_for_synthetic_runs(monkeypatch, run_cfg):
    """Verify that synthetic runs build a `Grid` from the configured size."""
    # Arrange a grid stub so the constructor call is observable.
    _GridStub.reset()
    monkeypatch.setattr(build_module, "_get_grid_cls", lambda: _GridStub)

    # Act by asking the build layer for a synthetic graph.
    graph = build_module.build_graph(run_cfg)

    # Assert that the graph was built from the configured dimensions.
    assert _GridStub.calls == [(3, 4)], \
        "build_graph should construct exactly one Grid for synthetic runs."
    assert isinstance(graph, _GridStub), \
        "build_graph should return the synthetic Grid instance."
    assert (graph.m, graph.n) == (3, 4), \
        "build_graph should pass the run config grid size into Grid."
    pass


def test_spni_build_build_graph_delegates_to_real_world_import(monkeypatch):
    """Verify that real-world runs call the CSV importer exactly once."""
    # Arrange a run config that selects the real-world graph path.
    run_cfg = SPNIRunConfig(
        base_cfg=SimpleNamespace(label="base"),
        grid_size=(3, 4),
        num_features=5,
        num_train_samples=10,
        num_val_samples=2,
        num_test_samples=3,
        batch_size=4,
        budget=1,
        num_scenarios=6,
        deg=2,
        noise_width=0.25,
        benders_max_count=7,
        benders_eps=1e-4,
        lsd=1e-5,
        seed=11,
        random_seed=13,
        intd_seed=17,
        loader_seed=19,
        pred_model="linear",
        pfl_epochs=8,
        dfl_epochs=9,
        pfl_lr=1e-3,
        dfl_lr=2e-3,
        compute_asym_intd=True,
        compute_wrong_asym_intd=False,
        load_real_world_graph="graphs/sample.csv",
        metadata={"source_type": "SimpleNamespace"},
    )
    importer = _CsvGraphStub()
    _CsvGraphStub.reset()
    monkeypatch.setattr(
        build_module,
        "_resolve_real_world_graph_loader",
        lambda: importer,
    )

    # Act by asking the build layer for a real-world graph.
    graph = build_module.build_graph(run_cfg)

    # Assert that the importer saw the requested CSV exactly once.
    assert _CsvGraphStub.calls == ["graphs/sample.csv"], \
        "build_graph should call the CSV importer once for real-world runs."
    assert graph.path == "graphs/sample.csv", \
        "build_graph should return the graph produced by the importer."
    assert graph.kind == "real_world_graph", \
        "build_graph should preserve the importer's graph-like payload."
    pass


def test_spni_build_build_graph_imports_tntp_real_world_graph(
        run_cfg: SPNIRunConfig,
        tmp_path: Path,
    ):
    """Verify that build_graph imports TNTP paths as real-world graphs."""
    # Arrange a compact TNTP file and a config that points to it.
    graph_path = tmp_path / "sample.tntp"
    graph_path.write_text(
        "<NUMBER OF NODES> 4\n"
        "<NUMBER OF LINKS> 3\n"
        "<END OF METADATA>\n"
        "~ init_node term_node capacity length free_flow_time b power speed "
        "toll link_type ;\n"
        "1 2 100 10 1.5 0.15 4 10 0 1 ;\n"
        "2 4 100 10 2.5 0.15 4 10 0 1 ;\n"
        "1 3 100 10 3.5 0.15 4 10 0 1 ;\n",
        encoding="utf-8",
    )
    real_world_cfg = replace(
        run_cfg,
        load_real_world_graph=str(graph_path),
        source_node=1,
        target_node=4,
    )

    # Act by importing the graph through the build stage.
    graph = build_module.build_graph(real_world_cfg)
    solution, objective = graph.solve()

    # Assert that build_graph used the TNTP dispatcher and applied terminals.
    assert graph.real_world_metadata["format"] == "tntp", \
        "build_graph should dispatch .tntp files to the TNTP importer."
    assert graph.arcs == [(1, 2), (2, 4), (1, 3)], \
        "build_graph should preserve TNTP directed arc ordering."
    assert np.allclose(graph.cost, np.ones(3, dtype=float)), \
        "build_graph should import TNTP topology with unit costs by default."
    assert graph.source == 1, \
        "build_graph should apply the configured TNTP source node."
    assert graph.target == 4, \
        "build_graph should apply the configured TNTP target node."
    assert objective == pytest.approx(2.0), \
        "The TNTP graph should solve on topology-only unit costs."
    assert float(np.sum(solution)) == pytest.approx(2.0), \
        "The TNTP shortest path should select two directed arcs."
    pass


########################################
### test build_opt_model(...) ###
########################################


def test_spni_build_build_opt_model_wraps_graph_once(monkeypatch, run_cfg):
    """Verify that `build_opt_model` wraps the supplied graph exactly once."""
    # Arrange a solver stub that records its graph input.
    _OptModelStub.reset()
    monkeypatch.setattr(
        build_module,
        "_get_shortest_path_model_cls",
        lambda: _OptModelStub,
    )
    graph = SimpleNamespace(name="graph")

    # Act by building the shortest-path optimization model.
    opt_model = build_module.build_opt_model(run_cfg, graph)

    # Assert that the solver saw the graph once and returned the stub model.
    assert _OptModelStub.calls == [graph], \
        "build_opt_model should instantiate ShortestPathGrb exactly once."
    assert isinstance(opt_model, _OptModelStub), \
        "build_opt_model should return the wrapped solver instance."
    assert opt_model.graph is graph, \
        "build_opt_model should preserve the original graph object."
    pass


############################################
### test build_problem_bundle(...) ###
############################################


def test_spni_build_build_problem_bundle_records_graph_metadata(monkeypatch):
    """Verify that `build_problem_bundle` records graph provenance metadata."""
    # Arrange a real-world run config and stub stage builders.
    run_cfg = SPNIRunConfig(
        base_cfg=SimpleNamespace(label="base"),
        grid_size=(3, 4),
        num_features=5,
        num_train_samples=10,
        num_val_samples=2,
        num_test_samples=3,
        batch_size=4,
        budget=1,
        num_scenarios=6,
        deg=2,
        noise_width=0.25,
        benders_max_count=7,
        benders_eps=1e-4,
        lsd=1e-5,
        seed=11,
        random_seed=13,
        intd_seed=17,
        loader_seed=19,
        pred_model="linear",
        pfl_epochs=8,
        dfl_epochs=9,
        pfl_lr=1e-3,
        dfl_lr=2e-3,
        compute_asym_intd=True,
        compute_wrong_asym_intd=False,
        load_real_world_graph="graphs/sample.csv",
        metadata={"source_type": "SimpleNamespace"},
    )
    graph = SimpleNamespace(name="graph")
    opt_model = SimpleNamespace(name="model")
    monkeypatch.setattr(build_module, "build_graph", lambda cfg: graph)
    monkeypatch.setattr(build_module, "build_opt_model", lambda cfg, g: opt_model)

    # Act by asking for the full graph-facing bundle.
    bundle = build_module.build_problem_bundle(run_cfg)

    # Assert that the bundle preserves graph/model objects and provenance.
    assert bundle.graph is graph, \
        "build_problem_bundle should keep the constructed graph."
    assert bundle.opt_model is opt_model, \
        "build_problem_bundle should keep the constructed solver model."
    assert bundle.graph_source == "graphs/sample.csv", \
        "build_problem_bundle should store the real-world source path."
    assert bundle.graph_kind == "real_world", \
        "build_problem_bundle should label the graph as real-world."
    assert bundle.diagnostics["graph_class"] == "SimpleNamespace", \
        "build_problem_bundle should record the graph class name."
    assert bundle.diagnostics["opt_model_class"] == "SimpleNamespace", \
        "build_problem_bundle should record the solver class name."
    assert bundle.diagnostics["num_vertices"] is None, \
        "build_problem_bundle should allow missing vertex metadata."
    assert bundle.diagnostics["num_arcs"] is None, \
        "build_problem_bundle should allow missing arc metadata."
    assert bundle.diagnostics["graph_path"] == "graphs/sample.csv", \
        "build_problem_bundle should record the real-world graph path."
    pass


def test_spni_build_build_problem_bundle_records_synthetic_grid_metadata(
    monkeypatch,
    run_cfg,
):
    """Verify that synthetic bundles keep grid metadata in diagnostics."""
    # Arrange synthetic graph/model stubs with explicit graph dimensions.
    graph = SimpleNamespace(
        name="grid",
        vertices=np.arange(12),
        arcs=[(0, 1), (1, 2), (2, 3)],
    )
    opt_model = SimpleNamespace(name="model")
    monkeypatch.setattr(build_module, "build_graph", lambda cfg: graph)
    monkeypatch.setattr(build_module, "build_opt_model", lambda cfg, g: opt_model)

    # Act by asking for the synthetic graph-facing bundle.
    bundle = build_module.build_problem_bundle(run_cfg)

    # Assert that synthetic metadata is exposed for debugging.
    assert bundle.graph_kind == "synthetic", \
        "build_problem_bundle should label synthetic graphs correctly."
    assert bundle.graph_source is None, \
        "build_problem_bundle should leave graph_source unset for grids."
    assert bundle.diagnostics["num_vertices"] == 12, \
        "build_problem_bundle should count synthetic graph vertices."
    assert bundle.diagnostics["num_arcs"] == 3, \
        "build_problem_bundle should count synthetic graph arcs."
    assert bundle.diagnostics["grid_size"] == [3, 4], \
        "build_problem_bundle should record the configured grid size."
    pass
