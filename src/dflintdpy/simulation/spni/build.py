"""Graph and optimization-model construction for SPNI runs.

This module should own only the object-construction part of the simulation
workflow. It must not generate data, train predictors, or evaluate outcomes.
"""

from __future__ import annotations

from importlib import import_module
from importlib.machinery import SourcelessFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Any, Callable

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import GraphBundle


def _get_grid_cls():
    """Return the grid class used for synthetic SPNI graphs."""

    from dflintdpy.models.grid import Grid

    return Grid


def _get_shortest_path_model_cls():
    """Return the shortest-path solver class used by the build stage."""

    from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb

    return ShortestPathGrb


def _load_sourceless_real_world_module():
    """Load the legacy real-world graph helper from bytecode when needed.

    The repository currently ships the helper only as compiled bytecode under
    `utils/__pycache__`. This fallback keeps the new orchestration layer aligned
    with that existing import path without reconstructing the helper here.
    """

    cache_dir = Path(__file__).resolve().parents[2] / "utils" / "__pycache__"
    candidates = sorted(cache_dir.glob("real_world_spni_data_handling*.pyc"))
    if not candidates:
        raise ModuleNotFoundError(
            "Could not locate the real-world SPNI graph helper module."
        )

    module_name = "dflintdpy.utils.real_world_spni_data_handling"
    loader = SourcelessFileLoader(module_name, str(candidates[0]))
    spec = spec_from_loader(module_name, loader)
    if spec is None:
        raise ImportError("Could not create a module spec for csv_to_graph.")

    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


def _resolve_real_world_graph_loader() -> Callable[[str], Any]:
    """Return the legacy CSV-to-graph helper used by real-world SPNI runs."""

    module_name = "dflintdpy.utils.real_world_spni_data_handling"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        module = _load_sourceless_real_world_module()

    loader = getattr(module, "csv_to_graph", None)
    if not callable(loader):
        raise AttributeError(
            "The real-world SPNI graph helper does not expose csv_to_graph."
        )
    return loader


def _safe_len(value: Any) -> int | None:
    """Return `len(value)` as an int when available, else `None`."""

    try:
        return int(len(value))
    except TypeError:
        return None


def build_graph(run_cfg: SPNIRunConfig):
    """Construct the graph used by one SPNI run.

    Behavior:
    - create a synthetic grid when no real-world graph path is given
    - otherwise delegate to the legacy CSV importer
    - return the graph instance only, without wrapping it in an opt model
    """

    if run_cfg.load_real_world_graph is not None:
        csv_to_graph = _resolve_real_world_graph_loader()
        return csv_to_graph(run_cfg.load_real_world_graph)

    grid_cls = _get_grid_cls()
    return grid_cls(*run_cfg.grid_size)


def build_opt_model(run_cfg: SPNIRunConfig, graph):
    """Construct the shortest-path optimization model for one graph.

    The run config is accepted for API consistency with the rest of the stage
    functions, even though model construction currently depends only on the
    graph object itself.
    """

    del run_cfg
    shortest_path_model_cls = _get_shortest_path_model_cls()
    return shortest_path_model_cls(graph)


def build_problem_bundle(run_cfg: SPNIRunConfig) -> GraphBundle:
    """Build and return the graph-facing bundle for one run.

    This keeps graph creation separately testable from optimization-model
    creation while still returning one typed stage artifact for the pipeline.
    """

    graph = build_graph(run_cfg)
    opt_model = build_opt_model(run_cfg, graph)

    graph_source = run_cfg.load_real_world_graph
    graph_kind = "real_world" if graph_source is not None else "synthetic"
    diagnostics = {
        "graph_class": type(graph).__name__,
        "opt_model_class": type(opt_model).__name__,
        "num_vertices": _safe_len(getattr(graph, "vertices", None)),
        "num_arcs": _safe_len(getattr(graph, "arcs", None)),
    }
    if graph_kind == "synthetic":
        diagnostics["grid_size"] = list(run_cfg.grid_size)
    else:
        diagnostics["graph_path"] = graph_source

    return GraphBundle(
        graph=graph,
        opt_model=opt_model,
        graph_kind=graph_kind,
        graph_source=graph_source,
        diagnostics=diagnostics,
    )
