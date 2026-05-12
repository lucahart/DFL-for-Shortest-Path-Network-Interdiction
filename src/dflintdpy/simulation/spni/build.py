"""Graph and optimization-model construction for SPNI runs.

This module owns the first concrete stage of the SPNI pipeline: producing the
graph object and its shortest-path optimization model. It stays intentionally
small so graph construction can be tested independently from the rest of the
simulation flow.

Non-goals:
- no dataset generation
- no predictor training
- no evaluation or summary logic
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
    """Return the grid class used for synthetic SPNI graphs.

    The import stays local so importing the orchestration layer does not
    eagerly import solver-heavy graph modules.
    """

    from dflintdpy.models.grid import Grid

    return Grid


def _get_shortest_path_model_cls():
    """Return the shortest-path solver class used by the build stage.

    The local import keeps the pipeline module light to import in tests.
    """

    from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb

    return ShortestPathGrb


def _load_sourceless_real_world_module():
    """Load the legacy real-world graph helper from bytecode when needed.

    The repository currently ships the helper only as compiled bytecode under
    `utils/__pycache__`. This fallback keeps the new orchestration layer aligned
    with that existing import path without reconstructing the helper here.
    """

    # The repository currently carries this helper only as compiled bytecode,
    # so the build stage has to resolve it lazily at runtime.
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
    """Return the graph-file helper used by real-world SPNI runs.

    The orchestration layer delegates graph-file parsing to the real-world
    helper module and keeps the import path explicit here. The source helper
    exposes a format-dispatching loader; the legacy bytecode fallback only
    exposes ``csv_to_graph`` and remains supported for older local checkouts.
    """

    module_name = "dflintdpy.utils.real_world_spni_data_handling"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        module = _load_sourceless_real_world_module()

    loader = getattr(module, "real_world_graph_to_graph", None)
    if not callable(loader):
        loader = getattr(module, "csv_to_graph", None)
    if not callable(loader):
        raise AttributeError(
            "The real-world SPNI graph helper does not expose a graph loader."
        )
    return loader


def _safe_len(value: Any) -> int | None:
    """Return ``len(value)`` as an int when available, else ``None``.

    Diagnostics should never fail just because a graph object does not expose
    a normal container interface.
    """

    try:
        return int(len(value))
    except TypeError:
        return None


def build_graph(run_cfg: SPNIRunConfig):
    """Construct the graph used by one SPNI run.

    Behavior:
    - create a synthetic grid when no real-world graph path is given
    - otherwise delegate to the real-world graph file importer
    - return the graph instance only, without wrapping it in an opt model
    """

    # Real-world runs defer to the graph file loader; synthetic runs build the
    # in-memory grid directly from the normalized config.
    if run_cfg.load_real_world_graph is not None:
        load_graph = _resolve_real_world_graph_loader()
        graph = load_graph(run_cfg.load_real_world_graph)
        if run_cfg.source_node is not None or run_cfg.target_node is not None:
            graph.setObj(
                graph.cost,
                source=run_cfg.source_node,
                target=run_cfg.target_node,
            )
        return graph

    grid_cls = _get_grid_cls()
    graph = grid_cls(*run_cfg.grid_size)
    if run_cfg.source_node is not None or run_cfg.target_node is not None:
        graph.setObj(
            graph.cost,
            source=run_cfg.source_node,
            target=run_cfg.target_node,
        )
    return graph


def build_opt_model(run_cfg: SPNIRunConfig, graph):
    """Construct the shortest-path optimization model for one graph.

    The run config is accepted for API consistency with the rest of the stage
    functions, even though model construction currently depends only on the
    graph object itself.
    """

    # The model currently depends only on the graph, but keeping the normalized
    # config in the signature makes the stage API consistent across modules.
    del run_cfg
    shortest_path_model_cls = _get_shortest_path_model_cls()
    return shortest_path_model_cls(graph)


def build_problem_bundle(run_cfg: SPNIRunConfig) -> GraphBundle:
    """Build and return the graph-facing bundle for one run.

    This keeps graph creation separately testable from optimization-model
    creation while still returning one typed stage artifact for the pipeline.
    """

    # Build the graph first, then immediately wrap it in the solver model that
    # later stages expect to receive.
    graph = build_graph(run_cfg)
    opt_model = build_opt_model(run_cfg, graph)

    graph_source = run_cfg.load_real_world_graph
    graph_kind = "real_world" if graph_source is not None else "synthetic"
    # These diagnostics are intentionally lightweight and JSON-friendly so they
    # can be surfaced in tests and logs without serializing the graph itself.
    diagnostics = {
        "graph_class": type(graph).__name__,
        "opt_model_class": type(opt_model).__name__,
        "num_vertices": _safe_len(getattr(graph, "vertices", None)),
        "num_arcs": _safe_len(getattr(graph, "arcs", None)),
        "source_node": getattr(graph, "source", None),
        "target_node": getattr(graph, "target", None),
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
