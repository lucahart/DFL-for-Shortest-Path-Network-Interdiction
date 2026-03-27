"""Graph and optimization-model construction for SPNI runs.

This module should own only the object-construction part of the simulation
workflow. It must not generate data, train predictors, or evaluate outcomes.
"""

from __future__ import annotations

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import GraphBundle


def build_graph(run_cfg: SPNIRunConfig):
    """Construct the graph used by one SPNI run.

    Future implementation responsibilities:
    - create a synthetic grid when no real-world graph path is given
    - otherwise delegate to the real-world graph importer
    - return the graph instance only, without wrapping it in an opt model
    """

    raise NotImplementedError("Specification scaffold only.")


def build_opt_model(run_cfg: SPNIRunConfig, graph):
    """Construct the shortest-path optimization model for one graph.

    Future implementation responsibilities:
    - wrap the graph in `ShortestPathGrb`
    - return only the optimization model object
    - keep graph creation and model creation separately testable
    """

    raise NotImplementedError("Specification scaffold only.")


def build_problem_bundle(run_cfg: SPNIRunConfig) -> GraphBundle:
    """Build and return the graph-facing bundle for one run.

    Future implementation responsibilities:
    - call `build_graph`
    - call `build_opt_model`
    - attach graph provenance metadata for debugging
    """

    raise NotImplementedError("Specification scaffold only.")

