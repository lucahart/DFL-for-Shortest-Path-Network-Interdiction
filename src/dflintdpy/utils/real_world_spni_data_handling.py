"""Real-world graph import helpers for SPNI simulations.

The SPNI experiments use real graph topology with synthetically generated
cost data. These helpers therefore default to topology-only imports: the edge
set comes from the input file, while every imported arc starts with unit cost.
Optional cost-column loading exists for notebooks and inspection workflows.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from dflintdpy.models.graph import Graph


_TNTP_DEFAULT_COLUMNS = [
    "init_node",
    "term_node",
    "capacity",
    "length",
    "free_flow_time",
    "b",
    "power",
    "speed",
    "toll",
    "link_type",
]


def _normalize_column_name(name: str) -> str:
    """Return a normalized column identifier for loose file headers."""

    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def _coerce_node(value: str, column: str, path: Path) -> int:
    """Coerce one node value to an integer with a file-aware error."""

    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not parse {column}={value!r} as an integer node in {path}."
        ) from exc


def _coerce_float(value: str, column: str, path: Path) -> float:
    """Coerce one numeric value to float with a file-aware error."""

    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not parse {column}={value!r} as a numeric value in {path}."
        ) from exc


def _find_column(
    fieldnames: Iterable[str],
    candidates: Iterable[str],
    path: Path,
    label: str,
) -> str:
    """Find a column using normalized candidate names."""

    by_normalized = {
        _normalize_column_name(name): name
        for name in fieldnames
    }
    for candidate in candidates:
        normalized = _normalize_column_name(candidate)
        if normalized in by_normalized:
            return by_normalized[normalized]
    raise ValueError(
        f"Could not find {label} column in {path}. "
        f"Available columns: {sorted(by_normalized)}."
    )


def _dedupe_arcs(
    arcs: list[tuple[int, int]],
    costs: list[float] | None,
) -> tuple[list[tuple[int, int]], list[float] | None]:
    """Remove duplicate arcs while preserving first-seen order."""

    seen = set()
    deduped_arcs: list[tuple[int, int]] = []
    deduped_costs: list[float] | None = [] if costs is not None else None

    for index, arc in enumerate(arcs):
        if arc in seen:
            continue
        seen.add(arc)
        deduped_arcs.append(arc)
        if deduped_costs is not None and costs is not None:
            deduped_costs.append(costs[index])

    return deduped_arcs, deduped_costs


def _build_graph(
    arcs: list[tuple[int, int]],
    vertices: Iterable[int],
    costs: list[float] | None,
    *,
    metadata: dict,
) -> Graph:
    """Build a Graph from parsed real-world topology pieces."""

    if not arcs:
        raise ValueError("Real-world graph import did not find any arcs.")

    deduped_arcs, deduped_costs = _dedupe_arcs(arcs, costs)
    vertex_array = np.array(sorted(set(vertices)), dtype=int)
    cost_array = (
        np.array(deduped_costs, dtype=float)
        if deduped_costs is not None
        else np.ones(len(deduped_arcs), dtype=float)
    )

    graph = Graph(
        arcs=deduped_arcs,
        vertices=vertex_array,
        cost=cost_array,
    )
    graph.real_world_metadata = metadata
    return graph


def csv_to_graph(
    path: str | Path,
    *,
    topology_only: bool = True,
    cost_column: str | None = None,
) -> Graph:
    """Import a real-world arc CSV as a :class:`Graph`.

    CSV files must expose tail and head columns named like ``From``/``To``.
    By default only those topology columns are used and every arc receives unit
    cost. Set ``topology_only=False`` and provide ``cost_column`` to import a
    specific numeric edge-cost column for inspection workflows.
    """

    csv_path = Path(path)
    arcs: list[tuple[int, int]] = []
    vertices: set[int] = set()
    costs: list[float] | None = [] if not topology_only else None

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        from_col = _find_column(
            fieldnames,
            ["from", "tail", "init_node", "source"],
            csv_path,
            "tail-node",
        )
        to_col = _find_column(
            fieldnames,
            ["to", "head", "term_node", "target"],
            csv_path,
            "head-node",
        )
        resolved_cost_col = None
        if not topology_only:
            if cost_column is None:
                raise ValueError(
                    "cost_column must be provided when topology_only=False "
                    "for CSV graph imports."
                )
            resolved_cost_col = _find_column(
                fieldnames,
                [cost_column],
                csv_path,
                "cost",
            )

        for row in reader:
            tail_value = row.get(from_col, "")
            head_value = row.get(to_col, "")
            if str(tail_value).strip() == "" and str(head_value).strip() == "":
                continue

            tail = _coerce_node(tail_value, from_col, csv_path)
            head = _coerce_node(head_value, to_col, csv_path)
            arcs.append((tail, head))
            vertices.update((tail, head))
            if costs is not None and resolved_cost_col is not None:
                costs.append(
                    _coerce_float(
                        row.get(resolved_cost_col, ""),
                        resolved_cost_col,
                        csv_path,
                    )
                )

    return _build_graph(
        arcs,
        vertices,
        costs,
        metadata={
            "format": "csv",
            "path": str(csv_path),
            "topology_only": topology_only,
            "cost_column": cost_column,
        },
    )


def _parse_tntp_metadata_line(line: str) -> tuple[str, str] | None:
    """Parse one TNTP metadata line when present."""

    match = re.match(r"<([^>]+)>\s*(.*)", line.strip())
    if match is None:
        return None
    key = _normalize_column_name(match.group(1))
    value = match.group(2).strip()
    return key, value


def _parse_tntp_fields(line: str) -> list[str]:
    """Parse whitespace-delimited TNTP fields before the semicolon."""

    content = line.split(";", 1)[0].strip()
    if content.startswith("~"):
        content = content[1:].strip()
    return content.split()


def _metadata_vertices(
    metadata: dict[str, str],
    observed_vertices: set[int],
) -> set[int]:
    """Return vertices implied by TNTP metadata and observed endpoints."""

    vertices = set(observed_vertices)
    raw_count = metadata.get("number_of_nodes")
    if raw_count is None:
        return vertices

    node_count = int(float(raw_count))
    if observed_vertices and min(observed_vertices) == 0:
        vertices.update(range(node_count))
    else:
        vertices.update(range(1, node_count + 1))
    return vertices


def tntp_to_graph(
    path: str | Path,
    *,
    topology_only: bool = True,
    cost_column: str | None = None,
) -> Graph:
    """Import a TNTP transportation network file as a :class:`Graph`.

    The parser reads rows after ``<END OF METADATA>`` and expects TNTP columns
    such as ``init_node`` and ``term_node``. By default, edge attributes from
    the file are ignored and unit costs are assigned. Set
    ``topology_only=False`` to load one numeric TNTP column as the Graph cost
    vector; when no ``cost_column`` is supplied, ``free_flow_time`` is used.
    """

    tntp_path = Path(path)
    metadata: dict[str, str] = {}
    columns: list[str] | None = None
    arcs: list[tuple[int, int]] = []
    observed_vertices: set[int] = set()
    costs: list[float] | None = [] if not topology_only else None
    in_data = False

    for raw_line in tntp_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        metadata_entry = _parse_tntp_metadata_line(line)
        if metadata_entry is not None:
            key, value = metadata_entry
            metadata[key] = value
            if key == "end_of_metadata":
                in_data = True
            continue
        if not in_data:
            continue

        if line.startswith("~"):
            columns = [
                _normalize_column_name(field)
                for field in _parse_tntp_fields(line)
            ]
            continue

        row_values = _parse_tntp_fields(line)
        if not row_values:
            continue
        if columns is None:
            columns = list(_TNTP_DEFAULT_COLUMNS)
        if len(row_values) < len(columns):
            raise ValueError(
                f"TNTP row in {tntp_path} has {len(row_values)} fields, "
                f"expected at least {len(columns)}."
            )

        row = dict(zip(columns, row_values))
        tail = _coerce_node(row["init_node"], "init_node", tntp_path)
        head = _coerce_node(row["term_node"], "term_node", tntp_path)
        arcs.append((tail, head))
        observed_vertices.update((tail, head))

        if costs is not None:
            resolved_cost_col = _normalize_column_name(
                cost_column or "free_flow_time"
            )
            if resolved_cost_col not in row:
                raise ValueError(
                    f"Could not find cost column {cost_column!r} in {tntp_path}."
                )
            costs.append(
                _coerce_float(row[resolved_cost_col], resolved_cost_col, tntp_path)
            )

    vertices = _metadata_vertices(metadata, observed_vertices)
    return _build_graph(
        arcs,
        vertices,
        costs,
        metadata={
            "format": "tntp",
            "path": str(tntp_path),
            "topology_only": topology_only,
            "cost_column": cost_column,
            "metadata": metadata,
            "columns": columns or list(_TNTP_DEFAULT_COLUMNS),
        },
    )


def real_world_graph_to_graph(
    path: str | Path,
    *,
    topology_only: bool = True,
    cost_column: str | None = None,
    format: str | None = None,
) -> Graph:
    """Dispatch a real-world graph file to the appropriate parser."""

    graph_path = Path(path)
    normalized_format = (
        _normalize_column_name(format)
        if format is not None
        else graph_path.suffix.lower().lstrip(".")
    )

    if normalized_format == "csv":
        return csv_to_graph(
            graph_path,
            topology_only=topology_only,
            cost_column=cost_column,
        )
    if normalized_format == "tntp":
        return tntp_to_graph(
            graph_path,
            topology_only=topology_only,
            cost_column=cost_column,
        )
    raise ValueError(
        f"Unsupported real-world graph format {normalized_format!r} for "
        f"{graph_path}. Supported formats are: csv, tntp."
    )
