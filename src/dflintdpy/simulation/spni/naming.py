"""Filename helpers for SPNI output artefacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read one config value from legacy or typed config objects."""
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def safe_filename_token(raw_value: Any, *, default: str = "graph") -> str:
    """Return a filesystem-safe token derived from one raw value."""
    token = Path(str(raw_value)).stem
    token = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").lower()
    return token or default


def real_world_graph_filename_marker(cfg: Any) -> str:
    """Return a filename marker for real-world graph runs, or an empty string."""
    graph_path = cfg_get(cfg, "load_real_world_graph", None)
    if graph_path is None:
        return ""
    graph_token = safe_filename_token(graph_path)
    return f"real_world_{graph_token}"


def real_world_graph_filename_suffix(cfg: Any) -> str:
    """Return the real-world graph filename marker with a leading underscore."""
    marker = real_world_graph_filename_marker(cfg)
    return f"_{marker}" if marker else ""
