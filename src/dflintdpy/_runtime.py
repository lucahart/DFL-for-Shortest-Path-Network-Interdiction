"""Runtime helpers for terminal entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def _ensure_directory(path: Path) -> Path | None:
    """Create ``path`` when possible and return it if usable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


def configure_terminal_cache_environment() -> None:
    """Set writable cache defaults for command-line runs.

    Matplotlib and some ML/data libraries write cache files while importing or
    plotting. Terminal entrypoints should be usable in restricted environments
    without requiring callers to spell out ``MPLCONFIGDIR`` or
    ``XDG_CACHE_HOME`` on every command, but explicit user-provided values must
    still win.
    """
    cache_root = Path(tempfile.gettempdir()) / "dflintdpy"

    if not os.environ.get("XDG_CACHE_HOME"):
        xdg_cache_home = (
            _ensure_directory(cache_root / "xdg-cache")
            or Path(tempfile.mkdtemp(prefix="dflintdpy-cache-"))
        )
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache_home)

    if not os.environ.get("MPLCONFIGDIR"):
        mpl_config_dir = _ensure_directory(
            Path(os.environ["XDG_CACHE_HOME"]) / "matplotlib"
        )
        if mpl_config_dir is None:
            mpl_config_dir = (
                _ensure_directory(cache_root / "matplotlib")
                or Path(tempfile.mkdtemp(prefix="dflintdpy-mpl-"))
            )
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
