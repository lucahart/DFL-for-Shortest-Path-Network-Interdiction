"""Console-script wrapper for the SPNI pipeline CLI."""

from __future__ import annotations

from collections.abc import Sequence

from dflintdpy._runtime import configure_terminal_cache_environment


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    """Run the SPNI pipeline CLI and return a console-script exit code."""
    configure_terminal_cache_environment()

    from dflintdpy.simulation.spni.pipeline import cli

    cli(argv, prog=prog)
    return 0
