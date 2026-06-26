"""Top-level ``dflintd`` command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from dflintdpy._runtime import configure_terminal_cache_environment


def _build_parser() -> argparse.ArgumentParser:
    """Return the top-level DFLIntdPy CLI parser."""
    parser = argparse.ArgumentParser(
        prog="dflintd",
        description="DFLIntdPy command-line tools.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["spni"],
        help="Tool to run. Use 'spni' for SPNI simulations and replots.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected tool.",
    )
    return parser


def _run_spni(argv: Sequence[str] | None = None) -> int:
    """Run the SPNI CLI command."""
    from dflintdpy.cli.spni import main as spni_main

    return spni_main(argv, prog="dflintd spni")


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one DFLIntdPy command-line invocation."""
    configure_terminal_cache_environment()
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 0

    if parsed.command == "spni":
        return _run_spni(parsed.args)

    parser.error(f"Unsupported command: {parsed.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
