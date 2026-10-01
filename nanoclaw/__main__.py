"""Command-line entry point. Grows into a terminal chat in layer 10."""

import sys

from nanoclaw import __version__


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args in (["--version"], ["-V"]):
        print(f"nanoclaw {__version__}")
        return 0
    print("nanoclaw: the agent is not wired up yet. Try --version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
