"""Command-line entry point. Grows into a terminal chat in layer 10."""

from __future__ import annotations

import argparse
import asyncio
import sys

from nanoclaw import __version__
from nanoclaw.providers import LLMProvider, provider_from_env


async def _ask(provider: LLMProvider, prompt: str) -> int:
    response = await provider.safe_chat(messages=[{"role": "user", "content": prompt}])
    if response.is_error:
        print(response.content, file=sys.stderr)
        return 1
    print(response.content or "")
    return 0


def main(argv: list[str] | None = None, provider: LLMProvider | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanoclaw", description="A minimal personal AI agent.")
    parser.add_argument("-V", "--version", action="version", version=f"nanoclaw {__version__}")
    commands = parser.add_subparsers(dest="command")
    ask = commands.add_parser("ask", help="send one message to the model (no tools yet)")
    ask.add_argument("prompt")

    args = parser.parse_args(argv)
    if args.command == "ask":
        return asyncio.run(_ask(provider or provider_from_env(), args.prompt))
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
