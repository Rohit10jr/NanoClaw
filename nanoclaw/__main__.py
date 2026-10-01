"""Command-line entry point. Grows into a terminal chat in layer 10."""

from __future__ import annotations

import argparse
import asyncio
import sys

from nanoclaw import __version__
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.providers import LLMProvider, provider_from_env


async def _ask(provider: LLMProvider, prompt: str) -> int:
    # No tools are registered until layer 3, but the full agent loop runs.
    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": prompt}],
            tools=ToolRegistry(),
            provider=provider,
        )
    )
    if result.stop_reason == "error":
        print(result.final_content, file=sys.stderr)
        return 1
    print(result.final_content or "")
    return 0


def main(argv: list[str] | None = None, provider: LLMProvider | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanoclaw", description="A minimal personal AI agent.")
    parser.add_argument("-V", "--version", action="version", version=f"nanoclaw {__version__}")
    commands = parser.add_subparsers(dest="command")
    ask = commands.add_parser("ask", help="run one agent turn (no tools until layer 3)")
    ask.add_argument("prompt")

    args = parser.parse_args(argv)
    if args.command == "ask":
        return asyncio.run(_ask(provider or provider_from_env(), args.prompt))
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
