"""Command-line entry point. Grows into a terminal chat in layer 10."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from nanoclaw import __version__
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import build_default_tools
from nanoclaw.providers import LLMProvider, provider_from_env

# nanobot's default is ~/.nanobot/workspace.
DEFAULT_WORKSPACE = Path("~/.nanoclaw/workspace")


async def _ask(provider: LLMProvider, prompt: str, workspace: Path) -> int:
    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": prompt}],
            tools=build_default_tools(workspace),
            provider=provider,
            workspace=workspace,
        )
    )
    if result.tools_used:
        print(f"[tools used: {', '.join(result.tools_used)}]", file=sys.stderr)
    if result.stop_reason == "error":
        print(result.final_content, file=sys.stderr)
        return 1
    print(result.final_content or "")
    return 0


def main(argv: list[str] | None = None, provider: LLMProvider | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanoclaw", description="A minimal personal AI agent.")
    parser.add_argument("-V", "--version", action="version", version=f"nanoclaw {__version__}")
    commands = parser.add_subparsers(dest="command")
    ask = commands.add_parser("ask", help="run one agent turn with file, shell and web tools")
    ask.add_argument("prompt")
    ask.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="directory the agent works in (default: ~/.nanoclaw/workspace)",
    )

    args = parser.parse_args(argv)
    if args.command == "ask":
        workspace = args.workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        return asyncio.run(_ask(provider or provider_from_env(), args.prompt, workspace))
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
