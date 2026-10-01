"""Command-line entry point. Grows into a terminal chat in layer 10."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from nanoclaw import __version__
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import build_default_tools
from nanoclaw.providers import LLMProvider, provider_from_env
from nanoclaw.runtime_context import current_time_block
from nanoclaw.utils.templates import sync_workspace_templates

# nanobot's default is ~/.nanobot/workspace.
DEFAULT_WORKSPACE = Path("~/.nanoclaw/workspace")
CHANNEL = "cli"


def _prepare_workspace(path: Path) -> Path:
    """Resolve the workspace and create any missing SOUL.md / USER.md / AGENTS.md."""
    workspace = path.expanduser().resolve()
    for name in sync_workspace_templates(workspace):
        print(f"[created {workspace / name}]", file=sys.stderr)
    return workspace


async def _ask(provider: LLMProvider, prompt: str, workspace: Path) -> int:
    messages = ContextBuilder(workspace).build_messages(
        [], prompt, channel=CHANNEL, runtime_context_blocks=[current_time_block()]
    )
    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=messages,
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

    workspace_help = "directory the agent works in (default: ~/.nanoclaw/workspace)"
    ask = commands.add_parser("ask", help="run one agent turn with file, shell and web tools")
    ask.add_argument("prompt")
    ask.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)
    show = commands.add_parser("prompt", help="print the system prompt the model receives (no API key needed)")
    show.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)

    args = parser.parse_args(argv)
    if args.command == "ask":
        workspace = _prepare_workspace(args.workspace)
        return asyncio.run(_ask(provider or provider_from_env(), args.prompt, workspace))
    if args.command == "prompt":
        workspace = _prepare_workspace(args.workspace)
        print(ContextBuilder(workspace).build_system_prompt(channel=CHANNEL))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
