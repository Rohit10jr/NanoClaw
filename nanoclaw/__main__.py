"""Command-line entry point. Grows into a terminal chat in layer 10."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from nanoclaw import __version__
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.providers import LLMProvider, provider_from_env
from nanoclaw.utils.templates import sync_workspace_templates

# nanobot's default is ~/.nanobot/workspace.
DEFAULT_WORKSPACE = Path("~/.nanoclaw/workspace")
CHANNEL = "cli"
# nanobot's terminal chat uses the same key.
DEFAULT_SESSION = "cli:direct"


def _prepare_workspace(path: Path) -> Path:
    """Resolve the workspace and create any missing SOUL.md / USER.md / AGENTS.md."""
    workspace = path.expanduser().resolve()
    for name in sync_workspace_templates(workspace):
        print(f"[created {workspace / name}]", file=sys.stderr)
    return workspace


async def _ask(provider: LLMProvider, prompt: str, workspace: Path, session_key: str, new: bool) -> int:
    loop = AgentLoop(provider, workspace)
    if new:
        loop.new_session(session_key)
    result = await loop.process_direct(prompt, session_key=session_key, channel=CHANNEL)
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
    ask = commands.add_parser("ask", help="send a message; the conversation continues across calls")
    ask.add_argument("prompt")
    ask.add_argument(
        "-s", "--session", default=DEFAULT_SESSION, help=f"conversation to continue (default: {DEFAULT_SESSION})"
    )
    ask.add_argument("--new", action="store_true", help="forget the conversation and start over")
    ask.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)
    show = commands.add_parser("prompt", help="print the system prompt the model receives (no API key needed)")
    show.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)

    args = parser.parse_args(argv)
    if args.command == "ask":
        workspace = _prepare_workspace(args.workspace)
        return asyncio.run(
            _ask(provider or provider_from_env(), args.prompt, workspace, args.session, args.new)
        )
    if args.command == "prompt":
        workspace = _prepare_workspace(args.workspace)
        print(ContextBuilder(workspace).build_system_prompt(channel=CHANNEL))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
