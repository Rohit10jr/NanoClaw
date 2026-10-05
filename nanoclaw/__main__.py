"""Command-line entry point. Grows into a terminal chat in layer 10."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from nanoclaw import __version__
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import DEFAULT_CONTEXT_WINDOW_TOKENS, AgentLoop, ContextWindowExceededError
from nanoclaw.providers import LLMProvider, provider_from_env
from nanoclaw.utils.templates import sync_workspace_templates

# nanobot's default is ~/.nanobot/workspace.
DEFAULT_WORKSPACE = Path("~/.nanoclaw/workspace")
CHANNEL = "cli"
# nanobot's terminal chat uses the same key.
DEFAULT_SESSION = "cli:direct"
# nanobot's out-of-the-box agent settings (config AgentDefaults). The provider's
# own defaults (0.7, 4096) only apply when no configuration is given.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 8192


def _prepare_workspace(path: Path) -> Path:
    """Resolve the workspace and create any missing SOUL.md / USER.md / AGENTS.md."""
    workspace = path.expanduser().resolve()
    for name in sync_workspace_templates(workspace):
        print(f"[created {workspace / name}]", file=sys.stderr)
    return workspace


def _context_window() -> int:
    """``NANOCLAW_CONTEXT_WINDOW`` (tokens), else the default."""
    value = os.environ.get("NANOCLAW_CONTEXT_WINDOW", "").strip()
    try:
        return int(value) if value else DEFAULT_CONTEXT_WINDOW_TOKENS
    except ValueError:
        raise SystemExit(f"NANOCLAW_CONTEXT_WINDOW must be a whole number of tokens, got {value!r}")


def _make_loop(provider: LLMProvider, workspace: Path) -> AgentLoop:
    return AgentLoop(
        provider,
        workspace,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        context_window_tokens=_context_window(),
    )


async def _ask(provider: LLMProvider, prompt: str, workspace: Path, session_key: str, new: bool) -> int:
    loop = _make_loop(provider, workspace)
    if new:
        await loop.new_session(session_key)
    try:
        try:
            result = await loop.process_direct(prompt, session_key=session_key, channel=CHANNEL)
        except ContextWindowExceededError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if result.tools_used:
            print(f"[tools used: {', '.join(result.tools_used)}]", file=sys.stderr)
        status = 0
        if result.stop_reason == "error":
            print(result.final_content, file=sys.stderr)
            status = 1
        else:
            print(result.final_content or "")
        # Subagents started before an error still report.
        return max(status, await _deliver_subagent_results(loop))
    finally:
        await loop.close()


async def _deliver_subagent_results(loop: AgentLoop) -> int:
    """Wait for background subagents and let the main agent report each result.

    nanobot's one-shot ``agent -m`` cancels them at exit and loses their
    results; its chat and server modes deliver them as they finish. ``ask``
    waits instead, so a background spawn is useful before the layer 10 chat.
    """
    status = 0
    while (report := await loop.next_subagent_result()) is not None:
        print(f"[subagent '{report.label}' finished]", file=sys.stderr)
        try:
            result = await loop.process_subagent_result(report)
        except ContextWindowExceededError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            status = 1
            continue
        if result.tools_used:
            print(f"[tools used: {', '.join(result.tools_used)}]", file=sys.stderr)
        if result.stop_reason == "error":
            print(result.final_content, file=sys.stderr)
            status = 1
        elif result.final_content and result.final_content.strip():
            print(result.final_content)
    return status


async def _compact(provider: LLMProvider, workspace: Path, session_key: str) -> int:
    summary = await _make_loop(provider, workspace).compact_session(session_key)
    if summary == "":
        print("Nothing to compact.")
    elif summary is None:
        print("Compaction failed.", file=sys.stderr)
        return 1
    else:
        print(f"Conversation compacted. Summary:\n{summary}")
    return 0


def _skills(workspace: Path) -> int:
    """List the skills (nanobot's ``/skill`` command)."""
    loader = ContextBuilder(workspace).skills
    skills = loader.list_skills(filter_unavailable=False)
    if not skills:
        print("No skills available.")
        return 0
    lines = [f"Available skills ({len(skills)}):", ""]
    for entry in skills:
        lines.append(f"- **{entry['name']}** — {loader.get_skill_description(entry['name'])}")
    print("\n".join(lines))
    return 0


async def _dream(provider: LLMProvider, workspace: Path) -> int:
    print(await _make_loop(provider, workspace).run_dream())
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
    compact = commands.add_parser("compact", help="summarize the conversation so far into a short checkpoint")
    compact.add_argument(
        "-s", "--session", default=DEFAULT_SESSION, help=f"conversation to compact (default: {DEFAULT_SESSION})"
    )
    compact.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)
    dream = commands.add_parser("dream", help="update long-term memory (memory/MEMORY.md, USER.md, SOUL.md) from the history journal")
    dream.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)
    skills = commands.add_parser("skills", help="list the skills the agent can use (no API key needed)")
    skills.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)
    show = commands.add_parser("prompt", help="print the system prompt the model receives (no API key needed)")
    show.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)

    args = parser.parse_args(argv)
    if args.command == "ask":
        workspace = _prepare_workspace(args.workspace)
        return asyncio.run(
            _ask(provider or provider_from_env(), args.prompt, workspace, args.session, args.new)
        )
    if args.command == "compact":
        workspace = _prepare_workspace(args.workspace)
        return asyncio.run(_compact(provider or provider_from_env(), workspace, args.session))
    if args.command == "dream":
        workspace = _prepare_workspace(args.workspace)
        return asyncio.run(_dream(provider or provider_from_env(), workspace))
    if args.command == "skills":
        return _skills(_prepare_workspace(args.workspace))
    if args.command == "prompt":
        workspace = _prepare_workspace(args.workspace)
        print(ContextBuilder(workspace).build_system_prompt(channel=CHANNEL))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
