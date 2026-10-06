"""Command-line entry point: ``nanoclaw chat`` (interactive), ``nanoclaw ask`` (one message), and the rest."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from nanoclaw import __version__
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import DEFAULT_CONTEXT_WINDOW_TOKENS, AgentLoop, ContextWindowExceededError
from nanoclaw.agent.memory import COMPACTION_MODES, COMPACTION_NANOBOT
from nanoclaw.providers import LLMProvider, provider_from_env
from nanoclaw.security.network import configure_ssrf_whitelist_from_env
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


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _restrict_from_env() -> bool:
    """``NANOCLAW_RESTRICT_TO_WORKSPACE`` (nanobot config ``tools.restrictToWorkspace``)."""
    return os.environ.get("NANOCLAW_RESTRICT_TO_WORKSPACE", "").strip().lower() in _TRUE_VALUES


def _compaction() -> str:
    """``NANOCLAW_COMPACTION``: ``nanobot`` (default, like upstream) or ``fixed``
    (docs/UPSTREAM_BUGS.md, bug 4)."""
    value = os.environ.get("NANOCLAW_COMPACTION", "").strip().lower() or COMPACTION_NANOBOT
    if value not in COMPACTION_MODES:
        raise SystemExit(f"NANOCLAW_COMPACTION must be one of {', '.join(COMPACTION_MODES)}, got {value!r}")
    return value


def _make_loop(provider: LLMProvider, workspace: Path, *, restrict: bool = False) -> AgentLoop:
    return AgentLoop(
        provider,
        workspace,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        context_window_tokens=_context_window(),
        restrict_to_workspace=restrict,
        compaction=_compaction(),
    )


async def _print_progress_to_stderr(content: str, *, tool_hint: bool = False, **_: object) -> None:
    """Progress lines for ``ask``, on stderr so stdout holds only the reply.

    nanobot's ``agent -m`` prints the same "↳" lines (cli/agent.py:199).
    """
    if content.strip():
        print(f"  ↳ {content}", file=sys.stderr)


async def _ask(
    provider: LLMProvider, prompt: str, workspace: Path, session_key: str, new: bool, restrict: bool = False
) -> int:
    loop = _make_loop(provider, workspace, restrict=restrict)
    if new:
        await loop.new_session(session_key)
    try:
        try:
            result = await loop.process_direct(
                prompt, session_key=session_key, channel=CHANNEL, on_progress=_print_progress_to_stderr
            )
        except ContextWindowExceededError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
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
    results; its chat and server modes deliver them as they finish (as
    ``nanoclaw chat`` does). ``ask`` waits instead, so a background spawn is
    still useful in one-shot use.
    """
    status = 0
    while (report := await loop.next_subagent_result()) is not None:
        print(f"[subagent '{report.label}' finished]", file=sys.stderr)
        try:
            result = await loop.process_subagent_result(report, on_progress=_print_progress_to_stderr)
        except ContextWindowExceededError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            status = 1
            continue
        if result.stop_reason == "error":
            print(result.final_content, file=sys.stderr)
            status = 1
        elif result.final_content and result.final_content.strip():
            print(result.final_content)
    return status


async def _chat(
    provider: LLMProvider, workspace: Path, session_key: str, restrict: bool, markdown: bool
) -> int:
    """The interactive chat (nanobot ``agent`` in classic mode)."""
    from nanoclaw.cli.chat import Chat, install_exit_signal_handlers
    from nanoclaw.cli.terminal import PromptInput, make_console, print_above_prompt, stdin_is_interactive

    console = make_console()
    loop = _make_loop(provider, workspace, restrict=restrict)
    if stdin_is_interactive():
        read_input, print_async = PromptInput(), print_above_prompt
    else:
        # Piped input (scripts, tests): plain lines, no prompt_toolkit.
        async def read_input() -> str:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                raise EOFError
            return line.rstrip("\n")

        from nanoclaw.cli.chat import _print_directly as print_async
    install_exit_signal_handlers(console)
    chat = Chat(
        loop,
        session_key=session_key,
        console=console,
        read_input=read_input,
        render_markdown=markdown,
        print_async=print_async,
    )
    await chat.run()
    return 0


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
    chat = commands.add_parser("chat", help="chat with the agent in the terminal (type exit or press Ctrl+C to quit)")
    chat.add_argument(
        "-s", "--session", default=DEFAULT_SESSION, help=f"conversation to continue (default: {DEFAULT_SESSION})"
    )
    chat.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)
    chat.add_argument(
        "--restrict",
        action="store_true",
        help="keep file tools and shell commands inside the workspace (also NANOCLAW_RESTRICT_TO_WORKSPACE=1)",
    )
    chat.add_argument(
        "--markdown", action=argparse.BooleanOptionalAction, default=True, help="render replies as Markdown (default: on)"
    )
    ask = commands.add_parser("ask", help="send one message and exit; the conversation continues across calls")
    ask.add_argument("prompt")
    ask.add_argument(
        "-s", "--session", default=DEFAULT_SESSION, help=f"conversation to continue (default: {DEFAULT_SESSION})"
    )
    ask.add_argument("--new", action="store_true", help="forget the conversation and start over")
    ask.add_argument("-w", "--workspace", type=Path, default=DEFAULT_WORKSPACE, help=workspace_help)
    ask.add_argument(
        "--restrict",
        action="store_true",
        help="keep file tools and shell commands inside the workspace (also NANOCLAW_RESTRICT_TO_WORKSPACE=1)",
    )
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
    # Trusted private ranges web_fetch may reach (nanobot tools.ssrfWhitelist).
    configure_ssrf_whitelist_from_env()
    if args.command == "chat":
        workspace = _prepare_workspace(args.workspace)
        restrict = args.restrict or _restrict_from_env()
        return asyncio.run(
            _chat(provider or provider_from_env(), workspace, args.session, restrict, args.markdown)
        )
    if args.command == "ask":
        workspace = _prepare_workspace(args.workspace)
        restrict = args.restrict or _restrict_from_env()
        return asyncio.run(
            _ask(provider or provider_from_env(), args.prompt, workspace, args.session, args.new, restrict)
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
