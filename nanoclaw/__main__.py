"""Command-line entry point: ``nanoclaw chat`` (interactive), ``nanoclaw ask`` (one message), and the rest."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from nanoclaw import __version__
from nanoclaw.cli.log_control import configure_console_logging, set_nanoclaw_logs
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import AgentLoop, ContextWindowExceededError
from nanoclaw.agent.runner import AgentRunResult
from nanoclaw.agent.tools.mcp import MCPProvider
from nanoclaw.bus.outbound_events import ProgressEvent, StreamDeltaEvent, StreamEndEvent
from nanoclaw.config import (
    ChannelsConfig,
    Config,
    ConfigEnvError,
    ConfigLoadError,
    get_config_path,
    load_runtime_config,
    set_config_path,
)
from nanoclaw.events import AgentEvent, EventSink
from nanoclaw.providers import LLMProvider
from nanoclaw.providers.factory import provider_from_config
from nanoclaw.security.network import configure_ssrf_whitelist
from nanoclaw.security.workspace_access import WorkspaceScopeError, project_workspace_scope
from nanoclaw.utils.templates import sync_workspace_templates

CHANNEL = "cli"
# The project folder for chat/ask/prompt when --project is not given.
ENV_PROJECT = "NANOCLAW_PROJECT"
# nanobot's terminal chat uses the same key.
DEFAULT_SESSION = "cli:direct"


def _prepare_workspace(path: Path) -> Path:
    """Resolve the workspace and create any missing SOUL.md / USER.md / AGENTS.md."""
    workspace = path.expanduser().resolve()
    for name in sync_workspace_templates(workspace):
        print(f"[created {workspace / name}]", file=sys.stderr)
    return workspace


def _project_dir(value: Path | None, workspace: Path) -> Path | None:
    """``--project DIR`` or ``$NANOCLAW_PROJECT``: an existing directory,
    relative to the current one; None (work in the workspace) when neither
    is set. A bad folder stops the command with a readable message.

    nanobot picks a project per chat in its WebUI and keeps it in the
    session; NanoClaw takes it per run, for every session of the run.
    """
    if value is None:
        env = os.environ.get(ENV_PROJECT, "").strip()
        value = Path(env) if env else None
    if value is None:
        return None
    try:
        scope = project_workspace_scope(value, workspace=workspace, restrict_to_workspace=False)
    except WorkspaceScopeError as exc:
        raise SystemExit(f"--project {value}: {exc}") from None
    return scope.project_path


def _runtime_config() -> Config:
    """config.json (or defaults) with ``${VAR}`` resolved and ``NANOCLAW_*`` applied.

    A broken file or variable stops the command with a readable message.
    """
    try:
        return load_runtime_config()
    except (ConfigLoadError, ConfigEnvError) as exc:
        raise SystemExit(str(exc)) from None


def _compaction() -> str:
    """``agents.defaults.compaction`` / ``NANOCLAW_COMPACTION``: ``nanobot``
    (default, like upstream) or ``fixed`` (docs/UPSTREAM_BUGS.md, bug 4)."""
    return _runtime_config().agents.defaults.compaction


def _make_loop(
    provider: LLMProvider,
    workspace: Path,
    *,
    restrict: bool = False,
    config: Config | None = None,
    project: Path | None = None,
) -> AgentLoop:
    """The agent loop with the configured settings (nanobot builds it from
    ``agents.defaults``, cli/agent.py) and ``tools.exec``. ``restrict``
    (``--restrict``) can only turn the workspace restriction on.

    The caller closes the loop (``aclose``) when done, which also ends the
    exec sessions it started (nanobot ``agent -m``'s ``_close_runtime``).

    ``project`` (``--project``) is the folder the agent works on; its own
    files stay in ``workspace``."""
    config = config or _runtime_config()
    defaults = config.agents.defaults
    return AgentLoop(
        provider,
        workspace,
        max_iterations=defaults.max_tool_iterations,
        max_tool_result_chars=defaults.max_tool_result_chars,
        max_concurrent_subagents=defaults.max_concurrent_subagents,
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        context_window_tokens=defaults.context_window_tokens,
        restrict_to_workspace=restrict or config.tools.restrict_to_workspace,
        compaction=defaults.compaction,
        provider_retry_mode=defaults.provider_retry_mode,
        exec_config=config.tools.exec,
        project=project,
    )


def _mcp_provider(loop: AgentLoop, config: Config | None = None) -> MCPProvider:
    """The configured MCP servers (``tools.mcpServers``), registering their
    tools into the loop's registry. As in nanobot (cli/agent.py), the caller
    connects before the first turn and closes it after the loop. Subagents
    build their own registries and get no MCP tools, as in nanobot."""
    config = config or _runtime_config()
    return MCPProvider(config.tools.mcp_servers, loop.tools)


class _AskOutput:
    """``ask``'s output for one turn: the reply on stdout, the rest on stderr.

    nanobot's ``agent -m`` streams everything to stdout through a
    StreamRenderer (cli/agent.py:250-279). ``ask`` keeps stdout for the reply
    so it can be piped: the reply streams as plain text only when stdout is a
    terminal (``stream``), otherwise it is printed once at the end. Progress,
    tool hints, retry waits and compaction are "↳" lines on stderr, reasoning
    "✻" lines, filtered by ``channels`` like nanobot's ``_cli_progress``
    (cli/agent.py:212-245). A streamed segment before tool calls is the
    model's text before the tools: it stays on stdout (it was already shown).
    """

    def __init__(self, channels: ChannelsConfig, *, stream: bool) -> None:
        from nanoclaw.cli.stream import ReasoningBuffer

        self.channels = channels
        self.stream = stream
        self.streamed_content = False
        self._segment = ""
        self._reasoning = ReasoningBuffer()

    @property
    def events(self) -> EventSink:
        return EventSink(self.publish)

    @staticmethod
    def _err(line: str) -> None:
        print(line, file=sys.stderr)

    def _reasoning_line(self, text: str | None) -> None:
        if text:
            self._err(f"✻ {text}")

    async def publish(self, event: AgentEvent) -> None:
        from nanoclaw.cli.terminal import progress_event_text

        if isinstance(event, StreamDeltaEvent):
            # Like the renderer, nothing shows before the first visible text.
            if not self._segment and not event.content.strip():
                return
            self._segment += event.content
            sys.stdout.write(event.content)
            sys.stdout.flush()
            return
        if isinstance(event, StreamEndEvent):
            if self._segment and not self._segment.endswith("\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
            self.streamed_content, self._segment = bool(self._segment), ""
            return
        if (text := progress_event_text(event)) is not None:
            if text.strip():
                self._err(f"  ↳ {text}")
            return
        if not isinstance(event, ProgressEvent):
            return
        channels = self.channels
        if event.reasoning_end or event.reasoning or event.reasoning_delta:
            if not channels.show_reasoning:
                self._reasoning.clear()
            elif event.reasoning_end:
                self._reasoning_line(self._reasoning.flush())
            else:
                self._reasoning_line(self._reasoning.add(event.content))
            return
        if event.tool_hint and not channels.send_tool_hints:
            return
        if not event.tool_hint and not channels.send_progress:
            return
        if event.content.strip():
            self._err(f"  ↳ {event.content}")

    def finish(self, result: AgentRunResult) -> int:
        """Print the reply unless it streamed; errors go to stderr. Returns the exit status."""
        if result.stop_reason == "error":
            print(result.final_content, file=sys.stderr)
            return 1
        if not (self.streamed_content and self.stream):
            print(result.final_content or "")
        return 0


def _ask_output(config: Config | None) -> _AskOutput:
    channels = (config or _runtime_config()).channels
    return _AskOutput(channels, stream=sys.stdout.isatty())


async def _ask(
    provider: LLMProvider,
    prompt: str,
    workspace: Path,
    session_key: str,
    new: bool,
    restrict: bool = False,
    config: Config | None = None,
    project: Path | None = None,
) -> int:
    loop = _make_loop(provider, workspace, restrict=restrict, config=config, project=project)
    mcp = _mcp_provider(loop, config)
    if new:
        await loop.new_session(session_key)
    try:
        await mcp.connect()
        output = _ask_output(config)
        try:
            result = await loop.process_direct(
                prompt, session_key=session_key, channel=CHANNEL, events=output.events, streaming=output.stream
            )
        except ContextWindowExceededError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        status = output.finish(result)
        # Subagents started before an error still report.
        return max(status, await _deliver_subagent_results(loop, config))
    finally:
        try:
            await loop.close()
        finally:
            await mcp.aclose()


async def _deliver_subagent_results(loop: AgentLoop, config: Config | None = None) -> int:
    """Wait for background subagents and let the main agent report each result.

    nanobot's one-shot ``agent -m`` cancels them at exit and loses their
    results; its chat and server modes deliver them as they finish (as
    ``nanoclaw chat`` does). ``ask`` waits instead, so a background spawn is
    still useful in one-shot use.
    """
    status = 0
    while (report := await loop.next_subagent_result()) is not None:
        print(f"[subagent '{report.label}' finished]", file=sys.stderr)
        output = _ask_output(config)
        try:
            result = await loop.process_subagent_result(report, events=output.events, streaming=output.stream)
        except ContextWindowExceededError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            status = 1
            continue
        if result.stop_reason == "error" or (result.final_content and result.final_content.strip()):
            status = max(status, output.finish(result))
    return status


async def _chat(
    provider: LLMProvider,
    workspace: Path,
    session_key: str,
    restrict: bool,
    markdown: bool,
    config: Config | None = None,
    project: Path | None = None,
) -> int:
    """The interactive chat (nanobot ``agent`` in classic mode, over the bus).

    In a terminal the prompt stays open while a turn runs (``type_ahead``):
    ``/stop`` cancels the turn, other text is injected into it. Piped input
    is read a line at a time after each reply, as nanobot does.
    """
    from nanoclaw.cli.chat import Chat, install_exit_signal_handlers
    from nanoclaw.cli.terminal import PromptInput, make_console, stdin_is_interactive

    console = make_console()
    config = config or _runtime_config()
    loop = _make_loop(provider, workspace, restrict=restrict, config=config, project=project)
    interactive = stdin_is_interactive()
    prompt: PromptInput | None = None
    if interactive:
        prompt = PromptInput(raw_stdout=True)
        read_input = prompt
    else:
        # Piped input (scripts, tests): plain lines, no prompt_toolkit.
        async def read_input() -> str:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                raise EOFError
            return line.rstrip("\n")

    install_exit_signal_handlers(console)
    chat = Chat(
        loop,
        session_key=session_key,
        console=console,
        read_input=read_input,
        render_markdown=markdown,
        channels=config.channels,
        type_ahead=interactive,
    )
    if prompt is not None:
        prompt.bottom_toolbar = chat.status_toolbar
    mcp = _mcp_provider(loop, config)
    try:
        await mcp.connect()
        await chat.run()  # closes the loop when it ends
    finally:
        await mcp.aclose()
    return 0


async def _compact(provider: LLMProvider, workspace: Path, session_key: str, config: Config | None = None) -> int:
    summary = await _make_loop(provider, workspace, config=config).compact_session(session_key)
    if summary == "":
        print("Nothing to compact.")
    elif summary is None:
        print("Compaction failed.", file=sys.stderr)
        return 1
    else:
        print(f"Conversation compacted. Summary:\n{summary}")
    return 0


def _skills(workspace: Path) -> int:
    """List the skills (nanobot's ``/skill`` command).

    Unlike nanobot, skills whose requirements are missing are marked with what
    is missing: since nanobot 32f5cc7 the prompt's skill list leaves them out,
    so this is the one place that says why a skill is not offered.
    """
    loader = ContextBuilder(workspace).skills
    skills = loader.list_skills(filter_unavailable=False)
    if not skills:
        print("No skills available.")
        return 0
    lines = [f"Available skills ({len(skills)}):", ""]
    for entry in skills:
        line = f"- **{entry['name']}** — {loader.get_skill_description(entry['name'])}"
        available, missing = loader.get_skill_availability(entry["name"])
        if not available:
            line += f" (unavailable: {missing})" if missing else " (unavailable)"
        lines.append(line)
    print("\n".join(lines))
    return 0


async def _dream(provider: LLMProvider, workspace: Path, config: Config | None = None) -> int:
    print(await _make_loop(provider, workspace, config=config).run_dream())
    return 0


def _config_command(action: str | None) -> int:
    """``nanoclaw config path|show|init``.

    nanobot writes its config with ``onboard`` (cli/commands.py:155) and shows
    it piecemeal in ``status`` (commands.py:644); NanoClaw keeps both small
    under one command.
    """
    from nanoclaw.config.loader import config_to_json, mask_secrets, refresh_config_file, save_config

    path = get_config_path()
    if action == "path":
        print(path)
        return 0
    if action == "show":
        # The settings a run would use: the file (or defaults) with ${VAR}
        # resolved and NANOCLAW_* applied, API keys masked.
        config = _runtime_config()
        source = str(path) if path.exists() else f"{path} (not found; defaults)"
        print(f"# {source}", file=sys.stderr)
        print(json.dumps(mask_secrets(config_to_json(config)), indent=2, ensure_ascii=False))
        return 0
    if action == "init":
        try:
            if not path.exists():
                save_config(Config(), path)
                print(f"Created config at {path}")
            elif refresh_config_file(path):
                print(f"Config refreshed at {path} (existing values preserved)")
            else:
                print(f"Config already up to date at {path}")
        except ConfigLoadError as exc:
            raise SystemExit(str(exc)) from None
        return 0
    return 2


def main(argv: list[str] | None = None, provider: LLMProvider | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanoclaw", description="A minimal personal AI agent.")
    parser.add_argument("-V", "--version", action="version", version=f"nanoclaw {__version__}")
    commands = parser.add_subparsers(dest="command")

    # Every command takes -c/--config (nanobot: --config on each command).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="config file (default: $NANOCLAW_CONFIG or ~/.nanoclaw/config.json)",
    )
    # Commands that run the agent take --logs (nanobot agent --logs/--no-logs,
    # cli/agent.py:67; off by default).
    with_logs = argparse.ArgumentParser(add_help=False)
    with_logs.add_argument(
        "--logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show nanoclaw runtime logs on stderr (default: off)",
    )
    workspace_help = "directory the agent works in (default: agents.defaults.workspace, ~/.nanoclaw/workspace)"
    restrict_help = (
        "keep file tools and shell commands inside the workspace, or the project with --project "
        "(also tools.restrictToWorkspace or NANOCLAW_RESTRICT_TO_WORKSPACE=1)"
    )
    project_help = (
        f"folder to work on, if not the workspace (also {ENV_PROJECT}); "
        "memory, skills, SOUL.md and USER.md stay in the workspace, AGENTS.md is read from the project"
    )
    chat = commands.add_parser(
        "chat", parents=[common, with_logs], help="chat with the agent in the terminal (type exit or press Ctrl+C to quit)"
    )
    chat.add_argument(
        "-s", "--session", default=DEFAULT_SESSION, help=f"conversation to continue (default: {DEFAULT_SESSION})"
    )
    chat.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    chat.add_argument("--restrict", action="store_true", help=restrict_help)
    chat.add_argument("-p", "--project", type=Path, default=None, help=project_help)
    chat.add_argument(
        "--markdown", action=argparse.BooleanOptionalAction, default=True, help="render replies as Markdown (default: on)"
    )
    ask = commands.add_parser(
        "ask", parents=[common, with_logs], help="send one message and exit; the conversation continues across calls"
    )
    ask.add_argument("prompt")
    ask.add_argument(
        "-s", "--session", default=DEFAULT_SESSION, help=f"conversation to continue (default: {DEFAULT_SESSION})"
    )
    ask.add_argument("--new", action="store_true", help="forget the conversation and start over")
    ask.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    ask.add_argument("--restrict", action="store_true", help=restrict_help)
    ask.add_argument("-p", "--project", type=Path, default=None, help=project_help)
    compact = commands.add_parser(
        "compact", parents=[common, with_logs], help="summarize the conversation so far into a short checkpoint"
    )
    compact.add_argument(
        "-s", "--session", default=DEFAULT_SESSION, help=f"conversation to compact (default: {DEFAULT_SESSION})"
    )
    compact.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    dream = commands.add_parser(
        "dream",
        parents=[common, with_logs],
        help="update long-term memory (memory/MEMORY.md, USER.md, SOUL.md) from the history journal",
    )
    dream.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    skills = commands.add_parser("skills", parents=[common], help="list the skills the agent can use (no API key needed)")
    skills.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    show = commands.add_parser(
        "prompt", parents=[common], help="print the system prompt the model receives (no API key needed)"
    )
    show.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    show.add_argument("-p", "--project", type=Path, default=None, help=project_help)
    config_cmd = commands.add_parser("config", help="show or create the config file (config.json)")
    config_actions = config_cmd.add_subparsers(dest="config_action", required=True)
    config_actions.add_parser("path", parents=[common], help="print where the config file is")
    config_actions.add_parser(
        "show", parents=[common], help="print the settings in effect (file + NANOCLAW_* variables), keys masked"
    )
    config_actions.add_parser(
        "init", parents=[common], help="write a default config file, or add missing settings to an existing one"
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    set_config_path(args.config)
    if getattr(args, "logs", False):
        configure_console_logging(sys.stderr)  # INFO and above, like nanobot's CLI sink
    set_nanoclaw_logs(getattr(args, "logs", False))
    if args.command == "config":
        return _config_command(args.config_action)

    config = _runtime_config()
    # Trusted private ranges web_fetch may reach (nanobot tools.ssrfWhitelist,
    # applied at load time in nanobot loader.py:133).
    configure_ssrf_whitelist(config.tools.ssrf_whitelist)
    workspace_arg = args.workspace or config.workspace_path
    # Checked first, so a bad --project creates no workspace files.
    project = _project_dir(args.project, workspace_arg) if args.command in ("chat", "ask", "prompt") else None
    workspace = _prepare_workspace(workspace_arg)

    def get_provider() -> LLMProvider:
        return provider or provider_from_config(config)

    if args.command == "chat":
        return asyncio.run(
            _chat(get_provider(), workspace, args.session, args.restrict, args.markdown, config, project=project)
        )
    if args.command == "ask":
        return asyncio.run(
            _ask(get_provider(), args.prompt, workspace, args.session, args.new, args.restrict, config, project=project)
        )
    if args.command == "compact":
        return asyncio.run(_compact(get_provider(), workspace, args.session, config))
    if args.command == "dream":
        return asyncio.run(_dream(get_provider(), workspace, config))
    if args.command == "skills":
        return _skills(workspace)
    if args.command == "prompt":
        print(ContextBuilder(workspace).build_system_prompt(channel=CHANNEL, workspace=project))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
