"""Command-line entry point: ``nanoclaw chat`` (interactive), ``nanoclaw ask`` (one message),
``nanoclaw gateway`` (long-running), and the rest."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from nanoclaw import __version__
from nanoclaw.cli.log_control import configure_console_logging, set_nanoclaw_logs
from nanoclaw.cli.plugins import add_plugins_parser, plugins_command
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import AgentLoop, ContextWindowExceededError
from nanoclaw.agent.runner import AgentRunResult
from nanoclaw.agent.tools.mcp import MCPProvider, configured_servers
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
from nanoclaw.agent.model_presets import (
    PresetRuntimeLoader,
    configured_model_presets,
    make_preset_runtime_loader,
    normalize_preset_name,
)
from nanoclaw.providers import LLMProvider
from nanoclaw.providers.factory import chain_context_window_tokens, provider_from_config
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


def _with_preset(config: Config, name: str | None) -> Config:
    """``--preset NAME``: the config with ``agents.defaults.modelPreset`` set to NAME.

    NanoClaw only (nanobot sets the preset in the config or with ``/model``).
    The name matches case-insensitively, as ``/model`` does.
    """
    if name is None:
        return config
    try:
        name = normalize_preset_name(name, configured_model_presets(config))
    except (KeyError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        raise SystemExit(f"--preset: {message}") from None
    config = config.model_copy(deep=True)
    config.agents.defaults.model_preset = None if name == "default" else name
    return config


def _make_loop(
    provider: LLMProvider,
    workspace: Path,
    *,
    restrict: bool = False,
    config: Config | None = None,
    project: Path | None = None,
    preset_loader: PresetRuntimeLoader | None = None,
) -> AgentLoop:
    """The agent loop with the configured settings (nanobot builds it from
    ``agents.defaults``, cli/agent.py) and ``tools.exec``. ``restrict``
    (``--restrict``) can only turn the workspace restriction on.

    The caller closes the loop (``aclose``) when done, which also ends the
    exec sessions it started (nanobot ``agent -m``'s ``_close_runtime``).

    ``project`` (``--project``) is the folder the agent works on; its own
    files stay in ``workspace``.

    Model settings come from the active preset (``agents.defaults.modelPreset``,
    else agents.defaults). ``preset_loader`` builds a provider for each
    preset ``/model`` picks; without it (a provider handed in, as in tests)
    presets reuse ``provider``.

    As in nanobot's ``agent`` command (cli/agent.py:177), the loop gets a
    ``CronService`` on ``<workspace>/cron/jobs.json`` for the ``cron`` tool,
    but the service is not started: jobs created here go to the store's
    action log and fire only while the gateway runs."""
    from nanoclaw.cron.service import CronService

    config = config or _runtime_config()
    defaults = config.agents.defaults
    preset = config.resolve_preset()
    return AgentLoop(
        provider,
        workspace,
        # A named preset's model; otherwise the provider's default model.
        model=preset.model if defaults.model_preset else None,
        max_iterations=defaults.max_tool_iterations,
        max_tool_result_chars=defaults.max_tool_result_chars,
        max_concurrent_subagents=defaults.max_concurrent_subagents,
        temperature=preset.temperature,
        max_tokens=preset.max_tokens,
        # The smallest window of the model and its fallbacks (nanobot build_provider_snapshot).
        context_window_tokens=chain_context_window_tokens(config, preset),
        restrict_to_workspace=restrict or config.tools.restrict_to_workspace,
        compaction=defaults.compaction,
        provider_retry_mode=defaults.provider_retry_mode,
        exec_config=config.tools.exec,
        project=project,
        model_presets=configured_model_presets(config),
        model_preset=defaults.model_preset,
        dream_model_preset=defaults.dream.model_override,
        preset_runtime_loader=preset_loader,
        cron_service=CronService(workspace / "cron" / "jobs.json"),
        timezone=defaults.timezone,
        # Idle auto-compaction on the bus loop's idle tick (nanobot from_config, loop.py:524).
        session_ttl_minutes=defaults.session_ttl_minutes,
        idle_compact_check_interval_seconds=defaults.idle_compact_check_interval_seconds,
    )


def _mcp_provider(loop: AgentLoop, config: Config | None = None) -> MCPProvider:
    """The configured MCP servers (``tools.mcpServers``), registering their
    tools into the loop's registry. As in nanobot (cli/agent.py), the caller
    connects before the first turn and closes it after the loop. Subagents
    build their own registries and get no MCP tools, as in nanobot. Enabled
    Agent Plugins add their stdio servers (layer 25)."""
    config = config or _runtime_config()
    return MCPProvider(configured_servers(config, loop.workspace), loop.tools)


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
    preset_loader: PresetRuntimeLoader | None = None,
) -> int:
    loop = _make_loop(
        provider, workspace, restrict=restrict, config=config, project=project, preset_loader=preset_loader
    )
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

    Since layer 19 ``ask`` normally finds none: with no bus loop running, the
    ``subagent`` tool runs every child inline (``wait`` or not) and its result
    is the tool result, as in nanobot's one-shot ``agent -m`` (c4dac64). Kept
    for reports that still reach the bus (e.g. a child started with
    ``SubagentManager.spawn`` directly).
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
    preset_loader: PresetRuntimeLoader | None = None,
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
    loop = _make_loop(
        provider, workspace, restrict=restrict, config=config, project=project, preset_loader=preset_loader
    )
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


async def _compact(
    provider: LLMProvider,
    workspace: Path,
    session_key: str,
    config: Config | None = None,
    preset_loader: PresetRuntimeLoader | None = None,
) -> int:
    loop = _make_loop(provider, workspace, config=config, preset_loader=preset_loader)
    try:
        summary = await loop.compact_session(session_key)
    finally:
        await loop.aclose()
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


async def _dream(
    provider: LLMProvider, workspace: Path, config: Config | None = None, preset_loader: PresetRuntimeLoader | None = None
) -> int:
    loop = _make_loop(provider, workspace, config=config, preset_loader=preset_loader)
    try:
        print(await loop.run_dream())
    finally:
        await loop.aclose()
    return 0


def _gateway(
    provider: LLMProvider,
    workspace: Path,
    config: Config,
    *,
    port: int | None = None,
    preset_loader: PresetRuntimeLoader | None = None,
) -> int:
    """``nanoclaw gateway``: the agent loop, MCP and ``/health`` in the
    foreground until SIGINT/SIGTERM (nanoclaw/cli/gateway.py)."""
    from nanoclaw.cli.gateway import run_gateway
    from nanoclaw.cli.terminal import make_console

    def make_agent() -> tuple[AgentLoop, MCPProvider]:
        loop = _make_loop(provider, workspace, config=config, preset_loader=preset_loader)
        return loop, _mcp_provider(loop, config)

    return run_gateway(config, make_agent, port=port, config_path=get_config_path(), console=make_console())


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


def _channels_command(action: str, name: str | None) -> int:
    """``nanoclaw channels status`` (nanobot ``channels status``, cli/commands.py:499) and
    ``nanoclaw channels validate telegram`` (nanobot's WebUI channel check,
    channels/validation.py ``validate_channel_config``)."""
    try:
        config = _runtime_config()
    except ConfigLoadError as exc:
        raise SystemExit(str(exc)) from None
    if action == "status":
        print("Channel   Enabled")
        for label, section in (("Telegram", config.channels.telegram), ("WebSocket", config.channels.websocket)):
            print(f"{label:<9} {'yes' if section.enabled else 'no'}")
        return 0
    from nanoclaw.channels.telegram.validation import validate_channel_config

    result = validate_channel_config(name or "telegram")
    print(f"{result['name']}: {result['status']} - {result['message']}")
    for item in result["checks"]:
        print(f"  [{item['status']}] {item['label']}: {item.get('message', '')}")
    if identity := result.get("identity"):
        print(f"  bot: @{identity.get('name', '?')} (id {identity.get('account', '?')})")
    return 0 if result["status"] in ("connected", "configured") else 1


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
    preset_help = "model preset to use (a modelPresets name or default; overrides agents.defaults.modelPreset)"
    chat = commands.add_parser(
        "chat", parents=[common, with_logs], help="chat with the agent in the terminal (type exit or press Ctrl+C to quit)"
    )
    chat.add_argument(
        "-s", "--session", default=DEFAULT_SESSION, help=f"conversation to continue (default: {DEFAULT_SESSION})"
    )
    chat.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    chat.add_argument("--restrict", action="store_true", help=restrict_help)
    chat.add_argument("-p", "--project", type=Path, default=None, help=project_help)
    chat.add_argument("--preset", default=None, metavar="NAME", help=preset_help)
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
    ask.add_argument("--preset", default=None, metavar="NAME", help=preset_help)
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
    gateway = commands.add_parser(
        "gateway",
        parents=[common],
        help="run the agent as a long-running process (idle compaction, /health endpoint) until Ctrl+C or SIGTERM",
    )
    gateway.add_argument("-p", "--port", type=int, default=None, help="health endpoint port (default: gateway.port, 18790)")
    gateway.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    # nanobot's gateway keeps its INFO log sink on; -v switches it to DEBUG.
    gateway.add_argument(
        "--logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show nanoclaw runtime logs on stderr (default: on)",
    )
    gateway.add_argument("-v", "--verbose", action="store_true", help="verbose output (DEBUG logs)")
    skills = commands.add_parser("skills", parents=[common], help="list the skills the agent can use (no API key needed)")
    skills.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    add_plugins_parser(commands, common, workspace_help)
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

    channels_cmd = commands.add_parser("channels", help="show the chat channels, or check a bot token")
    channels_actions = channels_cmd.add_subparsers(dest="channels_action", required=True)
    channels_actions.add_parser("status", parents=[common], help="list the channels and whether they are enabled")
    channels_validate = channels_actions.add_parser(
        "validate", parents=[common], help="check a channel's settings (telegram: the bot token, via getMe)"
    )
    channels_validate.add_argument("name", nargs="?", default="telegram", choices=["telegram"])

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    set_config_path(args.config)
    verbose = getattr(args, "verbose", False)
    if getattr(args, "logs", False) or verbose:
        # INFO and above, like nanobot's CLI sink; DEBUG with gateway -v.
        configure_console_logging(sys.stderr, level="DEBUG" if verbose else "INFO")
    set_nanoclaw_logs(getattr(args, "logs", False) or verbose)
    if args.command == "config":
        return _config_command(args.config_action)
    if args.command == "channels":
        return _channels_command(args.channels_action, getattr(args, "name", None))

    config = _with_preset(_runtime_config(), getattr(args, "preset", None))
    # Trusted private ranges web_fetch may reach (nanobot tools.ssrfWhitelist,
    # applied at load time in nanobot loader.py:133).
    configure_ssrf_whitelist(config.tools.ssrf_whitelist)
    workspace_arg = args.workspace or config.workspace_path
    # Checked first, so a bad --project creates no workspace files.
    project = _project_dir(args.project, workspace_arg) if args.command in ("chat", "ask", "prompt") else None
    workspace = _prepare_workspace(workspace_arg)

    def get_provider() -> LLMProvider:
        if provider is not None:
            return provider
        try:
            return provider_from_config(config)
        except ValueError as exc:  # e.g. "No API key configured for provider 'deepseek'."
            raise SystemExit(f"Provider setup error: {exc}") from None

    # Presets picked with /model get their own provider, built from the config.
    loader = None if provider is not None else make_preset_runtime_loader(config)
    if args.command == "chat":
        return asyncio.run(
            _chat(
                get_provider(),
                workspace,
                args.session,
                args.restrict,
                args.markdown,
                config,
                project=project,
                preset_loader=loader,
            )
        )
    if args.command == "ask":
        return asyncio.run(
            _ask(
                get_provider(),
                args.prompt,
                workspace,
                args.session,
                args.new,
                args.restrict,
                config,
                project=project,
                preset_loader=loader,
            )
        )
    if args.command == "compact":
        return asyncio.run(_compact(get_provider(), workspace, args.session, config, preset_loader=loader))
    if args.command == "gateway":
        return _gateway(get_provider(), workspace, config, port=args.port, preset_loader=loader)
    if args.command == "dream":
        return asyncio.run(_dream(get_provider(), workspace, config, preset_loader=loader))
    if args.command == "skills":
        return _skills(workspace)
    if args.command == "plugins":
        return plugins_command(args, workspace)
    if args.command == "prompt":
        builder = ContextBuilder(workspace)
        builder.include_scheduling = True  # chat and ask have the cron tool
        print(builder.build_system_prompt(channel=CHANNEL, workspace=project))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
