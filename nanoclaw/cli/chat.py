"""The interactive terminal chat (``nanoclaw chat``).

Mirrors nanobot's classic terminal chat (``nanobot agent --classic``,
nanobot/cli/agent.py:266-443) and the slash commands it shares with the other
channels (nanobot/command/builtin.py, command/router.py). One difference in
shape: nanobot's chat talks to the agent over its message bus; NanoClaw calls
the loop directly and watches the subagent result queue (layer 8).

Since layer 13 replies stream: each turn gets a new StreamRenderer and a
:class:`TurnOutput` that draws the turn's events (nanobot cli/agent.py:310-443).
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.console import Console

from nanoclaw import __version__
from nanoclaw.agent.loop import DREAM_NO_INPUT_MESSAGE, LAST_USAGE_KEY, AgentLoop, ContextWindowExceededError
from nanoclaw.agent.runner import AgentRunResult
from nanoclaw.cli.stream import StreamRenderer
from nanoclaw.cli.terminal import (
    BOT_NAME,
    ThinkingSpinner,
    TurnOutput,
    is_exit_command,
    print_agent_response,
    print_progress_line,
    progress_event_text,
    restore_terminal,
    sanitize_surrogates,
)
from nanoclaw.events import AgentEvent, EventSink

if TYPE_CHECKING:
    from nanoclaw.config import ChannelsConfig

CHANNEL = "cli"
ReadInput = Callable[[], Awaitable[str]]
# Wraps output written while the prompt may be waiting (a background
# subagent's follow-up turn), so it lands above the prompt.
OutputAbovePrompt = Callable[[], AbstractAsyncContextManager[None]]

_HISTORY_DEFAULT_COUNT = 10
_HISTORY_MAX_COUNT = 50
_HISTORY_MAX_CONTENT_CHARS = 200
_CONTEXT_SAFETY_BUFFER = 1024


@dataclass(frozen=True)
class CommandSpec:
    """A slash command, as nanobot's ``BuiltinCommandSpec`` (command/builtin.py:69)."""

    command: str
    description: str
    arg_hint: str = ""
    accepts_args: bool = False


# nanobot's commands that NanoClaw has, in nanobot's order and with its descriptions.
COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("/new", "Reset this chat and start a fresh conversation."),
    CommandSpec("/compact", "Compact this chat's context and continue the conversation."),
    CommandSpec("/stop", "Cancel the active agent turn for this chat."),
    CommandSpec("/status", "Display runtime, provider, and channel status."),
    CommandSpec("/history", "Print the last N persisted conversation messages.", "[n]", True),
    CommandSpec("/dream", "Manually trigger memory consolidation."),
    CommandSpec("/skill", "List all enabled skills available to the agent."),
    CommandSpec("/help", "List available slash commands."),
)


def build_help_text() -> str:
    """nanobot ``build_help_text`` (command/builtin.py:1056)."""
    lines = [f"{BOT_NAME} commands:"]
    for spec in COMMANDS:
        command = f"{spec.command} {spec.arg_hint}" if spec.arg_hint else spec.command
        lines.append(f"{command} — {spec.description}")
    return "\n".join(lines)


_BOT_SUFFIX_RE = re.compile(r"^[A-Za-z0-9_]+$")


def normalize_command_text(text: str) -> str:
    """Strip a transport ``@bot`` suffix from the command word (nanobot router.py:26-42).

    ``/cmd@bot args`` becomes ``/cmd args``; arguments stay verbatim.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return stripped
    first, sep, rest = stripped.partition(" ")
    if "@" not in first:
        return stripped
    command, suffix = first.rsplit("@", 1)
    if command and suffix and _BOT_SUFFIX_RE.fullmatch(suffix):
        return f"{command}{sep}{rest}" if sep else command
    return stripped


def command_name(text: str) -> str | None:
    """The registered command *text* starts with, or None for ordinary chat
    text (nanobot ``CommandRouter.command_name``, router.py:98-105).

    A leading slash alone is not a command: absolute paths such as
    ``/tmp/foo.py explain`` and unknown names go to the model.
    """
    parts = normalize_command_text(text).split(maxsplit=1)
    commands = {spec.command.lower(): spec.command for spec in COMMANDS}
    return commands.get(parts[0].lower()) if parts else None


def invalid_command_text(raw: str) -> str | None:
    """What a registered command used with wrong arguments gets, or None when
    *raw* isn't a registered command (nanobot router.py:149-168)."""
    canonical = command_name(raw)
    if canonical is None:
        return None
    entered = normalize_command_text(raw).split(maxsplit=1)[0]
    if any(spec.accepts_args and spec.command == canonical for spec in COMMANDS):
        return f'Invalid command "{entered}". Use "/help" to list available commands.'
    return f'Command "{canonical}" does not accept arguments. Did you mean "{canonical}"?'


def _format_history_message(msg: dict[str, Any]) -> str | None:
    """One message for /history, or None to skip it. (nanobot command/builtin.py:845)"""
    role = msg.get("role")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        )
    content = str(content).strip()
    if not content:
        return None
    if len(content) > _HISTORY_MAX_CONTENT_CHARS:
        content = content[:_HISTORY_MAX_CONTENT_CHARS] + "…"
    label = "👤 You" if role == "user" else "🤖 Bot"
    return f"{label}: {content}"


def build_status_content(
    *,
    model: str,
    start_time: float,
    last_usage: dict[str, Any] | None,
    context_window_tokens: int,
    session_msg_count: int,
    context_tokens_estimate: int,
    active_task_count: int,
    max_completion_tokens: int,
) -> str:
    """Runtime status for /status. (nanobot ``build_status_content``, utils/helpers.py:900)"""
    uptime_s = int(time.time() - start_time)
    uptime = (
        f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m" if uptime_s >= 3600 else f"{uptime_s // 60}m {uptime_s % 60}s"
    )
    last_in = int((last_usage or {}).get("input_tokens") or 0)
    last_out = int((last_usage or {}).get("output_tokens") or 0)
    cached = int((last_usage or {}).get("cache_read_tokens") or 0)
    token_line = f"📊 Tokens: {last_in} in / {last_out} out"
    if cached and last_in:
        token_line += f" ({cached * 100 // last_in}% cached)"
    ctx_total = max(context_window_tokens, 0)
    ctx_budget = max(ctx_total - int(max_completion_tokens) - _CONTEXT_SAFETY_BUFFER, 1)
    ctx_pct = min(int((context_tokens_estimate / ctx_budget) * 100), 999) if ctx_budget > 0 else 0
    ctx_used_str = f"{context_tokens_estimate // 1000}k" if context_tokens_estimate >= 1000 else str(context_tokens_estimate)
    ctx_total_str = f"{ctx_total // 1000}k" if ctx_total > 0 else "n/a"
    return "\n".join(
        [
            f"{BOT_NAME} v{__version__}",
            f"🧠 Model: {model}",
            token_line,
            f"📚 Context: {ctx_used_str}/{ctx_total_str} ({ctx_pct}% of input budget)",
            f"💬 Session: {session_msg_count} messages",
            f"⏱ Uptime: {uptime}",
            f"⚡ Tasks: {active_task_count} active",
        ]
    )


@asynccontextmanager
async def _output_directly() -> AsyncIterator[None]:
    yield


class Chat:
    """One interactive chat session in the terminal."""

    def __init__(
        self,
        loop: AgentLoop,
        *,
        session_key: str = "cli:direct",
        console: Console,
        read_input: ReadInput,
        render_markdown: bool = True,
        output_above_prompt: OutputAbovePrompt = _output_directly,
        channels: ChannelsConfig | None = None,
    ) -> None:
        self.loop = loop
        self.session_key = session_key
        self.console = console
        self.read_input = read_input
        self.render_markdown = render_markdown
        self.output_above_prompt = output_above_prompt
        # sendProgress / sendToolHints / showReasoning (nanobot ``channels_config``).
        self.channels = channels
        self._turn_lock = asyncio.Lock()
        self._renderer: StreamRenderer | None = None

    # -- output -----------------------------------------------------------

    def _reply(self, text: str, *, as_text: bool = False, show_header: bool = True) -> None:
        print_agent_response(
            self.console, text, render_markdown=self.render_markdown, render_as_text=as_text, show_header=show_header
        )

    def _new_turn_output(self) -> TurnOutput:
        """A fresh renderer for each turn (nanobot cli/agent.py:391)."""
        self._renderer = StreamRenderer(render_markdown=self.render_markdown, bot_name=BOT_NAME, console=self.console)
        return TurnOutput(self._renderer, self.channels)

    async def _streamed_turn(self, run: Callable[[EventSink], Awaitable[AgentRunResult]]) -> None:
        """Run a turn with streaming output, then print the reply unless it streamed.

        nanobot cli/agent.py:405-428: when the answer was not streamed (or is
        an error) the renderer is closed and the whole reply printed, without
        the name line if the renderer already printed it.
        """
        output = self._new_turn_output()
        renderer = output.renderer
        try:
            result = await run(output.events)
        except ContextWindowExceededError as exc:
            await renderer.close()
            self._reply(f"Error: {exc}", as_text=True, show_header=not renderer.header_printed)
            return
        except BaseException:
            await renderer.close()
            raise
        reply = result.final_content or ""
        if reply and not output.answer_on_screen(result.stop_reason):
            await renderer.close()
            self._reply(reply, show_header=not renderer.header_printed)
        elif not renderer.streamed:
            await renderer.close()

    def banner(self) -> None:
        """nanobot agent.py:271 (with the model name)."""
        model = self.loop.model or self.loop.provider.default_model
        self.console.print(
            f"{BOT_NAME} Interactive mode [bold blue]({model})[/bold blue] "
            "— type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit\n"
        )

    # -- main loop --------------------------------------------------------

    async def run(self) -> None:
        self.banner()
        watcher = asyncio.create_task(self._deliver_subagent_results())
        try:
            while True:
                try:
                    # Stop the spinner before the prompt (nanobot cli/agent.py:363).
                    if self._renderer:
                        self._renderer.stop_for_input()
                    user_input = sanitize_surrogates(await self.read_input())
                except (KeyboardInterrupt, EOFError):
                    restore_terminal()
                    self.console.print("\nGoodbye!")
                    break
                command = user_input.strip()
                if not command:
                    continue
                if is_exit_command(command):
                    restore_terminal()
                    self.console.print("\nGoodbye!")
                    break
                # Only registered command names are commands; other slash text
                # (paths, unknown names) is chat (nanobot router.py:107-122).
                if command_name(command) is not None:
                    await self.handle_command(command)
                    continue
                await self.turn(user_input)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            await self.loop.close()

    async def turn(self, content: str) -> None:
        """One user message: the reply streams in, with progress lines while it works."""
        async with self._turn_lock:
            await self._streamed_turn(
                lambda events: self.loop.process_direct(
                    content, session_key=self.session_key, channel=CHANNEL, events=events, streaming=True
                )
            )

    async def _deliver_subagent_results(self) -> None:
        """Background subagent results become follow-up turns that stream like
        user turns, above the prompt if it's waiting (nanobot agent.py:296-350)."""
        while True:
            report = await self.loop.subagents.results.get()
            async with self._turn_lock, self.output_above_prompt():
                await self._streamed_turn(
                    lambda events: self.loop.process_subagent_result(report, events=events, streaming=True)
                )
                if self._renderer:
                    self._renderer.stop_for_input()

    # -- slash commands ---------------------------------------------------

    async def handle_command(self, raw: str) -> None:
        """Route a registered slash command (nanobot ``CommandRouter.dispatch``,
        router.py:132-147)."""
        cmd = normalize_command_text(raw)
        lower = cmd.lower()
        exact: dict[str, Callable[[], Awaitable[None]]] = {
            "/new": self.cmd_new,
            "/compact": self.cmd_compact,
            "/stop": self.cmd_stop,
            "/status": self.cmd_status,
            "/history": lambda: self.cmd_history(""),
            "/dream": self.cmd_dream,
            "/skill": self.cmd_skill,
            "/help": self.cmd_help,
        }
        if handler := exact.get(lower):
            await handler()
        elif lower.startswith("/history "):
            await self.cmd_history(cmd[len("/history "):])
        elif (text := invalid_command_text(cmd)) is not None:
            self._reply(text, as_text=True)

    async def cmd_new(self) -> None:
        """Stop this chat's work, archive and clear the conversation (nanobot ``cmd_new``)."""
        await self.loop.subagents.cancel_by_session(self.session_key)
        await self.loop.new_session(self.session_key)
        self._reply("New session started.")

    async def cmd_compact(self) -> None:
        """Summarize now; the compaction events are printed as progress lines
        (nanobot command/builtin.py:358, terminal.py:372). Nothing to compact
        prints nothing."""

        async def show(event: AgentEvent) -> None:
            if (text := progress_event_text(event)) is not None:
                print_progress_line(self.console, text)

        await self.loop.compact_session(self.session_key, events=EventSink(show))

    async def cmd_stop(self) -> None:
        """Cancel this chat's background subagents (nanobot ``cmd_stop``)."""
        total = await self.loop.subagents.cancel_by_session(self.session_key)
        self._reply(f"Stopped {total} task(s)." if total else "No active task to stop.")

    async def cmd_status(self) -> None:
        loop = self.loop
        session = loop.sessions.get_or_create(self.session_key)
        try:
            estimate, _ = loop.consolidator.estimate_session_prompt_tokens(session, runtime=loop.runtime)
        except Exception:
            estimate = 0
        last_usage = session.metadata.get(LAST_USAGE_KEY)
        if estimate <= 0 and isinstance(last_usage, dict):
            estimate = int(last_usage.get("input_tokens") or 0)
        content = build_status_content(
            model=loop.model or loop.provider.default_model,
            start_time=loop.start_time,
            last_usage=last_usage if isinstance(last_usage, dict) else None,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history()),
            context_tokens_estimate=estimate,
            active_task_count=loop.subagents.get_running_count_by_session(self.session_key),
            max_completion_tokens=loop.max_tokens,
        )
        self._reply(content, as_text=True)

    async def cmd_history(self, args: str) -> None:
        """The last N messages (default 10, max 50). (nanobot ``cmd_history``)"""
        count = _HISTORY_DEFAULT_COUNT
        if args.strip():
            try:
                count = max(1, min(int(args.strip()), _HISTORY_MAX_COUNT))
            except ValueError:
                self._reply("Usage: /history [count] — e.g. /history 5 (default: 10, max: 50)")
                return
        session = self.loop.sessions.get_or_create(self.session_key)
        visible = [m for m in (_format_history_message(m) for m in session.get_history(include_runtime_context=False)) if m]
        recent = visible[-count:]
        if not recent:
            self._reply("No conversation history yet.")
            return
        self._reply(f"Last {len(recent)} message(s):\n" + "\n".join(recent), as_text=True)

    async def cmd_dream(self) -> None:
        """nanobot replies "Dreaming..." and reports when Dream finishes."""
        self._reply("Dreaming...")
        spinner = ThinkingSpinner(self.console)
        with spinner:
            result = await self.loop.run_dream()
        # nanobot shows the nothing-to-do message as plain text (keeps its line breaks).
        self._reply(result, as_text=result == DREAM_NO_INPUT_MESSAGE)

    async def cmd_skill(self) -> None:
        """nanobot ``cmd_skill``."""
        loader = self.loop.context.skills
        skills = loader.list_skills(filter_unavailable=False)
        if not skills:
            self._reply("No skills available.")
            return
        lines = [f"Available skills ({len(skills)}):", ""]
        for entry in skills:
            lines.append(f"- **{entry['name']}** — {loader.get_skill_description(entry['name'])}")
        self._reply("\n".join(lines))

    async def cmd_help(self) -> None:
        self._reply(build_help_text(), as_text=True)


def install_exit_signal_handlers(console: Console) -> None:
    """Ctrl+C (or a termination signal) while the agent works ends the chat,
    as in nanobot (agent.py:283). The interrupted message gets the
    "interrupted" reply on the next turn (layer 5 recovery)."""
    import signal

    def _handle(signum: int, _frame: Any) -> None:
        restore_terminal()
        console.print(f"\nReceived {signal.Signals(signum).name}, goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle)
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

