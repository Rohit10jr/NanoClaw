"""The interactive terminal chat (``nanoclaw chat``).

Mirrors nanobot's classic terminal chat (``nanobot agent --classic``,
nanobot/cli/agent.py:266-443). Slash commands go through the channel-independent
router in nanoclaw/command/ (nanobot command/); their replies are printed here.
One difference in shape: nanobot's chat talks to the agent over its message
bus; NanoClaw calls the loop directly and watches the subagent result queue
(layer 8).

Since layer 13 replies stream: each turn gets a new StreamRenderer and a
:class:`TurnOutput` that draws the turn's events (nanobot cli/agent.py:310-443).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any

from rich.console import Console

from nanoclaw.agent.loop import AgentLoop, ContextWindowExceededError
from nanoclaw.agent.runner import AgentRunResult
from nanoclaw.bus import META_RENDER_AS, InboundMessage, MessageBus, OutboundMessage
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
from nanoclaw.command import CommandContext, CommandRouter, register_builtin_commands
from nanoclaw.command.builtin import _format_history_message as _format_history_message
from nanoclaw.command.builtin import build_help_text as build_help_text
from nanoclaw.command.builtin import build_status_content as build_status_content
from nanoclaw.command.router import normalize_command_text as normalize_command_text
from nanoclaw.events import AgentEvent, EventSink

if TYPE_CHECKING:
    from nanoclaw.config import ChannelsConfig

CHANNEL = "cli"
ReadInput = Callable[[], Awaitable[str]]
# Wraps output written while the prompt may be waiting (a background
# subagent's follow-up turn), so it lands above the prompt.
OutputAbovePrompt = Callable[[], AbstractAsyncContextManager[None]]

# The slash commands live in nanoclaw/command/ (nanobot command/); these
# module-level helpers use a router with the built-in commands.
_ROUTER = CommandRouter()
register_builtin_commands(_ROUTER)


def command_name(text: str) -> str | None:
    """The registered command *text* starts with, or None for ordinary chat
    text (nanobot ``CommandRouter.command_name``, router.py:98-105)."""
    return _ROUTER.command_name(text)


def invalid_command_text(raw: str) -> str | None:
    """What a registered command used with wrong arguments gets, or None when
    *raw* isn't a registered command (nanobot router.py:149-168)."""
    return _ROUTER.invalid_command_text(raw)


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
        self.router = CommandRouter()
        register_builtin_commands(self.router)
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
                if self.router.command_name(command) is not None:
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
        """Run a registered slash command through the router and print its
        reply (nanobot loop.py: ``dispatch_priority`` for /stop and /status,
        else ``dispatch``; router.py:124-147)."""

        async def show(event: AgentEvent) -> None:
            # /compact's progress events (nanobot terminal.py:372).
            if (text := progress_event_text(event)) is not None:
                print_progress_line(self.console, text)

        # The chat has no bus consumer; follow-up replies (/dream's result)
        # are collected here and printed when the command's work is done.
        followups = MessageBus()
        msg = InboundMessage(
            channel=CHANNEL, sender_id="user", chat_id="direct", content=raw, session_key_override=self.session_key
        )
        ctx = CommandContext(
            msg=msg,
            session=None,
            key=self.session_key,
            raw=raw,
            loop=self.loop,
            is_user_turn=True,
            events=EventSink(show),
            bus=followups,
        )
        if self.router.is_priority(raw):
            result = await self.router.dispatch_priority(ctx)
        else:
            result = await self.router.dispatch(ctx)
        if isinstance(result, OutboundMessage):
            self._print_outbound(result)
        elif isinstance(result, InboundMessage):
            await self.turn(result.content)
        if ctx.tasks:
            # nanobot shows "Dreaming..." and later the result; the chat waits
            # for it with a spinner, so the reply lands before the next prompt.
            with ThinkingSpinner(self.console):
                await asyncio.gather(*ctx.tasks, return_exceptions=True)
        while followups.outbound_size:
            self._print_outbound(await followups.consume_outbound())

    def _print_outbound(self, msg: OutboundMessage) -> None:
        self._reply(msg.content, as_text=msg.metadata.get(META_RENDER_AS) == "text")


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

