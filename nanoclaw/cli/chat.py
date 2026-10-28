"""The interactive terminal chat (``nanoclaw chat``).

Mirrors nanobot's classic terminal chat (``nanobot agent --classic``,
nanobot/cli/agent.py:283-443). Since layer 14 it talks to the agent over the
message bus, as nanobot's does: ``AgentLoop.run`` runs as a task, every line
typed is published as an :class:`InboundMessage` (channel ``cli``, asking
for stream events), and a consumer task draws what comes back on the
outbound queue: stream deltas and ends, progress lines, command replies and
final answers. Slash commands are handled by the loop's router; a
background subagent's report reaches the loop on the bus too.

Two input modes:

- ``type_ahead=False`` (piped input, tests): like nanobot, the next line is
  read only after the reply to the previous one arrived. Output is drawn
  live: spinner, then the streamed Markdown (layer 13's StreamRenderer).
- ``type_ahead=True`` (a terminal): the prompt stays open while a turn
  runs, so ``/stop`` cancels it (a priority command) and other text is
  injected into the running turn. nanobot's classic chat waits for the
  reply instead; its WebUI behaves like this. Output lands above the prompt
  (``patch_stdout``), each streamed round printed when it ends.

Command replies and answers that did not stream are printed whole; a
streamed answer is not printed again (nanobot cli/agent.py:405-428). Ctrl+C
or Ctrl+D at the prompt, or an exit word, ends the chat.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.console import Console

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.turn_delivery import META_WANTS_STREAM
from nanoclaw.bus import META_RENDER_AS, InboundMessage, OutboundMessage
from nanoclaw.bus.outbound_events import StreamedResponseEvent
from nanoclaw.bus.runtime_events import TurnRunStatusChanged
from nanoclaw.cli.stream import StreamRenderer
from nanoclaw.cli.terminal import (
    BOT_NAME,
    ThinkingSpinner,
    TurnOutput,
    is_exit_command,
    print_agent_response,
    restore_terminal,
    sanitize_surrogates,
)
from nanoclaw.command import CommandRouter, register_builtin_commands
from nanoclaw.command.builtin import _format_history_message as _format_history_message
from nanoclaw.command.builtin import build_help_text as build_help_text
from nanoclaw.command.builtin import build_status_content as build_status_content
from nanoclaw.command.router import normalize_command_text as normalize_command_text

if TYPE_CHECKING:
    from nanoclaw.config import ChannelsConfig

CHANNEL = "cli"
ReadInput = Callable[[], Awaitable[str]]
# Wraps output written while the prompt is waiting (piped mode: a background
# subagent's reply arriving between lines), so it lands above the prompt.
OutputAbovePrompt = Callable[[], AbstractAsyncContextManager[None]]
# Inbound metadata naming which typed line a reply answers (NanoClaw's own;
# replies copy their input's metadata, nanobot loop.py:2066).
META_CLI_INPUT_ID = "_cli_input_id"

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


def split_session_key(session_key: str) -> tuple[str, str]:
    """The chat's channel and chat id: ``cli`` and the part after the colon.
    (nanobot cli/agent.py:291-294)"""
    if ":" in session_key:
        return CHANNEL, session_key.split(":", 1)[1]
    return CHANNEL, session_key


class Chat:
    """One interactive chat session in the terminal, over the agent's bus."""

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
        type_ahead: bool = False,
    ) -> None:
        self.loop = loop
        self.bus = loop.bus
        self.session_key = session_key
        self.channel, self.chat_id = split_session_key(session_key)
        self.console = console
        self.read_input = read_input
        self.render_markdown = render_markdown
        self.output_above_prompt = output_above_prompt
        # sendProgress / sendToolHints / showReasoning (nanobot ``channels_config``).
        self.channels = channels
        self.type_ahead = type_ahead
        self.router = CommandRouter()
        register_builtin_commands(self.router)
        self._input_ids = itertools.count(1)
        self._command_input_ids: set[int] = set()
        self._output: TurnOutput | None = None
        self._output_scope: AsyncExitStack | None = None
        self._renderer: StreamRenderer | None = None
        self._at_prompt = False
        self._reply_seen = asyncio.Event()

    # -- output -----------------------------------------------------------

    def _reply(self, text: str, *, as_text: bool = False, show_header: bool = True) -> None:
        print_agent_response(
            self.console, text, render_markdown=self.render_markdown, render_as_text=as_text, show_header=show_header
        )

    async def _start_output(self, *, above_prompt: bool) -> TurnOutput:
        """A fresh renderer for a turn (nanobot cli/agent.py:391). Without type-ahead
        it draws live; with it, rounds are printed whole above the open prompt."""
        await self._end_output()
        if above_prompt:
            self._output_scope = AsyncExitStack()
            await self._output_scope.enter_async_context(self.output_above_prompt())
        self._renderer = StreamRenderer(
            render_markdown=self.render_markdown, bot_name=BOT_NAME, console=self.console, live=not self.type_ahead
        )
        self._output = TurnOutput(self._renderer, self.channels)
        return self._output

    async def _end_output(self) -> None:
        output, scope = self._output, self._output_scope
        self._output = self._output_scope = None
        if output is not None:
            await output.renderer.close()
        if scope is not None:
            await scope.aclose()

    @property
    def busy(self) -> bool:
        """Whether a turn's output is being drawn."""
        return self._output is not None

    def status_toolbar(self) -> str:
        """Text under the open prompt (type-ahead mode)."""
        return f"{BOT_NAME} is working… type /stop to cancel" if self.busy else ""

    def banner(self) -> None:
        """nanobot agent.py:288 (with the model name and preset, cli/runtime_config.py:30)."""
        model = self.loop.model or self.loop.provider.default_model
        preset = self.loop.model_preset
        preset_tag = f" (preset: {preset})" if preset else ""
        self.console.print(
            f"{BOT_NAME} Interactive mode [bold blue]({model})[/bold blue]{preset_tag} "
            "— type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit\n"
        )

    # -- main loop --------------------------------------------------------

    async def run(self) -> None:
        self.banner()
        unsubscribe = self.bus.subscribe(self._on_run_status, TurnRunStatusChanged)
        bus_task = asyncio.create_task(self.loop.run())
        outbound_task = asyncio.create_task(self._consume_outbound())
        try:
            while True:
                try:
                    # Stop the spinner before the prompt (nanobot cli/agent.py:363).
                    if self._renderer and not self.type_ahead:
                        self._renderer.stop_for_input()
                    self._at_prompt = True
                    try:
                        user_input = sanitize_surrogates(await self.read_input())
                    finally:
                        self._at_prompt = False
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
                await self.send(user_input)
                if not self.type_ahead:
                    await self._wait_for_reply()
        finally:
            unsubscribe()
            self.loop.stop()
            bus_task.cancel()
            await asyncio.gather(bus_task, return_exceptions=True)
            await self.loop.aclose()
            outbound_task.cancel()
            await asyncio.gather(outbound_task, return_exceptions=True)
            await self._end_output()

    async def send(self, text: str) -> None:
        """Publish one typed line on the bus (nanobot cli/agent.py:405-413).

        Ordinary text starts a turn's output unless one is being drawn (then
        the loop injects the text into that turn). Commands get no turn output:
        their reply is printed on its own.
        """
        input_id = next(self._input_ids)
        if self.router.command_name(text.strip()) is not None:
            self._command_input_ids.add(input_id)
        elif not (self.type_ahead and self.busy):
            await self._start_output(above_prompt=False)
        self._reply_seen.clear()
        await self.bus.publish_inbound(
            InboundMessage(
                channel=self.channel,
                sender_id="user",
                chat_id=self.chat_id,
                content=text,
                metadata={META_WANTS_STREAM: True, META_CLI_INPUT_ID: input_id},
                session_key_override=self.session_key,
            )
        )

    async def _wait_for_reply(self) -> None:
        """Wait for the reply, as nanobot's classic chat does (``turn_done``),
        then for work the command started in the background (/dream, shown
        with a spinner) and until everything on the outbound queue is drawn."""
        await self._reply_seen.wait()
        if self.loop._background_tasks:
            with ThinkingSpinner(self.console):
                await self.loop.wait_for_background_tasks()
        await self.bus.outbound.join()

    async def _on_run_status(self, event: TurnRunStatusChanged) -> None:
        """A turn of this chat went idle: queue that behind its output, so the
        consumer closes the turn's renderer in order (also after /stop)."""
        if event.status == "idle" and event.context.session_key == self.session_key:
            await self.bus.publish_outbound(
                OutboundMessage(channel=self.channel, chat_id=self.chat_id, content="", event=event)
            )

    # -- drawing the outbound queue ---------------------------------------

    async def _consume_outbound(self) -> None:
        """(nanobot cli/agent.py:310-350)"""
        while True:
            msg = await self.bus.consume_outbound()
            try:
                await self._show(msg)
            except Exception:
                logger.exception("Could not show outbound message")
            finally:
                self.bus.outbound.task_done()

    async def _show(self, msg: OutboundMessage) -> None:
        event = msg.event
        if isinstance(event, TurnRunStatusChanged):
            await self._end_output()
            return
        if event is not None and not isinstance(event, StreamedResponseEvent):
            # Stream and progress events. A turn nobody typed for (a
            # subagent report's follow-up) gets its output here.
            output = self._output or await self._start_output(above_prompt=self._at_prompt)
            await output.publish(event)
            return
        await self._show_reply(msg)
        self._reply_seen.set()

    async def _show_reply(self, msg: OutboundMessage) -> None:
        """A command reply or a turn's final answer. (nanobot cli/agent.py:322-347, 415-428)"""
        as_text = msg.metadata.get(META_RENDER_AS) == "text"
        output = self._output
        if output is None or msg.metadata.get(META_CLI_INPUT_ID) in self._command_input_ids:
            if msg.content:
                scope = self.output_above_prompt() if self._at_prompt else contextlib.nullcontext()
                async with _async_scope(scope):
                    with output.renderer.pause() if output is not None else contextlib.nullcontext():
                        self._reply(msg.content, as_text=as_text)
            return
        renderer = output.renderer
        if isinstance(msg.event, StreamedResponseEvent):
            # Already on screen, unless nothing reached the renderer.
            if msg.content and not renderer.streamed:
                await renderer.close()
                self._reply(msg.content, as_text=as_text, show_header=not renderer.header_printed)
        elif msg.content:
            await renderer.close()
            self._reply(msg.content, as_text=as_text, show_header=not renderer.header_printed)
        elif not renderer.streamed:
            await renderer.close()


@asynccontextmanager
async def _async_scope(scope: Any) -> AsyncIterator[None]:
    """Enter a sync or async context manager."""
    if hasattr(scope, "__aenter__"):
        async with scope:
            yield
    else:
        with scope:
            yield


def install_exit_signal_handlers(console: Console) -> None:
    """Ctrl+C (or a termination signal) outside the prompt ends the chat, as
    in nanobot (agent.py:296). An interrupted turn is saved like a cancelled
    one only when the loop gets to close; otherwise its message gets the
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
