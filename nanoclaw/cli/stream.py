"""Streaming replies in the terminal.

Mirrors nanobot/cli/stream.py (StreamRenderer, ThinkingSpinner) and the
classic-chat helpers that drive it from nanobot/cli/terminal.py (progress
lines, the reasoning buffer and the dim "✻" reasoning lines).

While a reply streams, rich ``Live`` (``transient=True``) redraws it in place;
at the end of a round the live area is erased and the reply is printed once
so it stays on screen. A "thinking" spinner runs until the first visible
delta, and again after a round that ends with ``resuming=True`` (tools run,
then the model answers again).

With ``live=False`` (NanoClaw's own; the chat uses it while the prompt stays
open during a turn, since a live area and the prompt can't share the screen)
nothing is redrawn and there is no spinner: the text so far is printed once
it pauses for a moment (outside a code block) and the rest when the round
ends, below the header.

Everything printed as rich markup is escaped: nanobot prints progress and
reasoning text as markup, so ``[/x]`` crashes the turn and ``[red]`` vanishes
(UPSTREAM_BUGS.md, bug 6). Reply deltas are safe in both: they go through
``Markdown``/``Text``, which never parse markup.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Literal

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text

BOT_NAME = "nanoclaw"

# nanobot stream.py:23: at most one live redraw per 50 ms; deltas in between are coalesced.
_STREAM_REFRESH_INTERVAL = 0.05
# live=False: print the buffered text after this long without a delta.
_QUIET_FLUSH_SECONDS = 0.3
# nanobot terminal.py:52-53: reasoning is printed a sentence (or 60 chars) at a time.
_REASONING_SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？")
_REASONING_FLUSH_CHARS = 60


def _clear_current_line(console: Console) -> None:
    """Erase the spinner's line before printing lasting output. (nanobot ``stream.py:26``)"""
    file = console.file
    if not getattr(file, "isatty", lambda: False)():
        return
    file.write("\r\x1b[2K")
    file.flush()


def _make_console() -> Console:
    """Plain text when stdout is not a terminal, so pipes get no spinner frames or
    cursor codes. (nanobot ``stream.py:36``, nanobot #3265)"""
    return Console(file=sys.stdout, force_terminal=sys.stdout.isatty())


def progress_line_markup(text: str) -> str:
    """A dim "↳" line. The text is escaped: nanobot prints it as rich markup,
    so a command like ``echo "[/x]"`` crashes its turn (UPSTREAM_BUGS.md, bug 6)."""
    return f"  [dim]↳ {escape(text)}[/dim]"


def reasoning_markup(text: str) -> str:
    """A dim italic "✻" reasoning line, escaped like progress lines.
    (nanobot ``terminal.py:334`` prints the text unescaped: bug 6 again)"""
    return f"[dim italic]✻ {escape(text)}[/dim italic]"


class ThinkingSpinner:
    """Shows '<name> is thinking...' while the agent works; pauses for output lines.

    (nanobot ``stream.py:50`` ``ThinkingSpinner``)
    """

    def __init__(self, console: Console | None = None, bot_name: str = BOT_NAME) -> None:
        self._console = console or _make_console()
        self._spinner = self._console.status(f"[dim]{escape(bot_name)} is thinking...[/dim]", spinner="dots")
        self._active = False

    def __enter__(self) -> ThinkingSpinner:
        self._spinner.start()
        self._active = True
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self._active = False
        self._spinner.stop()
        _clear_current_line(self._console)
        return False

    @contextmanager
    def pause(self) -> Iterator[None]:
        """Temporarily stop the spinner so a line prints cleanly."""
        if self._active:
            self._spinner.stop()
            _clear_current_line(self._console)
        try:
            yield
        finally:
            if self._active:
                self._spinner.start()


class ReasoningBuffer:
    """Collects reasoning deltas and hands them out a sentence at a time.

    (nanobot ``terminal.py:292`` ``_ReasoningBuffer``)
    """

    def __init__(self) -> None:
        self._text = ""

    def add(self, text: str) -> str | None:
        """Add a delta; return the text to print when a newline, a sentence end
        or 60 buffered characters is reached."""
        if not text:
            return None
        self._text += text
        if self._should_flush(text):
            return self.flush()
        return None

    def flush(self) -> str | None:
        text = self._text.strip()
        self._text = ""
        return text or None

    def clear(self) -> None:
        self._text = ""

    def _should_flush(self, text: str) -> bool:
        return (
            "\n" in text
            or text.rstrip().endswith(_REASONING_SENTENCE_ENDINGS)
            or len(self._text) >= _REASONING_FLUSH_CHARS
        )


class StreamRenderer:
    """Renders one turn's streamed reply: spinner, header, live Markdown, final print.

    Per round: spinner -> first visible delta -> header + live updates ->
    ``on_end`` -> live erased + final render (-> spinner again if resuming).
    Progress and reasoning lines can come at any point; they stop the live
    view, print under the header, and the next delta starts a fresh live view
    from the buffer.

    nanobot's classic chat (cli/agent.py:256-279, 405-445) makes one per turn
    and passes ``on_delta``/``on_end`` as the stream callbacks; progress and
    reasoning go through ``print_progress_line``/``on_reasoning_delta``/
    ``on_reasoning_end`` (nanobot terminal.py:276-344). When nothing streamed
    (``streamed`` is False) it calls ``close()`` and prints the whole reply,
    without the header if ``header_printed``.

    (nanobot ``stream.py:88`` ``StreamRenderer``)
    """

    def __init__(
        self,
        render_markdown: bool = True,
        show_spinner: bool = True,
        bot_name: str = BOT_NAME,
        bot_icon: str = "",
        *,
        console: Console | None = None,
        live: bool = True,
    ) -> None:
        self._md = render_markdown
        self._live_updates = live
        self._show_spinner = show_spinner and live
        self._bot_name = bot_name
        self._bot_icon = bot_icon
        self._buf = ""
        self.streamed = False
        self._console = console or _make_console()
        self._live: Live | None = None
        self._refresh_handle: asyncio.TimerHandle | None = None
        self._next_refresh_at = 0.0
        self._spinner: ThinkingSpinner | None = None
        self._header_printed = False
        self._reasoning = ReasoningBuffer()
        self._start_spinner()

    def _renderable(self) -> Markdown | Text:
        if self._md and self._buf:
            return Markdown(self._buf)
        return Text(self._buf)

    def _render_str(self) -> str:
        with self._console.capture() as cap:
            self._console.print(self._renderable())
        return cap.get()

    def _start_spinner(self) -> None:
        if self._show_spinner:
            # nanobot stream.py:134 lets the spinner make its own stdout console;
            # it shares ours so everything goes to one place.
            self._spinner = ThinkingSpinner(self._console, bot_name=self._bot_name)
            self._spinner.__enter__()

    def _stop_spinner(self) -> None:
        if self._spinner:
            self._spinner.__exit__(None, None, None)
            self._spinner = None

    @property
    def console(self) -> Console:
        return self._console

    @property
    def header_printed(self) -> bool:
        """Whether this turn has already printed the bot's name line."""
        return self._header_printed

    def ensure_header(self) -> None:
        """Stop the spinner and print the bot's name line once per turn. (nanobot ``stream.py:152``)"""
        # A turn can print progress lines, restart the spinner for tools, then
        # stream the answer: the spinner must stop even though the header exists.
        self._stop_spinner()
        if self._header_printed:
            return
        self._console.print()
        header = f"{self._bot_icon} {self._bot_name}" if self._bot_icon else self._bot_name
        self._console.print(f"[cyan]{escape(header)}[/cyan]")
        self._header_printed = True

    def pause_spinner(self) -> AbstractContextManager[None]:
        """Stop the live view and pause the spinner while a line prints. (nanobot ``stream.py:165``)"""

        @contextmanager
        def _pause() -> Iterator[None]:
            self._cancel_refresh()
            if not self._live_updates:
                self._quiet_flush()  # the text so far stays above the line
            if self._live:
                # Stop the transient live view first so it doesn't leave a partial
                # frame above the line. The next delta starts a new one from the
                # buffer; if none comes, on_end prints the buffer once.
                self._live.stop()
                self._live = None
            with self._spinner.pause() if self._spinner else nullcontext():
                yield

        return _pause()

    async def on_delta(self, delta: str) -> None:
        """Add a reply delta. Whitespace before the first visible text shows nothing.
        (nanobot ``stream.py:187``)"""
        self.streamed = True
        self._buf += delta
        if not self._live_updates:
            self._cancel_refresh()
            self._refresh_handle = asyncio.get_running_loop().call_later(_QUIET_FLUSH_SECONDS, self._quiet_flush)
            return
        if self._live is None:
            if not self._buf.strip():
                return
            self.ensure_header()
            self._live = Live(self._renderable(), console=self._console, auto_refresh=False, transient=True)
            self._live.start()
            self._live.refresh()
            self._next_refresh_at = time.monotonic() + _STREAM_REFRESH_INTERVAL
        elif self._refresh_handle is None:
            delay = max(0.0, self._next_refresh_at - time.monotonic())
            if delay == 0:
                self._refresh_stream()
            else:
                self._refresh_handle = asyncio.get_running_loop().call_later(delay, self._refresh_stream)

    def _quiet_flush(self) -> None:
        """live=False: print what streamed so far, unless a code block is open."""
        self._refresh_handle = None
        if not self._buf.strip() or self._buf.count("```") % 2:
            return
        self.ensure_header()
        out = self._console.file
        out.write(self._render_str())
        out.flush()
        self._buf = ""

    def _refresh_stream(self) -> None:
        self._refresh_handle = None
        if self._live is None:
            return
        self._live.update(self._renderable())
        self._live.refresh()
        # Leave a quiet interval even when rendering takes longer than a frame.
        self._next_refresh_at = time.monotonic() + _STREAM_REFRESH_INTERVAL

    def _cancel_refresh(self) -> None:
        if self._refresh_handle is not None:
            self._refresh_handle.cancel()
            self._refresh_handle = None

    async def on_end(self, *, resuming: bool = False) -> None:
        """End a streamed round: erase the live view and print the reply for good.
        ``resuming`` means the model continues after tools: clear the buffer and
        restart the spinner. (nanobot ``stream.py:226``)"""
        self._cancel_refresh()
        if self._live:
            # Refresh twice so Live knows its height before stop() erases it.
            self._live.refresh()
            self._live.update(self._renderable())
            self._live.refresh()
            self._live.stop()
            self._live = None
        self._stop_spinner()
        if self._buf.strip():
            self.ensure_header()  # already printed by the first delta unless live=False
            # nanobot stream.py:238 writes to sys.stdout; we write to our console's
            # file (the same thing unless a console was passed in).
            out = self._console.file
            out.write(self._render_str())
            out.flush()
        if resuming:
            self._buf = ""
            self._start_spinner()

    def print_progress_line(self, text: str) -> None:
        """A dim "↳" line under the header. Blank text prints nothing.
        (nanobot ``terminal.py:276`` ``_print_cli_progress_line``)"""
        if not text.strip():
            return
        with self.pause_spinner():
            self.ensure_header()
            self._console.print(progress_line_markup(text))

    def print_reasoning(self, text: str) -> None:
        """A dim italic "✻" reasoning line under the header. Blank text prints nothing.
        (nanobot ``terminal.py:321`` ``_print_cli_reasoning``)"""
        if not text.strip():
            return
        with self.pause_spinner():
            self.ensure_header()
            self._console.print(reasoning_markup(text))

    def on_reasoning_delta(self, delta: str) -> None:
        """Buffer a reasoning delta; print it a sentence at a time.
        (nanobot ``agent.py:235-242``)"""
        text = self._reasoning.add(delta)
        if text:
            self.print_reasoning(text)

    def on_reasoning_end(self) -> None:
        """Print what's left of the reasoning. (nanobot ``terminal.py:337`` ``_flush_cli_reasoning``)"""
        text = self._reasoning.flush()
        if text:
            self.print_reasoning(text)

    def clear_reasoning(self) -> None:
        """Drop buffered reasoning (nanobot does this when ``show_reasoning`` is off, agent.py:229)."""
        self._reasoning.clear()

    def stop_for_input(self) -> None:
        """Stop the spinner before the prompt reads input. (nanobot ``stream.py:245``)"""
        self._stop_spinner()

    def pause(self) -> AbstractContextManager[None]:
        """Pause the spinner for outside output; no-op once it has stopped. (nanobot ``stream.py:249``)"""
        if self._spinner:
            return self._spinner.pause()
        return nullcontext()

    async def close(self) -> None:
        """Stop the spinner and live view without printing the buffer. (nanobot ``stream.py:255``)"""
        self._cancel_refresh()
        if self._live:
            self._live.stop()
            self._live = None
        self._stop_spinner()
