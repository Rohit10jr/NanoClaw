"""Streaming renderer (nanoclaw/cli/stream.py).

Adapted from nanobot tests/cli/test_stream_refresh.py, test_cli_input.py
(spinner, renderer, progress-line tests), test_bot_identity.py and
test_interactive_retry_wait.py (reasoning buffer), plus output checks on a
real rich console. The expected outputs below are what nanobot's renderer
prints for the same calls (escaped lines aside: nanobot bug 6).
"""

from __future__ import annotations

import asyncio
import io
import sys
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from nanoclaw.cli import stream, terminal
from nanoclaw.cli.stream import ReasoningBuffer, StreamRenderer, ThinkingSpinner


class TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def plain_console() -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    return Console(file=out, width=80, color_system=None, force_terminal=False), out


def mock_console() -> tuple[MagicMock, MagicMock]:
    """A console whose spinner (``status``) is a mock; its file is a TTY."""
    console = MagicMock()
    console.file = TTY()
    spinner = MagicMock()
    console.status.return_value = spinner
    return console, spinner


def lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.split("\n")]


@pytest.fixture
def fixed_clock():
    # Keep a burst inside one refresh window without changing asyncio's timer clock.
    with patch.object(stream, "time") as clock:
        clock.monotonic.return_value = 0.0
        yield


# -- refresh coalescing (nanobot test_stream_refresh.py) --


async def test_burst_is_coalesced_and_flushed_during_a_pause(fixed_clock):
    with patch.object(stream, "Live") as live_class:
        live = live_class.return_value
        renderer = StreamRenderer(show_spinner=False, console=MagicMock())
        await renderer.on_delta("first")
        live.refresh.assert_called_once()
        for _ in range(100):
            await renderer.on_delta(" token")
        live.update.assert_not_called()
        await asyncio.sleep(stream._STREAM_REFRESH_INTERVAL * 2)
        live.update.assert_called_once()
        assert live.update.call_args.args[0].markup == "first" + " token" * 100
        assert live.refresh.call_count == 2
        await renderer.close()


@pytest.mark.parametrize("finish", ["end", "close", "pause"])
async def test_pending_refresh_cannot_write_after_teardown(fixed_clock, finish):
    with patch.object(stream, "Live") as live_class:
        live = live_class.return_value
        console = MagicMock()
        renderer = StreamRenderer(show_spinner=False, console=console)
        await renderer.on_delta("first")
        await renderer.on_delta(" last")
        handle = renderer._refresh_handle
        assert handle is not None
        if finish == "end":
            with patch.object(renderer, "_render_str", return_value="first last\n"):
                await renderer.on_end(resuming=True)
            console.file.write.assert_called_once_with("first last\n")
            assert renderer._buf == ""
        elif finish == "close":
            await renderer.close()
        else:
            with renderer.pause_spinner():
                pass
            assert renderer._buf == "first last"
        assert handle.cancelled()
        live.reset_mock()
        await asyncio.sleep(stream._STREAM_REFRESH_INTERVAL * 2)
        live.update.assert_not_called()
        live.refresh.assert_not_called()
        await renderer.close()


async def test_resuming_after_a_line_renders_the_buffered_tail(fixed_clock):
    with patch.object(stream, "Live") as live_class:
        live_class.side_effect = [MagicMock(), MagicMock()]
        renderer = StreamRenderer(show_spinner=False, console=MagicMock())
        await renderer.on_delta("before")
        await renderer.on_delta(" trace")
        with renderer.pause_spinner():
            pass
        await renderer.on_delta(" after")
        assert live_class.call_args.args[0].markup == "before trace after"
        await renderer.close()


# -- spinner and renderer state (nanobot test_cli_input.py) --


def test_spinner_still_importable_from_terminal():
    assert terminal.ThinkingSpinner is ThinkingSpinner
    assert terminal.progress_line_markup is stream.progress_line_markup


def test_spinner_pause_clears_its_line():
    console, _ = mock_console()
    with ThinkingSpinner(console) as thinking, thinking.pause():
        pass
    assert "\r\x1b[2K" in console.file.getvalue()


def test_spinner_shows_bot_name():
    console, _ = mock_console()
    ThinkingSpinner(console, bot_name="mybot")
    assert "mybot is thinking..." in console.status.call_args.args[0]


def test_answer_stops_spinner_even_after_header_printed():
    console, spinner = mock_console()
    renderer = StreamRenderer(show_spinner=True, console=console)
    renderer._header_printed = True
    renderer.ensure_header()
    spinner.stop.assert_called_once()
    assert "\r\x1b[2K" in console.file.getvalue()


def test_progress_line_pauses_spinner_and_opens_header_first():
    order: list[str] = []
    console, spinner = mock_console()
    spinner.start.side_effect = lambda: order.append("start")
    spinner.stop.side_effect = lambda: order.append("stop")
    console.print.side_effect = lambda *a, **k: order.append(f"print {a[0] if a else ''}")
    renderer = StreamRenderer(show_spinner=True, console=console)
    renderer.print_progress_line("tool running")
    # The pause stops the spinner, then the header stops it for good (only
    # on_end(resuming=True) restarts it), so the pause doesn't restart it.
    assert order == ["start", "stop", "stop", "print ", "print [cyan]nanoclaw[/cyan]", "print   [dim]↳ tool running[/dim]"]


def test_progress_line_stops_live_before_printing():
    live = MagicMock()
    renderer = StreamRenderer(show_spinner=False, console=plain_console()[0])
    renderer._live = live
    renderer.print_progress_line("tool running")
    live.stop.assert_called_once()
    assert renderer._live is None


def test_stop_for_input_stops_spinner():
    console, spinner = mock_console()
    renderer = StreamRenderer(show_spinner=True, console=console)
    spinner.start.assert_called_once()
    renderer.stop_for_input()
    spinner.stop.assert_called_once()


async def test_on_end_stops_live_then_writes_final_render():
    live = MagicMock()
    console = MagicMock()
    console.capture.return_value.__enter__.return_value.get.return_value = "final output\n"
    renderer = StreamRenderer(show_spinner=False, console=console)
    renderer._live = live
    renderer._buf = "final output"
    await renderer.on_end()
    live.stop.assert_called_once()
    assert renderer._live is None
    console.file.write.assert_called_once_with("final output\n")


async def test_on_end_resuming_clears_buffer_and_restarts_spinner():
    console, spinner = mock_console()
    console.capture.return_value.__enter__.return_value.get.return_value = ""
    renderer = StreamRenderer(show_spinner=True, console=console)
    renderer._buf = "some content"
    await renderer.on_end(resuming=True)
    assert renderer._buf == ""
    assert spinner.start.call_count == 2


@pytest.mark.parametrize("tty", [True, False])
def test_default_console_follows_stdout_isatty(tty):
    with patch.object(sys.stdout, "isatty", return_value=tty):
        assert stream._make_console()._force_terminal is tty


# -- reasoning buffer (nanobot test_interactive_retry_wait.py) --


def test_reasoning_buffers_until_a_sentence_ends():
    buf = ReasoningBuffer()
    assert buf.add("The") is None
    assert buf.add(" user asked.") == "The user asked."
    assert buf.add("") is None


def test_reasoning_flushes_on_newline_and_length():
    buf = ReasoningBuffer()
    assert buf.add("first line\n") == "first line"
    assert buf.add("x" * 59) is None
    assert buf.add("y") == "x" * 59 + "y"


def test_reasoning_end_flushes_and_clear_drops():
    console, out = plain_console()
    renderer = StreamRenderer(show_spinner=False, console=console)
    renderer.on_reasoning_delta("The user asked")
    assert out.getvalue() == ""
    renderer.on_reasoning_end()
    renderer.on_reasoning_delta("hidden")
    renderer.clear_reasoning()
    renderer.on_reasoning_end()
    assert lines(out.getvalue()) == ["", "nanoclaw", "✻ The user asked", ""]


# -- output on a real console (same as nanobot's for the same calls) --


async def test_plain_text_stream_output():
    console, out = plain_console()
    renderer = StreamRenderer(render_markdown=False, show_spinner=False, console=console)
    for delta in ("Hello", " world", "!\nline two"):
        await renderer.on_delta(delta)
    await renderer.on_end()
    assert renderer.streamed and renderer.header_printed
    assert lines(out.getvalue()) == ["", "nanoclaw", "Hello world!", "line two", ""]


async def test_markdown_stream_output():
    console, out = plain_console()
    renderer = StreamRenderer(show_spinner=False, console=console)
    await renderer.on_delta("Some **bold** and `code`.\n\n")
    await renderer.on_delta("- a\n- b\n")
    await renderer.on_end()
    assert lines(out.getvalue()) == ["", "nanoclaw", "Some bold and code.", "", " • a", " • b", ""]


async def test_resumed_stream_keeps_one_header_and_lines_between():
    console, out = plain_console()
    renderer = StreamRenderer(show_spinner=True, console=console)
    await renderer.on_delta("Let me check.")
    await renderer.on_end(resuming=True)
    renderer.print_progress_line('read_file("a.txt")')
    await renderer.on_delta("The file says *hi*.")
    await renderer.on_end()
    renderer.stop_for_input()
    assert lines(out.getvalue()) == [
        "",
        "nanoclaw",
        "Let me check.",
        '  ↳ read_file("a.txt")',
        "The file says hi.",
        "",
    ]


async def test_reasoning_then_answer_output():
    console, out = plain_console()
    renderer = StreamRenderer(show_spinner=False, console=console)
    for delta in ("The", " user asked", " something. Then", " more"):
        renderer.on_reasoning_delta(delta)
    renderer.on_reasoning_end()
    await renderer.on_delta("Answer.")
    await renderer.on_end()
    assert lines(out.getvalue()) == ["", "nanoclaw", "✻ The user asked something. Then more", "Answer.", ""]


async def test_progress_mid_stream_prints_before_the_buffered_reply():
    # nanobot behaviour: the live view is erased for the line; the whole reply
    # is printed at the end, below it.
    console, out = plain_console()
    renderer = StreamRenderer(show_spinner=False, console=console)
    await renderer.on_delta("Part one")
    renderer.print_progress_line("tool running")
    await renderer.on_delta(" and part two.")
    await renderer.on_end()
    assert lines(out.getvalue()) == ["", "nanoclaw", "  ↳ tool running", "Part one and part two.", ""]


@pytest.mark.parametrize("deltas", [[], ["  \n"]])
async def test_empty_or_blank_stream_prints_nothing(deltas):
    console, out = plain_console()
    renderer = StreamRenderer(show_spinner=True, console=console)
    for delta in deltas:
        await renderer.on_delta(delta)
    await renderer.on_end()
    assert out.getvalue() == ""
    assert renderer.streamed is bool(deltas)
    assert not renderer.header_printed


async def test_close_without_stream_leaves_header_state_for_the_caller():
    console, out = plain_console()
    renderer = StreamRenderer(show_spinner=True, console=console)
    renderer.print_progress_line("working")
    renderer.print_progress_line("   ")  # blank: nothing
    await renderer.close()
    assert not renderer.streamed and renderer.header_printed
    assert lines(out.getvalue()) == ["", "nanoclaw", "  ↳ working", ""]


async def test_icon_header():
    console, out = plain_console()
    renderer = StreamRenderer(show_spinner=False, bot_name="mybot", bot_icon="🤖", console=console)
    await renderer.on_delta("hi")
    await renderer.on_end()
    assert lines(out.getvalue())[:2] == ["", "🤖 mybot"]


async def test_terminal_output_erases_live_frames_then_prints_once():
    out = TTY()
    console = Console(file=out, width=20, color_system=None, force_terminal=True)
    renderer = StreamRenderer(show_spinner=False, console=console)
    await renderer.on_delta("ok")
    await renderer.on_end()
    text = out.getvalue()
    assert text.startswith("\nnanoclaw\n\x1b[?25l")  # header, then the live view hides the cursor
    assert text.endswith("\x1b[?25h\r\x1b[1A\x1b[2Kok" + " " * 18 + "\n")  # erased, printed for good


# -- markup is never interpreted (nanobot bug 6) --


async def test_markup_in_lines_is_shown_as_typed():
    console, out = plain_console()
    renderer = StreamRenderer(show_spinner=False, bot_name="[bot]", console=console)
    renderer.print_progress_line('echo "[/x]" [red]hi[/red]')  # nanobot: MarkupError
    renderer.on_reasoning_delta("use [bold]x[/bold] here.")  # nanobot: "use x here."
    await renderer.on_delta("array[0] and [/x]")
    await renderer.on_end()
    assert lines(out.getvalue()) == [
        "",
        "[bot]",
        '  ↳ echo "[/x]" [red]hi[/red]',
        "✻ use [bold]x[/bold] here.",
        "array[0] and [/x]",
        "",
    ]


async def test_markup_in_plain_text_deltas_is_shown_as_typed():
    console, out = plain_console()
    renderer = StreamRenderer(render_markdown=False, show_spinner=False, console=console)
    await renderer.on_delta("[bold]x[/bold] [/x]")
    await renderer.on_end()
    assert lines(out.getvalue())[2] == "[bold]x[/bold] [/x]"
