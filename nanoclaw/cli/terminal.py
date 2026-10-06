"""Terminal input and output for the chat.

Mirrors the classic chat of nanobot/cli/terminal.py and nanobot/cli/stream.py:
a prompt_toolkit prompt with saved input history, rich for Markdown replies,
dim "↳" progress lines, and a "thinking" spinner. nanobot's streaming
renderer and reasoning display are left for phase 2 (streaming).
"""

from __future__ import annotations

import os
import select
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext, suppress
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text

# nanobot terminal.py:51
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
BOT_NAME = "nanoclaw"
# nanobot keeps it in ~/.nanobot/history/cli_history (config/paths.py:64).
DEFAULT_HISTORY_PATH = Path("~/.nanoclaw/history/cli_history")


def sanitize_surrogates(text: str) -> str:
    """Rebuild surrogate pairs and replace lone surrogates with U+FFFD.

    Some consoles (notably Windows) deliver emoji as UTF-16 surrogates, which
    can't be encoded as UTF-8 for the history file or the model request.
    (nanobot ``utils/helpers.py`` ``sanitize_surrogates``)
    """
    if not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        return text
    return text.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace")


def is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


def make_console(**kwargs: Any) -> Console:
    return Console(**kwargs)


def response_renderable(content: str, render_markdown: bool, *, render_as_text: bool = False) -> Text | Markdown:
    """Plain text for command output (keeps line breaks), else Markdown. (nanobot ``_response_renderable``)"""
    if not render_markdown or render_as_text:
        return Text(content)
    return Markdown(content)


def print_agent_response(
    console: Console,
    response: str,
    *,
    render_markdown: bool,
    render_as_text: bool = False,
) -> None:
    """A reply under the bot's name, with a blank line before and after. (nanobot ``_print_agent_response``)"""
    console.print()
    console.print(f"[cyan]{BOT_NAME}[/cyan]")
    console.print(response_renderable(response or "", render_markdown, render_as_text=render_as_text))
    console.print()


def progress_line_markup(text: str) -> str:
    """A dim "↳" line. The text is escaped: nanobot prints it as rich markup,
    so a command like ``echo "[/x]"`` crashes its turn (UPSTREAM_BUGS.md, bug 6)."""
    return f"  [dim]↳ {escape(text)}[/dim]"


def _clear_current_line(console: Console) -> None:
    """Erase the spinner's line before printing lasting output. (nanobot ``stream.py:26``)"""
    file = console.file
    if not getattr(file, "isatty", lambda: False)():
        return
    file.write("\r\x1b[2K")
    file.flush()


class ThinkingSpinner:
    """Shows '<name> is thinking...' while the agent works; pauses for output lines.

    (nanobot ``cli/stream.py`` ``ThinkingSpinner``)
    """

    def __init__(self, console: Console, bot_name: str = BOT_NAME) -> None:
        self._console = console
        self._spinner = console.status(f"[dim]{bot_name} is thinking...[/dim]", spinner="dots")
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


def print_progress_line(console: Console, text: str, spinner: ThinkingSpinner | None = None) -> None:
    """Print a progress line, pausing the spinner. Blank text prints nothing. (nanobot ``_print_cli_progress_line``)"""
    if not text.strip():
        return
    with spinner.pause() if spinner else nullcontext():
        console.print(progress_line_markup(text))


def build_key_bindings() -> Any:
    """Enter sends; Alt+Enter (or Shift+Enter where the terminal reports it) adds
    a new line. (nanobot ``_build_cli_key_bindings``, terminal.py:132)"""
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    # prompt_toolkit doesn't know the CSI-u Shift+Enter sequence; map it to a spare key.
    with suppress(Exception):
        from prompt_toolkit.input import ansi_escape_sequences as _aes

        _aes.ANSI_SEQUENCES.setdefault("\x1b[13;2u", Keys.ControlF3)

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter (ESC + CR)
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("escape", Keys.ControlJ)  # Alt+Enter on terminals that send LF
    def _newline_lf(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add(Keys.ControlF3)  # Shift+Enter (CSI-u terminals)
    def _newline_shift(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    return kb


def flush_pending_tty_input() -> None:
    """Drop keys typed while the agent was working, so they aren't sent as the
    next message. (nanobot ``_flush_pending_tty_input``, terminal.py:98)"""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return
    with suppress(Exception):
        import termios

        termios.tcflush(fd, termios.TCIFLUSH)
        return
    with suppress(Exception):
        while select.select([fd], [], [], 0)[0]:
            if not os.read(fd, 4096):
                break


_saved_term_attrs: list[Any] | None = None


def save_terminal() -> None:
    """Remember the terminal settings, to restore them on exit. (nanobot ``_init_prompt_session``)"""
    global _saved_term_attrs
    with suppress(Exception):
        import termios

        _saved_term_attrs = termios.tcgetattr(sys.stdin.fileno())


def restore_terminal() -> None:
    """Put the terminal back as it was (echo, line buffering). (nanobot ``_restore_terminal``)"""
    if _saved_term_attrs is None:
        return
    with suppress(Exception):
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _saved_term_attrs)


class PromptInput:
    """Reads input with prompt_toolkit: ↑/↓ history saved to a file, Enter to
    send, Alt+Enter for a new line, multi-line paste.

    (nanobot ``_init_prompt_session``, ``_read_interactive_input_async``)
    """

    def __init__(self, history_path: Path = DEFAULT_HISTORY_PATH, *, input: Any = None, output: Any = None) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        class SafeFileHistory(FileHistory):
            """Writes surrogate-free text (nanobot ``SafeFileHistory``, issue #2846)."""

            def store_string(self, string: str) -> None:
                super().store_string(sanitize_surrogates(string))

        save_terminal()
        path = history_path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._session: Any = PromptSession(
            history=SafeFileHistory(str(path)),
            enable_open_in_editor=False,
            multiline=True,
            key_bindings=build_key_bindings(),
            input=input,  # tests pass a pipe input and a dummy output
            output=output,
        )

    async def __call__(self) -> str:
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.patch_stdout import patch_stdout

        flush_pending_tty_input()
        try:
            with patch_stdout():
                return await self._session.prompt_async(HTML("<b fg='ansiblue'>You:</b> "))
        except EOFError as exc:  # Ctrl+D ends the chat like Ctrl+C
            raise KeyboardInterrupt from exc


async def print_above_prompt(write: Callable[[], None]) -> None:
    """Run ``write`` so its output appears above a waiting prompt. (nanobot ``run_in_terminal``)"""
    from prompt_toolkit.application import run_in_terminal

    await run_in_terminal(write)


def stdin_is_interactive() -> bool:
    return os.isatty(0) and os.isatty(1)
