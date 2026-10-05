"""Terminal input and output for the chat.

Mirrors the classic chat of nanobot/cli/terminal.py and nanobot/cli/stream.py:
a prompt_toolkit prompt with saved input history, rich for Markdown replies,
dim "↳" progress lines, and a "thinking" spinner. nanobot's streaming
renderer and reasoning display are left for phase 2 (streaming).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
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
        return False

    @contextmanager
    def pause(self) -> Iterator[None]:
        """Temporarily stop the spinner so a line prints cleanly."""
        if self._active:
            self._spinner.stop()
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


class PromptInput:
    """Reads a line with prompt_toolkit: ↑/↓ history saved to a file, multi-line paste.

    (nanobot ``_init_prompt_session``, ``_read_interactive_input_async``)
    """

    def __init__(self, history_path: Path = DEFAULT_HISTORY_PATH) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        path = history_path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._session: Any = PromptSession(history=FileHistory(str(path)), enable_open_in_editor=False)

    async def __call__(self) -> str:
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.patch_stdout import patch_stdout

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
