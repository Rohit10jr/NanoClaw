"""Layer 10: the terminal chat, slash commands, progress lines and tool hints."""

import asyncio
import contextlib
import io
import re
import signal

import pytest
from rich.console import Console

import nanoclaw.cli.chat as chat_module
from nanoclaw.__main__ import main
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.cli.chat import Chat, build_help_text, build_status_content, command_name, invalid_command_text
from nanoclaw.cli.terminal import EXIT_COMMANDS, is_exit_command, progress_line_markup
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.utils.templates import sync_workspace_templates
from nanoclaw.utils.tool_hints import format_tool_hints
from toy_tools import EchoTool


def call(name, args, call_id="c1", content=None):
    return LLMResponse(
        content=content, tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=args)], finish_reason="tool_calls"
    )


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    sync_workspace_templates(path)
    return path


class Output(io.StringIO):
    """Console output with the padding rich adds to Markdown lines removed."""

    def getvalue(self):
        return "\n".join(line.rstrip() for line in super().getvalue().split("\n"))


def make_chat(ws, tmp_path, script, inputs, *, markdown=True, tools=None, **loop_kwargs):
    provider = FakeProvider(list(script))
    loop = AgentLoop(
        provider, ws, sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"), tools=tools, **loop_kwargs
    )
    out = Output()
    console = Console(file=out, width=120, color_system=None, force_terminal=False)
    pending = list(inputs)

    async def read_input():
        if not pending:
            raise EOFError
        item = pending.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return await item()
        return item

    chat = Chat(loop, console=console, read_input=read_input, render_markdown=markdown)
    return chat, provider, out


async def test_turns_continue_the_conversation(ws, tmp_path):
    chat, provider, out = make_chat(ws, tmp_path, ["Hi Rohit.", "Your name is Rohit."], ["My name is Rohit", "", "What's my name?"])
    await chat.run()
    text = out.getvalue()
    assert text.startswith("nanoclaw Interactive mode (fake-model) — type exit or Ctrl+C to quit\n")
    assert "\nnanoclaw\nHi Rohit.\n\n" in text
    assert "\nnanoclaw\nYour name is Rohit.\n\n" in text
    assert text.rstrip().endswith("Goodbye!")  # input ran out (Ctrl+D)
    second = provider.calls[1].messages
    assert [m["role"] for m in second[1:]] == ["user", "assistant", "user"]
    assert len(provider.calls) == 2  # the blank line was skipped


@pytest.mark.parametrize("word", sorted(EXIT_COMMANDS) + ["EXIT", "Quit"])
async def test_exit_words(ws, tmp_path, word):
    chat, provider, out = make_chat(ws, tmp_path, [], [f"  {word}  ", "never sent"])
    await chat.run()
    assert out.getvalue().rstrip().endswith("Goodbye!")
    assert provider.calls == []


async def test_ctrl_c_at_the_prompt_says_goodbye(ws, tmp_path):
    chat, provider, out = make_chat(ws, tmp_path, [], [KeyboardInterrupt()])
    await chat.run()
    assert out.getvalue().endswith("\nGoodbye!\n")


def test_exit_signal_handler(monkeypatch):
    handlers = {}
    restored = []
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    monkeypatch.setattr(chat_module, "restore_terminal", lambda: restored.append(1))
    out = io.StringIO()
    chat_module.install_exit_signal_handlers(Console(file=out, color_system=None))
    with pytest.raises(SystemExit) as exc:
        handlers[signal.SIGINT](signal.SIGINT, None)
    assert exc.value.code == 0
    assert restored == [1]  # terminal put back before leaving
    assert out.getvalue() == "\nReceived SIGINT, goodbye!\n"
    if hasattr(signal, "SIGPIPE"):  # Windows has no SIGPIPE; the app checks the same way
        assert handlers[signal.SIGPIPE] is signal.SIG_IGN


# --- progress lines and tool hints ----------------------------------------


async def test_progress_lines_show_model_text_and_tool_hints(ws, tmp_path):
    script = [
        call("exec", {"command": "echo hi"}, content="Let me check."),
        call("list_dir", {"path": "."}, call_id="c2"),
        "All done.",
    ]
    chat, _, out = make_chat(ws, tmp_path, script, ["go"])
    await chat.run()
    text = out.getvalue()
    assert "  ↳ Let me check.\n  ↳ $ echo hi\n  ↳ ls .\n" in text
    assert "\nnanoclaw\nAll done.\n" in text


async def test_progress_text_is_not_treated_as_markup(ws, tmp_path):
    """nanobot crashes on this (UPSTREAM_BUGS.md, bug 6)."""
    script = [call("exec", {"command": 'echo "[/x]" [red]hi [a-z]'}), "ok"]
    chat, _, out = make_chat(ws, tmp_path, script, ["go"])
    await chat.run()
    assert '  ↳ $ echo "[/x]" [red]hi [a-z]\n' in out.getvalue()
    assert progress_line_markup("[/x]") == "  [dim]↳ \\[/x][/dim]"


def test_tool_hints():
    def tc(name, args, i="1"):
        return ToolCallRequest(id=i, name=name, arguments=args)

    assert format_tool_hints([tc("read_file", {"path": "notes.txt"}), tc("read_file", {"path": "notes.txt"}, "2")]) == (
        "read notes.txt × 2"
    )
    assert format_tool_hints([tc("web_search", {"query": "weather in Paris"})]) == 'search "weather in Paris"'
    assert format_tool_hints([tc("web_fetch", {"url": "https://example.com/a/very/long/path/to/the/page.html"})]) == (
        "fetch example.com/…/path/to/the/page.html"
    )
    assert format_tool_hints([tc("spawn", {"task": "summarize"})]) == 'spawn("summarize")'
    assert format_tool_hints([]) == ""


async def test_runner_reports_progress_only_for_tool_calls():
    tools = ToolRegistry()
    tools.register(EchoTool())
    seen = []

    async def on_progress(content, *, tool_hint=False):
        seen.append((content, tool_hint))

    provider = FakeProvider(
        [call("echo", {"text": "a"}, content="<think>hidden</think>Thinking aloud"), call("echo", {"text": "b"}, "c2"), "done"]
    )
    await AgentRunner().run(AgentRunSpec([{"role": "user", "content": "go"}], tools, provider, on_progress=on_progress))
    assert seen == [("Thinking aloud", False), ('echo("a")', True), ('echo("b")', True)]


async def test_markdown_rendering(ws, tmp_path):
    chat, _, out = make_chat(ws, tmp_path, ["**bold** and `code`"], ["hi"])
    await chat.run()
    assert "bold and code" in out.getvalue()
    chat, _, out = make_chat(ws, tmp_path, ["**bold** and `code`"], ["hi"], markdown=False)
    await chat.run()
    assert "**bold** and `code`" in out.getvalue()


# --- slash commands --------------------------------------------------------


def test_help_text():
    assert build_help_text() == "\n".join(
        [
            "nanoclaw commands:",
            "/new — Reset this chat and start a fresh conversation.",
            "/compact — Compact this chat's context and continue the conversation.",
            "/stop — Cancel the active agent turn for this chat.",
            "/status — Display runtime, provider, and channel status.",
            "/history [n] — Print the last N persisted conversation messages.",
            "/dream — Manually trigger memory consolidation.",
            "/skill — List all enabled skills available to the agent.",
            "/help — List available slash commands.",
        ]
    )


@pytest.mark.parametrize(
    "raw, text",
    [
        # nanobot tests/agent/test_loop_save_turn.py (7e00d6a) and test_router_dispatchable.py
        ("/new now", 'Command "/new" does not accept arguments. Did you mean "/new"?'),
        ("/NEW please", 'Command "/new" does not accept arguments. Did you mean "/new"?'),
        ("/stop now", 'Command "/stop" does not accept arguments. Did you mean "/stop"?'),
        ("/status now", 'Command "/status" does not accept arguments. Did you mean "/status"?'),
        ("/STOP extra", 'Command "/stop" does not accept arguments. Did you mean "/stop"?'),
        ("/help@nanoclaw me", 'Command "/help" does not accept arguments. Did you mean "/help"?'),
        ("/history\t5", 'Invalid command "/history". Use "/help" to list available commands.'),
    ],
)
def test_registered_commands_with_wrong_arguments(raw, text):
    assert invalid_command_text(raw) == text


# nanobot tests/command/test_router_dispatchable.py::test_non_command_returns_none (7e00d6a)
NOT_COMMANDS = [
    "hello world", "/", "/tmp", "/tmp 看看这个目录", "/home/user/project",
    "/Users/alice/report.md", "/new/file.txt", "/model/preset", "/neaw",
    "/totally-unknown-command", "/tmp/file@new", "请看看 /home/user/project",
    "/nwe", "/historyx", "/xyz",
]


@pytest.mark.parametrize("content", NOT_COMMANDS)
def test_unregistered_slash_text_is_not_a_command(content):
    assert command_name(content) is None
    assert invalid_command_text(content) is None


def test_command_names_ignore_case_and_bot_suffix():
    assert command_name("  /new  ") == "/new"
    assert command_name("/NEW@nanobot") == "/new"
    assert command_name("/history 5") == "/history"
    assert command_name("/STOP extra") == "/stop"


async def test_slash_commands_never_reach_the_model(ws, tmp_path):
    chat, provider, out = make_chat(ws, tmp_path, [], ["/help", "/new now", "/HELP", "/Help@nanoclaw"])
    await chat.run()
    assert provider.calls == []
    text = out.getvalue()
    assert text.count("nanoclaw commands:") == 3
    assert 'Command "/new" does not accept arguments. Did you mean "/new"?' in text


@pytest.mark.parametrize("content", ["/tmp 看看这个目录", "/home/user/project", "/new/file.txt", "/neaw", "/tmp/foo.py explain"])
async def test_paths_and_unknown_slash_text_reach_the_model(ws, tmp_path, content):
    """nanobot tests/agent/test_loop_save_turn.py::test_absolute_path_reaches_provider_as_chat (7e00d6a)."""
    chat, provider, out = make_chat(ws, tmp_path, ["Looked."], [content])
    await chat.run()
    assert len(provider.calls) == 1
    assert any(m["role"] == "user" and content in str(m["content"]) for m in provider.calls[0].messages)
    text = out.getvalue()
    assert "Looked." in text and "Unknown command" not in text
    session = chat.loop.sessions.get_or_create(chat.session_key)
    assert session.get_history(include_runtime_context=False)[0]["content"] == content
    assert not session.messages[0].get("_command")


async def test_new_history_and_status(ws, tmp_path):
    chat, provider, out = make_chat(
        ws,
        tmp_path,
        ["Hello!", "- [ephemeral] said hi"],
        ["/history", "hi there", "/history", "/history 1", "/history abc", "/status", "/new", "/history"],
    )
    await chat.run()
    text = out.getvalue()
    assert "No conversation history yet." in text
    assert "Last 2 message(s):\n👤 You: hi there\n🤖 Bot: Hello!\n" in text  # no time block shown
    assert "Last 1 message(s):\n🤖 Bot: Hello!\n" in text
    assert "Usage: /history [count] — e.g. /history 5 (default: 10, max: 50)" in text
    # FakeProvider reports no usage, so /status shows the runner's estimate.
    assert re.search(r"🧠 Model: fake-model\n📊 Tokens: [1-9]\d* in / [1-9]\d* out\n", text)
    assert "💬 Session: 2 messages" in text and "⚡ Tasks: 0 active" in text
    assert "New session started." in text
    assert text.count("No conversation history yet.") == 2  # cleared by /new
    assert (ws / "memory" / "history.jsonl").read_text(encoding="utf-8").count("said hi") == 1  # archived first


def test_status_content_uses_last_usage():
    content = build_status_content(
        model="m",
        start_time=0,
        last_usage={"input_tokens": 1200, "output_tokens": 30},
        context_window_tokens=128_000,
        session_msg_count=4,
        context_tokens_estimate=11_900,
        active_task_count=2,
        max_completion_tokens=8192,
    )
    assert "📊 Tokens: 1200 in / 30 out" in content
    assert "📚 Context: 11k/128k (10% of input budget)" in content
    assert "⚡ Tasks: 2 active" in content


def test_status_content_shows_cache_share():
    """nanobot utils/helpers.py:880: the cached share of the last input."""
    content = build_status_content(
        model="m",
        start_time=0,
        last_usage={"input_tokens": 1000, "output_tokens": 30, "cache_read_tokens": 250},
        context_window_tokens=128_000,
        session_msg_count=4,
        context_tokens_estimate=0,
        active_task_count=0,
        max_completion_tokens=8192,
    )
    assert "📊 Tokens: 1000 in / 30 out (25% cached)" in content


async def test_last_usage_is_saved(ws, tmp_path):
    from nanoclaw.providers.base import LLMUsage

    chat, _, _ = make_chat(ws, tmp_path, [LLMResponse(content="ok", usage=LLMUsage(input_tokens=50, output_tokens=7))], ["hi"])
    await chat.run()
    meta = chat.loop.sessions.get_or_create("cli:direct").metadata
    assert meta["_last_usage"] == LLMUsage(input_tokens=50, output_tokens=7).to_dict()
    assert meta["_last_usage"]["total_tokens"] == 57


async def test_compact_skill_and_dream(ws, tmp_path):
    chat, _, out = make_chat(ws, tmp_path, ["one", "two", "- [ephemeral] summary", "Nothing to change."], ["a", "b", "/compact", "/compact", "/skill", "/dream"])
    await chat.run()
    text = out.getvalue()
    assert "  ↳ Compressing context…\n  ↳ Context compacted.\n" in text
    assert text.count("Compressing context…") == 1  # the second /compact has nothing to do
    assert "Available skills (" in text and "weather — Get current weather" in text
    assert "Dreaming..." in text and "Dream completed in" in text


async def test_stop_cancels_background_subagents(ws, tmp_path):
    gate = asyncio.Event()
    chat, provider, out = make_chat(ws, tmp_path, [call("spawn", {"task": "slow job"}), "Started."], ["go", "/stop", "/stop"])
    original = provider.chat

    async def chat_fn(messages, *args, **kwargs):
        if messages[0]["content"].startswith("# Subagent"):
            await gate.wait()
        return await original(messages, *args, **kwargs)

    provider.chat = chat_fn
    await chat.run()
    text = out.getvalue()
    assert "Stopped 1 task(s)." in text
    assert "No active task to stop." in text


async def test_background_result_is_printed_while_waiting_for_input(ws, tmp_path):
    printed = asyncio.Event()
    chat, provider, out = make_chat(
        ws, tmp_path, [call("spawn", {"task": "check weather", "label": "Weather"}), "On it.", "It is sunny."], []
    )
    original_print = chat.print_async

    async def print_async(write):
        await original_print(write)
        printed.set()

    chat.print_async = print_async

    async def wait_then_exit():
        await asyncio.wait_for(printed.wait(), 5)
        return "exit"

    child_answers = iter(["Sunny, 20C."])
    original = provider.chat

    async def chat_fn(messages, *args, **kwargs):
        if messages[0]["content"].startswith("# Subagent"):
            return LLMResponse(content=next(child_answers))
        return await original(messages, *args, **kwargs)

    provider.chat = chat_fn
    chat.read_input = make_reader(["weather?", wait_then_exit])
    await chat.run()
    text = out.getvalue()
    assert text.index("On it.") < text.index("It is sunny.") < text.index("Goodbye!")


def make_reader(items):
    pending = list(items)

    async def read():
        item = pending.pop(0)
        return await item() if callable(item) else item

    return read


def test_exit_word_check():
    assert is_exit_command("/QUIT") and is_exit_command(":q") and not is_exit_command("exit now")


# --- CLI -------------------------------------------------------------------


def test_cli_chat_with_piped_input(ws, monkeypatch, capsys):
    import sys

    seen = {}
    original = AgentLoop.__init__

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(AgentLoop, "__init__", spy)
    monkeypatch.setattr(chat_module, "install_exit_signal_handlers", lambda console: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello\n/help\nexit\n"))
    provider = FakeProvider(["Hi!"])
    assert main(["chat", "-w", str(ws), "-s", "cli:other", "--restrict", "--no-markdown"], provider=provider) == 0
    out = capsys.readouterr().out
    assert "Interactive mode" in out and "Hi!" in out and "nanoclaw commands:" in out and "Goodbye!" in out
    assert seen["restrict_to_workspace"] is True
    sessions = SessionManager(ws.resolve())
    assert sessions.get_or_create("cli:other").messages[-1]["content"] == "Hi!"


def test_cli_ask_prints_progress_on_stderr(ws, capsys):
    provider = FakeProvider([call("exec", {"command": "echo hi"}, content="Checking."), "Done."])
    assert main(["ask", "go", "-w", str(ws)], provider=provider) == 0
    captured = capsys.readouterr()
    assert captured.out == "Done.\n"
    assert "  ↳ Checking.\n  ↳ $ echo hi\n" in captured.err


def test_history_without_runtime_context(ws, tmp_path):
    from nanoclaw.runtime_context import RUNTIME_CONTEXT_TAG

    sessions = SessionManager(ws, sessions_root=tmp_path / "s")
    loop = AgentLoop(FakeProvider(["ok"]), ws, sessions=sessions)
    asyncio.run(loop.process_direct("hello"))
    session = sessions.get_or_create("cli:direct")
    assert RUNTIME_CONTEXT_TAG in session.get_history()[0]["content"]
    assert session.get_history(include_runtime_context=False)[0]["content"] == "hello"


def test_blank_progress_lines_print_nothing():
    from nanoclaw.cli.terminal import print_progress_line

    out = io.StringIO()
    console = Console(file=out, color_system=None)
    print_progress_line(console, "")
    print_progress_line(console, "   ")
    assert out.getvalue() == ""


async def test_history_limits_truncation_and_roles(ws, tmp_path):
    chat, _, out = make_chat(ws, tmp_path, [], ["/history 1000", "/history 0"])
    session = chat.loop.sessions.get_or_create("cli:direct")
    for i in range(30):
        session.add_message("user", f"question {i}")
        session.add_message("assistant", f"answer {i}")
    session.add_message("user", "x" * 250)
    session.add_message("assistant", None, tool_calls=[{"id": "t1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}])
    session.add_message("tool", "tool output", tool_call_id="t1", name="exec")
    session.add_message("assistant", "final")
    chat.loop.sessions.save(session)
    await chat.run()
    text = out.getvalue()
    assert "Last 50 message(s):" in text  # capped at 50
    assert "Last 1 message(s):\n🤖 Bot: final" in text  # at least 1
    assert "x" * 200 + "…" in text.replace("\n", "")  # long messages cut at 200 characters
    assert "x" * 201 not in text.replace("\n", "")
    assert "tool output" not in text  # only user and assistant messages


async def test_new_and_status_with_a_running_subagent(ws, tmp_path):
    gate = asyncio.Event()
    chat, provider, out = make_chat(ws, tmp_path, [call("spawn", {"task": "slow job"}), "Started."], ["go", "/status", "/new", "/status"])
    original = provider.chat

    async def chat_fn(messages, *args, **kwargs):
        if messages[0]["content"].startswith("# Subagent"):
            await gate.wait()
        return await original(messages, *args, **kwargs)

    provider.chat = chat_fn
    await chat.run()
    text = out.getvalue()
    assert text.index("⚡ Tasks: 1 active") < text.index("New session started.") < text.index("⚡ Tasks: 0 active")


async def test_compact_failure_is_reported(ws, tmp_path, monkeypatch):
    chat, _, out = make_chat(ws, tmp_path, [], ["/compact"])

    async def failed(session_key):
        return None  # archiving produced nothing

    monkeypatch.setattr(chat.loop, "compact_session", failed)
    await chat.run()
    assert "  ↳ Compressing context…\n  ↳ Unable to compact context.\n" in out.getvalue()


def test_cli_chat_markdown_option(ws, monkeypatch, capsys):
    import sys

    monkeypatch.setattr(chat_module, "install_exit_signal_handlers", lambda console: None)
    for flag, expected in (([], "Hi there"), (["--no-markdown"], "**Hi** there")):
        monkeypatch.setattr(sys, "stdin", io.StringIO("hello\n"))
        assert main(["chat", "-w", str(ws), *flag], provider=FakeProvider(["**Hi** there"])) == 0
        out = capsys.readouterr().out
        assert expected in out
        if not flag:
            assert "**Hi**" not in out


# --- second review: nanobot's terminal handling ------------------------------


@contextlib.contextmanager
def dummy_terminal():
    """A pipe for keys and a dummy screen, also as the app session's default,
    so ``patch_stdout`` never asks for a real console (on Windows that raises
    NoConsoleScreenBufferError under pytest: docs/WINDOWS.md, cause G)."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        yield pipe


async def test_prompt_enter_sends_and_alt_enter_adds_a_line(tmp_path):
    from prompt_toolkit.output import DummyOutput

    from nanoclaw.cli.terminal import PromptInput

    with dummy_terminal() as pipe:
        reader = PromptInput(tmp_path / "hist", input=pipe, output=DummyOutput())
        pipe.send_text("line one\x1b\rline two\r")
        assert await reader() == "line one\nline two"
        pipe.send_text("second\r")
        assert await reader() == "second"
        from prompt_toolkit.filters import to_filter

        assert to_filter(reader._session.multiline)()  # a multi-line input box, as nanobot
    history = (tmp_path / "hist").read_text(encoding="utf-8")
    assert "+line one\n+line two\n" in history and "+second\n" in history


def test_history_file_never_gets_broken_surrogates(tmp_path):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from nanoclaw.cli.terminal import PromptInput

    with create_pipe_input() as pipe:
        reader = PromptInput(tmp_path / "hist", input=pipe, output=DummyOutput())
        reader._session.history.store_string("bad \ud83d end")
    assert "bad � end" in (tmp_path / "hist").read_text(encoding="utf-8")


def test_sanitize_surrogates():
    from nanoclaw.cli.terminal import sanitize_surrogates

    assert sanitize_surrogates("plain ✓") == "plain ✓"
    assert sanitize_surrogates("😀") == "😀"  # a pair becomes the emoji
    assert sanitize_surrogates("a\ud800b") == "a�b"  # a lone half is replaced


async def test_input_is_sanitized_before_the_model_sees_it(ws, tmp_path):
    chat, provider, _ = make_chat(ws, tmp_path, ["ok"], ["smile 😀 and \ud800"])
    await chat.run()
    assert provider.calls[0].messages[-1]["content"].startswith("smile 😀 and �")


async def test_terminal_is_restored_on_every_exit(ws, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(chat_module, "restore_terminal", lambda: calls.append(1))
    for inputs in (["exit"], [KeyboardInterrupt()], []):
        chat, _, _ = make_chat(ws, tmp_path, [], inputs)
        await chat.run()
    assert len(calls) == 3


def test_flush_and_restore_are_safe_without_a_terminal():
    from nanoclaw.cli import terminal

    terminal.flush_pending_tty_input()  # stdin is not a TTY under pytest: nothing happens
    terminal.restore_terminal()


async def test_dream_with_nothing_to_do_is_plain_text(ws, tmp_path):
    chat, _, out = make_chat(ws, tmp_path, [], ["/dream"])
    await chat.run()
    text = out.getvalue()
    assert "Dream has no conversation history to process yet.\n\nDream reads new entries" in text
    assert "`memory/history.jsonl`" in text  # backticks kept: not rendered as Markdown


def test_spinner_clears_its_line_on_a_terminal():
    from nanoclaw.cli.terminal import ThinkingSpinner

    class TTY(io.StringIO):
        def isatty(self):
            return True

    out = TTY()
    console = Console(file=out, force_terminal=True, color_system=None)
    with ThinkingSpinner(console):
        pass
    assert out.getvalue().endswith("\r\x1b[2K")


async def test_typed_ahead_keys_are_dropped_before_each_prompt(tmp_path, monkeypatch):
    from prompt_toolkit.output import DummyOutput

    from nanoclaw.cli import terminal

    flushed = []
    monkeypatch.setattr(terminal, "flush_pending_tty_input", lambda: flushed.append(1))
    with dummy_terminal() as pipe:
        reader = terminal.PromptInput(tmp_path / "hist", input=pipe, output=DummyOutput())
        pipe.send_text("hi\r")
        await reader()
    assert flushed == [1]
