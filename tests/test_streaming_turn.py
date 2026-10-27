"""Layer 13: streaming turns end to end.

FakeProvider streams scripted replies through AgentLoop (runner + nanobot's
hook chain) into a recording EventSink, and into the terminal chat's
StreamRenderer on a StringIO console. The event sequences match nanobot
c4dac64's AgentRunner + AgentProgressHook for the same scripts (see the task
13.4 report).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import sys

import pytest
from rich.console import Console

from nanoclaw.__main__ import main
from nanoclaw.agent.hook import AgentHook
from nanoclaw.agent.loop import AgentLoop, ContextWindowExceededError
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.agent.tools.spawn import SpawnTool
from nanoclaw.bus.outbound_events import ProgressEvent, StreamDeltaEvent, StreamEndEvent
from nanoclaw.cli.chat import Chat
from nanoclaw.config import ChannelsConfig
from nanoclaw.events import ContextCompactionEvent, EventSink, RetryWaitEvent
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.providers.fake import ScriptedStream
from nanoclaw.session import SessionManager
from nanoclaw.utils.templates import sync_workspace_templates
from toy_tools import EchoTool


def call(text, content=None, call_id="c1"):
    return LLMResponse(
        content=content,
        tool_calls=[ToolCallRequest(id=call_id, name="echo", arguments={"text": text})],
        finish_reason="tool_calls",
    )


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    sync_workspace_templates(path)
    return path


def make_loop(ws, tmp_path, script, **kwargs):
    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = FakeProvider(list(script), retries=kwargs.pop("retries", False))
    loop = AgentLoop(
        provider, ws, tools=tools, sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"), **kwargs
    )
    return loop, provider


class Recorder:
    """An EventSink that keeps every event."""

    def __init__(self):
        self.events = []
        self.sink = EventSink(self.publish)

    async def publish(self, event):
        self.events.append(event)

    def summary(self):
        """(type, text) pairs; tool-event-only progress shows its phases."""
        out = []
        for e in self.events:
            if isinstance(e, ProgressEvent):
                if e.reasoning_end:
                    out.append(("reasoning_end", ""))
                elif e.reasoning_delta:
                    out.append(("reasoning", e.content))
                elif e.tool_hint:
                    out.append(("tool_hint", e.content))
                elif e.content:
                    out.append(("progress", e.content))
                else:
                    out.append(("tool_events", ",".join(t["phase"] for t in e.tool_events or [])))
            elif isinstance(e, StreamDeltaEvent):
                out.append(("delta", e.content))
            elif isinstance(e, StreamEndEvent):
                out.append(("end", "resuming" if e.resuming else "final"))
            elif isinstance(e, RetryWaitEvent):
                out.append(("retry_wait", e.content))
            elif isinstance(e, ContextCompactionEvent):
                out.append(("compaction", f"{e.phase}:{e.reason}:{e.notify}"))
            else:
                out.append((type(e).__name__, ""))
        return out


def think_tool_answer():
    """A reply with <think> and text before a tool call, then the answer.

    New objects each time: the runner strips <think> from the response in place.
    """
    return [call("hi", "<think>plan it</think>Let me check."), "All done."]


# --- the loop's events ---------------------------------------------------------


async def test_streaming_turn_events(ws, tmp_path):
    loop, provider = make_loop(ws, tmp_path, think_tool_answer())
    rec = Recorder()
    result = await loop.process_direct("go", events=rec.sink, streaming=True)
    assert result.final_content == "All done."
    assert [c.streamed for c in provider.calls] == [True, True]
    assert rec.summary() == [
        ("reasoning", "plan it"),
        ("reasoning_end", ""),
        # Re-cleaned buffer: a trailing space waits for the next delta (nanobot).
        ("delta", "Let"),
        ("delta", " me"),
        ("delta", " check."),
        ("end", "resuming"),
        ("tool_hint", 'echo("hi")'),
        ("tool_events", "end"),
        ("delta", "All"),
        ("delta", " done."),
        ("end", "final"),
    ]
    # The tool-hint event also announces the call (version-1 payload).
    assert rec.events[6].tool_events[0]["phase"] == "start"
    # <think> never reaches the saved history.
    saved = loop.sessions.get_or_create("cli:direct").messages
    assert [m.get("content") for m in saved if m["role"] == "assistant"] == ["Let me check.", "All done."]


async def test_non_streaming_turn_events(ws, tmp_path):
    loop, provider = make_loop(ws, tmp_path, think_tool_answer())
    rec = Recorder()
    result = await loop.process_direct("go", events=rec.sink)
    assert result.final_content == "All done."
    # Every request goes through chat_stream_with_retry, as in nanobot.
    assert [c.streamed for c in provider.calls] == [True, True]
    assert rec.summary() == [
        ("reasoning", "plan it"),
        ("reasoning_end", ""),
        ("progress", "Let me check."),
        ("tool_hint", 'echo("hi")'),
        ("tool_events", "end"),
    ]


async def test_callbacks_are_adapted_like_nanobot_process_direct(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, think_tool_answer())
    seen = []

    async def on_progress(content, *, tool_hint=False):
        seen.append(("progress", content, tool_hint))

    async def on_stream(delta):
        seen.append(("delta", delta))

    async def on_stream_end(*, resuming=False):
        seen.append(("end", resuming))

    await loop.process_direct("go", on_progress=on_progress, on_stream=on_stream, on_stream_end=on_stream_end)
    # Reasoning and tool-finish payloads are dropped for a callback that
    # doesn't take them (nanobot output_events).
    assert seen == [
        # Re-cleaned buffer: a trailing space waits for the next delta (nanobot).
        ("delta", "Let"),
        ("delta", " me"),
        ("delta", " check."),
        ("end", True),
        ("progress", 'echo("hi")', True),
        ("delta", "All"),
        ("delta", " done."),
        ("end", False),
    ]


async def test_native_reasoning_streams_as_reasoning_events(ws, tmp_path):
    script = [ScriptedStream(content_deltas=["Answer."], thinking_deltas=["<think>deep ", "thought</think>"])]
    loop, _ = make_loop(ws, tmp_path, script)
    rec = Recorder()
    await loop.process_direct("go", events=rec.sink, streaming=True)
    assert rec.summary() == [
        ("reasoning", "deep"),
        ("reasoning", " thought"),
        ("reasoning_end", ""),
        ("delta", "Answer."),
        ("end", "final"),
    ]


async def test_retry_waits_become_events(ws, tmp_path, monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    script = [LLMResponse(content="Error: 503 server error", finish_reason="error"), "Fine."]
    loop, _ = make_loop(ws, tmp_path, script, retries=True)
    rec = Recorder()
    result = await loop.process_direct("go", events=rec.sink, streaming=True)
    assert result.final_content == "Fine."
    assert rec.summary() == [
        ("retry_wait", "Model request failed, retry in 1s (attempt 1)."),
        ("delta", "Fine."),
        ("end", "final"),
    ]


async def test_stalled_stream_starts_a_new_segment(ws, tmp_path, monkeypatch):
    """A stream that times out after text is retried; the shown text ends as
    its own segment (on_stream_recover -> on_stream_end(resuming=True))."""

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    stalled = ScriptedStream(
        content_deltas=["Hal"], response=LLMResponse(content="timed out", finish_reason="error", error_kind="timeout")
    )
    loop, _ = make_loop(ws, tmp_path, [stalled, "Hello."], retries=True)
    rec = Recorder()
    await loop.process_direct("go", events=rec.sink, streaming=True)
    kinds = [k for k, _ in rec.summary()]
    assert kinds[:2] == ["delta", "end"] and rec.events[1].resuming is True
    assert ("delta", "Hello.") in rec.summary() and kinds[-1] == "end"


async def test_max_iterations_answer_is_pushed_through_the_stream(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, [call("a"), call("b", call_id="c2"), "Summary."], max_iterations=2)
    rec = Recorder()
    result = await loop.process_direct("go", events=rec.sink, streaming=True)
    assert result.stop_reason == "max_iterations"
    assert rec.summary()[-2:] == [("delta", "Summary."), ("end", "final")]


async def test_runner_calls_hooks_in_nanobot_order(ws, tmp_path):
    calls = []

    class Spy(AgentHook):
        def wants_streaming(self):
            return True

        async def before_run(self, context):
            calls.append("before_run")

        async def before_iteration(self, context):
            calls.append(f"before_iteration:{context.iteration}")

        async def on_stream(self, context, delta):
            if not calls or calls[-1] != "on_stream":
                calls.append("on_stream")

        async def on_stream_end(self, context, *, resuming):
            calls.append(f"on_stream_end:{resuming}")

        async def before_execute_tools(self, context):
            calls.append("before_execute_tools")

        async def before_execute_tool(self, context, tool_call, tool, params):
            calls.append(f"before_execute_tool:{tool_call.name}")

        async def after_execute_tool(self, context, tool_call, tool, params, result):
            calls.append(f"after_execute_tool:{result}")

        async def on_execute_tool_error(self, context, tool_call, tool, params, error):
            calls.append("on_execute_tool_error")

        async def after_iteration(self, context):
            calls.append(f"after_iteration:{[e['status'] for e in context.tool_events]}:{context.stop_reason}")

        def finalize_content(self, context, content):
            calls.append("finalize_content")
            return content

        async def after_run(self, context):
            calls.append(f"after_run:{context.stop_reason}:{len(context.tool_events)}")

        async def on_finally(self, context):
            calls.append("on_finally")

    loop, _ = make_loop(ws, tmp_path, [call("hi", "Checking."), "Done."])
    await loop.process_direct("go", hooks=[Spy()])
    assert calls == [
        "before_run",
        "before_iteration:0",
        "on_stream",
        "on_stream_end:True",
        "before_execute_tools",
        "before_execute_tool:echo",
        "after_execute_tool:hi",
        "after_iteration:['ok']:None",
        "before_iteration:1",
        "on_stream",
        "finalize_content",
        "on_stream_end:False",
        "after_iteration:[]:completed",
        "after_run:completed:1",
        "on_finally",
    ]


async def test_temporary_turns_skip_extra_hooks(ws, tmp_path):
    calls = []

    class Spy(AgentHook):
        async def before_iteration(self, context):
            calls.append("before_iteration")

    loop, _ = make_loop(ws, tmp_path, ["ok"])
    await loop.process_direct("go", ephemeral=True, hooks=[Spy()])
    assert calls == []  # nanobot turn_hooks.py:61


async def test_tool_errors_are_reported_to_hooks_and_events(ws, tmp_path):
    rec = Recorder()
    loop, _ = make_loop(ws, tmp_path, [call("x"), "ok"])
    loop.tools.get("echo").execute = _boom
    result = await loop.process_direct("go", events=rec.sink)
    assert result.final_content == "ok"
    finish = rec.events[-1].tool_events[0]
    assert finish["phase"] == "error" and "boom" in finish["error"]


async def _boom(**_kwargs):
    raise RuntimeError("boom")


async def test_runner_without_a_hook_still_strips_think_and_reports_progress():
    tools = ToolRegistry()
    tools.register(EchoTool())
    seen = []

    async def on_progress(content, *, tool_hint=False):
        seen.append((content, tool_hint))

    provider = FakeProvider([call("a", "<think>x</think>Hmm"), "<think>y</think>Done"])
    result = await AgentRunner().run(
        AgentRunSpec([{"role": "user", "content": "go"}], tools, provider, on_progress=on_progress)
    )
    assert result.final_content == "Done"
    assert seen == [("Hmm", False), ('echo("a")', True)]
    assert result.tool_events == [{"name": "echo", "status": "ok", "detail": "a"}]


# --- compaction events ---------------------------------------------------------


async def test_compact_command_emits_manual_compaction_events(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, ["a", "- likes tea"])
    await loop.process_direct("I like tea")
    rec = Recorder()
    assert await loop.compact_session(events=rec.sink) == "- likes tea"
    assert rec.summary() == [("compaction", "started:manual:True"), ("compaction", "succeeded:manual:True")]
    rec = Recorder()
    assert await loop.compact_session(events=rec.sink) == ""
    assert rec.events == []  # nothing to compact: no events (nanobot memory.py:1243)


async def test_context_pressure_emits_compaction_events(ws, tmp_path):
    from test_compaction import THIRD, two_big_turns
    from test_compaction import make_loop as compaction_loop

    window = await two_big_turns(ws, tmp_path)
    loop, _ = compaction_loop(ws, tmp_path, "- summary", "answer", window=window)
    rec = Recorder()
    await loop.process_direct(THIRD, events=rec.sink)
    assert rec.summary() == [
        ("compaction", "started:context_pressure:False"),
        ("compaction", "succeeded:context_pressure:False"),
    ]


async def test_failed_context_compaction_is_reported(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, ["a"], max_tokens=100, context_window_tokens=100 + 1024 + 50)
    rec = Recorder()

    async def no_summary(*args, **kwargs):
        return None

    loop.consolidator.summarize_transcript = no_summary
    with pytest.raises(ContextWindowExceededError):
        await loop.process_direct("hello", events=rec.sink)
    assert rec.summary() == [
        ("compaction", "started:context_pressure:False"),
        ("compaction", "failed:context_pressure:False"),
    ]


# --- the terminal chat -----------------------------------------------------------


def make_chat(ws, tmp_path, script, inputs, *, channels=None, **kwargs):
    loop, provider = make_loop(ws, tmp_path, script, **kwargs)
    out = io.StringIO()
    console = Console(file=out, width=100, color_system=None, force_terminal=False)
    pending = list(inputs)

    async def read_input():
        if not pending:
            raise EOFError
        item = pending.pop(0)
        return await item() if callable(item) else item

    chat = Chat(loop, console=console, read_input=read_input, channels=channels)
    return chat, provider, out


def text_of(out):
    return "\n".join(line.rstrip() for line in out.getvalue().split("\n"))


async def test_chat_streams_the_reply_once(ws, tmp_path):
    chat, provider, out = make_chat(ws, tmp_path, think_tool_answer(), ["go"])
    await chat.run()
    text = text_of(out)
    # The same lines nanobot's renderer prints for this turn.
    assert "\nnanoclaw\n✻ plan it\nLet me check.\n  ↳ echo(\"hi\")\nAll done.\n" in text
    assert text.count("All done.") == 1 and text.count("\nnanoclaw\n") == 1
    assert all(c.streamed for c in provider.calls)


async def test_chat_prints_a_reply_that_did_not_stream(ws, tmp_path):
    error = LLMResponse(content="Error: bad request", finish_reason="error")
    chat, _, out = make_chat(ws, tmp_path, [error], ["go"])
    await chat.run()
    assert "\nnanoclaw\nError: bad request\n\n" in text_of(out)


async def test_chat_prints_header_once_when_answer_is_printed_after_progress(ws, tmp_path):
    error = LLMResponse(content="Error: bad request", finish_reason="error")
    chat, _, out = make_chat(ws, tmp_path, [call("hi"), error], ["go"])
    await chat.run()
    text = text_of(out)
    assert "\nnanoclaw\n  ↳ echo(\"hi\")\nError: bad request\n" in text
    assert text.count("nanoclaw\n") == 1


async def test_chat_shows_retry_waits_and_markdown_option(ws, tmp_path, monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    script = [LLMResponse(content="Error: 503 server error", finish_reason="error"), "**Fine**"]
    chat, _, out = make_chat(ws, tmp_path, script, ["go"], retries=True)
    chat.render_markdown = False
    await chat.run()
    text = text_of(out)
    assert "\nnanoclaw\n  ↳ Model request failed, retry in 1s (attempt 1).\n**Fine**\n" in text


async def test_chat_channel_settings_hide_lines(ws, tmp_path):
    channels = ChannelsConfig(show_reasoning=False, send_tool_hints=False)
    script = [call("hi", "<think>plan</think>"), "Done."]
    chat, _, out = make_chat(ws, tmp_path, script, ["go"], channels=channels)
    await chat.run()
    text = text_of(out)
    assert "plan" not in text and "echo" not in text
    assert "\nnanoclaw\nDone.\n" in text


async def test_chat_compact_prints_compaction_events(ws, tmp_path):
    chat, _, out = make_chat(ws, tmp_path, ["a", "- s"], ["hi", "/compact"])
    await chat.run()
    assert "  ↳ Compressing context…\n  ↳ Context compacted.\n" in text_of(out)


async def test_subagent_follow_up_streams_in_chat(ws, tmp_path):
    done = asyncio.Event()
    chat, provider, out = make_chat(ws, tmp_path, [], [])
    chat.loop.tools.register(SpawnTool(chat.loop.subagents))
    provider.queue(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="s1", name="spawn", arguments={"task": "check", "label": "Weather"})],
            finish_reason="tool_calls",
        ),
        "On it.",
    )
    original = provider.chat_stream

    async def chat_stream(messages, *args, **kwargs):
        if messages[0]["content"].startswith("# Subagent"):
            return LLMResponse(content="Sunny.")
        return await original(messages, *args, **kwargs)

    provider.chat_stream = chat_stream
    provider.queue("It is sunny.")
    original_output = chat.output_above_prompt

    @contextlib.asynccontextmanager
    async def output_above_prompt():
        async with original_output():
            yield
        done.set()

    chat.output_above_prompt = output_above_prompt

    async def wait_then_exit():
        await asyncio.wait_for(done.wait(), 5)
        return "exit"

    chat.read_input = _reader(["weather?", wait_then_exit])
    await chat.run()
    text = text_of(out)
    assert text.index("On it.") < text.index("It is sunny.") < text.index("Goodbye!")
    # The follow-up streamed: its request had delta callbacks.
    assert provider.calls[-1].streamed


def _reader(items):
    pending = list(items)

    async def read():
        item = pending.pop(0)
        return await item() if callable(item) else item

    return read


# --- nanoclaw ask -----------------------------------------------------------------


def list_dir_call(content):
    return LLMResponse(
        content=content,
        tool_calls=[ToolCallRequest(id="c1", name="list_dir", arguments={"path": "."})],
        finish_reason="tool_calls",
    )


def test_ask_prints_reply_once_when_stdout_is_not_a_terminal(ws, capsys):
    provider = FakeProvider([list_dir_call("<think>plan.</think>Checking."), "Done."])
    assert main(["ask", "go", "-w", str(ws)], provider=provider) == 0
    captured = capsys.readouterr()
    assert captured.out == "Done.\n"
    assert "✻ plan.\n  ↳ Checking.\n  ↳ ls .\n" in captured.err


def test_ask_streams_to_a_terminal(ws, capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    provider = FakeProvider([list_dir_call("Checking."), "Done here."])
    assert main(["ask", "go", "-w", str(ws)], provider=provider) == 0
    captured = capsys.readouterr()
    # Streamed: the text before the tool, then the answer, each ending its line.
    assert captured.out == "Checking.\nDone here.\n"
    assert "  ↳ ls .\n" in captured.err and "Checking." not in captured.err


def test_ask_channel_settings(ws, capsys, tmp_path):
    import json

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"channels": {"sendProgress": False, "showReasoning": False}}), encoding="utf-8")
    provider = FakeProvider([list_dir_call("<think>plan.</think>Checking."), "Done."])
    assert main(["ask", "go", "-w", str(ws), "-c", str(config)], provider=provider) == 0
    captured = capsys.readouterr()
    assert captured.out == "Done.\n"
    assert "plan." not in captured.err and "Checking." not in captured.err and "  ↳ ls .\n" in captured.err


def test_ask_keeps_errors_on_stderr_when_streaming(ws, capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    provider = FakeProvider([LLMResponse(content="Error: bad request", finish_reason="error")])
    assert main(["ask", "go", "-w", str(ws)], provider=provider) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "Error: bad request" in captured.err
