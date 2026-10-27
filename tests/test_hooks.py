"""AgentHook, CompositeHook, AgentProgressHook and the turn hook builder.

Adapted from nanobot tests/agent/test_hook_composite.py, test_turn_hooks.py
and test_progress_inline_reasoning.py (runner/loop integration tests wait for
the runner wiring).
"""

from __future__ import annotations

import pytest
from loguru import logger

from nanoclaw.agent.hook import (
    AgentHook,
    AgentHookContext,
    AgentRunHookContext,
    AgentTurnHookContext,
    CompositeHook,
    SDKCaptureHook,
)
from nanoclaw.agent.progress_hook import AgentProgressHook
from nanoclaw.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
from nanoclaw.bus.outbound_events import ProgressEvent, StreamDeltaEvent, StreamEndEvent
from nanoclaw.events import AgentEvent, EventSink
from nanoclaw.providers import LLMResponse, LLMUsage, ToolCallRequest
from nanoclaw.utils.progress_events import output_events


def _ctx() -> AgentHookContext:
    return AgentHookContext(iteration=0, messages=[])


def _run_ctx() -> AgentRunHookContext:
    return AgentRunHookContext(messages=[])


def _recorder() -> tuple[list[AgentEvent], EventSink]:
    events: list[AgentEvent] = []

    async def publish(event: AgentEvent) -> None:
        events.append(event)

    return events, EventSink(publish=publish)


ALL_ASYNC = [
    "before_run", "before_iteration", "emit_reasoning", "emit_reasoning_end",
    "on_stream", "on_stream_end", "on_provider_tool_event", "before_execute_tools",
    "before_execute_tool", "after_execute_tool", "on_execute_tool_error",
    "after_iteration", "after_run", "on_error", "on_finally",
]


async def _call_all(hook: AgentHook) -> None:
    ctx, run_ctx = _ctx(), _run_ctx()
    await hook.before_run(run_ctx)
    await hook.before_iteration(ctx)
    await hook.emit_reasoning("thinking...")
    await hook.emit_reasoning_end()
    await hook.on_stream(ctx, "hi")
    await hook.on_stream_end(ctx, resuming=True)
    await hook.on_provider_tool_event(ctx, {"phase": "start"})
    await hook.before_execute_tools(ctx)
    await hook.before_execute_tool(ctx, object(), object(), {})
    await hook.after_execute_tool(ctx, object(), object(), {}, "ok")
    await hook.on_execute_tool_error(ctx, object(), object(), {}, "err")
    await hook.after_iteration(ctx)
    await hook.after_run(run_ctx)
    await hook.on_error(run_ctx)
    await hook.on_finally(run_ctx)


def _recording_hook_class(calls: list[str], label: str, fail: bool = False) -> type[AgentHook]:
    """A hook class overriding all 15 async callbacks to record (or raise)."""

    def make(name: str):
        async def method(self, *args, **kwargs):
            if fail:
                raise RuntimeError(name)
            calls.append(f"{label}:{name}")
        return method

    return type(f"Rec{label}", (AgentHook,), {name: make(name) for name in ALL_ASYNC})


# --- AgentHook base -------------------------------------------------------


async def test_base_hook_callbacks_are_noops() -> None:
    hook = AgentHook()
    assert await hook.emit_reasoning("should not raise") is None
    await _call_all(hook)
    assert hook.finalize_content(_ctx(), "x") == "x"
    assert hook.wants_streaming() is False


def test_context_defaults() -> None:
    ctx = _ctx()
    assert ctx.response is None and ctx.usage is None
    assert ctx.tool_calls == [] and ctx.tool_results == [] and ctx.tool_events == []
    assert not ctx.streamed_content and not ctx.streamed_reasoning
    assert not ctx.stream_continues_current_message
    run = _run_ctx()
    assert run.tools_used == [] and run.exception is None and run.stop_reason is None


# --- CompositeHook --------------------------------------------------------


async def test_composite_fans_out_before_iteration_in_order() -> None:
    calls: list[str] = []

    class H(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            calls.append(f"A:{context.iteration}")

    class H2(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            calls.append(f"B:{context.iteration}")

    await CompositeHook([H(), H2()]).before_iteration(_ctx())
    assert calls == ["A:0", "B:0"]


async def test_composite_fans_out_all_async_methods() -> None:
    calls: list[str] = []
    rec = _recording_hook_class(calls, "R")
    await _call_all(CompositeHook([rec(), rec()]))
    assert calls == [f"R:{name}" for name in ALL_ASYNC for _ in range(2)]


async def test_composite_isolates_errors_in_every_callback() -> None:
    calls: list[str] = []
    bad = _recording_hook_class(calls, "bad", fail=True)
    good = _recording_hook_class(calls, "good")
    await _call_all(CompositeHook([bad(), good()]))
    assert calls == [f"good:{name}" for name in ALL_ASYNC]


async def test_composite_reraises_for_reraise_hooks() -> None:
    calls: list[str] = []

    class Strict(AgentHook):
        def __init__(self) -> None:
            super().__init__(reraise=True)

        async def before_iteration(self, context: AgentHookContext) -> None:
            raise RuntimeError("strict")

    class After(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            calls.append("after")

    with pytest.raises(RuntimeError, match="strict"):
        await CompositeHook([Strict(), After()]).before_iteration(_ctx())
    assert calls == []


def test_composite_finalize_content_none_passthrough() -> None:
    assert CompositeHook([AgentHook()]).finalize_content(_ctx(), None) is None


def test_composite_finalize_content_is_a_pipeline_without_isolation() -> None:
    steps: list[str] = []

    class H1(AgentHook):
        def finalize_content(self, context, content):
            steps.append(f"H1:{content}")
            return content.upper()

    class H2(AgentHook):
        def finalize_content(self, context, content):
            steps.append(f"H2:{content}")
            return content + "!"

    assert CompositeHook([H1(), H2()]).finalize_content(_ctx(), "hi") == "HI!"
    assert steps == ["H1:hi", "H2:HI"]

    class Bad(AgentHook):
        def finalize_content(self, context, content):
            raise ValueError("surface me")

    with pytest.raises(ValueError):
        CompositeHook([Bad()]).finalize_content(_ctx(), "x")


def test_composite_wants_streaming_any() -> None:
    class Yes(AgentHook):
        def wants_streaming(self):
            return True

    assert CompositeHook([AgentHook(), Yes(), AgentHook()]).wants_streaming() is True
    assert CompositeHook([AgentHook(), AgentHook()]).wants_streaming() is False
    assert CompositeHook([]).wants_streaming() is False


def test_has_run_hooks_detects_run_snapshot_consumers() -> None:
    assert AgentHook().has_run_hooks() is False
    assert AgentProgressHook().has_run_hooks() is False
    assert SDKCaptureHook().has_run_hooks() is True
    progress = AgentProgressHook()
    assert CompositeHook([AgentHook(), CompositeHook([progress])]).has_run_hooks() is False
    assert CompositeHook([progress, CompositeHook([SDKCaptureHook()])]).has_run_hooks() is True

    class SubComposite(CompositeHook):
        async def on_finally(self, context):
            pass

    assert SubComposite([]).has_run_hooks() is True


async def test_composite_empty_hooks_no_ops() -> None:
    hook = CompositeHook([])
    assert hook.has_run_hooks() is False
    await _call_all(hook)
    assert hook.finalize_content(_ctx(), "test") == "test"


async def test_composite_supports_hook_init_without_super() -> None:
    calls: list[str] = []

    class LegacyHook(AgentHook):
        def __init__(self, label: str) -> None:
            self.label = label

        async def before_iteration(self, context: AgentHookContext) -> None:
            calls.append(self.label)

    await CompositeHook([LegacyHook("legacy")]).before_iteration(_ctx())
    assert calls == ["legacy"]


async def test_composite_can_wrap_another_composite() -> None:
    calls: list[str] = []

    class Inner(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            calls.append("inner")

    await CompositeHook([CompositeHook([Inner()])]).before_iteration(_ctx())
    assert calls == ["inner"]


async def test_sdk_capture_hook_records_iteration_and_run() -> None:
    hook = SDKCaptureHook()
    ctx = _ctx()
    ctx.tool_calls = [ToolCallRequest(id="1", name="read_file", arguments={})]
    ctx.messages.append({"role": "user", "content": "hi"})
    ctx.usage = LLMUsage(input_tokens=3, output_tokens=2)
    await hook.after_iteration(ctx)
    assert hook.tools_used == ["read_file"]
    assert hook.messages == [{"role": "user", "content": "hi"}]
    assert hook.usage.total_tokens == 5

    await hook.after_run(AgentRunHookContext(
        messages=[], tools_used=["exec"], stop_reason="completed",
    ))
    assert hook.tools_used == ["exec"]
    assert hook.stop_reason == "completed"


# --- AgentProgressHook ----------------------------------------------------


async def test_progress_hook_matches_runner_on_progress_lines() -> None:
    """Without streaming: the model's text, then the tool hints (runner._report_progress)."""
    events, sink = _recorder()
    hook = AgentProgressHook(sink)
    ctx = _ctx()
    calls = [
        ToolCallRequest(id="c1", name="read_file", arguments={"path": "notes.txt"}),
        ToolCallRequest(id="c2", name="exec", arguments={"command": "ls"}),
    ]
    ctx.response = LLMResponse(content="<think>plan</think>Let me look.", tool_calls=calls)
    ctx.tool_calls = calls
    await hook.before_execute_tools(ctx)

    assert [(e.content, e.tool_hint) for e in events] == [
        ("Let me look.", False),
        ("read notes.txt, $ ls", True),
    ]
    assert [p["name"] for p in events[1].tool_events] == ["read_file", "exec"]
    assert events[1].tool_events[0] == {
        "version": 1, "phase": "start", "call_id": "c1", "name": "read_file",
        "arguments": {"path": "notes.txt"}, "result": None, "error": None,
        "files": [], "embeds": [],
    }


async def test_progress_hook_skips_text_when_streamed() -> None:
    events, sink = _recorder()
    hook = AgentProgressHook(sink)
    ctx = _ctx()
    ctx.response = LLMResponse(content="already shown", tool_calls=[])
    ctx.streamed_content = True
    await hook.before_execute_tools(ctx)
    assert [(e.content, e.tool_hint) for e in events] == [("", True)]


async def test_progress_hook_publishes_tool_finish_payloads() -> None:
    events, sink = _recorder()
    hook = AgentProgressHook(sink)
    ctx = _ctx()
    ctx.tool_calls = [
        ToolCallRequest(id="a", name="read_file", arguments={}),
        ToolCallRequest(id="b", name="exec", arguments={}),
        ToolCallRequest(id="c", name="exec", arguments={}),
    ]
    ctx.tool_results = [{"files": ["x.png"]}, "Error: boom", None]
    ctx.tool_events = [
        {"name": "read_file", "status": "ok", "detail": ""},
        {"name": "exec", "status": "error", "detail": "boom"},
        {"name": "exec", "status": "error", "detail": ""},
    ]
    await hook.after_iteration(ctx)
    (event,) = events
    phases = [(p["phase"], p["result"], p["error"], p["files"]) for p in event.tool_events]
    assert phases == [
        ("end", {"files": ["x.png"]}, None, ["x.png"]),
        ("error", None, "Error: boom", []),
        ("error", None, "Tool execution failed", []),
    ]


async def test_progress_hook_streams_deltas_and_split_think() -> None:
    events, sink = _recorder()
    hook = AgentProgressHook(sink, streaming=True)
    assert hook.wants_streaming() is True
    ctx = _ctx()
    for chunk in ["<th", "ink>step one", "</thi", "nk>Hel", "lo"]:
        await hook.on_stream(ctx, chunk)
    await hook.on_stream_end(ctx, resuming=True)

    assert events == [
        ProgressEvent(content="step one", reasoning_delta=True),
        ProgressEvent(reasoning_end=True),
        StreamDeltaEvent(content="Hel"),
        StreamDeltaEvent(content="lo"),
        StreamEndEvent(resuming=True),
    ]
    assert ctx.streamed_reasoning is True

    # The buffer and extractor reset after a stream end.
    events.clear()
    await hook.on_stream(ctx, "<think>step one</think>again")
    assert events[0] == ProgressEvent(content="step one", reasoning_delta=True)


async def test_progress_hook_holds_back_partial_tag() -> None:
    events, sink = _recorder()
    hook = AgentProgressHook(sink, streaming=True)
    ctx = _ctx()
    await hook.on_stream(ctx, "Answer <thi")
    await hook.on_stream(ctx, "s is fine")
    assert [e.content for e in events] == ["Answer", " <this is fine"]


async def test_progress_hook_without_streaming_publishes_no_deltas() -> None:
    events, sink = _recorder()
    hook = AgentProgressHook(sink, streaming=False)
    ctx = _ctx()
    await hook.on_stream(ctx, "<think>r</think>text")
    await hook.on_stream_end(ctx, resuming=False)
    assert events == [
        ProgressEvent(content="r", reasoning_delta=True),
        ProgressEvent(reasoning_end=True),
    ]


async def test_progress_hook_stream_end_merge_next() -> None:
    events, sink = _recorder()
    hook = AgentProgressHook(sink, streaming=True)
    ctx = _ctx()
    ctx.stream_continues_current_message = True
    await hook.on_stream_end(ctx, resuming=False)
    assert events == [StreamEndEvent(resuming=False, merge_next=True)]


async def test_progress_hook_reasoning_end_only_after_open() -> None:
    events, sink = _recorder()
    hook = AgentProgressHook(sink)
    await hook.emit_reasoning_end()
    await hook.emit_reasoning("")
    assert events == []
    await hook.emit_reasoning("deep")
    await hook.emit_reasoning_end()
    await hook.emit_reasoning_end()
    assert events == [
        ProgressEvent(content="deep", reasoning_delta=True),
        ProgressEvent(reasoning_end=True),
    ]


@pytest.mark.parametrize("chunks, expected", [
    (["<think>first paragraph</think><thought>second paragraph</thought>"],
     "first paragraph\n\nsecond paragraph"),
    (["<think>first paragraph</think>", "<thought>second paragraph</thought>"],
     "first paragraph\n\nsecond paragraph"),
    (["<think>first paragraph</think><think></think><thought>second paragraph</thought>"],
     "first paragraph\n\n\n\nsecond paragraph"),
    (["<think>first paragraph</think>", "<think></think>", "<thought>second paragraph</thought>"],
     "first paragraph\n\n\n\nsecond paragraph"),
])
async def test_inline_reasoning_preserves_paragraph_separators(chunks, expected) -> None:
    events, sink = _recorder()
    hook = AgentProgressHook(sink, streaming=True)
    ctx = _ctx()
    for chunk in chunks:
        await hook.on_stream(ctx, chunk)
    reasoning = "".join(
        e.content for e in events if isinstance(e, ProgressEvent) and e.reasoning_delta
    )
    assert reasoning == expected
    assert ctx.streamed_reasoning is True


async def test_progress_hook_provider_tool_events() -> None:
    events, sink = _recorder()
    hook = AgentProgressHook(sink, streaming=True)
    ctx = _ctx()
    await hook.on_stream(ctx, "pre")
    await hook.on_provider_tool_event(ctx, {
        "phase": "start", "name": "web_search", "call_id": "h1", "arguments": {"query": "x"},
    })
    await hook.on_provider_tool_event(ctx, {
        "phase": "end", "name": "web_search", "call_id": "h1", "result": "r",
    })
    await hook.on_provider_tool_event(ctx, {"phase": "bogus", "name": "x", "call_id": "1"})
    await hook.on_provider_tool_event(ctx, {"phase": "start", "name": "", "call_id": "1"})

    assert events[:2] == [StreamDeltaEvent(content="pre"), StreamEndEvent(resuming=True)]
    start, end = events[2:]
    assert (start.content, start.tool_hint) == ('search "x"', True)
    assert start.tool_events[0]["phase"] == "start"
    assert (end.content, end.tool_hint) == ("", False)
    assert end.tool_events[0]["result"] == "r"


async def test_progress_hook_errors_propagate_through_composite() -> None:
    async def publish(event: AgentEvent) -> None:
        raise RuntimeError("progress failed")

    hook = CompositeHook([AgentProgressHook(EventSink(publish=publish)), AgentHook()])
    with pytest.raises(RuntimeError, match="progress failed"):
        await hook.before_execute_tools(_ctx())


def test_progress_hook_finalize_strips_think() -> None:
    hook = AgentProgressHook()
    assert hook.finalize_content(_ctx(), "<think>a</think>b") == "b"
    assert hook.finalize_content(_ctx(), "<think>unclosed") is None
    assert hook.finalize_content(_ctx(), None) is None
    assert hook.finalize_content(_ctx(), "") is None


async def test_progress_hook_without_sink_is_silent() -> None:
    hook = AgentProgressHook()
    ctx = _ctx()
    ctx.tool_calls = [ToolCallRequest(id="1", name="exec", arguments={"command": "ls"})]
    await hook.before_execute_tools(ctx)
    await hook.on_stream(ctx, "<think>x</think>y")
    await hook.on_stream_end(ctx, resuming=False)


# --- build_agent_turn_hook ------------------------------------------------


class RecordingHook(AgentHook):
    def __init__(self, events: list[str], label: str = "hook") -> None:
        super().__init__()
        self._events = events
        self._label = label

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._events.append(f"{self._label}:{context.iteration}")


def test_turn_hook_builder_without_extras_returns_progress_hook() -> None:
    hook = build_agent_turn_hook(AgentTurnHookSpec(streaming=True))
    assert isinstance(hook, AgentProgressHook)
    assert hook.wants_streaming() is True


@pytest.mark.parametrize("ephemeral", [True, False])
async def test_tool_log_privacy_preserves_user_visible_progress(ephemeral) -> None:
    secret = "synthetic-private-argument-测试"
    tool_events: list[dict] = []

    async def on_progress(content, **kwargs):
        tool_events.extend(kwargs.get("tool_events") or [])

    hook = build_agent_turn_hook(AgentTurnHookSpec(
        ephemeral=ephemeral, events=output_events(on_progress=on_progress),
    ))
    context = AgentHookContext(iteration=1, messages=[], tool_calls=[
        ToolCallRequest(id="private-call", name="web_search", arguments={"query": secret}),
    ])
    logs: list[str] = []
    logger.enable("nanoclaw")
    sink = logger.add(lambda message: logs.append(str(message)), format="{message}")
    try:
        await hook.before_execute_tools(context)
    finally:
        logger.remove(sink)
        logger.disable("nanoclaw")

    assert tool_events[0]["arguments"] == {"query": secret}
    assert "web_search" in "\n".join(logs)
    assert (secret in "\n".join(logs)) is not ephemeral


def test_turn_hook_context_positional_arguments(tmp_path) -> None:
    context = AgentTurnHookContext(
        None, tmp_path, "sdk", "chat-a", "message-1", "sdk:chat-a", {"trusted": True}, True,
    )
    assert context.metadata == {"trusted": True}
    assert context.ephemeral is True
    assert context.attributes == {}


async def test_turn_hook_builder_runs_progress_then_registered_then_turn() -> None:
    events: list[str] = []
    hook = build_agent_turn_hook(AgentTurnHookSpec(
        registered_hooks=[RecordingHook(events, "registered")],
        turn_hooks=[RecordingHook(events, "turn")],
    ))
    assert isinstance(hook, CompositeHook)
    assert isinstance(hook._hooks[0], AgentProgressHook)
    await hook.before_iteration(AgentHookContext(iteration=2, messages=[]))
    assert events == ["registered:2", "turn:2"]


async def test_turn_hook_builder_runs_factories_in_registration_order(tmp_path) -> None:
    events: list[str] = []
    captured: list[AgentTurnHookContext] = []

    def factory(label: str):
        def _create(context: AgentTurnHookContext) -> AgentHook:
            captured.append(context)
            return RecordingHook(events, label)
        return _create

    def broken(context: AgentTurnHookContext) -> AgentHook:
        raise RuntimeError("factory boom")

    hook = build_agent_turn_hook(AgentTurnHookSpec(
        channel="websocket",
        chat_id="chat-1",
        message_id="msg-1",
        session_key="websocket:chat-1",
        workspace=tmp_path,
        metadata={"source": "test"},
        attributes={"tenant": "acme"},
        registered_hook_factories=[factory("registered_factory"), broken, lambda c: None],
        registered_hooks=[RecordingHook(events, "registered")],
        turn_hook_factories=[factory("turn_factory")],
        turn_hooks=[RecordingHook(events, "turn")],
    ))
    await hook.before_iteration(AgentHookContext(iteration=2, messages=[]))

    assert events == ["registered_factory:2", "registered:2", "turn_factory:2", "turn:2"]
    assert captured[0] is captured[1]
    ctx = captured[0]
    assert (ctx.workspace, ctx.channel, ctx.chat_id, ctx.message_id, ctx.session_key) == (
        tmp_path, "websocket", "chat-1", "msg-1", "websocket:chat-1",
    )
    assert ctx.metadata == {"source": "test"}
    assert ctx.attributes == {"tenant": "acme"}


async def test_turn_hook_builder_skips_extras_for_ephemeral_turns_by_default() -> None:
    events: list[str] = []
    factory_calls: list[str] = []

    def factory(context: AgentTurnHookContext) -> AgentHook:
        factory_calls.append(context.channel)
        return RecordingHook(events, "factory")

    hook = build_agent_turn_hook(AgentTurnHookSpec(
        registered_hook_factories=[factory],
        registered_hooks=[RecordingHook(events)],
        ephemeral=True,
    ))
    await hook.before_iteration(AgentHookContext(iteration=1, messages=[]))
    assert events == []
    assert factory_calls == []


async def test_turn_hook_builder_can_include_extras_for_ephemeral_turns() -> None:
    events: list[str] = []
    hook = build_agent_turn_hook(AgentTurnHookSpec(
        registered_hooks=[RecordingHook(events)],
        ephemeral=True,
        run_extra_hooks_for_ephemeral=True,
    ))
    await hook.before_iteration(AgentHookContext(iteration=1, messages=[]))
    assert events == ["hook:1"]
