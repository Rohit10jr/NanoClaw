"""Events, EventSink, outbound events, output_events and the think helpers.

Adapted from nanobot tests/bus/test_outbound_events.py (minus the
OutboundMessage parts; the bus comes in layer 14) and its output_events and
think-extraction coverage.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from nanoclaw.agent.messages import (
    IncrementalThinkExtractor,
    extract_think,
    strip_reasoning_tags,
    strip_think,
)
from nanoclaw.bus.outbound_events import (
    FileEditEvent,
    ProgressEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
    TurnEndEvent,
    _event_content,
)
from nanoclaw.events import (
    NO_EVENTS,
    AgentEvent,
    ContextCompactionEvent,
    EventSink,
    RetryStatusEvent,
    RetryWaitEvent,
)
from nanoclaw.providers import LLMUsage
from nanoclaw.utils.progress_events import output_events

# --- events / EventSink ---------------------------------------------------


def test_events_are_frozen_and_typed() -> None:
    event = ContextCompactionEvent("c1", "started", reason="manual")
    assert isinstance(event, AgentEvent)
    assert event.notify is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.phase = "succeeded"  # type: ignore[misc]
    status = RetryStatusEvent("waiting", 1, 3, "timeout", next_retry_at=5.0)
    assert (status.state, status.attempt, status.max_attempts) == ("waiting", 1, 3)


async def test_no_events_sink_drops_everything() -> None:
    assert NO_EVENTS.publish is None
    assert NO_EVENTS.accepts(ProgressEvent) is False
    await NO_EVENTS.emit(ProgressEvent(content="x"))


async def test_event_sink_emit_logs_errors_but_propagates_cancellation() -> None:
    async def failing(event: AgentEvent) -> None:
        raise RuntimeError("down")

    await EventSink(publish=failing).emit(RetryWaitEvent("wait"))

    async def cancelled(event: AgentEvent) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await EventSink(publish=cancelled).emit(RetryWaitEvent("wait"))


def test_event_sink_accepts_filters_by_type() -> None:
    async def publish(event: AgentEvent) -> None:
        pass

    sink = EventSink(publish=publish, accepts_type=lambda t: issubclass(t, ProgressEvent))
    assert sink.accepts(ProgressEvent) is True
    assert sink.accepts(FileEditEvent) is True
    assert sink.accepts(StreamDeltaEvent) is False
    assert EventSink(publish=publish).accepts(StreamDeltaEvent) is True


# --- outbound events ------------------------------------------------------


def test_compaction_events_have_readable_fallbacks() -> None:
    for event, expected in [
        (ContextCompactionEvent("1", "started"), "Compressing context…"),
        (ContextCompactionEvent("1", "succeeded"), "Context compacted."),
        (ContextCompactionEvent("1", "failed"), "Unable to compact context."),
        (ContextCompactionEvent("1", "cancelled"), "Context compaction cancelled."),
    ]:
        assert _event_content(event) == expected


def test_event_content_for_text_events() -> None:
    assert _event_content(ProgressEvent(content="working", tool_hint=True)) == "working"
    assert _event_content(StreamDeltaEvent(content="hel")) == "hel"
    assert _event_content(StreamEndEvent(resuming=True)) == ""
    assert _event_content(RetryWaitEvent("retrying in 2s")) == "retrying in 2s"
    assert _event_content(StreamedResponseEvent()) == ""
    assert _event_content(TurnEndEvent()) == ""


def test_progress_and_stream_event_fields() -> None:
    tool_events = [{"phase": "start", "name": "read_file"}]
    event = ProgressEvent(
        content="working", tool_hint=True, reasoning_delta=True,
        stream_id="r1", tool_events=tool_events,
    )
    assert event.tool_events == tool_events
    assert event.reasoning is False and event.reasoning_end is False
    assert isinstance(FileEditEvent(file_edit_events=[{"path": "a"}]), ProgressEvent)
    end = StreamEndEvent(stream_id="s1", resuming=True, merge_next=True)
    assert (end.resuming, end.merge_next, end.content) == (True, True, "")
    usage = LLMUsage(input_tokens=1, output_tokens=2)
    turn = TurnEndEvent(latency_ms=12, usage=usage, round_usages=(usage,))
    assert turn.outcome == "completed" and turn.usage.total_tokens == 3


# --- output_events --------------------------------------------------------


async def test_output_events_without_callbacks_returns_default() -> None:
    assert output_events() is NO_EVENTS


async def test_output_events_adapts_plain_on_progress() -> None:
    """NanoClaw's current on_progress(text, tool_hint=...) callbacks keep working."""
    calls: list[tuple] = []

    async def on_progress(text: str, tool_hint: bool = False) -> None:
        calls.append((text, tool_hint))

    sink = output_events(on_progress=on_progress)
    assert sink.accepts(ProgressEvent) is True
    assert sink.accepts(FileEditEvent) is False
    assert sink.accepts(StreamDeltaEvent) is False
    await sink.publish(ProgressEvent(content="thinking"))
    await sink.publish(ProgressEvent(content="$ ls", tool_hint=True, tool_events=[{"phase": "start"}]))
    # Finish payloads and reasoning are dropped for callbacks that can't take them.
    await sink.publish(ProgressEvent(tool_events=[{"phase": "end"}]))
    await sink.publish(ProgressEvent(content="r", reasoning_delta=True))
    await sink.publish(ProgressEvent(reasoning_end=True))
    assert calls == [("thinking", False), ("$ ls", True)]


async def test_output_events_passes_optional_fields_to_kwargs_callbacks() -> None:
    calls: list[tuple] = []
    stream: list[str] = []
    ends: list[dict] = []

    async def on_progress(text: str, **kwargs) -> None:
        calls.append((text, kwargs))

    async def on_stream(delta: str) -> None:
        stream.append(delta)

    async def on_stream_end(*, resuming: bool, merge_next: bool = False) -> None:
        ends.append({"resuming": resuming, "merge_next": merge_next})

    sink = output_events(on_progress=on_progress, on_stream=on_stream, on_stream_end=on_stream_end)
    await sink.publish(ProgressEvent(tool_events=[{"phase": "end"}]))
    await sink.publish(ProgressEvent(content="r", reasoning_delta=True))
    await sink.publish(ProgressEvent(reasoning_end=True))
    await sink.publish(StreamDeltaEvent(content="hi"))
    await sink.publish(StreamEndEvent(resuming=True, merge_next=True))
    assert calls == [
        ("", {"tool_hint": False, "tool_events": [{"phase": "end"}]}),
        ("r", {"reasoning": True}),
        ("", {"reasoning_end": True}),
    ]
    assert stream == ["hi"]
    assert ends == [{"resuming": True, "merge_next": True}]


async def test_output_events_forwards_other_events_to_default() -> None:
    seen: list[AgentEvent] = []

    async def publish(event: AgentEvent) -> None:
        seen.append(event)

    async def on_progress(text: str) -> None:
        pass

    sink = output_events(default=EventSink(publish=publish), on_progress=on_progress)
    event = RetryWaitEvent("wait")
    await sink.publish(event)
    await sink.publish(StreamDeltaEvent(content="x"))
    assert seen == [event, StreamDeltaEvent(content="x")]


# --- think helpers --------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("<think>a</think>b", "b"),
    ("<thinking>x</thinking> y", "y"),
    ("<think>open", ""),
    ("ans <tho", "ans"),
    ("ans <think", "ans"),
    ("<|channel|>hi", "hi"),
    ("<thinking/>z", "z"),
    ("mid </think> mid", "mid </think> mid"),
    ("</think>tail", "tail"),
    ("a<think广场", "a广场"),
    ("  plain  ", "plain"),
])
def test_strip_think(text: str, expected: str) -> None:
    assert strip_think(text) == expected


def test_strip_reasoning_tags_and_extract_think() -> None:
    assert strip_reasoning_tags("<think> deep </think>") == "deep"
    assert strip_reasoning_tags(None) == ""
    assert extract_think("<think>a</think><thought>b</thought>c") == ("a\n\nb", "c")
    assert extract_think("<think>open") == (None, "")
    assert extract_think("plain ") == (None, "plain")


@pytest.mark.parametrize("chunks, emitted, returns", [
    (["<think>first</think><thought>second</thought>"], ["first\n\nsecond"], [True]),
    (["<think>first</think>", "<thought>second</thought>"], ["first", "\n\nsecond"], [True, True]),
    (["<thi", "nk>abc", "</th", "ink>", "<think>def</think>"],
     ["abc", "\n\ndef"], [False, False, False, True, True]),
    (["<think> </think>", "<think>x</think>"], ["\n\nx"], [False, True]),
    (["no tags at all"], [], [False]),
])
async def test_incremental_think_extractor_on_split_streams(chunks, emitted, returns) -> None:
    extractor = IncrementalThinkExtractor()
    out: list[str] = []

    async def emit(text: str) -> None:
        out.append(text)

    buf = ""
    got = []
    for chunk in chunks:
        buf += chunk
        got.append(await extractor.feed(buf, emit))
    assert out == emitted
    assert got == returns


async def test_incremental_think_extractor_reset() -> None:
    extractor = IncrementalThinkExtractor()
    out: list[str] = []

    async def emit(text: str) -> None:
        out.append(text)

    await extractor.feed("<think>a</think>", emit)
    assert await extractor.feed("<think>a</think>", emit) is False
    extractor.reset()
    assert await extractor.feed("<think>a</think>", emit) is True
    assert out == ["a", "a"]
