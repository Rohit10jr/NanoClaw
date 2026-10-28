"""Web UI wire projection (layer 22.1): nanoclaw/webui/outbound_wire.py and outbound_projection.py.

The encoder tests are nanobot's tests/webui/test_outbound_wire.py; the
projector tests port the send_* tests of nanobot's
channels/websocket/tests/test_websocket_channel.py onto the socket-free
projector (frames instead of a mocked connection); the notification
fixtures are nanobot's packages/client-events/fixtures.json, the shapes the
frontend validates.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import nanoclaw.webui.outbound_wire as wire_mod
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.bus.events import OUTBOUND_META_AGENT_UI, InboundMessage, OutboundMessage
from nanoclaw.bus.outbound_events import (
    ContextCompactionEvent,
    FileEditEvent,
    GoalStateSyncEvent,
    GoalStatusEvent,
    ProgressEvent,
    RecoveryStateEvent,
    RetryStatusEvent,
    RetryWaitEvent,
    RuntimeModelUpdatedEvent,
    SessionUpdatedEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
    TurnEndEvent,
    TurnModelUpdatedEvent,
    UserInputEvent,
    outbound_message_for_event,
)
from nanoclaw.bus.runtime_events import (
    RuntimeEventContext,
    SubagentTaskChanged,
    TurnCompleted,
    TurnRunStatusChanged,
    TurnRuntimeAdmitted,
)
from nanoclaw.events import CompactionReason
from nanoclaw.providers import FakeProvider, LLMResponse, LLMUsage, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.utils.templates import sync_workspace_templates
from nanoclaw.webui.metadata import (
    WEBSOCKET_TURN_OWNER_METADATA_KEY,
    WEBUI_MESSAGE_SOURCE_METADATA_KEY,
    WEBUI_SYSTEM_COMMAND_TURN_PREFIX,
    WEBUI_TURN_METADATA_KEY,
)
from nanoclaw.webui.outbound_projection import (
    WebUIOutboundProjector,
    WireFrame,
    lifecycle_outbound,
)
from nanoclaw.webui.outbound_wire import (
    PERSISTED_WIRE_EVENTS,
    TRANSCRIPT_EVENTS,
    encode_context_compaction,
    encode_error,
    encode_recovery_state,
    encode_retry_status,
    encode_turn_end,
    is_persisted_wire_event,
)
from toy_tools import EchoTool

FIXTURES = Path(__file__).parent / "fixtures" / "webui_notification_fixtures.json"


def _msg(
    event=None, *, chat_id="chat-1", content=None, metadata=None, **kwargs
) -> OutboundMessage:
    if event is None:
        return OutboundMessage(
            channel="websocket",
            chat_id=chat_id,
            content=content or "",
            metadata=dict(metadata or {}),
            **kwargs,
        )
    return outbound_message_for_event(
        channel="websocket",
        chat_id=chat_id,
        event=event,
        content=content,
        metadata=metadata,
    )


def _payloads(frames: list[WireFrame]) -> list[dict]:
    return [frame.payload for frame in frames]


# --- encoders (nanobot tests/webui/test_outbound_wire.py) ---------------------


@pytest.mark.parametrize(
    "reason", [None, "idle", "context_pressure", "manual", "provider_native"]
)
def test_encode_context_compaction_projects_the_lifecycle(
    reason: CompactionReason | None,
) -> None:
    payload = encode_context_compaction(
        "chat-1",
        ContextCompactionEvent(
            compaction_id="compact-1", phase="succeeded", reason=reason
        ),
    )
    assert payload == {
        "event": "context_compaction",
        "chat_id": "chat-1",
        "compaction_id": "compact-1",
        "phase": "succeeded",
        **({"reason": reason} if reason is not None else {}),
    }


def test_encode_recovery_state_omits_absent_optional_fields() -> None:
    payload = encode_recovery_state(
        "chat-1",
        RecoveryStateEvent(
            status="running", recovery_id="recovery-1", reason="", attempts=0
        ),
    )
    assert payload == {
        "event": "recovery_state",
        "chat_id": "chat-1",
        "status": "running",
        "recovery_id": "recovery-1",
        "attempts": 0,
    }


def test_encode_recovery_state_preserves_false_can_continue() -> None:
    payload = encode_recovery_state(
        "chat-1",
        RecoveryStateEvent(
            status="awaiting_user",
            recovery_id="recovery-1",
            reason="tool_state_unknown",
            attempts=1,
            can_continue=False,
        ),
    )
    assert payload == {
        "event": "recovery_state",
        "chat_id": "chat-1",
        "status": "awaiting_user",
        "recovery_id": "recovery-1",
        "attempts": 1,
        "reason": "tool_state_unknown",
        "can_continue": False,
    }


def test_encode_retry_status_projects_relative_wait_and_turn_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nanoclaw.webui.outbound_wire.time.time", lambda: 120.0)
    payload = encode_retry_status(
        "chat-1",
        RetryStatusEvent(
            state="waiting",
            attempt=2,
            max_attempts=4,
            error_kind="connection",
            next_retry_at=123.5,
        ),
        {WEBUI_TURN_METADATA_KEY: "turn-1", "private": "must-not-leak"},
    )
    assert payload == {
        "event": "retry_status",
        "chat_id": "chat-1",
        "turn_id": "turn-1",
        "state": "waiting",
        "attempt": 2,
        "max_attempts": 4,
        "error_kind": "connection",
        "retry_after_s": 3.5,
    }


def test_encode_retry_status_never_projects_a_negative_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nanoclaw.webui.outbound_wire.time.time", lambda: 200.0)
    payload = encode_retry_status(
        "c", RetryStatusEvent("waiting", 1, 4, "server", next_retry_at=150.0)
    )
    assert payload["retry_after_s"] == 0.0


def test_encode_turn_end_omits_absent_fields_and_private_metadata() -> None:
    payload = encode_turn_end(
        "chat-1",
        TurnEndEvent(),
        {WEBUI_TURN_METADATA_KEY: 42, "private": "must-not-leak"},
    )
    assert payload == {"event": "turn_end", "chat_id": "chat-1"}


def test_encode_turn_end_projects_complete_wire_contract() -> None:
    usage = LLMUsage.reported(input_tokens=80, output_tokens=20, cache_read_tokens=40)
    goal_state = {"active": True, "ui_summary": "Explore codebase"}
    payload = encode_turn_end(
        "chat-1",
        TurnEndEvent(
            latency_ms=1500,
            goal_state=goal_state,
            usage=usage,
            round_usages=(usage,),
            context_window_tokens=128_000,
            outcome="error",
            failure_kind="provider",
            failure_error_kind="connection",
            failure_attempts=4,
            failure_message="Connection refused",
        ),
        {WEBUI_TURN_METADATA_KEY: "turn-1", "private": "must-not-leak"},
    )
    assert payload == {
        "event": "turn_end",
        "chat_id": "chat-1",
        "turn_id": "turn-1",
        "latency_ms": 1500,
        "goal_state": goal_state,
        "usage": usage.to_turn_dict(),
        "round_usages": [usage.to_turn_dict()],
        "context_window_tokens": 128_000,
        "outcome": "error",
        "failure_kind": "provider",
        "failure_error_kind": "connection",
        "failure_attempts": 4,
        "failure_message": "Connection refused",
    }


def test_usage_turn_dict_is_the_compact_web_ui_shape() -> None:
    usage = LLMUsage.reported(
        input_tokens=80, output_tokens=20, cache_read_tokens=40, cache_write_tokens=10
    )
    assert usage.to_turn_dict() == {
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 100,
        "request_count": 1,
        "estimated_tokens": 0,
        "context_tokens": 80,
        "cached_tokens": 40,
        "cache_write_tokens": 10,
    }
    assert LLMUsage().to_turn_dict() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
        "estimated_tokens": 0,
    }


def test_encode_error_is_the_control_event_shape() -> None:
    assert encode_error("invalid chat_id") == {
        "event": "error",
        "detail": "invalid chat_id",
    }
    assert encode_error(
        "workspace_scope_rejected", reason="outside", chat_id="c", turn_id="t"
    ) == {
        "event": "error",
        "detail": "workspace_scope_rejected",
        "reason": "outside",
        "chat_id": "c",
        "turn_id": "t",
    }


def _fixture_event(fixture: dict):
    kind = fixture["event"]
    if kind == "context_compaction":
        return ContextCompactionEvent(
            compaction_id=fixture["compaction_id"],
            phase=fixture["phase"],
            reason=fixture.get("reason"),
        )
    if kind == "recovery_state":
        return RecoveryStateEvent(
            status=fixture["status"],
            recovery_id=fixture["recovery_id"],
            reason=fixture.get("reason"),
            attempts=fixture["attempts"],
            can_continue=fixture.get("can_continue"),
            failure_error_kind=fixture.get("failure_error_kind"),
        )
    next_retry_at = (
        1000.0 + fixture["retry_after_s"] if "retry_after_s" in fixture else None
    )
    return RetryStatusEvent(
        fixture["state"],
        fixture["attempt"],
        fixture.get("max_attempts"),
        fixture["error_kind"],
        next_retry_at,
    )


@pytest.mark.parametrize(
    "fixture",
    json.loads(FIXTURES.read_text()),
    ids=lambda f: f"{f['event']}-{f.get('phase') or f.get('status') or f.get('state')}",
)
def test_projector_emits_the_frontend_notification_fixtures(
    fixture: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nanoclaw.webui.outbound_wire.time.time", lambda: 1000.0)
    frames = WebUIOutboundProjector().project(
        _msg(_fixture_event(fixture), chat_id=fixture["chat_id"])
    )
    assert _payloads(frames) == [fixture]


# --- persistence rule ------------------------------------------------------------


def test_persisted_wire_events_follow_nanobot() -> None:
    assert PERSISTED_WIRE_EVENTS == {
        "message",
        "file_edit",
        "stream_end",
        "reasoning_end",
        "turn_end",
        "context_compaction",
    }
    assert TRANSCRIPT_EVENTS == PERSISTED_WIRE_EVENTS | {"user_message"}
    assert is_persisted_wire_event(
        {"event": "context_compaction", "phase": "succeeded"}
    )
    assert not is_persisted_wire_event(
        {"event": "context_compaction", "phase": "started"}
    )
    for transient in (
        "delta",
        "reasoning_delta",
        "retry_status",
        "recovery_state",
        "goal_status",
        "goal_state",
        "session_updated",
        "turn_model_updated",
        "runtime_model_updated",
        "user_message",
        "error",
    ):
        assert not is_persisted_wire_event({"event": transient})


def test_frames_carry_records_exactly_for_persisted_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nanoclaw.webui.outbound_wire.time.time", lambda: 1000.0)
    projector = WebUIOutboundProjector()
    meta = {WEBUI_TURN_METADATA_KEY: "t"}
    events = [
        None,
        ProgressEvent(content="p"),
        ProgressEvent(content="r", reasoning_delta=True, stream_id="r"),
        ProgressEvent(reasoning_end=True, stream_id="r"),
        ProgressEvent(file_edit_events=[{"path": "a"}]),
        StreamDeltaEvent(content="x", stream_id="s"),
        StreamEndEvent(stream_id="s"),
        ContextCompactionEvent(compaction_id="k", phase="started"),
        ContextCompactionEvent(compaction_id="k", phase="succeeded"),
        RetryStatusEvent("waiting", 1, 4, "server", next_retry_at=1001.0),
        RecoveryStateEvent(status="resuming", recovery_id="r"),
        GoalStatusEvent(status="running", started_at=1.0),
        GoalStateSyncEvent(goal_state={"active": False}),
        UserInputEvent(content="u", created_at_ms=1, provenance={}),
        TurnModelUpdatedEvent(model="m"),
        RuntimeModelUpdatedEvent(model="m"),
        SessionUpdatedEvent(scope="metadata"),
        TurnEndEvent(),
    ]
    frames = [
        frame
        for event in events
        for frame in projector.project(_msg(event, content="a", metadata=meta))
    ]
    assert frames
    for frame in frames:
        assert frame.persisted == is_persisted_wire_event(frame.payload), frame.payload


# --- projector: messages and progress (nanobot test_websocket_channel.py send_*) ---


def test_message_with_media_and_reply() -> None:
    frames = WebUIOutboundProjector().project(
        _msg(
            content="hello",
            reply_to="m1",
            media=["/tmp/a.png"],
            buttons=[["Yes", "No"]],
        )
    )
    [frame] = frames
    assert frame.payload == {
        "event": "message",
        "chat_id": "chat-1",
        "text": "hello",
        "media": ["/tmp/a.png"],
        "reply_to": "m1",
    }
    assert frame.audience == "chat"
    assert frame.record == frame.payload
    assert frame.phase == "answer"


def test_message_media_urls_and_rewritten_text_come_from_the_media_hooks() -> None:
    projector = WebUIOutboundProjector(
        rewrite_text=lambda text: text.replace("img.png", "/api/media/x"),
        sign_media=lambda path: (
            {"name": Path(path).name, "url": "/api/media/clip"}
            if path.endswith(".mp4")
            else None
        ),
    )
    [frame] = projector.project(
        _msg(content="![a](img.png)", media=["/tmp/clip.mp4", "/tmp/doc.txt"])
    )
    assert frame.payload["text"] == "![a](/api/media/x)"
    assert frame.payload["media_urls"] == [
        {"name": "clip.mp4", "url": "/api/media/clip"}
    ]
    # The transcript keeps the text as the agent wrote it (nanobot transcript_overrides).
    assert frame.record is not None and frame.record["text"] == "![a](img.png)"


def test_user_shell_reply_is_literal_text() -> None:
    projector = WebUIOutboundProjector(rewrite_text=lambda text: "REWRITTEN")
    [frame] = projector.project(
        _msg(content="$ ls ![x](a.png)", metadata={"_user_shell": True})
    )
    assert frame.payload["text"] == "$ ls ![x](a.png)"
    assert frame.payload["render_as"] == "text"


def test_message_latency_turn_and_agent_ui() -> None:
    blob = {"kind": "panel", "data": {"version": 1, "event": "tick", "id": "r1"}}
    [frame] = WebUIOutboundProjector().project(
        _msg(
            ProgressEvent(content="progress · panel"),
            metadata={
                OUTBOUND_META_AGENT_UI: blob,
                "latency_ms": 12.9,
                WEBUI_TURN_METADATA_KEY: "turn-1",
            },
        )
    )
    assert frame.payload["kind"] == "progress"
    assert frame.payload["agent_ui"] == blob
    assert frame.payload["latency_ms"] == 12
    assert (
        frame.payload["turn_id"],
        frame.payload["turn_phase"],
        frame.payload["turn_seq"],
    ) == ("turn-1", "activity", 1)
    assert frame.phase == "activity"


def test_progress_includes_structured_tool_events() -> None:
    tool_event = {
        "version": 1,
        "phase": "start",
        "call_id": "call-1",
        "name": "web_search",
        "arguments": {"query": "hermes", "count": 8},
        "result": None,
        "error": None,
        "files": [],
        "embeds": [],
    }
    [frame] = WebUIOutboundProjector().project(
        _msg(
            ProgressEvent(
                content='search "hermes"', tool_hint=True, tool_events=[tool_event]
            ),
            metadata={"webui_turn_id": "turn-1"},
        )
    )
    payload = frame.payload
    assert payload["event"] == "message"
    assert payload["kind"] == "tool_hint"
    assert payload["turn_id"] == "turn-1"
    assert payload["turn_phase"] == "activity"
    assert payload["turn_seq"] == 1
    assert payload["tool_events"] == [tool_event]


def test_progress_omits_binary_tool_results_from_wire_and_transcript() -> None:
    data_url = f"data:image/png;base64,{'A' * (2 * 1024 * 1024)}"
    tool_events = [
        {
            "version": 1,
            "phase": "end",
            "call_id": "call-image",
            "name": "read_file",
            "arguments": {"path": "/media/screenshot.png"},
            "result": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                    "_meta": {"path": "/media/screenshot.png"},
                },
                {"type": "text", "text": "(Image file: /media/screenshot.png)"},
            ],
            "error": None,
            "files": [],
            "embeds": [],
        }
    ]
    [frame] = WebUIOutboundProjector().project(
        _msg(ProgressEvent(content="read_file completed", tool_events=tool_events))
    )
    [wire_event] = frame.payload["tool_events"]
    assert (
        wire_event["result"][0]["image_url"]["url"]
        == "[binary content omitted from WebUI]"
    )
    assert len(json.dumps(frame.payload).encode("utf-8")) < 4096
    assert (
        frame.record is not None
        and frame.record["tool_events"] == frame.payload["tool_events"]
    )
    assert data_url not in json.dumps(frame.record)
    # The projection must not alter the result the runner sends back to the model.
    assert tool_events[0]["result"][0]["image_url"]["url"] == data_url


@pytest.mark.parametrize("event_type", [ProgressEvent, FileEditEvent])
def test_file_edit_progress_uses_file_edit_event(event_type) -> None:
    edit = {
        "version": 1,
        "phase": "start",
        "call_id": "call-1",
        "tool": "write_file",
        "path": "src/app.py",
        "added": 12,
        "deleted": 2,
        "approximate": True,
        "status": "editing",
    }
    [frame] = WebUIOutboundProjector().project(
        _msg(event_type(file_edit_events=[edit]))
    )
    assert frame.payload == {"event": "file_edit", "chat_id": "chat-1", "edits": [edit]}
    assert frame.record == frame.payload


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({}, ["progress", "tool_hint"]),
        ({"send_progress": False}, ["tool_hint"]),
        ({"send_tool_hints": False}, ["progress"]),
    ],
)
def test_channel_progress_flags(flags, expected) -> None:
    projector = WebUIOutboundProjector(**flags)
    frames = [
        *projector.project(_msg(ProgressEvent(content="Looking..."))),
        *projector.project(_msg(ProgressEvent(content="x()", tool_hint=True))),
    ]
    assert [frame.payload["kind"] for frame in frames] == expected


# --- projector: streams ------------------------------------------------------------


def test_delta_and_stream_end() -> None:
    projector = WebUIOutboundProjector()
    frames = [
        *projector.project(_msg(StreamDeltaEvent(content="part", stream_id="sid"))),
        *projector.project(_msg(StreamEndEvent(stream_id="sid"))),
    ]
    first, second = frames
    assert first.payload == {
        "event": "delta",
        "chat_id": "chat-1",
        "text": "part",
        "stream_id": "sid",
    }
    assert first.record is None
    assert second.payload == {
        "event": "stream_end",
        "chat_id": "chat-1",
        "stream_id": "sid",
    }
    # The completed segment's record carries the full text (nanobot _persist_turn_stream_event).
    assert second.record == {**second.payload, "text": "part"}


def test_delta_preserves_webui_source_metadata() -> None:
    projector = WebUIOutboundProjector()
    source = {"kind": "cron", "label": "Repo check"}
    metadata = {WEBUI_MESSAGE_SOURCE_METADATA_KEY: source}
    first, second = (
        *projector.project(
            _msg(StreamDeltaEvent(stream_id="sid"), content="done", metadata=metadata)
        ),
        *projector.project(
            _msg(StreamEndEvent(stream_id="sid"), content="", metadata=metadata)
        ),
    )
    assert first.payload["source"] == source
    assert second.payload["source"] == source
    assert second.record is not None and second.record["text"] == "done"


def test_unknown_message_source_and_bad_response_sources_are_dropped() -> None:
    projector = WebUIOutboundProjector()
    [frame] = projector.project(
        _msg(
            content="x",
            metadata={
                WEBUI_MESSAGE_SOURCE_METADATA_KEY: {"kind": "nope"},
                "response_sources": [{"provider": "openai", "model": "m"}],
            },
        )
    )
    assert "source" not in frame.payload
    assert frame.payload["response_sources"] == []
    [frame] = projector.project(
        _msg(
            content="x",
            metadata={
                "response_sources": [
                    {"provider": "p", "model": "m", "preset": "s", "secret": 1}
                ]
                * 2
            },
        )
    )
    assert frame.payload["response_sources"] == [
        {"provider": "p", "model": "m", "preset": "s", "fallback": False}
    ]


def test_stream_end_marks_resuming() -> None:
    [frame] = WebUIOutboundProjector().project(
        _msg(StreamEndEvent(stream_id="sid", resuming=True), content="partial answer")
    )
    assert frame.payload["event"] == "stream_end"
    assert frame.payload["resuming"] is True


def test_stream_keeps_buffer_across_merged_stream_boundary() -> None:
    projector = WebUIOutboundProjector()
    meta = {WEBUI_TURN_METADATA_KEY: "t"}
    frames = [
        *projector.project(
            _msg(StreamDeltaEvent(stream_id="sid"), content="first ", metadata=meta)
        ),
        *projector.project(
            _msg(
                StreamEndEvent(stream_id="sid", resuming=True, merge_next=True),
                content="",
                metadata=meta,
            )
        ),
        *projector.project(
            _msg(StreamDeltaEvent(stream_id="sid"), content="second", metadata=meta)
        ),
        *projector.project(
            _msg(StreamEndEvent(stream_id="sid"), content="", metadata=meta)
        ),
    ]
    assert frames[1].payload["merge_next"] is True
    assert frames[1].payload["resuming"] is True
    assert [f.payload["text"] for f in frames if f.payload["event"] == "delta"] == [
        "first ",
        "second",
    ]
    assert not projector.has_open_stream("chat-1")
    records = [f.record for f in frames if f.record is not None]
    assert [r["event"] for r in records] == ["stream_end", "stream_end"]
    assert [r["text"] for r in records] == ["first ", "first second"]
    assert [f.payload["turn_seq"] for f in frames] == [1, 2, 3, 4]


def test_stream_end_includes_inline_final_text() -> None:
    [frame] = WebUIOutboundProjector().project(
        _msg(StreamEndEvent(stream_id="sid"), content="merged plain text")
    )
    assert frame.payload == {
        "event": "stream_end",
        "chat_id": "chat-1",
        "stream_id": "sid",
        "text": "merged plain text",
    }


def test_stream_end_carries_rewritten_full_text() -> None:
    projector = WebUIOutboundProjector(
        rewrite_text=lambda text: text.replace("diagram.png", "/api/media/d")
    )
    frames = [
        *projector.project(
            _msg(StreamDeltaEvent(stream_id="sid"), content="![Diagram](")
        ),
        *projector.project(
            _msg(StreamDeltaEvent(stream_id="sid"), content="diagram.png)")
        ),
        *projector.project(_msg(StreamEndEvent(stream_id="sid"), content="")),
    ]
    assert frames[2].payload["text"] == "![Diagram](/api/media/d)"
    assert (
        frames[2].record is not None
        and frames[2].record["text"] == "![Diagram](/api/media/d)"
    )


def test_interleaved_streams_keep_separate_buffers() -> None:
    projector = WebUIOutboundProjector()
    for stream_id, text in (("a", "A1"), ("b", "B1"), ("a", "A2")):
        projector.project(_msg(StreamDeltaEvent(stream_id=stream_id), content=text))
    [end_b] = projector.project(_msg(StreamEndEvent(stream_id="b")))
    [end_a] = projector.project(_msg(StreamEndEvent(stream_id="a")))
    assert end_b.record is not None and end_b.record["text"] == "B1"
    assert end_a.record is not None and end_a.record["text"] == "A1A2"


def test_streamed_response_and_retry_wait_send_nothing() -> None:
    projector = WebUIOutboundProjector()
    assert projector.project(_msg(StreamedResponseEvent(), content="final")) == []
    assert (
        projector.project(
            _msg(
                RetryWaitEvent(content="Model request failed, retry in 2s (attempt 1).")
            )
        )
        == []
    )


# --- projector: reasoning ---------------------------------------------------------


def test_reasoning_delta_and_end() -> None:
    projector = WebUIOutboundProjector()
    [delta] = projector.project(
        _msg(
            ProgressEvent(
                content="step-by-step thinking", reasoning_delta=True, stream_id="r1"
            )
        )
    )
    assert delta.payload == {
        "event": "reasoning_delta",
        "chat_id": "chat-1",
        "text": "step-by-step thinking",
        "stream_id": "r1",
    }
    assert delta.record is None
    [end] = projector.project(_msg(ProgressEvent(reasoning_end=True, stream_id="r1")))
    assert end.payload == {
        "event": "reasoning_end",
        "chat_id": "chat-1",
        "stream_id": "r1",
    }
    assert end.record == {**end.payload, "text": "step-by-step thinking"}


def test_reasoning_one_shot_expands_to_delta_plus_end() -> None:
    projector = WebUIOutboundProjector()
    first, second = projector.project(
        _msg(ProgressEvent(content="thinking", reasoning=True))
    )
    assert first.payload["event"] == "reasoning_delta"
    assert first.payload["text"] == "thinking"
    assert second.payload["event"] == "reasoning_end"
    assert projector.project(_msg(ProgressEvent(content="", reasoning=True))) == []


def test_reasoning_drops_empty_chunks_and_end_without_text_has_no_text() -> None:
    projector = WebUIOutboundProjector()
    assert (
        projector.project(_msg(ProgressEvent(content="", reasoning_delta=True))) == []
    )
    [end] = projector.project(_msg(ProgressEvent(reasoning_end=True)))
    assert end.record == {"event": "reasoning_end", "chat_id": "chat-1"}


def test_reasoning_hidden_when_the_channel_does_not_show_it() -> None:
    projector = WebUIOutboundProjector(show_reasoning=False)
    assert (
        projector.project(
            _msg(ProgressEvent(content="secret", reasoning_delta=True, stream_id="r"))
        )
        == []
    )
    assert (
        projector.project(_msg(ProgressEvent(reasoning_end=True, stream_id="r"))) == []
    )
    assert (
        projector.project(_msg(ProgressEvent(content="secret", reasoning=True))) == []
    )


# --- projector: turn end and notifications ------------------------------------------


def test_turn_end_discards_unclosed_stream_buffers() -> None:
    projector = WebUIOutboundProjector()
    projector.project(
        _msg(
            StreamDeltaEvent(stream_id="s1"), content="partial", chat_id="chat-unclosed"
        )
    )
    projector.project(
        _msg(
            ProgressEvent(content="thinking", reasoning_delta=True, stream_id="r1"),
            chat_id="chat-unclosed",
        )
    )
    projector.project(
        _msg(StreamDeltaEvent(stream_id="s1"), content="kept", chat_id="chat-2")
    )
    projector.project(_msg(TurnEndEvent(), chat_id="chat-unclosed"))
    assert not projector.has_open_stream("chat-unclosed")
    assert projector.has_open_stream("chat-2")


def test_turn_end_emits_turn_end_and_session_updated() -> None:
    frames = WebUIOutboundProjector().project(_msg(TurnEndEvent()))
    assert _payloads(frames) == [
        {"event": "turn_end", "chat_id": "chat-1"},
        {"event": "session_updated", "chat_id": "chat-1", "scope": "thread"},
    ]
    assert [frame.audience for frame in frames] == ["chat", "all"]
    assert frames[0].turn_complete and frames[0].record == frames[0].payload
    assert frames[1].record is None


def test_failed_turn_end_exposes_safe_terminal_outcome() -> None:
    [end, _updated] = WebUIOutboundProjector().project(
        _msg(
            TurnEndEvent(
                outcome="failed",
                failure_kind="model",
                failure_error_kind="connection",
                failure_attempts=4,
                failure_message="Model provider request failed.",
            )
        )
    )
    assert end.payload == {
        "event": "turn_end",
        "chat_id": "chat-1",
        "outcome": "failed",
        "failure_kind": "model",
        "failure_error_kind": "connection",
        "failure_attempts": 4,
        "failure_message": "Model provider request failed.",
    }


def test_system_command_turn_end_only_refreshes_session_metadata() -> None:
    frames = WebUIOutboundProjector().project(
        _msg(
            TurnEndEvent(),
            metadata={
                WEBUI_TURN_METADATA_KEY: f"{WEBUI_SYSTEM_COMMAND_TURN_PREFIX}model-switch"
            },
        )
    )
    assert _payloads(frames) == [
        {
            "event": "turn_end",
            "chat_id": "chat-1",
            "turn_id": f"{WEBUI_SYSTEM_COMMAND_TURN_PREFIX}model-switch",
            "turn_phase": "complete",
            "turn_seq": 1,
        },
        {"event": "session_updated", "chat_id": "chat-1", "scope": "metadata"},
    ]


def test_turn_end_carries_the_owner_and_ends_the_sequence() -> None:
    projector = WebUIOutboundProjector()
    meta = {WEBUI_TURN_METADATA_KEY: "t", WEBSOCKET_TURN_OWNER_METADATA_KEY: "owner-1"}
    projector.project(_msg(content="a", metadata=meta))
    [end, _] = projector.project(_msg(TurnEndEvent(), metadata=meta))
    assert (end.turn_owner, end.payload["turn_seq"]) == ("owner-1", 2)
    [again] = projector.project(_msg(content="b", metadata=meta))
    assert again.payload["turn_seq"] == 1


def test_inbound_annotation_shares_the_turn_sequence() -> None:
    projector = WebUIOutboundProjector()
    meta = {WEBUI_TURN_METADATA_KEY: "t"}
    user_row = {"event": "user_message", "chat_id": "chat-1", "text": "hi"}
    projector.annotate("chat-1", user_row, metadata=meta, phase="user")
    assert user_row["turn_seq"] == 1 and user_row["turn_phase"] == "user"
    [frame] = projector.project(_msg(content="answer", metadata=meta))
    assert frame.payload["turn_seq"] == 2
    projector.forget_chat("chat-1")
    [frame] = projector.project(_msg(content="answer", metadata=meta))
    assert frame.payload["turn_seq"] == 1


def test_retry_status_is_transient_and_turn_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nanoclaw.webui.outbound_wire.time.time", lambda: 120.0)
    [frame] = WebUIOutboundProjector().project(
        _msg(
            RetryStatusEvent(
                state="waiting",
                attempt=2,
                max_attempts=4,
                error_kind="connection",
                next_retry_at=123.5,
            ),
            metadata={WEBUI_TURN_METADATA_KEY: "turn-1"},
        )
    )
    assert frame.payload == {
        "event": "retry_status",
        "chat_id": "chat-1",
        "turn_id": "turn-1",
        "state": "waiting",
        "attempt": 2,
        "max_attempts": 4,
        "error_kind": "connection",
        "retry_after_s": 3.5,
    }
    assert frame.record is None


def test_context_compaction_started_is_live_only() -> None:
    [frame] = WebUIOutboundProjector().project(
        _msg(
            ContextCompactionEvent(
                compaction_id="compact-started", phase="started", notify=True
            )
        )
    )
    assert frame.payload == {
        "event": "context_compaction",
        "chat_id": "chat-1",
        "compaction_id": "compact-started",
        "phase": "started",
    }
    assert frame.record is None


@pytest.mark.parametrize("phase", ["succeeded", "failed", "cancelled"])
def test_context_compaction_is_structured_persisted_and_summary_free(phase) -> None:
    [frame] = WebUIOutboundProjector().project(
        _msg(
            ContextCompactionEvent(
                compaction_id="compact-1", phase=phase, notify=True, reason="manual"
            ),
            content="Context compacted · LLM summary",
            metadata={WEBUI_TURN_METADATA_KEY: "turn-1"},
        )
    )
    expected = {
        "event": "context_compaction",
        "chat_id": "chat-1",
        "compaction_id": "compact-1",
        "phase": phase,
        "reason": "manual",
        "turn_id": "turn-1",
        "turn_phase": "activity",
        "turn_seq": 1,
    }
    assert frame.payload == expected
    assert frame.record == expected


def test_recovery_state_is_a_structured_event_not_assistant_text() -> None:
    [frame] = WebUIOutboundProjector().project(
        _msg(
            RecoveryStateEvent(
                status="awaiting_user",
                recovery_id="recovery-1",
                reason="tool_state_unknown",
                attempts=1,
            )
        )
    )
    assert frame.payload == {
        "event": "recovery_state",
        "chat_id": "chat-1",
        "status": "awaiting_user",
        "recovery_id": "recovery-1",
        "reason": "tool_state_unknown",
        "attempts": 1,
    }
    assert frame.record is None


def test_goal_status_running_and_idle() -> None:
    projector = WebUIOutboundProjector()
    meta = {
        WEBUI_TURN_METADATA_KEY: "turn-1",
        WEBSOCKET_TURN_OWNER_METADATA_KEY: "owner-1",
    }
    [running] = projector.project(
        _msg(GoalStatusEvent(status="running", started_at=1234.5), metadata=meta)
    )
    assert running.payload == {
        "event": "goal_status",
        "chat_id": "chat-1",
        "status": "running",
        "started_at": 1234.5,
        "turn_id": "turn-1",
    }
    assert not running.release_turn_owner
    [idle] = projector.project(
        _msg(GoalStatusEvent(status="idle", started_at=5.0), metadata=meta)
    )
    assert idle.payload == {
        "event": "goal_status",
        "chat_id": "chat-1",
        "status": "idle",
        "turn_id": "turn-1",
    }
    assert idle.release_turn_owner and idle.turn_owner == "owner-1"
    assert projector.project(_msg(GoalStatusEvent(status="paused"))) == []


def test_goal_state_session_updated_and_user_input() -> None:
    projector = WebUIOutboundProjector()
    [state] = projector.project(_msg(GoalStateSyncEvent(goal_state={})))
    assert state.payload == {
        "event": "goal_state",
        "chat_id": "chat-1",
        "goal_state": {"active": False},
    }
    [updated] = projector.project(_msg(SessionUpdatedEvent(scope="metadata")))
    assert updated.payload == {
        "event": "session_updated",
        "chat_id": "chat-1",
        "scope": "metadata",
    }
    assert updated.audience == "all"
    [user] = projector.project(
        _msg(
            UserInputEvent(
                content="hello from another session",
                created_at_ms=1234,
                provenance={"name": "luma"},
            )
        )
    )
    assert user.payload == {
        "event": "user_message",
        "chat_id": "chat-1",
        "text": "hello from another session",
        "created_at_ms": 1234,
        "starts_turn": False,
        "provenance": {"name": "luma"},
    }
    assert user.record is None


def test_turn_and_runtime_model_updates() -> None:
    projector = WebUIOutboundProjector()
    [turn] = projector.project(
        _msg(
            TurnModelUpdatedEvent(
                model="deepseek/deepseek-chat",
                model_preset="Deep Research",
                context_window_tokens=128_000,
            )
        )
    )
    assert turn.payload == {
        "event": "turn_model_updated",
        "chat_id": "chat-1",
        "model_name": "deepseek/deepseek-chat",
        "model_preset": "Deep Research",
        "context_window_tokens": 128_000,
    }
    [fallback] = projector.project(
        _msg(
            TurnModelUpdatedEvent(
                model="m", fallback=True, reauth_provider="openai_codex"
            )
        )
    )
    assert fallback.payload["fallback"] is True
    assert fallback.payload["reauth_provider"] == "openai_codex"
    assert projector.project(_msg(TurnModelUpdatedEvent(model="  "))) == []
    [runtime] = projector.project(
        _msg(
            RuntimeModelUpdatedEvent(model="openai/gpt-4.1", model_preset="fast"),
            chat_id="*",
        )
    )
    assert runtime.payload == {
        "event": "runtime_model_updated",
        "model_name": "openai/gpt-4.1",
        "model_preset": "fast",
    }
    assert runtime.audience == "all"
    assert (
        projector.project(_msg(RuntimeModelUpdatedEvent(model=None), chat_id="*")) == []
    )


def test_subagent_task_cards_come_from_the_hook() -> None:
    ctx = RuntimeEventContext(
        channel="websocket", chat_id="chat-1", session_key="websocket:chat-1"
    )
    event = SubagentTaskChanged(context=ctx, task_id="t1")
    assert WebUIOutboundProjector().project(_msg(event)) == []
    projector = WebUIOutboundProjector(
        subagent_task=lambda chat, task: {"id": task, "status": "running"}
    )
    [frame] = projector.project(_msg(event))
    assert frame.payload == {
        "event": "subagent_task",
        "chat_id": "chat-1",
        "task": {"id": "t1", "status": "running"},
    }


# --- lifecycle translation (nanobot session/webui_turns.py) ------------------------


def _ctx(channel="websocket", metadata=None) -> RuntimeEventContext:
    return RuntimeEventContext(
        channel=channel,
        chat_id="chat-1",
        session_key=f"{channel}:chat-1",
        metadata=dict(metadata or {}),
    )


def test_lifecycle_run_status_becomes_goal_status() -> None:
    running = lifecycle_outbound(
        TurnRunStatusChanged(context=_ctx(), status="running", started_at=12.5)
    )
    assert running is not None and running.event == GoalStatusEvent(
        status="running", started_at=12.5
    )
    now = lifecycle_outbound(
        TurnRunStatusChanged(context=_ctx(), status="running"), now=lambda: 99.0
    )
    assert now is not None and now.event == GoalStatusEvent(
        status="running", started_at=99.0
    )
    idle = lifecycle_outbound(
        TurnRunStatusChanged(context=_ctx(), status="idle", started_at=12.5)
    )
    assert idle is not None and idle.event == GoalStatusEvent(
        status="idle", started_at=None
    )
    assert (
        lifecycle_outbound(TurnRunStatusChanged(context=_ctx("cli"), status="running"))
        is None
    )


def test_lifecycle_runtime_admitted_becomes_turn_model_updated() -> None:
    runtime = SimpleNamespace(
        model="openai/gpt-4.1", model_preset="fast", context_window_tokens=64_000
    )
    msg = lifecycle_outbound(
        TurnRuntimeAdmitted(
            context=_ctx(metadata={"webui_turn_id": "t"}), runtime=runtime
        )
    )  # type: ignore[arg-type]
    assert msg is not None
    assert msg.event == TurnModelUpdatedEvent(
        model="openai/gpt-4.1", model_preset="fast", context_window_tokens=64_000
    )
    assert msg.metadata == {"webui_turn_id": "t"}


def test_lifecycle_turn_completed_becomes_turn_end() -> None:
    usage = LLMUsage.reported(input_tokens=10, output_tokens=2)
    runtime = SimpleNamespace(context_window_tokens=64_000)
    msg = lifecycle_outbound(
        TurnCompleted(
            context=_ctx(),
            latency_ms=5,
            runtime=runtime,
            usage=usage,
            round_usages=(usage,),
        )  # type: ignore[arg-type]
    )
    assert msg is not None
    assert msg.event == TurnEndEvent(
        latency_ms=5,
        goal_state={"active": False},
        usage=usage,
        round_usages=(usage,),
        context_window_tokens=64_000,
    )
    failed = lifecycle_outbound(
        TurnCompleted(
            context=_ctx(),
            outcome="failed",
            failure_kind="model",
            failure_error_kind="server",
            failure_attempts=4,
        )
    )
    assert failed is not None and isinstance(failed.event, TurnEndEvent)
    assert (
        failed.event.failure_message is not None
        and failed.event.failure_message.startswith("Model provider")
    )
    internal = lifecycle_outbound(
        TurnCompleted(context=_ctx(), outcome="failed", failure_kind="internal")
    )
    assert (
        internal is not None
        and internal.event.failure_message == "This turn failed and has ended."
    )  # type: ignore[union-attr]


# --- a real turn through the loop -------------------------------------------------


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    sync_workspace_templates(path)
    return path


def _call(text: str, content: str | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[ToolCallRequest(id="c1", name="echo", arguments={"text": text})],
        finish_reason="tool_calls",
    )


async def _run_websocket_turn(
    ws, tmp_path, script
) -> tuple[list[OutboundMessage], AgentLoop]:
    """One websocket turn over the bus; the runtime events are translated the
    way layer 23's coordinator will (publish_outbound from the handler), so the
    outbound queue holds nanobot's order."""
    tools = ToolRegistry()
    tools.register(EchoTool())
    loop = AgentLoop(
        FakeProvider(script),
        ws,
        model="fake-model",
        tools=tools,
        sessions=SessionManager(ws, sessions_root=tmp_path / "s"),
    )

    async def translate(event) -> None:
        if (msg := lifecycle_outbound(event, now=lambda: 1.0)) is not None:
            await loop.bus.publish_outbound(msg)

    for event_type in (TurnRunStatusChanged, TurnRuntimeAdmitted, TurnCompleted):
        loop.bus.subscribe(translate, event_type)
    outbound: list[OutboundMessage] = []
    runner = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(
            InboundMessage(
                channel="websocket",
                sender_id="user",
                chat_id="chat-1",
                content="go",
                metadata={
                    "_wants_stream": True,
                    WEBUI_TURN_METADATA_KEY: "turn-1",
                    "webui": True,
                },
            )
        )
        async with asyncio.timeout(10):
            while True:
                msg = await loop.bus.consume_outbound()
                outbound.append(msg)
                if (
                    isinstance(msg.event, GoalStatusEvent)
                    and msg.event.status == "idle"
                ):
                    break
    finally:
        loop.stop()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
    return outbound, loop


def _strip(payload: dict) -> dict:
    return {
        k: v
        for k, v in payload.items()
        if k not in {"chat_id", "turn_id", "stream_id", "tool_events"}
    }


async def test_scripted_websocket_turn_projects_nanobots_wire_sequence(
    ws, tmp_path
) -> None:
    script = [_call("hi", "<think>plan it</think>Let me check."), "All done."]
    outbound, loop = await _run_websocket_turn(ws, tmp_path, script)
    projector = WebUIOutboundProjector()
    frames = [frame for msg in outbound for frame in projector.project(msg)]
    wire = [_strip(frame.payload) for frame in frames]
    model = loop.runtime.model
    window = loop.runtime.context_window_tokens
    # The runtime is admitted while the turn is built, before it runs (nanobot loop.py:2250, :2322).
    assert wire[:2] == [
        {
            "event": "turn_model_updated",
            "model_name": model,
            "context_window_tokens": window,
        },
        {
            "event": "goal_status",
            "status": "running",
            "started_at": wire[1]["started_at"],
        },
    ]
    events = [
        (w["event"], w["text"] if "text" in w else w.get("state"), w.get("kind"))
        for w in wire[2:]
    ]
    assert events == [
        # Each model request starts a fresh retry chain (nanobot base.py:1371).
        ("retry_status", "cleared", None),
        ("reasoning_delta", "plan it", None),
        ("reasoning_end", None, None),
        ("delta", "Let", None),
        ("delta", " me", None),
        ("delta", " check.", None),
        ("stream_end", None, None),
        ("message", 'echo("hi")', "tool_hint"),
        ("message", "", "progress"),
        ("retry_status", "cleared", None),
        ("delta", "All", None),
        ("delta", " done.", None),
        ("stream_end", None, None),
        ("turn_end", None, None),
        ("session_updated", None, None),
        ("goal_status", None, None),
    ]
    by_event = {frame.payload["event"]: frame.payload for frame in frames}
    # Every turn-scoped frame is numbered within turn-1, in order.
    seqs = [f.payload["turn_seq"] for f in frames if "turn_seq" in f.payload]
    assert seqs == list(range(1, len(seqs) + 1))
    assert all(
        f.payload.get("turn_id") == "turn-1"
        for f in frames
        if f.payload["event"] not in {"session_updated", "turn_model_updated"}
    )
    # The tool hint announces the call, the progress row ends it.
    tool_frames = [
        f.payload for f in frames if f.payload.get("kind") in ("tool_hint", "progress")
    ]
    assert [t["tool_events"][0]["phase"] for t in tool_frames] == ["start", "end"]
    # turn_end reports the two requests' usage.
    turn_end = by_event["turn_end"]
    assert turn_end["latency_ms"] >= 0
    assert turn_end["goal_state"] == {"active": False}
    assert turn_end["context_window_tokens"] == window
    assert len(turn_end["round_usages"]) == 2
    assert turn_end["usage"]["request_count"] == 2
    assert "outcome" not in turn_end
    # Persisted subset: reasoning, both completed segments, the tool rows, turn_end.
    records = [frame.record for frame in frames if frame.record is not None]
    assert [(r["event"], r.get("text")) for r in records] == [
        ("reasoning_end", "plan it"),
        ("stream_end", "Let me check."),
        ("message", 'echo("hi")'),
        ("message", ""),
        ("stream_end", "All done."),
        ("turn_end", None),
    ]


async def test_failed_websocket_turn_ends_with_safe_failure_and_no_error_reply(
    ws, tmp_path
) -> None:
    script = [
        LLMResponse(
            content="Error: invalid api key",
            finish_reason="error",
            error_status_code=401,
        )
    ]
    outbound, _loop = await _run_websocket_turn(ws, tmp_path, script)
    frames = [
        frame for msg in outbound for frame in WebUIOutboundProjector().project(msg)
    ]
    names = [frame.payload["event"] for frame in frames]
    # The web UI shows the failed turn_end, not the provider's error text.
    assert "message" not in names
    turn_end = next(
        frame.payload for frame in frames if frame.payload["event"] == "turn_end"
    )
    assert turn_end["outcome"] == "failed"
    assert turn_end["failure_kind"] == "model"
    assert turn_end["failure_error_kind"] == "authentication"
    assert turn_end["failure_message"].startswith("Model provider request failed.")
    assert "failure_attempts" not in turn_end


async def test_websocket_turn_reports_retry_status_and_exhaustion(
    ws, tmp_path, monkeypatch
) -> None:
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setattr(wire_mod.time, "time", lambda: 1000.0)
    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = FakeProvider(
        [
            LLMResponse(
                content="Error: 503", finish_reason="error", error_status_code=503
            )
        ]
        * 4,
        retries=True,
    )
    loop = AgentLoop(
        provider,
        ws,
        tools=tools,
        sessions=SessionManager(ws, sessions_root=tmp_path / "s"),
    )

    async def translate(event) -> None:
        if (msg := lifecycle_outbound(event, now=lambda: 1.0)) is not None:
            await loop.bus.publish_outbound(msg)

    loop.bus.subscribe(translate, TurnCompleted)
    outbound: list[OutboundMessage] = []
    runner = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(
            InboundMessage(
                channel="websocket",
                sender_id="u",
                chat_id="chat-1",
                content="go",
                metadata={WEBUI_TURN_METADATA_KEY: "turn-1"},
            )
        )
        async with asyncio.timeout(10):
            while True:
                msg = await loop.bus.consume_outbound()
                outbound.append(msg)
                if isinstance(msg.event, TurnEndEvent):
                    break
    finally:
        loop.stop()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
    projector = WebUIOutboundProjector()
    statuses = [
        (f.payload["state"], f.payload["attempt"])
        for msg in outbound
        for f in projector.project(msg)
        if f.payload["event"] == "retry_status"
    ]
    assert statuses == [
        ("cleared", 1),
        ("waiting", 1),
        ("waiting", 2),
        ("waiting", 3),
        ("exhausted", 4),
    ]
    turn_end = [m.event for m in outbound if isinstance(m.event, TurnEndEvent)][0]
    assert (
        turn_end.failure_kind,
        turn_end.failure_error_kind,
        turn_end.failure_attempts,
    ) == ("model", "server", 4)


async def test_projected_records_replay_as_the_webui_thread(ws, tmp_path) -> None:
    """Layer 22 end to end: a real websocket turn's persisted records, stored in
    the per-chat log, come back from ``webui-thread`` as the same turn."""
    from nanoclaw.webui.history import WebUIHistoryStore
    from nanoclaw.webui.transcript import (
        WebUITranscriptRecorder,
        build_webui_thread_payload,
    )

    script = [_call("hi", "<think>plan it</think>Let me check."), "All done."]
    outbound, loop = await _run_websocket_turn(ws, tmp_path, script)
    store = WebUIHistoryStore.for_sessions(loop.sessions)
    recorder = WebUITranscriptRecorder(store)
    projector = WebUIOutboundProjector()
    # The inbound path (layer 23) records the user's message.
    assert await recorder.append_user_message(
        "chat-1", "go", metadata={WEBUI_TURN_METADATA_KEY: "turn-1"}
    )
    for msg in outbound:
        for frame in projector.project(msg):
            if frame.record is not None:
                assert await recorder.append("chat-1", frame.record, metadata=frame.metadata)

    payload = build_webui_thread_payload("websocket:chat-1", store=store, sessions=loop.sessions)
    assert payload is not None
    assert payload["schemaVersion"] == 3 and payload["projection"] == "events"
    assert payload["completed_turn_ids"] == ["turn-1"]
    assert payload["active_turn_id"] is None
    replayed = [(e["event"], e.get("text")) for e in payload["events"]]
    assert replayed == [
        ("user_message", "go"),
        ("reasoning_end", "plan it"),
        ("stream_end", "Let me check."),
        ("message", 'echo("hi")'),
        ("message", ""),
        ("stream_end", "All done."),
        ("turn_end", None),
    ]
    # The session list shows the chat with its preview.
    from nanoclaw.webui.session_list_index import webui_sessions_payload

    rows = webui_sessions_payload(loop.sessions, store)["sessions"]
    assert [row["key"] for row in rows] == ["websocket:chat-1"]
