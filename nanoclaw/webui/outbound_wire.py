"""Encode typed outbound events as web UI wire JSON (``webui.core.v1``).

A port of nanobot/webui/outbound_wire.py (the notification and ``turn_end``
encoders, tool-event sanitizing) plus the payload builders nanobot keeps in
its WebSocket channel (channels/websocket/runtime.py ``send_*``:
``message``, ``delta``/``stream_end``, ``reasoning_delta``/``reasoning_end``,
``file_edit``, ``goal_status``, ``goal_state``, ``session_updated``,
``user_message``, ``runtime_model_updated``, ``turn_model_updated``) and
the control ``error`` event (nanobot webui/inbound_commands.py). Every
function returns a plain ``dict`` in the exact shape the frontend validates;
none of them touches a socket.

Persistence: :data:`PERSISTED_WIRE_EVENTS` and :func:`is_persisted_wire_event`
say which wire payloads nanobot writes to a chat's transcript (runtime.py
``_persist_turn_transcript_event`` / ``_persist_turn_stream_event`` and the
recorder's ``prepare_and_append_stream_event``): every ``message`` and
``file_edit``, the completed ``stream_end`` and ``reasoning_end`` (never
their chunks), ``turn_end``, and ``context_compaction`` except the
``started`` phase. ``user_message`` rows are written too, but by the inbound
path (``append_user_message``), not by the outbound projection.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict, cast

from nanoclaw.bus.events import OUTBOUND_META_AGENT_UI, OutboundMessage
from nanoclaw.bus.outbound_events import (
    ContextCompactionEvent,
    ProgressEvent,
    RecoveryStateEvent,
    RetryStatusEvent,
    TurnEndEvent,
)
from nanoclaw.events import AgentEvent, CompactionReason
from nanoclaw.webui.metadata import (
    INBOUND_META_USER_SHELL,
    WEBUI_MESSAGE_SOURCE_METADATA_KEY,
    WEBUI_TURN_METADATA_KEY,
)

_JSONValue: TypeAlias = (
    str | int | float | bool | None | list["_JSONValue"] | dict[str, "_JSONValue"]
)

_TOOL_EVENT_BINARY_OMISSION = "[binary content omitted from WebUI]"


def _is_base64_data_url(value: str) -> bool:
    candidate = value.lstrip()
    return (
        candidate[:5].casefold() == "data:" and ";base64," in candidate[:256].casefold()
    )


def _project_tool_event_value(value: object) -> _JSONValue:
    """Copy one dynamic value while removing inline binary payloads. (nanobot :32)"""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if _is_base64_data_url(value):
            return _TOOL_EVENT_BINARY_OMISSION
        return value
    if isinstance(value, dict):
        data = cast(dict[object, object], value)
        return {str(key): _project_tool_event_value(item) for key, item in data.items()}
    if isinstance(value, list):
        return [_project_tool_event_value(item) for item in cast(list[object], value)]
    if isinstance(value, tuple):
        return [
            _project_tool_event_value(item) for item in cast(tuple[object, ...], value)
        ]
    return str(value)


def project_tool_events(events: list[dict[str, Any]]) -> list[dict[str, _JSONValue]]:
    """Web-UI-safe tool events, without mutating model-facing results. (nanobot :52)"""
    projected_events: list[dict[str, _JSONValue]] = []
    for event in events:
        projected = _project_tool_event_value(cast(object, event))
        if not isinstance(projected, dict):
            continue
        projected_events.append(projected)
    return projected_events


class _ChatWirePayload(TypedDict):
    chat_id: str
    turn_id: NotRequired[str]


class RecoveryStateWirePayload(_ChatWirePayload):
    event: Literal["recovery_state"]
    status: str
    recovery_id: str
    attempts: int
    reason: NotRequired[str]
    can_continue: NotRequired[bool]
    failure_error_kind: NotRequired[str]


class RetryStatusWirePayload(_ChatWirePayload):
    event: Literal["retry_status"]
    state: str
    attempt: int
    max_attempts: NotRequired[int]
    error_kind: str
    retry_after_s: NotRequired[float]


class TurnEndWirePayload(_ChatWirePayload):
    event: Literal["turn_end"]
    latency_ms: NotRequired[int]
    goal_state: NotRequired[dict[str, Any]]
    usage: NotRequired[dict[str, int]]
    round_usages: NotRequired[list[dict[str, int]]]
    context_window_tokens: NotRequired[int]
    outcome: NotRequired[str]
    failure_kind: NotRequired[str]
    failure_error_kind: NotRequired[str]
    failure_attempts: NotRequired[int]
    failure_message: NotRequired[str]


class ContextCompactionWirePayload(_ChatWirePayload):
    event: Literal["context_compaction"]
    compaction_id: str
    phase: Literal["started", "succeeded", "failed", "cancelled"]
    reason: NotRequired[CompactionReason]


class SubagentTaskWirePayload(_ChatWirePayload):
    event: Literal["subagent_task"]
    task: dict[str, object]


WebUIWirePayload: TypeAlias = (
    RetryStatusWirePayload
    | ContextCompactionWirePayload
    | RecoveryStateWirePayload
    | TurnEndWirePayload
    | SubagentTaskWirePayload
)
# How a payload is persisted (nanobot :124): ``transient`` never,
# ``turn_activity`` as a turn activity row, ``turn_complete`` as the row that
# closes the turn (stream buffers are dropped afterwards).
WebUIWirePersistence: TypeAlias = Literal["transient", "turn_activity", "turn_complete"]


def encode_recovery_state(
    chat_id: str, event: RecoveryStateEvent
) -> RecoveryStateWirePayload:
    """One transient recovery transition. (nanobot :131)"""
    payload: RecoveryStateWirePayload = {
        "event": "recovery_state",
        "chat_id": chat_id,
        "status": event.status,
        "recovery_id": event.recovery_id,
        "attempts": event.attempts,
    }
    if event.reason:
        payload["reason"] = event.reason
    if event.can_continue is not None:
        payload["can_continue"] = event.can_continue
    if event.failure_error_kind is not None:
        payload["failure_error_kind"] = event.failure_error_kind
    return payload


def encode_context_compaction(
    chat_id: str, event: ContextCompactionEvent
) -> ContextCompactionWirePayload:
    """One summary-free compaction transition. (nanobot :152)"""
    payload: ContextCompactionWirePayload = {
        "event": "context_compaction",
        "chat_id": chat_id,
        "compaction_id": event.compaction_id,
        "phase": event.phase,
    }
    if event.reason is not None:
        payload["reason"] = event.reason
    return payload


@dataclass(frozen=True)
class NotificationProjection:
    """Allowlisted public payload and its durability policy. (nanobot :168)"""

    payload: WebUIWirePayload
    persistence: WebUIWirePersistence = "transient"
    deliver_offline: bool = False
    attach_turn_metadata: bool = False


def project_notification(
    chat_id: str,
    event: AgentEvent | None,
    metadata: Mapping[str, object] | None = None,
) -> NotificationProjection | None:
    """Notification payload and persistence decision for one event. (nanobot :178)"""
    if isinstance(event, ContextCompactionEvent):
        return NotificationProjection(
            encode_context_compaction(chat_id, event),
            persistence="transient" if event.phase == "started" else "turn_activity",
            deliver_offline=True,
            attach_turn_metadata=True,
        )
    if isinstance(event, RetryStatusEvent):
        return NotificationProjection(encode_retry_status(chat_id, event, metadata))
    if isinstance(event, RecoveryStateEvent):
        return NotificationProjection(encode_recovery_state(chat_id, event))
    return None


def _turn_id(metadata: Mapping[str, object] | None) -> str | None:
    turn_id = (metadata or {}).get(WEBUI_TURN_METADATA_KEY)
    return turn_id if isinstance(turn_id, str) and turn_id else None


def encode_retry_status(
    chat_id: str,
    event: RetryStatusEvent,
    metadata: Mapping[str, object] | None = None,
) -> RetryStatusWirePayload:
    """One transient retry transition; the wait is relative. (nanobot :196)"""
    payload: RetryStatusWirePayload = {
        "event": "retry_status",
        "chat_id": chat_id,
        "state": event.state,
        "attempt": event.attempt,
        "error_kind": event.error_kind,
    }
    if turn_id := _turn_id(metadata):
        payload["turn_id"] = turn_id
    if event.max_attempts is not None:
        payload["max_attempts"] = event.max_attempts
    if event.next_retry_at is not None:
        payload["retry_after_s"] = max(0.0, event.next_retry_at - time.time())
    return payload


def encode_turn_end(
    chat_id: str,
    event: TurnEndEvent,
    metadata: Mapping[str, object] | None = None,
) -> TurnEndWirePayload:
    """A completed turn, without internal metadata. (nanobot :218)"""
    payload: TurnEndWirePayload = {"event": "turn_end", "chat_id": chat_id}
    if turn_id := _turn_id(metadata):
        payload["turn_id"] = turn_id
    if event.latency_ms is not None:
        payload["latency_ms"] = int(event.latency_ms)
    if event.goal_state is not None:
        payload["goal_state"] = event.goal_state
    if event.usage is not None:
        payload["usage"] = event.usage.to_turn_dict()
    if event.round_usages:
        payload["round_usages"] = [item.to_turn_dict() for item in event.round_usages]
    if event.context_window_tokens is not None:
        payload["context_window_tokens"] = int(event.context_window_tokens)
    if event.outcome != "completed":
        payload["outcome"] = event.outcome
    if event.failure_kind is not None:
        payload["failure_kind"] = event.failure_kind
    if event.failure_error_kind is not None:
        payload["failure_error_kind"] = event.failure_error_kind
    if event.failure_attempts is not None:
        payload["failure_attempts"] = int(event.failure_attempts)
    if event.failure_message is not None:
        payload["failure_message"] = event.failure_message
    return payload


# --- channel payloads (nanobot channels/websocket/runtime.py send_*) --------


def encode_message(
    msg: OutboundMessage,
    progress_event: ProgressEvent | None,
    *,
    wire_text: str | None = None,
    media_urls: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """An ordinary outbound message or progress line. (nanobot runtime.py:1231)

    ``wire_text`` is the text after local markdown images were rewritten to
    signed URLs (the caller's media gateway); a user-shell reply keeps its
    text and is marked ``render_as: "text"``. ``kind`` is ``tool_hint`` or
    ``progress`` for progress events, absent for answers.
    """
    user_shell = msg.metadata.get(INBOUND_META_USER_SHELL) is True
    text = msg.content
    payload: dict[str, Any] = {
        "event": "message",
        "chat_id": msg.chat_id,
        "text": text if user_shell or wire_text is None else wire_text,
    }
    if user_shell:
        payload["render_as"] = "text"
    if turn_id := _turn_id(msg.metadata):
        payload["turn_id"] = turn_id
    if msg.media:
        payload["media"] = msg.media
        if media_urls:
            payload["media_urls"] = media_urls
    if msg.reply_to:
        payload["reply_to"] = msg.reply_to
    lat = msg.metadata.get("latency_ms")
    if isinstance(lat, (int, float)):
        payload["latency_ms"] = int(lat)
    if progress_event and progress_event.tool_events:
        payload["tool_events"] = project_tool_events(progress_event.tool_events)
    agent_ui = msg.metadata.get(OUTBOUND_META_AGENT_UI)
    if agent_ui is not None:
        payload["agent_ui"] = agent_ui
    if progress_event and progress_event.tool_hint:
        payload["kind"] = "tool_hint"
    elif progress_event:
        payload["kind"] = "progress"
    return payload


def encode_delta(
    chat_id: str, delta: str, *, stream_id: str | None = None
) -> dict[str, Any]:
    """One chunk of streamed answer text. (nanobot runtime.py:1416)"""
    body: dict[str, Any] = {"event": "delta", "chat_id": chat_id, "text": delta}
    if stream_id is not None:
        body["stream_id"] = stream_id
    return body


def encode_stream_end(
    chat_id: str,
    *,
    text: str | None = None,
    stream_id: str | None = None,
    resuming: bool = False,
    merge_next: bool = False,
) -> dict[str, Any]:
    """The end of a streamed segment. (nanobot runtime.py:1402)

    ``text`` is set only when the end carried its own text or the full text
    changed when images were rewritten; the persisted record always carries
    the full segment text.
    """
    body: dict[str, Any] = {"event": "stream_end", "chat_id": chat_id}
    if text is not None:
        body["text"] = text
    if stream_id is not None:
        body["stream_id"] = stream_id
    if resuming:
        body["resuming"] = True
    if merge_next:
        body["merge_next"] = True
    return body


def encode_reasoning_delta(
    chat_id: str, delta: str, *, stream_id: str | None = None
) -> dict[str, Any]:
    """One chunk of model reasoning. (nanobot runtime.py:1292)"""
    body: dict[str, Any] = {
        "event": "reasoning_delta",
        "chat_id": chat_id,
        "text": delta,
    }
    if stream_id is not None:
        body["stream_id"] = stream_id
    return body


def encode_reasoning_end(
    chat_id: str, *, stream_id: str | None = None
) -> dict[str, Any]:
    """Close the current reasoning segment. (nanobot runtime.py:1331)"""
    body: dict[str, Any] = {"event": "reasoning_end", "chat_id": chat_id}
    if stream_id is not None:
        body["stream_id"] = stream_id
    return body


def encode_file_edit(chat_id: str, edits: list[dict[str, Any]]) -> dict[str, Any]:
    """File activity of a tool call. (nanobot runtime.py:1362)"""
    return {"event": "file_edit", "chat_id": chat_id, "edits": edits}


def encode_goal_state(chat_id: str, blob: dict[str, Any]) -> dict[str, Any]:
    """A sustained-goal snapshot. (nanobot runtime.py:1490)"""
    return {"event": "goal_state", "chat_id": chat_id, "goal_state": blob}


def encode_goal_status(
    chat_id: str,
    status: str,
    *,
    started_at: float | None = None,
    turn_id: str | None = None,
) -> dict[str, Any]:
    """A turn started (``running``) or finished (``idle``). (nanobot runtime.py:1500)"""
    body: dict[str, Any] = {
        "event": "goal_status",
        "chat_id": chat_id,
        "status": status,
    }
    if status == "running" and started_at is not None:
        body["started_at"] = started_at
    if turn_id:
        body["turn_id"] = turn_id
    return body


def encode_session_updated(chat_id: str, *, scope: str | None = None) -> dict[str, Any]:
    """A session row should refresh. (nanobot runtime.py:1525)"""
    body: dict[str, Any] = {"event": "session_updated", "chat_id": chat_id}
    if scope:
        body["scope"] = scope
    return body


def encode_user_input(
    chat_id: str,
    *,
    content: str,
    created_at_ms: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """User input produced outside a WebSocket connection. (nanobot runtime.py:1537)"""
    body: dict[str, Any] = {
        "event": "user_message",
        "chat_id": chat_id,
        "text": content,
        "created_at_ms": created_at_ms,
        "starts_turn": False,
    }
    if provenance:
        body["provenance"] = provenance
    return body


def encode_runtime_model_updated(
    *, model_name: Any, model_preset: Any = None
) -> dict[str, Any] | None:
    """The runtime model changed; ``None`` without a model name. (nanobot runtime.py:1570)"""
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    body: dict[str, Any] = {
        "event": "runtime_model_updated",
        "model_name": model_name.strip(),
    }
    if isinstance(model_preset, str) and model_preset.strip():
        body["model_preset"] = model_preset.strip()
    return body


def encode_turn_model_updated(
    chat_id: str,
    *,
    model_name: Any,
    model_preset: Any = None,
    context_window_tokens: Any = None,
    fallback: bool = False,
    reauth_provider: str | None = None,
) -> dict[str, Any] | None:
    """The model handling one chat's turn; ``None`` without a model name. (nanobot runtime.py:1590)"""
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    body: dict[str, Any] = {
        "event": "turn_model_updated",
        "chat_id": chat_id,
        "model_name": model_name.strip(),
    }
    if isinstance(model_preset, str) and model_preset.strip():
        body["model_preset"] = model_preset.strip()
    if isinstance(context_window_tokens, int) and context_window_tokens > 0:
        body["context_window_tokens"] = context_window_tokens
    if fallback:
        body["fallback"] = True
        if reauth_provider:
            body["reauth_provider"] = reauth_provider
    return body


def encode_error(
    detail: str,
    *,
    chat_id: str | None = None,
    turn_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """A rejected request (nanobot webui/inbound_commands.py ``webui_send_event(..., "error")``).

    Sent to the one connection whose request failed; the frontend rejects a
    pending send whose ``turn_id`` matches (packages/client-events delivery.ts).
    """
    body: dict[str, Any] = {"event": "error", "detail": detail}
    if reason is not None:
        body["reason"] = reason
    if chat_id:
        body["chat_id"] = chat_id
    if turn_id:
        body["turn_id"] = turn_id
    return body


# --- transcript annotations (nanobot webui/transcript.py) --------------------

# Automation kinds a message source may name (nanobot session/automation_turns.py:56:
# "trigger" plus the cron and local-trigger automation specs).
_AUTOMATION_KINDS = frozenset({"trigger", "cron", "local_trigger"})


def webui_message_source(metadata: Mapping[str, Any] | None) -> dict[str, str] | None:
    """The automation that produced a message, if any. (nanobot transcript.py:511)"""
    raw = (metadata or {}).get(WEBUI_MESSAGE_SOURCE_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    source_metadata = cast(dict[str, Any], raw)
    kind = source_metadata.get("kind")
    if not isinstance(kind, str) or kind not in _AUTOMATION_KINDS:
        return None
    source: dict[str, str] = {"kind": kind}
    label = source_metadata.get("label")
    if isinstance(label, str) and label.strip():
        source["label"] = label.strip()
    return source


def response_sources(value: object) -> list[dict[str, str | bool]]:
    """Only recorded display identity, never settings or credentials. (nanobot transcript.py:93)"""
    if not isinstance(value, list):
        return []
    result: list[dict[str, str | bool]] = []
    keys = ("provider", "model", "preset")
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            return []
        fields = cast(dict[str, object], item)
        if any(not isinstance(fields.get(key), str) or not fields[key] for key in keys):
            return []
        source: dict[str, str | bool] = {key: cast(str, fields[key]) for key in keys}
        source["fallback"] = fields.get("fallback") is True
        if source not in result:
            result.append(source)
    return result


# --- persistence rule -------------------------------------------------------

# Wire events nanobot appends to a chat's transcript from the outbound side.
# ``context_compaction`` only when not ``started``; ``stream_end`` and
# ``reasoning_end`` carry the full segment text in the record.
PERSISTED_WIRE_EVENTS: frozenset[str] = frozenset(
    {
        "message",
        "file_edit",
        "stream_end",
        "reasoning_end",
        "turn_end",
        "context_compaction",
    }
)
# Every event kind a transcript can hold: the outbound ones plus the user's
# own rows, which the inbound path writes (nanobot transcript.py
# ``append_user_message``).
TRANSCRIPT_EVENTS: frozenset[str] = PERSISTED_WIRE_EVENTS | {"user_message"}


def is_persisted_wire_event(payload: Mapping[str, Any]) -> bool:
    """Whether nanobot writes this outbound wire payload to the transcript."""
    event = payload.get("event")
    if event not in PERSISTED_WIRE_EVENTS:
        return False
    return not (event == "context_compaction" and payload.get("phase") == "started")
