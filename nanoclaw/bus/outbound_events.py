"""Typed events for user-facing output: progress, streaming, turn end.

Copied from nanobot/bus/outbound_events.py. In nanobot these ride on an
``OutboundMessage`` through the message bus; NanoClaw has no bus yet
(layer 14), so ``outbound_message_for_event`` / ``replace_outbound_event``
are not here. The events are published straight to an
:class:`~nanoclaw.events.EventSink`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoclaw.events import AgentEvent
from nanoclaw.events import ContextCompactionEvent as ContextCompactionEvent
from nanoclaw.events import RecoveryStateEvent as RecoveryStateEvent
from nanoclaw.events import RetryStatusEvent as RetryStatusEvent
from nanoclaw.events import RetryWaitEvent as RetryWaitEvent
from nanoclaw.providers import LLMUsage


@dataclass(frozen=True)
class ProgressEvent(AgentEvent):
    """Progress while a turn works. (nanobot bus/outbound_events.py:24)

    Plain text (the model's words before tools), a tool hint
    (``tool_hint=True``), a reasoning chunk (``reasoning_delta``), the end of a
    reasoning segment (``reasoning_end``), or structured ``tool_events``
    (version-1 payloads: phase start/end/error, call_id, name, arguments,
    result, error, files, embeds).
    """

    content: str = ""
    tool_hint: bool = False
    reasoning: bool = False
    reasoning_delta: bool = False
    reasoning_end: bool = False
    stream_id: str | None = None
    tool_events: list[dict[str, Any]] | None = None
    file_edit_events: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class FileEditEvent(ProgressEvent):
    """File activity; only produced when a consumer wants it. (nanobot :36)"""


@dataclass(frozen=True)
class StreamDeltaEvent(AgentEvent):
    """New answer text while streaming. (nanobot :41)"""

    content: str = ""
    stream_id: str | None = None


@dataclass(frozen=True)
class StreamEndEvent(AgentEvent):
    """End of a streamed segment. (nanobot :47)

    ``resuming=True``: more model output follows in this turn (tools are
    about to run), so keep the message open. ``merge_next``: the next segment
    continues the same message.
    """

    content: str = ""
    stream_id: str | None = None
    resuming: bool = False
    merge_next: bool = False


@dataclass(frozen=True)
class StreamedResponseEvent(AgentEvent):
    """The final answer was already shown by streaming. (nanobot :55)"""


@dataclass(frozen=True)
class TurnEndEvent(AgentEvent):
    """A turn finished. (nanobot :60)"""

    latency_ms: int | None = None
    goal_state: dict[str, Any] | None = None
    usage: LLMUsage | None = None
    round_usages: tuple[LLMUsage, ...] = ()
    context_window_tokens: int | None = None
    outcome: str = "completed"
    failure_kind: str | None = None
    failure_error_kind: str | None = None
    failure_attempts: int | None = None
    failure_message: str | None = None


# The events below have no producer in NanoClaw yet (goals, session sync and
# model switching come in later layers). They are plain data, kept so the
# module matches nanobot's.


@dataclass(frozen=True)
class GoalStatusEvent(AgentEvent):
    """(nanobot :74)"""

    status: str
    started_at: float | None = None


@dataclass(frozen=True)
class GoalStateSyncEvent(AgentEvent):
    """(nanobot :80)"""

    goal_state: dict[str, Any]


@dataclass(frozen=True)
class SessionUpdatedEvent(AgentEvent):
    """(nanobot :85)"""

    scope: str | None = None


@dataclass(frozen=True)
class UserInputEvent(AgentEvent):
    """A user-input row projected by an edge adapter. (nanobot :90)"""

    content: str
    created_at_ms: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class RuntimeModelUpdatedEvent(AgentEvent):
    """(nanobot :99)"""

    model: str | None
    model_preset: str | None = None


@dataclass(frozen=True)
class TurnModelUpdatedEvent(AgentEvent):
    """The preset and concrete model handling one chat turn. (nanobot :105)"""

    model: str
    model_preset: str | None = None
    context_window_tokens: int | None = None
    fallback: bool = False
    reauth_provider: str | None = None


def _event_content(event: AgentEvent) -> str:
    """Plain-text fallback for an event, for channels that only show text.

    (nanobot :149; used there by ``outbound_message_for_event``)
    """
    if isinstance(
        event,
        ProgressEvent | RetryWaitEvent | StreamDeltaEvent | StreamEndEvent | UserInputEvent,
    ):
        return event.content
    if isinstance(event, ContextCompactionEvent):
        if event.phase == "started":
            return "Compressing context…"
        if event.phase == "failed":
            return "Unable to compact context."
        if event.phase == "cancelled":
            return "Context compaction cancelled."
        return "Context compacted."
    return ""
