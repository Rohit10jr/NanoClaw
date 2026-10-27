"""Typed notifications agent operations emit, independent of any transport.

Copied from nanobot/events.py. An operation gets an :class:`EventSink` and
publishes events to it; whoever built the sink (CLI, a channel, the web UI)
decides what to do with them. No sink means the events go nowhere.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from loguru import logger


class AgentEvent:
    """Base class for typed events. (nanobot events.py:12)"""


@dataclass(frozen=True)
class ResponseSource:
    """Which provider/model produced an answer. (nanobot events.py:17)

    ``preset`` is the model preset name; NanoClaw has no presets yet, so
    callers pass the model name or "".
    """

    provider: str
    model: str
    preset: str
    fallback: bool = False


@dataclass(frozen=True)
class ResponseSourceEvent(AgentEvent):
    """Sets the source of the answer text that follows. (nanobot events.py:27)"""

    source: ResponseSource | None
    # Completed provider text; runner/command messages must not inherit the
    # last model's name.
    content: str | None = None


CompactionReason = Literal["idle", "context_pressure", "manual", "provider_native"]


@dataclass(frozen=True)
class ContextCompactionEvent(AgentEvent):
    """Lifecycle of one context compaction. (nanobot events.py:40)"""

    compaction_id: str
    phase: Literal["started", "succeeded", "failed", "cancelled"]
    # Ask for a chat notice (e.g. /compact). Interactive clients consume
    # every phase regardless.
    notify: bool = False
    reason: CompactionReason | None = None


@dataclass(frozen=True)
class RetryWaitEvent(AgentEvent):
    """Human-readable "retrying in Ns" text. (nanobot events.py:50)"""

    content: str = ""


@dataclass(frozen=True)
class RetryStatusEvent(AgentEvent):
    """Sanitized retry lifecycle for one model request chain. (nanobot events.py:55)"""

    state: Literal["waiting", "recovered", "cleared", "exhausted"]
    attempt: int
    max_attempts: int | None
    error_kind: str
    next_retry_at: float | None = None


@dataclass(frozen=True)
class RecoveryStateEvent(AgentEvent):
    """State of a turn-recovery attempt. (nanobot events.py:66)"""

    status: str
    recovery_id: str
    reason: str | None = None
    attempts: int = 0
    can_continue: bool | None = None
    failure_error_kind: str | None = None


@dataclass(frozen=True)
class EventSink:
    """A send callback bound to one operation's destination. (nanobot events.py:76)

    ``emit`` is best effort (errors are logged); hooks that must surface
    output failures await ``publish`` directly. Both let cancellation
    through. ``accepts`` lets expensive producers skip work nobody consumes.
    """

    publish: Callable[[AgentEvent], Awaitable[None]] | None = None
    accepts_type: Callable[[type[AgentEvent]], bool] | None = None

    def accepts(self, event_type: type[AgentEvent]) -> bool:
        """Whether an event of this type has a consumer."""
        return self.publish is not None and (
            self.accepts_type is None or self.accepts_type(event_type)
        )

    async def emit(self, event: AgentEvent) -> None:
        if self.publish is None:
            return
        try:
            await self.publish(event)
        except Exception:
            logger.exception("Failed to publish {}", type(event).__name__)


NO_EVENTS = EventSink()
