"""Turn lifecycle facts published to local bus subscribers.

A slim copy of nanobot/bus/runtime_events.py: the events a turn's delivery
publishes with ``MessageBus.publish`` (never to channels) so local consumers,
such as the terminal chat, know when a turn starts, runs and goes idle.

Since layer 19 also ``SubagentTaskChanged`` (a subagent task's status
changed; nanobot's WebUI task cards read it, nothing in NanoClaw consumes it
yet). Left out until something consumes them: ``UserInputAccepted``,
``TurnRuntimeAdmitted``, ``SessionTurnPersisted``, goal events, and the
usage/runtime bookkeeping on ``TurnCompleted`` (nanobot records the runtime and every round's usage for its WebUI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nanoclaw.events import AgentEvent

if TYPE_CHECKING:
    from nanoclaw.bus.events import InboundMessage
    from nanoclaw.bus.queue import MessageBus


@dataclass(frozen=True)
class RuntimeEventContext:
    """Routing context common to turn-scoped runtime events. (nanobot runtime_events.py:18)"""

    channel: str
    chat_id: str
    session_key: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionTurnStarted(AgentEvent):
    """A turn has loaded its session and is about to build context. (nanobot :30)"""

    context: RuntimeEventContext


@dataclass(frozen=True)
class TurnRunStatusChanged(AgentEvent):
    """Visible run status changed for a turn: ``running`` or ``idle``. (nanobot :53)"""

    context: RuntimeEventContext
    status: str
    started_at: float | None = None


@dataclass(frozen=True)
class TurnCompleted(AgentEvent):
    """A turn has delivered its final user-visible response. (nanobot :61)"""

    context: RuntimeEventContext
    latency_ms: int | None = None
    outcome: str = "completed"
    failure_kind: str | None = None


@dataclass(frozen=True)
class SubagentTaskChanged(AgentEvent):
    """A parent-owned task observation changed. (nanobot :95)"""

    context: RuntimeEventContext
    task_id: str


class RuntimeEventPublisher:
    """Builds the event contexts and publishes them. (nanobot runtime_events.py:108)"""

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        self._turn_latency_ms: dict[str, int] = {}

    @staticmethod
    def _context(*, channel: str, chat_id: str, session_key: str, metadata: dict[str, Any] | None) -> RuntimeEventContext:
        return RuntimeEventContext(channel=channel, chat_id=chat_id, session_key=session_key, metadata=dict(metadata or {}))

    def record_turn_latency(self, session_key: str, latency_ms: int | None) -> None:
        if latency_ms is not None:
            self._turn_latency_ms[session_key] = int(latency_ms)

    def clear_turn(self, session_key: str) -> None:
        self._turn_latency_ms.pop(session_key, None)

    async def session_turn_started(self, msg: InboundMessage, session_key: str) -> None:
        await self.bus.publish(
            SessionTurnStarted(
                context=self._context(
                    channel=msg.channel, chat_id=msg.chat_id, session_key=session_key, metadata=msg.metadata
                )
            )
        )

    async def run_status_changed(
        self, msg: InboundMessage, session_key: str, status: str, *, started_at: float | None = None
    ) -> None:
        await self.bus.publish(
            TurnRunStatusChanged(
                context=self._context(
                    channel=msg.channel, chat_id=msg.chat_id, session_key=session_key, metadata=msg.metadata
                ),
                status=status,
                started_at=started_at,
            )
        )

    async def turn_completed(
        self,
        *,
        channel: str,
        chat_id: str,
        session_key: str,
        metadata: dict[str, Any] | None,
        outcome: str = "completed",
        failure_kind: str | None = None,
    ) -> None:
        await self.bus.publish(
            TurnCompleted(
                context=self._context(channel=channel, chat_id=chat_id, session_key=session_key, metadata=metadata),
                latency_ms=self._turn_latency_ms.pop(session_key, None),
                outcome=outcome,
                failure_kind=failure_kind,
            )
        )
