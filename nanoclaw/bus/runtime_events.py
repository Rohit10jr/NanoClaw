"""Turn lifecycle facts published to local bus subscribers.

A slim copy of nanobot/bus/runtime_events.py: the events a turn's delivery
publishes with ``MessageBus.publish`` (never to channels) so local consumers,
such as the terminal chat, know when a turn starts, runs and goes idle.

Since layer 19 also ``SubagentTaskChanged`` (a subagent task's status
changed; nanobot's WebUI task cards read it, nothing in NanoClaw consumes it
yet). Since layer 22 ``TurnRuntimeAdmitted`` and the runtime, usage and
failure details on ``TurnCompleted``, which the web UI projects as
``turn_model_updated`` and ``turn_end`` (nanoclaw/webui/outbound_projection.py).
Left out until something consumes them: ``UserInputAccepted``,
``SessionTurnPersisted`` and goal events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nanoclaw.events import AgentEvent
from nanoclaw.providers.base import LLMUsage
from nanoclaw.utils.llm_runtime import LLMRuntime

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
class TurnRuntimeAdmitted(AgentEvent):
    """The model runtime selected for one admitted turn. (nanobot :45)"""

    context: RuntimeEventContext
    runtime: LLMRuntime


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
    runtime: LLMRuntime | None = None
    usage: LLMUsage | None = None
    # Logical model rounds in display order.
    round_usages: tuple[LLMUsage, ...] = ()
    outcome: str = "completed"
    failure_kind: str | None = None
    failure_error_kind: str | None = None
    failure_attempts: int | None = None


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
        self._turn_runtime: dict[str, LLMRuntime] = {}
        self._turn_usage: dict[str, LLMUsage] = {}
        self._turn_round_usages: dict[str, tuple[LLMUsage, ...]] = {}

    @staticmethod
    def _context(*, channel: str, chat_id: str, session_key: str, metadata: dict[str, Any] | None) -> RuntimeEventContext:
        return RuntimeEventContext(channel=channel, chat_id=chat_id, session_key=session_key, metadata=dict(metadata or {}))

    def record_turn_runtime(self, session_key: str, runtime: LLMRuntime) -> None:
        self._turn_runtime[session_key] = runtime

    def record_turn_latency(self, session_key: str, latency_ms: int | None) -> None:
        if latency_ms is not None:
            self._turn_latency_ms[session_key] = int(latency_ms)

    def record_turn_usage(self, session_key: str, round_usages: list[LLMUsage]) -> None:
        """Add one run's per-request usage to the turn. (nanobot :148)"""
        if not round_usages:
            return
        usage = round_usages[0]
        for round_usage in round_usages[1:]:
            usage += round_usage
        previous = self._turn_usage.get(session_key)
        self._turn_usage[session_key] = usage if previous is None else previous + usage
        self._turn_round_usages[session_key] = (*self._turn_round_usages.get(session_key, ()), *round_usages)

    def clear_turn(self, session_key: str) -> None:
        self._turn_latency_ms.pop(session_key, None)
        self._turn_runtime.pop(session_key, None)
        self._turn_usage.pop(session_key, None)
        self._turn_round_usages.pop(session_key, None)

    async def session_turn_started(self, msg: InboundMessage, session_key: str) -> None:
        await self.bus.publish(
            SessionTurnStarted(
                context=self._context(
                    channel=msg.channel, chat_id=msg.chat_id, session_key=session_key, metadata=msg.metadata
                )
            )
        )

    async def turn_runtime_admitted(self, msg: InboundMessage, session_key: str, runtime: LLMRuntime) -> None:
        """Record and publish the runtime selected for one turn. (nanobot :207)"""
        self.record_turn_runtime(session_key, runtime)
        await self.bus.publish(
            TurnRuntimeAdmitted(
                context=self._context(
                    channel=msg.channel, chat_id=msg.chat_id, session_key=session_key, metadata=msg.metadata
                ),
                runtime=runtime,
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
        failure_error_kind: str | None = None,
        failure_attempts: int | None = None,
    ) -> None:
        await self.bus.publish(
            TurnCompleted(
                context=self._context(channel=channel, chat_id=chat_id, session_key=session_key, metadata=metadata),
                latency_ms=self._turn_latency_ms.pop(session_key, None),
                runtime=self._turn_runtime.pop(session_key, None),
                usage=self._turn_usage.pop(session_key, None),
                round_usages=self._turn_round_usages.pop(session_key, ()),
                outcome=outcome,
                failure_kind=failure_kind,
                failure_error_kind=failure_error_kind,
                failure_attempts=failure_attempts,
            )
        )
