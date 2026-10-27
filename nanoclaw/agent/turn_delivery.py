"""Route and publish the user-visible lifecycle of an agent turn.

A slim copy of nanobot/agent/turn_delivery.py. A :class:`TurnDelivery` is
made per bus turn (``AgentLoop._dispatch_one``). It

- decides where the turn's output goes (:class:`TurnRoute`: the inbound
  message's channel and chat, or for a system message such as a subagent
  report the ``channel:chat_id`` it names),
- filters and publishes the turn's events on the bus (progress only for
  user turns, streams only when the channel asked for them with the
  ``_wants_stream`` metadata) and gives each streamed segment a stream id,
- publishes the final :class:`OutboundMessage`, the error reply, and the
  lifecycle events (started, running, completed, idle; local subscribers only,
  nanoclaw/bus/runtime_events.py).

Left out: route policies and admission (channel plugins), session
generations, response-source attribution (no ResponseSourceEvent producer
yet), scheduled-delivery correlation and the per-turn runtime and usage
bookkeeping for the WebUI.
"""

from __future__ import annotations

import dataclasses
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from nanoclaw.bus.events import InboundMessage, OutboundMessage
from nanoclaw.bus.outbound_events import (
    FileEditEvent,
    ProgressEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
)
from nanoclaw.bus.queue import MessageBus
from nanoclaw.bus.runtime_events import RuntimeEventPublisher
from nanoclaw.events import (
    AgentEvent,
    ContextCompactionEvent,
    EventSink,
    RecoveryStateEvent,
    ResponseSourceEvent,
    RetryStatusEvent,
    RetryWaitEvent,
)

# Which operation events a channel gets (nanobot bus/notification_delivery.py:21):
# "lifecycle" only for routes that publish a turn's lifecycle (user turns),
# "interactive" only for the WebUI, "channel" always.
_NOTIFICATION_AUDIENCES: dict[type[AgentEvent], str] = {
    ProgressEvent: "lifecycle",
    FileEditEvent: "lifecycle",
    StreamDeltaEvent: "channel",
    StreamEndEvent: "channel",
    ContextCompactionEvent: "channel",
    RetryWaitEvent: "lifecycle",
    RecoveryStateEvent: "interactive",
    RetryStatusEvent: "interactive",
    ResponseSourceEvent: "interactive",
}

# Inbound metadata a channel sets to get the answer as stream events
# (nanobot turn_delivery.py:266, set by its CLI and WebUI).
META_WANTS_STREAM = "_wants_stream"


def notification_is_deliverable(event_type: type[AgentEvent], *, channel: str, publish_lifecycle: bool) -> bool:
    """(nanobot bus/notification_delivery.py:34)"""
    audience = _NOTIFICATION_AUDIENCES.get(event_type)
    if audience is None:
        return False
    if audience == "lifecycle":
        return publish_lifecycle
    if audience == "interactive":
        return channel == "websocket" and (event_type is not RetryStatusEvent or publish_lifecycle)
    return True


@dataclass(frozen=True)
class TurnRoute:
    """Where a turn's output goes, and whether its lifecycle is published. (nanobot :44)"""

    channel: str
    chat_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    publish_lifecycle: bool = False


def _bind_events(bus: MessageBus, route: TurnRoute) -> EventSink:
    """An event sink publishing deliverable events to *route*. (nanobot :61)"""
    channel, chat_id = route.channel, route.chat_id
    metadata = deepcopy(route.metadata)

    def accepts(event_type: type[AgentEvent]) -> bool:
        return notification_is_deliverable(event_type, channel=channel, publish_lifecycle=route.publish_lifecycle)

    async def publish(event: AgentEvent) -> None:
        if not accepts(type(event)):
            return
        await bus.publish_event(event, channel=channel, chat_id=chat_id, metadata=deepcopy(metadata))

    return EventSink(publish, accepts)


def _turn_outcome(stop_reason: object) -> tuple[str, str | None]:
    """(nanobot :84)"""
    if stop_reason == "error":
        return "failed", "model"
    if stop_reason == "tool_error":
        return "failed", "tool"
    return "completed", None


class TurnDeliveryFactory:
    """Creates a turn's delivery from its inbound message. (nanobot :92)"""

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        self.runtime_event_publisher = RuntimeEventPublisher(bus)

    def create(self, msg: InboundMessage, session_key: str, *, enable_stream: bool = False) -> TurnDelivery:
        return TurnDelivery(
            bus=self.bus,
            runtime_event_publisher=self.runtime_event_publisher,
            input_message=msg,
            session_key=session_key,
            route=self._default_route(msg),
            enable_stream=enable_stream,
        )

    def unrouted(self, msg: InboundMessage, session_key: str) -> TurnDelivery:
        """A fallback that answers the inbound chat directly. (nanobot :126)"""
        return TurnDelivery(
            bus=self.bus,
            runtime_event_publisher=self.runtime_event_publisher,
            input_message=msg,
            session_key=session_key,
            route=TurnRoute(channel=msg.channel, chat_id=msg.chat_id, metadata=dict(msg.metadata or {})),
        )

    @staticmethod
    def _default_route(msg: InboundMessage) -> TurnRoute:
        """User input answers its own chat; a system message names the chat as
        ``channel:chat_id``. (nanobot :184; ``system_reply_metadata`` only
        matters for Slack threads.)"""
        if msg.channel != "system":
            return TurnRoute(
                channel=msg.channel, chat_id=msg.chat_id, metadata=dict(msg.metadata or {}), publish_lifecycle=True
            )
        channel, chat_id = msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
        metadata: dict[str, Any] = {}
        if origin_message_id := msg.metadata.get("origin_message_id"):
            metadata["origin_message_id"] = origin_message_id
        return TurnRoute(channel=channel, chat_id=chat_id, metadata=metadata)


@dataclass
class TurnDelivery:
    """Routing, event publication and lifecycle for one turn. (nanobot :201)"""

    bus: MessageBus
    runtime_event_publisher: RuntimeEventPublisher
    input_message: InboundMessage
    session_key: str
    route: TurnRoute
    enable_stream: bool = False
    delivery_message: InboundMessage = field(init=False)
    lifecycle_message: InboundMessage = field(init=False)
    events: EventSink = field(init=False)
    _routed_events: EventSink = field(init=False)
    _stream_base_id: str | None = field(init=False, default=None)
    _stream_segment: int = field(init=False, default=0)
    _stream_open: bool = field(init=False, default=False)
    _stop_reason: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._routed_events = _bind_events(self.bus, self.route)
        self.events = EventSink(self._publish_event, self._routed_events.accepts)
        self.delivery_message = dataclasses.replace(
            self.input_message,
            channel=self.route.channel,
            chat_id=self.route.chat_id,
            metadata=dict(self.route.metadata),
        )
        self.lifecycle_message = self.delivery_message if self.route.publish_lifecycle else self.input_message
        if self.enable_stream and self.delivery_message.metadata.get(META_WANTS_STREAM):
            self._stream_base_id = f"{self.session_key}:{time.time_ns()}"

    @property
    def streaming(self) -> bool:
        """Whether the channel gets the answer as stream events. (nanobot :248)"""
        return self._stream_base_id is not None

    async def started(self) -> None:
        if self.route.publish_lifecycle:
            await self.runtime_event_publisher.session_turn_started(self.delivery_message, self.session_key)

    async def running(self, *, started_at: float) -> None:
        if self.route.publish_lifecycle:
            await self.runtime_event_publisher.run_status_changed(
                self.delivery_message, self.session_key, "running", started_at=started_at
            )

    def record_latency(self, latency_ms: int | None) -> None:
        self.runtime_event_publisher.record_turn_latency(self.session_key, latency_ms)

    def record_stop_reason(self, stop_reason: str) -> None:
        self._stop_reason = stop_reason

    def background_response(
        self, content: str | None, *, stop_reason: str, streamed: bool, latency_ms: int | None
    ) -> OutboundMessage:
        """The reply of a system turn (a subagent report). (nanobot :327)"""
        metadata = dict(self.route.metadata)
        if self.route.publish_lifecycle and latency_ms is not None:
            metadata["latency_ms"] = int(latency_ms)
        event = (
            StreamedResponseEvent()
            if self.route.publish_lifecycle and streamed and stop_reason not in {"error", "tool_error"}
            else None
        )
        return OutboundMessage(
            channel=self.route.channel,
            chat_id=self.route.chat_id,
            content=content or "Background task completed.",
            metadata=metadata,
            event=event,
        )

    async def complete(self, response: OutboundMessage | None, *, publish_completion: bool) -> None:
        """Publish the final reply (an empty one to the CLI when there is none,
        so it stops waiting) and the completion event. (nanobot :350)"""
        stop_reason = self._stop_reason
        if stop_reason is None and response is not None:
            stop_reason = response.metadata.get("_stop_reason")
        completed_channel = self.lifecycle_message.channel
        completed_chat_id = self.lifecycle_message.chat_id
        if response is not None:
            completed_channel, completed_chat_id = response.channel, response.chat_id
            await self.bus.publish_outbound(response)
        elif self.lifecycle_message.channel == "cli":
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=self.lifecycle_message.channel,
                    chat_id=self.lifecycle_message.chat_id,
                    content="",
                    metadata=dict(self.lifecycle_message.metadata or {}),
                )
            )
        if publish_completion:
            outcome, failure_kind = _turn_outcome(stop_reason)
            await self.runtime_event_publisher.turn_completed(
                channel=completed_channel,
                chat_id=completed_chat_id,
                session_key=self.session_key,
                metadata=self.lifecycle_message.metadata,
                outcome=outcome,
                failure_kind=failure_kind,
            )

    async def fail(self, *, publish_completion: bool) -> None:
        """The generic error reply after an unexpected failure. (nanobot :410)"""
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=self.lifecycle_message.channel,
                chat_id=self.lifecycle_message.chat_id,
                content="Sorry, I encountered an error.",
                metadata=dict(self.lifecycle_message.metadata or {}),
            )
        )
        if publish_completion:
            await self.runtime_event_publisher.turn_completed(
                channel=self.lifecycle_message.channel,
                chat_id=self.lifecycle_message.chat_id,
                session_key=self.session_key,
                metadata=self.lifecycle_message.metadata,
                outcome="failed",
                failure_kind="internal",
            )

    async def idle(self) -> None:
        """(nanobot :430)"""
        await self.runtime_event_publisher.run_status_changed(self.lifecycle_message, self.session_key, "idle")
        self.runtime_event_publisher.clear_turn(self.session_key)

    def _stream_id(self) -> str:
        assert self._stream_base_id is not None
        return f"{self._stream_base_id}:{self._stream_segment}"

    async def _publish_event(self, event: AgentEvent) -> None:
        """Stream events get the segment's stream id (and are dropped when the
        channel did not ask to stream); others go through the route's filter.
        (nanobot :441)"""
        if isinstance(event, ResponseSourceEvent):
            return
        if isinstance(event, StreamDeltaEvent | StreamEndEvent):
            if not self.streaming:
                return
            event = dataclasses.replace(event, stream_id=self._stream_id())
            if self._routed_events.accepts(type(event)):
                await self.bus.publish_event(
                    event, channel=self.route.channel, chat_id=self.route.chat_id, metadata=deepcopy(self.route.metadata)
                )
        elif self._routed_events.publish is not None:
            await self._routed_events.publish(event)
        if isinstance(event, StreamDeltaEvent):
            self._stream_open = True
        elif isinstance(event, StreamEndEvent):
            self._stream_open = event.merge_next
            if not event.merge_next:
                self._stream_segment += 1

    async def abort_stream(self) -> None:
        """Close an interrupted stream so the channel can release it. (nanobot :484)"""
        if self._stream_open:
            await self._publish_event(StreamEndEvent())
