"""Queued delivery of messages and typed events between the agent and channels.

Copied from nanobot/bus/queue.py. Left out: ``RuntimeAdmission`` (nanobot
runtime_admission.py; refuses new input while the process shuts down, marked
SKIP in the research notes) and ``outbound_recorder`` (a debug hook).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar, overload

from loguru import logger

from nanoclaw.bus.events import InboundMessage, OutboundMessage
from nanoclaw.bus.outbound_events import outbound_message_for_event
from nanoclaw.events import AgentEvent

_EventT = TypeVar("_EventT", bound=AgentEvent)
EventHandler = Callable[[AgentEvent], Awaitable[None] | None]


class MessageBus:
    """Async message bus between chat channels and the agent. (nanobot queue.py:20)

    Channels push messages to the inbound queue. The agent publishes text,
    media or typed events to the outbound queue; channel adapters decide how
    to show them.

    Local subscribers are awaited by ``publish``; channel delivery is queued by
    ``publish_event``. Local state changes never wait for network sends.
    """

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._handlers: list[EventHandler] = []
        self._pending: set[asyncio.Task[None]] = set()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """A message from a channel to the agent. (nanobot queue.py:40)"""
        self.inbound.put_nowait(msg)

    async def consume_inbound(self) -> InboundMessage:
        """The next inbound message; waits until there is one. (nanobot queue.py:46)"""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Queue a routed message or event for its channel. (nanobot queue.py:50)"""
        await self.outbound.put(msg)

    async def publish_event(
        self,
        event: AgentEvent,
        *,
        channel: str,
        chat_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Queue a typed event wrapped in an OutboundMessage. (nanobot queue.py:56)

        The bus does not look inside events; known ones get their text
        fallback as ``content``.
        """
        await self.publish_outbound(
            outbound_message_for_event(channel=channel, chat_id=chat_id, event=event, metadata=metadata)
        )

    async def consume_outbound(self) -> OutboundMessage:
        """The next outbound message; waits until there is one. (nanobot queue.py:74)"""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()

    @property
    def has_pending_events(self) -> bool:
        """Scheduled local dispatches still running. (nanobot queue.py:89)"""
        return bool(self._pending)

    @overload
    def subscribe(
        self, handler: Callable[[_EventT], Awaitable[None] | None], event_type: type[_EventT]
    ) -> Callable[[], None]: ...

    @overload
    def subscribe(self, handler: EventHandler, event_type: None = None) -> Callable[[], None]: ...

    def subscribe(
        self,
        handler: Callable[..., Awaitable[None] | None],
        event_type: type[AgentEvent] | None = None,
    ) -> Callable[[], None]:
        """Add an ordered, awaited handler (only for *event_type* if given).

        Returns a function that removes it; calling that twice is harmless.
        (nanobot queue.py:104)
        """
        active = True

        def entry(event: AgentEvent) -> Awaitable[None] | None:
            if active and (event_type is None or isinstance(event, event_type)):
                return handler(event)
            return None

        self._handlers.append(entry)

        def _unsubscribe() -> None:
            nonlocal active
            active = False
            with contextlib.suppress(ValueError):
                self._handlers.remove(entry)

        return _unsubscribe

    async def publish(self, event: AgentEvent) -> None:
        """Await local subscribers in the order they subscribed. (nanobot queue.py:130)

        Nothing goes to channels. A failing handler is logged and the rest
        still run.
        """
        for handler in list(self._handlers):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("event handler failed for {}", type(event).__name__)

    def publish_nowait(self, event: AgentEvent) -> asyncio.Task[None] | None:
        """Schedule ``publish`` without waiting, keeping the task until it ends.
        (nanobot queue.py:140)

        Handlers still run in order for one event, but separate publications
        may interleave. Without a running event loop the event is dropped.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("dropping event without a running loop: {}", type(event).__name__)
            return None
        task = loop.create_task(self.publish(event))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def drain(self) -> None:
        """Finish scheduled dispatches after producers stop. (nanobot queue.py:157)"""
        while self._pending:
            pending = tuple(self._pending)
            await asyncio.gather(*pending, return_exceptions=True)
            self._pending.difference_update(pending)
