"""The manager's per-event dispatch for channels that do not project events
themselves (``projects_outbound = False``, Telegram): nanobot's dispatcher
filters (channels/manager.py:930-958) and ``_send_once`` (:1057)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nanoclaw.bus.delivery import DeliveryRejectedError
from nanoclaw.bus.events import OutboundMessage
from nanoclaw.bus.outbound_events import (
    ProgressEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
    outbound_message_for_event,
)
from nanoclaw.bus.queue import MessageBus
from nanoclaw.channels.base import BaseChannel
from nanoclaw.channels.manager import ChannelManager
from nanoclaw.config import Config
from nanoclaw.events import RetryWaitEvent


class EventChannel(BaseChannel):
    name = "tg"
    projects_outbound = False

    def __init__(self, bus: MessageBus) -> None:
        super().__init__({"streaming": True, "allow_from": ["*"]}, bus)
        self.calls: list[tuple[str, Any]] = []
        self.reject = 0

    async def start(self) -> None:
        self._running = True
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self._running = False

    async def send(self, msg: OutboundMessage) -> None:
        if self.reject:
            self.reject -= 1
            raise DeliveryRejectedError("bad chat")
        self.calls.append(("send", msg.content))

    async def send_delta(self, chat_id, delta, metadata=None, **kw) -> None:
        self.calls.append(("delta", (delta, kw)))

    async def send_reasoning_delta(self, chat_id, delta, metadata=None, *, stream_id=None) -> None:
        self.calls.append(("reasoning_delta", delta))

    async def send_reasoning_end(self, chat_id, metadata=None, *, stream_id=None) -> None:
        self.calls.append(("reasoning_end", None))

    async def send_file_edit_events(self, chat_id, edits, metadata=None) -> None:
        self.calls.append(("file_edits", edits))


def _msg(event) -> OutboundMessage:
    return outbound_message_for_event(channel="tg", chat_id="1", event=event)


async def _dispatch(manager: ChannelManager, channel: EventChannel, bus: MessageBus, *msgs, expect: int) -> None:
    for msg in msgs:
        await bus.publish_outbound(msg)
    task = asyncio.create_task(manager._dispatch_outbound_loop())
    try:
        async with asyncio.timeout(5):
            while len(channel.calls) < expect or manager.has_pending_delivery:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def setup():
    bus = MessageBus()
    channel = EventChannel(bus)
    return bus, channel, ChannelManager(Config(), bus, channels={"tg": channel})


async def test_each_event_reaches_its_channel_method(setup) -> None:
    bus, channel, manager = setup
    await _dispatch(
        manager,
        channel,
        bus,
        _msg(StreamDeltaEvent(content="Hel", stream_id="s")),
        _msg(StreamEndEvent(stream_id="s", merge_next=True)),
        _msg(ProgressEvent(content="why", reasoning_delta=True)),
        _msg(ProgressEvent(reasoning_end=True)),
        _msg(ProgressEvent(content="whole", reasoning=True)),
        _msg(ProgressEvent(file_edit_events=[{"path": "a"}])),
        _msg(StreamedResponseEvent()),
        OutboundMessage(channel="tg", chat_id="1", content="final"),
        expect=8,
    )
    assert channel.calls == [
        ("delta", ("Hel", {"stream_id": "s", "stream_end": False, "resuming": False, "merge_next": False})),
        ("delta", ("", {"stream_id": "s", "stream_end": True, "resuming": False, "merge_next": True})),
        ("reasoning_delta", "why"),
        ("reasoning_end", None),
        ("reasoning_delta", "whole"),
        ("reasoning_end", None),
        ("file_edits", [{"path": "a"}]),
        ("send", "final"),
    ]


async def test_unwanted_progress_reasoning_and_retry_waits_are_dropped(setup) -> None:
    bus, channel, manager = setup
    channel.send_progress = False
    channel.send_tool_hints = True
    channel.show_reasoning = False
    await _dispatch(
        manager,
        channel,
        bus,
        _msg(ProgressEvent(content="thinking out loud")),
        _msg(ProgressEvent(content="read a.txt", tool_hint=True)),
        _msg(ProgressEvent(content="hidden", reasoning_delta=True)),
        _msg(RetryWaitEvent(content="retrying in 2s")),
        OutboundMessage(channel="tg", chat_id="1", content="done"),
        expect=2,
    )
    assert channel.calls == [("send", "read a.txt"), ("send", "done")]


async def test_a_rejected_delivery_is_not_retried(setup) -> None:
    bus, channel, manager = setup
    channel.reject = 1
    await _dispatch(
        manager,
        channel,
        bus,
        OutboundMessage(channel="tg", chat_id="1", content="lost"),
        OutboundMessage(channel="tg", chat_id="1", content="next"),
        expect=1,
    )
    assert channel.calls == [("send", "next")]
