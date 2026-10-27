"""The message bus (nanobot bus/events.py, bus/queue.py, bus/outbound_events.py)."""

import asyncio
from dataclasses import dataclass

from nanoclaw.bus import (
    InboundMessage,
    MessageBus,
    OutboundMessage,
    outbound_message_for_event,
    replace_outbound_event,
)
from nanoclaw.bus.outbound_events import ProgressEvent, StreamDeltaEvent
from nanoclaw.events import AgentEvent, ContextCompactionEvent


@dataclass(frozen=True)
class Ping(AgentEvent):
    n: int = 0


@dataclass(frozen=True)
class Pong(AgentEvent):
    n: int = 0


def test_inbound_session_key_and_user_input():
    msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="42", content="hi")
    assert msg.session_key == "telegram:42"
    assert msg.is_user_input
    assert msg.media == [] and msg.metadata == {}
    assert InboundMessage("cli", "u", "direct", "x", session_key_override="cli:other").session_key == "cli:other"
    assert not InboundMessage("system", "subagent", "direct", "report").is_user_input
    assert InboundMessage("system", "u", "d", "x", input_role="user").is_user_input
    assert not InboundMessage("cli", "u", "d", "x", input_role="system").is_user_input


def test_outbound_defaults():
    msg = OutboundMessage(channel="cli", chat_id="direct", content="hi")
    assert (msg.reply_to, msg.media, msg.metadata, msg.buttons, msg.event) == (None, [], {}, [], None)


async def test_inbound_and_outbound_queues_are_fifo():
    bus = MessageBus()
    first = InboundMessage("cli", "u", "direct", "one")
    second = InboundMessage("cli", "u", "direct", "two")
    await bus.publish_inbound(first)
    await bus.publish_inbound(second)
    assert bus.inbound_size == 2
    assert await bus.consume_inbound() is first
    assert await bus.consume_inbound() is second

    out = OutboundMessage("cli", "direct", "reply")
    await bus.publish_outbound(out)
    assert bus.outbound_size == 1
    assert await bus.consume_outbound() is out
    assert bus.outbound_size == 0


async def test_consume_waits_for_a_message():
    bus = MessageBus()
    waiting = asyncio.create_task(bus.consume_inbound())
    await asyncio.sleep(0)
    assert not waiting.done()
    msg = InboundMessage("cli", "u", "direct", "late")
    await bus.publish_inbound(msg)
    assert await waiting is msg


async def test_publish_event_wraps_the_event_with_its_text_fallback():
    bus = MessageBus()
    event = ProgressEvent(content="Reading files")
    await bus.publish_event(event, channel="cli", chat_id="direct", metadata={"message_id": 7})
    msg = await bus.consume_outbound()
    assert (msg.channel, msg.chat_id, msg.content, msg.event, msg.metadata) == (
        "cli",
        "direct",
        "Reading files",
        event,
        {"message_id": 7},
    )


def test_outbound_message_for_event_and_replace():
    started = ContextCompactionEvent(compaction_id="c1", phase="started")
    metadata = {"thread": "t"}
    msg = outbound_message_for_event(channel="cli", chat_id="direct", event=started, metadata=metadata)
    assert msg.content == "Compressing context…" and msg.event is started
    assert msg.metadata == metadata and msg.metadata is not metadata  # copied
    assert outbound_message_for_event(channel="c", chat_id="d", event=started, content="custom").content == "custom"
    assert outbound_message_for_event(channel="c", chat_id="d", event=Ping()).content == ""

    done = ContextCompactionEvent(compaction_id="c1", phase="succeeded")
    replaced = replace_outbound_event(msg, done)
    assert replaced.content == "Context compacted." and replaced.event is done
    assert replaced.metadata == metadata and msg.event is started  # original unchanged
    assert replace_outbound_event(msg, StreamDeltaEvent(content="x"), content="y").content == "y"


async def test_subscribers_run_in_order_with_type_filter():
    bus = MessageBus()
    seen = []

    async def first(event):
        seen.append(("first", event))

    bus.subscribe(first)
    bus.subscribe(lambda event: seen.append(("pings", event)), Ping)
    await bus.publish(Ping(1))
    await bus.publish(Pong(2))
    assert seen == [("first", Ping(1)), ("pings", Ping(1)), ("first", Pong(2))]
    assert bus.outbound_size == 0  # local events never reach channels


async def test_unsubscribe_is_idempotent_and_failures_are_isolated():
    bus = MessageBus()
    seen = []

    def broken(event):
        raise RuntimeError("boom")

    bus.subscribe(broken)
    off = bus.subscribe(seen.append)
    await bus.publish(Ping(1))
    off()
    off()
    await bus.publish(Ping(2))
    assert seen == [Ping(1)]


async def test_unsubscribe_during_dispatch_skips_the_removed_handler():
    bus = MessageBus()
    seen = []
    offs = []

    def remover(event):
        offs[0]()

    bus.subscribe(remover)
    offs.append(bus.subscribe(seen.append))
    await bus.publish(Ping())
    assert seen == []


async def test_publish_nowait_and_drain():
    bus = MessageBus()
    gate = asyncio.Event()
    seen = []

    async def slow(event):
        await gate.wait()
        seen.append(event)

    bus.subscribe(slow)
    task = bus.publish_nowait(Ping(1))
    assert task is not None and bus.has_pending_events
    await asyncio.sleep(0)
    assert seen == []
    gate.set()
    await bus.drain()
    assert seen == [Ping(1)] and not bus.has_pending_events


def test_publish_nowait_without_a_running_loop_drops_the_event():
    assert MessageBus().publish_nowait(Ping()) is None
