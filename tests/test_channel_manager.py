"""The channel manager (layer 23): coalescing, duplicate suppression, ordering, retry, status.

Coalescing tests adapted from nanobot tests/channels/test_channel_manager_delta_coalescing.py
(c4dac64); the rest from nanobot's manager behaviour (channels/manager.py).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.bus.events import OutboundMessage
from nanoclaw.bus.outbound_events import (
    ProgressEvent,
    StreamDeltaEvent,
    StreamEndEvent,
    outbound_message_for_event,
)
from nanoclaw.bus.queue import MessageBus
from nanoclaw.channels import manager as manager_module
from nanoclaw.channels.base import BaseChannel
from nanoclaw.channels.manager import ChannelManager
from nanoclaw.cli.gateway import GatewayHealthIdentity, health_payload
from nanoclaw.config import Config
from nanoclaw.providers import FakeProvider
from nanoclaw.session import SessionManager


class MockChannel(BaseChannel):
    name = "mock"
    display_name = "Mock"

    def __init__(self, config: Any, bus: MessageBus) -> None:
        super().__init__(config, bus)
        self.sent: list[OutboundMessage] = []
        self.fail_times = 0
        self.started = asyncio.Event()

    async def start(self) -> None:
        self._running = True
        self.started.set()
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self._running = False

    async def send(self, msg: OutboundMessage) -> None:
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("flaky")
        self.sent.append(msg)


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus()


@pytest.fixture
def channel(bus: MessageBus) -> MockChannel:
    return MockChannel({"streaming": True, "allow_from": ["*"]}, bus)


@pytest.fixture
def manager(bus: MessageBus, channel: MockChannel) -> ChannelManager:
    return ChannelManager(Config(), bus, channels={"mock": channel})


def _delta(content: str, *, chat_id: str = "chat1", stream_id: str | None = None) -> OutboundMessage:
    return outbound_message_for_event(channel="mock", chat_id=chat_id, event=StreamDeltaEvent(content=content, stream_id=stream_id))


def _end(content: str = "", *, chat_id: str = "chat1", stream_id: str | None = None, **kw: bool) -> OutboundMessage:
    return outbound_message_for_event(
        channel="mock", chat_id=chat_id, event=StreamEndEvent(content=content, stream_id=stream_id, **kw)
    )


# -- coalescing (nanobot test_channel_manager_delta_coalescing.py) ----------------------------


async def test_response_source_change_is_a_coalescing_boundary(manager, bus) -> None:
    first, second = _delta("primary"), _delta("backup")
    first.metadata["response_sources"] = [{"provider": "openai", "model": "m", "preset": "a"}]
    second.metadata["response_sources"] = [{"provider": "openai", "model": "m", "preset": "b"}]
    await bus.publish_outbound(second)
    merged, pending = manager._coalesce_stream_deltas(first)
    assert merged.content == "primary"
    assert pending == [second]


async def test_multiple_deltas_coalesced(manager, bus) -> None:
    for text in ["Hello", " ", "world", "!"]:
        await bus.publish_outbound(_delta(text))
    merged, pending = manager._coalesce_stream_deltas(await bus.consume_outbound())
    assert merged.content == "Hello world!" and isinstance(merged.event, StreamDeltaEvent)
    assert pending == []


async def test_deltas_different_chats_not_coalesced(manager, bus) -> None:
    await bus.publish_outbound(_delta("Hello", chat_id="chat1"))
    await bus.publish_outbound(_delta("World", chat_id="chat2"))
    merged, pending = manager._coalesce_stream_deltas(await bus.consume_outbound())
    assert (merged.content, merged.chat_id) == ("Hello", "chat1")
    assert [(m.chat_id, m.content) for m in pending] == [("chat2", "World")]


async def test_deltas_different_stream_ids_not_coalesced(manager, bus) -> None:
    await bus.publish_outbound(_delta("A1", stream_id="a"))
    await bus.publish_outbound(_delta("B1", stream_id="b"))
    merged, pending = manager._coalesce_stream_deltas(await bus.consume_outbound())
    assert merged.content == "A1" and merged.event.stream_id == "a"
    assert pending[0].content == "B1"


async def test_stream_end_terminates_coalescing(manager, bus) -> None:
    await bus.publish_outbound(_delta("Hello"))
    await bus.publish_outbound(_end(" world", resuming=True, merge_next=True))
    merged, pending = manager._coalesce_stream_deltas(await bus.consume_outbound())
    assert merged.content == "Hello world"
    assert isinstance(merged.event, StreamEndEvent) and merged.event.resuming and merged.event.merge_next
    assert pending == []


async def test_coalescing_stops_at_first_non_matching_boundary(manager, bus) -> None:
    await bus.publish_outbound(_delta("Hello", stream_id="seg-1"))
    await bus.publish_outbound(_end(stream_id="seg-1"))
    await bus.publish_outbound(_delta("world", stream_id="seg-2"))
    merged, pending = manager._coalesce_stream_deltas(await bus.consume_outbound())
    assert merged.content == "Hello"
    assert len(pending) == 1 and isinstance(pending[0].event, StreamEndEvent)
    assert (await bus.consume_outbound()).content == "world"


async def test_non_delta_message_preserved(manager, bus) -> None:
    await bus.publish_outbound(_delta("Delta"))
    await bus.publish_outbound(OutboundMessage(channel="mock", chat_id="chat1", content="Final"))
    merged, pending = manager._coalesce_stream_deltas(await bus.consume_outbound())
    assert merged.content == "Delta" and pending[0].content == "Final" and pending[0].event is None


@pytest.mark.parametrize("end_stream", [False, True])
async def test_dispatch_coalesces_and_delivers_in_order(manager, bus, channel, end_stream) -> None:
    await bus.publish_outbound(_delta("A", stream_id="s"))
    await bus.publish_outbound(_delta("B", stream_id="s"))
    if end_stream:
        await bus.publish_outbound(_end(stream_id="s"))
    await bus.publish_outbound(OutboundMessage(channel="mock", chat_id="chat1", content="after"))
    task = asyncio.create_task(manager._dispatch_outbound_loop())
    try:
        async with asyncio.timeout(3):
            while not channel.sent or channel.sent[-1].content != "after":
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert channel.sent[0].content == "AB"
    assert [m.content for m in channel.sent][-1] == "after"
    if end_stream:
        assert isinstance(channel.sent[1].event, StreamEndEvent)


# -- suppression, retry, unknown channels ---------------------------------------------------------


def test_duplicate_reply_to_same_origin_is_suppressed(manager) -> None:
    first = OutboundMessage(channel="mock", chat_id="c", content="Same  answer", metadata={"origin_message_id": "m1"})
    again = OutboundMessage(channel="mock", chat_id="c", content="Same answer", metadata={"origin_message_id": "m1"})
    other = OutboundMessage(channel="mock", chat_id="c", content="Same answer", metadata={"origin_message_id": "m2"})
    assert not manager._should_suppress_outbound(first)
    assert manager._should_suppress_outbound(again)
    assert not manager._should_suppress_outbound(other)
    progress = outbound_message_for_event(
        channel="mock", chat_id="c", event=ProgressEvent(content="x"), metadata={"origin_message_id": "m1"}
    )
    assert not manager._should_suppress_outbound(progress)


async def test_send_is_retried_with_backoff(manager, channel, monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(manager_module.asyncio, "sleep", fake_sleep)
    channel.fail_times = 2
    await manager._send_with_retry(channel, OutboundMessage(channel="mock", chat_id="c", content="x"))
    assert sleeps == [1, 2] and len(channel.sent) == 1


async def test_send_gives_up_after_max_attempts(bus, channel, monkeypatch) -> None:
    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(manager_module.asyncio, "sleep", fake_sleep)
    config = Config.model_validate({"channels": {"sendMaxRetries": 2}})
    mgr = ChannelManager(config, bus, channels={"mock": channel})
    channel.fail_times = 5
    await mgr._send_with_retry(channel, OutboundMessage(channel="mock", chat_id="c", content="x"))
    assert channel.sent == [] and channel.fail_times == 3


async def test_messages_for_unknown_channels_are_dropped(manager, bus, channel) -> None:
    await bus.publish_outbound(OutboundMessage(channel="cli", chat_id="direct", content="x"))
    await bus.publish_outbound(OutboundMessage(channel="mock", chat_id="c", content="y"))
    task = asyncio.create_task(manager._dispatch_outbound_loop())
    try:
        async with asyncio.timeout(3):
            while not channel.sent:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert [m.content for m in channel.sent] == ["y"]


# -- lifecycle and status -------------------------------------------------------------------------------


async def test_start_and_stop_report_status(manager, channel) -> None:
    assert manager.get_status()["mock"]["state"] == "stopped"
    runner = asyncio.create_task(manager.start_all())
    await asyncio.wait_for(channel.started.wait(), 3)
    status = manager.get_status()["mock"]
    assert status == {"enabled": True, "running": True, "state": "running", "owner": "mock", "instance_id": "default"}
    await manager.stop_all()
    await asyncio.wait_for(runner, 3)
    assert manager.get_status()["mock"]["running"] is False


async def test_failed_start_is_reported(bus) -> None:
    class Broken(MockChannel):
        async def start(self) -> None:
            raise OSError("port in use")

    mgr = ChannelManager(Config(), bus, channels={"mock": Broken({}, bus)})
    await mgr.start_all()
    status = mgr.get_status()["mock"]
    assert status["state"] == "failed" and status["error"] == "Channel failed to start. Check gateway logs."
    await mgr.stop_all()


def test_no_channels_without_websocket_enabled(bus) -> None:
    mgr = ChannelManager(Config(), bus)
    assert mgr.enabled_channels == [] and mgr.get_status() == {}


def test_websocket_channel_built_with_flag_overrides(tmp_path, bus) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    agent = AgentLoop(FakeProvider([]), ws, sessions=SessionManager(ws, sessions_root=tmp_path / "s"), bus=bus)
    config = Config.model_validate(
        {"channels": {"sendProgress": False, "websocket": {"enabled": True, "port": 0, "showReasoning": False}}}
    )
    mgr = ChannelManager(config, bus, agent=agent)
    ch = mgr.get_channel("websocket")
    assert mgr.enabled_channels == ["websocket"] and ch is not None
    assert (ch.send_progress, ch.send_tool_hints, ch.show_reasoning) == (False, True, False)


def test_websocket_without_agent_is_a_failed_channel(bus) -> None:
    config = Config.model_validate({"channels": {"websocket": {"enabled": True}}})
    mgr = ChannelManager(config, bus)
    assert mgr.enabled_channels == []
    assert mgr.get_status()["websocket"]["state"] == "failed"


def test_health_reports_websocket_state(manager, bus) -> None:
    identity = GatewayHealthIdentity.create(Path("c.json"), Path("w"))

    class WsStatus:
        enabled_channels = ["websocket"]

        def __init__(self, state: str, running: bool) -> None:
            self._status = {"websocket": {"running": running, "state": state}}

        def get_status(self) -> dict[str, Any]:
            return self._status

    ready, body = health_payload(WsStatus("running", True), identity)
    assert ready and body["websocket"] == "running" and body["status"] == "ok"
    ready, body = health_payload(WsStatus("starting", False), identity)
    assert not ready and body["websocket"] == "starting" and body["status"] == "degraded"
    ready, body = health_payload(ChannelManager(Config(), bus), identity)
    assert ready and body["websocket"] == "disabled"
