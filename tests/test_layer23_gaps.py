"""Layer 23 behaviour pinned down by mutation checks.

Each test here failed with a mutant of the layer 23 code (channels, the web UI
listener, the REST API) that the other tests let through.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from nanoclaw.bus.events import OutboundMessage
from nanoclaw.bus.queue import MessageBus
from nanoclaw.channels.base import BaseChannel
from nanoclaw.channels.manager import ChannelManager
from nanoclaw.config import Config
import test_websocket_channel
import webui_history_fixture
from test_websocket_channel import Harness


# The shared fixtures of the channel and history tests.
harness = test_websocket_channel.harness
history = webui_history_fixture.history


class _Channel(BaseChannel):
    name = "probe"

    async def start(self) -> None:  # pragma: no cover - not started
        pass

    async def stop(self) -> None:  # pragma: no cover
        pass

    async def send(self, msg: OutboundMessage) -> None:  # pragma: no cover
        pass


# -- channels/base.py ----------------------------------------------------------------


def test_streaming_flag_is_read_from_a_dict_config() -> None:
    bus = MessageBus()
    assert _Channel({"streaming": True}, bus).supports_streaming is True
    assert _Channel({"streaming": False}, bus).supports_streaming is False
    assert _Channel({}, bus).supports_streaming is False


def test_allow_list_accepts_camel_case_key() -> None:
    bus = MessageBus()
    channel = _Channel({"allowFrom": ["alice"]}, bus)
    assert channel.is_allowed("alice")
    assert not channel.is_allowed("bob")


async def test_handle_message_denies_unlisted_senders_and_keeps_session_override() -> None:
    bus = MessageBus()
    channel = _Channel({"allow_from": ["alice"]}, bus)
    assert await channel._handle_message("bob", "c", "hi") is False
    assert bus.inbound.empty()
    assert await channel._handle_message("alice", "c", "hi", session_key="websocket:shared") is True
    msg = bus.inbound.get_nowait()
    assert msg.session_key == "websocket:shared"
    assert msg.sender_id == "alice" and msg.content == "hi"
    assert "_wants_stream" not in msg.metadata


# -- channels/manager.py -------------------------------------------------------------


class _Mock(BaseChannel):
    name = "mock"

    def __init__(self, bus: MessageBus, *, name: str = "mock") -> None:
        super().__init__({}, bus)
        self.name = name
        self.fail_start = 0
        self.fail_stop = False
        self.block_send = False
        self.sent: list[OutboundMessage] = []
        self.stopped = False

    async def start(self) -> None:
        if self.fail_start:
            self.fail_start -= 1
            raise OSError("port in use")
        self._running = True
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self._running = False
        self.stopped = True
        if self.fail_stop:
            raise RuntimeError("stop failed")

    async def send(self, msg: OutboundMessage) -> None:
        if self.block_send:
            await asyncio.Event().wait()
        self.sent.append(msg)


async def _wait_for(predicate: Any, timeout: float = 3) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_restart_clears_a_previous_start_failure() -> None:
    bus = MessageBus()
    channel = _Mock(bus)
    channel.fail_start = 1
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})
    await mgr.start_all()
    assert mgr.get_status()["mock"]["state"] == "failed"
    await mgr.stop_all()
    runner = asyncio.create_task(mgr.start_all())
    await _wait_for(lambda: channel.is_running)
    status = mgr.get_status()["mock"]
    assert status["state"] == "running" and "error" not in status
    await mgr.stop_all()
    await asyncio.wait_for(runner, 3)


async def test_no_channels_leaves_the_outbound_queue_alone() -> None:
    bus = MessageBus()
    mgr = ChannelManager(Config(), bus)
    await bus.publish_outbound(OutboundMessage(channel="cli", chat_id="direct", content="x"))
    await mgr.start_all()
    await asyncio.sleep(0.05)
    assert bus.outbound_size == 1
    assert mgr._dispatch_task is None
    await mgr.stop_all()


async def test_stop_cancels_in_flight_sends() -> None:
    bus = MessageBus()
    channel = _Mock(bus)
    channel.block_send = True
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})
    runner = asyncio.create_task(mgr.start_all())
    await _wait_for(lambda: channel.is_running)
    await bus.publish_outbound(OutboundMessage(channel="mock", chat_id="c", content="stuck"))
    await _wait_for(lambda: bool(mgr._outbound_tasks))
    await asyncio.wait_for(mgr.stop_all(), 3)
    await asyncio.sleep(0)
    assert not mgr._outbound_tasks
    assert not mgr.has_pending_delivery
    await asyncio.wait_for(runner, 3)


async def test_a_failing_stop_does_not_keep_other_channels_running() -> None:
    bus = MessageBus()
    first, second = _Mock(bus, name="first"), _Mock(bus, name="second")
    first.fail_stop = True
    mgr = ChannelManager(Config(), bus, channels={"first": first, "second": second})
    runner = asyncio.create_task(mgr.start_all())
    await _wait_for(lambda: first.is_running and second.is_running)
    await asyncio.wait_for(mgr.stop_all(), 3)
    assert first.stopped and second.stopped
    await asyncio.wait_for(runner, 3)


def test_empty_replies_are_never_suppressed_as_duplicates() -> None:
    mgr = ChannelManager(Config(), MessageBus(), channels={})
    for _ in range(2):
        msg = OutboundMessage(channel="mock", chat_id="c", content="  ", metadata={"origin_message_id": "m1"})
        assert not mgr._should_suppress_outbound(msg)
    assert not mgr._origin_reply_fingerprints


def test_reply_fingerprints_are_bounded_least_recent_first() -> None:
    from nanoclaw.channels.manager import ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE

    mgr = ChannelManager(Config(), MessageBus(), channels={})
    for index in range(ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE + 1):
        msg = OutboundMessage(
            channel="mock", chat_id="c", content=f"a{index}", metadata={"origin_message_id": f"m{index}"}
        )
        assert not mgr._should_suppress_outbound(msg)
    assert len(mgr._origin_reply_fingerprints) == ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE
    assert ("mock", "c", "m0") not in mgr._origin_reply_fingerprints
    assert ("mock", "c", f"m{ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE}") in mgr._origin_reply_fingerprints


def _reply(content: str, chat_id: str = "c", **metadata: Any) -> OutboundMessage:
    return OutboundMessage(channel="mock", chat_id=chat_id, content=content, metadata=metadata)


def test_a_changed_reply_refreshes_its_fingerprint_position() -> None:
    from nanoclaw.channels.manager import ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE

    mgr = ChannelManager(Config(), MessageBus(), channels={})
    for index in range(ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE):
        assert not mgr._should_suppress_outbound(_reply(f"a{index}", origin_message_id=f"m{index}"))
    assert not mgr._should_suppress_outbound(_reply("changed", origin_message_id="m0"))
    assert not mgr._should_suppress_outbound(_reply("new", origin_message_id="overflow"))
    assert ("mock", "c", "m0") in mgr._origin_reply_fingerprints
    assert ("mock", "c", "m1") not in mgr._origin_reply_fingerprints
    assert mgr._should_suppress_outbound(_reply("changed", origin_message_id="m0"))


def test_progress_repeating_a_reply_is_not_suppressed() -> None:
    from nanoclaw.bus.outbound_events import ProgressEvent

    mgr = ChannelManager(Config(), MessageBus(), channels={})
    assert not mgr._should_suppress_outbound(_reply("Working on it", origin_message_id="m1"))
    progress = OutboundMessage(
        channel="mock", chat_id="c", content="Working on it", metadata={"origin_message_id": "m1"},
        event=ProgressEvent(content="Working on it"),
    )
    assert not mgr._should_suppress_outbound(progress)


def test_reply_to_a_sent_message_id_repeating_it_is_suppressed() -> None:
    mgr = ChannelManager(Config(), MessageBus(), channels={})
    assert not mgr._should_suppress_outbound(_reply("Done.", message_id="out-1"))
    assert mgr._should_suppress_outbound(_reply("Done.", origin_message_id="out-1"))


def test_same_origin_id_in_another_chat_is_not_a_duplicate() -> None:
    mgr = ChannelManager(Config(), MessageBus(), channels={})
    assert not mgr._should_suppress_outbound(_reply("Same", chat_id="a", origin_message_id="m1"))
    assert not mgr._should_suppress_outbound(_reply("Same", chat_id="b", origin_message_id="m1"))


async def test_pending_delivery_covers_every_stage() -> None:
    bus = MessageBus()
    channel = _Mock(bus)
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})
    assert not mgr.has_pending_delivery
    await bus.publish_outbound(_reply("queued"))
    assert mgr.has_pending_delivery
    await bus.consume_outbound()
    assert not mgr.has_pending_delivery
    mgr._outbound_pending.append(_reply("held"))
    assert mgr.has_pending_delivery
    mgr._outbound_pending.clear()

    class Busy(_Mock):
        @property
        def has_pending_delivery(self) -> bool:
            return True

    assert ChannelManager(Config(), bus, channels={"mock": Busy(bus)}).has_pending_delivery
    channel.block_send = True
    await mgr._queue_outbound(channel, _reply("in flight"))
    assert mgr.has_pending_delivery
    await mgr._cancel_outbound()


async def _run_dispatch(mgr: ChannelManager, until: Any) -> None:
    task = asyncio.create_task(mgr._dispatch_outbound_loop())
    try:
        await _wait_for(until)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_sends_to_one_chat_keep_their_order() -> None:
    bus = MessageBus()

    class Slow(_Mock):
        async def send(self, msg: OutboundMessage) -> None:
            await asyncio.sleep(0.1 if msg.content == "first" else 0)
            self.sent.append(msg)

    channel = Slow(bus)
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})
    await bus.publish_outbound(_reply("first"))
    await bus.publish_outbound(_reply("second"))
    await _run_dispatch(mgr, lambda: len(channel.sent) == 2)
    assert [m.content for m in channel.sent] == ["first", "second"]


async def test_delivered_sends_release_their_slot_and_task() -> None:
    from nanoclaw.channels.manager import _OUTBOUND_PENDING_LIMIT

    bus = MessageBus()
    channel = _Mock(bus)
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})
    total = _OUTBOUND_PENDING_LIMIT + 10
    for index in range(total):
        await bus.publish_outbound(_reply(f"m{index}", chat_id=f"c{index}"))
    await _run_dispatch(mgr, lambda: len(channel.sent) == total)
    await asyncio.sleep(0)
    assert not mgr._outbound_tasks and not mgr.has_pending_delivery


async def test_model_update_without_web_ui_is_dropped_quietly() -> None:
    from loguru import logger

    from nanoclaw.bus.outbound_events import RuntimeModelUpdatedEvent, outbound_message_for_event

    bus = MessageBus()
    channel = _Mock(bus)
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})
    warnings: list[str] = []
    logger.enable("nanoclaw")
    sink = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        await bus.publish_outbound(
            outbound_message_for_event(channel="websocket", chat_id="c", event=RuntimeModelUpdatedEvent(model="m"))
        )
        await bus.publish_outbound(OutboundMessage(channel="nowhere", chat_id="c", content="lost"))
        await bus.publish_outbound(_reply("after"))
        await _run_dispatch(mgr, lambda: bool(channel.sent))
    finally:
        logger.remove(sink)
        logger.disable("nanoclaw")
    assert [w for w in warnings if "Unknown channel" in w] == ["Unknown channel: nowhere"]


async def test_identical_stream_deltas_are_never_suppressed() -> None:
    from nanoclaw.bus.outbound_events import StreamDeltaEvent, outbound_message_for_event

    bus = MessageBus()
    channel = _Mock(bus)
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})

    def delta() -> OutboundMessage:
        return outbound_message_for_event(
            channel="mock", chat_id="c", event=StreamDeltaEvent(content="ha", stream_id="s"),
            metadata={"origin_message_id": "m1"},
        )

    await bus.publish_outbound(delta())
    await bus.publish_outbound(_reply("between"))
    await bus.publish_outbound(delta())
    await _run_dispatch(mgr, lambda: len(channel.sent) == 3)
    assert [m.content for m in channel.sent] == ["ha", "between", "ha"]


async def test_dispatcher_drops_a_repeated_reply() -> None:
    bus = MessageBus()
    channel = _Mock(bus)
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})
    await bus.publish_outbound(_reply("answer", origin_message_id="m1"))
    await bus.publish_outbound(_reply("answer", origin_message_id="m1"))
    await bus.publish_outbound(_reply("done"))
    await _run_dispatch(mgr, lambda: bool(channel.sent) and channel.sent[-1].content == "done")
    assert [m.content for m in channel.sent] == ["answer", "done"]


async def test_coalescing_stops_after_a_stream_end_with_text() -> None:
    from nanoclaw.bus.outbound_events import StreamDeltaEvent, StreamEndEvent, outbound_message_for_event

    bus = MessageBus()
    mgr = ChannelManager(Config(), bus, channels={})

    def msg(event: Any) -> OutboundMessage:
        return outbound_message_for_event(channel="mock", chat_id="c", event=event)

    await bus.publish_outbound(msg(StreamEndEvent(content="b", stream_id="s")))
    await bus.publish_outbound(msg(StreamDeltaEvent(content="c", stream_id="s")))
    merged, pending = mgr._coalesce_stream_deltas(msg(StreamDeltaEvent(content="a", stream_id="s")))
    assert merged.content == "ab" and isinstance(merged.event, StreamEndEvent)
    assert pending == []
    assert (await bus.consume_outbound()).content == "c"


async def test_non_retryable_send_errors_are_not_retried(monkeypatch) -> None:
    from nanoclaw.channels import manager as manager_module

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(manager_module.asyncio, "sleep", fake_sleep)
    bus = MessageBus()

    class Fatal(_Mock):
        attempts = 0

        async def send(self, msg: OutboundMessage) -> None:
            self.attempts += 1
            raise ValueError("bad payload")

        def should_retry_send_error(self, error: Exception) -> bool:
            return not isinstance(error, ValueError)

    channel = Fatal(bus)
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})
    await mgr._send_with_retry(channel, _reply("x"))
    assert channel.attempts == 1 and sleeps == []


async def test_retry_backoff_doubles_then_stays_at_four_seconds(monkeypatch) -> None:
    from nanoclaw.channels import manager as manager_module

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(manager_module.asyncio, "sleep", fake_sleep)
    bus = MessageBus()

    class Flaky(_Mock):
        failures = 4

        async def send(self, msg: OutboundMessage) -> None:
            if self.failures:
                self.failures -= 1
                raise ConnectionError("flaky")
            self.sent.append(msg)

    channel = Flaky(bus)
    mgr = ChannelManager(Config.model_validate({"channels": {"sendMaxRetries": 5}}), bus, channels={"mock": channel})
    await mgr._send_with_retry(channel, _reply("x"))
    assert sleeps == [1, 2, 4, 4] and len(channel.sent) == 1


async def test_status_is_starting_until_the_channel_runs() -> None:
    bus = MessageBus()
    gate = asyncio.Event()

    class SlowStart(_Mock):
        async def start(self) -> None:
            await gate.wait()
            await super().start()

    channel = SlowStart(bus)
    mgr = ChannelManager(Config(), bus, channels={"mock": channel})
    runner = asyncio.create_task(mgr.start_all())
    await _wait_for(lambda: "mock" in mgr._channel_tasks)
    assert mgr.get_status()["mock"]["state"] == "starting"
    gate.set()
    await _wait_for(lambda: channel.is_running)
    assert mgr.get_status()["mock"]["state"] == "running"
    await mgr.stop_all()
    await asyncio.wait_for(runner, 3)


# -- channels/websocket/runtime.py ----------------------------------------------------

def test_legacy_payload_skips_blank_fields() -> None:
    from nanoclaw.channels.websocket.runtime import _parse_inbound_payload

    assert _parse_inbound_payload('{"content": "  ", "text": "t"}') == "t"


async def test_legacy_frames_carry_the_peer_and_skip_contentless_json(harness: Harness) -> None:
    client = await harness.connect()
    await client.send('{"other": 1}')
    await client.send("hi")
    await asyncio.sleep(0.1)
    [msg] = harness.inbound()
    assert msg.content == "hi"
    remote = msg.metadata["remote"]
    assert isinstance(remote, tuple) and remote[0] == "127.0.0.1" and isinstance(remote[1], int)
    await client.ws.close()


async def test_upgrade_needs_a_connection_upgrade_header(harness: Harness) -> None:
    resp = await harness.get("/", {"Upgrade": "websocket"})
    assert resp.status == 200 and await resp.read() == b"<html>app</html>"


@pytest.mark.ws_config({"path": "/ws"})
async def test_socket_only_upgrades_on_its_path(harness: Harness) -> None:
    import aiohttp

    token = (await harness.bootstrap())["token"]
    with pytest.raises(aiohttp.WSServerHandshakeError):
        await harness.http.ws_connect(f"{harness.base}/?token={token}")
    ws = await harness.http.ws_connect(f"{harness.base}/ws?token={token}")
    assert (await ws.receive_json())["event"] == "ready"
    await ws.close()


@pytest.mark.ws_config({"token": "static"})
async def test_bootstrap_token_opens_a_trusted_socket_beside_a_static_token(harness: Harness) -> None:
    client = await harness.connect(headers={"X-Nanobot-Auth": "static"})
    await asyncio.sleep(0.05)
    assert [c.trusted_webui for c in harness.channel._connections] == [True]
    await client.ws.close()


@pytest.mark.ws_config({"websocketRequiresToken": False})
async def test_optional_token_sockets_are_untrusted_without_a_bootstrap_token(harness: Harness) -> None:
    anon = await harness.connect(trusted=False)
    bogus = await harness.connect("?token=bogus", trusted=False)
    trusted = await harness.connect()
    await asyncio.sleep(0.05)
    by_id = {c.client_id: c.trusted_webui for c in harness.channel._connections}
    assert by_id == {
        anon.ready["client_id"]: False, bogus.ready["client_id"]: False, trusted.ready["client_id"]: True,
    }
    for client in (anon, bogus, trusted):
        await client.ws.close()


async def test_terminal_protocol_needs_this_gateway_instance(harness: Harness) -> None:
    import aiohttp

    instance = harness.channel.tokens.instance_id
    token = (await harness.bootstrap())["token"]
    with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
        await harness.http.ws_connect(f"{harness.base}/?token={token}&terminal_protocol=1&terminal_instance=other")
    assert exc.value.status == 409
    client = await harness.connect(f"terminal_protocol=1&terminal_instance={instance}")
    assert client.ready["terminal"]["gatewayId"] == instance
    plain = await harness.connect()
    assert "terminal" not in plain.ready
    await client.ws.close()
    await plain.ws.close()


async def test_client_ids_are_truncated_or_generated(harness: Harness) -> None:
    long = await harness.connect("client_id=" + "x" * 200)
    assert long.ready["client_id"] == "x" * 128
    a, b = await harness.connect(), await harness.connect()
    assert a.ready["client_id"].startswith("anon-") and len(a.ready["client_id"]) == len("anon-") + 12
    assert a.ready["client_id"] != b.ready["client_id"]
    for client in (long, a, b):
        await client.ws.close()


@pytest.mark.ws_config({"maxMessageBytes": 1024, "pingIntervalS": 7})
async def test_socket_limits_frames_and_pings_on_the_configured_interval(harness: Harness) -> None:
    import aiohttp

    client = await harness.connect()
    await asyncio.sleep(0.05)
    [conn] = harness.channel._connections
    assert conn.ws._heartbeat == 7.0
    await client.send("x" * 2000)
    msg = await asyncio.wait_for(client.ws.receive(), 3)
    assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING)
    assert harness.inbound() == []


async def test_binary_frames_are_read_as_utf8_and_bad_bytes_are_skipped(harness: Harness) -> None:
    client = await harness.connect()
    await client.ws.send_bytes(b"\xff\xfe")
    await client.ws.send_bytes(json.dumps({"type": "bogus"}).encode())
    assert await client.recv() == {"event": "error", "detail": "unknown type: 'bogus'"}
    await client.ws.close()


def test_half_configured_tls_is_rejected(tmp_path) -> None:
    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.channels.websocket import WebSocketChannel
    from nanoclaw.config.schema import WebSocketConfig
    from nanoclaw.providers import FakeProvider
    from nanoclaw.session import SessionManager

    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    agent = AgentLoop(FakeProvider([]), ws_dir, sessions=SessionManager(ws_dir, sessions_root=tmp_path / "s"))
    for cert, key in (("cert.pem", ""), ("", "key.pem")):
        cfg = WebSocketConfig(enabled=True, port=0, ssl_certfile=cert, ssl_keyfile=key)
        channel = WebSocketChannel(cfg, agent.bus, agent=agent)
        with pytest.raises(ValueError, match="both be set"):
            channel._ssl_context()
    channel = WebSocketChannel(WebSocketConfig(enabled=True, port=0), agent.bus, agent=agent)
    assert channel._ssl_context() is None


async def test_stopped_channel_ignores_lifecycle_events(harness: Harness) -> None:
    from nanoclaw.bus.runtime_events import RuntimeEventContext, TurnRunStatusChanged

    ch = harness.channel
    await ch.stop()
    await asyncio.wait_for(harness.task, 5)
    ctx = RuntimeEventContext(channel="websocket", chat_id="c", session_key="websocket:c", metadata={})
    await ch.bus.publish(TurnRunStatusChanged(context=ctx, status="running", started_at=5.0))
    assert ch.bus.outbound.empty()
    assert ch.turns.wall_started_at("c") is None


async def test_stop_drops_connections_tokens_requests_and_background_work(harness: Harness) -> None:
    from aiohttp import web

    ch = harness.channel
    started = asyncio.Event()

    async def hang(payload: dict[str, Any]) -> web.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    ch.context.mutations["sidebar.update"] = hang
    client = await harness.connect()
    await harness.bootstrap()
    await client.send({"type": "webui_request", "request_id": "r1", "action": "sidebar.update", "payload": {}})
    await asyncio.wait_for(started.wait(), 3)
    [operation] = ch.commands.request_operations.values()
    background = asyncio.create_task(asyncio.Event().wait())
    ch._background.add(background)
    try:
        await asyncio.wait_for(ch.stop(), 3)
    finally:
        cancelled_by_stop = background.cancelled()
        background.cancel()
        ch._background.discard(background)
    assert cancelled_by_stop
    assert not ch._connections
    assert operation.task.cancelled()
    assert not ch.tokens.issued_tokens and not ch.tokens.api_tokens


async def test_a_closed_socket_is_forgotten_everywhere(harness: Harness) -> None:
    ch = harness.channel
    client = await harness.connect()
    await client.send({"type": "attach", "chat_id": "c"})
    await client.recv()
    await client.request(request_id="r1", action="nope.nope", payload={})
    assert ch.commands.request_locks
    await client.ws.close()
    await _wait_for(lambda: not ch._connections)
    await asyncio.sleep(0.05)
    assert ch._subs == {} and ch._conn_chats == {} and ch._conn_default == {}
    assert ch.commands.request_locks == {}


@pytest.mark.ws_config({"token": "static"})
async def test_web_ui_events_skip_untrusted_sockets(harness: Harness) -> None:
    from aiohttp import web

    from nanoclaw.webui.inbound_commands import mutation_json

    async def sidebar_update(payload: dict[str, Any]) -> web.Response:
        return mutation_json(payload["state"])

    harness.channel.context.mutations["sidebar.update"] = sidebar_update
    trusted = await harness.connect(headers={"X-Nanobot-Auth": "static"})
    plain = await harness.connect("?token=static", trusted=False)
    await trusted.send({"type": "set_sidebar_state", "state": {"pinned_keys": ["k"]}})
    assert (await trusted.recv())["event"] == "sidebar_state_updated"
    assert await plain.recv_all() == []
    await harness.channel.broadcast("c", {"event": "probe"}, audience="webui")
    assert await trusted.recv() == {"event": "probe"}
    assert await plain.recv_all() == []
    await trusted.ws.close()
    await plain.ws.close()


async def test_send_session_updated_goes_to_every_socket_with_its_scope(harness: Harness) -> None:
    a, b = await harness.connect(), await harness.connect()
    await harness.channel.send_session_updated("x", scope="metadata")
    expected = {"event": "session_updated", "chat_id": "x", "scope": "metadata"}
    assert await a.recv() == expected and await b.recv() == expected
    await harness.channel.send_session_updated("x")
    assert await a.recv() == {"event": "session_updated", "chat_id": "x"}
    await a.ws.close()
    await b.ws.close()


class _FakeWS:
    def __init__(self, *, block: bool = True) -> None:
        self.block = block
        self.closed = False
        self.sent: list[str] = []
        self.close_codes: list[int] = []

    async def send_str(self, raw: str) -> None:
        if self.block:
            await asyncio.Event().wait()
        self.sent.append(raw)

    async def close(self, *, code: int, message: bytes) -> None:
        self.close_codes.append(code)
        self.closed = True


def _fake_connection(harness: Harness, *, block: bool = True, max_frames: int = 256) -> Any:
    from nanoclaw.channels.websocket.runtime import WebUIConnection

    conn = WebUIConnection(
        ws=_FakeWS(block=block), client_id="fake", remote=None,  # type: ignore[arg-type]
        queue=asyncio.Queue(maxsize=max_frames),
    )
    harness.channel._connections.add(conn)
    return conn


async def test_a_full_frame_queue_retires_the_socket(harness: Harness) -> None:
    conn = _fake_connection(harness, max_frames=2)
    for raw in ("a", "b", "c", "d"):
        harness.channel._enqueue(conn, raw)
    await asyncio.sleep(0.05)
    assert conn.ws.close_codes == [1013]


async def test_the_byte_budget_retires_a_slow_socket(harness: Harness, monkeypatch) -> None:
    from nanoclaw.channels.websocket import runtime

    monkeypatch.setattr(runtime, "_OUTBOUND_QUEUE_MAX_BYTES", 10)
    conn = _fake_connection(harness)
    harness.channel._enqueue(conn, "12345678")
    assert conn.buffered_bytes == 8 and harness.channel.has_pending_delivery
    harness.channel._enqueue(conn, "abcd")
    await asyncio.sleep(0.05)
    assert conn.ws.close_codes == [1013]


async def test_drained_frames_release_their_bytes(harness: Harness) -> None:
    conn = _fake_connection(harness, block=False)
    harness.channel._enqueue(conn, "one")
    harness.channel._enqueue(conn, "two")
    await asyncio.sleep(0.05)
    assert conn.ws.sent == ["one", "two"]
    assert conn.buffered_bytes == 0 and not harness.channel.has_pending_delivery


async def test_a_stuck_send_times_out_and_retires_once(harness: Harness, monkeypatch) -> None:
    from nanoclaw.channels.websocket import runtime

    monkeypatch.setattr(runtime, "_OUTBOUND_SEND_TIMEOUT_S", 0.05)
    conn = _fake_connection(harness)
    harness.channel._enqueue(conn, "x")
    await asyncio.sleep(0.3)
    assert conn.ws.close_codes == [1013]
    other = _fake_connection(harness)
    harness.channel._retire(other, code=1000, reason="")
    harness.channel._retire(other, code=1013, reason="again")
    await asyncio.sleep(0.05)
    assert other.ws.close_codes == [1000]


async def test_closed_connections_get_no_frames_or_chats(harness: Harness) -> None:
    conn = _fake_connection(harness)
    conn.closing = True
    harness.channel._enqueue(conn, "x")
    assert conn.queue.empty() and conn.buffered_bytes == 0 and conn.writer is None
    harness.channel.attach(conn, "c")
    assert harness.channel.subscribers("c") == ()


async def test_attach_reports_the_cached_session_facts(harness: Harness) -> None:
    from nanoclaw.providers.base import LLMUsage

    usage = LLMUsage(input_tokens=10, output_tokens=5)
    session = harness.channel.sessions.get_or_create("websocket:c")
    session.metadata.update({
        "_nanobot_model_preset": "fast",
        "webui_recovery": {"status": "resuming", "recovery_id": "r1"},
        "_last_usage": usage.to_dict(),
    })
    client = await harness.connect()
    await client.send({"type": "attach", "chat_id": "c"})
    attached = await client.recv()
    assert attached["model_preset"] == "fast"
    assert attached["recovery_state"] == {"status": "resuming", "recovery_id": "r1"}
    assert attached["usage"] == usage.to_turn_dict()
    await client.ws.close()


async def test_channel_flags_reach_the_projector_at_send_time(harness: Harness) -> None:
    from nanoclaw.bus.outbound_events import ProgressEvent
    from test_websocket_channel import out

    ch = harness.channel
    client = await harness.connect()
    await client.send({"type": "attach", "chat_id": "c"})
    await client.recv()
    await ch.send(out("c", ProgressEvent(), "working"))
    await ch.send(out("c", ProgressEvent(reasoning_delta=True), "hmm"))
    assert [f["event"] for f in await client.recv_all()] == ["message", "reasoning_delta"]
    ch.send_progress = False
    ch.show_reasoning = False
    await ch.send(out("c", ProgressEvent(), "working"))
    await ch.send(out("c", ProgressEvent(reasoning_delta=True), "hmm"))
    assert await client.recv_all() == []
    await client.ws.close()


def _recording_append(monkeypatch, ch: Any, fail: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    async def append(chat_id: str, event: dict[str, Any], *, metadata: dict[str, Any]) -> bool:
        records.append(event)
        return not fail(event)

    monkeypatch.setattr(ch.recorder, "append", append)
    return records


async def test_turn_end_after_a_lost_answer_is_marked_incomplete(harness: Harness, monkeypatch) -> None:
    from nanoclaw.bus.outbound_events import TurnEndEvent
    from nanoclaw.webui.transcript import WEBUI_TRANSCRIPT_INCOMPLETE_KEY
    from test_websocket_channel import out

    ch = harness.channel
    ch.turns.mark_running("c", "o1", "t1", 1.0)
    records = _recording_append(monkeypatch, ch, lambda event: event.get("event") == "message")
    meta = {"webui": True, "webui_turn_id": "t1", "_websocket_turn_owner": "o1"}
    await ch.send(out("c", content="answer", **meta))
    await ch.send(out("c", TurnEndEvent(goal_state={"active": False}), **meta))
    assert records[-1]["event"] == "turn_end" and records[-1][WEBUI_TRANSCRIPT_INCOMPLETE_KEY] is True


async def test_non_webui_turn_end_is_not_marked_incomplete(harness: Harness, monkeypatch) -> None:
    from nanoclaw.bus.outbound_events import TurnEndEvent
    from nanoclaw.webui.transcript import WEBUI_TRANSCRIPT_INCOMPLETE_KEY
    from test_websocket_channel import out

    ch = harness.channel
    ch.turns.mark_running("c", "o1", "t1", 1.0)
    ch.turns.mark_transcript_persistence_failed("c", "o1")
    records = _recording_append(monkeypatch, ch, lambda event: False)
    await ch.send(out("c", TurnEndEvent(goal_state={"active": False}), _websocket_turn_owner="o1"))
    assert records[-1]["event"] == "turn_end" and WEBUI_TRANSCRIPT_INCOMPLETE_KEY not in records[-1]


async def test_only_lost_webui_answers_mark_the_turn(harness: Harness, monkeypatch) -> None:
    from nanoclaw.bus.outbound_events import ProgressEvent
    from test_websocket_channel import out

    ch = harness.channel
    ch.turns.mark_running("c", "o1", "t1", 1.0)
    records = _recording_append(monkeypatch, ch, lambda event: True)
    meta = {"webui": True, "webui_turn_id": "t1", "_websocket_turn_owner": "o1"}
    await ch.send(out("c", ProgressEvent(tool_hint=True), "read_file(a)", **meta))
    await ch.send(out("c", content="answer", _websocket_turn_owner="o1"))
    assert len(records) == 2
    assert not ch.turns.transcript_persistence_failed("c", "o1")


# -- turn lifecycle subscriptions -------------------------------------------------------


def _ctx(channel: str = "websocket", chat_id: str = "c", **metadata: Any) -> Any:
    from nanoclaw.bus.runtime_events import RuntimeEventContext

    return RuntimeEventContext(channel=channel, chat_id=chat_id, session_key=f"{channel}:{chat_id}", metadata=metadata)


async def test_session_turn_started_marks_only_websocket_sessions(harness: Harness) -> None:
    from nanoclaw.bus.runtime_events import SessionTurnStarted

    ch = harness.channel
    scope = {"project_path": "/p", "access_mode": "full"}
    web = ch.sessions.get_or_create("websocket:c")
    other = ch.sessions.get_or_create("cli:c")
    plain = ch.sessions.get_or_create("websocket:plain")
    await ch.bus.publish(SessionTurnStarted(context=_ctx(webui=True, workspace_scope=scope)))
    await ch.bus.publish(SessionTurnStarted(context=_ctx("cli", webui=True, workspace_scope=scope)))
    await ch.bus.publish(SessionTurnStarted(context=_ctx(chat_id="plain", webui=1, workspace_scope=scope)))
    assert web.metadata["webui"] is True and web.metadata["workspace_scope"] == scope
    assert "webui" not in other.metadata and "workspace_scope" not in other.metadata
    assert "webui" not in plain.metadata and plain.metadata["workspace_scope"] == scope


async def test_run_status_from_other_channels_registers_nothing(harness: Harness) -> None:
    from nanoclaw.bus.runtime_events import TurnRunStatusChanged

    ch = harness.channel
    await ch.bus.publish(TurnRunStatusChanged(context=_ctx("cli", "direct"), status="running", started_at=5.0))
    assert ch.turns.wall_started_at("direct") is None and ch._minted_owners == {}


async def test_running_without_a_start_time_uses_the_same_clock_for_both(harness: Harness, monkeypatch) -> None:
    import itertools
    import time

    from nanoclaw.bus.runtime_events import TurnRunStatusChanged

    ch = harness.channel
    clock = itertools.count(1_700_000_000, 100)
    monkeypatch.setattr(time, "time", lambda: float(next(clock)))
    await ch.bus.publish(TurnRunStatusChanged(context=_ctx(), status="running", started_at=0))
    msg = ch.bus.outbound.get_nowait()
    started = ch.turns.wall_started_at("c")
    assert started is not None and started >= 1_700_000_000
    assert msg.event.started_at == started


async def test_idle_without_a_running_turn_mints_no_owner(harness: Harness) -> None:
    from nanoclaw.bus.runtime_events import TurnRunStatusChanged

    ch = harness.channel
    await ch.bus.publish(TurnRunStatusChanged(context=_ctx(), status="idle"))
    msg = ch.bus.outbound.get_nowait()
    assert "_websocket_turn_owner" not in msg.metadata


async def test_each_run_without_an_owner_gets_a_fresh_one(harness: Harness) -> None:
    from nanoclaw.bus.runtime_events import TurnRunStatusChanged

    ch = harness.channel
    owners = []
    for status in ("running", "idle", "running"):
        await ch.bus.publish(TurnRunStatusChanged(context=_ctx(), status=status, started_at=5.0))
        owners.append(ch.bus.outbound.get_nowait().metadata["_websocket_turn_owner"])
    assert owners[0] == owners[1] and owners[2] != owners[0]


async def test_turn_completed_carries_the_minted_owner(harness: Harness) -> None:
    from nanoclaw.bus.runtime_events import TurnCompleted, TurnRunStatusChanged

    ch = harness.channel
    await ch.bus.publish(TurnRunStatusChanged(context=_ctx(), status="running", started_at=5.0))
    owner = ch.bus.outbound.get_nowait().metadata["_websocket_turn_owner"]
    await ch.bus.publish(TurnCompleted(context=_ctx()))
    assert ch.bus.outbound.get_nowait().metadata["_websocket_turn_owner"] == owner


async def test_a_generated_title_announces_a_metadata_update(harness: Harness) -> None:
    from nanoclaw.bus.outbound_events import SessionUpdatedEvent
    from nanoclaw.bus.runtime_events import TurnCompleted
    from nanoclaw.providers import FakeProvider
    from nanoclaw.utils.llm_runtime import LLMRuntime

    ch = harness.channel
    session = ch.sessions.get_or_create("websocket:c")
    session.metadata["webui"] = True
    session.add_message("user", "How do I bake bread?")
    session.add_message("assistant", "Mix flour and water.")
    ch.sessions.save(session)
    runtime = LLMRuntime(
        provider=FakeProvider(["Baking bread"]), model="m", max_tokens=100, temperature=0.0, context_window_tokens=1000
    )
    await ch.bus.publish(TurnCompleted(context=_ctx(webui=True), runtime=runtime))
    await ch.wait_for_background()
    events = []
    while not ch.bus.outbound.empty():
        events.append(ch.bus.outbound.get_nowait().event)
    assert session.metadata["title"] == "Baking bread"
    assert [e.scope for e in events if isinstance(e, SessionUpdatedEvent)] == ["metadata"]


# -- message envelopes ---------------------------------------------------------------------


async def test_new_chat_subscribes_the_socket(harness: Harness) -> None:
    from test_websocket_channel import out

    client = await harness.connect()
    await client.send({"type": "new_chat"})
    attached = await client.recv()
    await client.recv()
    await harness.channel.send(out(attached["chat_id"], content="hello"))
    frames = await client.recv_all()
    assert [f["event"] for f in frames] == ["message"] and frames[0]["text"] == "hello"
    await client.ws.close()


async def test_user_message_echo_reports_the_running_turn_start(harness: Harness) -> None:
    a, b = await harness.connect(), await harness.connect()
    await b.send({"type": "attach", "chat_id": "c"})
    await b.recv()
    await a.send({"type": "message", "chat_id": "c", "content": "hello", "turn_id": "t1", "webui": True})
    await a.recv()
    echo = await b.recv()
    assert echo["started_at"] == harness.channel.turns.wall_started_at("c")
    harness.channel.turns.clear("c")
    harness.channel.turns.mark_running("c", "o", None, 5.0)
    await a.send({"type": "message", "chat_id": "c", "content": "again", "turn_id": "t2", "webui": True})
    echo = [f for f in await b.recv_all() if f["event"] == "user_message"][0]
    assert "active_turn_id" not in echo and "started_at" not in echo
    await a.ws.close()
    await b.ws.close()


async def test_an_empty_turn_id_counts_as_none(harness: Harness) -> None:
    a, b = await harness.connect(), await harness.connect()
    await b.send({"type": "attach", "chat_id": "c"})
    await b.recv()
    await a.send({"type": "message", "chat_id": "c", "content": "hello", "turn_id": "", "webui": True})
    echo = await b.recv()
    assert echo["event"] == "user_message" and "turn_id" not in echo and "active_turn_id" not in echo
    assert await a.recv_all() == []
    await a.ws.close()
    await b.ws.close()


async def test_concurrent_retries_of_one_turn_dispatch_once(harness: Harness, monkeypatch) -> None:
    ch = harness.channel
    original = ch.recorder.append_user_message

    async def slow(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.05)
        return await original(*args, **kwargs)

    monkeypatch.setattr(ch.recorder, "append_user_message", slow)
    a, b = await harness.connect(), await harness.connect()
    envelope = {"type": "message", "chat_id": "c", "content": "hello", "turn_id": "t1", "webui": True}
    await a.send(envelope)
    await b.send(envelope)
    frames_a, frames_b = await a.recv_all(0.5), await b.recv_all(0.5)
    assert len(harness.inbound()) == 1
    assert any(f["event"] == "message_accepted" for f in frames_a)
    assert any(f["event"] == "message_accepted" for f in frames_b)
    assert ch.commands._message_locks == {}
    await a.ws.close()
    await b.ws.close()


@pytest.mark.ws_config({"allowFrom": ["alice"], "websocketRequiresToken": False})
async def test_a_sender_removed_from_the_allow_list_is_refused_before_recording(harness: Harness) -> None:
    client = await harness.connect("?client_id=alice", trusted=False)
    harness.channel.config.allow_from = ["bob"]
    await client.send({"type": "message", "chat_id": "c", "content": "hi", "turn_id": "t1", "webui": True})
    assert await client.recv() == {"event": "error", "detail": "access_denied", "chat_id": "c", "turn_id": "t1"}
    assert harness.channel.history.read_facts("websocket:c")[0] == []
    await client.ws.close()


async def test_accepted_turns_expire_after_the_dedupe_window(harness: Harness, monkeypatch) -> None:
    from nanoclaw.webui import inbound_commands

    monkeypatch.setattr(inbound_commands, "_ACCEPTED_MESSAGE_TTL_S", 0.05)
    client = await harness.connect()
    envelope = {"type": "message", "chat_id": "c", "content": "hello", "turn_id": "t1", "webui": True}
    await client.send(envelope)
    await client.recv_all()
    await asyncio.sleep(0.1)
    await client.send(envelope)
    await client.recv_all()
    assert len(harness.inbound()) == 2
    await client.ws.close()


async def test_dedupe_window_evicts_the_oldest_turn(harness: Harness, monkeypatch) -> None:
    from nanoclaw.webui import inbound_commands

    monkeypatch.setattr(inbound_commands, "_ACCEPTED_MESSAGES_MAX", 2)
    client = await harness.connect()
    for turn in ("t1", "t2", "t3", "t3", "t1"):
        await client.send({"type": "message", "chat_id": "c", "content": turn, "turn_id": turn})
        await asyncio.sleep(0.05)
    assert [m.content for m in harness.inbound()] == ["t1", "t2", "t3", "t1"]
    assert len(harness.channel.commands._accepted_messages) == 2


async def test_a_refused_dispatch_is_not_remembered_and_frees_its_turn(harness: Harness, monkeypatch) -> None:
    ch = harness.channel
    original = ch.dispatch_inbound
    answers = [False]

    async def dispatch(**kwargs: Any) -> bool:
        if answers:
            return answers.pop()
        return await original(**kwargs)

    monkeypatch.setattr(ch, "dispatch_inbound", dispatch)
    client = await harness.connect()
    envelope = {"type": "message", "chat_id": "c", "content": "hello", "turn_id": "t1", "webui": True}
    await client.send(envelope)
    assert await client.recv() == {"event": "error", "detail": "access_denied", "chat_id": "c", "turn_id": "t1"}
    assert ch.turns.wall_started_at("c") is None
    await client.send(envelope)
    accepted = await client.recv()
    assert accepted["event"] == "message_accepted" and accepted["starts_turn"] is True
    assert len(harness.inbound()) == 1
    await client.ws.close()


async def test_envelope_messages_carry_the_peer(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "message", "chat_id": "c", "content": "x"})
    await asyncio.sleep(0.05)
    [msg] = harness.inbound()
    assert isinstance(msg.metadata["remote"], tuple) and msg.metadata["remote"][0] == "127.0.0.1"
    await client.ws.close()


async def test_only_a_true_webui_flag_records_and_acknowledges(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "message", "chat_id": "c", "content": "x", "turn_id": "t1", "webui": 1})
    await client.send({"type": "message", "chat_id": "d", "content": "y", "turn_id": "t2"})
    assert await client.recv_all() == []
    msgs = harness.inbound()
    assert len(msgs) == 2 and all("webui" not in m.metadata for m in msgs)
    assert harness.channel.history.read_facts("websocket:c")[0] == []


# -- webui/gateway_http.py -------------------------------------------------------------------


def _gateway_http(model: Any = None, **config: Any) -> Any:
    from nanoclaw.config.schema import WebSocketConfig
    from nanoclaw.webui.gateway_http import WebUIGatewayHTTP
    from nanoclaw.webui.gateway_tokens import GatewayTokenStore

    return WebUIGatewayHTTP(
        WebSocketConfig(**config), GatewayTokenStore(), static_dist_path=None, runtime_model_name=model
    )


def test_bootstrap_model_name_is_stripped_and_blank_is_empty() -> None:
    assert _gateway_http(lambda: "  m-2 ")._model_name() == "m-2"
    assert _gateway_http(lambda: "   ")._model_name() == ""


def test_bootstrap_ws_url_scheme_and_path() -> None:
    from types import SimpleNamespace

    def url(headers: dict[str, str], **config: Any) -> str:
        return _gateway_http(**config).bootstrap_ws_url(SimpleNamespace(headers=headers))

    assert url({"Host": "h:1"}, path="/ws") == "ws://h:1/ws"
    assert url({"Host": "h:1", "X-Forwarded-Proto": "https, http"}) == "wss://h:1/"
    assert url({"Host": "h:1"}, ssl_certfile="cert.pem") == "wss://h:1/"


@pytest.mark.ws_config({"path": "/ws"})
async def test_bootstrap_reports_the_socket_path_and_capabilities(harness: Harness) -> None:
    body = await harness.bootstrap()
    assert body["ws_path"] == "/ws" and body["ws_url"].endswith("/ws")
    assert body["runtime_capabilities"] == {
        "can_restart_engine": False, "can_open_logs": False, "can_export_diagnostics": False,
    }


def test_static_paths_with_dot_segments_are_refused(tmp_path) -> None:
    from test_websocket_channel import make_dist

    http = _gateway_http()
    http.static_dist_path = make_dist(tmp_path).resolve()
    assert http.serve_static("/assets/../index.html").status == 403
    assert http.serve_static("/../secret.txt").status == 403
    assert http.serve_static("/index.html").status == 200


def test_static_symlinks_out_of_the_build_are_refused(tmp_path) -> None:
    from test_websocket_channel import make_dist

    dist = make_dist(tmp_path)
    (dist / "link.txt").symlink_to(tmp_path / "secret.txt")
    http = _gateway_http()
    http.static_dist_path = dist.resolve()
    assert http.serve_static("/link.txt").status == 403


def test_svg_is_served_compressed_without_charset(tmp_path) -> None:
    import gzip

    from test_websocket_channel import make_dist

    dist = make_dist(tmp_path)
    (dist / "icon.svg").write_text("<svg/>")
    (dist / "icon.svg.gz").write_bytes(gzip.compress(b"<svg/>"))
    http = _gateway_http()
    http.static_dist_path = dist.resolve()
    resp = http.serve_static("/icon.svg", accept_encoding="gzip")
    assert resp.headers["Content-Type"] == "image/svg+xml"
    assert resp.headers["Content-Encoding"] == "gzip" and resp.headers["Vary"] == "Accept-Encoding"


async def test_static_gzip_reads_repeated_accept_encoding_headers(harness: Harness) -> None:
    resp = await harness.http.get(
        harness.base + "/assets/app-1a2b.js", headers=[("Accept-Encoding", "br"), ("Accept-Encoding", "gzip")]
    )
    await resp.read()
    assert resp.headers.get("Content-Encoding") == "gzip"


def test_configured_dist_wins_over_the_environment(tmp_path) -> None:
    from nanoclaw.webui.gateway_http import resolve_webui_dist

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert resolve_webui_dist(str(a), {"NANOCLAW_WEBUI_DIST": str(b)}) == a.resolve()


def test_http_helpers_strip_slashes_and_whitespace() -> None:
    from nanoclaw.webui.http_utils import header, strip_trailing_slash

    assert strip_trailing_slash("/webui/bootstrap/") == "/webui/bootstrap"
    assert strip_trailing_slash("/a//") == "/a"
    assert strip_trailing_slash("/") == "/" and strip_trailing_slash("") == "/"
    assert header({"Host": "  box.lan  "}, "Host") == "box.lan"


@pytest.mark.ws_config({"token": "static"})
async def test_a_static_token_is_also_the_bootstrap_secret(harness: Harness) -> None:
    assert (await harness.get("/webui/bootstrap")).status == 401
    assert (await harness.get("/webui/bootstrap", {"X-Nanobot-Auth": "static"})).status == 200


@pytest.mark.ws_config({"tokenTtlS": 60, "maxMessageBytes": 4096, "tokenIssuePath": "/auth/token"})
async def test_bootstrap_and_issue_use_the_configured_ttl_and_frame_limit(harness: Harness) -> None:
    import time

    tokens = harness.channel.tokens
    body = await harness.bootstrap()
    assert body["expires_in"] == 60
    assert body["limits"]["transport"]["max_frame_bytes"] == 4096
    now = time.monotonic()
    assert 55 < tokens.issued_tokens[body["token"]] - now <= 60
    assert 55 < tokens.api_tokens[body["api_token"]] - now <= 60
    issued = await (await harness.get("/auth/token")).json()
    assert issued["expires_in"] == 60
    assert 55 < tokens.issued_tokens[issued["token"]] - time.monotonic() <= 60


@pytest.mark.ws_config({"tokenIssuePath": "/auth/token"})
async def test_token_routes_refuse_when_too_many_tokens_are_outstanding(harness: Harness) -> None:
    tokens = harness.channel.tokens
    tokens.max_tokens = 1
    tokens.issue_token(300)
    resp = await harness.get("/auth/token")
    assert resp.status == 429 and await resp.json() == {"error": "too many outstanding tokens"}
    assert (await harness.get("/webui/bootstrap")).status == 429
    tokens.issued_tokens.clear()
    tokens.issue_api_token(300)
    assert (await harness.get("/auth/token")).status == 200
    tokens.issued_tokens.clear()
    assert (await harness.get("/webui/bootstrap")).status == 429


async def test_stop_closes_sockets_before_it_returns(harness: Harness) -> None:
    client = await harness.connect()
    await asyncio.sleep(0.05)
    assert harness.channel._connections
    await harness.channel.stop()
    assert not harness.channel._connections
    await client.ws.close()


# -- webui/http_utils.py ---------------------------------------------------------------------


def test_http_utils_host_and_encoding_rules() -> None:
    from multidict import CIMultiDict

    from nanoclaw.webui.http_utils import accepts_gzip, combined_list_header, host_for_url, safe_host_header

    assert combined_list_header(CIMultiDict([("Accept-Encoding", "br"), ("Accept-Encoding", "gzip")]),
                                "Accept-Encoding") == "br, gzip"
    assert safe_host_header("[::1]:8765") == "[::1]:8765"
    assert safe_host_header("evil.com/path") == "" and safe_host_header("a b") == ""
    assert host_for_url("::", 8765) == "127.0.0.1:8765"
    assert host_for_url("0.0.0.0", 1) == "127.0.0.1:1"
    assert host_for_url("::1", 1) == "[::1]:1"
    assert not accepts_gzip("*;q=0")
    assert accepts_gzip("GZIP")
    assert not accepts_gzip("gzip;q=abc")


def test_http_utils_loopback_rules() -> None:
    from nanoclaw.webui.http_utils import is_local_browser_request, is_localhost, is_loopback_host

    assert is_localhost("::ffff:127.0.0.1")
    assert is_loopback_host("[::1]:8765")
    assert is_loopback_host("localhost.")
    assert is_loopback_host("::ffff:127.0.0.1")
    local = {"Host": "127.0.0.1:8765"}
    assert is_local_browser_request("127.0.0.1", {**local, "Forwarded": 'for="127.0.0.1";host=localhost'})
    for extra in (
        {"X-Real-IP": "10.0.0.1"},
        {"X-Forwarded-Host": "example.com"},
        {"Forwarded": "for=10.0.0.1"},
        {"Forwarded": "host=example.com"},
        {"Forwarded": "For=10.0.0.1"},
    ):
        assert not is_local_browser_request("127.0.0.1", {**local, **extra}), extra


def test_bearer_scheme_is_case_insensitive() -> None:
    from nanoclaw.webui.http_utils import bearer_token, issue_route_secret_matches

    assert bearer_token({"Authorization": "bearer tok"}) == "tok"
    assert issue_route_secret_matches({"Authorization": "bearer s"}, "s")
    assert not issue_route_secret_matches({"Authorization": "bearer t", "X-Nanobot-Auth": "s"}, "s")


# -- webui/gateway_tokens.py ---------------------------------------------------------------------


async def test_token_store_limits_expiry_and_query_tokens() -> None:
    from types import SimpleNamespace

    from nanoclaw.webui.gateway_tokens import GatewayTokenStore

    assert GatewayTokenStore().max_tokens == 10_000
    store = GatewayTokenStore(max_tokens=1)
    assert store.can_issue()
    store.issue_token(300)
    assert not store.can_issue()
    store.issued_tokens.clear()
    api = store.issue_api_token(300)
    assert store.can_issue() and not store.can_issue(include_api_token=True)
    assert store.check_api_token(SimpleNamespace(headers={}, query={"token": api}))
    store.clear()
    short = store.issue_token(0.01)
    short_api = store.issue_api_token(0.01)
    await asyncio.sleep(0.03)
    assert store.can_issue(include_api_token=True)
    assert store.take_issued_token_audience(short) is None
    assert not store.check_api_token(SimpleNamespace(headers={"Authorization": f"Bearer {short_api}"}, query={}))


# -- webui/ingress_policy.py, client_contract.py ---------------------------------------------------


def test_ingress_limits_count_bytes_and_report_nanobot_defaults() -> None:
    from nanoclaw.webui.ingress_policy import DEFAULT_WEBUI_INGRESS_POLICY as policy

    assert policy.validate_text("é" * 40_000) == "text_too_large"
    assert policy.validate_text("x" * 65_536) is None
    assert policy.validate_text("x" * 65_537) == "text_too_large"
    assert policy.bootstrap_limits(max_frame_bytes=9) == {
        "transport": {"max_frame_bytes": 9, "envelope_reserve_bytes": 65_536},
        "message": {"max_text_bytes": 65_536},
        "attachments": {"max_count": 4, "max_file_bytes": 6 * 1024 * 1024, "max_total_bytes": 24 * 1024 * 1024},
    }


def test_webui_contract_advertises_protocol_one() -> None:
    from nanoclaw import __version__
    from nanoclaw.webui.client_contract import webui_contract

    assert webui_contract() == {
        "version": __version__, "min_protocol": 1, "max_protocol": 1, "capabilities": ["webui.core.v1"],
    }


# -- config/schema.py --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    {"token_ttl_s": 29}, {"token_ttl_s": 86_401},
    {"max_message_bytes": 1023}, {"max_message_bytes": 41_943_041},
    {"ping_interval_s": 4.9}, {"ping_interval_s": 301},
    {"token_issue_path": "token"},
    {"public_ws_url": "wss://u:p@h/"}, {"public_ws_url": "wss://h/?a=1"}, {"public_ws_url": "wss://h/#f"},
    {"public_ws_url": "wss:///"},
    {"host": "::"},
])
def test_websocket_config_rejects(bad: dict[str, Any]) -> None:
    from pydantic import ValidationError

    from nanoclaw.config.schema import WebSocketConfig

    with pytest.raises(ValidationError):
        WebSocketConfig(**bad)


def test_websocket_config_accepts_edges_and_cleans_paths() -> None:
    from pydantic import ValidationError

    from nanoclaw.config.schema import ChannelsConfig, WebSocketConfig

    assert WebSocketConfig().max_message_bytes == 37_748_736
    assert WebSocketConfig(token_ttl_s=30).token_ttl_s == 30
    assert WebSocketConfig(token_issue_path="  ").token_issue_path == ""
    assert WebSocketConfig(token_issue_path=" /auth/token/ ").token_issue_path == "/auth/token"
    assert WebSocketConfig(host="0.0.0.0", token="t").host == "0.0.0.0"
    assert WebSocketConfig(host="::", token_issue_secret="s").host == "::"
    for retries in (-1, 11):
        with pytest.raises(ValidationError):
            ChannelsConfig(send_max_retries=retries)
    assert ChannelsConfig(send_max_retries=0).send_max_retries == 0


# -- cli/gateway.py ------------------------------------------------------------------------------


async def test_gateway_shutdown_stops_the_channels_first() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from nanoclaw.cli.gateway import _close_gateway_runtime

    channels = MagicMock(stop_all=AsyncMock())
    await _close_gateway_runtime(
        MagicMock(aclose=AsyncMock()), MagicMock(aclose=AsyncMock()), [], None, channels=channels
    )
    channels.stop_all.assert_awaited_once()


def test_run_gateway_refuses_a_busy_web_ui_port(tmp_path) -> None:
    import socket

    from rich.console import Console

    from nanoclaw.cli import gateway as gw

    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        health_port = free.getsockname()[1]
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        config = Config.model_validate({"channels": {"websocket": {"enabled": True, "port": busy.getsockname()[1]}}})
        console = Console(record=True, width=200)
        built: list[int] = []
        status = gw.run_gateway(config, lambda: built.append(1), port=health_port, console=console)
    assert status == 1 and built == []
    assert "web UI port" in console.export_text()


def test_banner_says_when_there_is_no_frontend_build(tmp_path, monkeypatch) -> None:
    from rich.console import Console

    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.cli.gateway import Gateway
    from nanoclaw.providers import FakeProvider
    from nanoclaw.session import SessionManager

    monkeypatch.delenv("NANOCLAW_WEBUI_DIST", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    agent = AgentLoop(FakeProvider([]), ws, sessions=SessionManager(ws, sessions_root=tmp_path / "s"))
    config = Config.model_validate({"channels": {"websocket": {"enabled": True, "port": 8799}}})
    console = Console(record=True, width=200)
    Gateway(config, agent, object(), console=console, health_server_enabled=False).print_banner()  # type: ignore[arg-type]
    text = console.export_text()
    assert "Web UI server: http://127.0.0.1:8799 (no frontend build" in text


# -- webui/rest_api.py ---------------------------------------------------------------------------

REST_TOKEN = "nbwt_test"
REST_AUTH = {"Authorization": f"Bearer {REST_TOKEN}"}


class _RestTurns:
    def __init__(self) -> None:
        self.started: dict[str, float] = {}
        self.ids: dict[str, str] = {}

    def wall_started_at(self, chat_id: str) -> float | None:
        return self.started.get(chat_id)

    def turn_id(self, chat_id: str) -> str | None:
        return self.ids.get(chat_id)


@pytest.fixture
def rest_ctx(history: Any, tmp_path: Any) -> Any:
    from types import SimpleNamespace

    from nanoclaw.cron.service import CronService
    from nanoclaw.webui.context import WebUIContext

    def check(request: Any) -> bool:
        return request.headers.get("Authorization") == f"Bearer {REST_TOKEN}"

    return WebUIContext(
        config=Config(),
        sessions=history.sessions,
        history=history.store,
        loop=None,
        workspace=history.sessions.workspace,
        check_auth=check,
        channel=SimpleNamespace(turns=_RestTurns()),  # type: ignore[arg-type]
        cron=CronService(tmp_path / "cron" / "jobs.json"),
    )


@pytest.fixture
async def rest(rest_ctx: Any):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from nanoclaw.webui.rest_api import register_api_fallback, register_rest_routes

    app = web.Application()
    register_rest_routes(app, rest_ctx)
    register_api_fallback(app)
    async with TestClient(TestServer(app)) as client:
        yield client


def _seed_chat(ctx: Any, key: str, *texts: str) -> None:
    session = ctx.sessions.get_or_create(key)
    for index, text in enumerate(texts):
        session.add_message("user" if index % 2 == 0 else "assistant", text)
    ctx.sessions.save(session)


def test_rest_json_gzip_starts_at_four_kib() -> None:
    from nanoclaw.webui.rest_api import _json_response

    at = _json_response("x" * 4094, accept_encoding="gzip")
    below = _json_response("x" * 4093, accept_encoding="gzip")
    assert at.headers["Content-Encoding"] == "gzip"
    assert "Content-Encoding" not in below.headers


def test_rest_gzip_negotiation() -> None:
    from nanoclaw.webui.rest_api import _accepts_gzip

    assert _accepts_gzip("GZIP") and _accepts_gzip("*") and _accepts_gzip("br;q=1, *;q=0.5")
    assert not _accepts_gzip("*;q=0") and not _accepts_gzip("br")


async def test_rest_auth_check_errors_deny(rest, rest_ctx) -> None:
    def broken(request: Any) -> bool:
        raise RuntimeError("auth backend down")

    rest_ctx.check_auth = broken
    resp = await rest.get("/api/commands", headers=REST_AUTH)
    assert resp.status == 401


async def test_rest_session_keys_are_decoded_from_the_raw_path(rest, rest_ctx) -> None:
    from yarl import URL

    _seed_chat(rest_ctx, "websocket:abc", "hi")
    base = str(rest.make_url(""))
    resp = await rest.session.get(URL(base + "/api/sessions/websocket%3Aabc/context", encoded=True), headers=REST_AUTH)
    assert resp.status == 200 and (await resp.json())["session_key"] == "websocket:abc"
    resp = await rest.session.get(URL(base + "/api/sessions/websocket%3Aa%2Fb/context", encoded=True), headers=REST_AUTH)
    assert resp.status == 400 and await resp.text() == "invalid session key"
    # Decoded once, as nanobot reads the raw path: a double-encoded colon stays invalid.
    resp = await rest.session.get(URL(base + "/api/sessions/websocket%253Aabc/context", encoded=True), headers=REST_AUTH)
    assert resp.status == 400
    resp = await rest.get("/api/sessions/websocket%3A" + "a" * 120 + "/context", headers=REST_AUTH)
    assert resp.status == 400


async def test_rest_repeated_query_parameters_use_the_first(rest, rest_ctx) -> None:
    _seed_chat(rest_ctx, "websocket:abc", "hi", "hello")
    resp = await rest.get("/api/sessions/websocket%3Aabc/webui-thread?limit=5&limit=bad", headers=REST_AUTH)
    assert resp.status == 200
    resp = await rest.get("/api/sessions/websocket%3Aabc/webui-thread?limit=%20", headers=REST_AUTH)
    assert resp.status == 200


async def test_rest_after_cursor_excludes_other_navigation(rest, rest_ctx) -> None:
    from nanoclaw.webui.transcript import _encode_page_cursor

    _seed_chat(rest_ctx, "websocket:abc", "hi", "hello")
    cursor = _encode_page_cursor(1)
    url = "/api/sessions/websocket%3Aabc/webui-thread"
    assert (await rest.get(f"{url}?after={cursor}", headers=REST_AUTH)).status != 400
    for extra in (f"&before={cursor}", "&direction=latest"):
        resp = await rest.get(f"{url}?after={cursor}{extra}", headers=REST_AUTH)
        assert resp.status == 400 and await resp.text() == "invalid navigation cursor"


async def test_rest_thread_etag_follows_the_turn_start_time(rest, rest_ctx) -> None:
    _seed_chat(rest_ctx, "websocket:abc", "hi", "hello")
    url = "/api/sessions/websocket%3Aabc/webui-thread"
    rest_ctx.channel.turns.ids["abc"] = "t1"
    rest_ctx.channel.turns.started["abc"] = 100.0
    first = (await rest.get(url, headers=REST_AUTH)).headers["ETag"]
    rest_ctx.channel.turns.started["abc"] = 200.0
    second = await rest.get(url, headers={**REST_AUTH, "If-None-Match": first})
    assert second.status == 200 and second.headers["ETag"] != first


async def test_rest_session_context_counts(rest, rest_ctx) -> None:
    from nanoclaw.runtime_context import RUNTIME_CONTEXT_HISTORY_META

    suffix = "[Runtime Context]\n" + "time: now " * 200
    session = rest_ctx.sessions.get_or_create("websocket:ctx")
    session.add_message("user", "hi\n\n" + suffix)
    session.messages[-1][RUNTIME_CONTEXT_HISTORY_META] = {"version": 1, "suffix": suffix}
    session.add_message("assistant", "hello")
    session.last_archived = 0
    session.metadata["_last_summary"] = {"text": "s" * 5000, "last_active": "2026-01-01T00:00:00"}
    rest_ctx.sessions.save(session)
    _seed_chat(rest_ctx, "websocket:plain", "hi", "hello")
    url = "/api/sessions/websocket%3A{}/context"
    body = await (await rest.get(url.format("ctx"), headers=REST_AUTH)).json()
    plain = await (await rest.get(url.format("plain"), headers=REST_AUTH)).json()
    assert body["estimated_replay_tokens"] == plain["estimated_replay_tokens"]
    assert body["estimated_summary_tokens"] > 0
    assert body["estimated_session_tokens"] == body["estimated_replay_tokens"] + body["estimated_summary_tokens"]
    assert len(body["archived_summary"]) < 5000
    session.last_archived = 10
    rest_ctx.sessions.save(session)
    body = await (await rest.get(url.format("ctx"), headers=REST_AUTH)).json()
    assert body["archived_messages"] == 2


async def test_rest_automation_pending_flags_cover_origin_and_disabled_jobs(rest, rest_ctx) -> None:
    from nanoclaw.cron.types import CronSchedule

    cron = rest_ctx.cron
    hourly = CronSchedule(kind="every", every_ms=3_600_000)
    by_origin = cron.add_job(name="origin", schedule=hourly, message="m",
                             origin_channel="websocket", origin_chat_id="o")
    disabled = cron.add_job(name="off", schedule=hourly, message="m", session_key="websocket:d",
                            origin_channel="websocket", origin_chat_id="d")
    cron.enable_job(disabled.id, enabled=False)
    pending = {"websocket:o": {by_origin.id}, "websocket:d": {disabled.id}}

    class Loop:
        def pending_cron_job_ids_for_session(self, key: str) -> set[str]:
            return pending.get(key, set())

    rest_ctx.loop = Loop()
    jobs = (await (await rest.get("/api/webui/automations", headers=REST_AUTH)).json())["jobs"]
    assert {job["name"]: job["state"]["pending"] for job in jobs} == {"origin": True, "off": True}


async def test_rest_unbound_automation_can_be_disabled_and_run_is_not_forced(rest, rest_ctx) -> None:
    from nanoclaw.cron.types import CronSchedule

    cron = rest_ctx.cron
    hourly = CronSchedule(kind="every", every_ms=3_600_000)
    loose = cron.add_job(name="loose", schedule=hourly, message="m")
    response = await rest_ctx.mutations["automation.disable"]({"id": loose.id})
    assert response.status == 200 and cron.get_job(loose.id).enabled is False
    bound = cron.add_job(name="bound", schedule=hourly, message="m", session_key="websocket:b",
                         origin_channel="websocket", origin_chat_id="b")
    calls: list[tuple[str, bool]] = []

    async def run_job(job_id: str, *, force: bool = False) -> bool:
        calls.append((job_id, force))
        return True

    cron.run_job = run_job  # type: ignore[method-assign]
    response = await rest_ctx.mutations["automation.run"]({"id": bound.id})
    await asyncio.sleep(0)
    assert response.status == 200 and calls == [(bound.id, False)]


def test_sidebar_state_normalization_limits() -> None:
    from nanoclaw.webui.rest_api import normalize_webui_sidebar_state as norm

    state = norm({
        "pinned_keys": [f"k{i}" for i in range(2_100)],
        "collapsed_groups": {"g": 1},
        "title_overrides": {"k": "t" * 200},
        "view": {"show_archived": 1},
    })
    assert len(state["pinned_keys"]) == 2_000
    assert state["collapsed_groups"]["g"] is True
    assert len(state["title_overrides"]["k"]) == 160
    assert state["view"]["show_archived"] is True


def _tab(**fields: Any) -> dict[str, Any]:
    from nanoclaw.webui.rest_api import normalize_webui_sidebar_state as norm

    return norm({"workbench": {"version": 1, "tabs": {"t": fields}}})["workbench"]["tabs"].get("t")


def test_sidebar_workbench_normalization() -> None:
    from nanoclaw.webui.rest_api import normalize_webui_sidebar_state as norm

    tab = _tab(paneKeys=["a", "b"], splitRatios=[True, 0.123456, 0.5, 0.6, 0.7], layoutPaneKeys=["zz", "b"])
    assert tab["splitRatios"] == [0.1235, 0.5]
    assert tab["layoutPaneKeys"] == ["b", "a"]
    assert _tab(paneKeys=list("abcdef"))["paneKeys"] == list("abcd")
    assert _tab(paneKeys=["a"]) is None
    assert _tab(paneKeys=["a"], explicit=1) is None
    assert _tab(paneKeys=["a"], explicit=True)["paneKeys"] == ["a"]
    assert norm({"workbench": {"version": 2, "tabs": {"t": {"paneKeys": ["a", "b"]}}}})["workbench"]["tabs"] == {}
    tabs = norm({"workbench": {"version": 1, "tabs": {
        "one": {"paneKeys": ["a", "b"]}, "two": {"paneKeys": ["b", "c"], "explicit": True},
    }}})["workbench"]["tabs"]
    assert tabs["one"]["paneKeys"] == ["a", "b"] and tabs["two"]["paneKeys"] == ["c"]


def test_sidebar_state_keeps_order_project_names_and_tab_fields(tmp_path) -> None:
    from nanoclaw.webui.rest_api import normalize_webui_sidebar_state as norm
    from nanoclaw.webui.rest_api import read_webui_sidebar_state

    state = norm({"session_order": ["b", "a"], "project_name_overrides": {"/p": " Proj "}})
    assert state["session_order"] == ["b", "a"] and state["project_name_overrides"] == {"/p": "Proj"}
    tab = _tab(paneKeys=["a", "b"], layout="spiral", title="t" * 200)
    assert tab["layout"] == "columns" and len(tab["title"]) == 160
    assert _tab(paneKeys=["a", "b"], layout="grid")["layout"] == "grid"
    path = tmp_path / "sidebar-state.json"
    path.write_text(json.dumps({"pinned_keys": [" k ", "k", 3], "junk": True, "view": {"sort": "nope"}}))
    read = read_webui_sidebar_state(path)
    assert read["pinned_keys"] == ["k"] and "junk" not in read and read["view"]["sort"] == "updated_desc"


async def _settings(rest: Any, rest_ctx: Any, raw: dict[str, Any]) -> dict[str, Any]:
    rest_ctx.config = Config.model_validate(raw)
    resp = await rest.get("/api/settings", headers=REST_AUTH)
    assert resp.status == 200
    return await resp.json()


async def test_settings_provider_rows_follow_the_registry_rules(rest, rest_ctx) -> None:
    body = await _settings(rest, rest_ctx, {
        "agents": {"defaults": {"model": "anthropic/claude-x"}},
        "providers": {
            "anthropic": {"apiKey": "sk-ant-0123456789"},
            "custom": {"apiKey": "k-only"},
            "openai": {"apiBase": "https://gw.example/v1"},
            "ollama": {"apiBase": "http://localhost:11434/v1"},
            "openrouter": {"extraBody": {"api_key": "x", "n": 1}, "extraQuery": {"token": "t", "q": "v"}},
        },
    })
    rows = {row["name"]: row for row in body["providers"]}
    assert rows["custom"]["configured"] is False  # a custom endpoint needs its base
    assert rows["openai"]["configured"] is False  # a hosted provider needs its key
    assert rows["ollama"]["configured"] is True  # a local one needs either
    assert [rows[n]["api_key_required"] for n in ("custom", "ollama", "openai")] == [False, False, True]
    assert {n: rows[n]["model_catalog"] for n in ("anthropic", "ollama", "openrouter", "openai", "custom")} == {
        "anthropic": "unsupported", "ollama": "local", "openrouter": "catalog", "openai": "official", "custom": "custom",
    }
    assert rows["anthropic"]["request_apis"] == ["anthropic_messages"]
    assert rows["openai"]["request_apis"] == ["chat_completions"]
    assert "thinking_style" not in rows["openai"]["advanced_fields"]
    assert rows["custom"]["default_api_base"] is None
    assert rows["openrouter"]["default_api_base"] == "https://openrouter.ai/api/v1"
    assert rows["openrouter"]["extra_body"] == {"api_key": "••••••••", "n": 1}
    assert rows["openrouter"]["extra_query"] == {"token": "••••••••", "q": "v"}


async def test_settings_active_provider_without_credentials_is_not_configured(rest, rest_ctx) -> None:
    body = await _settings(rest, rest_ctx, {"agents": {"defaults": {"model": "gpt-4o", "provider": "openai"}}})
    rows = {row["name"]: row for row in body["providers"]}
    assert rows["openai"]["configured"] is False
    assert body["agent"]["has_api_key"] is False


async def test_settings_active_custom_entry_with_only_a_key_is_configured(rest, rest_ctx) -> None:
    body = await _settings(rest, rest_ctx, {
        "agents": {"defaults": {"model": "companyProxy/gpt-4"}},
        "providers": {"companyProxy": {"apiKey": "k"}, "spare": {"apiKey": "k"}},
    })
    rows = {row["name"]: row for row in body["providers"]}
    assert rows["companyProxy"]["configured"] is True
    assert rows["spare"]["configured"] is False


async def test_settings_presets_labels_extras_and_effort_values(rest, rest_ctx) -> None:
    body = await _settings(rest, rest_ctx, {
        "agents": {"defaults": {"model": "anthropic/claude-x"}},
        "modelPresets": {
            "loose": {"model": "unknown-model", "extraHeaders": {"Authorization": "Bearer s", "X-Team": "a"}},
            "think": {"model": "magistral-medium", "provider": "mistral"},
            "plain": {"model": "mistral-large", "provider": "mistral"},
            "mini": {"model": "MiniMax-M1", "provider": "minimax"},
            "own": {"model": "companyProxy/x", "provider": "companyProxy"},
        },
    })
    presets = {p["name"]: p for p in body["model_presets"]}
    assert presets["default"]["label"] == "Default" and presets["loose"]["label"] == "loose"
    assert "extra_headers" not in presets["default"]
    assert presets["loose"]["extra_headers"] == {"Authorization": "••••••••", "X-Team": "a"}
    assert presets["loose"]["resolved_provider"] == "auto"
    assert presets["think"]["reasoning_effort_values"] == [""]
    assert presets["plain"]["reasoning_effort_values"] == ["", "high"]
    assert presets["mini"]["reasoning_effort_values"] == ["", "low", "medium", "high"]
    assert presets["own"]["reasoning_effort_values"] == ["", "low", "medium", "high"]


def test_settings_secret_redaction_rules() -> None:
    from nanoclaw.webui.rest_api import _mask_secret_hint, _redact_request_secrets

    assert _redact_request_secrets({
        "X-Api-Key": "v", "Authorization": "", "nested": {"client_secret": "s", "ok": 1},
        "items": [{"password": "p"}], "plain": "keep",
    }) == {
        "X-Api-Key": "••••••••", "Authorization": "", "nested": {"client_secret": "••••••••", "ok": 1},
        "items": [{"password": "••••••••"}], "plain": "keep",
    }
    assert _mask_secret_hint("12345678") == "••••"
    assert _mask_secret_hint("123456789") == "1234••••6789"


async def test_settings_runtime_and_advanced_sections_read_the_config(rest, rest_ctx) -> None:
    body = await _settings(rest, rest_ctx, {
        "agents": {"defaults": {"model": "anthropic/claude-x", "timezone": "Europe/Paris", "maxTokens": 1234}},
        "gateway": {"port": 1999, "heartbeat": {"enabled": False, "intervalS": 60}},
        "tools": {"restrictToWorkspace": True, "ssrfWhitelist": ["10.0.0.0/8"],
                  "mcpServers": {"x": {"command": "echo"}}, "exec": {"enable": False}},
    })
    assert body["agent"]["timezone"] == "Europe/Paris" and body["agent"]["max_tokens"] == 1234
    assert body["runtime"]["workspace_path"] == str(rest_ctx.workspace)
    assert body["runtime"]["gateway_port"] == 1999
    assert body["runtime"]["heartbeat"] == {"enabled": False, "interval_s": 60}
    advanced = body["advanced"]
    assert advanced["restrict_to_workspace"] is True
    assert (advanced["ssrf_whitelist_count"], advanced["mcp_server_count"], advanced["exec_enabled"]) == (1, 1, False)


async def test_skills_list_workspace_skills_first(rest, rest_ctx) -> None:
    skill = rest_ctx.workspace / "skills" / "zz-mine"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: zz-mine\ndescription: Mine\n---\nBody\n")
    skills = (await (await rest.get("/api/webui/skills", headers=REST_AUTH)).json())["skills"]
    assert skills[0]["name"] == "zz-mine" and skills[0]["deletable"] is True
    assert len(skills) > 1 and all(s["source"] != "workspace" for s in skills[1:])


async def test_skills_show_unavailable_ones_with_reason_and_fallback_description(rest, rest_ctx) -> None:
    root = rest_ctx.workspace / "skills"
    for name, meta in (("needs-tool", 'metadata: {"nanobot":{"requires":{"bins":["no-such-bin-xyz"]}}}\n'),
                       ("bare", "")):
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(f"---\nname: {name}\n{meta}---\nBody\n")
    skills = {s["name"]: s for s in (await (await rest.get("/api/webui/skills", headers=REST_AUTH)).json())["skills"]}
    assert skills["needs-tool"]["available"] is False
    assert skills["needs-tool"]["unavailable_reason"] == "CLI: no-such-bin-xyz"
    assert skills["bare"]["description"] == "bare"


# -- webui/session_automations.py ----------------------------------------------------------------


def test_automation_details_history_origin_and_binding(rest_ctx) -> None:
    from nanoclaw.cron.types import CronRunRecord, CronSchedule
    from nanoclaw.webui.session_automations import serialize_automation_jobs

    cron = rest_ctx.cron
    hourly = CronSchedule(kind="every", every_ms=3_600_000)
    bound = cron.add_job(name="bound", schedule=hourly, message="m", session_key="websocket:b",
                         origin_channel="websocket", origin_chat_id="b")
    bound.state.run_history = [CronRunRecord(run_at_ms=i, status="ok") for i in range(7)]
    bound.state.run_history[-1].session_key = "cli:direct"
    bound.state.last_run_at_ms = 6
    bound.state.last_error = "boom"
    loose = cron.add_job(name="loose", schedule=hourly, message="m", origin_channel="websocket")
    session = rest_ctx.sessions.get_or_create("websocket:b")  # cached, never saved
    session.metadata["title"] = "Bread chat"
    session.add_message("assistant", "Welcome back")
    session.add_message("user", "hidden", _hidden_history=True)
    session.add_message("assistant", "Second")
    session.add_message("user", "How long to proof?")
    [b, lo] = serialize_automation_jobs([bound, loose], include_details=True, sessions=rest_ctx.sessions)
    history = b["state"]["run_history"]
    assert [r["run_at_ms"] for r in history] == [2, 3, 4, 5, 6]
    assert [r["webui_session_key"] for r in history] == ["websocket:b"] * 4 + [None]
    assert (b["state"]["last_run_at_ms"], b["state"]["last_error"]) == (6, "boom")
    assert b["origin"]["title"] == "Bread chat" and b["origin"]["preview"] == "How long to proof?"
    assert "chat_binding_revision" in b and "chat_binding_revision" not in lo
    assert lo["origin"] is None


def test_automation_origin_preview_prefers_the_first_visible_user_text(rest_ctx) -> None:
    from nanoclaw.webui.session_automations import _session_preview

    assert _session_preview([
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "secret", "_hidden_history": True},
        {"role": "assistant", "content": "Second answer"},
        {"role": "tool", "content": "tool output"},
    ]) == "First answer"
    assert _session_preview([
        {"role": "user", "content": "secret", "_hidden_history": True},
        {"role": "user", "content": "Visible"},
    ]) == "Visible"
    assert _session_preview([{"role": "tool", "content": "tool output"}]) == ""


# -- webui_request and session.delete --------------------------------------------------------------


def test_mutation_query_values() -> None:
    from nanoclaw.webui.inbound_commands import mutation_query_value

    assert mutation_query_value({}, "x") is None
    assert mutation_query_value({"x": None}, "x") == ""
    assert mutation_query_value({"x": True}, "x") == "true"


async def test_webui_request_ids_payload_order_and_conflicts(harness: Harness) -> None:
    from aiohttp import web

    from nanoclaw.webui.inbound_commands import mutation_json

    async def echo(payload: dict[str, Any]) -> web.Response:
        return mutation_json(payload)

    ch = harness.channel
    ch.context.mutations["probe.a"] = echo
    ch.context.mutations["probe.b"] = echo
    client = await harness.connect()
    assert await client.request(request_id="r" * 129, action="probe.a", payload={}) == [
        {"event": "error", "detail": "invalid webui request_id"}
    ]
    first = await client.request(request_id="r1", action="probe.a", payload={"a": 1, "b": 2})
    assert first[0]["ok"] is True
    assert await client.request(request_id="r1", action="probe.a", payload={"b": 2, "a": 1}) == first
    conflict = await client.request(request_id="r1", action="probe.b", payload={"a": 1, "b": 2})
    assert conflict[0]["error"]["status"] == 409
    await client.ws.close()


async def test_webui_request_cache_expires_and_is_bounded(harness: Harness, monkeypatch) -> None:
    from aiohttp import web

    from nanoclaw.webui import inbound_commands
    from nanoclaw.webui.inbound_commands import mutation_json

    runs: list[str] = []

    async def count(payload: dict[str, Any]) -> web.Response:
        runs.append(payload["n"])
        return mutation_json({"n": payload["n"]})

    ch = harness.channel
    ch.context.mutations["probe.count"] = count
    monkeypatch.setattr(inbound_commands, "_WEBUI_REQUEST_CACHE_MAX", 2)
    client = await harness.connect()
    for n in ("1", "2", "3"):
        await client.request(request_id=f"r{n}", action="probe.count", payload={"n": n})
    assert set(ch.commands.request_operations) == {"r2", "r3"}
    await client.request(request_id="r3", action="probe.count", payload={"n": "3"})
    assert runs == ["1", "2", "3"]
    monkeypatch.setattr(inbound_commands, "_WEBUI_REQUEST_CACHE_TTL_S", 0.05)
    await asyncio.sleep(0.1)
    await client.request(request_id="r3", action="probe.count", payload={"n": "3"})
    assert runs == ["1", "2", "3", "3"]
    await client.ws.close()


async def test_webui_request_duplicates_and_replays_keep_order(harness: Harness) -> None:
    from aiohttp import web

    from nanoclaw.webui.inbound_commands import mutation_json

    gate = asyncio.Event()

    async def fast(payload: dict[str, Any]) -> web.Response:
        return mutation_json({"fast": True})

    async def slow(payload: dict[str, Any]) -> web.Response:
        await gate.wait()
        return mutation_json({"slow": True})

    ch = harness.channel
    ch.context.mutations["probe.fast"] = fast
    ch.context.mutations["probe.slow"] = slow
    client = await harness.connect()
    await client.request(request_id="f1", action="probe.fast", payload={})
    await client.send({"type": "webui_request", "request_id": "s1", "action": "probe.slow", "payload": {}})
    await client.send({"type": "webui_request", "request_id": "s1", "action": "probe.slow", "payload": {}})
    await client.send({"type": "webui_request", "request_id": "f1", "action": "probe.fast", "payload": {}})
    assert await client.recv_all(0.2) == []  # the replay waits behind the running mutation
    gate.set()
    frames = await client.recv_all(0.3)
    assert sorted(f["request_id"] for f in frames) == ["f1", "s1"]  # one answer per request id
    await client.ws.close()


async def test_webui_request_errors_and_sidebar_broadcast(harness: Harness) -> None:
    from aiohttp import web

    async def broken(payload: dict[str, Any]) -> web.Response:
        raise RuntimeError("boom")

    ch = harness.channel
    ch.context.mutations["probe.broken"] = broken
    a, b = await harness.connect(), await harness.connect()
    assert await a.request(request_id="x1", action="probe.broken", payload={}) == [{
        "event": "webui_response", "request_id": "x1", "ok": False,
        "error": {"status": 500, "message": "WebUI mutation failed"},
    }]
    frames = await a.request(request_id="s1", action="sidebar.update", payload={"state": {"pinned_keys": ["k"]}})
    events = sorted(f["event"] for f in frames)
    assert events == ["sidebar_state_updated", "webui_response"]
    assert (await b.recv())["event"] == "sidebar_state_updated"
    await a.ws.close()
    await b.ws.close()


@pytest.mark.ws_config({"token": "static"})
async def test_set_sidebar_state_needs_trust_and_a_successful_save(harness: Harness) -> None:
    from aiohttp import web

    from nanoclaw.webui.inbound_commands import mutation_error

    plain = await harness.connect("?token=static", trusted=False)
    await plain.send({"type": "set_sidebar_state", "state": {"pinned_keys": ["k"]}})
    assert await plain.recv() == {"event": "error", "detail": "access_denied"}

    async def refuse(payload: dict[str, Any]) -> web.Response:
        return mutation_error(400, "sidebar state is too large")

    harness.channel.context.mutations["sidebar.update"] = refuse
    trusted = await harness.connect(headers={"X-Nanobot-Auth": "static"})
    await trusted.send({"type": "set_sidebar_state", "state": {"pinned_keys": ["k"]}})
    assert await trusted.recv() == {"event": "error", "detail": "invalid_sidebar_state"}
    await plain.ws.close()
    await trusted.ws.close()


async def test_session_delete_validates_keys_and_releases_chat_state(harness: Harness, monkeypatch) -> None:
    ch = harness.channel
    delete = ch.context.mutations["session.delete"]
    resp = await delete({"key": "   "})
    assert resp.status == 400 and resp.text == "missing session key"
    resp = await delete({"key": "websocket:" + "a" * 120})
    assert resp.status == 400 and resp.text == "invalid session key"
    calls: list[tuple[str, str]] = []

    async def cancel(key: str) -> int:
        calls.append(("cancel", key))
        return 0

    monkeypatch.setattr(ch.agent, "cancel_active_tasks", cancel)
    monkeypatch.setattr(ch.agent, "discard_session_file_state", lambda key: calls.append(("discard", key)))
    monkeypatch.setattr(ch.projector, "forget_chat", lambda chat_id: calls.append(("forget", chat_id)))
    resp = await delete({"key": "websocket:gone"})
    assert resp.status == 200
    assert calls == [("cancel", "websocket:gone"), ("discard", "websocket:gone"), ("forget", "gone")]


def test_rest_gateway_instance_sources(rest_ctx) -> None:
    from types import SimpleNamespace

    from nanoclaw.webui.rest_api import _gateway_instance

    assert _gateway_instance(rest_ctx) is None
    rest_ctx.extras["instance_id"] = "from-extras"
    assert _gateway_instance(rest_ctx) == "from-extras"
    rest_ctx.channel = SimpleNamespace(turns=None, tokens=SimpleNamespace(instance_id="from-tokens"))
    assert _gateway_instance(rest_ctx) == "from-tokens"


# -- session/webui_turns.py -----------------------------------------------------------------


def test_a_resumed_owner_becomes_the_latest_turn() -> None:
    from nanoclaw.session.webui_turns import WebsocketTurnRegistry

    reg = WebsocketTurnRegistry()
    reg.mark_running("c", "o1", "t1", 1.0)
    reg.mark_running("c", "o2", "t2", 2.0)
    reg.mark_running("c", "o1", "t1", 3.0)
    assert reg.turn_id("c") == "t1" and reg.wall_started_at("c") == 3.0


def test_persistence_failure_is_read_per_owner() -> None:
    from nanoclaw.session.webui_turns import WebsocketTurnRegistry

    reg = WebsocketTurnRegistry()
    reg.mark_running("c", "o1", "t1", 1.0)
    reg.mark_running("c", "o2", "t2", 2.0)
    assert reg.mark_transcript_persistence_failed("c", "o1")
    assert reg.transcript_persistence_failed("c", "o1")
    assert not reg.transcript_persistence_failed("c", "o2")
    assert not reg.transcript_persistence_failed("c")
