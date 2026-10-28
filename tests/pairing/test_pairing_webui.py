"""Pairing in the web UI: ``GET /api/settings/pairing``, the
``settings.pairing.approve`` / ``deny`` mutations, ``pairing_changed`` frames,
and the whole flow from an unknown DM sender to an approved one.

Ported from nanobot tests/webui/test_settings_system.py
(``test_pairing_payload_preserves_optional_delivery_state``),
channels/websocket/tests/test_websocket_http_routes.py (the pairing routes),
test_websocket_channel.py (``test_pairing_invalidation_reaches_only_authenticated_webui_connections``),
test_websocket_runtime_lifecycle.py (PairingChanged → ``websocket`` / ``*``)
and tests/test_fixture_isolation.py (c4dac64). The flow tests are NanoClaw's.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nanoclaw.bus.events import OutboundMessage
from nanoclaw.bus.queue import MessageBus
from nanoclaw.channels.base import BaseChannel
from nanoclaw.config.schema import Config
from nanoclaw.pairing import store
from nanoclaw.pairing.events import PairingChanged
from nanoclaw.webui.context import WebUIContext
from nanoclaw.webui.outbound_projection import WebUIOutboundProjector
from nanoclaw.webui.rest_api import register_api_fallback, register_rest_routes
from nanoclaw.webui.settings_system import pairing_payload
from test_websocket_channel import Harness, harness  # noqa: F401

TOKEN = "nbwt_test"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _check_auth(request: web.Request) -> bool:
    return request.headers.get("Authorization") == f"Bearer {TOKEN}"


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus()


@pytest.fixture
def ctx(history, bus: MessageBus) -> WebUIContext:  # noqa: F811
    return WebUIContext(
        config=Config(),
        sessions=history.sessions,
        history=history.store,
        loop=None,
        workspace=history.sessions.workspace,
        check_auth=_check_auth,
        channel=SimpleNamespace(bus=bus),
    )


@pytest.fixture
async def client(ctx: WebUIContext):
    app = web.Application()
    register_rest_routes(app, ctx)
    register_api_fallback(app)
    async with TestClient(TestServer(app)) as c:
        yield c


class _FakeDMChannel(BaseChannel):
    name = "fake"
    display_name = "Fake"

    def __init__(self, bus: MessageBus) -> None:
        super().__init__({"allowFrom": []}, bus)
        self.sent: list[OutboundMessage] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, msg: OutboundMessage) -> None:
        self.sent.append(msg)


# --- pairing_payload ------------------------------------------------------------------


@pytest.mark.parametrize("status", [None, "sending", "sent", "failed", "unknown", []])
def test_pairing_payload_preserves_optional_delivery_state(status):
    item = {"code": "ABCD-EFGH", "channel": "feishu", "sender_id": "alice",
            "created_at": 1000, "expires_at": 1600}
    if status is not None:
        item["delivery_status"] = status
    [request] = pairing_payload(lambda: [item], now=1100)["requests"]
    assert request["code"] == "ABCD-EFGH"
    assert request["expires_in_seconds"] == 500
    if isinstance(status, str) and status in {"sending", "sent", "failed"}:
        assert request["delivery_status"] == status
    else:
        assert "delivery_status" not in request


# --- REST route and mutations -------------------------------------------------------------


async def test_pairing_routes_require_token_and_approve_or_deny(client, ctx, monkeypatch) -> None:
    pending = [
        {
            "code": "ABCD-EFGH",
            "channel": "feishu",
            "sender_id": "ou_123",
            "created_at": 1_000.0,
            "expires_at": 1_600.0,
        }
    ]
    approved: list[str] = []
    denied: list[str] = []
    monkeypatch.setattr("nanoclaw.webui.rest_api.list_pending", lambda: list(pending))
    monkeypatch.setattr(
        "nanoclaw.webui.rest_api.approve_code",
        lambda code, **_kwargs: approved.append(code) or ("feishu", "ou_123") if code == "ABCD-EFGH" else None,
    )
    monkeypatch.setattr(
        "nanoclaw.webui.rest_api.deny_code",
        lambda code, **_kwargs: denied.append(code) or code == "ABCD-EFGH",
    )

    assert (await client.get("/api/settings/pairing")).status == 401
    listed = await client.get("/api/settings/pairing", headers=AUTH)
    assert listed.status == 200
    body = await listed.json()
    assert body["requests"][0]["code"] == "ABCD-EFGH"
    assert body["requests"][0]["channel"] == "feishu"
    assert body["requests"][0]["sender_id"] == "ou_123"
    assert body["requests"][0]["created_at_ms"] == 1_000_000
    assert body["requests"][0]["expires_at_ms"] == 1_600_000

    approved_response = await ctx.mutations["settings.pairing.approve"]({"code": "ABCD-EFGH"})
    assert approved_response.status == 200
    import json

    body = json.loads(approved_response.text)
    assert body["last_action"] == {
        "ok": True, "action": "approve", "message": "Approved ou_123 for feishu",
        "channel": "feishu", "sender_id": "ou_123", "code": "ABCD-EFGH",
    }
    assert approved == ["ABCD-EFGH"]

    denied_action = await ctx.mutations["settings.pairing.deny"]({"code": "ABCD-EFGH"})
    assert denied_action.status == 200
    assert json.loads(denied_action.text)["last_action"] == {
        "ok": True, "action": "deny", "message": "Denied pairing code ABCD-EFGH", "code": "ABCD-EFGH",
    }
    assert denied == ["ABCD-EFGH"]

    missing_code = await ctx.mutations["settings.pairing.approve"]({})
    assert missing_code.status == 400
    assert "Missing pairing code" in missing_code.text
    unknown = await ctx.mutations["settings.pairing.deny"]({"code": "NOPE-NOPE"})
    assert (unknown.status, unknown.text) == (404, "Pairing code not found or expired")


@pytest.mark.parametrize("action", ["approve", "deny"])
async def test_pairing_mutation_publishes_committed_snapshot(action, client, ctx, bus) -> None:
    code = store.generate_code("telegram", "alice")
    snapshots = []
    bus.subscribe(lambda _event: snapshots.append(store.list_pending()), PairingChanged)
    result = await ctx.mutations[f"settings.pairing.{action}"]({"code": code})
    await bus.drain()
    assert result.status == 200 and snapshots == [[]]
    assert store.is_approved("telegram", "alice") == (action == "approve")
    missing = await ctx.mutations[f"settings.pairing.{action}"]({"code": code})
    await bus.drain()
    assert missing.status == 404 and snapshots == [[]]


@pytest.mark.parametrize("action", ["approve", "deny"])
async def test_pairing_mutation_paths_reject_plain_http(client, action) -> None:
    resp = await client.post(f"/api/settings/pairing/{action}?code=X", headers=AUTH)
    assert resp.status == 405
    assert await resp.text() == "WebUI mutations require an authenticated WebSocket"


# --- pairing_changed ------------------------------------------------------------------------


def test_projector_sends_pairing_changed_to_webui_clients_only() -> None:
    [frame] = WebUIOutboundProjector().project(
        OutboundMessage(channel="websocket", chat_id="*", content="", event=PairingChanged())
    )
    assert frame.payload == {"event": "pairing_changed"}
    assert frame.audience == "webui" and frame.record is None


async def test_pairing_invalidation_reaches_only_authenticated_webui_connections(harness: Harness) -> None:  # noqa: F811
    webui = await harness.connect()
    harness.channel.config.token = "static"
    ordinary = await harness.connect("?token=static", trusted=False)
    await asyncio.sleep(0.05)
    await harness.channel.send(OutboundMessage(channel="websocket", chat_id="*", content="", event=PairingChanged()))
    assert await webui.recv_all() == [{"event": "pairing_changed"}]
    assert await ordinary.recv_all() == []
    await webui.ws.close()
    await ordinary.ws.close()


async def test_running_websocket_queues_pairing_changes_for_all_webui_clients(harness: Harness) -> None:  # noqa: F811
    bus = harness.channel.bus
    await bus.publish(PairingChanged())
    notification = await asyncio.wait_for(bus.consume_outbound(), 1)
    assert notification.channel == "websocket" and notification.chat_id == "*"
    assert isinstance(notification.event, PairingChanged)


async def test_stopped_websocket_does_not_queue_pairing_changes(harness: Harness) -> None:  # noqa: F811
    await harness.channel.stop()
    await asyncio.wait_for(harness.task, 5)
    await harness.channel.bus.publish(PairingChanged())
    assert harness.channel.bus.outbound_size == 0


# --- isolation ------------------------------------------------------------------------------


@pytest.mark.parametrize("_iteration", range(2))
def test_pairing_store_is_isolated_between_tests(_iteration: int) -> None:
    """(nanobot tests/test_fixture_isolation.py) The store lives in the data
    directory (``~/.nanoclaw``), and every test gets its own HOME."""
    assert store._store_path() == Path.home() / ".nanoclaw" / "pairing.json"
    assert not store.is_approved("test", "isolated-user")
    code = store.generate_code("test", "isolated-user")
    assert store.approve_code(code) is not None
    assert store.is_approved("test", "isolated-user")


# --- the whole flow -------------------------------------------------------------------------


async def _pairing_code_for_stranger(channel: _FakeDMChannel) -> str:
    assert await channel._handle_message("stranger", "dm-1", "hello", is_dm=True) is False
    assert channel.bus.inbound_size == 0
    [reply] = channel.sent
    code = reply.metadata["_pairing_code"]
    assert reply.chat_id == "dm-1" and f"`/pairing approve {code}`" in reply.content
    assert store.list_pending()[0]["delivery_status"] == "sent"
    # A second DM before approval gets the same code.
    await channel._handle_message("stranger", "dm-1", "hello again", is_dm=True)
    assert channel.sent[-1].metadata["_pairing_code"] == code
    return code


async def _assert_stranger_now_allowed(channel: _FakeDMChannel) -> None:
    sent_before = len(channel.sent)
    assert await channel._handle_message("stranger", "dm-1", "hello", is_dm=True) is True
    inbound = channel.bus.inbound.get_nowait()
    assert (inbound.channel, inbound.sender_id, inbound.content) == ("fake", "stranger", "hello")
    assert len(channel.sent) == sent_before
    assert store.list_pending() == []


async def test_flow_cli_pairing_approve_lets_the_sender_in(harness: Harness) -> None:  # noqa: F811
    """Unknown DM sender → code → ``/pairing approve <code>`` in the terminal chat → allowed."""
    agent = harness.channel.agent
    fake = _FakeDMChannel(agent.bus)
    code = await _pairing_code_for_stranger(fake)

    listed = await agent.process_direct("/pairing")
    assert listed.final_content.startswith("Pending pairing requests:\n- `" + code + "` | fake | stranger | ")
    result = await agent.process_direct(f"/pairing approve {code}")
    assert result.final_content == f"Approved pairing code `{code}` — stranger can now access fake"
    await _assert_stranger_now_allowed(fake)

    result = await agent.process_direct("/pairing revoke fake stranger")
    assert result.final_content == "Revoked stranger from fake"
    assert fake.is_allowed("stranger") is False


async def test_flow_webui_approval_lets_the_sender_in(harness: Harness) -> None:  # noqa: F811
    """Unknown DM sender → code → the web UI lists and approves it → allowed;
    every change reaches the web UI as ``pairing_changed``."""
    bus = harness.channel.bus
    fake = _FakeDMChannel(bus)
    client = await harness.connect()
    await asyncio.sleep(0.05)

    code = await _pairing_code_for_stranger(fake)
    await bus.drain()
    # The DM's store changes reached the web UI (through the manager in the gateway).
    changes = 0
    while bus.outbound_size:
        msg = bus.outbound.get_nowait()
        assert isinstance(msg.event, PairingChanged)
        await harness.channel.send(msg)
        changes += 1
    assert changes >= 1
    assert {"event": "pairing_changed"} in await client.recv_all()

    token = (await harness.bootstrap())["api_token"]
    resp = await harness.get("/api/settings/pairing", {"Authorization": f"Bearer {token}"})
    body: dict[str, Any] = await resp.json()
    [request] = body["requests"]
    assert (request["code"], request["channel"], request["sender_id"], request["delivery_status"]) == (
        code, "fake", "stranger", "sent",
    )

    [frame] = await client.request(request_id="p1", action="settings.pairing.approve", payload={"code": code})
    assert frame["ok"] is True
    assert frame["result"]["requests"] == []
    assert frame["result"]["last_action"]["message"] == "Approved stranger for fake"
    await bus.drain()
    msg = await asyncio.wait_for(bus.consume_outbound(), 1)
    await harness.channel.send(msg)
    assert await client.recv_all() == [{"event": "pairing_changed"}]
    await _assert_stranger_now_allowed(fake)
    await client.ws.close()
