"""The web UI WebSocket channel (layer 23).

Adapted from nanobot nanobot/channels/websocket/tests/test_websocket_channel.py
and test_websocket_http_routes.py (c4dac64): config validation, frame
parsing, bootstrap and token rules, the handshake, the envelope router
(validation, dedupe, attach / new_chat, broadcast, webui_request,
session.delete), outbound projection and persistence, the active-turn
registry, static files. A real aiohttp listener on a free port per test.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from pydantic import ValidationError

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.bus.events import OutboundMessage
from nanoclaw.bus.outbound_events import (
    GoalStatusEvent,
    ProgressEvent,
    SessionUpdatedEvent,
    StreamDeltaEvent,
    StreamEndEvent,
    TurnEndEvent,
    outbound_message_for_event,
)
from nanoclaw.bus.runtime_events import RuntimeEventContext, TurnRunStatusChanged
from nanoclaw.channels.websocket import WebSocketChannel
from nanoclaw.channels.websocket.runtime import _parse_envelope, _parse_inbound_payload
from nanoclaw.config.schema import Config, WebSocketConfig
from nanoclaw.providers import FakeProvider
from nanoclaw.session import SessionManager
from nanoclaw.session.webui_turns import WebsocketTurnRegistry
from nanoclaw.webui.http_utils import accepts_gzip, is_local_browser_request, issue_route_secret_matches
from nanoclaw.webui.inbound_commands import mutation_json

# -- helpers ---------------------------------------------------------------------------------


class Harness:
    def __init__(self, channel: WebSocketChannel, task: asyncio.Task[None], session: aiohttp.ClientSession) -> None:
        self.channel = channel
        self.task = task
        self.http = session
        self.port = channel.bound_port
        self.base = f"http://127.0.0.1:{self.port}"

    def inbound(self) -> list[Any]:
        out = []
        while not self.channel.bus.inbound.empty():
            out.append(self.channel.bus.inbound.get_nowait())
        return out

    async def get(self, path: str, headers: dict[str, str] | None = None) -> aiohttp.ClientResponse:
        resp = await self.http.get(self.base + path, headers=headers or {}, allow_redirects=False)
        await resp.read()
        return resp

    async def bootstrap(self, headers: dict[str, str] | None = None) -> dict[str, Any]:
        resp = await self.get("/webui/bootstrap", headers)
        assert resp.status == 200, await resp.text()
        return await resp.json()

    async def connect(self, query: str = "", *, trusted: bool = True, headers: dict[str, str] | None = None) -> Client:
        if trusted and "token=" not in query:
            token = (await self.bootstrap(headers))["token"]
            query = f"?token={token}" + (f"&{query.lstrip('?')}" if query else "")
        ws = await self.http.ws_connect(f"{self.base}/{query}")
        client = Client(ws)
        client.ready = (await client.recv())
        return client


class Client:
    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self.ws = ws
        self.ready: dict[str, Any] = {}

    async def send(self, obj: Any) -> None:
        await self.ws.send_str(obj if isinstance(obj, str) else json.dumps(obj))

    async def recv(self, timeout: float = 3) -> dict[str, Any]:
        msg = await asyncio.wait_for(self.ws.receive(), timeout)
        assert msg.type == aiohttp.WSMsgType.TEXT, msg
        return json.loads(msg.data)

    async def recv_all(self, quiet: float = 0.2) -> list[dict[str, Any]]:
        frames = []
        while True:
            try:
                msg = await asyncio.wait_for(self.ws.receive(), quiet)
            except TimeoutError:
                return frames
            if msg.type != aiohttp.WSMsgType.TEXT:
                return frames
            frames.append(json.loads(msg.data))

    async def request(self, conn_frames_quiet: float = 0.3, **envelope: Any) -> list[dict[str, Any]]:
        await self.send({"type": "webui_request", **envelope})
        return await self.recv_all(conn_frames_quiet)


def make_dist(root: Path) -> Path:
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    js = b"console.log('x');" * 20
    (dist / "assets" / "app-1a2b.js").write_bytes(js)
    (dist / "assets" / "app-1a2b.js.gz").write_bytes(gzip.compress(js))
    (dist / "logo.png").write_bytes(b"\x89PNG")
    (root / "secret.txt").write_text("secret")
    return dist


@pytest.fixture
async def harness(tmp_path, request):
    marker = request.node.get_closest_marker("ws_config")
    overrides: dict[str, Any] = dict(marker.args[0]) if marker else {}
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    agent = AgentLoop(
        FakeProvider([]), ws_dir, sessions=SessionManager(ws_dir, sessions_root=tmp_path / "sessions"), model="m-1"
    )
    cfg = WebSocketConfig.model_validate(
        {"enabled": True, "port": 0, "webuiDist": str(make_dist(tmp_path)), **overrides}
    )
    channel = WebSocketChannel(cfg, agent.bus, agent=agent, app_config=Config())
    task = asyncio.create_task(channel.start())
    async with asyncio.timeout(5):
        while not channel.is_running:
            await asyncio.sleep(0.01)
    async with aiohttp.ClientSession(auto_decompress=False) as session:
        yield Harness(channel, task, session)
    await channel.stop()
    await asyncio.wait_for(task, 5)


def out(chat_id: str, event: Any = None, content: str = "", **metadata: Any) -> OutboundMessage:
    if event is None:
        return OutboundMessage(channel="websocket", chat_id=chat_id, content=content, metadata=metadata)
    msg = outbound_message_for_event(channel="websocket", chat_id=chat_id, event=event, metadata=metadata)
    if content:
        msg.content = content
    return msg


# -- config and parsing (nanobot test_websocket_channel.py) -------------------------------


def test_default_config_is_loopback_streaming_and_disabled() -> None:
    cfg = WebSocketConfig()
    assert (cfg.host, cfg.port, cfg.path, cfg.streaming, cfg.token_ttl_s) == ("127.0.0.1", 8765, "/", True, 300)
    assert cfg.enabled is False  # NanoClaw: opt-in
    assert cfg.websocket_requires_token is True
    assert cfg.allow_from == ["*"]


def test_websocket_config_path_must_start_with_slash() -> None:
    with pytest.raises(ValidationError):
        WebSocketConfig(path="ws")
    assert WebSocketConfig(path="/ws/").path == "/ws"


def test_token_issue_path_must_differ_from_websocket_path() -> None:
    with pytest.raises(ValidationError):
        WebSocketConfig(path="/ws", token_issue_path="/ws/")


def test_wildcard_host_requires_auth() -> None:
    with pytest.raises(ValidationError):
        WebSocketConfig(host="0.0.0.0")
    assert WebSocketConfig(host="0.0.0.0", token_issue_secret="s").host == "0.0.0.0"


def test_public_ws_url_must_match_path() -> None:
    with pytest.raises(ValidationError):
        WebSocketConfig(path="/ws", public_ws_url="wss://example.com/other")
    with pytest.raises(ValidationError):
        WebSocketConfig(public_ws_url="https://example.com/")
    assert WebSocketConfig(path="/ws", public_ws_url="wss://example.com/ws/").public_ws_url == "wss://example.com/ws"


def test_config_accepts_camel_case_under_channels() -> None:
    config = Config.model_validate(
        {"channels": {"sendMaxRetries": 5, "websocket": {"enabled": True, "tokenIssueSecret": "x", "webuiDist": "/d"}}}
    )
    assert config.channels.send_max_retries == 5
    assert config.channels.websocket.token_issue_secret == "x"
    assert config.channels.websocket.webui_dist == "/d"


def test_parse_inbound_payload() -> None:
    assert _parse_inbound_payload("hello") == "hello"
    assert _parse_inbound_payload('{"content": "hi"}') == "hi"
    assert _parse_inbound_payload('{"text": "t"}') == "t"
    assert _parse_inbound_payload('{"message": "m"}') == "m"
    assert _parse_inbound_payload('{"other": 1}') is None
    assert _parse_inbound_payload("   ") is None
    assert _parse_inbound_payload("{not json") == "{not json"
    assert _parse_inbound_payload("[1, 2]") == "[1, 2]"


def test_parse_envelope() -> None:
    assert _parse_envelope('{"type": "attach", "chat_id": "c"}') == {"type": "attach", "chat_id": "c"}
    assert _parse_envelope('{"content": "x"}') is None
    assert _parse_envelope('{"type": 3}') is None
    assert _parse_envelope("plain") is None


def test_issue_route_secret_matches_bearer_and_header() -> None:
    assert issue_route_secret_matches({"Authorization": "Bearer s"}, "s")
    assert issue_route_secret_matches({"X-Nanobot-Auth": "s"}, "s")
    assert not issue_route_secret_matches({"X-Nanobot-Auth": "t"}, "s")
    assert not issue_route_secret_matches({}, "s")
    assert issue_route_secret_matches({}, "")


def test_local_browser_request_rules() -> None:
    assert is_local_browser_request("127.0.0.1", {"Host": "127.0.0.1:8765"})
    assert is_local_browser_request("::1", {"Host": "localhost"})
    assert not is_local_browser_request("10.0.0.2", {"Host": "127.0.0.1"})
    assert not is_local_browser_request("127.0.0.1", {"Host": "example.com"})
    assert not is_local_browser_request("127.0.0.1", {"Host": "127.0.0.1", "X-Forwarded-For": "10.0.0.8"})


def test_accepts_gzip() -> None:
    assert accepts_gzip("gzip, br")
    assert not accepts_gzip("gzip;q=0")
    assert accepts_gzip("*")
    assert not accepts_gzip("br")


# -- bootstrap and handshake ----------------------------------------------------------------


async def test_bootstrap_issues_tokens_for_local_browser(harness: Harness) -> None:
    body = await harness.bootstrap()
    assert body["ws_path"] == "/"
    assert body["ws_url"] == f"ws://127.0.0.1:{harness.port}/"
    assert body["token"].startswith("nbwt_") and body["api_token"].startswith("nbwt_")
    assert body["expires_in"] == 300
    assert body["model_name"] == "m-1"
    assert body["runtime_surface"] == "browser"
    assert body["terminal"]["webui"]["capabilities"] == ["webui.core.v1"]
    assert body["limits"]["message"] == {"max_text_bytes": 65536}
    resp = await harness.get("/webui/bootstrap")
    assert resp.headers["Cache-Control"] == "no-store"


async def test_bootstrap_is_localhost_only_without_secret(harness: Harness) -> None:
    resp = await harness.get("/webui/bootstrap", {"Host": "example.com"})
    assert resp.status == 403
    assert await resp.text() == "bootstrap is localhost-only"
    resp = await harness.get("/webui/bootstrap", {"X-Forwarded-For": "192.168.1.5"})
    assert resp.status == 403


@pytest.mark.ws_config({"tokenIssueSecret": "s3cret"})
async def test_bootstrap_with_secret_allows_remote_browser(harness: Harness) -> None:
    assert (await harness.get("/webui/bootstrap")).status == 401
    assert (await harness.get("/webui/bootstrap", {"X-Nanobot-Auth": "nope"})).status == 401
    body = await harness.bootstrap({"X-Nanobot-Auth": "s3cret", "Host": "box.lan:8765", "X-Forwarded-Proto": "https"})
    assert body["ws_url"] == "wss://box.lan:8765/"
    assert "api_token" in body


@pytest.mark.ws_config({"publicWsUrl": "wss://chat.example.com/"})
async def test_bootstrap_prefers_public_ws_url(harness: Harness) -> None:
    assert (await harness.bootstrap())["ws_url"] == "wss://chat.example.com/"


async def test_handshake_requires_a_one_time_token(harness: Harness) -> None:
    with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
        await harness.http.ws_connect(harness.base + "/")
    assert exc.value.status == 401
    token = (await harness.bootstrap())["token"]
    client = await harness.connect(f"?token={token}&client_id=me")
    assert client.ready["event"] == "ready" and client.ready["client_id"] == "me"
    assert "upload" not in client.ready
    with pytest.raises(aiohttp.WSServerHandshakeError):
        await harness.http.ws_connect(f"{harness.base}/?token={token}")  # consumed
    await client.ws.close()


async def test_bootstrap_token_marks_trusted_webui_connection(harness: Harness) -> None:
    client = await harness.connect()
    await asyncio.sleep(0.05)
    assert [c.trusted_webui for c in harness.channel._connections] == [True]
    await client.ws.close()


@pytest.mark.ws_config({"token": "static"})
async def test_static_token_opens_an_untrusted_connection(harness: Harness) -> None:
    client = await harness.connect("?token=static", trusted=False)
    frames = await client.request(request_id="r1", action="session.delete", payload={"key": "websocket:x"})
    assert frames == [
        {"event": "webui_response", "request_id": "r1", "ok": False, "error": {"status": 403, "message": "access_denied"}}
    ]
    await client.ws.close()


@pytest.mark.ws_config({"websocketRequiresToken": False})
async def test_no_token_needed_when_not_required(harness: Harness) -> None:
    client = await harness.connect(trusted=False)
    assert client.ready["event"] == "ready"
    await client.ws.close()


@pytest.mark.ws_config({"allowFrom": ["alice"], "websocketRequiresToken": False})
async def test_allow_from_rejects_unauthorized_client_id(harness: Harness) -> None:
    with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
        await harness.http.ws_connect(f"{harness.base}/?client_id=mallory")
    assert exc.value.status == 403
    client = await harness.connect("?client_id=alice", trusted=False)
    assert client.ready["client_id"] == "alice"
    await client.ws.close()


@pytest.mark.ws_config({"tokenIssuePath": "/auth/token", "tokenIssueSecret": "s"})
async def test_token_issue_route_requires_secret_then_token_opens_socket(harness: Harness) -> None:
    assert (await harness.get("/auth/token")).status == 401
    resp = await harness.get("/auth/token", {"Authorization": "Bearer s"})
    body = await resp.json()
    assert body["expires_in"] == 300
    client = await harness.connect(f"?token={body['token']}", trusted=False)
    assert client.ready["event"] == "ready"
    await asyncio.sleep(0.05)
    assert [c.trusted_webui for c in harness.channel._connections] == [False]
    await client.ws.close()


async def test_terminal_probe_and_protocol_check(harness: Harness) -> None:
    resp = await harness.get("/webui/terminal")
    body = await resp.json()
    assert body["protocolVersion"] == 1 and body["gatewayId"] == harness.channel.tokens.instance_id
    token = (await harness.bootstrap())["token"]
    with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
        await harness.http.ws_connect(f"{harness.base}/?token={token}&terminal_protocol=2")
    assert exc.value.status == 409


# -- envelopes ------------------------------------------------------------------------------


async def test_unknown_type_and_invalid_attach(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "bogus"})
    assert await client.recv() == {"event": "error", "detail": "unknown type: 'bogus'"}
    await client.send({"type": "attach", "chat_id": "bad id!"})
    assert await client.recv() == {"event": "error", "detail": "invalid chat_id"}
    await client.send({"type": "fork_chat"})
    assert (await client.recv())["detail"] == "unknown type: 'fork_chat'"
    await client.ws.close()


async def test_new_chat_attaches_with_workspace_scope(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "new_chat"})
    attached, updated = await client.recv(), await client.recv()
    assert attached["event"] == "attached" and attached["model_preset"] is None
    assert updated["event"] == "session_updated" and updated["chat_id"] == attached["chat_id"]
    assert updated["workspace_scope"]["project_path"] == str(harness.channel.agent.workspace)
    assert updated["workspace_scope"]["access_mode"] == "full"
    # No session is created until a message arrives (nanobot behaves the same).
    assert harness.channel.sessions.list_sessions() == []
    await client.ws.close()


async def test_message_validation_errors_carry_chat_and_turn(harness: Harness) -> None:
    client = await harness.connect()
    cases = [
        ({"chat_id": "bad id!", "content": "x"}, {"event": "error", "detail": "invalid chat_id"}),
        ({"chat_id": "c1", "turn_id": "t"}, {"event": "error", "detail": "missing content", "chat_id": "c1", "turn_id": "t"}),
        ({"chat_id": "c1", "turn_id": "t", "content": "  "}, {"event": "error", "detail": "missing content", "chat_id": "c1", "turn_id": "t"}),
        (
            {"chat_id": "c1", "turn_id": "t", "content": "x" * 70_000},
            {"event": "error", "detail": "message_rejected", "reason": "text_too_large", "chat_id": "c1", "turn_id": "t"},
        ),
        (
            {"chat_id": "c1", "turn_id": "t", "content": "x", "media": "no"},
            {"event": "error", "detail": "attachment_rejected", "reason": "malformed", "chat_id": "c1", "turn_id": "t"},
        ),
        (
            {"chat_id": "c1", "turn_id": "t", "content": "x", "media": [{"reference": "r"}]},
            {"event": "error", "detail": "attachment_rejected", "reason": "Attachments are not supported", "chat_id": "c1", "turn_id": "t"},
        ),
    ]
    for envelope, expected in cases:
        await client.send({"type": "message", **envelope})
        assert await client.recv() == expected
    assert harness.inbound() == []
    await client.ws.close()


async def test_webui_message_is_dispatched_recorded_and_acknowledged(harness: Harness) -> None:
    client = await harness.connect("client_id=me")
    await client.send({"type": "message", "chat_id": "chat-a", "content": "hello", "turn_id": "t1", "webui": True})
    accepted = await client.recv()
    assert accepted["event"] == "message_accepted"
    assert accepted["turn_id"] == "t1" and accepted["starts_turn"] is True and accepted["active_turn_id"] == "t1"
    assert isinstance(accepted["started_at"], float)
    [msg] = harness.inbound()
    assert (msg.channel, msg.chat_id, msg.sender_id, msg.content) == ("websocket", "chat-a", "me", "hello")
    assert msg.session_key == "websocket:chat-a"
    assert msg.metadata["webui"] is True and msg.metadata["webui_turn_id"] == "t1"
    assert msg.metadata["_wants_stream"] is True
    assert msg.metadata["workspace_scope"] == {"project_path": str(harness.channel.agent.workspace), "access_mode": "full"}
    owner = msg.metadata["_websocket_turn_owner"]
    assert harness.channel.turns.owner_is_registered("chat-a", owner, "t1")
    facts, _ = harness.channel.history.read_facts("websocket:chat-a")
    assert len(facts) == 1
    await client.ws.close()


async def test_duplicate_turn_is_acknowledged_but_not_dispatched_twice(harness: Harness) -> None:
    client = await harness.connect()
    envelope = {"type": "message", "chat_id": "chat-a", "content": "hello", "turn_id": "t1", "webui": True}
    await client.send(envelope)
    await client.recv()
    await client.send(envelope)
    assert await client.recv() == {"event": "message_accepted", "chat_id": "chat-a", "turn_id": "t1", "starts_turn": False}
    assert len(harness.inbound()) == 1
    await client.ws.close()


async def test_user_messages_fan_out_to_other_clients_on_the_same_chat(harness: Harness) -> None:
    a = await harness.connect()
    b = await harness.connect()
    await b.send({"type": "attach", "chat_id": "chat-a"})
    assert (await b.recv())["event"] == "attached"
    await a.send({"type": "message", "chat_id": "chat-a", "content": "hello", "turn_id": "t1", "webui": True})
    assert (await a.recv())["event"] == "message_accepted"
    echo = await b.recv()
    assert echo["event"] == "user_message" and echo["text"] == "hello" and echo["starts_turn"] is True
    assert echo["turn_id"] == "t1" and echo["active_turn_id"] == "t1"
    assert await a.recv_all() == []  # never echoed to the sender
    await a.ws.close()
    await b.ws.close()


async def test_second_message_while_running_does_not_start_a_turn(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "message", "chat_id": "c", "content": "one", "turn_id": "t1", "webui": True})
    await client.recv()
    await client.send({"type": "message", "chat_id": "c", "content": "two", "turn_id": "t2", "webui": True})
    frames = await client.recv_all()
    # Hydration replays the running turn first (nanobot does the same).
    assert frames[0]["event"] == "goal_status" and frames[0]["turn_id"] == "t1"
    second = frames[-1]
    assert second["event"] == "message_accepted"
    assert second["starts_turn"] is False and second["active_turn_id"] == "t1"
    await client.ws.close()


async def test_side_channel_command_does_not_register_queued_turn(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "message", "chat_id": "c", "content": "/help", "turn_id": "t1", "webui": True})
    accepted = await client.recv()
    assert accepted["starts_turn"] is False
    assert harness.channel.turns.wall_started_at("c") is None
    await client.ws.close()


async def test_plain_message_does_not_mark_webui_or_ack(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "message", "chat_id": "c", "content": "plain"})
    assert await client.recv_all() == []
    [msg] = harness.inbound()
    assert "webui" not in msg.metadata
    assert harness.channel.history.read_facts("websocket:c")[0] == []
    await client.ws.close()


async def test_legacy_text_frame_goes_to_default_chat(harness: Harness) -> None:
    client = await harness.connect()
    await client.send("hi there")
    await asyncio.sleep(0.1)
    [msg] = harness.inbound()
    assert msg.chat_id == client.ready["chat_id"] and msg.content == "hi there"
    await client.ws.close()


async def test_attach_hydrates_running_turn(harness: Harness) -> None:
    harness.channel.turns.mark_running("c", "owner", "t7", 123.0)
    client = await harness.connect()
    await client.send({"type": "attach", "chat_id": "c"})
    attached, running = await client.recv(), await client.recv()
    assert attached == {"event": "attached", "chat_id": "c", "model_preset": None}
    assert running == {"event": "goal_status", "chat_id": "c", "status": "running", "started_at": 123.0, "turn_id": "t7"}
    await client.ws.close()


async def test_set_sidebar_state_uses_registered_mutation_and_broadcasts(harness: Harness) -> None:
    saved: list[dict[str, Any]] = []

    async def sidebar_update(payload: dict[str, Any]) -> web.Response:
        saved.append(payload["state"])
        return mutation_json({"schema_version": 1, **payload["state"]})

    harness.channel.context.mutations["sidebar.update"] = sidebar_update
    a, b = await harness.connect(), await harness.connect()
    await a.send({"type": "set_sidebar_state", "state": {"pinned_keys": ["k"]}})
    expected = {"event": "sidebar_state_updated", "state": {"schema_version": 1, "pinned_keys": ["k"]}}
    assert await a.recv() == expected and await b.recv() == expected
    await a.send({"type": "set_sidebar_state", "state": []})
    assert await a.recv() == {"event": "error", "detail": "invalid_sidebar_state"}
    assert saved == [{"pinned_keys": ["k"]}]
    await a.ws.close()
    await b.ws.close()


# -- webui_request ---------------------------------------------------------------------------


async def test_webui_request_validation(harness: Harness) -> None:
    client = await harness.connect()
    assert await client.request(request_id="bad id!", action="x", payload={}) == [
        {"event": "error", "detail": "invalid webui request_id"}
    ]

    def err(rid: str, status: int, message: str) -> list[dict[str, Any]]:
        return [{"event": "webui_response", "request_id": rid, "ok": False, "error": {"status": status, "message": message}}]

    assert await client.request(request_id="r1", action="Bad", payload={}) == err("r1", 400, "invalid WebUI mutation action")
    assert await client.request(request_id="r2", action="a.b", payload=[]) == err(
        "r2", 400, "WebUI mutation payload must be an object"
    )
    assert await client.request(request_id="r3", action="settings.agent.update", payload={}) == err(
        "r3", 404, "unknown WebUI mutation action"
    )
    assert await client.request(request_id="r4", action="session.delete", payload={}) == err("r4", 400, "missing session key")
    assert await client.request(request_id="r5", action="session.delete", payload={"key": "cli:direct"}) == err(
        "r5", 404, "session not found"
    )
    assert await client.request(request_id="r6", action="session.delete", payload={"key": "bad key!"}) == err(
        "r6", 400, "invalid session key"
    )
    await client.ws.close()


async def test_session_delete_removes_session_history_and_turn(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "message", "chat_id": "gone", "content": "hello", "turn_id": "t1", "webui": True})
    await client.recv()
    sessions = harness.channel.sessions
    session = sessions.get_or_create("websocket:gone")
    session.add_message("user", "hello")
    sessions.save(session)
    assert harness.channel.history.exists("websocket:gone")
    frames = await client.request(request_id="d1", action="session.delete", payload={"key": "websocket:gone"})
    assert frames == [{"event": "webui_response", "request_id": "d1", "ok": True, "result": {"deleted": True}}]
    assert not harness.channel.history.exists("websocket:gone")
    assert sessions.get_existing("websocket:gone") is None
    assert harness.channel.turns.wall_started_at("gone") is None
    # Replay answers the cached result; a different payload under the same id conflicts.
    assert await client.request(request_id="d1", action="session.delete", payload={"key": "websocket:gone"}) == frames
    conflict = await client.request(request_id="d1", action="session.delete", payload={"key": "websocket:x"})
    assert conflict[0]["error"]["status"] == 409
    fresh = await client.request(request_id="d2", action="session.delete", payload={"key": "websocket:gone"})
    assert fresh[0]["result"] == {"deleted": False}
    await client.ws.close()


async def test_http_mutation_paths_require_the_socket(harness: Harness) -> None:
    for path in ("/api/sessions/websocket%3Ax/delete", "/api/webui/sidebar-state/update"):
        resp = await harness.get(path)
        assert resp.status == 405
        assert await resp.text() == "WebUI mutations require an authenticated WebSocket"


# -- outbound ---------------------------------------------------------------------------------


async def test_send_streams_to_subscribers_and_persists_completed_segment(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "attach", "chat_id": "c"})
    await client.recv()
    meta = {"webui": True, "webui_turn_id": "t1"}
    ch = harness.channel
    await ch.send(out("c", StreamDeltaEvent(stream_id="s:0"), "Hel", **meta))
    await ch.send(out("c", StreamDeltaEvent(stream_id="s:0"), "lo", **meta))
    await ch.send(out("c", StreamEndEvent(stream_id="s:0"), **meta))
    frames = await client.recv_all()
    assert [f["event"] for f in frames] == ["delta", "delta", "stream_end"]
    assert [f["turn_seq"] for f in frames] == [1, 2, 3]
    assert "text" not in frames[2]
    facts, _ = ch.history.read_facts("websocket:c")
    from nanoclaw.webui.history import record_from_fact

    [record] = [record_from_fact(f, "websocket:c") for f in facts]
    assert record["event"] == "stream_end" and record["text"] == "Hello"
    await client.ws.close()


async def test_send_persists_without_subscribers(harness: Harness) -> None:
    await harness.channel.send(out("nobody", content="offline reply", webui=True, webui_turn_id="t"))
    facts, _ = harness.channel.history.read_facts("websocket:nobody")
    assert len(facts) == 1


async def test_progress_tool_hint_and_flags(harness: Harness) -> None:
    client = await harness.connect()
    await client.send({"type": "attach", "chat_id": "c"})
    await client.recv()
    await harness.channel.send(out("c", ProgressEvent(tool_hint=True), "read_file(a)"))
    [hint] = await client.recv_all()
    assert hint["kind"] == "tool_hint" and hint["text"] == "read_file(a)"
    harness.channel.send_tool_hints = False
    await harness.channel.send(out("c", ProgressEvent(tool_hint=True), "read_file(b)"))
    assert await client.recv_all() == []
    await client.ws.close()


async def test_turn_end_releases_owner_and_announces_session_update_to_all(harness: Harness) -> None:
    a, b = await harness.connect(), await harness.connect()
    await a.send({"type": "attach", "chat_id": "c"})
    await a.recv()
    ch = harness.channel
    ch.turns.mark_running("c", "o1", "t1", 1.0)
    meta = {"webui": True, "webui_turn_id": "t1", "_websocket_turn_owner": "o1"}
    await ch.send(out("c", TurnEndEvent(latency_ms=12, goal_state={"active": False}), **meta))
    end, updated = await a.recv(), await a.recv()
    assert end["event"] == "turn_end" and end["latency_ms"] == 12 and end["turn_phase"] == "complete"
    assert updated == {"event": "session_updated", "chat_id": "c", "scope": "thread"}
    assert await b.recv() == updated  # audience "all"
    assert ch.turns.wall_started_at("c") is None
    await a.ws.close()
    await b.ws.close()


async def test_turn_end_keeps_registry_when_transcript_persistence_fails(harness: Harness, monkeypatch) -> None:
    ch = harness.channel
    ch.turns.mark_running("c", "o1", "t1", 1.0)

    async def failing_append(chat_id, event, *, metadata):
        return False

    monkeypatch.setattr(ch.recorder, "append", failing_append)
    meta = {"webui": True, "webui_turn_id": "t1", "_websocket_turn_owner": "o1"}
    await ch.send(out("c", content="answer", **meta))
    assert ch.turns.transcript_persistence_failed("c", "o1")
    await ch.send(out("c", TurnEndEvent(goal_state={"active": False}), **meta))
    await ch.send(out("c", GoalStatusEvent(status="idle"), **meta))
    assert ch.turns.wall_started_at("c") == 1.0  # kept for HTTP recovery


async def test_idle_goal_status_releases_owner(harness: Harness) -> None:
    ch = harness.channel
    ch.turns.mark_running("c", "o1", "t1", 1.0)
    await ch.send(out("c", GoalStatusEvent(status="idle"), _websocket_turn_owner="o1"))
    assert ch.turns.wall_started_at("c") is None


async def test_send_session_updated_reaches_every_connection(harness: Harness) -> None:
    a, b = await harness.connect(), await harness.connect()
    await harness.channel.send(out("other", SessionUpdatedEvent(scope="metadata")))
    expected = {"event": "session_updated", "chat_id": "other", "scope": "metadata"}
    assert await a.recv() == expected and await b.recv() == expected
    await a.ws.close()
    await b.ws.close()


async def test_running_lifecycle_registers_turn_and_publishes_goal_status(harness: Harness) -> None:
    ch = harness.channel
    ctx = RuntimeEventContext(
        channel="websocket",
        chat_id="c",
        session_key="websocket:c",
        metadata={"webui": True, "webui_turn_id": "t1", "_websocket_turn_owner": "o1"},
    )
    await ch.bus.publish(TurnRunStatusChanged(context=ctx, status="running", started_at=50.0))
    assert ch.turns.turn_id("c") == "t1" and ch.turns.wall_started_at("c") == 50.0
    msg = ch.bus.outbound.get_nowait()
    assert isinstance(msg.event, GoalStatusEvent) and msg.event.started_at == 50.0
    assert msg.metadata["_websocket_turn_owner"] == "o1"
    other = RuntimeEventContext(channel="cli", chat_id="direct", session_key="cli:direct")
    await ch.bus.publish(TurnRunStatusChanged(context=other, status="running"))
    assert ch.bus.outbound.empty()


async def test_running_lifecycle_without_owner_mints_one_for_the_turn(harness: Harness) -> None:
    ch = harness.channel
    ctx = RuntimeEventContext(channel="websocket", chat_id="c", session_key="websocket:c", metadata={})
    await ch.bus.publish(TurnRunStatusChanged(context=ctx, status="running", started_at=5.0))
    running = ch.bus.outbound.get_nowait()
    owner = running.metadata["_websocket_turn_owner"]
    await ch.bus.publish(TurnRunStatusChanged(context=ctx, status="idle"))
    idle = ch.bus.outbound.get_nowait()
    assert idle.metadata["_websocket_turn_owner"] == owner
    await ch.send(idle)
    assert ch.turns.wall_started_at("c") is None


# -- static files and routing (nanobot ws_http.py _serve_static) ------------------------------


async def test_static_index_assets_and_spa_fallback(harness: Harness) -> None:
    resp = await harness.get("/")
    assert resp.status == 200 and resp.headers["Cache-Control"] == "no-cache"
    assert resp.headers["Content-Type"] == "text/html; charset=utf-8"
    resp = await harness.get("/assets/app-1a2b.js", {"Accept-Encoding": "identity"})
    assert resp.headers["Cache-Control"] == "public, max-age=31536000, immutable"
    assert resp.headers["Content-Type"] == "application/javascript; charset=utf-8"
    assert resp.headers["Vary"] == "Accept-Encoding" and "Content-Encoding" not in resp.headers
    resp = await harness.get("/assets/app-1a2b.js", {"Accept-Encoding": "gzip, br"})
    assert resp.headers["Content-Encoding"] == "gzip"
    assert gzip.decompress(await resp.read()).startswith(b"console.log")
    resp = await harness.get("/logo.png")
    assert resp.headers["Content-Type"] == "image/png" and "Vary" not in resp.headers
    resp = await harness.get("/settings/deep/link")
    assert resp.status == 200 and await resp.read() == b"<html>app</html>"


async def test_static_rejects_traversal_and_api_misses_are_404(harness: Harness) -> None:
    resp = await harness.get("/%2e%2e/secret.txt")
    assert resp.status in (403, 200) and await resp.read() != b"secret"
    resp = await harness.get("/api/nope")
    assert resp.status == 404 and await resp.text() == "API route not found"


async def test_without_build_unknown_paths_are_404(tmp_path) -> None:
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    agent = AgentLoop(FakeProvider([]), ws_dir, sessions=SessionManager(ws_dir, sessions_root=tmp_path / "s"))
    channel = WebSocketChannel(WebSocketConfig(enabled=True, port=0), agent.bus, agent=agent)
    assert channel.static_dist_path is None
    task = asyncio.create_task(channel.start())
    async with asyncio.timeout(5):
        while not channel.is_running:
            await asyncio.sleep(0.01)
    async with aiohttp.ClientSession() as s:
        async with s.get(f"http://127.0.0.1:{channel.bound_port}/") as resp:
            assert resp.status == 404 and await resp.text() == "Not Found"
    await channel.stop()
    await asyncio.wait_for(task, 5)


def test_webui_dist_from_environment(tmp_path, monkeypatch) -> None:
    from nanoclaw.webui.gateway_http import resolve_webui_dist

    dist = tmp_path / "d"
    dist.mkdir()
    assert resolve_webui_dist("", {"NANOCLAW_WEBUI_DIST": str(dist)}) == dist.resolve()
    assert resolve_webui_dist(str(dist), {}) == dist.resolve()
    assert resolve_webui_dist("", {}) is None
    assert resolve_webui_dist(str(tmp_path / "missing"), {}) is None


# -- lifecycle ----------------------------------------------------------------------------------


async def test_stop_closes_connections_and_is_idempotent(harness: Harness) -> None:
    client = await harness.connect()
    await harness.channel.stop()
    msg = await asyncio.wait_for(client.ws.receive(), 3)
    assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING)
    await harness.channel.stop()


def test_turn_registry_keeps_owners_in_admission_order() -> None:
    reg = WebsocketTurnRegistry()
    first = reg.register_queued_if_idle("c", "t1")
    assert first is not None and reg.register_queued_if_idle("c", "t2") is None
    reg.mark_running("c", "o2", "t2", 2.0)
    assert reg.turn_id("c") == "t2"
    assert reg.clear_if_current("c", "o2")
    assert reg.turn_id("c") == "t1"
    assert not reg.clear_if_current("c", "nope")
    reg.mark_transcript_persistence_failed("c", first)
    assert not reg.clear_if_current("c", first, preserve_persistence_failure=True)
    assert reg.clear_if_current("c", first)
    assert reg.wall_started_at("c") is None and reg.active_chat_ids() == []
