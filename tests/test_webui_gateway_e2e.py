"""The web UI against a running ``nanoclaw gateway`` (layer 23), in process.

A gateway with a FakeProvider and ``channels.websocket`` enabled on a free
port: bootstrap over HTTP, open the socket with the one-time token, send a
``message`` envelope and read the turn as the browser does (message_accepted,
goal_status, streamed deltas, stream_end, turn_end, idle), then read the
chat's thread back from history.
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest
from rich.console import Console

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools.mcp import MCPProvider
from nanoclaw.cli.gateway import Gateway
from nanoclaw.config import Config
from nanoclaw.providers import FakeProvider
from nanoclaw.providers.fake import ScriptedStream
from nanoclaw.session import SessionManager
from nanoclaw.webui.session_identity import webui_session_key
from nanoclaw.webui.transcript import build_webui_thread_payload


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NoMCP(MCPProvider):
    def __init__(self) -> None:
        super().__init__({}, MagicMock())

    async def connect(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


async def _until(predicate, timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def _recv_until(ws: aiohttp.ClientWebSocketResponse, event: str, frames: list[dict[str, Any]], timeout: float = 10):
    async with asyncio.timeout(timeout):
        while True:
            msg = await ws.receive()
            assert msg.type == aiohttp.WSMsgType.TEXT, msg
            data = json.loads(msg.data)
            frames.append(data)
            if data.get("event") == event:
                return data


@pytest.fixture
async def gateway(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    provider = FakeProvider([ScriptedStream(content_deltas=["Hel", "lo ", "there"]), "Greeting chat"])
    agent = AgentLoop(provider, ws_dir, sessions=SessionManager(ws_dir, sessions_root=tmp_path / "sessions"))
    port = free_port()
    config = Config.model_validate({"channels": {"websocket": {"enabled": True, "port": port}}, "gateway": {"heartbeat": {"enabled": False}}})
    gw = Gateway(config, agent, _NoMCP(), console=Console(file=None, quiet=True), health_server_enabled=False)
    run = asyncio.create_task(gw.run())
    channel = gw.channels.get_channel("websocket")
    assert channel is not None
    await _until(lambda: channel.is_running)
    try:
        yield gw, channel, port, provider
    finally:
        gw.shutdown_event.set()
        await asyncio.wait_for(run, 20)


async def test_full_turn_over_the_socket(gateway) -> None:
    gw, channel, port, _provider = gateway
    base = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession() as http:
        async with http.get(f"{base}/webui/bootstrap") as resp:
            assert resp.status == 200
            boot = await resp.json()
        assert boot["ws_path"] == "/"
        assert boot["terminal"]["webui"]["capabilities"] == ["webui.core.v1"]
        # The loop names no model (an injected provider): the configured one.
        assert boot["model_name"] == "gpt-4o-mini"
        async with http.ws_connect(f"ws://127.0.0.1:{port}/?token={boot['token']}") as ws:
            ready = json.loads((await ws.receive()).data)
            assert ready["event"] == "ready"
            await ws.send_json({"type": "new_chat"})
            frames: list[dict[str, Any]] = []
            attached = await _recv_until(ws, "attached", frames)
            chat_id = attached["chat_id"]
            await _recv_until(ws, "session_updated", frames)
            await ws.send_json({"type": "message", "chat_id": chat_id, "content": "hi", "turn_id": "t1", "webui": True})
            frames = []
            await _recv_until(ws, "turn_end", frames)
            await _recv_until(ws, "goal_status", frames)
            events = [f["event"] for f in frames]
            assert "message_accepted" in events
            accepted = next(f for f in frames if f["event"] == "message_accepted")
            assert accepted["turn_id"] == "t1" and accepted["starts_turn"] is True
            running = next(f for f in frames if f["event"] == "goal_status")
            assert running["status"] == "running" and running["turn_id"] == "t1"
            deltas = [f["text"] for f in frames if f["event"] == "delta"]
            assert "".join(deltas) == "Hello there"
            assert any(f["event"] == "stream_end" for f in frames)
            end = next(f for f in frames if f["event"] == "turn_end")
            assert end["turn_id"] == "t1" and end["turn_phase"] == "complete"
            assert frames[-1] == {"event": "goal_status", "chat_id": chat_id, "status": "idle", "turn_id": "t1"}
            # The title arrives after the turn.
            await _recv_until(ws, "session_updated", frames)
        assert channel.turns.wall_started_at(chat_id) is None
        # The REST side (nanoclaw/webui/rest_api.py) replays the same turn.
        auth = {"Authorization": f"Bearer {boot['api_token']}"}
        async with http.get(f"{base}/api/sessions/websocket%3A{chat_id}/webui-thread", headers=auth) as resp:
            assert resp.status == 200
            thread = await resp.json()
        assert [e["event"] for e in thread["events"]][0] == "user_message"
        assert "t1" in thread.get("completed_turn_ids", [])
        async with http.get(f"{base}/api/sessions", headers=auth) as resp:
            rows = (await resp.json())["sessions"]
        assert [r["key"] for r in rows] == [f"websocket:{chat_id}"] and rows[0]["title"] == "Greeting chat"

    key = webui_session_key(chat_id)
    session = channel.sessions.get_existing(key)
    assert session is not None and session.metadata.get("webui") is True
    assert session.metadata.get("title") == "Greeting chat"
    payload = build_webui_thread_payload(key, sessions=channel.sessions, store=channel.history)
    assert payload is not None
    kinds = [e.get("event") for e in payload["events"]]
    assert kinds[0] == "user_message"
    assert "stream_end" in kinds and "turn_end" in kinds
    stream_end = next(e for e in payload["events"] if e.get("event") == "stream_end")
    assert stream_end["text"] == "Hello there"
    assert "t1" in payload.get("completed_turn_ids", [])


async def test_health_reports_websocket_running(gateway) -> None:
    gw, _channel, _port, _provider = gateway
    ready, body = gw.health()
    assert ready is True
    assert body["websocket"] == "running"
