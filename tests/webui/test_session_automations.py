"""The web UI automation routes and mutations (``nanoclaw/webui/rest_api.py``,
``nanoclaw/webui/session_automations.py``) through aiohttp's test client.

Ported from nanobot channels/websocket/tests/test_websocket_http_routes.py
(c4dac64): ``test_session_automations_route_filters_by_webui_session``,
``test_session_automations_route_ignores_unified_owner`` and
``test_webui_automations_route_lists_all_jobs_and_allows_user_actions``
(without its ``automation.update`` steps, which NanoClaw does not serve).
Left out: the local trigger tests (NanoClaw has no local triggers). The
``session.delete`` automation check is in tests/test_websocket_channel.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nanoclaw.config.schema import Config
from nanoclaw.cron.service import CronService
from nanoclaw.cron.types import CronJob, CronPayload, CronSchedule
from nanoclaw.webui.context import WebUIContext
from nanoclaw.webui.rest_api import register_api_fallback, register_rest_routes
from nanoclaw.webui.session_automations import serialize_automation_jobs

TOKEN = "nbwt_test"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
HOURLY = CronSchedule(kind="every", every_ms=3_600_000)
DAILY = CronSchedule(kind="every", every_ms=86_400_000)


def _check_auth(request: web.Request) -> bool:
    return request.headers.get("Authorization") == f"Bearer {TOKEN}"


class _Loop:
    def __init__(self) -> None:
        self.pending: dict[str, set[str]] = {}

    def pending_cron_job_ids_for_session(self, session_key: str) -> set[str]:
        return set(self.pending.get(session_key, set()))


@pytest.fixture
def cron(tmp_path: Path) -> CronService:
    return CronService(tmp_path / "cron" / "jobs.json")


@pytest.fixture
def ctx(history, cron: CronService) -> WebUIContext:
    return WebUIContext(
        config=Config(),
        sessions=history.sessions,
        history=history.store,
        loop=_Loop(),  # type: ignore[arg-type]
        workspace=history.sessions.workspace,
        check_auth=_check_auth,
        channel=SimpleNamespace(turns=None),  # type: ignore[arg-type]
        cron=cron,
    )


@pytest.fixture
async def client(ctx: WebUIContext):
    app = web.Application()
    register_rest_routes(app, ctx)
    register_api_fallback(app)
    async with TestClient(TestServer(app)) as c:
        yield c


def _seed_session(ctx: WebUIContext, key: str, text: str = "hi") -> None:
    session = ctx.sessions.get_or_create(key)
    session.add_message("user", text)
    ctx.sessions.save(session)


def _bound(cron: CronService, name: str, chat: str, *, schedule: CronSchedule = HOURLY, **kw: Any) -> CronJob:
    return cron.add_job(
        name=name,
        schedule=schedule,
        message=kw.pop("message", f"{name} message"),
        session_key=f"websocket:{chat}",
        origin_channel="websocket",
        origin_chat_id=chat,
        **kw,
    )


async def _mutate(ctx: WebUIContext, action: str, payload: dict[str, Any]) -> tuple[int, Any]:
    response = await ctx.mutations[action](payload)
    text = response.text or ""
    if response.content_type == "application/json":
        return response.status, json.loads(text)
    return response.status, text


# --- GET /api/sessions/<key>/automations -----------------------------------------


async def test_session_automations_route_filters_by_webui_session(client, ctx, cron) -> None:
    pending_job_id = ""
    for name, message, to in (
        ("Morning check", "Check the project status", "abc"),
        ("Other session", "Do not show", "other"),
    ):
        job = _bound(cron, name, to, message=message)
        if name == "Morning check":
            pending_job_id = job.id
    cron.add_job(
        name="Legacy same target",
        schedule=HOURLY,
        message="Legacy job should be migrated",
        deliver=True,
        channel="websocket",
        to="abc",
        session_key="websocket:abc",
    )
    cron.register_system_job(
        CronJob(
            id="heartbeat",
            name="heartbeat",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            payload=CronPayload(kind="system_event"),
        )
    )
    _seed_session(ctx, "websocket:abc")
    ctx.loop.pending["websocket:abc"] = {pending_job_id}  # type: ignore[union-attr]

    deny = await client.get("/api/sessions/websocket:abc/automations")
    assert deny.status == 401
    assert await deny.text() == "Unauthorized"

    resp = await client.get("/api/sessions/websocket%3Aabc/automations", headers=AUTH)
    assert resp.status == 200
    body = await resp.json()
    assert [job["name"] for job in body["jobs"]] == ["Morning check", "Legacy same target"]
    job = body["jobs"][0]
    assert job["schedule"]["kind"] == "every"
    assert job["schedule"]["every_ms"] == 3_600_000
    assert job["payload"]["message"] == "Check the project status"
    assert job["state"]["pending"] is True
    assert body["jobs"][1]["state"]["pending"] is False
    # The detailed shape (include_details=True).
    assert job["protected"] is False
    assert job["payload"]["kind"] == "agent_turn"
    assert job["origin"] == {
        "session_key": "websocket:abc",
        "channel": "websocket",
        "chat_id": "abc",
        "title": "",
        "preview": "hi",
    }
    assert isinstance(job["chat_binding_revision"], str)
    assert job["state"]["run_history"] == []


async def test_session_automations_route_ignores_unified_owner(client, ctx, cron) -> None:
    cron.add_job(
        name="Unified check",
        schedule=HOURLY,
        message="Check the shared session",
        session_key="unified:default",
        origin_channel="websocket",
        origin_chat_id="abc",
    )
    _bound(cron, "Visible chat job", "abc")
    _seed_session(ctx, "websocket:abc")

    resp = await client.get("/api/sessions/websocket%3Aabc/automations", headers=AUTH)
    assert resp.status == 200
    assert [job["name"] for job in (await resp.json())["jobs"]] == ["Visible chat job"]

    resp = await client.get("/api/sessions/websocket%3Aother/automations", headers=AUTH)
    assert resp.status == 200
    assert (await resp.json())["jobs"] == []


async def test_session_automations_route_key_errors(client, ctx) -> None:
    resp = await client.get("/api/sessions/cli%3Adirect/automations", headers=AUTH)
    assert (resp.status, await resp.text()) == (404, "session not found")
    resp = await client.get("/api/sessions/bad%20key/automations", headers=AUTH)
    assert (resp.status, await resp.text()) == (400, "invalid session key")


async def test_session_automations_route_without_cron_lists_nothing(client, ctx) -> None:
    ctx.cron = None
    resp = await client.get("/api/sessions/websocket%3Aabc/automations", headers=AUTH)
    assert resp.status == 200
    assert await resp.json() == {"jobs": []}


async def test_session_automations_shapes_for_each_schedule_kind(client, ctx, cron) -> None:
    at_ms = 4_102_444_800_000  # 2100-01-01
    _bound(cron, "Every", "abc")
    cron_job = _bound(cron, "Cron", "abc", schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="UTC"))
    _bound(cron, "At", "abc", schedule=CronSchedule(kind="at", at_ms=at_ms), delete_after_run=True)
    cron.enable_job(cron_job.id, enabled=False)
    resp = await client.get("/api/sessions/websocket%3Aabc/automations", headers=AUTH)
    jobs = {job["name"]: job for job in (await resp.json())["jobs"]}
    assert jobs["Cron"]["enabled"] is False
    assert jobs["Cron"]["schedule"] == {"kind": "cron", "at_ms": None, "every_ms": None, "expr": "0 9 * * *", "tz": "UTC"}
    assert jobs["Cron"]["state"]["next_run_at_ms"] is None
    assert jobs["At"]["schedule"]["at_ms"] == at_ms
    assert jobs["At"]["delete_after_run"] is True
    assert jobs["At"]["state"]["next_run_at_ms"] == at_ms
    assert jobs["Every"]["enabled"] is True
    assert jobs["Every"]["origin"]["preview"] == ""  # no saved session


def test_serialize_automation_jobs_summary_shape(cron) -> None:
    job = _bound(cron, "Daily check", "doomed", schedule=DAILY, message="Check the repo")
    [row] = serialize_automation_jobs([job])
    assert row == {
        "id": job.id,
        "name": "Daily check",
        "enabled": True,
        "schedule": {"kind": "every", "at_ms": None, "every_ms": 86_400_000, "expr": None, "tz": None},
        "payload": {"message": "Check the repo"},
        "state": {"next_run_at_ms": job.state.next_run_at_ms, "last_status": None, "pending": False},
    }


# --- GET /api/webui/automations and automation.* mutations ------------------------


async def test_webui_automations_route_lists_all_jobs_and_allows_user_actions(client, ctx, cron) -> None:
    user_job = _bound(cron, "Daily repo check", "abc", schedule=DAILY, message="Check the repo status")
    incomplete_job = cron.add_job(
        name="english-quiz",
        schedule=HOURLY,
        message="Practice English",
        session_key="unified:default",
    )
    external_job = cron.add_job(
        name="WeChat quiz",
        schedule=HOURLY,
        message="Send a quiz",
        session_key="weixin:wx-chat",
        origin_channel="weixin",
        origin_chat_id="wx-chat",
    )
    cron.register_system_job(
        CronJob(
            id="heartbeat",
            name="heartbeat",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            payload=CronPayload(kind="system_event"),
        )
    )
    _seed_session(ctx, "websocket:abc")
    _seed_session(ctx, "weixin:wx-chat", "Scheduled cron job triggered")
    ctx.loop.pending["websocket:abc"] = {user_job.id}  # type: ignore[union-attr]

    deny = await client.get("/api/webui/automations")
    assert deny.status == 401

    resp = await client.get("/api/webui/automations", headers=AUTH)
    assert resp.status == 200
    text = await resp.text()
    assert "wx-chat" not in text
    assert "unified:default" not in text
    by_id = {job["id"]: job for job in json.loads(text)["jobs"]}
    assert by_id[user_job.id]["protected"] is False
    assert by_id[user_job.id]["state"]["pending"] is True
    assert by_id[user_job.id]["state"]["run_history"] == []
    assert by_id[user_job.id]["origin"]["session_key"] == "websocket:abc"
    assert by_id[user_job.id]["origin"]["preview"] == "hi"
    assert "session_key" not in by_id[incomplete_job.id]["payload"]
    assert by_id[incomplete_job.id]["origin"] is None
    assert "session_key" not in by_id[external_job.id]["payload"]
    assert by_id[external_job.id]["origin"]["channel"] == "weixin"
    assert "session_key" not in by_id[external_job.id]["origin"]
    assert "chat_id" not in by_id[external_job.id]["origin"]
    assert by_id[external_job.id]["origin"]["preview"] == ""
    assert by_id["heartbeat"]["protected"] is True

    status, body = await _mutate(ctx, "automation.disable", {"id": user_job.id})
    assert status == 200
    assert {job["id"]: job for job in body["jobs"]}[user_job.id]["enabled"] is False

    assert (await _mutate(ctx, "automation.run", {"id": user_job.id})) == (409, "automation is disabled")
    status, text = await _mutate(ctx, "automation.run", {"id": incomplete_job.id})
    assert status == 409 and "no linked chat" in text
    status, text = await _mutate(ctx, "automation.enable", {"id": incomplete_job.id})
    assert status == 409 and "no linked chat" in text

    for action in ("delete", "disable", "run"):
        assert (await _mutate(ctx, f"automation.{action}", {"id": "heartbeat"})) == (
            403, "system automation is protected"
        )

    status, body = await _mutate(ctx, "automation.enable", {"id": user_job.id})
    assert status == 200
    assert {job["id"]: job for job in body["jobs"]}[user_job.id]["enabled"] is True

    status, body = await _mutate(ctx, "automation.delete", {"id": user_job.id})
    assert status == 200
    assert user_job.id not in {job["id"] for job in body["jobs"]}
    assert "heartbeat" in {job["id"] for job in body["jobs"]}


async def test_automation_action_errors(client, ctx, cron) -> None:
    assert (await _mutate(ctx, "automation.enable", {})) == (400, "missing automation id")
    assert (await _mutate(ctx, "automation.enable", {"id": "  "})) == (400, "missing automation id")
    assert (await _mutate(ctx, "automation.delete", {"id": "nope"})) == (404, "automation not found")
    assert "automation.update" not in ctx.mutations
    ctx.cron = None
    assert (await _mutate(ctx, "automation.enable", {"id": "x"})) == (503, "automation service unavailable")


async def test_automation_run_starts_the_job(client, ctx, cron) -> None:
    ran: list[str] = []

    async def on_job(job: CronJob) -> str:
        ran.append(job.id)
        return "done"

    cron.on_job = on_job
    job = _bound(cron, "Run me", "abc")
    status, body = await _mutate(ctx, "automation.run", {"job_id": job.id})
    assert status == 200
    assert [row["id"] for row in body["jobs"]] == [job.id]
    async with asyncio.timeout(3):
        while not ran:
            await asyncio.sleep(0.01)
    assert ran == [job.id]


async def test_automation_mutation_paths_answer_405_over_http(client) -> None:
    for action in ("enable", "disable", "delete", "run", "update", "change-chat"):
        resp = await client.get(f"/api/webui/automations/{action}", headers=AUTH)
        assert resp.status == 405
        assert await resp.text() == "WebUI mutations require an authenticated WebSocket"
