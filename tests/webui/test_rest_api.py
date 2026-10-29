"""The web UI REST routes (``nanoclaw/webui/rest_api.py``) through aiohttp's test client.

Ported from nanobot channels/websocket/tests/test_websocket_http_routes.py
(the session, thread, commands, workspaces and sidebar routes),
tests/utils/test_webui_sidebar_state.py and tests/webui/test_session_context.py.
Left out: bootstrap/token tests (task 23.1's server), session delete (the
channel's ``session.delete`` mutation), automations (tests/webui/test_session_automations.py), skills management,
settings mutations, remote instances and media, which NanoClaw does not serve;
pairing (route and mutations) is in tests/pairing/test_pairing_webui.py.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import nanoclaw.webui.rest_api as rest_api
from nanoclaw.config.schema import Config
from nanoclaw.webui.context import WebUIContext
from nanoclaw.webui.rest_api import register_api_fallback, register_rest_routes
from webui_history_fixture import history  # noqa: F401

TOKEN = "nbwt_test"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _check_auth(request: web.Request) -> bool:
    return request.headers.get("Authorization") == f"Bearer {TOKEN}" or request.query.get("token") == TOKEN


class _Turns:
    def __init__(self) -> None:
        self.started: dict[str, float] = {}
        self.ids: dict[str, str] = {}

    def wall_started_at(self, chat_id: str) -> float | None:
        return self.started.get(chat_id)

    def turn_id(self, chat_id: str) -> str | None:
        return self.ids.get(chat_id)


@pytest.fixture
def ctx(history, tmp_path: Path) -> WebUIContext:  # noqa: F811
    config = Config.model_validate({
        "agents": {"defaults": {"model": "anthropic/claude-x", "timezone": "UTC"}},
        "modelPresets": {"fast": {"model": "openai/gpt-fast", "provider": "openai", "maxTokens": 1024}},
    })
    return WebUIContext(
        config=config,
        sessions=history.sessions,
        history=history.store,
        loop=None,
        workspace=history.sessions.workspace,
        check_auth=_check_auth,
        channel=SimpleNamespace(turns=_Turns()),
    )


@pytest.fixture
async def client(ctx: WebUIContext):
    app = web.Application()
    register_rest_routes(app, ctx)
    register_api_fallback(app)
    async with TestClient(TestServer(app)) as c:
        yield c


def _seed(ctx: WebUIContext, key: str, *texts: str) -> None:
    session = ctx.sessions.get_or_create(key)
    for index, text in enumerate(texts):
        session.add_message("user" if index % 2 == 0 else "assistant", text)
    ctx.sessions.save(session)


def _thread_url(key: str, suffix: str = "") -> str:
    from urllib.parse import quote

    return f"/api/sessions/{quote(key, safe='')}/webui-thread{suffix}"


# --- auth -------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "/api/sessions",
    "/api/sessions/websocket%3Aabc/webui-thread",
    "/api/sessions/websocket%3Aabc/webui-thread/prompts",
    "/api/sessions/websocket%3Aabc/webui-thread/trace-detail?ref=x",
    "/api/sessions/websocket%3Aabc/context",
    "/api/commands",
    "/api/workspaces",
    "/api/webui/sidebar-state",
    "/api/settings",
    "/api/settings/pairing",
    "/api/webui/skills",
])
async def test_every_route_requires_the_api_token(client, path):
    deny = await client.get(path)
    assert deny.status == 401
    assert await deny.text() == "Unauthorized"
    assert deny.headers["Content-Type"].startswith("text/plain")
    wrong = await client.get(path, headers={"Authorization": "Bearer nope"})
    assert wrong.status == 401


async def test_query_token_is_accepted_like_bearer(client):
    resp = await client.get(f"/api/commands?token={TOKEN}")
    assert resp.status == 200


# --- sessions ---------------------------------------------------------------------


async def test_sessions_list_requires_bearer_token(client, ctx):
    _seed(ctx, "websocket:abc", "hi", "hello back")
    listing = await client.get("/api/sessions", headers=AUTH)
    assert listing.status == 200
    assert listing.headers["Content-Type"] == "application/json; charset=utf-8"
    assert listing.headers["Vary"] == "Accept-Encoding"
    sessions = (await listing.json())["sessions"]
    assert [s["key"] for s in sessions] == ["websocket:abc"]
    # Filesystem paths and internal index fields never reach the wire.
    assert all("path" not in s for s in sessions)
    assert all(not any(k.startswith("_") for k in s) for s in sessions)
    assert sessions[0]["preview"] == "hi"


async def test_sessions_list_only_returns_websocket_sessions(client, ctx):
    for key in ("cli:direct", "slack:C123", "websocket:alpha", "websocket:beta"):
        _seed(ctx, key, f"hi from {key}")
    body = await (await client.get("/api/sessions", headers=AUTH)).json()
    rows = {row["key"]: row for row in body["sessions"]}
    assert set(rows) == {"websocket:alpha", "websocket:beta"}
    scope = rows["websocket:alpha"]["workspace_scope"]
    assert scope["project_path"] == str(ctx.workspace)
    assert scope["access_mode"] == "full"
    assert "recovery_state" not in rows["websocket:alpha"]


async def test_sessions_list_reports_running_turns(client, ctx):
    _seed(ctx, "websocket:busy", "working")
    _seed(ctx, "websocket:idle", "resting")
    ctx.channel.turns.started["busy"] = 1234.5
    rows = {r["key"]: r for r in (await (await client.get("/api/sessions", headers=AUTH)).json())["sessions"]}
    assert rows["websocket:busy"]["run_started_at"] == 1234.5
    assert "run_started_at" not in rows["websocket:idle"]


async def test_sessions_list_negotiates_gzip_across_repeated_headers(client, ctx):
    for index in range(80):
        _seed(ctx, f"websocket:gzip-{index:03d}", f"hi from {index}")
    resp = await client.get("/api/sessions", headers=[
        ("Authorization", f"Bearer {TOKEN}"),
        ("Accept-Encoding", "identity;q=0"),
        ("Accept-Encoding", "gzip"),
    ])
    assert resp.status == 200
    assert resp.headers["Content-Encoding"] == "gzip"
    assert resp.headers["Vary"] == "Accept-Encoding"
    assert len((await resp.json())["sessions"]) == 80


async def test_sessions_list_trailing_slash(client):
    assert (await client.get("/api/sessions/", headers=AUTH)).status == 200


# --- webui-thread -----------------------------------------------------------------


async def test_thread_restores_display_without_model_messages(client, ctx, history):  # noqa: F811
    key = "websocket:restored-history"
    # nanobot's display-history append creates the session row; NanoClaw lists
    # session files, so the chat's (message-less) session is saved first.
    ctx.sessions.save(ctx.sessions.get_or_create(key))
    history.append(key, {"event": "user", "chat_id": "restored-history", "text": "original question"})
    history.append(key, {"event": "message", "chat_id": "restored-history", "text": "original answer"})
    listing = await (await client.get("/api/sessions", headers=AUTH)).json()
    assert [row["key"] for row in listing["sessions"]] == [key]
    assert listing["sessions"][0]["preview"] == "original question"
    thread = await client.get(_thread_url(key), headers=AUTH)
    assert thread.status == 200
    body = await thread.json()
    assert body["projection"] == "events"
    assert body["schemaVersion"] == 3
    assert "messages" not in body
    assert [e["text"] for e in body["events"] if e["event"] in {"user_message", "message", "stream_end"}] == [
        "original question", "original answer",
    ]
    assert body["workspace_scope"]["project_path"] == str(ctx.workspace)


async def test_thread_backfills_from_session_messages(client, ctx):
    _seed(ctx, "websocket:backfill", "question", "answer")
    body = await (await client.get(_thread_url("websocket:backfill"), headers=AUTH)).json()
    texts = [e.get("text") for e in body["events"] if e["event"] in {"user_message", "message"}]
    assert texts == ["question", "answer"]


async def test_thread_404_for_unknown_chat_and_non_websocket_keys(client):
    missing = await client.get(_thread_url("websocket:nothing"), headers=AUTH)
    assert missing.status == 404
    assert await missing.text() == "webui thread not found"
    other = await client.get(_thread_url("cli:direct"), headers=AUTH)
    assert other.status == 404
    assert await other.text() == "session not found"


async def test_thread_rejects_invalid_keys_and_queries(client):
    bad_key = await client.get("/api/sessions/web%20socket/webui-thread", headers=AUTH)
    assert bad_key.status == 400
    assert await bad_key.text() == "invalid session key"
    slash_key = await client.get("/api/sessions/websocket%3Aa%2Fb/webui-thread", headers=AUTH)
    assert slash_key.status in {400, 404}
    for query, message in (
        ("?limit=abc", "invalid limit"),
        ("?direction=oldest", "invalid direction"),
        ("?after=garbage", "invalid navigation cursor"),
    ):
        resp = await client.get(_thread_url("websocket:x", query), headers=AUTH)
        assert resp.status == 400
        assert await resp.text() == message


async def test_thread_complete_transcript_with_paging_params(client, history):  # noqa: F811
    key = "websocket:fast-thread"
    for event in (
        {"event": "user", "chat_id": "fast-thread", "text": "hi"},
        {"event": "message", "chat_id": "fast-thread", "text": "hello back"},
        {"event": "turn_end", "chat_id": "fast-thread"},
    ):
        history.append(key, event)
    resp = await client.get(_thread_url(key, "?limit=160&direction=latest"), headers=AUTH)
    assert resp.status == 200
    body = await resp.json()
    assert [e["text"] for e in body["events"] if "text" in e] == ["hi", "hello back"]
    assert body["page"]["has_more_before"] is False


async def test_thread_pages_older_turns_with_before_cursor(client, history):  # noqa: F811
    key = "websocket:paged"
    for index in range(5):
        history.append(key, {"event": "user", "chat_id": "paged", "text": f"q{index}"})
        history.append(key, {"event": "message", "chat_id": "paged", "text": f"a{index}"})
        history.append(key, {"event": "turn_end", "chat_id": "paged"})
    latest = await (await client.get(_thread_url(key, "?limit=2&direction=latest"), headers=AUTH)).json()
    assert latest["page"]["has_more_before"] is True
    cursor = latest["page"]["before_cursor"]
    older = await client.get(_thread_url(key, f"?limit=2&before={cursor}"), headers=AUTH)
    assert older.status == 200
    older_texts = [e["text"] for e in (await older.json())["events"] if e["event"] == "user_message"]
    latest_texts = [e["text"] for e in latest["events"] if e["event"] == "user_message"]
    assert older_texts and not set(older_texts) & set(latest_texts)


async def test_thread_negotiates_gzip_for_large_payloads(client, history):  # noqa: F811
    key = "websocket:gzip-thread"
    history.append(key, {"event": "user", "chat_id": "gzip-thread", "text": "compress me " * 1_000})
    url = _thread_url(key, "?limit=80&direction=latest")
    compressed = await client.get(url, headers={**AUTH, "Accept-Encoding": "br, gzip"}, auto_decompress=False)
    assert compressed.status == 200
    assert compressed.headers["Content-Encoding"] == "gzip"
    assert compressed.headers["Vary"] == "Accept-Encoding"
    raw = await compressed.read()
    body = json.loads(gzip.decompress(raw))
    assert body["events"][0]["text"].startswith("compress me")
    identity = await client.get(url, headers={**AUTH, "Accept-Encoding": "gzip;q=0, br"})
    assert "Content-Encoding" not in identity.headers
    assert await identity.json() == body
    unauthorized = await client.get(url, headers={"Accept-Encoding": "gzip"})
    assert unauthorized.status == 401
    assert "Content-Encoding" not in unauthorized.headers


async def test_thread_revalidates_and_loads_large_trace_details(client, history):  # noqa: F811
    key = "websocket:revalidated-thread"
    trace = f'exec({json.dumps({"command": "x" * 40_000})})'
    for event in (
        {"event": "user", "chat_id": "revalidated-thread", "text": "run it"},
        {"event": "message", "chat_id": "revalidated-thread", "kind": "progress", "text": trace},
        {"event": "message", "chat_id": "revalidated-thread", "text": "done"},
        {"event": "turn_end", "chat_id": "revalidated-thread"},
    ):
        history.append(key, event)
    url = _thread_url(key, "?limit=40&direction=latest")
    first = await client.get(url, headers=AUTH)
    body = await first.json()
    assert first.status == 200
    assert first.headers["Cache-Control"] == "no-store"
    assert first.headers["ETag"] == f'"{body["revision"]}"'
    trace_event = next(e for e in body["events"] if e.get("kind") == "progress")
    assert trace_event["text"] == "exec(…)"

    unchanged = await client.get(url, headers={**AUTH, "If-None-Match": first.headers["ETag"]})
    assert unchanged.status == 304
    assert await unchanged.read() == b""
    assert unchanged.headers["ETag"] == first.headers["ETag"]
    assert unchanged.headers["Cache-Control"] == "no-store"
    weak = await client.get(url, headers={**AUTH, "If-None-Match": f'"other", W/{first.headers["ETag"]}'})
    assert weak.status == 304
    star = await client.get(url, headers={**AUTH, "If-None-Match": "*"})
    assert star.status == 304

    # A different page is a different representation.
    other_page = await client.get(_thread_url(key, "?limit=41&direction=latest"),
                                  headers={**AUTH, "If-None-Match": first.headers["ETag"]})
    assert other_page.status == 200

    detail = await client.get(
        _thread_url(key, f"/trace-detail?ref={trace_event['trace_detail']['ref']}"), headers=AUTH,
    )
    assert detail.status == 200
    assert detail.headers["Cache-Control"] == "no-store"
    detail_body = await detail.json()
    assert detail_body["message_id"] == trace_event["projection_id"]
    assert detail_body["events"][0]["text"] == trace

    history.append(key, {"event": "user", "chat_id": "revalidated-thread", "text": "again"})
    changed = await client.get(url, headers={**AUTH, "If-None-Match": first.headers["ETag"]})
    assert changed.status == 200
    assert changed.headers["ETag"] != first.headers["ETag"]


async def test_thread_etag_changes_with_the_active_turn(client, ctx, history):  # noqa: F811
    key = "websocket:live"
    history.append(key, {"event": "user", "chat_id": "live", "text": "go"})
    first = await client.get(_thread_url(key), headers=AUTH)
    ctx.channel.turns.started["live"] = 99.0
    ctx.channel.turns.ids["live"] = "turn-1"
    second = await client.get(_thread_url(key), headers={**AUTH, "If-None-Match": first.headers["ETag"]})
    assert second.status == 200
    assert (await second.json())["active_turn_id"] == "turn-1"


async def test_trace_detail_errors(client, history):  # noqa: F811
    key = "websocket:t"
    history.append(key, {"event": "user", "chat_id": "t", "text": "x"})
    missing_ref = await client.get(_thread_url(key, "/trace-detail"), headers=AUTH)
    assert missing_ref.status == 400
    assert await missing_ref.text() == "missing trace detail ref"
    unknown = await client.get(_thread_url(key, "/trace-detail?ref=nope"), headers=AUTH)
    assert unknown.status == 404
    assert await unknown.text() == "trace detail not found"


async def test_thread_omits_validator_when_transcript_changes_during_replay(client, history, monkeypatch):  # noqa: F811
    key = "websocket:changing-thread"
    history.append(key, {"event": "user", "chat_id": "changing-thread", "text": "hello"})
    monkeypatch.setattr(rest_api, "webui_transcript_revision", lambda *_a, **_k: "before-replay")
    resp = await client.get(_thread_url(key, "?limit=40&direction=latest"), headers=AUTH)
    assert resp.status == 200
    assert "ETag" not in resp.headers
    assert "revision" not in await resp.json()


# --- prompt index -----------------------------------------------------------------


async def test_prompt_index_revalidates_and_sends_deltas(client, history):  # noqa: F811
    key = "websocket:prompts"
    history.append(key, {"event": "user", "chat_id": "prompts", "text": "first question"})
    history.append(key, {"event": "message", "chat_id": "prompts", "text": "first answer"})
    history.append(key, {"event": "turn_end", "chat_id": "prompts"})
    first = await client.get(_thread_url(key, "/prompts"), headers=AUTH)
    assert first.status == 200
    assert first.headers["Cache-Control"] == "no-store"
    body = await first.json()
    assert body["replaceFrom"] == 0
    assert [(p["preview"], p["answerPreview"]) for p in body["prompts"]] == [("first question", "first answer")]
    assert first.headers["ETag"] == f'"{body["revision"]}"'

    same = await client.get(_thread_url(key, "/prompts"), headers={**AUTH, "If-None-Match": first.headers["ETag"]})
    assert same.status == 304

    history.append(key, {"event": "user", "chat_id": "prompts", "text": "second question"})
    delta = await (await client.get(_thread_url(key, f"/prompts?since={body['revision']}"), headers=AUTH)).json()
    assert delta["replaceFrom"] == 1
    assert [p["preview"] for p in delta["prompts"]] == ["second question"]


# --- context ----------------------------------------------------------------------


async def test_session_context_reports_replay_and_summary(client, ctx):
    session = ctx.sessions.get_or_create("websocket:ctx")
    for role, text in (("user", "old"), ("assistant", "old answer"), ("user", "new"), ("assistant", "new answer")):
        session.add_message(role, text)
    session.last_archived = 2
    session.metadata["_last_summary"] = {"text": "  user asked old things  ", "last_active": "2026-01-01T00:00:00"}
    from nanoclaw.providers.base import LLMUsage

    session.metadata["_last_usage"] = LLMUsage(input_tokens=10, output_tokens=5).to_dict()
    ctx.sessions.save(session)
    body = await (await client.get("/api/sessions/websocket%3Actx/context", headers=AUTH)).json()
    assert body["schema_version"] == 1
    assert body["session_key"] == "websocket:ctx"
    assert body["total_messages"] == 4
    assert body["archived_messages"] == 2
    assert body["replay_messages"] == 2
    assert body["archived_summary"] == "user asked old things"
    assert body["archived_summary_at"] == "2026-01-01T00:00:00"
    assert body["estimated_session_tokens"] == body["estimated_replay_tokens"] + body["estimated_summary_tokens"]
    assert body["last_usage"]["prompt_tokens"] == 10


async def test_session_context_404(client):
    resp = await client.get("/api/sessions/websocket%3Amissing/context", headers=AUTH)
    assert resp.status == 404
    assert await resp.text() == "session not found"


# --- commands, workspaces, settings, pairing, skills --------------------------------


async def test_commands_lists_builtin_palette(client):
    from nanoclaw.command.builtin import BUILTIN_COMMAND_SPECS

    body = await (await client.get("/api/commands", headers=AUTH)).json()
    commands = body["commands"]
    assert [c["command"] for c in commands] == [s.command for s in BUILTIN_COMMAND_SPECS]
    lifecycles = {"side_channel", "finalize_active_turn", "stop_active_turn", "agent_turn", "agent_turn_with_args"}
    for row in commands:
        assert {"command", "title", "description", "icon", "lifecycle", "accepts_args"} <= set(row)
        assert row["lifecycle"] in lifecycles


async def test_workspaces_is_the_fixed_default_scope(client, ctx):
    body = await (await client.get("/api/workspaces", headers=AUTH)).json()
    assert body["schema_version"] == 1
    assert body["default_access_mode"] == "default"
    assert body["default_scope"]["project_path"] == str(ctx.workspace)
    assert body["controls"] == {
        "can_change_project": False, "can_browse_directories": False,
        "can_resolve_project": False, "can_manage_favorites": False,
    }
    assert body["favorite_projects"] == [] and body["recent_projects"] == []
    assert set(body["host"]) == {"name", "platform"}


async def test_workspaces_uses_the_loops_project_scope(client, ctx, tmp_path):
    from nanoclaw.security.workspace_access import build_workspace_scope

    project = tmp_path / "project"
    project.mkdir()
    ctx.loop = SimpleNamespace(workspace_scope=build_workspace_scope(project, "restricted"))
    body = await (await client.get("/api/workspaces", headers=AUTH)).json()
    assert body["default_scope"]["project_path"] == str(project.resolve())
    assert body["default_scope"]["access_mode"] == "restricted"


async def test_settings_has_model_and_presets(client):
    body = await (await client.get("/api/settings", headers=AUTH)).json()
    assert body["agent"]["model"] == "anthropic/claude-x"
    assert body["agent"]["model_preset"] == "default"
    presets = {p["name"]: p for p in body["model_presets"]}
    assert presets["default"]["active"] is True and presets["default"]["is_default"] is True
    assert presets["fast"]["model"] == "openai/gpt-fast"
    assert presets["fast"]["max_tokens"] == 1024
    assert presets["fast"]["active"] is False
    # Every field the frontend's SettingsPayload requires is present.
    for field in ("agent", "model_presets", "model_call_order", "model_call_order_editable", "providers",
                  "web_search", "web", "image_generation", "runtime", "advanced", "requires_restart"):
        assert field in body
    assert body["surface"] == body["runtime_surface"] == "browser"
    assert body["apply_state"] == {"status": "idle", "sections": []}


async def test_settings_follows_the_active_preset(client, ctx):
    ctx.config.agents.defaults.model_preset = "fast"
    body = await (await client.get("/api/settings", headers=AUTH)).json()
    assert body["agent"]["model"] == "openai/gpt-fast"
    assert body["agent"]["model_preset"] == "fast"
    assert {p["name"]: p["active"] for p in body["model_presets"]} == {"default": False, "fast": True}


def _needs_setup(settings: dict, scoped_preset: str | None = None) -> bool:
    """The frontend's gate on sending (webui/src/components/thread/model-preset.ts
    ``toModelBadgeInfo``): the composer shows "Choose your AI" and refuses to
    send while this is true. ``inferProviderFromModelName`` is not ported: it
    only matters when a preset names no provider at all."""
    presets = settings.get("model_presets") or []
    if scoped_preset:
        preset = next((p for p in presets if p["name"] == scoped_preset), None)
    else:
        configured = settings["agent"].get("model_preset") or "default"
        preset = next((p for p in presets if p["name"] == configured), None) or next(
            (p for p in presets if p["active"]), None)
    model = (preset or {}).get("model") if scoped_preset else (settings["agent"].get("model") or None)
    raw = (preset or {}).get("provider") or (None if scoped_preset else settings["agent"].get("provider")) or None
    if raw == "auto":
        provider = (preset or {}).get("resolved_provider") or (
            None if scoped_preset else settings["agent"].get("resolved_provider")) or None
    else:
        provider = raw
    row = next((r for r in settings.get("providers") or [] if r["name"] == provider), None) if provider else None
    return not model or not provider or row is None or not row["configured"]


async def _settings_for(client, ctx, raw_config: dict, env: dict | None = None) -> dict:
    from nanoclaw.config.loader import apply_env_overrides

    config = Config.model_validate(raw_config)
    ctx.config = apply_env_overrides(config, env or {}) if env is not None else config
    resp = await client.get("/api/settings", headers=AUTH)
    assert resp.status == 200
    return await resp.json()


async def test_settings_active_provider_row_is_configured(client, ctx):
    body = await _settings_for(client, ctx, {
        "agents": {"defaults": {"model": "anthropic/claude-sonnet-4"}},
        "providers": {"anthropic": {"apiKey": "sk-ant-0123456789abcdef"}},
    })
    assert body["agent"]["provider"] == body["agent"]["resolved_provider"] == "anthropic"
    rows = {row["name"]: row for row in body["providers"]}
    assert rows["anthropic"]["configured"] is True
    assert rows["anthropic"]["label"] == "Anthropic"
    assert rows["anthropic"]["api_key_hint"] == "sk-a••••cdef"
    assert "sk-ant-0123456789abcdef" not in json.dumps(body)
    assert rows["openai"]["configured"] is False
    assert rows["custom"]["is_custom"] is False and rows["custom"]["model_catalog"] == "custom"
    assert not _needs_setup(body)


async def test_settings_env_only_key_configures_custom(client, ctx):
    """Phase 1's ``NANOCLAW_API_KEY`` alone: ``custom`` gets the key (no base,
    NanoClaw then calls api.openai.com); the UI must still let the user send."""
    body = await _settings_for(client, ctx, {}, env={"NANOCLAW_API_KEY": "sk-env-0123456789", "NANOCLAW_MODEL": "gpt-4o"})
    assert body["agent"]["model"] == "gpt-4o"
    assert body["agent"]["resolved_provider"] == "custom"
    rows = {row["name"]: row for row in body["providers"]}
    assert rows["custom"]["configured"] is True
    assert rows["custom"]["api_base"] is None
    assert not _needs_setup(body)


async def test_settings_env_key_and_base(client, ctx):
    body = await _settings_for(client, ctx, {}, env={
        "NANOCLAW_API_KEY": "k", "NANOCLAW_API_BASE": "http://localhost:9999/v1", "NANOCLAW_MODEL": "my-model"})
    rows = {row["name"]: row for row in body["providers"]}
    assert body["agent"]["resolved_provider"] == "custom"
    assert rows["custom"]["configured"] is True
    assert rows["custom"]["api_base"] == "http://localhost:9999/v1"
    assert rows["custom"]["api_key_hint"] == "••••"
    assert not _needs_setup(body)


async def test_settings_without_any_provider_needs_setup(client, ctx):
    body = await _settings_for(client, ctx, {"agents": {"defaults": {"model": "gpt-4o"}}})
    assert body["agent"]["resolved_provider"] == "auto"
    assert not any(row["configured"] for row in body["providers"])
    assert _needs_setup(body)


async def test_settings_forced_provider_and_scoped_preset(client, ctx):
    body = await _settings_for(client, ctx, {
        "agents": {"defaults": {"model": "deepseek-chat", "provider": "deepseek"}},
        "providers": {"deepseek": {"apiKey": "sk-deep-123456789"}},
        "modelPresets": {
            "fast": {"model": "gpt-4o-mini", "provider": "openai"},
            "local": {"model": "llama3.2", "provider": "ollama"},
        },
    })
    assert body["agent"]["provider"] == "deepseek"
    assert not _needs_setup(body)
    # A chat pinned to a preset whose provider has no key asks for setup, as in nanobot.
    assert _needs_setup(body, "fast")
    assert _needs_setup(body, "local")  # Ollama without an apiBase is not configured


async def test_settings_custom_provider_entries_and_redaction(client, ctx):
    body = await _settings_for(client, ctx, {
        "agents": {"defaults": {"model": "companyProxy/gpt-4"}},
        "providers": {"companyProxy": {"apiBase": "https://proxy.example/v1", "displayName": "Company",
                                       "extraHeaders": {"Authorization": "Bearer s3cret", "X-Team": "ai"}}},
    })
    rows = {row["name"]: row for row in body["providers"]}
    row = rows["companyProxy"]
    assert row["is_custom"] is True and row["label"] == "Company" and row["configured"] is True
    assert row["extra_headers"] == {"Authorization": "••••••••", "X-Team": "ai"}
    assert "thinking_style" in row["advanced_fields"]
    assert body["agent"]["resolved_provider"] == "companyProxy"
    assert not _needs_setup(body)
    assert list(rows)[-1] == "companyProxy"  # custom entries after the registry rows


async def test_pairing_is_empty(client):
    resp = await client.get("/api/settings/pairing", headers=AUTH)
    assert resp.status == 200
    assert await resp.json() == {"requests": []}


async def test_skills_lists_builtin_skills_without_paths(client):
    body = await (await client.get("/api/webui/skills", headers=AUTH)).json()
    names = {s["name"] for s in body["skills"]}
    assert "weather" in names
    for skill in body["skills"]:
        assert set(skill) == {"name", "description", "source", "enabled", "deletable", "available",
                              "unavailable_reason"}
        assert "/" not in json.dumps(skill.get("description", ""))[:1]


# --- sidebar state ----------------------------------------------------------------


async def test_sidebar_state_default_and_save_round_trip(client, ctx):
    initial = await client.get("/api/webui/sidebar-state", headers=AUTH)
    assert initial.status == 200
    assert await initial.json() == rest_api.default_webui_sidebar_state()

    payload = {
        "pinned_keys": ["websocket:sidebar"],
        "archived_keys": ["websocket:old"],
        "session_order": ["websocket:old", "websocket:sidebar"],
        "title_overrides": {"websocket:sidebar": "Pinned work"},
        "view": {"density": "compact", "show_archived": True},
    }
    updated = await ctx.mutations["sidebar.update"]({"state": payload})
    assert updated.status == 200
    body = json.loads(updated.body)
    assert body["pinned_keys"] == ["websocket:sidebar"]
    assert body["title_overrides"] == {"websocket:sidebar": "Pinned work"}
    assert body["view"] == {"density": "compact", "show_archived": True, "sort": "updated_desc"}
    assert isinstance(body["updated_at"], str)

    state_path = Path(ctx.history.directory) / "sidebar-state.json"
    assert json.loads(state_path.read_text(encoding="utf-8"))["pinned_keys"] == ["websocket:sidebar"]
    reread = await (await client.get("/api/webui/sidebar-state", headers=AUTH)).json()
    assert reread == body


async def test_sidebar_update_rejects_bad_payloads(client, ctx):
    missing = await ctx.mutations["sidebar.update"]({})
    assert missing.status == 400 and missing.text == "missing state"
    wrong = await ctx.mutations["sidebar.update"]({"state": []})
    assert wrong.status == 400 and wrong.text == "state must be an object"
    huge = await ctx.mutations["sidebar.update"]({"state": {"title_overrides": {
        f"websocket:{i}": "t" * 150 for i in range(2_000)}}})
    assert huge.status == 400 and huge.text == "sidebar state is too large"


async def test_sidebar_state_ignores_corrupt_or_oversized_files(client, ctx):
    path = Path(ctx.history.directory) / "sidebar-state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert await (await client.get("/api/webui/sidebar-state", headers=AUTH)).json() == \
        rest_api.default_webui_sidebar_state()
    path.write_text(json.dumps({"pinned_keys": ["a"], "x": "y" * (300 * 1024)}), encoding="utf-8")
    assert (await (await client.get("/api/webui/sidebar-state", headers=AUTH)).json())["pinned_keys"] == []


def test_sidebar_state_normalization():
    """nanobot tests/utils/test_webui_sidebar_state.py."""
    state = rest_api.normalize_webui_sidebar_state({
        "pinned_keys": [" a ", "a", "", 3, "b"],
        "tags_by_key": {"k": ["x" * 50, "y"] + [f"t{i}" for i in range(20)], "empty": []},
        "collapsed_groups": {"g": 1, "": True},
        "view": {"density": "huge", "sort": "manual", "show_archived": 1},
        "workbench": {"version": 1, "tabs": {
            "t1": {"paneKeys": ["p1", "p2", "p1"], "layoutPaneKeys": ["p2"], "layout": "grid",
                   "splitRatios": [0.0, 2, float("nan"), True], "title": " T "},
            "t2": {"paneKeys": ["p1"], "explicit": False},
            "t3": {"paneKeys": ["p3"], "explicit": True},
        }},
        "updated_at": 5,
    })
    assert state["pinned_keys"] == ["a", "b"]
    assert state["tags_by_key"]["k"][0] == "x" * 40 and len(state["tags_by_key"]["k"]) == 12
    assert "empty" not in state["tags_by_key"]
    assert state["collapsed_groups"] == {"g": True}
    assert state["view"] == {"density": "comfortable", "show_archived": True, "sort": "manual"}
    tabs = state["workbench"]["tabs"]
    assert tabs["t1"] == {"explicit": False, "title": "T", "paneKeys": ["p1", "p2"],
                          "layoutPaneKeys": ["p2", "p1"], "layout": "grid", "splitRatios": [0.05, 0.95]}
    assert "t2" not in tabs and tabs["t3"]["paneKeys"] == ["p3"]
    assert state["updated_at"] is None
    assert rest_api.normalize_webui_sidebar_state("bad") == rest_api.default_webui_sidebar_state()
    assert rest_api.normalize_webui_sidebar_state({"workbench": {"version": 2}})["workbench"] == \
        {"version": 1, "tabs": {}}


# --- mutation paths, unknown routes -----------------------------------------------


@pytest.mark.parametrize("path", [
    "/api/webui/sidebar-state/update",
    "/api/sessions/websocket%3Aabc/delete",
    "/api/workspaces/favorite",
    "/api/workspaces/resolve-project",
])
async def test_plain_http_mutations_answer_405(client, path):
    for method in ("GET", "POST"):
        resp = await client.request(method, path, headers=AUTH)
        assert resp.status == 405
        assert await resp.text() == "WebUI mutations require an authenticated WebSocket"


async def test_unknown_api_route_is_404_not_the_spa(client):
    resp = await client.get("/api/nope", headers=AUTH)
    assert resp.status == 404
    assert await resp.text() == "API route not found"
