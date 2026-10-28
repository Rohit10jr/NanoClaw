"""The web UI's REST routes (``GET /api/...``), for nanobot's React frontend.

Ported from nanobot/webui/ws_http.py (c4dac64): the session routes
(``_dispatch_session_routes`` :831, ``_handle_sessions_list`` :978,
``_handle_webui_thread_get`` :1193, ``_handle_prompt_index_get`` :1093,
``_handle_webui_trace_detail_get`` :1302, ``_handle_session_context_get``
:960), ``_handle_commands`` :1737, ``_handle_workspaces`` :1742, the sidebar
state routes :1976 and a read-only ``GET /api/settings`` (nanobot
settings_routes.py ``_handle_settings`` with settings_api.py
``settings_payload``). Same paths, query parameters, status codes, headers and
JSON shapes; errors are plain-text bodies (nanobot http_utils ``http_error``)
and every route answers ``401 Unauthorized`` without a valid API token.

nanobot runs mutations only through the socket's ``webui_request``; a plain
HTTP call to a mutation path answers 405 (ws_http.py:483). This module
registers the ``sidebar.update`` mutation in ``ctx.mutations`` (the channel
owns ``session.delete``) and answers 405 on the mutation paths it knows.

What differs from nanobot (deliberately):

- Workspace scope: NanoClaw has one project folder per run (``--project``),
  so every chat reports the loop's ``workspace_scope`` (or the agent
  workspace), ``/api/workspaces`` is that fixed scope with every project
  control off, and per-chat scope metadata is not read.
- ``/api/settings`` is read-only and small: ``agent``, ``model_presets`` and
  ``runtime`` come from NanoClaw's config; providers, usage, version and the
  other sections are nanobot's defaults or empty. The ``settings.*``
  mutations are not served. It reads ``ctx.config`` (nanobot reloads
  ``config.json`` on each request).
- Sidebar state lives at ``<webui history dir>/sidebar-state.json``
  (``~/.nanoclaw/webui/<workspace-id>/``), per workspace like NanoClaw's
  sessions; nanobot keeps one per data directory.
- No gateway-update maintenance (503) or slow-request diagnostics.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import platform
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from aiohttp import web
from loguru import logger

from nanoclaw.command.builtin import builtin_command_palette
from nanoclaw.webui.context import WebUIContext, default_workspace_scope_payload
from nanoclaw.webui.session_identity import is_webui_session_key
from nanoclaw.webui.session_list_index import webui_sessions_payload
from nanoclaw.webui.transcript import (
    build_webui_thread_payload,
    build_webui_trace_detail_response,
    decode_transcript_cursor,
    prompt_index_response,
    prompt_index_revision,
    read_session_messages,
    session_file_version,
    webui_thread_history,
    webui_transcript_revision,
)

__all__ = [
    "api_not_found",
    "register_api_fallback",
    "webui_skills_payload",
    "default_webui_sidebar_state",
    "normalize_webui_sidebar_state",
    "read_webui_sidebar_state",
    "register_rest_routes",
    "settings_payload",
    "webui_sidebar_state_path",
    "workspaces_payload",
    "write_webui_sidebar_state",
]

_NO_STORE_HEADERS = {"Cache-Control": "no-store"}
_JSON_GZIP_MIN_BYTES = 4 * 1024
_JSON_GZIP_LEVEL = 5
_API_KEY_RE = re.compile(r"^[A-Za-z0-9_:.-]{1,128}$")
_CTX_KEY = web.AppKey("nanoclaw_webui_context", WebUIContext)


# --- responses (nanobot webui/http_utils.py) ------------------------------------


def _accepts_gzip(value: str) -> bool:
    wildcard_quality: float | None = None
    for item in value.split(","):
        name, *params = (part.strip() for part in item.split(";"))
        quality = 1.0
        for param in params:
            key, separator, raw_value = param.partition("=")
            if separator and key.strip().lower() == "q":
                try:
                    quality = float(raw_value.strip())
                except ValueError:
                    quality = 0.0
                break
        if name.lower() == "gzip":
            return quality > 0
        if name == "*":
            wildcard_quality = quality
    return wildcard_quality is not None and wildcard_quality > 0


def _accept_encoding(request: web.Request) -> str:
    """Repeated ``Accept-Encoding`` headers combined (http_utils ``combined_list_header``)."""
    return ", ".join(v.strip() for v in request.headers.getall("Accept-Encoding", []) if v.strip())


def _json_response(
    data: Any,
    *,
    status: int = 200,
    accept_encoding: str | None = None,
    headers: dict[str, str] | None = None,
) -> web.Response:
    """http_utils ``http_json_response``: gzip at 4 KiB when the caller negotiates it."""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    out: dict[str, str] = {}
    if accept_encoding is not None:
        out["Vary"] = "Accept-Encoding"
        if len(body) >= _JSON_GZIP_MIN_BYTES and _accepts_gzip(accept_encoding):
            body = gzip.compress(body, compresslevel=_JSON_GZIP_LEVEL, mtime=0)
            out["Content-Encoding"] = "gzip"
    if headers:
        out.update(headers)
    return web.Response(status=status, body=body, headers=out,
                        content_type="application/json", charset="utf-8")


def _error(status: int, message: str) -> web.Response:
    """http_utils ``http_error``: a plain-text body."""
    return web.Response(status=status, text=message, content_type="text/plain", charset="utf-8")


def _not_modified(etag: str) -> web.Response:
    return web.Response(status=304, headers={**_NO_STORE_HEADERS, "ETag": etag})


def _unauthorized() -> web.Response:
    return _error(401, "Unauthorized")


def _quoted_etag(revision: str) -> str:
    return f'"{revision}"'


def _etag_matches(value: str, etag: str) -> bool:
    return any(candidate.strip().removeprefix("W/") in {"*", etag} for candidate in value.split(","))


def _if_none_match(request: web.Request) -> str:
    return (request.headers.get("If-None-Match") or "").strip()


# --- request helpers ------------------------------------------------------------


def _ctx(request: web.Request) -> WebUIContext:
    return request.app[_CTX_KEY]


def _authorized(request: web.Request) -> bool:
    try:
        return bool(_ctx(request).check_auth(request))
    except Exception:
        logger.exception("webui auth check failed")
        return False


def _raw_session_segment(request: web.Request) -> str:
    """The still-encoded ``<key>`` path segment (nanobot matches the raw path)."""
    path = urlsplit(request.raw_path).path
    match = re.match(r"^/api/sessions/([^/]+)/", path)
    return match.group(1) if match else request.match_info.get("key", "")


def _decode_api_key(raw_key: str) -> str | None:
    key = unquote(raw_key)
    return key if _API_KEY_RE.match(key) else None


def _query_first(request: web.Request, key: str) -> str | None:
    values = request.query.getall(key, [])
    return values[0] if values else None


def _session_key_or_error(request: web.Request) -> str | web.Response:
    """Auth, decode and the ``websocket:`` check every session route starts with."""
    if not _authorized(request):
        return _unauthorized()
    key = _decode_api_key(_raw_session_segment(request))
    if key is None:
        return _error(400, "invalid session key")
    if not is_webui_session_key(key):
        return _error(404, "session not found")
    if _ctx(request).sessions is None:
        return _error(503, "session manager unavailable")
    return key


def workspace_scope_payload(ctx: WebUIContext) -> dict[str, Any]:
    """The run's project scope: the loop's (``--project``), else the workspace."""
    scope = getattr(ctx.loop, "workspace_scope", None)
    if scope is not None and hasattr(scope, "payload"):
        return cast(dict[str, Any], scope.payload())
    restrict = bool(getattr(getattr(ctx.config, "tools", None), "restrict_to_workspace", False))
    return default_workspace_scope_payload(ctx.workspace, restrict_to_workspace=restrict)


def _active_turn(ctx: WebUIContext, chat_id: str) -> tuple[float | None, str | None]:
    turns = getattr(ctx.channel, "turns", None)
    if turns is None:
        return None, None
    try:
        return turns.wall_started_at(chat_id), turns.turn_id(chat_id)
    except Exception:
        logger.exception("webui active turn lookup failed")
        return None, None


def _gateway_instance(ctx: WebUIContext) -> str | None:
    for owner in (ctx.channel, getattr(ctx.channel, "tokens", None)):
        value = getattr(owner, "instance_id", None)
        if isinstance(value, str):
            return value
    value = ctx.extras.get("instance_id")
    return value if isinstance(value, str) else None


# --- session routes -------------------------------------------------------------


async def _handle_sessions_list(request: web.Request) -> web.Response:
    """GET /api/sessions (ws_http.py:978)."""
    if not _authorized(request):
        return _unauthorized()
    ctx = _ctx(request)
    if ctx.sessions is None:
        return _error(503, "session manager unavailable")
    scope = workspace_scope_payload(ctx)
    payload = webui_sessions_payload(
        ctx.sessions, ctx.history,
        run_started_at=lambda chat_id: _active_turn(ctx, chat_id)[0],
        workspace_scope=lambda _present, _raw: scope,
    )
    return _json_response(payload, accept_encoding=_accept_encoding(request))


async def _handle_webui_thread(request: web.Request) -> web.Response:
    """GET /api/sessions/<key>/webui-thread (ws_http.py:1193)."""
    key = _session_key_or_error(request)
    if isinstance(key, web.Response):
        return key
    ctx = _ctx(request)
    raw_limit = _query_first(request, "limit")
    limit: int | None = None
    if raw_limit is not None and raw_limit.strip():
        try:
            limit = int(raw_limit)
        except ValueError:
            return _error(400, "invalid limit")
    direction = _query_first(request, "direction")
    if direction is not None and direction not in {"latest"}:
        return _error(400, "invalid direction")
    before = _query_first(request, "before")
    after = _query_first(request, "after")
    if after is not None and (
        before is not None or direction is not None or decode_transcript_cursor(after) is None
    ):
        return _error(400, "invalid navigation cursor")

    chat_id = key.split(":", 1)[1]
    started_at, turn_id = _active_turn(ctx, chat_id)
    scope = workspace_scope_payload(ctx)
    variant: dict[str, Any] = {
        "active_turn_id": turn_id,
        "active_turn_started_at": started_at,
        "active_turn_transcript_persistence_failed": False,
        "before": before,
        "after": after,
        "direction": direction,
        "gateway_instance": _gateway_instance(ctx),
        "limit": limit,
        "workspace_scope": scope,
    }
    history = webui_thread_history(key, store=ctx.history, sessions=ctx.sessions,
                                   active=started_at is not None)
    initial = webui_transcript_revision(key, history=history, variant={
        **variant, "session_version": session_file_version(ctx.sessions, key),
    })
    if initial is not None and _etag_matches(_if_none_match(request), _quoted_etag(initial)):
        return _not_modified(_quoted_etag(initial))
    data = build_webui_thread_payload(
        key, store=ctx.history, sessions=ctx.sessions,
        active_turn_started_at=started_at, active_turn_id=turn_id,
        limit=limit, direction=direction, before=before, after=after,
        revision_variant=variant,
    )
    if data is None:
        return _error(404, "webui thread not found")
    data["workspace_scope"] = scope
    headers = dict(_NO_STORE_HEADERS)
    final = data.pop("revision", None)
    if final is not None and final == initial:
        headers["ETag"] = _quoted_etag(final)
        data["revision"] = final
    return _json_response(data, accept_encoding=_accept_encoding(request), headers=headers)


async def _handle_prompt_index(request: web.Request) -> web.Response:
    """GET /api/sessions/<key>/webui-thread/prompts[?since=<revision>] (ws_http.py:1093)."""
    key = _session_key_or_error(request)
    if isinstance(key, web.Response):
        return key
    ctx = _ctx(request)
    history = webui_thread_history(key, store=ctx.history, sessions=ctx.sessions)
    revision = prompt_index_revision(key, session_file_version(ctx.sessions, key), history=history)
    etag = _quoted_etag(revision) if revision else None
    if etag and _etag_matches(_if_none_match(request), etag):
        return _not_modified(etag)
    data = prompt_index_response(
        key, _query_first(request, "since"), history=history,
        session_messages_loader=lambda: read_session_messages(ctx.sessions, key),
        session_version_loader=lambda: session_file_version(ctx.sessions, key),
    )
    headers = dict(_NO_STORE_HEADERS)
    if data.get("revision"):
        headers["ETag"] = _quoted_etag(cast(str, data["revision"]))
    return _json_response(dict(data), accept_encoding=_accept_encoding(request), headers=headers)


async def _handle_trace_detail(request: web.Request) -> web.Response:
    """GET /api/sessions/<key>/webui-thread/trace-detail?ref=... (ws_http.py:1302)."""
    key = _session_key_or_error(request)
    if isinstance(key, web.Response):
        return key
    ctx = _ctx(request)
    detail_ref = _query_first(request, "ref")
    if not detail_ref:
        return _error(400, "missing trace detail ref")
    history = webui_thread_history(key, store=ctx.history, sessions=ctx.sessions)
    data = build_webui_trace_detail_response(key, detail_ref, history=history)
    if data is None:
        return _error(404, "trace detail not found")
    return _json_response(data, accept_encoding=_accept_encoding(request), headers=dict(_NO_STORE_HEADERS))


def session_context_payload(session: Any) -> dict[str, Any]:
    """nanobot webui/session_context.py: the session-owned part of the prompt."""
    from nanoclaw.agent.messages import truncate_text
    from nanoclaw.providers.base import LLMUsage
    from nanoclaw.utils.tokens import estimate_message_tokens

    replay = session.get_history(include_runtime_context=False)
    raw_summary = session.metadata.get("_last_summary")
    summary = summary_preview = ""
    summary_at: str | None = None
    if isinstance(raw_summary, dict):
        text, last_active = raw_summary.get("text"), raw_summary.get("last_active")
        if isinstance(text, str):
            summary = text.strip()
            summary_preview = truncate_text(summary, 4_000)
        if isinstance(last_active, str):
            summary_at = last_active
    replay_tokens = sum(estimate_message_tokens(message) for message in replay)
    summary_tokens = estimate_message_tokens({"role": "system", "content": summary}) if summary else 0
    usage = LLMUsage.from_dict(session.metadata.get("_last_usage"))
    return {
        "schema_version": 1,
        "session_key": session.key,
        "total_messages": len(session.messages),
        "archived_messages": min(session.last_archived, len(session.messages)),
        "replay_messages": len(replay),
        "estimated_replay_tokens": replay_tokens,
        "estimated_summary_tokens": summary_tokens,
        "estimated_session_tokens": replay_tokens + summary_tokens,
        "archived_summary": summary_preview or None,
        "archived_summary_at": summary_at,
        "last_usage": usage.to_turn_dict() if usage is not None else None,
    }


async def _handle_session_context(request: web.Request) -> web.Response:
    """GET /api/sessions/<key>/context (ws_http.py:960)."""
    key = _session_key_or_error(request)
    if isinstance(key, web.Response):
        return key
    session = _ctx(request).sessions.get_existing(key)
    if session is None:
        return _error(404, "session not found")
    return _json_response(session_context_payload(session))


# --- commands, workspaces -------------------------------------------------------


async def _handle_commands(request: web.Request) -> web.Response:
    """GET /api/commands (ws_http.py:1737)."""
    if not _authorized(request):
        return _unauthorized()
    return _json_response({"commands": builtin_command_palette()})


def workspaces_payload(ctx: WebUIContext) -> dict[str, Any]:
    """nanobot webui/workspaces.py ``workspaces_payload`` with NanoClaw's fixed scope."""
    return {
        "schema_version": 1,
        "default_access_mode": "default",
        "default_scope": workspace_scope_payload(ctx),
        "host": {"name": socket.gethostname(), "platform": platform.system()},
        "favorite_projects": [],
        "recent_projects": [],
        "controls": {
            "can_change_project": False,
            "can_browse_directories": False,
            "can_resolve_project": False,
            "can_manage_favorites": False,
        },
    }


async def _handle_workspaces(request: web.Request) -> web.Response:
    """GET /api/workspaces (ws_http.py:1742)."""
    if not _authorized(request):
        return _unauthorized()
    return _json_response(workspaces_payload(_ctx(request)))


# --- sidebar state (nanobot webui/sidebar_state.py) -------------------------------

WEBUI_SIDEBAR_STATE_SCHEMA_VERSION = 1
_MAX_STATE_FILE_BYTES = 256 * 1024
_MAX_LIST_ITEMS = 2_000
_MAX_MAP_ITEMS = 2_000
_MAX_KEY_LEN = 512
_MAX_TITLE_LEN = 160
_MAX_TAG_LEN = 40
_MAX_WORKBENCH_PANES = 4
_ALLOWED_DENSITIES = {"comfortable", "compact"}
_ALLOWED_SORTS = {"updated_desc", "created_desc", "title_asc", "manual"}
_ALLOWED_WORKBENCH_LAYOUTS = {"columns", "rows", "grid", "bsp", "main-stack"}
_SIDEBAR_STATE_WRITE_LOCK = threading.Lock()


def webui_sidebar_state_path(ctx: WebUIContext) -> Path:
    return Path(ctx.history.directory) / "sidebar-state.json"


def default_webui_sidebar_state() -> dict[str, Any]:
    return {
        "schema_version": WEBUI_SIDEBAR_STATE_SCHEMA_VERSION,
        "pinned_keys": [],
        "archived_keys": [],
        "session_order": [],
        "title_overrides": {},
        "project_name_overrides": {},
        "tags_by_key": {},
        "collapsed_groups": {},
        "workbench": {"version": 1, "tabs": {}},
        "view": {"density": "comfortable", "show_archived": False, "sort": "updated_desc"},
        "updated_at": None,
    }


def _clean_string(value: Any, *, max_len: int = _MAX_KEY_LEN) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned[:max_len] if cleaned else None


def _clean_string_list(value: Any, *, max_len: int = _MAX_KEY_LEN) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value[:_MAX_LIST_ITEMS]:
        cleaned = _clean_string(item, max_len=max_len)
        if cleaned is None or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _clean_split_ratios(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    ratios: list[float] = []
    for raw_ratio in value[: _MAX_WORKBENCH_PANES - 1]:
        if isinstance(raw_ratio, bool) or not isinstance(raw_ratio, (int, float)):
            continue
        ratio = float(raw_ratio)
        if math.isfinite(ratio):
            ratios.append(round(min(0.95, max(0.05, ratio)), 4))
    return ratios


def _clean_bool_map(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, bool] = {}
    for key, raw in list(value.items())[:_MAX_MAP_ITEMS]:
        cleaned_key = _clean_string(key)
        if cleaned_key is not None:
            out[cleaned_key] = bool(raw)
    return out


def _clean_title_overrides(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw_title in list(value.items())[:_MAX_MAP_ITEMS]:
        cleaned_key = _clean_string(key)
        cleaned_title = _clean_string(raw_title, max_len=_MAX_TITLE_LEN)
        if cleaned_key is not None and cleaned_title is not None:
            out[cleaned_key] = cleaned_title
    return out


def _clean_tags_by_key(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, raw_tags in list(value.items())[:_MAX_MAP_ITEMS]:
        cleaned_key = _clean_string(key)
        if cleaned_key is None:
            continue
        tags = _clean_string_list(raw_tags, max_len=_MAX_TAG_LEN)[:12]
        if tags:
            out[cleaned_key] = tags
    return out


def _clean_view(value: Any) -> dict[str, Any]:
    default: dict[str, Any] = default_webui_sidebar_state()["view"]
    if not isinstance(value, dict):
        return dict(default)
    density, sort = value.get("density"), value.get("sort")
    return {
        "density": density if density in _ALLOWED_DENSITIES else default["density"],
        "show_archived": bool(value.get("show_archived", default["show_archived"])),
        "sort": sort if sort in _ALLOWED_SORTS else default["sort"],
    }


def _clean_workbench(value: Any) -> dict[str, Any]:
    empty = {"version": 1, "tabs": {}}
    if not isinstance(value, dict) or value.get("version") != 1:
        return empty
    raw_tabs = value.get("tabs")
    if not isinstance(raw_tabs, dict):
        return empty
    tabs: dict[str, dict[str, Any]] = {}
    claimed_panes: set[str] = set()
    for raw_tab_key, tab in list(raw_tabs.items())[:_MAX_MAP_ITEMS]:
        tab_key = _clean_string(raw_tab_key)
        if tab_key is None or not isinstance(tab, dict):
            continue
        pane_keys = [key for key in _clean_string_list(tab.get("paneKeys"))
                     if key not in claimed_panes][:_MAX_WORKBENCH_PANES]
        if not pane_keys:
            continue
        explicit = tab.get("explicit") is True
        if not explicit and len(pane_keys) == 1:
            continue
        requested = [key for key in _clean_string_list(tab.get("layoutPaneKeys")) if key in pane_keys]
        claimed_panes.update(pane_keys)
        raw_layout = tab.get("layout")
        tabs[tab_key] = {
            "explicit": explicit,
            "title": _clean_string(tab.get("title"), max_len=_MAX_TITLE_LEN),
            "paneKeys": pane_keys,
            "layoutPaneKeys": requested + [key for key in pane_keys if key not in requested],
            "layout": raw_layout if raw_layout in _ALLOWED_WORKBENCH_LAYOUTS else "columns",
            "splitRatios": _clean_split_ratios(tab.get("splitRatios")),
        }
    return {"version": 1, "tabs": tabs}


def normalize_webui_sidebar_state(raw: Any) -> dict[str, Any]:
    """Return a validated canonical sidebar state."""
    if not isinstance(raw, dict):
        raw = {}
    state = default_webui_sidebar_state()
    state["pinned_keys"] = _clean_string_list(raw.get("pinned_keys"))
    state["archived_keys"] = _clean_string_list(raw.get("archived_keys"))
    state["session_order"] = _clean_string_list(raw.get("session_order"))
    state["title_overrides"] = _clean_title_overrides(raw.get("title_overrides"))
    state["project_name_overrides"] = _clean_title_overrides(raw.get("project_name_overrides"))
    state["tags_by_key"] = _clean_tags_by_key(raw.get("tags_by_key"))
    state["collapsed_groups"] = _clean_bool_map(raw.get("collapsed_groups"))
    state["workbench"] = _clean_workbench(raw.get("workbench"))
    state["view"] = _clean_view(raw.get("view"))
    updated_at = raw.get("updated_at")
    state["updated_at"] = updated_at if isinstance(updated_at, str) else None
    return state


def read_webui_sidebar_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return default_webui_sidebar_state()
    try:
        if path.stat().st_size > _MAX_STATE_FILE_BYTES:
            logger.warning("webui sidebar state too large, ignoring: {}", path)
            return default_webui_sidebar_state()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("read webui sidebar state failed {}: {}", path, e)
        return default_webui_sidebar_state()
    return normalize_webui_sidebar_state(raw)


def write_webui_sidebar_state(path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Validate, stamp ``updated_at`` and replace the file atomically."""
    with _SIDEBAR_STATE_WRITE_LOCK:
        state = normalize_webui_sidebar_state(raw)
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        encoded = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        if len(encoded) > _MAX_STATE_FILE_BYTES:
            raise ValueError("sidebar state is too large")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "wb") as f:
            f.write(encoded + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return state


async def _handle_sidebar_state(request: web.Request) -> web.Response:
    """GET /api/webui/sidebar-state (ws_http.py:1976)."""
    if not _authorized(request):
        return _unauthorized()
    return _json_response(read_webui_sidebar_state(webui_sidebar_state_path(_ctx(request))))


def _sidebar_update_mutation(ctx: WebUIContext):
    async def sidebar_update(payload: dict[str, Any]) -> web.Response:
        """``webui_request`` ``sidebar.update`` (ws_http.py:1981)."""
        state_value = payload.get("state") if isinstance(payload, dict) else None
        if state_value is None:
            return _error(400, "missing state")
        if not isinstance(state_value, dict):
            return _error(400, "state must be an object")
        try:
            state = write_webui_sidebar_state(webui_sidebar_state_path(ctx), state_value)
        except ValueError as e:
            return _error(400, str(e))
        except OSError:
            logger.exception("failed to write webui sidebar state")
            return _error(500, "failed to write sidebar state")
        return _json_response(state)

    return sidebar_update


async def _handle_mutation_path(request: web.Request) -> web.Response:
    """Plain HTTP on a mutation path (ws_http.py:483)."""
    return _error(405, "WebUI mutations require an authenticated WebSocket")


# --- settings (read-only) ---------------------------------------------------------

_DEFAULT_REASONING_EFFORT_VALUES = ("", "low", "medium", "high")


def _reasoning_effort_values(provider_name: str | None, model: str) -> list[str]:
    """nanobot settings_models.py ``reasoning_effort_values_for`` (without model metadata)."""
    from nanoclaw.providers.registry import find_by_name

    spec = find_by_name(provider_name) if provider_name else None
    if spec is None:
        return list(_DEFAULT_REASONING_EFFORT_VALUES)
    model_lower = (model or "").lower()
    implicit = getattr(spec, "implicit_reasoning_models", ())
    if implicit and any(pattern in model_lower for pattern in implicit):
        return [""]
    remap = getattr(spec, "reasoning_effort_remap", ())
    if remap:
        wire: list[str] = []
        for _user, value in remap:
            if value and value != "none" and value not in wire:
                wire.append(value)
        return ["", *wire]
    return list(_DEFAULT_REASONING_EFFORT_VALUES)


def _preset_row(config: Any, name: str, preset: Any, *, active: bool, is_default: bool) -> dict[str, Any]:
    resolved = config.get_provider_name(preset.model, preset=preset)
    row: dict[str, Any] = {
        "name": name,
        "label": "Default" if is_default else name,
        "active": active,
        "is_default": is_default,
        "model": preset.model,
        "provider": preset.provider,
        "resolved_provider": resolved if is_default else (resolved or preset.provider),
        "max_tokens": preset.max_tokens,
        "context_window_tokens": preset.context_window_tokens,
        "temperature": preset.temperature,
        "reasoning_effort": preset.reasoning_effort,
        "api": None,
    }
    if not is_default:
        for field in ("extra_headers", "extra_body", "extra_query"):
            value = getattr(preset, field, None)
            row[field] = {k: "********" for k in value} if field == "extra_headers" and value else value
    row["reasoning_effort_values"] = _reasoning_effort_values(resolved or preset.provider, preset.model)
    return row


def settings_payload(ctx: WebUIContext) -> dict[str, Any]:
    """Read-only ``GET /api/settings`` (nanobot settings_api.py ``settings_payload``)."""
    from nanoclaw.config.loader import get_config_path
    from nanoclaw.security.workspace_access import workspace_sandbox_status

    config = ctx.config
    defaults = config.agents.defaults
    active = defaults.model_preset or "default"
    effective = config.resolve_preset()
    provider_name = config.get_provider_name(effective.model, preset=effective) or effective.provider
    provider = config.get_provider(effective.model, preset=effective)
    presets = [_preset_row(config, "default", config.resolve_default_preset(),
                           active=active == "default", is_default=True)]
    presets += [_preset_row(config, name, preset, active=active == name, is_default=False)
                for name, preset in config.model_presets.items()]
    restrict = bool(config.tools.restrict_to_workspace)
    workspace = str(ctx.workspace)
    sandbox = workspace_sandbox_status(restrict_to_workspace=restrict, workspace=Path(workspace)).as_dict()
    gateway = config.gateway
    payload: dict[str, Any] = {
        "model_api_resolution_supported": False,
        "provider_api_configuration_supported": False,
        "agent": {
            "model": effective.model,
            "provider": provider_name if effective.provider == "auto" else effective.provider,
            "resolved_provider": provider_name,
            "has_api_key": bool(provider and getattr(provider, "api_key", None)),
            "model_preset": active,
            "max_tokens": effective.max_tokens,
            "context_window_tokens": effective.context_window_tokens,
            "temperature": effective.temperature,
            "reasoning_effort": effective.reasoning_effort,
            "api": None,
            "timezone": defaults.timezone,
            "tool_hint_max_length": 40,
        },
        "model_presets": presets,
        "model_request_overrides_supported": False,
        "model_call_order": [],
        "model_call_order_editable": False,
        "model_configuration_migratable": False,
        "providers": [],
        "web_search": {
            "provider": "duckduckgo", "api_key_hint": None, "base_url": None,
            "max_results": 5, "timeout": 30,
            "providers": [{"name": "duckduckgo", "label": "DuckDuckGo", "credential": "none"}],
        },
        "web": {"enable": True, "use_proxy": False, "user_agent": None,
                "search": {"max_results": 5, "timeout": 30}, "fetch": {"use_jina_reader": False}},
        "image_generation": {
            "enabled": False, "provider": "", "provider_configured": False, "model": "",
            "default_aspect_ratio": "1:1", "default_image_size": "1K", "max_images_per_turn": 0,
            "save_dir": "", "request_overrides_supported": False,
            "extra_headers": None, "extra_body": None, "providers": [],
        },
        "runtime": {
            "config_path": str(get_config_path()),
            "workspace_path": workspace,
            "gateway_host": gateway.host,
            "gateway_port": gateway.port,
            "heartbeat": {"enabled": gateway.heartbeat.enabled, "interval_s": gateway.heartbeat.interval_s},
            "dream": {"schedule": defaults.dream.describe_schedule()},
            "unified_session": False,
        },
        "advanced": {
            "restrict_to_workspace": restrict,
            "workspace_sandbox": sandbox,
            "webui_allow_local_service_access": False,
            "allow_local_preview_access": False,
            "webui_default_access_mode": "default",
            "private_service_protection_enabled": True,
            "ssrf_whitelist_count": len(config.tools.ssrf_whitelist),
            "mcp_server_count": len(config.tools.mcp_servers),
            "exec_enabled": bool(config.tools.exec.enable),
            "exec_max_runtime_limit": 86400,
            "exec_sandbox": getattr(config.tools.exec, "sandbox", None) or None,
            "exec_path_prepend_set": False,
            "exec_path_append_set": False,
        },
        "requires_restart": False,
    }
    # settings_api.py ``decorate_settings_payload`` (browser surface).
    payload.update({
        "surface": "browser",
        "runtime_surface": "browser",
        "runtime_capabilities": {"can_restart_engine": False, "can_open_logs": False,
                                 "can_export_diagnostics": False},
        "restart_behavior_by_section": {
            "appearance": "none", "models": "none", "providers": "none",
            "runtime": "engineRestart", "browser": "engineRestart", "image": "engineRestart",
            "apps": "engineRestart", "advanced": "appRestart",
        },
        "restart_required_sections": [],
        "apply_state": {"status": "idle", "sections": []},
    })
    return payload


async def _handle_settings(request: web.Request) -> web.Response:
    """GET /api/settings (settings_routes.py:500)."""
    if not _authorized(request):
        return _unauthorized()
    try:
        return _json_response(settings_payload(_ctx(request)))
    except Exception:
        logger.exception("webui settings payload failed")
        return _error(500, "failed to read settings")


# --- pairing, skills ---------------------------------------------------------------


async def _handle_pairing(request: web.Request) -> web.Response:
    """GET /api/settings/pairing (settings_system.py ``pairing_payload``).

    NanoClaw has no DM pairing, so there is never a pending request.
    """
    if not _authorized(request):
        return _unauthorized()
    return _json_response({"requests": []})


def webui_skills_payload(workspace: Path) -> dict[str, Any]:
    """nanobot webui/skills_api.py ``webui_skills_payload``: skills without local paths."""
    from nanoclaw.agent.skills import SkillsLoader

    loader = SkillsLoader(workspace)
    entries = sorted(loader.list_skills(filter_unavailable=False),
                     key=lambda entry: (entry.get("source") != "workspace", entry["name"]))
    skills = []
    for entry in entries:
        name = entry["name"]
        metadata = loader.get_skill_metadata(name)
        raw_description = metadata.get("description") if metadata else None
        available, unavailable_reason = loader.get_skill_availability(name)
        source = entry.get("source", "unknown")
        skills.append({
            "name": name,
            "description": (raw_description.strip() if isinstance(raw_description, str)
                            and raw_description.strip() else name),
            "source": source,
            "enabled": True,
            "deletable": source == "workspace",
            "available": available,
            "unavailable_reason": unavailable_reason,
        })
    return {"skills": skills}


async def _handle_webui_skills(request: web.Request) -> web.Response:
    """GET /api/webui/skills (ws_http.py:1803)."""
    if not _authorized(request):
        return _unauthorized()
    import asyncio

    return _json_response(await asyncio.to_thread(webui_skills_payload, _ctx(request).workspace))


async def api_not_found(request: web.Request) -> web.Response:
    """Any other ``/api/*`` path (ws_http.py:637): never the SPA page."""
    return _error(404, "API route not found")


def register_api_fallback(app: web.Application) -> None:
    """Register ``/api/{tail}`` → 404 last, after every other ``/api`` route."""
    app.router.add_route("*", "/api/{tail:.*}", api_not_found)


# --- registration -----------------------------------------------------------------

_SLASH = "{_slash:/?}"  # nanobot strips a trailing slash before routing


def register_rest_routes(app: web.Application, ctx: WebUIContext) -> None:
    """Add the ``/api`` read routes to ``app`` and ``sidebar.update`` to ``ctx.mutations``.

    Unknown ``/api/*`` paths are not caught here, so the server can add its
    own ``/api`` routes after this; it calls ``register_api_fallback(app)``
    last (nanobot: ``404 API route not found``, never the SPA page).
    """
    app[_CTX_KEY] = ctx
    ctx.mutations["sidebar.update"] = _sidebar_update_mutation(ctx)
    for path in (
        "/api/webui/sidebar-state/update",
        "/api/sessions/{key}/delete",
        "/api/workspaces/favorite",
        "/api/workspaces/resolve-project",
    ):
        app.router.add_route("*", path + _SLASH, _handle_mutation_path)
    routes = (
        ("/api/sessions", _handle_sessions_list),
        ("/api/sessions/{key}/webui-thread/prompts", _handle_prompt_index),
        ("/api/sessions/{key}/webui-thread/trace-detail", _handle_trace_detail),
        ("/api/sessions/{key}/webui-thread", _handle_webui_thread),
        ("/api/sessions/{key}/context", _handle_session_context),
        ("/api/commands", _handle_commands),
        ("/api/workspaces", _handle_workspaces),
        ("/api/webui/sidebar-state", _handle_sidebar_state),
        ("/api/settings", _handle_settings),
        ("/api/settings/pairing", _handle_pairing),
        ("/api/webui/skills", _handle_webui_skills),
    )
    for path, handler in routes:
        app.router.add_get(path + _SLASH, handler)
