"""Sidebar rows for ``GET /api/sessions``, and WebUI session deletion.

Ported from nanobot/webui/session_list_index.py plus the row cleaning in
nanobot ws_http.py ``_sessions_list_payload`` and the preview/title helpers of
nanobot session/manager.py and sqlite_store.py. nanobot reads one SQLite
snapshot whose ``sessions`` row already holds ``title``, ``preview`` and
``visible_updated_at`` (computed on save) and whose ``history_heads`` row holds
the display-history preview and activity time (computed on append). NanoClaw
computes the same values from the session JSONL file (cached by file
size/mtime) and the chat's event log.

Row (``list_webui_sessions``)::

    {"key", "created_at", "updated_at", "title", "preview", "model_preset",
     "recovery_state", "_workspace_scope_present", "_workspace_scope_value", "path"}

Payload (``webui_sessions_payload``): ``{"sessions": [row...]}`` with only
``websocket:`` keys, newest first, internal fields and ``path`` dropped,
``recovery_state`` dropped when ``None``, and ``run_started_at``,
``workspace_scope``, ``handle`` added by the caller's hooks.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from nanoclaw.agent.messages import strip_think
from nanoclaw.runtime_context import public_history_message
from nanoclaw.session.display_history import HistoryKind
from nanoclaw.session.history_visibility import is_hidden_history_message
from nanoclaw.session.manager import _sanitize_assistant_replay_text
from nanoclaw.session.model_selection import model_preset_from_metadata
from nanoclaw.webui.history import WebUIHistoryStore
from nanoclaw.webui.session_identity import is_webui_session_key

if TYPE_CHECKING:
    from nanoclaw.session.manager import SessionManager

WORKSPACE_SCOPE_METADATA_KEY = "workspace_scope"  # nanobot security/workspace_access.py
RECOVERY_METADATA_KEY = "webui_recovery"  # nanobot session/recovery.py
SESSION_TYPE_KEY = "session_type"  # nanobot session/types.py
_PUBLIC_SESSION_TYPES = frozenset({"conversation"})
_RECOVERY_STATUSES = frozenset({"resuming", "awaiting_user", "recovered", "failed"})
_SESSION_PREVIEW_MAX_CHARS = 120
_SUBAGENT_CHANNEL_RESULT_MAX_CHARS = 800

_WORKSPACE_SCOPE_PRESENT_FIELD = "_workspace_scope_present"
_WORKSPACE_SCOPE_VALUE_FIELD = "_workspace_scope_value"
WEBUI_SESSION_INDEX_INTERNAL_FIELDS = frozenset(
    {_WORKSPACE_SCOPE_PRESENT_FIELD, _WORKSPACE_SCOPE_VALUE_FIELD}
)
_INDEXED_WORKSPACE_SCOPE_KEYS = ("project_path", "path", "access_mode")
_MAX_INDEXED_WORKSPACE_SCOPE_BYTES = 4096


# --- previews and titles (nanobot session/manager.py) ---------------------------


def scrub_subagent_announce_body(content: str) -> str:
    """Channel-safe text from a full subagent announce blob
    (nanobot utils/subagent_channel_display.py)."""
    stripped = content.replace("\r\n", "\n").strip()
    lines = stripped.splitlines()
    header = lines[0].strip() if lines and lines[0].startswith("[Subagent") else ""
    lower = stripped.lower()
    key = "\nresult:\n"
    ri = lower.find(key)
    if ri == -1:
        key = "\nresult:"
        ri = lower.find(key)
    if ri == -1:
        return header if header else stripped
    after = stripped[ri + len(key):].lstrip()
    si = after.lower().find("summarize this naturally")
    if si != -1:
        after = after[:si].rstrip()
    body = after.strip()
    limit = _SUBAGENT_CHANNEL_RESULT_MAX_CHARS
    if limit and len(body) > limit:
        body = body[: limit - 1].rstrip() + "…"
    if header and body:
        return f"{header}\n\n{body}"
    return header or body or stripped


def _text_preview(content: object) -> str:
    """Return compact display text for session lists."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in cast(list[object], content):
            if isinstance(block, dict):
                block_data = cast(dict[object, object], block)
                if block_data.get("type") != "text":
                    continue
                value = block_data.get("text")
                if isinstance(value, str):
                    parts.append(value)
        text = " ".join(parts)
    else:
        return ""
    text = _sanitize_assistant_replay_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _SESSION_PREVIEW_MAX_CHARS:
        text = text[: _SESSION_PREVIEW_MAX_CHARS - 1].rstrip() + "…"
    return text


def message_preview_text(message: dict[str, Any]) -> str:
    """Session list preview text; subagent inject blobs are shortened for display."""
    message = public_history_message(message)
    content = cast(object, message.get("content"))
    if message.get("injected_event") == "subagent_result" and isinstance(content, str):
        content = scrub_subagent_announce_body(content)
    return _text_preview(content)


def metadata_title(metadata: object) -> str:
    if not isinstance(metadata, dict):
        return ""
    metadata_data = cast(dict[object, object], metadata)
    title = metadata_data.get("title")
    if not isinstance(title, str):
        return ""
    if metadata_data.get("title_user_edited") is True:
        return title
    return strip_think(title)


def recovery_state_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a sanitized recovery state suitable for the WebSocket wire."""
    value = metadata.get(RECOVERY_METADATA_KEY) if metadata else None
    if not isinstance(value, dict):
        return None
    state = cast(dict[str, Any], value)
    status = state.get("status")
    recovery_id = state.get("recovery_id")
    if status not in _RECOVERY_STATUSES or not isinstance(recovery_id, str):
        return None
    payload: dict[str, Any] = {"status": status, "recovery_id": recovery_id}
    reason = state.get("reason")
    if isinstance(reason, str) and reason:
        payload["reason"] = reason
    attempts = state.get("attempts")
    if isinstance(attempts, int) and attempts >= 0:
        payload["attempts"] = attempts
    can_continue = state.get("can_continue")
    if isinstance(can_continue, bool):
        payload["can_continue"] = can_continue
    failure_error_kind = state.get("failure_error_kind")
    if isinstance(failure_error_kind, str) and failure_error_kind in {
        "authentication", "billing", "connection", "timeout", "rate_limit", "server", "unknown",
    }:
        payload["failure_error_kind"] = failure_error_kind
    return payload


def public_history(metadata: Mapping[str, object]) -> bool:
    """Ordinary conversations are listed; typed sessions (subagent children) are not.
    (nanobot ``SessionTypes.public_history`` with only the conversation type public)"""
    name = metadata.get(SESSION_TYPE_KEY, "conversation")
    return isinstance(name, str) and name in _PUBLIC_SESSION_TYPES


# --- session file summary (nanobot sqlite_store.py ``save``) ----------------------

_SUMMARY_CACHE: dict[Path, tuple[tuple[int, int, int], tuple[str, str | None]]] = {}
_SUMMARY_LOCK = threading.Lock()


def _messages_summary(messages: list[dict[str, Any]]) -> tuple[str, str | None]:
    """(preview, last visible message timestamp) as nanobot stores on save."""
    preview = ""
    fallback = ""
    scanned_chars = 0
    for message in messages[:200]:
        scanned_chars += len(json.dumps(message, ensure_ascii=False, separators=(",", ":"))) + 1
        if scanned_chars > 1_000_000:
            break
        if is_hidden_history_message(message):
            continue
        text = message_preview_text(message)
        if message.get("role") == "user" and text:
            preview = text
            break
        if message.get("role") == "assistant" and not fallback:
            fallback = text
    last_visible = next((m["timestamp"] for m in reversed(messages)
                         if m.get("timestamp") and not is_hidden_history_message(m)
                         and m.get("role") in {"user", "assistant"}), None)
    return preview or fallback, last_visible


def _session_file_summary(path: Path) -> tuple[str, str | None]:
    try:
        stat = path.stat()
    except OSError:
        return "", None
    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    with _SUMMARY_LOCK:
        cached = _SUMMARY_CACHE.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
    messages: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and value.get("_type") != "metadata":
                    messages.append(cast(dict[str, Any], value))
    except OSError:
        return "", None
    summary = _messages_summary(messages)
    with _SUMMARY_LOCK:
        _SUMMARY_CACHE[path] = (signature, summary)
        while len(_SUMMARY_CACHE) > 512:
            _SUMMARY_CACHE.pop(next(iter(_SUMMARY_CACHE)))
    return summary


def history_head(store: WebUIHistoryStore | None, key: str) -> tuple[int, str]:
    """(latest display-history ``created_at_ms``, display preview) for a chat
    (nanobot ``history_heads``: the user preview, else the first answer)."""
    if store is None:
        return 0, ""
    facts, _ = store.read_facts(key)
    updated_at_ms = 0
    preview = user_preview = ""
    for fact in facts:
        updated_at_ms = max(updated_at_ms, fact.created_at_ms)
        if fact.kind in {HistoryKind.INPUT, HistoryKind.ANSWER, HistoryKind.ANSWER_SEGMENT}:
            text = message_preview_text({"content": fact.content.get("text", "")})
            if not text:
                continue
            if fact.kind is HistoryKind.INPUT:
                user_preview = user_preview or text
            else:
                preview = preview or text
    return updated_at_ms, user_preview or preview


# --- the list ---------------------------------------------------------------------


def list_session_metadata(
    session_manager: SessionManager, store: WebUIHistoryStore | None = None,
) -> list[dict[str, Any]]:
    """nanobot ``list_session_metadata`` rows from the session files and event logs."""
    rows: list[dict[str, Any]] = []
    for path in sorted(session_manager.sessions_dir.glob("*.jsonl")):
        header = session_manager._read_header(path)
        if header is None or not isinstance(header.get("key"), str):
            continue
        key = header["key"]
        try:
            fallback_time = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        except OSError:
            continue
        preview, visible_updated_at = _session_file_summary(path)
        history_updated_at_ms, history_preview = history_head(store, key)
        metadata = header["metadata"]
        rows.append({
            "key": key,
            "created_at": header.get("created_at") or fallback_time,
            "updated_at": header.get("updated_at") or fallback_time,
            "title": metadata_title(metadata), "preview": preview,
            "visible_updated_at": visible_updated_at, "metadata": metadata,
            "history_updated_at_ms": history_updated_at_ms, "history_preview": history_preview,
            "path": str(path),
        })
    return sorted(rows, key=lambda row: row["updated_at"], reverse=True)


def list_webui_sessions(
    session_manager: SessionManager,
    store: WebUIHistoryStore | None = None,
    stored_sessions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One row per public session, newest message activity first."""
    stored = list_session_metadata(session_manager, store) if stored_sessions is None else stored_sessions
    rows: list[dict[str, Any]] = []
    for row in stored:
        metadata = row["metadata"]
        if not public_history(metadata):
            continue
        millis = row["history_updated_at_ms"]
        activity = datetime.fromtimestamp(millis / 1000).isoformat() if millis else None
        rows.append({
            "key": row["key"], "created_at": row["created_at"],
            "updated_at": _visible_activity_updated_at(row["updated_at"], row["visible_updated_at"], activity),
            "title": row["title"], "preview": row["preview"] or row["history_preview"],
            "model_preset": model_preset_from_metadata(metadata),
            "recovery_state": recovery_state_from_metadata(metadata),
            **_indexed_workspace_scope_fields(metadata), "path": row["path"],
        })
    return sorted(rows, key=lambda row: row.get("updated_at") or "", reverse=True)


def webui_sessions_payload(
    session_manager: SessionManager,
    store: WebUIHistoryStore | None = None,
    *,
    run_started_at: Callable[[str], float | None] | None = None,
    workspace_scope: Callable[[bool, object], dict[str, Any] | None] | None = None,
    handle: Callable[[str], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """``GET /api/sessions`` body: ``{"sessions": [...]}`` for ``websocket:`` keys.

    ``run_started_at(chat_id)`` is the running turn's wall start time,
    ``workspace_scope(present, raw)`` builds the row's scope payload from the
    indexed metadata value, ``handle(key)`` the session handle payload; each
    field is left out when its hook is missing or returns ``None``.
    """
    cleaned: list[dict[str, Any]] = []
    for s in list_webui_sessions(session_manager, store):
        key = s.get("key")
        if not (isinstance(key, str) and is_webui_session_key(key)):
            continue
        row = {k: v for k, v in s.items() if k != "path" and k not in WEBUI_SESSION_INDEX_INTERNAL_FIELDS}
        # Keep the additive recovery field absent for ordinary sessions.
        if row.get("recovery_state") is None:
            row.pop("recovery_state", None)
        chat_id = key.split(":", 1)[1]
        if run_started_at is not None and (started_at := run_started_at(chat_id)) is not None:
            row["run_started_at"] = started_at
        if workspace_scope is not None:
            scope = workspace_scope(*indexed_workspace_scope(s))
            if scope is not None:
                row["workspace_scope"] = scope
        if handle is not None and (payload := handle(key)) is not None:
            row["handle"] = payload
        cleaned.append(row)
    return {"sessions": cleaned}


def delete_webui_session(
    session_manager: SessionManager, store: WebUIHistoryStore | None, key: str,
) -> bool:
    """``session.delete``: remove the session, its descendants and their event logs."""
    removed = session_manager.delete_session_family(key)
    transcript_removed = False
    if store is not None:
        for removed_key in {key, *removed}:
            transcript_removed |= store.delete(removed_key)
    return bool(removed) or transcript_removed


def indexed_workspace_scope(row: dict[str, Any]) -> tuple[bool, object]:
    """Return the cached sidebar scope value while preserving missing vs null."""
    return (
        row.get(_WORKSPACE_SCOPE_PRESENT_FIELD) is True,
        cast(object, row.get(_WORKSPACE_SCOPE_VALUE_FIELD)),
    )


def _indexed_workspace_scope_fields(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict):
        return {_WORKSPACE_SCOPE_PRESENT_FIELD: False, _WORKSPACE_SCOPE_VALUE_FIELD: None}
    metadata_data = cast(dict[str, Any], metadata)
    if WORKSPACE_SCOPE_METADATA_KEY not in metadata_data:
        return {_WORKSPACE_SCOPE_PRESENT_FIELD: False, _WORKSPACE_SCOPE_VALUE_FIELD: None}
    raw_scope = metadata_data.get(WORKSPACE_SCOPE_METADATA_KEY)
    indexed_scope: object = False
    if raw_scope is None:
        indexed_scope = None
    elif isinstance(raw_scope, dict):
        scope_data = cast(dict[object, object], raw_scope)
        recognized = {key: scope_data[key] for key in _INDEXED_WORKSPACE_SCOPE_KEYS if key in scope_data}
        try:
            encoded = json.dumps(recognized, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
        else:
            if len(encoded.encode("utf-8")) <= _MAX_INDEXED_WORKSPACE_SCOPE_BYTES:
                indexed_scope = cast(object, json.loads(encoded))
    return {_WORKSPACE_SCOPE_PRESENT_FIELD: True, _WORKSPACE_SCOPE_VALUE_FIELD: indexed_scope}


def _timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _latest_updated_at(stored: str | None, activity: str | None) -> str | None:
    if _timestamp(activity) > _timestamp(stored):
        return activity
    return stored


def _visible_activity_updated_at(
    stored: str | None,
    visible_message_at: str | None,
    webui_activity: str | None,
) -> str | None:
    return _latest_updated_at(visible_message_at, webui_activity) or stored


__all__ = [
    "WEBUI_SESSION_INDEX_INTERNAL_FIELDS",
    "delete_webui_session",
    "history_head",
    "indexed_workspace_scope",
    "list_session_metadata",
    "list_webui_sessions",
    "message_preview_text",
    "metadata_title",
    "public_history",
    "recovery_state_from_metadata",
    "webui_sessions_payload",
]
