"""Sidebar rows for ``GET /api/sessions``.

Ported from nanobot tests/webui/test_session_list_index.py. Left out: tests of
SQLite snapshot isolation, provider state, model-preset renames and session
handles, which NanoClaw's JSONL sessions do not have.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import nanoclaw.webui.session_list_index as session_list_index
from nanoclaw.cron.session_turns import CRON_HISTORY_META
from nanoclaw.session.automation_spec import AUTOMATION_HISTORY_META
from nanoclaw.session.history_visibility import HIDDEN_HISTORY_META
from nanoclaw.session.manager import SessionManager
from nanoclaw.session.model_selection import SESSION_MODEL_PRESET_METADATA_KEY
from nanoclaw.webui.history import WebUIHistoryStore
from nanoclaw.webui.session_list_index import (
    RECOVERY_METADATA_KEY,
    SESSION_TYPE_KEY,
    WORKSPACE_SCOPE_METADATA_KEY,
)


def _manager(tmp_path: Path) -> SessionManager:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return SessionManager(workspace, sessions_root=tmp_path / "sessions")


def list_webui_sessions(manager: SessionManager) -> list[dict]:
    return session_list_index.list_webui_sessions(manager, WebUIHistoryStore.for_sessions(manager))


def test_webui_session_list_reads_previews(tmp_path):
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:indexed")
    session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = "fast"
    session.add_message("user", "indexed preview")
    manager.save(session)
    rows = list_webui_sessions(manager)
    assert rows[0]["key"] == "websocket:indexed"
    assert rows[0]["preview"] == "indexed preview"
    assert rows[0]["model_preset"] == "fast"


def test_webui_session_list_surfaces_pending_recovery_state(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:needs-attention")
    session.add_message("user", "the interrupted task")
    session.metadata[RECOVERY_METADATA_KEY] = {
        "status": "awaiting_user",
        "recovery_id": "recovery-123",
        "reason": "uncertain_tool_state",
        "attempts": 1,
        # Private checkpoint details must never leak into the sidebar index.
        "checkpoint": {"tool_args": "secret"},
    }
    manager.save(session)

    row = list_webui_sessions(manager)[0]

    assert row["recovery_state"] == {
        "status": "awaiting_user",
        "recovery_id": "recovery-123",
        "reason": "uncertain_tool_state",
        "attempts": 1,
    }


def test_webui_session_list_indexes_workspace_scope_and_preserves_null(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    scoped = manager.get_or_create("websocket:scoped")
    scoped.metadata[WORKSPACE_SCOPE_METADATA_KEY] = {
        "project_path": str(project),
        "access_mode": "full",
        "future_extension": "x" * 5000,
    }
    manager.save(scoped)
    explicit_null = manager.get_or_create("websocket:null")
    explicit_null.metadata[WORKSPACE_SCOPE_METADATA_KEY] = None
    manager.save(explicit_null)
    manager.save(manager.get_or_create("websocket:missing"))

    rows = {row["key"]: row for row in list_webui_sessions(manager)}

    assert session_list_index.indexed_workspace_scope(rows["websocket:scoped"]) == (
        True,
        {"project_path": str(project), "access_mode": "full"},
    )
    assert session_list_index.indexed_workspace_scope(rows["websocket:null"]) == (True, None)
    assert session_list_index.indexed_workspace_scope(rows["websocket:missing"]) == (False, None)

    scoped.metadata[WORKSPACE_SCOPE_METADATA_KEY]["access_mode"] = "restricted"
    manager.save(scoped)

    refreshed = {row["key"]: row for row in list_webui_sessions(manager)}
    assert session_list_index.indexed_workspace_scope(refreshed["websocket:scoped"])[1] == {
        "project_path": str(project),
        "access_mode": "restricted",
    }


def test_webui_session_list_rejects_invalid_internal_model_preset_metadata(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:custom-metadata")
    session.metadata["model_preset"] = 7
    session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = {"invalid": True}
    session.add_message("user", "custom metadata")
    manager.save(session)

    with pytest.raises(ValueError, match="session model preset must be a non-empty string"):
        list_webui_sessions(manager)

    assert manager.get_or_create(session.key).metadata["model_preset"] == 7


def test_webui_session_list_reflects_changed_preview(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    first = manager.get_or_create("websocket:first")
    first.add_message("user", "first")
    manager.save(first)
    second = manager.get_or_create("websocket:second")
    second.add_message("user", "second before")
    manager.save(second)

    assert {row["preview"] for row in list_webui_sessions(manager)} == {"first", "second before"}

    second.messages.clear()
    second.add_message("user", "second after")
    manager.save(second)

    assert {row["preview"] for row in list_webui_sessions(manager)} == {"first", "second after"}


def test_webui_session_list_drops_deleted_index_rows(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:deleted")
    session.add_message("user", "gone")
    manager.save(session)

    assert list_webui_sessions(manager)[0]["key"] == "websocket:deleted"

    assert manager.delete_session("websocket:deleted") is True

    assert list_webui_sessions(manager) == []


def test_webui_session_list_skips_cron_internal_user_preview(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:cron-preview")
    session.add_message(
        "user",
        "Scheduled cron job triggered: 30s-test\n\nInternal reminder prompt",
        **{CRON_HISTORY_META: True},
    )
    session.add_message("assistant", "提醒已经到期。")
    manager.save(session)

    assert list_webui_sessions(manager)[0]["preview"] == "提醒已经到期。"


def test_webui_session_list_skips_trigger_internal_user_preview(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:trigger-preview")
    session.add_message(
        "user",
        "Local trigger received: PR review",
        **{AUTOMATION_HISTORY_META: {"kind": "local_trigger", "trigger_id": "trg_123"}},
    )
    session.add_message("assistant", "PR #4502 已经开始 review。")
    manager.save(session)

    assert list_webui_sessions(manager)[0]["preview"] == "PR #4502 已经开始 review。"


def test_webui_session_list_skips_hidden_history_user_preview(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:hidden-preview")
    session.add_message(
        "user",
        "internal subagent result",
        **{HIDDEN_HISTORY_META: {"kind": "subagent_result", "subagent_task_id": "sub-1"}},
    )
    session.add_message("assistant", "subagent summary")
    manager.save(session)

    assert list_webui_sessions(manager)[0]["preview"] == "subagent summary"


def test_webui_session_list_sorts_by_message_activity_not_maintenance_timestamp(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    old = manager.get_or_create("websocket:old")
    old.created_at = datetime(2026, 6, 1, 10, 0, 0)
    old.add_message("user", "old first visible activity")
    old.messages[-1]["timestamp"] = "2026-06-01T10:00:00"
    old.add_message("assistant", "automation result")
    old.messages[-1]["timestamp"] = "2026-06-05T10:00:00"
    old.updated_at = datetime(2026, 6, 30, 17, 40, 0)
    manager.save(old)

    newer = manager.get_or_create("websocket:newer")
    newer.created_at = datetime(2026, 6, 4, 10, 0, 0)
    newer.add_message("user", "newer real activity")
    newer.messages[-1]["timestamp"] = "2026-06-04T10:00:00"
    newer.updated_at = datetime(2026, 6, 4, 10, 0, 0)
    manager.save(newer)

    rows = list_webui_sessions(manager)

    assert [row["key"] for row in rows] == ["websocket:old", "websocket:newer"]
    assert rows[0]["updated_at"] == "2026-06-05T10:00:00"


def test_webui_session_list_fallback_time_when_missing(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.get_session_path("websocket:missing-time").write_text(
        '{"_type": "metadata", "key": "websocket:missing-time"}\n'
        '{"_type": "message", "role": "user", "content": "hello"}\n',
        encoding="utf-8",
    )

    rows = list_webui_sessions(manager)
    assert len(rows) == 1
    assert rows[0]["key"] == "websocket:missing-time"
    assert rows[0]["preview"] == "hello"
    datetime.fromisoformat(rows[0]["created_at"])
    datetime.fromisoformat(rows[0]["updated_at"])


def test_sidebar_uses_durable_display_summary_and_filters_private_sessions(history):
    sessions = history.sessions
    for key in ("websocket:visible", "websocket:private"):
        sessions.save(sessions.get_or_create(key))
        history.append(key, {"event": "user", "text": "  display\n preview  ", "created_at_ms": 1760000001000})
        history.append(key, {"event": "message", "text": "answer", "created_at_ms": 1760000002000})
    private = sessions.get_or_create("websocket:private")
    private.metadata[SESSION_TYPE_KEY] = "internal"
    sessions.save(private)
    rows = list_webui_sessions(sessions)
    assert [row["key"] for row in rows] == ["websocket:visible"]
    assert rows[0]["preview"] == "display preview"
    assert datetime.fromisoformat(rows[0]["updated_at"]).timestamp() == 1760000002
    history.append("websocket:visible", {"event": "turn_end", "created_at_ms": 1760000003000})
    assert datetime.fromisoformat(list_webui_sessions(sessions)[0]["updated_at"]).timestamp() == 1760000003
    sessions.delete_session("websocket:visible")
    assert list_webui_sessions(sessions) == []


# --- NanoClaw additions -------------------------------------------------------------


def test_payload_lists_only_websocket_sessions_and_drops_internal_fields(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    for key, text in (("websocket:a", "web chat"), ("cli:direct", "terminal chat"), ("telegram:1", "tg")):
        session = manager.get_or_create(key)
        session.add_message("user", text)
        manager.save(session)
    subagent = manager.get_or_create("websocket:child")
    subagent.metadata[SESSION_TYPE_KEY] = "subagent"
    manager.save(subagent)

    payload = session_list_index.webui_sessions_payload(manager, WebUIHistoryStore.for_sessions(manager))

    assert [row["key"] for row in payload["sessions"]] == ["websocket:a"]
    row = payload["sessions"][0]
    assert set(row) == {"key", "created_at", "updated_at", "title", "preview", "model_preset"}
    assert row["preview"] == "web chat"
    assert row["title"] == "" and row["model_preset"] is None


def test_payload_hooks_add_run_state_scope_and_handle(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:hooked")
    session.metadata[WORKSPACE_SCOPE_METADATA_KEY] = {"project_path": "/p", "access_mode": "full"}
    session.metadata["title"] = "<think>hmm</think>Generated title"
    manager.save(session)

    payload = session_list_index.webui_sessions_payload(
        manager,
        None,
        run_started_at=lambda chat_id: 12.5 if chat_id == "hooked" else None,
        workspace_scope=lambda present, raw: {"present": present, "raw": raw},
        handle=lambda key: {"name": key.split(":")[1]},
    )

    [row] = payload["sessions"]
    assert row["run_started_at"] == 12.5
    assert row["workspace_scope"] == {"present": True, "raw": {"project_path": "/p", "access_mode": "full"}}
    assert row["handle"] == {"name": "hooked"}
    assert row["title"] == "Generated title"


def test_user_edited_title_is_kept_verbatim(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:titled")
    session.metadata.update({"title": "<think>kept</think>", "title_user_edited": True})
    manager.save(session)
    assert list_webui_sessions(manager)[0]["title"] == "<think>kept</think>"


def test_subagent_announce_preview_is_scrubbed(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:announce")
    session.add_message(
        "assistant",
        "[Subagent 'scan' completed]\n\nTask: secret prompt\n\nResult:\nall good\n\nSummarize this naturally",
        injected_event="subagent_result",
    )
    manager.save(session)
    assert list_webui_sessions(manager)[0]["preview"] == "[Subagent 'scan' completed] all good"


def test_delete_webui_session_removes_session_descendants_and_event_logs(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    store = WebUIHistoryStore.for_sessions(manager)
    parent = manager.get_or_create("websocket:parent")
    parent.add_message("user", "hi")
    manager.save(parent)
    child = manager.get_or_create("subagent:websocket:parent:task1")
    child.metadata.update({SESSION_TYPE_KEY: "subagent", "parent_session_key": "websocket:parent"})
    manager.save(child)
    other = manager.get_or_create("websocket:other")
    manager.save(other)
    store.append_records("websocket:parent", [{"event": "user", "text": "hi"}])
    store.append_records("websocket:other", [{"event": "user", "text": "keep"}])

    assert session_list_index.delete_webui_session(manager, store, "websocket:parent") is True

    assert manager.get_existing("websocket:parent") is None
    assert manager.get_existing("subagent:websocket:parent:task1") is None
    assert not store.exists("websocket:parent")
    assert store.exists("websocket:other") and manager.get_existing("websocket:other") is not None
    assert session_list_index.delete_webui_session(manager, store, "websocket:parent") is False


def test_delete_removes_an_orphan_event_log(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    store = WebUIHistoryStore.for_sessions(manager)
    store.append_records("websocket:orphan", [{"event": "user", "text": "x"}])
    assert session_list_index.delete_webui_session(manager, store, "websocket:orphan") is True
    assert not store.path("websocket:orphan").exists()


def test_session_file_summary_is_cached_until_the_file_changes(tmp_path: Path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:cached")
    session.add_message("user", "first")
    manager.save(session)
    assert list_webui_sessions(manager)[0]["preview"] == "first"
    calls = []
    original = session_list_index._messages_summary
    monkeypatch.setattr(session_list_index, "_messages_summary", lambda m: calls.append(1) or original(m))
    assert list_webui_sessions(manager)[0]["preview"] == "first"
    assert calls == []
    session.messages[0]["content"] = "changed"
    manager.save(session)
    assert list_webui_sessions(manager)[0]["preview"] == "changed"
    assert calls == [1]
    assert json.loads(manager.get_session_path(session.key).read_text().splitlines()[1])["content"] == "changed"
