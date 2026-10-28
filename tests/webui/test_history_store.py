"""The per-chat JSONL event log, session backfill, and the payload entry point.

Includes adapted nanobot tests/session/test_display_history.py cases (facts
survive reopen, the recorder keeps completed segments, the record adapter).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanoclaw.session.display_history import HistoryFact, HistoryKind
from nanoclaw.session.manager import SessionManager
from nanoclaw.session.summary import HIDDEN_HISTORY_META, SUMMARY_CONTINUATION_TEXT
from nanoclaw.session.webui_turns import (
    clean_generated_title,
    maybe_generate_webui_title,
    maybe_generate_webui_title_after_turn,
)
from nanoclaw.webui.history import WebUIHistoryStore, fact_from_record, record_from_fact, webui_history_dir
from nanoclaw.webui.transcript import (
    WebUITranscriptRecorder,
    build_session_thread_response,
    build_webui_thread_payload,
    build_webui_thread_response,
    session_backfill_history,
)


def _manager(tmp_path: Path) -> SessionManager:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return SessionManager(workspace, sessions_root=tmp_path / "sessions")


# --- storage -----------------------------------------------------------------------


def test_store_layout_sits_beside_sessions_per_workspace(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    store = WebUIHistoryStore.for_sessions(manager)
    assert webui_history_dir(manager) == tmp_path / "webui" / manager.sessions_dir.name
    store.append_records("websocket:c1", [{"event": "user", "text": "hi", "created_at_ms": 5}])
    path = store.path("websocket:c1")
    assert path.parent == store.directory
    assert path.name == f"{SessionManager.storage_key('websocket:c1')}.jsonl"
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"event": "user", "chat_id": "c1", "text": "hi", "created_at_ms": 5},
    ]
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_display_facts_survive_reopen(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    key = "websocket:durable"
    facts = [
        HistoryFact(HistoryKind.INPUT, {"text": "question"}, "turn-1", "user", 1, 1000),
        HistoryFact(HistoryKind.REASONING_SEGMENT, {"text": "reasoning"}, "turn-1", "reasoning", 2, 1001),
        HistoryFact(HistoryKind.FILE_ACTIVITY, {"edits": [{"path": "a.py", "diff": "-old\n+new"}]}, "turn-1", "activity", 3, 1002),
        HistoryFact(HistoryKind.ANSWER_SEGMENT, {"text": "answer", "response_sources": [{
            "provider": "test", "model": "actual-fallback", "preset": "default", "fallback": True,
        }]}, "turn-1", "answer", 4, 1003),
        HistoryFact(HistoryKind.COMPLETION, {"usage": {"total_tokens": 30}, "round_usages": [{"total_tokens": 30}],
                    "latency_ms": 40, "outcome": "failed", "failure_message": "model unavailable"},
                    "turn-1", "complete", 5, 1004),
    ]
    WebUIHistoryStore.for_sessions(manager).append(key, facts)

    reopened = WebUIHistoryStore.for_sessions(_manager(tmp_path))
    history = reopened.reader(key)
    assert history.all_facts() == facts
    payload = build_webui_thread_response(key, history=history)
    assert payload["completed_turn_ids"] == ["turn-1"]
    terminal = next(event for event in payload["events"] if event["event"] == "turn_end")
    assert terminal["failure_message"] == "model unavailable" and terminal["outcome"] == "failed"
    assert terminal["round_usages"] == [{"total_tokens": 30}]
    assert any(event.get("edits") == facts[2].content["edits"] for event in payload["events"])
    assert next(event for event in payload["events"] if event["event"] == "stream_end")["response_sources"] == \
        facts[3].content["response_sources"]


def test_message_projection_adapter_preserves_reply_and_agent_activity() -> None:
    record = {
        "event": "message", "chat_id": "activity", "kind": "progress",
        "text": "Inspecting workspace", "reply_to": "message-1", "created_at_ms": 1000,
        "agent_ui": {"version": 1, "status": "working", "label": "Inspecting workspace"},
    }
    fact = HistoryFact.decode(fact_from_record(record).encode())
    assert record_from_fact(fact, "websocket:activity") == record


def test_invalid_records_are_rejected_before_writing(tmp_path: Path) -> None:
    store = WebUIHistoryStore(tmp_path / "webui")
    with pytest.raises(ValueError):
        store.append_records("websocket:c", [{"event": "user", "text": "ok"}, {"event": "bogus"}])
    assert not store.exists("websocket:c")
    with pytest.raises(ValueError):
        store.append_records("websocket:c", [{"event": "message", "kind": "secret"}])


def test_unreadable_lines_are_skipped(tmp_path: Path) -> None:
    store = WebUIHistoryStore(tmp_path / "webui")
    store.append_records("websocket:c", [{"event": "user", "text": "one"}])
    with store.path("websocket:c").open("a") as handle:
        handle.write("{not json\n[1]\n" + json.dumps({"event": "nope"}) + "\n")
    store.append_records("websocket:c", [{"event": "turn_end"}])
    assert [fact.kind for fact in store.reader("websocket:c").all_facts()] == [HistoryKind.INPUT, HistoryKind.COMPLETION]


def test_symlinked_history_is_refused(tmp_path: Path) -> None:
    store = WebUIHistoryStore(tmp_path / "webui")
    store.directory.mkdir(parents=True)
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("")
    store.path("websocket:c").symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        store.append_records("websocket:c", [{"event": "user", "text": "x"}])
    assert target.read_text() == ""


def test_revision_changes_with_every_append_and_is_none_without_a_file(tmp_path: Path) -> None:
    store = WebUIHistoryStore(tmp_path / "webui")
    assert store.reader("websocket:c").revision is None
    store.append_records("websocket:c", [{"event": "user", "text": "x"}])
    first = store.reader("websocket:c").revision
    store.append_records("websocket:c", [{"event": "turn_end"}])
    second = store.reader("websocket:c").revision
    assert first is not None and second is not None and first != second
    assert store.delete("websocket:c") is True
    assert store.delete("websocket:c") is False
    assert store.reader("websocket:c").revision is None


# --- recorder ----------------------------------------------------------------------


async def test_recorder_keeps_completed_segments_not_every_transport_delta(tmp_path: Path) -> None:
    store = WebUIHistoryStore(tmp_path / "webui")
    recorder = WebUITranscriptRecorder(store)
    metadata = {"webui_turn_id": "turn"}
    for _ in range(8):
        assert await recorder.prepare_and_append_stream_event(
            "stream", {"event": "delta", "text": "x"}, completed_text=None, metadata=metadata, phase="answer")
    assert await recorder.prepare_and_append_stream_event(
        "stream", {"event": "stream_end"}, completed_text="x" * 8, metadata=metadata, phase="answer")
    facts = store.reader("websocket:stream").all_facts()
    assert [fact.kind for fact in facts] == [HistoryKind.ANSWER_SEGMENT]
    assert facts[0].content["text"] == "x" * 8
    assert (facts[0].turn_id, facts[0].phase, facts[0].sequence) == ("turn", "answer", 9)


async def test_recorder_round_trip_of_a_hand_written_turn(tmp_path: Path) -> None:
    store = WebUIHistoryStore(tmp_path / "webui")
    recorder = WebUITranscriptRecorder(store)
    meta = {"webui_turn_id": "t1", "response_sources": [{"provider": "p", "model": "m", "preset": "default"}]}
    assert await recorder.append_user_message("c1", "hello", metadata=meta) is True
    assert await recorder.append_user_message("c1", "/stop", metadata=meta) is False
    await recorder.prepare_and_append("c1", {"event": "message", "chat_id": "c1", "kind": "tool_hint",
                                             "text": 'read_file("a")', "tool_events": [
                                                 {"phase": "start", "name": "read_file", "call_id": "1"}]},
                                      metadata=meta, phase="activity")
    await recorder.prepare_and_append_stream_event("c1", {"event": "stream_end", "chat_id": "c1"},
                                                   completed_text="Hi there", metadata=meta, phase="answer",
                                                   include_source=True)
    await recorder.prepare_and_append("c1", {"event": "turn_end", "chat_id": "c1", "latency_ms": 12},
                                      metadata=meta, phase="complete")

    payload = build_webui_thread_response("websocket:c1", history=store.reader("websocket:c1"))
    assert payload["completed_turn_ids"] == ["t1"]
    assert payload["has_pending_tool_calls"] is False
    assert [(e["event"], e.get("turn_phase"), e.get("turn_seq")) for e in payload["events"]] == [
        ("user_message", "user", 1), ("message", "activity", 2), ("stream_end", "answer", 3),
        ("turn_end", "complete", 4),
    ]
    assert payload["events"][0]["starts_turn"] is True
    assert payload["events"][2]["text"] == "Hi there"
    assert payload["events"][2]["response_sources"] == [{"provider": "p", "model": "m", "preset": "default",
                                                         "fallback": False}]
    assert payload["events"][3]["latency_ms"] == 12
    assert payload["page"]["user_message_offset"] == 0


async def test_recorder_without_store_is_transport_only() -> None:
    assert await WebUITranscriptRecorder(None).append("c", {"event": "user", "text": "x"}, metadata={}) is True


# --- backfill ----------------------------------------------------------------------


def _backfill_session(manager: SessionManager, key: str = "websocket:legacy"):
    session = manager.get_or_create(key)
    session.created_at = datetime(2026, 1, 1, 9, 0, 0)
    session.messages = [
        {"role": "user", "content": "old summary marker", HIDDEN_HISTORY_META: True},
        {"role": "assistant", "content": SUMMARY_CONTINUATION_TEXT, HIDDEN_HISTORY_META: True},
        {"role": "user", "content": "list files", "timestamp": "2026-01-01T09:00:00"},
        {"role": "assistant", "content": "Looking.", "reasoning_content": "think first",
         "tool_calls": [{"id": "1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}],
         "tool_events": [{"phase": "end", "name": "exec", "call_id": "1", "arguments": {"command": "ls"}},
                         {"phase": "end", "name": "exec", "call_id": "2", "result": "data:image/png;base64,AAAA"}],
         "file_edit_events": [{"call_id": "1", "path": "a.py", "added": 1, "deleted": 0}],
         "timestamp": "2026-01-01T09:00:01"},
        {"role": "tool", "tool_call_id": "1", "content": "a.py"},
        {"role": "assistant", "content": "One file: a.py", "timestamp": "2026-01-01T09:00:02"},
        {"role": "user", "content": "thanks", "timestamp": "2026-01-01T09:01:00"},
        {"role": "assistant", "content": "You're welcome", "timestamp": "2026-01-01T09:01:01"},
    ]
    manager.save(session)
    return session


def test_payload_backfills_from_session_messages_when_no_log_exists(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    store = WebUIHistoryStore.for_sessions(manager)
    session = _backfill_session(manager)

    payload = build_webui_thread_payload(session.key, store=store, sessions=manager)

    assert payload is not None
    kinds = [(e["event"], e.get("kind")) for e in payload["events"]]
    assert kinds == [
        ("user_message", None), ("message", "reasoning"), ("message", "progress"), ("message", "tool_hint"),
        ("file_edit", None), ("message", None), ("turn_end", None),
        ("user_message", None), ("message", None), ("turn_end", None),
    ]
    tool_hint = payload["events"][3]
    assert tool_hint["text"] == 'exec({"command": "ls"})\nexec()'
    assert tool_hint["tool_events"][1]["result"] == "[binary content omitted from WebUI]"
    assert payload["completed_turn_ids"] == [f"session:{session.key}:2", f"session:{session.key}:6"]
    assert payload["has_pending_tool_calls"] is False
    assert payload["events"][0]["created_at_ms"] == int(datetime(2026, 1, 1, 9).timestamp() * 1000)
    # Same rows as nanobot's session thread projection, except the per-turn ends.
    session_view = build_session_thread_response(session)["events"]
    strip = {"projection_id", "created_at_ms", "turn_phase", "latency_ms"}
    assert [{k: v for k, v in e.items() if k not in strip} for e in payload["events"] if e["event"] != "turn_end"] == \
        [{k: v for k, v in e.items() if k not in strip} for e in session_view if e["event"] != "turn_end"]
    assert not store.exists(session.key)  # backfill is a read-time projection


def test_backfill_pages_by_turn(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    store = WebUIHistoryStore.for_sessions(manager)
    session = _backfill_session(manager)
    latest = build_webui_thread_payload(session.key, store=store, sessions=manager, limit=2)
    assert [e["text"] for e in latest["events"] if e["event"] == "user_message"] == ["thanks"]
    assert latest["page"]["has_more_before"] is True and latest["page"]["user_message_offset"] == 1
    older = build_webui_thread_payload(session.key, store=store, sessions=manager, limit=2,
                                       before=latest["page"]["before_cursor"])
    assert [e["text"] for e in older["events"] if e["event"] == "user_message"] == ["list files"]
    assert older["page"]["has_more_before"] is False


def test_backfill_of_a_running_turn_leaves_it_open(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:running")
    session.add_message("user", "go")
    manager.save(session)
    history = session_backfill_history(session, active=True)
    payload = build_webui_thread_response(session.key, history=history, active_turn_started_at=1.0,
                                          active_turn_id="live")
    assert [e["event"] for e in payload["events"]] == ["user_message"]
    assert payload["has_pending_tool_calls"] is True
    assert payload["active_turn_id"] == "live"


def test_recorded_log_wins_over_backfill_and_missing_everything_is_404(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    store = WebUIHistoryStore.for_sessions(manager)
    session = _backfill_session(manager)
    store.append_records(session.key, [{"event": "user", "text": "recorded"}, {"event": "turn_end"}])
    payload = build_webui_thread_payload(session.key, store=store, sessions=manager)
    assert [e["text"] for e in payload["events"] if e["event"] == "user_message"] == ["recorded"]
    assert build_webui_thread_payload("websocket:none", store=store, sessions=manager) is None
    empty = manager.get_or_create("websocket:empty")
    manager.save(empty)
    assert build_webui_thread_payload("websocket:empty", store=store, sessions=manager) is None


def test_payload_revision_tracks_appends_and_variants(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    store = WebUIHistoryStore.for_sessions(manager)
    key = "websocket:rev"
    store.append_records(key, [{"event": "user", "text": "a"}])
    first = build_webui_thread_payload(key, store=store, sessions=manager, revision_variant={"limit": None})
    again = build_webui_thread_payload(key, store=store, sessions=manager, revision_variant={"limit": None})
    other = build_webui_thread_payload(key, store=store, sessions=manager, revision_variant={"limit": 5})
    assert first["revision"] == again["revision"] != other["revision"]
    store.append_records(key, [{"event": "turn_end"}])
    assert build_webui_thread_payload(key, store=store, sessions=manager,
                                      revision_variant={"limit": None})["revision"] != first["revision"]
    assert "revision" not in build_webui_thread_payload(key, store=store, sessions=manager)


# --- titles ------------------------------------------------------------------------


class _TitleProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    async def chat_stream_with_retry(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return SimpleNamespace(content=self.content, finish_reason="stop")


def test_clean_generated_title() -> None:
    assert clean_generated_title('Title: "Fixing the build."') == "Fixing the build"
    assert clean_generated_title("标题：<think>x</think>部署 计划。") == "部署 计划"
    assert len(clean_generated_title("word " * 40)) == 60


async def test_title_is_generated_once_for_webui_sessions(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.get_or_create("websocket:t")
    session.metadata["webui"] = True
    session.add_message("user", "How do I deploy?")
    session.add_message("assistant", "Use the script.")
    manager.save(session)
    provider = _TitleProvider("Deploying with the script.")

    assert await maybe_generate_webui_title_after_turn(
        channel="websocket", chat_id="t", metadata={"webui": True}, sessions=manager,
        session_key="websocket:t", provider=provider, model="m") is True
    assert manager.read_session_metadata("websocket:t")["metadata"]["title"] == "Deploying with the script"
    assert "User: How do I deploy?" in provider.calls[0]["messages"][1]["content"]
    assert await maybe_generate_webui_title(sessions=manager, session_key="websocket:t",
                                            provider=provider, model="m") is False
    assert len(provider.calls) == 1


async def test_title_respects_user_edits_and_non_webui_sessions(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    provider = _TitleProvider("New title")
    edited = manager.get_or_create("websocket:edited")
    edited.metadata.update({"webui": True, "title": "Mine", "title_user_edited": True})
    edited.add_message("user", "hi")
    manager.save(edited)
    plain = manager.get_or_create("cli:direct")
    plain.add_message("user", "hi")
    manager.save(plain)
    assert await maybe_generate_webui_title(sessions=manager, session_key="websocket:edited",
                                            provider=provider, model="m") is False
    assert await maybe_generate_webui_title(sessions=manager, session_key="cli:direct",
                                            provider=provider, model="m") is False
    assert await maybe_generate_webui_title_after_turn(
        channel="telegram", chat_id="1", metadata={"webui": True}, sessions=manager,
        session_key="cli:direct", provider=provider, model="m") is False
    assert provider.calls == []
