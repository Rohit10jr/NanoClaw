"""Session (history replay) and SessionManager (JSONL storage)."""

import json
import os
from pathlib import Path

import pytest

from nanoclaw.session import Session, SessionManager
from nanoclaw.session.manager import find_legal_message_start


def assistant_call(call_id, content=""):
    return {"role": "assistant", "content": content, "tool_calls": [{"id": call_id, "type": "function"}]}


def tool(call_id, content="out"):
    return {"role": "tool", "tool_call_id": call_id, "name": "echo", "content": content}


# --- Session ------------------------------------------------------------------------------


def test_add_message_stamps_time_and_keeps_extras():
    s = Session("cli:direct")
    s.add_message("user", "hi", _runtime_context={"version": 1})
    assert s.messages[0]["role"] == "user"
    assert s.messages[0]["_runtime_context"] == {"version": 1}
    assert "timestamp" in s.messages[0]


def test_history_keeps_only_replay_keys():
    s = Session("k")
    s.add_message("user", "q", _runtime_context={"version": 1, "suffix": "x"})
    s.messages.append({**assistant_call("c1"), "timestamp": "t", "latency_ms": 5, "reasoning_content": "r"})
    s.messages.append({**tool("c1"), "timestamp": "t"})
    s.add_message("assistant", "done")
    assert s.get_history() == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function"}], "reasoning_content": "r"},
        {"role": "tool", "content": "out", "tool_call_id": "c1", "name": "echo"},
        {"role": "assistant", "content": "done"},
    ]


def test_history_starts_at_first_user_message():
    s = Session("k", messages=[{"role": "assistant", "content": "leftover"}, {"role": "user", "content": "q"}])
    assert s.get_history() == [{"role": "user", "content": "q"}]


def test_history_without_any_user_message_is_kept():
    s = Session("k", messages=[{"role": "assistant", "content": "a"}])
    assert s.get_history() == [{"role": "assistant", "content": "a"}]


def test_history_drops_orphan_tool_results():
    s = Session("k", messages=[tool("gone"), {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    assert [m["role"] for m in s.get_history()] == ["user", "assistant"]
    # Orphans after the first user message cut the window just past them.
    s = Session("k", messages=[{"role": "user", "content": "q1"}, tool("gone"), {"role": "user", "content": "q2"}])
    assert s.get_history() == [{"role": "user", "content": "q2"}]


def test_find_legal_message_start():
    assert find_legal_message_start([assistant_call("a"), tool("a")]) == 0
    assert find_legal_message_start([tool("x"), assistant_call("a"), tool("a")]) == 1


def test_history_skips_empty_assistant_messages():
    s = Session("k", messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "  "},
        {"role": "assistant", "content": "", "reasoning_content": "thought"},
        {"role": "assistant", "content": None},
    ])
    out = s.get_history()
    # Blank text is skipped; reasoning is kept; None is not a string, so it stays (as in nanobot).
    assert out == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "reasoning_content": "thought"},
        {"role": "assistant", "content": None},
    ]


def test_history_starts_after_archived_messages():
    msgs = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "new"}]
    assert Session("k", messages=msgs, last_archived=2).get_history() == [{"role": "user", "content": "new"}]


@pytest.mark.parametrize("bad", [-1, 99, "2", True, None])
def test_out_of_range_archive_offset_is_reset(bad):
    assert Session("k", messages=[{"role": "user", "content": "q"}], last_archived=bad).last_archived == 0


def test_clear():
    s = Session("k", messages=[{"role": "user", "content": "q"}], last_archived=1)
    s.clear()
    assert s.messages == [] and s.last_archived == 0


# --- SessionManager -------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def root(tmp_path):
    return tmp_path / "sessions"


def test_save_and_load_round_trip(ws, root):
    mgr = SessionManager(ws, sessions_root=root)
    s = mgr.get_or_create("cli:direct")
    s.add_message("user", "héllo ✓")
    s.messages.append({**assistant_call("c1"), "timestamp": "t"})
    s.metadata["x"] = 1
    mgr.save(s)

    loaded = SessionManager(ws, sessions_root=root).get_or_create("cli:direct")
    assert loaded is not s
    assert loaded.messages == s.messages
    assert loaded.metadata == {"x": 1}
    assert loaded.created_at == s.created_at and loaded.updated_at == s.updated_at


def test_file_format_is_metadata_line_then_messages(ws, root):
    mgr = SessionManager(ws, sessions_root=root)
    s = mgr.get_or_create("cli:direct")
    s.add_message("user", "hi")
    s.add_message("assistant", "hello")
    mgr.save(s)
    path = mgr.get_session_path("cli:direct")
    assert path.name == "Y2xpOmRpcmVjdA.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["_type"] == "metadata"
    assert lines[0]["key"] == "cli:direct"
    assert lines[0]["last_archived"] == 0
    assert [line["content"] for line in lines[1:]] == ["hi", "hello"]


def test_unicode_is_stored_readably(ws, root):
    mgr = SessionManager(ws, sessions_root=root)
    s = mgr.get_or_create("k")
    s.add_message("user", "héllo ✓")
    mgr.save(s)
    assert "héllo ✓" in mgr.get_session_path("k").read_text(encoding="utf-8")


@pytest.mark.parametrize("key", ["telegram:-100123", "a/b\\c", "../../etc/passwd", "ключ:1"])
def test_any_key_gives_a_safe_file_name(ws, root, key):
    mgr = SessionManager(ws, sessions_root=root)
    path = mgr.get_session_path(key)
    assert path.parent == mgr.sessions_dir
    assert "/" not in path.name and "\\" not in path.name and ":" not in path.name


def test_save_leaves_no_temp_files(ws, root):
    mgr = SessionManager(ws, sessions_root=root)
    mgr.save(mgr.get_or_create("k"))
    assert [p.name for p in mgr.sessions_dir.iterdir()] == [mgr.get_session_path("k").name]


def test_corrupt_lines_are_skipped(ws, root):
    mgr = SessionManager(ws, sessions_root=root)
    s = mgr.get_or_create("k")
    s.add_message("user", "q")
    s.add_message("assistant", "a")
    mgr.save(s)
    path = mgr.get_session_path("k")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], "{not json", "[1, 2]", lines[1], lines[2][:10]]) + "\n", encoding="utf-8")
    loaded = SessionManager(ws, sessions_root=root).get_or_create("k")
    assert [m["content"] for m in loaded.messages] == ["q"]


def test_missing_session_is_new_and_not_written_until_saved(ws, root):
    mgr = SessionManager(ws, sessions_root=root)
    s = mgr.get_or_create("k")
    assert s.messages == [] and not mgr.get_session_path("k").exists()


def test_cache_returns_same_object_and_invalidate_reloads(ws, root):
    mgr = SessionManager(ws, sessions_root=root)
    s = mgr.get_or_create("k")
    assert mgr.get_or_create("k") is s
    s.add_message("user", "unsaved")
    mgr.invalidate("k")
    assert mgr.get_or_create("k").messages == []


def test_cache_evicts_least_recently_used(ws, root):
    mgr = SessionManager(ws, sessions_root=root, max_cached_sessions=2)
    a = mgr.get_or_create("a")
    mgr.get_or_create("b")
    mgr.get_or_create("a")  # a is now most recent
    mgr.get_or_create("c")  # evicts b
    assert set(mgr._cache) == {"a", "c"}
    assert mgr.get_or_create("a") is a


# --- where sessions live ----------------------------------------------------------------------


def test_default_location_is_outside_the_workspace(ws):
    mgr = SessionManager(ws)  # HOME is a temporary folder in tests (conftest.py)
    assert mgr.sessions_dir.parent == Path(os.path.expanduser("~/.nanoclaw/sessions")).resolve()
    assert not mgr.sessions_dir.is_relative_to(ws)


def test_storage_inside_the_workspace_is_refused(ws):
    with pytest.raises(RuntimeError, match="outside the agent workspace"):
        SessionManager(ws, sessions_root=ws / "sessions")
    with pytest.raises(RuntimeError, match="outside the agent workspace"):
        SessionManager(ws, sessions_root=ws)


def test_workspace_id_is_created_once_and_reused(ws, root):
    first = SessionManager(ws, sessions_root=root)
    marker = ws / ".nanoclaw" / "workspace-id"
    workspace_id = marker.read_text().strip()
    assert len(workspace_id) == 32
    assert first.sessions_dir == root.resolve() / workspace_id
    assert SessionManager(ws, sessions_root=root).sessions_dir == first.sessions_dir


def test_each_workspace_gets_its_own_folder(tmp_path, root):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = SessionManager(tmp_path / "a", sessions_root=root)
    b = SessionManager(tmp_path / "b", sessions_root=root)
    assert a.sessions_dir != b.sessions_dir


def test_history_follows_a_moved_workspace(tmp_path, root):
    old = tmp_path / "old"
    old.mkdir()
    mgr = SessionManager(old, sessions_root=root)
    s = mgr.get_or_create("k")
    s.add_message("user", "remember me")
    mgr.save(s)
    new = tmp_path / "new"
    old.rename(new)
    moved = SessionManager(new, sessions_root=root).get_or_create("k")
    assert moved.messages[0]["content"] == "remember me"


def test_invalid_workspace_id_is_an_error(ws, root):
    (ws / ".nanoclaw").mkdir()
    (ws / ".nanoclaw" / "workspace-id").write_text("not-an-id\n")
    with pytest.raises(RuntimeError, match="workspace identity marker is invalid"):
        SessionManager(ws, sessions_root=root)


def test_symlinked_workspace_id_is_refused(ws, root, tmp_path):
    (ws / ".nanoclaw").mkdir()
    target = tmp_path / "elsewhere"
    target.write_text("0" * 32 + "\n")
    (ws / ".nanoclaw" / "workspace-id").symlink_to(target)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        SessionManager(ws, sessions_root=root)


def test_sessions_root_is_private(ws, root):
    SessionManager(ws, sessions_root=root)
    assert (root.stat().st_mode & 0o777) == 0o700
