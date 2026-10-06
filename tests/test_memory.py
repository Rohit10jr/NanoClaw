"""Memory store, history journal, archiving and compaction (nanobot agent/memory.py)."""

import json
from datetime import datetime

import pytest

from conftest import REAL_GET_TOKEN_ENCODING
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.memory import _ARCHIVE_TOOL_RESULT, Consolidator, MemoryArchiver, MemoryStore
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.session import Session, SessionManager
from nanoclaw.session.summary import (
    SUMMARY_CONTINUATION_TEXT,
    is_summary_checkpoint,
    session_summary_from_metadata,
)
from nanoclaw.utils import tokens
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.templates import load_bundled_template, sync_workspace_templates


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    sync_workspace_templates(path)
    return path


@pytest.fixture
def store(ws):
    return MemoryStore(ws)


def runtime(provider, window=128_000, max_tokens=1000):
    return LLMRuntime(provider=provider, model=None, max_tokens=max_tokens, temperature=0.1, context_window_tokens=window)


# --- history journal ------------------------------------------------------------------------


def test_append_history_numbers_entries(store):
    assert store.append_history("first", session_key="cli:direct") == 1
    assert store.append_history("second") == 2
    lines = [json.loads(line) for line in store.history_file.read_text().splitlines()]
    assert [e["cursor"] for e in lines] == [1, 2]
    assert lines[0]["session_key"] == "cli:direct" and "session_key" not in lines[1]
    datetime.strptime(lines[0]["timestamp"], "%Y-%m-%d %H:%M")
    assert store._cursor_file.read_text() == "2"


def test_append_history_cleans_and_caps(store):
    store.append_history("<think>hidden</think>visible  \n")
    store.append_history("x" * 100, max_chars=10)
    contents = [e["content"] for e in store.read_unprocessed_history(0)]
    assert contents == ["visible", "x" * 10 + "\n... (truncated)"]


def test_cursor_keeps_counting_after_the_journal_is_trimmed(store):
    for i in range(3):
        store.append_history(f"e{i}")
    store._write_entries([])  # journal emptied, counter file remains
    assert store.append_history("next") == 4


def test_invalid_entries_are_ignored(store):
    store.history_file.write_text(
        "\n".join([
            json.dumps({"cursor": 1, "timestamp": "t", "content": "ok"}),
            "{broken",
            json.dumps({"cursor": True, "timestamp": "t", "content": "bool cursor"}),
            json.dumps({"cursor": 3, "timestamp": "t"}),
            json.dumps({"cursor": 4, "timestamp": "t", "content": "ok too", "session_key": 5}),
            json.dumps({"cursor": 5, "timestamp": "t", "content": "last"}),
        ]) + "\n"
    )
    assert [e["content"] for e in store.read_unprocessed_history(0)] == ["ok", "last"]
    assert [e["content"] for e in store.read_unprocessed_history(1)] == ["last"]


def test_compact_history_never_drops_unread_entries(ws):
    store = MemoryStore(ws, max_history_entries=3)
    for i in range(6):
        store.append_history(f"e{i + 1}")
    store.set_last_dream_cursor(1)  # Dream has read only entry 1
    store.compact_history()
    assert [e["cursor"] for e in store._read_entries()] == [2, 3, 4, 5, 6]
    store.set_last_dream_cursor(6)
    store.compact_history()
    assert [e["cursor"] for e in store._read_entries()] == [4, 5, 6]


def test_dream_cursor(store):
    assert store.get_last_dream_cursor() == 0
    store.set_last_dream_cursor(7)
    assert store.get_last_dream_cursor() == 7


# --- Dream prompt and tools -------------------------------------------------------------------


def test_dream_prompt(store):
    assert store.build_dream_prompt() is None
    for i in range(25):
        store.append_history(f"fact {i}" + ("y" * 2000 if i == 0 else ""))
    prompt, last = store.build_dream_prompt()
    assert prompt.startswith("You are running Dream.")
    assert "## Conversation History\n[" in prompt
    assert last == 20  # at most 20 entries per run
    assert "fact 19" in prompt and "fact 20" not in prompt
    assert "y" * 1001 not in prompt  # each entry cut at 1000 chars
    store.set_last_dream_cursor(20)
    assert store.build_dream_prompt()[1] == 25


async def test_dream_tools_write_only_memory_files(ws, store, tmp_path):
    tools = store.build_dream_tools()
    assert sorted(tools.tool_names) == ["edit_file", "read_file", "write_file"]
    ok = await tools.execute("write_file", {"path": "memory/MEMORY.md", "content": "# Facts\n- likes tea\n"})
    assert not getattr(ok, "is_error", False)
    assert (ws / "memory/MEMORY.md").read_text() == "# Facts\n- likes tea\n"
    assert not getattr(await tools.execute("write_file", {"path": "USER.md", "content": "Name: R"}), "is_error", False)
    refused = await tools.execute("write_file", {"path": "notes.txt", "content": "x"})
    assert refused.is_error
    assert refused.startswith(f"Error: Path notes.txt is outside allowed directory {ws.resolve() / 'skills'} (this is a hard policy boundary")
    # Since layer 7 Dream may write skills, and read the built-in ones.
    assert not getattr(await tools.execute("write_file", {"path": "skills/tea/SKILL.md", "content": "x"}), "is_error", False)
    assert "name: weather" in await tools.execute("read_file", {"path": "skills/weather/SKILL.md"})
    assert not (ws / "notes.txt").exists()
    refused = await tools.execute("edit_file", {"path": "AGENTS.md", "old_text": "a", "new_text": "b"})
    assert refused.is_error
    outside = tmp_path / "secret.txt"
    outside.write_text("s")
    refused = await tools.execute("read_file", {"path": str(outside)})
    assert refused.is_error and "outside allowed directory" in refused
    assert "likes tea" in await tools.execute("read_file", {"path": "memory/MEMORY.md"})


# --- raw fallback ------------------------------------------------------------------------------


def test_raw_archive_journals_chunks_without_runtime_context(store):
    marker = {"version": 1, "sources": ["time"], "suffix": "[T]"}
    messages = [
        {"role": "user", "content": "hello\n\n[T]", "_runtime_context": marker, "timestamp": "2026-10-01T10:00:00"},
        {"role": "assistant", "content": "a" * 50, "timestamp": "2026-10-01T10:01:00"},
    ]
    checkpoint = store.raw_archive(messages, max_chars=40, session_key="k")
    entries = store.read_unprocessed_history(0)
    assert len(entries) > 1 and entries[0]["content"].startswith(f"[RAW] 2 messages (part 1/{len(entries)})")
    joined = "".join(e["content"].split("\n", 1)[1] for e in entries)
    assert joined.startswith("[2026-10-01T10:00] USER: hello\n[2026-10-01T10:01] ASSISTANT: aaa")
    assert "[T]" not in joined
    assert checkpoint.startswith("[RAW] 2 messages\n") and checkpoint.endswith("... (truncated)")


def test_combine_raw_checkpoint_keeps_both_sides():
    combine = MemoryArchiver._combine_raw_checkpoint
    assert combine("RAW", previous_summary=None, max_tokens=100) == "RAW"
    assert combine("RAW", previous_summary="OLD", max_tokens=1000) == (
        "[Previous archived context]\nOLD\n\n[Newly archived raw context]\nRAW"
    )
    tight = combine("R" * 500, previous_summary="O" * 500, max_tokens=200)
    assert "O" in tight and "R" in tight and len(tight.encode()) <= 200


# --- archiving with the model --------------------------------------------------------------------


def archiver(ws, store):
    return MemoryArchiver(store, ContextBuilder(ws).build_messages, lambda: [])


SOURCE = [{"role": "user", "content": "I like tea"}, {"role": "assistant", "content": "Noted."}]
HISTORY = [{"role": "system", "content": "SYS"}, *SOURCE]


async def test_archive_asks_the_model_and_journals_the_summary(ws, store):
    fake = FakeProvider(["- [permanent] likes tea"])
    summary = await archiver(ws, store).archive(SOURCE, runtime=runtime(fake), session_key="k", history=HISTORY, request_tools=[])
    assert summary == "- [permanent] likes tea"
    sent = fake.calls[0].messages
    assert sent[:3] == HISTORY
    assert sent[-1] == {"role": "user", "content": load_bundled_template("agent/consolidator_archive.md").rstrip()}
    assert [e["content"] for e in store.read_unprocessed_history(0)] == ["- [permanent] likes tea"]


async def test_nothing_summary_is_not_journaled(ws, store):
    summary = await archiver(ws, store).archive(
        SOURCE, runtime=runtime(FakeProvider(["(nothing)"])), session_key="k", history=HISTORY, request_tools=[]
    )
    assert summary == "(nothing)" and store.read_unprocessed_history(0) == []


@pytest.mark.parametrize(
    "reply",
    [
        LLMResponse(content="Error calling LLM: boom", finish_reason="error"),
        LLMResponse(content="cut off", finish_reason="length"),
        LLMResponse(content="   "),
    ],
)
async def test_failed_archive_falls_back_to_raw(ws, store, reply):
    summary = await archiver(ws, store).archive(
        SOURCE, runtime=runtime(FakeProvider([reply])), session_key="k", history=HISTORY, request_tools=[]
    )
    assert summary.startswith("[RAW] 2 messages")
    assert store.read_unprocessed_history(0)[0]["content"].startswith("[RAW] 2 messages")


async def test_tool_calls_are_refused_once_then_raw(ws, store):
    call = LLMResponse(content=None, tool_calls=[ToolCallRequest(id="c1", name="read_file", arguments={"path": "x"})], finish_reason="tool_calls")
    fake = FakeProvider([call, "- [durable] summary"])
    summary = await archiver(ws, store).archive(SOURCE, runtime=runtime(fake), session_key="k", history=HISTORY, request_tools=[{"x": 1}])
    assert summary == "- [durable] summary"
    assert fake.calls[1].messages[-1] == {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": _ARCHIVE_TOOL_RESULT}
    fake = FakeProvider([call, call])
    summary = await archiver(ws, store).archive(SOURCE, runtime=runtime(fake), session_key="k", history=HISTORY, request_tools=[{"x": 1}])
    assert summary.startswith("[RAW]")


async def test_archive_that_does_not_fit_goes_raw_without_calling_the_model(ws, store):
    fake = FakeProvider(["unused"])
    summary = await archiver(ws, store).archive(
        SOURCE, runtime=runtime(fake), session_key="k", history=HISTORY, request_tools=[], input_token_budget=10
    )
    assert summary.startswith("[RAW]") and fake.calls == []


# --- session checkpoints ------------------------------------------------------------------------


def test_commit_summary_checkpoint_and_history():
    s = Session("k", messages=[{"role": "user", "content": "old"}, {"role": "assistant", "content": "a"}])
    s.commit_summary_checkpoint("- summary")
    assert s.last_archived == 2
    assert is_summary_checkpoint(s.messages[2]) and s.messages[2]["content"] == SUMMARY_CONTINUATION_TEXT
    assert s.metadata["_last_summary"]["text"] == "- summary"
    assert s.get_history() == []
    s.add_message("user", "new")
    assert s.get_history() == [{"role": "user", "content": "new"}]
    s.clear()
    assert "_last_summary" not in s.metadata


def test_history_skips_command_messages():
    s = Session("k", messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "x", "_command": True}])
    assert s.get_history() == [{"role": "user", "content": "q"}]


def test_history_token_limit_keeps_recent_messages_from_a_user_turn():
    msgs = [
        {"role": "user", "content": "u1" * 50},
        {"role": "assistant", "content": "a1" * 50},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    s = Session("k", messages=msgs)
    assert [m["content"] for m in s.get_history(max_tokens=30)] == ["u2", "a2"]
    # Nothing but an assistant message fits: the nearest user message is recovered.
    s = Session("k", messages=[{"role": "user", "content": "u"}, {"role": "assistant", "content": "a" * 200}])
    assert [m["role"] for m in s.get_history(max_tokens=5)] == ["user", "assistant"]


def test_session_summary_from_metadata():
    now = datetime(2026, 10, 1, 12, 0)
    assert session_summary_from_metadata({}, fallback_last_active=now) is None
    assert session_summary_from_metadata({"_last_summary": {"text": ""}}, fallback_last_active=now) is None
    good = {"_last_summary": {"text": "s", "last_active": "2026-09-30T08:00:00"}}
    assert session_summary_from_metadata(good, fallback_last_active=now) == {"text": "s", "last_active": "2026-09-30T08:00:00"}
    bad_time = {"_last_summary": {"text": "s", "last_active": "yesterday"}}
    assert session_summary_from_metadata(bad_time, fallback_last_active=now)["last_active"] == now.isoformat()


# --- Consolidator ---------------------------------------------------------------------------------


def consolidator(ws, tmp_path, store):
    sessions = SessionManager(ws, sessions_root=tmp_path / "sessions")
    return Consolidator(store, sessions, ContextBuilder(ws).build_messages, lambda: []), sessions


async def test_compact_idle_session(ws, tmp_path, store):
    cons, sessions = consolidator(ws, tmp_path, store)
    s = sessions.get_or_create("cli:direct")
    s.add_message("user", "I like tea")
    s.add_message("assistant", "Noted.")
    sessions.save(s)
    fake = FakeProvider(["- [permanent] likes tea"])
    assert await cons.compact_idle_session("cli:direct", runtime=runtime(fake)) == "- [permanent] likes tea"
    # The archive request is the system prompt, the history, then the archive prompt.
    roles = [m["role"] for m in fake.calls[0].messages]
    assert roles == ["system", "user", "assistant", "user"]
    reloaded = SessionManager(ws, sessions_root=tmp_path / "sessions").get_or_create("cli:direct")
    assert reloaded.last_archived == 2 and reloaded.metadata["_last_summary"]["text"] == "- [permanent] likes tea"
    assert await cons.compact_idle_session("cli:direct", runtime=runtime(fake)) == ""  # nothing new


async def test_summarize_transcript_caps_the_summary(ws, tmp_path, store):
    cons, _ = consolidator(ws, tmp_path, store)
    fake = FakeProvider(["z" * 500])
    summary = await cons.summarize_transcript(HISTORY, None, runtime=runtime(fake, max_tokens=50), session_key="k", tools=[])
    assert len(summary.encode()) <= 50 and summary.endswith("... (truncated)")
    assert await cons.summarize_transcript([{"role": "system", "content": "S"}], None, runtime=runtime(fake), session_key="k", tools=[]) is None


# --- system prompt ---------------------------------------------------------------------------------


def test_memory_section_only_when_edited(ws):
    builder = ContextBuilder(ws)
    assert "# Memory" not in builder.build_system_prompt()
    (ws / "memory/MEMORY.md").write_text("- likes tea\n")
    prompt = builder.build_system_prompt()
    assert "---\n\n# Memory\n\n## Long-term Memory\n- likes tea\n\n\n---\n\n# Skills\n" in prompt
    assert prompt.index("# Tool Usage Notes") < prompt.index("# Memory")


def test_summary_section(ws):
    builder = ContextBuilder(ws)
    prompt = builder.build_system_prompt(session_summary={"text": "- s", "last_active": "2026-10-01T10:00:00"})
    assert prompt.endswith(
        "[Archived Context Summary]\n\nPrevious conversation summary (last active 2026-10-01T10:00:00):\n- s"
    )
    assert "Archived" not in builder.build_system_prompt(session_summary={"text": "(nothing)", "last_active": "x"})


def test_build_messages_without_a_new_message(ws):
    msgs = ContextBuilder(ws).build_messages([{"role": "user", "content": "q"}], None)
    assert [m["role"] for m in msgs] == ["system", "user"]


# --- token counting ----------------------------------------------------------------------------------


def test_byte_estimates():
    msgs = [{"role": "user", "content": "héllo"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]}]
    payload = "héllo\n" + json.dumps([{"id": "c"}])
    assert tokens.estimate_prompt_tokens(msgs) == (len(payload.encode()) + 8, "heuristic")
    tools = [{"name": "t"}]
    assert tokens.estimate_prompt_tokens(msgs, tools)[0] == len((payload + "\n" + json.dumps(tools)).encode()) + 8
    assert tokens.estimate_message_tokens({"role": "user", "content": ""}) == 4
    assert tokens.truncate_text_to_tokens("abc", 10) == "abc"
    cut = tokens.truncate_text_to_tokens("x" * 100, 30)
    assert cut.endswith(tokens.TRUNCATED_SUFFIX) and len(cut.encode()) <= 30


def test_tiktoken_estimates_when_available(monkeypatch):
    monkeypatch.setattr(tokens, "get_token_encoding", REAL_GET_TOKEN_ENCODING)
    if REAL_GET_TOKEN_ENCODING() is None:
        pytest.skip("tiktoken vocabulary not available offline")
    count, source = tokens.estimate_prompt_tokens([{"role": "user", "content": "hello world"}])
    assert (count, source) == (2 + 4, "tiktoken")
    cut = tokens.truncate_text_to_tokens("word " * 200, 20)
    assert cut.endswith(tokens.TRUNCATED_SUFFIX) and len(REAL_GET_TOKEN_ENCODING().encode_ordinary(cut)) <= 20


async def test_archive_session_dumps_raw_when_the_stretch_does_not_fit(ws, tmp_path, store):
    cons, sessions = consolidator(ws, tmp_path, store)
    s = sessions.get_or_create("cli:direct")
    s.add_message("user", "u" * 3000)
    s.add_message("assistant", "a" * 3000)
    sessions.save(s)
    fake = FakeProvider(["unused"])
    # Budget = window - max_tokens - 1024 = 2000: positive, but the 6000-byte stretch can't be replayed.
    summary = await cons.compact_idle_session("cli:direct", runtime=runtime(fake, window=4024, max_tokens=1000))
    assert summary.startswith("[RAW] 2 messages") and fake.calls == []


async def test_archive_session_never_summarizes_only_part_of_the_stretch(ws, tmp_path, store):
    cons, sessions = consolidator(ws, tmp_path, store)
    s = sessions.get_or_create("cli:direct")
    s.add_message("user", "u" * 8000)
    s.add_message("assistant", "a" * 8000)
    s.add_message("user", "recent")
    s.add_message("assistant", "ok")
    sessions.save(s)
    fake = FakeProvider(["- partial summary"])
    # Budget 12000: the two recent messages (plus system prompt and archive
    # prompt) would fit, the old ones wouldn't. Summarizing only the recent
    # part would silently lose the old messages, so a raw dump is used.
    summary = await cons.compact_idle_session("cli:direct", runtime=runtime(fake, window=12000 + 1000 + 1024, max_tokens=1000))
    assert summary.startswith("[RAW] 4 messages") and fake.calls == []
