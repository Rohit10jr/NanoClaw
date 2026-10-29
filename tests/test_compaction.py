"""Compaction in the AgentLoop, /compact, /new archiving, Dream and the CLI."""

import re

import pytest

from nanoclaw.__main__ import main
from nanoclaw.agent.context_governance import merge_adjacent_user_messages_for_model
from nanoclaw.agent.loop import DREAM_NO_INPUT_MESSAGE, AgentLoop, ContextWindowExceededError
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.session.summary import is_summary_checkpoint
from nanoclaw.utils.templates import sync_workspace_templates
from nanoclaw.utils.tokens import estimate_prompt_tokens
from toy_tools import EchoTool

BIG = "x" * 3000


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    sync_workspace_templates(path)
    return path


def make_loop(ws, tmp_path, *script, window=0):
    tools = ToolRegistry()
    tools.register(EchoTool())
    fake = FakeProvider(list(script))
    loop = AgentLoop(
        fake,
        ws,
        tools=tools,
        sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"),
        max_tokens=100,
        context_window_tokens=window,
    )
    return loop, fake


def request_size(loop, history, text):
    messages = loop.context.build_messages(history, text, channel="cli")
    return estimate_prompt_tokens(merge_adjacent_user_messages_for_model(messages), loop.tools.get_definitions())[0]


# The next message is long enough that the archive request (history + the
# archive prompt, without this message) fits where the turn didn't.
THIRD = "third " + "y" * 2000


async def two_big_turns(ws, tmp_path, overflow=1):
    """A session with two long turns, and a window the next request overflows by ``overflow``.

    Like nanobot, the model is only asked for a summary when the archive
    request fits: its budget is the window minus the reply, 1024 more than the
    turn's. A larger overflow gives a raw checkpoint (docs/UPSTREAM_BUGS.md).
    """
    loop, _ = make_loop(ws, tmp_path, BIG, BIG)
    await loop.process_direct("first " + BIG)
    await loop.process_direct("second " + BIG)
    history = loop.sessions.get_or_create("cli:direct").get_history()
    full = request_size(loop, history, THIRD)
    # The time block varies by a few bytes between calls; overflow by a margin.
    return full - overflow - 50 + 100 + 1024


async def test_turn_compacts_when_the_request_does_not_fit(ws, tmp_path):
    window = await two_big_turns(ws, tmp_path)
    loop, fake = make_loop(ws, tmp_path, "- [ephemeral] two long messages", "answer", window=window)
    result = await loop.process_direct(THIRD)
    assert result.final_content == "answer"

    archive_call, turn_call = fake.calls
    # The archive request: system prompt + the old history + the archive prompt.
    assert [m["role"] for m in archive_call.messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert archive_call.messages[-1]["content"].startswith("Create a compact replacement checkpoint")
    assert archive_call.tools  # tool definitions are sent, as in nanobot
    # The turn: summary in the system prompt, then only the new message.
    assert [m["role"] for m in turn_call.messages] == ["system", "user"]
    assert "[Archived Context Summary]" in turn_call.messages[0]["content"]
    assert "- [ephemeral] two long messages" in turn_call.messages[0]["content"]
    assert turn_call.messages[1]["content"].startswith(THIRD)

    # Saved: old messages kept, then the marker, then this turn.
    session = SessionManager(ws, sessions_root=tmp_path / "sessions").get_or_create("cli:direct")
    roles = [m["role"] for m in session.messages]
    assert roles == ["user", "assistant", "user", "assistant", "user", "user", "assistant"]
    assert is_summary_checkpoint(session.messages[4]) and session.last_archived == 4
    assert session.metadata["_last_summary"]["text"] == "- [ephemeral] two long messages"
    assert [e["content"] for e in loop.memory.read_unprocessed_history(0)] == ["- [ephemeral] two long messages"]


async def test_next_turn_replays_from_the_summary(ws, tmp_path):
    window = await two_big_turns(ws, tmp_path)
    loop, fake = make_loop(ws, tmp_path, "- s1", "answer", "next answer", window=window)
    await loop.process_direct(THIRD)
    await loop.process_direct("fourth")
    sent = fake.calls[2].messages
    assert "- s1" in sent[0]["content"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"].startswith(THIRD) and sent[3]["content"].startswith("fourth")


async def test_no_compaction_when_the_request_fits(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "a", "b", window=1_000_000)
    await loop.process_direct("one")
    await loop.process_direct("two")
    assert len(fake.calls) == 2 and loop.memory.read_unprocessed_history(0) == []


async def test_window_zero_disables_compaction(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "a", window=0)
    await loop.process_direct("one " + BIG)
    assert len(fake.calls) == 1


async def test_second_compaction_updates_the_previous_summary(ws, tmp_path):
    window = await two_big_turns(ws, tmp_path)
    loop, fake = make_loop(ws, tmp_path, "- s1", BIG, "- s2", "ok", window=window)
    await loop.process_direct(THIRD)
    # A fourth message that just overflows the budget again.
    session = loop.sessions.get_or_create("cli:direct")
    summary = {"text": "- s1", "last_active": session.updated_at.isoformat()}
    messages = loop.context.build_messages(session.get_history(), "fourth ", channel="cli", session_summary=summary)
    base = estimate_prompt_tokens(merge_adjacent_user_messages_for_model(messages), loop.tools.get_definitions())[0]
    await loop.process_direct("fourth " + "z" * (loop._input_budget() - base + 60))
    archive2 = fake.calls[2].messages
    assert "[Archived Context Summary]" in archive2[0]["content"] and "- s1" in archive2[0]["content"]
    session = loop.sessions.get_or_create("cli:direct")
    assert session.metadata["_last_summary"]["text"] == "- s2"
    assert sum(is_summary_checkpoint(m) for m in session.messages) == 2


async def test_still_too_big_after_compaction_is_an_error(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "- s", window=100 + 1024 + 50)
    with pytest.raises(ContextWindowExceededError, match="exceeds the local context budget"):
        await loop.process_direct("hello")


async def test_large_overflow_gets_a_raw_checkpoint_like_nanobot(ws, tmp_path):
    window = await two_big_turns(ws, tmp_path, overflow=2000)
    loop, fake = make_loop(ws, tmp_path, "answer", window=window)
    assert (await loop.process_direct(THIRD)).final_content == "answer"
    assert len(fake.calls) == 1  # the model was never asked to summarize
    summary = loop.sessions.get_or_create("cli:direct").metadata["_last_summary"]["text"]
    assert summary.startswith("[RAW] 4 messages\n[")
    assert "[Archived Context Summary]" in fake.calls[0].messages[0]["content"]
    # The raw journal entry has the messages as the model saw them (no timestamps: "[?]").
    assert loop.memory.read_unprocessed_history(0)[0]["content"].startswith("[RAW] 4 messages\n[?] USER: first")


async def test_crashed_turn_does_not_commit_its_summary(ws, tmp_path):
    window = await two_big_turns(ws, tmp_path)
    loop, fake = make_loop(ws, tmp_path, "- s1", window=window)

    # Compaction now runs inside the runner, so the crash comes after it: in
    # the turn's model request (the archive request uses chat_with_retry).
    async def boom(**kwargs):
        raise RuntimeError("died")

    fake.chat_stream_with_retry = boom
    with pytest.raises(RuntimeError):
        await loop.process_direct(THIRD)
    session = SessionManager(ws, sessions_root=tmp_path / "sessions").get_or_create("cli:direct")
    assert not any(is_summary_checkpoint(m) for m in session.messages)
    # The summary did reach the journal (nanobot journals before the turn is saved too).
    assert loop.memory.read_unprocessed_history(0)[0]["content"] == "- s1"


# --- /compact -------------------------------------------------------------------------------


async def test_compact_session(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "a", "- [permanent] likes tea", "b", window=128_000)
    assert await loop.compact_session() == ""  # empty session
    await loop.process_direct("I like tea")
    assert await loop.compact_session() == "- [permanent] likes tea"
    await loop.process_direct("what do I like?")
    sent = fake.calls[2].messages
    assert [m["role"] for m in sent] == ["system", "user"]
    assert "- [permanent] likes tea" in sent[0]["content"]


# --- ephemeral turns ----------------------------------------------------------------------------


async def test_ephemeral_turn_saves_nothing_and_uses_its_own_tools(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "regular", "temp")
    await loop.process_direct("keep me")
    only_echo = ToolRegistry()
    only_echo.register(EchoTool())
    await loop.process_direct("temporary", session_key="dream:x", ephemeral=True, tools=only_echo)
    assert [m["role"] for m in fake.calls[1].messages] == ["system", "user"]
    assert not loop.sessions.get_session_path("dream:x").exists()
    assert "pending_user_turn" not in loop.sessions.get_or_create("dream:x").metadata


# --- Dream ---------------------------------------------------------------------------------------


def write_memory_call(content):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="w1", name="write_file", arguments={"path": "memory/MEMORY.md", "content": content})],
        finish_reason="tool_calls",
    )


async def test_dream_with_nothing_to_do(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path)
    assert await loop.run_dream() == DREAM_NO_INPUT_MESSAGE
    assert fake.calls == []


async def test_dream_updates_memory_and_advances_the_cursor(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, write_memory_call("# Memory\n- likes tea\n"), "Updated MEMORY.md.")
    loop.memory.append_history("- [permanent] likes tea")
    message = await loop.run_dream()
    assert message.startswith("Dream completed in") and "no memory changes" not in message
    assert (ws / "memory/MEMORY.md").read_text(encoding="utf-8") == "# Memory\n- likes tea\n"
    assert loop.memory.get_last_dream_cursor() == 1
    # Dream's request: normal system prompt, then its instructions with the journal.
    first = fake.calls[0]
    assert first.messages[1]["content"].startswith("You are running Dream.")
    assert "- [permanent] likes tea" in first.messages[1]["content"]
    assert sorted(t["function"]["name"] for t in first.tools) == ["apply_patch", "edit_file", "read_file", "write_file"]
    # Nothing saved for the Dream session; the new memory shows up in the next prompt.
    assert not any(p.name.startswith("ZHJlYW") for p in loop.sessions.sessions_dir.iterdir())
    assert "- likes tea" in loop.context.build_system_prompt()


async def test_dream_reports_no_changes(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, "Memory is already current.")
    loop.memory.append_history("- small talk")
    message = await loop.run_dream()
    # "No memory changes" comes from the git diff of the three files. The
    # moved .dream_cursor is still committed (nanobot stages it by name, past
    # .gitignore), so the reply names a commit, as nanobot's /dream does.
    assert re.fullmatch(r"Dream completed in \d+\.\ds; no memory changes\. \(commit [0-9a-f]{8}\)", message)
    assert loop.memory.get_last_dream_cursor() == 1
    assert [c.message for c in loop.memory.git.log()] == ["dream: manual run", "init: nanoclaw memory store"]


async def test_incomplete_dream_keeps_the_cursor(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, LLMResponse(content="Error calling LLM: down", finish_reason="error"))
    loop.memory.append_history("- fact")
    message = await loop.run_dream()
    assert "did not complete" in message and "stop_reason: error" in message
    assert loop.memory.get_last_dream_cursor() == 0


# --- CLI --------------------------------------------------------------------------------------------


def test_cli_compact_and_dream(capsys, ws):
    fake = FakeProvider(["hi", "- [permanent] user is Rohit", write_memory_call("- user is Rohit\n"), "done"])
    assert main(["ask", "I am Rohit", "-w", str(ws)], provider=fake) == 0
    assert main(["compact", "-w", str(ws)], provider=fake) == 0
    assert "Conversation compacted. Summary:\n- [permanent] user is Rohit" in capsys.readouterr().out
    assert main(["dream", "-w", str(ws)], provider=fake) == 0
    assert "Dream completed" in capsys.readouterr().out
    assert (ws / "memory/MEMORY.md").read_text(encoding="utf-8") == "- user is Rohit\n"
    assert main(["compact", "-w", str(ws)], provider=fake) == 0
    assert "Nothing to compact." in capsys.readouterr().out


def test_cli_context_window_setting(monkeypatch, capsys, ws):
    monkeypatch.setenv("NANOCLAW_CONTEXT_WINDOW", "9000")
    fake = FakeProvider(["- s"])
    assert main(["ask", "hello", "-w", str(ws)], provider=fake) == 1  # 9000 - 8192 - 1024 < 0
    assert "exceeds the local context budget" in capsys.readouterr().err
    monkeypatch.setenv("NANOCLAW_CONTEXT_WINDOW", "lots")
    with pytest.raises(SystemExit, match="NANOCLAW_CONTEXT_WINDOW"):
        main(["ask", "hello", "-w", str(ws)], provider=fake)


def test_cli_default_context_window(ws):
    from nanoclaw.__main__ import _make_loop

    assert _make_loop(FakeProvider([]), ws).context_window_tokens == 128_000


async def test_new_message_too_big_even_after_compaction_is_an_error(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "a", window=0)
    await loop.process_direct("short question")
    system = request_size(loop, [], "")
    # Room for the archive request, but not for the system prompt plus this message.
    loop.context_window_tokens = system + 2500 + 100 + 1024
    fake.queue("- summary")
    with pytest.raises(ContextWindowExceededError):
        await loop.process_direct("huge " + "q" * 5000)
    assert fake.calls[-1].messages[-1]["content"].startswith("Create a compact replacement checkpoint")


async def test_dream_trims_the_journal_afterwards(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, "Nothing to change.")
    loop.memory.max_history_entries = 1
    for i in range(3):
        loop.memory.append_history(f"- fact {i}")
    await loop.run_dream()
    assert [e["cursor"] for e in loop.memory._read_entries()] == [3]


async def test_oversized_ephemeral_turn_is_refused_like_nanobot(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "unused", window=100 + 1024 + 3000)
    with pytest.raises(ContextWindowExceededError):
        await loop.process_direct("big " + "q" * 20000, session_key="dream:x", ephemeral=True)
    assert fake.calls == [] and loop.memory.read_unprocessed_history(0) == []


async def test_compaction_without_persist_journals_nothing(ws, tmp_path):
    # Was a direct call of the removed AgentLoop._compact_if_needed(persist=False).
    # Now: a temporary turn (persist=False) whose tool result overflows mid-turn.
    loop, fake = make_loop(ws, tmp_path, window=0)
    system = request_size(loop, [], "")
    loop.context_window_tokens = system + 6800 + 100 + 1024
    echo = ToolCallRequest(id="e1", name="echo", arguments={"text": "r" * 3000})
    fake.queue(LLMResponse(content=None, tool_calls=[echo], finish_reason="tool_calls"), "- S", "answer")
    result = await loop.process_direct("go " + "q" * 1500, session_key="dream:x", ephemeral=True)
    assert result.final_content == "answer"
    archive_call, turn_call = fake.calls[1], fake.calls[2]
    assert archive_call.messages[-1]["content"].startswith("Create a compact replacement checkpoint")
    assert [m["role"] for m in turn_call.messages] == ["system", "user", "assistant", "tool"]
    assert "- S" in turn_call.messages[0]["content"]
    assert loop.memory.read_unprocessed_history(0) == []