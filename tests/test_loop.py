"""AgentLoop: turns that remember the conversation."""

import asyncio

import pytest

from nanoclaw.__main__ import main
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.messages import EMPTY_FINAL_RESPONSE_MESSAGE, MODEL_ERROR_PLACEHOLDER
from nanoclaw.agent.runner import AgentRunResult
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.providers.base import sanitize_messages
from nanoclaw.runtime_context import RUNTIME_CONTEXT_TAG
from nanoclaw.session import SessionManager
from nanoclaw.session.recovery import INTERRUPTED_TURN_MESSAGE, restore_pending_interruption
from toy_tools import EchoTool


def wants_echo(call_id="call_1", text="hi"):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name="echo", arguments={"text": text})], finish_reason="tool_calls"
    )


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


def make_loop(ws, tmp_path, *script):
    tools = ToolRegistry()
    tools.register(EchoTool())
    fake = FakeProvider(list(script))
    sessions = SessionManager(ws, sessions_root=tmp_path / "sessions")
    return AgentLoop(fake, ws, tools=tools, sessions=sessions), fake


def visible(messages):
    """(role, content-before-runtime-context) for each non-system message."""
    out = []
    for m in messages:
        if m["role"] == "system":
            continue
        content = m.get("content")
        if isinstance(content, str) and RUNTIME_CONTEXT_TAG in content:
            content = content.split("\n\n" + RUNTIME_CONTEXT_TAG)[0]
        out.append((m["role"], content))
    return out


async def test_second_turn_sees_the_first(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "Nice to meet you, Rohit.", "Your name is Rohit.")
    first = await loop.process_direct("My name is Rohit")
    second = await loop.process_direct("What's my name?")
    assert (first.final_content, second.final_content) == ("Nice to meet you, Rohit.", "Your name is Rohit.")
    assert visible(fake.calls[1].messages) == [
        ("user", "My name is Rohit"),
        ("assistant", "Nice to meet you, Rohit."),
        ("user", "What's my name?"),
    ]


async def test_history_survives_a_restart(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, "noted")
    await loop.process_direct("remember 42")
    loop2, fake2 = make_loop(ws, tmp_path, "42")
    await loop2.process_direct("what number?")
    assert ("user", "remember 42") in visible(fake2.calls[0].messages)


async def test_old_turns_replay_byte_identical(ws, tmp_path):
    """The saved user message includes its time block, so replay never changes."""
    loop, fake = make_loop(ws, tmp_path, "a1", "a2", "a3")
    await loop.process_direct("q1")
    await loop.process_direct("q2")
    await loop.process_direct("q3")
    sent_q1 = fake.calls[0].messages[1]
    assert RUNTIME_CONTEXT_TAG in sent_q1["content"]
    assert fake.calls[1].messages[1]["content"] == sent_q1["content"]
    # What a provider actually sends (``_meta`` is dropped) is identical up to the new message.
    assert sanitize_messages(fake.calls[2].messages[:4]) == sanitize_messages(fake.calls[1].messages[:4])


async def test_saved_user_message_has_runtime_context_marker(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, "ok")
    await loop.process_direct("hello")
    user = loop.sessions.get_or_create("cli:direct").messages[0]
    marker = user["_runtime_context"]
    assert marker["version"] == 1 and marker["sources"] == ["time"]
    assert user["content"] == "hello\n\n" + marker["suffix"]
    assert "_meta" not in user


async def test_tool_messages_are_saved_and_replayed(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, wants_echo(), "echoed", "again")
    result = await loop.process_direct("echo hi")
    assert result.tools_used == ["echo"]
    saved = loop.sessions.get_or_create("cli:direct").messages
    assert [m["role"] for m in saved] == ["user", "assistant", "tool", "assistant"]
    assert saved[2]["tool_call_id"] == "call_1" and saved[2]["content"] == "hi"
    assert all("timestamp" in m for m in saved)

    await loop.process_direct("next")
    roles = [m["role"] for m in fake.calls[2].messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    # Replayed messages carry no bookkeeping keys.
    assert all("timestamp" not in m and "_runtime_context" not in m for m in fake.calls[2].messages)


async def test_user_message_is_saved_before_the_run(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path)

    async def boom(spec):
        raise RuntimeError("crash mid-turn")

    loop.runner.run = boom
    with pytest.raises(RuntimeError):
        await loop.process_direct("don't lose me")
    reloaded = SessionManager(ws, sessions_root=tmp_path / "sessions").get_or_create("cli:direct")
    assert reloaded.messages[0]["content"].startswith("don't lose me")


async def test_after_an_interrupted_turn_the_next_message_is_merged(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, wants_echo(), "done")
    session = loop.sessions.get_or_create("cli:direct")
    session.add_message("user", "unanswered")
    loop.sessions.save(session)
    await loop.process_direct("new question")
    # One user message is sent (merged), never two in a row.
    sent = visible(fake.calls[0].messages)
    assert [role for role, _ in sent] == ["user"]
    assert sent[0][1].startswith("unanswered\n\nnew question")
    # Both stay separate in the session and the whole turn is saved.
    saved = [m["role"] for m in loop.sessions.get_or_create("cli:direct").messages]
    assert saved == ["user", "user", "assistant", "tool", "assistant"]


async def test_model_error_is_saved_with_placeholder(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, LLMResponse(content="Error calling LLM: boom", finish_reason="error"), "ok")
    result = await loop.process_direct("hi")
    assert result.stop_reason == "error"
    saved = loop.sessions.get_or_create("cli:direct").messages
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[1]["content"] == MODEL_ERROR_PLACEHOLDER


async def test_blank_final_content_becomes_the_empty_response_message(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, "   ", "  ", "", "")
    result = await loop.process_direct("hi")
    assert result.final_content == EMPTY_FINAL_RESPONSE_MESSAGE


def test_save_turn_drops_bad_messages(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path)
    session = loop.sessions.get_or_create("k")
    turn = [
        {"role": "assistant", "content": ""},  # empty: dropped
        {"role": "tool", "tool_call_id": "nope", "content": "x"},  # undeclared: dropped
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "first"},
        {"role": "tool", "tool_call_id": "c1", "content": "duplicate"},  # already answered: dropped
        {"role": "tool", "content": "no id"},  # dropped
        {"role": "assistant", "content": "done", "timestamp": "kept"},
    ]
    loop._save_turn(session, turn, 0)
    assert [(m["role"], m["content"]) for m in session.messages] == [
        ("assistant", None),
        ("tool", "first"),
        ("assistant", "done"),
    ]
    assert session.messages[-1]["timestamp"] == "kept"


async def test_sessions_are_separate(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "a", "b")
    await loop.process_direct("in work", session_key="cli:work")
    await loop.process_direct("in home", session_key="cli:home")
    assert visible(fake.calls[1].messages) == [("user", "in home")]


async def test_new_session_forgets_after_archiving_to_the_journal(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "a", "- [ephemeral] user said old", "b")
    await loop.process_direct("old")
    await loop.new_session()
    # The old conversation was summarized into the history journal first.
    assert fake.calls[1].messages[-1]["content"].startswith("Create a compact replacement checkpoint")
    assert [e["content"] for e in loop.memory.read_unprocessed_history(0)] == ["- [ephemeral] user said old"]
    await loop.process_direct("fresh")
    assert visible(fake.calls[2].messages) == [("user", "fresh")]
    assert len(loop.sessions.get_or_create("cli:direct").messages) == 2


async def test_turns_in_one_session_run_one_at_a_time(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path)
    order = []

    async def slow_run(spec):
        order.append("start")
        await asyncio.sleep(0.01)
        order.append("end")
        return AgentRunResult("ok", spec.initial_messages + [{"role": "assistant", "content": "ok"}], [], None, "completed")

    loop.runner.run = slow_run
    await asyncio.gather(loop.process_direct("a"), loop.process_direct("b"))
    assert order == ["start", "end", "start", "end"]
    roles = [m["role"] for m in loop.sessions.get_or_create("cli:direct").messages]
    assert roles == ["user", "assistant", "user", "assistant"]


# --- CLI ------------------------------------------------------------------------------------


def test_cli_ask_continues_the_conversation(capsys, ws):
    fake = FakeProvider(["hello Rohit", "you are Rohit", "- [permanent] user is Rohit", "who?"])
    assert main(["ask", "I am Rohit", "-w", str(ws)], provider=fake) == 0
    assert main(["ask", "who am I?", "-w", str(ws)], provider=fake) == 0
    assert ("user", "I am Rohit") in visible(fake.calls[1].messages)
    assert main(["ask", "who am I?", "--new", "-w", str(ws)], provider=fake) == 0
    assert visible(fake.calls[3].messages) == [("user", "who am I?")]  # call 2 archived the old chat
    assert "who?" in capsys.readouterr().out


def test_cli_session_option(ws):
    fake = FakeProvider(["a", "b"])
    main(["ask", "one", "-s", "cli:other", "-w", str(ws)], provider=fake)
    main(["ask", "two", "-w", str(ws)], provider=fake)
    assert visible(fake.calls[1].messages) == [("user", "two")]


def test_save_turn_moves_the_runtime_context_marker_onto_the_user_message(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path)
    session = loop.sessions.get_or_create("k")
    marker = {"version": 1, "sources": ["time"], "suffix": "[T]"}
    loop._save_turn(session, [{"role": "user", "content": "q\n\n[T]", "_meta": {"runtime_context": marker}}], 0)
    assert session.messages[0]["_runtime_context"] == marker
    assert "_meta" not in session.messages[0]


# --- interrupted turns (nanobot session/recovery.py) -----------------------------------------


async def test_crashed_turn_gets_an_error_reply_before_the_next_turn(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path)

    async def boom(spec):
        raise RuntimeError("process died")

    loop.runner.run = boom
    with pytest.raises(RuntimeError):
        await loop.process_direct("first question")
    session = SessionManager(ws, sessions_root=tmp_path / "sessions").get_or_create("cli:direct")
    assert session.metadata["pending_user_turn"] is True

    loop2, fake = make_loop(ws, tmp_path, "answer")
    await loop2.process_direct("second question")
    assert visible(fake.calls[0].messages) == [
        ("user", "first question"),
        ("assistant", INTERRUPTED_TURN_MESSAGE),
        ("user", "second question"),
    ]
    saved = loop2.sessions.get_or_create("cli:direct")
    assert [m["role"] for m in saved.messages] == ["user", "assistant", "user", "assistant"]
    assert saved.messages[1]["_recovery_interrupted"] is True
    assert "pending_user_turn" not in saved.metadata


async def test_pending_flag_is_cleared_after_a_normal_turn(ws, tmp_path):
    loop, _ = make_loop(ws, tmp_path, "ok")
    await loop.process_direct("hi")
    reloaded = SessionManager(ws, sessions_root=tmp_path / "sessions").get_or_create("cli:direct")
    assert "pending_user_turn" not in reloaded.metadata


def test_restore_pending_interruption_rules():
    from nanoclaw.session import Session

    s = Session("k", messages=[{"role": "user", "content": "q"}], metadata={"pending_user_turn": True})
    assert restore_pending_interruption(s) is True
    assert s.messages[-1]["content"] == INTERRUPTED_TURN_MESSAGE
    assert restore_pending_interruption(s) is False  # flag gone
    # Flag set but the last message is not a user message: only the flag is cleared.
    s = Session("k", messages=[{"role": "assistant", "content": "a"}], metadata={"pending_user_turn": True})
    assert restore_pending_interruption(s) is True
    assert len(s.messages) == 1 and "pending_user_turn" not in s.metadata


# --- generation settings ----------------------------------------------------------------------


async def test_loop_passes_generation_settings(ws, tmp_path):
    loop, fake = make_loop(ws, tmp_path, "ok")
    loop.temperature, loop.max_tokens = 0.3, 1234
    await loop.process_direct("hi")
    assert (fake.calls[0].temperature, fake.calls[0].max_tokens) == (0.3, 1234)


def test_cli_uses_nanobots_agent_defaults(ws):
    fake = FakeProvider(["ok"])
    main(["ask", "hi", "-w", str(ws)], provider=fake)
    assert (fake.calls[0].temperature, fake.calls[0].max_tokens) == (0.1, 8192)
