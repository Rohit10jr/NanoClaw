"""Layer 15: a turn whose process died is restored at the next turn.

nanobot session/recovery.py (``runtime_checkpoint_is_well_formed`` :216,
``restore_runtime_checkpoint`` :278, ``discard_runtime_checkpoint`` :376,
``restore_pending_interruption`` :386) and the restore stage of
``_restore_turn`` (agent/loop.py:2134: the checkpoint first, then a user-only
tail). Adapted from nanobot tests/agent/test_loop_save_turn.py (restore and
``test_stop_preserves_runtime_checkpoint_for_next_turn``) and
tests/agent/test_auto_compact.py (a checkpoint restored on resume).
"""

import subprocess
import sys
import textwrap
from typing import Any

import pytest

from nanoclaw.__main__ import DEFAULT_SESSION, main
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools import Tool, ToolRegistry
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.runtime_context import RUNTIME_CONTEXT_TAG
from nanoclaw.session import Session, SessionManager
from nanoclaw.session.recovery import (
    INTERRUPTED_TOOL_MESSAGE,
    INTERRUPTED_TURN_MESSAGE,
    PENDING_USER_TURN_KEY,
    RUNTIME_CHECKPOINT_KEY,
    discard_runtime_checkpoint,
    restore_pending_interruption,
    restore_runtime_checkpoint,
    runtime_checkpoint_is_well_formed,
)
from toy_tools import EchoTool


def tc(call_id, name="exec"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


ASSISTANT = {"role": "assistant", "content": "working", "tool_calls": [tc("c1", "read_file"), tc("c2")]}
DONE_1 = {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "ok"}
DONE_2 = {"role": "tool", "tool_call_id": "c2", "name": "exec", "content": "done"}
FINAL = {
    "phase": "final_response",
    "assistant_message": {"role": "assistant", "content": "finished"},
    "completed_tool_results": [],
    "pending_tool_calls": [],
}
AWAITING = {
    "phase": "awaiting_tools",
    "assistant_message": ASSISTANT,
    "completed_tool_results": [],
    "pending_tool_calls": [tc("c1", "read_file"), tc("c2")],
}
COMPLETED = {
    "phase": "tools_completed",
    "assistant_message": ASSISTANT,
    "completed_tool_results": [DONE_1, DONE_2],
    "pending_tool_calls": [],
}


def crashed(checkpoint, *rows, flag=True):
    """A session as a killed turn leaves it: the user message, the flag, the checkpoint."""
    session = Session(key="cli:a")
    session.messages = [{"role": "user", "content": "go"}, *(dict(r) for r in rows)]
    if flag:
        session.metadata[PENDING_USER_TURN_KEY] = True
    if checkpoint is not None:
        session.metadata[RUNTIME_CHECKPOINT_KEY] = checkpoint
    return session


def shape(session):
    return [(m["role"], m.get("tool_call_id"), m.get("content")) for m in session.messages]


# --- runtime_checkpoint_is_well_formed ------------------------------------------------


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        (AWAITING, True),
        (COMPLETED, True),
        (FINAL, True),
        ({**AWAITING, "completed_tool_results": [DONE_1], "pending_tool_calls": [tc("c2")]}, False),
        ({**COMPLETED, "completed_tool_results": [DONE_1], "pending_tool_calls": [tc("c2")]}, False),
        ({**FINAL, "assistant_message": {"role": "assistant", "content": "  "}}, False),
        ({**FINAL, "phase": "error"}, False),
        ({k: v for k, v in FINAL.items() if k != "phase"}, False),
        ({**FINAL, "assistant_message": {"role": "user", "content": "x"}}, False),
        ({**COMPLETED, "completed_tool_results": "x"}, False),
        ({**AWAITING, "pending_tool_calls": [tc("c1"), {"id": "c2", "function": "bad"}]}, False),
        # Every call pending, but a result already recorded: not awaiting any more.
        ({**AWAITING, "completed_tool_results": [DONE_1]}, False),
        # The right number of rows, but for a call the assistant never made.
        ({**AWAITING, "pending_tool_calls": [tc("c1"), tc("c3")]}, False),
        ({**COMPLETED, "completed_tool_results": [DONE_1, dict(DONE_2, tool_call_id="c3")]}, False),
        # Every call answered, but one still listed as pending.
        ({**COMPLETED, "pending_tool_calls": [tc("c2")]}, False),
        # A result row that is not a tool message.
        ({**COMPLETED, "completed_tool_results": [DONE_1, dict(DONE_2, role="user")]}, False),
        (
            {
                "phase": "awaiting_tools",
                "assistant_message": {"role": "assistant", "content": None, "tool_calls": [tc("c1"), tc("c1")]},
                "completed_tool_results": [],
                "pending_tool_calls": [tc("c1"), tc("c1")],
            },
            False,
        ),
    ],
    ids=[
        "awaiting",
        "completed",
        "final",
        "awaiting-with-results",
        "completed-with-pending",
        "final-blank",
        "error-phase",
        "no-phase",
        "not-assistant",
        "rows-not-a-list",
        "call-without-function",
        "awaiting-all-pending-with-a-result",
        "awaiting-unknown-call",
        "completed-unknown-call",
        "completed-all-answered-with-pending",
        "result-not-a-tool-row",
        "duplicate-ids",
    ],
)
def test_well_formed_matches_nanobot(checkpoint, expected):
    assert runtime_checkpoint_is_well_formed(checkpoint) is expected


def test_discard_drops_the_checkpoint_without_saving_it():
    session = crashed(AWAITING)
    assert discard_runtime_checkpoint(session) is True
    assert shape(session) == [("user", None, "go")]
    assert RUNTIME_CHECKPOINT_KEY not in session.metadata
    assert discard_runtime_checkpoint(session) is False


# --- restore_runtime_checkpoint -----------------------------------------------------------


def test_awaiting_tools_restores_every_call_as_interrupted():
    session = crashed(AWAITING)
    assert restore_runtime_checkpoint(session) is True
    assert shape(session)[1:] == [
        ("assistant", None, "working"),
        ("tool", "c1", INTERRUPTED_TOOL_MESSAGE),
        ("tool", "c2", INTERRUPTED_TOOL_MESSAGE),
    ]
    assert [m["name"] for m in session.messages[2:]] == ["read_file", "exec"]
    assert all(m["_recovery_interrupted"] is True for m in session.messages[2:])
    assert PENDING_USER_TURN_KEY not in session.metadata and RUNTIME_CHECKPOINT_KEY not in session.metadata


def test_interrupted_tool_text_is_nanobots():
    assert INTERRUPTED_TOOL_MESSAGE == "Error: Task interrupted before this tool finished."
    assert INTERRUPTED_TURN_MESSAGE == "Error: Task interrupted before a response was generated."


def test_tools_completed_restores_the_results_as_they_were():
    session = crashed(COMPLETED)
    assert restore_runtime_checkpoint(session) is True
    assert shape(session)[1:] == [("assistant", None, "working"), ("tool", "c1", "ok"), ("tool", "c2", "done")]
    assert not any(m.get("_recovery_interrupted") for m in session.messages)


def test_final_response_restores_the_answer():
    session = crashed(FINAL)
    assert restore_runtime_checkpoint(session) is True
    assert shape(session) == [("user", None, "go"), ("assistant", None, "finished")]


def test_completed_and_pending_rows_are_rehydrated():
    """nanobot test_restore_runtime_checkpoint_rehydrates_completed_and_pending_tools."""
    session = crashed({"assistant_message": ASSISTANT, "completed_tool_results": [DONE_1], "pending_tool_calls": [tc("c2")]})
    assert restore_runtime_checkpoint(session) is True
    assert shape(session)[1:] == [
        ("assistant", None, "working"),
        ("tool", "c1", "ok"),
        ("tool", "c2", INTERRUPTED_TOOL_MESSAGE),
    ]


@pytest.mark.parametrize(
    ("saved", "checkpoint", "expected"),
    [
        # nanobot test_restore_runtime_checkpoint_dedupes_overlapping_tail
        ([ASSISTANT, DONE_1], {**AWAITING, "completed_tool_results": [DONE_1], "pending_tool_calls": [tc("c2")]}, 4),
        ([ASSISTANT], COMPLETED, 4),
        ([ASSISTANT, DONE_1, DONE_2], COMPLETED, 4),
        # Only an exact tail counts: a different assistant row is kept and the checkpoint added.
        ([dict(ASSISTANT, content="other")], COMPLETED, 5),
    ],
    ids=["assistant-and-result", "assistant-only", "all-saved", "content-differs"],
)
def test_rows_already_saved_are_not_added_twice(saved, checkpoint, expected):
    session = crashed(checkpoint, *saved)
    assert restore_runtime_checkpoint(session) is True
    assert len(session.messages) == expected
    assert [m.get("tool_call_id") for m in session.messages[-2:]] == ["c1", "c2"]


def test_the_longest_saved_overlap_wins():
    """Only the whole matching tail counts, even when a shorter one matches too
    (two identical interrupted rows: a checkpoint repeating a call id)."""
    checkpoint = {"assistant_message": None, "completed_tool_results": [], "pending_tool_calls": [tc("c1"), tc("c1")]}
    session = crashed(checkpoint)
    restore_runtime_checkpoint(session)
    before = shape(session)
    assert len(before) == 3
    session.metadata[RUNTIME_CHECKPOINT_KEY] = checkpoint
    restore_runtime_checkpoint(session)
    assert shape(session) == before


def test_restoring_twice_adds_nothing():
    session = crashed(AWAITING)
    restore_runtime_checkpoint(session)
    before = shape(session)
    session.metadata[RUNTIME_CHECKPOINT_KEY] = AWAITING  # the same checkpoint again (a /stop racing a restart)
    assert restore_runtime_checkpoint(session) is True
    # The interrupted rows built again match the saved ones exactly.
    assert shape(session) == before


@pytest.mark.parametrize("checkpoint", ["garbage", [1, 2], 3], ids=["string", "list", "number"])
def test_a_checkpoint_that_is_not_a_dict_is_ignored(checkpoint):
    """As nanobot: nothing is restored and the value stays until the next save
    clears it; the user-only tail is then closed as usual."""
    session = crashed(checkpoint)
    assert restore_runtime_checkpoint(session) is False
    assert session.metadata[RUNTIME_CHECKPOINT_KEY] == checkpoint
    assert restore_pending_interruption(session) is True
    assert shape(session) == [("user", None, "go"), ("assistant", None, INTERRUPTED_TURN_MESSAGE)]


def test_malformed_rows_are_left_out():
    session = crashed(
        {
            "phase": "awaiting_tools",
            "assistant_message": {"role": "user", "content": "not an assistant"},
            "completed_tool_results": ["junk", {"role": "user"}, DONE_1],
            "pending_tool_calls": [{"id": ""}, "junk", {"id": "c9"}, {"id": "c8", "function": "bad"}, tc("c2")],
        }
    )
    assert restore_runtime_checkpoint(session) is True
    assert [(m["role"], m.get("tool_call_id"), m.get("name")) for m in session.messages[1:]] == [
        ("tool", "c1", "read_file"),
        ("tool", "c9", "tool"),
        ("tool", "c8", "tool"),
        ("tool", "c2", "exec"),
    ]


def test_fields_of_the_wrong_type_restore_only_the_assistant():
    session = crashed({**COMPLETED, "completed_tool_results": "x", "pending_tool_calls": None})
    assert restore_runtime_checkpoint(session) is True
    assert shape(session)[1:] == [("assistant", None, "working")]


def test_a_restored_checkpoint_leaves_no_user_only_tail_to_close():
    """restore_runtime_checkpoint clears the pending flag (nanobot recovery.py:370),
    so the turn-start stage adds no "interrupted" reply after it."""
    session = crashed(FINAL)
    assert restore_runtime_checkpoint(session) is True
    assert restore_pending_interruption(session) is False
    assert shape(session)[-1] == ("assistant", None, "finished")


# --- the next turn restores a crashed one -------------------------------------------------


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


def make_loop(ws, tmp_path, *script, tools=None):
    if tools is None:
        tools = ToolRegistry()
        tools.register(EchoTool())
    fake = FakeProvider(list(script))
    return AgentLoop(fake, ws, tools=tools, sessions=SessionManager(ws, sessions_root=tmp_path / "sessions")), fake


def write_crashed_session(tmp_path, checkpoint, *rows, flag=True, key="cli:direct"):
    """Write the session file the way a turn killed after its checkpoint leaves it."""
    sessions = SessionManager(tmp_path / "ws", sessions_root=tmp_path / "sessions")
    session = sessions.get_or_create(key)
    session.add_message("user", "go")
    session.messages.extend(dict(r) for r in rows)
    if flag:
        session.metadata[PENDING_USER_TURN_KEY] = True
    if checkpoint is not None:
        session.metadata[RUNTIME_CHECKPOINT_KEY] = checkpoint
    sessions.save(session)


def on_disk(tmp_path, key="cli:direct"):
    return SessionManager(tmp_path / "ws", sessions_root=tmp_path / "sessions").get_or_create(key)


def plain(messages):
    out = []
    for m in messages:
        if m["role"] == "system":
            continue
        content = m.get("content")
        if isinstance(content, str) and RUNTIME_CONTEXT_TAG in content:
            content = content.split("\n\n" + RUNTIME_CONTEXT_TAG)[0]
        out.append((m["role"], m.get("tool_call_id"), content))
    return out


@pytest.mark.parametrize(
    ("checkpoint", "restored"),
    [
        (
            AWAITING,
            [
                ("assistant", None, "working"),
                ("tool", "c1", INTERRUPTED_TOOL_MESSAGE),
                ("tool", "c2", INTERRUPTED_TOOL_MESSAGE),
            ],
        ),
        (COMPLETED, [("assistant", None, "working"), ("tool", "c1", "ok"), ("tool", "c2", "done")]),
        (FINAL, [("assistant", None, "finished")]),
    ],
    ids=["awaiting_tools", "tools_completed", "final_response"],
)
async def test_the_next_turn_restores_the_checkpoint_first(ws, tmp_path, checkpoint, restored):
    write_crashed_session(tmp_path, checkpoint)
    loop, fake = make_loop(ws, tmp_path, "Back again.")
    result = await loop.process_direct("hello")
    assert result.final_content == "Back again."
    expected = [("user", None, "go"), *restored, ("user", None, "hello"), ("assistant", None, "Back again.")]
    assert plain(fake.calls[0].messages) == expected[:-1]  # the model saw the restored rows
    saved = on_disk(tmp_path)
    assert plain(saved.messages) == expected
    assert RUNTIME_CHECKPOINT_KEY not in saved.metadata and PENDING_USER_TURN_KEY not in saved.metadata


async def test_already_saved_rows_are_not_repeated_at_the_next_turn(ws, tmp_path):
    write_crashed_session(tmp_path, COMPLETED, ASSISTANT, DONE_1)
    loop, _ = make_loop(ws, tmp_path, "Back again.")
    await loop.process_direct("hello")
    assert [m.get("tool_call_id") for m in on_disk(tmp_path).messages] == [None, None, "c1", "c2", None, None]


async def test_a_user_only_tail_gets_the_interrupted_reply(ws, tmp_path):
    write_crashed_session(tmp_path, None)
    loop, fake = make_loop(ws, tmp_path, "Back again.")
    await loop.process_direct("hello")
    expected = [
        ("user", None, "go"),
        ("assistant", None, INTERRUPTED_TURN_MESSAGE),
        ("user", None, "hello"),
        ("assistant", None, "Back again."),
    ]
    assert plain(fake.calls[0].messages) == expected[:-1]
    assert plain(on_disk(tmp_path).messages) == expected


async def test_a_malformed_checkpoint_is_cleared_by_the_next_save(ws, tmp_path):
    write_crashed_session(tmp_path, "garbage")
    loop, _ = make_loop(ws, tmp_path, "Back again.")
    await loop.process_direct("hello")
    saved = on_disk(tmp_path)
    assert [c for _, _, c in plain(saved.messages)] == ["go", INTERRUPTED_TURN_MESSAGE, "hello", "Back again."]
    assert RUNTIME_CHECKPOINT_KEY not in saved.metadata


async def test_a_command_after_a_crash_still_restores_first(ws, tmp_path):
    """The restore stage runs before command dispatch (nanobot ``_process_message`` order)."""
    write_crashed_session(tmp_path, FINAL)
    loop, fake = make_loop(ws, tmp_path)
    await loop.process_direct("/status")
    assert fake.calls == []
    saved = on_disk(tmp_path)
    assert plain(saved.messages)[:2] == [("user", None, "go"), ("assistant", None, "finished")]
    assert RUNTIME_CHECKPOINT_KEY not in saved.metadata
    # The command's own saved input is answered, so it leaves no crash marker.
    assert PENDING_USER_TURN_KEY not in saved.metadata


async def test_a_subagent_report_after_a_crash_restores_first(ws, tmp_path):
    """System turns go through the same restore stage."""
    from nanoclaw.bus import InboundMessage

    write_crashed_session(tmp_path, AWAITING)
    loop, fake = make_loop(ws, tmp_path, "Noted.")
    # The report's child session still belongs to cli:direct (otherwise the
    # report is dropped, as in nanobot: ``accepts_result``).
    loop.sessions.save(
        Session(key="subagent:t1", metadata={"session_type": "subagent", "parent_session_key": "cli:direct"})
    )
    report = InboundMessage(
        channel="system",
        sender_id="subagent",
        chat_id="cli:direct",
        content="Subagent finished.",
        metadata={"subagent_task_id": "t1"},
    )
    await loop.process_subagent_result(report)
    roles = [(m["role"], m.get("tool_call_id")) for m in on_disk(tmp_path).messages]
    assert roles[:4] == [("user", None), ("assistant", None), ("tool", "c1"), ("tool", "c2")]
    assert plain(fake.calls[0].messages)[1:4] == plain(on_disk(tmp_path).messages)[1:4]


# --- a process that dies mid-tool ---------------------------------------------------------


class Killed(BaseException):
    """Stands in for the process dying: nothing in the runner or loop catches it."""


class DieTool(Tool):
    @property
    def name(self) -> str:
        return "die"

    @property
    def description(self) -> str:
        return "Kill the process."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self) -> str:
        raise Killed


def calls(*pairs, content=None):
    return LLMResponse(
        content=content,
        tool_calls=[ToolCallRequest(id=i, name=n, arguments=a) for i, n, a in pairs],
        finish_reason="tool_calls",
    )


async def test_a_turn_killed_mid_tool_is_restored_by_a_new_loop(ws, tmp_path):
    """The checkpoint written before the tool ran survives the death; a new
    loop on the same sessions restores it (nanobot
    test_stop_preserves_runtime_checkpoint_for_next_turn, with a kill instead
    of /stop)."""
    tools = ToolRegistry()
    tools.register(EchoTool())
    tools.register(DieTool())
    first, _ = make_loop(
        ws,
        tmp_path,
        calls(("e1", "echo", {"text": "a"})),
        calls(("c1", "echo", {"text": "b"}), ("c2", "die", {}), content="Checking."),
        tools=tools,
    )
    with pytest.raises(Killed):
        await first.process_direct("go")

    left = on_disk(tmp_path)
    assert left.metadata[PENDING_USER_TURN_KEY] is True
    assert left.metadata[RUNTIME_CHECKPOINT_KEY]["phase"] == "awaiting_tools"
    assert plain(left.messages) == [("user", None, "go")]

    second, fake = make_loop(ws, tmp_path, "Back again.")
    await second.process_direct("continue here")
    # As nanobot, only the latest round survives; both of its calls are
    # interrupted, since the checkpoint was written before either finished.
    expected = [
        ("user", None, "go"),
        ("assistant", None, "Checking."),
        ("tool", "c1", INTERRUPTED_TOOL_MESSAGE),
        ("tool", "c2", INTERRUPTED_TOOL_MESSAGE),
        ("user", None, "continue here"),
        ("assistant", None, "Back again."),
    ]
    assert plain(fake.calls[0].messages) == expected[:-1]
    saved = on_disk(tmp_path)
    assert plain(saved.messages) == expected
    assert RUNTIME_CHECKPOINT_KEY not in saved.metadata and PENDING_USER_TURN_KEY not in saved.metadata


CHILD = textwrap.dedent(
    """
    import asyncio, sys
    from pathlib import Path
    from nanoclaw.__main__ import CHANNEL, DEFAULT_SESSION
    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.agent.tools import Tool, ToolRegistry
    from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest

    class Hang(Tool):
        name = "hang"
        description = "Never returns."
        parameters = {"type": "object", "properties": {}}

        async def execute(self):
            print("READY", flush=True)
            await asyncio.Event().wait()

    tools = ToolRegistry()
    tools.register(Hang())
    reply = LLMResponse(
        content="Let me wait.",
        tool_calls=[ToolCallRequest(id="c1", name="hang", arguments={})],
        finish_reason="tool_calls",
    )
    loop = AgentLoop(FakeProvider([reply]), Path(sys.argv[1]), tools=tools)
    asyncio.run(loop.process_direct("go", session_key=DEFAULT_SESSION, channel=CHANNEL))
    """
)


def test_nanoclaw_ask_restores_a_process_killed_mid_tool(ws, tmp_path, capsys):
    """End to end: a real process is killed while its tool runs, then
    ``nanoclaw ask`` continues the conversation."""
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(ws)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        line = child.stdout.readline()
        assert line.strip() == "READY", child.stderr.read()
    finally:
        child.kill()
        child.communicate()

    sessions = SessionManager(ws)  # ask's default sessions folder (~ is a test folder)
    left = sessions.get_or_create(DEFAULT_SESSION)
    assert left.metadata[RUNTIME_CHECKPOINT_KEY]["phase"] == "awaiting_tools"

    fake = FakeProvider(["Back again."])
    assert main(["ask", "hello", "-w", str(ws)], provider=fake) == 0
    assert capsys.readouterr().out.strip() == "Back again."
    expected = [
        ("user", None, "go"),
        ("assistant", None, "Let me wait."),
        ("tool", "c1", INTERRUPTED_TOOL_MESSAGE),
        ("user", None, "hello"),
        ("assistant", None, "Back again."),
    ]
    assert plain(fake.calls[0].messages) == expected[:-1]
    saved = SessionManager(ws).get_or_create(DEFAULT_SESSION)
    assert plain(saved.messages) == expected
    assert RUNTIME_CHECKPOINT_KEY not in saved.metadata and PENDING_USER_TURN_KEY not in saved.metadata
