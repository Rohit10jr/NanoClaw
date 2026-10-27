"""Turns that stop at the tool-iteration limit (layer 11).

Adapted from nanobot tests/agent/test_loop_iteration_limit.py: after a user
turn runs out of iterations, the next user turn is told so, once, and the
model is asked not to resume the old work unless the user wants that.
"""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.subagent import SubagentResult
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.providers.base import sanitize_messages
from nanoclaw.runtime_context import (
    ITERATION_LIMIT_META,
    RUNTIME_CONTEXT_END,
    RUNTIME_CONTEXT_TAG,
    iteration_limit_block,
    public_history_message,
    record_iteration_limit,
)
from nanoclaw.session import Session, SessionManager

_STOP_MARKER = "previous turn stopped at the tool-iteration limit"


class BudgetProvider(FakeProvider):
    """Calls exec for "run the tests..." while tools are offered; otherwise says "done"."""

    def __init__(self, requests: list[list[dict[str, Any]]]) -> None:
        super().__init__()
        self.requests = requests
        self.schemas: list[Any] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7, tool_choice=None):
        # What a real API receives (nanobot records `_sanitize_empty_content(messages)`).
        self.requests.append(copy.deepcopy(sanitize_messages(messages)))
        self.schemas.append(copy.deepcopy(tools))
        last_user = next(m for m in reversed(messages) if m.get("role") == "user")
        if str(last_user.get("content")).startswith("run the tests") and tools is not None:
            return LLMResponse(
                content="running",
                tool_calls=[ToolCallRequest(id=f"call_{len(self.requests)}", name="exec", arguments={"command": "make test"})],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="done")


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


def make_loop(ws, tmp_path, requests: list[list[dict[str, Any]]]) -> AgentLoop:
    sessions = SessionManager(ws, sessions_root=tmp_path / "sessions")
    loop = AgentLoop(BudgetProvider(requests), ws, sessions=sessions, max_iterations=2)
    exec_tool = loop.tools.get("exec")
    assert exec_tool is not None
    exec_tool.execute = AsyncMock(return_value="Error: make test failed")
    return loop


def last_user_content(messages: list[dict[str, Any]]) -> str:
    return str(next(m for m in reversed(messages) if m.get("role") == "user")["content"])


def test_block_text_matches_nanobot():
    # nanobot session/turn_continuation.py:54 iteration_limit_context
    assert iteration_limit_block({}) is None
    block = iteration_limit_block({ITERATION_LIMIT_META: True})
    assert block is not None and block.source == "iteration_limit"
    assert block.content == (
        f"{RUNTIME_CONTEXT_TAG}\n"
        "The previous turn stopped at the tool-iteration limit without finishing its work.\n"
        "Follow the current request. Resume unfinished work only if the user asks for it.\n"
        "Completed tool calls and their results remain in the history; do not repeat them blindly.\n"
        f"{RUNTIME_CONTEXT_END}"
    )
    metadata: dict[str, Any] = {}
    record_iteration_limit(metadata, "max_iterations")
    assert metadata == {ITERATION_LIMIT_META: True}
    record_iteration_limit(metadata, "completed")
    assert metadata == {}


async def test_next_turn_is_told_not_to_resume_after_iteration_limit(ws, tmp_path):
    requests: list[list[dict[str, Any]]] = []
    loop = make_loop(ws, tmp_path, requests)
    result = await loop.process_direct("run the tests", session_key="cli:limit")
    assert result.stop_reason == "max_iterations"
    assert _STOP_MARKER not in last_user_content(requests[0])

    await loop.process_direct("what is 2 + 2?", session_key="cli:limit")
    content = last_user_content(requests[-1])
    assert _STOP_MARKER in content
    # Its own block, after the time block (nanobot: after the workspace block).
    assert content.index("Current Time:") < content.index(_STOP_MARKER)
    assert content.endswith(iteration_limit_block({ITERATION_LIMIT_META: True}).content)

    # Later inputs get no new marker; the earlier marked input stays in history.
    await loop.process_direct("thanks", session_key="cli:limit")
    assert _STOP_MARKER not in last_user_content(requests[-1])
    assert _STOP_MARKER in json.dumps(requests[-1])


async def test_stop_survives_restart_and_keeps_history_prefix(ws, tmp_path):
    previous: list[list[dict[str, Any]]] = []
    loop = make_loop(ws, tmp_path, previous)
    await loop.process_direct("hello", session_key="cli:limit")
    await loop.process_direct("run the tests", session_key="cli:limit")
    schemas = loop.provider.schemas

    current: list[list[dict[str, Any]]] = []
    loop = make_loop(ws, tmp_path, current)
    await loop.process_direct("what is 2 + 2?", session_key="cli:limit")
    assert _STOP_MARKER in last_user_content(current[0])
    # Ordinary prefix and tool schemas are unchanged. Failed tool calls remain.
    assert json.dumps(previous[0]) == json.dumps(current[0][: len(previous[0])])
    assert schemas[0] == loop.provider.schemas[0]
    assert [m for m in previous[-1] if m["role"] == "tool"] == [m for m in current[0] if m["role"] == "tool"]
    await loop.process_direct("thanks", session_key="cli:limit")
    assert json.dumps(current[0]) == json.dumps(current[1][: len(current[0])])
    assert _STOP_MARKER not in last_user_content(current[1])
    session = loop.sessions.get_or_create("cli:limit")
    assert _STOP_MARKER not in json.dumps([public_history_message(m) for m in session.messages])

    # Another session must not inherit this stopped turn.
    await loop.process_direct("hello", session_key="cli:other")
    assert _STOP_MARKER not in json.dumps(current[-1])


async def test_subagent_follow_up_neither_gets_nor_clears_the_notice(ws, tmp_path):
    # NanoClaw's stand-in for nanobot's goal-continuation case: only user turns
    # read or write the flag (nanobot loop.py:2253 and :2385 check TurnKind.USER).
    requests: list[list[dict[str, Any]]] = []
    loop = make_loop(ws, tmp_path, requests)
    await loop.process_direct("run the tests", session_key="cli:limit")
    report = SubagentResult("abcd1234", "x", "[Subagent 'x' completed successfully]", "ok", "cli:limit", "cli")
    next_request = len(requests)
    await loop.process_subagent_result(report)
    assert _STOP_MARKER not in json.dumps(requests[next_request:])
    assert loop.sessions.get_or_create("cli:limit").metadata.get(ITERATION_LIMIT_META) is True

    await loop.process_direct("what happened?", session_key="cli:limit")
    assert _STOP_MARKER in last_user_content(requests[-1])


async def test_new_history_does_not_inherit_abandoned_turn(ws, tmp_path):
    requests: list[list[dict[str, Any]]] = []
    loop = make_loop(ws, tmp_path, requests)
    await loop.process_direct("run the tests", session_key="cli:limit")
    await loop.new_session("cli:limit")
    await loop.process_direct("hello", session_key="cli:limit")
    assert _STOP_MARKER not in last_user_content(requests[-1])


def test_session_clear_drops_the_flag():
    session = Session(key="cli:x", metadata={ITERATION_LIMIT_META: True, "other": 1})
    session.clear()
    assert session.metadata == {"other": 1}


async def test_temporary_turn_skips_hint_and_explicit_retry_can_use_tools(ws, tmp_path):
    requests: list[list[dict[str, Any]]] = []
    loop = make_loop(ws, tmp_path, requests)
    await loop.process_direct("run the tests", session_key="cli:limit")
    await loop.process_direct("hello", session_key="cli:limit", ephemeral=True)
    assert _STOP_MARKER not in last_user_content(requests[-1])
    next_request = len(requests)
    await loop.process_direct("run the tests again", session_key="cli:limit")
    assert _STOP_MARKER in last_user_content(requests[next_request])
    assert "only if the user asks" in last_user_content(requests[next_request])
    assert loop.tools.get("exec").execute.await_count == 4
    # The retry hit the limit again, so the flag is still set.
    assert loop.sessions.get_or_create("cli:limit").metadata.get(ITERATION_LIMIT_META) is True
