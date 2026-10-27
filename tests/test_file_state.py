"""read_file deduplication: "[File unchanged since last read: ...]" only while the
earlier result is still in the model's input.

Ports nanobot tests/tools/test_file_state_store.py, the read-dedup parts of
tests/tools/test_read_enhancements.py, tests/agent/test_runner_file_reads.py
(without the provider-native compaction test: NanoClaw has none) and the
subagent isolation test of tests/agent/test_subagent.py.
"""

import asyncio
import os
from copy import deepcopy

import pytest

from nanoclaw.agent.hook import AgentHook, AgentHookContext
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.subagent import SubagentManager
from nanoclaw.agent.tools import ToolRegistry, build_default_tools
from nanoclaw.agent.tools import file_state
from nanoclaw.agent.tools.apply_patch import ApplyPatchTool
from nanoclaw.agent.tools.file_state import (
    FileStates,
    FileStateStore,
    bind_file_states,
    reset_file_states,
)
from nanoclaw.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest

STUB = "[File unchanged since last read: data.txt]"


# ---------------------------------------------------------------------------
# FileStateStore (nanobot test_file_state_store.py)
# ---------------------------------------------------------------------------


def test_file_state_store_evicts_least_recently_used_session() -> None:
    store = FileStateStore(max_sessions=2)
    first = store.for_session("first")
    second = store.for_session("second")

    assert store.for_session("first") is first
    store.for_session("third")

    assert store.for_session("first") is first
    assert store.for_session("second") is not second


def test_file_state_store_discards_reset_session() -> None:
    store = FileStateStore()
    previous = store.for_session("websocket:chat")

    store.discard("websocket:chat")

    assert store.for_session("websocket:chat") is not previous


def test_file_state_store_requires_positive_capacity() -> None:
    with pytest.raises(ValueError, match="max_sessions must be positive"):
        FileStateStore(max_sessions=0)


# ---------------------------------------------------------------------------
# ReadFileTool with an explicit read context (nanobot test_read_enhancements.py)
# ---------------------------------------------------------------------------


class TestReadDedup:
    @pytest.fixture()
    def tool(self, tmp_path):
        return ReadFileTool(workspace=tmp_path)

    @pytest.fixture()
    def write_tool(self, tmp_path):
        return WriteFileTool(workspace=tmp_path)

    async def test_second_read_returns_unchanged_stub(self, tool, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")
        with file_state.file_read_context("read-1", lambda: {}):
            first = await tool.execute(path=str(f))
        assert "line 0" in first
        with file_state.file_read_context("read-2", lambda: {"read-1": first}):
            second = await tool.execute(path=str(f))
        assert second == f"[File unchanged since last read: {f}]"

    async def test_force_returns_contents_again(self, tool, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("alpha\n", encoding="utf-8")
        with file_state.file_read_context("read-1", lambda: {}):
            first = await tool.execute(path=str(f))
        with file_state.file_read_context("read-2", lambda: {"read-1": first}):
            forced = await tool.execute(path=str(f), force=True)
        assert forced == first
        # The forced read is now the receipt: a later read checks it.
        with file_state.file_read_context("read-3", lambda: {"read-2": forced}):
            assert "unchanged" in (await tool.execute(path=str(f))).lower()

    async def test_read_after_external_modification_returns_full(self, tool, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("original", encoding="utf-8")
        with file_state.file_read_context("read-1", lambda: {}):
            first = await tool.execute(path=str(f))
        f.write_text("modified content", encoding="utf-8")
        with file_state.file_read_context("read-2", lambda: {"read-1": first}):
            second = await tool.execute(path=str(f))
        assert "modified content" in second

    async def test_different_offset_returns_full(self, tool, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1, 21)), encoding="utf-8")
        with file_state.file_read_context("read-1", lambda: {}):
            first = await tool.execute(path=str(f), offset=1, limit=5)
        with file_state.file_read_context("read-2", lambda: {"read-1": first}):
            second = await tool.execute(path=str(f), offset=6, limit=5)
        assert "line 6" in second

    async def test_first_read_after_write_returns_full_content(self, tool, write_tool, tmp_path):
        f = tmp_path / "fresh.txt"
        result = await write_tool.execute(path=str(f), content="hello")
        assert "Successfully" in result
        read_result = await tool.execute(path=str(f))
        assert "hello" in read_result
        assert "unchanged" not in read_result.lower()

    @pytest.mark.parametrize("writer", ["write_file", "edit_file", "apply_patch"])
    async def test_write_forgets_the_read(self, tmp_path, writer):
        states = FileStates()
        f = tmp_path / "data.txt"
        f.write_text("alpha\n", encoding="utf-8")
        tool = ReadFileTool(workspace=tmp_path, file_states=states)
        with file_state.file_read_context("read-1", lambda: {}):
            await tool.execute(path="data.txt")
        assert states.get(f) is not None
        if writer == "write_file":
            await WriteFileTool(workspace=tmp_path, file_states=states).execute(path="data.txt", content="alpha\n")
        elif writer == "edit_file":
            await EditFileTool(workspace=tmp_path, file_states=states).execute(
                path="data.txt", old_text="alpha", new_text="beta"
            )
        else:
            await ApplyPatchTool(workspace=tmp_path, file_states=states).execute(
                edits=[{"action": "replace", "path": "data.txt", "old_text": "alpha", "new_text": "beta"}]
            )
        assert states.get(f) is None

    async def test_read_returns_changed_content_when_mtime_same(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("original", encoding="utf-8")
        tool = ReadFileTool(workspace=tmp_path)
        with file_state.file_read_context("read-1", lambda: {}):
            first = await tool.execute(path=str(f))
        original_mtime = os.path.getmtime(f)
        f.write_text("modified", encoding="utf-8")
        os.utime(f, (original_mtime, original_mtime))
        with file_state.file_read_context("read-2", lambda: {"read-1": first}):
            second = await tool.execute(path=str(f))
        assert "modified" in second
        assert "unchanged" not in second.lower()

    async def test_force_is_in_the_schema(self, tool):
        assert tool.parameters["properties"]["force"] == {
            "type": "boolean",
            "description": "Return an unchanged range again",
            "default": False,
        }


class TestReadDedupSessionIsolation:
    async def test_separate_tools_do_not_share_dedup_state(self, tmp_path):
        f = tmp_path / "shared.txt"
        f.write_text("\n".join(f"line {i}" for i in range(10)), encoding="utf-8")
        first = await ReadFileTool(workspace=tmp_path).execute(path=str(f))
        second = await ReadFileTool(workspace=tmp_path).execute(path=str(f))
        assert "line 0" in first
        assert "unchanged" not in second.lower()

    async def test_shared_loop_tool_uses_bound_session_state(self, tmp_path):
        f = tmp_path / "shared.txt"
        f.write_text("\n".join(f"line {i}" for i in range(10)), encoding="utf-8")
        shared_tool = ReadFileTool(workspace=tmp_path)
        session_a, session_b = FileStates(), FileStates()

        token = bind_file_states(session_a)
        try:
            with file_state.file_read_context("read-a1", lambda: {}):
                first = await shared_tool.execute(path=str(f))
            with file_state.file_read_context("read-a2", lambda: {"read-a1": first}):
                repeat = await shared_tool.execute(path=str(f))
        finally:
            reset_file_states(token)
        assert "line 0" in first
        assert "unchanged" in repeat.lower()

        token = bind_file_states(session_b)
        try:
            with file_state.file_read_context("read-b1", lambda: {"read-a1": first}):
                other = await shared_tool.execute(path=str(f))
        finally:
            reset_file_states(token)
        assert "line 0" in other


# ---------------------------------------------------------------------------
# Through the runner (nanobot tests/agent/test_runner_file_reads.py)
# ---------------------------------------------------------------------------


def _tools(tmp_path):
    (tmp_path / "data.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(workspace=tmp_path))
    return tools


def _read_call(call_id, **extra):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name="read_file", arguments={"path": "data.txt", **extra})],
        finish_reason="tool_calls",
    )


async def _read(tools, call_id, history=None, **kwargs):
    fake = FakeProvider([_read_call(call_id), "done"])
    initial = [*(history or []), {"role": "user", "content": "Read data.txt again."}]
    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=initial,
            tools=tools,
            provider=fake,
            max_iterations=2,
            max_tool_result_chars=128_000,
            **kwargs,
        )
    )
    observation = next(
        m["content"] for m in result.messages if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    )
    return result, observation, fake


@pytest.mark.parametrize("retained", ["original", "summary", "truncated", "stub_only", "orphan"])
async def test_repeat_requires_original_read_result_in_current_context(tmp_path, retained):
    tools = _tools(tmp_path)
    first, contents, _ = await _read(tools, "read-1")
    second, stub, _ = await _read(tools, "read-2", first.messages)
    assert "1| alpha" in contents
    assert stub == STUB

    history = deepcopy(second.messages)
    if retained == "summary":
        history = [{"role": "user", "content": "Summary: data.txt contains alpha and beta."}]
    elif retained == "truncated":
        for message in history:
            if message.get("tool_call_id") == "read-1":
                message["content"] = "1| alpha\n[truncated]"
    elif retained in {"stub_only", "orphan"}:
        history = [
            message
            for message in history
            if not any(call["id"] == "read-1" for call in message.get("tool_calls", []))
            and (retained == "orphan" or message.get("tool_call_id") != "read-1")
        ]

    _, repeated, _ = await _read(tools, "read-3", history)
    assert repeated == (stub if retained == "original" else contents)


async def test_dedup_uses_compacted_request_instead_of_raw_transcript(tmp_path, monkeypatch):
    tools = _tools(tmp_path)
    first, contents, _ = await _read(tools, "read-1")

    def estimate(messages, _tools):
        contains_original = any(message.get("tool_call_id") == "read-1" for message in messages)
        return (600 if contains_original else 100), "test-counter"

    monkeypatch.setattr("nanoclaw.agent.context_governance.estimate_prompt_tokens", estimate)
    summaries = []

    async def consolidate(accepted, previous):
        summaries.append(accepted)
        return "The file was inspected; its original text was omitted."

    result, repeated, fake = await _read(
        tools,
        "read-2",
        first.messages,
        context_window_tokens=1_624,
        max_tokens=100,
        consolidate_history=consolidate,
        accepted_boundary=len(first.messages),
        summary_transcript_builder=lambda summary: [{"role": "system", "content": summary}],
    )

    assert any(message.get("tool_call_id") == "read-1" for message in result.messages)
    assert not any(message.get("tool_call_id") == "read-1" for message in fake.calls[0].messages)
    assert repeated == contents
    assert len(summaries) == 1
    assert result.summary_checkpoint is not None


async def test_direct_read_without_model_context_never_omits_contents(tmp_path):
    tools = _tools(tmp_path)
    first = await tools.execute("read_file", {"path": "data.txt"})
    second = await tools.execute("read_file", {"path": "data.txt"})
    assert second == first
    assert "1| alpha" in second


@pytest.mark.parametrize("operation", ["list_dir", "first_read", "forced_read", "repeat_batch"])
async def test_only_repeat_reads_scan_model_results_once_per_batch(tmp_path, operation):
    tools = _tools(tmp_path)
    tools.register(ListDirTool(workspace=tmp_path))
    first, contents, _ = await _read(tools, "read-1")

    class CountedMessages(list):
        scans = 0

        def __iter__(self):
            self.scans += 1
            return super().__iter__()

    messages = CountedMessages(first.messages)
    name, arguments = "read_file", {"path": "data.txt"}
    if operation == "list_dir":
        name, arguments = "list_dir", {"path": "."}
    elif operation == "first_read":
        (tmp_path / "fresh.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        arguments = {"path": "fresh.txt"}
    elif operation == "forced_read":
        arguments["force"] = True
    calls = [ToolCallRequest(id="batch-1", name=name, arguments=arguments)]
    if operation == "repeat_batch":
        calls.append(ToolCallRequest(id="batch-2", name=name, arguments=dict(arguments)))

    runner = AgentRunner()
    spec = AgentRunSpec(initial_messages=[], tools=tools, provider=FakeProvider([]))
    read_results = runner._model_tool_results(messages)
    results = [
        (await runner._execute_tool_call(spec, call, read_results=read_results))[0] for call in calls
    ]

    if operation == "repeat_batch":
        assert results == [STUB] * 2
        assert messages.scans == 1
    else:
        assert messages.scans == 0
        if operation == "list_dir":
            assert "data.txt" in results[0]
        else:
            assert results == [contents]
    # A later direct read has no model input proving that its contents are visible.
    assert await tools.execute("read_file", {"path": "data.txt"}) == contents


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_failed_file_read_restores_context(tmp_path, monkeypatch, error):
    tools = _tools(tmp_path)
    first, contents, _ = await _read(tools, "read-1")
    tool = tools.get("read_file")
    original_execute = tool.execute

    async def fail(**_kwargs):
        raise error("read interrupted")

    monkeypatch.setattr(tool, "execute", fail)
    runner = AgentRunner()
    spec = AgentRunSpec(initial_messages=[], tools=tools, provider=FakeProvider([]))
    request = runner._execute_tool_call(
        spec,
        ToolCallRequest(id="failed-read", name="read_file", arguments={"path": "data.txt"}),
        hook=AgentHook(),
        context=AgentHookContext(iteration=0, messages=[]),
        read_results=runner._model_tool_results(first.messages),
    )
    if error is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            await request
    else:
        result, _ = await request
        assert result.startswith("Error: RuntimeError: read interrupted")

    monkeypatch.setattr(tool, "execute", original_execute)
    assert await tools.execute("read_file", {"path": "data.txt"}) == contents


async def test_concurrent_sessions_keep_separate_read_contexts(tmp_path, monkeypatch):
    tools = _tools(tmp_path)
    session_a, session_b = FileStates(), FileStates()

    async def session_read(states, call_id, history=None):
        token = bind_file_states(states)
        try:
            return await _read(tools, call_id, history)
        finally:
            reset_file_states(token)

    first, contents, _ = await session_read(session_a, "read-a1")
    tool = tools.get("read_file")
    execute = tool.execute

    async def interleave(**kwargs):
        await asyncio.sleep(0)
        return await execute(**kwargs)

    monkeypatch.setattr(tool, "execute", interleave)
    (_, repeated, _), (_, other_session, _) = await asyncio.gather(
        session_read(session_a, "read-a2", first.messages),
        session_read(session_b, "read-b1"),
    )
    assert repeated == STUB
    assert other_session == contents


async def test_two_reads_in_one_response_both_return_contents(tmp_path):
    """The first read's result is not in the input that asked for both."""
    tools = _tools(tmp_path)
    fake = FakeProvider([
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(id="r1", name="read_file", arguments={"path": "data.txt"}),
                ToolCallRequest(id="r2", name="read_file", arguments={"path": "data.txt"}),
            ],
            finish_reason="tool_calls",
        ),
        "done",
    ])
    result = await AgentRunner().run(
        AgentRunSpec(initial_messages=[{"role": "user", "content": "x"}], tools=tools, provider=fake)
    )
    observations = [m["content"] for m in result.messages if m.get("role") == "tool"]
    assert observations[0] == observations[1]
    assert "1| alpha" in observations[0]


# ---------------------------------------------------------------------------
# Whose state: subagents, Dream, the loop's sessions
# ---------------------------------------------------------------------------


async def test_subagent_build_tools_isolates_file_read_state(tmp_path):
    """Each spawned subagent needs a fresh file-state cache (nanobot test_subagent.py)."""
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    sm = SubagentManager(tmp_path)
    first_read = sm._build_tools().get("read_file")
    second_read = sm._build_tools().get("read_file")

    assert first_read is not second_read
    assert first_read._file_states is not second_read._file_states
    # Bound session state is ignored: the child has its own.
    token = bind_file_states(FileStates())
    try:
        assert first_read._file_states is not file_state.current_file_states(FileStates())
    finally:
        reset_file_states(token)
    assert (await first_read.execute(path="note.txt")).startswith("1| hello")
    second_result = await second_read.execute(path="note.txt")
    assert second_result.startswith("1| hello")
    assert "File unchanged" not in second_result


def test_dream_tools_share_their_own_state(tmp_path):
    """nanobot memory.py:567: one FileStates for Dream's file tools, not the session's."""
    from nanoclaw.agent.memory import MemoryStore

    tools = MemoryStore(tmp_path).build_dream_tools()
    states = {id(tools.get(name)._inner._file_states) for name in ("read_file", "edit_file", "write_file")}
    assert len(states) == 1
    bound = FileStates()
    token = bind_file_states(bound)
    try:
        assert tools.get("read_file")._inner._file_states is not bound
    finally:
        reset_file_states(token)


def test_default_tools_share_one_explicit_state(tmp_path):
    states = FileStates()
    tools = build_default_tools(tmp_path, file_states=states)
    for name in ("read_file", "write_file", "edit_file", "list_dir"):
        assert tools.get(name)._file_states is states


async def test_loop_binds_session_state_and_new_session_discards_it(tmp_path):
    (tmp_path / "data.txt").write_text("alpha\n", encoding="utf-8")
    fake = FakeProvider([_read_call("r1"), "done", _read_call("r2"), "done"])
    loop = AgentLoop(fake, tmp_path)
    await loop.process_direct("read it", session_key="cli:a")
    states = loop._file_state_store.for_session("cli:a")
    assert states.get(tmp_path / "data.txt") is not None
    await loop.process_direct("read it again", session_key="cli:a")
    tool_messages = [m for m in loop.sessions.get_or_create("cli:a").messages if m.get("role") == "tool"]
    assert tool_messages[-1]["content"] == STUB

    await loop.new_session("cli:a")
    assert loop._file_state_store.for_session("cli:a") is not states
