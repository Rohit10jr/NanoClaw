"""Context governance: the model-facing request, and compaction before every
model request (nanobot agent/context_governance.py, c4dac64)."""

import asyncio
import copy
from types import SimpleNamespace
from typing import Any

import pytest

from nanoclaw.agent.context_governance import (
    BACKFILL_CONTENT,
    PENDING_READ_NOTICE,
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    ContextWindowExceededError,
    ModelRequestState,
)
from nanoclaw.agent.context_governance import merge_adjacent_user_messages_for_model as merge
from nanoclaw.config import Config
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.messages import MALFORMED_TOOL_CALL_NOTE
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.tools import Tool, ToolRegistry
from nanoclaw.events import ContextCompactionEvent, EventSink
from nanoclaw.providers import FakeProvider, LLMResponse, LLMUsage, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.session.summary import SUMMARY_CONTINUATION_TEXT, SessionSummaryCheckpoint, is_summary_checkpoint
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.templates import sync_workspace_templates
from nanoclaw.utils.tokens import estimate_prompt_tokens
from nanoclaw.runtime_context import (
    RuntimeContextBlock,
    append_runtime_context,
    detach_runtime_context,
    reattach_runtime_context,
)


def user(text, *blocks):
    content, marker = append_runtime_context(text, [RuntimeContextBlock(s, t) for s, t in blocks])
    message = {"role": "user", "content": content}
    if marker:
        message["_meta"] = {"runtime_context": marker}
    return message


def test_runtime_context_stays_at_the_end_after_a_merge():
    out = merge([user("hi", ("time", "[T]")), {"role": "user", "content": "Please answer"}])
    assert len(out) == 1
    assert out[0]["content"] == "hi\n\nPlease answer\n\n[T]"
    assert out[0]["_meta"]["runtime_context"]["suffix"] == "[T]"


def test_two_messages_with_runtime_context_keep_both_blocks_in_order():
    out = merge([user("a", ("time", "[T1]")), user("b", ("time", "[T2]"))])
    assert out[0]["content"] == "a\n\nb\n\n[T1]\n\n[T2]"
    assert out[0]["_meta"]["runtime_context"]["sources"] == ["time", "time"]


def test_plain_merges_and_non_adjacent_messages():
    assert merge([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]) == [
        {"role": "user", "content": "a\n\nb"}
    ]
    assert merge([{"role": "user", "content": ""}, {"role": "user", "content": "b"}])[0]["content"] == "b"
    apart = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "b"}]
    assert merge(apart) == apart


def test_a_marker_that_no_longer_matches_falls_back_to_a_plain_merge():
    stale = {"role": "user", "content": "edited", "_meta": {"runtime_context": {"version": 1, "sources": ["t"], "suffix": "[T]"}}}
    assert merge([stale, {"role": "user", "content": "b"}])[0]["content"] == "edited\n\nb"


def test_merge_does_not_change_the_input():
    messages = [user("hi", ("time", "[T]")), {"role": "user", "content": "x"}]
    before = copy.deepcopy(messages)
    merge(messages)
    assert messages == before


def test_detach_and_reattach_round_trip():
    content, marker = append_runtime_context("hi", [RuntimeContextBlock("t", "[T]")])
    clean, sources, blocks = detach_runtime_context(content, marker)
    assert (clean, sources, blocks) == ("hi", ["t"], [{"type": "text", "text": "[T]"}])
    assert reattach_runtime_context(clean, sources, blocks) == (content, marker)
    assert detach_runtime_context("changed", marker) is None
    assert detach_runtime_context("hi[T]", marker) is None  # the blank-line separator must be there
    assert detach_runtime_context("[T]", marker)[0] == ""
    parts, marker = append_runtime_context([{"type": "text", "text": "x"}], [RuntimeContextBlock("t", "[T]")])
    assert detach_runtime_context(parts, marker)[0] == [{"type": "text", "text": "x"}]


def test_summary_continuation_and_hidden_messages_are_not_merged():
    # nanobot context_governance.py:334-343
    hidden = {"role": "user", "content": "report", "_hidden_history": {"kind": "subagent_result"}}
    continuation = {"role": "user", "content": SUMMARY_CONTINUATION_TEXT}
    plain = {"role": "user", "content": "q"}
    assert len(merge([plain, hidden, plain])) == 3
    assert len(merge([continuation, plain])) == 2


# --- transcript repair (model-facing copy only) ----------------------------------

GOV = ContextGovernor()
CONFIG = ContextGovernanceConfig(session_key="s", max_tool_result_chars=50)


def tool_call(call_id, name="fetch"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_repair_strips_placeholders_nameless_calls_orphans_and_backfills():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "[Previous assistant message omitted.]"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call("a1"), tool_call("a2")]},
        {"role": "tool", "tool_call_id": "a1", "name": "fetch", "content": "one"},
        {"role": "tool", "tool_call_id": "zz", "name": "fetch", "content": "orphan"},
        {"role": "assistant", "content": "text", "tool_calls": [tool_call("b1", name="")]},
        {"role": "tool", "tool_call_id": "b1", "name": "", "content": "result of a nameless call"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call("c1", name=None)]},
        {"role": "tool", "tool_call_id": "a1", "name": "fetch", "content": "duplicate"},
        {"role": "user", "content": "q2"},
    ]
    before = copy.deepcopy(messages)
    out = GOV.prepare_for_model(ContextGovernanceConfig(session_key="s", max_tool_result_chars=1000), messages)
    assert messages == before  # the saved transcript is never changed
    assert out == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call("a1"), tool_call("a2")]},
        {"role": "tool", "tool_call_id": "a1", "name": "fetch", "content": "one"},
        {"role": "tool", "tool_call_id": "a2", "name": "fetch", "content": BACKFILL_CONTENT},
        {"role": "assistant", "content": "text"},  # its only call had no name; the empty one is dropped
        {"role": "user", "content": "q2"},
    ]


def test_repair_returns_the_same_list_when_nothing_changes():
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    for repair in (
        GOV.strip_placeholder_assistant_messages,
        GOV.strip_malformed_tool_calls,
        GOV.drop_orphan_tool_results,
        GOV.backfill_missing_tool_results,
    ):
        assert repair(messages) is messages
    assert GOV.apply_tool_result_budget(CONFIG, messages) is messages


def test_tool_result_budget_also_applies_to_older_results():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [tool_call("a1"), tool_call("r1", "read_file")]},
        {"role": "tool", "tool_call_id": "a1", "name": "fetch", "content": "z" * 200},
        {"role": "tool", "tool_call_id": "r1", "name": "read_file", "content": "r" * 200},
    ]
    out = GOV.apply_tool_result_budget(CONFIG, messages)
    assert out[1]["content"].startswith("z" * 50) and out[1]["content"].endswith("(truncated)")
    assert out[2]["content"] == "r" * 200  # read_file pages itself; never cut
    assert messages[1]["content"] == "z" * 200
    empty = [{"role": "tool", "tool_call_id": "x", "name": "exec", "content": ""}]
    assert GOV.apply_tool_result_budget(CONFIG, empty)[0]["content"] == "(exec completed with no output)"
    blocks = [{"type": "text", "text": "b" * 200}, {"type": "image_url", "image_url": {"url": "x"}}]
    out = GOV.normalize_tool_result(CONFIG, "c", "fetch", blocks)
    assert out[0]["text"].endswith("(truncated)") and out[1] == blocks[1]


# --- measuring ----------------------------------------------------------------------

BUDGETED = ContextGovernanceConfig(
    session_key="s", max_tool_result_chars=1000, context_window_tokens=1000 + 100 + 1024, max_tokens=100
)


def pressure(messages, usage, **flags):
    defaults = {"usage_matches_messages": False, "usage_prefixes_messages": False, "tool_definitions": None}
    return GOV.request_pressure(BUDGETED, messages, usage, **{**defaults, **flags})


def test_pressure_uses_the_estimate_then_provider_counts():
    small = [{"role": "user", "content": "hi"}]
    assert GOV.input_budget(BUDGETED) == 1000
    assert pressure(small, None) is None
    measured, _ = pressure([{"role": "user", "content": "x" * 1200}], None)
    assert measured > 1000
    # Exactly this request measured by the provider: its count wins either way.
    exact = LLMUsage.reported(input_tokens=1000, output_tokens=1)
    assert pressure(small, exact, usage_matches_messages=True) == (1000, "matching provider usage")
    # A prefix measured: the provider's count only when it is higher (D8, faf1efa).
    higher = LLMUsage.reported(input_tokens=1500, output_tokens=1)
    assert pressure(small, higher, usage_prefixes_messages=True) == (1500, "prior provider usage plus pending messages")
    assert pressure(small, higher) is None  # not a prefix: the estimate only
    estimated = LLMUsage.estimated(input_tokens=1500, output_tokens=1)
    assert pressure(small, estimated, usage_prefixes_messages=True) is None  # an estimate is no measurement


def test_no_window_means_no_pressure():
    config = ContextGovernanceConfig(session_key="s", max_tool_result_chars=10)
    assert GOV.input_budget(config) == 0
    flags = {"usage_matches_messages": False, "usage_prefixes_messages": False, "tool_definitions": None}
    assert GOV.request_pressure(config, [{"role": "user", "content": "x" * 10**6}], None, **flags) is None


def test_transcript_state_keeps_h_and_delta_apart():
    raw = [SYSTEM, {"role": "user", "content": "old"}, {"role": "assistant", "content": "a"}, TASK]
    state = ContextCompactionState.from_transcript(
        raw,
        accepted_boundary=3,
        active_summary="prev",
        summary_transcript_builder=lambda summary: [SYSTEM],
        consolidate_history=None,
    )
    assert state.request_messages(raw) == raw and state.delta_after_accepted(raw) == [TASK]
    sent = [{"role": "system", "content": "as sent"}, TASK]
    state.accepted_messages, state.raw_accepted_boundary = sent, 4
    raw.append({"role": "assistant", "content": "b"})
    assert state.request_messages(raw) == [*sent, {"role": "assistant", "content": "b"}]


def test_standalone_transcripts_put_the_summary_in_the_last_system_message():
    state = ContextCompactionState.from_messages([SYSTEM, TASK], None)
    assert state.raw_accepted_boundary == 2
    assert state.summary_transcript_builder("- S") == [
        {
            "role": "system",
            "content": "You are a test agent.\n\n---\n\n[Archived Context Summary]\n\nPrevious conversation summary:\n- S",
        }
    ]
    assert ContextCompactionState.from_messages([TASK], None).summary_transcript_builder("- S")[0]["role"] == "system"


# --- the runner: compaction before every request ------------------------------------------


class Fetch(Tool):
    """Returns ``size`` characters."""

    def __init__(self, size=3000, name="fetch"):
        self.size, self._name = size, name

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "Fetch."

    @property
    def parameters(self):
        return {"type": "object", "properties": {"id": {"type": "string"}}}

    async def execute(self, **kwargs: Any) -> str:
        return "L" * self.size


def fetch_call(call_id, name="fetch", usage=None):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments={"id": call_id})],
        finish_reason="tool_calls",
        usage=usage,
    )


class Summarizer:
    """A ``consolidate_history`` that records what it was asked to summarize."""

    def __init__(self, *summaries):
        self.summaries = list(summaries)
        self.calls: list[tuple[list[dict[str, Any]], str | None]] = []

    async def __call__(self, accepted, previous):
        self.calls.append((accepted, previous))
        return self.summaries.pop(0) if self.summaries else None


SYSTEM = {"role": "system", "content": "You are a test agent."}
TASK = {"role": "user", "content": "fetch logs"}


def registry(*tools):
    reg = ToolRegistry()
    for tool in tools or (Fetch(),):
        reg.register(tool)
    return reg


def window_between(low, high):
    """A window whose input budget lies between two request sizes."""
    return (low + high) // 2 + 100 + 1024


def sizes_of(fake):
    return [estimate_prompt_tokens(c.messages, c.tools)[0] for c in fake.calls]


async def sizes_without_limit(script, reg):
    fake = FakeProvider(list(script))
    await AgentRunner().run(AgentRunSpec(initial_messages=[SYSTEM, TASK], tools=reg, provider=fake, max_tokens=100))
    return sizes_of(fake)


async def run_spec(fake, reg, window, summarizer, *, events=None):
    return await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[SYSTEM, TASK],
            tools=reg,
            provider=fake,
            max_tokens=100,
            context_window_tokens=window,
            consolidate_history=summarizer,
            events=events or EventSink(),
        )
    )


async def test_a_turn_that_overflows_between_tool_calls_is_compacted():
    reg = registry()
    script = [fetch_call("c1"), fetch_call("c2"), "done"]
    sizes = await sizes_without_limit(script, reg)
    fake, summarizer = FakeProvider(list(script)), Summarizer("- S")
    seen = []

    async def publish(event):
        if isinstance(event, ContextCompactionEvent):
            seen.append((event.phase, event.reason))

    result = await run_spec(fake, reg, window_between(sizes[1], sizes[2]), summarizer, events=EventSink(publish))
    assert result.final_content == "done" and len(fake.calls) == 3
    # H is the second request exactly as sent: system, task, first call and result.
    accepted, previous = summarizer.calls[0]
    assert previous is None and [m["role"] for m in accepted] == ["system", "user", "assistant", "tool"]
    # The third request: the summary in the system prompt, the resume instruction, then only Δ.
    third = fake.calls[2].messages
    assert third[0]["content"].endswith("[Archived Context Summary]\n\nPrevious conversation summary:\n- S")
    assert third[1] == {"role": "user", "content": SUMMARY_CONTINUATION_TEXT}
    assert [m["role"] for m in third[2:]] == ["assistant", "tool"] and third[2]["tool_calls"][0]["id"] == "c2"
    assert seen == [("started", "context_pressure"), ("succeeded", "context_pressure")]
    # The summary replaces the transcript before the second call; the transcript keeps everything.
    assert result.summary_checkpoint == SessionSummaryCheckpoint("- S", 4)
    roles = ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert [m["role"] for m in result.messages] == roles


async def test_later_requests_build_on_the_compacted_request():
    reg = registry()
    script = [fetch_call(f"c{i}") for i in range(1, 7)] + ["done"]
    sizes = await sizes_without_limit(script, reg)
    fake, summarizer = FakeProvider(list(script)), Summarizer("- S1", "- S2")
    # Requests 4 and 6 overflow (three steps each); a compacted request keeps one step.
    result = await run_spec(fake, reg, window_between(sizes[2], sizes[3]), summarizer)
    assert result.final_content == "done" and len(fake.calls) == 7
    # The fifth request is the fourth (compacted) one plus the new step, not the raw transcript.
    fourth, fifth = fake.calls[3].messages, fake.calls[4].messages
    assert fifth[: len(fourth)] == fourth and len(fifth) == len(fourth) + 2
    # The second compaction summarizes the compacted request and updates the first summary.
    assert len(summarizer.calls) == 2 and summarizer.calls[1][1] == "- S1"
    assert summarizer.calls[1][0][1] == {"role": "user", "content": SUMMARY_CONTINUATION_TEXT}
    final_system = fake.calls[-1].messages[0]["content"]
    assert "- S2" in final_system and "- S1" not in final_system
    assert result.summary_checkpoint == SessionSummaryCheckpoint("- S2", 10)


async def test_a_compaction_without_a_summary_fails_the_run():
    reg = registry()
    script = [fetch_call("c1"), fetch_call("c2"), "done"]
    sizes = await sizes_without_limit(script, reg)
    seen = []

    async def publish(event):
        seen.append(event.phase)

    with pytest.raises(ContextWindowExceededError, match="exceeds the local context budget"):
        window = window_between(sizes[1], sizes[2])
        await run_spec(FakeProvider(list(script)), reg, window, Summarizer(), events=EventSink(publish))
    assert seen == ["started", "failed"]


async def test_a_window_needs_a_summarizer():
    spec = AgentRunSpec(initial_messages=[TASK], tools=registry(), provider=FakeProvider(["x"]), context_window_tokens=10)
    with pytest.raises(ValueError, match="consolidate_history"):
        await AgentRunner().run(spec)


async def test_an_unread_file_that_still_does_not_fit_is_replaced_by_a_notice():
    # nanobot D9 (c9bd5eb): the largest unread read_file results become a recoverable error.
    reg = registry(Fetch(size=20_000, name="read_file"))
    first = (await sizes_without_limit(["x"], reg))[0]
    fake, summarizer = FakeProvider([fetch_call("r1", "read_file"), "done"]), Summarizer("- S")
    result = await run_spec(fake, reg, first + 3000 + 100 + 1024, summarizer)
    assert result.final_content == "done"
    assert fake.calls[1].messages[-1]["content"] == PENDING_READ_NOTICE
    # The run's transcript says the same, so a replay never shows what the model never read.
    assert result.messages[3]["content"] == PENDING_READ_NOTICE


async def test_too_big_with_no_unread_file_to_replace_is_an_error():
    reg = registry(Fetch(size=20_000))
    first = (await sizes_without_limit(["x"], reg))[0]
    with pytest.raises(ContextWindowExceededError):
        await run_spec(FakeProvider([fetch_call("c1"), "done"]), reg, first + 3000 + 100 + 1024, Summarizer("- S"))


async def test_provider_reported_context_tokens_trigger_compaction():
    # nanobot D8 (faf1efa): the previous request is a prefix of this one and the
    # provider counted more than the estimate.
    reg = registry(Fetch(size=10))
    first = (await sizes_without_limit(["x"], reg))[0]
    window = first + 2000 + 100 + 1024
    usage = LLMUsage.reported(input_tokens=first + 2000, output_tokens=5)
    fake, summarizer = FakeProvider([fetch_call("c1", usage=usage), "done"]), Summarizer("- S")
    result = await run_spec(fake, reg, window, summarizer)
    assert len(summarizer.calls) == 1 and result.summary_checkpoint == SessionSummaryCheckpoint("- S", 2)
    # Without the report, the same run fits.
    fake, summarizer = FakeProvider([fetch_call("c1"), "done"]), Summarizer("- S")
    result = await run_spec(fake, reg, window, summarizer)
    assert summarizer.calls == [] and result.summary_checkpoint is None


async def test_a_fallback_models_reported_tokens_do_not_trigger_compaction():
    # Mutation-testing gap (layer 20): another model's count cannot bound the
    # next request (nanobot runner.py:1340), so the run falls back to estimates.
    reg = registry(Fetch(size=10))
    first = (await sizes_without_limit(["x"], reg))[0]
    window = first + 2000 + 100 + 1024
    answer = fetch_call("c1", usage=LLMUsage.reported(input_tokens=first + 2000, output_tokens=5))
    answer.is_fallback = True
    fake, summarizer = FakeProvider([answer, "done"]), Summarizer("- S")
    result = await run_spec(fake, reg, window, summarizer)
    assert summarizer.calls == [] and result.summary_checkpoint is None


async def test_a_retry_note_stays_in_the_accepted_request():
    # nanobot runner.py:1076 adds the note to the request as sent, which becomes H.
    nameless = LLMResponse(
        content=None, tool_calls=[ToolCallRequest(id="x", name="", arguments={})], finish_reason="tool_calls"
    )
    fake = FakeProvider([nameless, fetch_call("c1"), "done"])
    await AgentRunner().run(AgentRunSpec(initial_messages=[SYSTEM, TASK], tools=registry(), provider=fake))
    last = fake.calls[-1].messages
    assert MALFORMED_TOOL_CALL_NOTE in last[1]["content"]  # merged into the task message
    assert [m["role"] for m in last] == ["system", "user", "assistant", "tool"]


# --- the loop: the summary is committed where H ends -----------------------------------------


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    sync_workspace_templates(path)
    return path


def make_loop(ws, tmp_path, script=(), *, window=0, size=3000, compaction="nanobot"):
    return AgentLoop(
        FakeProvider(list(script)),
        ws,
        tools=registry(Fetch(size=size)),
        sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"),
        max_tokens=100,
        context_window_tokens=window,
        compaction=compaction,
    )


async def turn_sizes(ws, tmp_path, script, text):
    loop = make_loop(ws, tmp_path, script)
    await loop.process_direct(text, session_key="probe:x", ephemeral=True)
    return sizes_of(loop.provider)


async def test_mid_turn_compaction_marks_the_summary_inside_the_turn(ws, tmp_path):
    script = [fetch_call("c1"), fetch_call("c2"), "done"]
    sizes = await turn_sizes(ws, tmp_path, script, "fetch two")
    # A temporary turn has no time block; leave room for it.
    loop = make_loop(ws, tmp_path, [script[0], script[1], "- S", "done"], window=window_between(sizes[1], sizes[2]) + 40)
    result = await loop.process_direct("fetch two")
    assert result.final_content == "done"
    archive = loop.provider.calls[2].messages
    assert archive[-1]["content"].startswith("Create a compact replacement checkpoint")
    assert [m["role"] for m in archive] == ["system", "user", "assistant", "tool", "user"]

    session = SessionManager(ws, sessions_root=tmp_path / "sessions").get_or_create("cli:direct")
    roles = ["marker" if is_summary_checkpoint(m) else m["role"] for m in session.messages]
    assert roles == ["user", "assistant", "tool", "marker", "assistant", "tool", "assistant"]
    assert session.last_archived == 3 and session.metadata["_last_summary"]["text"] == "- S"
    assert [e["content"] for e in loop.memory.read_unprocessed_history(0)] == ["- S"]

    # The next turn replays from the summary: the second step and the answer.
    loop.provider.queue("next")
    await loop.process_direct("and now?")
    sent = loop.provider.calls[-1].messages
    assert "- S" in sent[0]["content"]
    assert [m["role"] for m in sent] == ["system", "assistant", "tool", "assistant", "user"]


async def test_temporary_turns_never_commit_a_mid_turn_summary(ws, tmp_path):
    script = [fetch_call("c1"), fetch_call("c2"), "done"]
    sizes = await turn_sizes(ws, tmp_path, script, "fetch two")
    loop = make_loop(ws, tmp_path, [script[0], script[1], "- S", "done"], window=window_between(sizes[1], sizes[2]))
    result = await loop.process_direct("fetch two", session_key="tmp:x", ephemeral=True)
    assert result.final_content == "done" and len(loop.provider.calls) == 4
    assert loop.memory.read_unprocessed_history(0) == []
    assert loop.sessions.get_or_create("tmp:x").messages == []


@pytest.mark.parametrize("mode", ["nanobot", "fixed"])
async def test_the_bug_4_switch_applies_to_mid_turn_compaction(ws, tmp_path, mode):
    """docs/notes/06-memory.md "Compaction A/B": the accepted history is too big
    for one archive request; ``nanobot`` raw-dumps it, ``fixed`` summarizes it in parts."""
    loop = make_loop(ws, tmp_path, ["a1", "a2"], size=600, compaction=mode)
    await loop.process_direct("first " + "f" * 3000)
    await loop.process_direct("second " + "s" * 3000)
    session = loop.sessions.get_or_create("cli:direct")
    turn = loop.context.build_messages(session.get_history(), "go", channel="cli")
    base = estimate_prompt_tokens(turn, loop.tools.get_definitions())[0]
    # The turn's first request fits (with its time block); the tool result doesn't.
    loop.context_window_tokens = base + 300 + 100 + 1024
    if mode == "nanobot":
        loop.provider.queue(fetch_call("c1"), "done")
    else:
        loop.provider.queue(fetch_call("c1"), "- P1", "- P2", "done")
    result = await loop.process_direct("go")
    assert result.final_content == "done"
    journal = [e["content"] for e in loop.memory.read_unprocessed_history(0)]
    summary = loop.sessions.get_or_create("cli:direct").metadata["_last_summary"]["text"]
    if mode == "nanobot":
        assert journal and all(entry.startswith("[RAW]") for entry in journal)
        assert summary.startswith("[RAW]")
    else:
        assert journal == ["- P1", "- P2"] and summary == "- P1\n\n- P2"
        assert len(loop.provider.calls) == 2 + 4


async def test_subagents_compact_their_own_runs_without_journaling(ws, tmp_path):
    loop = make_loop(ws, tmp_path)
    manager = loop.subagents
    assert manager.consolidator is loop.consolidator
    manager._build_tools = lambda *_a, **_k: registry(Fetch(size=3000))
    script = [fetch_call("c1"), fetch_call("c2"), "child done"]
    probe = FakeProvider(list(script))
    await manager.run_inline("task", runtime=LLMRuntime(probe, "m", 100, 0.1, context_window_tokens=0))
    sizes = sizes_of(probe)

    fake = FakeProvider([script[0], script[1], "- child summary", "child done"])
    runtime = LLMRuntime(fake, "m", 100, 0.1, context_window_tokens=window_between(sizes[1], sizes[2]))
    seen = []
    original = loop.consolidator.summarize_transcript

    async def spy(*args, **kwargs):
        seen.append((kwargs["session_key"], kwargs["persist"]))
        return await original(*args, **kwargs)

    loop.consolidator.summarize_transcript = spy
    assert await manager.run_inline("task", runtime=runtime) == "child done"
    assert len(seen) == 1 and seen[0][0].startswith("subagent:") and seen[0][1] is False
    last = fake.calls[-1].messages
    assert "[Archived Context Summary]" in last[0]["content"] and "- child summary" in last[0]["content"]
    assert last[1] == {"role": "user", "content": SUMMARY_CONTINUATION_TEXT}
    assert loop.memory.read_unprocessed_history(0) == []


def test_save_turn_ignores_a_boundary_outside_the_turn(ws, tmp_path):
    # nanobot loop.py:2470 ``_validated_checkpoint_boundary``
    loop = make_loop(ws, tmp_path)
    session = loop.sessions.get_or_create("k")
    turn = [SYSTEM, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    loop._save_turn(session, turn, 1, summary_checkpoint=SessionSummaryCheckpoint("- S", 7))
    assert not any(is_summary_checkpoint(m) for m in session.messages)
    loop._save_turn(session, turn, 1, summary_checkpoint=SessionSummaryCheckpoint("- S", 3))
    assert is_summary_checkpoint(session.messages[-1]) and session.last_archived == 4


async def test_a_retry_after_compaction_resends_the_compacted_request():
    # The note goes on the request as sent (nanobot runner.py:1076), so the
    # retry does not bring back the history that was just summarized.
    reg = registry()
    nameless = LLMResponse(
        content=None, tool_calls=[ToolCallRequest(id="x", name="", arguments={})], finish_reason="tool_calls"
    )
    sizes = await sizes_without_limit([fetch_call("c1"), fetch_call("c2"), "done"], reg)
    fake, summarizer = FakeProvider([fetch_call("c1"), fetch_call("c2"), nameless, "done"]), Summarizer("- S")
    result = await run_spec(fake, reg, window_between(sizes[1], sizes[2]), summarizer)
    assert result.final_content == "done" and len(summarizer.calls) == 1
    retry = fake.calls[-1].messages
    assert "- S" in retry[0]["content"] and MALFORMED_TOOL_CALL_NOTE in retry[-1]["content"]


# --- found by mutation checks (layer 15) ---------------------------------------------------


def test_list_tool_results_get_the_empty_marker_and_read_file_is_exempt():
    marker = "(exec completed with no output)"
    assert GOV.normalize_tool_result(CONFIG, "c", "exec", []) == marker
    assert GOV.normalize_tool_result(CONFIG, "c", "exec", [{"type": "text", "text": "  "}]) == marker
    long_read = [{"type": "text", "text": "r" * 200}]
    assert GOV.normalize_tool_result(CONFIG, "c", "read_file", long_read) == long_read


def test_repair_keeps_a_placeholder_with_calls_and_calls_named_at_the_top_level():
    placeholder = [
        {"role": "assistant", "content": "[Previous assistant message omitted.]", "tool_calls": [tool_call("p")]}
    ]
    assert GOV.strip_placeholder_assistant_messages(placeholder) is placeholder
    top_level = [{"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "name": "fetch"}]}]
    assert GOV.strip_malformed_tool_calls(top_level) is top_level


def test_backfill_puts_each_result_after_its_own_call():
    a1 = {"role": "assistant", "content": None, "tool_calls": [tool_call("a1")]}
    a2 = {"role": "assistant", "content": None, "tool_calls": [tool_call("b1")]}
    out = GOV.backfill_missing_tool_results([a1, {"role": "user", "content": "q"}, a2])
    assert [(m["role"], m.get("tool_call_id")) for m in out] == [
        ("assistant", None),
        ("tool", "a1"),
        ("user", None),
        ("assistant", None),
        ("tool", "b1"),
    ]


def test_a_prefix_measurement_never_lowers_the_estimate():
    big = [{"role": "user", "content": "x" * 1200}]
    low = LLMUsage.reported(input_tokens=10, output_tokens=1)
    measured, _ = pressure(big, low, usage_prefixes_messages=True)
    assert measured > 1000


def test_a_request_exactly_at_the_budget_fits():
    messages = [{"role": "user", "content": "hello there"}]
    estimate = estimate_prompt_tokens(messages, None)[0]
    config = ContextGovernanceConfig(
        session_key="s", max_tool_result_chars=1000, context_window_tokens=estimate + 100 + 1024, max_tokens=100
    )
    assert GOV.ensure_request_fits(config, messages, tool_definitions=None) is messages


OTHER_TOOLS = [{"type": "function", "function": {"name": "x"}}]


def _measured_state(measured, tools):
    """A run whose last request (``measured``, sent with ``tools``) the provider counted at 1500 > budget."""
    summarizer = Summarizer("- S")
    state = ModelRequestState(config=BUDGETED, compaction=ContextCompactionState.from_messages([TASK], summarizer))
    state.messages, state.tool_definitions = measured, tools
    state.usage = LLMUsage.reported(input_tokens=1500, output_tokens=1)
    return state, summarizer


@pytest.mark.parametrize(
    ("measured", "measured_tools", "current", "compacts"),
    [
        ([TASK], None, [TASK], True),  # exactly this request: the provider's count
        ([TASK], None, [TASK, {"role": "assistant", "content": "a"}], True),  # a prefix (D8)
        ([TASK], OTHER_TOOLS, [TASK], False),  # same messages, other tools
        ([TASK], OTHER_TOOLS, [TASK, {"role": "assistant", "content": "a"}], False),  # a prefix, other tools
        ([{"role": "user", "content": "other"}], None, [TASK, {"role": "assistant", "content": "a"}], False),
    ],
    ids=["same-request", "prefix", "same-messages-other-tools", "prefix-other-tools", "not-a-prefix"],
)
async def test_provider_counts_apply_only_to_the_same_request_or_a_prefix(measured, measured_tools, current, compacts):
    state, summarizer = _measured_state(measured, measured_tools)
    await GOV.prepare_request(state, current, tool_definitions=None)
    assert bool(summarizer.calls) is compacts


def test_the_largest_unread_file_is_replaced_first():
    messages = [
        SYSTEM,
        {"role": "assistant", "content": None, "tool_calls": [tool_call("r1", "read_file"), tool_call("r2", "read_file")]},
        {"role": "tool", "tool_call_id": "r1", "name": "read_file", "content": "a" * 8000},
        {"role": "tool", "tool_call_id": "r2", "name": "read_file", "content": "b" * 4000},
    ]
    without_largest = [*messages[:2], {**messages[2], "content": PENDING_READ_NOTICE}, messages[3]]
    estimate = estimate_prompt_tokens(without_largest, None)[0]
    config = ContextGovernanceConfig(
        session_key="s", max_tool_result_chars=10**6, context_window_tokens=estimate + 100 + 1024, max_tokens=100
    )
    prepared, replacements = GOV._fit_pending_file_reads(config, messages, messages[1:], tool_definitions=None)
    assert replacements == {"r1": PENDING_READ_NOTICE}
    assert prepared[3]["content"] == "b" * 4000


async def test_an_empty_summary_fails_the_run():
    reg = registry()
    script = [fetch_call("c1"), fetch_call("c2"), "done"]
    sizes = await sizes_without_limit(script, reg)
    with pytest.raises(ContextWindowExceededError):
        await run_spec(FakeProvider(list(script)), reg, window_between(sizes[1], sizes[2]), Summarizer(""))


async def test_a_cancelled_compaction_reports_cancelled():
    reg = registry()
    script = [fetch_call("c1"), fetch_call("c2"), "done"]
    sizes = await sizes_without_limit(script, reg)
    seen = []

    async def publish(event):
        seen.append(event.phase)

    async def cancelled(accepted, previous):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        window = window_between(sizes[1], sizes[2])
        await run_spec(FakeProvider(list(script)), reg, window, cancelled, events=EventSink(publish))
    assert seen == ["started", "cancelled"]


async def test_the_max_iterations_summary_request_builds_on_the_compacted_request():
    reg = registry()
    script = [fetch_call("c1"), fetch_call("c2"), fetch_call("c3"), "wrap-up"]
    sizes = await sizes_without_limit(script, reg)
    fake, summarizer = FakeProvider(list(script)), Summarizer("- S1", "- S2")
    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[SYSTEM, TASK],
            tools=reg,
            provider=fake,
            max_tokens=100,
            max_iterations=3,
            context_window_tokens=window_between(sizes[1], sizes[2]),
            consolidate_history=summarizer,
        )
    )
    assert result.stop_reason == "max_iterations" and result.final_content == "wrap-up"
    final = fake.calls[-1].messages
    # Compacted again from the third (compacted) request: Δ is only the last step and the prompt.
    assert len(summarizer.calls) == 2 and "- S2" in final[0]["content"]
    assert [(m["role"], m.get("tool_call_id")) for m in final] == [
        ("system", None),
        ("assistant", None),
        ("tool", "c3"),
        ("user", None),
    ]


async def test_the_loop_passes_the_sessions_summary_to_the_runner(ws, tmp_path):
    loop = make_loop(ws, tmp_path, ["hi"])
    session = loop.sessions.get_or_create("cli:direct")
    session.metadata["_last_summary"] = {"text": "- earlier", "last_active": "2026-01-01T00:00:00"}
    seen = []
    original = loop.runner.run

    async def spy(spec):
        seen.append(spec.active_summary)
        return await original(spec)

    loop.runner.run = spy
    await loop.process_direct("hello")
    assert seen == ["- earlier"]


async def test_ask_reports_a_subagent_result_that_does_not_fit(capsys):
    from nanoclaw.__main__ import _deliver_subagent_results

    class Loop:
        reports = [SimpleNamespace(label="child")]

        async def next_subagent_result(self):
            return self.reports.pop() if self.reports else None

        async def process_subagent_result(self, report, **kwargs):
            raise ContextWindowExceededError(session_key="s", estimated_tokens=9, input_budget=1, source="estimate")

    assert await _deliver_subagent_results(Loop(), Config()) == 1
    assert "exceeds the local context budget" in capsys.readouterr().err
