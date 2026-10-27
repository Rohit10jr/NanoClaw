"""NANOCLAW_COMPACTION=fixed: the A/B alternative to nanobot's compaction (UPSTREAM_BUGS.md, bug 4).

"nanobot" (the default) is pinned by tests/test_compaction.py and
tests/test_memory.py; these tests check "fixed" and the switch itself.
"""

import pytest

from nanoclaw.__main__ import _compaction
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.memory import COMPACTION_FIXED, Consolidator, MemoryArchiver, MemoryStore
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.tools import ToolRegistry
from nanoclaw.providers import FakeProvider, LLMResponse
from nanoclaw.session import SessionManager
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.templates import sync_workspace_templates
from nanoclaw.utils.tokens import TRUNCATED_PREFIX, estimate_prompt_tokens, truncate_text_to_tokens_keep_end
from test_compaction import BIG, THIRD, request_size

ARCHIVE_MARKER = "Create a compact replacement checkpoint"


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    sync_workspace_templates(path)
    return path


def loop_for(ws, tmp_path, *script, window=0, compaction=COMPACTION_FIXED):
    fake = FakeProvider(list(script))
    loop = AgentLoop(
        fake,
        ws,
        tools=ToolRegistry(),
        sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"),
        max_tokens=100,
        context_window_tokens=window,
        compaction=compaction,
    )
    return loop, fake


async def three_big_turns(ws, tmp_path, overflow):
    """Three long turns; returns a window the next request (THIRD) overflows by ``overflow``."""
    loop, _ = loop_for(ws, tmp_path, BIG, BIG, BIG)
    for word in ("first ", "second ", "middle "):
        await loop.process_direct(word + BIG)
    history = loop.sessions.get_or_create("cli:direct").get_history()
    return request_size(loop, history, THIRD) - overflow - 50 + 100 + 1024


# --- the switch ---------------------------------------------------------------------------------


def test_default_is_nanobot(ws, tmp_path, monkeypatch):
    monkeypatch.delenv("NANOCLAW_COMPACTION", raising=False)
    assert _compaction() == "nanobot"
    loop = AgentLoop(FakeProvider([]), ws, sessions=SessionManager(ws, sessions_root=tmp_path / "s"))
    assert loop.consolidator.archiver.compaction == "nanobot"


def test_env_switch(monkeypatch):
    monkeypatch.setenv("NANOCLAW_COMPACTION", " Fixed ")
    assert _compaction() == "fixed"
    monkeypatch.setenv("NANOCLAW_COMPACTION", "smart")
    with pytest.raises(SystemExit, match="nanobot, fixed"):
        _compaction()


def test_unknown_mode_is_rejected(ws):
    with pytest.raises(ValueError):
        MemoryArchiver(MemoryStore(ws), ContextBuilder(ws).build_messages, lambda: [], compaction="smart")


# --- in the loop --------------------------------------------------------------------------------


async def test_small_overflow_is_the_same_as_nanobot(ws, tmp_path):
    window = await three_big_turns(ws, tmp_path, overflow=200)
    loop, fake = loop_for(ws, tmp_path, "- SUM", "answer", window=window)
    await loop.process_direct(THIRD)
    roles = [m["role"] for m in fake.calls[0].messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
    assert loop.sessions.get_or_create("cli:direct").metadata["_last_summary"]["text"] == "- SUM"
    assert [e["content"] for e in loop.memory.read_unprocessed_history(0)] == ["- SUM"]


def archive_calls(fake):
    return [c.messages for c in fake.calls if str(c.messages[-1]["content"]).startswith(ARCHIVE_MARKER)]


@pytest.mark.parametrize(("overflow", "parts"), [(2000, 2), (8000, 3)])
async def test_large_overflow_is_summarized_in_parts(ws, tmp_path, overflow, parts):
    window = await three_big_turns(ws, tmp_path, overflow)
    history = loop_for(ws, tmp_path)[0].sessions.get_or_create("cli:direct").get_history()
    summaries = [f"- S{i}" for i in range(1, parts + 1)]
    loop, fake = loop_for(ws, tmp_path, *summaries, "answer", window=window)
    assert (await loop.process_direct(THIRD)).final_content == "answer"

    calls = archive_calls(fake)
    assert len(calls) == parts
    covered = []
    for request in calls:
        assert request[0]["role"] == "system" and request[1]["role"] == "user"  # each part starts at a user message
        assert estimate_prompt_tokens(request, [])[0] <= window - 100  # and fits one request
        covered += request[1:-1]
    # Oldest part first; together they are the whole history, each message once.
    assert [m["content"] for m in covered] == [m["content"] for m in history]

    assert [e["content"] for e in loop.memory.read_unprocessed_history(0)] == summaries
    assert loop.sessions.get_or_create("cli:direct").metadata["_last_summary"]["text"] == "\n\n".join(summaries)


@pytest.mark.parametrize("failing", [0, 1])
async def test_model_failure_gives_a_raw_dump_that_keeps_the_end(ws, tmp_path, failing):
    window = await three_big_turns(ws, tmp_path, overflow=2000)
    failed = LLMResponse(content="Error calling LLM: boom", finish_reason="error")
    script = ["- S1", "- S2"]
    script[failing] = failed
    loop, fake = loop_for(ws, tmp_path, *script[: failing + 1], "answer", window=window)
    await loop.process_direct(THIRD)
    text = loop.sessions.get_or_create("cli:direct").metadata["_last_summary"]["text"]
    assert text.startswith("[RAW] 6 messages\n" + TRUNCATED_PREFIX)
    assert text.endswith(BIG[-20:]) and "USER: first" not in text
    # Nothing is journaled before every part succeeded: the journal has each message once, no summary.
    journal = [e["content"] for e in loop.memory.read_unprocessed_history(0)]
    assert all(e.startswith("[RAW] 6 messages") for e in journal)
    joined = "".join(journal)
    assert joined.count("USER: first") == 1 and joined.count("USER: middle") == 1


async def test_nanobot_mode_still_keeps_the_start(ws, tmp_path):
    window = await three_big_turns(ws, tmp_path, overflow=2000)
    loop, fake = loop_for(ws, tmp_path, "answer", window=window, compaction="nanobot")
    await loop.process_direct(THIRD)
    assert len(fake.calls) == 1
    text = loop.sessions.get_or_create("cli:direct").metadata["_last_summary"]["text"]
    assert text.startswith("[RAW] 6 messages\n[?] USER: first") and text.endswith("... (truncated)")


# --- the archiver -------------------------------------------------------------------------------


def fixed_archiver(ws, store):
    return MemoryArchiver(store, ContextBuilder(ws).build_messages, lambda: [], compaction="fixed")


def turns(*texts):
    out = []
    for text in texts:
        out += [{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}]
    return out


async def test_long_summaries_share_the_checkpoint(ws):
    store = MemoryStore(ws)
    source = turns("A" * 2500, "B" * 2500)
    fake = FakeProvider(["- old " + "o" * 500 + " OLD-END", "- new " + "n" * 500 + " NEW-END"])
    runtime = LLMRuntime(provider=fake, model=None, max_tokens=400, temperature=0.1, context_window_tokens=10_000)
    text = await fixed_archiver(ws, store).archive(
        source, runtime=runtime, session_key="k", history=[{"role": "system", "content": "SYS"}, *source],
        request_tools=[], input_token_budget=5000,
    )
    assert len(fake.calls) == 2
    old, new = text.split("\n\n")
    # Both parts are cut to the same share, so the newest summary isn't lost.
    assert old.startswith("- old") and new.startswith("- new") and len(text.encode()) <= 400
    # The journal keeps the full summaries.
    assert [e["content"][-7:] for e in store.read_unprocessed_history(0)] == ["OLD-END", "NEW-END"]


async def test_temporary_turns_journal_nothing(ws):
    store = MemoryStore(ws)
    source = turns("A" * 2500, "B" * 2500)
    runtime = LLMRuntime(
        provider=FakeProvider(["- A", "- B"]), model=None, max_tokens=400, temperature=0.1,
        context_window_tokens=10_000,
    )
    text = await fixed_archiver(ws, store).archive(
        source, runtime=runtime, session_key="k", history=[{"role": "system", "content": "SYS"}, *source],
        request_tools=[], input_token_budget=5000, persist=False,
    )
    assert text == "- A\n\n- B" and store.read_unprocessed_history(0) == []


async def test_nothing_parts_are_left_out(ws):
    store = MemoryStore(ws)
    source = turns("A" * 2500, "B" * 2500)
    runtime = LLMRuntime(
        provider=FakeProvider(["(nothing)", "- B"]), model=None, max_tokens=400, temperature=0.1,
        context_window_tokens=10_000,
    )
    text = await fixed_archiver(ws, store).archive(
        source, runtime=runtime, session_key="k", history=[{"role": "system", "content": "SYS"}, *source],
        request_tools=[], input_token_budget=5000,
    )
    assert text == "- B" and [e["content"] for e in store.read_unprocessed_history(0)] == ["- B"]


async def test_a_turn_too_big_for_any_request_gives_a_raw_dump(ws):
    store = MemoryStore(ws)
    source = turns("small", "Z" * 9000)
    fake = FakeProvider([])
    runtime = LLMRuntime(provider=fake, model=None, max_tokens=400, temperature=0.1, context_window_tokens=10_000)
    text = await fixed_archiver(ws, store).archive(
        source, runtime=runtime, session_key="k", history=[{"role": "system", "content": "SYS"}, *source],
        request_tools=[], input_token_budget=5000,
    )
    assert fake.calls == [] and text.startswith("[RAW] 4 messages\n") and text.endswith("ASSISTANT: ok")


# --- /compact -----------------------------------------------------------------------------------


async def test_compact_session_summarizes_in_parts(ws, tmp_path):
    loop, _ = loop_for(ws, tmp_path, "a", "b", "c")
    for word in ("first ", "second ", "third "):
        await loop.process_direct(word + "z" * 4000)
    fake = FakeProvider(["- S1", "- S2", "- S3"])
    store = MemoryStore(ws)
    sessions = SessionManager(ws, sessions_root=tmp_path / "sessions")
    consolidator = Consolidator(store, sessions, ContextBuilder(ws).build_messages, lambda: [], compaction="fixed")
    runtime = LLMRuntime(provider=fake, model=None, max_tokens=100, temperature=0.1, context_window_tokens=11_200)
    assert await consolidator.compact_idle_session("cli:direct", runtime=runtime) == "- S1\n\n- S2\n\n- S3"
    firsts = [call.messages[1]["content"][:6] for call in fake.calls]
    assert firsts == ["first ", "second", "third "]
    assert [e["content"] for e in store.read_unprocessed_history(0)] == ["- S1", "- S2", "- S3"]


# --- helpers ------------------------------------------------------------------------------------


def test_keep_end_truncation():
    assert truncate_text_to_tokens_keep_end("short", 100) == "short"
    text = "start " + "word " * 500 + "THE END"
    cut = truncate_text_to_tokens_keep_end(text, 50)
    assert cut.startswith(TRUNCATED_PREFIX) and cut.endswith("THE END") and "start" not in cut


def test_keep_end_truncation_with_tiktoken(monkeypatch):
    from conftest import REAL_GET_TOKEN_ENCODING
    from nanoclaw.utils import tokens

    monkeypatch.setattr(tokens, "get_token_encoding", REAL_GET_TOKEN_ENCODING)
    enc = REAL_GET_TOKEN_ENCODING()
    if enc is None:
        pytest.skip("tiktoken vocabulary not available offline")
    text = "start " + "word " * 500 + "THE END"
    cut = truncate_text_to_tokens_keep_end(text, 50)
    assert cut.startswith(TRUNCATED_PREFIX) and cut.endswith("THE END") and "start" not in cut
    assert len(enc.encode_ordinary(cut)) <= 50


def test_raw_checkpoint_chars_keep_the_end(ws):
    store = MemoryStore(ws)
    messages = turns("FIRST " + "a" * 500, "b" * 500 + " LAST")
    cut = store._build_raw_checkpoint(messages, max_chars=200, keep_end=True)
    assert cut.startswith("[RAW] 4 messages\n... (earlier part truncated)\n") and cut.endswith("ASSISTANT: ok")
    assert "LAST" in cut and "FIRST" not in cut
    assert store._build_raw_checkpoint(messages, max_chars=200).startswith("[RAW] 4 messages\n[?] USER: FIRST")


def test_combine_keep_end_keeps_the_header_and_the_newest_part():
    raw = "[RAW] 9 messages\n" + "old " * 400 + "NEWEST"
    out = MemoryArchiver._combine_raw_checkpoint(raw, previous_summary=None, max_tokens=60, keep_end=True)
    assert out.startswith("[RAW] 9 messages\n" + TRUNCATED_PREFIX) and out.endswith("NEWEST")
    both = MemoryArchiver._combine_raw_checkpoint(raw, previous_summary="P " * 400, max_tokens=400, keep_end=True)
    assert both.startswith("[Previous archived context]\nP") and both.endswith("NEWEST")


# --- the A/B script -----------------------------------------------------------------------------


async def test_compare_script_dry_run(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_compaction.py"
    spec = importlib.util.spec_from_file_location("compare_compaction", path)
    script = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "compare_compaction", script)  # dataclasses look the module up
    spec.loader.exec_module(script)
    # Token counts are bytes in the tests, so the sizes are bytes too.
    argv = ["--dry-run", "--runs", "1", "--window", "9000", "--max-tokens", "500", "--filler-tokens", "800",
            "--out", str(tmp_path / "ab")]
    assert await script.main(argv) == 0
    report = (tmp_path / "ab" / "report.md").read_text(encoding="utf-8")
    assert "| | nanobot | fixed |" in report and "| Quiz: total recall |" in report
    for mode in ("nanobot", "fixed"):
        folder = tmp_path / "ab" / "run1" / mode
        assert (folder / "quiz.md").exists() and (folder / "checkpoints.md").exists()
        assert (folder / "workspace" / "memory" / "history.jsonl").exists()
    results = __import__("json").loads((tmp_path / "ab" / "results.json").read_text(encoding="utf-8"))
    assert [r["mode"] for r in results] == ["nanobot", "fixed"]
    assert all(r["compactions"] >= 1 for r in results)  # the scenario really compacts
