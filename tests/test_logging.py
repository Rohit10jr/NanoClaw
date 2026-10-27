"""Runtime logs (loguru): off by default, and the warnings nanobot logs when enabled."""

import io
import sys

import httpx
import pytest
from loguru import logger

from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.memory import MemoryArchiver, MemoryStore
from nanoclaw.agent.subagent import SubagentManager
from nanoclaw.agent.tool_results import normalize_tool_result
from nanoclaw.agent.tools import is_tool_error
from nanoclaw.agent.tools.web import WebFetchTool, WebSearchTool, _redact_url_for_log
from nanoclaw.cli.log_control import configure_console_logging, set_nanoclaw_logs
from nanoclaw.providers import FakeProvider, LLMResponse
from nanoclaw.session import SessionManager
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.templates import sync_workspace_templates

SOURCE = [{"role": "user", "content": "I like tea"}, {"role": "assistant", "content": "Noted."}]
HISTORY = [{"role": "system", "content": "SYS"}, *SOURCE]


def _capture():
    records: list[tuple[str, str]] = []
    sink = logger.add(lambda m: records.append((m.record["level"].name, m.record["message"])), level="DEBUG")
    return records, sink


@pytest.fixture
def logs():
    """Turn NanoClaw's logs on and collect ``(level, message)`` pairs; off again afterwards."""
    records, sink = _capture()
    set_nanoclaw_logs(True)
    try:
        yield records
    finally:
        set_nanoclaw_logs(False)
        logger.remove(sink)


def messages(records, level=None):
    return [msg for lvl, msg in records if level is None or lvl == level]


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    sync_workspace_templates(path)
    return path


def corrupt_session(tmp_path):
    mgr = SessionManager(tmp_path / "ws", sessions_root=tmp_path / "sessions")
    session = mgr.get_or_create("cli:direct")
    session.add_message("user", "hello")
    mgr.save(session)
    with open(mgr.get_session_path("cli:direct"), "a", encoding="utf-8") as handle:
        handle.write("{not json\n[1, 2]\n")
    return SessionManager(tmp_path / "ws", sessions_root=tmp_path / "sessions")


def runtime(provider, max_tokens=1000):
    return LLMRuntime(provider=provider, model=None, max_tokens=max_tokens, temperature=0.1, context_window_tokens=128_000)


# --- on/off -------------------------------------------------------------------------------


def test_logs_are_off_by_default(tmp_path):
    records, sink = _capture()
    try:
        session = corrupt_session(tmp_path).get_or_create("cli:direct")
    finally:
        logger.remove(sink)
    assert [m["content"] for m in session.messages] == ["hello"]
    assert records == []


def test_set_nanoclaw_logs_turns_package_logs_on_and_off(tmp_path):
    records, sink = _capture()
    try:
        set_nanoclaw_logs(True)
        corrupt_session(tmp_path / "a").get_or_create("cli:direct")
        set_nanoclaw_logs(False)
        corrupt_session(tmp_path / "b").get_or_create("cli:direct")
    finally:
        set_nanoclaw_logs(False)
        logger.remove(sink)
    assert messages(records) == ["Skipped 2 corrupt lines in session cli:direct"]


def test_console_sink_shows_info_and_up(tmp_path, ws):
    stream = io.StringIO()
    handler = configure_console_logging(stream)
    set_nanoclaw_logs(True)
    try:
        MemoryStore(ws).append_history("<think>only reasoning</think>")  # a DEBUG record
        corrupt_session(tmp_path).get_or_create("cli:direct")  # a WARNING record
    finally:
        set_nanoclaw_logs(False)
        logger.remove(handler)
        logger.add(sys.stderr)  # loguru's default handler, removed by configure_console_logging
    out = stream.getvalue()
    assert "| WARNING | nanoclaw.session.manager | Skipped 2 corrupt lines in session cli:direct" in out
    assert "stripped to empty" not in out


# --- sessions and memory ------------------------------------------------------------------


def test_corrupt_session_lines_are_warned(tmp_path, logs):
    corrupt_session(tmp_path).get_or_create("cli:direct")
    assert ("WARNING", "Skipped 2 corrupt lines in session cli:direct") in logs


async def test_raw_checkpoint_fallback_is_warned(ws, logs):
    store = MemoryStore(ws)
    archiver = MemoryArchiver(store, ContextBuilder(ws).build_messages, lambda: [])
    reply = LLMResponse(content="Error calling LLM: boom", finish_reason="error")
    summary = await archiver.archive(
        SOURCE, runtime=runtime(FakeProvider([reply])), session_key="k", history=HISTORY, request_tools=[]
    )
    assert summary.startswith("[RAW] 2 messages")
    assert messages(logs, "WARNING") == [
        "Memory archive provider did not complete (error); using raw checkpoint",
        "Memory consolidation degraded: raw-archived 2 messages in 1 entries",
    ]


async def test_archive_that_does_not_fit_is_logged_at_debug(ws, logs):
    store = MemoryStore(ws)
    archiver = MemoryArchiver(store, ContextBuilder(ws).build_messages, lambda: [])
    await archiver.archive(
        SOURCE, runtime=runtime(FakeProvider([])), session_key="k", history=HISTORY, request_tools=[], input_token_budget=10
    )
    debug = messages(logs, "DEBUG")
    assert len(debug) == 1 and debug[0].startswith("Memory archive input does not fit for k: ")
    assert "/10 via " in debug[0] and debug[0].endswith("; using raw checkpoint")


def test_corruption_is_logged_exactly_once_per_store(ws, logs):
    """nanobot tests/agent/test_cursor_recovery.py: the first invalid cursor emits
    one warning; later reads on the same store stay quiet."""
    store = MemoryStore(ws)
    store.history_file.write_text(
        '{"cursor": "bad1", "timestamp": "2026-04-01 10:00", "content": "x"}\n'
        '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "y"}\n',
        encoding="utf-8",
    )
    store._cursor_file.unlink(missing_ok=True)
    store.read_unprocessed_history(since_cursor=0)
    store.read_unprocessed_history(since_cursor=0)
    store.append_history("another")
    warnings = [m for m in messages(logs, "WARNING") if "invalid cursor" in m]
    assert warnings == [
        "history.jsonl contains an invalid cursor ('bad1'); dropping it. "
        "Usually caused by an external writer; further occurrences suppressed."
    ]


def test_malformed_entry_is_logged_once(ws, logs):
    store = MemoryStore(ws)
    store.history_file.write_text('{"cursor": 1, "timestamp": 5, "content": "x"}\n', encoding="utf-8")
    assert store.read_unprocessed_history(0) == []
    assert store.read_unprocessed_history(0) == []
    assert [m for m in messages(logs) if "malformed entry" in m] == [
        "history.jsonl contains a malformed entry at cursor 1; dropping it. "
        "Usually caused by an external writer; further occurrences suppressed."
    ]


def test_oversized_history_entry_is_warned_once(ws, logs):
    store = MemoryStore(ws)
    store.append_history("x" * 50, max_chars=10)
    store.append_history("y" * 50, max_chars=10)
    assert [m for m in messages(logs) if "exceeds" in m] == [
        "history entry exceeds 10 chars (50); truncating. "
        "Usually means a caller forgot its own cap; further occurrences suppressed."
    ]


# --- subagents ----------------------------------------------------------------------------


async def test_subagent_start_and_failure_are_logged(ws, logs, monkeypatch):
    manager = SubagentManager(ws)

    async def broken_run(spec):
        raise RuntimeError("child crashed")

    monkeypatch.setattr(manager.runner, "run", broken_run)
    result = await manager.run_inline("look around", label="scout", runtime=runtime(FakeProvider([])))
    assert is_tool_error(result)
    info = messages(logs, "INFO")
    assert len(info) == 2
    assert info[0].startswith("Started subagent [") and info[0].endswith("]: scout")
    assert info[1].endswith("] starting task: scout")
    errors = messages(logs, "ERROR")
    assert len(errors) == 1 and errors[0].startswith("Subagent [") and errors[0].endswith("] failed")


# --- tools --------------------------------------------------------------------------------


async def test_web_fetch_error_is_warned_without_url_path(fake_dns, logs):
    def handler(request):
        raise httpx.ConnectError("refused")

    url = "https://user:secret@example.test/private?token=abc"
    out = await WebFetchTool(transport=httpx.MockTransport(handler)).execute(url=url)
    assert is_tool_error(out)
    assert messages(logs, "WARNING") == ["WebFetch error for https://example.test (ConnectError)"]


async def test_web_search_failure_is_warned(logs):
    def broken(query, n):
        raise RuntimeError("network down")

    out = await WebSearchTool(search_fn=broken).execute(query="x")
    assert is_tool_error(out)
    assert messages(logs, "WARNING") == ["DuckDuckGo search failed: network down"]


def test_log_label_excludes_every_credential_bearing_component():
    # nanobot tests/tools/test_web_fetch_jina_privacy.py
    url = "https://user:secret@example.com:8443/private/webhook-token?token=abc#secret"
    assert _redact_url_for_log(url) == "https://example.com:8443"


def test_log_label_preserves_ipv6_origin_without_credentials():
    url = "https://user:secret@[2001:db8::1]:8443/private?token=abc"
    assert _redact_url_for_log(url) == "https://[2001:db8::1]:8443"


def test_tool_result_persist_failure_falls_back_to_truncation(tmp_path, logs):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".nanoclaw").write_text("not a folder", encoding="utf-8")
    out = normalize_tool_result("exec", "c1", "x" * 500, workspace=workspace, max_chars=100, bucket="cli:direct")
    assert len(out) <= 200 and out.startswith("x")
    assert messages(logs, "ERROR") == ["Tool result persist failed for c1 in cli:direct; using raw result"]
