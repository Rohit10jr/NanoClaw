"""Runtime logs (loguru): off by default, and the warnings nanobot logs when enabled."""

import io
import sys

import httpx
import pytest
from loguru import logger

from conftest import REAL_GET_TOKEN_ENCODING
from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.memory import MemoryArchiver, MemoryStore
from nanoclaw.agent.runner import AgentRunner, AgentRunSpec
from nanoclaw.agent.subagent import SubagentManager
from nanoclaw.agent.tool_results import normalize_tool_result
from nanoclaw.agent.tools import ToolRegistry, is_tool_error
from nanoclaw.agent.tools.violations import classify_violation
from nanoclaw.agent.tools.web import WebFetchTool, WebSearchTool, _redact_url_for_log
from nanoclaw.cli.log_control import configure_console_logging, set_nanoclaw_logs
from nanoclaw.providers import FakeProvider, LLMResponse, ToolCallRequest
from nanoclaw.session import SessionManager
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.templates import sync_workspace_templates
from toy_tools import EchoTool

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


# --- agent runner and loop ----------------------------------------------------------------


def echo_registry():
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


def wants_tools(*calls):
    return LLMResponse(content=None, tool_calls=list(calls), finish_reason="tool_calls")


async def run_spec(provider, **kwargs):
    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "hi"}], tools=echo_registry(), provider=provider, **kwargs
    )
    return await AgentRunner().run(spec)


async def test_runner_logs_nothing_by_default():
    records, sink = _capture()
    try:
        result = await run_spec(FakeProvider(["", "", ""]), session_key="cli:direct")
    finally:
        logger.remove(sink)
    assert result.stop_reason == "empty_final_response"
    assert records == []


async def test_empty_response_retry_is_warned(logs):
    result = await run_spec(FakeProvider(["", "", "Recovered."]), session_key="cli:direct")
    assert result.final_content == "Recovered."
    assert messages(logs, "WARNING") == [
        "Empty response on turn 0 for cli:direct (1/2); retrying",
        "Empty response on turn 1 for cli:direct after 2 retries; attempting finalization",
    ]


async def test_malformed_tool_calls_are_warned(logs):
    nameless = ToolCallRequest(id="bad", name="", arguments={})
    result = await run_spec(FakeProvider([wants_tools(nameless), wants_tools(nameless), "Plain answer."]))
    assert result.final_content == "Plain answer."
    dropped = "Dropped 1 malformed tool call(s) with missing/non-string name from LLM response (finish_reason='tool_calls')"
    assert messages(logs, "WARNING") == [
        dropped,
        "Retrying LLM request after all 1 malformed tool call(s) were dropped",
        dropped,
        "Malformed tool calls persisted after retry; falling back to no-tools request",
    ]


async def test_max_iterations_is_warned(tmp_path, ws, logs):
    script = [wants_tools(ToolCallRequest(id=f"c{i}", name="echo", arguments={"text": "x"})) for i in range(2)]
    loop = AgentLoop(
        FakeProvider([*script, "Summary."]),
        ws,
        tools=echo_registry(),
        sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"),
        max_iterations=2,
    )
    result = await loop.process_direct("hello there")
    assert result.stop_reason == "max_iterations"
    # The progress hook logs each tool call, as nanobot's does (progress_hook.py:151).
    assert messages(logs, "INFO") == ["Processing message from cli:user: hello there"] + [
        'Tool call: echo({"text": "x"})'
    ] * 2
    assert messages(logs, "WARNING") == ["Max iterations (2) reached"]


async def test_dream_with_nothing_to_process_is_logged(tmp_path, ws, logs):
    loop = AgentLoop(FakeProvider([]), ws, sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"))
    await loop.run_dream()
    info = messages(logs, "INFO")
    assert info[0].startswith("Registered ") and "'spawn'" in info[0]
    assert info[1:] == ["Dream: nothing to process"]


def test_token_encoding_unavailable_is_warned(monkeypatch, logs):
    import tiktoken

    def broken(name):
        raise OSError("offline")

    monkeypatch.setattr(tiktoken, "get_encoding", broken)
    REAL_GET_TOKEN_ENCODING.cache_clear()
    try:
        assert REAL_GET_TOKEN_ENCODING() is None
    finally:
        REAL_GET_TOKEN_ENCODING.cache_clear()
    assert messages(logs, "WARNING") == [
        "Token encoding unavailable (offline); using UTF-8 byte estimates until restart."
    ]


def test_repeated_workspace_violation_is_warned(tmp_path, logs):
    target = tmp_path / "secret.txt"
    counts: dict[str, int] = {}
    outs = [
        classify_violation(
            raw_text="Error: path is outside the configured workspace",
            soft_payload="soft",
            tool_name="read_file",
            arguments={"path": str(target)},
            workspace_violation_counts=counts,
        )
        for _ in range(3)
    ]
    assert outs[:2] == ["soft", "soft"] and outs[2].startswith("Error: refusing repeated workspace-bypass")
    signature = f"violation:{target.resolve().as_posix()}".lower()
    assert messages(logs, "WARNING") == [
        f"Escalating repeated workspace bypass attempt {signature} (attempt 3)",
        "Tool read_file hit workspace boundary repeatedly; escalating hint",
    ]


# --- off by default, --logs, retry log lines -------------------------------------------------


def test_logs_are_off_in_a_fresh_interpreter():
    """Importing nanoclaw turns its logs off; nothing reaches a sink until enabled."""
    import subprocess

    code = (
        "import sys, tempfile; from pathlib import Path; from loguru import logger; "
        "from nanoclaw.agent.memory import MemoryStore; logger.remove(); "
        "logger.add(sys.stdout, format='{message}'); "
        "store = MemoryStore(Path(tempfile.mkdtemp())); "
        "store.append_history('x' * 70000); print('done')"  # oversized entry: a WARNING when logs are on
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "done"


def test_cli_logs_flag_turns_logs_on(tmp_path, capsys):
    from nanoclaw.__main__ import main

    try:
        assert main(["ask", "hi", "--logs", "-w", str(tmp_path / "ws")], provider=FakeProvider(["hello"])) == 0
        err = capsys.readouterr().err
        assert "Processing message from cli:user" in err
        assert main(["ask", "hi again", "-w", str(tmp_path / "ws")], provider=FakeProvider(["hello"])) == 0
        assert "Processing message" not in capsys.readouterr().err  # off again without --logs
    finally:
        set_nanoclaw_logs(False)
        logger.remove()


async def test_retry_steps_are_logged(logs, monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("nanoclaw.providers.base.asyncio.sleep", no_sleep)
    fake = FakeProvider([LLMResponse(content="HTTP 503 down", finish_reason="error", error_status_code=503)] * 4)
    fake.retries = True
    await fake.chat_with_retry(messages=[{"role": "user", "content": "x"}])
    warnings = messages(logs, "WARNING")
    assert warnings[0] == "LLM transient error (attempt 1/3), retrying in 1s: http 503 down"
    assert warnings[-1] == "LLM request failed after 4 attempts, giving up: http 503 down"
