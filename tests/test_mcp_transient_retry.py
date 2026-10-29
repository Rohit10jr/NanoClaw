"""Tests for MCP tool transient error retry.

Ported from nanobot tests/agent/test_mcp_transient_retry.py (c4dac64); the
resource and prompt wrappers are not ported.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mcp import types as mcp_types
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from nanoclaw.agent.tools.base import is_tool_error
from nanoclaw.agent.tools.mcp import (
    MCPToolWrapper,
    _is_session_terminated,
    _is_transient,
)

# ---------------------------------------------------------------------------
# _is_transient helper
# ---------------------------------------------------------------------------


class _FakeClosedResourceError(Exception):
    pass


_FakeClosedResourceError.__name__ = "ClosedResourceError"


class _FakeEndOfStreamError(Exception):
    pass


_FakeEndOfStreamError.__name__ = "EndOfStream"


def _session_terminated_error() -> McpError:
    return McpError(ErrorData(code=-32000, message="Session terminated"))


def _connection_closed_error() -> McpError:
    return McpError(ErrorData(code=-32000, message="Connection closed"))


def test_is_transient_recognizes_closed_resource():
    assert _is_transient(_FakeClosedResourceError("gone"))


def test_is_transient_recognizes_broken_pipe():
    assert _is_transient(BrokenPipeError("pipe"))


def test_is_transient_recognizes_connection_reset():
    assert _is_transient(ConnectionResetError("reset"))


def test_is_transient_recognizes_connection_refused():
    assert _is_transient(ConnectionRefusedError("refused"))


def test_is_transient_recognizes_end_of_stream():
    assert _is_transient(_FakeEndOfStreamError("eof"))


def test_is_transient_rejects_value_error():
    assert not _is_transient(ValueError("nope"))


def test_is_transient_rejects_runtime_error():
    assert not _is_transient(RuntimeError("nope"))


def test_is_transient_rejects_timeout():
    assert not _is_transient(TimeoutError("timeout"))


def test_is_session_terminated_recognizes_mcp_error():
    assert _is_session_terminated(_session_terminated_error())


def test_is_session_terminated_recognizes_connection_closed_mcp_error():
    assert _is_session_terminated(_connection_closed_error())


# ---------------------------------------------------------------------------
# MCPToolWrapper retry behaviour
# ---------------------------------------------------------------------------


def _make_tool_def(name="test_tool"):
    return SimpleNamespace(
        name=name,
        description="A test tool",
        inputSchema={"type": "object", "properties": {}},
    )


def _make_tool_result(text):
    """Build a mock tool result with proper MCP TextContent."""
    return SimpleNamespace(content=[mcp_types.TextContent(type="text", text=text)])


@pytest.mark.asyncio
async def test_tool_retries_on_transient_error():
    """Tool should retry once when a transient error occurs, then succeed."""
    session = AsyncMock()
    result = _make_tool_result("ok")
    exc = _FakeClosedResourceError("connection lost")
    session.call_tool = AsyncMock(side_effect=[exc, result])

    wrapper = MCPToolWrapper(session, "test_server", _make_tool_def(), tool_timeout=5)

    with patch("nanoclaw.agent.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
        output = await wrapper.execute(foo="bar")

    assert output == "ok"
    assert session.call_tool.call_count == 2


@pytest.mark.asyncio
async def test_tool_fails_after_retry_exhausted():
    """Tool should fail with retry message when both attempts hit transient errors."""
    session = AsyncMock()
    exc1 = _FakeClosedResourceError("still dead")
    exc2 = _FakeClosedResourceError("still dead again")
    session.call_tool = AsyncMock(side_effect=[exc1, exc2])

    wrapper = MCPToolWrapper(session, "test_server", _make_tool_def(), tool_timeout=5)

    with patch("nanoclaw.agent.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
        output = await wrapper.execute()

    assert "failed after retry" in output
    assert "ClosedResourceError" in output
    assert is_tool_error(output)
    assert session.call_tool.call_count == 2


@pytest.mark.asyncio
async def test_tool_no_retry_on_non_transient_error():
    """Tool should NOT retry on non-transient errors like ValueError."""
    session = AsyncMock()
    session.call_tool = AsyncMock(side_effect=ValueError("bad input"))

    wrapper = MCPToolWrapper(session, "test_server", _make_tool_def(), tool_timeout=5)
    output = await wrapper.execute()

    assert "ValueError" in output
    assert "retry" not in output
    assert session.call_tool.call_count == 1


@pytest.mark.asyncio
async def test_tool_no_retry_on_timeout():
    """Timeouts should not trigger retry (they have their own handling)."""
    session = AsyncMock()
    session.call_tool = AsyncMock(side_effect=asyncio.TimeoutError())

    wrapper = MCPToolWrapper(session, "test_server", _make_tool_def(), tool_timeout=5)
    output = await wrapper.execute()

    assert "timed out" in output
    assert session.call_tool.call_count == 1


@pytest.mark.asyncio
async def test_tool_success_on_first_try_no_retry():
    """Normal success path — no retry logic involved."""
    session = AsyncMock()
    result = _make_tool_result("hello")
    session.call_tool = AsyncMock(return_value=result)

    wrapper = MCPToolWrapper(session, "test_server", _make_tool_def(), tool_timeout=5)
    output = await wrapper.execute()

    assert output == "hello"
    assert session.call_tool.call_count == 1


@pytest.mark.asyncio
async def test_tool_does_not_retry_on_cancelled_error():
    """`asyncio.CancelledError` must short-circuit the retry loop.

    Regression guard: the retry branch lives under ``except Exception``,
    but ``CancelledError`` inherits from ``BaseException``, not
    ``Exception``, so it naturally bypasses the retry branch today.  If a
    future refactor ever widens the retry branch to ``BaseException`` (or
    re-orders the handlers), ``/stop`` would start retrying instead of
    cancelling — this test pins that invariant.
    """
    session = AsyncMock()
    session.call_tool = AsyncMock(side_effect=asyncio.CancelledError())

    wrapper = MCPToolWrapper(session, "test_server", _make_tool_def(), tool_timeout=5)

    with patch("nanoclaw.agent.tools.mcp.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        output = await wrapper.execute()

    assert "cancelled" in output
    assert session.call_tool.call_count == 1
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_tool_retry_on_connection_reset():
    """ConnectionResetError (a stdlib exception) should also trigger retry."""
    session = AsyncMock()
    result = _make_tool_result("recovered")
    session.call_tool = AsyncMock(
        side_effect=[ConnectionResetError("reset by peer"), result]
    )

    wrapper = MCPToolWrapper(session, "test_server", _make_tool_def(), tool_timeout=5)

    with patch("nanoclaw.agent.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
        output = await wrapper.execute()

    assert output == "recovered"
    assert session.call_tool.call_count == 2


@pytest.mark.asyncio
async def test_tool_retry_on_end_of_stream():
    """EndOfStream (anyio) should trigger retry."""
    session = AsyncMock()
    result = _make_tool_result("back")
    session.call_tool = AsyncMock(side_effect=[_FakeEndOfStreamError("eof"), result])

    wrapper = MCPToolWrapper(session, "test_server", _make_tool_def(), tool_timeout=5)

    with patch("nanoclaw.agent.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
        output = await wrapper.execute()

    assert output == "back"
    assert session.call_tool.call_count == 2


@pytest.mark.asyncio
async def test_tool_reconnects_on_transient_failure():
    """Tool should reconnect when a stale session reports a transient stream failure."""
    old_session = AsyncMock()
    old_session.call_tool = AsyncMock(side_effect=_FakeClosedResourceError("closed"))
    new_session = AsyncMock()
    new_session.call_tool = AsyncMock(return_value=_make_tool_result("fresh"))

    wrapper = MCPToolWrapper(old_session, "test_server", _make_tool_def(), tool_timeout=5)
    replacement = MCPToolWrapper(new_session, "test_server", _make_tool_def(), tool_timeout=5)

    async def reconnect(server_name: str, tool_name: str, stale_tool):
        assert server_name == "test_server"
        assert tool_name == "mcp_test_server_test_tool"
        assert stale_tool is wrapper
        return replacement

    wrapper.set_reconnect_handler(reconnect)

    with patch("nanoclaw.agent.tools.mcp.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        output = await wrapper.execute(foo="bar")

    assert output == "fresh"
    assert old_session.call_tool.call_count == 1
    assert new_session.call_tool.call_count == 1
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_tool_reconnects_on_session_terminated():
    """NanoClaw: the tool counterpart of nanobot's resource/prompt
    session-terminated tests: one reconnect, then the call on the new session."""
    old_session = AsyncMock()
    old_session.call_tool = AsyncMock(side_effect=_session_terminated_error())
    new_session = AsyncMock()
    new_session.call_tool = AsyncMock(return_value=_make_tool_result("fresh"))

    wrapper = MCPToolWrapper(old_session, "test_server", _make_tool_def(), tool_timeout=5)
    replacement = MCPToolWrapper(new_session, "test_server", _make_tool_def(), tool_timeout=5)
    calls = 0

    async def reconnect(server_name: str, tool_name: str, stale_tool):
        nonlocal calls
        calls += 1
        return replacement

    wrapper.set_reconnect_handler(reconnect)

    output = await wrapper.execute()

    assert output == "fresh"
    assert calls == 1
    assert old_session.call_tool.call_count == 1


@pytest.mark.asyncio
async def test_tool_reports_failure_when_session_stays_terminated():
    """A second session-terminated error after one refresh is not retried again."""
    session = AsyncMock()
    session.call_tool = AsyncMock(side_effect=_session_terminated_error())
    wrapper = MCPToolWrapper(session, "test_server", _make_tool_def(), tool_timeout=5)

    async def reconnect(server_name: str, tool_name: str, stale_tool):
        return wrapper  # same (dead) session

    wrapper.set_reconnect_handler(reconnect)

    output = await wrapper.execute()

    assert output == "(MCP tool call failed: McpError)"
    assert is_tool_error(output)
    assert session.call_tool.call_count == 2
