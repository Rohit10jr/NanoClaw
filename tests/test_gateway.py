"""``nanoclaw gateway`` (layer 21): teardown, signals, ``/health``, the run, the CLI.

The teardown, readiness, signal and health-handler tests are adapted from
nanobot tests/cli/test_gateway_runtime.py and tests/cli/test_commands.py
(c4dac64), without channels. The subprocess tests start the real command.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from nanoclaw import __version__
from nanoclaw.__main__ import main
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.tools.mcp import MCPProvider
from nanoclaw.cli import gateway as gw
from nanoclaw.cli.gateway import (
    Gateway,
    GatewayHealthIdentity,
    _close_gateway_runtime,
    _gateway_readiness_payload,
    _install_gateway_shutdown_handlers,
    health_payload,
    make_health_handler,
)
from nanoclaw.config import Config
from nanoclaw.providers import FakeProvider
from nanoclaw.session import SessionManager

WINDOWS = sys.platform == "win32"
REPO = Path(__file__).resolve().parents[1]


# -- teardown (nanobot test_gateway_runtime.py) ------------------------------------------


class _FakeAgent:
    def __init__(self, events: list[str] | None = None) -> None:
        self.close_calls = 0
        self.events = events if events is not None else []
        self.hang_on_close = False
        self.raise_on_close = False
        self.background: asyncio.Task[None] | None = None

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.hang_on_close:
            await asyncio.sleep(3600)
        if self.raise_on_close:
            raise RuntimeError("cleanup exploded")
        if self.background is not None:
            await self.background
        self.events.append("aclose")


class _FakeMCPProvider:
    def __init__(self, events: list[str] | None = None) -> None:
        self.close_calls = 0
        self.events = events if events is not None else []

    async def aclose(self) -> None:
        self.close_calls += 1
        self.events.append("mcp_closed")


async def _cancellable_task(events: list[str]) -> None:
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        events.append("cancelled")
        raise


async def _stubborn_task(events: list[str]) -> None:
    """Swallows cancellation and keeps running."""
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        events.append("swallowed")
        await asyncio.sleep(3600)


async def test_runtime_tasks_cancelled_before_resources_closed() -> None:
    events: list[str] = []
    agent, provider = _FakeAgent(events), _FakeMCPProvider(events)
    task = asyncio.create_task(_cancellable_task(events))
    await asyncio.sleep(0)
    await _close_gateway_runtime(agent, provider, [task], None)
    assert events == ["cancelled", "aclose", "mcp_closed"]
    assert agent.close_calls == 1
    assert task.cancelled()


async def test_pending_background_work_is_drained_before_close_returns() -> None:
    agent, provider = _FakeAgent(), _FakeMCPProvider()
    done = {"done": False}

    async def background_work() -> None:
        await asyncio.sleep(0.01)
        done["done"] = True

    agent.background = asyncio.create_task(background_work())
    await _close_gateway_runtime(agent, provider, [], None)
    assert done["done"] is True


async def test_stubborn_task_does_not_block_past_wait_timeout() -> None:
    agent, provider = _FakeAgent(), _FakeMCPProvider()
    events: list[str] = []
    task = asyncio.create_task(_stubborn_task(events))
    await asyncio.sleep(0)
    runtime_tasks = asyncio.gather(task)
    start = time.monotonic()
    await _close_gateway_runtime(agent, provider, [task], runtime_tasks, task_wait_timeout=0.05)
    elapsed = time.monotonic() - start
    for _ in range(10):
        await asyncio.sleep(0)
    assert "swallowed" in events
    assert task.done()
    assert runtime_tasks.done()
    assert agent.close_calls == 1 and provider.close_calls == 1
    assert elapsed < 1.0


async def test_hanging_close_is_bounded_and_does_not_raise() -> None:
    agent, provider = _FakeAgent(), _FakeMCPProvider()
    agent.hang_on_close = True
    start = time.monotonic()
    await _close_gateway_runtime(agent, provider, [], None, close_timeout=0.05)
    assert agent.close_calls == 1 and provider.close_calls == 1
    assert time.monotonic() - start < 1.0


async def test_failing_close_is_logged_but_shutdown_proceeds() -> None:
    agent, provider = _FakeAgent(), _FakeMCPProvider()
    agent.raise_on_close = True
    await _close_gateway_runtime(agent, provider, [], None)
    assert agent.close_calls == 1 and provider.close_calls == 1


async def test_duplicate_cleanup_is_idempotent() -> None:
    agent, provider = _FakeAgent(), _FakeMCPProvider()
    task = asyncio.create_task(_cancellable_task([]))
    await _close_gateway_runtime(agent, provider, [task], None)
    await _close_gateway_runtime(agent, provider, [task], None)
    assert agent.close_calls == 2 and provider.close_calls == 2
    assert task.cancelled()


async def test_finished_runtime_tasks_gather_is_retrieved() -> None:
    agent, provider = _FakeAgent(), _FakeMCPProvider()
    finished = asyncio.get_running_loop().create_future()
    finished.set_result(None)
    runtime_tasks = asyncio.gather(finished)
    await asyncio.sleep(0)
    await _close_gateway_runtime(agent, provider, [], runtime_tasks)
    assert runtime_tasks.done()
    assert agent.close_calls == 1 and provider.close_calls == 1


async def test_cancelled_runtime_tasks_gather_does_not_raise() -> None:
    agent, provider = _FakeAgent(), _FakeMCPProvider()
    runtime_tasks = asyncio.gather(asyncio.sleep(3600))
    runtime_tasks.cancel()
    await _close_gateway_runtime(agent, provider, [], runtime_tasks)
    with suppress(asyncio.CancelledError):
        await runtime_tasks
    assert runtime_tasks.done()
    assert agent.close_calls == 1 and provider.close_calls == 1


# -- readiness ----------------------------------------------------------------------------


def test_gateway_readiness_is_degraded_when_required_websocket_is_unavailable() -> None:
    channels = type(
        "Channels",
        (),
        {
            "enabled_channels": ["websocket"],
            "get_status": lambda self: {"websocket": {"enabled": True, "running": False, "state": "starting"}},
        },
    )()
    ready, payload = _gateway_readiness_payload(channels)
    assert ready is False
    assert payload == {"status": "degraded", "process": "alive", "ready": False, "websocket": "starting"}


def test_gateway_readiness_with_a_running_websocket() -> None:
    channels = type("C", (), {"get_status": lambda self: {"websocket": {"running": True, "state": "running"}}})()
    assert _gateway_readiness_payload(channels) == (
        True,
        {"status": "ok", "process": "alive", "ready": True, "websocket": "running"},
    )


def test_gateway_readiness_without_channels_is_ok() -> None:
    """NanoClaw has no channel manager yet: nanobot's no-WebSocket answer."""
    assert _gateway_readiness_payload(None) == (
        True,
        {"status": "ok", "process": "alive", "ready": True, "websocket": "disabled"},
    )


def test_health_payload_adds_identity_and_maintenance(tmp_path) -> None:
    identity = GatewayHealthIdentity.create(tmp_path / "config.json", tmp_path / "ws")
    ready, payload = health_payload(None, identity)
    assert ready is True
    assert payload["maintenance"] is False
    runtime = payload["runtime"]
    assert runtime["version"] == __version__
    assert len(runtime["instance_id"]) == 64 and len(runtime["run_id"]) == 32
    # Same config and workspace, same instance; every run has its own id.
    again = GatewayHealthIdentity.create(tmp_path / "config.json", tmp_path / "ws")
    assert again.instance_id == identity.instance_id and again.run_id != identity.run_id


# -- signals (nanobot test_commands.py:147-237) ------------------------------------------------


def test_gateway_signal_handler_first_signal_stops_and_second_forces() -> None:
    class _FakeLoop:
        def __init__(self) -> None:
            self.handlers: dict[int, tuple[object, tuple[object, ...]]] = {}
            self.removed: list[int] = []

        def add_signal_handler(self, signum, callback, *args) -> None:
            self.handlers[int(signum)] = (callback, args)

        def remove_signal_handler(self, signum) -> bool:
            self.removed.append(int(signum))
            self.handlers.pop(int(signum), None)
            return True

    async def _run() -> None:
        loop = _FakeLoop()
        shutdown_event = asyncio.Event()
        task = asyncio.create_task(asyncio.Event().wait())
        output: list[str] = []
        tasks: list[asyncio.Task[object]] = []
        restore = _install_gateway_shutdown_handlers(loop, shutdown_event, tasks, output.append)
        tasks.append(task)
        try:
            callback, args = loop.handlers[int(signal.SIGINT)]
            callback(*args)
            assert shutdown_event.is_set()
            assert output == ["\nShutting down... Press Ctrl+C again to force."]
            assert not task.done()
            callback(*args)
            await asyncio.sleep(0)
            assert task.cancelled()
        finally:
            restore()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        assert int(signal.SIGINT) in loop.removed
        assert int(signal.SIGTERM) in loop.removed

    asyncio.run(_run())


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGBREAK"])
async def test_gateway_signal_fallback_requests_shutdown_then_forces_and_restores(monkeypatch, signal_name) -> None:
    sigbreak = getattr(signal, "SIGBREAK", 21)
    monkeypatch.setattr(signal, "SIGBREAK", sigbreak, raising=False)
    previous = object()
    handlers = {}
    monkeypatch.setattr(signal, "getsignal", lambda _signum: previous)
    monkeypatch.setattr(signal, "signal", lambda signum, handler: handlers.update({signum: handler}))
    loop = MagicMock()
    loop.add_signal_handler.side_effect = NotImplementedError
    shutdown_event = asyncio.Event()
    task = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    tasks: list[asyncio.Task[object]] = []
    output: list[str] = []
    restore = _install_gateway_shutdown_handlers(loop, shutdown_event, tasks, output.append)
    tasks.append(task)
    signum = getattr(signal, signal_name)
    try:
        handlers[signum](signum, None)
        assert shutdown_event.is_set()
        assert output == ["\nShutting down... Press Ctrl+C again to force."]
        assert not task.done()
        handlers[signum](signum, None)
        await asyncio.sleep(0)
        assert task.cancelled()
    finally:
        restore()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert handlers == dict.fromkeys((signal.SIGINT, signal.SIGTERM, sigbreak), previous)


# -- the health handler (nanobot test_commands.py:3893) -----------------------------------------


class _FakeReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def read(self, _size: int) -> bytes:
        return self.payload


class _FakeWriter:
    def __init__(self) -> None:
        self.output = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.output += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _ok_payload():
    return True, {"status": "ok"}


async def _call(handler, path: str) -> tuple[str, _FakeWriter]:
    writer = _FakeWriter()
    await handler(_FakeReader(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()), writer)
    return writer.output.decode(), writer


async def test_health_handler_serves_expected_responses() -> None:
    handler = make_health_handler(_ok_payload)
    root, root_writer = await _call(handler, "/")
    assert root_writer.closed is True
    assert "HTTP/1.0 404 Not Found" in root and "Connection: close" in root
    assert root.endswith("\r\n\r\nNot Found")
    health, writer = await _call(handler, "/health")
    assert writer.closed is True
    assert "HTTP/1.0 200 OK" in health and "Content-Type: application/json" in health
    assert json.loads(health.split("\r\n\r\n", 1)[1]) == {"status": "ok"}
    missing, _ = await _call(handler, "/missing")
    assert missing.endswith("\r\n\r\nNot Found")
    post = _FakeWriter()
    await handler(_FakeReader(b"POST /health HTTP/1.1\r\n\r\n"), post)
    assert b"404 Not Found" in post.output


async def test_health_handler_not_ready_is_503() -> None:
    handler = make_health_handler(lambda: (False, {"status": "degraded"}))
    response, _ = await _call(handler, "/health")
    assert "HTTP/1.0 503 Service Unavailable" in response


async def test_health_handler_connection_limit() -> None:
    limit = gw.GATEWAY_HEALTH_MAX_CONNECTIONS
    assert limit == 64
    handler = make_health_handler(_ok_payload)
    release, all_started = asyncio.Event(), asyncio.Event()
    started = 0

    class _BlockingReader:
        async def read(self, _size: int) -> bytes:
            nonlocal started
            started += 1
            if started == limit:
                all_started.set()
            await release.wait()
            return b"GET /health HTTP/1.1\r\n\r\n"

    writers = [_FakeWriter() for _ in range(limit)]
    tasks = [asyncio.create_task(handler(_BlockingReader(), w)) for w in writers]
    await asyncio.wait_for(all_started.wait(), timeout=1)
    overflow = _FakeWriter()
    await handler(_FakeReader(b"GET /health HTTP/1.1\r\n\r\n"), overflow)
    assert overflow.closed is True and overflow.output == b""
    release.set()
    await asyncio.gather(*tasks)
    assert all(b"200 OK" in w.output for w in writers)


async def test_health_handler_read_timeout_closes_silently() -> None:
    class _NeverRespondingReader:
        async def read(self, _size: int) -> bytes:
            await asyncio.Event().wait()

    handler = make_health_handler(_ok_payload, read_timeout=0.01)
    writer = _FakeWriter()
    await handler(_NeverRespondingReader(), writer)
    assert writer.closed is True and writer.output == b""


@pytest.mark.parametrize(
    ("host", "display_url", "warns"),
    [("127.0.0.1", "http://127.0.0.1:18791/health", False), ("0.0.0.0", "http://127.0.0.1:18791/health", True)],
)
def test_health_endpoint_line_and_public_bind_warning(tmp_path, host, display_url, warns) -> None:
    config = Config()
    config.gateway.host = host
    console = Console(record=True, width=300)
    agent = MagicMock(workspace=tmp_path)
    gateway = Gateway(config, agent, MagicMock(), port=18791, console=console)
    gateway._print_health_endpoint()
    text = console.export_text()
    assert f"Health endpoint: {display_url}" in text
    assert ("unauthenticated health endpoint" in text) is warns
    assert ("listening on 0.0.0.0" in text) is warns


# -- the run ------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_agent(tmp_path, *script, ttl: int = 0) -> AgentLoop:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return AgentLoop(
        FakeProvider(list(script)),
        ws,
        sessions=SessionManager(ws, sessions_root=tmp_path / "sessions"),
        session_ttl_minutes=ttl,
    )


class _TrackingMCP(MCPProvider):
    def __init__(self) -> None:
        super().__init__({}, MagicMock())
        self.connected = 0
        self.closed = 0

    async def connect(self) -> None:
        self.connected += 1

    async def aclose(self) -> None:
        self.closed += 1


async def _until(predicate, timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def test_gateway_runs_until_shutdown_then_closes_everything(tmp_path) -> None:
    agent = make_agent(tmp_path)
    mcp = _TrackingMCP()
    gateway = Gateway(Config(), agent, mcp, port=free_port(), console=Console(record=True), health_server_enabled=False)
    run = asyncio.create_task(gateway.run())
    await _until(lambda: agent._running)
    assert mcp.connected == 1  # MCP connects inside the agent task
    gateway.shutdown_event.set()
    await asyncio.wait_for(run, 10)
    assert agent._running is False
    assert mcp.closed >= 1
    assert all(t.done() for t in gateway.tasks)
    assert gateway.crashed is False


async def test_gateway_serves_health_over_http(tmp_path) -> None:
    port = free_port()
    agent = make_agent(tmp_path)
    console = Console(record=True, width=200)
    gateway = Gateway(Config(), agent, _TrackingMCP(), port=port, console=console)
    run = asyncio.create_task(gateway.run())
    try:
        await _until(lambda: "Health endpoint" in console.export_text(clear=False))

        def fetch(path: str):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
                    return r.status, r.read().decode()
            except urllib.error.HTTPError as exc:
                return exc.code, exc.read().decode()

        status, body = await asyncio.to_thread(fetch, "/health")
        assert status == 200
        payload = json.loads(body)
        runtime = payload.pop("runtime")
        assert payload == {"status": "ok", "process": "alive", "ready": True, "websocket": "disabled", "maintenance": False}
        assert runtime == {
            "instance_id": gateway.health_identity.instance_id,
            "run_id": gateway.health_identity.run_id,
            "version": __version__,
        }
        assert await asyncio.to_thread(fetch, "/") == (404, "Not Found")
    finally:
        gateway.shutdown_event.set()
        await asyncio.wait_for(run, 10)


async def test_gateway_crashing_task_ends_the_run_and_is_reported(tmp_path) -> None:
    agent = make_agent(tmp_path)
    console = Console(record=True, width=200)
    gateway = Gateway(Config(), agent, _TrackingMCP(), console=console, health_server_enabled=False)

    async def boom() -> None:
        raise RuntimeError("task exploded")

    gateway.extra_tasks.append(("boom", boom))
    await asyncio.wait_for(gateway.run(), 10)
    assert gateway.crashed is True
    assert "Gateway crashed unexpectedly" in console.export_text()
    assert agent._running is False


async def test_gateway_compacts_idle_sessions(tmp_path) -> None:
    """The gateway's agent loop runs idle compaction on its idle tick."""
    agent = make_agent(tmp_path, "Gateway idle summary.", ttl=15)
    session = agent.sessions.get_or_create("cli:direct")
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    session.updated_at = datetime.now() - timedelta(minutes=30)
    agent.sessions.save(session)
    gateway = Gateway(Config(), agent, _TrackingMCP(), console=Console(record=True), health_server_enabled=False)
    run = asyncio.create_task(gateway.run())
    try:
        await _until(lambda: "_last_summary" in agent.sessions.get_or_create("cli:direct").metadata)
    finally:
        gateway.shutdown_event.set()
        await asyncio.wait_for(run, 10)
    assert agent.sessions.get_or_create("cli:direct").metadata["_last_summary"]["text"] == "Gateway idle summary."


def test_banner_lines(tmp_path) -> None:
    console = Console(record=True, width=200)
    config = Config()
    gateway = Gateway(config, MagicMock(workspace=tmp_path), MagicMock(), console=console)
    gateway.print_banner()
    gateway.register_system_jobs()
    text = console.export_text()
    assert "Warning: No channels enabled" in text
    assert "Idle compaction: after 15 min" in text
    assert "Heartbeat: every 1800s" in text
    config.gateway.heartbeat.enabled = False
    config.agents.defaults.session_ttl_minutes = 0
    gateway.print_banner()
    gateway.register_system_jobs()
    text = console.export_text()
    assert "Idle compaction: disabled" in text and "Heartbeat: disabled" in text


# -- the CLI ------------------------------------------------------------------------------------


def test_gateway_config_section_defaults() -> None:
    config = Config()
    assert (config.gateway.host, config.gateway.port) == ("127.0.0.1", 18790)
    assert (config.gateway.heartbeat.enabled, config.gateway.heartbeat.interval_s) == (True, 1800)
    loaded = Config.model_validate({"gateway": {"port": 1234, "heartbeat": {"intervalS": 60, "enabled": False}}})
    assert loaded.gateway.port == 1234
    assert (loaded.gateway.heartbeat.enabled, loaded.gateway.heartbeat.interval_s) == (False, 60)


@pytest.mark.parametrize(("argv", "expected"), [([], None), (["--port", "18792"], 18792), (["-p", "9"], 9)])
def test_cli_passes_port_and_builds_the_agent(monkeypatch, tmp_path, argv, expected) -> None:
    captured = {}

    def fake_run_gateway(config, make_agent, *, port, config_path, console):
        captured["port"] = port
        agent, mcp = make_agent()
        captured["agent"] = agent
        captured["mcp"] = mcp
        return 0

    monkeypatch.setattr(gw, "run_gateway", fake_run_gateway)
    assert main(["gateway", "-w", str(tmp_path / "ws"), "--no-logs", *argv], provider=FakeProvider([])) == 0
    assert captured["port"] == expected
    assert isinstance(captured["agent"], AgentLoop)
    assert captured["agent"].workspace == (tmp_path / "ws").resolve()
    assert captured["agent"].auto_compact._ttl == 15  # agents.defaults.idleCompactAfterMinutes
    assert isinstance(captured["mcp"], MCPProvider)


def test_run_gateway_refuses_a_port_in_use(tmp_path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        console = Console(record=True, width=200)
        built = []
        status = gw.run_gateway(Config(), lambda: built.append(1), port=port, console=console)
    assert status == 1
    assert built == []  # nothing was built
    assert "already in use" in console.export_text()


def test_run_gateway_reports_a_startup_failure(tmp_path) -> None:
    def broken():
        raise RuntimeError("no loop for you")

    console = Console(record=True, width=200)
    assert gw.run_gateway(Config(), broken, port=free_port(), console=console) == 1
    assert "Error: no loop for you" in console.export_text()


# -- a real process -----------------------------------------------------------------------------


def _poll_health(port: int, proc: subprocess.Popen, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"gateway exited early: {proc.communicate()}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                assert response.status == 200
                return json.loads(response.read())
        except OSError:
            time.sleep(0.1)
    raise AssertionError("gateway /health never answered")


def _start(args: list[str], home: Path) -> subprocess.Popen:
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home), PYTHONPATH=str(REPO))
    env.pop("NANOCLAW_CONFIG", None)
    return subprocess.Popen(
        [sys.executable, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, cwd=str(home)
    )


FAKE_GATEWAY = textwrap.dedent(
    """
    import sys
    from nanoclaw.__main__ import main
    from nanoclaw.providers import FakeProvider
    raise SystemExit(main(sys.argv[1:], provider=FakeProvider([])))
    """
)


@pytest.mark.skipif(WINDOWS, reason="SIGTERM to a child is a hard kill on Windows")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_subprocess_gateway_with_fake_provider_health_and_clean_exit(tmp_path, sig) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"agents": {"defaults": {"idleCompactAfterMinutes": 5}}}))
    port = free_port()
    proc = _start(
        ["-c", FAKE_GATEWAY, "gateway", "-c", str(home / "config.json"), "--port", str(port), "-w", str(tmp_path / "ws")],
        home,
    )
    try:
        payload = _poll_health(port, proc)
        proc.send_signal(sig)
        out, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, err
    assert payload["status"] == "ok" and payload["websocket"] == "disabled" and payload["maintenance"] is False
    assert payload["runtime"]["version"] == __version__
    assert f"Starting nanoclaw gateway version {__version__} on port {port}..." in out
    assert "Warning: No channels enabled" in out
    assert "Idle compaction: after 5 min" in out
    assert f"Health endpoint: http://127.0.0.1:{port}/health" in out
    assert "Shutting down... Press Ctrl+C again to force." in out
    assert "Agent loop started" in err  # logs are on by default, as in nanobot
    assert f"Gateway shutdown requested by {signal.Signals(sig).name}" in err
    assert "Traceback" not in err


@pytest.mark.skipif(WINDOWS, reason="SIGTERM to a child is a hard kill on Windows")
def test_subprocess_real_cli_with_configured_provider(tmp_path) -> None:
    """``python -m nanoclaw gateway`` builds its provider from config.json;
    with no channels nothing calls it."""
    home = tmp_path / "home"
    home.mkdir()
    config = {"agents": {"defaults": {"provider": "deepseek", "model": "deepseek-chat"}},
              "providers": {"deepseek": {"apiKey": "test-key"}}}
    (home / "config.json").write_text(json.dumps(config))
    port = free_port()
    proc = _start(
        ["-m", "nanoclaw", "gateway", "-c", str(home / "config.json"), "--port", str(port),
         "-w", str(tmp_path / "ws"), "--no-logs"],
        home,
    )
    try:
        _poll_health(port, proc)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, err
    assert "Starting nanoclaw gateway" in out
    assert "Agent loop started" not in err  # --no-logs
