"""``nanoclaw gateway``: the long-running agent process, in the foreground.

Mirrors nanobot's foreground gateway (cli/gateway.py ``gateway`` and
cli/gateway_runtime.py ``_run_claimed_gateway`` / ``run``, c4dac64), cut down
to what NanoClaw has. One asyncio process runs:

- the agent loop (``AgentLoop.run``) on a message bus, with the configured
  MCP servers connected inside its task, as nanobot's ``_run_agent`` does;
  its 1 s idle tick runs idle auto-compaction (agent/autocompact.py);
- the ``/health`` endpoint on ``gateway.host``:``gateway.port`` (default
  127.0.0.1:18790; ``GET /health``, at most 64 connections, nanobot's JSON).

SIGINT / SIGTERM set a shutdown event (a second one cancels the tasks), then
the tasks are cancelled and awaited (bounded), the loop and MCP closed and the
bus drained (nanobot ``_install_gateway_shutdown_handlers``,
``_close_gateway_runtime``). The process then exits with status 0.

No channels yet (layers 23/24 add WebSocket and Telegram), so nothing feeds
the bus from outside. nanobot behaves the same with no channel enabled: it
prints "Warning: No channels enabled" and keeps running (cron, Dream,
heartbeat and idle compaction still work, ``/health`` answers ``ok`` because
the WebSocket channel is not required). Layer 21.4 adds the cron service with
Dream and heartbeat as system jobs: see ``Gateway.register_system_jobs`` and
the ``cron`` marks in ``Gateway.run``.

Left out of nanobot's runtime: background mode (``--background``,
``gateway status/stop/logs/restart``) and OS services
(``install-service``), on-demand client leases, install/profile claims and
the update machinery, the config-file watcher, local triggers, WebUI
recovery and dev server, the dock icon, ``/restart``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import signal
import socket
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from rich.console import Console

from nanoclaw import __version__
from nanoclaw.security.network import is_loopback_host

if TYPE_CHECKING:
    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.agent.tools.mcp import MCPProvider
    from nanoclaw.config import Config

# nanobot cli/gateway_runtime.py:309-310.
GATEWAY_HEALTH_MAX_CONNECTIONS = 64
GATEWAY_HEALTH_READ_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class GatewayHealthIdentity:
    """Opaque local instance/run identity; never configuration paths or
    credentials. (nanobot gateway/runtime.py:51)

    nanobot's ``instance_id`` hashes its per-config runtime state path (update
    profile receipt); NanoClaw has no state file, so it hashes the config
    path and workspace instead. ``run_id`` is new per process.
    """

    instance_id: str
    run_id: str
    version: str

    @classmethod
    def create(cls, config_path: Path, workspace: Path) -> GatewayHealthIdentity:
        seed = f"{config_path.expanduser().resolve(strict=False)}\0{workspace.expanduser().resolve(strict=False)}"
        return cls(hashlib.sha256(seed.encode()).hexdigest(), uuid.uuid4().hex, __version__)


def _signal_name(signum: int) -> str:
    with suppress(ValueError):
        return signal.Signals(signum).name
    return f"signal {signum}"


def _install_gateway_shutdown_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
    tasks: list[asyncio.Task[Any]],
    print_status: Callable[[str], None],
) -> Callable[[], None]:
    """Install foreground gateway signal handlers and return a restore callback.
    (nanobot gateway_runtime.py:171)"""
    loop_signals: list[int] = []
    previous_handlers: list[tuple[int, Any]] = []
    shutdown_requested = False

    def request_shutdown(signum: int) -> None:
        nonlocal shutdown_requested
        sig_name = _signal_name(signum)
        if shutdown_requested:
            logger.warning("Forcing gateway shutdown after repeated {}", sig_name)
            for task in tasks:
                if not task.done():
                    task.cancel()
            return
        shutdown_requested = True
        logger.info("Gateway shutdown requested by {}", sig_name)
        print_status("\nShutting down... Press Ctrl+C again to force.")
        shutdown_event.set()

    shutdown_signals = [signal.SIGINT, signal.SIGTERM]
    if sigbreak := getattr(signal, "SIGBREAK", None):
        shutdown_signals.append(sigbreak)

    for signum in shutdown_signals:
        try:
            loop.add_signal_handler(signum, request_shutdown, signum)
        except (NotImplementedError, RuntimeError, ValueError):
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, lambda sig, _frame: request_shutdown(sig))
            except (RuntimeError, ValueError):
                logger.debug("Could not install gateway handler for {}", _signal_name(signum))
                continue
            previous_handlers.append((signum, previous))
        else:
            loop_signals.append(signum)

    def restore() -> None:
        for signum in loop_signals:
            with suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(signum)
        for signum, handler in previous_handlers:
            with suppress(RuntimeError, ValueError):
                signal.signal(signum, handler)

    return restore


def _gateway_readiness_payload(channels: Any) -> tuple[bool, dict[str, object]]:
    """Describe process liveness separately from required WebSocket readiness.
    (nanobot gateway_runtime.py:329)

    ``channels`` is a channel manager (``get_status()``, ``enabled_channels``);
    NanoClaw has none yet and passes None: WebSocket "disabled", ready.
    """
    channel_status: dict[str, Any] = {}
    get_status = getattr(channels, "get_status", None)
    if callable(get_status):
        try:
            raw_status = get_status()
            if isinstance(raw_status, dict):
                channel_status = cast(dict[str, Any], raw_status)
        except Exception:
            logger.exception("Gateway readiness could not read channel status")

    websocket = channel_status.get("websocket")
    websocket_required = websocket is not None or "websocket" in getattr(channels, "enabled_channels", ())
    if not websocket_required:
        websocket_state = "disabled"
        ready = True
    elif isinstance(websocket, dict):
        websocket_status = cast(dict[str, Any], websocket)
        ready = websocket_status.get("running") is True
        state = websocket_status.get("state")
        websocket_state = str(state) if isinstance(state, str) else "unavailable"
    else:
        ready = False
        websocket_state = "unavailable"

    return ready, {
        "status": "ok" if ready else "degraded",
        "process": "alive",
        "ready": ready,
        "websocket": websocket_state,
    }


def health_payload(
    channels: Any, identity: GatewayHealthIdentity, *, maintenance: bool = False
) -> tuple[bool, dict[str, object]]:
    """The ``/health`` body. (nanobot gateway_runtime.py:905-908; ``maintenance``
    is the updater's pause flag, always False in NanoClaw.)"""
    ready, payload = _gateway_readiness_payload(channels)
    payload["runtime"] = asdict(identity)
    payload["maintenance"] = maintenance
    return ready, payload


def _host_for_local_browser(host: str) -> str:
    """(nanobot cli/webui_support.py:344)"""
    if host in {"", "0.0.0.0"}:
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def gateway_health_url(host: str, port: int) -> str:
    """A health URL that can be opened from this device. (nanobot webui_support.py:357)"""
    return f"http://{_host_for_local_browser(host)}:{port}/health"


def _tcp_endpoint_reachable(host: str, port: int, *, timeout_s: float = 0.25) -> bool:
    """Whether a local TCP endpoint accepts connections. (nanobot webui_support.py:454)"""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


HealthPayload = Callable[[], tuple[bool, dict[str, object]]]
HealthHandler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


def make_health_handler(
    payload: HealthPayload,
    *,
    max_connections: int = GATEWAY_HEALTH_MAX_CONNECTIONS,
    read_timeout: float = GATEWAY_HEALTH_READ_TIMEOUT_SECONDS,
) -> HealthHandler:
    """The connection handler of the health server: ``GET /health`` answers
    the payload (200, or 503 when not ready), anything else 404; at most
    ``max_connections`` at once, the rest are closed unanswered.
    (nanobot ``_health_server.handle``, gateway_runtime.py:882)"""
    connection_slots = asyncio.Semaphore(max_connections)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if connection_slots.locked():
            writer.close()
            return

        async with connection_slots:
            try:
                data = await asyncio.wait_for(reader.read(4096), timeout=read_timeout)
                request_line = data.split(b"\r\n", 1)[0].decode("utf-8", errors="replace")
                method, path = "", ""
                parts = request_line.split(" ")
                if len(parts) >= 2:
                    method, path = parts[0], parts[1]

                if method == "GET" and path == "/health":
                    ready, body_payload = payload()
                    body = json.dumps(body_payload)
                    status = "200 OK" if ready else "503 Service Unavailable"
                    content_type = "application/json"
                else:
                    body = "Not Found"
                    status = "404 Not Found"
                    content_type = "text/plain"

                resp = (
                    f"HTTP/1.0 {status}\r\n"
                    f"Content-Type: {content_type}\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n"
                    f"\r\n{body}"
                )
                writer.write(resp.encode())
                await writer.drain()
            except (asyncio.TimeoutError, ConnectionError):
                pass
            finally:
                writer.close()

    return handle


async def serve_health(
    host: str,
    port: int,
    payload: HealthPayload,
    *,
    on_listening: Callable[[], None] | None = None,
) -> None:
    """Lightweight HTTP health endpoint on the gateway port; runs until cancelled.
    (nanobot ``_health_server``, gateway_runtime.py:876)"""
    server = await asyncio.start_server(make_health_handler(payload), host, port)
    if on_listening is not None:
        on_listening()
    async with server:
        await server.serve_forever()


async def _close_gateway_runtime(
    agent: AgentLoop,
    mcp_provider: MCPProvider,
    tasks: list[asyncio.Task[Any]],
    runtime_tasks: asyncio.Future[list[Any]] | None,
    *,
    task_wait_timeout: float = 15.0,
    close_timeout: float = 15.0,
) -> None:
    """Cancel runtime tasks, then close the loop and MCP, each bounded.
    (nanobot gateway_runtime.py:365; no channels to stop first)"""
    for task in tasks:
        if not task.done():
            task.cancel()
    pending: set[asyncio.Task[Any]] = set()
    if tasks:
        # Bounded: a coroutine that swallows cancellation must not hold the
        # stop open; whatever is still pending is re-cancelled and closed
        # underneath.
        _done, pending = await asyncio.wait(tasks, timeout=task_wait_timeout)
        for task in pending:
            task.cancel()
    if runtime_tasks is not None and not runtime_tasks.done():
        runtime_tasks.cancel()
    for label, close in (("agent", agent.aclose), ("MCP provider", mcp_provider.aclose)):
        try:
            await asyncio.wait_for(close(), timeout=close_timeout)
        except BaseException as exc:  # noqa: BLE001 - shutdown must proceed
            logger.warning("Gateway shutdown: {} cleanup incomplete: {}", label, exc)
    # Retrieve an already-finished gather so it logs nothing, but never wait for it.
    if runtime_tasks is not None and runtime_tasks.done():
        with suppress(asyncio.CancelledError, Exception):
            await runtime_tasks


class Gateway:
    """One foreground gateway run: an agent loop, its MCP servers and ``/health``.

    The caller builds the loop (``__main__._make_loop``) and MCP provider as
    ``chat`` does. ``run`` blocks until SIGINT/SIGTERM or until a runtime task
    ends; ``health_server_enabled=False`` leaves the endpoint out (tests).
    """

    def __init__(
        self,
        config: Config,
        agent: AgentLoop,
        mcp_provider: MCPProvider,
        *,
        port: int | None = None,
        config_path: Path | None = None,
        console: Console | None = None,
        health_server_enabled: bool = True,
    ) -> None:
        self.config = config
        self.agent = agent
        self.bus = agent.bus
        self.mcp_provider = mcp_provider
        self.host = config.gateway.host
        self.port = port if port is not None else config.gateway.port
        self.console = console or Console()
        self.health_server_enabled = health_server_enabled
        self.health_identity = GatewayHealthIdentity.create(config_path or Path("config.json"), agent.workspace)
        # No channel manager yet (layers 23/24); /health treats it as "no channels".
        self.channels: Any = None
        self.enabled_channels: list[str] = []
        self.shutdown_event = asyncio.Event()
        self.tasks: list[asyncio.Task[Any]] = []
        # Layer 21.4: the CronService (workspace/cron/jobs.json) goes here,
        # with ``cron.on_job`` dispatching dream / heartbeat / bound agent jobs
        # (nanobot gateway_runtime.py:655-805).
        self.cron: Any = None
        # Extra long-running coroutines to run beside the loop (named tasks).
        self.extra_tasks: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
        self.crashed = False

    # -- startup ----------------------------------------------------------------

    def health(self) -> tuple[bool, dict[str, object]]:
        return health_payload(self.channels, self.health_identity)

    def print_banner(self) -> None:
        """The startup lines nanobot prints after building the runtime
        (gateway_runtime.py:849-852); the "Starting ..." line comes first, from
        ``run_gateway``. The idle-compaction line is NanoClaw's."""
        if self.enabled_channels:
            self.console.print(f"[green]✓[/green] Channels enabled: {', '.join(self.enabled_channels)}")
        else:
            self.console.print("[yellow]Warning: No channels enabled[/yellow]")
        ttl = self.config.agents.defaults.session_ttl_minutes
        if ttl > 0:
            self.console.print(f"[green]✓[/green] Idle compaction: after {ttl} min")
        else:
            self.console.print("[yellow]○[/yellow] Idle compaction: disabled")

    def register_system_jobs(self) -> None:
        """Register the protected cron system jobs. (nanobot gateway_runtime.py:934-965)

        Layer 21.4 fills this in: Dream every ``agents.defaults.dream``
        interval (``cron.register_system_job(CronJob(id="dream", ...))``, or
        ``remove_system_job("dream")`` when disabled) and the heartbeat every
        ``gateway.heartbeat.intervalS`` (id "heartbeat"), then prints
        "✓ Dream: ...", "✓ Heartbeat: every Ns" and "✓ Cron: N scheduled jobs".
        Until then the heartbeat settings are only reported.
        """
        hb_cfg = self.config.gateway.heartbeat
        if hb_cfg.enabled:
            self.console.print(f"[yellow]○[/yellow] Heartbeat: every {hb_cfg.interval_s}s (not scheduled yet)")
        else:
            self.console.print("[yellow]✗[/yellow] Heartbeat: disabled")

    def _print_health_endpoint(self) -> None:
        """(nanobot ``_print_gateway_health_endpoint``, gateway_runtime.py:313)"""
        note = "" if is_loopback_host(self.host) else f" [dim](listening on {self.host})[/dim]"
        self.console.print(f"[green]✓[/green] Health endpoint: {gateway_health_url(self.host, self.port)}{note}")
        if is_loopback_host(self.host):
            return
        self.console.print(
            "[yellow]Warning: the unauthenticated health endpoint is listening beyond loopback "
            "and may be reachable from other devices. "
            f"Keep port {self.port} private or protect it with a firewall or reverse proxy.[/yellow]"
        )

    # -- the run ------------------------------------------------------------------

    async def _run_agent(self) -> None:
        """MCP connects inside the loop's task and closes when it ends (nanobot :1035)."""
        try:
            await self.mcp_provider.connect()
            await self.agent.run()
        finally:
            await self.mcp_provider.aclose()

    async def run(self) -> None:
        """nanobot ``run()`` (gateway_runtime.py:1013), minus updates, leases,
        recovery, triggers, the config watcher and the WebUI."""
        tasks = self.tasks
        shutdown_task: asyncio.Task[Any] | None = None
        runtime_tasks: asyncio.Future[list[Any]] | None = None
        startup_complete = False
        shutdown_event = self.shutdown_event
        restore_shutdown_handlers = _install_gateway_shutdown_handlers(
            asyncio.get_running_loop(), shutdown_event, tasks, self.console.print
        )
        try:
            # Layer 21.4: ``await self.cron.start()`` here (nanobot :1031),
            # after interrupted subagents are recovered (AgentLoop.run does that).
            tasks.append(asyncio.create_task(self._run_agent(), name="nanoclaw-agent-loop"))
            if self.health_server_enabled:
                tasks.append(
                    asyncio.create_task(
                        serve_health(self.host, self.port, self.health, on_listening=self._print_health_endpoint),
                        name="nanoclaw-health-server",
                    )
                )
            for name, factory in self.extra_tasks:
                tasks.append(asyncio.create_task(factory(), name=name))
            runtime_tasks = asyncio.gather(*tasks)
            startup_complete = True
            shutdown_task = asyncio.create_task(shutdown_event.wait(), name="nanoclaw-gateway-shutdown")
            done, _pending = await asyncio.wait({runtime_tasks, shutdown_task}, return_when=asyncio.FIRST_COMPLETED)
            if runtime_tasks in done:
                await runtime_tasks
            else:
                runtime_tasks.cancel()
        except KeyboardInterrupt:
            self.console.print("\nShutting down...")
        except asyncio.CancelledError:
            # A forced shutdown (second signal) cancelled the runtime tasks.
            if not shutdown_event.is_set():
                raise
        except Exception:
            import traceback

            self.console.print("\n[red]Error: Gateway crashed unexpectedly[/red]")
            self.console.print(traceback.format_exc())
            # nanobot exits 0 after a crash once started; NanoClaw reports it.
            self.crashed = True
            if not startup_complete:
                raise
        finally:
            try:
                if shutdown_task and not shutdown_task.done():
                    shutdown_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await shutdown_task
                # Layer 21.4: ``self.cron.stop()`` here (nanobot :1137).
                self.agent.stop()
                await _close_gateway_runtime(self.agent, self.mcp_provider, tasks, runtime_tasks)
                await self.bus.drain()
            finally:
                restore_shutdown_handlers()


AgentFactory = Callable[[], "tuple[AgentLoop, MCPProvider]"]


def run_gateway(
    config: Config,
    make_agent: AgentFactory,
    *,
    port: int | None = None,
    config_path: Path | None = None,
    console: Console | None = None,
) -> int:
    """Start the foreground gateway and block until it stops; returns the exit status.

    ``make_agent`` builds the agent loop and its MCP provider inside the
    event loop (``chat`` does the same). Startup order as nanobot: port
    check, banner, the runtime, system jobs, then the tasks.
    """
    console = console or Console()
    port = port if port is not None else config.gateway.port
    if _tcp_endpoint_reachable(_host_for_local_browser(config.gateway.host).strip("[]"), port):
        console.print(
            f"[red]Cannot start the gateway: {gateway_health_url(config.gateway.host, port)} is already in use.[/red]"
        )
        console.print("Stop the other instance, or choose an unused port with [cyan]--port[/cyan].")
        return 1
    console.print(f"Starting nanoclaw gateway version {__version__} on port {port}...")

    async def main() -> int:
        agent, mcp_provider = make_agent()
        gateway = Gateway(config, agent, mcp_provider, port=port, config_path=config_path, console=console)
        gateway.print_banner()
        gateway.register_system_jobs()
        await gateway.run()
        return 1 if gateway.crashed else 0

    try:
        return asyncio.run(main())
    except Exception as exc:  # startup failed before any task ran (nanobot: typer.Exit(1))
        console.print(f"[red]Error: {exc}[/red]")
        return 1
