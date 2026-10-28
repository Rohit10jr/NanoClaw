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
the WebSocket channel is not required).

The cron service (``<workspace>/cron/jobs.json``, the same one the loop's
``cron`` tool writes to) starts before the loop and stops first on shutdown.
``Gateway.on_cron_job`` runs its due jobs as nanobot's ``on_cron_job``
(gateway_runtime.py:655): the protected ``dream`` system job runs Dream
directly (``agents.defaults.dream``: every ``intervalH`` hours, default 2, or
a legacy ``cron`` expression), the protected ``heartbeat`` job reads
``HEARTBEAT.md`` every ``gateway.heartbeat.intervalS`` seconds (default 1800),
session-bound jobs run as turns of their session, other agent jobs are
skipped. The heartbeat needs a delivery target: the most recently used
session of an enabled channel other than the CLI. With no channels (until
layers 23/24) it finds none and does nothing, as nanobot does with only the
CLI.

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
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from rich.console import Console

from nanoclaw import __version__
from nanoclaw.bus.events import OutboundMessage
from nanoclaw.cron.service import CronJobSkippedError, CronService
from nanoclaw.cron.session_turns import is_bound_cron_job
from nanoclaw.cron.types import CronJob, CronPayload, CronRunResult, CronSchedule
from nanoclaw.security.network import is_loopback_host
from nanoclaw.session.keys import HEARTBEAT_SESSION_KEY, session_key_for_channel
from nanoclaw.utils.evaluator import evaluate_response, resolve_evaluator_prompt

if TYPE_CHECKING:
    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.agent.memory import MemoryStore
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


def _advance_dream_cursor_if_behind(memory: MemoryStore) -> None:
    """With Dream disabled, mark the whole journal as processed.
    (nanobot gateway_runtime.py:225)"""
    latest = memory.get_latest_cursor()
    if memory.get_last_dream_cursor() < latest:
        memory.set_last_dream_cursor(latest)


_HEARTBEAT_PREAMBLE = (
    "[Your response will be delivered directly to the user's messaging app. "
    "Output ONLY the final user-facing message. Never reference internal "
    "files (HEARTBEAT.md, AWARENESS.md, etc.), your instructions, or your "
    "decision process. If nothing needs reporting, respond with just "
    "'All clear.' and nothing else.]\n\n"
)


def _heartbeat_has_active_tasks(content: str) -> bool:
    """True if HEARTBEAT.md has task lines, ignoring headers, blanks and comments.
    (nanobot gateway_runtime.py:254)"""
    in_comment = False
    in_active_section: bool = False
    for line in content.splitlines():
        stripped = line.strip()
        if in_comment:
            if "-->" in stripped:
                in_comment = False
            continue
        if not stripped or stripped.startswith("#"):
            if stripped.startswith("##") and not stripped.startswith("###"):
                heading = stripped.lstrip("#").strip().lower()
                in_active_section = heading.startswith("active tasks")
            continue
        if stripped.startswith("<!--"):
            if "-->" not in stripped[4:]:
                in_comment = True
            continue
        if in_active_section is False:
            continue
        return True
    return False


def _pick_heartbeat_target_from_sessions(
    *,
    enabled_channels: Iterable[str],
    sessions: Iterable[dict[str, Any]],
    archived_keys: Iterable[str],
) -> tuple[str, str]:
    """The first ``channel:chat_id`` session of an enabled channel other than
    the CLI, in the given order (most recently updated first); ``("cli",
    "direct")`` when there is none. (nanobot gateway_runtime.py:279, without
    the unified session; ``archived_keys`` are nanobot's WebUI-archived chats,
    none in NanoClaw.)"""
    enabled = set(enabled_channels)
    archived = set(archived_keys)
    for item in sessions:
        key = item.get("key") or ""
        if key in archived:
            continue
        if ":" not in key:
            continue
        channel, chat_id = key.split(":", 1)
        if channel in {"cli", "system"}:
            continue
        if channel in enabled and chat_id:
            return channel, chat_id
    return "cli", "direct"


def _sessions_most_recent_first(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """nanobot's session store lists ``ORDER BY updated_at DESC``; NanoClaw's
    lists by file name."""
    return sorted(sessions, key=lambda item: str(item.get("updated_at") or ""), reverse=True)


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
        # One cron service per workspace store: the loop's (its ``cron`` tool
        # writes to it), else a new one on <workspace>/cron/jobs.json.
        cron = getattr(agent, "cron_service", None)
        if not isinstance(cron, CronService):
            cron = CronService(agent.workspace / "cron" / "jobs.json")
        self.cron: CronService = cron
        self.cron.on_job = self.on_cron_job
        # Extra long-running coroutines to run beside the loop (named tasks).
        self.extra_tasks: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
        self.crashed = False

    # -- startup ----------------------------------------------------------------

    def health(self) -> tuple[bool, dict[str, object]]:
        return health_payload(self.channels, self.health_identity)

    def print_banner(self) -> None:
        """The startup lines nanobot prints after building the runtime
        (gateway_runtime.py:849-858); the "Starting ..." line comes first, from
        ``run_gateway``. The idle-compaction line is NanoClaw's."""
        if self.enabled_channels:
            self.console.print(f"[green]✓[/green] Channels enabled: {', '.join(self.enabled_channels)}")
        else:
            self.console.print("[yellow]Warning: No channels enabled[/yellow]")
        hb_cfg = self.config.gateway.heartbeat
        if hb_cfg.enabled:
            self.console.print(f"[green]✓[/green] Heartbeat: every {hb_cfg.interval_s}s")
        else:
            self.console.print("[yellow]✗[/yellow] Heartbeat: disabled")
        ttl = self.config.agents.defaults.session_ttl_minutes
        if ttl > 0:
            self.console.print(f"[green]✓[/green] Idle compaction: after {ttl} min")
        else:
            self.console.print("[yellow]○[/yellow] Idle compaction: disabled")

    def register_system_jobs(self) -> None:
        """Register (or remove) the protected cron system jobs, idempotent on
        restart. (nanobot gateway_runtime.py:934-965)"""
        defaults = self.config.agents.defaults
        dream_cfg = defaults.dream
        if dream_cfg.enabled:
            self.cron.register_system_job(
                CronJob(
                    id="dream",
                    name="dream",
                    schedule=dream_cfg.build_schedule(defaults.timezone),
                    payload=CronPayload(kind="system_event"),
                )
            )
            self.console.print(f"[green]✓[/green] Dream: {dream_cfg.describe_schedule()}")
        else:
            self.console.print("[yellow]○[/yellow] Dream: disabled")
            # Cursor repair must not depend on a healthy cron store.
            _advance_dream_cursor_if_behind(self.agent.context.memory)
            self.cron.remove_system_job("dream")

        hb_cfg = self.config.gateway.heartbeat
        if hb_cfg.enabled:
            self.cron.register_system_job(
                CronJob(
                    id="heartbeat",
                    name="heartbeat",
                    schedule=CronSchedule(kind="every", every_ms=hb_cfg.interval_s * 1000, tz=defaults.timezone),
                    payload=CronPayload(kind="system_event"),
                )
            )
        else:
            self.cron.remove_system_job("heartbeat")

        cron_job_count = cast(int, self.cron.status()["jobs"])
        if cron_job_count > 0:
            self.console.print(f"[green]✓[/green] Cron: {cron_job_count} scheduled jobs")

    # -- cron jobs ----------------------------------------------------------------

    async def on_cron_job(self, job: CronJob) -> str | CronRunResult | None:
        """Execute a due cron job. (nanobot ``on_cron_job``, gateway_runtime.py:655)"""
        if job.name == "dream":
            await self.run_dream_job()
            return None
        if job.name == "heartbeat":
            return await self.run_heartbeat_job()
        if is_bound_cron_job(job):
            from nanoclaw.cron.bound_runner import run_bound_cron_job

            return await run_bound_cron_job(job, agent=self.agent, cron=self.cron)

        reason = "unbound agent cron job must be recreated from a chat session"
        logger.warning("Cron: skipped unbound agent job '{}' ({}): {}", job.name, job.id, reason)
        raise CronJobSkippedError(reason)

    async def run_dream_job(self) -> None:
        """Dream is an internal job: run directly, not through the bus.
        (nanobot gateway_runtime.py:660-722)

        As ``AgentLoop.run_dream`` (``/dream``) but with no reply and
        nanobot's scheduled commit (``commit_dream_changes``: message
        ``dream: periodic memory consolidation``, only when a memory file
        changed). nanobot then prunes its saved Dream sessions; NanoClaw's
        Dream turns are never saved.
        """
        from nanoclaw.agent.memory import MemoryStore, commit_dream_changes

        agent = self.agent
        store = agent.context.memory
        try:
            result = store.build_dream_prompt()
            if result is None:
                logger.info("Dream: nothing to process")
                return
            prompt, last_cursor = result
            dream_runtime = agent.dream_runtime()
            await self.mcp_provider.connect()
            resp = await agent.process_direct(
                prompt,
                session_key=MemoryStore.dream_session_key(),
                ephemeral=True,
                tools=store.build_dream_tools(),
                # Dream edits the agent's own files, never the project.
                workspace_scope=agent.agent_workspace_scope,
                runtime=dream_runtime,
            )
            # The real file delta grounds the audit record; normal completion
            # decides whether this history batch has finished processing.
            diff_body = store.dream_content_diff()
            if resp.stop_reason == "completed":
                store.set_last_dream_cursor(last_cursor)
                if diff_body:
                    logger.info("Dream cron job completed, cursor advanced to {}", last_cursor)
                else:
                    logger.info("Dream cron job completed with no memory changes; cursor advanced to {}", last_cursor)
            else:
                logger.warning(
                    "Dream cron job did not complete ({}); cursor remains at {}",
                    f"stop_reason: {resp.stop_reason}",
                    store.get_last_dream_cursor(),
                )
        except Exception:
            logger.exception("Dream cron job failed")
        finally:
            sha = commit_dream_changes(store)
            if sha:
                logger.info("Dream commit: {}", sha)
            store.compact_history()

    def pick_heartbeat_target(self) -> tuple[str, str]:
        """Pick a routable channel/chat target for heartbeat-triggered messages.
        (nanobot ``_pick_heartbeat_target``, gateway_runtime.py:830)"""
        return _pick_heartbeat_target_from_sessions(
            enabled_channels=self.enabled_channels,
            sessions=_sessions_most_recent_first(self.agent.sessions.list_sessions()),
            archived_keys=(),
        )

    async def run_heartbeat_job(self) -> str | None:
        """Check HEARTBEAT.md for active tasks, run them in the ``heartbeat``
        session, and deliver the result to the target chat if the evaluator
        says so. (nanobot gateway_runtime.py:725-800)

        nanobot switches its ``message`` tool to suppressed delivery for this
        turn so every word passes the evaluator. NanoClaw has no message tool,
        and a ``process_direct`` turn publishes nothing on the bus, so the
        effect is the same: the only delivery is the gated one below.
        """
        heartbeat_file = self.agent.workspace / "HEARTBEAT.md"
        try:
            content = heartbeat_file.read_text(encoding="utf-8")
        except OSError:
            logger.debug("Heartbeat: HEARTBEAT.md missing")
            return None
        if not _heartbeat_has_active_tasks(content):
            logger.debug("Heartbeat: HEARTBEAT.md has no active tasks")
            return None

        channel, chat_id = self.pick_heartbeat_target()
        if channel == "cli":
            return None
        delivery_key = session_key_for_channel(channel, chat_id)
        if self.agent.sessions.get_existing(delivery_key) is None:
            return None

        prompt = (
            _HEARTBEAT_PREAMBLE
            + "You are executing periodic heartbeat tasks. Read the active tasks below, "
            f"perform each one, and report what you did:\n\n{content}"
        )
        await self.mcp_provider.connect()
        resp = await self.agent.process_direct(
            prompt,
            session_key=HEARTBEAT_SESSION_KEY,
            channel=channel,
            chat_id=chat_id,
        )
        if not resp or not resp.final_content:
            return None
        response = resp.final_content

        evaluator_prompt = resolve_evaluator_prompt(self.agent.workspace)
        # Fail closed: stay silent on evaluator failure instead of notifying.
        should_notify = await evaluate_response(
            response=response,
            task_context=prompt,
            provider=self.agent.provider,
            model=self.agent.model,  # None: the provider's default model
            evaluator_prompt=evaluator_prompt,
            default_notify=False,
        )
        if should_notify:
            logger.info("Heartbeat: completed, delivering response")
            await self.deliver_to_channel(
                OutboundMessage(channel=channel, chat_id=chat_id, content=response),
                record=True,
                session_key=delivery_key,
            )
        else:
            logger.info("Heartbeat: silenced by post-run evaluation")
        return response

    async def deliver_to_channel(
        self, msg: OutboundMessage, *, record: bool = False, session_key: str | None = None
    ) -> None:
        """Publish a user-visible message and mirror it into its session.
        (nanobot ``_deliver_to_channel``, gateway_runtime.py:72)

        The recorded copy is an assistant message marked ``_channel_delivery``,
        so a reply to it replays it as context. nanobot checks the session's
        generation (a reset in between drops the record); NanoClaw has no
        session generations and records under the session lock instead.
        """
        key = session_key or session_key_for_channel(msg.channel, msg.chat_id)
        if record and msg.channel != "cli" and msg.content.strip():
            extra: dict[str, Any] = {"_channel_delivery": True}
            if msg.media:
                extra["media"] = list(msg.media)
            async with self.agent._get_session_lock(key):
                session = self.agent.sessions.get_or_create(key)
                session.add_message("assistant", msg.content, **extra)
                self.agent.sessions.save(session)
        await self.bus.publish_outbound(msg)

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
            # nanobot :1031. (nanobot recovers interrupted subagents first;
            # NanoClaw's AgentLoop.run does that when its task starts.)
            await self.cron.start()
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
                self.cron.stop()  # nanobot :1137
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
