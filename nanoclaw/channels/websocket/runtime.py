"""The web UI channel: one aiohttp listener for HTTP, the WebSocket and the frontend.

A port of nanobot channels/websocket/runtime.py ``WebSocketChannel`` onto
aiohttp (phase 2 decision 13; nanobot runs ``websockets`` with an aiohttp
bridge for uploads). One port (``channels.websocket.host``/``port``,
default 127.0.0.1:8765) serves:

- the WebSocket upgrade on ``path`` (``?token=<one-time token>``, see
  ``authorize_websocket_handshake``) with nanobot's envelope protocol
  (nanoclaw/webui/inbound_commands.py);
- ``GET /webui/bootstrap`` and the optional token issue route
  (nanoclaw/webui/gateway_http.py);
- the REST routes of nanoclaw/webui/rest_api.py (``register_rest_routes``),
  then its 404 for other ``/api/*`` paths;
- the built frontend (``webuiDist`` / ``NANOCLAW_WEBUI_DIST``) with SPA
  fallback.

Outbound: the channel manager hands every bus message for ``websocket`` to
``send``; ``WebUIOutboundProjector.project`` turns it into wire frames, each
frame's ``record`` is appended to the chat's history
(``WebUITranscriptRecorder``) and its payload goes to its audience. The
frames' turn hints drive the active-turn registry (``self.turns``,
nanobot's module-level ``_WEBSOCKET_ACTIVE_TURNS``).

Lifecycle (nanobot ``WebuiTurnCoordinator``, session/webui_turns.py:587):
the channel subscribes to the turn runtime events and publishes
``lifecycle_outbound(...)`` for websocket turns, marks a session opened by
the web UI (``metadata["webui"]``) and, after a web UI turn completes,
generates the chat title and announces ``session_updated``.

Each connection has a bounded outbound queue (256 frames, 8 MiB) drained by
its own writer task, so a slow browser never blocks the others (nanobot
runtime.py:70-77).

Left out: unix sockets, the listener restart loop, binary uploads, media
signing, temporary chats, pairing, subagent cards, terminal leases.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import ssl
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from aiohttp import WSMsgType, web

from nanoclaw.bus.events import OutboundMessage
from nanoclaw.bus.outbound_events import SessionUpdatedEvent, outbound_message_for_event
from nanoclaw.bus.runtime_events import (
    SessionTurnStarted,
    TurnCompleted,
    TurnRunStatusChanged,
    TurnRuntimeAdmitted,
)
from nanoclaw.channels.base import BaseChannel
from nanoclaw.session.webui_turns import (
    WEBUI_SESSION_METADATA_KEY,
    WebsocketTurnRegistry,
    maybe_generate_webui_title_after_turn,
)
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.webui.context import WebUIContext
from nanoclaw.webui.gateway_http import WebUIGatewayHTTP, http_error, resolve_webui_dist
from nanoclaw.webui.gateway_tokens import GatewayTokenStore
from nanoclaw.webui.history import WebUIHistoryStore
from nanoclaw.webui.http_utils import normalize_config_path, strip_trailing_slash
from nanoclaw.webui.inbound_commands import WebUICommandRouter, mutation_error, mutation_json
from nanoclaw.webui.metadata import WEBSOCKET_TURN_OWNER_METADATA_KEY, WEBUI_TURN_METADATA_KEY
from nanoclaw.webui.outbound_projection import WebUIOutboundProjector, WireFrame, lifecycle_outbound
from nanoclaw.webui.outbound_wire import encode_goal_status
from nanoclaw.webui.session_identity import is_webui_session_key, webui_chat_id, webui_session_key
from nanoclaw.webui.session_list_index import (
    WORKSPACE_SCOPE_METADATA_KEY,
    delete_webui_session,
    recovery_state_from_metadata,
)
from nanoclaw.webui.transcript import WEBUI_TRANSCRIPT_INCOMPLETE_KEY, WebUITranscriptRecorder

if TYPE_CHECKING:
    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.bus.queue import MessageBus
    from nanoclaw.config import Config
    from nanoclaw.config.schema import WebSocketConfig

# nanobot runtime.py:74-77.
_OUTBOUND_QUEUE_MAX_FRAMES = 256
_OUTBOUND_QUEUE_MAX_BYTES = 8 * 1024 * 1024
_OUTBOUND_SEND_TIMEOUT_S = 10.0

_SESSION_KEY_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:.-")


def _parse_inbound_payload(raw: str) -> str | None:
    """A legacy frame's text: plain text or ``{"content"|"text"|"message": ...}``. (nanobot runtime.py:309)"""
    text = raw.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            data = cast(object, json.loads(text))
        except json.JSONDecodeError:
            return text
        if isinstance(data, dict):
            payload = cast(dict[str, Any], data)
            for key in ("content", "text", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return None
    return text


def _parse_envelope(raw: str) -> dict[str, Any] | None:
    """A JSON object with a string ``type``, else None (legacy frame). (nanobot runtime.py:334)"""
    text = raw.strip()
    if not text.startswith("{"):
        return None
    try:
        data = cast(object, json.loads(text))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    envelope = cast(dict[str, Any], data)
    return envelope if isinstance(envelope.get("type"), str) else None


def _config_model(config: Config | None) -> str | None:
    """The configured model when the loop names none (nanobot
    ``_default_model_name_from_config``, ws_http.py:324)."""
    if config is None:
        return None
    try:
        return config.resolve_preset().model.strip() or None
    except Exception:
        return None


def _default_config() -> Config:
    from nanoclaw.config import Config

    return Config()


def _peer_address(request: web.Request) -> Any:
    """The TCP peer as ``(host, port)`` (nanobot's ``remote_address``), else the host."""
    transport = request.transport
    peer = transport.get_extra_info("peername") if transport is not None else None
    if isinstance(peer, tuple) and len(peer) >= 2:
        return (peer[0], peer[1])
    return request.remote


def _is_websocket_upgrade(request: web.Request) -> bool:
    """(nanobot gateway_endpoint.py:28)"""
    upgrade = request.headers.get("Upgrade", "")
    connection = request.headers.get("Connection", "")
    return "websocket" in upgrade.lower() and "upgrade" in connection.lower()


@dataclass(eq=False)
class WebUIConnection:
    """One open socket and its outbound queue (nanobot ``_ConnectionOutbound``)."""

    ws: web.WebSocketResponse
    client_id: str
    remote: Any  # the peer (host, port), as nanobot's ``connection.remote_address``
    trusted_webui: bool = False
    queue: asyncio.Queue[tuple[str, int]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_OUTBOUND_QUEUE_MAX_FRAMES)
    )
    buffered_bytes: int = 0
    writer: asyncio.Task[None] | None = None
    closing: bool = False

    @property
    def closed(self) -> bool:
        return self.closing or self.ws.closed


class WebSocketChannel(BaseChannel):
    """Serve the web UI on one local listener; forward its messages to the bus."""

    name = "websocket"
    display_name = "WebSocket"

    def __init__(
        self,
        config: WebSocketConfig,
        bus: MessageBus,
        *,
        agent: AgentLoop,
        app_config: Config | None = None,
    ) -> None:
        super().__init__(config, bus)
        self.config: WebSocketConfig = config
        self.agent = agent
        self.sessions = agent.sessions
        self.history = WebUIHistoryStore.for_sessions(self.sessions)
        self.projector = WebUIOutboundProjector(
            show_reasoning=self.show_reasoning,
            send_progress=self.send_progress,
            send_tool_hints=self.send_tool_hints,
        )
        self.recorder = WebUITranscriptRecorder(self.history)
        # One turn sequence per (chat, turn) for the user row (recorder) and
        # the reply rows (projector), as nanobot's single recorder keeps it.
        self.recorder._turn_sequences = self.projector._turn_sequences
        self.turns = WebsocketTurnRegistry()
        self.tokens = GatewayTokenStore()
        self.static_dist_path: Path | None = resolve_webui_dist(config.webui_dist)
        self.http = WebUIGatewayHTTP(
            config,
            self.tokens,
            static_dist_path=self.static_dist_path,
            runtime_model_name=lambda: agent.model or _config_model(app_config),
        )
        self.context = WebUIContext(
            config=app_config if app_config is not None else _default_config(),
            sessions=self.sessions,
            history=self.history,
            loop=agent,
            workspace=agent.workspace,
            check_auth=self.tokens.check_api_token,
            channel=self,
        )
        self.context.mutations["session.delete"] = self._mutation_session_delete
        self.commands = WebUICommandRouter(self)
        self._connections: set[WebUIConnection] = set()
        self._subs: dict[str, set[WebUIConnection]] = {}
        self._conn_chats: dict[WebUIConnection, set[str]] = {}
        self._conn_default: dict[WebUIConnection, str] = {}
        self._minted_owners: dict[str, str] = {}
        self._background: set[asyncio.Task[Any]] = set()
        self._unsubscribe: list[Any] = []
        self._stop_event: asyncio.Event | None = None
        self._runner: web.AppRunner | None = None
        self.bound_port: int | None = None
        self.app = self.build_app()

    # -- the aiohttp application -------------------------------------------------

    def build_app(self) -> web.Application:
        """The REST routes (nanoclaw/webui/rest_api.py), the ``/api`` 404
        fallback, then one catch-all for the socket, bootstrap, static files
        and 404s (aiohttp matches routes in registration order)."""
        from nanoclaw.webui.rest_api import register_api_fallback, register_rest_routes

        app = web.Application(client_max_size=self.config.max_message_bytes)
        register_rest_routes(app, self.context)
        register_api_fallback(app)
        app.router.add_route("*", "/{tail:.*}", self._dispatch_http)
        return app

    async def _dispatch_http(self, request: web.Request) -> web.StreamResponse:
        """Route one request. (nanobot gateway_endpoint.py ``process_request`` and
        ws_http.py ``_dispatch_resolved``)"""
        got = strip_trailing_slash(request.path)
        if got == normalize_config_path(self.config.path) and _is_websocket_upgrade(request):
            return await self._handle_websocket(request)
        if self.config.token_issue_path and got == normalize_config_path(self.config.token_issue_path):
            return self.http.handle_token_issue(request)
        if got == "/webui/bootstrap":
            return self.http.handle_bootstrap(request)
        if got == "/webui/terminal":
            return self.http.handle_bootstrap(request, terminal_probe=True)
        # /api/* never reaches here: rest_api answers its routes, 405 for
        # mutation paths and 404 for the rest.
        return self.http.static_or_404(request, got)

    def authorize_websocket_handshake(self, request: web.Request) -> tuple[web.Response | None, bool]:
        """Check the upgrade's token; ``(rejection, trusted_webui)``. (nanobot gateway_endpoint.py:83)

        A one-time token from ``/webui/bootstrap`` (audience ``webui``)
        marks the connection as a trusted web UI connection, required for
        ``webui_request`` and ``set_sidebar_state``. A static ``token`` also
        opens the socket; with ``websocketRequiresToken`` false no token is
        needed.
        """
        supplied = request.query.get("token")
        static_token = self.config.token.strip()
        if static_token:
            if supplied and hmac.compare_digest(supplied, static_token):
                return None, False
            if supplied:
                audience = self.tokens.take_issued_token_audience(supplied)
                if audience is not None:
                    return None, audience == "webui"
            return http_error(401, "Unauthorized"), False
        if self.config.websocket_requires_token:
            if supplied:
                audience = self.tokens.take_issued_token_audience(supplied)
                if audience is not None:
                    return None, audience == "webui"
            return http_error(401, "Unauthorized"), False
        if supplied:
            return None, self.tokens.take_issued_token_audience(supplied) == "webui"
        return None, False

    async def _handle_websocket(self, request: web.Request) -> web.StreamResponse:
        terminal_version = request.query.get("terminal_protocol")
        if terminal_version is not None and (
            terminal_version != "1" or request.query.get("terminal_instance") != self.tokens.instance_id
        ):
            return http_error(409, "Terminal instance or protocol changed")
        client_id = (request.query.get("client_id") or "").strip()[:128]
        if not self.is_allowed(client_id):
            return http_error(403, "Forbidden")
        rejection, trusted = self.authorize_websocket_handshake(request)
        if rejection is not None:
            return rejection
        ws = web.WebSocketResponse(
            max_msg_size=self.config.max_message_bytes,
            heartbeat=self.config.ping_interval_s,
        )
        await ws.prepare(request)
        if not client_id:
            client_id = f"anon-{uuid.uuid4().hex[:12]}"
        connection = WebUIConnection(ws=ws, client_id=client_id, remote=_peer_address(request), trusted_webui=trusted)
        await self._connection_loop(connection, terminal=terminal_version == "1")
        return ws

    async def _connection_loop(self, connection: WebUIConnection, *, terminal: bool = False) -> None:
        """(nanobot runtime.py:840)"""
        from nanoclaw.webui.client_contract import gateway_identity

        default_chat_id = str(uuid.uuid4())
        ready: dict[str, Any] = {"event": "ready", "chat_id": default_chat_id, "client_id": connection.client_id}
        if terminal:
            ready["terminal"] = gateway_identity(self.tokens.instance_id)
        try:
            await connection.ws.send_str(json.dumps(ready, ensure_ascii=False))
            self._connections.add(connection)
            self._conn_default[connection] = default_chat_id
            self.attach(connection, default_chat_id)
            await self.hydrate(default_chat_id)
            async for frame in connection.ws:
                if frame.type == WSMsgType.TEXT:
                    raw = cast(str, frame.data)
                elif frame.type == WSMsgType.BINARY:
                    try:
                        raw = cast(bytes, frame.data).decode("utf-8")
                    except UnicodeDecodeError:
                        self.logger.warning("ignoring non-utf8 binary frame")
                        continue
                else:
                    break
                envelope = _parse_envelope(raw)
                if envelope is not None:
                    if connection.closed:
                        break
                    await self.commands.dispatch(connection, connection.client_id, envelope)
                    continue
                content = _parse_inbound_payload(raw)
                if content is None:
                    continue
                # Authenticated at the handshake; a legacy frame goes to the
                # connection's default chat.
                await self.dispatch_inbound(
                    sender_id=connection.client_id,
                    chat_id=default_chat_id,
                    content=content,
                    metadata={"remote": connection.remote},
                )
        except Exception as e:
            self.logger.debug("connection ended: {}", e)
        finally:
            await self._cleanup_connection(connection)

    # -- lifecycle -------------------------------------------------------------------

    def _ssl_context(self) -> ssl.SSLContext | None:
        """(nanobot runtime.py:566)"""
        cert = self.config.ssl_certfile.strip()
        key = self.config.ssl_keyfile.strip()
        if not cert and not key:
            return None
        if not cert or not key:
            raise ValueError("ssl_certfile and ssl_keyfile must both be set for WSS, or both left empty")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(certfile=cert, keyfile=key)
        return ctx

    def _subscribe_lifecycle(self) -> None:
        self._unsubscribe = [
            self.bus.subscribe(self._on_session_turn_started, SessionTurnStarted),
            self.bus.subscribe(self._on_run_status_changed, TurnRunStatusChanged),
            self.bus.subscribe(self._on_runtime_admitted, TurnRuntimeAdmitted),
            self.bus.subscribe(self._on_turn_completed, TurnCompleted),
        ]

    async def start(self) -> None:
        """Listen until ``stop``. (nanobot runtime.py:640)"""
        ssl_context = self._ssl_context()
        scheme = "wss" if ssl_context else "ws"
        stop_event = asyncio.Event()
        self._stop_event = stop_event
        self._subscribe_lifecycle()
        await self.recorder.initialize()
        runner = web.AppRunner(self.app, access_log=None, handle_signals=False)
        await runner.setup()
        self._runner = runner
        try:
            site = web.TCPSite(runner, self.config.host, self.config.port, ssl_context=ssl_context)
            await site.start()
            server = getattr(site, "_server", None)
            sockets = getattr(server, "sockets", None) or ()
            self.bound_port = sockets[0].getsockname()[1] if sockets else self.config.port
            self._running = True
            self.logger.info(
                "WebSocket server listening on {}://{}:{}{}", scheme, self.config.host, self.bound_port, self.config.path
            )
            if self.config.token_issue_path:
                self.logger.info(
                    "WebSocket token issue route: {}://{}:{}{}",
                    scheme,
                    self.config.host,
                    self.bound_port,
                    self.config.token_issue_path,
                )
            if self.static_dist_path is not None:
                self.logger.info("Web UI served from {}", self.static_dist_path)
            await stop_event.wait()
        finally:
            self._running = False
            for unsubscribe in self._unsubscribe:
                unsubscribe()
            self._unsubscribe = []
            for connection in tuple(self._connections):
                await self._close_connection(connection, code=1001, reason="server shutdown")
            await runner.cleanup()
            self._runner = None

    async def stop(self) -> None:
        """(nanobot runtime.py:925)"""
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()
        for connection in tuple(self._connections):
            await self._close_connection(connection, code=1001, reason="server shutdown")
        await self.commands.close()
        for task in tuple(self._background):
            task.cancel()
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        self.tokens.clear()
        self._subs.clear()
        self._conn_chats.clear()
        self._conn_default.clear()

    # -- subscriptions -----------------------------------------------------------------

    def attach(self, connection: WebUIConnection, chat_id: str) -> None:
        """Idempotently subscribe *connection* to *chat_id*."""
        if connection.closed:
            return
        self._subs.setdefault(chat_id, set()).add(connection)
        self._conn_chats.setdefault(connection, set()).add(chat_id)

    def detach(self, connection: WebUIConnection, chat_id: str) -> None:
        chats = self._conn_chats.get(connection)
        if chats is not None:
            chats.discard(chat_id)
            if not chats:
                self._conn_chats.pop(connection, None)
        subscribers = self._subs.get(chat_id)
        if subscribers is not None:
            subscribers.discard(connection)
            if not subscribers:
                self._subs.pop(chat_id, None)

    def subscribers(self, chat_id: str) -> tuple[WebUIConnection, ...]:
        return tuple(self._subs.get(chat_id, ()))

    def webui_connections(self) -> tuple[WebUIConnection, ...]:
        return tuple(c for c in self._connections if c.trusted_webui and not c.closed)

    async def _cleanup_connection(self, connection: WebUIConnection) -> None:
        connection.closing = True
        self._connections.discard(connection)
        if connection.writer is not None and connection.writer is not asyncio.current_task():
            connection.writer.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await connection.writer
        connection.writer = None
        while not connection.queue.empty():
            connection.queue.get_nowait()
        connection.buffered_bytes = 0
        for chat_id in tuple(self._conn_chats.get(connection, ())):
            self.detach(connection, chat_id)
        self._conn_default.pop(connection, None)
        await self.commands.cleanup_connection(connection)

    async def _close_connection(self, connection: WebUIConnection, *, code: int, reason: str) -> None:
        with suppress(Exception):
            async with asyncio.timeout(1.0):
                await connection.ws.close(code=code, message=reason.encode())
        await self._cleanup_connection(connection)

    # -- connection output ---------------------------------------------------------------

    async def send_event(self, connection: WebUIConnection, event: str, **fields: Any) -> None:
        """A control event to one connection. (nanobot runtime.py:542)"""
        await self.send_payload_to(connection, {"event": event, **fields})

    async def send_payload_to(self, connection: WebUIConnection, payload: dict[str, Any]) -> None:
        self._enqueue(connection, json.dumps(payload, ensure_ascii=False))
        await asyncio.sleep(0)

    def _enqueue(self, connection: WebUIConnection, raw: str) -> None:
        """Queue one frame without blocking other clients. (nanobot ``_safe_send_to``, runtime.py:1100)"""
        if connection.closed:
            return
        size = len(raw.encode("utf-8"))
        if connection.queue.full() or connection.buffered_bytes + size > _OUTBOUND_QUEUE_MAX_BYTES:
            self.logger.warning(
                "disconnecting slow WebSocket connection: outbound queue full ({} frames, {} bytes)",
                connection.queue.qsize(),
                connection.buffered_bytes,
            )
            self._retire(connection, code=1013, reason="outbound queue full")
            return
        connection.queue.put_nowait((raw, size))
        connection.buffered_bytes += size
        if connection.writer is None or connection.writer.done():
            connection.writer = asyncio.create_task(self._drain(connection), name="websocket-outbound")

    async def _drain(self, connection: WebUIConnection) -> None:
        """(nanobot ``_drain_connection_outbound``, runtime.py:992)"""
        while not connection.closed:
            try:
                raw, size = connection.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                async with asyncio.timeout(_OUTBOUND_SEND_TIMEOUT_S):
                    await connection.ws.send_str(raw)
            except TimeoutError:
                self.logger.warning("connection send timed out")
                self._retire(connection, code=1013, reason="outbound send timeout")
                return
            except (ConnectionError, RuntimeError):
                self.logger.warning("connection gone")
                self._retire(connection, code=1000, reason="")
                return
            finally:
                connection.buffered_bytes = max(0, connection.buffered_bytes - size)

    def _retire(self, connection: WebUIConnection, *, code: int, reason: str) -> None:
        if connection.closing:
            return
        connection.closing = True
        task = asyncio.create_task(self._close_connection(connection, code=code, reason=reason))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    @property
    def has_pending_delivery(self) -> bool:
        return any(c.buffered_bytes or not c.queue.empty() for c in self._connections)

    def _targets(self, chat_id: str, audience: str) -> tuple[WebUIConnection, ...]:
        if audience == "all":
            return tuple(self._conn_chats)
        if audience == "webui":
            return self.webui_connections()
        return self.subscribers(chat_id)

    async def broadcast(self, chat_id: str, payload: dict[str, Any], *, audience: str = "chat") -> None:
        targets = self._targets(chat_id, audience)
        if not targets:
            return
        raw = json.dumps(payload, ensure_ascii=False)
        for connection in targets:
            self._enqueue(connection, raw)
        await asyncio.sleep(0)

    async def send_session_updated(self, chat_id: str, *, scope: str | None = None) -> None:
        """Tell every connection a sidebar row changed. (nanobot runtime.py:1525)"""
        body: dict[str, Any] = {"event": "session_updated", "chat_id": chat_id}
        if scope:
            body["scope"] = scope
        await self.broadcast(chat_id, body, audience="all")

    # -- inbound ----------------------------------------------------------------------

    async def dispatch_inbound(self, *, sender_id: str, chat_id: str, content: str, metadata: dict[str, Any]) -> bool:
        return await self._handle_message(sender_id=sender_id, chat_id=chat_id, content=content, metadata=metadata)

    def attach_fields(self, chat_id: str) -> dict[str, Any]:
        """Session facts sent with ``attached``. (nanobot session_projection.py:51)"""
        from nanoclaw.providers.base import LLMUsage
        from nanoclaw.session.model_selection import model_preset_from_metadata

        key = webui_session_key(chat_id)
        cached = self.sessions.get_cached(key)
        if cached is not None:
            metadata: Any = cached.metadata
        else:
            header = self.sessions.read_session_metadata(key)
            metadata = header.get("metadata") if header else None
        fields: dict[str, Any] = {}
        try:
            fields["model_preset"] = model_preset_from_metadata(metadata)
        except ValueError:
            fields["model_preset"] = None
        if not isinstance(metadata, dict):
            return fields
        recovery_state = recovery_state_from_metadata(metadata)
        if recovery_state is not None:
            fields["recovery_state"] = recovery_state
        usage = LLMUsage.from_dict(metadata.get("_last_usage"))
        if usage is not None:
            fields["usage"] = usage.to_turn_dict()
        return fields

    def workspace_scope_payload(self) -> dict[str, Any]:
        return self.agent.workspace_scope.payload()

    async def hydrate(self, chat_id: str) -> None:
        """Replay a running turn to a chat's subscribers. (nanobot outbound_projection.py:126)"""
        started_at = self.turns.wall_started_at(chat_id)
        if started_at is None:
            return
        await self.broadcast(
            chat_id, encode_goal_status(chat_id, "running", started_at=started_at, turn_id=self.turns.turn_id(chat_id))
        )

    # -- outbound ---------------------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        """Project one bus message and deliver its frames."""
        self.projector.show_reasoning = self.show_reasoning
        self.projector.send_progress = self.send_progress
        self.projector.send_tool_hints = self.send_tool_hints
        for frame in self.projector.project(msg):
            await self._deliver_frame(msg.chat_id, frame)

    async def _deliver_frame(self, chat_id: str, frame: WireFrame) -> None:
        """Persist the frame's record, update the turn registry, fan out.
        (nanobot runtime.py ``_persist_turn_transcript_event`` / ``send_payload``)"""
        metadata = frame.metadata or {}
        if frame.record is not None:
            record = frame.record
            if frame.turn_complete and metadata.get("webui") is True and self.turns.transcript_persistence_failed(
                chat_id, frame.turn_owner
            ):
                record = {**record, WEBUI_TRANSCRIPT_INCOMPLETE_KEY: True}
            persisted = await self.recorder.append(chat_id, record, metadata=metadata)
            if not persisted and frame.phase in {"answer", "complete"} and metadata.get("webui") is True:
                owner = metadata.get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
                self.turns.mark_transcript_persistence_failed(chat_id, owner if isinstance(owner, str) else None)
            if frame.turn_complete and persisted:
                self.turns.clear_if_current(chat_id, frame.turn_owner)
        if frame.release_turn_owner:
            self.turns.clear_if_current(chat_id, frame.turn_owner, preserve_persistence_failure=True)
        await self.broadcast(chat_id, frame.payload, audience=frame.audience)

    # -- turn lifecycle (nanobot WebuiTurnCoordinator) -----------------------------------

    def _lifecycle_owner(self, session_key: str, metadata: dict[str, Any], *, mint: bool) -> str | None:
        owner = metadata.get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
        if isinstance(owner, str) and owner:
            return owner
        if mint:
            return self._minted_owners.setdefault(session_key, uuid.uuid4().hex)
        return self._minted_owners.get(session_key)

    async def _publish_lifecycle(self, msg: OutboundMessage | None, owner: str | None) -> None:
        if msg is None:
            return
        if owner is not None:
            msg.metadata = {**msg.metadata, WEBSOCKET_TURN_OWNER_METADATA_KEY: owner}
        await self.bus.publish_outbound(msg)

    async def _on_session_turn_started(self, event: SessionTurnStarted) -> None:
        """Mark a session opened by the web UI and keep its workspace scope
        for the sidebar. (nanobot webui_turns.py:704 and ``persist_scope``,
        which nanobot runs before dispatch)"""
        ctx = event.context
        if ctx.channel != self.name:
            return
        session = self.sessions.get_cached(ctx.session_key)
        if session is None:
            return
        scope = ctx.metadata.get(WORKSPACE_SCOPE_METADATA_KEY)
        if isinstance(scope, dict):
            session.metadata[WORKSPACE_SCOPE_METADATA_KEY] = dict(scope)
        if ctx.metadata.get(WEBUI_SESSION_METADATA_KEY) is True:
            session.metadata[WEBUI_SESSION_METADATA_KEY] = True

    async def _on_run_status_changed(self, event: TurnRunStatusChanged) -> None:
        """``goal_status`` running / idle; a running turn becomes the chat's
        active turn. (nanobot ``publish_turn_run_status``, webui_turns.py:432)"""
        ctx = event.context
        if ctx.channel != self.name:
            return
        chat_id = str(ctx.chat_id)
        running = event.status == "running"
        started_at = (
            float(event.started_at)
            if isinstance(event.started_at, int | float) and event.started_at > 0
            else time.time()
        )
        owner = self._lifecycle_owner(ctx.session_key, ctx.metadata, mint=running)
        if running and owner is not None:
            turn_id = ctx.metadata.get(WEBUI_TURN_METADATA_KEY)
            self.turns.mark_running(chat_id, owner, turn_id if isinstance(turn_id, str) and turn_id else None, started_at)
        if not running:
            self._minted_owners.pop(ctx.session_key, None)
        await self._publish_lifecycle(lifecycle_outbound(event, now=lambda: started_at), owner)

    async def _on_runtime_admitted(self, event: TurnRuntimeAdmitted) -> None:
        if event.context.channel != self.name:
            return
        await self._publish_lifecycle(lifecycle_outbound(event), None)

    async def _on_turn_completed(self, event: TurnCompleted) -> None:
        """``turn_end``, then the title. (nanobot webui_turns.py:740)"""
        ctx = event.context
        if ctx.channel != self.name:
            return
        owner = self._lifecycle_owner(ctx.session_key, ctx.metadata, mint=False)
        await self._publish_lifecycle(lifecycle_outbound(event), owner)
        self._schedule_title_update(event)

    def _schedule_title_update(self, event: TurnCompleted) -> None:
        """(nanobot ``_schedule_title_update_from_event``, webui_turns.py:853)"""
        runtime = event.runtime if isinstance(event.runtime, LLMRuntime) else None
        ctx = event.context
        if ctx.metadata.get("webui") is not True or runtime is None:
            return
        # None: the provider's default model (a provider given without a model).
        model = cast(str, runtime.model)

        async def generate() -> None:
            try:
                generated = await maybe_generate_webui_title_after_turn(
                    channel=ctx.channel,
                    chat_id=ctx.chat_id,
                    metadata=ctx.metadata,
                    sessions=self.sessions,
                    session_key=ctx.session_key,
                    provider=runtime.provider,
                    model=model,
                )
            except Exception:
                self.logger.opt(exception=True).debug("title generation failed for {}", ctx.session_key)
                return
            if generated:
                await self.bus.publish_outbound(
                    outbound_message_for_event(
                        channel=ctx.channel,
                        chat_id=ctx.chat_id,
                        event=SessionUpdatedEvent(scope="metadata"),
                        metadata=ctx.metadata,
                    )
                )

        task = asyncio.create_task(generate(), name="webui-title")
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def wait_for_background(self) -> None:
        """Wait for scheduled title generation (tests)."""
        while self._background:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)

    # -- mutations -------------------------------------------------------------------------

    async def _mutation_session_delete(self, payload: dict[str, Any]) -> web.Response:
        """``session.delete``: stop the chat's work, delete its session and
        history. (nanobot ws_http.py:1367 ``_handle_session_delete``, without
        the automation check: NanoClaw keeps no session-bound automations list)"""
        raw_key = payload.get("key")
        if not isinstance(raw_key, str) or not raw_key.strip():
            return mutation_error(400, "missing session key")
        key = raw_key
        if not 1 <= len(key) <= 128 or any(ch not in _SESSION_KEY_CHARS for ch in key):
            return mutation_error(400, "invalid session key")
        if not is_webui_session_key(key):
            return mutation_error(404, "session not found")
        await self.agent.cancel_active_tasks(key)
        self.agent.discard_session_file_state(key)
        deleted = delete_webui_session(self.sessions, self.history, key)
        chat_id = webui_chat_id(key)
        if chat_id is not None:
            self.turns.clear(chat_id)
            self.projector.forget_chat(chat_id)
        return mutation_json({"deleted": deleted})
