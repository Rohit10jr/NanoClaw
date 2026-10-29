"""Typed web UI socket commands (nanobot webui/inbound_commands.py, a subset).

The browser sends JSON envelopes with a string ``type``:

- ``message``: validated (chat id, access, content, size, attachments),
  deduplicated by ``(chat_id, turn_id)`` for 10 minutes, recorded in the
  chat's history (the user row), published to the bus as
  ``InboundMessage(channel="websocket", ...)``, broadcast as ``user_message``
  to the chat's other connections and acknowledged with
  ``message_accepted``; a rejection is an ``error`` with ``chat_id`` and
  ``turn_id``;
- ``new_chat``: a fresh chat id, attached (``attached`` +
  ``session_updated`` with the workspace scope);
- ``attach``: subscribe to a chat (``attached``, then ``goal_status
  running`` when a turn is in flight);
- ``set_sidebar_state``: saved through the ``sidebar.update`` mutation and
  broadcast as ``sidebar_state_updated``;
- ``webui_request``: one mutation RPC (``session.delete``,
  ``sidebar.update``; anything else answers nanobot's 404), idempotent per
  ``request_id`` for 5 minutes, only on bootstrap-authenticated
  connections.

Not supported (answered as an unknown type): temporary chats
(``new_temporary_chat``, ``discard_temporary_chat``), ``fork_chat``,
``set_workspace_scope`` (NanoClaw has one project folder) and
``transcribe_audio``. Messages may not carry attachments or ``!`` shell
commands; ``cli_apps``, ``mcp_presets``, ``session_mentions``,
``quoted_context`` and ``intent`` are ignored.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from aiohttp import web
from loguru import logger

from nanoclaw.command.builtin import builtin_command_starts_agent_turn
from nanoclaw.webui.ingress_policy import DEFAULT_WEBUI_INGRESS_POLICY, WebUIIngressPolicy
from nanoclaw.webui.metadata import WEBSOCKET_TURN_OWNER_METADATA_KEY
from nanoclaw.webui.session_identity import is_valid_webui_chat_id
from nanoclaw.webui.session_list_index import WORKSPACE_SCOPE_METADATA_KEY

if TYPE_CHECKING:
    from nanoclaw.channels.websocket.runtime import WebSocketChannel, WebUIConnection

_WEBUI_REQUEST_CACHE_TTL_S = 5 * 60.0
_WEBUI_REQUEST_CACHE_MAX = 256
_ACCEPTED_MESSAGE_TTL_S = 600
_ACCEPTED_MESSAGES_MAX = 1024


@dataclass(frozen=True)
class WebUIRequestResult:
    result: Any = None
    status: int | None = None
    message: str | None = None


@dataclass
class WebUIRequestOperation:
    action: str
    payload_digest: bytes
    task: asyncio.Task[WebUIRequestResult]
    completed_at: float | None = None


class WebUICommandRouter:
    """Command semantics; the channel owns the raw connections. (nanobot inbound_commands.py:142)"""

    def __init__(self, channel: WebSocketChannel, *, ingress: WebUIIngressPolicy = DEFAULT_WEBUI_INGRESS_POLICY) -> None:
        self.channel = channel
        self.ingress = ingress
        self.request_tasks: dict[tuple[WebUIConnection, str], asyncio.Task[None]] = {}
        self.request_operations: dict[str, WebUIRequestOperation] = {}
        self.request_locks: dict[WebUIConnection, asyncio.Lock] = {}
        self._accepted_messages: dict[tuple[str, str], float] = {}
        self._message_locks: dict[tuple[str, str], tuple[asyncio.Lock, int]] = {}

    async def _send(self, connection: WebUIConnection, event: str, **fields: Any) -> None:
        await self.channel.send_event(connection, event, **fields)

    # -- dispatch -----------------------------------------------------------------

    async def dispatch(self, connection: WebUIConnection, client_id: str, envelope: dict[str, Any]) -> None:
        """Execute one typed command. (nanobot inbound_commands.py:296)"""
        command_type = envelope.get("type")
        channel = self.channel
        if command_type == "webui_request":
            await self.start_webui_request(connection, envelope)
            return
        if command_type == "new_chat":
            new_id = str(uuid.uuid4())
            channel.attach(connection, new_id)
            await self._send(connection, "attached", chat_id=new_id, **channel.attach_fields(new_id))
            await self._send(
                connection,
                "session_updated",
                chat_id=new_id,
                scope="metadata",
                workspace_scope=channel.workspace_scope_payload(),
            )
            await channel.hydrate(new_id)
            return
        if command_type == "attach":
            chat_id = envelope.get("chat_id")
            if not is_valid_webui_chat_id(chat_id):
                await self._send(connection, "error", detail="invalid chat_id")
                return
            channel.attach(connection, chat_id)
            await self._send(connection, "attached", chat_id=chat_id, **channel.attach_fields(chat_id))
            await channel.hydrate(chat_id)
            return
        if command_type == "set_sidebar_state":
            if not connection.trusted_webui:
                await self._send(connection, "error", detail="access_denied")
                return
            state = envelope.get("state")
            if not isinstance(state, dict):
                await self._send(connection, "error", detail="invalid_sidebar_state")
                return
            result = await self._run_mutation("sidebar.update", {"state": state})
            if result.status is not None or not isinstance(result.result, dict):
                await self._send(connection, "error", detail="invalid_sidebar_state")
                return
            await self.broadcast_webui_event("sidebar_state_updated", state=result.result)
            return
        if command_type == "message":
            await self._dispatch_message(connection, client_id, envelope)
            return
        await self._send(connection, "error", detail=f"unknown type: {command_type!r}")

    async def broadcast_webui_event(self, event: str, **fields: Any) -> None:
        for connection in self.channel.webui_connections():
            await self._send(connection, event, **fields)

    async def broadcast_user_message(
        self, origin: WebUIConnection, chat_id: str, text: str, *, turn_id: str | None, starts_turn: bool
    ) -> None:
        """The accepted user message to the chat's other tabs. (nanobot inbound_commands.py:235)"""
        body: dict[str, Any] = {"event": "user_message", "chat_id": chat_id, "text": text, "starts_turn": starts_turn}
        if turn_id is not None:
            body["turn_id"] = turn_id
        turns = self.channel.turns
        active_turn_id = turns.turn_id(chat_id)
        if active_turn_id is not None:
            body["active_turn_id"] = active_turn_id
        started_at = turns.wall_started_at(chat_id)
        if active_turn_id is not None and started_at is not None:
            body["started_at"] = started_at
        for connection in self.channel.subscribers(chat_id):
            if connection is not origin:
                await self.channel.send_payload_to(connection, body)

    # -- message --------------------------------------------------------------------

    async def _dispatch_message(self, connection: WebUIConnection, client_id: str, envelope: dict[str, Any]) -> None:
        """Serialize retries of one ``(chat_id, turn_id)``. (nanobot inbound_commands.py:493)"""
        chat_id, turn_id = envelope.get("chat_id"), envelope.get("turn_id")
        if not isinstance(chat_id, str) or not isinstance(turn_id, str) or not turn_id:
            await self._dispatch_message_once(connection, client_id, envelope)
            return
        key = (chat_id, turn_id)
        lock, users = self._message_locks.get(key, (asyncio.Lock(), 0))
        self._message_locks[key] = (lock, users + 1)
        try:
            async with lock:
                await self._dispatch_message_once(connection, client_id, envelope)
        finally:
            _, users = self._message_locks[key]
            if users == 1:
                del self._message_locks[key]
            else:
                self._message_locks[key] = (lock, users - 1)

    async def _dispatch_message_once(
        self, connection: WebUIConnection, client_id: str, envelope: dict[str, Any]
    ) -> None:
        """(nanobot inbound_commands.py:511)"""
        channel = self.channel
        chat_id = envelope.get("chat_id")
        content = envelope.get("content")
        if not is_valid_webui_chat_id(chat_id):
            await self._send(connection, "error", detail="invalid chat_id")
            return
        raw_turn_id = envelope.get("turn_id")
        turn_id = raw_turn_id if isinstance(raw_turn_id, str) and raw_turn_id else None
        rejection_fields: dict[str, Any] = {"chat_id": chat_id, **({"turn_id": turn_id} if turn_id else {})}
        if not channel.is_allowed(client_id):
            await self._send(connection, "error", detail="access_denied", **rejection_fields)
            return
        if not isinstance(content, str):
            await self._send(connection, "error", detail="missing content", **rejection_fields)
            return
        message_rejection = self.ingress.validate_text(content)
        if message_rejection is not None:
            await self._send(connection, "error", detail="message_rejected", reason=message_rejection, **rejection_fields)
            return

        # A client may retry a turn after a lost ACK: never dispatch it twice.
        now = time.monotonic()
        self._accepted_messages = {k: exp for k, exp in self._accepted_messages.items() if exp > now}
        if turn_id and (chat_id, turn_id) in self._accepted_messages:
            await self._send(connection, "message_accepted", chat_id=chat_id, turn_id=turn_id, starts_turn=False)
            return

        raw_media = envelope.get("media", [])
        if not isinstance(raw_media, list):
            await self._send(connection, "error", detail="attachment_rejected", reason="malformed", **rejection_fields)
            return
        if raw_media:
            # NanoClaw has no upload store (nanobot resolves references and paths here).
            await self._send(
                connection,
                "error",
                detail="attachment_rejected",
                reason="Attachments are not supported",
                **rejection_fields,
            )
            return
        if not content.strip():
            await self._send(connection, "error", detail="missing content", **rejection_fields)
            return

        channel.attach(connection, chat_id)
        await channel.hydrate(chat_id)

        metadata: dict[str, Any] = {"remote": connection.remote}
        is_webui = envelope.get("webui") is True
        if is_webui:
            metadata["webui"] = True
            metadata.update(channel.recorder.client_turn_metadata(envelope.get("turn_id")))
        # The chat's project scope (nanobot resolves a per-chat scope; NanoClaw
        # has the gateway's one project folder).
        metadata[WORKSPACE_SCOPE_METADATA_KEY] = channel.agent.workspace_scope.metadata()
        queued_owner = None
        if is_webui and builtin_command_starts_agent_turn(content):
            queued_owner = channel.turns.register_queued_if_idle(chat_id, turn_id)
        if is_webui:
            # nanobot's WebuiTurnRoutePolicy gives every web UI turn a
            # server-owned lifecycle owner (the queued one when registered).
            metadata[WEBSOCKET_TURN_OWNER_METADATA_KEY] = queued_owner or uuid.uuid4().hex

        accepted = False
        try:
            if is_webui:
                await channel.recorder.append_user_message(chat_id, content, metadata=metadata)
            accepted = await channel.dispatch_inbound(sender_id=client_id, chat_id=chat_id, content=content, metadata=metadata)
            if accepted and turn_id:
                self._accepted_messages[(chat_id, turn_id)] = time.monotonic() + _ACCEPTED_MESSAGE_TTL_S
                if len(self._accepted_messages) > _ACCEPTED_MESSAGES_MAX:
                    del self._accepted_messages[next(iter(self._accepted_messages))]
        finally:
            if not accepted and queued_owner is not None:
                channel.turns.clear_if_current(chat_id, queued_owner)
        if not accepted:
            await self._send(connection, "error", detail="access_denied", **rejection_fields)
            return

        if is_webui:
            await self.broadcast_user_message(
                connection, chat_id, content, turn_id=turn_id, starts_turn=queued_owner is not None
            )
        if is_webui and turn_id:
            active_turn_id = channel.turns.turn_id(chat_id)
            started_at = channel.turns.wall_started_at(chat_id)
            await self._send(
                connection,
                "message_accepted",
                chat_id=chat_id,
                turn_id=turn_id,
                starts_turn=queued_owner is not None,
                **({"active_turn_id": active_turn_id} if active_turn_id is not None else {}),
                **({"started_at": started_at} if active_turn_id is not None and started_at is not None else {}),
            )

    # -- webui_request ----------------------------------------------------------------

    async def start_webui_request(self, connection: WebUIConnection, envelope: dict[str, Any]) -> None:
        """(nanobot inbound_commands.py:896)"""
        request_id = envelope.get("request_id")
        if not isinstance(request_id, str) or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id) is None:
            await self._send(connection, "error", detail="invalid webui request_id")
            return
        if not connection.trusted_webui:
            await self.send_webui_response(connection, request_id, status=403, message="access_denied")
            return
        action = envelope.get("action")
        payload = envelope.get("payload")
        if not isinstance(action, str) or re.fullmatch(r"[a-z][a-z0-9_.]{0,127}", action) is None:
            await self.send_webui_response(connection, request_id, status=400, message="invalid WebUI mutation action")
            return
        if not isinstance(payload, dict):
            await self.send_webui_response(
                connection, request_id, status=400, message="WebUI mutation payload must be an object"
            )
            return
        payload = cast(dict[str, Any], payload)
        payload_digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        self.prune_request_operations()
        operation = self.request_operations.get(request_id)
        is_replay = operation is not None
        if operation is not None and (operation.action != action or operation.payload_digest != payload_digest):
            await self.send_webui_response(
                connection,
                request_id,
                status=409,
                message="request_id was already used for a different WebUI mutation",
            )
            return
        if operation is None:
            operation_task = asyncio.create_task(self.execute_webui_request(connection, action, payload))
            new_operation = WebUIRequestOperation(action=action, payload_digest=payload_digest, task=operation_task)
            operation = new_operation
            self.request_operations[request_id] = new_operation

            def mark_complete(_task: asyncio.Task[WebUIRequestResult]) -> None:
                if self.request_operations.get(request_id) is not new_operation:
                    return
                new_operation.completed_at = time.monotonic()
                self.prune_request_operations()

            operation_task.add_done_callback(mark_complete)

        key = (connection, request_id)
        if key in self.request_tasks:
            return
        self.request_tasks[key] = asyncio.create_task(
            self.deliver_webui_request(connection, request_id, operation.task, sequence=is_replay)
        )

    def prune_request_operations(self) -> None:
        now = time.monotonic()
        for request_id, operation in tuple(self.request_operations.items()):
            if operation.completed_at is not None and now - operation.completed_at >= _WEBUI_REQUEST_CACHE_TTL_S:
                self.request_operations.pop(request_id, None)
        completed = sorted(
            (
                (operation.completed_at, request_id)
                for request_id, operation in self.request_operations.items()
                if operation.completed_at is not None
            ),
            key=lambda item: item[0],
        )
        for _, request_id in completed[:-_WEBUI_REQUEST_CACHE_MAX]:
            self.request_operations.pop(request_id, None)

    def discard_request_lock_if_idle(self, connection: WebUIConnection) -> None:
        if connection.trusted_webui and not connection.closed:
            return
        if any(task_connection is connection for task_connection, _ in self.request_tasks):
            return
        self.request_locks.pop(connection, None)

    async def deliver_webui_request(
        self,
        connection: WebUIConnection,
        request_id: str,
        operation_task: asyncio.Task[WebUIRequestResult],
        *,
        sequence: bool = False,
    ) -> None:
        try:
            if sequence:
                lock = self.request_locks.setdefault(connection, asyncio.Lock())
                async with lock:
                    result = await asyncio.shield(operation_task)
                    await self.send_webui_response(
                        connection, request_id, result=result.result, status=result.status, message=result.message
                    )
                return
            result = await asyncio.shield(operation_task)
            await self.send_webui_response(
                connection, request_id, result=result.result, status=result.status, message=result.message
            )
        finally:
            self.request_tasks.pop((connection, request_id), None)
            self.discard_request_lock_if_idle(connection)

    async def _run_mutation(self, action: str, payload: dict[str, Any]) -> WebUIRequestResult:
        """Run one registered mutation and read its HTTP-style answer.
        (nanobot ``execute_webui_request`` body and ``_webui_mutation_path``)"""
        if action == "session.delete":
            key = payload.get("key")
            if not isinstance(key, str) or not key.strip():
                return WebUIRequestResult(status=400, message="missing session key")
        handler = self.channel.context.mutations.get(action)
        if handler is None:
            return WebUIRequestResult(status=404, message="unknown WebUI mutation action")
        response = await handler(payload)
        status = response.status
        body = (response.body.decode("utf-8", errors="replace") if isinstance(response.body, bytes) else (response.text or "")).strip()
        if 200 <= status < 300:
            try:
                return WebUIRequestResult(result=json.loads(body))
            except json.JSONDecodeError:
                return WebUIRequestResult(status=502, message="WebUI mutation returned an invalid response")
        return WebUIRequestResult(status=status, message=body or response.reason)

    async def execute_webui_request(
        self, connection: WebUIConnection, action: str, payload: dict[str, Any]
    ) -> WebUIRequestResult:
        """(nanobot inbound_commands.py:1144)"""
        try:
            lock = self.request_locks.setdefault(connection, asyncio.Lock())
            async with lock:
                result = await self._run_mutation(action, payload)
                if result.status is None and action == "sidebar.update" and isinstance(result.result, dict):
                    await self.broadcast_webui_event("sidebar_state_updated", state=result.result)
                return result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("WebUI mutation '{}' failed", action)
            return WebUIRequestResult(status=500, message="WebUI mutation failed")

    async def send_webui_response(
        self,
        connection: WebUIConnection,
        request_id: str,
        *,
        result: Any = None,
        status: int | None = None,
        message: str | None = None,
    ) -> None:
        if status is None:
            await self._send(connection, "webui_response", request_id=request_id, ok=True, result=result)
            return
        await self._send(
            connection,
            "webui_response",
            request_id=request_id,
            ok=False,
            error={"status": status, "message": message or "WebUI mutation failed"},
        )

    async def cleanup_connection(self, connection: WebUIConnection) -> None:
        self.discard_request_lock_if_idle(connection)

    async def close(self) -> None:
        """Cancel command work. (nanobot inbound_commands.py:1210)"""
        delivery_tasks = tuple(self.request_tasks.values())
        operation_tasks = tuple(operation.task for operation in self.request_operations.values())
        for task in (*delivery_tasks, *operation_tasks):
            task.cancel()
        if delivery_tasks or operation_tasks:
            await asyncio.gather(*delivery_tasks, *operation_tasks, return_exceptions=True)
        self.request_tasks.clear()
        self.request_locks.clear()
        self.request_operations.clear()


def mutation_json(data: Any, *, status: int = 200) -> web.Response:
    """A mutation answer (what nanobot's internal HTTP route returns)."""
    return web.Response(status=status, text=json.dumps(data, ensure_ascii=False), content_type="application/json")


def mutation_error(status: int, message: str) -> web.Response:
    return web.Response(status=status, text=message, content_type="text/plain")


def mutation_query_value(payload: dict[str, Any], key: str) -> str | None:
    """A mutation payload field as nanobot's handlers read it: the payload
    becomes the internal request's query (ws_http.py:304 ``_request_query``),
    so ``True`` reads ``"true"``, ``None`` ``""``, an object its JSON."""
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)
