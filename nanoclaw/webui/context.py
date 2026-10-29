"""What the web UI's HTTP handlers need from the running gateway.

nanobot passes its websocket channel object (``WebSocketChannel``) to the
REST handlers in ws_http.py and reaches the session manager, the WebUI
history store, the agent loop and the auth helpers through it. NanoClaw
gives the REST module (``nanoclaw/webui/rest_api.py``) one small context
object instead, built by ``WebSocketChannel`` when it builds its aiohttp
application.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import web

    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.channels.websocket import WebSocketChannel
    from nanoclaw.config import Config
    from nanoclaw.cron.service import CronService
    from nanoclaw.session.manager import SessionManager
    from nanoclaw.webui.history import WebUIHistoryStore

# A web UI mutation (a socket ``webui_request`` action, nanobot
# ws_http.py ``_WEBUI_MUTATION_PATHS``): takes the request's ``payload`` and
# answers like nanobot's internal HTTP route does: a 2xx JSON body becomes
# ``webui_response {ok: true, result: <body>}``, any other status
# ``{ok: false, error: {status, message: <body text>}}``.
WebUIMutation = Callable[[dict[str, Any]], Awaitable["web.Response"]]


@dataclass
class WebUIContext:
    """Shared state for the web UI REST routes and socket mutations.

    ``check_auth(request)`` is True when the request carries a valid API
    token (``Authorization: Bearer <api_token>`` from ``/webui/bootstrap``,
    or ``?token=``), as nanobot's ``check_api_token``; a handler answers
    ``401 Unauthorized`` otherwise.

    ``channel`` is the running ``WebSocketChannel`` (None in unit tests). Its
    turn registry answers "is a turn running in this chat":
    ``channel.turns.wall_started_at(chat_id)`` / ``channel.turns.turn_id(chat_id)``
    (nanobot ``websocket_turn_wall_started_at`` / ``websocket_turn_id``);
    ``await channel.send_session_updated(chat_id, scope=...)`` tells every
    open connection to refresh a sidebar row.

    ``mutations`` maps ``webui_request`` actions to handlers. The channel
    registers ``session.delete``; the REST module registers
    ``sidebar.update`` (payload ``{"state": {...}}``, answers the saved
    state) and ``automation.enable`` / ``disable`` / ``delete`` / ``run``.
    An action with no handler answers nanobot's
    ``404 unknown WebUI mutation action``.

    ``cron`` is the gateway's cron service (nanobot ``cron_service``): the
    loop's ``cron_service`` when the channel is built, replaced by the
    gateway's own one (``Gateway``); None lists no automations.
    """

    config: Config
    sessions: SessionManager
    history: WebUIHistoryStore
    loop: AgentLoop | None
    workspace: Path
    check_auth: Callable[[web.Request], bool]
    channel: WebSocketChannel | None = None
    cron: CronService | None = None
    mutations: dict[str, WebUIMutation] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def default_workspace_scope_payload(workspace: Path, *, restrict_to_workspace: bool) -> dict[str, Any]:
    """The workspace scope payload of a chat (nanobot ``WorkspaceScope.payload()``,
    security/workspace_access.py:86) when the project is the workspace.
    With a loop at hand prefer ``loop.workspace_scope.payload()``: the
    gateway's project folder (``--project``), which every web UI chat uses."""
    from nanoclaw.security.workspace_access import default_workspace_scope

    return default_workspace_scope(workspace, restrict_to_workspace).payload()
