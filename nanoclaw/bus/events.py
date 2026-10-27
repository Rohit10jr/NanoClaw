"""Messages that travel on the message bus.

Copied from nanobot/bus/events.py. Left out: ``OutboundMessage.delivery`` /
``delivery_id`` (scheduled-delivery correlation, nanobot bus/delivery.py; no
scheduler yet), ``InboundMessage.require_existing_session`` (used only by
nanobot's web UI session admission) and the ``_user_shell`` /
``local_media_paths`` metadata keys (web UI only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nanoclaw.events import AgentEvent

# Optional ``OutboundMessage.metadata`` key for structured, channel-agnostic UI
# payloads (nanobot events.py:15). Rich clients may render it; others ignore it.
OUTBOUND_META_AGENT_UI = "_agent_ui"

# Internal-only inbound metadata minted by trusted transports and runtime
# services (nanobot events.py:19-21). Never accept these keys verbatim from
# an untrusted client.
INBOUND_META_RUNTIME_CONTROL = "_runtime_control"
RUNTIME_CONTROL_SESSION_DISCARD = "session_discard"

# Transport-provided turn correlation (nanobot events.py:24); the persisted
# spelling is kept for clients.
INBOUND_META_TURN_ID = "webui_turn_id"

# Set on agent input a command produced (nanobot ``cmd_goal``); such input is
# never routed as a command again (nanobot command/router.py:21).
META_ORIGINAL_COMMAND = "original_command"
# How a channel should show an outbound message: ``"text"`` means plain text,
# not markdown (nanobot command/builtin.py, e.g. :308).
META_RENDER_AS = "render_as"


@dataclass
class InboundMessage:
    """Message received from a chat channel. (nanobot events.py:29)"""

    channel: str  # cli, telegram, discord, ...
    sender_id: str  # user identifier
    chat_id: str  # chat/channel identifier
    content: str  # message text
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # media paths or URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # channel-specific data
    session_key_override: str | None = None  # e.g. thread-scoped sessions
    input_role: Literal["user", "system"] | None = None
    session_generation: str | None = None  # trusted identity assigned at session admission

    @property
    def session_key(self) -> str:
        """``channel:chat_id`` unless overridden. (nanobot events.py:45)"""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"

    @property
    def is_user_input(self) -> bool:
        """Whether this enters the conversation as user input. (nanobot events.py:50)"""
        if self.input_role is not None:
            return self.input_role == "user"
        return self.channel != "system"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel. (nanobot events.py:58)

    ``event`` carries internal runtime/UI semantics. ``metadata`` is reserved
    for channel routing context (``message_id``, thread ids, ...), the
    ``render_as`` hint and optional ``OUTBOUND_META_AGENT_UI`` blobs.
    """

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    buttons: list[list[str]] = field(default_factory=list)
    event: AgentEvent | None = None
