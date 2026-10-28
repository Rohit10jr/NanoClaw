"""The chat channel interface (nanobot channels/base.py, slimmed).

A channel listens on its platform (``start``), turns what arrives into
:class:`InboundMessage` on the bus (``_handle_message``) and shows what the
agent publishes (``send``). nanobot's manager calls a different channel
method per outbound event (``send_delta``, ``send_reasoning_delta``,
``send_file_edit_events``, ...); NanoClaw's only channel so far, the web
UI, projects every event itself (nanoclaw/webui/outbound_projection.py), so
``send`` is the one outbound entry point.

Unknown DM senders get a pairing code (nanobot pairing/, layer 24): an
unapproved sender of a direct message (``is_dm=True``) is answered with
``format_pairing_reply(code)`` instead of being published; once the owner
approves the code (``/pairing approve <code>`` or the web UI) ``is_allowed``
lets the sender in.

Left out: ``transcribe_audio``, ``login``, plugin metadata hooks, the
``require_existing_session`` argument of ``_handle_message`` (NanoClaw's
``InboundMessage`` has no such field). ``_handle_message`` returns whether
the message was published (nanobot returns None).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from loguru import logger

from nanoclaw.bus.events import InboundMessage, OutboundMessage
from nanoclaw.bus.queue import MessageBus
from nanoclaw.pairing import (
    PAIRING_CODE_META_KEY,
    format_pairing_reply,
    generate_code,
    is_approved,
    record_delivery,
)
from nanoclaw.pairing.events import PairingChanged


class BaseChannel(ABC):
    """Abstract base class for chat channel implementations. (nanobot base.py:24)"""

    name: str = "base"
    display_name: str = "Base"
    send_progress: bool = True
    send_tool_hints: bool = True
    show_reasoning: bool = True

    def __init__(self, config: Any, bus: MessageBus) -> None:
        self.config = config
        self.logger = logger.bind(channel=self.name)
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Start listening; a long-running task that publishes inbound messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Deliver one outbound message; raise on failure so the manager can retry."""

    @property
    def has_pending_delivery(self) -> bool:
        """Whether delivery outlives an awaited ``send``. (nanobot base.py:121)"""
        return False

    def should_retry_send_error(self, error: Exception) -> bool:
        """Whether the manager should retry a failed ``send``. (nanobot base.py:136)"""
        return True

    @property
    def supports_streaming(self) -> bool:
        """True when the config enables streaming. (nanobot base.py:256; every
        NanoClaw channel streams through ``send``.)"""
        cfg = self.config
        if isinstance(cfg, dict):
            return bool(cast(dict[str, Any], cfg).get("streaming", False))
        return bool(getattr(cfg, "streaming", False))

    def is_allowed(self, sender_id: str) -> bool:
        """Check sender permission: star > allowlist > pairing store > deny. (nanobot base.py:261)"""
        if isinstance(self.config, dict):
            data = cast(dict[str, Any], self.config)
            allow_list: Any = data.get("allow_from") or data.get("allowFrom") or []
        else:
            allow_list = getattr(self.config, "allow_from", None) or []
        if "*" in allow_list:
            return True
        # allowFrom entries are opaque tokens — must match exactly.
        if str(sender_id) in allow_list:
            return True
        if is_approved(self.name, str(sender_id)):
            return True
        return False

    def _pairing_changed(self) -> None:
        self.bus.publish_nowait(PairingChanged())

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        is_dm: bool = False,
        authorization_id: str | None = None,
        session_generation: str | None = None,
    ) -> bool:
        """Check the authorization subject and publish the message; False when
        not published. (nanobot base.py:282)

        ``sender_id`` is the identity recorded on the inbound message.  Channels
        where access is scoped to another entity (for example, a group or room)
        can pass that entity as ``authorization_id`` without changing the
        sender's identity.  When omitted, authorization remains sender-based.
        An unapproved DM sender (``is_dm``) is sent a pairing code instead.
        """
        permission_id = authorization_id if authorization_id is not None else sender_id
        if not self.is_allowed(permission_id):
            if is_dm:
                try:
                    code = generate_code(self.name, str(sender_id), on_change=self._pairing_changed)
                except OSError:
                    # Transient pairing-store I/O failure: skip the pairing
                    # reply for this message rather than crash the handler.
                    self.logger.warning("Pairing store unavailable; dropping DM from {}", sender_id)
                    return False
                record_delivery(code, "sending", on_change=self._pairing_changed)
                try:
                    await self.send(
                        OutboundMessage(
                            channel=self.name,
                            chat_id=str(chat_id),
                            content=format_pairing_reply(code),
                            metadata={PAIRING_CODE_META_KEY: code},
                        )
                    )
                except Exception:
                    record_delivery(code, "failed", on_change=self._pairing_changed)
                    raise
                record_delivery(code, "sent", on_change=self._pairing_changed)
                self.logger.info(
                    "Sent pairing code {} to sender {} in chat {}",
                    code, sender_id, chat_id,
                )
            else:
                self.logger.warning(
                    "Access denied for sender {}. "
                    "Add them to allowFrom list in config to grant access.",
                    sender_id,
                )
            return False
        meta = metadata or {}
        if self.supports_streaming:
            meta = {**meta, "_wants_stream": True}
        await self.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender_id=str(sender_id),
                chat_id=str(chat_id),
                content=content,
                media=media or [],
                metadata=meta,
                session_key_override=session_key,
                session_generation=session_generation,
            )
        )
        return True

    @property
    def is_running(self) -> bool:
        return self._running
