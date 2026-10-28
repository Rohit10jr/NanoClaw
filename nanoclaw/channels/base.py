"""The chat channel interface (nanobot channels/base.py, slimmed).

A channel listens on its platform (``start``), turns what arrives into
:class:`InboundMessage` on the bus (``_handle_message``) and shows what the
agent publishes (``send``). nanobot's manager calls a different channel
method per outbound event (``send_delta``, ``send_reasoning_delta``,
``send_file_edit_events``, ...); NanoClaw's only channel so far, the web
UI, projects every event itself (nanoclaw/webui/outbound_projection.py), so
``send`` is the one outbound entry point.

Left out: pairing codes for unknown DM senders (nanobot pairing/),
``transcribe_audio``, ``login``, plugin metadata hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from loguru import logger

from nanoclaw.bus.events import InboundMessage, OutboundMessage
from nanoclaw.bus.queue import MessageBus


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
        """``"*"`` allows everyone, else ``allowFrom`` must list the sender. (nanobot base.py:267)"""
        if isinstance(self.config, dict):
            data = cast(dict[str, Any], self.config)
            allow_list: Any = data.get("allow_from") or data.get("allowFrom") or []
        else:
            allow_list = getattr(self.config, "allow_from", None) or []
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> bool:
        """Check the sender and publish the message; False when denied. (nanobot base.py:288)"""
        if not self.is_allowed(sender_id):
            self.logger.warning(
                "Access denied for sender {}. Add them to allowFrom list in config to grant access.",
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
            )
        )
        return True

    @property
    def is_running(self) -> bool:
        return self._running
