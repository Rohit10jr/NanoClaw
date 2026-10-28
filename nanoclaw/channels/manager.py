"""Start the enabled channels and deliver the agent's outbound messages to them.

A slim port of nanobot channels/manager.py ``ChannelManager``: builds the
enabled channels from config (``websocket``, the web UI, and ``telegram``),
starts and stops them, and runs the outbound dispatcher that consumes the bus:

- consecutive stream deltas of one ``(channel, chat_id, stream_id)`` are
  coalesced into one (``_coalesce_stream_deltas``, manager.py:1081);
- a repeated reply to the same source message is suppressed
  (``_should_suppress_outbound``, manager.py:747);
- sends keep FIFO order per ``(channel, chat_id)`` with at most 32 at once
  and 256 queued (manager.py:796), and are retried 1 s / 2 s / 4 s up to
  ``channels.sendMaxRetries`` attempts (``_send_with_retry``, :1150).

nanobot's dispatcher also drops reasoning / progress / tool hints a
channel does not want and ``RetryWaitEvent``, then calls the channel method
for the event (``_send_once``: ``send_delta``, ``send_reasoning_delta``, ...).
The web UI does both in ``WebUIOutboundProjector.project`` (layer 22), so a
channel with ``projects_outbound`` (the default) gets every event through
``send``; for the others (Telegram) the manager applies nanobot's filters
and ``_send_once``. A ``DeliveryRejectedError`` is never retried.
Left out: the scheduled-delivery outbox, restart notices, plugin discovery,
hot reload and channel feature actions.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanoclaw.bus.delivery import DeliveryRejectedError
from nanoclaw.bus.events import OutboundMessage
from nanoclaw.bus.outbound_events import (
    ProgressEvent,
    RetryWaitEvent,
    RuntimeModelUpdatedEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
    replace_outbound_event,
)
from nanoclaw.channels.base import BaseChannel

if TYPE_CHECKING:
    from nanoclaw.agent.loop import AgentLoop
    from nanoclaw.bus.queue import MessageBus
    from nanoclaw.config import Config

# nanobot manager.py:66-71.
_SEND_RETRY_DELAYS = (1, 2, 4)
_OUTBOUND_CONCURRENCY = 32
_OUTBOUND_PENDING_LIMIT = 256
ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE = 1000


class ChannelManager:
    """Manage chat channels and route outbound messages. (nanobot manager.py:89)

    ``agent`` is the gateway's loop; the web UI channel reads its sessions,
    history and model from it. ``channels`` replaces the config-built set
    (tests).
    """

    def __init__(
        self,
        config: Config,
        bus: MessageBus,
        *,
        agent: AgentLoop | None = None,
        config_path: Path | None = None,
        channels: dict[str, BaseChannel] | None = None,
    ) -> None:
        self.config = config
        self.bus = bus
        self.agent = agent
        self.config_path = config_path
        self.channels: dict[str, BaseChannel] = {}
        self._channel_errors: dict[str, str] = {}
        self._channel_tasks: dict[str, asyncio.Task[None]] = {}
        self._dispatch_task: asyncio.Task[None] | None = None
        self._outbound_tasks: dict[asyncio.Task[None], tuple[str, str]] = {}
        self._outbound_tails: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._outbound_pending: list[OutboundMessage] = []
        self._outbound_dispatching = False
        self._outbound_slots = asyncio.Semaphore(_OUTBOUND_PENDING_LIMIT)
        self._outbound_sends = asyncio.Semaphore(_OUTBOUND_CONCURRENCY)
        self._origin_reply_fingerprints: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        if channels is not None:
            self.channels = dict(channels)
        else:
            self._init_channels()

    # -- construction -----------------------------------------------------------

    def _init_channels(self) -> None:
        """Build the enabled channels. (nanobot manager.py:268, without plugins)"""
        self._init_websocket()
        self._init_telegram()
        self._validate_allow_from()

    def _init_telegram(self) -> None:
        tg_config = self.config.channels.telegram
        if not tg_config.enabled:
            return
        try:
            from nanoclaw.channels.telegram import TelegramChannel

            channel: BaseChannel = TelegramChannel(tg_config, self.bus)
        except Exception as exc:
            self._channel_errors["telegram"] = "Channel runtime could not be loaded. Check gateway logs."
            logger.warning("telegram channel not available: {}", exc)
            return
        self._apply_flags(channel, tg_config)
        self.channels[channel.name] = channel
        logger.info("{} channel enabled as {}", channel.display_name, channel.name)

    def _validate_allow_from(self) -> None:
        """(nanobot manager.py:361; nanobot checks for an omitted key, NanoClaw's
        typed config has an empty list then)"""
        for name, ch in self.channels.items():
            if not getattr(ch.config, "allow_from", None):
                logger.info('"{}" has no allowFrom; unapproved users will receive a pairing code', name)

    def _init_websocket(self) -> None:
        ws_config = self.config.channels.websocket
        if not ws_config.enabled:
            return
        try:
            from nanoclaw.channels.websocket import WebSocketChannel

            if self.agent is None:
                raise RuntimeError("the web UI channel needs the agent loop")
            channel = WebSocketChannel(ws_config, self.bus, agent=self.agent, app_config=self.config)
        except Exception as exc:
            self._channel_errors["websocket"] = "Channel runtime could not be loaded. Check gateway logs."
            logger.warning("websocket channel not available: {}", exc)
            return
        self._apply_flags(channel, ws_config)
        self.channels[channel.name] = channel
        logger.info("{} channel enabled as {}", channel.display_name, channel.name)

    def _apply_flags(self, channel: BaseChannel, section: Any) -> None:
        """Per-channel overrides of the ``channels`` defaults. (nanobot manager.py:236-246)"""
        defaults = self.config.channels
        for key in ("send_progress", "send_tool_hints", "show_reasoning", "show_compaction_notices"):
            value = getattr(section, key, None)
            setattr(channel, key, value if isinstance(value, bool) else getattr(defaults, key))

    # -- lifecycle ----------------------------------------------------------------

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """Start a channel and log any exceptions. (nanobot manager.py:406)"""
        self._channel_errors.pop(name, None)
        try:
            await channel.start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._channel_errors[name] = "Channel failed to start. Check gateway logs."
            logger.error("Failed to start channel {}: {}", name, exc)

    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher; returns when every
        channel's ``start`` has returned. (nanobot manager.py:632)"""
        if not self.channels:
            logger.warning("No channels enabled")
            return
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound_loop(), name="nanoclaw-outbound")
        tasks: list[asyncio.Task[None]] = []
        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            task = asyncio.create_task(self._start_channel(name, channel), name=f"nanoclaw-channel-{name}")
            self._channel_tasks[name] = task
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop the dispatcher, then every channel. (nanobot manager.py:708)"""
        logger.info("Stopping all channels...")
        if self._dispatch_task:
            self._dispatch_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._dispatch_task
            self._dispatch_task = None
        await self._cancel_outbound()
        for name, channel in list(self.channels.items()):
            task = self._channel_tasks.pop(name, None)
            try:
                await channel.stop()
                logger.info("Stopped {} channel", name)
            except Exception:
                logger.exception("Error stopping {}", name)
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    # -- outbound -----------------------------------------------------------------

    @staticmethod
    def _fingerprint_content(content: str) -> str:
        normalized = " ".join(content.split())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest() if normalized else ""

    def _remember_origin_reply_fingerprint(self, key: tuple[str, str, str], fingerprint: str) -> None:
        self._origin_reply_fingerprints[key] = fingerprint
        self._origin_reply_fingerprints.move_to_end(key)
        while len(self._origin_reply_fingerprints) > ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE:
            self._origin_reply_fingerprints.popitem(last=False)

    def _should_suppress_outbound(self, msg: OutboundMessage) -> bool:
        """The same text answering the same source message twice. (nanobot manager.py:747)"""
        metadata = msg.metadata or {}
        if isinstance(msg.event, ProgressEvent):
            return False
        fingerprint = self._fingerprint_content(msg.content)
        if not fingerprint:
            return False
        origin_message_id = metadata.get("origin_message_id")
        if isinstance(origin_message_id, str) and origin_message_id:
            key = (msg.channel, msg.chat_id, origin_message_id)
            if self._origin_reply_fingerprints.get(key) == fingerprint:
                self._origin_reply_fingerprints.move_to_end(key)
                return True
            self._remember_origin_reply_fingerprint(key, fingerprint)
        message_id = metadata.get("message_id")
        if isinstance(message_id, str) and message_id:
            self._remember_origin_reply_fingerprint((msg.channel, msg.chat_id, message_id), fingerprint)
        return False

    @property
    def has_pending_delivery(self) -> bool:
        """(nanobot manager.py:776)"""
        return bool(
            self.bus.outbound_size
            or self._outbound_pending
            or self._outbound_dispatching
            or self._outbound_tasks
            or any(channel.has_pending_delivery for channel in self.channels.values())
        )

    async def _cancel_outbound(self) -> None:
        tasks = list(self._outbound_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _queue_outbound(self, channel: BaseChannel, msg: OutboundMessage) -> None:
        """Bounded sends, FIFO per destination. (nanobot manager.py:796)"""
        await self._outbound_slots.acquire()
        key = (msg.channel, msg.chat_id)
        previous = self._outbound_tails.get(key)

        async def send() -> None:
            if previous is not None:
                await asyncio.shield(asyncio.gather(previous, return_exceptions=True))
            async with self._outbound_sends:
                await self._send_with_retry(channel, msg)

        task = asyncio.create_task(send(), name=f"outbound-{msg.channel}-{msg.chat_id}")
        self._outbound_tasks[task] = key
        self._outbound_tails[key] = task

        def finished(done: asyncio.Task[None]) -> None:
            self._outbound_tasks.pop(done, None)
            if self._outbound_tails.get(key) is done:
                self._outbound_tails.pop(key, None)
            self._outbound_slots.release()
            if not done.cancelled() and (error := done.exception()) is not None:
                logger.error("Outbound delivery to {}:{} failed: {}", *key, error)

        task.add_done_callback(finished)

    async def _dispatch_outbound_loop(self) -> None:
        """Consume the outbound queue and hand each message to its channel. (nanobot manager.py:898)"""
        logger.info("Outbound dispatcher started")
        pending = self._outbound_pending
        while True:
            try:
                if pending:
                    msg = pending.pop(0)
                else:
                    async with asyncio.timeout(1.0):
                        msg = await self.bus.consume_outbound()
                self._outbound_dispatching = True
                event = msg.event
                target = self.channels.get(msg.channel)
                if target is not None and not target.projects_outbound and self._drop_for_channel(target, msg):
                    continue
                if (
                    isinstance(event, RuntimeModelUpdatedEvent)
                    and msg.channel == "websocket"
                    and "websocket" not in self.channels
                ):
                    continue
                if isinstance(event, StreamDeltaEvent):
                    msg, extra_pending = self._coalesce_stream_deltas(msg)
                    pending.extend(extra_pending)
                    event = msg.event
                channel = self.channels.get(msg.channel)
                if channel is None:
                    if msg.channel != "cli":
                        logger.warning("Unknown channel: {}", msg.channel)
                    continue
                if not isinstance(event, StreamDeltaEvent | StreamEndEvent | StreamedResponseEvent):
                    if self._should_suppress_outbound(msg):
                        logger.info("Suppressing duplicate outbound message to {}:{}", msg.channel, msg.chat_id)
                        continue
                await self._queue_outbound(channel, msg)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            finally:
                self._outbound_dispatching = False

    @staticmethod
    def _drop_for_channel(channel: BaseChannel, msg: OutboundMessage) -> bool:
        """nanobot's dispatcher filters (manager.py:930-958) for a channel that
        does not project events itself: reasoning unless ``show_reasoning``,
        progress / tool hints unless ``send_progress`` / ``send_tool_hints``,
        and retry-wait notices."""
        event = msg.event
        if isinstance(event, ProgressEvent):
            if event.reasoning_delta or event.reasoning_end or event.reasoning:
                return not channel.show_reasoning
            return not (channel.send_tool_hints if event.tool_hint else channel.send_progress)
        return isinstance(event, RetryWaitEvent)

    @staticmethod
    async def _send_once(channel: BaseChannel, msg: OutboundMessage) -> None:
        """Call the channel method for the event. (nanobot manager.py:1057)"""
        if channel.projects_outbound:
            await channel.send(msg)
            return
        event = msg.event
        if isinstance(event, ProgressEvent) and event.reasoning_end:
            await channel.send_reasoning_end(msg.chat_id, msg.metadata, stream_id=event.stream_id)
        elif isinstance(event, ProgressEvent) and event.reasoning_delta:
            await channel.send_reasoning_delta(msg.chat_id, msg.content, msg.metadata, stream_id=event.stream_id)
        elif isinstance(event, ProgressEvent) and event.reasoning:
            await channel.send_reasoning(msg)
        elif isinstance(event, ProgressEvent) and event.file_edit_events:
            await channel.send_file_edit_events(msg.chat_id, event.file_edit_events, msg.metadata)
        elif isinstance(event, StreamDeltaEvent | StreamEndEvent):
            await channel.send_delta(
                msg.chat_id,
                msg.content,
                msg.metadata,
                stream_id=event.stream_id,
                stream_end=isinstance(event, StreamEndEvent),
                resuming=event.resuming if isinstance(event, StreamEndEvent) else False,
                merge_next=event.merge_next if isinstance(event, StreamEndEvent) else False,
            )
        elif not isinstance(event, StreamedResponseEvent):
            await channel.send(msg)

    def _coalesce_stream_deltas(self, first_msg: OutboundMessage) -> tuple[OutboundMessage, list[OutboundMessage]]:
        """Merge consecutive deltas of one stream already queued. (nanobot manager.py:1081)"""
        first_event = first_msg.event
        first_stream_id = first_event.stream_id if isinstance(first_event, StreamDeltaEvent) else None
        target_key = (first_msg.channel, first_msg.chat_id, first_stream_id)
        combined_content = first_msg.content
        final_event: StreamDeltaEvent | StreamEndEvent = (
            first_event if isinstance(first_event, StreamDeltaEvent) else StreamDeltaEvent(stream_id=first_stream_id)
        )
        non_matching: list[OutboundMessage] = []
        while True:
            try:
                next_msg = self.bus.outbound.get_nowait()
            except asyncio.QueueEmpty:
                break
            next_event = next_msg.event
            next_stream_id = next_event.stream_id if isinstance(next_event, StreamDeltaEvent | StreamEndEvent) else None
            same_target = (next_msg.channel, next_msg.chat_id, next_stream_id) == target_key
            is_delta = isinstance(next_event, StreamDeltaEvent)
            is_end = isinstance(next_event, StreamEndEvent)
            same_response_sources = next_msg.metadata.get("response_sources") == first_msg.metadata.get(
                "response_sources"
            )
            if same_target and same_response_sources and (is_delta or (is_end and next_msg.content)):
                combined_content += next_msg.content
                if isinstance(next_event, StreamEndEvent):
                    final_event = StreamEndEvent(
                        stream_id=next_stream_id, resuming=next_event.resuming, merge_next=next_event.merge_next
                    )
                    break
            else:
                non_matching.append(next_msg)
                break
        merged = replace_outbound_event(first_msg, final_event, content=combined_content)
        return merged, non_matching

    async def _send_with_retry(self, channel: BaseChannel, msg: OutboundMessage) -> None:
        """Send with exponential backoff; cancellation propagates. (nanobot manager.py:1150)"""
        max_attempts = max(self.config.channels.send_max_retries, 1)
        attempt = 0
        while True:
            attempt += 1
            try:
                await self._send_once(channel, msg)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if isinstance(e, DeliveryRejectedError) or not channel.should_retry_send_error(e):
                    logger.error("Send to {} failed with a non-retryable {}: {}", msg.channel, type(e).__name__, e)
                    return
                if attempt >= max_attempts:
                    logger.exception("Failed to send to {} after {} attempts", msg.channel, attempt)
                    return
                delay = _SEND_RETRY_DELAYS[min(attempt - 1, len(_SEND_RETRY_DELAYS) - 1)]
                logger.warning(
                    "Send to {} failed (attempt {}/{}): {}, retrying in {}s",
                    msg.channel,
                    attempt,
                    max_attempts,
                    type(e).__name__,
                    delay,
                )
                await asyncio.sleep(delay)

    # -- status ---------------------------------------------------------------------

    def get_channel(self, name: str) -> BaseChannel | None:
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Runtime state of each enabled channel, including ones that failed. (nanobot manager.py:1214)"""
        status: dict[str, Any] = {}
        for name in [*self.channels, *(n for n in self._channel_errors if n not in self.channels)]:
            channel = self.channels.get(name)
            task = self._channel_tasks.get(name)
            error = self._channel_errors.get(name)
            running = bool(channel and channel.is_running)
            if error:
                state = "failed"
            elif running:
                state = "running"
            elif task is not None and not task.done():
                state = "starting"
            else:
                state = "stopped"
            status[name] = {"enabled": True, "running": running, "state": state, "owner": name, "instance_id": "default"}
            if error:
                status[name]["error"] = error
        return status

    @property
    def enabled_channels(self) -> list[str]:
        return list(self.channels.keys())
