"""Project bus outbound messages onto web UI wire frames.

A port of nanobot's outbound path for its WebSocket channel, without the
socket: the dispatcher's per-event routing (nanobot channels/manager.py
``_dispatch_outbound_loop`` filters and ``_send_once``), the application
projector (webui/outbound_projection.py ``WebUIOutboundProjector.send``)
and the channel's ``send_*`` methods (channels/websocket/runtime.py),
including the state nanobot keeps per chat on the channel and the transcript
recorder: stream and reasoning text buffers keyed by ``(chat_id,
stream_id)`` and the turn sequence counters keyed by ``(chat_id, turn_id)``
(webui/transcript.py ``_annotate_turn``).

:class:`WebUIOutboundProjector` turns one :class:`OutboundMessage` into a
list of :class:`WireFrame`. A frame holds the exact wire payload, who gets
it (the chat's subscribers, every connection, or only bootstrap-authenticated
web UI connections) and, when nanobot writes it to the chat's transcript,
the ``record`` to append (the payload as nanobot persists it: a completed
``stream_end`` / ``reasoning_end`` carries the full segment text, a
``message`` its text before image rewriting). A transport sends
``payload`` to the audience and hands ``record`` to the transcript store;
nanobot persists such records even when no client is subscribed.

Lifecycle: :func:`lifecycle_outbound` translates the runtime events a turn
publishes (nanoclaw/bus/runtime_events.py) into the outbound events nanobot's
``WebuiTurnCoordinator`` publishes for ``websocket`` chats (session/webui_turns.py):
``goal_status`` running/idle, ``turn_model_updated`` and ``turn_end``.

Left to the transport (layer 23): stream-delta coalescing on the queue,
duplicate suppression, the per-chat active-turn registry (frames say when
to release an owner), hydration on subscribe, temporary chats (never
persisted), markdown image rewriting and media signing (pass
``rewrite_text`` / ``sign_media``) and subagent task cards (pass
``subagent_task``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from nanoclaw.bus.events import OutboundMessage
from nanoclaw.bus.outbound_events import (
    GoalStateSyncEvent,
    GoalStatusEvent,
    ProgressEvent,
    RetryWaitEvent,
    RuntimeModelUpdatedEvent,
    SessionUpdatedEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
    TurnEndEvent,
    TurnModelUpdatedEvent,
    UserInputEvent,
    outbound_message_for_event,
)
from nanoclaw.bus.runtime_events import (
    SubagentTaskChanged,
    TurnCompleted,
    TurnRunStatusChanged,
    TurnRuntimeAdmitted,
)
from nanoclaw.events import AgentEvent
from nanoclaw.webui.metadata import (
    WEBSOCKET_TURN_OWNER_METADATA_KEY,
    WEBUI_SYSTEM_COMMAND_TURN_PREFIX,
    WEBUI_TURN_METADATA_KEY,
)
from nanoclaw.webui.outbound_wire import (
    WebUIWirePersistence,
    encode_delta,
    encode_file_edit,
    encode_goal_state,
    encode_goal_status,
    encode_message,
    encode_reasoning_delta,
    encode_reasoning_end,
    encode_runtime_model_updated,
    encode_session_updated,
    encode_stream_end,
    encode_turn_end,
    encode_turn_model_updated,
    encode_user_input,
    is_persisted_wire_event,
    project_notification,
    response_sources,
    webui_message_source,
)

__all__ = [
    "WebUIOutboundProjector",
    "WireAudience",
    "WireFrame",
    "is_persisted_wire_event",
    "lifecycle_outbound",
]

# ``chat``: the chat's subscribers. ``all``: every open connection
# (nanobot ``send_session_updated`` / ``send_runtime_model_updated``).
# ``webui``: bootstrap-authenticated web UI connections only.
WireAudience = Literal["chat", "all", "webui"]

# The text of a failed turn_end (nanobot session/webui_turns.py:840).
_MODEL_FAILURE_MESSAGE = "Model provider request failed. Check the provider configuration or service status, then try again."
_TURN_FAILURE_MESSAGE = "This turn failed and has ended."


@dataclass(frozen=True)
class WireFrame:
    """One wire payload and what the transport does with it.

    ``record``: the transcript row to append, ``None`` when nanobot does not
    persist this payload; ``phase``: the turn phase it was annotated with
    (``user``, ``activity``, ``answer``, ``reasoning``, ``complete``).
    ``turn_complete``: a ``turn_end``; once its record is stored the
    transport releases the turn owner ``turn_owner``. ``release_turn_owner``:
    an idle ``goal_status``; release ``turn_owner`` unless its transcript
    failed to persist (nanobot ``clear_websocket_turn_if_current(...,
    preserve_persistence_failure=True)``).
    """

    payload: dict[str, Any]
    audience: WireAudience = "chat"
    record: dict[str, Any] | None = None
    phase: str | None = None
    metadata: dict[str, Any] | None = None
    turn_complete: bool = False
    release_turn_owner: bool = False
    turn_owner: str | None = None

    @property
    def persisted(self) -> bool:
        return self.record is not None


def _turn_owner(metadata: Mapping[str, Any] | None) -> str | None:
    owner = (metadata or {}).get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
    return owner if isinstance(owner, str) else None


class WebUIOutboundProjector:
    """Turn bus outbound messages into web UI wire frames, keeping per-chat state.

    ``show_reasoning`` / ``send_progress`` / ``send_tool_hints`` are the
    channel flags nanobot's dispatcher checks (channels/base.py:36-38).
    ``rewrite_text(text)`` rewrites local markdown images to signed URLs and
    ``sign_media(path)`` signs one media path (both nanobot's media gateway;
    default: unchanged / unsigned). ``subagent_task(chat_id, task_id)``
    returns a task card for ``SubagentTaskChanged`` (nanobot session
    projection; default: no card).
    """

    def __init__(
        self,
        *,
        show_reasoning: bool = True,
        send_progress: bool = True,
        send_tool_hints: bool = True,
        rewrite_text: Callable[[str], str] | None = None,
        sign_media: Callable[[str], dict[str, str] | None] | None = None,
        subagent_task: Callable[[str, str], dict[str, Any] | None] | None = None,
    ) -> None:
        self.show_reasoning = show_reasoning
        self.send_progress = send_progress
        self.send_tool_hints = send_tool_hints
        self._rewrite_text = rewrite_text or (lambda text: text)
        self._sign_media = sign_media
        self._subagent_task = subagent_task
        self._stream_text_buffers: dict[tuple[str, str], list[str]] = {}
        self._reasoning_text_buffers: dict[tuple[str, str], list[str]] = {}
        self._turn_sequences: dict[tuple[str, str], int] = {}

    # -- state ----------------------------------------------------------------

    def clear_stream_buffers(self, chat_id: str) -> None:
        """Drop unclosed stream text of one chat. (nanobot runtime.py:507)"""
        for key in tuple(self._stream_text_buffers):
            if key[0] == chat_id:
                self._stream_text_buffers.pop(key, None)
        for key in tuple(self._reasoning_text_buffers):
            if key[0] == chat_id:
                self._reasoning_text_buffers.pop(key, None)

    def forget_chat(self, chat_id: str) -> None:
        """Drop all state of a discarded chat."""
        self.clear_stream_buffers(chat_id)
        for key in tuple(self._turn_sequences):
            if key[0] == chat_id:
                self._turn_sequences.pop(key, None)

    def has_open_stream(self, chat_id: str) -> bool:
        return any(
            key[0] == chat_id
            for key in (*self._stream_text_buffers, *self._reasoning_text_buffers)
        )

    def annotate(
        self,
        chat_id: str,
        event: dict[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        phase: str | None = None,
        include_source: bool = False,
    ) -> None:
        """Add source and turn fields to a payload in place. (nanobot transcript.py:542)

        With a turn id in ``metadata`` the payload gets ``turn_id``,
        ``turn_phase`` and the next ``turn_seq`` of that turn (the
        ``complete`` phase ends the sequence). The inbound path calls this
        too (``phase="user"`` for the user's row), so a turn's rows share one
        sequence.
        """
        if include_source and (source := webui_message_source(metadata)):
            event["source"] = source
        if include_source and metadata is not None and "response_sources" in metadata:
            event["response_sources"] = response_sources(metadata["response_sources"])
        turn_id = (metadata or {}).get(WEBUI_TURN_METADATA_KEY)
        if phase is None or not isinstance(turn_id, str) or not turn_id:
            return
        key = (chat_id, turn_id)
        sequence = self._turn_sequences.get(key, 0) + 1
        self._turn_sequences[key] = sequence
        event.update(turn_id=turn_id, turn_phase=phase, turn_seq=sequence)
        if phase == "complete":
            self._turn_sequences.pop(key, None)

    # -- projection -------------------------------------------------------------

    def project(self, msg: OutboundMessage) -> list[WireFrame]:
        """The frames one bus message becomes; ``[]`` when nanobot sends nothing."""
        event = msg.event
        progress_event = event if isinstance(event, ProgressEvent) else None
        # Dispatcher filters (nanobot manager.py:931-958).
        if progress_event and (
            progress_event.reasoning_delta
            or progress_event.reasoning_end
            or progress_event.reasoning
        ):
            if not self.show_reasoning:
                return []
        elif progress_event:
            if not (
                self.send_tool_hints if progress_event.tool_hint else self.send_progress
            ):
                return []
        if isinstance(event, RetryWaitEvent):
            return []
        # Per-event routing (nanobot manager.py:1057 ``_send_once``).
        if progress_event and progress_event.reasoning_end:
            return [
                self._reasoning_end(msg.chat_id, msg.metadata, progress_event.stream_id)
            ]
        if progress_event and progress_event.reasoning_delta:
            return self._reasoning_delta(
                msg.chat_id, msg.content, msg.metadata, progress_event.stream_id
            )
        if progress_event and progress_event.reasoning:
            # One-shot reasoning is a delta plus an end (nanobot channels/base.py:225).
            if not msg.content:
                return []
            return [
                *self._reasoning_delta(
                    msg.chat_id, msg.content, msg.metadata, progress_event.stream_id
                ),
                self._reasoning_end(
                    msg.chat_id, msg.metadata, progress_event.stream_id
                ),
            ]
        if progress_event and progress_event.file_edit_events:
            return [
                self._file_edit(
                    msg.chat_id, progress_event.file_edit_events, msg.metadata
                )
            ]
        if isinstance(event, StreamDeltaEvent):
            return [
                self._delta(
                    msg.chat_id, msg.content, msg.metadata, stream_id=event.stream_id
                )
            ]
        if isinstance(event, StreamEndEvent):
            return [
                self._stream_end(
                    msg.chat_id,
                    msg.content,
                    msg.metadata,
                    stream_id=event.stream_id,
                    resuming=event.resuming,
                    merge_next=event.merge_next,
                )
            ]
        if isinstance(event, StreamedResponseEvent):
            return []
        return self._project_application(msg, progress_event)

    def _project_application(
        self, msg: OutboundMessage, progress_event: ProgressEvent | None
    ) -> list[WireFrame]:
        """nanobot webui/outbound_projection.py:140 ``send``."""
        event = msg.event
        chat_id = msg.chat_id
        if isinstance(event, RuntimeModelUpdatedEvent):
            body = encode_runtime_model_updated(
                model_name=event.model, model_preset=event.model_preset
            )
            return [WireFrame(body, audience="all")] if body is not None else []
        if isinstance(event, TurnModelUpdatedEvent):
            body = encode_turn_model_updated(
                chat_id,
                model_name=event.model,
                model_preset=event.model_preset,
                context_window_tokens=event.context_window_tokens,
                fallback=event.fallback,
                reauth_provider=event.reauth_provider,
            )
            return [WireFrame(body)] if body is not None else []
        if isinstance(event, UserInputEvent):
            # Persisted before publication by whoever accepted the input.
            body = encode_user_input(
                chat_id,
                content=event.content,
                created_at_ms=event.created_at_ms,
                provenance=event.provenance,
            )
            return [WireFrame(body)]
        notification = project_notification(chat_id, event, msg.metadata)
        if notification is not None:
            metadata = msg.metadata if notification.attach_turn_metadata else None
            return [
                self._payload_frame(
                    chat_id,
                    dict(notification.payload),
                    notification.persistence,
                    metadata,
                )
            ]
        if isinstance(event, GoalStateSyncEvent):
            return [
                WireFrame(
                    encode_goal_state(chat_id, event.goal_state or {"active": False})
                )
            ]
        if isinstance(event, GoalStatusEvent):
            turn_id = (msg.metadata or {}).get(WEBUI_TURN_METADATA_KEY)
            owner = _turn_owner(msg.metadata)
            frames: list[WireFrame] = []
            if event.status in ("running", "idle"):
                body = encode_goal_status(
                    chat_id,
                    event.status,
                    started_at=event.started_at,
                    turn_id=turn_id if isinstance(turn_id, str) else None,
                )
                frames.append(
                    WireFrame(
                        body,
                        release_turn_owner=event.status == "idle",
                        turn_owner=owner,
                    )
                )
            return frames
        if isinstance(event, TurnEndEvent):
            turn_id = (msg.metadata or {}).get(WEBUI_TURN_METADATA_KEY)
            scope = (
                "metadata"
                if isinstance(turn_id, str)
                and turn_id.startswith(WEBUI_SYSTEM_COMMAND_TURN_PREFIX)
                else "thread"
            )
            end = self._payload_frame(
                chat_id,
                dict(encode_turn_end(chat_id, event, msg.metadata)),
                "turn_complete",
                msg.metadata,
                turn_owner=_turn_owner(msg.metadata),
            )
            return [
                end,
                WireFrame(encode_session_updated(chat_id, scope=scope), audience="all"),
            ]
        if isinstance(event, SubagentTaskChanged):
            task = (
                self._subagent_task(chat_id, event.task_id)
                if self._subagent_task is not None
                else None
            )
            if task is None:
                return []
            return [
                WireFrame({"event": "subagent_task", "chat_id": chat_id, "task": task})
            ]
        if isinstance(event, SessionUpdatedEvent):
            return [
                WireFrame(
                    encode_session_updated(chat_id, scope=event.scope), audience="all"
                )
            ]
        if progress_event and progress_event.file_edit_events:
            return [
                self._file_edit(chat_id, progress_event.file_edit_events, msg.metadata)
            ]
        return [self._message(msg, progress_event)]

    # -- channel sends (nanobot channels/websocket/runtime.py) -------------------

    def _message(
        self, msg: OutboundMessage, progress_event: ProgressEvent | None
    ) -> WireFrame:
        """nanobot runtime.py:1231 ``send_projected_message``."""
        media_urls: list[dict[str, str]] = []
        if msg.media and self._sign_media is not None:
            for entry in msg.media:
                if (signed := self._sign_media(entry)) is not None:
                    media_urls.append(signed)
        payload = encode_message(
            msg,
            progress_event,
            wire_text=self._rewrite_text(msg.content),
            media_urls=media_urls or None,
        )
        phase = (
            "activity" if payload.get("kind") in ("tool_hint", "progress") else "answer"
        )
        self.annotate(
            msg.chat_id,
            payload,
            metadata=msg.metadata,
            phase=phase,
            include_source=True,
        )
        record = {**payload, "text": msg.content}
        return WireFrame(payload, record=record, phase=phase, metadata=msg.metadata)

    def _file_edit(
        self, chat_id: str, edits: list[dict[str, Any]], metadata: dict[str, Any] | None
    ) -> WireFrame:
        """nanobot runtime.py:1362 ``send_file_edit_events``."""
        payload = encode_file_edit(chat_id, edits)
        self.annotate(chat_id, payload, metadata=metadata, phase="activity")
        return WireFrame(
            payload, record=dict(payload), phase="activity", metadata=metadata
        )

    def _reasoning_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None,
        stream_id: str | None,
    ) -> list[WireFrame]:
        """nanobot runtime.py:1292 ``send_reasoning_delta``: chunks are never persisted."""
        if not delta:
            return []
        payload = encode_reasoning_delta(chat_id, delta, stream_id=stream_id)
        self._reasoning_text_buffers.setdefault(
            (chat_id, str(stream_id or "")), []
        ).append(delta)
        self.annotate(chat_id, payload, metadata=metadata or {}, phase="reasoning")
        return [WireFrame(payload, phase="reasoning", metadata=metadata)]

    def _reasoning_end(
        self, chat_id: str, metadata: dict[str, Any] | None, stream_id: str | None
    ) -> WireFrame:
        """nanobot runtime.py:1331 ``send_reasoning_end``: the record carries the full text."""
        payload = encode_reasoning_end(chat_id, stream_id=stream_id)
        text = "".join(
            self._reasoning_text_buffers.pop((chat_id, str(stream_id or "")), [])
        )
        self.annotate(chat_id, payload, metadata=metadata or {}, phase="reasoning")
        record = dict(payload)
        if text:
            record["text"] = text
        return WireFrame(payload, record=record, phase="reasoning", metadata=metadata)

    def _delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None,
        *,
        stream_id: str | None,
    ) -> WireFrame:
        """nanobot runtime.py:1386 ``send_delta`` (a chunk)."""
        payload = encode_delta(chat_id, delta, stream_id=stream_id)
        self._stream_text_buffers.setdefault(
            (chat_id, str(stream_id or "")), []
        ).append(delta)
        self.annotate(
            chat_id,
            payload,
            metadata=metadata or {},
            phase="answer",
            include_source=True,
        )
        return WireFrame(payload, phase="answer", metadata=metadata)

    def _stream_end(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None,
        *,
        stream_id: str | None,
        resuming: bool,
        merge_next: bool,
    ) -> WireFrame:
        """nanobot runtime.py:1386 ``send_delta(stream_end=True)``.

        The segment's chunks are joined into the record's text; with
        ``merge_next`` the buffer stays open, so the next segment's end
        carries both.
        """
        key = (chat_id, str(stream_id or ""))
        buffered = (
            self._stream_text_buffers.setdefault(key, [])
            if merge_next
            else self._stream_text_buffers.pop(key, [])
        )
        if delta:
            buffered.append(delta)
        full_text = "".join(buffered)
        rewritten = self._rewrite_text(full_text)
        payload = encode_stream_end(
            chat_id,
            text=rewritten if delta or rewritten != full_text else None,
            stream_id=stream_id,
            resuming=resuming,
            merge_next=merge_next,
        )
        self.annotate(
            chat_id,
            payload,
            metadata=metadata or {},
            phase="answer",
            include_source=True,
        )
        record = {**payload, "text": rewritten}
        return WireFrame(payload, record=record, phase="answer", metadata=metadata)

    def _payload_frame(
        self,
        chat_id: str,
        body: dict[str, Any],
        persistence: WebUIWirePersistence,
        metadata: dict[str, Any] | None,
        *,
        turn_owner: str | None = None,
    ) -> WireFrame:
        """nanobot runtime.py:1442 ``send_payload``."""
        if persistence == "turn_activity":
            self.annotate(chat_id, body, metadata=metadata, phase="activity")
            return WireFrame(
                body, record=dict(body), phase="activity", metadata=metadata
            )
        if persistence == "turn_complete":
            self.annotate(chat_id, body, metadata=metadata, phase="complete")
            self.clear_stream_buffers(chat_id)
            return WireFrame(
                body,
                record=dict(body),
                phase="complete",
                metadata=metadata,
                turn_complete=True,
                turn_owner=turn_owner,
            )
        return WireFrame(body, metadata=metadata)


# -- turn lifecycle (nanobot session/webui_turns.py WebuiTurnCoordinator) -------


def lifecycle_outbound(
    event: AgentEvent,
    *,
    goal_state: dict[str, Any] | None = None,
    now: Callable[[], float] = time.time,
) -> OutboundMessage | None:
    """The outbound event nanobot publishes for a ``websocket`` turn's runtime event.

    - ``TurnRunStatusChanged`` -> ``GoalStatusEvent`` (``publish_turn_run_status``,
      webui_turns.py:432; a running turn without a start time starts now);
    - ``TurnRuntimeAdmitted`` -> ``TurnModelUpdatedEvent`` (:724);
    - ``TurnCompleted`` -> ``TurnEndEvent`` (``handle_turn_end``, :808;
      ``goal_state`` is the session's goal blob, ``{"active": False}``
      without a goal, as NanoClaw has none).

    ``None`` for other events and other channels.
    """
    context = getattr(event, "context", None)
    if context is None or context.channel != "websocket":
        return None
    if isinstance(event, TurnRunStatusChanged):
        started_at: float | None = None
        if event.status == "running":
            started_at = (
                float(event.started_at)
                if isinstance(event.started_at, int | float) and event.started_at > 0
                else now()
            )
        return outbound_message_for_event(
            channel=context.channel,
            chat_id=str(context.chat_id),
            event=GoalStatusEvent(status=event.status, started_at=started_at),
            metadata=context.metadata,
        )
    if isinstance(event, TurnRuntimeAdmitted):
        return outbound_message_for_event(
            channel=context.channel,
            chat_id=context.chat_id,
            event=TurnModelUpdatedEvent(
                model=event.runtime.model or "",
                model_preset=event.runtime.model_preset,
                context_window_tokens=event.runtime.context_window_tokens,
            ),
            metadata=context.metadata,
        )
    if isinstance(event, TurnCompleted):
        return outbound_message_for_event(
            channel=context.channel,
            chat_id=context.chat_id,
            event=TurnEndEvent(
                latency_ms=event.latency_ms,
                goal_state=goal_state if goal_state is not None else {"active": False},
                usage=event.usage,
                round_usages=event.round_usages,
                context_window_tokens=event.runtime.context_window_tokens
                if event.runtime is not None
                else None,
                outcome=event.outcome,
                failure_kind=event.failure_kind,
                failure_error_kind=event.failure_error_kind,
                failure_attempts=event.failure_attempts,
                failure_message=(
                    _MODEL_FAILURE_MESSAGE
                    if event.failure_kind == "model"
                    else _TURN_FAILURE_MESSAGE
                    if event.outcome == "failed"
                    else None
                ),
            ),
            metadata=context.metadata,
        )
    return None
