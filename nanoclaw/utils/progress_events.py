"""Helpers for structured progress events. Copied from nanobot/utils/progress_events.py.

``output_events`` adapts plain callbacks (NanoClaw's ``on_progress(text,
tool_hint=...)``, plus ``on_stream`` / ``on_stream_end``) into an
:class:`EventSink`, so callers that pass callbacks keep working once the
runner publishes events. The ``build_tool_event_*`` helpers make the
version-1 tool-event payloads the web UI reads.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from nanoclaw.bus.outbound_events import (
    FileEditEvent,
    ProgressEvent,
    StreamDeltaEvent,
    StreamEndEvent,
)
from nanoclaw.events import NO_EVENTS, AgentEvent, EventSink

if TYPE_CHECKING:
    from nanoclaw.agent.hook import AgentHookContext


def _on_progress_accepts(cb: Callable[..., Any], name: str) -> bool:
    """Whether ``cb`` takes keyword ``name`` (or ``**kwargs``). (nanobot :19)"""
    try:
        sig = inspect.signature(cb)
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return name in sig.parameters


def output_events(
    *,
    default: EventSink = NO_EVENTS,
    on_progress: Callable[..., Awaitable[None]] | None = None,
    on_stream: Callable[[str], Awaitable[None]] | None = None,
    on_stream_end: Callable[..., Awaitable[None]] | None = None,
) -> EventSink:
    """Wrap direct-call output callbacks in an EventSink. (nanobot :29)

    Optional keyword fields (tool_events, reasoning, ...) are only passed to
    callbacks that declare them; anything the callbacks don't take goes to
    ``default``.
    """
    if on_progress is None and on_stream is None and on_stream_end is None:
        return default
    progress_fields = {
        name for name in ("tool_events", "file_edit_events", "reasoning", "reasoning_end")
        if on_progress is not None and _on_progress_accepts(on_progress, name)
    }
    merge_next = on_stream_end is not None and _on_progress_accepts(on_stream_end, "merge_next")

    def accepts(event_type: type[AgentEvent]) -> bool:
        if issubclass(event_type, FileEditEvent) and on_progress is not None:
            return "file_edit_events" in progress_fields
        if issubclass(event_type, ProgressEvent) and on_progress is not None:
            return True
        if issubclass(event_type, StreamDeltaEvent) and on_stream is not None:
            return True
        if issubclass(event_type, StreamEndEvent) and on_stream_end is not None:
            return True
        return default.accepts(event_type)

    async def publish(event: AgentEvent) -> None:
        if isinstance(event, ProgressEvent) and on_progress is not None:
            if event.file_edit_events:
                if "file_edit_events" in progress_fields:
                    await on_progress(event.content, file_edit_events=event.file_edit_events)
                return
            if event.reasoning_delta or event.reasoning_end:
                name = "reasoning" if event.reasoning_delta else "reasoning_end"
                if name in progress_fields:
                    await on_progress(event.content, **{name: True})
                return
            # Tool-finish payloads carry no text; old callbacks would print "".
            if event.tool_events and not event.tool_hint and "tool_events" not in progress_fields:
                return
            kwargs: dict[str, Any] = {"tool_hint": event.tool_hint}
            if event.tool_events and "tool_events" in progress_fields:
                kwargs["tool_events"] = event.tool_events
            await on_progress(event.content, **kwargs)
        elif isinstance(event, StreamDeltaEvent) and on_stream is not None:
            await on_stream(event.content)
        elif isinstance(event, StreamEndEvent) and on_stream_end is not None:
            end_kwargs: dict[str, Any] = {"resuming": event.resuming}
            if event.merge_next and merge_next:
                end_kwargs["merge_next"] = True
            await on_stream_end(**end_kwargs)
        elif default.publish is not None:
            await default.publish(event)

    return EventSink(publish, accepts)


def _tool_event_arguments(tool_call: Any) -> dict[str, Any]:
    arguments = getattr(tool_call, "arguments", {}) or {}
    return cast(dict[str, Any], arguments) if isinstance(arguments, dict) else {}


def build_tool_event_start_payload(tool_call: Any) -> dict[str, Any]:
    """Payload announcing a tool call. (nanobot :91)"""
    return {
        "version": 1,
        "phase": "start",
        "call_id": str(getattr(tool_call, "id", "") or ""),
        "name": getattr(tool_call, "name", ""),
        "arguments": _tool_event_arguments(tool_call),
        "result": None,
        "error": None,
        "files": [],
        "embeds": [],
    }


def tool_event_result_extras(result: Any) -> tuple[list[Any], list[Any]]:
    """``files`` / ``embeds`` lists from a dict tool result. (nanobot :105)"""
    if not isinstance(result, dict):
        return [], []
    result_data = cast(dict[str, Any], result)
    raw_files = result_data.get("files")
    raw_embeds = result_data.get("embeds")
    files: list[Any] = cast(list[Any], raw_files) if isinstance(raw_files, list) else []
    embeds: list[Any] = cast(list[Any], raw_embeds) if isinstance(raw_embeds, list) else []
    return files, embeds


def build_tool_event_finish_payloads(context: AgentHookContext) -> list[dict[str, Any]]:
    """End/error payloads from the iteration's tool calls, results and events.

    (nanobot :116)
    """
    payloads: list[dict[str, Any]] = []
    count = min(len(context.tool_calls), len(context.tool_results), len(context.tool_events))
    for idx in range(count):
        tool_call = context.tool_calls[idx]
        result = context.tool_results[idx]
        event = context.tool_events[idx]
        phase = "end" if event.get("status") == "ok" else "error"
        files, embeds = tool_event_result_extras(result)
        payload: dict[str, Any] = {
            "version": 1,
            "phase": phase,
            "call_id": str(getattr(tool_call, "id", "") or ""),
            "name": getattr(tool_call, "name", ""),
            "arguments": _tool_event_arguments(tool_call),
            "result": result if phase == "end" else None,
            "error": None,
            "files": files,
            "embeds": embeds,
        }
        if phase == "error":
            if isinstance(result, str) and result.strip():
                payload["error"] = result.strip()
            else:
                payload["error"] = str(event.get("detail") or "Tool execution failed")
        payloads.append(payload)
    return payloads
