"""The agent loop: call the model, run the tools it asks for, repeat.

Mirrors nanobot/agent/runner.py (AgentRunner.run / _run_core) and
nanobot/agent/tools/execution.py (execute_tool_calls). The runner is
model-facing only: it knows nothing about sessions, channels or prompts.
Its caller builds ``initial_messages`` and decides what to do with the result.

Since layer 13 the runner reports what it does through an :class:`AgentHook`
(``spec.hook``), called at the same points and in the same order as nanobot's
runner, and every model request goes through ``chat_stream_with_retry``
(deltas only reach the hook when it ``wants_streaming()``). Retry waits are
published to ``spec.events`` as :class:`RetryWaitEvent`.

Since layer 14 a caller can fold messages that arrive mid-turn into the run
(``injection_callback``, drained before every model call and after an answer;
``terminal_injection_callback``, consulted only when the turn is about to end,
e.g. to wait for running subagents), and can keep the latest finished step of
the turn (``checkpoint_callback``) so a cancelled turn can still be saved.

Since layer 15 every model request goes through
``ContextGovernor.prepare_request`` (context_governance.py): the transcript is
repaired and merged for the model, measured, and compacted when it no longer
fits ``context_window_tokens``, also between tool calls (nanobot runner.py:877).
A compaction's summary comes back as ``AgentRunResult.summary_checkpoint``.

Since layer 17 each ``read_file`` call runs inside ``file_read_context``
(agent/tools/file_state.py) with the tool results of the request the model
just answered, so a repeat read is answered with "[File unchanged since last
read: ...]" only while the earlier result is really in the model's input
(nanobot execution.py:62-72, :159-162).
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import nullcontext
from contextvars import ContextVar
from functools import cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanoclaw.agent.context_governance import (
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    HistoryConsolidator,
    ModelRequestState,
    SummaryTranscriptBuilder,
)
from nanoclaw.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext
from nanoclaw.agent.messages import (
    BUDGET_EXHAUSTED_FINALIZATION_PROMPT,
    DEFAULT_ERROR_MESSAGE,
    EMPTY_FINAL_RESPONSE_MESSAGE,
    FINALIZATION_RETRY_PROMPT,
    MAX_ITERATIONS_MESSAGE,
    MODEL_ERROR_PLACEHOLDER,
    build_assistant_message,
    build_malformed_tool_call_note,
    build_tool_message,
    extract_reasoning,
    is_blank,
    strip_reasoning_tags,
)
from nanoclaw.agent.progress_hook import AgentProgressHook
from nanoclaw.agent.tools import ToolRegistry, is_tool_error
from nanoclaw.agent.tools.file_state import file_read_context
from nanoclaw.agent.tools.registry import with_retry_hint
from nanoclaw.agent.tools.violations import classify_violation, is_ssrf_violation
from nanoclaw.agent.tools.context import current_request_context
from nanoclaw.bus.outbound_events import TurnModelUpdatedEvent
from nanoclaw.events import NO_EVENTS, EventSink, RetryWaitEvent
from nanoclaw.providers import LLMProvider, LLMResponse, LLMUsage, ToolCallRequest
from nanoclaw.providers.fallback_provider import FallbackModelSelection
from nanoclaw.session.summary import SessionSummaryCheckpoint
from nanoclaw.utils.progress_events import output_events

# Called as on_progress(text) for the model's text before tool calls, and
# on_progress(hint, tool_hint=True) with the tool hints (nanobot's progress
# events). Kept as a compatibility input: the runner wraps it in an
# AgentProgressHook (see AgentRunSpec.on_progress).
ProgressCallback = Callable[..., Awaitable[None]]

# Called with each checkpoint payload: {"phase": "awaiting_tools" |
# "tools_completed" | "final_response", "iteration", "model",
# "assistant_message", "completed_tool_results", "pending_tool_calls"}
# (nanobot runner.py:70, without "provider_state").
CheckpointCallback = Callable[[dict[str, Any]], Awaitable[None]]

# Returns one snapshot of pending input (nanobot runner.py:71). Items may be
# {"role": "user", "content": ...} dicts (kept as they are, extra keys too),
# objects with a ``.content`` (e.g. InboundMessage) or anything else (str()).
InjectionCallback = Callable[[], Awaitable[Iterable[Any] | None]]


# The events of the model request in flight, so the process-level fallback
# observer can address the turn that asked (nanobot routes it through
# ``current_request_context()``, webui_turns.py:555).
_request_events: ContextVar[EventSink | None] = ContextVar("nanoclaw_request_events", default=None)


async def publish_fallback_model(selection: FallbackModelSelection) -> None:
    """A ``FallbackProvider`` observer: report the fallback model that answered.

    nanobot's ``build_webui_fallback_model_observer`` publishes a
    ``TurnModelUpdatedEvent(fallback=True)`` to the WebUI chat of the current
    request; NanoClaw publishes it to the running request's event sink, where
    the terminal and text channels show it as a progress line.
    """
    events = _request_events.get()
    if events is None or not events.accepts(TurnModelUpdatedEvent):
        return
    context = current_request_context()
    await events.emit(
        TurnModelUpdatedEvent(
            model=selection.model,
            model_preset=context.runtime.model_preset if context is not None and context.runtime is not None else None,
            fallback=True,
            reauth_provider=selection.reauth_provider,
        )
    )


def _raise_if_cancelling() -> None:
    """Don't resume model work when a dependency swallowed a cancellation.

    (nanobot runner.py:64)
    """
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


# After this many empty replies in a row, ask once more with tools off.
_MAX_EMPTY_RETRIES = 2

# finish_reasons after which an empty reply is final rather than retried.
_NO_RETRY_FINISH_REASONS = frozenset({"error", "length", "refusal", "content_filter"})


@dataclass
class AgentRunSpec:
    """Everything one run needs."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    provider: LLMProvider
    model: str | None = None
    max_iterations: int = 20  # nanobot's default is 200
    max_tool_result_chars: int = 16_000
    # Where oversized tool results are saved; without it they are truncated.
    workspace: Path | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    # Saved tool results go in a folder per session (nanobot: per session key).
    session_key: str | None = None
    # When the iterations run out: ask the model for a summary with tools off
    # (main agent), or just end with the fixed message (subagents set False).
    finalize_on_max_iterations: bool = True
    # Replaces the default out-of-iterations message ({max_iterations} is filled in).
    max_iterations_message: str | None = None
    # Compatibility input (before layer 13 the runner's only output): used
    # only when ``hook`` is None, wrapped as AgentProgressHook(output_events(
    # on_progress=...)). See ProgressCallback.
    on_progress: ProgressCallback | None = None
    # "standard" or "persistent" (nanobot ``provider_retry_mode``); see
    # LLMProvider.chat_with_retry.
    provider_retry_mode: str = "standard"
    # Lifecycle callbacks (nanobot runner.py:109). None: a non-streaming
    # AgentProgressHook (fed by ``on_progress`` if given), so <think> blocks
    # are still removed from the answer; nanobot's default is a bare
    # AgentHook, but every nanobot caller passes a progress hook.
    hook: AgentHook | None = None
    # Where retry waits go, as RetryWaitEvent (nanobot runner.py:125
    # ``events``, read by the provider's retry notifications, base.py:1357).
    events: EventSink = NO_EVENTS
    # The latest finished step of the turn, so a cancelled turn can keep its
    # completed tool results (nanobot runner.py:116; the loop stores it and
    # restores it on cancel, loop.py:1082 and :1740).
    checkpoint_callback: CheckpointCallback | None = None
    # Pending input drained before every model call and after an answer
    # (nanobot runner.py:119; the loop's ``_drain_pending``, loop.py:1098).
    injection_callback: InjectionCallback | None = None
    # Asked only when the turn would otherwise end with an answer; may wait
    # for running subagents (nanobot runner.py:120; the loop's
    # ``_wait_for_pending``, loop.py:1217).
    terminal_injection_callback: InjectionCallback | None = None
    # Context governance (nanobot runner.py:107-118, context_governance.py).
    # 0: no budget, so never compacts (the transcript is still repaired and
    # tool results budgeted). Otherwise ``consolidate_history`` is required:
    # it summarizes the accepted history (``Consolidator.summarize_transcript``).
    context_window_tokens: int = 0
    consolidate_history: HistoryConsolidator | None = None
    # A turn built from a session (nanobot ``transcript_input``): the number of
    # leading messages taken as already accepted (system prompt + history),
    # the summary in effect, and how to rebuild the system prompt around a new
    # summary. Without a builder the whole initial transcript counts as
    # accepted (nanobot ``ContextCompactionState.from_messages``).
    accepted_boundary: int | None = None
    active_summary: str | None = None
    summary_transcript_builder: SummaryTranscriptBuilder | None = None


@dataclass
class AgentRunResult:
    """What a run produced.

    ``stop_reason`` is one of ``completed``, ``error``,
    ``empty_final_response`` or ``max_iterations``. ``tool_events`` has one
    ``{"name", "status", "detail"}`` dict per tool call of the run
    (nanobot runner.py:142). ``had_injections``: at least one injected
    message was folded into the run. ``summary_checkpoint``: the run
    compacted; its summary replaces ``messages[:transcript_boundary]``.
    """

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False
    summary_checkpoint: SessionSummaryCheckpoint | None = field(default=None, repr=False)


def _add_usage(total: LLMUsage | None, new: LLMUsage | None) -> LLMUsage | None:
    """Sum a turn's usage across requests. (nanobot runner.py:1364 ``_merge_usage``)"""
    if new is None:
        return total
    if total is None:
        return new
    return total + new


def _event_detail(prefix: str, text: str, limit: int = 160) -> str:
    """nanobot tools/execution.py:296."""
    return (prefix + text.replace("\n", " ").strip())[:limit]


def _short_detail(text: str) -> str:
    """One line of at most 120 characters (nanobot tools/execution.py:191)."""
    return text.replace("\n", " ").strip()[:120]


def _ok_detail(result: Any) -> str:
    """The tool event detail for a successful call (nanobot tools/execution.py:210)."""
    detail = ("" if result is None else str(result)).replace("\n", " ").strip()
    if not detail:
        return "(empty)"
    return detail[:120] + "..." if len(detail) > 120 else detail


class AgentRunner:
    """Runs one agent turn against a provider and a tool registry."""

    def __init__(self) -> None:
        self.context_governor = ContextGovernor()

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        """Run hooks around the core loop, in nanobot's order (runner.py:308-372)."""
        hook = spec.hook if spec.hook is not None else self._default_hook(spec)
        has_run_hooks = hook.has_run_hooks()
        messages: list[dict[str, Any]] = []
        context = AgentRunHookContext(messages=[])
        try:
            try:
                # Work on a copy so the caller's list is never modified.
                messages = copy.deepcopy(spec.initial_messages)
                context.messages = self._snapshot(messages) if has_run_hooks else []
                await hook.before_run(context)
                result = await self._run_core(spec, hook, messages)
            except Exception as exc:
                context.messages = self._snapshot(messages) if has_run_hooks else []
                context.stop_reason = "error"
                context.error = f"Error: {type(exc).__name__}: {exc}"
                context.exception = exc
                await hook.on_error(context)
                raise
            else:
                context.messages = self._snapshot(result.messages) if has_run_hooks else []
                context.final_content = result.final_content
                context.tools_used = list(result.tools_used)
                context.usage = result.usage
                context.stop_reason = result.stop_reason
                context.error = result.error
                context.tool_events = copy.deepcopy(result.tool_events)
                context.had_injections = result.had_injections
                context.exception = None
                if context.error is not None:
                    await hook.on_error(context)
                await hook.after_run(context)
                return result
        except asyncio.CancelledError as exc:
            context.stop_reason = "cancelled"
            context.error = None
            context.exception = exc
            if has_run_hooks:
                context.messages = self._snapshot(messages)
            raise
        finally:
            if context.exception is None:
                await hook.on_finally(context)
            else:
                try:
                    await hook.on_finally(context)
                except Exception:
                    logger.opt(exception=True).error(
                        "AgentHook.on_finally error after {}",
                        context.stop_reason or "run exception",
                    )

    @staticmethod
    def _default_hook(spec: AgentRunSpec) -> AgentHook:
        """The hook for a spec without one: progress lines to ``on_progress``, if given."""
        return AgentProgressHook(output_events(on_progress=spec.on_progress), session_key=spec.session_key)

    @staticmethod
    def _snapshot(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run-hook copy of the messages (nanobot runner.py:164 ``_run_context_snapshot``)."""
        return copy.deepcopy(messages) if messages else []

    # --- mid-turn injections (nanobot runner.py:170-306) ---------------------

    @staticmethod
    def _append_injected_messages(messages: list[dict[str, Any]], injections: list[dict[str, Any]]) -> None:
        """Append injected messages as they are; the model copy merges adjacent users.

        (nanobot runner.py:170)
        """
        messages.extend(injections)

    async def _drain_injections(self, spec: AgentRunSpec, *, terminal: bool = False) -> list[dict[str, Any]]:
        """One snapshot of pending input as user messages (nanobot runner.py:262).

        Dicts are kept only when they are user messages; other objects become
        ``{"role": "user", "content": item.content}`` (or ``str(item)``).
        Blank content is skipped; a failing callback counts as no input.
        """
        callback = spec.terminal_injection_callback if terminal else spec.injection_callback
        if callback is None:
            return []
        try:
            items = await callback()
        except Exception:
            logger.opt(exception=True).error("injection_callback failed")
            return []
        if not items:
            return []
        injected: list[dict[str, Any]] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                content = item.get("content")
                if item.get("role") == "user" and "content" in item and self._has_injection_content(content):
                    injected.append(item)
                continue
            content = item.content if hasattr(item, "content") else str(item)
            if self._has_injection_content(content):
                injected.append({"role": "user", "content": content})
        return injected

    @staticmethod
    def _has_injection_content(content: Any) -> bool:
        """nanobot runner.py:299."""
        if content is None:
            return False
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return bool(content)
        return True

    async def _try_drain_injections(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any] | None,
        injection_cycles: int,
        *,
        phase: str = "after error",
        iteration: int | None = None,
        wait_at_terminal: bool = False,
        drain_callback: bool = True,
    ) -> tuple[bool, int]:
        """Append one pending-input snapshot; return whether the run continues.

        When input arrives after an answer, the answer (``assistant_message``)
        is kept in history first, with a ``final_response`` checkpoint.
        ``wait_at_terminal`` asks the terminal callback when nothing is
        pending; ``drain_callback=False`` (no iteration left) consumes nothing,
        so the input stays queued for the next turn. (nanobot runner.py:179,
        without the goal ``continuation_callback``.)
        """
        injections = await self._drain_injections(spec) if drain_callback else []
        if not injections and wait_at_terminal and drain_callback:
            injections = await self._drain_injections(spec, terminal=True)
        if not injections:
            return False, injection_cycles
        injection_cycles += 1
        if assistant_message is not None:
            messages.append(assistant_message)
            if iteration is not None:
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "final_response",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [],
                    },
                )
        self._append_injected_messages(messages, injections)
        preview = "\n\n".join(m["content"] for m in injections if isinstance(m.get("content"), str))
        preview = preview[:80] + "..." if len(preview) > 80 else preview
        logger.info(
            "Injected {} follow-up message(s) {} (snapshot {}): {}",
            len(injections),
            phase,
            injection_cycles,
            preview,
        )
        return True, injection_cycles

    @staticmethod
    def _request_state(spec: AgentRunSpec, messages: list[dict[str, Any]]) -> ModelRequestState:
        """The run's governance config and H/Δ state (nanobot runner.py:379-457).

        ``messages`` is the run's own transcript, so a pending-read notice
        written during compaction is part of the result.
        """
        if spec.context_window_tokens and spec.consolidate_history is None:
            raise ValueError("consolidate_history is required with context_window_tokens")
        if spec.summary_transcript_builder is not None:
            compaction = ContextCompactionState.from_transcript(
                messages,
                accepted_boundary=len(messages) if spec.accepted_boundary is None else spec.accepted_boundary,
                active_summary=spec.active_summary,
                summary_transcript_builder=spec.summary_transcript_builder,
                consolidate_history=spec.consolidate_history,
            )
        else:
            compaction = ContextCompactionState.from_messages(messages, spec.consolidate_history)
        config = ContextGovernanceConfig(
            session_key=spec.session_key,
            max_tool_result_chars=spec.max_tool_result_chars,
            workspace=spec.workspace,
            context_window_tokens=spec.context_window_tokens,
            max_tokens=spec.max_tokens,
        )
        return ModelRequestState(config=config, compaction=compaction, events=spec.events)

    @staticmethod
    async def _emit_checkpoint(spec: AgentRunSpec, payload: dict[str, Any]) -> None:
        """nanobot runner.py:1380."""
        if spec.checkpoint_callback is not None:
            await spec.checkpoint_callback(payload)

    async def _run_core(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        messages: list[dict[str, Any]],
    ) -> AgentRunResult:
        """The iterations (nanobot runner.py:400-876)."""
        final_content: str | None = None
        stop_reason = "completed"
        error: str | None = None
        tools_used: list[str] = []
        tool_events: list[dict[str, str]] = []
        usage: LLMUsage | None = None
        empty_retries = 0
        # Per-run throttle for repeated attempts at the same outside target.
        workspace_violation_counts: dict[str, int] = {}
        had_injections = False
        injection_cycles = 0
        state = self._request_state(spec, messages)

        for iteration in range(spec.max_iterations):
            # A resumed iteration must not inherit a previous iteration's
            # failure (nanobot runner.py:472, fix 5825e15).
            stop_reason = "completed"
            error = None
            # Pending input is cut into a finite snapshot before every model
            # call: follow-ups that arrived before the first request, during
            # the previous request or while tools ran (nanobot runner.py:476).
            drained, injection_cycles = await self._try_drain_injections(
                spec, messages, None, injection_cycles, phase="before model call"
            )
            if drained:
                had_injections = True
            context = AgentHookContext(iteration=iteration, messages=messages, session_key=spec.session_key)
            await hook.before_iteration(context)
            # H plus what was added since; the request just sent becomes the
            # new H (nanobot runner.py:495-513).
            request_message_count = len(messages)
            request_messages = state.compaction.request_messages(messages)
            response, raw_usage = await self._request_model(spec, state, request_messages, hook, context)
            assert state.messages is not None
            messages_for_model = state.messages
            self.context_governor.accept_request(state, messages_for_model, raw_boundary=request_message_count)
            context.response = response
            context.tool_calls = list(response.tool_calls)
            # Reasoning out, inline <think> stripped (nanobot runner.py:516-531).
            reasoning_text, response.content = extract_reasoning(
                response.reasoning_content, response.thinking_blocks, response.content
            )
            context.usage = raw_usage
            usage = _add_usage(usage, raw_usage)
            if reasoning_text and not context.streamed_reasoning:
                await hook.emit_reasoning(reasoning_text)
                await hook.emit_reasoning_end()
                context.streamed_reasoning = True

            # 1. The model wants tools: run them and go round again (runner.py:532-621).
            if response.should_execute_tools:
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)
                assistant_message = build_assistant_message(
                    response.content,
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                messages.append(assistant_message)
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "awaiting_tools",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                    },
                )
                await hook.before_execute_tools(context)
                results: list[Any] = []
                new_events: list[dict[str, str]] = []
                completed_tool_results: list[dict[str, Any]] = []
                read_results = self._model_tool_results(messages_for_model)
                for call in response.tool_calls:
                    raw, event = await self._execute_tool_call(
                        spec, call, workspace_violation_counts, hook=hook, context=context,
                        read_results=read_results,
                    )
                    results.append(raw)
                    new_events.append(event)
                    content = self.context_governor.normalize_tool_result(
                        state.config, call.id, call.name, "" if raw is None else str(raw)
                    )
                    tool_message = build_tool_message(call.id, call.name, content)
                    messages.append(tool_message)
                    completed_tool_results.append(tool_message)
                tool_events.extend(new_events)
                tools_used.extend(
                    call.name for call, event in zip(response.tool_calls, new_events) if event["status"] == "ok"
                )
                context.tool_results = results
                context.tool_events = list(new_events)
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results,
                        "pending_tool_calls": [],
                    },
                )
                empty_retries = 0
                await hook.after_iteration(context)
                continue

            # Tool calls that arrive with a cut-off or filtered reply may be
            # incomplete; they are dropped and the reply is treated as text.
            if response.has_tool_calls:
                logger.warning(
                    "Ignoring tool calls under finish_reason='{}' for {}",
                    response.finish_reason,
                    spec.session_key or "default",
                )
            clean = hook.finalize_content(context, response.content)

            # 2. Empty reply: retry, then ask once more with tools off (runner.py:632-672).
            if response.finish_reason not in _NO_RETRY_FINISH_REASONS and is_blank(clean):
                empty_retries += 1
                if empty_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty response on turn {} for {} ({}/{}); retrying",
                        iteration,
                        spec.session_key or "default",
                        empty_retries,
                        _MAX_EMPTY_RETRIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=False)
                    await hook.after_iteration(context)
                    continue
                logger.warning(
                    "Empty response on turn {} for {} after {} retries; attempting finalization",
                    iteration,
                    spec.session_key or "default",
                    empty_retries,
                )
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=False)
                # The request just sent plus the prompt (nanobot runner.py:1169).
                response = await self._request(
                    spec,
                    state,
                    messages_for_model + [{"role": "user", "content": FINALIZATION_RETRY_PROMPT}],
                    with_tools=False,
                )
                usage = _add_usage(usage, response.usage)
                context.response = response
                context.usage = _add_usage(raw_usage, response.usage)
                context.tool_calls = list(response.tool_calls)
                clean = hook.finalize_content(context, response.content)

            final_message: dict[str, Any] | None = None
            if not response.is_error and not is_blank(clean):
                final_message = build_assistant_message(
                    clean, reasoning_content=response.reasoning_content, thinking_blocks=response.thinking_blocks
                )

            # Input that arrived during the answer continues the turn. It is
            # checked BEFORE the stream is closed, so a streaming channel keeps
            # its message open (resuming=True). Only an answer may wait for the
            # terminal callback (nanobot runner.py:729-756).
            can_make_followup_request = iteration + 1 < spec.max_iterations
            should_continue, injection_cycles = await self._try_drain_injections(
                spec,
                messages,
                final_message,
                injection_cycles,
                phase="after final response",
                iteration=iteration,
                wait_at_terminal=(
                    final_message is not None and response.finish_reason not in _NO_RETRY_FINISH_REASONS
                ),
                drain_callback=can_make_followup_request,
            )
            if should_continue:
                had_injections = True
            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=should_continue)
            if should_continue:
                await hook.after_iteration(context)
                continue

            # 3. Model error: stop, keeping history well formed; input that
            # arrived meanwhile still gets an answer (runner.py:758-782).
            if response.is_error:
                final_content = clean or DEFAULT_ERROR_MESSAGE
                stop_reason, error = "error", final_content
                self._append_model_error_placeholder(messages)
                context.final_content, context.error, context.stop_reason = final_content, error, stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec,
                    messages,
                    None,
                    injection_cycles,
                    phase="after LLM error",
                    drain_callback=can_make_followup_request,
                )
                if should_continue:
                    had_injections = True
                    continue
                break

            # 4. Still nothing to say (runner.py:783-801).
            if final_message is None:
                final_content = EMPTY_FINAL_RESPONSE_MESSAGE
                stop_reason, error = "empty_final_response", final_content
                messages.append(build_assistant_message(final_content))
                context.final_content, context.error, context.stop_reason = final_content, error, stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec,
                    messages,
                    None,
                    injection_cycles,
                    phase="after empty response",
                    drain_callback=can_make_followup_request,
                )
                if should_continue:
                    had_injections = True
                    continue
                break

            # 5. A normal final answer (runner.py:803-835).
            messages.append(final_message)
            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": spec.model,
                    "assistant_message": final_message,
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                },
            )
            final_content = clean
            context.final_content, context.stop_reason = final_content, stop_reason
            await hook.after_iteration(context)
            break
        else:
            # Out of iterations: ask for a summary with tools off, else a fixed message.
            stop_reason = "max_iterations"
            final_content, finalize_usage = await self._finalize_after_max_iterations(spec, state, hook, messages)
            usage = _add_usage(usage, finalize_usage)
            messages.append(build_assistant_message(final_content))

        return AgentRunResult(
            final_content,
            messages,
            tools_used,
            usage,
            stop_reason,
            error,
            tool_events,
            had_injections=had_injections,
            summary_checkpoint=state.compaction.summary_checkpoint,
        )

    def _retry_wait_callback(self, spec: AgentRunSpec) -> Callable[[str], Awaitable[None]] | None:
        """Retry waits become RetryWaitEvent (nanobot providers/base.py:1357 ``_retry_notifications``)."""
        if spec.events.publish is None:
            return None

        async def publish(content: str) -> None:
            await spec.events.emit(RetryWaitEvent(content))

        return publish

    async def _request(
        self,
        spec: AgentRunSpec,
        state: ModelRequestState,
        messages: list[dict[str, Any]],
        *,
        with_tools: bool,
        stream: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """One provider call through ``chat_stream_with_retry``, retried on transient errors.

        The messages first go through ``ContextGovernor.prepare_request``
        (repaired, measured, compacted if needed; the exact request is kept in
        ``state.messages``). ``stream`` holds the delta callbacks (only when
        the hook streams). ``usage`` is always set: estimated if the provider
        sent none (on the raw content, before <think> is stripped, since those
        tokens were generated too), and becomes the next request's baseline.
        nanobot runner.py:877, :1015, :1254 and :1321.
        """
        tools = spec.tools.get_definitions() if with_tools else None
        _raise_if_cancelling()
        prepared = await self.context_governor.prepare_request(state, messages, tool_definitions=tools)
        kwargs: dict[str, Any] = {
            "messages": prepared,
            "tools": tools or None,
            "model": spec.model,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
        }
        # A dependency that swallowed a cancellation must not start or finish
        # model work (nanobot runner.py:919-1035 and :1257-1273).
        _raise_if_cancelling()
        events_token = _request_events.set(spec.events)
        try:
            response = await spec.provider.chat_stream_with_retry(
                **kwargs,
                **(stream or {}),
                retry_mode=spec.provider_retry_mode,
                on_retry_wait=self._retry_wait_callback(spec),
            )
        finally:
            _request_events.reset(events_token)
        _raise_if_cancelling()
        response.usage = self._usage_or_estimate(spec.provider, response, kwargs)
        # Another model's count cannot bound the next request (nanobot
        # runner.py:1340; NanoClaw has no provider-native compaction).
        state.usage = None if response.is_fallback else response.usage
        return response

    @staticmethod
    def _usage_or_estimate(provider: LLMProvider, response: LLMResponse, kwargs: dict[str, Any]) -> LLMUsage:
        """The request's usage; estimated if missing, empty for a failed request.

        (nanobot runner.py:1297)
        """
        usage = response.usage
        if response.is_error:
            return usage if usage is not None and usage.total_tokens > 0 else LLMUsage.empty_request()
        return provider.usage_for_call(response, kwargs) or LLMUsage.empty_request()

    @staticmethod
    def _stream_callbacks(
        hook: AgentHook, context: AgentHookContext
    ) -> tuple[dict[str, Any], Callable[[], Awaitable[None]]]:
        """Delta callbacks for a streaming hook, and the closer for native reasoning.

        Content deltas go to ``hook.on_stream`` (the progress hook separates
        inline <think> text there); native thinking deltas become
        ``emit_reasoning`` chunks; a recovered stream ends the segment with
        ``on_stream_end(resuming=True)``; provider-hosted tool events go to
        ``on_provider_tool_event``. (nanobot runner.py:925-1018, without the
        timing fields and the shielded close task.)
        """
        thinking_buf = ""
        native_reasoning_open = False

        async def close_native_reasoning() -> None:
            nonlocal native_reasoning_open
            if native_reasoning_open:
                native_reasoning_open = False
                await hook.emit_reasoning_end()

        async def on_content_delta(delta: str) -> None:
            if delta:
                context.streamed_content = True
                await close_native_reasoning()
            await hook.on_stream(context, delta)

        async def on_thinking_delta(delta: str) -> None:
            nonlocal thinking_buf, native_reasoning_open
            if not delta:
                return
            prev_clean = strip_reasoning_tags(thinking_buf)
            thinking_buf += delta
            incremental = strip_reasoning_tags(thinking_buf)[len(prev_clean):]
            if incremental:
                context.streamed_reasoning = True
                native_reasoning_open = True
                await hook.emit_reasoning(incremental)

        async def on_tool_call_delta(event: dict[str, Any]) -> None:
            # Only provider-hosted tools are reported; function-call deltas
            # are assembled by the provider (nanobot runner.py:978).
            if event.get("kind") != "hosted_tool":
                return
            await close_native_reasoning()
            await hook.on_provider_tool_event(context, event)

        async def on_stream_recover() -> None:
            await close_native_reasoning()
            await hook.on_stream_end(context, resuming=True)

        callbacks = {
            "on_content_delta": on_content_delta,
            "on_thinking_delta": on_thinking_delta,
            "on_tool_call_delta": on_tool_call_delta,
            "on_stream_recover": on_stream_recover,
        }
        return callbacks, close_native_reasoning

    async def _request_model(
        self,
        spec: AgentRunSpec,
        state: ModelRequestState,
        messages: list[dict[str, Any]],
        hook: AgentHook,
        context: AgentHookContext,
        *,
        malformed_retry: bool = False,
    ) -> tuple[LLMResponse, LLMUsage | None]:
        """Request with tools, recovering from tool calls that have no name.

        Mirrors nanobot runner.py _request_model: nameless calls are dropped.
        If that leaves none, the request is retried once with a note (added
        to the request as sent, nanobot runner.py:1076); if the retry is
        malformed too, one final request is made with tools off.
        """
        stream: dict[str, Any] | None = None
        close_native_reasoning: Callable[[], Awaitable[None]] | None = None
        if hook.wants_streaming():
            stream, close_native_reasoning = self._stream_callbacks(hook, context)
        try:
            response = await self._request(spec, state, messages, with_tools=True, stream=stream)
        finally:
            if close_native_reasoning is not None:
                await close_native_reasoning()
        usage = response.usage
        dropped, all_dropped, original_finish = self._drop_malformed_tool_calls(response)
        if not all_dropped or original_finish not in ("tool_calls", "function_call"):
            return response, usage

        assert state.messages is not None
        retry_messages = state.messages + [build_malformed_tool_call_note(response.content)]
        if not malformed_retry:
            logger.warning(
                "Retrying LLM request after all {} malformed tool call(s) were dropped",
                dropped,
            )
            retry, retry_usage = await self._request_model(
                spec, state, retry_messages, hook, context, malformed_retry=True
            )
            return retry, _add_usage(usage, retry_usage)
        logger.warning(
            "Malformed tool calls persisted after retry; falling back to no-tools request",
        )
        fallback = await self._request(spec, state, retry_messages, with_tools=False)
        return fallback, _add_usage(usage, fallback.usage)

    @staticmethod
    def _drop_malformed_tool_calls(response: LLMResponse) -> tuple[int, bool, str]:
        """Remove tool calls with an empty name.

        Returns (dropped_count, all_dropped, original_finish_reason).

        A nameless call can't run, and if saved to history the API would reject
        every later request in the conversation. (nanobot runner.py:1068)
        """
        original_finish = response.finish_reason
        if not response.tool_calls:
            return 0, False, original_finish
        valid = [tc for tc in response.tool_calls if tc.has_valid_name()]
        if len(valid) == len(response.tool_calls):
            return 0, False, original_finish
        dropped = len(response.tool_calls) - len(valid)
        logger.warning(
            "Dropped {} malformed tool call(s) with missing/non-string name "
            "from LLM response (finish_reason={!r})",
            dropped,
            original_finish,
        )
        response.tool_calls = valid
        if not valid:
            response.finish_reason = "stop"
        return dropped, not valid, original_finish

    @staticmethod
    def _model_tool_results(model_messages: list[dict[str, Any]]) -> Callable[[], Mapping[str, str]]:
        """The tool results in the request the model just answered, by call id.

        Handed to each ``read_file`` call (``file_read_context``), so a repeat
        read only returns ``[File unchanged since last read: ...]`` while the
        earlier result is really in the model's input: not after compaction
        summarized it away, nor after it was truncated or replaced by the
        pending-read notice. Indexed once, on the first check in a batch.
        (nanobot execution.py:62-72; NanoClaw has no provider-native
        compaction, so there is no ``compacted_tool_results`` set to exclude.)
        """

        @cache
        def read_results() -> dict[str, str]:
            return {
                message["tool_call_id"]: message["content"]
                for message in model_messages
                if message.get("role") == "tool"
                and isinstance(message.get("tool_call_id"), str)
                and isinstance(message.get("content"), str)
            }

        return read_results

    async def _execute_tool_call(
        self,
        spec: AgentRunSpec,
        call: ToolCallRequest,
        workspace_violation_counts: dict[str, int] | None = None,
        *,
        hook: AgentHook | None = None,
        context: AgentHookContext | None = None,
        read_results: Callable[[], Mapping[str, str]] | None = None,
    ) -> tuple[Any, dict[str, str]]:
        """Run one call; return its result and its tool event.

        The result is the tool's raw output on success, else the error text
        for the model. The event is ``{"name", "status": "ok"|"error",
        "detail"}``. Hooks: ``before_execute_tool``, then
        ``on_execute_tool_error`` or ``after_execute_tool``. Mirrors nanobot
        agent/tools/execution.py:110-213; boundary errors (private network,
        outside the workspace) are reported as nanobot does
        (``classify_violation``).
        """
        counts = workspace_violation_counts if workspace_violation_counts is not None else {}
        hook = hook or AgentHook()
        context = context or AgentHookContext(iteration=0, messages=[])

        def failed(raw_text: str, payload: str, detail: str) -> tuple[str, dict[str, str]]:
            event = {"name": call.name, "status": "error", "detail": detail}
            handled = classify_violation(
                raw_text=raw_text,
                soft_payload=payload,
                tool_name=call.name,
                arguments=call.arguments,
                workspace_violation_counts=counts,
            )
            if handled is None:
                return payload, event
            # nanobot execution.py:242-271 labels boundary errors in the event.
            if is_ssrf_violation(raw_text):
                prefix = "ssrf_violation: "
            elif handled != payload:
                prefix = "workspace_violation_escalated: "
            else:
                prefix = "workspace_violation: "
            event["detail"] = _event_detail(prefix, raw_text)
            return handled, event

        tool, params, error = spec.tools.prepare_call(call.name, call.arguments)
        if error:
            return failed(error, with_retry_hint(error), error.split(": ", 1)[-1][:120])
        assert tool is not None
        await hook.before_execute_tool(context, call, tool, params)
        try:
            # read_file sees what the model's input holds (nanobot execution.py:159-162).
            with (
                file_read_context(call.id, read_results)
                if call.name == "read_file" and read_results is not None
                else nullcontext()
            ):
                result = await tool.execute(**params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await hook.on_execute_tool_error(context, call, tool, params, exc)
            return failed(str(exc), with_retry_hint(f"Error: {type(exc).__name__}: {exc}"), str(exc))
        if is_tool_error(result):
            await hook.on_execute_tool_error(context, call, tool, params, result)
            return failed(str(result), with_retry_hint(str(result)), _short_detail(str(result)))
        await hook.after_execute_tool(context, call, tool, params, result)
        return result, {"name": call.name, "status": "ok", "detail": _ok_detail(result)}

    async def _finalize_after_max_iterations(
        self,
        spec: AgentRunSpec,
        state: ModelRequestState,
        hook: AgentHook,
        messages: list[dict[str, Any]],
    ) -> tuple[str, LLMUsage | None]:
        """Ask for a summary with tools off (nanobot runner.py:1196 ``_try_finalize_after_max_iterations``)."""
        if not spec.finalize_on_max_iterations:
            return self._max_iterations_fallback(spec), None
        response = await self._request(
            spec,
            state,
            state.compaction.request_messages(messages)
            + [{"role": "user", "content": BUDGET_EXHAUSTED_FINALIZATION_PROMPT}],
            with_tools=False,
        )
        if response.is_error or response.has_tool_calls:
            logger.warning(
                "Budget-exhausted finalization returned finish_reason='{}' "
                "with {} tool call(s) for {}; using fallback",
                response.finish_reason,
                len(response.tool_calls),
                spec.session_key or "default",
            )
            return self._max_iterations_fallback(spec), response.usage
        context = AgentHookContext(
            iteration=spec.max_iterations,
            messages=messages,
            response=response,
            usage=response.usage,
            session_key=spec.session_key,
        )
        clean = hook.finalize_content(context, response.content)
        if is_blank(clean):
            return self._max_iterations_fallback(spec), response.usage
        return clean or "", response.usage

    @staticmethod
    def _max_iterations_fallback(spec: AgentRunSpec) -> str:
        """The fixed out-of-iterations message. (nanobot runner.py:1249)"""
        return (spec.max_iterations_message or MAX_ITERATIONS_MESSAGE).format(max_iterations=spec.max_iterations)

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict[str, Any]]) -> None:
        """Close the turn with an assistant message so the history can be replayed."""
        last = messages[-1] if messages else None
        if last and last.get("role") == "assistant" and not last.get("tool_calls"):
            return
        messages.append(build_assistant_message(MODEL_ERROR_PLACEHOLDER))
