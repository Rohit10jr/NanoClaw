"""Shaping and sizing every request sent to the model.

Mirrors nanobot/agent/context_governance.py (c4dac64). Before EVERY model
request of a run (the first one, the ones between tool calls, retries and the
final summary request; in subagents too) the runner calls
``ContextGovernor.prepare_request``, which:

1. Builds the model-facing copy of the transcript (``prepare_messages_for_model``):
   removes ``[Previous assistant message omitted.]`` placeholders, tool calls
   without a name, tool results without a call (or answered twice); adds an
   error result for a call whose result is missing; keeps every tool result
   within the tool-result budget; merges adjacent user messages. The saved
   transcript is never changed by this.
2. Measures the request (``request_pressure``): the provider's own count when
   the previous request was this one or a prefix of it, else tiktoken.
3. If it is at or over the input budget, compacts (``_compact_request_history``):
   the history the model already accepted (H) is summarized, and the request
   becomes ``[system prompt + summary] + continuation + Δ`` (Δ: everything new
   since). If it still doesn't fit, the largest unread ``read_file`` results
   in Δ are replaced by a notice (``_fit_pending_file_reads``). The summary is
   returned as a :class:`SessionSummaryCheckpoint`, which the loop commits
   where H ends when it saves the turn.

H and Δ (``ContextCompactionState``): after each request, its exact
model-facing copy becomes the new H (``accept_request``); later requests send
H plus the raw messages added since.

Which summary is written depends on the bug 4 switch (``NANOCLAW_COMPACTION``,
docs/notes/06-memory.md "Compaction A/B"): the caller's ``consolidate_history``
is ``Consolidator.summarize_transcript``, whose archiver either raw-dumps an
archive request that doesn't fit (``nanobot``) or summarizes it in parts
(``fixed``).

Left out of nanobot's module: provider conversation state (Responses-API
``previous_response_id`` and provider-native compaction,
``summarize_provider_compaction``), and running the preparation in a worker
thread (``prepare_in_background``; NanoClaw runs it inline, same results).
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanoclaw.agent.tool_results import OFFLOAD_EXEMPT_TOOLS, normalize_tool_result
from nanoclaw.events import NO_EVENTS, ContextCompactionEvent, EventSink
from nanoclaw.runtime_context import (
    RUNTIME_CONTEXT_MESSAGE_META,
    detach_runtime_context,
    reattach_runtime_context,
)
from nanoclaw.session.summary import (
    SUMMARY_CONTINUATION_TEXT,
    SessionSummaryCheckpoint,
    is_hidden_history_message,
)
from nanoclaw.utils.tokens import estimate_prompt_tokens

# Headroom for token-estimate drift (nanobot providers/base.py).
CONTEXT_SAFETY_BUFFER = 1024
BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"
PLACEHOLDER_TEXTS = frozenset({"[Previous assistant message omitted.]"})
# nanobot context_governance.py:467
PENDING_READ_NOTICE = (
    "Error: read_file result was not sent because the combined request exceeds "
    "the context budget. No file was modified by this recovery. Read one file at a time "
    "with a smaller offset/limit range; for large single-line JSON, use exec "
    "to select only the needed records or fields."
)

# Summarizes H: (accepted messages, previous summary) -> summary or None.
# The loop and subagents pass ``Consolidator.summarize_transcript``.
HistoryConsolidator = Callable[[list[dict[str, Any]], str | None], Awaitable[str | None]]
# Rebuilds the stable system prefix around a new summary.
SummaryTranscriptBuilder = Callable[[str], list[dict[str, Any]]]


class ContextWindowExceededError(RuntimeError):
    """The request is still too big for the model after compacting. (nanobot :79)"""

    def __init__(self, *, session_key: str | None, estimated_tokens: int, input_budget: int, source: str) -> None:
        self.session_key = session_key
        self.estimated_tokens = estimated_tokens
        self.input_budget = input_budget
        self.source = source
        super().__init__(
            "Model input exceeds the local context budget "
            f"for {session_key or 'default'}: {estimated_tokens}/{input_budget} via {source}"
        )


def _request_extends_measured(measured: list[dict[str, Any]], current: list[dict[str, Any]]) -> bool:
    """Whether the measured request is an unchanged prefix of this one. (nanobot :100, D8)"""
    return len(current) >= len(measured) and all(current[i] == measured[i] for i in range(len(measured)))


def _tool_call_name_is_valid(tool_call: Any) -> bool:
    """A saved tool call with a usable name. (nanobot :110)"""
    if not isinstance(tool_call, dict):
        return False
    fn = tool_call.get("function")
    name = fn.get("name") if isinstance(fn, dict) else tool_call.get("name")
    return isinstance(name, str) and bool(name)


@dataclass(slots=True)
class ContextGovernanceConfig:
    """What the governor needs to know about a run. (nanobot :125)"""

    session_key: str | None
    max_tool_result_chars: int
    workspace: Path | None = None
    context_window_tokens: int = 0
    max_tokens: int = 4096


@dataclass(slots=True)
class ContextCompactionState:
    """The accepted history H, kept apart from the unsent delta. (nanobot :137)

    ``raw_messages`` is the runner's own transcript list (the one it returns),
    so the pending-read notice reaches the saved turn.
    """

    raw_messages: list[dict[str, Any]]
    accepted_messages: list[dict[str, Any]]
    raw_accepted_boundary: int
    active_summary: str | None
    summary_transcript_builder: SummaryTranscriptBuilder
    consolidate_history: HistoryConsolidator | None
    summary_checkpoint: SessionSummaryCheckpoint | None = None

    @classmethod
    def from_transcript(
        cls,
        messages: list[dict[str, Any]],
        *,
        accepted_boundary: int,
        active_summary: str | None,
        summary_transcript_builder: SummaryTranscriptBuilder,
        consolidate_history: HistoryConsolidator | None,
    ) -> ContextCompactionState:
        """A turn built from a session: the system prompt and history are H.

        (nanobot ``from_transcript``, :151; ``accepted_boundary`` is its
        ``1 + len(transcript_input.history)``.)
        """
        return cls(
            raw_messages=messages,
            accepted_messages=copy.deepcopy(messages[:accepted_boundary]),
            raw_accepted_boundary=accepted_boundary,
            active_summary=active_summary,
            summary_transcript_builder=summary_transcript_builder,
            consolidate_history=consolidate_history,
        )

    @classmethod
    def from_messages(
        cls, messages: list[dict[str, Any]], consolidate_history: HistoryConsolidator | None
    ) -> ContextCompactionState:
        """A standalone transcript (subagents): all of it is H. (nanobot :192)"""
        instruction_prefix: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") not in {"system", "developer"}:
                break
            instruction_prefix.append(dict(message))

        def build_summary_transcript(summary: str) -> list[dict[str, Any]]:
            archived_context = f"[Archived Context Summary]\n\nPrevious conversation summary:\n{summary}"
            prefix = copy.deepcopy(instruction_prefix)
            for index in range(len(prefix) - 1, -1, -1):
                content = prefix[index].get("content")
                if isinstance(content, str):
                    prefix[index]["content"] = f"{content}\n\n---\n\n{archived_context}"
                    return prefix
            return [{"role": "system", "content": archived_context}, *prefix]

        return cls(
            raw_messages=messages,
            accepted_messages=copy.deepcopy(messages),
            raw_accepted_boundary=len(messages),
            active_summary=None,
            summary_transcript_builder=build_summary_transcript,
            consolidate_history=consolidate_history,
        )

    def request_messages(self, raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """H plus the raw messages added since. (nanobot :232)"""
        return [*copy.deepcopy(self.accepted_messages), *copy.deepcopy(raw_messages[self.raw_accepted_boundary:])]

    def delta_after_accepted(self, request_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """nanobot :241."""
        return copy.deepcopy(request_messages[len(self.accepted_messages):])


@dataclass(slots=True)
class ModelRequestState:
    """Shared by every request of one run. (nanobot :247, without provider state)

    ``messages`` / ``tool_definitions`` are the last request exactly as sent;
    ``usage`` its usage, the baseline for the next measurement.
    """

    config: ContextGovernanceConfig
    compaction: ContextCompactionState
    usage: Any = None  # LLMUsage | None
    messages: list[dict[str, Any]] | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    events: EventSink = field(default=NO_EVENTS)


def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
    if isinstance(left, str) and isinstance(right, str):
        return f"{left}\n\n{right}" if left else right

    def to_blocks(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
        if value is None:
            return []
        return [{"type": "text", "text": str(value)}]

    return to_blocks(left) + to_blocks(right)


def _runtime_marker(message: dict[str, Any]) -> dict[str, Any] | None:
    meta = message.get("_meta")
    marker = meta.get(RUNTIME_CONTEXT_MESSAGE_META) if isinstance(meta, dict) else None
    return marker if isinstance(marker, dict) else None


def _mergeable(message: dict[str, Any]) -> bool:
    """A user message another may merge with. (nanobot :334-343)

    The summary continuation and hidden records (subagent reports) stay
    separate. nanobot also keeps provider-state boundaries apart; NanoClaw has
    no provider conversation state.
    """
    return (
        message.get("role") == "user"
        and message.get("content") != SUMMARY_CONTINUATION_TEXT
        and not is_hidden_history_message(message)
    )


def merge_adjacent_user_messages_for_model(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent user messages; the input list is not changed. (nanobot :326)

    E.g. the user's message (ending in a time block) followed by a "please
    answer" prompt becomes one message: text, prompt, then the time block.
    """
    prepared: list[dict[str, Any]] = []
    for source in messages:
        message = copy.deepcopy(source)
        if not (prepared and _mergeable(message) and _mergeable(prepared[-1])):
            prepared.append(message)
            continue
        merged = dict(prepared[-1])
        left_marker, right_marker = _runtime_marker(merged), _runtime_marker(message)
        left = (
            detach_runtime_context(merged.get("content"), left_marker)
            if left_marker is not None
            else (merged.get("content"), [], [])
        )
        right = (
            detach_runtime_context(message.get("content"), right_marker)
            if right_marker is not None
            else (message.get("content"), [], [])
        )
        if left is not None and right is not None:
            content = _merge_message_content(left[0], right[0])
            blocks = [*left[2], *right[2]]
            if blocks:
                content, marker = reattach_runtime_context(content, [*left[1], *right[1]], blocks)
                meta = dict(merged.get("_meta") or {})
                for key, value in (message.get("_meta") or {}).items():
                    meta.setdefault(key, value)
                meta[RUNTIME_CONTEXT_MESSAGE_META] = marker
                merged["_meta"] = meta
            merged["content"] = content
        else:
            merged["content"] = _merge_message_content(merged.get("content"), message.get("content"))
        prepared[-1] = merged
    return prepared


def _ensure_nonempty_tool_result(tool_name: str, content: Any) -> Any:
    """nanobot utils/runtime.py:48 (NanoClaw's tool results are strings; lists come from old history)."""
    marker = f"({tool_name} completed with no output)"
    if content is None or (isinstance(content, str) and not content.strip()):
        return marker
    if isinstance(content, list):
        if not content:
            return marker
        texts = [b.get("text") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if len(texts) == len(content) and all(isinstance(t, str) for t in texts) and not "".join(texts).strip():
            return marker
    return content


class ContextGovernor:
    """Owns the model-facing request; never changes the saved history. (nanobot :262)"""

    # -- H/Δ ------------------------------------------------------------------

    @staticmethod
    def accept_request(state: ModelRequestState, model_messages: list[dict[str, Any]], *, raw_boundary: int) -> None:
        """The request just sent becomes H. (nanobot :297)"""
        state.compaction.accepted_messages = copy.deepcopy(model_messages)
        state.compaction.raw_accepted_boundary = raw_boundary

    # -- the model-facing copy --------------------------------------------------

    def prepare_messages_for_model(
        self, config: ContextGovernanceConfig, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """The normalized model-facing copy of a raw transcript. (nanobot :412)"""
        return merge_adjacent_user_messages_for_model(self.prepare_for_model(config, messages))

    def prepare_for_model(self, config: ContextGovernanceConfig, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """nanobot :421."""
        updated = self.strip_placeholder_assistant_messages(messages)
        updated = self.strip_malformed_tool_calls(updated)
        updated = self.drop_orphan_tool_results(updated)
        updated = self.backfill_missing_tool_results(updated)
        return self.apply_tool_result_budget(config, updated)

    # -- measuring ----------------------------------------------------------------

    @staticmethod
    def input_budget(config: ContextGovernanceConfig) -> int:
        """Window minus the reply and a safety margin; 0 means no budget. (nanobot :815)"""
        if not config.context_window_tokens:
            return 0
        budget = config.context_window_tokens - config.max_tokens - CONTEXT_SAFETY_BUFFER
        return budget if budget > 0 else 0

    def ensure_request_fits(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Check an exact request without dropping anything. (nanobot :432)"""
        if not config.context_window_tokens:
            return messages
        budget = self.input_budget(config)
        estimated, source = estimate_prompt_tokens(messages, tool_definitions)
        if budget > 0 and estimated <= budget:
            return messages
        raise ContextWindowExceededError(
            session_key=config.session_key, estimated_tokens=estimated, input_budget=budget, source=source
        )

    def request_pressure(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        usage: Any,
        *,
        usage_matches_messages: bool,
        usage_prefixes_messages: bool,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> tuple[int, str] | None:
        """The measurement, when the request is at or over budget. (nanobot :506)

        The provider's count is used when it measured exactly this request,
        or a prefix of it and counted more than the estimate (D8, faf1efa).
        """
        if not config.context_window_tokens:
            return None
        budget = self.input_budget(config)
        if usage_matches_messages and usage is not None and usage.context_tokens is not None:
            measured, source = usage.context_tokens, "matching provider usage"
        else:
            measured, source = estimate_prompt_tokens(messages, tool_definitions)
            if (
                usage_prefixes_messages
                and usage is not None
                and usage.reported_tokens > 0
                and usage.context_tokens is not None
                and usage.context_tokens > measured
            ):
                measured, source = usage.context_tokens, "prior provider usage plus pending messages"
        if budget > 0 and measured < budget:
            return None
        return measured, source

    # -- compacting ------------------------------------------------------------------

    def _fit_pending_file_reads(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        delta_messages: list[dict[str, Any]],
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Replace the largest unread read_file results until the request fits. (nanobot :458, D9 c9bd5eb)

        Returns the request and ``{tool_call_id: notice}``; raises
        ContextWindowExceededError when nothing is left to replace.
        """
        pending_ids = {
            message.get("tool_call_id")
            for message in delta_messages
            if message.get("role") == "tool"
            and message.get("name") == "read_file"
            and isinstance(message.get("tool_call_id"), str)
        }
        candidates = iter(
            sorted(
                (
                    index
                    for index, message in enumerate(messages)
                    if message.get("role") == "tool"
                    and message.get("tool_call_id") in pending_ids
                    and isinstance(message.get("content"), str)
                    and len(message["content"].encode("utf-8")) > len(PENDING_READ_NOTICE)
                ),
                key=lambda index: len(messages[index]["content"].encode("utf-8")),
                reverse=True,
            )
        )
        prepared = list(messages)
        replacements: dict[str, str] = {}
        while True:
            try:
                return self.ensure_request_fits(config, prepared, tool_definitions=tool_definitions), replacements
            except ContextWindowExceededError:
                index = next(candidates, None)
                if index is None:
                    raise
                # Only unread results change, and the saved turn gets the same
                # text, so a later replay never shows what the model never saw.
                replacements[prepared[index]["tool_call_id"]] = PENDING_READ_NOTICE
                prepared[index] = {**prepared[index], "content": PENDING_READ_NOTICE}

    async def _compact_request_history(
        self,
        state: ModelRequestState,
        messages: list[dict[str, Any]],
        pressure: tuple[int, str],
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Replace H with a summary, keeping Δ. (nanobot :623)

        Publishes ContextCompactionEvent (reason ``context_pressure``) started,
        then succeeded, failed or cancelled.
        """
        compaction = state.compaction
        measured, source = pressure
        compaction_id = uuid.uuid4().hex
        await state.events.emit(ContextCompactionEvent(compaction_id, "started", reason="context_pressure"))
        try:
            delta_messages = compaction.delta_after_accepted(messages)
            consolidation_prefix = self.prepare_messages_for_model(state.config, compaction.accepted_messages)
            summary = (
                await compaction.consolidate_history(copy.deepcopy(consolidation_prefix), compaction.active_summary)
                if compaction.consolidate_history is not None
                else None
            )
            if not summary:
                raise ContextWindowExceededError(
                    session_key=state.config.session_key,
                    estimated_tokens=measured,
                    input_budget=self.input_budget(state.config),
                    source=source,
                )
            compaction.active_summary = summary
            # New user input defines the next task; only ongoing work without
            # it needs the instruction to resume after H is replaced.
            continuation = (
                []
                if any(message.get("role") == "user" for message in delta_messages)
                else [{"role": "user", "content": SUMMARY_CONTINUATION_TEXT}]
            )
            prepared = self.prepare_messages_for_model(
                state.config,
                [*compaction.summary_transcript_builder(summary), *continuation, *delta_messages],
            )
            prepared, replacements = self._fit_pending_file_reads(
                state.config, prepared, delta_messages, tool_definitions=tool_definitions
            )
            # Commit only once the whole request fits. Earlier results and all
            # call/result pairs stay as they are.
            for message in compaction.raw_messages[compaction.raw_accepted_boundary:]:
                if message.get("role") == "tool" and message.get("tool_call_id") in replacements:
                    message["content"] = replacements[message["tool_call_id"]]
            state.usage = None
            compaction.summary_checkpoint = SessionSummaryCheckpoint(
                summary=summary, transcript_boundary=compaction.raw_accepted_boundary
            )
        except (Exception, asyncio.CancelledError) as exc:
            phase = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            await state.events.emit(ContextCompactionEvent(compaction_id, phase, reason="context_pressure"))
            raise
        await state.events.emit(ContextCompactionEvent(compaction_id, "succeeded", reason="context_pressure"))
        return prepared

    async def prepare_request(
        self,
        state: ModelRequestState,
        messages: list[dict[str, Any]],
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """The exact messages to send, compacted if needed; recorded in ``state``. (nanobot :716)"""
        prepared = self.prepare_messages_for_model(state.config, messages)
        usage_matches_messages = (
            state.messages is not None and prepared == state.messages and tool_definitions == state.tool_definitions
        )
        usage_prefixes_messages = (
            state.messages is not None
            and tool_definitions == state.tool_definitions
            and _request_extends_measured(state.messages, prepared)
        )
        pressure = self.request_pressure(
            state.config,
            prepared,
            state.usage,
            usage_matches_messages=usage_matches_messages,
            usage_prefixes_messages=usage_prefixes_messages,
            tool_definitions=tool_definitions,
        )
        if pressure is not None:
            prepared = await self._compact_request_history(state, messages, pressure, tool_definitions=tool_definitions)
        state.messages, state.tool_definitions = copy.deepcopy(prepared), copy.deepcopy(tool_definitions)
        return prepared

    # -- tool results ---------------------------------------------------------------

    @staticmethod
    def normalize_tool_result(config: ContextGovernanceConfig, tool_call_id: str, tool_name: str, result: Any) -> Any:
        """The empty-result marker, then the tool-result budget. (nanobot :831)

        A result over ``max_tool_result_chars`` is saved to the workspace and
        replaced by a preview (truncated without a workspace); read_file is
        exempt. Text blocks of a list result are handled one by one.
        """
        bucket = config.session_key or "default"
        if isinstance(result, str):
            return normalize_tool_result(
                tool_name,
                tool_call_id,
                result,
                workspace=config.workspace,
                max_chars=config.max_tool_result_chars,
                bucket=bucket,
            )
        result = _ensure_nonempty_tool_result(tool_name, result)
        if tool_name in OFFLOAD_EXEMPT_TOOLS or not isinstance(result, list):
            return result
        blocks: list[dict[str, Any]] = []
        for block in result:
            if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                return result
            if block["type"] == "text" and not isinstance(block.get("text"), str):
                return result
            blocks.append(block)
        # Image blocks don't change how the text blocks next to them are handled.
        return [
            {
                **block,
                "text": normalize_tool_result(
                    tool_name,
                    f"{tool_call_id}_text_{index}",
                    block["text"],
                    workspace=config.workspace,
                    max_chars=config.max_tool_result_chars,
                    bucket=bucket,
                ),
            }
            if block["type"] == "text"
            else block
            for index, block in enumerate(blocks)
        ]

    def apply_tool_result_budget(
        self, config: ContextGovernanceConfig, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Every tool result within budget (also ones from older turns). (nanobot :1052)"""
        updated = messages
        for index, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            normalized = self.normalize_tool_result(
                config,
                str(message.get("tool_call_id") or f"tool_{index}"),
                str(message.get("name") or "tool"),
                message.get("content"),
            )
            if normalized != message.get("content"):
                if updated is messages:
                    updated = [dict(m) for m in messages]
                updated[index]["content"] = normalized
        return updated

    # -- transcript repair (model-facing copy only) ------------------------------------

    @staticmethod
    def strip_placeholder_assistant_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop ``[Previous assistant message omitted.]`` placeholders. (nanobot :884)

        They carry nothing and can make a model repeat failed tool calls.
        Returns the same list when nothing changes.
        """
        updated: list[dict[str, Any]] | None = None
        for index, message in enumerate(messages):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                text = content if isinstance(content, str) else ""
                if text.strip() in PLACEHOLDER_TEXTS and not message.get("tool_calls"):
                    if updated is None:
                        updated = list(messages[:index])
                    logger.debug("Stripping placeholder assistant message from history: {!r}", text[:60])
                    continue
            if updated is not None:
                updated.append(message)
        return messages if updated is None else updated

    @staticmethod
    def strip_malformed_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop saved tool calls without a name. (nanobot :922)

        A nameless call makes providers reject every later request; once it
        is gone, ``drop_orphan_tool_results`` removes its result. An assistant
        message left with neither text nor calls is dropped too.
        """
        updated: list[dict[str, Any]] | None = None
        for index, message in enumerate(messages):
            calls = message.get("tool_calls") if message.get("role") == "assistant" else None
            kept = [call for call in calls if _tool_call_name_is_valid(call)] if calls else None
            if kept is None or len(kept) == len(calls):
                if updated is not None:
                    updated.append(message)
                continue
            if updated is None:
                updated = [dict(m) for m in messages[:index]]
            logger.warning(
                "Stripping {} malformed tool_call(s) with missing/non-string name from assistant history before request",
                len(calls) - len(kept),
            )
            repaired = dict(message)
            if kept:
                repaired["tool_calls"] = kept
            else:
                repaired.pop("tool_calls", None)
            if not kept and not repaired.get("content"):
                continue
            updated.append(repaired)
        return messages if updated is None else updated

    @staticmethod
    def drop_orphan_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop tool results without an earlier call, or answering one twice. (nanobot :977)"""
        declared: set[str] = set()
        fulfilled: set[str] = set()
        updated: list[dict[str, Any]] | None = None
        for index, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    if isinstance(call, dict) and call.get("id"):
                        declared.add(str(call["id"]))
            if role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                if not call_id or call_id not in declared or call_id in fulfilled:
                    if updated is None:
                        updated = [dict(m) for m in messages[:index]]
                    continue
                fulfilled.add(call_id)
            if updated is not None:
                updated.append(dict(message))
        return messages if updated is None else updated

    @staticmethod
    def backfill_missing_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add an error result for each call whose result is missing. (nanobot :1008)"""
        declared: list[tuple[int, str, str]] = []
        fulfilled: set[str] = set()
        for index, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    if isinstance(call, dict) and call.get("id"):
                        func = call.get("function")
                        name = ""
                        if isinstance(func, dict):
                            raw_name = func.get("name", "")
                            name = raw_name if isinstance(raw_name, str) else str(raw_name)
                        declared.append((index, str(call["id"]), name))
            elif role == "tool" and message.get("tool_call_id"):
                fulfilled.add(str(message["tool_call_id"]))

        missing = [(index, call_id, name) for index, call_id, name in declared if call_id not in fulfilled]
        if not missing:
            return messages
        updated = list(messages)
        offset = 0
        for assistant_index, call_id, name in missing:
            insert_at = assistant_index + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(
                insert_at, {"role": "tool", "tool_call_id": call_id, "name": name, "content": BACKFILL_CONTENT}
            )
            offset += 1
        return updated
