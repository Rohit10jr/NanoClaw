"""One conversation turn: load the session, run the agent, save the result.

Mirrors nanobot/agent/loop.py (``AgentLoop.process_direct`` and the turn stages
``_build_turn``, ``_run_turn``, ``_persist_turn`` and ``_save_turn``). nanobot's
loop also serves a message bus with many chat channels, queues, cancellation
and crash recovery; here only the direct path is kept: text in, reply out.

Since layer 6 a turn also compacts the conversation when it no longer fits the
model's context window, and the loop runs nanobot's ``/compact``, ``/new``
(with archiving) and ``/dream`` commands.

Since layer 8 the main agent has the ``spawn`` tool. A background subagent's
report becomes a follow-up turn (``process_subagent_result``): saved as an
assistant message and shown to the model as new input, so the main agent can
tell the user about it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.context_governance import merge_adjacent_user_messages_for_model
from nanoclaw.agent.memory import COMPACTION_NANOBOT, Consolidator, MemoryStore
from nanoclaw.agent.messages import EMPTY_FINAL_RESPONSE_MESSAGE, is_blank
from nanoclaw.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec, ProgressCallback
from nanoclaw.agent.subagent import (
    DEFAULT_MAX_CONCURRENT_SUBAGENTS,
    SUBAGENT_RESULT_EVENT,
    SubagentManager,
    SubagentResult,
)
from nanoclaw.agent.tools import ToolRegistry, build_default_tools
from nanoclaw.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanoclaw.agent.tools.spawn import SpawnTool
from nanoclaw.providers import LLMProvider
from nanoclaw.runtime_context import (
    RUNTIME_CONTEXT_HISTORY_META,
    RUNTIME_CONTEXT_MESSAGE_META,
    RuntimeContextBlock,
    append_runtime_context,
    current_time_block,
    iteration_limit_block,
    record_iteration_limit,
)
from nanoclaw.session import Session, SessionManager
from nanoclaw.session.recovery import PENDING_USER_TURN_KEY, restore_pending_interruption
from nanoclaw.session.summary import SessionSummary, session_summary_from_metadata
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.tokens import estimate_prompt_tokens

DREAM_NO_INPUT_MESSAGE = "\n".join([
    "Dream has no conversation history to process yet.",
    "",
    "Dream reads new entries from `memory/history.jsonl` after the current Dream cursor.",
    "Conversations only reach that file when they are compacted (`nanoclaw compact`, or",
    "automatically when they grow too long) or cleared with `nanoclaw ask --new`.",
])
# Headroom for token-estimate drift (nanobot providers/base.py).
CONTEXT_SAFETY_BUFFER = 1024
# nanobot's default is 200,000; NanoClaw's default model, gpt-4o-mini, has 128,000.
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
DEFAULT_SESSION_KEY = "cli:direct"
LAST_USAGE_KEY = "_last_usage"


class ContextWindowExceededError(RuntimeError):
    """The request is still too big for the model after compacting."""

    def __init__(self, *, session_key: str | None, estimated_tokens: int, input_budget: int, source: str) -> None:
        self.session_key = session_key
        self.estimated_tokens = estimated_tokens
        self.input_budget = input_budget
        self.source = source
        super().__init__(
            "Model input exceeds the local context budget "
            f"for {session_key or 'default'}: {estimated_tokens}/{input_budget} via {source}"
        )


class AgentLoop:
    """Runs turns against saved sessions."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        *,
        model: str | None = None,
        tools: ToolRegistry | None = None,
        sessions: SessionManager | None = None,
        # nanobot config max_tool_iterations (schema.py:180); phase 2 decision 7.
        max_iterations: int = 200,
        max_tool_result_chars: int = 16_000,
        max_concurrent_subagents: int = DEFAULT_MAX_CONCURRENT_SUBAGENTS,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
        restrict_to_workspace: bool = False,
        compaction: str = COMPACTION_NANOBOT,
        provider_retry_mode: str = "standard",
    ) -> None:
        self.provider = provider
        # For the main agent's model calls only; subagents and memory archiving
        # retry in standard mode, as in nanobot (loop.py:367, :1346).
        self.provider_retry_mode = provider_retry_mode
        self.restrict_to_workspace = restrict_to_workspace
        self.workspace = workspace
        self.model = model
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_window_tokens = context_window_tokens
        self.context = ContextBuilder(workspace)
        self.memory: MemoryStore = self.context.memory
        self.subagents = SubagentManager(
            workspace,
            max_tool_result_chars=max_tool_result_chars,
            max_iterations=max_iterations,
            max_concurrent_subagents=max_concurrent_subagents,
            restrict_to_workspace=restrict_to_workspace,
        )
        if tools is None:
            # The main agent's tools: the standard set plus spawn. Subagents and
            # Dream build their own sets without it.
            tools = build_default_tools(workspace, restrict_to_workspace=restrict_to_workspace)
            tools.register(SpawnTool(self.subagents))
        self.tools = tools
        self.sessions = sessions if sessions is not None else SessionManager(workspace)
        self.consolidator = Consolidator(
            self.memory, self.sessions, self.context.build_messages, self.tools.get_definitions, compaction=compaction
        )
        self.runner = AgentRunner()
        self.start_time = time.time()
        self._session_locks: dict[str, asyncio.Lock] = {}

    @property
    def runtime(self) -> LLMRuntime:
        return LLMRuntime(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            context_window_tokens=self.context_window_tokens,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        *,
        ephemeral: bool = False,
        tools: ToolRegistry | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> AgentRunResult:
        """Run one turn in the session and return the result.

        Turns in the same session run one at a time. An ``ephemeral`` turn
        starts from an empty session and saves nothing (used by Dream); ``tools``
        replaces the default tool set for this turn. ``on_progress`` receives
        the model's text and tool hints while the turn works (nanobot ``on_progress``).
        """
        result = await self._process_turn(
            content, session_key, channel, ephemeral=ephemeral, tools=tools, on_progress=on_progress
        )
        if is_blank(result.final_content):
            result = dataclasses.replace(result, final_content=EMPTY_FINAL_RESPONSE_MESSAGE)
        return result

    async def process_subagent_result(
        self, report: SubagentResult, *, on_progress: ProgressCallback | None = None
    ) -> AgentRunResult:
        """Let the main agent react to a finished background subagent.

        Mirrors nanobot's system turn for a subagent result (``_build_turn``
        with ``is_subagent``): the report is saved as an assistant message
        (once per task) and given to the model as new input, without the time
        block. The model's reply is what the user sees.
        """
        return await self._process_turn(
            report.content,
            report.session_key or DEFAULT_SESSION_KEY,
            report.channel,
            subagent=report,
            on_progress=on_progress,
        )

    async def next_subagent_result(self) -> SubagentResult | None:
        """The next background subagent report, waiting for running subagents.

        Returns ``None`` when no report is waiting and no subagent is running.
        """
        while True:
            if not self.subagents.results.empty():
                return self.subagents.results.get_nowait()
            running = self.subagents.running_tasks()
            if not running:
                return None
            # A subagent queues its report before its task ends.
            await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

    async def close(self) -> None:
        """Cancel running subagents (nanobot ``AgentLoop.aclose``)."""
        await self.subagents.close()

    async def _process_turn(
        self,
        content: str,
        session_key: str,
        channel: str,
        *,
        ephemeral: bool = False,
        tools: ToolRegistry | None = None,
        subagent: SubagentResult | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> AgentRunResult:
        tools = tools if tools is not None else self.tools
        # Children follow the loop's current iteration limit (nanobot
        # _sync_subagent_runtime_limits).
        self.subagents.max_iterations = self.max_iterations
        async with self._get_session_lock(session_key):
            if ephemeral:
                session = Session(key=session_key)
            else:
                session = self.sessions.get_or_create(session_key)
                # A turn that died before it was saved gets an error reply first.
                if restore_pending_interruption(session):
                    self.sessions.save(session)

            # Build: history first, then save the user's message before running,
            # so it survives a crash or an error mid-turn.
            summary = session_summary_from_metadata(session.metadata, fallback_last_active=session.updated_at)
            history = session.get_history()
            boundary = len(session.messages)
            blocks: list[RuntimeContextBlock] = []
            if subagent is not None:
                # Saved as the agent's own record; it reaches the model as input.
                if self._persist_subagent_followup(session, subagent):
                    self.sessions.save(session)
                persisted_early = True
            else:
                blocks.append(current_time_block())
                # The previous user turn hit the iteration limit: say so, once
                # (nanobot loop.py:663 registers this provider; user turns only, :2253).
                if not ephemeral and (limit_block := iteration_limit_block(session.metadata)) is not None:
                    blocks.append(limit_block)
                # Skills named as $skill-name are part of the saved message too (nanobot
                # adds them while resolving runtime context, before saving).
                skill_context = self.context.skills.build_explicit_skill_runtime_context(content)
                if skill_context is not None and skill_context not in blocks:
                    blocks.append(skill_context)
                persisted_early = not ephemeral and self._persist_user_message_early(content, session, blocks)
            messages = self.context.build_messages(
                history, content, channel=channel, runtime_context_blocks=blocks, session_summary=summary
            )
            # Temporary turns are checked too, but never journal or save a summary.
            messages, checkpoint = await self._compact_if_needed(
                session, messages, history, summary, content, channel, blocks, tools, persist=not ephemeral
            )

            # Run. Tools such as spawn read the turn's session and model settings.
            token = bind_request_context(RequestContext(channel=channel, session_key=session.key, runtime=self.runtime))
            try:
                result = await self.runner.run(
                    AgentRunSpec(
                        initial_messages=messages,
                        tools=tools,
                        provider=self.provider,
                        model=self.model,
                        max_iterations=self.max_iterations,
                        max_tool_result_chars=self.max_tool_result_chars,
                        # Temporary turns never write spill files.
                        workspace=None if ephemeral else self.workspace,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        session_key=session.key,
                        on_progress=on_progress,
                        provider_retry_mode=self.provider_retry_mode,
                    )
                )
            finally:
                reset_request_context(token)

            # Persist: the summary boundary (if compacted), then everything the
            # runner added after the user's message.
            if not ephemeral:
                if checkpoint is not None:
                    session.commit_summary_checkpoint(checkpoint, insert_at=boundary)
                skip = len(messages) if persisted_early else len(messages) - 1
                # Token use of the last turn, for /status (nanobot ``_last_usage``).
                if result.usage is not None:
                    session.metadata[LAST_USAGE_KEY] = result.usage.to_dict()  # nanobot loop.py:2384
                # Only user turns set or clear the iteration-limit flag; a
                # subagent follow-up leaves it alone (nanobot loop.py:2385).
                if subagent is None:
                    record_iteration_limit(session.metadata, result.stop_reason)
                self._save_turn(session, result.messages, skip)
                session.metadata.pop(PENDING_USER_TURN_KEY, None)
                self.sessions.save(session)
        return result

    def _input_budget(self) -> int:
        """Tokens the request may use: window minus the reply and a safety margin."""
        if not self.context_window_tokens:
            return 0
        budget = self.context_window_tokens - self.max_tokens - CONTEXT_SAFETY_BUFFER
        return budget if budget > 0 else 0

    async def _compact_if_needed(
        self,
        session: Session,
        messages: list[dict[str, Any]],
        history: list[dict[str, Any]],
        summary: SessionSummary | None,
        content: str,
        channel: str,
        blocks: list[RuntimeContextBlock],
        tools: ToolRegistry,
        *,
        persist: bool = True,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Summarize the history if the request doesn't fit the context window.

        Returns the request to send and the new summary (``None`` if nothing
        was compacted). Mirrors nanobot's compaction before a model request
        (``ContextGovernor.prepare_request``) at the first request of a turn:
        the history (everything before the new message) is replaced by a
        summary in the system prompt, and the new message is kept. nanobot
        checks before every request of a turn; NanoClaw only before the first
        (see docs/notes/06-memory.md).
        """
        if not self.context_window_tokens:
            return messages, None
        definitions = tools.get_definitions()
        budget = self._input_budget()
        measured, source = estimate_prompt_tokens(merge_adjacent_user_messages_for_model(messages), definitions)
        if budget > 0 and measured < budget:
            return messages, None

        accepted = merge_adjacent_user_messages_for_model(messages[: 1 + len(history)])
        new_summary = await self.consolidator.summarize_transcript(
            accepted,
            summary["text"] if summary else None,
            runtime=self.runtime,
            session_key=session.key,
            tools=definitions,
            persist=persist,
        )
        if not new_summary:
            raise ContextWindowExceededError(
                session_key=session.key, estimated_tokens=measured, input_budget=budget, source=source
            )
        compacted = self.context.build_messages(
            [],
            content,
            channel=channel,
            runtime_context_blocks=blocks,
            session_summary={"text": new_summary, "last_active": datetime.now().astimezone().isoformat()},
        )
        measured, source = estimate_prompt_tokens(merge_adjacent_user_messages_for_model(compacted), definitions)
        if budget <= 0 or measured >= budget:
            raise ContextWindowExceededError(
                session_key=session.key, estimated_tokens=measured, input_budget=budget, source=source
            )
        return compacted, new_summary

    async def new_session(self, session_key: str = "cli:direct") -> None:
        """Start the conversation over (nanobot's ``/new`` command).

        Messages not yet archived are first summarized into the history journal,
        so Dream can still learn from them. nanobot does that in the background
        and ignores failures; NanoClaw waits for it and ignores failures.
        """
        session = self.sessions.get_or_create(session_key)
        snapshot = list(session.messages)
        if session.last_archived < len(snapshot):
            archived = Session(
                key=session.key,
                messages=snapshot,
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata=dict(session.metadata),
                last_archived=session.last_archived,
            )
            try:
                await self.consolidator.archive_session(archived, archive_end=len(snapshot), runtime=self.runtime)
            except Exception:
                pass
        session.clear()
        self.sessions.save(session)
        self.sessions.invalidate(session_key)

    async def compact_session(self, session_key: str = "cli:direct") -> str | None:
        """Summarize the whole conversation now (nanobot's ``/compact`` command).

        Returns the summary, ``""`` if there was nothing new, ``None`` on failure.
        """
        async with self._get_session_lock(session_key):
            return await self.consolidator.compact_idle_session(session_key, runtime=self.runtime)

    async def run_dream(self) -> str:
        """Let Dream fold new journal entries into the memory files (nanobot ``/dream``).

        Runs one temporary turn with Dream's instructions and restricted file
        tools. The Dream cursor only moves forward if the run completed, so an
        interrupted run sees the same entries next time.
        """
        start = time.monotonic()
        result = self.memory.build_dream_prompt()
        if result is None:
            return DREAM_NO_INPUT_MESSAGE
        prompt, last_cursor = result
        # nanobot reads the change from its git store; NanoClaw compares the files.
        files = (self.memory.memory_file, self.memory.soul_file, self.memory.user_file)
        before = [MemoryStore.read_file(f) for f in files]
        try:
            response = await self.process_direct(
                prompt, session_key=MemoryStore.dream_session_key(), ephemeral=True, tools=self.memory.build_dream_tools()
            )
            elapsed = time.monotonic() - start
            if response.stop_reason == "completed":
                self.memory.set_last_dream_cursor(last_cursor)
                if [MemoryStore.read_file(f) for f in files] != before:
                    return f"Dream completed in {elapsed:.1f}s."
                return f"Dream completed in {elapsed:.1f}s; no memory changes."
            return (
                f"Dream did not complete after {elapsed:.1f}s (stop_reason: {response.stop_reason}); "
                "memory cursor was not advanced."
            )
        except Exception as exc:
            return f"Dream failed after {time.monotonic() - start:.1f}s: {exc}"
        finally:
            self.memory.compact_history()

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        return self._session_locks.setdefault(session_key, asyncio.Lock())

    def _persist_user_message_early(
        self, content: str, session: Session, blocks: list[RuntimeContextBlock]
    ) -> bool:
        """Save the user's message, with its runtime context, before the run.

        The message is saved exactly as the model sees it (time included), with a
        marker recording the appended part. Replaying it unchanged keeps earlier
        turns byte-identical between requests, which helps provider caching.
        """
        if not (content.strip() or blocks):
            return False
        text, marker = append_runtime_context(content, blocks)
        extra = {RUNTIME_CONTEXT_HISTORY_META: marker} if marker is not None else {}
        session.add_message("user", text, **extra)
        # Cleared once the turn is saved; still set next time means it crashed.
        session.metadata[PENDING_USER_TURN_KEY] = True
        self.sessions.save(session)
        return True

    @staticmethod
    def _persist_subagent_followup(session: Session, report: SubagentResult) -> bool:
        """Save a subagent report as an assistant message, once per task.

        Returns False if the report is empty or this task's report is already
        saved. (nanobot ``_persist_subagent_followup``)
        """
        if not report.content:
            return False
        task_id = report.task_id
        if task_id and any(
            m.get("injected_event") == SUBAGENT_RESULT_EVENT and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            report.content,
            sender_id="subagent",
            injected_event=SUBAGENT_RESULT_EVENT,
            subagent_task_id=task_id,
        )
        return True

    @staticmethod
    def _save_turn(session: Session, messages: list[dict[str, Any]], skip: int) -> None:
        """Append the turn's new messages to the session.

        Skips empty assistant messages (they confuse later turns) and tool
        results whose call is unknown or already answered (providers reject
        those). Every message gets a timestamp.
        """
        declared = {
            str(call["id"])
            for m in session.messages
            if m.get("role") == "assistant"
            for call in m.get("tool_calls") or []
            if isinstance(call, dict) and call.get("id")
        }
        fulfilled = {
            str(m["tool_call_id"]) for m in session.messages if m.get("role") == "tool" and m.get("tool_call_id")
        }
        for message in messages[skip:]:
            entry = dict(message)
            meta = entry.pop("_meta", None)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue
            if role == "tool":
                call_id = str(entry.get("tool_call_id") or "")
                if not call_id or call_id not in declared or call_id in fulfilled:
                    continue
                fulfilled.add(call_id)
            elif role == "user" and isinstance(meta, dict):
                marker = meta.get(RUNTIME_CONTEXT_MESSAGE_META)
                if isinstance(marker, dict):
                    entry[RUNTIME_CONTEXT_HISTORY_META] = marker
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
            if role == "assistant":
                declared.update(
                    str(call["id"])
                    for call in entry.get("tool_calls") or []
                    if isinstance(call, dict) and call.get("id")
                )
        session.updated_at = datetime.now()

