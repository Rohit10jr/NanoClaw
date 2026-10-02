"""One conversation turn: load the session, run the agent, save the result.

Mirrors nanobot/agent/loop.py (``AgentLoop.process_direct`` and the turn stages
``_build_turn``, ``_run_turn``, ``_persist_turn`` and ``_save_turn``). nanobot's
loop also serves a message bus with many chat channels, queues, cancellation
and crash recovery; here only the direct path is kept: text in, reply out.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any

from nanoclaw.agent.context import ContextBuilder
from nanoclaw.agent.messages import EMPTY_FINAL_RESPONSE_MESSAGE, is_blank
from nanoclaw.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec
from nanoclaw.agent.tools import ToolRegistry, build_default_tools
from nanoclaw.providers import LLMProvider
from nanoclaw.runtime_context import (
    RUNTIME_CONTEXT_HISTORY_META,
    RUNTIME_CONTEXT_MESSAGE_META,
    RuntimeContextBlock,
    append_runtime_context,
    current_time_block,
)
from nanoclaw.session import Session, SessionManager


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
        max_iterations: int = 20,
        max_tool_result_chars: int = 16_000,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.model = model
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.context = ContextBuilder(workspace)
        self.tools = tools if tools is not None else build_default_tools(workspace)
        self.sessions = sessions if sessions is not None else SessionManager(workspace)
        self.runner = AgentRunner()
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
    ) -> AgentRunResult:
        """Run one turn in the session and return the result.

        Turns in the same session run one at a time.
        """
        async with self._get_session_lock(session_key):
            # Build: history first, then save the user's message before running,
            # so it survives a crash or an error mid-turn.
            session = self.sessions.get_or_create(session_key)
            history = session.get_history()
            blocks = [current_time_block()]
            persisted_early = self._persist_user_message_early(content, session, blocks)
            messages = self.context.build_messages(
                history, content, channel=channel, runtime_context_blocks=blocks
            )

            # Run.
            result = await self.runner.run(
                AgentRunSpec(
                    initial_messages=messages,
                    tools=self.tools,
                    provider=self.provider,
                    model=self.model,
                    max_iterations=self.max_iterations,
                    max_tool_result_chars=self.max_tool_result_chars,
                    workspace=self.workspace,
                )
            )

            # Persist: everything the runner added after the user's message.
            skip = len(messages) if persisted_early else len(messages) - 1
            self._save_turn(session, result.messages, skip)
            self.sessions.save(session)

        if is_blank(result.final_content):
            result = dataclasses.replace(result, final_content=EMPTY_FINAL_RESPONSE_MESSAGE)
        return result

    def new_session(self, session_key: str = "cli:direct") -> None:
        """Start the conversation over (nanobot's ``/new`` command)."""
        session = self.sessions.get_or_create(session_key)
        session.clear()
        self.sessions.save(session)
        self.sessions.invalidate(session_key)

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
        self.sessions.save(session)
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

