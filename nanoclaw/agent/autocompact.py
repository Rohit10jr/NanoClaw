"""Idle auto-compaction: summarize sessions nobody has used for a while.

Mirrors nanobot/agent/autocompact.py (c4dac64). On the loop's idle tick
(``AgentLoop._check_expired_sessions_if_due``) every saved session idle for at
least ``session_ttl_minutes`` (config ``agents.defaults.idleCompactAfterMinutes``)
whose transcript still has unarchived messages is compacted in the background
with ``compact_idle_session(reason="idle")``: the same summary checkpoint
``/compact`` writes. The next turn of that session starts from the summary.

Skipped: Dream sessions (``dream:``), subagent sessions (``subagent:``, nanobot
688976b), sessions with a busy worker (``active_session_keys``) and sessions
already being archived. The archive takes the session lock and checks the
idle time again, so a turn that started meanwhile wins (nanobot 9650f4f).

NanoClaw's session store is synchronous (one JSONL file per session), so the
``sessions.state.*`` awaits of nanobot become plain calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Collection
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

from nanoclaw.events import NO_EVENTS, EventSink
from nanoclaw.session.manager import Session, SessionManager
from nanoclaw.session.summary import SessionSummary, session_summary_from_metadata

if TYPE_CHECKING:
    from nanoclaw.agent.memory import Consolidator
    from nanoclaw.utils.llm_runtime import LLMRuntime

SessionEventFactory = Callable[[str], EventSink]


def is_dream_session(key: str) -> bool:
    """The per-run session namespace owned by Dream. (nanobot session/keys.py:13)"""
    return key.startswith("dream:")


class AutoCompact:
    def __init__(
        self,
        sessions: SessionManager,
        consolidator: Consolidator,
        session_ttl_minutes: int = 0,
        bind_events: SessionEventFactory | None = None,
        *,
        session_lock: Callable[[str], asyncio.Lock],
    ):
        self.sessions = sessions
        self.consolidator = consolidator
        self._ttl = session_ttl_minutes
        self._archiving: set[str] = set()
        self._summaries: dict[str, SessionSummary] = {}
        self._bind_events = bind_events
        self._session_lock = session_lock

    def _is_expired(self, ts: datetime | str | None, now: datetime | None = None) -> bool:
        if self._ttl <= 0 or not ts:
            return False
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            current = now or datetime.now()
            if getattr(ts, "tzinfo", None) is not None or current.tzinfo is not None:
                idle_seconds = current.timestamp() - ts.timestamp()
            else:
                idle_seconds = (current - ts).total_seconds()
        except (OSError, OverflowError, TypeError, ValueError):
            # list_sessions() forwards raw saved metadata; an unusable value
            # must not escape the idle scan and stop the agent loop.
            return False
        return idle_seconds >= self._ttl * 60

    async def check_expired(
        self,
        schedule_background: Callable[[Coroutine[Any, Any, None]], None],
        resolve_runtime: Callable[[Session], Awaitable[LLMRuntime]],
        active_session_keys: Collection[str] = (),
    ) -> None:
        """Schedule idle archival without blocking the event loop."""
        now = datetime.now()
        active_keys = set(active_session_keys)
        for info in self.sessions.list_sessions():
            key = info.get("key", "")
            if not key or is_dream_session(key) or key.startswith("subagent:") or key in self._archiving:
                continue
            if key in active_keys or not self._is_expired(info.get("updated_at"), now):
                continue
            if not self.sessions.has_unarchived_messages(key):
                continue
            session = self.sessions.get_or_create(key)
            try:
                runtime = await resolve_runtime(session)
            except (KeyError, ValueError):
                continue
            self._archiving.add(key)
            schedule_background(self._archive(key, runtime=runtime))

    async def _archive(self, key: str, *, runtime: LLMRuntime) -> None:
        if is_dream_session(key):
            self._archiving.discard(key)
            return
        try:
            async with self._session_lock(key):
                # A turn may have refreshed activity while this archive waited.
                session = self.sessions.get_or_create(key)
                if not self._is_expired(session.updated_at):
                    return
                # Idle compaction and turns share one transcript boundary.
                summary = await self.consolidator.compact_idle_session(
                    key,
                    runtime=runtime,
                    events=self._bind_events(key) if self._bind_events else NO_EVENTS,
                    reason="idle",
                )
                if summary:
                    session = self.sessions.get_or_create(key)
                    stored = session_summary_from_metadata(session.metadata, fallback_last_active=session.updated_at)
                    if stored is not None:
                        self._summaries[key] = stored
        except Exception:
            logger.exception("Auto-compact: failed for {}", key)
        finally:
            self._archiving.discard(key)

    async def prepare_session(self, session: Session, key: str) -> tuple[Session, SessionSummary | None]:
        """The summary a turn of ``session`` starts from, if any."""
        if is_dream_session(key):
            self._archiving.discard(key)
            self._summaries.pop(key, None)
            return session, None
        # Hot path: summary from the in-memory dict (process hasn't restarted).
        entry = self._summaries.pop(key, None)
        if entry:
            return session, entry
        # Cold path: summary saved in the session metadata (process restarted).
        return session, session_summary_from_metadata(session.metadata, fallback_last_active=session.updated_at)
