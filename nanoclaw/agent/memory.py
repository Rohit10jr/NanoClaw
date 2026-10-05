"""Long-term memory, the history journal, and compaction.

Mirrors nanobot/agent/memory.py:

- ``MemoryStore``: the files. ``memory/MEMORY.md`` (long-term facts, shown in
  the system prompt) and ``memory/history.jsonl`` (an append-only journal of
  conversation summaries, numbered by a cursor).
- ``MemoryArchiver``: asks the model to summarize a stretch of conversation
  into a checkpoint, falling back to a raw transcript dump if that fails.
- ``Consolidator``: compacts a session, i.e. replaces its old messages with a
  checkpoint so the next request fits the model's context window.

Dream (``AgentLoop.run_dream``) later reads new journal entries and edits
``MEMORY.md``, ``USER.md`` and ``SOUL.md`` with restricted file tools.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from nanoclaw.agent.messages import build_assistant_message, strip_think, truncate_text
from nanoclaw.agent.tools import Tool, ToolRegistry, ToolResult
from nanoclaw.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from nanoclaw.runtime_context import public_history_message
from nanoclaw.session.manager import Session, SessionManager
from nanoclaw.session.summary import is_summary_checkpoint, session_summary_from_metadata
from nanoclaw.utils.llm_runtime import LLMRuntime
from nanoclaw.utils.templates import render_template
from nanoclaw.utils.tokens import estimate_prompt_tokens, truncate_text_to_tokens

_RAW_ARCHIVE_MAX_CHARS = 16_000  # fallback dump (model failed)
_HISTORY_ENTRY_HARD_CAP = 64_000  # emergency cap in append_history
# nanobot security/workspace_policy.py: appended to path-boundary errors.
WORKSPACE_BOUNDARY_NOTE = (
    " (this is a hard policy boundary, not a transient failure; "
    "do not retry with shell tricks or alternative tools, and ask "
    "the user how to proceed if the resource is genuinely required)"
)
_ARCHIVE_TOOL_RESULT = (
    "Session archival does not execute tools. Use only the supplied conversation and "
    "return the requested compact checkpoint now; do not call another tool."
)


class MemoryStore:
    """File I/O for ``memory/MEMORY.md`` and ``memory/history.jsonl``."""

    _DEFAULT_MAX_HISTORY = 1000

    def __init__(self, workspace: Path, max_history_entries: int = _DEFAULT_MAX_HISTORY) -> None:
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    # -- MEMORY.md -------------------------------------------------------------

    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    # -- history.jsonl: append-only journal --------------------------------------

    def _normalize_history_entry(self, entry: str, *, max_chars: int | None = None) -> str:
        """Clean model leftovers (``<think>``) and cap the size."""
        limit = max_chars if max_chars is not None else _HISTORY_ENTRY_HARD_CAP
        content = strip_think(entry.rstrip())
        return truncate_text(content, limit) if len(content) > limit else content

    def append_history(self, entry: str, *, max_chars: int | None = None, session_key: str | None = None) -> int:
        """Append an entry and return its cursor (1, 2, 3, ...)."""
        return self._append_history_record(
            self._normalize_history_entry(entry, max_chars=max_chars), session_key=session_key
        )

    def _append_history_record(self, content: str, *, session_key: str | None = None) -> int:
        cursor = self._next_cursor()
        record: dict[str, Any] = {
            "cursor": cursor,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "content": content,
        }
        if session_key:
            record["session_key"] = session_key
        with open(self.history_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    @staticmethod
    def _valid_cursor(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @staticmethod
    def _valid_history_payload(entry: dict[str, Any]) -> bool:
        session_key = entry.get("session_key")
        return (
            isinstance(entry.get("timestamp"), str)
            and isinstance(entry.get("content"), str)
            and (session_key is None or isinstance(session_key, str))
        )

    def _iter_valid_entries(self) -> Iterator[tuple[dict[str, Any], int]]:
        for entry in self._read_entries():
            cursor = self._valid_cursor(entry.get("cursor"))
            if cursor is not None and self._valid_history_payload(entry):
                yield entry, cursor

    def _read_cursor_counter(self) -> int | None:
        with suppress(ValueError, OSError):
            cursor = int(self._cursor_file.read_text(encoding="utf-8").strip())
            if cursor >= 0:
                return cursor
        return None

    def _next_cursor(self) -> int:
        """One more than the highest cursor seen (counter file or journal)."""
        counter = self._read_cursor_counter()
        last = self._read_last_entry() or {}
        last_cursor = self._valid_cursor(last.get("cursor"))
        if counter is not None:
            if last_cursor is not None:
                return max(counter, last_cursor) + 1
            return max(counter, max((c for _, c in self._iter_valid_entries()), default=0)) + 1
        if last_cursor is not None:
            return last_cursor + 1
        return max((c for _, c in self._iter_valid_entries()), default=0) + 1

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        return [entry for entry, cursor in self._iter_valid_entries() if cursor > since_cursor]

    def compact_history(self) -> None:
        """Keep the newest ``max_history_entries``, never dropping entries Dream hasn't read."""
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        last_dream = self.get_last_dream_cursor()
        first_unprocessed = next(
            (
                index
                for index, entry in enumerate(entries)
                if (cursor := self._valid_cursor(entry.get("cursor"))) is not None and cursor > last_dream
            ),
            len(entries),
        )
        keep_from = min(len(entries) - self.max_history_entries, first_unprocessed)
        self._write_entries(entries[keep_from:])

    def _read_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with suppress(FileNotFoundError):
            with open(self.history_file, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        entries.append(parsed)
        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        try:
            with open(self.history_file, "rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                if size == 0:
                    return None
                handle.seek(size - min(size, 4096))
                lines = [line for line in handle.read().decode("utf-8").split("\n") if line.strip()]
                if not lines:
                    return None
                parsed = json.loads(lines[-1])
                return parsed if isinstance(parsed, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        tmp = self.history_file.with_name(f".{self.history_file.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "x", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            os.replace(tmp, self.history_file)
        finally:
            tmp.unlink(missing_ok=True)

    # -- Dream ---------------------------------------------------------------------

    def get_last_dream_cursor(self) -> int:
        with suppress(ValueError, OSError):
            return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    def get_latest_cursor(self) -> int:
        return max(self._next_cursor() - 1, 0)

    def build_dream_prompt(self, *, max_entries: int = 20) -> tuple[str, int] | None:
        """Dream's instructions plus up to ``max_entries`` unread journal entries.

        Returns ``(prompt, last_cursor)``, or ``None`` if there is nothing new.
        The current memory files reach Dream through the normal system prompt.
        """
        entries = self.read_unprocessed_history(since_cursor=self.get_last_dream_cursor())
        if not entries:
            return None
        batch = entries[:max_entries]
        history_text = "\n".join(f"[{e['timestamp']}] {truncate_text(e['content'], 1000)}" for e in batch)
        template = self.default_dream_prompt()
        return f"{template}\n\n## Conversation History\n{history_text}", batch[-1]["cursor"]

    def build_dream_tools(self) -> ToolRegistry:
        """Dream's restricted file tools (nanobot ``build_dream_tools``).

        Read: the workspace and the built-in skills. Write: the workspace's
        ``skills/`` folder plus ``MEMORY.md``, ``SOUL.md`` and ``USER.md``.
        nanobot also gives Dream ``apply_patch``, which NanoClaw doesn't have.
        """
        from nanoclaw.agent.skills import BUILTIN_SKILLS_DIR

        workspace = self.workspace.resolve()
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        read_roots = [workspace, BUILTIN_SKILLS_DIR.resolve()]
        editable = {self.memory_file.resolve(), self.soul_file.resolve(), self.user_file.resolve()}

        def within(path: Path, root: Path) -> bool:
            return path == root or path.is_relative_to(root)

        def readable(raw: str, path: Path) -> str | None:
            if any(within(path, root) for root in read_roots):
                return None
            return f"Error: Path {raw} is outside allowed directory {workspace}{WORKSPACE_BOUNDARY_NOTE}"

        def writable(raw: str, path: Path) -> str | None:
            if within(path, skills_dir) or path in editable:
                return None
            return f"Error: Path {raw} is outside allowed directory {skills_dir}{WORKSPACE_BOUNDARY_NOTE}"

        tools = ToolRegistry()
        tools.register(_PathGuard(ReadFileTool(workspace), workspace, readable))
        tools.register(_PathGuard(EditFileTool(workspace), workspace, writable))
        tools.register(_PathGuard(WriteFileTool(workspace), workspace, writable))
        return tools

    @staticmethod
    def default_dream_prompt() -> str:
        """Dream's instructions, pointing at the bundled skill-creator skill."""
        from nanoclaw.agent.skills import BUILTIN_SKILLS_DIR

        return render_template(
            "agent/dream.md", strip=True, skill_creator_path=str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md")
        )

    @staticmethod
    def dream_session_key() -> str:
        """A unique key per Dream run, e.g. ``dream:20260528-100000``."""
        return f"dream:{datetime.now():%Y%m%d-%H%M%S}"

    # -- raw fallback ----------------------------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            content = message.get("content", "")
            if not content:
                continue
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            tools_used = message.get("tools_used")
            tools = f" [tools: {', '.join(tools_used)}]" if tools_used else ""
            timestamp = str(message.get("timestamp")) if message.get("timestamp") is not None else "?"
            role = str(message.get("role") or "unknown")
            lines.append(f"[{timestamp[:16]}] {role.upper()}{tools}: {content}")
        return "\n".join(lines)

    def raw_archive(
        self, messages: list[dict[str, Any]], *, max_chars: int | None = None, session_key: str | None = None
    ) -> str:
        """Journal the raw messages in bounded chunks and return a checkpoint."""
        limit = max_chars if max_chars is not None else _RAW_ARCHIVE_MAX_CHARS
        chunk_size = min(max(1, limit), _HISTORY_ENTRY_HARD_CAP - 1_000)
        formatted = strip_think(self._format_messages([public_history_message(m) for m in messages]))
        chunks = [formatted[i:i + chunk_size] for i in range(0, len(formatted), chunk_size)] or [""]
        for part, chunk in enumerate(chunks, start=1):
            suffix = f" (part {part}/{len(chunks)})" if len(chunks) > 1 else ""
            self._append_history_record(f"[RAW] {len(messages)} messages{suffix}\n{chunk}", session_key=session_key)
        return self._normalize_history_entry(f"[RAW] {len(messages)} messages\n{formatted}", max_chars=limit)

    def _build_raw_checkpoint(self, messages: list[dict[str, Any]], *, max_chars: int | None = None) -> str:
        """Same checkpoint as ``raw_archive``, without writing to the journal."""
        limit = max_chars if max_chars is not None else _RAW_ARCHIVE_MAX_CHARS
        formatted = self._format_messages([public_history_message(m) for m in messages])
        return self._normalize_history_entry(f"[RAW] {len(messages)} messages\n{formatted}", max_chars=limit)


class _PathGuard(Tool):
    """Wraps a file tool so it only accepts paths that ``check`` allows.

    Stands in for nanobot's ``allowed_dir`` / ``extra_write_allowed_files``
    options on its file tools; general path restriction is layer 9.
    """

    def __init__(self, inner: Tool, workspace: Path, check: Callable[[str, Path], str | None]) -> None:
        self._inner = inner
        self._workspace = workspace
        self._check = check

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._inner.parameters

    async def execute(self, **kwargs: Any) -> Any:
        raw = kwargs.get("path")
        if isinstance(raw, str):
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self._workspace / path
            problem = self._check(raw, path.resolve(strict=False))
            if problem:
                return ToolResult.error(problem)
        return await self._inner.execute(**kwargs)


class MemoryArchiver:
    """Asks the model for a checkpoint of a stretch of conversation."""

    def __init__(
        self,
        store: MemoryStore,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
    ) -> None:
        self.store = store
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions

    def _raw_checkpoint(
        self,
        messages: list[dict[str, Any]],
        *,
        session_key: str,
        previous_summary: str | None,
        max_tokens: int,
        persist: bool = True,
    ) -> str:
        raw = (
            self.store.raw_archive(messages, session_key=session_key)
            if persist
            else self.store._build_raw_checkpoint(messages)
        )
        return self._combine_raw_checkpoint(raw, previous_summary=previous_summary, max_tokens=max_tokens)

    @staticmethod
    def _combine_raw_checkpoint(raw: str, *, previous_summary: str | None, max_tokens: int) -> str:
        """Keep the previous summary and the new raw dump, within ``max_tokens``."""
        limit = max(1, max_tokens)
        if not previous_summary:
            return truncate_text_to_tokens(raw, limit)
        combined = f"[Previous archived context]\n{previous_summary}\n\n[Newly archived raw context]\n{raw}"
        bounded = truncate_text_to_tokens(combined, limit)
        if bounded == combined:
            return combined
        section = max(1, (limit - 32) // 2)
        return truncate_text_to_tokens(
            "[Previous archived context]\n"
            f"{truncate_text_to_tokens(previous_summary, section)}\n\n"
            "[Newly archived raw context]\n"
            f"{truncate_text_to_tokens(raw, section)}",
            limit,
        )

    async def archive(
        self,
        source_messages: list[dict[str, Any]],
        *,
        runtime: LLMRuntime,
        session_key: str,
        history: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
        previous_summary: str | None = None,
        input_token_budget: int | None = None,
        fallback_max_tokens: int | None = None,
        persist: bool = True,
    ) -> str | None:
        """Summarize ``source_messages``; journal and return the checkpoint.

        ``history`` is the request as the model saw it (system prompt included);
        the archive prompt is appended to it. If the request doesn't fit, the
        model fails, or it keeps calling tools, a raw dump is used instead.
        """
        if not source_messages:
            return None

        def raw_fallback() -> str:
            return self._raw_checkpoint(
                source_messages,
                session_key=session_key,
                previous_summary=previous_summary,
                max_tokens=fallback_max_tokens if fallback_max_tokens is not None else runtime.max_tokens,
                persist=persist,
            )

        prompt = render_template("agent/consolidator_archive.md", strip=True, archive_count=len(source_messages))
        request_messages = [*[dict(m) for m in history], {"role": "user", "content": prompt}]
        if input_token_budget is not None:
            estimated, _ = estimate_prompt_tokens(request_messages, request_tools)
            if input_token_budget <= 0 or estimated > input_token_budget:
                return raw_fallback()

        response = None
        for attempt in range(2):
            response = await runtime.provider.safe_chat(
                messages=request_messages,
                tools=request_tools or None,
                model=runtime.model,
                max_tokens=runtime.max_tokens,
                temperature=runtime.temperature,
            )
            if response.should_execute_tools is not True or attempt == 1:
                break
            # The model wanted tools: answer each call with a refusal and ask once more.
            request_messages = [
                *request_messages,
                build_assistant_message(
                    response.content,
                    tool_calls=[call.to_openai_tool_call() for call in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                ),
                *[
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": _ARCHIVE_TOOL_RESULT}
                    for call in response.tool_calls
                ],
            ]
        assert response is not None
        if response.finish_reason in {"error", "length"} or response.has_tool_calls:
            return raw_fallback()
        summary = response.content
        if not summary or not summary.strip():
            return raw_fallback()
        summary = self.store._normalize_history_entry(summary)
        if not summary:
            return raw_fallback()
        if persist and summary != "(nothing)":
            self.store.append_history(summary, session_key=session_key)
        return summary

    async def archive_session(
        self, session: Session, *, archive_end: int, runtime: LLMRuntime, input_token_budget: int
    ) -> str | None:
        """Archive ``session.messages[last_archived:archive_end]`` without changing the session."""
        messages = [
            m
            for m in session.messages[session.last_archived:archive_end]
            if not m.get("_command") and not is_summary_checkpoint(m)
        ]
        if not messages:
            return None
        summary_meta = session_summary_from_metadata(session.metadata, fallback_last_active=session.updated_at)
        previous_summary = summary_meta["text"] if summary_meta else None
        if input_token_budget <= 0:
            return self._raw_checkpoint(
                messages, session_key=session.key, previous_summary=previous_summary, max_tokens=runtime.max_tokens
            )
        prefix = Session(key=session.key, messages=list(session.messages[:archive_end]), last_archived=session.last_archived)
        history = prefix.get_history(max_tokens=input_token_budget)
        archive_history = Session(key=session.key, messages=messages).get_history()
        if not archive_history or history[-len(archive_history):] != archive_history:
            # The stretch to archive doesn't fit in one request: dump it raw.
            return self._raw_checkpoint(
                messages, session_key=session.key, previous_summary=previous_summary, max_tokens=runtime.max_tokens
            )
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        history_messages = self._build_messages(
            history=history, current_message=None, channel=channel, session_summary=summary_meta
        )
        return await self.archive(
            messages,
            runtime=runtime,
            session_key=session.key,
            history=history_messages,
            request_tools=self._get_tool_definitions(),
            previous_summary=previous_summary,
            input_token_budget=input_token_budget,
        )


class Consolidator:
    """Compacts sessions through ``MemoryArchiver``."""

    _SAFETY_BUFFER = 1024  # headroom for token-estimate drift

    def __init__(
        self,
        store: MemoryStore,
        sessions: SessionManager,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
    ) -> None:
        self.store = store
        self.sessions = sessions
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self.archiver = MemoryArchiver(store, build_messages, get_tool_definitions)

    async def summarize_transcript(
        self,
        accepted_messages: list[dict[str, Any]],
        previous_summary: str | None,
        *,
        runtime: LLMRuntime,
        session_key: str,
        tools: list[dict[str, Any]],
        persist: bool = True,
    ) -> str | None:
        """Summarize the request prefix the model has already seen (system prompt + history)."""
        source = [dict(m) for m in accepted_messages if m.get("role") != "system"]
        if not source:
            return None
        max_output = max(0, runtime.max_tokens)
        input_budget = runtime.context_window_tokens - max_output
        checkpoint_tokens = min(max_output, max(1, (input_budget - self._SAFETY_BUFFER) // 2))
        summary = await self.archiver.archive(
            source,
            runtime=runtime,
            session_key=session_key,
            history=accepted_messages,
            request_tools=tools,
            previous_summary=previous_summary,
            input_token_budget=input_budget,
            fallback_max_tokens=max(1, checkpoint_tokens),
            persist=persist,
        )
        if summary is None:
            return None
        return truncate_text_to_tokens(summary, max(1, max_output))

    def estimate_session_prompt_tokens(self, session: Session, *, runtime: LLMRuntime) -> tuple[int, str]:
        """Estimated size of the next request for this session (with a placeholder message)."""
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        summary = session_summary_from_metadata(session.metadata, fallback_last_active=session.updated_at)
        probe = self._build_messages(
            history=session.get_history() if session.messages else [],
            current_message="[token-probe]",
            channel=channel,
            session_summary=summary,
        )
        return estimate_prompt_tokens(probe, self._get_tool_definitions())

    def _input_token_budget(self, runtime: LLMRuntime) -> int:
        return runtime.context_window_tokens - runtime.max_tokens - self._SAFETY_BUFFER

    async def archive_session(self, session: Session, *, archive_end: int, runtime: LLMRuntime) -> str | None:
        return await self.archiver.archive_session(
            session, archive_end=archive_end, runtime=runtime, input_token_budget=self._input_token_budget(runtime)
        )

    async def compact_idle_session(self, session_key: str, *, runtime: LLMRuntime) -> str | None:
        """Replace the session's unarchived messages with a summary checkpoint.

        Returns the summary, ``""`` if there was nothing to compact, or ``None``
        if archiving produced nothing. (nanobot uses this for ``/compact`` and
        idle sessions.)
        """
        self.sessions.invalidate(session_key)
        session = self.sessions.get_or_create(session_key)
        start = session.last_archived
        to_archive = list(session.messages[start:])
        if not any(not m.get("_command") and not is_summary_checkpoint(m) for m in to_archive):
            return ""
        last_active = session.updated_at
        end = start + len(to_archive)
        summary = await self.archive_session(session, archive_end=end, runtime=runtime)
        if not summary:
            return None
        session.commit_summary_checkpoint(summary, insert_at=end, last_active=last_active)
        self.sessions.save(session)
        return summary
