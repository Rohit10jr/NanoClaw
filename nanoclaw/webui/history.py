"""WebUI display history: one append-only JSONL event log per chat.

Mirrors nanobot/webui/history.py (``fact_from_record``/``record_from_fact``/
``turn_records``). nanobot stores display facts in SQLite next to the session;
NanoClaw keeps the JSONL layout nanobot used before its SQLite migration (its
"legacy" ``~/.nanobot/webui/<stem>.jsonl`` files, which nanobot still imports):
each line is one persisted wire record such as

    {"event": "user", "chat_id": "c1", "text": "hi", "created_at_ms": 1700000000000,
     "turn_id": "t1", "turn_phase": "user", "turn_seq": 1}

Storage layout::

    ~/.nanoclaw/sessions/<workspace-id>/<storage-key>.jsonl   session (SessionManager)
    ~/.nanoclaw/webui/<workspace-id>/<storage-key>.jsonl      display history (this module)

``<storage-key>`` is the session file's own collision-free name
(``SessionManager.storage_key``, base64url of ``websocket:<chat_id>``); the
workspace ID keeps two workspaces' chats apart, as the sessions directory does.
Records are validated and normalised through ``fact_from_record`` before they
are written, so a file only ever holds the fields nanobot persists.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from nanoclaw.session.display_history import HistoryFact, HistoryKind, HistoryReader
from nanoclaw.webui.session_identity import webui_chat_id

if TYPE_CHECKING:
    from nanoclaw.session.manager import SessionManager

# nanobot bus/events.py INBOUND_META_LOCAL_MEDIA: uploads that live on this host.
INBOUND_META_LOCAL_MEDIA = "local_media_paths"

_EVENTS = {
    "user": HistoryKind.INPUT,
    "message": HistoryKind.ANSWER,
    "stream_end": HistoryKind.ANSWER_SEGMENT,
    "reasoning_end": HistoryKind.REASONING_SEGMENT,
    "file_edit": HistoryKind.FILE_ACTIVITY,
    "turn_end": HistoryKind.COMPLETION,
    "fork_marker": HistoryKind.FORK_BOUNDARY,
    "context_compaction": HistoryKind.COMPACTION,
    "delta": HistoryKind.ANSWER_FRAGMENT,
    "reasoning_delta": HistoryKind.REASONING_FRAGMENT,
}
_MESSAGE_KINDS = {
    "reasoning": HistoryKind.REASONING,
    "progress": HistoryKind.PROGRESS,
    "tool_hint": HistoryKind.TOOL_ACTIVITY,
}
_CONTENT_FIELDS = frozenset({
    "text", "media_paths", "media", "media_urls", "reply_to", "agent_ui", "cli_apps", "mcp_presets", "session_mentions",
    "session_message", INBOUND_META_LOCAL_MEDIA, "tool_events", "edits", "source", "response_sources", "resuming",
    "merge_next", "latency_ms", "usage", "round_usages", "context_window_tokens", "goal_state",
    "outcome", "failure_kind", "failure_error_kind", "failure_attempts", "failure_message",
    "transcript_incomplete", "stream_id", "compaction_id", "phase", "reason", "render_as",
})


def fact_from_record(record: Mapping[str, Any]) -> HistoryFact:
    """Parse the channel boundary once; core storage never sees event/chat_id keys."""
    event = record.get("event")
    if event is None and record.get("role") in {"user", "assistant"}:
        event = "user" if record["role"] == "user" else "message"
        record = {**record, "text": record.get("text", record.get("content", ""))}
    if event not in _EVENTS:
        raise ValueError(f"unsupported legacy history event: {event!r}")
    kind = _EVENTS[event]
    if event == "message":
        message_kind = record.get("kind")
        if message_kind is not None and message_kind not in _MESSAGE_KINDS:
            raise ValueError(f"unsupported history message kind: {message_kind!r}")
        if isinstance(message_kind, str):
            kind = _MESSAGE_KINDS[message_kind]
    created = record.get("created_at_ms", 0)
    if isinstance(created, bool) or not isinstance(created, int | float) or not 0 <= created < 10**16:
        raise ValueError("invalid history timestamp")
    sequence = record.get("turn_seq")
    if sequence is not None and (isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0):
        raise ValueError("invalid history sequence")
    turn_id, phase = record.get("turn_id"), record.get("turn_phase")
    if turn_id is not None and (not isinstance(turn_id, str) or len(turn_id) > 128):
        raise ValueError("invalid history turn identity")
    if phase is not None and not isinstance(phase, str):
        raise ValueError("invalid history phase")
    return HistoryFact(kind, {key: value for key, value in record.items() if key in _CONTENT_FIELDS},
                       turn_id, phase, sequence, int(created))


def record_from_fact(fact: HistoryFact, key: str) -> dict[str, Any]:
    if fact.kind in _MESSAGE_KINDS.values():
        event = "message"
        message_kind = next(name for name, kind in _MESSAGE_KINDS.items() if kind is fact.kind)
    else:
        event = next(name for name, kind in _EVENTS.items() if kind is fact.kind)
        message_kind = None
    record = {"event": event, "chat_id": webui_chat_id(key), **fact.content,
              "created_at_ms": fact.created_at_ms}
    if message_kind:
        record["kind"] = message_kind
    if fact.turn_id is not None:
        record["turn_id"] = fact.turn_id
    if fact.phase is not None:
        record["turn_phase"] = fact.phase
    if fact.sequence is not None:
        record["turn_seq"] = fact.sequence
    return record


def turn_records(
    history: HistoryReader, ordinal: int, *, max_records: int | None = None, max_bytes: int | None = None,
) -> list[dict[str, Any]]:
    return [{**record_from_fact(fact, history.key),
             "_webui_replay_identity": f"turn:{ordinal}:record:{position}"}
            for position, fact in history.read_entries(ordinal, max_records=max_records, max_bytes=max_bytes)]


def webui_history_dir(sessions: SessionManager) -> Path:
    """``<sessions root>/../webui/<workspace-id>``: ``~/.nanoclaw/webui/<id>`` by default."""
    return sessions.sessions_dir.parent.parent / "webui" / sessions.sessions_dir.name


class WebUIHistoryStore:
    """Per-chat append-only display history files (see the module docstring)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).expanduser()
        self._lock = threading.Lock()
        self._cache: dict[Path, tuple[tuple[int, int, int, int], list[HistoryFact]]] = {}

    @classmethod
    def for_sessions(cls, sessions: SessionManager) -> WebUIHistoryStore:
        return cls(webui_history_dir(sessions))

    def path(self, key: str) -> Path:
        from nanoclaw.session.manager import SessionManager

        return self.directory / f"{SessionManager.storage_key(key)}.jsonl"

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def _ensure_directory(self) -> None:
        if self.directory.is_symlink():
            raise ValueError(f"webui history directory must not be a symlink: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.directory, 0o700)  # conversations are private
        except OSError:
            pass

    def append(self, key: str, facts: Sequence[HistoryFact]) -> None:
        """Append facts in order, one JSON line each. Validation happens first,
        so a bad fact leaves the file unchanged."""
        lines = []
        for fact in facts:
            fact.encode()  # enforces the 8 MiB per-fact limit
            lines.append(json.dumps(record_from_fact(fact, key), ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")))
        if not lines:
            return
        with self._lock:
            self._ensure_directory()
            path = self.path(key)
            if path.is_symlink():
                raise ValueError(f"webui history must not be a symlink: {path}")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write("".join(f"{line}\n" for line in lines))

    def append_records(self, key: str, records: Sequence[Mapping[str, Any]]) -> None:
        self.append(key, [fact_from_record(record) for record in records])

    def read_facts(self, key: str) -> tuple[list[HistoryFact], tuple[str, int] | None]:
        """All facts plus the file revision; ``([], None)`` when there is no file."""
        path = self.path(key)
        with self._lock:
            try:
                stat = path.stat()
            except OSError:
                self._cache.pop(path, None)
                return [], None
            signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            cached = self._cache.get(path)
            if cached is None or cached[0] != signature:
                cached = (signature, self._parse(path, key))
                self._cache[path] = cached
                while len(self._cache) > 32:
                    self._cache.pop(next(iter(self._cache)))
            facts = cached[1]
        return list(facts), (f"{stat.st_dev}:{stat.st_ino}", len(facts))

    @staticmethod
    def _parse(path: Path, key: str) -> list[HistoryFact]:
        facts: list[HistoryFact] = []
        skipped = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value: object = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("record is not an object")
                    facts.append(fact_from_record(cast(dict[str, Any], value)))
                except (ValueError, TypeError):
                    skipped += 1
        if skipped:
            logger.warning("Skipped {} unreadable lines in webui history {}", skipped, key)
        return facts

    def reader(self, key: str) -> HistoryReader:
        facts, revision = self.read_facts(key)
        return HistoryReader(key, facts, storage_path=self.path(key), revision=revision)

    def delete(self, key: str) -> bool:
        path = self.path(key)
        with self._lock:
            self._cache.pop(path, None)
            try:
                path.unlink()
            except FileNotFoundError:
                return False
        return True
