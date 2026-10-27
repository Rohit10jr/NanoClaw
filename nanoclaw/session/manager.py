"""Conversation sessions, saved as JSONL files.

Mirrors nanobot/session/manager.py (``Session``, ``JsonlSessionStore`` and
``SessionManager``, here folded into one class). A session is one conversation:
the list of messages in OpenAI chat format, plus a little metadata. Its file has
one metadata line followed by one line per message.

Sessions live outside the workspace, in ``~/.nanoclaw/sessions/<workspace-id>/``,
so the agent's file tools can't read or rewrite their own history. The workspace
ID is a random value saved in ``<workspace>/.nanoclaw/workspace-id``: it moves
with the folder, so a renamed workspace keeps its conversations.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanoclaw.runtime_context import public_history_message
from nanoclaw.session.summary import HIDDEN_HISTORY_META, SUMMARY_CONTINUATION_TEXT, is_summary_checkpoint
from nanoclaw.utils.tokens import estimate_message_tokens

# nanobot keeps sessions under its config directory (~/.nanobot/sessions).
DEFAULT_SESSIONS_ROOT = Path("~/.nanoclaw/sessions")
SESSION_CACHE_MAX_SIZE = 128
WORKSPACE_STATE_DIR = ".nanoclaw"
WORKSPACE_ID_FILE = "workspace-id"

_WORKSPACE_ID_RE = re.compile(r"[0-9a-f]{32}")
# Internal markers a model may have copied into its own replies (nanobot).
_MESSAGE_TIME_PREFIX_RE = re.compile(r"^\[Message Time: [^\]]+\]\n?")
_LOCAL_IMAGE_BREADCRUMB_RE = re.compile(r"^\[image: (?:/|~)[^\]]+\]\s*$")
_TOOL_CALL_ECHO_RE = re.compile(r"^\s*(?:generate_image|message)\([^)]*\)\s*$")
# Keys replayed to the model besides role and content.
_REPLAY_KEYS = ("tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks")


def _sanitize_assistant_replay_text(content: str) -> str:
    """Trim an assistant reply before replaying it.

    Removes internal markers the model may have copied (a message-time prefix,
    local image paths, echoed tool calls), which would otherwise become
    examples for it to repeat, and surrounding whitespace.
    (nanobot session/manager.py ``_sanitize_assistant_replay_text``)
    """
    content = _MESSAGE_TIME_PREFIX_RE.sub("", content, count=1)
    lines = [
        line
        for line in content.splitlines()
        if not _LOCAL_IMAGE_BREADCRUMB_RE.match(line) and not _TOOL_CALL_ECHO_RE.match(line)
    ]
    return "\n".join(lines).strip()


def find_legal_message_start(messages: list[dict[str, Any]]) -> int:
    """The first index after which every tool result has its assistant tool call.

    A window that starts in the middle of a turn can begin with tool results
    whose calls were cut off; providers reject those. (nanobot utils/helpers.py)
    """
    declared: set[str] = set()
    start = 0
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    declared.add(str(call["id"]))
        elif role == "tool":
            tid = msg.get("tool_call_id")
            if tid and str(tid) not in declared:
                start = i + 1
                declared.clear()
    return start


@dataclass
class Session:
    """A conversation session."""

    key: str  # channel:chat_id, e.g. "cli:direct"
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Messages before this index were summarized (compacted) and are no longer
    # replayed; the summary goes into the system prompt instead.
    last_archived: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        # An out-of-range offset (corrupt file) would hide all history; reset it.
        offset = self.last_archived
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= len(self.messages):
            self.last_archived = 0

    def add_message(self, role: str, content: Any, **kwargs: Any) -> None:
        """Append a message, stamped with the current time."""
        self.messages.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs}
        )
        self.updated_at = datetime.now()

    def commit_summary_checkpoint(
        self, summary: str, *, insert_at: int | None = None, last_active: datetime | None = None
    ) -> None:
        """Replace everything before ``insert_at`` with ``summary`` in future replays.

        The messages stay in the file; a hidden marker records where the summary
        ends, ``last_archived`` moves to it, and the summary text is stored in
        the metadata for the system prompt.
        """
        boundary = len(self.messages) if insert_at is None else insert_at
        self.messages.insert(boundary, {
            "role": "user",
            "content": SUMMARY_CONTINUATION_TEXT,
            HIDDEN_HISTORY_META: True,
            "timestamp": datetime.now().isoformat(),
        })
        self.metadata["_last_summary"] = {
            "text": summary,
            "last_active": (last_active or self.updated_at).isoformat(),
        }
        self.last_archived = boundary

    def get_history(self, *, max_tokens: int = 0, include_runtime_context: bool = True) -> list[dict[str, Any]]:
        """The messages to replay to the model, in OpenAI chat format.

        Starts after the last summary and at the first user message (never
        mid-turn), drops tool results whose call is missing, skips empty
        assistant messages and the summary marker, and keeps only the keys the
        model needs. A positive ``max_tokens`` keeps only the most recent
        messages that fit. ``include_runtime_context=False`` removes the
        appended runtime context (the time block), as shown to the user.
        """
        sliced = self.messages[self.last_archived:]
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break
        sliced = sliced[find_legal_message_start(sliced):]

        out: list[dict[str, Any]] = []
        for message in sliced:
            if message.get("_command") or is_summary_checkpoint(message):
                continue
            if not include_runtime_context:
                message = public_history_message(message)
            role = message.get("role")
            content = message.get("content", "")
            if role == "assistant" and isinstance(content, str):
                content = _sanitize_assistant_replay_text(content)
            if role == "assistant" and isinstance(content, str) and not content.strip():
                if not any(key in message for key in ("tool_calls", "reasoning_content", "thinking_blocks")):
                    continue
            entry: dict[str, Any] = {"role": role, "content": content}
            for key in _REPLAY_KEYS:
                if key in message:
                    entry[key] = message[key]
            out.append(entry)

        if max_tokens > 0 and out:
            kept: list[dict[str, Any]] = []
            used = 0
            for message in reversed(out):
                tokens = estimate_message_tokens(message)
                if kept and used + tokens > max_tokens:
                    break
                kept.append(message)
                used += tokens
            kept.reverse()
            # Start at a user message; if none fits, recover the nearest one.
            first_user = next((i for i, m in enumerate(kept) if m.get("role") == "user"), None)
            if first_user is not None:
                kept = kept[first_user:]
            else:
                recovered = next((i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "user"), None)
                if recovered is not None:
                    kept = out[recovered:]
            out = kept[find_legal_message_start(kept):]
        return out

    def clear(self) -> None:
        """Forget all messages (the ``/new`` command)."""
        self.messages = []
        self.last_archived = 0
        self.updated_at = datetime.now()
        self.metadata.pop("_last_summary", None)
        # A fresh conversation has no cut-off turn to mention (nanobot session/manager.py:355).
        self.metadata.pop("_stopped_at_iteration_limit", None)


class SessionManager:
    """Loads, caches and saves the sessions of one workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        sessions_root: Path | None = None,
        max_cached_sessions: int = SESSION_CACHE_MAX_SIZE,
    ) -> None:
        workspace = Path(workspace).expanduser().resolve()
        root = Path(sessions_root or DEFAULT_SESSIONS_ROOT).expanduser().resolve()
        if root == workspace or root.is_relative_to(workspace):
            raise RuntimeError(f"session storage must be outside the agent workspace: {root}")
        root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(root, 0o700)  # conversations are private
        except OSError:
            pass
        self.workspace = workspace
        self.sessions_dir = root / self._load_or_create_workspace_id(workspace)
        self.sessions_dir.mkdir(exist_ok=True)
        self._max_cached_sessions = max_cached_sessions
        self._cache: OrderedDict[str, Session] = OrderedDict()

    # --- workspace identity ---------------------------------------------------

    @staticmethod
    def _load_or_create_workspace_id(workspace: Path) -> str:
        state_dir = workspace / WORKSPACE_STATE_DIR
        if state_dir.is_symlink():
            raise RuntimeError(f"workspace state directory must not be a symlink: {state_dir}")
        state_dir.mkdir(parents=True, exist_ok=True)
        marker = state_dir / WORKSPACE_ID_FILE
        if not (marker.exists() or marker.is_symlink()):
            workspace_id = secrets.token_hex(16)
            try:
                # O_EXCL: if two processes race, the second reads the first's ID.
                fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "w", encoding="ascii") as handle:
                    handle.write(f"{workspace_id}\n")
                return workspace_id
        if marker.is_symlink():
            raise RuntimeError(f"workspace identity marker must not be a symlink: {marker}")
        value = marker.read_text(encoding="utf-8").strip()
        if not _WORKSPACE_ID_RE.fullmatch(value):
            raise RuntimeError(
                f"workspace identity marker is invalid: {marker}; "
                "restore its original 32-character identifier before starting nanoclaw"
            )
        return value

    # --- paths ------------------------------------------------------------------

    @staticmethod
    def storage_key(key: str) -> str:
        """A file-name-safe form of any session key ("cli:direct" -> "Y2xpOmRpcmVjdA")."""
        return base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")

    def get_session_path(self, key: str) -> Path:
        return self.sessions_dir / f"{self.storage_key(key)}.jsonl"

    # --- cache --------------------------------------------------------------------

    def get_or_create(self, key: str) -> Session:
        """The cached session, else the saved one, else a new empty one."""
        session = self._cache.get(key)
        if session is None:
            session = self._load(key) or Session(key=key)
        self._remember(session)
        return session

    def invalidate(self, key: str) -> None:
        """Drop a session from the cache; the next ``get_or_create`` reloads it."""
        self._cache.pop(key, None)

    def _remember(self, session: Session) -> None:
        self._cache[session.key] = session
        self._cache.move_to_end(session.key)
        while len(self._cache) > self._max_cached_sessions:
            self._cache.popitem(last=False)

    # --- storage ------------------------------------------------------------------

    def save(self, session: Session) -> None:
        """Write the whole session file atomically, and keep it cached."""
        metadata_line = {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "last_archived": session.last_archived,
        }
        lines = [json.dumps(metadata_line, ensure_ascii=False)]
        lines.extend(json.dumps(message, ensure_ascii=False) for message in session.messages)
        path = self.get_session_path(session.key)
        # A crash mid-write leaves the old file intact: write a temp file, then rename.
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "x", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        self._remember(session)

    def _load(self, key: str) -> Session | None:
        """Read a session file; unreadable lines are skipped (nanobot ``repair``)."""
        path = self.get_session_path(key)
        if not path.exists():
            return None
        messages: list[dict[str, Any]] = []
        header: dict[str, Any] = {}
        skipped = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if not isinstance(data, dict):
                    skipped += 1
                    continue
                if data.get("_type") == "metadata":
                    header = data
                else:
                    messages.append(data)
        if skipped:
            logger.warning("Skipped {} corrupt lines in session {}", skipped, key)
        return Session(
            key=key,
            messages=messages,
            created_at=_parse_time(header.get("created_at")),
            updated_at=_parse_time(header.get("updated_at")),
            metadata=header.get("metadata", {}),
            last_archived=header.get("last_archived", 0),
        )


def _parse_time(value: Any) -> datetime:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now()
