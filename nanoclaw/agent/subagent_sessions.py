"""Private child sessions for subagent transcripts and task observations.

Mirrors nanobot/agent/subagent_sessions.py (c4dac64). Each task gets a
session ``subagent:<task id>`` in the common session store: its metadata
names the session type (``subagent``), the parent session that owns it and
the task's status (``subagent_task``); its messages are the child's
transcript (the task, then what the runner added). Only the parent session
can read or control it.

NanoClaw stores sessions as JSONL files (SessionManager, outside the
workspace) instead of nanobot's SQLite state store, so:

- there are no session generations or conflict errors: a child is saved by
  rewriting its file, and a running child's ``Session`` object is the one
  every save writes (``_live``), so status and transcript saves never race;
- nanobot keeps a non-persisted session (``SessionPolicy(persist=False)``,
  a temporary turn's child) in its state cache; here such children and
  their parents are kept in memory only (``_transient``), never written;
- nanobot's session-type registry (``SessionType("subagent", ...)``, which
  hides child sessions from public session lists) has no counterpart: NanoClaw
  has no session list yet.

Later layers: nanobot's idle auto-compaction skips ``subagent:`` sessions
(autocompact.py:77, drift row D13); NanoClaw has no idle compaction yet
(layer 21), so whoever adds it must skip these keys too. Deleting a parent
session should also stop its children's reports (``accepts_result``); NanoClaw
never deletes session files today (``/new`` clears the parent in place).
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from collections import OrderedDict
from typing import Any

from loguru import logger

from nanoclaw.agent.subagent_status import SubagentSessionError as SubagentSessionError
from nanoclaw.agent.subagent_status import SubagentStatus
from nanoclaw.providers import LLMUsage
from nanoclaw.session.manager import Session, SessionManager, SessionPolicy

# nanobot session/types.py
SESSION_TYPE_KEY = "session_type"
PARENT_SESSION_KEY = "parent_session_key"
SUBAGENT_SESSION_TYPE = "subagent"
TASK_METADATA_KEY = "subagent_task"
_ACTIVE_STATES = {"queued", "running", "stopping"}
# How many temporary (never saved) children stay readable.
MAX_TRANSIENT = 128
_STATUS_FIELDS = {f.name for f in dataclasses.fields(SubagentStatus)}


class SubagentSessions:
    """Use the common session store; keep task access owned by the parent."""

    def __init__(self, sessions: SessionManager):
        self.sessions = sessions
        # Running children: the Session object every save writes.
        self._live: dict[str, Session] = {}
        # Children of temporary turns (never saved) and their parents.
        self._transient: OrderedDict[str, Session] = OrderedDict()
        self._transient_owners: set[str] = set()

    @staticmethod
    def key(task_id: str) -> str:
        return f"subagent:{task_id}"

    # -- reading -----------------------------------------------------------

    def _metadata(self, key: str) -> dict[str, Any] | None:
        session = self._transient.get(key) or self._live.get(key) or self.sessions.get_cached(key)
        if session is not None:
            return session.metadata
        header = self.sessions.read_session_metadata(key)
        return header["metadata"] if header is not None else None

    def _session(self, key: str) -> Session | None:
        return self._transient.get(key) or self._live.get(key) or self.sessions.get_existing(key)

    def exists(self, owner: str) -> bool:
        return (
            owner in self._transient_owners
            or self.sessions.get_cached(owner) is not None
            or self.sessions.get_session_path(owner).exists()
        )

    def contains(self, task_id: str, owner: str) -> bool:
        metadata = self._metadata(self.key(task_id)) or {}
        return metadata.get(SESSION_TYPE_KEY) == SUBAGENT_SESSION_TYPE and metadata.get(PARENT_SESSION_KEY) == owner

    def snapshot(self, task_id: str, owner: str) -> Session | None:
        """Read a detached child only while its parent still owns it."""
        if not self.exists(owner) or not self.contains(task_id, owner):
            return None
        session = self._session(self.key(task_id))
        return copy.deepcopy(session) if session is not None else None

    @staticmethod
    def _payload(status: SubagentStatus) -> dict[str, Any]:
        """The status as saved: every field but the monotonic clock readings,
        plus the elapsed time (nanobot dumps the dataclass with pydantic)."""
        payload: dict[str, Any] = {}
        for item in dataclasses.fields(status):
            if item.name in {"started_at", "finished_at"}:
                continue
            value = getattr(status, item.name)
            if item.name == "usage" and value is not None:
                value = dataclasses.asdict(value)
            payload[item.name] = copy.deepcopy(value)
        payload["elapsed_seconds"] = status.as_dict()["elapsed_seconds"]
        return payload

    # -- writing -----------------------------------------------------------

    def create(self, status: SubagentStatus, *, policy: SessionPolicy | None = None) -> Session:
        """Establish the child (and its parent, if new) before execution starts.
        (nanobot ``SessionState.create_child``)"""
        policy = policy or SessionPolicy()
        owner = status.owner
        if not self.exists(owner):
            if policy.persist:
                self.sessions.save(self.sessions.get_or_create(owner))
            else:
                self._transient_owners.add(owner)
        child = Session(
            key=self.key(status.task_id),
            metadata={SESSION_TYPE_KEY: SUBAGENT_SESSION_TYPE, TASK_METADATA_KEY: self._payload(status)},
        )
        child.metadata[PARENT_SESSION_KEY] = owner
        child.add_message("user", status.task_description)
        if policy.persist:
            self._live[child.key] = child
            self.sessions.save(child)
        else:
            self._transient[child.key] = child
            while len(self._transient) > MAX_TRANSIENT:
                self._transient.popitem(last=False)
        return child

    def save(self, status: SubagentStatus) -> bool:
        key = self.key(status.task_id)
        session = self._session(key)
        if session is None:
            return False
        session.metadata[TASK_METADATA_KEY] = self._payload(status)
        if key not in self._transient:
            self.sessions.save(session)
        return True

    def save_transcript(self, session: Session) -> None:
        """Commit the child's transcript (nanobot ``finish_turn`` on the child)."""
        if session.key in self._transient:
            return
        if session.key in self._live or self.sessions.get_session_path(session.key).exists():
            self.sessions.save(session)

    def release(self, task_id: str) -> None:
        """The child stopped running; later reads come from its saved file."""
        self._live.pop(self.key(task_id), None)

    # -- statuses ----------------------------------------------------------

    def status(self, task_id: str, owner: str) -> SubagentStatus | None:
        metadata = self._metadata(self.key(task_id))
        if metadata is None:
            return None
        return self._status_from_metadata(task_id, owner, metadata)

    @staticmethod
    def _status_from_metadata(task_id: str, owner: str, metadata: dict[str, Any]) -> SubagentStatus | None:
        if metadata.get(SESSION_TYPE_KEY) != SUBAGENT_SESSION_TYPE or metadata.get(PARENT_SESSION_KEY) != owner:
            return None
        raw: object = metadata.get(TASK_METADATA_KEY)
        if not isinstance(raw, dict):
            raise SubagentSessionError("invalid task session metadata")
        value = dict(raw)
        elapsed = value.pop("elapsed_seconds", None)
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise SubagentSessionError("invalid task duration")
        now = time.monotonic()
        try:
            if set(value) - _STATUS_FIELDS:
                raise TypeError("unknown status fields")
            usage = value.get("usage")
            if usage is not None:
                value["usage"] = LLMUsage(**usage)
            status = SubagentStatus(
                **value,
                started_at=now - elapsed,
                finished_at=now if value.get("completed_at") is not None else None,
            )
        except (TypeError, ValueError) as exc:
            raise SubagentSessionError("invalid task session metadata") from exc
        if not isinstance(status.task_id, str) or not isinstance(status.owner, str):
            raise SubagentSessionError("invalid task session metadata")
        if status.owner != owner or status.task_id != task_id:
            raise SubagentSessionError("invalid task ownership")
        return status

    def load(self, owner: str) -> dict[str, SubagentStatus]:
        """Every saved task status of one parent session."""
        rows = [(header["key"], header["metadata"]) for header in self.sessions.list_session_headers()]
        rows += [(key, session.metadata) for key, session in self._transient.items()]
        result: dict[str, SubagentStatus] = {}
        for key, metadata in rows:
            task_id = key.removeprefix("subagent:")
            if key != self.key(task_id) or metadata.get(PARENT_SESSION_KEY) != owner:
                continue
            status = self._status_from_metadata(task_id, owner, metadata)
            if status is not None:
                result[task_id] = status
        return result

    async def interrupt_pending(self) -> None:
        """Seal abandoned work after the host acquires execution ownership."""
        for header in self.sessions.list_session_headers():
            key = header["key"]
            metadata = header["metadata"]
            if metadata.get(SESSION_TYPE_KEY) != SUBAGENT_SESSION_TYPE:
                continue
            owner = metadata.get(PARENT_SESSION_KEY)
            if not isinstance(owner, str):
                continue
            try:
                status = self.status(key.removeprefix("subagent:"), owner)
                if status is None or status.state not in _ACTIVE_STATES:
                    continue
                status.state = status.phase = "interrupted"
                status.revision += 1
                status.stop_reason = "host_restarted"
                status.finished_at = time.monotonic()
                status.completed_at = time.time()
                status.partial = bool(status.result)
                status.tool_events = []
                status.receipts = {
                    message_id: "undelivered" if receipt == "accepted" else receipt
                    for message_id, receipt in status.receipts.items()
                }
                self.save(status)
            except (OSError, SubagentSessionError):
                logger.exception("Could not recover subagent session {}", key)
