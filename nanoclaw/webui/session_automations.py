"""Automation payloads for the embedded web UI.

Ported from nanobot/webui/session_automations.py (c4dac64): the jobs the
Session details popover lists (``GET /api/sessions/<key>/automations``), the
automation manager's list (``GET /api/webui/automations``, also the answer of
every ``automation.*`` mutation) and the ``automations`` list a blocked
``session.delete`` answers. Same JSON shapes.

What differs from nanobot (deliberately): NanoClaw has no local triggers
(``nanobot trigger``), so only cron jobs are listed; the origin chat's title
and preview are read from NanoClaw's session store.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol, cast

from nanoclaw.cron.binding import binding_revision
from nanoclaw.cron.session_turns import is_bound_cron_job
from nanoclaw.cron.types import CronJob
from nanoclaw.session.history_visibility import is_hidden_history_message
from nanoclaw.webui.session_identity import is_webui_session_key
from nanoclaw.webui.session_list_index import message_preview_text, metadata_title

__all__ = [
    "all_automations_payload",
    "serialize_automation_jobs",
    "session_automation_jobs",
    "session_automations_payload",
]


class _CronServiceLike(Protocol):
    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]: ...

    def list_bound_cron_jobs_for_session(
        self,
        session_key: str,
        *,
        include_disabled: bool = True,
    ) -> list[CronJob]: ...


def session_automation_jobs(cron_service: _CronServiceLike | None, session_key: str) -> list[CronJob]:
    """Return user automations attached to the WebUI session."""
    if cron_service is None:
        return []
    return list(cron_service.list_bound_cron_jobs_for_session(session_key, include_disabled=True))


def session_automations_payload(
    cron_service: _CronServiceLike | None,
    session_key: str,
    *,
    sessions: Any = None,
    pending_job_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Return user-created automation jobs attached to a WebUI session."""
    return {
        "jobs": serialize_automation_jobs(
            session_automation_jobs(cron_service, session_key),
            pending_job_ids=pending_job_ids,
            include_details=True,
            sessions=sessions,
        )
    }


def all_automations_payload(
    cron_service: _CronServiceLike | None,
    *,
    sessions: Any = None,
    pending_job_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Return all cron jobs visible to the WebUI automation manager."""
    jobs = cron_service.list_jobs(include_disabled=True) if cron_service is not None else []
    return {
        "jobs": serialize_automation_jobs(
            jobs,
            pending_job_ids=pending_job_ids,
            include_details=True,
            sessions=sessions,
        )
    }


def serialize_automation_jobs(
    jobs: list[CronJob],
    *,
    pending_job_ids: Collection[str] | None = None,
    include_details: bool = False,
    sessions: Any = None,
) -> list[dict[str, Any]]:
    return [
        _serialize_job(
            job,
            pending=job.id in (pending_job_ids or ()),
            include_details=include_details,
            sessions=sessions,
        )
        for job in jobs
    ]


def _serialize_job(
    job: CronJob,
    *,
    pending: bool = False,
    include_details: bool = False,
    sessions: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": job.id,
        "name": job.name,
        "enabled": job.enabled,
        "schedule": {
            "kind": job.schedule.kind,
            "at_ms": job.schedule.at_ms,
            "every_ms": job.schedule.every_ms,
            "expr": job.schedule.expr,
            "tz": job.schedule.tz,
        },
        "payload": {
            "message": job.payload.message,
        },
        "state": {
            "next_run_at_ms": job.state.next_run_at_ms,
            "last_status": job.state.last_status,
            "pending": pending,
        },
    }
    if not include_details:
        return payload

    payload["protected"] = job.payload.kind == "system_event"
    payload["delete_after_run"] = job.delete_after_run
    payload["created_at_ms"] = job.created_at_ms
    payload["updated_at_ms"] = job.updated_at_ms
    payload["payload"].update({"kind": job.payload.kind})
    run_history: list[dict[str, Any]] = []
    for record in job.state.run_history[-5:]:
        session_key = record.session_key or job.payload.session_key or ""
        run_history.append({
            "run_at_ms": record.run_at_ms,
            "status": record.status,
            "duration_ms": record.duration_ms,
            "error": record.error,
            "webui_session_key": session_key if is_webui_session_key(session_key) else None,
        })
    payload["state"].update(
        {
            "last_run_at_ms": job.state.last_run_at_ms,
            "last_error": job.state.last_error,
            "run_history": run_history,
        }
    )
    payload["origin"] = _origin_payload(job, sessions)
    if is_bound_cron_job(job):
        payload["chat_binding_revision"] = binding_revision(job)
    return payload


def _origin_payload(job: CronJob, sessions: Any) -> dict[str, Any] | None:
    channel = job.payload.origin_channel
    chat_id = job.payload.origin_chat_id
    if not channel or not chat_id:
        return None
    if channel != "websocket":
        return {
            "channel": channel,
            "title": "",
            "preview": "",
        }

    session_key = f"{channel}:{chat_id}"
    title = ""
    preview = ""
    data = _read_session(sessions, session_key)
    if data is not None:
        title = metadata_title(data.get("metadata"))
        preview = _session_preview(data.get("messages"))
    return {
        "session_key": session_key,
        "channel": channel,
        "chat_id": chat_id,
        "title": title,
        "preview": preview,
    }


def _read_session(sessions: Any, key: str) -> dict[str, Any] | None:
    """The session's metadata and messages without caching it (nanobot
    ``SessionManager.read_session_file``)."""
    if sessions is None:
        return None
    session = sessions.get_cached(key)
    if session is None:
        session = sessions._load(key)  # pyright: ignore[reportPrivateUsage]
    if session is None:
        return None
    return {"metadata": session.metadata, "messages": session.messages}


def _session_preview(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    fallback_preview = ""
    for message_value in cast(list[object], messages):
        if not isinstance(message_value, dict):
            continue
        message = cast(dict[str, Any], message_value)
        if is_hidden_history_message(message):
            continue
        text = message_preview_text(message)
        if not text:
            continue
        if message.get("role") == "user":
            return text
        if not fallback_preview and message.get("role") == "assistant":
            fallback_preview = text
    return fallback_preview
