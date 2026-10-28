"""WebUI session titles (the title parts of nanobot session/webui_turns.py).

A WebUI session is marked ``metadata["webui"] = True`` by its first inbound
frame. After a turn, ``maybe_generate_webui_title_after_turn`` asks the turn's
model for a 3-8 word title once (never over a user-edited title) and stores it
in ``metadata["title"]``, which the sidebar shows (``session_list_index``).
Layer 23 calls it from the websocket channel when a turn completes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from nanoclaw.agent.messages import strip_think, truncate_text
from nanoclaw.runtime_context import public_history_message
from nanoclaw.session.history_visibility import is_hidden_history_message

if TYPE_CHECKING:
    from nanoclaw.providers.base import LLMProvider
    from nanoclaw.session.manager import Session, SessionManager

WEBUI_SESSION_METADATA_KEY = "webui"
WEBUI_TITLE_METADATA_KEY = "title"
WEBUI_TITLE_USER_EDITED_METADATA_KEY = "title_user_edited"
TITLE_MAX_CHARS = 60
TITLE_GENERATION_MAX_TOKENS = 96


@dataclass(frozen=True)
class _WebsocketTurn:
    started_at: float
    turn_id: str | None
    transcript_persistence_failed: bool = False


@dataclass
class WebsocketTurnRegistry:
    """In-flight web UI turns per chat (nanobot webui_turns.py:71-430).

    nanobot keeps these as module globals (``_WEBSOCKET_ACTIVE_TURNS`` and the
    latest-owner projections); NanoClaw keeps one registry on the websocket
    channel, with the functions as methods of the same names. Every
    in-flight lifecycle owner of a chat is kept in admission order; the
    latest one answers ``wall_started_at`` / ``turn_id``. The state lives as
    long as the gateway process.
    """

    _turns: dict[str, dict[str, _WebsocketTurn]] = field(default_factory=dict)

    def _latest(self, chat_id: str) -> _WebsocketTurn | None:
        turns = self._turns.get(chat_id)
        if not turns:
            return None
        return turns[next(reversed(turns))]

    def wall_started_at(self, chat_id: str) -> float | None:
        """``time.time()`` when the active user turn began. (``websocket_turn_wall_started_at``)"""
        turn = self._latest(chat_id)
        return turn.started_at if turn is not None else None

    def turn_id(self, chat_id: str) -> str | None:
        """The web UI identity of the active turn. (``websocket_turn_id``)"""
        turn = self._latest(chat_id)
        return turn.turn_id if turn is not None else None

    def active_chat_ids(self) -> list[str]:
        return [chat_id for chat_id, turns in self._turns.items() if turns]

    def register_queued_if_idle(self, chat_id: str, turn_id: str | None) -> str | None:
        """Track an accepted turn while it waits for the loop; None when one
        is already running. (``register_queued_websocket_turn_if_idle``)"""
        if self.wall_started_at(chat_id) is not None:
            return None
        owner = uuid4().hex
        self._turns.setdefault(chat_id, {})[owner] = _WebsocketTurn(started_at=time.time(), turn_id=turn_id)
        return owner

    def owner_is_registered(self, chat_id: str, owner: str, turn_id: str | None) -> bool:
        """(``websocket_turn_owner_is_registered``)"""
        turn = self._turns.get(chat_id, {}).get(owner)
        return turn is not None and turn.turn_id == turn_id

    def mark_running(self, chat_id: str, owner: str, turn_id: str | None, started_at: float) -> None:
        """A turn started running: make its owner the latest. (``publish_turn_run_status``)"""
        turns = self._turns.setdefault(chat_id, {})
        turns.pop(owner, None)
        turns[owner] = _WebsocketTurn(started_at=started_at, turn_id=turn_id)

    def transcript_persistence_failed(self, chat_id: str, owner: str | None = None) -> bool:
        """(``websocket_turn_transcript_persistence_failed``)"""
        turns = self._turns.get(chat_id)
        if not turns:
            return False
        turn = turns.get(owner or next(reversed(turns)))
        return turn.transcript_persistence_failed if turn is not None else False

    def mark_transcript_persistence_failed(self, chat_id: str, owner: str | None) -> bool:
        """(``mark_websocket_turn_transcript_persistence_failed``)"""
        if not owner:
            return False
        turns = self._turns.get(chat_id)
        if turns is None or owner not in turns:
            return False
        turns[owner] = replace(turns[owner], transcript_persistence_failed=True)
        return True

    def clear_if_current(
        self, chat_id: str, owner: str | None, *, preserve_persistence_failure: bool = False
    ) -> bool:
        """Release one owner without disturbing concurrent turns. (``clear_websocket_turn_if_current``)"""
        if not owner:
            return False
        turns = self._turns.get(chat_id)
        if turns is None or owner not in turns:
            return False
        if preserve_persistence_failure and turns[owner].transcript_persistence_failed:
            return False
        turns.pop(owner)
        if not turns:
            self._turns.pop(chat_id, None)
        return True

    def clear(self, chat_id: str) -> None:
        """Forget a discarded chat. (``clear_websocket_turns``)"""
        self._turns.pop(chat_id, None)


def mark_webui_session(session: Session, metadata: dict[str, Any]) -> bool:
    """Persist a WebUI marker only when the inbound websocket frame opted in."""
    if metadata.get(WEBUI_SESSION_METADATA_KEY) is not True:
        return False
    session.metadata[WEBUI_SESSION_METADATA_KEY] = True
    return True


def clean_generated_title(raw: str | None) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"^\s*(title|标题)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`“”‘’")
    text = strip_think(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip("。.!！?？,，;；:")
    if len(text) > TITLE_MAX_CHARS:
        text = text[: TITLE_MAX_CHARS - 1].rstrip() + "…"
    return text


def _title_inputs(session: Session, *, latest: bool = False) -> tuple[str, str]:
    """First (or, with ``latest``, last) visible user and assistant texts."""
    user_text = ""
    assistant_text = ""
    for message in (reversed(session.messages) if latest else session.messages):
        if message.get("_command") is True or is_hidden_history_message(message):
            continue
        message = public_history_message(message)
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        content = strip_think(content)
        if not content:
            continue
        if role == "user" and not user_text:
            user_text = content.strip()
        elif role == "assistant" and not assistant_text:
            assistant_text = content.strip()
        if user_text and assistant_text:
            break
    return user_text, assistant_text


async def maybe_generate_webui_title(
    *,
    sessions: SessionManager,
    session_key: str,
    provider: LLMProvider,
    model: str,
    target_session_key: str | None = None,
) -> bool:
    """Generate and persist a short title for a WebUI-owned session.

    ``session_key`` owns the conversation content; ``target_session_key`` (when
    different) is the per-chat session that receives the title.
    """
    routed_session = sessions.get_existing(session_key)
    if routed_session is None:
        return False
    target_is_routed = target_session_key is None or target_session_key == session_key
    target_session = routed_session if target_is_routed else sessions.get_or_create(target_session_key or session_key)
    if (
        routed_session.metadata.get(WEBUI_SESSION_METADATA_KEY) is not True
        and target_session.metadata.get(WEBUI_SESSION_METADATA_KEY) is not True
    ):
        return False
    if target_session.metadata.get(WEBUI_TITLE_USER_EDITED_METADATA_KEY) is True:
        return False
    current_title = target_session.metadata.get(WEBUI_TITLE_METADATA_KEY)
    if isinstance(current_title, str) and current_title.strip():
        cleaned_current_title = clean_generated_title(current_title)
        if cleaned_current_title:
            if cleaned_current_title != current_title:
                target_session.metadata[WEBUI_TITLE_METADATA_KEY] = cleaned_current_title
                sessions.save(target_session)
            return False
        target_session.metadata.pop(WEBUI_TITLE_METADATA_KEY, None)
        current_title = None

    user_text, assistant_text = _title_inputs(routed_session, latest=not target_is_routed)
    if not user_text:
        return False

    prompt = (
        "Generate a concise title for this chat.\n"
        "Rules:\n"
        "- Use the same language as the user when practical.\n"
        "- 3 to 8 words.\n"
        "- No quotes.\n"
        "- No punctuation at the end.\n"
        "- Return only the title.\n\n"
        f"User: {truncate_text(user_text, 1_000)}"
    )
    if assistant_text:
        prompt += f"\nAssistant: {truncate_text(assistant_text, 1_000)}"
    try:
        response = await provider.chat_stream_with_retry(
            [
                {"role": "system", "content": "You write short, neutral chat titles. Return only the title text."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            model=model,
            max_tokens=TITLE_GENERATION_MAX_TOKENS,
            temperature=0.2,
            retry_mode="standard",
        )
    except Exception:
        logger.opt(exception=True).debug("Failed to generate webui session title for {}", session_key)
        return False

    title = clean_generated_title(response.content)
    if not title or title.lower().startswith("error"):
        logger.debug("WebUI title generation returned no usable title for {} (finish_reason={})",
                     session_key, response.finish_reason)
        return False
    if target_session.metadata.get(WEBUI_TITLE_USER_EDITED_METADATA_KEY) is True:
        return False
    if target_session.metadata.get(WEBUI_TITLE_METADATA_KEY) != current_title:
        return False
    target_session.metadata[WEBUI_TITLE_METADATA_KEY] = title
    sessions.save(target_session)
    return True


async def maybe_generate_webui_title_after_turn(
    *,
    channel: str,
    chat_id: str,
    metadata: dict[str, Any],
    sessions: SessionManager,
    session_key: str,
    provider: LLMProvider,
    model: str,
) -> bool:
    if channel != "websocket" or metadata.get(WEBUI_SESSION_METADATA_KEY) is not True:
        return False
    origin_session_key = f"{channel}:{chat_id}"
    return await maybe_generate_webui_title(
        sessions=sessions,
        session_key=session_key,
        provider=provider,
        model=model,
        target_session_key=origin_session_key if origin_session_key != session_key else None,
    )
