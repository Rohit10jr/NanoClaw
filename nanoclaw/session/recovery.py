"""Closing a turn that was interrupted before it could be saved.

Mirrors the part of nanobot/session/recovery.py that needs no checkpoints. The
loop flags the session when it saves the user's message and clears the flag
once the turn is saved. If the flag is still set when the next turn starts, the
process died mid-turn: the unanswered message gets an honest error reply, so the
model doesn't treat the old request as part of the new one.
"""

from __future__ import annotations

from datetime import datetime

from nanoclaw.session.manager import Session

PENDING_USER_TURN_KEY = "pending_user_turn"
INTERRUPTED_TURN_MESSAGE = "Error: Task interrupted before a response was generated."


def restore_pending_interruption(session: Session) -> bool:
    """Close a saved user-only turn without pretending it was answered.

    Returns True if the session changed (and should be saved).
    """
    if not session.metadata.get(PENDING_USER_TURN_KEY):
        return False
    if session.messages and session.messages[-1].get("role") == "user":
        session.messages.append(
            {
                "role": "assistant",
                "content": INTERRUPTED_TURN_MESSAGE,
                "timestamp": datetime.now().isoformat(),
                "_recovery_interrupted": True,
            }
        )
        session.updated_at = datetime.now()
    session.metadata.pop(PENDING_USER_TURN_KEY, None)
    return True
