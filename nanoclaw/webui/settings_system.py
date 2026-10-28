"""The pairing part of nanobot's system settings domain (webui/settings_system.py).

``pairing_payload`` is what ``GET /api/settings/pairing`` answers and what
the ``settings.pairing.approve`` / ``settings.pairing.deny`` mutations
answer on success (with ``last_action``); ``pairing_action`` is nanobot's
``SystemSettingsHandler._pairing_action``. Both take the store operations
as arguments, as nanobot's ``SystemSettingsOperations`` does, so callers
can bind ``on_change`` (the ``PairingChanged`` publication).

The rest of nanobot's system settings (channels, features, updates, CLI
apps) is not ported.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

ListPendingPairings = Callable[[], Iterable[dict[str, Any]]]
ApprovePairingCode = Callable[[str], "tuple[str, str] | None"]
DenyPairingCode = Callable[[str], bool]


def pairing_payload(
    list_pending: ListPendingPairings,
    last_action: dict[str, Any] | None = None,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """(nanobot settings_system.py:411)"""
    current_time = time.time() if now is None else now
    requests: list[dict[str, Any]] = []
    for item in list_pending():
        expires_at = float(item.get("expires_at", 0) or 0)
        created_at = float(item.get("created_at", 0) or 0)
        delivery_status = item.get("delivery_status")
        requests.append(
            {
                "code": str(item.get("code", "")),
                "channel": str(item.get("channel", "")),
                "sender_id": str(item.get("sender_id", "")),
                "created_at_ms": int(created_at * 1000) if created_at else None,
                "expires_at_ms": int(expires_at * 1000) if expires_at else None,
                "expires_in_seconds": (
                    max(0, int(expires_at - current_time)) if expires_at else None
                ),
                **(
                    {"delivery_status": delivery_status}
                    if isinstance(delivery_status, str)
                    and delivery_status in {"sending", "sent", "failed"}
                    else {}
                ),
            }
        )
    payload: dict[str, Any] = {"requests": requests}
    if last_action is not None:
        payload["last_action"] = last_action
    return payload


def pairing_action(
    code: str | None,
    action: str,
    *,
    list_pending: ListPendingPairings,
    approve_code: ApprovePairingCode,
    deny_code: DenyPairingCode,
) -> tuple[int, dict[str, Any] | str]:
    """Approve or deny one code: ``(200, payload)`` or ``(status, error text)``.
    (nanobot settings_system.py:1154 ``_pairing_action``)"""
    code = (code or "").strip()
    if not code:
        return 400, "Missing pairing code"
    if action == "approve":
        result = approve_code(code)
        if result is None:
            return 404, "Pairing code not found or expired"
        channel, sender_id = result
        return 200, pairing_payload(
            list_pending,
            {
                "ok": True,
                "action": "approve",
                "message": f"Approved {sender_id} for {channel}",
                "channel": channel,
                "sender_id": sender_id,
                "code": code,
            },
        )
    if not deny_code(code):
        return 404, "Pairing code not found or expired"
    return 200, pairing_payload(
        list_pending,
        {
            "ok": True,
            "action": "deny",
            "message": f"Denied pairing code {code}",
            "code": code,
        },
    )
