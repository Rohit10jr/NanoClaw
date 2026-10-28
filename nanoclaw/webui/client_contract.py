"""The contract between a shipped web UI and the gateway (nanobot webui/client_contract.py).

NanoClaw advertises only ``webui.core.v1``: the frontend then hides the
subagent panels, the self-update flow, prompt navigation, automation chats
and binary uploads, none of which NanoClaw serves.
"""

from __future__ import annotations

from nanoclaw import __version__

WEBUI_PROTOCOL = 1
CORE_CAPABILITY = "webui.core.v1"


def webui_contract() -> dict[str, object]:
    return {
        "version": __version__,
        "min_protocol": WEBUI_PROTOCOL,
        "max_protocol": WEBUI_PROTOCOL,
        "capabilities": [CORE_CAPABILITY],
    }


def gateway_identity(gateway_id: str) -> dict[str, object]:
    """``terminal`` in bootstrap (``protocolVersion`` is the terminal transport's)."""
    return {"protocolVersion": 1, "gatewayId": gateway_id, "webui": webui_contract()}
