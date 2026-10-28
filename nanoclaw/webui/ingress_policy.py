"""Limits for inbound web UI messages (nanobot webui/ingress_policy.py).

NanoClaw accepts no attachments; the attachment limits are still reported in
bootstrap (nanobot's defaults, channels/websocket/attachment_policy.py:7) so
the frontend's size estimates match nanobot's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MessageRejection = Literal["text_too_large"]


@dataclass(frozen=True)
class MessageIngressLimits:
    max_text_bytes: int = 64 * 1024


@dataclass(frozen=True)
class AttachmentIngressLimits:
    max_count: int = 4
    max_file_bytes: int = 6 * 1024 * 1024
    max_total_bytes: int = 24 * 1024 * 1024


@dataclass(frozen=True)
class WebUIIngressPolicy:
    """Limits applied after the channel has decoded the transport envelope."""

    message: MessageIngressLimits = field(default_factory=MessageIngressLimits)
    attachments: AttachmentIngressLimits = field(default_factory=AttachmentIngressLimits)
    envelope_reserve_bytes: int = 64 * 1024

    def validate_text(self, content: str) -> MessageRejection | None:
        if len(content.encode("utf-8")) > self.message.max_text_bytes:
            return "text_too_large"
        return None

    def bootstrap_limits(self, *, max_frame_bytes: int) -> dict[str, object]:
        return {
            "transport": {
                "max_frame_bytes": max_frame_bytes,
                "envelope_reserve_bytes": self.envelope_reserve_bytes,
            },
            "message": {"max_text_bytes": self.message.max_text_bytes},
            "attachments": {
                "max_count": self.attachments.max_count,
                "max_file_bytes": self.attachments.max_file_bytes,
                "max_total_bytes": self.attachments.max_total_bytes,
            },
        }


DEFAULT_WEBUI_INGRESS_POLICY = WebUIIngressPolicy()
