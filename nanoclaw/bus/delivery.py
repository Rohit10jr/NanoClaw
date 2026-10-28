"""Delivery outcome errors a channel raises from ``send``. (nanobot bus/delivery.py:11-16)

nanobot's scheduled-delivery requests and outbox are not ported; the manager
only uses these to decide whether a failed send is retried.
"""

from __future__ import annotations


class ChannelNotReadyError(RuntimeError):
    """No part of the message was sent; retry after the channel reconnects."""


class DeliveryRejectedError(RuntimeError):
    """No part of the message was sent; configuration or destination needs repair."""
