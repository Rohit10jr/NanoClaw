"""Chat channels (nanobot channels/): the base interface and the manager."""

from nanoclaw.channels.base import BaseChannel
from nanoclaw.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
