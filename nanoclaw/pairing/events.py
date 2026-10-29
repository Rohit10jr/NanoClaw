"""Transport-independent pairing snapshot invalidation."""

from dataclasses import dataclass

from nanoclaw.events import AgentEvent


@dataclass(frozen=True)
class PairingChanged(AgentEvent):
    """Pairing state may have changed; consumers must read the current store."""
