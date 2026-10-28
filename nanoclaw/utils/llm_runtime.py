"""The model settings a request runs with.

Mirrors nanobot/utils/llm_runtime.py (``LLMRuntime``), flattened: nanobot keeps
the generation settings (temperature, max_tokens, reasoning_effort) in a
nested ``GenerationSettings`` and adds a provider snapshot signature for
config hot reload, which NanoClaw does not have.

``reasoning_effort`` is informational here: the provider built for a preset
already carries it (``OpenAICompatProvider(reasoning_effort=...)``), because
NanoClaw's ``chat`` calls take no effort argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the providers package imports this module (factory)
    from nanoclaw.providers.base import LLMProvider


@dataclass(frozen=True)
class LLMRuntime:
    provider: LLMProvider
    model: str | None
    max_tokens: int
    temperature: float
    context_window_tokens: int
    reasoning_effort: str | None = None
    # The preset this runtime came from (None: agents.defaults, unnamed).
    model_preset: str | None = None
