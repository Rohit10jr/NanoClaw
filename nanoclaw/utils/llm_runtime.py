"""The model settings a request runs with.

Mirrors nanobot/utils/llm_runtime.py (``LLMRuntime``), flattened: nanobot keeps
the generation settings in a nested ``GenerationSettings`` and adds model
presets and provider snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanoclaw.providers import LLMProvider


@dataclass(frozen=True)
class LLMRuntime:
    provider: LLMProvider
    model: str | None
    max_tokens: int
    temperature: float
    context_window_tokens: int
