"""Helpers for runtime model preset selection (nanobot agent/model_presets.py).

Without nanobot's provider snapshots: a preset resolves straight to an
``LLMRuntime``, either with a provider built for it from the config
(``make_preset_runtime_loader``) or reusing the current provider with the
preset's model and limits (``build_static_preset_runtime``, used when no
config is at hand, e.g. a provider handed in by a test).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from nanoclaw.config.schema import Config, ModelPresetConfig
from nanoclaw.providers.base import LLMProvider
from nanoclaw.utils.llm_runtime import LLMRuntime

__all__ = [
    "PresetRuntimeLoader",
    "build_runtime_preset",
    "build_static_preset_runtime",
    "configured_model_presets",
    "make_preset_runtime_loader",
    "normalize_preset_name",
]

PresetRuntimeLoader = Callable[[str], LLMRuntime]


def configured_model_presets(config: Config) -> dict[str, ModelPresetConfig]:
    """The named presets plus ``default`` (agents.defaults) (model_presets.py:24)."""
    return {**config.model_presets, "default": config.resolve_default_preset()}


def make_preset_runtime_loader(config: Config) -> PresetRuntimeLoader:
    """Build a fresh provider for a preset name (model_presets.py:43)."""
    from nanoclaw.providers.factory import build_preset_runtime

    return lambda name: build_preset_runtime(config, preset_name=name)


def build_static_preset_runtime(
    provider: LLMProvider, name: str, preset: ModelPresetConfig
) -> LLMRuntime:
    """The current provider with the preset's settings (model_presets.py:52)."""
    return LLMRuntime(
        provider=provider,
        model=preset.model,
        max_tokens=preset.max_tokens,
        temperature=preset.temperature,
        context_window_tokens=preset.context_window_tokens,
        reasoning_effort=preset.reasoning_effort,
        model_preset=name,
    )


def build_runtime_preset(
    *,
    name: str,
    presets: dict[str, ModelPresetConfig],
    provider: LLMProvider,
    loader: PresetRuntimeLoader | None,
) -> LLMRuntime:
    """model_presets.py:66 (``build_runtime_preset_snapshot``)."""
    if loader is not None:
        return replace(loader(name), model_preset=name)
    return build_static_preset_runtime(provider, name, presets[name])


def normalize_preset_name(
    name: str | None, presets: dict[str, ModelPresetConfig]
) -> str:
    """The configured spelling of ``name``, matched case-insensitively (model_presets.py:78)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model_preset must be a non-empty string")
    name = name.strip()
    if name in presets:
        return name
    matches = [
        candidate for candidate in presets if candidate.casefold() == name.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"model_preset {name!r} not found. Available: {', '.join(presets) or '(none)'}"
    )
