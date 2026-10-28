"""Model selection for the agent loop (nanobot agent/model_runtime.py ``ModelRuntimeResolver``).

The resolver owns the default ``LLMRuntime`` (provider, model, limits) and
the named presets. ``resolve_preset`` builds a preset's runtime once and
caches it, so sessions that picked a preset with ``/model`` share one
provider; ``aclose`` closes every provider it handed out.

Left out from nanobot: config hot reload (``invalidate``/``refresh`` and
provider signatures), per-run SDK overrides (``resolve_override``) and
provider generation tracking.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast

from nanoclaw.agent import model_presets as preset_helpers
from nanoclaw.config.schema import ModelPresetConfig
from nanoclaw.providers.base import LLMProvider
from nanoclaw.utils.llm_runtime import LLMRuntime

__all__ = ["ModelRuntimeResolver"]


class ModelRuntimeResolver:
    """Own model selection and resolve it to immutable runtime values."""

    def __init__(
        self,
        initial_runtime: LLMRuntime,
        *,
        model_presets: Mapping[str, ModelPresetConfig] | None = None,
        preset_runtime_loader: preset_helpers.PresetRuntimeLoader | None = None,
    ) -> None:
        self._runtime = initial_runtime
        self._providers: list[LLMProvider] = [initial_runtime.provider]
        self._model_presets = dict(model_presets or {})
        self._preset_runtime_loader = preset_runtime_loader
        self._resolved_presets: dict[str, LLMRuntime] = {}

    @property
    def runtime(self) -> LLMRuntime:
        """The default runtime for turns without a session preset."""
        return self._runtime

    @property
    def model_presets(self) -> Mapping[str, ModelPresetConfig]:
        return MappingProxyType(
            {
                name: preset.model_copy(deep=True)
                for name, preset in self._model_presets.items()
            }
        )

    @property
    def model_preset(self) -> str | None:
        return self._runtime.model_preset

    def _track(self, provider: LLMProvider) -> None:
        if not any(known is provider for known in self._providers):
            self._providers.append(provider)

    def update_default(self, **changes: Any) -> LLMRuntime:
        """Change fields of the default runtime (the loop's ``provider``, ``model``, ... setters)."""
        self._runtime = replace(self._runtime, **changes)
        self._track(self._runtime.provider)
        return self._runtime

    def resolve_preset(self, name: str | None) -> LLMRuntime:
        """A named preset's runtime, without changing the default (model_runtime.py:186)."""
        normalized = preset_helpers.normalize_preset_name(name, self._model_presets)
        cached = self._resolved_presets.get(normalized)
        if cached is not None:
            return cached
        runtime = preset_helpers.build_runtime_preset(
            name=normalized,
            presets=self._model_presets,
            provider=self._runtime.provider,
            loader=self._preset_runtime_loader,
        )
        self._track(runtime.provider)
        self._resolved_presets[normalized] = runtime
        return runtime

    def select_preset(self, name: str | None) -> LLMRuntime:
        """Make a named preset the default for future turns (model_runtime.py:204)."""
        self._runtime = self.resolve_preset(name)
        return self._runtime

    def select_model(self, model: str) -> LLMRuntime:
        """Change the default model on the current provider (model_runtime.py:211)."""
        if not isinstance(cast(object, model), str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        self._runtime = replace(self._runtime, model=model.strip(), model_preset=None)
        return self._runtime

    def select_context_window(self, context_window_tokens: int) -> LLMRuntime:
        """Change the default context limit (model_runtime.py:222)."""
        raw = cast(object, context_window_tokens)
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise TypeError("context_window_tokens must be an integer")
        self._runtime = replace(
            self._runtime, context_window_tokens=context_window_tokens
        )
        return self._runtime

    async def aclose(self) -> None:
        """Close every provider handed out, including replaced ones (model_runtime.py:166)."""
        providers, self._providers = self._providers, [self._runtime.provider]
        self._resolved_presets.clear()
        closing = []
        for provider in providers:
            # Test doubles may not be LLMProviders; a closed client reopens lazily.
            close = getattr(provider, "aclose", None)
            result = close() if callable(close) else None
            if inspect.isawaitable(result):
                closing.append(result)
        results = await asyncio.gather(*closing, return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup("failed to close model providers", errors)
