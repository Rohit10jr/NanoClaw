"""Create LLM providers from config.

Mirrors nanobot/providers/factory.py (``make_provider``,
``_make_provider_core``, ``_provider_extra_headers``) for NanoClaw's two
backends: ``openai_compat`` (``OpenAICompatProvider``) and ``anthropic``
(``nanoclaw.providers.anthropic_provider.AnthropicProvider``, imported only
when a route needs it). nanobot's provider snapshots and signatures (used
for config hot reload) are left out; ``build_preset_runtime`` gives the
per-preset ``LLMRuntime`` instead.

With ``agents.defaults.fallbackModels`` set, ``make_provider`` wraps the
provider in a ``FallbackProvider`` whose factory builds plain providers for the
fallback presets (``_resolve_fallback_presets``, ``_inline_fallback_preset``),
and ``build_preset_runtime`` uses the smallest context window of the chain.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from nanoclaw.config.loader import (
    ENV_API_BASE,
    ENV_API_KEY,
    ENV_MODEL,
    apply_env_overrides,
)
from nanoclaw.config.schema import (
    DEFAULT_MODEL,
    Config,
    InlineFallbackConfig,
    ModelPresetConfig,
    ProviderConfig,
)
from nanoclaw.providers.base import LLMProvider
from nanoclaw.providers.fallback_provider import FallbackProvider
from nanoclaw.providers.openai_compat import OpenAICompatProvider
from nanoclaw.providers.registry import ProviderSpec
from nanoclaw.providers.routing import ProviderRoute, resolve_provider_route
from nanoclaw.utils.llm_runtime import LLMRuntime

__all__ = [
    "DEFAULT_MODEL",
    "ENV_API_BASE",
    "ENV_API_KEY",
    "ENV_MODEL",
    "build_preset_runtime",
    "chain_context_window_tokens",
    "make_provider",
    "provider_from_config",
    "provider_from_env",
]

# The Anthropic Messages adapter (task 20.2). Imported lazily so the
# ``anthropic`` SDK is only needed by configs that route to it.
ANTHROPIC_PROVIDER_MODULE = "nanoclaw.providers.anthropic_provider"


def _resolve_model_preset(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
) -> ModelPresetConfig:
    return preset if preset is not None else config.resolve_preset(preset_name)


def _provider_extra_headers(
    spec: ProviderSpec | None, provider_config: ProviderConfig | None
) -> dict[str, str] | None:
    """The spec's default headers, overridden by the config's (factory.py:45)."""
    headers = dict(spec.default_extra_headers) if spec else {}
    if provider_config and provider_config.extra_headers:
        names = {key.lower() for key in provider_config.extra_headers}
        headers = {
            key: value for key, value in headers.items() if key.lower() not in names
        }
        headers.update(provider_config.extra_headers)
    return headers or None


def _make_anthropic_provider(
    route: ProviderRoute, preset: ModelPresetConfig
) -> LLMProvider:
    """The ``anthropic`` backend (nanobot factory.py:117)."""
    try:
        module = importlib.import_module(ANTHROPIC_PROVIDER_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name not in (ANTHROPIC_PROVIDER_MODULE, "anthropic"):
            raise
        raise ValueError(
            f"Provider '{route.provider_name}' uses the Anthropic Messages API, "
            f"which this NanoClaw build cannot load ({exc.name} is missing)."
        ) from None
    p, spec = route.provider_config, route.spec
    # A custom OpenAI-compatible endpoint that speaks Anthropic Messages.
    custom_spec = spec if spec and spec.backend == "openai_compat" else None
    api_key = p.api_key if p else None
    if custom_spec is not None:
        api_key = api_key or "no-key"
    return module.AnthropicProvider(
        api_key=api_key,
        api_base=route.api_base,
        default_model=route.model,
        extra_headers=_provider_extra_headers(spec, p),
        extra_body=p.extra_body if p else None,
        extra_query=p.extra_query if p else None,
        spec=custom_spec,
        provider_name=route.provider_name,
        # NanoClaw's stand-in for nanobot's provider.generation.reasoning_effort.
        reasoning_effort=preset.reasoning_effort,
    )


def _make_provider_core(
    config: Config, *, preset: ModelPresetConfig, model: str | None = None
) -> LLMProvider:
    """A plain provider for ``preset`` (factory.py:56). Raises ``ValueError`` for a bad setup."""
    route = resolve_provider_route(config, preset=preset, model=model)
    if route.backend == "anthropic":
        return _make_anthropic_provider(route, preset)
    p = route.provider_config
    return OpenAICompatProvider(
        api_key=p.api_key if p else None,
        api_base=route.api_base,
        default_model=route.model,
        extra_headers=_provider_extra_headers(route.spec, p),
        spec=route.spec,
        extra_body=p.extra_body if p else None,
        extra_query=p.extra_query if p else None,
        reasoning_effort=preset.reasoning_effort,
        provider_name=route.provider_name or "openai",
    )


def _inline_fallback_preset(primary: ModelPresetConfig, fallback: InlineFallbackConfig) -> ModelPresetConfig:
    """An inline fallback as a preset; unset limits come from the primary (factory.py:174)."""
    return ModelPresetConfig(
        model=fallback.model,
        provider=fallback.provider,
        max_tokens=fallback.max_tokens if fallback.max_tokens is not None else primary.max_tokens,
        context_window_tokens=(
            fallback.context_window_tokens
            if fallback.context_window_tokens is not None
            else primary.context_window_tokens
        ),
        temperature=fallback.temperature if fallback.temperature is not None else primary.temperature,
        reasoning_effort=fallback.reasoning_effort,
        extra_headers=fallback.extra_headers,
        extra_query=fallback.extra_query,
        extra_body=fallback.extra_body,
    )


def _resolve_fallback_presets(config: Config, primary: ModelPresetConfig) -> list[ModelPresetConfig]:
    """``agents.defaults.fallbackModels`` as presets, in order (factory.py:193)."""
    presets: list[ModelPresetConfig] = []
    for fallback in config.agents.defaults.fallback_models:
        if isinstance(fallback, str):
            presets.append(config.model_presets[fallback])
        else:
            presets.append(_inline_fallback_preset(primary, fallback))
    return presets


def make_provider(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
    model: str | None = None,
) -> LLMProvider:
    """The provider the config (its active preset, or ``preset_name``) implies (factory.py:202).

    ``model`` overrides the preset's model. With fallback models configured
    the provider is a ``FallbackProvider``; its fallback providers are built
    when a request needs them, so a fallback with a bad setup fails then
    (and is skipped), not here.
    """
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    provider = _make_provider_core(config, preset=resolved, model=model)
    fallback_presets = _resolve_fallback_presets(config, resolved)

    if fallback_presets:
        provider = FallbackProvider(
            primary=provider,
            fallback_presets=fallback_presets,
            provider_factory=lambda fb: _make_provider_core(config, preset=fb),
            primary_context_window_tokens=resolved.context_window_tokens,
            fallback_preset_names=[
                fallback if isinstance(fallback, str) else None
                for fallback in config.agents.defaults.fallback_models
            ],
        )

    return provider


def chain_context_window_tokens(config: Config, preset: ModelPresetConfig) -> int:
    """The smallest context window of ``preset`` and its fallbacks.

    A fallback may answer any request, so the context budget must fit every
    model of the chain (nanobot ``build_provider_snapshot``, factory.py:296).
    """
    fallback_windows = [fallback.context_window_tokens for fallback in _resolve_fallback_presets(config, preset)]
    return min([preset.context_window_tokens, *fallback_windows])


def build_preset_runtime(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
) -> LLMRuntime:
    """A new provider and the turn settings for a preset (nanobot
    ``build_provider_snapshot`` + ``runtime_from_provider_snapshot``).

    ``preset_name`` None means the active preset (``agents.defaults.modelPreset``).
    """
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    selected = (
        config.agents.defaults.model_preset
        if preset_name is None and preset is None
        else preset_name
    )
    return LLMRuntime(
        provider=make_provider(config, preset=resolved),
        model=resolved.model,
        max_tokens=resolved.max_tokens,
        temperature=resolved.temperature,
        context_window_tokens=chain_context_window_tokens(config, resolved),
        reasoning_effort=resolved.reasoning_effort,
        model_preset=selected,
    )


def provider_from_config(config: Config) -> LLMProvider:
    """The provider for ``config``'s active preset (env overrides already applied).

    NanoClaw's name for ``make_provider(config)``.
    """
    return make_provider(config)


def provider_from_env(env: Mapping[str, str] | None = None) -> LLMProvider:
    """Create the provider from ``NANOCLAW_*`` variables alone (no config file)."""
    return provider_from_config(apply_env_overrides(Config(), env))
