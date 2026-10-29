"""Resolve the provider connection for a preset without creating a client.

Mirrors nanobot/providers/routing.py (``ProviderRoute``,
``resolve_provider_route``, ``validate_provider_setup``) and
providers/request_overrides.py, without nanobot's request-API declarations
(``api``), proxies and the backends NanoClaw does not have.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from nanoclaw.config.schema import (
    Config,
    ModelPresetConfig,
    ProviderConfig,
    RequestOverrides,
)
from nanoclaw.providers.registry import ProviderSpec, create_dynamic_spec, find_by_name

__all__ = [
    "ProviderRoute",
    "apply_request_overrides",
    "isolate_request_overrides",
    "resolve_provider_route",
    "validate_provider_setup",
]


@dataclass(frozen=True)
class ProviderRoute:
    """Validated connection and adapter choice, without a client (routing.py:17).

    ``provider_name`` is None only when nothing is configured at all; NanoClaw
    then talks to api.openai.com without a key (nanobot refuses; see
    ``resolve_provider_route``).
    """

    model: str
    provider_name: str | None
    provider_config: ProviderConfig | None
    spec: ProviderSpec | None
    backend: str
    api_base: str | None


def _provider_spec_for_config(
    provider_name: str, provider_config: ProviderConfig | None
) -> ProviderSpec | None:
    """The registry spec, or a dynamic one for a custom provider (routing.py:35)."""
    spec = find_by_name(provider_name)
    if spec is None and provider_config is not None:
        if not provider_config.api_base:
            raise ValueError(f"Provider '{provider_name}' requires api_base in config.")
        spec = create_dynamic_spec(
            provider_name,
            display_name=provider_config.display_name or "",
            thinking_style=provider_config.thinking_style or "",
        )
    return spec


def isolate_request_overrides(
    candidate: ModelPresetConfig, *, previous_provider: str | None, provider: str | None
) -> ModelPresetConfig:
    """Keep inherited request settings only within the same connection (request_overrides.py:8)."""
    if previous_provider is not None and previous_provider == provider:
        return candidate
    return candidate.model_copy(
        update={"extra_headers": None, "extra_query": None, "extra_body": None}
    )


def apply_request_overrides(
    connection: ProviderConfig | None, overrides: RequestOverrides
) -> ProviderConfig | None:
    """Overlay a preset's extra headers/query/body on a copy of the connection (request_overrides.py:26).

    Headers match without case sensitivity; query and body keys are exact;
    nested body values replace as a unit. None and empty maps inherit.
    """
    if not any((overrides.extra_headers, overrides.extra_query, overrides.extra_body)):
        return connection.model_copy(deep=True) if connection is not None else None
    result = (
        connection.model_copy(deep=True) if connection is not None else ProviderConfig()
    )
    for field in ("extra_headers", "extra_query", "extra_body"):
        values = getattr(overrides, field)
        if not values:
            continue
        merged = dict(getattr(result, field) or {})
        if field == "extra_headers":
            names = {key.lower() for key in values}
            merged = {
                key: value for key, value in merged.items() if key.lower() not in names
            }
        merged.update(deepcopy(values))
        setattr(result, field, merged)
    return result


def resolve_provider_route(
    config: Config, *, preset: ModelPresetConfig, model: str | None = None
) -> ProviderRoute:
    """Resolve and validate the provider for ``preset`` (routing.py:95).

    Raises ``ValueError`` with nanobot's messages for a provider that needs an
    apiBase or an API key it lacks. Two deliberate differences:

    - A ``custom`` entry without an apiBase means api.openai.com (NanoClaw's
      phase 1 setup, ``NANOCLAW_API_KEY`` alone); nanobot requires apiBase.
    - Nothing configured at all under ``provider: "auto"`` is not an error
      here: the route has no provider name and requests go to api.openai.com
      without a key, failing with the server's 401 (nanobot refuses with "No
      provider is configured"). A forced provider that matches nothing is
      refused, as in nanobot.
    """
    if model is not None:
        preset = isolate_request_overrides(
            preset.model_copy(update={"model": model}),
            previous_provider=config.get_provider_name(preset=preset),
            provider=config.get_provider_name(model, preset=preset),
        )
    model = model or preset.model
    provider_name = config.get_provider_name(model, preset=preset)
    provider_config = config.get_provider(model, preset=preset)
    if not provider_name:
        if preset.provider != "auto":
            raise ValueError(f"No provider is configured for model '{model}'.")
        return ProviderRoute(
            model=model,
            provider_name=None,
            provider_config=apply_request_overrides(None, preset),
            spec=None,
            backend="openai_compat",
            api_base=None,
        )
    spec = _provider_spec_for_config(provider_name, provider_config)
    backend = spec.backend if spec else "openai_compat"
    provider_config = apply_request_overrides(provider_config, preset)
    api_base = config.get_api_base(model, preset=preset)

    if (
        spec
        and spec.is_direct
        and not spec.default_api_base
        and not api_base
        and spec.name != "custom"
    ):
        raise ValueError(f"Provider '{provider_name}' requires api_base in config.")
    needs_key = not (provider_config and provider_config.api_key)
    exempt = spec and (spec.is_local or spec.is_direct)
    if needs_key and not exempt:
        raise ValueError(f"No API key configured for provider '{provider_name}'.")

    return ProviderRoute(
        model=model,
        provider_name=provider_name,
        provider_config=provider_config,
        spec=spec,
        backend=backend,
        api_base=api_base,
    )


def validate_provider_setup(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
    model: str | None = None,
) -> None:
    """Validate provider/model settings without creating a client (routing.py:196)."""
    resolved = preset if preset is not None else config.resolve_preset(preset_name)
    resolve_provider_route(config, preset=resolved, model=model)
