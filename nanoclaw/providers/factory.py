"""Build a provider from settings.

nanobot resolves providers through a registry of ~30 specs plus config
presets (providers/registry.py, providers/factory.py). NanoClaw has one
OpenAI-compatible provider: the config picks an entry under ``providers``
(``Config.resolve_provider``) and the ``NANOCLAW_*`` variables override it.
"""

from __future__ import annotations

from collections.abc import Mapping

from nanoclaw.config.loader import ENV_API_BASE, ENV_API_KEY, ENV_MODEL, apply_env_overrides
from nanoclaw.config.schema import DEFAULT_MODEL, Config
from nanoclaw.providers.openai_compat import OpenAICompatProvider

__all__ = ["DEFAULT_MODEL", "ENV_API_BASE", "ENV_API_KEY", "ENV_MODEL", "provider_from_config", "provider_from_env"]


def provider_from_config(config: Config) -> OpenAICompatProvider:
    """Create the OpenAI-compatible provider for ``config`` (env overrides already applied).

    No API base means api.openai.com. Examples: OpenRouter
    ``https://openrouter.ai/api/v1``, DeepSeek ``https://api.deepseek.com``,
    local Ollama ``http://localhost:11434/v1``.
    """
    return OpenAICompatProvider(
        api_key=config.get_api_key(),
        api_base=config.get_api_base(),
        default_model=config.agents.defaults.model or DEFAULT_MODEL,
    )


def provider_from_env(env: Mapping[str, str] | None = None) -> OpenAICompatProvider:
    """Create the provider from ``NANOCLAW_*`` variables alone (no config file)."""
    return provider_from_config(apply_env_overrides(Config(), env))
