"""Build a provider from settings.

nanobot resolves providers through a registry of ~30 specs plus config
presets (providers/registry.py, providers/factory.py). NanoClaw reads three
environment variables instead.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from nanoclaw.providers.openai_compat import OpenAICompatProvider

ENV_API_KEY = "NANOCLAW_API_KEY"
ENV_API_BASE = "NANOCLAW_API_BASE"
ENV_MODEL = "NANOCLAW_MODEL"

DEFAULT_MODEL = "gpt-4o-mini"


def provider_from_env(env: Mapping[str, str] | None = None) -> OpenAICompatProvider:
    """Create an OpenAI-compatible provider from ``NANOCLAW_*`` variables.

    ``NANOCLAW_API_BASE`` unset means api.openai.com. Examples:
    OpenRouter ``https://openrouter.ai/api/v1``, DeepSeek
    ``https://api.deepseek.com``, local Ollama ``http://localhost:11434/v1``.
    """
    env = os.environ if env is None else env
    return OpenAICompatProvider(
        api_key=env.get(ENV_API_KEY) or None,
        api_base=env.get(ENV_API_BASE) or None,
        default_model=env.get(ENV_MODEL) or DEFAULT_MODEL,
    )
