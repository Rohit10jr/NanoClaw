"""Provider registry: the metadata of each LLM provider NanoClaw knows.

Mirrors nanobot/providers/registry.py (``ProviderSpec``, ``PROVIDERS``,
``find_by_name``, ``create_dynamic_spec``) for the providers worth having in
a small agent, with nanobot's names, order and values. Each spec keeps only
the fields NanoClaw reads; nanobot's model catalogs, Responses-API
capabilities, OAuth and transcription flags are left out.

Order matters, as in nanobot: it is the priority for keyword matching and for
the "first provider with a key" fallback (``Config._match_provider``).
nanobot's Azure OpenAI, AWS Bedrock, OAuth (Codex, Copilot, xAI) and
regional gateways (VolcEngine, BytePlus, ...) are not here; a config entry
with one of those names is treated as a custom OpenAI-compatible provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic.alias_generators import to_snake

__all__ = ["PROVIDERS", "ProviderSpec", "create_dynamic_spec", "find_by_name"]


@dataclass(frozen=True)
class ProviderSpec:
    """One provider's metadata (nanobot registry.py:50, the fields NanoClaw uses)."""

    # identity
    name: str  # config key, e.g. "dashscope"
    keywords: tuple[str, ...]  # model-name keywords for matching (lowercase)
    env_key: str  # the provider's usual API-key variable (informational: NanoClaw reads NANOCLAW_API_KEY)
    display_name: str = ""

    # which provider implementation to use: "openai_compat" or "anthropic"
    backend: str = "openai_compat"

    # request headers the provider integration always sends
    default_extra_headers: tuple[tuple[str, str], ...] = ()

    # gateway / local detection
    is_gateway: bool = False  # routes any model (OpenRouter, AiHubMix)
    is_local: bool = False  # local deployment (vLLM, Ollama)
    # Kept as in nanobot; nanobot's routing no longer reads it (only the
    # Web UI does), so neither does NanoClaw's.
    detect_by_key_prefix: str = ""
    detect_by_base_keyword: str = ""  # substring of api_base: picks a local provider
    default_api_base: str = ""  # OpenAI-compatible base URL for this provider

    # gateway behavior
    strip_model_prefix: bool = False  # strip "provider/" before sending to gateway
    strip_model_prefixes: tuple[
        str, ...
    ] = ()  # strip only when the first model segment matches
    supports_max_completion_tokens: bool = False

    # per-model param overrides, e.g. (("kimi-k2.7", {"temperature": 1.0}),)
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()

    # Direct providers skip API-key validation (user supplies everything)
    is_direct: bool = False

    # Provider supports cache_control on content blocks (Anthropic prompt
    # caching): read by the Anthropic adapter, and by the OpenAI-compatible
    # adapter for Claude models through a caching gateway (OpenRouter).
    supports_prompt_caching: bool = False

    # How to send the thinking on/off switch in extra_body when a
    # reasoning_effort is set (nanobot openai_compat_provider.py:116):
    # "thinking_type" {"thinking": {"type": ...}}, "enable_thinking"
    # {"enable_thinking": bool}, "reasoning_split" {"reasoning_split": bool}.
    thinking_style: str = ""

    # Gateway-native reasoning control paired with model-level thinking
    # styles: "reasoning_effort" sends {"reasoning": {"effort": ...}} (OpenRouter).
    gateway_reasoning_style: str = ""

    # reasoning_effort (OpenAI vocabulary) -> the value this provider accepts;
    # an empty mapped value omits the field.
    reasoning_effort_remap: tuple[tuple[str, str], ...] = ()

    # Models that always reason and reject the reasoning_effort field
    # (substring of the lowercased model name).
    implicit_reasoning_models: tuple[str, ...] = ()

    # Drop reasoning_content from assistant history before sending (Mistral
    # rejects unknown fields).
    strip_history_reasoning_content: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.name.title()

    def request_model_name(self, model: str) -> str:
        """Remove only the provider prefixes owned by this adapter (nanobot registry.py:157)."""
        if "/" not in model:
            return model
        if self.strip_model_prefix:
            return model.rsplit("/", 1)[-1]
        prefix, routed_model = model.split("/", 1)
        prefix_key = to_snake(prefix.replace("-", "_")).lower()
        if any(
            to_snake(value.replace("-", "_")).lower() == prefix_key
            for value in self.strip_model_prefixes
        ):
            return routed_model
        return model


# ---------------------------------------------------------------------------
# PROVIDERS: the registry. Order = priority (nanobot registry.py:219).
# ---------------------------------------------------------------------------

PROVIDERS: tuple[ProviderSpec, ...] = (
    # === Custom (direct OpenAI-compatible endpoint), registry.py:224 =========
    ProviderSpec(
        name="custom",
        keywords=(),
        env_key="",
        display_name="Custom",
        is_direct=True,
    ),
    # === Gateways (detected by api_key / api_base, not model name) ===========
    # OpenRouter: global gateway, keys start with "sk-or-" (registry.py:271)
    ProviderSpec(
        name="openrouter",
        keywords=("openrouter",),
        env_key="OPENROUTER_API_KEY",
        display_name="OpenRouter",
        is_gateway=True,
        detect_by_key_prefix="sk-or-",
        detect_by_base_keyword="openrouter",
        default_api_base="https://openrouter.ai/api/v1",
        supports_prompt_caching=True,
        gateway_reasoning_style="reasoning_effort",
    ),
    # Hugging Face Inference Providers router (registry.py:359)
    ProviderSpec(
        name="huggingface",
        keywords=("huggingface", "hugging-face"),
        env_key="HF_TOKEN",
        display_name="Hugging Face",
        is_gateway=True,
        detect_by_key_prefix="hf_",
        detect_by_base_keyword="huggingface",
        default_api_base="https://router.huggingface.co/v1",
    ),
    # AiHubMix: global gateway, sends the bare model name (registry.py:388)
    ProviderSpec(
        name="aihubmix",
        keywords=("aihubmix",),
        env_key="OPENAI_API_KEY",
        display_name="AiHubMix",
        is_gateway=True,
        detect_by_base_keyword="aihubmix",
        default_api_base="https://aihubmix.com/v1",
        strip_model_prefix=True,
    ),
    # SiliconFlow (registry.py:401)
    ProviderSpec(
        name="siliconflow",
        keywords=("siliconflow",),
        env_key="OPENAI_API_KEY",
        display_name="SiliconFlow",
        is_gateway=True,
        detect_by_base_keyword="siliconflow",
        default_api_base="https://api.siliconflow.cn/v1",
    ),
    # === Standard providers (matched by model-name keywords) =================
    # Anthropic: native Messages API (registry.py:488)
    ProviderSpec(
        name="anthropic",
        keywords=("anthropic", "claude"),
        env_key="ANTHROPIC_API_KEY",
        display_name="Anthropic",
        backend="anthropic",
        supports_prompt_caching=True,
    ),
    # OpenAI (registry.py:498)
    ProviderSpec(
        name="openai",
        keywords=("openai", "gpt"),
        env_key="OPENAI_API_KEY",
        display_name="OpenAI",
        supports_max_completion_tokens=True,
    ),
    # DeepSeek (registry.py:637)
    ProviderSpec(
        name="deepseek",
        keywords=("deepseek",),
        env_key="DEEPSEEK_API_KEY",
        display_name="DeepSeek",
        default_api_base="https://api.deepseek.com",
        thinking_style="thinking_type",
    ),
    # Gemini through Google's OpenAI-compatible endpoint (registry.py:679)
    ProviderSpec(
        name="gemini",
        keywords=("gemini", "gemma"),
        env_key="GEMINI_API_KEY",
        display_name="Gemini",
        default_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    # Zhipu AI (registry.py:689)
    ProviderSpec(
        name="zhipu",
        keywords=("zhipu", "glm", "zai"),
        env_key="ZAI_API_KEY",
        display_name="Zhipu AI",
        default_api_base="https://open.bigmodel.cn/api/paas/v4",
    ),
    # DashScope / Qwen (registry.py:700)
    ProviderSpec(
        name="dashscope",
        keywords=("qwen", "dashscope"),
        env_key="DASHSCOPE_API_KEY",
        display_name="DashScope",
        default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        thinking_style="enable_thinking",
    ),
    # Moonshot / Kimi (registry.py:726)
    ProviderSpec(
        name="moonshot",
        keywords=("moonshot", "kimi"),
        env_key="MOONSHOT_API_KEY",
        display_name="Moonshot",
        default_api_base="https://api.moonshot.ai/v1",
        model_overrides=(
            ("kimi-k2.7", {"temperature": 1.0}),
            ("kimi-k2.7-code", {"temperature": 1.0}),
            ("kimi-k2.7-code-highspeed", {"temperature": 1.0}),
        ),
    ),
    # MiniMax (registry.py:753)
    ProviderSpec(
        name="minimax",
        keywords=("minimax",),
        env_key="MINIMAX_API_KEY",
        display_name="MiniMax",
        default_api_base="https://api.minimax.io/v1",
        thinking_style="reasoning_split",
        reasoning_effort_remap=(
            ("none", "low"),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
        ),
    ),
    # Mistral (registry.py:795)
    ProviderSpec(
        name="mistral",
        keywords=("mistral", "magistral", "ministral", "codestral", "devstral"),
        env_key="MISTRAL_API_KEY",
        display_name="Mistral",
        default_api_base="https://api.mistral.ai/v1",
        reasoning_effort_remap=(
            ("minimal", "none"),
            ("low", "none"),
            ("medium", "high"),
            ("high", "high"),
            ("none", "none"),
        ),
        implicit_reasoning_models=("magistral",),
        strip_history_reasoning_content=True,
    ),
    # === Local deployments ===================================================
    # vLLM / any OpenAI-compatible local server (registry.py:861)
    ProviderSpec(
        name="vllm",
        keywords=("vllm",),
        env_key="HOSTED_VLLM_API_KEY",
        display_name="vLLM",
        is_local=True,
    ),
    # Ollama (registry.py:871)
    ProviderSpec(
        name="ollama",
        keywords=("ollama", "nemotron"),
        env_key="OLLAMA_API_KEY",
        display_name="Ollama",
        is_local=True,
        detect_by_base_keyword="11434",
        default_api_base="http://localhost:11434/v1",
    ),
    # LM Studio (registry.py:883)
    ProviderSpec(
        name="lm_studio",
        keywords=("lm-studio", "lmstudio", "lm_studio"),
        env_key="LM_STUDIO_API_KEY",
        display_name="LM Studio",
        is_local=True,
        detect_by_base_keyword="1234",
        default_api_base="http://localhost:1234/v1",
    ),
    # NVIDIA NIM: keys start with "nvapi-" (registry.py:919)
    ProviderSpec(
        name="nvidia",
        keywords=("nvidia", "nemotron", "nvapi"),
        env_key="NVIDIA_NIM_API_KEY",
        display_name="NVIDIA NIM",
        detect_by_key_prefix="nvapi-",
        detect_by_base_keyword="nvidia.com",
        default_api_base="https://integrate.api.nvidia.com/v1",
    ),
    # === Auxiliary ===========================================================
    # Groq (registry.py:934)
    ProviderSpec(
        name="groq",
        keywords=("groq",),
        env_key="GROQ_API_KEY",
        display_name="Groq",
        default_api_base="https://api.groq.com/openai/v1",
    ),
)


def find_by_name(name: str) -> ProviderSpec | None:
    """The spec for a config name, e.g. "dashscope", "lmStudio" or "lm-studio" (registry.py:972)."""
    normalized = to_snake(name.replace("-", "_"))
    for spec in PROVIDERS:
        if spec.name == normalized:
            return spec
    return None


def create_dynamic_spec(
    name: str, *, display_name: str = "", thinking_style: str = ""
) -> ProviderSpec:
    """A spec for a custom provider named in ``providers`` (registry.py:981)."""
    normalized = to_snake(name.replace("-", "_"))
    strip_prefixes = tuple(dict.fromkeys((name, normalized)))
    return ProviderSpec(
        name=normalized,
        keywords=(),
        env_key="",
        display_name=display_name or name.replace("-", " ").replace("_", " ").title(),
        is_direct=True,
        strip_model_prefixes=strip_prefixes,
        thinking_style=thinking_style,
    )
