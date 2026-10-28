"""Provider registry, routing and the provider factory (layer 20).

Adapted from nanobot tests/config/test_model_presets.py (routing and preset
resolution), tests/providers/test_openai_compat_spec.py (spec checks),
test_mistral_provider.py, test_local_endpoint_detection.py and
test_provider_default_headers.py, plus NanoClaw's deliberate differences.
"""

import sys

import pytest

from nanoclaw.config.schema import Config, ProviderConfig
from nanoclaw.providers import OpenAICompatProvider
from nanoclaw.providers.factory import (
    _provider_extra_headers,
    build_preset_runtime,
    make_provider,
)
from nanoclaw.providers.registry import PROVIDERS, create_dynamic_spec, find_by_name
from nanoclaw.providers.routing import (
    apply_request_overrides,
    resolve_provider_route,
    validate_provider_setup,
)


def cfg(providers=None, presets=None, **defaults) -> Config:
    data: dict = {"agents": {"defaults": defaults}, "providers": providers or {}}
    if presets is not None:
        data["modelPresets"] = presets
    return Config.model_validate(data)


# --- the registry --------------------------------------------------------------------------------


def test_registry_order_and_names_follow_nanobot():
    names = [spec.name for spec in PROVIDERS]
    assert names[0] == "custom"
    # Gateways come before the standard providers, local ones near the end.
    assert (
        names.index("openrouter")
        < names.index("anthropic")
        < names.index("openai")
        < names.index("ollama")
    )
    assert len(names) == len(set(names))


def test_find_by_name_accepts_config_spellings():
    assert find_by_name("lmStudio").name == "lm_studio"
    assert find_by_name("lm-studio").name == "lm_studio"
    assert find_by_name("lm_studio").name == "lm_studio"
    assert find_by_name("volcengine") is None  # not in NanoClaw's registry


def test_openrouter_spec_is_gateway():
    spec = find_by_name("openrouter")
    assert spec.is_gateway and spec.default_api_base == "https://openrouter.ai/api/v1"
    assert spec.detect_by_key_prefix == "sk-or-"


def test_gemini_spec_keeps_openai_compat_base():
    spec = find_by_name("gemini")
    assert spec.backend == "openai_compat"
    assert (
        spec.default_api_base
        == "https://generativelanguage.googleapis.com/v1beta/openai/"
    )


def test_anthropic_spec_uses_the_anthropic_backend():
    spec = find_by_name("anthropic")
    assert (spec.backend, spec.supports_prompt_caching) == ("anthropic", True)


def test_mistral_provider_in_registry():
    spec = find_by_name("mistral")
    assert spec.default_api_base == "https://api.mistral.ai/v1"
    assert spec.implicit_reasoning_models == ("magistral",)
    assert spec.strip_history_reasoning_content


@pytest.mark.parametrize(
    "spec_name, model, expected",
    [
        ("aihubmix", "anthropic/claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("openrouter", "anthropic/claude-sonnet-4-5", "anthropic/claude-sonnet-4-5"),
        ("deepseek", "deepseek-chat", "deepseek-chat"),
        ("deepseek", "deepseek/deepseek-chat", "deepseek/deepseek-chat"),
    ],
)
def test_request_model_name(spec_name, model, expected):
    assert find_by_name(spec_name).request_model_name(model) == expected


def test_dynamic_spec_strips_its_own_prefix():
    spec = create_dynamic_spec("company-proxy", thinking_style="thinking_type")
    assert spec.name == "company_proxy" and spec.is_direct
    assert spec.display_name == "Company Proxy"
    assert spec.request_model_name("company-proxy/gpt-4o") == "gpt-4o"
    assert spec.request_model_name("company_proxy/gpt-4o") == "gpt-4o"
    assert spec.request_model_name("other/gpt-4o") == "other/gpt-4o"
    assert spec.thinking_style == "thinking_type"


# --- matching (nanobot Config._match_provider) ---------------------------------------------------


def test_custom_provider_fallback_uses_extra_keys():
    config = cfg(
        {"my-company-api": {"apiBase": "https://example.test/v1"}},
        model="unmatched-model",
    )
    assert config.get_provider_name() == "my-company-api"


def test_dynamic_custom_provider_prefix_matches_camel_case_key():
    config = cfg(
        {
            "otherProxy": {"apiBase": "https://other.example.test/v1"},
            "companyProxy": {"apiBase": "https://company.example.test/v1"},
        },
        model="companyProxy/gpt-4o-mini",
    )
    assert config.get_provider_name() == "companyProxy"
    assert config.get_api_base() == "https://company.example.test/v1"


def test_dynamic_custom_provider_prefix_does_not_fall_through_when_base_missing():
    config = cfg(
        {
            "otherProxy": {"apiBase": "https://other.example.test/v1"},
            "companyProxy": {"apiKey": "sk-company"},
        },
        model="companyProxy/gpt-4o-mini",
    )
    assert config.get_provider_name() == "companyProxy"
    assert config.get_api_base() is None
    with pytest.raises(
        ValueError, match="Provider 'companyProxy' requires api_base in config."
    ):
        make_provider(config)


@pytest.mark.parametrize("provider_name", ["lm-studio", "LM_STUDIO"])
def test_custom_provider_rejects_builtin_provider_aliases(provider_name):
    with pytest.raises(ValueError, match="conflicts with built-in provider"):
        cfg({provider_name: {"apiBase": "https://example.test/v1"}})


@pytest.mark.parametrize(
    "provider_name, model",
    [("openai", "openai/gpt-4.1"), ("anthropic", "anthropic/claude-opus-4-5")],
)
def test_match_provider_uses_model_preset(provider_name, model):
    config = cfg(
        {provider_name: {"apiKey": "sk-test"}},
        presets={"fast": {"model": model, "provider": provider_name}},
        modelPreset="fast",
    )
    assert config.get_provider_name() == provider_name


def test_keyword_match_follows_registry_order():
    config = cfg(
        {"openrouter": {"apiKey": "sk-or"}, "deepseek": {"apiKey": "ds"}},
        model="deepseek-chat",
    )
    assert (
        config.get_provider_name() == "deepseek"
    )  # keyword beats the gateway fallback
    config = cfg(
        {"deepseek": {"apiKey": "ds"}, "openrouter": {"apiKey": "sk-or"}},
        model="gpt-4o",
    )
    assert config.get_provider_name() == "openrouter"  # no match: gateways first


def test_explicit_prefix_needs_a_key_unless_local():
    config = cfg({"openrouter": {"apiKey": "sk-or"}}, model="deepseek/deepseek-chat")
    assert config.get_provider_name() == "openrouter"
    config = cfg({}, model="ollama/llama3.2")
    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434/v1"


def test_local_keyword_requires_configured_base_and_no_foreign_prefix():
    # nanobot test_local_endpoint_detection: a cloud-hosted nemotron is not Ollama's.
    providers = {
        "ollama": {"apiBase": "http://localhost:11434/v1"},
        "openrouter": {"apiKey": "sk-or"},
    }
    assert (
        cfg(providers, model="nvidia/nemotron-nano").get_provider_name() == "openrouter"
    )
    assert cfg(providers, model="nemotron-mini").get_provider_name() == "ollama"
    assert (
        cfg(
            {"openrouter": {"apiKey": "sk-or"}}, model="nemotron-mini"
        ).get_provider_name()
        == "openrouter"
    )


def test_local_fallback_prefers_base_matching_the_port_keyword():
    providers = {
        "vllm": {"apiBase": "http://h:9000/v1"},
        "ollama": {"apiBase": "http://h:11434/v1"},
    }
    assert cfg(providers, model="phi4").get_provider_name() == "ollama"
    assert (
        cfg({"vllm": {"apiBase": "http://h:9000/v1"}}, model="phi4").get_provider_name()
        == "vllm"
    )


def test_forced_provider_wins_over_model_keywords():
    config = cfg(
        {"openrouter": {"apiKey": "sk-or"}, "anthropic": {"apiKey": "a"}},
        model="claude-x",
        provider="openrouter",
    )
    assert config.get_provider_name() == "openrouter"
    # Forced and unconfigured: chosen anyway, then refused by the route (nanobot).
    config = cfg({}, model="deepseek-chat", provider="deepseek")
    assert config.get_provider_name() == "deepseek"
    with pytest.raises(
        ValueError, match="No API key configured for provider 'deepseek'."
    ):
        make_provider(config)


def test_forced_unknown_provider_is_refused():
    config = cfg({"openrouter": {"apiKey": "sk-or"}}, model="x", provider="nonexistent")
    assert config.get_provider_name() is None
    with pytest.raises(ValueError, match="No provider is configured for model 'x'."):
        make_provider(config)


# --- routes and the factory ----------------------------------------------------------------------


def test_route_carries_backend_base_and_model():
    config = cfg({"deepseek": {"apiKey": "ds"}}, model="deepseek-chat")
    route = resolve_provider_route(config, preset=config.resolve_preset())
    assert (route.provider_name, route.backend, route.api_base, route.model) == (
        "deepseek",
        "openai_compat",
        "https://api.deepseek.com",
        "deepseek-chat",
    )
    validate_provider_setup(config)


def test_route_model_override_drops_preset_overrides_for_another_connection():
    config = cfg(
        {"deepseek": {"apiKey": "ds"}, "openrouter": {"apiKey": "sk-or"}},
        presets={"fast": {"model": "deepseek-chat", "extraBody": {"seed": 1}}},
    )
    preset = config.resolve_preset("fast")
    same = resolve_provider_route(config, preset=preset, model="deepseek-reasoner")
    assert same.provider_config.extra_body == {"seed": 1}
    other = resolve_provider_route(config, preset=preset, model="gpt-4o")
    assert (
        other.provider_name == "openrouter" and other.provider_config.extra_body is None
    )


def test_apply_request_overrides_merges_headers_case_insensitively():
    connection = ProviderConfig(
        extra_headers={"X-A": "1", "X-B": "2"}, extra_body={"a": {"x": 1}}
    )
    preset = cfg(
        presets={
            "p": {
                "model": "m",
                "extraHeaders": {"x-a": "3"},
                "extraBody": {"a": {"y": 2}},
            }
        }
    ).resolve_preset("p")
    merged = apply_request_overrides(connection, preset)
    assert merged.extra_headers == {"X-B": "2", "x-a": "3"}
    assert merged.extra_body == {"a": {"y": 2}}  # nested values replace as a unit
    assert connection.extra_headers == {"X-A": "1", "X-B": "2"}  # a copy


def test_make_provider_builds_openai_compat_with_spec_and_extras():
    config = cfg(
        {
            "deepseek": {
                "apiKey": "ds",
                "extraHeaders": {"APP-Code": "x"},
                "extraBody": {"top_k": 5},
                "extraQuery": {"v": "1"},
            }
        },
        model="deepseek-chat",
        reasoningEffort="high",
    )
    provider = make_provider(config)
    assert isinstance(provider, OpenAICompatProvider)
    assert (provider.api_key, provider.api_base, provider.default_model) == (
        "ds",
        "https://api.deepseek.com",
        "deepseek-chat",
    )
    assert provider._spec is find_by_name("deepseek")
    assert provider.extra_headers == {"APP-Code": "x"}
    assert (provider._extra_body, provider._extra_query) == ({"top_k": 5}, {"v": "1"})
    assert (provider.reasoning_effort, provider.provider_name) == ("high", "deepseek")


def test_provider_config_extra_headers_override_spec_defaults():
    from dataclasses import replace

    spec = replace(
        find_by_name("moonshot"),
        default_extra_headers=(("User-Agent", "claude-code/0.1.0"),),
    )
    assert _provider_extra_headers(spec, ProviderConfig()) == {
        "User-Agent": "claude-code/0.1.0"
    }
    provider = ProviderConfig.model_validate(
        {"extraHeaders": {"user-agent": "custom/1.0", "X-Test": "1"}}
    )
    assert _provider_extra_headers(spec, provider) == {
        "user-agent": "custom/1.0",
        "X-Test": "1",
    }


def test_build_preset_runtime_uses_the_preset():
    config = cfg(
        {"deepseek": {"apiKey": "ds"}, "openrouter": {"apiKey": "sk-or"}},
        presets={
            "deep": {
                "model": "anthropic/claude-opus-4-5",
                "provider": "openrouter",
                "maxTokens": 100,
                "temperature": 0.5,
                "contextWindowTokens": 1000,
                "reasoningEffort": "low",
            }
        },
        model="deepseek-chat",
    )
    runtime = build_preset_runtime(config, preset_name="deep")
    assert (
        runtime.model,
        runtime.max_tokens,
        runtime.temperature,
        runtime.context_window_tokens,
    ) == (
        "anthropic/claude-opus-4-5",
        100,
        0.5,
        1000,
    )
    assert (runtime.reasoning_effort, runtime.model_preset) == ("low", "deep")
    assert runtime.provider.api_base == "https://openrouter.ai/api/v1"
    default = build_preset_runtime(config)
    assert (default.model, default.model_preset, default.provider.api_base) == (
        "deepseek-chat",
        None,
        "https://api.deepseek.com",
    )


# --- NanoClaw's deliberate differences ------------------------------------------------------------


def test_custom_without_base_means_api_openai_com():
    """nanobot requires apiBase for ``custom``; NanoClaw's phase 1 setup
    (``NANOCLAW_API_KEY`` alone) talks to api.openai.com."""
    provider = make_provider(cfg({"custom": {"apiKey": "k"}}, model="gpt-4o"))
    assert (provider.api_key, provider.api_base) == ("k", None)
    assert (
        provider.build_kwargs([{"role": "user", "content": "hi"}], max_tokens=5)[
            "max_completion_tokens"
        ]
        == 5
    )


def test_nothing_configured_gives_an_unconfigured_openai_client():
    """nanobot refuses ("No provider is configured"); NanoClaw keeps phase 1's client."""
    route = resolve_provider_route(Config(), preset=Config().resolve_preset())
    assert (route.provider_name, route.backend, route.api_base) == (
        None,
        "openai_compat",
        None,
    )
    provider = make_provider(Config())
    assert provider.api_key is None and provider.default_model == "gpt-4o-mini"


# --- the anthropic backend hook -------------------------------------------------------------------


def test_anthropic_backend_without_adapter_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "nanoclaw.providers.anthropic_provider", None)
    config = cfg({"anthropic": {"apiKey": "a"}}, model="claude-opus-4-5")
    assert (
        resolve_provider_route(config, preset=config.resolve_preset()).backend
        == "anthropic"
    )
    with pytest.raises(ValueError, match="uses the Anthropic Messages API"):
        make_provider(config)


def test_anthropic_backend_builds_the_adapter_lazily(monkeypatch):
    import types

    seen = {}

    class AnthropicProvider:
        def __init__(
            self,
            api_key=None,
            api_base=None,
            default_model="",
            extra_headers=None,
            extra_body=None,
        ):
            seen.update(
                api_key=api_key,
                api_base=api_base,
                default_model=default_model,
                extra_headers=extra_headers,
            )

    module = types.ModuleType("nanoclaw.providers.anthropic_provider")
    module.AnthropicProvider = AnthropicProvider
    monkeypatch.setitem(sys.modules, "nanoclaw.providers.anthropic_provider", module)
    config = cfg(
        {"anthropic": {"apiKey": "a", "extraHeaders": {"X": "1"}}},
        model="anthropic/claude-opus-4-5",
    )
    assert isinstance(make_provider(config), AnthropicProvider)
    assert seen == {
        "api_key": "a",
        "api_base": None,
        "default_model": "anthropic/claude-opus-4-5",
        "extra_headers": {"X": "1"},
    }
    # A setting the adapter cannot take is refused rather than dropped.
    effort = cfg(
        {"anthropic": {"apiKey": "a"}}, model="claude-opus-4-5", reasoningEffort="high"
    )
    with pytest.raises(ValueError, match="does not support reasoning_effort"):
        make_provider(effort)
