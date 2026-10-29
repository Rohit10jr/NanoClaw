"""Gaps found by mutation testing layer 20 (providers, presets, Anthropic, fallback).

Each test pins a behavior that a deliberate code change (a "mutant") broke
without any other test noticing.
"""

from __future__ import annotations

from nanoclaw.config.loader import apply_env_overrides
from nanoclaw.config.schema import Config, ModelPresetConfig, ProviderConfig
from nanoclaw.providers.factory import make_provider
from nanoclaw.providers.registry import ProviderSpec, create_dynamic_spec, find_by_name
from nanoclaw.providers.routing import apply_request_overrides, resolve_provider_route


def cfg(providers=None, presets=None, **defaults) -> Config:
    data: dict = {"agents": {"defaults": defaults}, "providers": providers or {}}
    if presets is not None:
        data["modelPresets"] = presets
    return Config.model_validate(data)


# --- registry ------------------------------------------------------------------------------------


def test_gateway_strip_keeps_only_the_last_model_segment():
    # nanobot registry.py:157: rsplit, so a doubly-prefixed model loses both prefixes.
    assert find_by_name("aihubmix").request_model_name("openrouter/anthropic/claude-x") == "claude-x"


def test_spec_label_prefers_the_display_name():
    assert find_by_name("lm_studio").label == "LM Studio"
    assert ProviderSpec(name="acme", keywords=(), env_key="").label == "Acme"
    assert create_dynamic_spec("company-proxy", display_name="ACME Gateway").label == "ACME Gateway"


# --- matching ------------------------------------------------------------------------------------


def test_forced_custom_provider_matches_ignoring_case_and_dashes():
    config = cfg({"company-proxy": {"apiBase": "https://c.test/v1"}}, model="m", provider="Company_Proxy")
    assert config.get_provider_name() == "company-proxy"


def test_dashed_keyword_matches_an_underscored_model_name():
    # "hugging-face" matches "hugging_face-..." only after normalizing dashes;
    # otherwise the gateway fallback (openrouter, first with a key) wins.
    config = cfg(
        {"openrouter": {"apiKey": "sk-or"}, "huggingface": {"apiKey": "hf"}},
        model="hugging_face-llama-3",
    )
    assert config.get_provider_name() == "huggingface"


def test_local_fallback_without_port_match_takes_the_first_local_provider():
    providers = {
        "vllm": {"apiBase": "http://h:9000/v1"},
        "lmStudio": {"apiBase": "http://h:8000/v1"},
    }
    assert cfg(providers, model="phi4").get_provider_name() == "vllm"


def test_custom_provider_without_api_base_is_not_a_last_resort():
    assert cfg({"companyProxy": {"apiKey": "k"}}, model="unmatched").get_provider_name() is None


def test_camel_case_registry_key_configures_the_provider():
    config = cfg({"lmStudio": {"apiBase": "http://h:8000/v1"}}, model="phi4")
    assert config.get_provider_name() == "lm_studio"
    assert config.get_api_base() == "http://h:8000/v1"


# --- routing -------------------------------------------------------------------------------------


def test_route_custom_spec_keeps_the_display_name():
    config = cfg(
        {"companyProxy": {"apiBase": "https://c.test/v1", "displayName": "ACME"}},
        model="companyProxy/gpt-4o",
    )
    route = resolve_provider_route(config, preset=config.resolve_preset())
    assert route.spec is not None and route.spec.display_name == "ACME"


def test_preset_header_override_replaces_a_lowercase_connection_header():
    connection = ProviderConfig(extra_headers={"x-a": "1"})
    preset = ModelPresetConfig(model="m", extra_headers={"X-A": "3"})
    assert apply_request_overrides(connection, preset).extra_headers == {"X-A": "3"}


# --- factory -------------------------------------------------------------------------------------


def test_anthropic_adapter_gets_the_configured_base_and_extras():
    config = cfg(
        {
            "anthropic": {
                "apiKey": "sk-ant-a",
                "apiBase": "https://proxy.test/v1",
                "extraBody": {"metadata": {"user_id": "u"}},
            }
        },
        model="claude-opus-4-5",
    )
    provider = make_provider(config)
    assert provider.api_base == "https://proxy.test/v1"
    assert provider._extra_body == {"metadata": {"user_id": "u"}}


def test_env_key_fills_a_forced_custom_provider_entry():
    forced = cfg({}, model="m", provider="myproxy")
    config = apply_env_overrides(forced, {"NANOCLAW_API_BASE": "https://p.test/v1"})
    assert config.providers["myproxy"].api_base == "https://p.test/v1"
    assert "custom" not in config.providers or not config.providers["custom"].api_base


def test_env_key_fills_the_existing_camel_case_entry():
    forced = cfg({"lmStudio": {"apiBase": "http://h:8000/v1"}}, model="phi4", provider="lm_studio")
    config = apply_env_overrides(forced, {"NANOCLAW_API_KEY": "k"})
    assert config.providers["lmStudio"].api_key == "k"
    assert "lm_studio" not in config.providers


# --- presets -------------------------------------------------------------------------------------


def test_model_preset_defaults():
    preset = ModelPresetConfig(model="m")
    assert (preset.provider, preset.max_tokens, preset.temperature, preset.reasoning_effort) == (
        "auto",
        8192,
        0.1,
        None,
    )


# --- OpenAI-compatible request rules -------------------------------------------------------------


def _oc(provider_name, model, reasoning_effort=None):
    from nanoclaw.providers import OpenAICompatProvider

    p = OpenAICompatProvider(
        api_key="k", default_model=model, spec=find_by_name(provider_name), reasoning_effort=reasoning_effort
    )
    return p.build_kwargs([{"role": "user", "content": "hi"}], None, model, 1024, 0.7)


def test_gateway_reasoning_field_only_for_models_with_a_thinking_switch():
    # OpenRouter's {"reasoning": {"effort"}} pairs with model-level thinking styles only.
    plain = _oc("openrouter", "openai/gpt-4o", reasoning_effort="high")
    assert "reasoning" not in plain.get("extra_body", {})
    kimi = _oc("openrouter", "moonshotai/kimi-k2.6", reasoning_effort="high")
    assert kimi["extra_body"]["reasoning"] == {"effort": "high"}


async def test_openai_compat_aclose_closes_the_client():
    from unittest.mock import AsyncMock, MagicMock

    from nanoclaw.providers import OpenAICompatProvider

    provider = OpenAICompatProvider(api_key="k")
    client = MagicMock()
    client.close = AsyncMock()
    provider._client = client
    await provider.aclose()
    client.close.assert_awaited_once()
    assert provider._client is None


# --- Anthropic message conversion ----------------------------------------------------------------


def _anthropic():
    from unittest.mock import MagicMock

    from nanoclaw.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(api_key="sk-test", client=MagicMock())


def _anthropic_messages(messages):
    return _anthropic()._build_kwargs(messages, None, None, 1024, 0.7, None, None, supports_caching=False)["messages"]


def test_anthropic_drops_empty_text_blocks():
    messages = _anthropic_messages(
        [{"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "hi"}]}]
    )
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_anthropic_wraps_a_single_dict_content_in_a_list():
    messages = _anthropic_messages([{"role": "user", "content": {"type": "text", "text": "hi"}}])
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_anthropic_assistant_without_content_gets_an_empty_text_block():
    _system, messages = _anthropic()._convert_messages(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "b"},
        ]
    )
    assert messages[1] == {"role": "assistant", "content": [{"type": "text", "text": ""}]}


def test_anthropic_tool_keeps_a_caller_cache_marker():
    from nanoclaw.providers.anthropic_provider import AnthropicProvider

    tools = [
        {
            "type": "function",
            "function": {"name": "a", "parameters": {"type": "object"}},
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert AnthropicProvider._convert_tools(tools)[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_cache_marker_goes_on_the_last_system_block():
    from nanoclaw.providers.anthropic_provider import AnthropicProvider

    system = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    marked, _messages, _tools = AnthropicProvider._apply_cache_control(system, [], None)
    assert marked == [{"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}]


def test_anthropic_two_messages_get_no_message_cache_marker():
    from nanoclaw.providers.anthropic_provider import AnthropicProvider

    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    _system, marked, _tools = AnthropicProvider._apply_cache_control("", messages, None)
    assert marked == messages


def _anthropic_kwargs(model, effort, max_tokens):
    from unittest.mock import MagicMock

    from nanoclaw.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-test", default_model=model, client=MagicMock())
    return provider._build_kwargs(
        [{"role": "user", "content": "hi"}], None, None, max_tokens, 0.7, effort, None, supports_caching=False
    )


def test_anthropic_high_budget_grows_with_max_tokens():
    kw = _anthropic_kwargs("claude-sonnet-4-6", "high", 16_000)
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 16_000}
    assert kw["max_tokens"] == 20_096


def test_anthropic_unknown_effort_on_a_budget_model_uses_the_medium_budget():
    kw = _anthropic_kwargs("claude-sonnet-4-6", "max", 1024)
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_anthropic_max_tokens_is_at_least_one():
    assert _anthropic_kwargs("claude-sonnet-4-6", None, 0)["max_tokens"] == 1


def test_anthropic_response_without_text_has_no_content():
    from types import SimpleNamespace

    from nanoclaw.providers.anthropic_provider import AnthropicProvider

    response = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="toolu_1", name="read", input={"p": 1})],
        stop_reason="tool_use",
        usage=None,
    )
    parsed = AnthropicProvider._parse_response(response)
    assert parsed.content is None and parsed.finish_reason == "tool_calls"


def test_anthropic_error_reads_the_status_from_the_response():
    from types import SimpleNamespace

    from nanoclaw.providers.anthropic_provider import AnthropicProvider

    exc = RuntimeError("upstream")
    exc.response = SimpleNamespace(status_code=503, headers={}, text="busy")  # type: ignore[attr-defined]
    response = AnthropicProvider._handle_error(exc)
    assert response.error_status_code == 503 and response.content == "Error: busy"
