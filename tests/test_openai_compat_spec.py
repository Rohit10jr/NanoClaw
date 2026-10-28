"""Spec-driven request rules of OpenAICompatProvider (layer 20).

Adapted from nanobot tests/providers/test_openai_compat_spec.py,
test_mistral_provider.py, test_prompt_cache_markers.py,
test_extra_body_config.py and test_custom_thinking_style.py, for the
Chat Completions path NanoClaw has. nanobot passes ``reasoning_effort`` per
call; NanoClaw's provider holds the preset's effort.
"""

import json

import httpx2
import pytest
from openai import AsyncOpenAI

import nanoclaw.providers.openai_compat as openai_compat
from nanoclaw.config.schema import Config
from nanoclaw.providers import OpenAICompatProvider
from nanoclaw.providers.factory import make_provider
from nanoclaw.providers.registry import create_dynamic_spec, find_by_name

HI = [{"role": "user", "content": "hi"}]


def _build_kwargs_for(
    provider_name, model, reasoning_effort=None, messages=None, **kwargs
):
    spec = find_by_name(provider_name)
    p = OpenAICompatProvider(
        api_key="k",
        default_model=model,
        spec=spec,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )
    return p.build_kwargs(messages or HI, None, model, 1024, 0.7)


# --- model names and token limits ----------------------------------------------------------------


def test_openrouter_keeps_model_name_intact():
    assert (
        _build_kwargs_for("openrouter", "anthropic/claude-sonnet-4-5")["model"]
        == "anthropic/claude-sonnet-4-5"
    )


def test_aihubmix_strips_model_prefix():
    assert (
        _build_kwargs_for("aihubmix", "anthropic/claude-sonnet-4-5")["model"]
        == "claude-sonnet-4-5"
    )


def test_standard_provider_passes_model_through():
    assert _build_kwargs_for("deepseek", "deepseek-chat")["model"] == "deepseek-chat"


def test_custom_provider_strips_its_own_prefix():
    p = OpenAICompatProvider(
        api_key="k",
        api_base="https://c.test/v1",
        spec=create_dynamic_spec("companyProxy"),
    )
    assert (
        p.build_kwargs(HI, model="companyProxy/gpt-4o-mini")["model"] == "gpt-4o-mini"
    )


def test_openai_spec_uses_max_completion_tokens_for_every_model():
    kw = _build_kwargs_for("openai", "gpt-4o", api_base="https://proxy.test/v1")
    assert kw["max_completion_tokens"] == 1024 and "max_tokens" not in kw


def test_other_specs_use_max_tokens_even_on_an_openai_looking_base():
    kw = _build_kwargs_for(
        "deepseek", "deepseek-chat", api_base="https://api.openai.com/v1"
    )
    assert kw["max_tokens"] == 1024 and "max_completion_tokens" not in kw


def test_reasoning_models_use_max_completion_tokens_on_any_spec():
    kw = _build_kwargs_for("openrouter", "openai/o3-mini")
    assert kw["max_completion_tokens"] == 1024 and "temperature" not in kw


# --- temperature ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, effort, supports_temperature",
    [
        ("gpt-6.1-sol", "high", False),
        ("gpt-6-sol", "high", False),
        ("gpt-6-sol", None, False),
        ("gpt-6-sol", "none", True),
    ],
)
def test_gpt6_temperature_requires_explicit_none_effort(
    model, effort, supports_temperature
):
    kw = _build_kwargs_for("openai", model, reasoning_effort=effort)
    assert ("temperature" in kw) is supports_temperature


@pytest.mark.parametrize(
    "provider_name, model",
    [("deepseek", "deepseek-v4-flash"), ("mistral", "mistral-medium-3-5")],
)
def test_compatible_provider_keeps_temperature_when_reasoning_effort_is_set(
    provider_name, model
):
    kw = _build_kwargs_for(provider_name, model, reasoning_effort="high")
    assert kw["temperature"] == 0.7
    assert kw["reasoning_effort"] == "high"


@pytest.mark.parametrize("model", ["kimi-k2.5", "kimi-k2.6"])
@pytest.mark.parametrize(
    "reasoning_effort", [None, "none", "minimal", "medium", "high"]
)
def test_moonshot_kimi_k25_k26_omit_temperature(model, reasoning_effort):
    assert "temperature" not in _build_kwargs_for(
        "moonshot", model, reasoning_effort=reasoning_effort
    )


def test_moonshot_kimi_k27_code_temperature_override():
    assert _build_kwargs_for("moonshot", "kimi-k2.7-code")["temperature"] == 1.0


# --- reasoning effort and thinking switches -------------------------------------------------------


@pytest.mark.parametrize(
    "provider, model, effort, thinking_type",
    [
        ("zhipu", "glm-4.6", "none", "disabled"),
        ("zhipu", "glm-5.2", "minimal", "disabled"),
        ("zhipu", "glm-4.7", "high", "enabled"),
        ("zhipu", "glm-4.7", None, None),
        ("zhipu", "glm-5.3", "none", None),
        ("custom", "glm-4.7", "none", None),
    ],
)
def test_zhipu_glm_thinking_controls(provider, model, effort, thinking_type):
    kw = _build_kwargs_for(provider, model, reasoning_effort=effort)
    assert kw.get("extra_body", {}).get("thinking", {}).get("type") == thinking_type
    if effort == "none":
        assert "reasoning_effort" not in kw


def test_dashscope_thinking_enabled_with_reasoning_effort():
    assert _build_kwargs_for("dashscope", "qwen3-plus", reasoning_effort="medium")[
        "extra_body"
    ] == {"enable_thinking": True}


@pytest.mark.parametrize("effort", ["minimal", "minimum"])
def test_dashscope_thinking_disabled_for_minimal(effort):
    kw = _build_kwargs_for("dashscope", "qwen3-plus", reasoning_effort=effort)
    assert kw["reasoning_effort"] == "minimum"
    assert kw["extra_body"] == {"enable_thinking": False}


def test_non_dashscope_minimal_not_retranslated():
    assert (
        _build_kwargs_for("openai", "gpt-5", reasoning_effort="minimal")[
            "reasoning_effort"
        ]
        == "minimal"
    )


@pytest.mark.parametrize(
    "provider_name, model",
    [
        ("dashscope", "qwen-turbo"),
        ("minimax", "MiniMax-M2.7"),
        ("deepseek", "deepseek-chat"),
        ("moonshot", "kimi-k2.5"),
        ("openrouter", "moonshotai/kimi-k2.5"),
        ("openrouter", "qwen/qwen3.6-flash"),
    ],
)
def test_provider_omits_extra_body_without_reasoning_effort(provider_name, model):
    assert "extra_body" not in _build_kwargs_for(
        provider_name, model, reasoning_effort=None
    )


def test_minimax_reasoning_split_enabled_with_reasoning_effort():
    assert _build_kwargs_for("minimax", "MiniMax-M2.7", reasoning_effort="medium")[
        "extra_body"
    ] == {"reasoning_split": True}


@pytest.mark.parametrize("effort", ["none", "minimal"])
def test_minimax_off_presets_map_to_low_effort(effort):
    kw = _build_kwargs_for("minimax", "MiniMax-M2.7", reasoning_effort=effort)
    assert kw["reasoning_effort"] == "low"
    assert kw["extra_body"] == {"reasoning_split": True}


def test_deepseek_thinking_enabled():
    assert _build_kwargs_for("deepseek", "deepseek-v4-pro", reasoning_effort="high")[
        "extra_body"
    ] == {"thinking": {"type": "enabled"}}


@pytest.mark.parametrize(
    "effort, wire_effort, thinking_type",
    [
        ("minimal", "low", "enabled"),
        ("low", "low", "enabled"),
        ("none", None, "disabled"),
        (None, None, None),
    ],
)
def test_deepseek_reasoning_effort_controls(effort, wire_effort, thinking_type):
    kw = _build_kwargs_for("deepseek", "deepseek-flash", reasoning_effort=effort)
    assert kw.get("reasoning_effort") == wire_effort
    assert kw.get("extra_body") == (
        {"thinking": {"type": thinking_type}} if thinking_type else None
    )


TOOL_HISTORY = [
    {"role": "user", "content": "hi"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call1", "content": "ok"},
    {"role": "user", "content": "next"},
]


def test_deepseek_thinking_backfills_missing_reasoning_content_on_tool_history():
    kw = _build_kwargs_for(
        "deepseek", "deepseek-chat", reasoning_effort="high", messages=TOOL_HISTORY
    )
    assistant = next(m for m in kw["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == ""


def test_backfill_does_not_touch_messages_when_thinking_explicitly_off():
    kw = _build_kwargs_for(
        "deepseek", "deepseek-chat", reasoning_effort="none", messages=TOOL_HISTORY
    )
    assert all("reasoning_content" not in m for m in kw["messages"])


def test_deepseek_reasoner_backfills_when_effort_implicit():
    kw = _build_kwargs_for("deepseek", "deepseek-reasoner", messages=TOOL_HISTORY)
    assert (
        next(m for m in kw["messages"] if m["role"] == "assistant")["reasoning_content"]
        == ""
    )


def test_deepseek_chat_keeps_tool_history_when_effort_implicit():
    kw = _build_kwargs_for("deepseek", "deepseek-chat", messages=TOOL_HISTORY)
    assert all("reasoning_content" not in m for m in kw["messages"])


@pytest.mark.parametrize(
    "model, effort",
    [("mistral-medium-3-5", "low"), ("mistral-vibe-cli-latest", "minimal")],
)
def test_mistral_omits_low_reasoning_effort(model, effort):
    assert "reasoning_effort" not in _build_kwargs_for(
        "mistral", model, reasoning_effort=effort
    )


@pytest.mark.parametrize("effort", ["medium", "high"])
def test_mistral_uses_high_reasoning_effort(effort):
    assert (
        _build_kwargs_for("mistral", "mistral-medium-3-5", reasoning_effort=effort)[
            "reasoning_effort"
        ]
        == "high"
    )


def test_magistral_strips_reasoning_effort():
    for effort in ("low", "medium", "high", "minimal"):
        assert "reasoning_effort" not in _build_kwargs_for(
            "mistral", "magistral-medium-latest", reasoning_effort=effort
        )


def test_mistral_strips_reasoning_content_from_history():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Hello!",
            "reasoning_content": "internal thoughts",
        },
        {"role": "user", "content": "follow up"},
    ]
    kw = _build_kwargs_for("mistral", "mistral-large-latest", messages=messages)
    assert all("reasoning_content" not in msg for msg in kw["messages"])
    assert kw["messages"][1]["content"] == "Hello!"
    # Other providers keep it (DeepSeek needs it on the wire).
    assert _build_kwargs_for("deepseek", "deepseek-chat", messages=messages)[
        "messages"
    ][1]["reasoning_content"]


@pytest.mark.parametrize(
    "provider_name, model, effort, thinking_type",
    [
        ("moonshot", "kimi-k2.5", "high", "enabled"),
        ("moonshot", "k2.6-code-preview", "high", "enabled"),
        ("deepseek", "deepseek-v4-pro", "none", "disabled"),
        ("moonshot", "kimi-k2.5", "none", "disabled"),
    ],
)
def test_provider_thinking_type_mapping(provider_name, model, effort, thinking_type):
    kw = _build_kwargs_for(provider_name, model, reasoning_effort=effort)
    assert kw.get("extra_body") == {"thinking": {"type": thinking_type}}
    assert "reasoning_effort" not in kw


@pytest.mark.parametrize(
    "model", ["kimi-k3", "kimi-k3.1-pro", "kimi-k3.1-pro-highspeed"]
)
def test_kimi_k3_rules(model):
    kw = _build_kwargs_for("moonshot", model)
    assert (
        "temperature" not in kw
        and "max_tokens" not in kw
        and kw["max_completion_tokens"] == 1024
    )
    assert "reasoning_effort" not in kw and "extra_body" not in kw
    assert (
        _build_kwargs_for("moonshot", model, reasoning_effort="high")[
            "reasoning_effort"
        ]
        == "high"
    )
    assert (
        _build_kwargs_for("moonshot", model, reasoning_effort="medium")[
            "reasoning_effort"
        ]
        == "max"
    )
    assert "reasoning_effort" not in _build_kwargs_for(
        "moonshot", model, reasoning_effort="none"
    )


@pytest.mark.parametrize(
    "model, effort",
    [("moonshotai/kimi-k2.5", "medium"), ("moonshotai/kimi-k2.7-code", "high")],
)
def test_openrouter_kimi_thinking_mapping(model, effort):
    kw = _build_kwargs_for("openrouter", model, reasoning_effort=effort)
    assert kw.get("extra_body") == {
        "thinking": {"type": "enabled"},
        "reasoning": {"effort": effort},
    }
    assert "reasoning_effort" not in kw


def test_kimi_k27_code_thinking_none_omits_disabled():
    for provider, model in (
        ("moonshot", "kimi-k2.7-code"),
        ("openrouter", "moonshotai/kimi-k2.7-code"),
    ):
        kw = _build_kwargs_for(provider, model, reasoning_effort="none")
        assert "extra_body" not in kw and "reasoning_effort" not in kw


def test_kimi_k2_series_no_thinking_injection():
    assert "extra_body" not in _build_kwargs_for(
        "moonshot", "kimi-k2", reasoning_effort="high"
    )
    assert "extra_body" not in _build_kwargs_for(
        "moonshot", "kimi-k2-thinking", reasoning_effort="high"
    )


def test_qwen_thinking_via_model_level_mapping():
    # OpenRouter also sends its own reasoning.effort next to the model's switch.
    assert _build_kwargs_for(
        "openrouter", "qwen/qwen3.6-flash", reasoning_effort="medium"
    )["extra_body"] == {
        "enable_thinking": True,
        "reasoning": {"effort": "medium"},
    }
    assert _build_kwargs_for(
        "openrouter", "qwen/qwen3.5-flash", reasoning_effort="none"
    )["extra_body"] == {
        "enable_thinking": False,
        "reasoning": {"effort": "none"},
    }
    kw = _build_kwargs_for("dashscope", "qwen3.6-plus", reasoning_effort="none")
    assert (
        kw.get("extra_body") == {"enable_thinking": False}
        and "reasoning_effort" not in kw
    )


def test_openai_no_thinking_extra_body():
    kw = _build_kwargs_for("openai", "gpt-5", reasoning_effort="high")
    assert "extra_body" not in kw and kw["reasoning_effort"] == "high"


def test_custom_provider_thinking_style():
    """nanobot test_custom_thinking_style: a custom provider's thinkingStyle."""
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "model": "r1",
                    "provider": "myds",
                    "reasoningEffort": "high",
                }
            },
            "providers": {
                "myds": {
                    "apiBase": "https://ds.test/v1",
                    "thinkingStyle": "thinking_type",
                }
            },
        }
    )
    kw = make_provider(config).build_kwargs(HI)
    assert kw["extra_body"] == {"thinking": {"type": "enabled"}}
    with pytest.raises(ValueError, match="Invalid thinking_style"):
        Config.model_validate(
            {"providers": {"myds": {"apiBase": "x", "thinkingStyle": "nope"}}}
        )


# --- extra_body, prompt caching -------------------------------------------------------------------


def test_extra_body_is_merged_last_and_deep():
    p = OpenAICompatProvider(
        api_key="k",
        spec=find_by_name("deepseek"),
        reasoning_effort="high",
        extra_body={"thinking": {"budget": 10}, "top_k": 5},
    )
    kw = p.build_kwargs(HI, model="deepseek-chat")
    assert kw["extra_body"] == {
        "thinking": {"type": "enabled", "budget": 10},
        "top_k": 5,
    }


def test_extra_body_tools_are_appended_not_replaced():
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    p = OpenAICompatProvider(
        api_key="k", extra_body={"tools": [{"type": "web_search"}]}
    )
    kw = p.build_kwargs(HI, tools=tools, model="m")
    assert kw["tools"] == [*tools, {"type": "web_search"}]
    assert "extra_body" not in kw


def _tools(*names):
    return [
        {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
        for n in names
    ]


def _marked(tools):
    return [t["function"]["name"] for t in tools or [] if "cache_control" in t]


def test_openai_compat_marks_builtin_boundary_and_tail_tool():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u"},
    ]
    _, marked = OpenAICompatProvider._apply_cache_control(
        messages, _tools("read_file", "write_file", "mcp_fs_ls", "mcp_git_status")
    )
    assert _marked(marked) == ["write_file", "mcp_git_status"]


def test_openai_compat_marks_only_tail_without_mcp():
    _, marked = OpenAICompatProvider._apply_cache_control(
        [{"role": "user", "content": "u"}], _tools("read_file", "write_file")
    )
    assert _marked(marked) == ["write_file"]


def test_openrouter_claude_gets_cache_markers_others_do_not():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "x"},
    ]
    kw = _build_kwargs_for(
        "openrouter", "anthropic/claude-sonnet-4-5", messages=messages
    )
    assert kw["messages"][0]["content"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]
    assert kw["messages"][2]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["messages"][3]["content"] == "x"
    assert (
        _build_kwargs_for("openrouter", "openai/gpt-4o", messages=messages)["messages"][
            0
        ]["content"]
        == "sys"
    )
    assert (
        _build_kwargs_for("deepseek", "claude-ish", messages=messages)["messages"][0][
            "content"
        ]
        == "sys"
    )


# --- on the wire: extra headers and query ---------------------------------------------------------


async def test_extra_headers_and_query_reach_the_server(monkeypatch):
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        body = {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx2.Response(200, json=body)

    real = AsyncOpenAI

    def client_factory(**kwargs):
        return real(
            **kwargs,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )

    monkeypatch.setattr(openai_compat, "AsyncOpenAI", client_factory)
    provider = OpenAICompatProvider(
        api_key="k",
        api_base="https://llm.test/v1",
        extra_headers={"APP-Code": "abc"},
        extra_query={"api-version": "2024-01"},
        extra_body={"top_k": 3},
    )
    response = await provider.chat(HI, model="m")
    assert response.content == "ok"
    request = seen[0]
    assert request.headers["APP-Code"] == "abc"
    assert request.url.params["api-version"] == "2024-01"
    assert json.loads(request.content)["top_k"] == 3
    await provider.aclose()
    assert provider._client is None
