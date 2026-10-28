"""Model presets, the runtime resolver, per-session /model and --preset (layer 20).

Adapted from nanobot tests/config/test_model_presets.py,
tests/agent/test_model_runtime_resolver.py,
tests/agent/test_session_model_runtime.py and
tests/command/test_model_command.py.
"""

import asyncio
import json

from pathlib import Path

import pytest

from nanoclaw.__main__ import _make_loop, main
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.model_presets import configured_model_presets, normalize_preset_name
from nanoclaw.agent.model_runtime import ModelRuntimeResolver
from nanoclaw.bus.events import InboundMessage
from nanoclaw.command import CommandContext, CommandRouter, register_builtin_commands
from nanoclaw.command.builtin import build_help_text, builtin_command_palette, cmd_model
from nanoclaw.config.schema import Config, ModelPresetConfig
from nanoclaw.providers import FakeProvider, LLMProvider, LLMResponse
from nanoclaw.session.model_selection import (
    SESSION_MODEL_PRESET_METADATA_KEY,
    model_preset_from_metadata,
)
from nanoclaw.utils.llm_runtime import LLMRuntime

# --- config (nanobot tests/config/test_model_presets.py) -----------------------------------------


def test_resolve_preset_returns_defaults_when_no_preset():
    config = Config()
    resolved = config.resolve_preset()
    d = config.agents.defaults
    assert (resolved.model, resolved.provider, resolved.max_tokens) == (
        d.model,
        d.provider,
        d.max_tokens,
    )
    assert (
        resolved.context_window_tokens,
        resolved.temperature,
        resolved.reasoning_effort,
    ) == (
        d.context_window_tokens,
        d.temperature,
        d.reasoning_effort,
    )


def test_legacy_defaults_config_without_presets_still_resolves():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "model": "openai/gpt-4.1",
                    "provider": "openai",
                    "maxTokens": 4096,
                    "contextWindowTokens": 128_000,
                    "temperature": 0.2,
                    "reasoningEffort": "low",
                }
            }
        }
    )
    resolved = config.resolve_preset()
    assert config.agents.defaults.model_preset is None and config.model_presets == {}
    assert (
        resolved.model,
        resolved.provider,
        resolved.max_tokens,
        resolved.temperature,
        resolved.reasoning_effort,
    ) == (
        "openai/gpt-4.1",
        "openai",
        4096,
        0.2,
        "low",
    )


def test_resolve_preset_returns_active_preset():
    config = Config.model_validate(
        {
            "model_presets": {
                "fast": {
                    "model": "openai/gpt-4.1",
                    "provider": "openai",
                    "maxTokens": 4096,
                    "contextWindowTokens": 32_768,
                    "temperature": 0.5,
                    "reasoningEffort": "low",
                }
            },
            "agents": {"defaults": {"modelPreset": "fast"}},
        }
    )
    resolved = config.resolve_preset()
    assert (
        resolved.model,
        resolved.max_tokens,
        resolved.context_window_tokens,
        resolved.temperature,
    ) == (
        "openai/gpt-4.1",
        4096,
        32_768,
        0.5,
    )


def test_default_preset_is_agents_defaults_even_when_named_preset_is_active():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "model": "openai/gpt-4.1",
                    "provider": "openai",
                    "modelPreset": "fast",
                }
            },
            "modelPresets": {
                "fast": {"model": "openai/gpt-4.1-mini", "provider": "openai"}
            },
        }
    )
    assert config.resolve_preset().model == "openai/gpt-4.1-mini"
    assert config.resolve_preset("default").model == "openai/gpt-4.1"
    assert (
        config.resolve_preset("default").context_window_tokens == 128_000
    )  # NanoClaw's default window


def test_model_presets_serialize_with_camel_case_root_key():
    config = Config.model_validate(
        {"model_presets": {"fast": {"model": "openai/gpt-4.1", "provider": "openai"}}}
    )
    dumped = config.model_dump(mode="json", by_alias=True)
    assert "modelPresets" in dumped and "model_presets" not in dumped
    assert dumped["modelPresets"]["fast"]["model"] == "openai/gpt-4.1"
    assert dumped["modelPresets"]["fast"]["extraHeaders"] is None


@pytest.mark.parametrize(
    "model_presets",
    [
        {"Default": {"model": "a"}},
        {"Fast": {"model": "a"}, "fast": {"model": "b"}},
        {" fast ": {"model": "a"}},
    ],
)
def test_preset_names_accepted_by_earlier_releases_remain_loadable(model_presets):
    assert list(
        Config.model_validate({"modelPresets": model_presets}).model_presets
    ) == list(model_presets)


def test_validator_rejects_unknown_and_reserved_presets():
    with pytest.raises(
        ValueError, match="model_preset 'unknown' not found in model_presets"
    ):
        Config.model_validate({"agents": {"defaults": {"modelPreset": "unknown"}}})
    with pytest.raises(ValueError, match="model_preset name 'default' is reserved"):
        Config.model_validate({"modelPresets": {"default": {"model": "custom-model"}}})
    with pytest.raises(ValueError, match="Dream model preset 'unknown' not found"):
        Config.model_validate(
            {"agents": {"defaults": {"dream": {"modelOverride": "unknown"}}}}
        )
    with pytest.raises(KeyError, match="model_preset 'missing' not found"):
        Config().resolve_preset("missing")


def test_dream_model_preset_accepts_legacy_aliases():
    for key in ("modelOverride", "model", "model_override"):
        config = Config.model_validate(
            {
                "modelPresets": {"dream": {"model": "m"}},
                "agents": {"defaults": {"dream": {key: "dream"}}},
            }
        )
        assert config.agents.defaults.dream.model_override == "dream"
    explicit_default = Config.model_validate(
        {"agents": {"defaults": {"model": "x", "modelPreset": "default"}}}
    )
    assert explicit_default.resolve_preset().model == "x"


def test_configured_model_presets_adds_default():
    config = Config.model_validate(
        {
            "modelPresets": {"fast": {"model": "f"}},
            "agents": {"defaults": {"model": "base"}},
        }
    )
    presets = configured_model_presets(config)
    assert list(presets) == ["fast", "default"] and presets["default"].model == "base"


def test_normalize_preset_name():
    presets = {
        "Deep Research": ModelPresetConfig(model="d"),
        "default": ModelPresetConfig(model="b"),
    }
    assert normalize_preset_name(" deep research ", presets) == "Deep Research"
    with pytest.raises(
        KeyError, match="model_preset 'x' not found. Available: Deep Research, default"
    ):
        normalize_preset_name("x", presets)
    with pytest.raises(ValueError, match="non-empty"):
        normalize_preset_name("  ", presets)


# --- the resolver (nanobot tests/agent/test_model_runtime_resolver.py) ----------------------------


class ClosingProvider(FakeProvider):
    def __init__(self, name: str, script=None) -> None:
        super().__init__(script)
        self.default_model = name
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


def _runtime(provider, model="base-model", window=1000, max_tokens=123):
    return LLMRuntime(
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=0.1,
        context_window_tokens=window,
    )


PRESETS = {
    "default": ModelPresetConfig(
        model="base-model", max_tokens=123, context_window_tokens=1000
    ),
    "fast": ModelPresetConfig(
        model="openai/gpt-4.1", max_tokens=4096, context_window_tokens=32_768
    ),
}


def test_resolver_resolves_preset_without_mutating_selected_runtime():
    base = ClosingProvider("base")
    resolver = ModelRuntimeResolver(_runtime(base), model_presets=PRESETS)
    fast = resolver.resolve_preset("fast")
    assert (
        fast.model,
        fast.max_tokens,
        fast.context_window_tokens,
        fast.model_preset,
    ) == ("openai/gpt-4.1", 4096, 32_768, "fast")
    assert fast.provider is base  # no loader: the current provider
    assert resolver.runtime.model == "base-model" and resolver.model_preset is None
    assert resolver.resolve_preset("FAST") is fast  # cached, case-insensitive


def test_resolver_builds_each_preset_once_with_the_loader_and_closes_all():
    base = ClosingProvider("base")
    built: list[str] = []

    def loader(name):
        built.append(name)
        return LLMRuntime(ClosingProvider(name), PRESETS[name].model, 1, 0.1, 2)

    resolver = ModelRuntimeResolver(
        _runtime(base), model_presets=PRESETS, preset_runtime_loader=loader
    )
    first = resolver.resolve_preset("fast")
    assert resolver.resolve_preset("fast") is first and built == ["fast"]
    assert first.model_preset == "fast"
    selected = resolver.select_preset("fast")
    assert resolver.runtime is selected and resolver.model_preset == "fast"
    asyncio.run(resolver.aclose())
    assert (base.closed, first.provider.closed) == (1, 1)


def test_resolver_model_presets_are_read_only():
    resolver = ModelRuntimeResolver(_runtime(FakeProvider()), model_presets=PRESETS)
    with pytest.raises(TypeError):
        resolver.model_presets["new"] = ModelPresetConfig(model="x")  # type: ignore[index]
    resolver.model_presets["fast"].model = "changed"
    assert resolver.model_presets["fast"].model == "openai/gpt-4.1"


def test_resolver_select_model_and_context_window():
    resolver = ModelRuntimeResolver(_runtime(FakeProvider()), model_presets=PRESETS)
    resolver.select_preset("fast")
    runtime = resolver.select_model(" other ")
    assert (runtime.model, runtime.model_preset) == ("other", None)
    assert resolver.select_context_window(5).context_window_tokens == 5
    with pytest.raises(ValueError):
        resolver.select_model(" ")
    with pytest.raises(TypeError):
        resolver.select_context_window(True)


def test_resolver_unknown_preset_keeps_the_default():
    resolver = ModelRuntimeResolver(_runtime(FakeProvider()), model_presets=PRESETS)
    with pytest.raises(KeyError):
        resolver.select_preset("missing")
    assert resolver.runtime.model == "base-model"


# --- sessions (nanobot tests/agent/test_session_model_runtime.py) ---------------------------------


class RecordingProvider(LLMProvider):
    def __init__(self, name: str) -> None:
        super().__init__(default_model=name)
        self.name = name
        self.calls: list[tuple[str | None, int, float]] = []

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        tool_choice=None,
    ):
        await asyncio.sleep(0)
        self.calls.append((model, max_tokens, temperature))
        return LLMResponse(content=f"reply from {self.name}", finish_reason="stop")


async def test_sessions_run_concurrently_with_isolated_model_presets(tmp_path):
    base, fast, deep = (
        RecordingProvider("base-model"),
        RecordingProvider("fast-model"),
        RecordingProvider("deep-model"),
    )
    providers = {"fast": fast, "deep": deep}
    load_counts = {"fast": 0, "deep": 0}
    presets = {
        "default": ModelPresetConfig(model="base-model", context_window_tokens=8_000),
        "fast": ModelPresetConfig(
            model="fast-model",
            context_window_tokens=16_000,
            max_tokens=111,
            temperature=0.4,
        ),
        "deep": ModelPresetConfig(model="deep-model", context_window_tokens=32_000),
    }

    def load_preset(name):
        load_counts[name] += 1
        p = presets[name]
        return LLMRuntime(
            providers[name],
            p.model,
            p.max_tokens,
            p.temperature,
            p.context_window_tokens,
        )

    loop = AgentLoop(
        base,
        tmp_path,
        model="base-model",
        context_window_tokens=8_000,
        model_presets=presets,
        preset_runtime_loader=load_preset,
    )
    await loop.set_session_model_preset("sdk:fast", "fast")
    await loop.set_session_model_preset("sdk:deep", "deep")
    fast_reply, deep_reply = await asyncio.gather(
        loop.process_direct("hello", session_key="sdk:fast"),
        loop.process_direct("hello", session_key="sdk:deep"),
    )
    assert fast_reply.final_content == "reply from fast-model"
    assert deep_reply.final_content == "reply from deep-model"
    assert fast.calls == [("fast-model", 111, 0.4)]
    assert [call[0] for call in deep.calls] == ["deep-model"]
    assert base.calls == []
    assert loop.provider is base and loop.model == "base-model"
    assert load_counts == {"fast": 1, "deep": 1}

    loop.sessions.invalidate("sdk:fast")
    assert (
        model_preset_from_metadata(loop.sessions.get_or_create("sdk:fast").metadata)
        == "fast"
    )

    # An explicit runtime (Dream's preset) wins for one turn.
    override = RecordingProvider("override-model")
    reply = await loop.process_direct(
        "hello",
        session_key="sdk:fast",
        runtime=LLMRuntime(override, "override-model", 10, 0.1, 24_000),
    )
    assert reply.final_content == "reply from override-model"
    assert fast.calls == [("fast-model", 111, 0.4)]


async def test_removed_session_model_preset_falls_back_and_clears_metadata(tmp_path):
    base = RecordingProvider("base-model")
    loop = AgentLoop(base, tmp_path, model="base-model", context_window_tokens=128_000)
    session = loop.sessions.get_or_create("sdk:removed")
    session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = "removed"
    loop.sessions.save(session)
    reply = await loop.process_direct("hello", session_key="sdk:removed")
    assert reply.final_content == "reply from base-model"
    assert [call[0] for call in base.calls] == ["base-model"]
    loop.sessions.invalidate("sdk:removed")
    assert (
        model_preset_from_metadata(loop.sessions.get_or_create("sdk:removed").metadata)
        is None
    )


def test_model_preset_from_metadata_rejects_blank_values():
    assert model_preset_from_metadata({}) is None
    assert model_preset_from_metadata(None) is None
    assert (
        model_preset_from_metadata({SESSION_MODEL_PRESET_METADATA_KEY: " fast "})
        == "fast"
    )
    with pytest.raises(ValueError):
        model_preset_from_metadata({SESSION_MODEL_PRESET_METADATA_KEY: " "})


# --- /model (nanobot tests/command/test_model_command.py) -----------------------------------------


def _make_loop_with_presets(
    tmp_path, *, preset_runtime_loader=None, model_presets=None
) -> AgentLoop:
    return AgentLoop(
        ClosingProvider("base-model"),
        tmp_path,
        model="base-model",
        max_tokens=123,
        context_window_tokens=1000,
        model_presets=model_presets or PRESETS,
        preset_runtime_loader=preset_runtime_loader,
    )


def _ctx(loop, raw, args=""):
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=raw)
    return CommandContext(
        msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop
    )


def _saved_model_preset(loop, session_key="cli:direct"):
    return model_preset_from_metadata(loop.sessions.get_or_create(session_key).metadata)


async def test_model_command_lists_current_and_available_presets(tmp_path):
    out = await cmd_model(_ctx(_make_loop_with_presets(tmp_path), "/model"))
    assert out.content == "\n".join(
        [
            "## Model",
            "- Current model: `base-model`",
            "- Current preset: `default`",
            "- Available presets: `default`, `fast`",
        ]
    )
    assert out.metadata == {"render_as": "text"}


async def test_model_command_switches_preset(tmp_path):
    loop = _make_loop_with_presets(tmp_path)
    out = await cmd_model(_ctx(loop, "/model fast", args="fast"))
    assert "Switched model preset to `fast`." in out.content
    assert (
        "Scope: current session" in out.content
        and "Model: `openai/gpt-4.1`" in out.content
    )
    assert _saved_model_preset(loop) == "fast"
    assert loop.model_preset is None and loop.model == "base-model"

    await loop.process_direct("/new", session_key="cli:direct")
    assert _saved_model_preset(loop) == "fast"  # /new keeps the choice
    status = await loop.process_direct("/status", session_key="cli:direct")
    assert "openai/gpt-4.1" in status.final_content


async def test_model_command_accepts_canonical_names_with_spaces(tmp_path):
    loop = _make_loop_with_presets(
        tmp_path,
        model_presets={
            "default": ModelPresetConfig(model="base-model"),
            "Deep Research": ModelPresetConfig(model="deep-model"),
        },
    )
    out = await cmd_model(_ctx(loop, "/model deep research", args="deep research"))
    assert "Switched model preset to `Deep Research`." in out.content
    assert _saved_model_preset(loop) == "Deep Research"


async def test_model_command_switches_back_to_default(tmp_path):
    loop = _make_loop_with_presets(tmp_path)
    await cmd_model(_ctx(loop, "/model fast", args="fast"))
    out = await cmd_model(_ctx(loop, "/model default", args="default"))
    assert "Switched model preset to `default`." in out.content
    assert _saved_model_preset(loop) == "default"
    assert (loop.model_preset, loop.model, loop.context_window_tokens) == (
        None,
        "base-model",
        1000,
    )


async def test_model_command_unknown_preset_keeps_old_state(tmp_path):
    loop = _make_loop_with_presets(tmp_path)
    out = await cmd_model(_ctx(loop, "/model missing", args="missing"))
    assert out.content == (
        "Could not switch model preset: model_preset 'missing' not found. Available: default, fast\n\n"
        "Available presets: `default`, `fast`"
    )
    assert loop.model_preset is None and loop.model == "base-model"
    assert _saved_model_preset(loop) is None


async def test_model_command_reports_provider_configuration_errors(tmp_path):
    def fail_preset(_name):
        raise ValueError("No API key configured for provider 'openai'.")

    loop = _make_loop_with_presets(tmp_path, preset_runtime_loader=fail_preset)
    switched = await cmd_model(_ctx(loop, "/model fast", args="fast"))
    session = loop.sessions.get_or_create("cli:direct")
    session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = "fast"
    loop.sessions.save(session)
    status = await cmd_model(_ctx(loop, "/model"))
    assert "Could not switch model preset" in switched.content
    assert "No API key configured for provider 'openai'." in switched.content
    assert "Current selection error" in status.content
    assert "No API key configured for provider 'openai'." in status.content


async def test_model_command_registered_as_exact_and_prefix(tmp_path):
    router = CommandRouter()
    register_builtin_commands(router)
    loop = _make_loop_with_presets(tmp_path)
    out = await router.dispatch(_ctx(loop, "/model fast"))
    assert (out.channel, out.chat_id, out.metadata) == (
        "cli",
        "direct",
        {"render_as": "text"},
    )
    assert out.content == "\n".join(
        [
            "Switched model preset to `fast`.",
            "- Scope: current session",
            "- Model: `openai/gpt-4.1`",
            "- Context window: 32768",
            "- Max output tokens: 4096",
        ]
    )


async def test_model_command_does_not_change_another_session(tmp_path):
    loop = _make_loop_with_presets(tmp_path)
    await cmd_model(_ctx(loop, "/model fast", args="fast"))
    other = InboundMessage(
        channel="cli", sender_id="user", chat_id="other", content="/model"
    )
    out = await cmd_model(
        CommandContext(
            msg=other, session=None, key=other.session_key, raw="/model", loop=loop
        )
    )
    assert "Current preset: `default`" in out.content
    assert _saved_model_preset(loop) == "fast"


async def test_model_command_reports_and_recovers_removed_session_preset(tmp_path):
    loop = _make_loop_with_presets(tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = "removed"
    loop.sessions.save(session)
    status = await loop.process_direct("/model", session_key="cli:direct")
    switched = await loop.process_direct("/model default", session_key="cli:direct")
    assert "model_preset 'removed' not found" in status.final_content
    assert "Available presets: `default`, `fast`" in status.final_content
    assert "Switch with `/model <preset>`" in status.final_content
    assert "Switched model preset to `default`." in switched.final_content
    assert _saved_model_preset(loop) == "default"


def test_model_command_in_help_and_palette():
    model = next(
        item for item in builtin_command_palette() if item["command"] == "/model"
    )
    assert (model["arg_hint"], model["lifecycle"], model["accepts_args"]) == (
        "[preset]",
        "side_channel",
        True,
    )
    assert (
        "/model [preset] — Show or switch the active model preset." in build_help_text()
    )


async def test_turn_after_model_switch_uses_the_preset_provider_and_closes_it(tmp_path):
    fast_provider = ClosingProvider("fast", ["from fast"])

    def loader(name):
        p = PRESETS[name]
        return LLMRuntime(
            fast_provider, p.model, p.max_tokens, p.temperature, p.context_window_tokens
        )

    loop = _make_loop_with_presets(tmp_path, preset_runtime_loader=loader)
    await loop.process_direct("/model fast")
    result = await loop.process_direct("hi")
    assert result.final_content == "from fast"
    assert [(c.model, c.max_tokens) for c in fast_provider.calls] == [
        ("openai/gpt-4.1", 4096)
    ]
    await loop.aclose()
    assert fast_provider.closed == 1 and loop.provider.closed == 1


# --- Dream, the CLI and the config file -----------------------------------------------------------


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


PRESET_CONFIG = {
    "agents": {
        "defaults": {"model": "base-model", "dream": {"modelOverride": "dream"}}
    },
    "providers": {"custom": {"apiKey": "k", "apiBase": "http://x/v1"}},
    "modelPresets": {
        "fast": {
            "model": "fast-model",
            "maxTokens": 2048,
            "contextWindowTokens": 64000,
            "temperature": 0.3,
        },
        "dream": {"model": "dream-model", "maxTokens": 512},
    },
}


def test_make_loop_takes_presets_from_the_config(tmp_path):
    config = Config.model_validate(
        {
            **PRESET_CONFIG,
            "agents": {
                "defaults": {
                    **PRESET_CONFIG["agents"]["defaults"],
                    "modelPreset": "fast",
                }
            },
        }
    )
    loop = _make_loop(FakeProvider([]), tmp_path, config=config)
    assert (
        loop.model,
        loop.max_tokens,
        loop.context_window_tokens,
        loop.temperature,
    ) == ("fast-model", 2048, 64000, 0.3)
    assert loop.model_preset == "fast" and loop.dream_model_preset == "dream"
    assert set(loop.model_presets) == {"fast", "dream", "default"}
    dream = loop.dream_runtime()
    assert (dream.model, dream.max_tokens) == ("dream-model", 512)
    # Without a named preset the loop keeps the provider's model (phase 1 behaviour).
    plain = _make_loop(
        FakeProvider([]), tmp_path, config=Config.model_validate(PRESET_CONFIG)
    )
    assert plain.model is None and plain.model_preset is None


async def test_dream_runs_with_its_preset(tmp_path):
    from nanoclaw.utils.templates import sync_workspace_templates

    sync_workspace_templates(tmp_path)
    fake = FakeProvider(["done"])
    loop = _make_loop(fake, tmp_path, config=Config.model_validate(PRESET_CONFIG))
    loop.memory.append_history("user: hi")
    await loop.run_dream()
    assert [(c.model, c.max_tokens) for c in fake.calls] == [("dream-model", 512)]


def test_cli_preset_flag(tmp_path, capsys):
    path = _write(tmp_path / "config.json", PRESET_CONFIG)
    fake = FakeProvider(["hi"])
    assert (
        main(
            [
                "ask",
                "hello",
                "-c",
                str(path),
                "-w",
                str(tmp_path / "ws"),
                "--preset",
                "FAST",
            ],
            provider=fake,
        )
        == 0
    )
    assert [(c.model, c.max_tokens, c.temperature) for c in fake.calls] == [
        ("fast-model", 2048, 0.3)
    ]
    with pytest.raises(SystemExit, match="--preset: model_preset 'nope' not found"):
        main(
            [
                "ask",
                "hello",
                "-c",
                str(path),
                "-w",
                str(tmp_path / "ws"),
                "--preset",
                "nope",
            ],
            provider=FakeProvider([]),
        )


def test_cli_reports_provider_setup_errors(tmp_path, monkeypatch):
    for key in ("NANOCLAW_API_KEY", "NANOCLAW_API_BASE", "NANOCLAW_MODEL"):
        monkeypatch.delenv(key, raising=False)
    path = _write(
        tmp_path / "config.json",
        {"agents": {"defaults": {"model": "deepseek-chat", "provider": "deepseek"}}},
    )
    with pytest.raises(
        SystemExit,
        match="Provider setup error: No API key configured for provider 'deepseek'.",
    ):
        main(["ask", "hello", "-c", str(path), "-w", str(tmp_path / "ws")])


def test_cli_config_show_lists_presets_with_masked_headers(tmp_path, capsys):
    data = {
        **PRESET_CONFIG,
        "providers": {
            "custom": {
                "apiBase": "http://x/v1",
                "extraHeaders": {"Authorization": "Bearer secret-token-9876"},
            }
        },
    }
    path = _write(tmp_path / "config.json", data)
    assert main(["config", "show", "-c", str(path)]) == 0
    out = capsys.readouterr().out
    shown = json.loads(out)
    assert shown["modelPresets"]["fast"]["model"] == "fast-model"
    assert shown["agents"]["defaults"]["dream"] == {"modelOverride": "dream"}
    assert shown["providers"]["custom"]["extraHeaders"] == {"Authorization": "****9876"}
    assert "secret-token" not in out


def test_banner_shows_the_preset(tmp_path):
    from io import StringIO

    from rich.console import Console

    from nanoclaw.cli.chat import Chat

    config = Config.model_validate(
        {**PRESET_CONFIG, "agents": {"defaults": {"modelPreset": "fast"}}}
    )
    loop = _make_loop(FakeProvider([]), tmp_path, config=config)
    buffer = StringIO()
    chat = Chat(
        loop,
        session_key="cli:direct",
        console=Console(file=buffer, width=200),
        read_input=None,
    )
    chat.banner()
    assert "(fast-model) (preset: fast)" in buffer.getvalue()
