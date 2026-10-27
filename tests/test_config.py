"""config.json (layer 12.1): a nanobot-compatible subset of nanobot config/.

Several tests are adapted from nanobot tests/config/ (test_env_interpolation.py,
test_config_load_errors.py, test_timezone.py, test_config_paths.py).
"""

import json

import pytest

from nanoclaw.__main__ import _make_loop, main
from nanoclaw.agent.loop import AgentLoop
from nanoclaw.agent.subagent import SubagentManager
from nanoclaw.config import (
    Config,
    ConfigEnvError,
    ConfigLoadError,
    apply_env_overrides,
    get_config_path,
    get_data_dir,
    load_config,
    load_runtime_config,
    resolve_config_env_vars,
    save_config,
    set_config_path,
)
from nanoclaw.config.loader import mask_secrets, merge_missing_defaults, refresh_config_file
from nanoclaw.providers import FakeProvider
from nanoclaw.providers.factory import provider_from_config, provider_from_env
from nanoclaw.security import network

ENV_NAMES = (
    "NANOCLAW_CONFIG",
    "NANOCLAW_API_KEY",
    "NANOCLAW_API_BASE",
    "NANOCLAW_MODEL",
    "NANOCLAW_CONTEXT_WINDOW",
    "NANOCLAW_RESTRICT_TO_WORKSPACE",
    "NANOCLAW_COMPACTION",
    "NANOCLAW_SSRF_WHITELIST",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    # Timezone detection reads the host; pin it.
    monkeypatch.setattr("nanoclaw.config.timezone._local_zone_name", lambda: "UTC")
    set_config_path(None)
    yield
    set_config_path(None)
    network.configure_ssrf_whitelist([])


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- schema ----------------------------------------------------------------------------------------


def test_defaults_match_nanobot_except_documented_differences():
    d = Config().agents.defaults
    # Same as nanobot config/schema.py AgentDefaults.
    assert (d.provider, d.max_tokens, d.temperature) == ("auto", 8192, 0.1)
    assert (d.max_tool_iterations, d.max_concurrent_subagents, d.max_tool_result_chars) == (200, 4, 16_000)
    assert (d.provider_retry_mode, d.timezone, d.timezone_mode) == ("standard", "UTC", "auto")
    # Deliberate differences.
    assert d.workspace == "~/.nanoclaw/workspace"
    assert d.model == "gpt-4o-mini"
    assert d.context_window_tokens == 128_000
    assert d.compaction == "nanobot"  # NanoClaw only
    tools = Config().tools
    assert tools.restrict_to_workspace is False and tools.ssrf_whitelist == []


def test_camel_case_and_snake_case_keys():
    camel = Config.model_validate({"agents": {"defaults": {"maxToolIterations": 7, "contextWindowTokens": 9}}})
    snake = Config.model_validate({"agents": {"defaults": {"max_tool_iterations": 7, "context_window_tokens": 9}}})
    assert camel == snake
    assert camel.agents.defaults.max_tool_iterations == 7
    dumped = camel.model_dump(by_alias=True)
    assert dumped["agents"]["defaults"]["maxToolIterations"] == 7
    assert "restrictToWorkspace" in dumped["tools"] and "ssrfWhitelist" in dumped["tools"]


def test_full_nanobot_config_loads_and_unknown_keys_are_ignored(tmp_path):
    path = write(
        tmp_path / "config.json",
        {
            "agents": {"defaults": {"model": "deepseek-chat", "botName": "nanobot", "dream": {"intervalH": 4}}},
            "channels": {"telegram": {"enabled": False}},
            "providers": {"deepseek": {"apiKey": "sk-1", "extraHeaders": {"a": "b"}, "thinkingStyle": None}},
            "gateway": {"port": 18790},
            "modelPresets": {},
            "tools": {"restrictToWorkspace": True, "web": {"search": {"maxResults": 3}}, "exec": {"timeout": 60}},
        },
    )
    config = load_config(path)
    assert config.agents.defaults.model == "deepseek-chat"
    assert config.tools.restrict_to_workspace is True
    assert config.get_api_key() == "sk-1"
    assert config.get_api_base() == "https://api.deepseek.com"  # nanobot registry default


# --- provider selection ----------------------------------------------------------------------------


def providers_config(providers, **defaults):
    return Config.model_validate({"agents": {"defaults": defaults}, "providers": providers})


def test_forced_provider():
    config = providers_config(
        {"openrouter": {"apiKey": "or"}, "lm_studio": {"apiBase": "http://localhost:1234/v1"}}, provider="lmStudio"
    )
    assert config.resolve_provider()[0] == "lm_studio"
    assert (config.get_api_key(), config.get_api_base()) == (None, "http://localhost:1234/v1")
    assert providers_config({}, provider="missing").resolve_provider() == (None, None)


def test_auto_prefers_the_model_prefix_then_the_first_configured_entry():
    providers = {"custom": {}, "deepseek": {"apiKey": "ds"}, "openrouter": {"apiKey": "or"}}
    assert providers_config(providers, model="openrouter/anthropic/claude").resolve_provider()[0] == "openrouter"
    config = providers_config(providers, model="gpt-4o")
    assert config.resolve_provider()[0] == "deepseek"
    assert config.get_api_base() == "https://api.deepseek.com"
    assert providers_config({"custom": {}}).resolve_provider() == (None, None)


def test_provider_from_config():
    config = providers_config({"openrouter": {"apiKey": "or"}}, model="x/y")
    provider = provider_from_config(config)
    assert (provider.api_key, provider.api_base, provider.default_model) == ("or", "https://openrouter.ai/api/v1", "x/y")


# --- loading, errors, saving -----------------------------------------------------------------------


def test_config_path_precedence(tmp_path, monkeypatch):
    assert get_config_path().name == "config.json" and get_config_path().parent.name == ".nanoclaw"
    monkeypatch.setenv("NANOCLAW_CONFIG", str(tmp_path / "env.json"))
    assert get_config_path() == tmp_path / "env.json"
    set_config_path(tmp_path / "flag" / "c.json")
    assert get_config_path() == tmp_path / "flag" / "c.json"
    assert get_data_dir() == tmp_path / "flag"  # nanobot: data dir = the config file's folder


def test_missing_file_uses_defaults(tmp_path):
    assert load_config(tmp_path / "missing.json") == Config()


def test_invalid_json_fails_fast(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{broken json", encoding="utf-8")
    with pytest.raises(ConfigLoadError) as exc_info:
        load_config(path)
    assert exc_info.value.kind == "invalid_json"
    assert "line 1, column 2" in str(exc_info.value)


@pytest.mark.parametrize(("content", "root_type"), [("[]", "list"), ("null", "NoneType"), ('"value"', "str")])
def test_rejects_non_object_root(tmp_path, content, root_type):
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigLoadError) as exc_info:
        load_config(path)
    assert exc_info.value.kind == "invalid_root"
    assert f"Expected an object, but found {root_type}." in str(exc_info.value)


def test_invalid_schema_is_readable_and_hides_values(tmp_path):
    secret = "should-never-appear"
    path = write(
        tmp_path / "config.json",
        {
            "agents": {"defaults": {"maxToolIterations": "lots", "maxConcurrentSubagents": 0}},
            "providers": {"openrouter": {"apiKey": [secret]}},
        },
    )
    with pytest.raises(ConfigLoadError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert exc_info.value.kind == "invalid_schema"
    assert "Found 3 invalid setting(s)." in message
    assert "agents.defaults.maxToolIterations" in message
    assert "Must be a valid integer" in message
    assert "agents.defaults.maxConcurrentSubagents" in message
    assert "Must be greater than or equal to 1." in message
    assert secret not in message and "errors.pydantic.dev" not in message


def test_invalid_enum_values(tmp_path):
    path = write(tmp_path / "c.json", {"agents": {"defaults": {"providerRetryMode": "forever", "compaction": "smart"}}})
    with pytest.raises(ConfigLoadError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "agents.defaults.providerRetryMode" in message and "agents.defaults.compaction" in message


def test_env_references_are_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "resolved-key")
    monkeypatch.setenv("HOST", "example.com")
    path = write(
        tmp_path / "config.json",
        {"providers": {"groq": {"apiKey": "${TEST_API_KEY}", "apiBase": "https://${HOST}/v1"}}},
    )
    raw = load_config(path)
    assert raw.providers["groq"].api_key == "${TEST_API_KEY}"
    resolved = resolve_config_env_vars(raw)
    assert resolved.providers["groq"].api_key == "resolved-key"
    assert resolved.providers["groq"].api_base == "https://example.com/v1"
    assert raw.providers["groq"].api_key == "${TEST_API_KEY}"  # the loaded config is not changed


def test_missing_env_reference_reports_the_field(tmp_path, monkeypatch):
    name = "NANOCLAW_TEST_MISSING_PROVIDER_KEY"
    monkeypatch.delenv(name, raising=False)
    config = Config.model_validate({"providers": {"openrouter": {"apiKey": f"${{{name}}}"}}})
    with pytest.raises(ConfigLoadError) as exc_info:
        resolve_config_env_vars(config, config_path=tmp_path / "config.json")
    assert exc_info.value.kind == "missing_env"
    assert "providers.openrouter.apiKey" in str(exc_info.value)
    assert name in str(exc_info.value)


def test_save_writes_camel_case_and_keeps_templates(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "real-token")
    path = write(tmp_path / "config.json", {"providers": {"custom": {"apiKey": "${MY_TOKEN}"}}})
    config = load_config(path)
    config.agents.defaults.max_tokens = 1234
    save_config(config, path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["providers"]["custom"]["apiKey"] == "${MY_TOKEN}"
    assert saved["agents"]["defaults"]["maxTokens"] == 1234
    assert load_config(path) == config
    assert not (tmp_path / "config.json.tmp").exists()


def test_merge_missing_defaults():
    assert merge_missing_defaults({"a": 1, "b": {"c": 2}}, {"a": 0, "b": {"c": 0, "d": 3}, "e": 4}) == {
        "a": 1,
        "b": {"c": 2, "d": 3},
        "e": 4,
    }
    assert merge_missing_defaults([1], {"a": 1}) == [1]


def test_refresh_keeps_nanobot_only_keys_and_adds_missing_settings(tmp_path):
    path = write(
        tmp_path / "config.json",
        {"agents": {"defaults": {"model": "m", "botName": "nb"}}, "gateway": {"port": 1}, "providers": {"x": {}}},
    )
    assert refresh_config_file(path) is True
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["gateway"] == {"port": 1} and saved["agents"]["defaults"]["botName"] == "nb"
    assert saved["agents"]["defaults"]["model"] == "m"
    assert saved["agents"]["defaults"]["maxToolIterations"] == 200
    assert saved["tools"]["ssrfWhitelist"] == []
    assert saved["providers"] == {"x": {}}  # no "custom" entry added
    assert refresh_config_file(path) is False


# --- timezone (nanobot tests/config/test_timezone.py) ----------------------------------------------


def test_new_config_detects_timezone(monkeypatch):
    monkeypatch.setattr("nanoclaw.config.timezone._local_zone_name", lambda: "Asia/Shanghai")
    d = Config().agents.defaults
    assert (d.timezone, d.timezone_mode) == ("Asia/Shanghai", "auto")


def test_explicit_timezone_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr("nanoclaw.config.timezone._local_zone_name", lambda: "Asia/Shanghai")
    path = write(tmp_path / "c.json", {"agents": {"defaults": {"timezone": "America/New_York"}}})
    d = load_config(path).agents.defaults
    assert (d.timezone, d.timezone_mode) == ("America/New_York", "manual")
    path = write(tmp_path / "c.json", {"agents": {"defaults": {"timezone": "UTC", "timezoneMode": "auto"}}})
    assert load_config(path).agents.defaults.timezone == "Asia/Shanghai"


def test_unknown_timezone_is_rejected(tmp_path, monkeypatch):
    path = write(tmp_path / "c.json", {"agents": {"defaults": {"timezone": "Mars/Base"}}})
    with pytest.raises(ConfigLoadError, match="agents.defaults.timezone"):
        load_config(path)
    from nanoclaw.config.timezone import detect_system_timezone

    monkeypatch.setattr("nanoclaw.config.timezone._local_zone_name", lambda: "Not/AZone")
    assert detect_system_timezone() == "UTC"
    monkeypatch.setattr("nanoclaw.config.timezone._local_zone_name", lambda: "Etc/UTC")
    assert detect_system_timezone() == "UTC"


# --- environment variables override the file -------------------------------------------------------


def test_env_overrides_the_file(tmp_path):
    path = write(
        tmp_path / "config.json",
        {
            "agents": {"defaults": {"model": "file-model", "contextWindowTokens": 50_000, "compaction": "fixed"}},
            "providers": {"openrouter": {"apiKey": "file-key"}},
            "tools": {"restrictToWorkspace": True, "ssrfWhitelist": ["10.0.0.0/8"]},
        },
    )
    file_only = load_runtime_config(path, env={})
    assert file_only.agents.defaults.model == "file-model"
    assert file_only.get_api_key() == "file-key"
    env = {
        "NANOCLAW_API_KEY": "env-key",
        "NANOCLAW_API_BASE": "http://localhost:11434/v1",
        "NANOCLAW_MODEL": "llama3.2",
        "NANOCLAW_CONTEXT_WINDOW": "9000",
        "NANOCLAW_RESTRICT_TO_WORKSPACE": "0",
        "NANOCLAW_COMPACTION": " Nanobot ",
        "NANOCLAW_SSRF_WHITELIST": " 100.64.0.0/10 , ,",
    }
    merged = load_runtime_config(path, env=env)
    d = merged.agents.defaults
    assert (d.model, d.context_window_tokens, d.compaction) == ("llama3.2", 9000, "nanobot")
    assert merged.tools.restrict_to_workspace is False
    assert merged.tools.ssrf_whitelist == ["100.64.0.0/10"]
    # The key and base go to the entry the config picked.
    assert merged.resolve_provider()[0] == "openrouter"
    assert (merged.get_api_key(), merged.get_api_base()) == ("env-key", "http://localhost:11434/v1")
    # Empty variables do not override.
    assert load_runtime_config(path, env={k: "" for k in env}) == file_only


def test_env_alone_fills_a_custom_entry():
    config = apply_env_overrides(Config(), {"NANOCLAW_API_KEY": "k", "NANOCLAW_RESTRICT_TO_WORKSPACE": "yes"})
    assert config.resolve_provider()[0] == "custom"
    assert config.get_api_key() == "k" and config.get_api_base() is None
    assert config.tools.restrict_to_workspace is True
    forced = Config.model_validate({"agents": {"defaults": {"provider": "deepseek"}}})
    config = apply_env_overrides(forced, {"NANOCLAW_API_KEY": "k"})
    assert (config.get_api_key(), config.get_api_base()) == ("k", "https://api.deepseek.com")
    assert forced.providers == Config().providers  # the input is not changed


def test_provider_from_env_still_works():
    p = provider_from_env({"NANOCLAW_API_KEY": "k", "NANOCLAW_MODEL": "m"})
    assert (p.api_key, p.api_base, p.default_model) == ("k", None, "m")


def test_env_errors_keep_their_messages():
    with pytest.raises(ConfigEnvError, match="NANOCLAW_CONTEXT_WINDOW must be a whole number of tokens, got 'lots'"):
        apply_env_overrides(Config(), {"NANOCLAW_CONTEXT_WINDOW": "lots"})
    with pytest.raises(ConfigEnvError, match="NANOCLAW_COMPACTION must be one of nanobot, fixed, got 'smart'"):
        apply_env_overrides(Config(), {"NANOCLAW_COMPACTION": "smart"})


def test_mask_secrets():
    data = {"providers": {"a": {"apiKey": "sk-1234567890abcd"}, "b": {"apiKey": "short"}, "c": {"apiKey": None}}}
    assert mask_secrets(data) == {
        "providers": {"a": {"apiKey": "****abcd"}, "b": {"apiKey": "****"}, "c": {"apiKey": None}}
    }


# --- iteration limit (phase 2 decision 7) ----------------------------------------------------------


def test_iteration_limit_defaults_to_200(tmp_path):
    loop = AgentLoop(FakeProvider([]), tmp_path)
    assert loop.max_iterations == 200 and loop.subagents.max_iterations == 200
    assert SubagentManager(tmp_path).max_iterations == 200
    assert _make_loop(FakeProvider([]), tmp_path).max_iterations == 200


def test_make_loop_uses_the_config(tmp_path):
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "maxToolIterations": 12,
                    "maxConcurrentSubagents": 2,
                    "maxToolResultChars": 999,
                    "maxTokens": 100,
                    "temperature": 0.5,
                    "contextWindowTokens": 60_000,
                    "compaction": "fixed",
                    "providerRetryMode": "persistent",
                }
            },
            "tools": {"restrictToWorkspace": True},
        }
    )
    loop = _make_loop(FakeProvider([]), tmp_path, config=config)
    assert (loop.max_iterations, loop.subagents.max_iterations, loop.subagents.max_concurrent_subagents) == (12, 12, 2)
    assert (loop.max_tool_result_chars, loop.max_tokens, loop.temperature) == (999, 100, 0.5)
    assert loop.context_window_tokens == 60_000 and loop.restrict_to_workspace is True
    assert loop.consolidator.archiver.compaction == "fixed"
    assert loop.provider_retry_mode == "persistent"
    # --restrict can turn the restriction on, not off.
    assert _make_loop(FakeProvider([]), tmp_path, restrict=True).restrict_to_workspace is True


# --- CLI -------------------------------------------------------------------------------------------


def test_cli_config_path(tmp_path, monkeypatch, capsys):
    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out.strip().endswith("config.json")
    monkeypatch.setenv("NANOCLAW_CONFIG", str(tmp_path / "env.json"))
    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "env.json")
    assert main(["config", "path", "-c", str(tmp_path / "flag.json")]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "flag.json")


def test_cli_config_init_and_show(tmp_path, monkeypatch, capsys):
    path = tmp_path / "c" / "config.json"
    assert main(["config", "init", "-c", str(path)]) == 0
    assert "Created config at" in capsys.readouterr().out
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["agents"]["defaults"]["maxToolIterations"] == 200
    assert saved["providers"] == {"custom": {"apiKey": None, "apiBase": None}}
    assert main(["config", "init", "-c", str(path)]) == 0
    assert "already up to date" in capsys.readouterr().out

    saved["providers"]["custom"]["apiKey"] = "sk-file-secret-1234"
    path.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setenv("NANOCLAW_MODEL", "env-model")
    assert main(["config", "show", "-c", str(path)]) == 0
    out = capsys.readouterr().out
    shown = json.loads(out)
    assert shown["agents"]["defaults"]["model"] == "env-model"
    assert shown["providers"]["custom"]["apiKey"] == "****1234"
    assert "sk-file-secret" not in out


def test_cli_reports_a_broken_config(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(SystemExit, match="JSON syntax error"):
        main(["ask", "hi", "-c", str(path), "-w", str(tmp_path / "ws")], provider=FakeProvider(["x"]))
    with pytest.raises(SystemExit, match="JSON syntax error"):
        main(["config", "show", "-c", str(path)])
    assert main(["config", "path", "-c", str(path)]) == 0  # path never reads the file


def test_cli_ask_reads_the_config(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "from-config"
    path = write(
        tmp_path / "config.json",
        {
            "agents": {"defaults": {"workspace": str(workspace), "contextWindowTokens": 9000}},
            "tools": {"ssrfWhitelist": ["10.0.0.0/8"]},
        },
    )
    fake = FakeProvider(["- s"])
    # 9000 - 8192 - 1024 < 0: the file's context window is used.
    assert main(["ask", "hello", "-c", str(path)], provider=fake) == 1
    assert "exceeds the local context budget" in capsys.readouterr().err
    assert (workspace / "AGENTS.md").exists()  # the configured workspace
    assert [str(n) for n in network._allowed_networks] == ["10.0.0.0/8"]
    # The environment beats the file; -w beats the configured workspace.
    monkeypatch.setenv("NANOCLAW_CONTEXT_WINDOW", "128000")
    monkeypatch.setenv("NANOCLAW_CONFIG", str(path))
    other = tmp_path / "flag-ws"
    assert main(["ask", "hello", "-w", str(other)], provider=FakeProvider(["hi"])) == 0
    assert capsys.readouterr().out.strip() == "hi"
    assert (other / "AGENTS.md").exists()
