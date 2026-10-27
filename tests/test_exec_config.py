"""``tools.exec`` config (layer 16.2): nanobot ``ExecToolConfig``.

The first part is ported from nanobot tests/config/test_shell_config.py. Left
out: the WebUI settings editor, the ``my`` tool and the ``NANOBOT_TOOLS__EXEC``
pydantic-settings precedence tests (NanoClaw has none of these).

The second part pins the results of loading the same ``config.json`` files with
nanobot c4dac64 (``load_config(path).tools.exec``): the expected values below
are what nanobot returned, and NanoClaw's dumps matched them exactly.
"""

import json

import pytest
from pydantic import ValidationError

from nanoclaw.__main__ import main
from nanoclaw.config import Config, ConfigLoadError, ExecToolConfig, load_config, save_config, set_config_path
from nanoclaw.config.loader import refresh_config_file


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("NANOCLAW_CONFIG", "NANOCLAW_MODEL", "NANOCLAW_API_KEY", "NANOCLAW_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("nanoclaw.config.timezone._local_zone_name", lambda: "UTC")
    set_config_path(None)
    yield
    set_config_path(None)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- ported from nanobot tests/config/test_shell_config.py -----------------------------------------


def test_exec_timing_defaults() -> None:
    config = Config().tools.exec

    assert config.max_runtime == 3600
    assert config.max_runtime_limit == 86400
    assert config.idle_timeout == 300
    assert config.idle_action == "report"
    assert config.timeout == config.max_runtime


@pytest.mark.parametrize(
    "values",
    [
        {"maxRuntime": 172800, "maxRuntimeLimit": 14400, "idleTimeout": 90, "idleAction": "terminate"},
        {"max_runtime": 172800, "max_runtime_limit": 14400, "idle_timeout": 90, "idle_action": "terminate"},
        {"timeout": 172800, "maxRuntimeLimit": 14400, "idleTimeout": 90, "idleAction": "terminate"},
    ],
)
def test_exec_timing_aliases_and_serialization(values) -> None:
    config = ExecToolConfig.model_validate(values)
    dumped = config.model_dump(by_alias=True)

    assert config.max_runtime == 172800
    assert config.max_runtime_limit == 14400
    assert config.idle_timeout == 90
    assert config.idle_action == "terminate"
    assert dumped["maxRuntime"] == 172800
    assert dumped["maxRuntimeLimit"] == 14400
    assert dumped["idleTimeout"] == 90
    assert dumped["idleAction"] == "terminate"
    assert not {"timeout", "max_runtime", "max_runtime_limit", "idle_timeout", "idle_action"} & dumped.keys()
    assert ExecToolConfig.model_validate_json(config.model_dump_json(by_alias=True)) == config


@pytest.mark.parametrize("name", ["timeout", "maxRuntime", "max_runtime"])
@pytest.mark.parametrize("runtime", [0, 60, 3600, 172800])
def test_exec_runtime_does_not_raise_default_model_limit(name: str, runtime: int) -> None:
    config = ExecToolConfig.model_validate({name: runtime})

    assert config.max_runtime == runtime
    assert config.timeout == runtime
    assert config.max_runtime_limit == 86400


@pytest.mark.parametrize("name", ["maxRuntime", "max_runtime"])
def test_exec_rejects_simultaneous_legacy_and_canonical_runtime(name: str) -> None:
    with pytest.raises(ValidationError, match="not both"):
        Config.model_validate({"tools": {"exec": {"timeout": 60, name: 60}}})


@pytest.mark.parametrize(
    "values",
    [
        {"maxRuntime": -1},
        {"timeout": -1},
        {"idleTimeout": -1},
        {"maxRuntime": 1.5},
        {"idleTimeout": 1.5},
        {"maxRuntimeLimit": 0},
        {"maxRuntimeLimit": -1},
        {"idleAction": "ignore"},
    ],
)
def test_exec_rejects_invalid_timing_configuration(values) -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"tools": {"exec": values}})


def test_exec_unlimited_default_keeps_a_positive_model_limit() -> None:
    config = ExecToolConfig(max_runtime=0, max_runtime_limit=1, idle_timeout=0)

    assert config.max_runtime == 0
    assert config.max_runtime_limit == 1
    assert config.idle_timeout == 0
    assert ExecToolConfig(max_runtime=1, max_runtime_limit=1).max_runtime == 1


def test_legacy_timeout_property_writes_canonical_runtime() -> None:
    config = ExecToolConfig()
    config.timeout = 42
    assert config.max_runtime == 42
    assert config.model_dump(by_alias=True)["maxRuntime"] == 42

    config.max_runtime = 0
    assert config.timeout == 0
    assert "timeout" not in config.model_dump()


def test_legacy_exec_config_loads_and_saves_canonical_fields(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tools": {"exec": {"timeout": 172800}}}), encoding="utf-8")

    config = load_config(path)
    save_config(config, path)
    restored = load_config(path).tools.exec
    saved = json.loads(path.read_text(encoding="utf-8"))["tools"]["exec"]

    assert restored.max_runtime == 172800
    assert restored.max_runtime_limit == 86400
    assert saved["maxRuntime"] == 172800
    assert saved["maxRuntimeLimit"] == 86400
    assert saved["idleTimeout"] == 300
    assert saved["idleAction"] == "report"
    assert "timeout" not in saved


def test_exec_load_rejects_runtime_aliases_from_the_same_file(tmp_path):
    path = write(tmp_path / "config.json", {"tools": {"exec": {"timeout": 60, "maxRuntime": 90}}})
    with pytest.raises(ConfigLoadError):
        load_config(path)


# --- same results as nanobot c4dac64 -----------------------------------------------------------------

DEFAULT_DUMP = {
    "enable": True,
    "maxRuntime": 3600,
    "maxRuntimeLimit": 86400,
    "idleTimeout": 300,
    "idleAction": "report",
    "pathPrepend": "",
    "pathAppend": "",
    "sandbox": "",
    "sandboxRoBinds": [],
    "sandboxRwBinds": [],
    "allowedEnvKeys": [],
    "allowPatterns": [],
    "denyPatterns": [],
}

CAMEL_ALL = {
    "enable": False,
    "maxRuntime": 7200,
    "maxRuntimeLimit": 14400,
    "idleTimeout": 90,
    "idleAction": "terminate",
    "pathPrepend": "/opt/bin",
    "pathAppend": "/usr/sbin",
    "sandbox": "bwrap",
    "sandboxRoBinds": ["/etc"],
    "sandboxRwBinds": ["/tmp/x"],
    "allowedEnvKeys": ["HOME", "LANG"],
    "allowPatterns": ["^ls"],
    "denyPatterns": ["rm -rf"],
}

SNAKE_ALL = {
    "max_runtime": 10,
    "max_runtime_limit": 20,
    "idle_timeout": 0,
    "idle_action": "terminate",
    "path_prepend": "a",
    "path_append": "b",
    "sandbox_ro_binds": ["r"],
    "sandbox_rw_binds": ["w"],
    "allowed_env_keys": ["K"],
    "allow_patterns": ["x"],
    "deny_patterns": ["y"],
}

SAME_AS_NANOBOT = [
    ("empty", {}, DEFAULT_DUMP),
    ("no_exec", {"tools": {"restrictToWorkspace": True}}, DEFAULT_DUMP),
    ("exec_empty", {"tools": {"exec": {}}}, DEFAULT_DUMP),
    ("camel_all", {"tools": {"exec": CAMEL_ALL}}, CAMEL_ALL),
    (
        "snake_all",
        {"tools": {"exec": SNAKE_ALL}},
        {
            **DEFAULT_DUMP,
            "maxRuntime": 10,
            "maxRuntimeLimit": 20,
            "idleTimeout": 0,
            "idleAction": "terminate",
            "pathPrepend": "a",
            "pathAppend": "b",
            "sandboxRoBinds": ["r"],
            "sandboxRwBinds": ["w"],
            "allowedEnvKeys": ["K"],
            "allowPatterns": ["x"],
            "denyPatterns": ["y"],
        },
    ),
    ("legacy_timeout", {"tools": {"exec": {"timeout": 172800}}}, {**DEFAULT_DUMP, "maxRuntime": 172800}),
    ("legacy_timeout_zero", {"tools": {"exec": {"timeout": 0}}}, {**DEFAULT_DUMP, "maxRuntime": 0}),
    (
        "unknown_keys",
        {"tools": {"exec": {"maxRuntime": 5, "fooBar": 1, "shell": "pwsh"}, "web": {"search": {"maxResults": 3}}}},
        {**DEFAULT_DUMP, "maxRuntime": 5},
    ),
    (
        "string_int",
        {"tools": {"exec": {"maxRuntime": "120", "enable": "false"}}},
        {**DEFAULT_DUMP, "maxRuntime": 120, "enable": False},
    ),
    (
        "env_ref_left_as_written",
        {"tools": {"exec": {"pathPrepend": "${HOME}/bin"}}},
        {**DEFAULT_DUMP, "pathPrepend": "${HOME}/bin"},
    ),
]


@pytest.mark.parametrize(("name", "data", "expected"), SAME_AS_NANOBOT, ids=[case[0] for case in SAME_AS_NANOBOT])
def test_exec_section_loads_as_in_nanobot(tmp_path, name, data, expected):
    loaded = load_config(write(tmp_path / "config.json", data)).tools.exec
    assert loaded.model_dump(mode="json", by_alias=True) == expected
    assert loaded.timeout == expected["maxRuntime"]


@pytest.mark.parametrize(
    "exec_section",
    [
        {"timeout": 60, "maxRuntime": 90},
        {"timeout": 60, "max_runtime": 90},
        {"maxRuntime": -1},
        {"idleTimeout": 1.5},
        {"maxRuntimeLimit": 0},
        {"idleAction": "ignore"},
    ],
)
def test_exec_section_rejected_as_in_nanobot(tmp_path, exec_section):
    with pytest.raises(ConfigLoadError):
        load_config(write(tmp_path / "config.json", {"tools": {"exec": exec_section}}))


# --- config init / show ----------------------------------------------------------------------------


def test_cli_config_init_writes_the_exec_section(tmp_path, capsys):
    path = tmp_path / "config.json"
    assert main(["config", "init", "-c", str(path)]) == 0
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["tools"]["exec"] == DEFAULT_DUMP
    assert list(saved["tools"])[0] == "exec"  # nanobot order: exec before restrictToWorkspace


def test_cli_config_init_adds_missing_exec_settings(tmp_path, capsys):
    path = write(tmp_path / "config.json", {"tools": {"exec": {"maxRuntime": 60, "nanobotOnly": 1}}})
    assert main(["config", "init", "-c", str(path)]) == 0
    assert "refreshed" in capsys.readouterr().out
    saved = json.loads(path.read_text(encoding="utf-8"))["tools"]["exec"]
    assert saved == {**DEFAULT_DUMP, "maxRuntime": 60, "nanobotOnly": 1}


def test_refresh_keeps_legacy_timeout_without_adding_max_runtime(tmp_path):
    path = write(tmp_path / "config.json", {"tools": {"exec": {"timeout": 60}}})
    assert refresh_config_file(path) is True
    saved = json.loads(path.read_text(encoding="utf-8"))["tools"]["exec"]
    assert saved["timeout"] == 60 and "maxRuntime" not in saved
    assert saved["idleTimeout"] == 300
    assert load_config(path).tools.exec.max_runtime == 60  # still valid
    assert refresh_config_file(path) is False


def test_cli_config_show_shows_the_exec_section(tmp_path, capsys):
    path = write(tmp_path / "config.json", {"tools": {"exec": {"timeout": 120, "allowedEnvKeys": ["LANG"]}}})
    assert main(["config", "show", "-c", str(path)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["tools"]["exec"] == {**DEFAULT_DUMP, "maxRuntime": 120, "allowedEnvKeys": ["LANG"]}
