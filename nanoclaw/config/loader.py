"""Load and save ``config.json``. Mirrors nanobot ``config/loader.py``.

Where the file lives: ``--config PATH`` (``set_config_path``), else
``NANOCLAW_CONFIG``, else ``~/.nanoclaw/config.json`` (nanobot:
``~/.nanobot/config.json``, loader.py:33). A missing file means defaults.

Precedence, lowest to highest:

1. defaults (``nanoclaw/config/schema.py``)
2. ``config.json``, with ``${VAR}`` references resolved
   (``resolve_config_env_vars``, nanobot loader.py:190)
3. ``NANOCLAW_*`` environment variables (``apply_env_overrides``)
4. command-line flags (``-w``, ``--restrict``; applied in ``nanoclaw/__main__.py``)

Left out: nanobot's ``--home`` instance roots (config/home.py), legacy-format
migrations (loader.py:344), the network proxy section and pydantic-settings
``NANOBOT_*`` variables.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from nanoclaw.config.errors import ConfigIssue, ConfigLoadError, validation_issues
from nanoclaw.config.schema import Config, ProviderConfig

CONFIG_PATH_ENV = "NANOCLAW_CONFIG"
DEFAULT_CONFIG_PATH = Path("~/.nanoclaw/config.json")

ENV_API_KEY = "NANOCLAW_API_KEY"
ENV_API_BASE = "NANOCLAW_API_BASE"
ENV_MODEL = "NANOCLAW_MODEL"
ENV_CONTEXT_WINDOW = "NANOCLAW_CONTEXT_WINDOW"
ENV_RESTRICT = "NANOCLAW_RESTRICT_TO_WORKSPACE"
ENV_COMPACTION = "NANOCLAW_COMPACTION"
ENV_SSRF_WHITELIST = "NANOCLAW_SSRF_WHITELIST"

_TRUE_VALUES = {"1", "true", "yes", "on"}

# Set by ``--config`` (nanobot loader.py:24, multi-instance support).
_current_config_path: Path | None = None


class ConfigEnvError(ValueError):
    """A ``NANOCLAW_*`` variable has a value NanoClaw cannot use."""


def set_config_path(path: Path | None) -> None:
    """Use ``path`` as the config file (``None`` goes back to the default)."""
    global _current_config_path
    _current_config_path = path.expanduser() if path is not None else None


def get_config_path() -> Path:
    """``--config``, else ``NANOCLAW_CONFIG``, else ``~/.nanoclaw/config.json``."""
    if _current_config_path is not None:
        return _current_config_path
    override = os.environ.get(CONFIG_PATH_ENV, "").strip()
    return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH.expanduser()


def get_data_dir() -> Path:
    """Instance data directory: the config file's folder (nanobot config/paths.py:21).

    NanoClaw keeps sessions and memory in the workspace, so nothing is stored
    here yet; future runtime data (cron, logs) belongs here.
    """
    return get_config_path().parent


def load_config(config_path: Path | None = None) -> Config:
    """Read the config file, or return defaults when it does not exist (nanobot loader.py:40).

    ``${VAR}`` references are left as written; see ``resolve_config_env_vars``.
    """
    path = config_path or get_config_path()
    if not path.exists():
        return Config()
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigLoadError(
            path,
            kind="invalid_json",
            summary=f"JSON syntax error at line {exc.lineno}, column {exc.colno}: {_sentence(exc.msg)}",
        ) from exc
    except UnicodeDecodeError as exc:
        raise ConfigLoadError(path, kind="io_error", summary="The file is not valid UTF-8.") from exc
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise ConfigLoadError(path, kind="io_error", summary=f"Unable to read the file: {_sentence(detail)}") from exc

    if not isinstance(data, dict):
        raise ConfigLoadError(
            path,
            kind="invalid_root",
            summary="The top level of config.json must be a JSON object.",
            issues=(ConfigIssue(path=(), message=f"Expected an object, but found {type(data).__name__}."),),
        )
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        issues = validation_issues(exc)
        raise ConfigLoadError(
            path, kind="invalid_schema", summary=f"Found {len(issues)} invalid setting(s).", issues=issues
        ) from exc


def config_to_json(config: Config) -> dict[str, Any]:
    """The camelCase JSON form written to ``config.json``."""
    return config.model_dump(mode="json", by_alias=True)


def save_config(config: Config, config_path: Path | None = None) -> None:
    """Write the config as camelCase JSON (nanobot loader.py:139).

    Written to a temporary file and renamed, so a crash cannot leave a
    truncated ``config.json``. Keys NanoClaw does not know are not kept; to
    add missing defaults to an existing file use ``refresh_config_file``.
    """
    _write_json(config_path or get_config_path(), config_to_json(config))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def merge_missing_defaults(existing: object, defaults: object) -> object:
    """Recursively add missing defaults without replacing configured values (nanobot loader.py:171)."""
    if not isinstance(existing, dict) or not isinstance(defaults, dict):
        return existing
    merged = dict(cast(dict[str, object], existing))
    for key, value in cast(dict[str, object], defaults).items():
        merged[key] = value if key not in merged else merge_missing_defaults(merged[key], value)
    return merged


def refresh_config_file(config_path: Path | None = None) -> bool:
    """Add NanoClaw's missing settings to an existing file, keeping everything
    else, including keys only nanobot reads. Returns whether the file changed.

    nanobot ``onboard --refresh`` loads and re-saves the whole config
    (commands.py:200); NanoClaw merges into the raw JSON instead, because a
    re-save would drop the nanobot sections it does not model.
    """
    path = config_path or get_config_path()
    config = load_config(path)  # validates first
    raw = json.loads(path.read_text(encoding="utf-8"))
    defaults = config_to_json(Config())
    # Provider entries are the user's choice; do not add a "custom" entry to a
    # file that already names its providers.
    if isinstance(raw.get("providers"), dict):
        defaults.pop("providers")
    # The detected timezone belongs to the loaded config, not to the defaults.
    defaults["agents"]["defaults"]["timezone"] = config.agents.defaults.timezone
    # A legacy ``tools.exec.timeout`` already sets the runtime; adding
    # ``maxRuntime`` beside it would make the file invalid ("not both").
    raw_exec = raw.get("tools", {}).get("exec") if isinstance(raw.get("tools"), dict) else None
    if isinstance(raw_exec, dict) and "timeout" in raw_exec:
        defaults["tools"]["exec"].pop("maxRuntime")
    merged = merge_missing_defaults(raw, defaults)
    if merged == raw:
        return False
    _write_json(path, merged)
    return True


_ENV_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_config_env_vars(config: Config, *, config_path: Path | None = None) -> Config:
    """Return ``config`` with ``${VAR}`` references resolved (nanobot loader.py:190).

    Raises ``ConfigLoadError`` listing every referenced variable that is not set.
    """
    missing = tuple(_missing_env_issues(config))
    if missing:
        raise ConfigLoadError(
            config_path or get_config_path(),
            kind="missing_env",
            summary=f"Found {len(missing)} missing environment variable reference(s).",
            issues=missing,
        )
    return _resolve_in_place(config)


def _resolve_in_place(obj: Any) -> Any:
    """Copy-on-write walk (nanobot loader.py:247); unchanged parts are shared."""
    if isinstance(obj, str):
        return _ENV_REF_PATTERN.sub(_env_replace, obj)
    if isinstance(obj, BaseModel):
        updates: dict[str, Any] = {}
        for name in type(obj).model_fields:
            old = getattr(obj, name)
            new = _resolve_in_place(old)
            if new is not old and new != old:
                updates[name] = new
        return obj.model_copy(update=updates) if updates else obj
    if isinstance(obj, dict):
        return {key: _resolve_in_place(value) for key, value in cast(dict[str, Any], obj).items()}
    if isinstance(obj, list):
        return [_resolve_in_place(value) for value in cast(list[Any], obj)]
    return obj


def _missing_env_issues(obj: Any, path: tuple[str | int, ...] = ()) -> list[ConfigIssue]:
    """nanobot loader.py:291; locations use the JSON (camelCase) names."""
    if isinstance(obj, str):
        return [
            ConfigIssue(path=path, message=f"Environment variable '{name}' is not set.")
            for name in dict.fromkeys(_ENV_REF_PATTERN.findall(obj))
            if name not in os.environ
        ]
    if isinstance(obj, BaseModel):
        issues: list[ConfigIssue] = []
        for name, field in type(obj).model_fields.items():
            alias = field.serialization_alias or field.alias or name
            issues.extend(_missing_env_issues(getattr(obj, name), (*path, alias)))
        return issues
    if isinstance(obj, dict):
        issues = []
        for key, value in cast(dict[str, Any], obj).items():
            issues.extend(_missing_env_issues(value, (*path, key)))
        return issues
    if isinstance(obj, list):
        issues = []
        for index, value in enumerate(cast(list[Any], obj)):
            issues.extend(_missing_env_issues(value, (*path, index)))
        return issues
    return []


def _env_replace(match: re.Match[str]) -> str:
    name = match.group(1)
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"Environment variable '{name}' referenced in config is not set")
    return value


def apply_env_overrides(config: Config, env: Mapping[str, str] | None = None) -> Config:
    """Return a copy of ``config`` with the ``NANOCLAW_*`` variables applied.

    Empty variables are ignored. ``NANOCLAW_API_KEY`` / ``NANOCLAW_API_BASE``
    set the provider entry ``resolve_provider`` picks (after
    ``NANOCLAW_MODEL`` is applied), or a ``custom`` entry when none is
    configured. Raises ``ConfigEnvError`` for an unusable value.
    """
    env = os.environ if env is None else env
    config = config.model_copy(deep=True)
    defaults = config.agents.defaults

    def get(name: str) -> str:
        return env.get(name, "").strip()

    if model := get(ENV_MODEL):
        defaults.model = model
    if value := get(ENV_CONTEXT_WINDOW):
        try:
            defaults.context_window_tokens = int(value)
        except ValueError:
            raise ConfigEnvError(f"{ENV_CONTEXT_WINDOW} must be a whole number of tokens, got {value!r}") from None
    if value := get(ENV_COMPACTION).lower():
        modes = ("nanobot", "fixed")
        if value not in modes:
            raise ConfigEnvError(f"{ENV_COMPACTION} must be one of {', '.join(modes)}, got {value!r}")
        defaults.compaction = cast(Any, value)
    if value := get(ENV_RESTRICT).lower():
        config.tools.restrict_to_workspace = value in _TRUE_VALUES
    if value := get(ENV_SSRF_WHITELIST):
        config.tools.ssrf_whitelist = [part.strip() for part in value.split(",") if part.strip()]

    api_key, api_base = get(ENV_API_KEY), get(ENV_API_BASE)
    if api_key or api_base:
        key, entry = config.resolve_provider()
        if entry is None:
            # Nothing configured: fill the named provider, or "custom" under
            # "auto" (then the only configured entry, so "auto" picks it).
            key = defaults.provider if defaults.provider != "auto" else "custom"
            entry = config.providers.setdefault(key, ProviderConfig())
        if api_key:
            entry.api_key = api_key
        if api_base:
            entry.api_base = api_base
    return config


def load_runtime_config(config_path: Path | None = None, env: Mapping[str, str] | None = None) -> Config:
    """The settings a run uses: file (or defaults), ``${VAR}`` resolved, then env overrides."""
    path = config_path or get_config_path()
    return apply_env_overrides(resolve_config_env_vars(load_config(path), config_path=path), env)


_SECRET_KEYS = {"apikey"}


def mask_secrets(data: object) -> object:
    """Hide API keys in JSON data for display (``nanoclaw config show``).

    Keeps the last four characters of long keys, so two keys can be told apart.
    """
    if isinstance(data, dict):
        return {
            key: _mask(value) if key.replace("_", "").lower() in _SECRET_KEYS else mask_secrets(value)
            for key, value in cast(dict[str, object], data).items()
        }
    if isinstance(data, list):
        return [mask_secrets(value) for value in cast(list[object], data)]
    return data


def _mask(value: object) -> object:
    if not isinstance(value, str) or not value:
        return value
    return f"****{value[-4:]}" if len(value) >= 12 else "****"


def _sentence(message: str) -> str:
    message = message.strip()
    if message and message[-1] not in ".!?":
        message += "."
    return message
