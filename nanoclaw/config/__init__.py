"""NanoClaw's ``config.json``: a nanobot-compatible subset (nanobot ``config/``).

See ``schema.py`` for the fields and ``loader.py`` for where the file lives
and how environment variables override it.
"""

from nanoclaw.config.base import Base
from nanoclaw.config.errors import ConfigIssue, ConfigLoadError
from nanoclaw.config.loader import (
    ConfigEnvError,
    apply_env_overrides,
    get_config_path,
    get_data_dir,
    load_config,
    load_runtime_config,
    resolve_config_env_vars,
    save_config,
    set_config_path,
)
from nanoclaw.config.schema import AgentDefaults, Config, ProviderConfig, ToolsConfig

__all__ = [
    "AgentDefaults",
    "Base",
    "Config",
    "ConfigEnvError",
    "ConfigIssue",
    "ConfigLoadError",
    "ProviderConfig",
    "ToolsConfig",
    "apply_env_overrides",
    "get_config_path",
    "get_data_dir",
    "load_config",
    "load_runtime_config",
    "resolve_config_env_vars",
    "save_config",
    "set_config_path",
]
