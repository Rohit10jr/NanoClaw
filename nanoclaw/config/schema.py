"""The ``config.json`` schema: a subset of nanobot ``config/schema.py``.

Every field here has nanobot's JSON name, nesting and default, except where a
comment says otherwise. nanobot's other sections (gateway, api, modelPresets,
mcpServers, the per-channel entries under channels, ...) are not read: unknown
keys are ignored, so a nanobot ``config.json`` loads unchanged.

Layout (camelCase in JSON)::

    {
      "agents": {"defaults": {"workspace", "model", "provider", "maxTokens",
                              "contextWindowTokens", "temperature",
                              "maxToolIterations", "maxConcurrentSubagents",
                              "maxToolResultChars", "providerRetryMode",
                              "timezone", "timezoneMode",
                              "compaction"  # NanoClaw only
                              }},
      "channels": {"sendProgress", "sendToolHints", "showReasoning"},
      "providers": {"<name>": {"apiKey", "apiBase"}, ...},
      "tools": {"exec": {"enable", "maxRuntime", "maxRuntimeLimit",
                         "idleTimeout", "idleAction", "pathPrepend",
                         "pathAppend", "sandbox", "sandboxRoBinds",
                         "sandboxRwBinds", "allowedEnvKeys",
                         "allowPatterns", "denyPatterns"},
                "restrictToWorkspace", "ssrfWhitelist"}
    }

nanobot builds ``Config`` on pydantic-settings, so ``NANOBOT_AGENTS__DEFAULTS__MODEL``
style variables override the file (schema.py:757). NanoClaw keeps its own
``NANOCLAW_*`` variables instead (see ``nanoclaw/config/loader.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, Field, field_validator, model_validator

from nanoclaw.config.base import Base
from nanoclaw.config.timezone import detect_system_timezone

# nanobot's default is ~/.nanobot/workspace (config/home.py:27).
DEFAULT_WORKSPACE = "~/.nanoclaw/workspace"
# nanobot's default is "anthropic/claude-opus-4-5" (schema.py:172), which needs
# nanobot's provider registry; NanoClaw talks to an OpenAI-compatible endpoint,
# api.openai.com by default.
DEFAULT_MODEL = "gpt-4o-mini"
# nanobot's default is 200,000 (schema.py:177); NanoClaw's default model,
# gpt-4o-mini, has 128,000 (layer 6, difference #2).
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
# nanobot schema.py:180 (decision 7 of phase 2).
DEFAULT_MAX_TOOL_ITERATIONS = 200

# NanoClaw only: the compaction switch (docs/UPSTREAM_BUGS.md, bug 4). Kept in
# sync with nanoclaw/agent/memory.py COMPACTION_MODES (not imported, so the
# schema stays free of agent imports, like nanobot's).
CompactionMode = Literal["nanobot", "fixed"]

# OpenAI-compatible base URLs nanobot's provider registry fills in when a
# provider entry has no apiBase (providers/registry.py ``default_api_base``).
# Only a few common gateways; others need an explicit apiBase. ``openai`` and
# ``custom`` without an apiBase mean api.openai.com, as in NanoClaw before.
DEFAULT_API_BASES = {
    "openrouter": "https://openrouter.ai/api/v1",  # registry.py:280
    "deepseek": "https://api.deepseek.com",  # registry.py:643
    "aihubmix": "https://aihubmix.com/v1",  # registry.py:396
    "siliconflow": "https://api.siliconflow.cn/v1",  # registry.py:409
    "huggingface": "https://router.huggingface.co/v1",  # registry.py:368
}


def _provider_key(name: str) -> str:
    """``lm_studio``, ``lmStudio`` and ``lm-studio`` name the same provider."""
    return name.replace("-", "").replace("_", "").lower()


class AgentDefaults(Base):
    """Default agent configuration (nanobot schema.py:167 ``AgentDefaults``, a subset)."""

    workspace: str = DEFAULT_WORKSPACE
    model: str = DEFAULT_MODEL
    provider: str = "auto"  # a key of ``providers``, or "auto" (schema.py:173)
    max_tokens: int = 8192
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    temperature: float = 0.1
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS
    max_concurrent_subagents: int = Field(default=4, ge=1)
    max_tool_result_chars: int = 16_000
    # Read by provider retries (schema.py:183); stored here, used once retries land.
    provider_retry_mode: Literal["standard", "persistent"] = "standard"
    # Stored for the time block; NanoClaw's time block still uses the host clock.
    timezone: str = "UTC"
    timezone_mode: Literal["auto", "manual"] = "auto"
    # NanoClaw only (not in nanobot): "nanobot" (default, like upstream) or
    # "fixed" (docs/UPSTREAM_BUGS.md, bug 4). nanobot ignores the key.
    compaction: CompactionMode = "nanobot"

    @model_validator(mode="before")
    @classmethod
    def resolve_timezone(cls, value: object) -> object:
        """Detect new defaults while preserving configured timezones (nanobot schema.py:211)."""
        if not isinstance(value, dict):
            return value
        data = dict(cast(dict[str, object], value))
        timezone_mode = data.get("timezoneMode", data.get("timezone_mode"))
        if timezone_mode is None:
            timezone_mode = "manual" if "timezone" in data else "auto"
            data["timezoneMode"] = timezone_mode
        if timezone_mode == "auto":
            data["timezone"] = detect_system_timezone()
        return data

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        # nanobot schema.py:227. "UTC" is always accepted: Windows has no tz
        # database unless the tzdata package is installed.
        if value == "UTC":
            return value
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown timezone {value!r}") from None
        return value


class AgentsConfig(Base):
    """Agent configuration (nanobot schema.py:239)."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(Base):
    """One provider entry (nanobot schema.py:245 ``ProviderConfig``, the two
    connection fields; extraHeaders, extraBody, ... are ignored)."""

    api_key: str | None = Field(default=None, repr=False)
    api_base: str | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key or self.api_base)


def _default_providers() -> dict[str, ProviderConfig]:
    # An empty "custom" entry (nanobot's "any OpenAI-compatible endpoint",
    # schema.py:301) so ``nanoclaw config init`` writes something to fill in.
    return {"custom": ProviderConfig()}


class ChannelsConfig(Base):
    """What the terminal shows while a turn works (nanobot schema.py:33 ``ChannelsConfig``, a subset).

    nanobot's CLI reads these three (cli/agent.py:222-245); the per-channel
    entries (telegram, slack, ...) and delivery settings are not read.
    """

    send_progress: bool = True  # schema.py:43, the model's text before tool calls
    send_tool_hints: bool = True  # schema.py:44, e.g. read notes.txt
    show_reasoning: bool = True  # schema.py:45, the model's reasoning


class ExecToolConfig(Base):
    """Shell ``exec`` tool settings: nanobot agent/tools/shell_config.py ``ExecToolConfig``.

    Same fields, JSON names, defaults and validation as nanobot. ``timeout`` is
    the legacy spelling of ``maxRuntime``: it is read, never written, and giving
    both is an error.

    ``sandbox``, ``sandboxRoBinds`` and ``sandboxRwBinds`` are read and kept but
    unused until layer 18 (the bubblewrap sandbox).
    """

    enable: bool = True
    max_runtime: int = Field(
        default=3600,
        ge=0,
        validation_alias=AliasChoices("maxRuntime", "max_runtime", "timeout"),
    )
    max_runtime_limit: int = Field(default=86400, gt=0)
    idle_timeout: int = Field(default=300, ge=0)
    idle_action: Literal["report", "terminate"] = "report"
    path_prepend: str = ""
    path_append: str = ""
    sandbox: str = ""  # unused until layer 18
    sandbox_ro_binds: list[str] = Field(default_factory=list)  # unused until layer 18
    sandbox_rw_binds: list[str] = Field(default_factory=list)  # unused until layer 18
    allowed_env_keys: list[str] = Field(default_factory=list)
    allow_patterns: list[str] = Field(default_factory=list)
    deny_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_runtime_alias_conflict(cls, value: object) -> object:
        if isinstance(value, Mapping):
            value = cast(Mapping[object, object], value)
            if "timeout" in value and ("maxRuntime" in value or "max_runtime" in value):
                raise ValueError("Use maxRuntime or legacy timeout, not both")
        return value

    @property
    def timeout(self) -> int:
        """Legacy hard-runtime setting backed by ``max_runtime``."""
        return self.max_runtime

    @timeout.setter
    def timeout(self, value: int) -> None:
        self.max_runtime = value


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP). nanobot schema.py:414.

    ``auth: "oauth"`` and ``useProxy`` are accepted so nanobot configs load,
    but NanoClaw skips OAuth servers and ignores ``useProxy`` (environment
    proxies still apply); see agent/tools/mcp.py.
    """

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    auth: Literal["oauth"] | None = None  # Remote MCP OAuth (not supported by NanoClaw)
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    cwd: str = ""  # Stdio: working directory for MCP server runtime artifacts
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    use_proxy: bool = False  # nanobot's shared proxy; ignored by NanoClaw
    tool_timeout: int = 30  # seconds before a tool call is cancelled
    # Only register these tools; accepts raw MCP names or wrapped
    # mcp_<server>_<tool> names; ["*"] = all, [] = none.
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])


class ToolsConfig(Base):
    """Tool settings (nanobot schema.py:430 ``ToolsConfig``, a subset)."""

    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)  # schema.py:434
    restrict_to_workspace: bool = False  # schema.py:442
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)  # schema.py:459
    ssrf_whitelist: list[str] = Field(default_factory=list)  # schema.py:460, CIDR ranges


class Config(Base):
    """Root configuration (nanobot schema.py:501, a subset)."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)  # schema.py:507
    # nanobot declares ~40 named providers plus extra keys for custom ones
    # (schema.py:292). NanoClaw has one OpenAI-compatible provider, so any key
    # is accepted and the active one is picked by ``resolve_provider``.
    providers: dict[str, ProviderConfig] = Field(default_factory=_default_providers)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """Expanded workspace path (nanobot schema.py:584)."""
        return Path(self.agents.defaults.workspace).expanduser()

    def find_provider(self, name: str) -> tuple[str, ProviderConfig] | None:
        """The entry whose key names ``name`` (ignoring case, ``-`` and ``_``)."""
        wanted = _provider_key(name)
        for key, entry in self.providers.items():
            if _provider_key(key) == wanted:
                return key, entry
        return None

    def resolve_provider(self) -> tuple[str | None, ProviderConfig | None]:
        """Pick the provider entry NanoClaw uses. Returns ``(key, entry)``.

        A simplified nanobot ``_match_provider`` (schema.py:588):

        1. ``agents.defaults.provider`` names an entry: that entry (configured
           or not, as nanobot).
        2. ``"auto"``: the entry named by the model's prefix
           (``openrouter/...``), if it has an apiKey or apiBase.
        3. ``"auto"``: the first entry, in file order, with an apiKey or
           apiBase. (nanobot uses its registry order, gateways first, and
           keyword matching; NanoClaw has no registry.)
        """
        forced = self.agents.defaults.provider
        if forced != "auto":
            found = self.find_provider(forced)
            return found if found else (None, None)
        model = self.agents.defaults.model
        if "/" in model:
            found = self.find_provider(model.split("/", 1)[0])
            if found and found[1].is_configured:
                return found
        for key, entry in self.providers.items():
            if entry.is_configured:
                return key, entry
        return None, None

    def get_api_key(self) -> str | None:
        _, entry = self.resolve_provider()
        return (entry.api_key or None) if entry else None

    def get_api_base(self) -> str | None:
        """The entry's apiBase, else nanobot's default base for that provider name."""
        key, entry = self.resolve_provider()
        if entry and entry.api_base:
            return entry.api_base
        return DEFAULT_API_BASES.get(_provider_key(key)) if key else None
