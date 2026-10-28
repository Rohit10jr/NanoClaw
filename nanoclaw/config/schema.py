"""The ``config.json`` schema: a subset of nanobot ``config/schema.py``.

Every field here has nanobot's JSON name, nesting and default, except where a
comment says otherwise. nanobot's other sections (api, the
per-channel entries under channels, ...) are not read: unknown keys are
ignored, so a nanobot ``config.json`` loads unchanged.

Layout (camelCase in JSON)::

    {
      "agents": {"defaults": {"workspace", "modelPreset", "model", "provider",
                              "maxTokens", "contextWindowTokens", "temperature",
                              "reasoningEffort", "fallbackModels",
                              "dream": {"enabled", "intervalH", "cron",
                                        "modelOverride"},
                              "maxToolIterations", "maxConcurrentSubagents",
                              "maxToolResultChars", "providerRetryMode",
                              "timezone", "timezoneMode",
                              "idleCompactAfterMinutes",
                              "idleCompactCheckIntervalSeconds",
                              "compaction"  # NanoClaw only
                              }},
      "channels": {"sendProgress", "sendToolHints", "showReasoning",
                   "sendMaxRetries",
                   "websocket": {"enabled", "host", "port", "path",
                                 "publicWsUrl", "token", "tokenIssuePath",
                                 "tokenIssueSecret", "tokenTtlS",
                                 "websocketRequiresToken", "allowFrom",
                                 "streaming", "maxMessageBytes",
                                 "pingIntervalS", "pingTimeoutS",
                                 "sslCertfile", "sslKeyfile",
                                 "sendProgress", "sendToolHints",
                                 "showReasoning",
                                 "webuiDist"  # NanoClaw only
                                 },
                   "showCompactionNotices",
                   "telegram": {"enabled", "token", "mode", "allowFrom",
                                "useProxy", "replyToMessage", "reactEmoji",
                                "groupPolicy", "connectionPoolSize",
                                "poolTimeout", "streaming",
                                "inlineKeyboards", "richMessages",
                                "streamEditInterval", "webhookUrl",
                                "webhookListenHost", "webhookListenPort",
                                "webhookPath", "webhookSecretToken",
                                "webhookMaxConnections", "sendProgress",
                                "sendToolHints", "showReasoning",
                                "showCompactionNotices"}},
      "gateway": {"host", "port", "heartbeat": {"enabled", "intervalS"}},
      "providers": {"<name>": {"apiKey", "apiBase", "extraHeaders",
                               "extraBody", "extraQuery", "thinkingStyle"}, ...},
      "modelPresets": {"<name>": {"model", "provider", "maxTokens",
                                  "contextWindowTokens", "temperature",
                                  "reasoningEffort", "extraHeaders",
                                  "extraQuery", "extraBody"}, ...},
      "tools": {"exec": {"enable", "maxRuntime", "maxRuntimeLimit",
                         "idleTimeout", "idleAction", "pathPrepend",
                         "pathAppend", "sandbox", "sandboxRoBinds",
                         "sandboxRwBinds", "allowedEnvKeys",
                         "allowPatterns", "denyPatterns"},
                "my": {"enable", "allowSet"},
                "restrictToWorkspace", "ssrfWhitelist"}
    }

nanobot builds ``Config`` on pydantic-settings, so ``NANOBOT_AGENTS__DEFAULTS__MODEL``
style variables override the file (schema.py:757). NanoClaw keeps its own
``NANOCLAW_*`` variables instead (see ``nanoclaw/config/loader.py``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from nanoclaw.config.base import Base
from nanoclaw.config.timezone import detect_system_timezone

if TYPE_CHECKING:
    from nanoclaw.cron.types import CronSchedule
    from nanoclaw.providers.registry import ProviderSpec

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


class DreamConfig(Base):
    """Dream settings (nanobot schema.py:64 ``DreamConfig``): the schedule of
    the gateway's protected ``dream`` system job and the model preset.

    ``enabled`` registers the job when ``nanoclaw gateway`` starts (default
    every ``intervalH`` = 2 hours; a legacy ``cron`` expression wins over the
    interval). ``nanoclaw dream`` and ``/dream`` run Dream on request either way.
    """

    _HOUR_MS: ClassVar[int] = 3_600_000

    enabled: bool = True  # Register the periodic Dream consolidation job on startup
    interval_h: int = Field(default=2, ge=1)  # Every 2 hours by default
    cron: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )  # Legacy cron expression override
    model_override: str | None = Field(
        default=None,
        validation_alias=AliasChoices("modelOverride", "model", "model_override"),
    )  # Model preset name for Dream runs

    def build_schedule(self, timezone: str) -> CronSchedule:
        """Build the runtime schedule, preferring the legacy cron override if present."""
        from nanoclaw.cron.types import CronSchedule

        if self.cron:
            return CronSchedule(kind="cron", expr=self.cron, tz=timezone)
        return CronSchedule(kind="every", every_ms=self.interval_h * self._HOUR_MS)

    def describe_schedule(self) -> str:
        """Return a human-readable summary for logs and startup output."""
        if self.cron:
            return f"cron {self.cron} (legacy)"
        hours = self.interval_h
        return f"every {hours}h"


class RequestOverrides(Base):
    """Optional request differences; omitted fields inherit connection defaults (schema.py:128)."""

    extra_headers: dict[str, str] | None = Field(default=None, repr=False)
    extra_query: dict[str, str] | None = Field(default=None, repr=False)
    extra_body: dict[str, Any] | None = Field(default=None, repr=False)


class InlineFallbackConfig(RequestOverrides):
    """One inline fallback model configuration (nanobot schema.py:133).

    nanobot's ``api`` (request-API selection) is not read, as for presets.
    Omitted ``maxTokens``, ``contextWindowTokens`` and ``temperature`` come
    from the primary preset; ``reasoningEffort`` does not.
    """

    model: str
    provider: str
    max_tokens: int | None = None
    context_window_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None


# A ``modelPresets`` name or an inline model (nanobot schema.py:145).
FallbackCandidate = str | InlineFallbackConfig


class ModelPresetConfig(RequestOverrides):
    """A named set of model + generation parameters for quick switching (schema.py:148).

    nanobot's ``api`` (request-API selection) is not read. The default
    window is NanoClaw's 128,000 (nanobot: 200,000; difference #1 of layer 12).
    """

    model: str
    provider: str = "auto"
    max_tokens: int = 8192
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    temperature: float = 0.1
    reasoning_effort: str | None = None


class AgentDefaults(Base):
    """Default agent configuration (nanobot schema.py:167 ``AgentDefaults``, a subset)."""

    workspace: str = DEFAULT_WORKSPACE
    model_preset: str | None = None  # Active preset name; takes precedence over the fields below (schema.py:171)
    model: str = DEFAULT_MODEL
    provider: str = "auto"  # a provider name ("deepseek", a ``providers`` key), or "auto" (schema.py:173)
    max_tokens: int = 8192
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    temperature: float = 0.1
    # Tried in order when the active model fails (schema.py:179; nanoclaw/providers/fallback_provider.py).
    fallback_models: list[FallbackCandidate] = Field(default_factory=list)
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS
    max_concurrent_subagents: int = Field(default=4, ge=1)
    max_tool_result_chars: int = 16_000
    # Read by provider retries (schema.py:183); stored here, used once retries land.
    provider_retry_mode: Literal["standard", "persistent"] = "standard"
    # low / medium / high / ... / none; None keeps the provider's default (schema.py:191).
    reasoning_effort: str | None = None
    # Stored for the time block; NanoClaw's time block still uses the host clock.
    timezone: str = "UTC"
    timezone_mode: Literal["auto", "manual"] = "auto"
    # NanoClaw only (not in nanobot): "nanobot" (default, like upstream) or
    # "fixed" (docs/UPSTREAM_BUGS.md, bug 4). nanobot ignores the key.
    compaction: CompactionMode = "nanobot"
    # Idle auto-compaction threshold in minutes, 0 = off (schema.py:199;
    # JSON ``idleCompactAfterMinutes``, legacy ``sessionTtlMinutes``).
    session_ttl_minutes: int = Field(
        default=15,
        ge=0,
        validation_alias=AliasChoices("idleCompactAfterMinutes", "sessionTtlMinutes", "session_ttl_minutes"),
        serialization_alias="idleCompactAfterMinutes",
    )
    # Minimum seconds between scans for idle sessions (schema.py:205).
    idle_compact_check_interval_seconds: int = Field(default=60, ge=0)
    dream: DreamConfig = Field(default_factory=DreamConfig)  # schema.py:212

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
    """One provider entry (nanobot schema.py:245 ``ProviderConfig``).

    nanobot's ``api`` (request-API declarations) and ``useProxy`` are not read.
    """

    display_name: str | None = Field(default=None, exclude_if=lambda value: value is None)
    api_key: str | None = Field(default=None, repr=False)
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)
    extra_body: dict[str, Any] | None = None  # Extra request fields, merged into the request body
    extra_query: dict[str, str] | None = None  # Extra query params (e.g. api-version)
    thinking_style: str | None = None  # Thinking/reasoning style for custom providers

    # nanobot schema.py:268: the keys of _THINKING_STYLE_MAP in the
    # OpenAI-compatible provider (not imported, to keep the schema free of
    # provider imports).
    _VALID_THINKING_STYLES: ClassVar[tuple[str, ...]] = ("thinking_type", "enable_thinking", "reasoning_split")

    @field_validator("thinking_style")
    @classmethod
    def _validate_thinking_style(cls, v: str | None) -> str | None:
        if not v:  # None or "" -> no injection
            return v
        if v not in cls._VALID_THINKING_STYLES:
            raise ValueError(
                f"Invalid thinking_style {v!r}. "
                f"Must be one of: {', '.join(repr(s) for s in cls._VALID_THINKING_STYLES)} "
                f"(or empty/omitted)."
            )
        return v


def _default_providers() -> dict[str, ProviderConfig]:
    # An empty "custom" entry (nanobot's "any OpenAI-compatible endpoint",
    # schema.py:301) so ``nanoclaw config init`` writes something to fill in.
    return {"custom": ProviderConfig()}


def _normalize_config_path(path: str) -> str:
    """nanobot webui/http_utils.py ``normalize_config_path``: no trailing slash."""
    if len(path) > 1 and path.endswith("/"):
        return path.rstrip("/") or "/"
    return path or "/"


class WebSocketConfig(Base):
    """``channels.websocket``: the web UI server (nanobot
    channels/websocket/runtime.py:180 ``WebSocketConfig``, a subset).

    One listener serves the built web UI, ``GET /webui/bootstrap``, the
    ``/api/*`` routes and the WebSocket (upgrade on ``path``). Without a
    ``token``/``tokenIssueSecret`` only loopback browsers may bootstrap; with
    one, a remote browser sends it as ``X-Nanobot-Auth`` (or
    ``Authorization: Bearer``).

    Differences: ``enabled`` defaults to False (nanobot enables the channel
    by default; NanoClaw ships no frontend build, so the web UI is opt-in);
    ``webuiDist`` is NanoClaw's (nanobot serves the build bundled in its
    package, ``nanobot/web/dist``); ``unixSocketPath`` and
    ``trustedProxyAuth`` are not supported. The per-channel
    ``sendProgress`` / ``sendToolHints`` / ``showReasoning`` overrides are
    nanobot's (channels/manager.py ``_resolve_bool_override``).
    """

    enabled: bool = False  # nanobot: True
    host: str = "127.0.0.1"
    port: int = 8765
    path: str = "/"
    public_ws_url: str = ""
    token: str = ""
    token_issue_path: str = ""
    token_issue_secret: str = ""
    token_ttl_s: int = Field(default=300, ge=30, le=86_400)
    websocket_requires_token: bool = True
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    streaming: bool = True
    max_message_bytes: int = Field(default=37_748_736, ge=1024, le=41_943_040)
    ping_interval_s: float = Field(default=20.0, ge=5.0, le=300.0)
    ping_timeout_s: float = Field(default=20.0, ge=5.0, le=300.0)
    ssl_certfile: str = ""
    ssl_keyfile: str = ""
    send_progress: bool | None = None
    send_tool_hints: bool | None = None
    show_reasoning: bool | None = None
    # NanoClaw only: the directory holding the built frontend (index.html and
    # assets/); empty: ``NANOCLAW_WEBUI_DIST``, else no static files.
    webui_dist: str = ""

    @field_validator("path")
    @classmethod
    def _path_must_start_with_slash(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError('path must start with "/"')
        return _normalize_config_path(value)

    @field_validator("token_issue_path")
    @classmethod
    def _token_issue_path_format(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        if not value.startswith("/"):
            raise ValueError('token_issue_path must start with "/"')
        return _normalize_config_path(value)

    @field_validator("public_ws_url")
    @classmethod
    def _public_ws_url_format(cls, value: str) -> str:
        from urllib.parse import urlsplit, urlunsplit

        value = value.strip()
        if not value:
            return ""
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"ws", "wss"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("public_ws_url must be an absolute ws:// or wss:// URL without credentials")
        return urlunsplit((parsed.scheme, parsed.netloc, _normalize_config_path(parsed.path or "/"), "", ""))

    @model_validator(mode="after")
    def _check_paths_and_auth(self) -> WebSocketConfig:
        from urllib.parse import urlsplit

        if self.public_ws_url and urlsplit(self.public_ws_url).path != self.path:
            raise ValueError("public_ws_url path must match path")
        if self.token_issue_path and self.token_issue_path == self.path:
            raise ValueError("token_issue_path must differ from path (the WebSocket upgrade path)")
        if self.host in ("0.0.0.0", "::") and not (self.token.strip() or self.token_issue_secret.strip()):
            raise ValueError(
                "host is 0.0.0.0 (all interfaces) but neither token nor token_issue_secret "
                "is set — set one to prevent unauthenticated access"
            )
        return self


class TelegramConfig(Base):
    """``channels.telegram``: the Telegram bot (nanobot
    channels/telegram/runtime.py:491 ``TelegramConfig``; same names, defaults
    and checks).

    ``useProxy`` is accepted and ignored: nanobot routes the bot through its
    shared ``network.proxy``, which NanoClaw does not have (like
    ``mcpServers.*.useProxy``); the standard ``HTTPS_PROXY`` / ``ALL_PROXY``
    variables apply. The per-channel ``sendProgress`` / ``sendToolHints`` /
    ``showReasoning`` / ``showCompactionNotices`` overrides are nanobot's
    (channels/manager.py ``_resolve_bool_override``).
    """

    enabled: bool = False
    token: str = ""
    mode: Literal["polling", "webhook"] = "polling"
    allow_from: list[str] = Field(default_factory=list)
    use_proxy: bool = False
    reply_to_message: bool = False
    react_emoji: str = "👀"
    group_policy: Literal["open", "mention"] = "mention"
    connection_pool_size: int = 32
    pool_timeout: float = 5.0
    streaming: bool = True
    inline_keyboards: bool = False
    rich_messages: bool = False
    stream_edit_interval: float = Field(default=0.6, ge=0.1)
    webhook_url: str = ""
    webhook_listen_host: str = "127.0.0.1"
    webhook_listen_port: int = Field(default=8081, ge=1, le=65535)
    webhook_path: str = "/telegram"
    webhook_secret_token: str = ""
    webhook_max_connections: int = Field(default=4, ge=1, le=100)
    send_progress: bool | None = None
    send_tool_hints: bool | None = None
    show_reasoning: bool | None = None
    show_compaction_notices: bool | None = None

    @field_validator("webhook_path")
    @classmethod
    def webhook_path_must_start_with_slash(cls, value: str) -> str:
        value = value.strip() or "/telegram"
        if not value.startswith("/"):
            raise ValueError('webhook_path must start with "/"')
        return value

    @model_validator(mode="after")
    def validate_webhook_config(self) -> TelegramConfig:
        if self.mode != "webhook":
            return self
        url = self.webhook_url.strip()
        if not url:
            raise ValueError("webhook_url is required when Telegram mode is webhook")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("webhook_url must be a public HTTPS URL")
        secret = self.webhook_secret_token.strip()
        if not secret:
            raise ValueError("webhook_secret_token is required when Telegram mode is webhook")
        if len(secret) > 256 or re.match(r"^[A-Za-z0-9_-]+$", secret) is None:
            raise ValueError("webhook_secret_token must be 1-256 characters using only A-Z, a-z, 0-9, _ and -")
        return self


class ChannelsConfig(Base):
    """Chat channel settings (nanobot schema.py:33 ``ChannelsConfig``, a subset).

    The first three also decide what the terminal shows while a turn works
    (nanobot cli/agent.py:222-245). ``websocket`` is the web UI server and
    ``telegram`` the Telegram bot (nanobot keeps both as extra keys the
    channel parses). Other channels (slack, discord, ...) are not read.
    """

    send_progress: bool = True  # schema.py:43, the model's text before tool calls
    send_tool_hints: bool = True  # schema.py:44, e.g. read notes.txt
    show_reasoning: bool = True  # schema.py:45, the model's reasoning
    show_compaction_notices: bool = False  # schema.py:46, automatic compaction chat notices
    send_max_retries: int = Field(default=3, ge=0, le=10)  # schema.py:48, delivery attempts
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


class ExecToolConfig(Base):
    """Shell ``exec`` tool settings: nanobot agent/tools/shell_config.py ``ExecToolConfig``.

    Same fields, JSON names, defaults and validation as nanobot. ``timeout`` is
    the legacy spelling of ``maxRuntime``: it is read, never written, and giving
    both is an error.

    ``sandbox`` is ``""`` (none) or ``"bwrap"`` (layer 18, ``agent/tools/sandbox.py``);
    nanobot also accepts ``"seatbelt"`` (macOS), which NanoClaw leaves out.
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
    sandbox: str = ""
    sandbox_ro_binds: list[str] = Field(default_factory=list)
    sandbox_rw_binds: list[str] = Field(default_factory=list)
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


class MyToolConfig(Base):
    """The ``my`` tool (nanobot agent/tools/self_config.py ``MyToolConfig``).

    ``enable`` registers the tool for the main agent; ``allow_set`` lets it
    change runtime settings (off: ``check`` only).
    """

    enable: bool = True
    allow_set: bool = False


class ToolsConfig(Base):
    """Tool settings (nanobot schema.py:430 ``ToolsConfig``, a subset)."""

    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)  # schema.py:434
    my: MyToolConfig = Field(default_factory=MyToolConfig)  # schema.py:437
    restrict_to_workspace: bool = False  # schema.py:442
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)  # schema.py:459
    ssrf_whitelist: list[str] = Field(default_factory=list)  # schema.py:460, CIDR ranges


class HeartbeatConfig(Base):
    """Heartbeat settings (nanobot schema.py:378): the gateway's protected
    ``heartbeat`` system job, every ``intervalS`` seconds."""

    enabled: bool = True
    interval_s: int = 30 * 60  # 30 minutes


class GatewayConfig(Base):
    """``nanoclaw gateway`` settings (nanobot schema.py:405 ``GatewayConfig``,
    without ``restartMode``: NanoClaw has no ``/restart``)."""

    host: str = "127.0.0.1"  # local-only bind by default
    port: int = 18790  # the /health endpoint
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class Config(Base):
    """Root configuration (nanobot schema.py:501, a subset)."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)  # schema.py:507
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)  # schema.py:512
    # nanobot declares a field per registry provider plus extra keys for custom
    # ones (schema.py:292). NanoClaw keeps one dict: a key naming a registry
    # provider (``deepseek``, ``lmStudio``) configures it, any other key is a
    # custom OpenAI-compatible provider (nanobot's ``model_extra``).
    providers: dict[str, ProviderConfig] = Field(default_factory=_default_providers)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    model_presets: dict[str, ModelPresetConfig] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("modelPresets", "model_presets"),
        serialization_alias="modelPresets",
    )  # schema.py:514

    @field_validator("providers")
    @classmethod
    def _reject_builtin_aliases(cls, providers: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        """A key spelling a registry name another way is an error (nanobot schema.py:349)."""
        from nanoclaw.providers.registry import find_by_name

        for key in providers:
            spec = find_by_name(key)
            if spec is not None and key not in (spec.name, to_camel(spec.name)):
                raise ValueError(
                    f"providers.{key} conflicts with built-in provider {spec.name!r}; "
                    "use the built-in provider key or choose a different custom provider name"
                )
        return providers

    @model_validator(mode="after")
    def _validate_model_preset(self) -> Config:
        """nanobot schema.py:550."""
        if "default" in self.model_presets:
            raise ValueError("model_preset name 'default' is reserved for agents.defaults")
        name = self.agents.defaults.model_preset
        if name and name != "default" and name not in self.model_presets:
            raise ValueError(f"model_preset {name!r} not found in model_presets")
        dream_name = self.agents.defaults.dream.model_override
        if dream_name and dream_name != "default" and dream_name not in self.model_presets:
            raise ValueError(f"Dream model preset {dream_name!r} not found in model_presets")
        for fallback in self.agents.defaults.fallback_models:
            if isinstance(fallback, str) and fallback not in self.model_presets:
                raise ValueError(f"fallback_models entry {fallback!r} not found in model_presets")
        return self

    def resolve_default_preset(self) -> ModelPresetConfig:
        """The implicit ``default`` preset from agents.defaults fields (schema.py:567)."""
        d = self.agents.defaults
        return ModelPresetConfig(
            model=d.model,
            provider=d.provider,
            max_tokens=d.max_tokens,
            context_window_tokens=d.context_window_tokens,
            temperature=d.temperature,
            reasoning_effort=d.reasoning_effort,
        )

    def resolve_preset(self, name: str | None = None) -> ModelPresetConfig:
        """Effective model params from a named preset or the implicit default (schema.py:576)."""
        name = self.agents.defaults.model_preset if name is None else name
        if not name or name == "default":
            return self.resolve_default_preset()
        if name not in self.model_presets:
            raise KeyError(f"model_preset {name!r} not found in model_presets")
        return self.model_presets[name]

    @property
    def workspace_path(self) -> Path:
        """Expanded workspace path (nanobot schema.py:584)."""
        return Path(self.agents.defaults.workspace).expanduser()

    # -- provider choice (nanobot schema.py:588-756) ------------------------

    def provider_entry(self, spec_name: str) -> tuple[str, ProviderConfig] | None:
        """The ``providers`` entry for a registry provider: key ``lm_studio`` or ``lmStudio``."""
        for key in (spec_name, to_camel(spec_name)):
            if key in self.providers:
                return key, self.providers[key]
        return None

    def custom_providers(self) -> dict[str, ProviderConfig]:
        """The entries that name no registry provider (nanobot ``providers.model_extra``)."""
        from nanoclaw.providers.registry import find_by_name

        return {key: entry for key, entry in self.providers.items() if find_by_name(key) is None}

    def _spec_provider(self, spec: ProviderSpec) -> ProviderConfig:
        # nanobot has a field (an empty ProviderConfig by default) for every
        # registry provider; a missing key here reads the same way.
        found = self.provider_entry(spec.name)
        return found[1] if found else ProviderConfig()

    def _match_provider(
        self,
        model: str | None = None,
        *,
        preset: ModelPresetConfig | None = None,
    ) -> tuple[ProviderConfig | None, str | None]:
        """Match the provider config and its name. Returns ``(config, name)`` (schema.py:588).

        1. A preset/default ``provider`` other than ``auto`` wins (a registry
           name, else a custom key).
        2. The model's prefix names a registry provider with a key (local
           and direct providers need none), or a custom provider.
        3. A model-name keyword (``claude``, ``deepseek``, ...), in registry order.
        4. A configured local provider (apiBase set), preferring one whose
           base matches its port keyword (``11434`` = Ollama).
        5. The first registry provider with a key, gateways first.
        6. The first custom provider with an apiBase.
        """
        from nanoclaw.providers.registry import PROVIDERS, find_by_name

        resolved = preset or self.resolve_preset()
        forced = resolved.provider
        custom = self.custom_providers()

        def _custom_provider_by_name(name: str) -> tuple[ProviderConfig, str] | None:
            normalized = name.replace("-", "_").lower()
            for attr_name, provider in custom.items():
                if attr_name.replace("-", "_").lower() == normalized:
                    return provider, attr_name
            return None

        if forced != "auto":
            spec = find_by_name(forced)
            if spec:
                return self._spec_provider(spec), spec.name
            found = _custom_provider_by_name(forced)
            if found is not None:
                return found
            return None, None

        model_lower = (model or resolved.model).lower()
        model_normalized = model_lower.replace("-", "_")
        model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
        normalized_prefix = model_prefix.replace("-", "_")
        prefixed_provider = find_by_name(model_prefix) if model_prefix else None

        def _kw_matches(kw: str) -> bool:
            kw = kw.lower()
            return kw in model_lower or kw.replace("-", "_") in model_normalized

        # Explicit provider prefix wins.
        for spec in PROVIDERS:
            p = self._spec_provider(spec)
            if model_prefix and normalized_prefix == spec.name:
                if spec.is_local or spec.is_direct or p.api_key:
                    return p, spec.name

        # A custom provider named by the prefix (e.g. "companyProxy/gpt-4"),
        # even without an apiBase, so a malformed prefix fails instead of
        # falling through to a different provider.
        if model_prefix:
            found = _custom_provider_by_name(normalized_prefix)
            if found is not None:
                return found

        # Match by keyword (registry order).
        for spec in PROVIDERS:
            p = self._spec_provider(spec)
            if any(_kw_matches(kw) for kw in spec.keywords):
                # A local keyword counts only for a configured local endpoint,
                # and never for a model qualified with another provider.
                if spec.is_local:
                    foreign_prefix = bool(prefixed_provider is not None and prefixed_provider.name != spec.name)
                    if not p.api_base or foreign_prefix:
                        continue
                if spec.is_local or spec.is_direct or p.api_key:
                    return p, spec.name

        # Configured local providers route models without keywords (plain
        # "llama3.2" on Ollama), preferring a base that matches the port keyword.
        local_fallback: tuple[ProviderConfig, str] | None = None
        if prefixed_provider is None:
            for spec in PROVIDERS:
                if not spec.is_local:
                    continue
                p = self._spec_provider(spec)
                if not p.api_base:
                    continue
                if spec.detect_by_base_keyword and spec.detect_by_base_keyword in p.api_base:
                    return p, spec.name
                if local_fallback is None:
                    local_fallback = (p, spec.name)
        if local_fallback:
            return local_fallback

        # Gateways first, then the others (registry order).
        for spec in PROVIDERS:
            p = self._spec_provider(spec)
            if p.api_key:
                return p, spec.name

        # Any custom provider with an apiBase.
        for attr_name, p in custom.items():
            if p.api_base:
                return p, attr_name

        return None, None

    def get_provider(self, model: str | None = None, *, preset: ModelPresetConfig | None = None) -> ProviderConfig | None:
        """The matched provider config (schema.py:722)."""
        p, _ = self._match_provider(model, preset=preset)
        return p

    def get_provider_name(self, model: str | None = None, *, preset: ModelPresetConfig | None = None) -> str | None:
        """The registry name (or custom key) of the matched provider (schema.py:732)."""
        _, name = self._match_provider(model, preset=preset)
        return name

    def get_api_key(self, model: str | None = None, *, preset: ModelPresetConfig | None = None) -> str | None:
        """The matched provider's API key (schema.py:742)."""
        p = self.get_provider(model, preset=preset)
        return p.api_key if p else None

    def get_api_base(self, model: str | None = None, *, preset: ModelPresetConfig | None = None) -> str | None:
        """The matched provider's apiBase, else the registry default (schema.py:752)."""
        from nanoclaw.providers.registry import find_by_name

        p, name = self._match_provider(model, preset=preset)
        if p and p.api_base:
            return p.api_base
        if name:
            spec = find_by_name(name)
            if spec and spec.default_api_base:
                return spec.default_api_base
        return None

    def resolve_provider(self) -> tuple[str | None, ProviderConfig | None]:
        """``(name, entry)`` of the provider the active preset uses (NanoClaw's
        shorthand for ``get_provider_name()`` and ``get_provider()``)."""
        p, name = self._match_provider()
        return name, p
