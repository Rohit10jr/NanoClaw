# 12 — Foundations: config file, logging, retries, token usage

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

Four things every later phase 2 layer relies on: one place for settings, logs
to see what happens inside, model requests that survive rate limits and flaky
servers, and token counts that are always there.

## Config file

`~/.nanoclaw/config.json` (another file with `-c FILE` or `NANOCLAW_CONFIG`).
It is a **subset of nanobot's `config.json`**: same JSON names (camelCase),
same nesting, same defaults except where noted. A full nanobot config loads;
sections NanoClaw doesn't use (channels, gateway, model presets, ...) are
ignored, as nanobot ignores unknown keys.

| JSON field | nanobot (`config/schema.py`), default | NanoClaw default |
|---|---|---|
| `agents.defaults.workspace` | `:170`, `~/.nanobot/workspace` | `~/.nanoclaw/workspace` |
| `agents.defaults.model` | `:172`, `anthropic/claude-opus-4-5` | `gpt-4o-mini` |
| `agents.defaults.provider` | `:173`, `auto` | `auto` |
| `agents.defaults.maxTokens` | `:176`, 8192 | 8192 |
| `agents.defaults.contextWindowTokens` | `:177`, 200000 | 128000 (layer 6 difference #2) |
| `agents.defaults.temperature` | `:178`, 0.1 | 0.1 |
| `agents.defaults.maxToolIterations` | `:180`, 200 | 200 (was 20 in phase 1) |
| `agents.defaults.maxConcurrentSubagents` | `:181`, 4 | 4 |
| `agents.defaults.maxToolResultChars` | `:182`, 16000 | 16000 |
| `agents.defaults.providerRetryMode` | `:183`, `standard` | `standard` |
| `agents.defaults.timezone`, `timezoneMode` | `:193-194`, `UTC`, `auto` | same (stored; the time block still uses the host clock) |
| `agents.defaults.compaction` | — | `nanobot` (NanoClaw only: the bug 4 switch) |
| `providers.<name>.apiKey`, `apiBase` | `:245`, `:292` | null; any provider name |
| `tools.restrictToWorkspace` | `:442`, false | false |
| `tools.ssrfWhitelist` | `:460`, [] | [] |

**Precedence**, lowest to highest: defaults → config file (`${VAR}` references
resolved; a missing variable is an error naming the field) → `NANOCLAW_*`
environment variables (the phase 1 names keep working) → flags (`-w`,
`--restrict`, which can only turn the restriction on).

**Which provider entry is used** (a simplified `_match_provider`,
`schema.py:588`): the one `agents.defaults.provider` names; under `auto`, the
entry named by the model's prefix (`openrouter/...`) if it has a key or base;
otherwise the first entry with a key or base. With no base, nanobot's default
base is used for openrouter, deepseek, aihubmix, siliconflow and huggingface;
otherwise api.openai.com.

Commands: `nanoclaw config path`, `config show` (keys masked), `config init`
(writes defaults, or adds only the missing settings to an existing file so
nanobot-only sections survive).

## Logging

loguru, as nanobot. Logs are off unless `--logs` is given to `chat`, `ask`,
`compact` or `dream` (nanobot `agent --logs`, `cli/agent.py:67`); with it,
INFO and above go to stderr in nanobot's format. About 45 log calls were
ported with nanobot's level and text: memory and compaction (degraded archive,
raw checkpoints, journal corruption), sessions (corrupt lines), subagents,
saved tool results, web fetch and search, the runner (empty replies, malformed
tool calls, truncated output), the loop (max iterations, model errors, saved
turns), Dream, the token counter and retries. Only places where NanoClaw has
the same code path got a log call.

## Retries

`LLMProvider.chat_with_retry` (nanobot `providers/base.py:1299`, `:1580`).
The runner and memory archiving use it.

| Error | Decision | Wait |
|---|---|---|
| 429 rate limit, 5xx, timeout, connection error, `x-should-retry: true` | retry | 1, 2, 4 s (4 attempts), or Retry-After + 1 s (headers incl. `retry-after-ms`, or "try again in 1m30s" text) |
| 429 `insufficient_quota`, 402, billing | stop | — |
| 401, 403, authentication | stop | — |
| `x-should-retry: false` | stop | — |

`"providerRetryMode": "persistent"` retries forever with waits capped at 60 s
and stops after 10 identical errors. Errors from the OpenAI SDK carry their
status, type, code and headers into `LLMResponse`, and the SDK's own retries
are off (`max_retries=0`), so requests aren't retried twice. Checked against
nanobot: the same 15 SDK errors give the same decision and wait in both.

## Token usage

`LLMUsage` has nanobot's fields (`llm_usage/types.py`): input, output, total,
cached tokens read and written (None when the provider doesn't say),
reported vs estimated tokens, context tokens and the number of requests.
Cached tokens are read from the OpenAI usage block. When a provider sends no
usage, it is estimated with tiktoken from the request actually sent; a failed
request counts as an empty request. So every turn has usage, and `/status`
shows the cached share, as nanobot does.

## How it was built

Three tasks in parallel (12.1 config, 12.2 logging, 12.3 retries and usage),
each by a sub-agent in its own worktree, reviewed against nanobot and merged
without conflicts. The main session then connected them: the retry mode comes
from the config, the `--logs` flag turns logging on, and the provider retry
log lines were added. A fourth task (12.4) added the runner, loop, Dream,
token-counter and violation log calls.

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Defaults | workspace `~/.nanobot/workspace`, model `anthropic/claude-opus-4-5`, window 200,000 | `~/.nanoclaw/workspace`, `gpt-4o-mini`, 128,000 | NanoClaw's own defaults (window: layer 6) |
| 2 | Environment variables | `NANOBOT_` prefix with `__` nesting (pydantic-settings) | The phase 1 `NANOCLAW_*` names, mapped by hand | Keep phase 1 working; no pydantic-settings |
| 3 | Provider choice | Registry of ~40 providers, prefix stripping | First configured entry; a few default bases | No registry until layer 20 |
| 4 | `config init` on an existing file | `onboard --refresh` re-saves the whole file | Adds only missing settings | Keeps nanobot-only sections |
| 5 | When logging is turned off | Per CLI command; library use logs | At package import | Quiet library use and tests; same for CLI users |
| 6 | Retry status in the UI | `RetryStatusEvent`s and a heartbeat every 30 s | A plain `on_retry_wait(text)` callback, not yet shown in the chat | Events arrive with layer 13 |
| 7 | Image retry | Retries once without images | None | NanoClaw sends no images |
| 8 | Usage history | SQLite usage store for the web UI chart | Per-turn and per-session totals only | The store arrives with the web UI (layer 23) if needed |
| 9 | `FakeProvider` | — | Doesn't retry unless `retries=True` | Each scripted reply answers one request in tests |
| 10 | Error text | `Error: <whole body>` | `Error calling LLM: HTTP 429: <message>` | Phase 1 text; classification uses the status, type and code fields |

## Left for later

Model presets, fallback models and `/model` (layer 20); showing retry waits in
the chat (layer 13); the timezone in the time block; the usage store.

## Tests

Layer 12 adds about 130 tests (784 → ~915): `tests/test_config.py`,
`tests/test_logging.py`, `tests/test_retry.py`, `tests/test_usage.py`, plus
additions to the provider, runner and chat tests, adapted from nanobot's
`tests/config/`, `test_provider_retry.py`, `test_provider_error_metadata.py`,
`test_usage_contract.py` and `test_cached_tokens.py`.
