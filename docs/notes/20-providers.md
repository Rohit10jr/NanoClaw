# 20 — Providers: registry, presets, `/model`, Anthropic, fallback

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package. Layer 12 already ported retries and usage, layer 13 streaming.

## The concept

Until now NanoClaw talked to one OpenAI-compatible endpoint, picked with a
simple rule from the config. nanobot knows dozens of providers and lets the
user name **model presets** (a model with its provider and settings), switch
between them in a chat with **`/model`**, fall back to other models when one
keeps failing, and talk to Claude through Anthropic's own Messages API. This
layer ports a slim version of each.

```
config ── resolve_provider_route ──▶ ProviderSpec (registry) ──▶ backend: openai_compat | anthropic
  modelPresets / --preset / /model ─▶ ModelRuntimeResolver ──▶ provider per preset (cached)
  fallbackModels ───────────────────▶ FallbackProvider(primary, [fallback, ...])
```

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `ProviderSpec`, `PROVIDERS` | 19 providers (custom, OpenRouter, Hugging Face, AiHubMix, SiliconFlow, Anthropic, OpenAI, DeepSeek, Gemini, Zhipu, DashScope, Moonshot, MiniMax, Mistral, vLLM, Ollama, LM Studio, NVIDIA, Groq): environment key, default base URL, gateway/local detection by key prefix or base keyword, model-prefix stripping, `max_completion_tokens` and temperature rules, prompt caching, thinking style | `providers/registry.py` |
| `resolve_provider_route` | Picks provider and backend: the configured `provider`, else the model's prefix, a key prefix, a base keyword, gateways first, then the first configured entry; custom providers under `providers` | `providers/routing.py:95` |
| `make_provider` | Builds the OpenAI-compatible or Anthropic backend with extra headers, body and query, and wraps it in the fallback chain | `providers/factory.py:57-226` |
| `modelPresets`, `modelPreset` | Named `{model, provider, maxTokens, contextWindowTokens, temperature, reasoningEffort}`; `--preset NAME` on `chat`/`ask` (NanoClaw), `dream.modelOverride` | `config/schema.py:148`, `agent/model_presets.py` |
| `ModelRuntimeResolver`, `/model` | One provider per preset, built once and closed when the loop closes; `/model` lists presets and switches the session's preset (saved in its metadata) | `agent/model_runtime.py`, `command/builtin.py` |
| `AnthropicProvider` | Official SDK: history converted to Messages API blocks (tool results in user turns, tool calls with thinking blocks first, same-role merge), tools and `tool_choice`, prompt-caching markers, thinking (disabled, adaptive, budgets with temperature 1.0), parsing incl. cache usage and thinking blocks, native streaming, the "streaming is required" fallback, error metadata for retries | `providers/anthropic_provider.py` |
| `thinking_blocks` | Kept on the assistant message and replayed in later requests | `providers/base.py`, `agent/runner.py` |
| `FallbackProvider` | Each request runs the primary's retries; on a timeout, 5xx, 429, quota or auth error the next fallback is tried (built for that attempt, possibly another backend); after 3 primary failures in a row the primary is skipped for 60 s; no switch after visible streamed text; the chain's smallest context window is used for compaction; `↳ Using fallback model X` in the chat | `providers/fallback_provider.py` |

```json
{
  "agents": { "defaults": { "modelPreset": "fast", "fallbackModels": ["smart"] } },
  "modelPresets": {
    "fast":  { "model": "openai/gpt-4o-mini" },
    "smart": { "model": "anthropic/claude-sonnet-4-6", "reasoningEffort": "medium" }
  },
  "providers": { "openai": { "apiKey": "${OPENAI_API_KEY}" }, "anthropic": { "apiKey": "${ANTHROPIC_API_KEY}" } }
}
```

## How it was built

Phase A in parallel: registry, presets and `/model` (20.1); the Anthropic
adapter (20.2). Integration: the factory builds the real adapter (its fallback
for a missing adapter removed), a test builds it from a config. Phase B: the
fallback chain (20.3).

| Task | Checked against nanobot |
|---|---|
| 20.1 Registry, presets, `/model` | 63 config × preset cases (provider, backend, base URL, model sent, full request, headers, errors): 60 identical, 3 deliberate (below); `/model` outputs 16/16 identical; re-run after integration |
| 20.2 Anthropic | 82 request comparisons (41 cases × `chat`/`chat_stream`): 2 deliberate (malformed replayed arguments); 52 response, stream, error and fallback comparisons: identical; re-run after integration |
| 20.3 Fallback | 72 scripted scenarios (29 error kinds, chain positions, standard and persistent retries, streaming with and without text, circuit breaker on a scripted clock, preset vs inline candidates, context windows): identical; re-run after merging |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Nothing configured, or `custom` without `apiBase` | Refuses to start / "requires api_base" | api.openai.com with `NANOCLAW_API_KEY` | Phase 1 behaviour keeps working |
| 2 | Default context window of a preset | 200,000 | 128,000 | Layer 6 decision |
| 3 | Registry | ~45 providers incl. Azure, Bedrock, OAuth logins | 19 | Decision 9 |
| 4 | OpenAI Responses API | Used for some OpenAI models | Chat Completions always | Works for every model |
| 5 | Malformed replayed tool arguments (Anthropic) | Repaired with `json_repair` | `{}` | As NanoClaw's OpenAI adapter |
| 6 | `reasoning_effort` | Per call | Set when the provider is built (per preset and fallback) | NanoClaw's `chat()` signature |
| 7 | `--preset` flag | — | Added | NanoClaw CLI |
| 8 | Fallback model notice | Web UI event only | Also a progress line in the chat | No web UI yet |
| 9 | Provider conversation state, native compaction, OAuth re-login hints | Yes | No | Not ported |

## Tests

Layer 20 adds 435 tests (2,226 → 2,661, of which 6 skip on Linux): `test_provider_registry.py`,
`test_openai_compat_spec.py`, `test_model_presets.py`,
`test_anthropic_provider.py`, `test_anthropic_sdk.py` (the real SDK over a mock
HTTP transport), `test_fallback_provider.py` and `test_provider_gaps.py`, adapted from nanobot's.

### Mutation checks

309 rules were broken on purpose (the run was interrupted by a container
restart; a second sub-agent finished it from the saved tests and logs):

| Area | Rules | Caught at once | Gaps closed with tests | Equivalent | Result |
|---|---|---|---|---|---|
| Registry, routing, factory | 64 | 45 | 14 | 5 | 59 caught |
| Presets, runtime, `/model`, OpenAI parameters | 85 | 68 | 17 | 0 | 85 caught |
| Anthropic adapter, thinking blocks | 90 | 75 | 13 | 2 | 88 caught |
| Fallback and its wiring | 70 | 52 | 16 | 2 | 68 caught |

59 new tests close the gaps, for example: `openrouter/anthropic/claude-x`
stripped to `claude-x`; dash/underscore normalisation in keyword matching; a
custom provider without a base never used as a last resort; the Anthropic
adapter receiving the configured base and extra body; presets saved on `/model`
and on recovery; empty text blocks dropped and cache markers on the last system
block; thinking budgets; every `_should_fallback` rule; a fallback that already
streamed text stops the chain; a fallback model's token count not triggering
compaction.

Equivalent (9): checks repeated later with the same message (custom provider
without a base), branches that can't be reached (an Anthropic backend is never
an `openai_compat` spec), names that are always canonical already, and token
tables where each entry is also matched by a broader rule.

The layer 16 split `wait_for` test, which had failed now and then in every
mutation run since layer 17, was finally explained here: its text `ab` appeared
in the random hex session id about 4% of the time. It now uses letters outside
hex.

