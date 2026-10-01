# 01 — Provider

Reference: nanobot at commit `d3d70f2`. Paths are relative to `nanobot/providers/`.

## The concept

The provider is the only part of the agent that talks to a language model.
Everything above it — the runner, tools, memory — asks it one question:
*"here is the conversation and the tools you may use; what do you say next?"*

It has two jobs:

1. **Send** — turn our messages + tool definitions into one API request.
2. **Receive** — turn the raw API response into one clean `LLMResponse`:
   reply text, tool calls, why the model stopped, token usage.

### One message format everywhere

nanobot (and NanoClaw) store every conversation in the **OpenAI chat format**,
whatever model is behind it:

```python
{"role": "system",    "content": "You are ..."}
{"role": "user",      "content": "read notes.md"}
{"role": "assistant", "content": None,
 "tool_calls": [{"id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\": \"notes.md\"}"}}]}
{"role": "tool",      "tool_call_id": "call_1", "content": "<file text>"}
{"role": "assistant", "content": "The notes say ..."}
```

An OpenAI-compatible provider sends this as-is. A provider for a different API
(Anthropic, Bedrock) translates it on the way out and back. Either way the rest
of the agent only ever sees this one shape.

## What we built

| NanoClaw | nanobot original | Notes |
|---|---|---|
| `providers/base.py` `ToolCallRequest` | `base.py:72` | `id`, `name`, `arguments`; `to_openai_tool_call()` writes it back into history |
| `providers/base.py` `parse_tool_arguments` | `base.py:118` | identical rules (see below) |
| `providers/base.py` `LLMUsage` | `base.py:282` (~280 lines) | just input/output token counts |
| `providers/base.py` `LLMResponse` | `base.py:567` | `content`, `tool_calls`, `finish_reason`, `usage`, `reasoning_content`; `should_execute_tools`, `is_error` |
| `providers/base.py` `sanitize_messages` | `base.py:834` `_sanitize_empty_content` | empty-string content → `None` / `"(empty)"` |
| `providers/base.py` `LLMProvider` | `base.py:636` | abstract `chat()`; `safe_chat()` = nanobot's `_safe_chat` (`base.py:1281`) |
| `providers/openai_compat.py` `OpenAICompatProvider` | `openai_compat_provider.py:510` | `build_kwargs` ↔ `_build_kwargs` (`:929`), `parse` ↔ `_parse` (`:1546`), `_error_response` ↔ `_handle_error` (`:1901`) |
| `providers/factory.py` `provider_from_env` | `registry.py` + `factory.py` | three env vars instead of a registry |
| `providers/fake.py` `FakeProvider` | — | scripted replies for tests |

About 520 lines including docstrings and comments, versus about 5,400 for the same files in nanobot.

## Design decisions worth knowing

**Errors are values, not exceptions.** A failed call returns
`LLMResponse(finish_reason="error", content="Error calling LLM: ...")`. The
runner (layer 2) can then treat a failure like any other response: stop, show
it, or later retry, with no `try/except` scattered around. `safe_chat` is the
safety net for anything unexpected. `asyncio.CancelledError` is deliberately
*not* caught, so Ctrl-C still interrupts a turn.

**Don't guess tool arguments.** `parse_tool_arguments` turns valid JSON into a
dict and empty arguments into `{}`, but leaves malformed JSON as the original
string. Running a tool with "repaired" arguments could, for example, write to
the wrong file. Instead the tool registry (layer 3) rejects the call and the
model sees the error and tries again.

**Only run tools after a normal stop.** `should_execute_tools` is true only
when the model asked for tools *and* `finish_reason` is `tool_calls` / `stop`.
If the reply was cut off (`length`) or filtered (`content_filter`), the tool
calls may be incomplete. nanobot added this check after gateways injected
tool calls into refused responses (their issue #3220).

**Unique tool-call IDs.** Every `tool` result message must answer exactly one
call ID. Some providers omit IDs or reuse one for parallel calls, so `parse`
generates a fresh 9-character ID when needed (nanobot's `_short_tool_id`).

**Empty content is rewritten.** Most chat APIs reject `"content": ""`. An
assistant message that only carries tool calls gets `None`; anything else gets
`"(empty)"`.

**Stored history is repaired before every request.** History is replayed on
every call, so one bad message would break every later request.
`OpenAICompatProvider._sanitize_messages` (nanobot
`openai_compat_provider.py:701`) cleans a *copy* of the messages before sending:

- drops keys the API doesn't know (anything outside `role`, `content`,
  `tool_calls`, `tool_call_id`, `name`, `reasoning_content`)
- rewrites each replayed tool call's `arguments` as a valid JSON object
  (`tool_arguments_json_for_replay`, nanobot `base.py:169`). Malformed
  arguments are kept *as-is* for the tool registry to reject when the call
  first runs, but replaying them as invalid JSON could get the whole request
  rejected, so on replay they become `"{}"`.
- merges two user (or two assistant) messages in a row and drops a trailing
  assistant message (`enforce_role_alternation`, nanobot `base.py:1144`).
  vLLM, Ollama, Azure and others reject both. The runner itself creates
  consecutive user messages when it adds a "please answer" prompt.

**Lazy client.** Creating a provider never touches the network; the SDK client
is built on the first call. nanobot does the same because client creation is
slow on some systems.

## What we left out

| nanobot feature | Where | Why it exists |
|---|---|---|
| Retry policy with backoff, `Retry-After`, quota detection | `base.py:1462–1980` | Rate limits and flaky gateways. We rely on the SDK's built-in 2 retries for now; nanobot sets `max_retries=0` and does its own. |
| Streaming | `chat_stream`, `_parse_chunks` | Show replies word by word |
| Detailed usage (cache reads/writes, timing) | `LLMUsage`, `llm_usage/` | Cost tracking |
| Per-model quirks | `_build_kwargs` | `max_completion_tokens` for o-series/GPT-5, no `temperature` for reasoning models, DeepSeek `reasoning_content` backfill, Kimi temperature, Gemini thought signatures, OpenRouter headers, prompt-cache markers |
| OpenAI Responses API + native compaction | `_build_responses_body`, `openai_responses/` | Newer OpenAI endpoint; server-side history compression |
| Tool calls written as text | `_extract_text_tool_calls` | Rescues weaker models that print JSON instead of calling tools |
| Tool calls spread across several `choices` | `_parse` loop | A few gateways do this; we read `choices[0]` only |
| Repairing malformed replayed arguments | `tool_arguments_object_for_replay` (uses `json_repair`) | Keeps what can be saved; we fall back to `{}` |
| Tool-call ID normalization on replay | `_sanitize_messages` `map_id` / `unique_tool_id` | Providers with strict ID formats (Mistral) |
| Role-alternation safety nets | `_enforce_role_alternation` | Synthetic user message for GLM-style providers; merging list (image) content |
| Provider registry, presets, fallback, OAuth | `registry.py`, `factory.py`, `fallback_provider.py`, `*_oauth.py` | ~30 providers, named model setups, backup models, Copilot/Codex logins |
| Anthropic / Bedrock / Azure adapters | `anthropic_provider.py`, … | Non-OpenAI wire formats |

## Try it

```bash
export NANOCLAW_API_KEY=sk-...
export NANOCLAW_MODEL=gpt-4o-mini            # or deepseek-chat, llama3.2, ...
export NANOCLAW_API_BASE=https://api.deepseek.com   # omit for OpenAI
uv run nanoclaw ask "Say hello in five words"
```

Without a key you get a clean error, not a traceback:

```
Error calling LLM: HTTP 401: Incorrect API key provided: no-key. ...
```

## Tests

- `tests/test_provider_base.py` — argument parsing, serialization, the
  `should_execute_tools` rule, sanitizing, `safe_chat`.
- `tests/test_openai_compat.py` — the real `openai` SDK against a mocked HTTP
  transport: the exact request body sent, text and tool-call parsing, malformed
  arguments, duplicate IDs, reasoning text, HTTP and connection errors.
- `tests/test_fake_provider_and_cli.py` — the fake provider and `nanoclaw ask`.
