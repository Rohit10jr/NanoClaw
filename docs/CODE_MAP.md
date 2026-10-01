# Code Map: NanoClaw ↔ nanobot

Every piece we have built, where it lives in NanoClaw, and the nanobot code it
comes from. Updated at the end of each layer. The plan for future layers is in
[`ROADMAP.md`](ROADMAP.md).

nanobot links point to commit
[`d3d70f2`](https://github.com/HKUDS/nanobot/tree/d3d70f20655bce5d3c7a017bfbe803efe09a2438),
so line numbers stay correct even as nanobot changes.

**How it compares** (last column):

- **Same:** same behaviour as nanobot, often the same code.
- **Simplified:** same idea, with production details removed (listed in the layer note).
- **Replaced:** nanobot does this a different, bigger way; we use a small stand-in.
- **New:** exists only in NanoClaw.


## Summary

| Layer | NanoClaw files | Lines | nanobot files | Lines |
|---|---|---|---|---|
| 1 Provider | `nanoclaw/providers/` (4 modules) | ~450 | `providers/base.py`, `openai_compat_provider.py`, `registry.py`, `factory.py` | ~5,400 |
| 2 Runner | `nanoclaw/agent/runner.py`, `messages.py`, `tools/base.py`, `tools/registry.py` | ~440 | `agent/runner.py`, `agent/tools/base.py`, `registry.py`, `execution.py` | ~2,250 |
| CLI (`ask`) | `nanoclaw/__main__.py` | ~45 | `cli/agent.py`, `cli/commands.py` | ~1,180 |

## Layer 1 — Provider

Note: [`notes/01-provider.md`](notes/01-provider.md)

### Types — `nanoclaw/providers/base.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`parse_tool_arguments`](../nanoclaw/providers/base.py#L20) | [`providers/base.py:118`][nb-pta] | Same rules: valid JSON → value, empty → `{}`, malformed → kept as string |
| [`ToolCallRequest`](../nanoclaw/providers/base.py#L43) | [`providers/base.py:72`][nb-tcr] | Simplified: `id`, `name`, `arguments` (no provider-specific extra fields) |
| ↳ `has_valid_name()` | [`base.py:81`][nb-tcr-hvn] | Same |
| ↳ `to_openai_tool_call()` | [`base.py:94`][nb-tcr-oai] | Same, minus extra fields |
| [`LLMUsage`](../nanoclaw/providers/base.py#L69) | [`providers/base.py:282`][nb-usage] | Simplified: input/output tokens only (no cache, timing or estimates) |
| [`LLMResponse`](../nanoclaw/providers/base.py#L81) | [`providers/base.py:567`][nb-resp] | Simplified: `content`, `tool_calls`, `finish_reason`, `usage`, `reasoning_content` |
| ↳ `has_tool_calls` | [`base.py:611`][nb-resp-htc] | Same |
| ↳ `should_execute_tools` | [`base.py:616`][nb-resp-set] | Same rule: tools run only after `tool_calls` / `function_call` / `stop` |
| ↳ `is_error` | — | New: shorthand for `finish_reason == "error"` |
| [`sanitize_messages`](../nanoclaw/providers/base.py#L109) | [`base.py:834`][nb-sanitize] `_sanitize_empty_content` | Simplified: empty-content rule only (no `_meta` stripping, no surrogate scrubbing) |
| [`LLMProvider`](../nanoclaw/providers/base.py#L127) | [`providers/base.py:636`][nb-prov] | Simplified: constructor + abstract `chat()` + `safe_chat()` |
| ↳ `chat()` (abstract) | [`base.py:943`][nb-prov-chat] | Same signature, minus `reasoning_effort` |
| ↳ `safe_chat()` | [`base.py:1281`][nb-safe] `_safe_chat` | Simplified: exception → error response; no call observer |
| — | [`base.py:1462`][nb-retry]–1980 `chat_with_retry`, `chat_stream*` | Not built (Phase 2: retries, streaming) |

### OpenAI-compatible adapter — `nanoclaw/providers/openai_compat.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`OpenAICompatProvider`](../nanoclaw/providers/openai_compat.py#L49) | [`openai_compat_provider.py:510`][nb-oc] | Simplified |
| ↳ `__init__` / `_get_client()` | [`:519`][nb-oc-init], [`:568`][nb-oc-client] `_build_client`, `:615` `_ensure_client` | Simplified: lazy SDK client; no proxy, local-endpoint or header handling |
| ↳ [`build_kwargs()`](../nanoclaw/providers/openai_compat.py#L81) | [`:929`][nb-oc-kwargs] `_build_kwargs` | Simplified: model, messages, max_tokens, temperature, tools, tool_choice. No per-model quirks |
| ↳ [`chat()`](../nanoclaw/providers/openai_compat.py#L102) | [`:1960`][nb-oc-chat] | Simplified: Chat Completions only (no Responses API) |
| ↳ [`parse()`](../nanoclaw/providers/openai_compat.py#L119) | [`:1546`][nb-oc-parse] `_parse` | Simplified: `choices[0]` only; same ID de-duplication; no text-tool-call rescue |
| ↳ `_error_response()` | [`:1901`][nb-oc-err] `_handle_error` | Simplified: readable message; no retry metadata |
| [`_short_tool_id`](../nanoclaw/providers/openai_compat.py#L28) | [`:231`][nb-oc-id] | Same |
| [`_extract_text`](../nanoclaw/providers/openai_compat.py#L33) | [`:1415`][nb-oc-text] `_extract_text_content` | Simplified |

### Choosing a provider — `nanoclaw/providers/factory.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`provider_from_env`](../nanoclaw/providers/factory.py#L22) | [`providers/factory.py:273`][nb-make] `make_provider` + [`providers/registry.py`][nb-registry] | Replaced: three env vars (`NANOCLAW_API_KEY`, `_API_BASE`, `_MODEL`) instead of a registry of ~30 providers and config presets |

### Testing — `nanoclaw/providers/fake.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`FakeProvider`](../nanoclaw/providers/fake.py#L21), [`RecordedCall`](../nanoclaw/providers/fake.py#L13) | — (nanobot tests use mocks) | New: scripted replies + recorded requests for testing later layers |

## Layer 2 — Runner

Note: [`notes/02-runner.md`](notes/02-runner.md)

### Tool interface — `nanoclaw/agent/tools/base.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`ToolResult`](../nanoclaw/agent/tools/base.py#L9) | [`agent/tools/base.py:149`][nb-l2-toolresult] | Same: a `str` with an `is_error` flag; `ToolResult.error(...)` |
| [`is_tool_error`](../nanoclaw/agent/tools/base.py#L29) | [`agent/tools/registry.py:15`][nb-l2-iste] `is_tool_error_result` | Same |
| [`Tool`](../nanoclaw/agent/tools/base.py#L33) | [`agent/tools/base.py:164`][nb-l2-tool] | Simplified: `name`, `description`, `parameters`, `execute`, `to_schema`. No concurrency flags, plugin metadata, `cast_params` or `validate_params` (layer 3) |
| ↳ `to_schema()` | [`base.py:313`][nb-l2-toschema] | Same |

### Registry — `nanoclaw/agent/tools/registry.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`ToolRegistry`](../nanoclaw/agent/tools/registry.py#L18) | [`agent/tools/registry.py:19`][nb-l2-registry] | Simplified |
| ↳ `register` / `unregister` / `get` / `has` / `tool_names` | [`:30`][nb-l2-register]–`:71`, `:204` | Same |
| ↳ [`get_definitions()`](../nanoclaw/agent/tools/registry.py#L46) | [`:86`][nb-l2-defs] | Simplified: sorted by name; no MCP grouping or caching |
| ↳ [`_suggest_name()`](../nanoclaw/agent/tools/registry.py#L54) | [`:58`][nb-l2-suggest] | Same: "Did you mean ...?" for case/punctuation differences, never auto-runs |
| ↳ [`prepare_call()`](../nanoclaw/agent/tools/registry.py#L65) | [`:110`][nb-l2-prepare] | Simplified: unknown tool, non-object arguments, missing required keys. Same messages. No argument unwrapping, casting or full schema validation |
| ↳ [`execute()`](../nanoclaw/agent/tools/registry.py#L89) | [`:187`][nb-l2-execute] | Same: never raises; errors get the retry hint |
| [`RETRY_HINT`](../nanoclaw/agent/tools/registry.py#L10), [`with_retry_hint`](../nanoclaw/agent/tools/registry.py#L13) | [`agent/tools/execution.py:23`][nb-l2-hint], `:50` `_with_retry_hint` | Same text; added once |

### Runner — `nanoclaw/agent/runner.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`AgentRunSpec`](../nanoclaw/agent/runner.py#L35) | [`agent/runner.py:89`][nb-l2-spec] | Simplified: 8 fields instead of ~25 (no hooks, callbacks, checkpoints, workspace) |
| [`AgentRunResult`](../nanoclaw/agent/runner.py#L49) | [`runner.py:119`][nb-l2-result] | Simplified: `final_content`, `messages`, `tools_used`, `usage`, `stop_reason`, `error` |
| [`AgentRunner.run()`](../nanoclaw/agent/runner.py#L78) | [`runner.py:284`][nb-l2-run] `run` + [`:368`][nb-l2-core] `_run_core` | Simplified: the same loop and stop rules, ~75 lines instead of ~520 |
| ↳ tool branch | [`runner.py:487`][nb-l2-toolbranch] | Simplified: sequential, no checkpoints or hooks |
| ↳ ignore calls in cut-off replies | [`runner.py:579`][nb-l2-ignore] | Same |
| ↳ empty-reply retry + finalization | [`runner.py:592`][nb-l2-empty], [`:1126`][nb-l2-finretry] `_request_finalization_retry` | Same rule (`_MAX_EMPTY_RETRIES = 2`) |
| ↳ model error | [`runner.py:714`][nb-l2-error] | Same, minus the arrearage message |
| ↳ empty final response | [`runner.py:737`][nb-l2-emptyfinal] | Same |
| [`_execute_tool_call()`](../nanoclaw/agent/runner.py#L170) | [`agent/tools/execution.py:57`][nb-l2-exec] `execute_tool_calls`, [`:115`][nb-l2-exec1] `_execute_tool_call` | Simplified: no batching, throttling or violation classification |
| ↳ empty-output marker + truncation | [`context_governance.py:716`][nb-l2-normalize] `normalize_tool_result` | Simplified: truncate in place; nanobot can offload huge results to files |
| [`_finalize_after_max_iterations()`](../nanoclaw/agent/runner.py#L183) | [`runner.py:1154`][nb-l2-finmax], [`:1249`][nb-l2-maxfallback] `_max_iterations_fallback` | Same |
| [`_append_model_error_placeholder()`](../nanoclaw/agent/runner.py#L198) | [`runner.py:1361`][nb-l2-placeholder] | Same |
| [`_add_usage()`](../nanoclaw/agent/runner.py#L64) | `runner.py:1326` `_merge_usage` | Simplified |

### Messages and fixed texts — `nanoclaw/agent/messages.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`build_assistant_message`](../nanoclaw/agent/messages.py#L40) | [`utils/helpers.py:753`][nb-l2-bam] | Simplified: no thinking blocks |
| [`build_tool_message`](../nanoclaw/agent/messages.py#L53) | inline in `runner.py:538` | New helper, same shape |
| [`truncate_text`](../nanoclaw/agent/messages.py#L57), `TRUNCATED_SUFFIX` | [`utils/helpers.py:400`][nb-l2-trunc], `:369` | Same |
| `FINALIZATION_RETRY_PROMPT`, `BUDGET_EXHAUSTED_FINALIZATION_PROMPT`, `EMPTY_FINAL_RESPONSE_MESSAGE` | [`utils/runtime.py:19–34`][nb-l2-runtime] | Same text |
| `MODEL_ERROR_PLACEHOLDER`, `DEFAULT_ERROR_MESSAGE` | [`runner.py:67–72`][nb-l2-consts] | Same text |
| `MAX_ITERATIONS_MESSAGE` | [`templates/agent/max_iterations_message.md`][nb-l2-maxtpl] | Same text (Python format instead of Jinja) |

## CLI — `nanoclaw/__main__.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`main()`](../nanoclaw/__main__.py#L31) `nanoclaw ask "..."` | [`cli/agent.py:55`][nb-cli] `nanobot agent -m "..."` | Simplified: one `AgentRunner` turn with an empty tool registry (tools arrive in layer 3). Switches to `AgentLoop` in layer 5 and a chat REPL in layer 10 |

## Tests

| File | Covers |
|---|---|
| `tests/test_provider_base.py` | argument parsing, tool-call serialization, `should_execute_tools`, sanitizing, `safe_chat` |
| `tests/test_openai_compat.py` | real `openai` SDK over a mocked HTTP transport: request body, text + tool-call parsing, malformed args, duplicate IDs, reasoning text, HTTP + connection errors, `provider_from_env` |
| `tests/test_fake_provider_and_cli.py` | `FakeProvider`, `nanoclaw ask` |
| `tests/toy_tools.py` | toy tools for tests: `echo`, `add`, `boom`, `fail`, `silent`, `big` |
| `tests/test_tools_registry.py` | `ToolResult`, `to_schema`, sorted definitions, every registry error message, retry hint once |
| `tests/test_runner.py` | the loop end to end with `FakeProvider`: exact history, requests sent, tool failures, truncation, cut-off replies, empty-reply handling, model errors, iteration limit, usage, no mutation, one result per call |
| `tests/test_cli_version.py` | `--version` |

[nb-pta]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L118
[nb-tcr]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L72
[nb-tcr-hvn]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L81
[nb-tcr-oai]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L94
[nb-usage]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L282
[nb-resp]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L567
[nb-resp-htc]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L611
[nb-resp-set]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L616
[nb-sanitize]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L834
[nb-prov]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L636
[nb-prov-chat]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L943
[nb-safe]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L1281
[nb-retry]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L1462
[nb-oc]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L510
[nb-oc-init]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L519
[nb-oc-client]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L568
[nb-oc-kwargs]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L929
[nb-oc-chat]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L1960
[nb-oc-parse]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L1546
[nb-oc-err]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L1901
[nb-oc-id]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L231
[nb-oc-text]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L1415
[nb-make]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/factory.py#L273
[nb-registry]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/registry.py
[nb-cli]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/agent.py#L55
[nb-l2-toolresult]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/base.py#L149
[nb-l2-iste]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/registry.py#L15
[nb-l2-tool]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/base.py#L164
[nb-l2-toschema]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/base.py#L313
[nb-l2-registry]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/registry.py#L19
[nb-l2-register]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/registry.py#L30
[nb-l2-defs]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/registry.py#L86
[nb-l2-suggest]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/registry.py#L58
[nb-l2-prepare]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/registry.py#L110
[nb-l2-execute]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/registry.py#L187
[nb-l2-hint]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/execution.py#L23
[nb-l2-spec]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L89
[nb-l2-result]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L119
[nb-l2-run]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L284
[nb-l2-core]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L368
[nb-l2-toolbranch]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L487
[nb-l2-ignore]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L579
[nb-l2-empty]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L592
[nb-l2-finretry]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L1126
[nb-l2-error]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L714
[nb-l2-emptyfinal]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L737
[nb-l2-exec]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/execution.py#L57
[nb-l2-exec1]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/execution.py#L115
[nb-l2-normalize]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context_governance.py#L716
[nb-l2-finmax]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L1154
[nb-l2-maxfallback]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L1249
[nb-l2-placeholder]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L1361
[nb-l2-bam]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L753
[nb-l2-trunc]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L400
[nb-l2-runtime]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/runtime.py#L19
[nb-l2-consts]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L67
[nb-l2-maxtpl]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/templates/agent/max_iterations_message.md
