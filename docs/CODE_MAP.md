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
| 1 Provider | `nanoclaw/providers/` (4 modules) | ~520 | `providers/base.py`, `openai_compat_provider.py`, `registry.py`, `factory.py` | ~5,400 |
| 2 Runner | `nanoclaw/agent/runner.py`, `messages.py`, `tools/base.py`, `tools/registry.py` | ~540 | `agent/runner.py`, `agent/tools/base.py`, `registry.py`, `execution.py` | ~2,250 |
| 3 Tools | `nanoclaw/agent/tools/schema.py`, `filesystem.py`, `shell.py`, `web.py`, `agent/tool_results.py` | ~1,420 | `agent/tools/filesystem.py`, `shell.py`, `web.py`, `schema.py`, `loader.py` | ~4,170 |
| 4 Context | `nanoclaw/agent/context.py`, `runtime_context.py`, `utils/templates.py` + `templates/` | ~260 | `agent/context.py`, `runtime_context.py`, `utils/prompt_templates.py` + `templates/` | ~640 |
| 5 Sessions + AgentLoop | `nanoclaw/session/`, `nanoclaw/agent/loop.py`, `agent/context_governance.py` | ~580 | `session/manager.py`, `agent/loop.py` | ~4,400 |
| 6 Memory + compaction | `nanoclaw/agent/memory.py`, `session/summary.py`, `utils/tokens.py`, `utils/llm_runtime.py` + loop/context changes | ~1,100 | `agent/memory.py`, `agent/autocompact.py`, `session/summary.py`, parts of `agent/context_governance.py`, `utils/helpers.py`, `command/builtin.py` | ~2,500 |
| 7 Skills | `nanoclaw/agent/skills.py` + 6 built-in skills, context/loop/read_file/Dream changes | ~290 + skills | `agent/skills.py`, `skills/`, parts of `context.py`, `filesystem.py`, `memory.py` | ~420 + skills |
| 8 Subagents | `nanoclaw/agent/subagent.py`, `tools/spawn.py`, `tools/context.py` + runner/loop/CLI changes | ~470 | `agent/subagent.py`, `agent/tools/spawn.py`, parts of `tools/context.py`, `runner.py`, `loop.py` | ~800 |
| 9 Safety | `nanoclaw/security/` (2 modules), `agent/tools/violations.py` + shell/file/web/runner changes | ~650 | `security/network.py`, `security/workspace_policy.py`, parts of `shell.py`, `filesystem.py`, `web.py`, `execution.py`, `utils/runtime.py` | ~1,100 |
| 10 Terminal chat | `nanoclaw/cli/` (2 modules), `utils/tool_hints.py`, `utils/path.py` + runner/loop changes | ~950 | classic parts of `cli/agent.py`, `cli/terminal.py`, `cli/stream.py`, parts of `command/builtin.py`, `command/router.py`, `utils/tool_hints.py`, `utils/path.py` | ~1,500 |
| CLI (`chat`, `ask`, `compact`, `dream`, `skills`, `prompt`) | `nanoclaw/__main__.py` | ~265 | `cli/agent.py`, `cli/commands.py` | ~1,180 |

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
| [`sanitize_messages`](../nanoclaw/providers/base.py#L160) | [`base.py:834`][nb-sanitize] `_sanitize_empty_content` | Simplified: empty-content rule only (no `_meta` stripping, no surrogate scrubbing) |
| [`tool_arguments_json_for_replay`](../nanoclaw/providers/base.py#L109) | [`base.py:169`][nb-replay-args] (+ `:141`) | Simplified: unparseable → `"{}"` (nanobot tries `json_repair` first) |
| [`enforce_role_alternation`](../nanoclaw/providers/base.py#L128) | [`base.py:1144`][nb-alternation] `_enforce_role_alternation` | Simplified: merge string content, drop trailing assistant. No GLM synthetic-user safety nets or list-content merging |
| [`LLMProvider`](../nanoclaw/providers/base.py#L180) | [`providers/base.py:636`][nb-prov] | Simplified: constructor + abstract `chat()` + `safe_chat()` |
| ↳ `chat()` (abstract) | [`base.py:943`][nb-prov-chat] | Same signature, minus `reasoning_effort` |
| ↳ `safe_chat()` | [`base.py:1281`][nb-safe] `_safe_chat` | Simplified: exception → error response; no call observer |
| — | [`base.py:1462`][nb-retry]–1980 `chat_with_retry`, `chat_stream*` | Not built (Phase 2: retries, streaming) |

### OpenAI-compatible adapter — `nanoclaw/providers/openai_compat.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`OpenAICompatProvider`](../nanoclaw/providers/openai_compat.py#L56) | [`openai_compat_provider.py:510`][nb-oc] | Simplified |
| ↳ `__init__` / `_get_client()` | [`:519`][nb-oc-init], [`:568`][nb-oc-client] `_build_client`, `:615` `_ensure_client` | Simplified: lazy SDK client; no proxy, local-endpoint or header handling |
| ↳ [`build_kwargs()`](../nanoclaw/providers/openai_compat.py#L88) | [`:929`][nb-oc-kwargs] `_build_kwargs` | Simplified: model, messages, max_tokens, temperature, tools, tool_choice. No per-model quirks |
| ↳ [`chat()`](../nanoclaw/providers/openai_compat.py#L137) | [`:1960`][nb-oc-chat] | Simplified: Chat Completions only (no Responses API) |
| ↳ [`parse()`](../nanoclaw/providers/openai_compat.py#L154) | [`:1546`][nb-oc-parse] `_parse` | Simplified: `choices[0]` only; same ID de-duplication; no text-tool-call rescue |
| ↳ `_error_response()` | [`:1901`][nb-oc-err] `_handle_error` | Simplified: readable message; no retry metadata |
| ↳ [`_sanitize_messages()`](../nanoclaw/providers/openai_compat.py#L110), `_ALLOWED_MSG_KEYS` | [`:701`][nb-oc-sanitize] `_sanitize_messages`, `:89` | Simplified: allowed keys, replayed arguments, empty content, role alternation. No tool-call ID normalization or provider-specific rewrites |
| [`_short_tool_id`](../nanoclaw/providers/openai_compat.py#L35) | [`:231`][nb-oc-id] | Same |
| [`_extract_text`](../nanoclaw/providers/openai_compat.py#L40) | [`:1415`][nb-oc-text] `_extract_text_content` | Simplified |

### Choosing a provider — `nanoclaw/providers/factory.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`provider_from_env`](../nanoclaw/providers/factory.py#L22) | [`providers/factory.py:273`][nb-make] `make_provider` + [`providers/registry.py`][nb-registry] | Replaced: three env vars (`NANOCLAW_API_KEY`, `_API_BASE`, `_MODEL`) instead of a registry of ~30 providers and config presets |

### Testing — `nanoclaw/providers/fake.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`FakeProvider`](../nanoclaw/providers/fake.py#L23), [`RecordedCall`](../nanoclaw/providers/fake.py#L13) | — (nanobot tests use mocks) | New: scripted replies + recorded requests (messages, tools, model, max_tokens, temperature) for testing later layers |

## Layer 2 — Runner

Note: [`notes/02-runner.md`](notes/02-runner.md)

### Tool interface — `nanoclaw/agent/tools/base.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`ToolResult`](../nanoclaw/agent/tools/base.py#L11) | [`agent/tools/base.py:149`][nb-l2-toolresult] | Same: a `str` with an `is_error` flag; `ToolResult.error(...)` |
| [`is_tool_error`](../nanoclaw/agent/tools/base.py#L31) | [`agent/tools/registry.py:15`][nb-l2-iste] `is_tool_error_result` | Same |
| [`Tool`](../nanoclaw/agent/tools/base.py#L35) | [`agent/tools/base.py:164`][nb-l2-tool] | Simplified: `name`, `description`, `parameters`, `execute`, `to_schema`. No concurrency flags, plugin metadata, `cast_params` or `validate_params` (layer 3) |
| ↳ `to_schema()` | [`base.py:313`][nb-l2-toschema] | Same |

### Registry — `nanoclaw/agent/tools/registry.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`ToolRegistry`](../nanoclaw/agent/tools/registry.py#L19) | [`agent/tools/registry.py:19`][nb-l2-registry] | Simplified |
| ↳ `register` / `unregister` / `get` / `has` / `tool_names` | [`:30`][nb-l2-register]–`:71`, `:204` | Same |
| ↳ [`get_definitions()`](../nanoclaw/agent/tools/registry.py#L47) | [`:86`][nb-l2-defs] | Simplified: sorted by name; no MCP grouping or caching |
| ↳ [`_suggest_name()`](../nanoclaw/agent/tools/registry.py#L55) | [`:58`][nb-l2-suggest] | Same: "Did you mean ...?" for case/punctuation differences, never auto-runs |
| ↳ [`prepare_call()`](../nanoclaw/agent/tools/registry.py#L66) | [`:110`][nb-l2-prepare] | Simplified: unknown tool, non-object arguments, missing required keys. Same messages. No argument unwrapping, casting or full schema validation |
| ↳ [`execute()`](../nanoclaw/agent/tools/registry.py#L119) | [`:187`][nb-l2-execute] | Same: never raises; errors get the retry hint. (The runner uses `prepare_call` + its own execution, like nanobot's runner) |
| [`RETRY_HINT`](../nanoclaw/agent/tools/registry.py#L11), [`with_retry_hint`](../nanoclaw/agent/tools/registry.py#L14) | [`agent/tools/execution.py:23`][nb-l2-hint], `:50` `_with_retry_hint` | Same text (since layer 11: nanobot `c4dac64` `registry.py:14` `_ERROR_HINT`); added once |

### Runner — `nanoclaw/agent/runner.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`AgentRunSpec`](../nanoclaw/agent/runner.py#L50) | [`agent/runner.py:89`][nb-l2-spec] | Simplified: 10 fields instead of ~25 (no hooks, callbacks or checkpoints). `workspace` (layer 3), `max_tokens`/`temperature` and `session_key` (for the tool-result folder) added since |
| [`AgentRunResult`](../nanoclaw/agent/runner.py#L75) | [`runner.py:119`][nb-l2-result] | Simplified: `final_content`, `messages`, `tools_used`, `usage`, `stop_reason`, `error` |
| [`AgentRunner.run()`](../nanoclaw/agent/runner.py#L104) | [`runner.py:284`][nb-l2-run] `run` + [`:368`][nb-l2-core] `_run_core` | Simplified: the same loop and stop rules, ~75 lines instead of ~520 |
| ↳ tool branch | [`runner.py:487`][nb-l2-toolbranch] | Simplified: sequential, no checkpoints or hooks |
| ↳ ignore calls in cut-off replies | [`runner.py:579`][nb-l2-ignore] | Same |
| [`_request_model()`](../nanoclaw/agent/runner.py#L223) | [`runner.py:852`][nb-l2-reqmodel] `_request_model` | Simplified: the malformed-call recovery only (retry with note, then no tools); no streaming, governance or compaction |
| [`_request()`](../nanoclaw/agent/runner.py#L203) | [`runner.py:864`][nb-l5-prepreq], [`:1214`][nb-l5-prepreq2] (`context_governor.prepare_request`) | Simplified: one provider call; since layer 5 adjacent user messages are merged first, as in nanobot; no token budgets |
| [`_drop_malformed_tool_calls()`](../nanoclaw/agent/runner.py#L250) | [`runner.py:1068`][nb-l2-dropmal] | Same |
| ↳ empty-reply retry + finalization | [`runner.py:592`][nb-l2-empty], [`:1126`][nb-l2-finretry] `_request_finalization_retry` | Same rule (`_MAX_EMPTY_RETRIES = 2`) |
| ↳ model error | [`runner.py:714`][nb-l2-error] | Same, minus the arrearage message |
| ↳ empty final response | [`runner.py:737`][nb-l2-emptyfinal] | Same |
| [`_execute_tool_call()`](../nanoclaw/agent/runner.py#L267) | [`agent/tools/execution.py:57`][nb-l2-exec] `execute_tool_calls`, [`:115`][nb-l2-exec1] `_execute_tool_call` | Simplified: `prepare_call`, run, `Error: <Type>: <msg>` on exceptions, same as nanobot. No batching, throttling or violation classification |
| ↳ empty-output marker + keeping results in budget | [`context_governance.py:716`][nb-l2-normalize] `normalize_tool_result` | Same since the layer 3 review: see [`normalize_tool_result`](../nanoclaw/agent/tool_results.py#L132) under Layer 3 |
| [`_finalize_after_max_iterations()`](../nanoclaw/agent/runner.py#L303) | [`runner.py:1154`][nb-l2-finmax], [`:1249`][nb-l2-maxfallback] `_max_iterations_fallback` | Same |
| [`_append_model_error_placeholder()`](../nanoclaw/agent/runner.py#L325) | [`runner.py:1361`][nb-l2-placeholder] | Same |
| [`_add_usage()`](../nanoclaw/agent/runner.py#L90) | `runner.py:1326` `_merge_usage` | Simplified |

### Messages and fixed texts — `nanoclaw/agent/messages.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`build_assistant_message`](../nanoclaw/agent/messages.py#L57) | [`utils/helpers.py:753`][nb-l2-bam] | Simplified: no thinking blocks |
| [`build_tool_message`](../nanoclaw/agent/messages.py#L70) | inline in `runner.py:538` | New helper, same shape |
| [`truncate_text`](../nanoclaw/agent/messages.py#L74), `TRUNCATED_SUFFIX` | [`utils/helpers.py:400`][nb-l2-trunc], `:369` | Same |
| [`strip_think`](../nanoclaw/agent/messages.py#L87) | [`utils/helpers.py:167`][nb-l2-think] `strip_think`, `:291` `extract_reasoning` | Simplified: closed blocks + unclosed leading block. No malformed-tag, channel-marker or streaming-fragment cleanup |
| [`build_malformed_tool_call_note`](../nanoclaw/agent/messages.py#L50), `MALFORMED_TOOL_CALL_NOTE` | [`runner.py:1106`][nb-l2-malnote] `_malformed_tool_call_retry_messages` | Same text |
| `FINALIZATION_RETRY_PROMPT`, `BUDGET_EXHAUSTED_FINALIZATION_PROMPT`, `EMPTY_FINAL_RESPONSE_MESSAGE` | [`utils/runtime.py:19–34`][nb-l2-runtime] | Same text |
| `MODEL_ERROR_PLACEHOLDER`, `DEFAULT_ERROR_MESSAGE` | [`runner.py:67–72`][nb-l2-consts] | Same text |
| `MAX_ITERATIONS_MESSAGE` | [`templates/agent/max_iterations_message.md`][nb-l2-maxtpl] | Same text (Python format instead of Jinja) |

## Layer 3 — Tools

Note: [`notes/03-tools.md`](notes/03-tools.md)

### Parameter checking — `nanoclaw/agent/tools/schema.py`, `registry.py`, `base.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`object_schema`](../nanoclaw/agent/tools/schema.py#L27) | [`agent/tools/schema.py:217`][nb-l3-tps] `tool_parameters_schema` | Simplified: plain dicts instead of `StringSchema`/`IntegerSchema` classes; same `additionalProperties: false` default |
| [`validate_json_schema_value`](../nanoclaw/agent/tools/schema.py#L57) | [`agent/tools/base.py:51`][nb-l3-validate] `Schema.validate_json_schema_value` | Same rules and messages |
| [`cast_params`](../nanoclaw/agent/tools/schema.py#L123), `_cast_value` | [`base.py:256`][nb-l3-cast] `Tool.cast_params`, `:263` `_cast_value` | Same |
| [`Tool.cast_params()`](../nanoclaw/agent/tools/base.py#L57), `validate_params()` | [`base.py:256`][nb-l3-cast], `:304` | Same |
| [`_coerce_params`](../nanoclaw/agent/tools/registry.py#L107) | [`agent/tools/registry.py:171`][nb-l3-coerce] | Same: JSON string → object; `{"arguments": ...}` unwrapped |
| [`prepare_call()`](../nanoclaw/agent/tools/registry.py#L66) | [`registry.py:110`][nb-l2-prepare] | Now the same: unknown tool → non-object → cast → validate |

### File tools — `nanoclaw/agent/tools/filesystem.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`_FsTool`](../nanoclaw/agent/tools/filesystem.py#L24) | [`agent/tools/filesystem.py:34`][nb-l3-fstool] | Simplified: workspace-relative paths; restriction added in layer 9 (see below); no extra allowed dirs |
| [`ReadFileTool`](../nanoclaw/agent/tools/filesystem.py#L141) | [`filesystem.py:270`][nb-l3-read] | Simplified: text only (no PDF/Office/images), no unchanged-file check. Same numbering, footers, limits, errors, BOM and Latin-1 decoding |
| [`_decode_bom_text`](../nanoclaw/agent/tools/filesystem.py#L129), `_TEXT_EXTENSIONS` | [`utils/document.py:607`][nb-l3-bom], `:643` `_is_text_extension` | Same |
| [`_is_blocked_device`](../nanoclaw/agent/tools/filesystem.py#L107) | [`filesystem.py:210`][nb-l3-device] | Same |
| [`WriteFileTool`](../nanoclaw/agent/tools/filesystem.py#L259) | [`filesystem.py:557`][nb-l3-write] | Same |
| [`EditFileTool`](../nanoclaw/agent/tools/filesystem.py#L511) | [`filesystem.py:877`][nb-l3-edit] | Simplified: no `line_hint` or `expected_replacements` |
| [`_reindent_like_match`](../nanoclaw/agent/tools/filesystem.py#L352), `_preserve_quote_style`, `_curly_double_quotes`, `_curly_single_quotes` | `filesystem.py` (same names) | Same (added in the layer 3 review) |
| [`_find_matches`](../nanoclaw/agent/tools/filesystem.py#L446) (+ `_find_exact_matches`, `_find_trim_matches`, `_find_quote_matches`) | [`filesystem.py:793`][nb-l3-match] | Same strategies, same order |
| [`_best_window`](../nanoclaw/agent/tools/filesystem.py#L478), `_diagnose_near_match`, `_not_found_msg`, `_file_not_found_msg` | [`filesystem.py:827`][nb-l3-best], `:1073`, `:1059` | Same |
| [`_count_changed_lines`](../nanoclaw/agent/tools/filesystem.py#L494) | `utils/file_edit_events.py:59` `FileDiff.from_text` | Simplified: `difflib` instead of `rapidfuzz` |
| [`ListDirTool`](../nanoclaw/agent/tools/filesystem.py#L692) | [`filesystem.py:1115`][nb-l3-list] | Same, except ignore rules only look below the listed folder (difference #3 in the note) |

### Shell — `nanoclaw/agent/tools/shell.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`ExecTool`](../nanoclaw/agent/tools/shell.py#L62) | [`agent/tools/shell.py:162`][nb-l3-exec] | *(Layer 16: rewritten on exec sessions with Windows support; see the layer 16 section)* Phase 1: bash only, no sessions |
| [`DEFAULT_DENY_PATTERNS`](../nanoclaw/agent/tools/shell.py#L32) | [`shell.py:215`][nb-l3-deny] | Same list as nanobot, including the memory-file patterns (`history.jsonl`, `.dream_cursor`) added at the end of layer 6 (checked: identical) |
| [`_guard_command`](../nanoclaw/agent/tools/shell.py#L197) | [`shell.py:840`][nb-l3-guard], called at [`:446`][nb-l3-guardcall] | **Different: deny list and internal-URL check always applied** (nanobot: only when `restrict_to_workspace` is on, off by default). Since layer 9 also nanobot's internal-URL and restricted-mode path checks (see the layer 9 section); no allowlist |
| [`_build_env`](../nanoclaw/agent/tools/shell.py#L405) | [`shell.py:791`][nb-l3-env] | Same (Unix) |
| [`_kill_process_tree`](../nanoclaw/agent/tools/shell.py#L420) | [`shell.py:730`][nb-l3-kill] | Simplified: `killpg` on the process group, then `process.kill()` |
| [`_resolve_timeout`](../nanoclaw/agent/tools/shell.py#L185) | [`shell.py:386`][nb-l3-timeout] | Same: model's choice capped at 600 s; configured default may be higher; 0 = no limit |

### Web — `nanoclaw/agent/tools/web.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`WebFetchTool`](../nanoclaw/agent/tools/web.py#L67) | [`agent/tools/web.py:1118`][nb-l3-fetch] | Simplified: readability path only; no Jina Reader or image fetching. Private-network checks, DNS pinning and environment proxies added in layer 9 |
| [`_get_with_redirects`](../nanoclaw/agent/tools/web.py#L146) | [`web.py:208`][nb-l3-redirects] `_get_with_safe_redirects` | Same checks on every hop since layer 9 (scheme, resolved addresses, pinned connection) |
| `_strip_tags`, `_normalize`, `_to_markdown`, `_extract_readable_html` | `web.py:86`, `:94`, `:1338`, `:1330` | Same. (`_validate_url`, nanobot `web.py:100`, was replaced by `_validate_url_safe` in layer 9 and removed; nanobot keeps it for search providers) |
| [`UNTRUSTED_BANNER`](../nanoclaw/agent/tools/web.py#L29) | `web.py:34` | Same text |
| [`WebSearchTool`](../nanoclaw/agent/tools/web.py#L224) | [`web.py:365`][nb-l3-search] | Simplified: DuckDuckGo only (nanobot's default of ~10 providers) |
| [`_format_results`](../nanoclaw/agent/tools/web.py#L202), `_duckduckgo_search` | [`web.py:306`][nb-l3-format], `:1024` | Same |

### Large tool results — `nanoclaw/agent/tool_results.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`normalize_tool_result`](../nanoclaw/agent/tool_results.py#L132) | [`agent/context_governance.py:716`][nb-l2-normalize] `ContextGovernor.normalize_tool_result` | Same order: empty marker → `read_file` exempt → truncate (no workspace) or save |
| [`OFFLOAD_EXEMPT_TOOLS`](../nanoclaw/agent/tool_results.py#L28) | [`context_governance.py:68`][nb-l3-exempt] `TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS` | Same: `{"read_file"}` |
| [`maybe_persist_tool_result`](../nanoclaw/agent/tool_results.py#L88) | [`utils/helpers.py:624`][nb-l3-persist] | Same folder per session and clean-up (since the end of layer 6). **Different file name:** `<id>-<content hash>.txt`, fixing nanobot's stale-output bug ([`UPSTREAM_BUGS.md`](UPSTREAM_BUGS.md) #3) |
| [`cleanup_tool_result_buckets`](../nanoclaw/agent/tool_results.py#L69) | [`utils/helpers.py:540`][nb-l6-cleanup] `_cleanup_tool_result_buckets` | Same: other sessions' folders removed after 7 days and beyond 32 |
| [`render_tool_result_reference`](../nanoclaw/agent/tool_results.py#L42), `safe_filename`, `ensure_nonempty_tool_result` | `utils/helpers.py:511`, `:372`; `utils/runtime.py:48` | Same text |

### Default tool set — `nanoclaw/agent/tools/__init__.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`build_default_tools`](../nanoclaw/agent/tools/__init__.py#L11) | [`agent/tools/loader.py:26`][nb-l3-loader] `ToolLoader` + `agent/loop.py:608` | Replaced: an explicit list instead of package scanning and plugins |

## Layer 4 — Context

Note: [`notes/04-context.md`](notes/04-context.md)

### System prompt and messages — `nanoclaw/agent/context.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`ContextBuilder`](../nanoclaw/agent/context.py#L29) | [`agent/context.py:89`][nb-l4-cb] | Simplified: one workspace folder; skills since layer 7 |
| ↳ `BOOTSTRAP_FILES`, `_SKIPPABLE_DEFAULTS` | [`context.py:92`][nb-l4-boot] | Same |
| ↳ [`build_system_prompt()`](../nanoclaw/agent/context.py#L45) | [`context.py:101`][nb-l4-bsp] | Same order and separator; memory and summary sections since layer 6, skills sections since layer 7 (whole prompt checked identical) |
| ↳ [`_get_identity()`](../nanoclaw/agent/context.py#L78) | [`context.py:154`][nb-l4-id] | Same; agent and project path are the same folder |
| ↳ [`_load_bootstrap_files()`](../nanoclaw/agent/context.py#L96) | [`context.py:194`][nb-l4-load] | Same rules, minus the legacy `SOUL.md` upgrade |
| ↳ [`_is_template_content()`](../nanoclaw/agent/context.py#L111) | [`context.py:224`][nb-l4-istpl] | Same |
| ↳ [`build_messages()`](../nanoclaw/agent/context.py#L140) | [`context.py:231`][nb-l4-bm] (+ `build_transcript`) | Simplified: merges a trailing user message (and its `_meta`); no media. Since layer 6: `current_message=None` and `session_summary` |
| ↳ [`build_current_message()`](../nanoclaw/agent/context.py#L118) | [`context.py:310`][nb-l4-bcm] | Simplified: text + runtime blocks, with the `_meta` marker, and since layer 7 the `$skill` block; no images |
| ↳ [`_merge_message_content()`](../nanoclaw/agent/context.py#L172) | [`context.py:172`][nb-l4-merge] | Same |

### Runtime context — `nanoclaw/runtime_context.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`RuntimeContextBlock`](../nanoclaw/runtime_context.py#L26) | [`runtime_context.py:25`][nb-l4-rcb] | Same |
| [`RUNTIME_CONTEXT_TAG`](../nanoclaw/runtime_context.py#L21), `RUNTIME_CONTEXT_END` | `runtime_context.py:17` | Same text |
| [`wrap_runtime_context_lines`](../nanoclaw/runtime_context.py#L36) | [`runtime_context.py:55`][nb-l4-wrap] | Same |
| [`append_runtime_context`](../nanoclaw/runtime_context.py#L44) | [`runtime_context.py:120`][nb-l4-append] | Same: returns the content and a marker of what was appended |
| [`detach_runtime_context`](../nanoclaw/runtime_context.py#L66), [`reattach_runtime_context`](../nanoclaw/runtime_context.py#L96) | [`runtime_context.py:148`][nb-l5-detach], [`:184`][nb-l5-reattach] | Same (added in layer 5) |
| [`current_time_block`](../nanoclaw/runtime_context.py#L136) | — | **New:** nanobot gives the model no date/time (difference #1 in the note) |

### Templates — `nanoclaw/utils/templates.py`, `nanoclaw/templates/`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`render_template`](../nanoclaw/utils/templates.py#L36), `_environment` | [`utils/prompt_templates.py:28`][nb-l4-render], `:18` | Same Jinja2 settings |
| [`load_bundled_template`](../nanoclaw/utils/templates.py#L42) | [`utils/helpers.py:1010`][nb-l4-loadtpl] | Same (reads from the package folder) |
| [`sync_workspace_templates`](../nanoclaw/utils/templates.py#L48) | [`utils/helpers.py:956`][nb-l4-sync] | Simplified: `AGENTS.md`, `SOUL.md`, `USER.md`, `memory/MEMORY.md`, `memory/history.jsonl`, `skills/`; no `HEARTBEAT.md`, `prompts/` or git store |
| `templates/agent/identity.md`, `platform_policy.md`, `tool_contract.md`, `_snippets/untrusted_content.md` | `templates/agent/` (same names) | Same text except: name, image line removed ("Dream" wording restored in layer 6) |
| `templates/SOUL.md`, `USER.md` | `templates/` (same names) | Same text, renamed |
| `templates/AGENTS.md` | `templates/AGENTS.md` | Same opening; cron/heartbeat/`apply_patch` guidance replaced by `edit_file`/`write_file` guidance |

## Layer 5 — Sessions + AgentLoop

Note: [`notes/05-sessions.md`](notes/05-sessions.md)

### Session — `nanoclaw/session/manager.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`Session`](../nanoclaw/session/manager.py#L87) | [`session/manager.py:172`][nb-l5-session] | Simplified: no provider state or session policy; `last_archived` without the legacy `last_consolidated` alias |
| ↳ [`add_message()`](../nanoclaw/session/manager.py#L107) | [`manager.py:208`][nb-l5-add] | Same |
| ↳ [`get_history()`](../nanoclaw/session/manager.py#L136) | [`manager.py:240`][nb-l5-hist] | Same output (checked by running both), including the summary marker, `max_tokens` limit and reply cleanup (layer 6); no `max_messages`, image or CLI-app breadcrumbs |
| ↳ [`clear()`](../nanoclaw/session/manager.py#L193) | [`manager.py:374`][nb-l5-clear] | Same, minus the summary and provider state |
| [`find_legal_message_start`](../nanoclaw/session/manager.py#L64) | [`utils/helpers.py:477`][nb-l5-legal] | Same |

### Storage — `nanoclaw/session/manager.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`SessionManager`](../nanoclaw/session/manager.py#L201) | [`SessionManager`][nb-l5-mgr] + [`JsonlSessionStore`][nb-l5-store] | Simplified: one class; no file locks, runtime checkpoints, migration or session listing |
| ↳ `__init__` | [`manager.py:448`][nb-l5-init] | Same rule: storage inside the workspace is refused; root folder made private |
| ↳ [`_load_or_create_workspace_id()`](../nanoclaw/session/manager.py#L229) | [`manager.py:568`][nb-l5-wsid], [`:517`][nb-l5-wsread] | Same file and format (`<workspace>/.nanoclaw/workspace-id`, 32 hex chars); no recovery of a lost ID |
| ↳ [`storage_key()`](../nanoclaw/session/manager.py#L259), [`get_session_path()`](../nanoclaw/session/manager.py#L263) | [`manager.py:903`][nb-l5-skey], [`:923`][nb-l5-path] | Same file names (base64 of the key) |
| ↳ [`get_or_create()`](../nanoclaw/session/manager.py#L268), [`invalidate()`](../nanoclaw/session/manager.py#L276), [`_remember()`](../nanoclaw/session/manager.py#L280) | [`manager.py:1613`][nb-l5-goc], [`:1738`][nb-l5-inv], [`:1538`][nb-l5-rem] | Simplified: one LRU cache of 128 (nanobot also keeps evicted sessions weakly) |
| ↳ [`save()`](../nanoclaw/session/manager.py#L288) | [`manager.py:1207`][nb-l5-save] | Same format (metadata line, then messages); atomic temp-file + rename; no fsync option |
| ↳ [`_load()`](../nanoclaw/session/manager.py#L311) | [`manager.py:935`][nb-l5-load], [`repair`][nb-l5-repair] | Simplified: always skips unreadable lines (nanobot's `repair` fallback), without logging. Files are readable by both projects (checked) |

### AgentLoop — `nanoclaw/agent/loop.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`AgentLoop`](../nanoclaw/agent/loop.py#L79) | [`agent/loop.py`][nb-l5-loop] `AgentLoop` | Simplified: direct path only; no bus, channels, queues, hooks, cancellation or checkpoint recovery. Takes `temperature` and `max_tokens` |
| ↳ [`process_direct()`](../nanoclaw/agent/loop.py#L139) | [`loop.py:2498`][nb-l5-pd] → `_build_turn` [`:2046`][nb-l5-build], `_run_turn` [`:2155`][nb-l5-run], `_persist_turn` [`:2195`][nb-l5-persist] | Same stages and order, including closing a crashed turn first and the `pending_user_turn` flag (set when the user message is saved, cleared after the turn); returns the `AgentRunResult` instead of an `OutboundMessage` |
| ↳ [`_persist_user_message_early()`](../nanoclaw/agent/loop.py#L426) | [`loop.py:663`][nb-l5-early] | Same: saved with its runtime context and the `_runtime_context` marker |
| ↳ [`_save_turn()`](../nanoclaw/agent/loop.py#L470) | [`loop.py:2320`][nb-l5-saveturn] | Same rules (empty assistant skipped, unknown/duplicate tool results dropped, timestamps); no latency, image redaction or summary checkpoints |
| ↳ [`new_session()`](../nanoclaw/agent/loop.py#L355) | [`command/builtin.py:314`][nb-l5-new] `/new` | Since layer 6: archives the old messages to the journal first, then clears (see the layer 6 table) |
| ↳ [`_get_session_lock()`](../nanoclaw/agent/loop.py#L423) | [`loop.py:2562`][nb-l5-lock] | Same: one `asyncio.Lock` per session key |

### Closing a crashed turn — `nanoclaw/session/recovery.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`restore_pending_interruption`](../nanoclaw/session/recovery.py#L20) | [`session/recovery.py:384`][nb-l5-restore] | Same message and marker (`_recovery_interrupted`); no "superseded" variant |
| [`PENDING_USER_TURN_KEY`](../nanoclaw/session/recovery.py#L16) | [`recovery.py:34`][nb-l5-pendkey], set/cleared in [`loop.py:2488`][nb-l5-mark] | Same key |

### Model-facing merge — `nanoclaw/agent/context_governance.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`merge_adjacent_user_messages_for_model`](../nanoclaw/agent/context_governance.py#L42) | [`agent/context_governance.py:282`][nb-l5-merge] `_merge_adjacent_user_messages_for_model` | Same output (checked on 9 cases); no summary or provider-state boundary checks, which don't exist yet |
| [`_merge_message_content`](../nanoclaw/agent/context_governance.py#L22) | [`context_governance.py:263`][nb-l5-mergec] | Same |

## Layer 6 — Memory and compaction

Note: [`notes/06-memory.md`](notes/06-memory.md)

### Memory files and journal — `nanoclaw/agent/memory.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`MemoryStore`](../nanoclaw/agent/memory.py#L60) | [`agent/memory.py:59`][nb-l6-store] | Simplified: no git store, legacy `HISTORY.md` migration, append lock or warnings |
| ↳ [`read_memory()`](../nanoclaw/agent/memory.py#L86), [`write_memory()`](../nanoclaw/agent/memory.py#L89) | [`memory.py:230`][nb-l6-readmem] | Same (no `read_soul`/`read_user` helpers; not needed) |
| ↳ [`append_history()`](../nanoclaw/agent/memory.py#L100), [`_normalize_history_entry()`](../nanoclaw/agent/memory.py#L94) | [`:283`][nb-l6-append], [`:260`][nb-l6-norm] | Same: `<think>` removed, 64,000-char cap, cursor + `.cursor` file |
| ↳ [`_next_cursor()`](../nanoclaw/agent/memory.py#L148), [`read_unprocessed_history()`](../nanoclaw/agent/memory.py#L161) | [`:384`][nb-l6-next], [`:402`][nb-l6-unproc] | Same (checked) |
| ↳ [`compact_history()`](../nanoclaw/agent/memory.py#L164) | [`:406`][nb-l6-compacth] | Same: keep 1,000, never drop entries Dream hasn't read |
| ↳ [`build_dream_prompt()`](../nanoclaw/agent/memory.py#L238) | [`:528`][nb-l6-dreamprompt] | Same (20 entries, 1,000 chars each); no `prompts/dream.md` override |
| ↳ [`build_dream_tools()`](../nanoclaw/agent/memory.py#L252) + [`_PathGuard`](../nanoclaw/agent/memory.py#L357) | [`:560`][nb-l6-dreamtools] | Same since layer 7: read the workspace and built-in skills; write `skills/` and `MEMORY.md`/`SOUL.md`/`USER.md`; nanobot's error text. No `apply_patch` |
| ↳ [`raw_archive()`](../nanoclaw/agent/memory.py#L318), [`_build_raw_checkpoint()`](../nanoclaw/agent/memory.py#L340), [`_format_messages()`](../nanoclaw/agent/memory.py#L303) | [`:650`][nb-l6-raw], [`:685`][nb-l6-rawcp], [`:628`][nb-l6-fmt] | Same output (checked); no image breadcrumbs |
| ↳ [`dream_session_key()`](../nanoclaw/agent/memory.py#L296) | [`:704`][nb-l6-dreamkey] | Same |

### Archiving and compaction — `nanoclaw/agent/memory.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`MemoryArchiver`](../nanoclaw/agent/memory.py#L393) | [`memory.py:763`][nb-l6-arch] | Simplified: no provider conversation state |
| ↳ [`archive()`](../nanoclaw/agent/memory.py#L476) | [`:836`][nb-l6-archive] | Same rules: archive prompt appended to the request, size check, one retry after refusing tool calls, raw fallback, `(nothing)` not journaled. With `NANOCLAW_COMPACTION=fixed` a request that doesn't fit goes to `_archive_in_parts()` (below) |
| ↳ [`archive_session()`](../nanoclaw/agent/memory.py#L641) | [`:1011`][nb-l6-archsess] | Same (never summarizes only part of the stretch); with `fixed`, a stretch that doesn't fit is summarized in parts |
| ↳ [`_raw_checkpoint()`](../nanoclaw/agent/memory.py#L415), [`_combine_raw_checkpoint()`](../nanoclaw/agent/memory.py#L446) | [`:783`][nb-l6-rawck], [`:805`][nb-l6-combine] | Same (checked) |
| [`Consolidator`](../nanoclaw/agent/memory.py#L684) | [`memory.py:1087`][nb-l6-cons] | Simplified: no locks or events |
| ↳ [`summarize_transcript()`](../nanoclaw/agent/memory.py#L704) | [`:1118`][nb-l6-summtr] | Same budgets (input = window − max tokens; raw fallback ≤ half of it) |
| ↳ [`estimate_session_prompt_tokens()`](../nanoclaw/agent/memory.py#L736), [`_input_token_budget()`](../nanoclaw/agent/memory.py#L748) | [`:1192`][nb-l6-estsess], [`:1218`][nb-l6-inbudget] | Same |
| ↳ [`compact_idle_session()`](../nanoclaw/agent/memory.py#L756) | [`:1241`][nb-l6-idle] | Same result (checked via `/compact`); no events |

### Compaction A/B (bug 4) — `nanoclaw/agent/memory.py`, `scripts/compare_compaction.py`

`NANOCLAW_COMPACTION=nanobot` (default) runs the rows above unchanged;
`fixed` is NanoClaw's opt-in fix for [upstream bug 4](UPSTREAM_BUGS.md#4-compaction-usually-falls-back-to-a-raw-dump),
being A/B tested (`docs/notes/06-memory.md`, "Compaction A/B").

| NanoClaw | nanobot | How it compares |
|---|---|---|
| `COMPACTION_MODES`, `MemoryArchiver(compaction=)`, `Consolidator(compaction=)`, `AgentLoop(compaction=)`, [`_compaction()`](../nanoclaw/__main__.py#L55) | — | NanoClaw only: the switch (`NANOCLAW_COMPACTION`) |
| ↳ [`_archive_in_parts()`](../nanoclaw/agent/memory.py#L568) | — (nanobot: raw fallback) | `fixed`: split into parts that each fit, summarize each, journal oldest first, join |
| ↳ [`_ask_for_summary()`](../nanoclaw/agent/memory.py#L531) | part of [`archive()`][nb-l6-archive] | The model call of `archive()` moved into a helper, unchanged |
| ↳ [`_fit_raw()`](../nanoclaw/agent/memory.py#L435), `raw_archive(keep_end=)`, [`truncate_text_to_tokens_keep_end`](../nanoclaw/utils/tokens.py#L159) | — | `fixed`: the raw dump keeps the newest part |
| [`scripts/compare_compaction.py`](../scripts/compare_compaction.py) | — | The A/B harness: same conversation in both modes, quiz, Dream, report |

### Compaction in the loop — `nanoclaw/agent/loop.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`_compact_if_needed()`](../nanoclaw/agent/loop.py#L297) | [`context_governance.py:617`][nb-l6-prepreq] `prepare_request`, [`:538`][nb-l6-compreq] `_compact_request_history`, [`:414`][nb-l6-pressure] `request_pressure` | **Different:** checked once before the turn; nanobot checks before every request (difference #1). Same budget, summary transcript and save position (checked) |
| [`_input_budget()`](../nanoclaw/agent/loop.py#L290), `CONTEXT_SAFETY_BUFFER` | [`context_governance.py:700`][nb-l6-inputbudget], [`providers/base.py:39`][nb-l6-buffer] | Same |
| [`ContextWindowExceededError`](../nanoclaw/agent/loop.py#L65) | [`context_governance.py:75`][nb-l6-cwee] | Same message |
| `DEFAULT_CONTEXT_WINDOW_TOKENS` | [`config/schema.py:127`][nb-l6-window] | **Different:** 128,000 instead of 200,000 (difference #2) |
| ↳ [`new_session()`](../nanoclaw/agent/loop.py#L355) | [`command/builtin.py:314`][nb-l5-new] `cmd_new` | Same request and journal entry (checked); waits instead of running in the background |
| ↳ [`compact_session()`](../nanoclaw/agent/loop.py#L381) | [`command/builtin.py:349`][nb-l6-compact] `cmd_compact` | Same |
| ↳ [`run_dream()`](../nanoclaw/agent/loop.py#L389) | [`command/builtin.py:453`][nb-l6-dream] `cmd_dream` | Same flow and messages; file comparison instead of a git diff; no schedule |
| ↳ `process_direct(ephemeral=, tools=)` | [`loop.py:2498`][nb-l5-pd] | Same: a temporary turn saves nothing, writes no spill files, and is checked against the context window without journaling |

### Sessions, prompt and helpers

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`Session.commit_summary_checkpoint()`](../nanoclaw/session/manager.py#L114) | [`session/manager.py:219`][nb-l6-commit] | Same (checked) |
| [`_sanitize_assistant_replay_text`](../nanoclaw/session/manager.py#L47) | [`session/manager.py:101`][nb-l6-sanitize] | Same (ported during the layer 6 review) |
| [`SUMMARY_CONTINUATION_TEXT`](../nanoclaw/session/summary.py#L16), [`is_summary_checkpoint`](../nanoclaw/session/summary.py#L29), [`session_summary_from_metadata`](../nanoclaw/session/summary.py#L34) | [`session/summary.py:12`][nb-l6-summary], [`:16`][nb-l6-issumm], [`:37`][nb-l6-summmeta] | Same |
| [`HIDDEN_HISTORY_META`](../nanoclaw/session/summary.py#L15), [`is_hidden_history_message`](../nanoclaw/session/summary.py#L24) | [`session/history_visibility.py:10`][nb-l6-hidden], [`:20`][nb-l6-ishidden] | Simplified: no automation messages |
| `build_system_prompt` memory + summary sections | [`agent/context.py:127`][nb-l6-ctxmem] | Same text (checked) |
| [`estimate_prompt_tokens`](../nanoclaw/utils/tokens.py#L60), [`estimate_message_tokens`](../nanoclaw/utils/tokens.py#L87), [`truncate_text_to_tokens`](../nanoclaw/utils/tokens.py#L131) | [`utils/helpers.py:774`][nb-l6-esttok], [`:840`][nb-l6-estmsg], [`:407`][nb-l6-trunctok] | Same counts (checked with tiktoken and bytes); no per-tools cache |
| [`get_token_encoding`](../nanoclaw/utils/tokens.py#L25) | [`utils/token_encoding.py:44`][nb-l6-enc] | Simplified: loaded on first use, not in a background thread |
| [`LLMRuntime`](../nanoclaw/utils/llm_runtime.py#L16) | [`utils/llm_runtime.py:15`][nb-l6-runtime] | Simplified: flat fields, no presets |
| [`public_history_message`](../nanoclaw/runtime_context.py#L111) | [`runtime_context.py:215`][nb-l6-public] | Same |
| `templates/memory/MEMORY.md`, `agent/consolidator_archive.md`, `agent/dream.md` | `templates/` (same names) | Same text (`dream.md` complete since layer 7) |

## Layer 7 — Skills

Note: [`notes/07-skills.md`](notes/07-skills.md)

### Skills loader — `nanoclaw/agent/skills.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`BUILTIN_SKILLS_DIR`](../nanoclaw/agent/skills.py#L27) | [`agent/skills.py:15`][nb-l7-builtin] | Same idea: `nanoclaw/skills/` |
| [`parse_skill_metadata`](../nanoclaw/agent/skills.py#L35), [`valid_skill_metadata`](../nanoclaw/agent/skills.py#L48) | [`skills.py:26`][nb-l7-parse], [`:39`][nb-l7-valid] | Same (PyYAML `safe_load`) |
| [`SkillsLoader`](../nanoclaw/agent/skills.py#L60) | [`skills.py:51`][nb-l7-loader] | Simplified: no plugin skills, CLI-app aliases or `disabled_skills` |
| ↳ [`list_skills()`](../nanoclaw/agent/skills.py#L84), [`_skill_entries_from_dir()`](../nanoclaw/agent/skills.py#L69) | [`:90`][nb-l7-list], [`:74`][nb-l7-entries] | Same order and precedence (workspace wins), unsorted as in nanobot |
| ↳ [`load_skill()`](../nanoclaw/agent/skills.py#L99), [`load_skills_for_context()`](../nanoclaw/agent/skills.py#L104) | [`:132`][nb-l7-load], [`:148`][nb-l7-loadctx] | Same |
| ↳ [`get_explicitly_invoked_skills()`](../nanoclaw/agent/skills.py#L113), [`build_explicit_skill_runtime_context()`](../nanoclaw/agent/skills.py#L125) | [`:165`][nb-l7-explicit], [`:182`][nb-l7-block] | Same (`$name`, word start, available only, always-on excluded) |
| ↳ [`build_skills_summary()`](../nanoclaw/agent/skills.py#L139) | [`:204`][nb-l7-summary] | Same text; no plugin group or absolute roots for a separate project folder |
| ↳ [`_requirement_lists()`](../nanoclaw/agent/skills.py#L173), [`_get_missing_requirements()`](../nanoclaw/agent/skills.py#L184), [`_check_requirements()`](../nanoclaw/agent/skills.py#L190), [`get_skill_availability()`](../nanoclaw/agent/skills.py#L194) | [`:266`][nb-l7-reqs], [`:277`][nb-l7-missing], [`:338`][nb-l7-check], [`:285`][nb-l7-avail] | Same (no `get_skill_requirements`; not used) |
| ↳ [`_parse_nanobot_metadata()`](../nanoclaw/agent/skills.py#L215) | [`:318`][nb-l7-meta] | Same, plus a `nanoclaw` key checked first (difference #2) |
| ↳ [`get_always_skills()`](../nanoclaw/agent/skills.py#L239), [`get_skill_description()`](../nanoclaw/agent/skills.py#L202), [`_strip_frontmatter()`](../nanoclaw/agent/skills.py#L208), [`get_skill_metadata()`](../nanoclaw/agent/skills.py#L248) | [`:350`][nb-l7-always], [`:301`][nb-l7-desc], [`:309`][nb-l7-strip], [`:362`][nb-l7-getmeta] | Same |

### Using skills

| NanoClaw | nanobot | How it compares |
|---|---|---|
| `build_system_prompt` `# Active Skills` and `# Skills` sections | [`agent/context.py:132`][nb-l7-ctxactive], [`:138`][nb-l7-ctxsummary] | Same (whole prompt checked identical) |
| [`build_current_message()`](../nanoclaw/agent/context.py#L118) `$skill` block | [`context.py:310`][nb-l4-bcm] | Same, with the same "not already added" check |
| `process_direct` adds the `$skill` block before saving | [`agent/loop.py:762`][nb-l7-loopskill] | Same: the block is saved with the user message |
| [`_builtin_skill_read_path`](../nanoclaw/agent/tools/filesystem.py#L85) | [`agent/tools/filesystem.py:234`][nb-l7-readfb], used at [`:313`][nb-l7-readfbuse] | Same |
| [`MemoryStore.build_dream_tools()`](../nanoclaw/agent/memory.py#L252) | [`agent/memory.py:560`][nb-l6-dreamtools] | Same rules and error text (`WORKSPACE_BOUNDARY_NOTE`, [`security/workspace_policy.py:13`][nb-l7-note]); no `apply_patch` |
| [`MemoryStore.default_dream_prompt()`](../nanoclaw/agent/memory.py#L287) | [`agent/memory.py:503`][nb-l7-dreamprompt] | Same (no `prompts/dream.md` override) |
| `templates/agent/skills_section.md`, `templates/agent/dream.md` | same files | Same text |
| `nanoclaw/skills/` (6 skills + README) | `nanobot/skills/` (11) | Same text with the name changed, except two lines adapted to NanoClaw's tools (difference #4); 5 left out (difference #1) |
| [`nanoclaw skills`](../nanoclaw/__main__.py#L170) | [`command/builtin.py:954`][nb-l7-cmdskill] `/skill` | Same list (checked); since layer 11 it also marks unavailable skills, which the prompt no longer lists |
| `sync_workspace_templates` creates `skills/` | [`utils/helpers.py:983`][nb-l7-syncskills] | Same |

## Layer 8 — Subagents

Note: [`notes/08-subagents.md`](notes/08-subagents.md)

### Subagent manager — `nanoclaw/agent/subagent.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`SubagentStatus`](../nanoclaw/agent/subagent.py#L46) | [`agent/subagent.py:57`][nb-l8-status] | Simplified: phase, stop reason and error only (no live iteration/tool events/usage; those come from hooks) |
| [`SubagentResult`](../nanoclaw/agent/subagent.py#L61) | `InboundMessage` published at [`:527`][nb-l8-announce] | Replaced: a report on the manager's `results` queue instead of the message bus |
| [`SubagentManager`](../nanoclaw/agent/subagent.py#L76) | [`:99`][nb-l8-manager] | Simplified: no exec sessions, workspace scopes, usage sources or deprecated constructor arguments |
| ↳ [`spawn()`](../nanoclaw/agent/subagent.py#L122) | [`:249`][nb-l8-spawn] | Same reply, ID and label rules |
| ↳ [`run_inline()`](../nanoclaw/agent/subagent.py#L146) | [`:313`][nb-l8-inline] | Same (error result marked as a tool error) |
| ↳ [`_run_subagent()`](../nanoclaw/agent/subagent.py#L177) | [`:376`][nb-l8-run] | Same 4-slot queue (`queued` → `initializing`) |
| ↳ [`_run_admitted_subagent()`](../nanoclaw/agent/subagent.py#L197) | [`:405`][nb-l8-admitted] | Same messages, runner settings and results; no hook, checkpoints or compaction |
| ↳ [`_announce_result()`](../nanoclaw/agent/subagent.py#L258) | [`:527`][nb-l8-announce] | Same text (`subagent_announce.md`) |
| ↳ [`_build_subagent_prompt()`](../nanoclaw/agent/subagent.py#L284) | [`:572`][nb-l8-prompt] | Same (one workspace, so no separate agent-workspace line) |
| ↳ [`_build_tools()`](../nanoclaw/agent/subagent.py#L100) | [`:227`][nb-l8-buildtools] | Same idea: the standard tools, no `spawn` (difference #5) |
| ↳ [`cancel_by_session()`](../nanoclaw/agent/subagent.py#L307), [`close()`](../nanoclaw/agent/subagent.py#L320), [`get_running_count()`](../nanoclaw/agent/subagent.py#L328), [`statuses_for_session()`](../nanoclaw/agent/subagent.py#L300) | [`:595`][nb-l8-cancel], [`:606`][nb-l8-close], [`:615`][nb-l8-count] | Same (no exec sessions to close) |
| [`DEFAULT_MAX_CONCURRENT_SUBAGENTS`](../nanoclaw/agent/subagent.py#L39) = 4 | [`config/schema.py:131`][nb-l8-config] | Same |

### Tools — `nanoclaw/agent/tools/spawn.py`, `context.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`SpawnTool`](../nanoclaw/agent/tools/spawn.py#L15) | [`agent/tools/spawn.py:48`][nb-l8-spawntool] | Same schema, description and replies (checked) |
| ↳ [`execute()`](../nanoclaw/agent/tools/spawn.py#L63) | [`:81`][nb-l8-spawnexec] | Same; no workspace scope or origin message ID |
| [`RequestContext`](../nanoclaw/agent/tools/context.py#L20), [`bind_request_context`](../nanoclaw/agent/tools/context.py#L29), [`current_request_context`](../nanoclaw/agent/tools/context.py#L37) | [`agent/tools/context.py:30`][nb-l8-reqctx], [`:52`][nb-l8-bind], [`:70`][nb-l8-current] | Simplified: channel, session key and model settings only |

### Runner and loop

| NanoClaw | nanobot | How it compares |
|---|---|---|
| `AgentRunSpec.finalize_on_max_iterations`, `max_iterations_message`; [`_max_iterations_fallback()`](../nanoclaw/agent/runner.py#L320) | [`agent/runner.py:101`][nb-l8-runspec], [`:795`][nb-l8-finalize], [`:1249`][nb-l8-fallback] | Same (`error_message` not needed: same default) |
| `AgentLoop.__init__`: `self.subagents`, `spawn` in the main tools | `agent/loop.py:392`, tool loader | Same: main agent only |
| `_process_turn` binds the request context around the run and copies the iteration limit to the subagents | `_request_context_for_turn`, [`_sync_subagent_runtime_limits` :517][nb-l8-sync] | Simplified / Same |
| [`process_subagent_result()`](../nanoclaw/agent/loop.py#L163) | [`agent/loop.py:2066`][nb-l8-isub] (system turn) | Same request and saved rows (checked); runs after the main turn, not injected into it ([`:1095`][nb-l8-inject], difference #3) |
| [`_persist_subagent_followup()`](../nanoclaw/agent/loop.py#L446) | [`:2454`][nb-l8-persist] | Same; the summary boundary is recorded before it, so a repeated report compacts correctly (nanobot: [bug 5](UPSTREAM_BUGS.md#5-summary-marker-one-message-too-early-on-a-repeated-subagent-report)) |
| [`next_subagent_result()`](../nanoclaw/agent/loop.py#L181) | `bus.consume_inbound` | **New:** reads the report queue, waiting for running children |
| [`close()`](../nanoclaw/agent/loop.py#L195) | [`aclose()` :1639][nb-l8-aclose] | Simplified: cancels subagents |
| [`_deliver_subagent_results()`](../nanoclaw/__main__.py#L111) | [`cli/agent.py:190`][nb-l8-cliclose] cancels them | Different: `ask` waits and reports (difference #1) |
| `templates/agent/subagent_system.md`, `subagent_announce.md` | [same][nb-l8-tsys] [files][nb-l8-tann] | Same text ("NanoClaw's agent workspace") |

## Layer 9 — Safety

Note: [`notes/09-safety.md`](notes/09-safety.md)

### Private-network guard — `nanoclaw/security/network.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`_BLOCKED_NETWORKS`](../nanoclaw/security/network.py#L28), [`_normalize_addr`](../nanoclaw/security/network.py#L65), [`_is_private`](../nanoclaw/security/network.py#L80) | [`security/network.py:16`][nb-l9-blocked], [`:56`][nb-l9-norm] | Same |
| [`configure_ssrf_whitelist`](../nanoclaw/security/network.py#L49), [`configure_ssrf_whitelist_from_env`](../nanoclaw/security/network.py#L59) | [`:46`][nb-l9-wl], [`config/loader.py:141`][nb-l9-loader] | Same; read from `NANOCLAW_SSRF_WHITELIST` instead of the config (difference #3) |
| [`resolve_url_target`](../nanoclaw/security/network.py#L87), [`validate_url_target`](../nanoclaw/security/network.py#L126) | [`:78`][nb-l9-resolve], [`:148`][nb-l9-validate] | Same results (checked on 27 URLs × 3 whitelists); no `allow_loopback` / `trust_remote_dns` |
| [`httpx_env_proxy_mounts`](../nanoclaw/security/network.py#L132) | [`:174`][nb-l9-mounts] | Same |
| [`pin_resolved_url_dns`](../nanoclaw/security/network.py#L173), [`PinnedDNSAsyncTransport`](../nanoclaw/security/network.py#L222), [`UnsafeURLRequestError`](../nanoclaw/security/network.py#L218) | [`:215`][nb-l9-pin], [`:264`][nb-l9-pinned] | Same |
| [`contains_internal_url`](../nanoclaw/security/network.py#L249) | [`:323`][nb-l9-contains] | Same |

### Workspace boundary — `nanoclaw/security/workspace_policy.py`, file tools, `exec`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`resolve_allowed_path`](../nanoclaw/security/workspace_policy.py#L96), [`is_path_within`](../nanoclaw/security/workspace_policy.py#L44), [`WORKSPACE_BOUNDARY_NOTE`](../nanoclaw/security/workspace_policy.py#L13) | [`security/workspace_policy.py`][nb-l9-policy] ([`:96`][nb-l9-allowed], [`:44`][nb-l9-within]) | Same file, copied unchanged (the note moved here from `memory.py`) |
| [`_FsTool`](../nanoclaw/agent/tools/filesystem.py#L24) `__init__` (`restrict_to_workspace`), [`_resolve_read()`](../nanoclaw/agent/tools/filesystem.py#L46), [`_resolve_write()`](../nanoclaw/agent/tools/filesystem.py#L57) | [`agent/tools/filesystem.py:84`][nb-l9-fscreate], [`:151`][nb-l9-rread], [`:182`][nb-l9-rwrite] | Same rules (built-in and workspace skills, the journal); no media folder, plugin skills or workspace scopes |
| `ExecTool.execute` `working_dir` check | [`agent/tools/shell.py:424`][nb-l9-wd] | Same |
| [`ExecTool._guard_command()`](../nanoclaw/agent/tools/shell.py#L197) | [`:840`][nb-l9-guard], called at [`:446`][nb-l9-guardcall] | Same checks (3,042 commands compared in restricted mode); deny list and internal-URL check always on (differences #1, #2); no allow patterns or sandbox roots |
| [`_extract_absolute_paths()`](../nanoclaw/agent/tools/shell.py#L261), [`_extract_posix_paths_from_token()`](../nanoclaw/agent/tools/shell.py#L293), [`_is_benign_device_path()`](../nanoclaw/agent/tools/shell.py#L254), [`_WORKSPACE_BOUNDARY_NOTE`](../nanoclaw/agent/tools/shell.py#L53) | [`:1032`][nb-l9-extract], [`:254`][nb-l9-benign], [`:85`][nb-l9-note] | Same (copied) |
| `build_default_tools(restrict_to_workspace=)`, `AgentLoop(restrict_to_workspace=)`, `SubagentManager(restrict_to_workspace=)` | config [`tools.restrictToWorkspace`][nb-l9-schema], `_subagent_tools_config` | Same default (off) |

### Web fetch — `nanoclaw/agent/tools/web.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`_validate_url_safe`](../nanoclaw/agent/tools/web.py#L46) | [`agent/tools/web.py:113`][nb-l9-safe] | Same |
| [`_fetch_client_kwargs`](../nanoclaw/agent/tools/web.py#L51) | [`:133`][nb-l9-kwargs] | Same, without the configured `web.proxy` option |
| [`WebFetchTool._get_with_redirects()`](../nanoclaw/agent/tools/web.py#L146) | [`:208`][nb-l9-redir] `_get_with_safe_redirects` | Same checks on every hop |

### Reporting blocked calls — `nanoclaw/agent/tools/violations.py`, runner

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`is_ssrf_violation`](../nanoclaw/agent/tools/violations.py#L49), `_SSRF_BOUNDARY_NOTE`, `_ssrf_soft_payload` | [`agent/tools/execution.py:26`][nb-l9-ssrf] | Same (note names `NANOCLAW_SSRF_WHITELIST`) |
| [`classify_violation`](../nanoclaw/agent/tools/violations.py#L125) | [`_classify_violation` :245][nb-l9-classify] | Same decisions (checked); no events or logging |
| [`workspace_violation_signature`](../nanoclaw/agent/tools/violations.py#L72), [`repeated_workspace_violation_error`](../nanoclaw/agent/tools/violations.py#L103) | [`utils/runtime.py:138`][nb-l9-sig], [`:173`][nb-l9-repeat] | Same (more than 2 attempts per run) |
| `AgentRunner._execute_tool_call` uses `classify_violation`; counts per run | `execution.py` `_execute_tool_call`, `runner.py:385` | Same |
| `nanoclaw ask --restrict`, `NANOCLAW_RESTRICT_TO_WORKSPACE`, `NANOCLAW_SSRF_WHITELIST` | config file | **Replaced:** flags and environment variables (difference #3) |

## Layer 10 — Terminal chat

Note: [`notes/10-terminal-chat.md`](notes/10-terminal-chat.md)

### Chat — `nanoclaw/cli/chat.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`Chat`](../nanoclaw/cli/chat.py#L153) | [`cli/agent.py:266`][nb-l10-classic] (classic interactive mode) | Simplified: direct loop calls instead of the message bus; same turns, exit handling and output |
| ↳ [`banner()`](../nanoclaw/cli/chat.py#L185), [`run()`](../nanoclaw/cli/chat.py#L195), [`turn()`](../nanoclaw/cli/chat.py#L222) | [`:271`][nb-l10-banner], `:384–425` | Same banner, exit words, Ctrl+C/Ctrl+D, spinner and reply |
| ↳ [`_deliver_subagent_results()`](../nanoclaw/cli/chat.py#L236) | `_consume_outbound` (`agent.py:296`) | Same idea: results printed above the prompt; read from the subagent queue |
| ↳ [`handle_command()`](../nanoclaw/cli/chat.py#L263) | [`command/router.py`][nb-l10-router] `dispatch` | Same matching (exact, `/history ` prefix, case-insensitive) |
| ↳ [`cmd_new()`](../nanoclaw/cli/chat.py#L284), [`cmd_compact()`](../nanoclaw/cli/chat.py#L290), [`cmd_stop()`](../nanoclaw/cli/chat.py#L298), [`cmd_status()`](../nanoclaw/cli/chat.py#L303), [`cmd_history()`](../nanoclaw/cli/chat.py#L325), [`cmd_dream()`](../nanoclaw/cli/chat.py#L342), [`cmd_skill()`](../nanoclaw/cli/chat.py#L351), [`cmd_help()`](../nanoclaw/cli/chat.py#L363) | [`cmd_new` :314][nb-l10-new], [`cmd_compact` :349][nb-l10-compact] (+ [events][nb-l10-compactevt]), [`cmd_stop` :214][nb-l10-stop], [`cmd_status` :267][nb-l10-status], [`cmd_history` :869][nb-l10-history], [`cmd_dream` :453][nb-l10-dream], [`cmd_skill` :954][nb-l10-skill] | Same texts; `/stop` cancels subagents only; `/dream` waits (differences #5, #6) |
| [`COMMANDS`](../nanoclaw/cli/chat.py#L56), [`build_help_text`](../nanoclaw/cli/chat.py#L68) | [`BUILTIN_COMMAND_SPECS` :69][nb-l10-specs], [`build_help_text` :1056][nb-l10-help] | Same descriptions for the 8 commands we have (checked) |
| [`invalid_command_text`](../nanoclaw/cli/chat.py#L77) | [`_invalid_command_response`][nb-l10-router] | Same texts (checked through nanobot's router) |
| [`_format_history_message`](../nanoclaw/cli/chat.py#L92) | [`:845`][nb-l10-histfmt] | Same |
| [`build_status_content`](../nanoclaw/cli/chat.py#L113) | [`utils/helpers.py:900`][nb-l10-statustext] | Same lines, without cache rate and search quota |
| [`install_exit_signal_handlers`](../nanoclaw/cli/chat.py#L367) | [`agent.py:283`][nb-l10-signal] | Same |

### Terminal — `nanoclaw/cli/terminal.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`EXIT_COMMANDS`](../nanoclaw/cli/terminal.py#L25), [`is_exit_command`](../nanoclaw/cli/terminal.py#L43) | [`cli/terminal.py:51`][nb-l10-exit] | Same |
| [`print_agent_response`](../nanoclaw/cli/terminal.py#L59), [`response_renderable`](../nanoclaw/cli/terminal.py#L52) | [`:216`][nb-l10-reply], [`:233`][nb-l10-renderable] | Same (name without emoji) |
| [`print_progress_line`](../nanoclaw/cli/terminal.py#L123), [`progress_line_markup`](../nanoclaw/cli/terminal.py#L73) | [`:276`][nb-l10-progress] | Same, but the text is escaped (bug 6) |
| [`ThinkingSpinner`](../nanoclaw/cli/terminal.py#L88) | [`cli/stream.py:50`][nb-l10-spinner] | Same |
| [`PromptInput`](../nanoclaw/cli/terminal.py#L206) | [`_init_prompt_session` :172][nb-l10-session], [`_read_interactive_input_async` :415][nb-l10-read] | Same (history in `~/.nanoclaw/history/`) |
| [`build_key_bindings`](../nanoclaw/cli/terminal.py#L131), [`flush_pending_tty_input`](../nanoclaw/cli/terminal.py#L164), [`save_terminal`](../nanoclaw/cli/terminal.py#L187) / [`restore_terminal`](../nanoclaw/cli/terminal.py#L196), [`sanitize_surrogates`](../nanoclaw/cli/terminal.py#L31), `SafeFileHistory`, `_clear_current_line` | [`_build_cli_key_bindings` :132][nb-l10-keys], [`_flush_pending_tty_input` :98][nb-l10-flush], [`_restore_terminal` :122][nb-l10-restore], `sanitize_surrogates`, [`SafeFileHistory` :86][nb-l10-safehist] | Same (added in the second review) |

### Tool hints and the progress callback

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`format_tool_hints`](../nanoclaw/utils/tool_hints.py#L39), [`abbreviate_path`](../nanoclaw/utils/path.py#L10) | [`utils/tool_hints.py:40`][nb-l10-hints], [`utils/path.py:10`][nb-l10-path] | Copied (2,000 random cases identical) |
| `AgentRunSpec.on_progress` (since layer 13 adapted into an `AgentProgressHook`; `_report_progress` removed) | [`progress_hook.py:135`][nb-l10-hook] | See layer 13 |
| `AgentLoop.process_direct(on_progress=)`, `process_subagent_result(on_progress=)`, `_last_usage`, `start_time` | `process_direct(on_progress=)`, [`loop.py:2217`][nb-l10-usage] | Same |
| `Session.get_history(include_runtime_context=)` | [`session/manager.py:240`][nb-l10-gethist] | Same option |
| [`_chat()`](../nanoclaw/__main__.py#L136), [`_print_progress_to_stderr()`](../nanoclaw/__main__.py#L76) | [`cli/agent.py:54`][nb-l10-agent] `agent` | `nanoclaw chat` = classic `nanobot agent`; `ask` = `agent -m` with progress on stderr |

## CLI — `nanoclaw/__main__.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`main()`](../nanoclaw/__main__.py#L200) `nanoclaw ask "..." [-w DIR] [-s KEY] [--new]` | [`cli/agent.py:55`][nb-cli] `nanobot agent -m "..."` | Simplified: creates missing workspace files, then one `AgentLoop.process_direct` turn in session `cli:direct` (nanobot's key), temperature 0.1 and 8192 max tokens (nanobot's config defaults). `--new` is nanobot's `/new` command. Since layer 6 also `nanoclaw compact` and `nanoclaw dream`, and `NANOCLAW_CONTEXT_WINDOW`. Since layer 8 `ask` waits for background subagents and prints each follow-up reply. Since layer 9 `--restrict`. Since layer 10 `nanoclaw chat` (above), and `ask` shows progress lines |
| `nanoclaw prompt [-w DIR]` | — | **New:** prints the system prompt (no API key needed) |

## Tests

| File | Covers |
|---|---|
| `tests/test_provider_base.py` | argument parsing, tool-call serialization, `should_execute_tools`, sanitizing, `safe_chat` |
| `tests/test_openai_compat.py` | real `openai` SDK over a mocked HTTP transport: request body, text + tool-call parsing, malformed args, duplicate IDs, reasoning text, HTTP + connection errors, `provider_from_env` |
| `tests/test_fake_provider_and_cli.py` | `FakeProvider`, `nanoclaw ask` |
| `tests/toy_tools.py` | toy tools for tests: `echo`, `add`, `boom`, `fail`, `silent`, `big` |
| `tests/test_tools_registry.py` | `ToolResult`, `to_schema`, sorted definitions, every registry error message, retry hint once |
| `tests/test_schema.py` | validation messages, casting, unknown parameters, `arguments` unwrapping |
| `tests/test_filesystem_tools.py` | the four file tools on real files |
| `tests/test_shell_tool.py` | `exec` with real bash: output, timeout, secrets hidden, deny list (including writes to the memory journal) |
| `tests/test_web_tools.py` | `web_fetch` over mocked HTTP; `web_search` with a fake search |
| `tests/test_tool_results.py` | large results saved with preview; `read_file` never cut; truncation without a workspace; a folder per session and clean-up of old folders |
| `tests/test_tools_end_to_end.py` | runner + default tools with a scripted model |
| `tests/test_context.py` | system prompt order, bootstrap file rules, channel/platform hints, runtime context, time block, message merging, `prompt` command |
| `tests/test_session.py` | history replay rules, file format, safe file names, atomic save, corrupt lines, cache, storage location, workspace ID |
| `tests/test_memory.py` | journal, cursors, trimming; Dream prompt and tool limits; raw fallback; archiving with the model; summary checkpoints; token limit in history; prompt sections; token counting |
| `tests/test_compaction.py` | a compacting turn end to end, the next turn, budgets, raw dump on large overflow, errors, `compact`, ephemeral turns, Dream, CLI commands and settings |
| `tests/test_subagents.py` | `spawn` schema; main vs child tools; `wait=true` (child request, answer, temperature, errors, out of iterations); child prompt; background reply, report and follow-up turn; replay; saved once; failures; 4 at once; cancel/close; waiting; runner options; `ask` waiting for children; iteration-limit sync; repeated-report compaction boundary |
| `tests/test_chat.py` | the chat with scripted input: turns, exit words and signals, progress lines and tool hints (escaped markup), Markdown, every slash command and unknown commands, subagent results while idle, CLI `chat` options, `ask` progress |
| `tests/test_skills.py` | frontmatter, discovery and precedence, requirements, always-on and `$skill` skills, summary, bundled skills, `read_file` fallback, prompt sections, saved `$skill` turn, Dream's skills access, CLI |
| `tests/test_loop.py` | two-turn memory, restart, byte-identical replay, tool messages saved and replayed, user message saved before the run, crashed turn closed, interrupted turn, model error, save rules, separate sessions, `--new`, per-session lock, generation settings, CLI options and defaults |
| `tests/test_context_governance.py` | merging adjacent user messages with the time block kept last; detach/reattach |
| `tests/conftest.py` | points `~` at a temporary folder for every test; `fake_dns` fixture (no real DNS or proxy) |
| `tests/test_safety.py` | blocked address ranges, whitelist, internal URLs in commands, proxies, `web_fetch` blocks and redirects, DNS pinning and rebinding, restricted file tools and `exec`, blocked-call reporting and escalation, wiring and CLI |
| `tests/test_runner.py` | the loop end to end with `FakeProvider`: exact history, requests sent, tool failures, truncation, cut-off replies, empty-reply handling, model errors, iteration limit, usage, no mutation, one result per call |
| `tests/test_cli_version.py` | `--version` |

## Layer 11 — Catch up with upstream

Note: [`notes/11-catch-up.md`](notes/11-catch-up.md). nanobot links in this
section point at `c4dac64`, the phase 2 reference.

| NanoClaw | nanobot (`c4dac64`) | How it compares |
|---|---|---|
| `ContextBuilder.build_system_prompt` memory part (`nanoclaw/agent/context.py`) | [`agent/context.py:113`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/context.py#L113) | Same: `MEMORY.md` as is |
| `templates/agent/identity.md`, `platform_policy.md` | [`templates/agent/identity.md`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/templates/agent/identity.md) | Same, apart from the NanoClaw name |
| `SkillsLoader.build_skills_summary` (`nanoclaw/agent/skills.py`) | [`agent/skills.py:270`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/skills.py#L270) | Same: available skills only, absolute built-in root |
| `record_iteration_limit`, `iteration_limit_block`, `ITERATION_LIMIT_META` (`nanoclaw/runtime_context.py`) | [`session/turn_continuation.py:44`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/session/turn_continuation.py#L44) | Same text and rules; no internal continuations |
| `Session.clear()` drops the flag (`nanoclaw/session/manager.py`) | `session/manager.py:355` | Same |
| `SubagentManager` outcomes, partial result, `_announce_result` (`nanoclaw/agent/subagent.py`) | [`agent/subagent.py:678-885`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/subagent.py#L678) | Same strings; partial result from the run's messages (no hook yet) |
| `templates/agent/subagent_announce.md` | [`templates/agent/subagent_announce.md`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/templates/agent/subagent_announce.md) | Identical |
| `WriteFileTool`/`EditFileTool` (`newline=""`), `_reindent_like_match`, `_find_trim_matches`, `_strip_trailing_ws` (`nanoclaw/agent/tools/filesystem.py`) | [`agent/tools/filesystem.py:580-1047`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/filesystem.py#L580) | Same behaviour |
| `ExecTool._guard_command` `~user` check (`nanoclaw/agent/tools/shell.py`) | [`agent/tools/shell.py:894`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/shell.py#L894) | Same error text |
| `pin_resolved_url_dns` (`nanoclaw/security/network.py`) | [`security/network.py:241`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/security/network.py#L241) | Same |
| `_is_json_integer`, `_json_values_equal`, `match_json_schema_type`, `_decode_json_container` (`nanoclaw/agent/tools/schema.py`) | [`agent/tools/base.py:31-367`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/base.py#L31) | Identical results on 7,922 cases |
| `RETRY_HINT` (`nanoclaw/agent/tools/registry.py`) | [`agent/tools/registry.py:14`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/registry.py#L14) `_ERROR_HINT` | Same text, NanoClaw name kept |
| `_search_error`, tool descriptions (`nanoclaw/agent/tools/web.py`) | [`agent/tools/web.py:1206`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/web.py#L1206) | Same texts |
| `_model_slug`, `_requires_max_completion_tokens`, `_supports_temperature`, `_is_direct_openai`, `_parse_usage` (`nanoclaw/providers/openai_compat.py`) | [`providers/openai_compat_provider.py:147-927`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/providers/openai_compat_provider.py#L147) | Same request settings in 17 cases; no effort setting or provider specs |
| `normalize_command_text`, `command_name`, `invalid_command_text` (`nanoclaw/cli/chat.py`) | [`command/router.py:26-168`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/command/router.py#L26) | Same results; functions instead of a router class |
| `requires_posix_shell(reason)`, `pid_running`, `symlink_or_skip` (`tests/conftest.py`) | — | NanoClaw only: Windows test skips and a portable process check (`WINDOWS.md`) |

## Layer 12 — Foundations

Note: [`notes/12-foundations.md`](notes/12-foundations.md). Links point at nanobot `c4dac64`.

| NanoClaw | nanobot (`c4dac64`) | How it compares |
|---|---|---|
| `nanoclaw/config/base.py` `Base` | [`config_base.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/config_base.py) | Same: camelCase aliases, snake_case accepted |
| `nanoclaw/config/schema.py` `Config`, `AgentDefaults`, `ProviderConfig`, `ToolsConfig` | [`config/schema.py:167`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/config/schema.py#L167) | Subset with the same names and defaults (except workspace, model, window); unknown keys ignored |
| `Config.resolve_provider` | [`config/schema.py:588`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/config/schema.py#L588) `_match_provider` | Simplified: named entry, model prefix, first configured |
| `nanoclaw/config/loader.py` `load_config`, `save_config`, `resolve_config_env_vars`, `refresh_config_file` | [`config/loader.py:33-344`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/config/loader.py#L33) | Same file location rules and `${VAR}` references; refresh only adds missing settings |
| `apply_env_overrides`, `load_runtime_config` | — | NanoClaw only: the phase 1 `NANOCLAW_*` variables override the file |
| `nanoclaw/config/errors.py` | [`config/errors.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/config/errors.py) | Copied |
| `nanoclaw/config/timezone.py` | [`config/timezone.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/config/timezone.py) | Same rules; no tzlocal |
| `nanoclaw config path/show/init` (`__main__.py`) | `nanobot config`, `onboard` | Simplified |
| `nanoclaw/cli/log_control.py` `set_nanoclaw_logs`, `configure_console_logging`; `--logs` | [`cli/log_control.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/cli/log_control.py), `cli/agent.py:67` | Same flag; logger disabled at import instead of per command |
| Log calls in memory, session, subagent, tool results, web, runner, loop, violations, tokens, retries | the same lines in nanobot (listed in note 12's tasks) | Same level and text where NanoClaw has the code path |
| `LLMResponse` error fields, `LLMProvider.chat_with_retry`, `_run_with_retry`, `is_transient_response`, `is_arrearage_response`, Retry-After helpers (`providers/base.py`) | [`providers/base.py:318, 738-900, 1299, 1441-1771`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/providers/base.py#L1299) | Same decisions and waits (15 SDK errors compared); no image retry or status events |
| `OpenAICompatProvider._extract_error_metadata`, `_error_response`, `max_retries=0` | [`providers/openai_compat_provider.py:1684-1767`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/providers/openai_compat_provider.py#L1684) | Same metadata; NanoClaw's error text |
| `LLMUsage` (`providers/base.py`) | [`llm_usage/types.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/llm_usage/types.py) | Same fields and sum rules; no timing fields |
| `LLMProvider.usage_for_call`, runner `_usage_or_estimate` | [`providers/base.py:449`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/providers/base.py#L449), `agent/runner.py:1297` | Same estimate rules |

## Layer 13 — Hooks, events, streaming

Note: [`notes/13-hooks-streaming.md`](notes/13-hooks-streaming.md). Links point at nanobot `c4dac64`.

| NanoClaw | nanobot (`c4dac64`) | How it compares |
|---|---|---|
| [`LLMProvider.chat_stream`, `safe_chat_stream`, `chat_stream_with_retry`, `resolve_stream_idle_timeout_s`](../nanoclaw/providers/base.py#L451) | [`providers/base.py:50, 1085, 1147, 1230`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/providers/base.py#L1230) | Same retry rule (no retry after text, timeout recovery); no timing |
| `OpenAICompatProvider.chat_stream`, `_parse_chunks` | [`providers/openai_compat_provider.py:1556, 1839`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/providers/openai_compat_provider.py#L1839) | Same assembled response (19/20 cases; no `<tool_call>` text rescue) |
| `FakeProvider` streaming, `ScriptedStream` | — | NanoClaw only (tests) |
| [`AgentHook`, `CompositeHook`, `AgentHookContext`, `AgentRunHookContext`, `AgentTurnHookContext`](../nanoclaw/agent/hook.py#L91) | [`agent/hook.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/hook.py) | Same callbacks and failure isolation (`had_injections` restored in layer 14) |
| [`AgentEvent`, `EventSink`, `ContextCompactionEvent`, `RetryWaitEvent`, …](../nanoclaw/events.py#L17) | [`events.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/events.py) | Same dataclasses |
| [`ProgressEvent`, `StreamDeltaEvent`, `StreamEndEvent`, …](../nanoclaw/bus/outbound_events.py#L24) | [`bus/outbound_events.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/bus/outbound_events.py) | Same; no `OutboundMessage` helpers until layer 14 |
| [`AgentProgressHook`](../nanoclaw/agent/progress_hook.py#L29) | [`agent/progress_hook.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/progress_hook.py) | Byte-identical events for the same calls |
| [`build_agent_turn_hook`, `AgentTurnHookSpec`](../nanoclaw/agent/turn_hooks.py#L74) | [`agent/turn_hooks.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/turn_hooks.py) | Same: progress hook first |
| `nanoclaw/utils/progress_events.py` (`output_events`, tool event payloads) | [`utils/progress_events.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/utils/progress_events.py) | Same payloads; `output_events` adapts callbacks |
| [`strip_think`, `IncrementalThinkExtractor`, `extract_reasoning`](../nanoclaw/agent/messages.py#L103) | [`utils/helpers.py:173, 261, 297`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/utils/helpers.py#L173) | Same |
| `AgentRunner` hook calls, `tool_events`, streaming requests (`nanoclaw/agent/runner.py`) | [`agent/runner.py:310-1244`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/runner.py#L310), [`agent/tools/execution.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/execution.py) | Same order and events; default hook strips `<think>` |
| `AgentLoop` turn hook, `events=`, compaction events (`nanoclaw/agent/loop.py`, `memory.py`) | [`agent/loop.py:640-700, 1386`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/loop.py#L640), [`agent/memory.py:1246`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/memory.py#L1246) | Same events (direct mode) |
| [`StreamRenderer`, `ThinkingSpinner`, `ReasoningBuffer`](../nanoclaw/cli/stream.py#L140) | [`cli/stream.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/cli/stream.py), [`cli/terminal.py:276-337`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/cli/terminal.py#L276) | Same output; lines escaped |
| [`TurnOutput`](../nanoclaw/cli/terminal.py#L115), chat and ask streaming | [`cli/agent.py:200-450`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/cli/agent.py#L200) | Same chat output; `ask` keeps stdout for the reply |
| `channels.sendProgress`, `sendToolHints`, `showReasoning` (`config/schema.py`) | [`config/schema.py:43-45`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/config/schema.py#L43) | Same names and defaults |

## Layer 14 — Bus, session workers, `/stop`, injections

Note: [`notes/14-bus-workers.md`](notes/14-bus-workers.md). Links point at nanobot `c4dac64`.

| NanoClaw | nanobot (`c4dac64`) | How it compares |
|---|---|---|
| `nanoclaw/bus/events.py` `InboundMessage`, `OutboundMessage` | [`bus/events.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/bus/events.py) | Same fields; no delivery ids |
| `nanoclaw/bus/queue.py` `MessageBus` | [`bus/queue.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/bus/queue.py) | Same; no runtime admission |
| `outbound_message_for_event`, `replace_outbound_event` (`bus/outbound_events.py`) | [`bus/outbound_events.py:115, 134`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/bus/outbound_events.py#L115) | Same |
| `nanoclaw/bus/runtime_events.py` | [`bus/runtime_events.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/bus/runtime_events.py) | Slim: turn lifecycle only |
| `nanoclaw/command/router.py` `CommandRouter`, `CommandContext`, `normalize_command_text` | [`command/router.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/command/router.py) | Same results on 30 inputs; context has `events`, `bus`, `tasks` |
| `nanoclaw/command/builtin.py` `BUILTIN_COMMAND_SPECS`, handlers, `register_builtin_commands` | [`command/builtin.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/command/builtin.py) | Same specs and registration for the 8 commands |
| `AgentRunSpec.injection_callback`, `terminal_injection_callback`, `checkpoint_callback`, `had_injections`, `_raise_if_cancelling` (`agent/runner.py`) | [`agent/runner.py:64`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/runner.py#L64) and injection points | Byte-identical on 15 scenarios; no goal continuation |
| `AgentLoop.run`, `stop`, `aclose`, session queues and workers, `_drain_pending`, `_wait_for_pending`, `cancel_active_tasks`, `_process_message` (`agent/loop.py`) | [`agent/loop.py:1096-1819`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/loop.py#L1096) | Same outbound and saved sessions on 12 scenarios |
| `nanoclaw/agent/turn_delivery.py` `TurnDelivery` | [`agent/turn_delivery.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/turn_delivery.py) | Slim: routing, stream ids, final and error replies, lifecycle |
| `restore_runtime_checkpoint`, `finish_turn` (`session/recovery.py`) | [`session/recovery.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/session/recovery.py) | Same rebuild; JSONL storage |
| Subagent reports on the bus (`agent/subagent.py`) | [`agent/subagent.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/subagent.py) | Same system-message report |
| `Chat` over the bus (`cli/chat.py`) | [`cli/agent.py:322-445`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/cli/agent.py#L322) | Prompt stays open during a turn (difference #1) |

## Layer 15 — Robustness

Note: [`notes/15-robustness.md`](notes/15-robustness.md). Links point at nanobot `c4dac64`.

| NanoClaw | nanobot (`c4dac64`) | How it compares |
|---|---|---|
| `restore_runtime_checkpoint`, `runtime_checkpoint_is_well_formed`, `discard_runtime_checkpoint` (`session/recovery.py`) | [`session/recovery.py:216, 278, 376`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/session/recovery.py#L216) | Byte-identical on 21 checkpoint cases; JSONL storage |
| Restore stage in `AgentLoop._restore_turn` (`agent/loop.py`) | [`agent/loop.py:2134`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/loop.py#L2134) | Same order: checkpoint, user-only tail, then commands |
| `ContextGovernor`, `ContextGovernanceConfig`, `ContextCompactionState`, `ModelRequestState` (`agent/context_governance.py`) | [`agent/context_governance.py:126-262`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/context_governance.py#L126) | Same; no provider conversation state or native compaction |
| `ContextGovernor.prepare_request`, `request_pressure`, `_compact_request_history`, `_fit_pending_file_reads` | [`:716, 506, 623, 458`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/context_governance.py#L716) | 0 differences on 109 checks |
| Transcript repair helpers, `apply_tool_result_budget`, `merge_adjacent_user_messages_for_model` | [`agent/context_governance.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/context_governance.py) | Same |
| `ContextWindowExceededError` (moved from `agent/loop.py`, re-exported there) | [`agent/context_governance.py:79`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/context_governance.py#L79) | Same |
| `AgentRunSpec.context_window_tokens`, `consolidate_history`, `accepted_boundary`, `active_summary`; `AgentRunResult.summary_checkpoint` (`agent/runner.py`) | [`agent/runner.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/runner.py) | Same call site, before every request |
| `SessionSummaryCheckpoint` (`session/summary.py`); commit in `AgentLoop._save_turn` | [`session/summary.py:30`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/session/summary.py#L30) | Same boundary rule |
| Subagent compaction, `persist=False` (`agent/subagent.py`) | [`agent/subagent.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/subagent.py) | Same; no child transcript saved |

## Layer 16 — Shell: exec sessions, Windows

Note: [`notes/16-shell.md`](notes/16-shell.md). Links point at nanobot `c4dac64`.

| NanoClaw | nanobot (`c4dac64`) | How it compares |
|---|---|---|
| `nanoclaw/agent/tools/exec_session.py` `ExecSessionManager`, `_ExecSession`, `_BoundedOutputBuffer`, `format_session_poll` | [`agent/tools/exec_session.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/exec_session.py) | Full port; 41 of 44 scenarios identical, 3 deliberate (guard) |
| `ExecSessionTool` (`exec_session`), `ListExecSessionsTool` (`list_exec_sessions`) | [`exec_session.py:777, 936`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/exec_session.py#L777) | Same names, schemas, texts; plain constructors (no `ToolContext`) |
| `nanoclaw/agent/tools/shell_output.py` `ShellOutputFilter` | [`agent/tools/shell_output.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/shell_output.py) | Same |
| `ExecTool.execute`, `_execute_session`, `execute_to_completion`, `_prepare_command`, `_spawn`, `_resolve_timeout`, `_exec_parameters` (`agent/tools/shell.py`) | [`agent/tools/shell.py:103-620`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/shell.py#L103) | Same; guard always on (difference #1) |
| `_resolve_shell`, `_normalize_powershell_command`, `_build_env`, `_kill_process_tree`, `_create_windows_job` | [`shell.py:620-819`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/shell.py#L620) | 486 Windows cases identical (monkeypatched) |
| `nanoclaw/agent/tools/_windows_job.py` | [`agent/tools/_windows_job.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/_windows_job.py) | Same; NanoClaw falls back to `taskkill` when a job can't be created |
| `current_request_session_key`, `request_context` (`agent/tools/context.py`) | [`agent/tools/context.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/context.py) | Session key for owning sessions |
| `ExecToolConfig` as `tools.exec` (`config/schema.py`) | [`agent/tools/shell_config.py`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/tools/shell_config.py) | Identical on 16 config files |
| `AgentLoop` exec manager, `cancel_active_tasks`, `aclose`, `_has_pending_work` (`agent/loop.py`) | [`agent/loop.py:397, 983, 1531, 1819`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/loop.py#L983) | Same on 21 loop-level checks |
| `SubagentManager` per-child exec managers, `close`, `has_pending_work` (`agent/subagent.py`) | [`agent/subagent.py:106, 274, 318, 999`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/agent/subagent.py#L318) | Same; child sessions owned by the parent's key |
| Windows paths in `_OUTSIDE_PATH_PATTERN`, `_normalize_violation_target` (`agent/tools/violations.py`) | [`utils/runtime.py:136`](https://github.com/HKUDS/nanobot/blob/c4dac64/nanobot/utils/runtime.py#L136) | NanoClaw only (nanobot has the gap) |

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
[nb-replay-args]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L169
[nb-alternation]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L1144
[nb-oc-sanitize]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/openai_compat_provider.py#L701
[nb-l2-reqmodel]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L852
[nb-l2-dropmal]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L1068
[nb-l2-think]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L167
[nb-l2-malnote]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L1106
[nb-l3-tps]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/schema.py#L217
[nb-l3-validate]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/base.py#L51
[nb-l3-cast]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/base.py#L256
[nb-l3-coerce]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/registry.py#L171
[nb-l3-fstool]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L34
[nb-l3-read]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L270
[nb-l3-device]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L210
[nb-l3-write]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L557
[nb-l3-edit]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L877
[nb-l3-match]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L793
[nb-l3-best]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L827
[nb-l3-list]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L1115
[nb-l3-exec]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L162
[nb-l3-deny]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L215
[nb-l3-guard]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L840
[nb-l3-guardcall]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L446
[nb-l3-env]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L791
[nb-l3-kill]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L730
[nb-l3-fetch]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/web.py#L1118
[nb-l3-redirects]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/web.py#L208
[nb-l3-search]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/web.py#L365
[nb-l3-format]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/web.py#L306
[nb-l3-loader]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/loader.py#L26
[nb-l3-bom]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/document.py#L607
[nb-l3-timeout]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L386
[nb-l3-exempt]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context_governance.py#L68
[nb-l3-persist]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L624
[nb-l4-cb]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L89
[nb-l4-boot]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L92
[nb-l4-bsp]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L101
[nb-l4-id]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L154
[nb-l4-load]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L194
[nb-l4-istpl]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L224
[nb-l4-bm]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L231
[nb-l4-bcm]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L310
[nb-l4-merge]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L172
[nb-l4-rcb]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/runtime_context.py#L25
[nb-l4-wrap]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/runtime_context.py#L55
[nb-l4-append]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/runtime_context.py#L120
[nb-l4-render]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/prompt_templates.py#L28
[nb-l4-loadtpl]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L1010
[nb-l4-sync]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L956
[nb-l5-session]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L172
[nb-l5-add]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L208
[nb-l5-hist]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L240
[nb-l5-clear]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L374
[nb-l5-legal]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L477
[nb-l5-mgr]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L1517
[nb-l5-store]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L445
[nb-l5-init]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L448
[nb-l5-wsid]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L568
[nb-l5-wsread]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L517
[nb-l5-skey]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L903
[nb-l5-path]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L923
[nb-l5-goc]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L1613
[nb-l5-inv]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L1738
[nb-l5-rem]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L1538
[nb-l5-save]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L1207
[nb-l5-load]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L935
[nb-l5-repair]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L1015
[nb-l5-loop]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py
[nb-l5-pd]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2498
[nb-l5-build]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2046
[nb-l5-run]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2155
[nb-l5-persist]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2195
[nb-l5-early]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L663
[nb-l5-saveturn]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2320
[nb-l5-new]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L314
[nb-l5-lock]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2562
[nb-l5-prepreq]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L864
[nb-l5-prepreq2]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L1214
[nb-l5-detach]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/runtime_context.py#L148
[nb-l5-reattach]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/runtime_context.py#L184
[nb-l5-restore]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/recovery.py#L384
[nb-l5-pendkey]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/recovery.py#L34
[nb-l5-mark]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2488
[nb-l5-merge]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context_governance.py#L282
[nb-l5-mergec]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context_governance.py#L263
[nb-l6-store]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L59
[nb-l6-readmem]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L230
[nb-l6-append]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L283
[nb-l6-norm]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L260
[nb-l6-next]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L384
[nb-l6-unproc]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L402
[nb-l6-compacth]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L406
[nb-l6-dreamprompt]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L528
[nb-l6-dreamtools]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L560
[nb-l6-raw]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L650
[nb-l6-rawcp]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L685
[nb-l6-fmt]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L628
[nb-l6-dreamkey]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L704
[nb-l6-arch]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L763
[nb-l6-archive]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L836
[nb-l6-archsess]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L1011
[nb-l6-rawck]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L783
[nb-l6-combine]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L805
[nb-l6-cons]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L1087
[nb-l6-summtr]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L1118
[nb-l6-estsess]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L1192
[nb-l6-inbudget]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L1218
[nb-l6-idle]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L1241
[nb-l6-prepreq]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context_governance.py#L617
[nb-l6-compreq]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context_governance.py#L538
[nb-l6-pressure]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context_governance.py#L414
[nb-l6-inputbudget]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context_governance.py#L700
[nb-l6-buffer]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/providers/base.py#L39
[nb-l6-cwee]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context_governance.py#L75
[nb-l6-window]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/config/schema.py#L127
[nb-l6-compact]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L349
[nb-l6-dream]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L453
[nb-l6-commit]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L219
[nb-l6-sanitize]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L101
[nb-l6-summary]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/summary.py#L12
[nb-l6-issumm]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/summary.py#L16
[nb-l6-summmeta]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/summary.py#L37
[nb-l6-hidden]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/history_visibility.py#L10
[nb-l6-ishidden]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/history_visibility.py#L20
[nb-l6-ctxmem]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L127
[nb-l6-esttok]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L774
[nb-l6-estmsg]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L840
[nb-l6-trunctok]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L407
[nb-l6-enc]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/token_encoding.py#L44
[nb-l6-runtime]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/llm_runtime.py#L15
[nb-l6-public]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/runtime_context.py#L215
[nb-l6-cleanup]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L540
[nb-l7-builtin]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L15
[nb-l7-parse]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L26
[nb-l7-valid]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L39
[nb-l7-loader]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L51
[nb-l7-list]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L90
[nb-l7-entries]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L74
[nb-l7-load]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L132
[nb-l7-loadctx]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L148
[nb-l7-explicit]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L165
[nb-l7-block]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L182
[nb-l7-summary]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L204
[nb-l7-reqs]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L266
[nb-l7-missing]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L277
[nb-l7-check]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L338
[nb-l7-avail]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L285
[nb-l7-meta]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L318
[nb-l7-always]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L350
[nb-l7-desc]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L301
[nb-l7-strip]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L309
[nb-l7-getmeta]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/skills.py#L362
[nb-l7-ctxactive]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L132
[nb-l7-ctxsummary]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/context.py#L138
[nb-l7-loopskill]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L762
[nb-l7-readfb]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L234
[nb-l7-readfbuse]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L313
[nb-l7-note]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/workspace_policy.py#L13
[nb-l7-dreamprompt]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/memory.py#L503
[nb-l7-cmdskill]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L954
[nb-l7-syncskills]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L983
[nb-l8-status]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L57
[nb-l8-manager]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L99
[nb-l8-buildtools]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L227
[nb-l8-spawn]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L249
[nb-l8-inline]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L313
[nb-l8-run]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L376
[nb-l8-admitted]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L405
[nb-l8-announce]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L527
[nb-l8-prompt]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L572
[nb-l8-cancel]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L595
[nb-l8-close]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L606
[nb-l8-count]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/subagent.py#L615
[nb-l8-spawntool]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/spawn.py#L48
[nb-l8-spawnexec]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/spawn.py#L81
[nb-l8-reqctx]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/context.py#L30
[nb-l8-bind]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/context.py#L52
[nb-l8-current]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/context.py#L70
[nb-l8-runspec]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L101
[nb-l8-finalize]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L795
[nb-l8-fallback]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/runner.py#L1249
[nb-l8-isub]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2066
[nb-l8-sync]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L517
[nb-l8-persist]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2454
[nb-l8-aclose]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L1639
[nb-l8-inject]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L1095
[nb-l8-config]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/config/schema.py#L131
[nb-l8-cliclose]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/agent.py#L190
[nb-l8-tsys]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/templates/agent/subagent_system.md
[nb-l8-tann]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/templates/agent/subagent_announce.md
[nb-l9-blocked]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/network.py#L16
[nb-l9-wl]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/network.py#L46
[nb-l9-norm]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/network.py#L56
[nb-l9-resolve]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/network.py#L78
[nb-l9-validate]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/network.py#L148
[nb-l9-mounts]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/network.py#L174
[nb-l9-pin]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/network.py#L215
[nb-l9-pinned]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/network.py#L264
[nb-l9-contains]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/network.py#L323
[nb-l9-policy]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/workspace_policy.py#L1
[nb-l9-allowed]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/workspace_policy.py#L96
[nb-l9-within]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/security/workspace_policy.py#L44
[nb-l9-safe]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/web.py#L113
[nb-l9-kwargs]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/web.py#L133
[nb-l9-redir]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/web.py#L208
[nb-l9-wd]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L424
[nb-l9-guardcall]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L446
[nb-l9-guard]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L840
[nb-l9-extract]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L1032
[nb-l9-note]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L85
[nb-l9-benign]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/shell.py#L254
[nb-l9-fscreate]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L84
[nb-l9-rread]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L151
[nb-l9-rwrite]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/filesystem.py#L182
[nb-l9-ssrf]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/execution.py#L26
[nb-l9-classify]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/execution.py#L245
[nb-l9-sig]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/runtime.py#L138
[nb-l9-repeat]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/runtime.py#L173
[nb-l9-loader]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/config/loader.py#L141
[nb-l9-schema]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/config/schema.py#L402
[nb-l10-agent]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/agent.py#L54
[nb-l10-classic]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/agent.py#L266
[nb-l10-banner]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/agent.py#L271
[nb-l10-signal]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/agent.py#L283
[nb-l10-exit]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L51
[nb-l10-session]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L172
[nb-l10-read]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L415
[nb-l10-reply]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L216
[nb-l10-renderable]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L233
[nb-l10-progress]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L276
[nb-l10-spinner]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/stream.py#L50
[nb-l10-specs]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L69
[nb-l10-help]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L1056
[nb-l10-router]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/router.py#L130
[nb-l10-new]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L314
[nb-l10-compact]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L349
[nb-l10-stop]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L214
[nb-l10-status]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L267
[nb-l10-statustext]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/helpers.py#L900
[nb-l10-history]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L869
[nb-l10-histfmt]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L845
[nb-l10-dream]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L453
[nb-l10-skill]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/command/builtin.py#L954
[nb-l10-hints]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/tool_hints.py#L40
[nb-l10-path]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/utils/path.py#L10
[nb-l10-hook]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/progress_hook.py#L135
[nb-l10-usage]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/loop.py#L2217
[nb-l10-gethist]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/session/manager.py#L240
[nb-l10-compactevt]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/bus/outbound_events.py#L155
[nb-l10-keys]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L132
[nb-l10-flush]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L98
[nb-l10-restore]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L122
[nb-l10-safehist]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/cli/terminal.py#L86
