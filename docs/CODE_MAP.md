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
| CLI (`ask`, `compact`, `dream`, `prompt`) | `nanoclaw/__main__.py` | ~120 | `cli/agent.py`, `cli/commands.py` | ~1,180 |

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
| [`RETRY_HINT`](../nanoclaw/agent/tools/registry.py#L11), [`with_retry_hint`](../nanoclaw/agent/tools/registry.py#L14) | [`agent/tools/execution.py:23`][nb-l2-hint], `:50` `_with_retry_hint` | Same text; added once |

### Runner — `nanoclaw/agent/runner.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`AgentRunSpec`](../nanoclaw/agent/runner.py#L43) | [`agent/runner.py:89`][nb-l2-spec] | Simplified: 8 fields instead of ~25 (no hooks, callbacks, checkpoints, workspace) |
| [`AgentRunResult`](../nanoclaw/agent/runner.py#L59) | [`runner.py:119`][nb-l2-result] | Simplified: `final_content`, `messages`, `tools_used`, `usage`, `stop_reason`, `error` |
| [`AgentRunner.run()`](../nanoclaw/agent/runner.py#L88) | [`runner.py:284`][nb-l2-run] `run` + [`:368`][nb-l2-core] `_run_core` | Simplified: the same loop and stop rules, ~75 lines instead of ~520 |
| ↳ tool branch | [`runner.py:487`][nb-l2-toolbranch] | Simplified: sequential, no checkpoints or hooks |
| ↳ ignore calls in cut-off replies | [`runner.py:579`][nb-l2-ignore] | Same |
| [`_request_model()`](../nanoclaw/agent/runner.py#L190) | [`runner.py:852`][nb-l2-reqmodel] `_request_model` | Simplified: the malformed-call recovery only (retry with note, then no tools); no streaming, governance or compaction |
| [`_request()`](../nanoclaw/agent/runner.py#L170) | [`runner.py:864`][nb-l5-prepreq], [`:1214`][nb-l5-prepreq2] (`context_governor.prepare_request`) | Simplified: one provider call; since layer 5 adjacent user messages are merged first, as in nanobot; no token budgets |
| [`_drop_malformed_tool_calls()`](../nanoclaw/agent/runner.py#L217) | [`runner.py:1068`][nb-l2-dropmal] | Same |
| ↳ empty-reply retry + finalization | [`runner.py:592`][nb-l2-empty], [`:1126`][nb-l2-finretry] `_request_finalization_retry` | Same rule (`_MAX_EMPTY_RETRIES = 2`) |
| ↳ model error | [`runner.py:714`][nb-l2-error] | Same, minus the arrearage message |
| ↳ empty final response | [`runner.py:737`][nb-l2-emptyfinal] | Same |
| [`_execute_tool_call()`](../nanoclaw/agent/runner.py#L234) | [`agent/tools/execution.py:57`][nb-l2-exec] `execute_tool_calls`, [`:115`][nb-l2-exec1] `_execute_tool_call` | Simplified: `prepare_call`, run, `Error: <Type>: <msg>` on exceptions, same as nanobot. No batching, throttling or violation classification |
| ↳ empty-output marker + keeping results in budget | [`context_governance.py:716`][nb-l2-normalize] `normalize_tool_result` | Same since the layer 3 review: see [`normalize_tool_result`](../nanoclaw/agent/tool_results.py#L96) under Layer 3 |
| [`_finalize_after_max_iterations()`](../nanoclaw/agent/runner.py#L255) | [`runner.py:1154`][nb-l2-finmax], [`:1249`][nb-l2-maxfallback] `_max_iterations_fallback` | Same |
| [`_append_model_error_placeholder()`](../nanoclaw/agent/runner.py#L270) | [`runner.py:1361`][nb-l2-placeholder] | Same |
| [`_add_usage()`](../nanoclaw/agent/runner.py#L74) | `runner.py:1326` `_merge_usage` | Simplified |

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
| [`_FsTool`](../nanoclaw/agent/tools/filesystem.py#L22) | [`agent/tools/filesystem.py:34`][nb-l3-fstool] | Simplified: workspace-relative paths only; no allowed dirs or restriction (layer 9) |
| [`ReadFileTool`](../nanoclaw/agent/tools/filesystem.py#L90) | [`filesystem.py:270`][nb-l3-read] | Simplified: text only (no PDF/Office/images), no unchanged-file check. Same numbering, footers, limits, errors, BOM and Latin-1 decoding |
| [`_decode_bom_text`](../nanoclaw/agent/tools/filesystem.py#L78), `_TEXT_EXTENSIONS` | [`utils/document.py:607`][nb-l3-bom], `:643` `_is_text_extension` | Same |
| [`_is_blocked_device`](../nanoclaw/agent/tools/filesystem.py#L56) | [`filesystem.py:210`][nb-l3-device] | Same |
| [`WriteFileTool`](../nanoclaw/agent/tools/filesystem.py#L206) | [`filesystem.py:557`][nb-l3-write] | Same |
| [`EditFileTool`](../nanoclaw/agent/tools/filesystem.py#L458) | [`filesystem.py:877`][nb-l3-edit] | Simplified: no `line_hint` or `expected_replacements` |
| [`_reindent_like_match`](../nanoclaw/agent/tools/filesystem.py#L299), `_preserve_quote_style`, `_curly_double_quotes`, `_curly_single_quotes` | `filesystem.py` (same names) | Same (added in the layer 3 review) |
| [`_find_matches`](../nanoclaw/agent/tools/filesystem.py#L393) (+ `_find_exact_matches`, `_find_trim_matches`, `_find_quote_matches`) | [`filesystem.py:793`][nb-l3-match] | Same strategies, same order |
| [`_best_window`](../nanoclaw/agent/tools/filesystem.py#L425), `_diagnose_near_match`, `_not_found_msg`, `_file_not_found_msg` | [`filesystem.py:827`][nb-l3-best], `:1073`, `:1059` | Same |
| [`_count_changed_lines`](../nanoclaw/agent/tools/filesystem.py#L441) | `utils/file_edit_events.py:59` `FileDiff.from_text` | Simplified: `difflib` instead of `rapidfuzz` |
| [`ListDirTool`](../nanoclaw/agent/tools/filesystem.py#L639) | [`filesystem.py:1115`][nb-l3-list] | Same, except ignore rules only look below the listed folder (difference #3 in the note) |

### Shell — `nanoclaw/agent/tools/shell.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`ExecTool`](../nanoclaw/agent/tools/shell.py#L39) | [`agent/tools/shell.py:162`][nb-l3-exec] | Simplified: bash only; no sessions, sandbox, shell choice, PATH config or Windows |
| [`DEFAULT_DENY_PATTERNS`](../nanoclaw/agent/tools/shell.py#L26) | [`shell.py:215`][nb-l3-deny] | Same general patterns; memory-file patterns come in layer 6 |
| [`_guard_command`](../nanoclaw/agent/tools/shell.py#L149) | [`shell.py:840`][nb-l3-guard], called at [`:446`][nb-l3-guardcall] | **Different: always applied.** nanobot only when `restrict_to_workspace` is on (off by default). Deny list only; no allowlist or internal-URL check |
| [`_build_env`](../nanoclaw/agent/tools/shell.py#L164) | [`shell.py:791`][nb-l3-env] | Same (Unix) |
| [`_kill_process_tree`](../nanoclaw/agent/tools/shell.py#L179) | [`shell.py:730`][nb-l3-kill] | Simplified: `killpg` on the process group, then `process.kill()` |
| [`_resolve_timeout`](../nanoclaw/agent/tools/shell.py#L137) | [`shell.py:386`][nb-l3-timeout] | Same: model's choice capped at 600 s; configured default may be higher; 0 = no limit |

### Web — `nanoclaw/agent/tools/web.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`WebFetchTool`](../nanoclaw/agent/tools/web.py#L56) | [`agent/tools/web.py:1118`][nb-l3-fetch] | Simplified: readability path only; no Jina Reader, image fetching, proxy or SSRF checks (layer 9) |
| [`_get_with_redirects`](../nanoclaw/agent/tools/web.py#L135) | [`web.py:208`][nb-l3-redirects] `_get_with_safe_redirects` | Simplified: scheme check per hop (nanobot also checks resolved IPs) |
| [`_validate_url`](../nanoclaw/agent/tools/web.py#L39), `_strip_tags`, `_normalize`, `_to_markdown`, `_extract_readable_html` | [`web.py:100`][nb-l3-validate-url], `:86`, `:94`, `:1338`, `:1330` | Same |
| [`UNTRUSTED_BANNER`](../nanoclaw/agent/tools/web.py#L22) | `web.py:34` | Same text |
| [`WebSearchTool`](../nanoclaw/agent/tools/web.py#L201) | [`web.py:365`][nb-l3-search] | Simplified: DuckDuckGo only (nanobot's default of ~10 providers) |
| [`_format_results`](../nanoclaw/agent/tools/web.py#L179), `_duckduckgo_search` | [`web.py:306`][nb-l3-format], `:1024` | Same |

### Large tool results — `nanoclaw/agent/tool_results.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`normalize_tool_result`](../nanoclaw/agent/tool_results.py#L96) | [`agent/context_governance.py:716`][nb-l2-normalize] `ContextGovernor.normalize_tool_result` | Same order: empty marker → `read_file` exempt → truncate (no workspace) or save |
| [`OFFLOAD_EXEMPT_TOOLS`](../nanoclaw/agent/tool_results.py#L23) | [`context_governance.py:68`][nb-l3-exempt] `TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS` | Same: `{"read_file"}` |
| [`maybe_persist_tool_result`](../nanoclaw/agent/tool_results.py#L57) | [`utils/helpers.py:624`][nb-l3-persist] | Simplified: one `default` folder, no clean-up. **Different file name:** `<id>-<content hash>.txt`, fixing nanobot's stale-output bug ([`UPSTREAM_BUGS.md`](UPSTREAM_BUGS.md) #3) |
| [`render_tool_result_reference`](../nanoclaw/agent/tool_results.py#L37), `safe_filename`, `ensure_nonempty_tool_result` | `utils/helpers.py:511`, `:372`; `utils/runtime.py:48` | Same text |

### Default tool set — `nanoclaw/agent/tools/__init__.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`build_default_tools`](../nanoclaw/agent/tools/__init__.py#L11) | [`agent/tools/loader.py:26`][nb-l3-loader] `ToolLoader` + `agent/loop.py:608` | Replaced: an explicit list instead of package scanning and plugins |

## Layer 4 — Context

Note: [`notes/04-context.md`](notes/04-context.md)

### System prompt and messages — `nanoclaw/agent/context.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`ContextBuilder`](../nanoclaw/agent/context.py#L29) | [`agent/context.py:89`][nb-l4-cb] | Simplified: one workspace folder; no skills yet (layer 7) |
| ↳ `BOOTSTRAP_FILES`, `_SKIPPABLE_DEFAULTS` | [`context.py:92`][nb-l4-boot] | Same |
| ↳ [`build_system_prompt()`](../nanoclaw/agent/context.py#L44) | [`context.py:101`][nb-l4-bsp] | Same order and separator for the sections we have; memory and summary sections since layer 6 |
| ↳ [`_get_identity()`](../nanoclaw/agent/context.py#L69) | [`context.py:154`][nb-l4-id] | Same; agent and project path are the same folder |
| ↳ [`_load_bootstrap_files()`](../nanoclaw/agent/context.py#L87) | [`context.py:194`][nb-l4-load] | Same rules, minus the legacy `SOUL.md` upgrade |
| ↳ [`_is_template_content()`](../nanoclaw/agent/context.py#L102) | [`context.py:224`][nb-l4-istpl] | Same |
| ↳ [`build_messages()`](../nanoclaw/agent/context.py#L123) | [`context.py:231`][nb-l4-bm] (+ `build_transcript`) | Simplified: merges a trailing user message (and its `_meta`); no media. Since layer 6: `current_message=None` and `session_summary` |
| ↳ [`build_current_message()`](../nanoclaw/agent/context.py#L109) | [`context.py:310`][nb-l4-bcm] | Simplified: text + runtime blocks, with the `_meta` marker; no images or skill context |
| ↳ [`_merge_message_content()`](../nanoclaw/agent/context.py#L155) | [`context.py:172`][nb-l4-merge] | Same |

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
| [`sync_workspace_templates`](../nanoclaw/utils/templates.py#L48) | [`utils/helpers.py:956`][nb-l4-sync] | Simplified: `AGENTS.md`, `SOUL.md`, `USER.md`, `memory/MEMORY.md`, `memory/history.jsonl`; `skills/` comes in layer 7; no `HEARTBEAT.md`, `prompts/` or git store |
| `templates/agent/identity.md`, `platform_policy.md`, `tool_contract.md`, `_snippets/untrusted_content.md` | `templates/agent/` (same names) | Same text except: name, image line removed ("Dream" wording restored in layer 6) |
| `templates/SOUL.md`, `USER.md` | `templates/` (same names) | Same text, renamed |
| `templates/AGENTS.md` | `templates/AGENTS.md` | Same opening; cron/heartbeat/`apply_patch` guidance replaced by `edit_file`/`write_file` guidance |

## Layer 5 — Sessions + AgentLoop

Note: [`notes/05-sessions.md`](notes/05-sessions.md)

### Session — `nanoclaw/session/manager.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`Session`](../nanoclaw/session/manager.py#L86) | [`session/manager.py:172`][nb-l5-session] | Simplified: no provider state or session policy; `last_archived` without the legacy `last_consolidated` alias |
| ↳ [`add_message()`](../nanoclaw/session/manager.py#L106) | [`manager.py:208`][nb-l5-add] | Same |
| ↳ [`get_history()`](../nanoclaw/session/manager.py#L135) | [`manager.py:240`][nb-l5-hist] | Same output (checked by running both), including the summary marker, `max_tokens` limit and reply cleanup (layer 6); no `max_messages`, image or CLI-app breadcrumbs |
| ↳ [`clear()`](../nanoclaw/session/manager.py#L189) | [`manager.py:374`][nb-l5-clear] | Same, minus the summary and provider state |
| [`find_legal_message_start`](../nanoclaw/session/manager.py#L63) | [`utils/helpers.py:477`][nb-l5-legal] | Same |

### Storage — `nanoclaw/session/manager.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`SessionManager`](../nanoclaw/session/manager.py#L197) | [`SessionManager`][nb-l5-mgr] + [`JsonlSessionStore`][nb-l5-store] | Simplified: one class; no file locks, runtime checkpoints, migration or session listing |
| ↳ `__init__` | [`manager.py:448`][nb-l5-init] | Same rule: storage inside the workspace is refused; root folder made private |
| ↳ [`_load_or_create_workspace_id()`](../nanoclaw/session/manager.py#L225) | [`manager.py:568`][nb-l5-wsid], [`:517`][nb-l5-wsread] | Same file and format (`<workspace>/.nanoclaw/workspace-id`, 32 hex chars); no recovery of a lost ID |
| ↳ [`storage_key()`](../nanoclaw/session/manager.py#L255), [`get_session_path()`](../nanoclaw/session/manager.py#L259) | [`manager.py:903`][nb-l5-skey], [`:923`][nb-l5-path] | Same file names (base64 of the key) |
| ↳ [`get_or_create()`](../nanoclaw/session/manager.py#L264), [`invalidate()`](../nanoclaw/session/manager.py#L272), [`_remember()`](../nanoclaw/session/manager.py#L276) | [`manager.py:1613`][nb-l5-goc], [`:1738`][nb-l5-inv], [`:1538`][nb-l5-rem] | Simplified: one LRU cache of 128 (nanobot also keeps evicted sessions weakly) |
| ↳ [`save()`](../nanoclaw/session/manager.py#L284) | [`manager.py:1207`][nb-l5-save] | Same format (metadata line, then messages); atomic temp-file + rename; no fsync option |
| ↳ [`_load()`](../nanoclaw/session/manager.py#L307) | [`manager.py:935`][nb-l5-load], [`repair`][nb-l5-repair] | Simplified: always skips unreadable lines (nanobot's `repair` fallback), without logging. Files are readable by both projects (checked) |

### AgentLoop — `nanoclaw/agent/loop.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`AgentLoop`](../nanoclaw/agent/loop.py#L69) | [`agent/loop.py`][nb-l5-loop] `AgentLoop` | Simplified: direct path only; no bus, channels, queues, hooks, cancellation or checkpoint recovery. Takes `temperature` and `max_tokens` |
| ↳ [`process_direct()`](../nanoclaw/agent/loop.py#L114) | [`loop.py:2498`][nb-l5-pd] → `_build_turn` [`:2046`][nb-l5-build], `_run_turn` [`:2155`][nb-l5-run], `_persist_turn` [`:2195`][nb-l5-persist] | Same stages and order, including closing a crashed turn first and the `pending_user_turn` flag (set when the user message is saved, cleared after the turn); returns the `AgentRunResult` instead of an `OutboundMessage` |
| ↳ [`_persist_user_message_early()`](../nanoclaw/agent/loop.py#L318) | [`loop.py:663`][nb-l5-early] | Same: saved with its runtime context and the `_runtime_context` marker |
| ↳ [`_save_turn()`](../nanoclaw/agent/loop.py#L338) | [`loop.py:2320`][nb-l5-saveturn] | Same rules (empty assistant skipped, unknown/duplicate tool results dropped, timestamps); no latency, image redaction or summary checkpoints |
| ↳ [`new_session()`](../nanoclaw/agent/loop.py#L247) | [`command/builtin.py:314`][nb-l5-new] `/new` | Since layer 6: archives the old messages to the journal first, then clears (see the layer 6 table) |
| ↳ [`_get_session_lock()`](../nanoclaw/agent/loop.py#L315) | [`loop.py:2562`][nb-l5-lock] | Same: one `asyncio.Lock` per session key |

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
| [`MemoryStore`](../nanoclaw/agent/memory.py#L46) | [`agent/memory.py:59`][nb-l6-store] | Simplified: no git store, legacy `HISTORY.md` migration, append lock or warnings |
| ↳ [`read_memory()`](../nanoclaw/agent/memory.py#L72), [`write_memory()`](../nanoclaw/agent/memory.py#L75) | [`memory.py:230`][nb-l6-readmem] | Same (no `read_soul`/`read_user` helpers; not needed) |
| ↳ [`append_history()`](../nanoclaw/agent/memory.py#L86), [`_normalize_history_entry()`](../nanoclaw/agent/memory.py#L80) | [`:283`][nb-l6-append], [`:260`][nb-l6-norm] | Same: `<think>` removed, 64,000-char cap, cursor + `.cursor` file |
| ↳ [`_next_cursor()`](../nanoclaw/agent/memory.py#L134), [`read_unprocessed_history()`](../nanoclaw/agent/memory.py#L147) | [`:384`][nb-l6-next], [`:402`][nb-l6-unproc] | Same (checked) |
| ↳ [`compact_history()`](../nanoclaw/agent/memory.py#L150) | [`:406`][nb-l6-compacth] | Same: keep 1,000, never drop entries Dream hasn't read |
| ↳ [`build_dream_prompt()`](../nanoclaw/agent/memory.py#L224) | [`:528`][nb-l6-dreamprompt] | Same (20 entries, 1,000 chars each); no `prompts/dream.md` override |
| ↳ [`build_dream_tools()`](../nanoclaw/agent/memory.py#L238) + [`_PathGuard`](../nanoclaw/agent/memory.py#L306) | [`:560`][nb-l6-dreamtools] | Simplified: read inside the workspace; write only `MEMORY.md`/`SOUL.md`/`USER.md` (no `skills/`, no `apply_patch`) |
| ↳ [`raw_archive()`](../nanoclaw/agent/memory.py#L286), [`_build_raw_checkpoint()`](../nanoclaw/agent/memory.py#L299), [`_format_messages()`](../nanoclaw/agent/memory.py#L271) | [`:650`][nb-l6-raw], [`:685`][nb-l6-rawcp], [`:628`][nb-l6-fmt] | Same output (checked); no image breadcrumbs |
| ↳ [`dream_session_key()`](../nanoclaw/agent/memory.py#L264) | [`:704`][nb-l6-dreamkey] | Same |

### Archiving and compaction — `nanoclaw/agent/memory.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`MemoryArchiver`](../nanoclaw/agent/memory.py#L342) | [`memory.py:763`][nb-l6-arch] | Simplified: no provider conversation state |
| ↳ [`archive()`](../nanoclaw/agent/memory.py#L390) | [`:836`][nb-l6-archive] | Same rules: archive prompt appended to the request, size check, one retry after refusing tool calls, raw fallback, `(nothing)` not journaled |
| ↳ [`archive_session()`](../nanoclaw/agent/memory.py#L465) | [`:1011`][nb-l6-archsess] | Same (never summarizes only part of the stretch) |
| ↳ [`_raw_checkpoint()`](../nanoclaw/agent/memory.py#L355), [`_combine_raw_checkpoint()`](../nanoclaw/agent/memory.py#L372) | [`:783`][nb-l6-rawck], [`:805`][nb-l6-combine] | Same (checked) |
| [`Consolidator`](../nanoclaw/agent/memory.py#L505) | [`memory.py:1087`][nb-l6-cons] | Simplified: no locks or events |
| ↳ [`summarize_transcript()`](../nanoclaw/agent/memory.py#L523) | [`:1118`][nb-l6-summtr] | Same budgets (input = window − max tokens; raw fallback ≤ half of it) |
| ↳ [`estimate_session_prompt_tokens()`](../nanoclaw/agent/memory.py#L555), [`_input_token_budget()`](../nanoclaw/agent/memory.py#L567) | [`:1192`][nb-l6-estsess], [`:1218`][nb-l6-inbudget] | Same |
| ↳ [`compact_idle_session()`](../nanoclaw/agent/memory.py#L575) | [`:1241`][nb-l6-idle] | Same result (checked via `/compact`); no events |

### Compaction in the loop — `nanoclaw/agent/loop.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`_compact_if_needed()`](../nanoclaw/agent/loop.py#L192) | [`context_governance.py:617`][nb-l6-prepreq] `prepare_request`, [`:538`][nb-l6-compreq] `_compact_request_history`, [`:414`][nb-l6-pressure] `request_pressure` | **Different:** checked once before the turn; nanobot checks before every request (difference #1). Same budget, summary transcript and save position (checked) |
| [`_input_budget()`](../nanoclaw/agent/loop.py#L185), `CONTEXT_SAFETY_BUFFER` | [`context_governance.py:700`][nb-l6-inputbudget], [`providers/base.py:39`][nb-l6-buffer] | Same |
| [`ContextWindowExceededError`](../nanoclaw/agent/loop.py#L55) | [`context_governance.py:75`][nb-l6-cwee] | Same message |
| `DEFAULT_CONTEXT_WINDOW_TOKENS` | [`config/schema.py:127`][nb-l6-window] | **Different:** 128,000 instead of 200,000 (difference #2) |
| ↳ [`new_session()`](../nanoclaw/agent/loop.py#L247) | [`command/builtin.py:314`][nb-l5-new] `cmd_new` | Same request and journal entry (checked); waits instead of running in the background |
| ↳ [`compact_session()`](../nanoclaw/agent/loop.py#L273) | [`command/builtin.py:349`][nb-l6-compact] `cmd_compact` | Same |
| ↳ [`run_dream()`](../nanoclaw/agent/loop.py#L281) | [`command/builtin.py:453`][nb-l6-dream] `cmd_dream` | Same flow and messages; file comparison instead of a git diff; no schedule |
| ↳ `process_direct(ephemeral=, tools=)` | [`loop.py:2498`][nb-l5-pd] | Same idea: a temporary turn saves nothing and writes no spill files |

### Sessions, prompt and helpers

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`Session.commit_summary_checkpoint()`](../nanoclaw/session/manager.py#L113) | [`session/manager.py:219`][nb-l6-commit] | Same (checked) |
| [`_sanitize_assistant_replay_text`](../nanoclaw/session/manager.py#L46) | [`session/manager.py:101`][nb-l6-sanitize] | Same (ported during the layer 6 review) |
| [`SUMMARY_CONTINUATION_TEXT`](../nanoclaw/session/summary.py#L16), [`is_summary_checkpoint`](../nanoclaw/session/summary.py#L29), [`session_summary_from_metadata`](../nanoclaw/session/summary.py#L34) | [`session/summary.py:12`][nb-l6-summary], [`:16`][nb-l6-issumm], [`:37`][nb-l6-summmeta] | Same |
| [`HIDDEN_HISTORY_META`](../nanoclaw/session/summary.py#L15), [`is_hidden_history_message`](../nanoclaw/session/summary.py#L24) | [`session/history_visibility.py:10`][nb-l6-hidden], [`:20`][nb-l6-ishidden] | Simplified: no automation messages |
| `build_system_prompt` memory + summary sections | [`agent/context.py:127`][nb-l6-ctxmem] | Same text (checked) |
| [`estimate_prompt_tokens`](../nanoclaw/utils/tokens.py#L59), [`estimate_message_tokens`](../nanoclaw/utils/tokens.py#L86), [`truncate_text_to_tokens`](../nanoclaw/utils/tokens.py#L130) | [`utils/helpers.py:774`][nb-l6-esttok], [`:840`][nb-l6-estmsg], [`:407`][nb-l6-trunctok] | Same counts (checked with tiktoken and bytes); no per-tools cache |
| [`get_token_encoding`](../nanoclaw/utils/tokens.py#L24) | [`utils/token_encoding.py:44`][nb-l6-enc] | Simplified: loaded on first use, not in a background thread |
| [`LLMRuntime`](../nanoclaw/utils/llm_runtime.py#L16) | [`utils/llm_runtime.py:15`][nb-l6-runtime] | Simplified: flat fields, no presets |
| [`public_history_message`](../nanoclaw/runtime_context.py#L111) | [`runtime_context.py:215`][nb-l6-public] | Same |
| `templates/memory/MEMORY.md`, `agent/consolidator_archive.md`, `agent/dream.md` | `templates/` (same names) | Same text; `dream.md` without its skills section (layer 7) |

## CLI — `nanoclaw/__main__.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`main()`](../nanoclaw/__main__.py#L90) `nanoclaw ask "..." [-w DIR] [-s KEY] [--new]` | [`cli/agent.py:55`][nb-cli] `nanobot agent -m "..."` | Simplified: creates missing workspace files, then one `AgentLoop.process_direct` turn in session `cli:direct` (nanobot's key), temperature 0.1 and 8192 max tokens (nanobot's config defaults). `--new` is nanobot's `/new` command. Since layer 6 also `nanoclaw compact` and `nanoclaw dream`, and `NANOCLAW_CONTEXT_WINDOW`. A chat REPL comes in layer 10 |
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
| `tests/test_shell_tool.py` | `exec` with real bash: output, timeout, secrets hidden, deny list |
| `tests/test_web_tools.py` | `web_fetch` over mocked HTTP; `web_search` with a fake search |
| `tests/test_tool_results.py` | large results saved with preview; `read_file` never cut; truncation without a workspace |
| `tests/test_tools_end_to_end.py` | runner + default tools with a scripted model |
| `tests/test_context.py` | system prompt order, bootstrap file rules, channel/platform hints, runtime context, time block, message merging, `prompt` command |
| `tests/test_session.py` | history replay rules, file format, safe file names, atomic save, corrupt lines, cache, storage location, workspace ID |
| `tests/test_memory.py` | journal, cursors, trimming; Dream prompt and tool limits; raw fallback; archiving with the model; summary checkpoints; token limit in history; prompt sections; token counting |
| `tests/test_compaction.py` | a compacting turn end to end, the next turn, budgets, raw dump on large overflow, errors, `compact`, ephemeral turns, Dream, CLI commands and settings |
| `tests/test_loop.py` | two-turn memory, restart, byte-identical replay, tool messages saved and replayed, user message saved before the run, crashed turn closed, interrupted turn, model error, save rules, separate sessions, `--new`, per-session lock, generation settings, CLI options and defaults |
| `tests/test_context_governance.py` | merging adjacent user messages with the time block kept last; detach/reattach |
| `tests/conftest.py` | points `~` at a temporary folder for every test |
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
[nb-l3-validate-url]: https://github.com/HKUDS/nanobot/blob/d3d70f20655bce5d3c7a017bfbe803efe09a2438/nanobot/agent/tools/web.py#L100
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
