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
| [`sanitize_messages`](../nanoclaw/providers/base.py#L160) | [`base.py:834`][nb-sanitize] `_sanitize_empty_content` | Simplified: empty-content rule only (no `_meta` stripping, no surrogate scrubbing) |
| [`tool_arguments_json_for_replay`](../nanoclaw/providers/base.py#L109) | [`base.py:169`][nb-replay-args] (+ `:141`) | Simplified: unparseable → `"{}"` (nanobot tries `json_repair` first) |
| [`enforce_role_alternation`](../nanoclaw/providers/base.py#L128) | [`base.py:1144`][nb-alternation] `_enforce_role_alternation` | Simplified: merge string content, drop trailing assistant. No GLM synthetic-user safety nets or list-content merging |
| [`LLMProvider`](../nanoclaw/providers/base.py#L178) | [`providers/base.py:636`][nb-prov] | Simplified: constructor + abstract `chat()` + `safe_chat()` |
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
| [`FakeProvider`](../nanoclaw/providers/fake.py#L21), [`RecordedCall`](../nanoclaw/providers/fake.py#L13) | — (nanobot tests use mocks) | New: scripted replies + recorded requests for testing later layers |

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
| [`AgentRunSpec`](../nanoclaw/agent/runner.py#L42) | [`agent/runner.py:89`][nb-l2-spec] | Simplified: 8 fields instead of ~25 (no hooks, callbacks, checkpoints, workspace) |
| [`AgentRunResult`](../nanoclaw/agent/runner.py#L58) | [`runner.py:119`][nb-l2-result] | Simplified: `final_content`, `messages`, `tools_used`, `usage`, `stop_reason`, `error` |
| [`AgentRunner.run()`](../nanoclaw/agent/runner.py#L87) | [`runner.py:284`][nb-l2-run] `run` + [`:368`][nb-l2-core] `_run_core` | Simplified: the same loop and stop rules, ~75 lines instead of ~520 |
| ↳ tool branch | [`runner.py:487`][nb-l2-toolbranch] | Simplified: sequential, no checkpoints or hooks |
| ↳ ignore calls in cut-off replies | [`runner.py:579`][nb-l2-ignore] | Same |
| [`_request_model()`](../nanoclaw/agent/runner.py#L189) | [`runner.py:852`][nb-l2-reqmodel] `_request_model` | Simplified: the malformed-call recovery only (retry with note, then no tools); no streaming, governance or compaction |
| [`_drop_malformed_tool_calls()`](../nanoclaw/agent/runner.py#L216) | [`runner.py:1068`][nb-l2-dropmal] | Same |
| ↳ empty-reply retry + finalization | [`runner.py:592`][nb-l2-empty], [`:1126`][nb-l2-finretry] `_request_finalization_retry` | Same rule (`_MAX_EMPTY_RETRIES = 2`) |
| ↳ model error | [`runner.py:714`][nb-l2-error] | Same, minus the arrearage message |
| ↳ empty final response | [`runner.py:737`][nb-l2-emptyfinal] | Same |
| [`_execute_tool_call()`](../nanoclaw/agent/runner.py#L233) | [`agent/tools/execution.py:57`][nb-l2-exec] `execute_tool_calls`, [`:115`][nb-l2-exec1] `_execute_tool_call` | Simplified: `prepare_call`, run, `Error: <Type>: <msg>` on exceptions, same as nanobot. No batching, throttling or violation classification |
| ↳ empty-output marker + keeping results in budget | [`context_governance.py:716`][nb-l2-normalize] `normalize_tool_result` | Same since the layer 3 review: see [`normalize_tool_result`](../nanoclaw/agent/tool_results.py#L88) under Layer 3 |
| [`_finalize_after_max_iterations()`](../nanoclaw/agent/runner.py#L254) | [`runner.py:1154`][nb-l2-finmax], [`:1249`][nb-l2-maxfallback] `_max_iterations_fallback` | Same |
| [`_append_model_error_placeholder()`](../nanoclaw/agent/runner.py#L269) | [`runner.py:1361`][nb-l2-placeholder] | Same |
| [`_add_usage()`](../nanoclaw/agent/runner.py#L73) | `runner.py:1326` `_merge_usage` | Simplified |

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
| [`normalize_tool_result`](../nanoclaw/agent/tool_results.py#L88) | [`agent/context_governance.py:716`][nb-l2-normalize] `ContextGovernor.normalize_tool_result` | Same order: empty marker → `read_file` exempt → truncate (no workspace) or save |
| [`OFFLOAD_EXEMPT_TOOLS`](../nanoclaw/agent/tool_results.py#L22) | [`context_governance.py:68`][nb-l3-exempt] `TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS` | Same: `{"read_file"}` |
| [`maybe_persist_tool_result`](../nanoclaw/agent/tool_results.py#L56) | [`utils/helpers.py:624`][nb-l3-persist] | Simplified: one `default` folder, overwrites, no clean-up (difference #5 in the note) |
| [`render_tool_result_reference`](../nanoclaw/agent/tool_results.py#L36), `safe_filename`, `ensure_nonempty_tool_result` | `utils/helpers.py:511`, `:372`; `utils/runtime.py:48` | Same text |

### Default tool set — `nanoclaw/agent/tools/__init__.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`build_default_tools`](../nanoclaw/agent/tools/__init__.py#L11) | [`agent/tools/loader.py:26`][nb-l3-loader] `ToolLoader` + `agent/loop.py:608` | Replaced: an explicit list instead of package scanning and plugins |

## Layer 4 — Context

Note: [`notes/04-context.md`](notes/04-context.md)

### System prompt and messages — `nanoclaw/agent/context.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`ContextBuilder`](../nanoclaw/agent/context.py#L22) | [`agent/context.py:89`][nb-l4-cb] | Simplified: one workspace folder; no memory, skills or summary yet (layers 6–7) |
| ↳ `BOOTSTRAP_FILES`, `_SKIPPABLE_DEFAULTS` | [`context.py:92`][nb-l4-boot] | Same |
| ↳ [`build_system_prompt()`](../nanoclaw/agent/context.py#L36) | [`context.py:101`][nb-l4-bsp] | Same order and separator for the sections we have |
| ↳ [`_get_identity()`](../nanoclaw/agent/context.py#L45) | [`context.py:154`][nb-l4-id] | Same; agent and project path are the same folder |
| ↳ [`_load_bootstrap_files()`](../nanoclaw/agent/context.py#L63) | [`context.py:194`][nb-l4-load] | Same rules, minus the legacy `SOUL.md` upgrade |
| ↳ [`_is_template_content()`](../nanoclaw/agent/context.py#L78) | [`context.py:224`][nb-l4-istpl] | Same |
| ↳ [`build_messages()`](../nanoclaw/agent/context.py#L95) | [`context.py:231`][nb-l4-bm] (+ `build_transcript`) | Simplified: merges a trailing user message; no media or `_meta` |
| ↳ [`build_current_message()`](../nanoclaw/agent/context.py#L85) | [`context.py:310`][nb-l4-bcm] | Simplified: text + runtime blocks; no images or skill context |
| ↳ [`_merge_message_content()`](../nanoclaw/agent/context.py#L121) | [`context.py:172`][nb-l4-merge] | Same |

### Runtime context — `nanoclaw/runtime_context.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`RuntimeContextBlock`](../nanoclaw/runtime_context.py#L21) | [`runtime_context.py:25`][nb-l4-rcb] | Same |
| [`RUNTIME_CONTEXT_TAG`](../nanoclaw/runtime_context.py#L16), `RUNTIME_CONTEXT_END` | `runtime_context.py:17` | Same text |
| [`wrap_runtime_context_lines`](../nanoclaw/runtime_context.py#L31) | [`runtime_context.py:55`][nb-l4-wrap] | Same |
| [`append_runtime_context`](../nanoclaw/runtime_context.py#L39) | [`runtime_context.py:120`][nb-l4-append] | Simplified: no marker returned for later removal |
| [`current_time_block`](../nanoclaw/runtime_context.py#L51) | — | **New:** nanobot gives the model no date/time (difference #1 in the note) |

### Templates — `nanoclaw/utils/templates.py`, `nanoclaw/templates/`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`render_template`](../nanoclaw/utils/templates.py#L36), `_environment` | [`utils/prompt_templates.py:28`][nb-l4-render], `:18` | Same Jinja2 settings |
| [`load_bundled_template`](../nanoclaw/utils/templates.py#L42) | [`utils/helpers.py:1010`][nb-l4-loadtpl] | Same (reads from the package folder) |
| [`sync_workspace_templates`](../nanoclaw/utils/templates.py#L48) | [`utils/helpers.py:956`][nb-l4-sync] | Simplified: `AGENTS.md`, `SOUL.md`, `USER.md` only; memory/skills files come in layers 6–7; no git store |
| `templates/agent/identity.md`, `platform_policy.md`, `tool_contract.md`, `_snippets/untrusted_content.md` | `templates/agent/` (same names) | Same text except: name, "Dream" made generic, image line removed |
| `templates/SOUL.md`, `USER.md` | `templates/` (same names) | Same text, renamed |
| `templates/AGENTS.md` | `templates/AGENTS.md` | Same opening; cron/heartbeat/`apply_patch` guidance replaced by `edit_file`/`write_file` guidance |

## CLI — `nanoclaw/__main__.py`

| NanoClaw | nanobot | How it compares |
|---|---|---|
| [`main()`](../nanoclaw/__main__.py#L52) `nanoclaw ask "..." [-w DIR]` | [`cli/agent.py:55`][nb-cli] `nanobot agent -m "..."` | Simplified: creates missing workspace files, builds messages with `ContextBuilder` (channel `cli`, time block), runs one `AgentRunner` turn. Switches to `AgentLoop` in layer 5 and a chat REPL in layer 10 |
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
