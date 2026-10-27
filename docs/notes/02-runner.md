# 02 — Runner

Reference: nanobot at commit `d3d70f2`. Paths are relative to its `nanobot/` package.

## The concept

Layer 1 gave us a model we can ask one question. Layer 2 makes it an
**agent**: the model can ask for tools, see their results, and keep going until
it has an answer.

```
messages = copy of initial_messages
for iteration in range(max_iterations):
    response = provider.safe_chat(messages, tools)
    if response.should_execute_tools:
        append assistant message (with tool_calls)
        for each call: append {"role": "tool", ...} with its result
        continue
    if empty   → retry; then one no-tools "please answer" call; then give up
    if error   → stop ("error")
    otherwise  → append final answer, stop ("completed")
else:
    one no-tools "budget exhausted, summarize" call → stop ("max_iterations")
```

That loop is the whole agent. Everything later in the roadmap either feeds it
better input (context, sessions, memory, skills) or is just another tool
(files, shell, web, subagents).

### What the history looks like

A run where the model adds two numbers with a tool:

```python
{"role": "user",      "content": "What is 2 + 3?"}
{"role": "assistant", "content": "",
 "tool_calls": [{"id": "c1", "type": "function",
                 "function": {"name": "add", "arguments": "{\"a\": 2, \"b\": 3}"}}]}
{"role": "tool",      "tool_call_id": "c1", "name": "add", "content": "5"}
{"role": "assistant", "content": "The sum is 5."}
```

The rule that keeps this valid: **every `tool_call` id gets exactly one `tool`
message**, even when the tool is unknown, the arguments are broken or the tool
crashes. APIs reject histories that break this rule, so a single bad tool call
would otherwise break the conversation.

## What we built

| File | Contents |
|---|---|
| `nanoclaw/agent/tools/base.py` | `Tool` (name, description, parameters, `execute`, `to_schema`), `ToolResult`, `is_tool_error` |
| `nanoclaw/agent/tools/registry.py` | `ToolRegistry`: register, look up, sorted definitions, `prepare_call`, `execute` |
| `nanoclaw/agent/runner.py` | `AgentRunSpec`, `AgentRunResult`, `AgentRunner` |
| `nanoclaw/agent/messages.py` | message builders, truncation, and the fixed texts copied from nanobot |

About 540 lines. nanobot's equivalent (`runner.py`, `tools/base.py`,
`tools/registry.py`, `tools/execution.py`) is about 2,250 lines.

Symbol-by-symbol mapping: [`../CODE_MAP.md`](../CODE_MAP.md#layer-2--runner).

## Design decisions worth knowing

**Tools never crash the run.** Every failure becomes an error *result*, which
the model reads. The runner runs calls the way nanobot's runner does
(`tools/execution.py` `_execute_tool_call`): `ToolRegistry.prepare_call` checks
the call, then the runner runs the tool itself and catches any exception.

| Situation | What the model sees |
|---|---|
| Unknown tool | `Error: Tool 'x' not found. Available: add, echo` |
| Name differs only in case/punctuation | `... Did you mean 'echo'? Tool names must match exactly.` (the call still does **not** run) |
| Arguments not a JSON object (e.g. malformed JSON from layer 1) | `Error: Tool 'echo' parameters must be a JSON object, got str. ...` |
| Missing required parameter | `Error: Invalid parameters for tool 'add': missing required b` |
| Tool raised | `Error: RuntimeError: kaboom` (`ToolRegistry.execute`, used outside the runner, says `Error executing boom: kaboom`, as in nanobot) |
| Tool returned `ToolResult.error(...)` | its own message |

Each one ends with nanobot's hint
`[Consider this result in the context of the user's goal and decide whether and how to proceed.]`
(added once; nanobot `c4dac64` text since layer 11, it was `[Analyze the error above and try a
different approach.]`). In
practice models usually fix their call on the next iteration.

**Tool calls without a name are dropped, never run or saved.** Some models
emit a tool call with an empty name. It can't run, and if it were saved the API
would reject every later request in the conversation. So it is removed from the
response before anything else happens (nanobot `runner.py:1068`). If that
leaves no calls, the request is retried once with nanobot's note *"The previous
model response attempted to call tools, but every tool call was malformed..."*;
if the retry is malformed too, one last request is made with tools off. Neither
the note nor the bad calls end up in history.

**Inline thinking is removed from answers.** Local reasoning models often reply
`<think>...</think>The answer.`. `strip_think` removes those blocks (and an
unclosed block at the start), so a reply that is *only* thinking counts as
empty. nanobot does this in `extract_reasoning` (`utils/helpers.py:291`) and
also handles malformed and streaming-split tags.

**Tools return `ToolResult.error` for expected failures.** `ToolResult` is a
`str` with an `is_error` flag. A tool like `read_file` can say "file not found"
without raising, and the runner still knows it failed (it is left out of
`tools_used`).

**Only run tools after a normal stop.** If the reply was cut off (`length`) or
filtered (`content_filter`), any tool calls in it are ignored and the reply is
treated as text. This uses `LLMResponse.should_execute_tools` from layer 1
(nanobot `runner.py:487` / `:579`).

**Empty replies get two more chances.** Some models occasionally return
nothing (unless the reply was cut off, refused or filtered, which counts as final). First the same request is retried. If it is empty again, one request
is sent with tools turned off and an extra user message, *"Please provide your
response to the user based on the conversation above."* If that is also empty,
the run stops with `empty_final_response` and nanobot's message *"I completed
the tool steps but couldn't produce a final answer..."*. (`runner.py:592`)

**Running out of iterations still produces an answer.** After
`max_iterations` (20 for us, 200 in nanobot) the runner sends one request with
tools off and nanobot's *"The tool-call budget for this turn is exhausted..."*
prompt, so the user gets a summary of what was done. If that fails, the fixed
message *"I reached the maximum number of tool call iterations (N)..."* is used.
(`runner.py:792`, `:1154`)

**Helper prompts are never saved.** The two extra user messages above are added
to that one request only, not to `messages`. The saved history contains only
what the user and model actually said. *(Updated in layer 5: before each
request, adjacent user messages are merged into one with the runtime-context
block kept last, as nanobot's context governance does. So the "please answer"
prompt reaches the model inside the user's message, not as a second one. See
`05-sessions.md`.)*

**A model error leaves history usable.** The run stops with `stop_reason =
"error"` and the error text as `final_content`, and the placeholder
`[Assistant reply unavailable due to model error.]` is appended so the next
turn doesn't start right after a dangling `user` or `tool` message.
(`runner.py:714`, `:1361`)

**Tool output is bounded.** Results longer than `max_tool_result_chars`
(16,000, nanobot's default) are cut with `\n... (truncated)`. Empty results
become `(<tool> completed with no output)` so the model doesn't think the tool
hung. *(Updated in layer 3: when the runner has a workspace, long results are
saved to a file with a preview instead, and `read_file` is never cut. See
`03-tools.md`, "Large tool results".)*

**The caller's messages are never changed.** The runner deep-copies
`initial_messages`; the full new history is returned in `result.messages`.

## What we left out

| nanobot feature | Where | Why it exists |
|---|---|---|
| Hooks (`before_iteration`, `before_execute_tool`, `on_stream`, ...) | `agent/hook.py`, used throughout `_run_core` | Plugins and UIs observe or change each step *(layer 10 added one slice: the `on_progress` callback for progress lines)* |
| Mid-turn injections | `_try_drain_injections` | Fold a user's follow-up message into the running turn |
| Checkpoints | `_emit_checkpoint` | Resume after a crash mid-turn |
| Length recovery | `build_length_recovery_message`, `_MAX_LENGTH_RECOVERIES` | Ask the model to continue a reply cut off by `max_tokens`, then join the parts |
| Parallel tools | `execution.py` `_partition_tool_batches`, `Tool.concurrency_safe` | Run read-only tools at the same time |
| Repeated-call throttling | `repeated_external_lookup_error`, workspace-violation counting | Stop the model hammering the same URL or forbidden path *(the forbidden-path part done in layer 9; repeated lookups: phase 2)* |
| SSRF / workspace violation classification | `execution.py` `_classify_violation` | Treat security blocks specially *(done in layer 9: `agent/tools/violations.py`)* |
| Argument casting + full schema validation | `Tool.cast_params`, `validate_params`, `tools/schema.py` | `"5"` → `5`, type checks; **comes in layer 3** (done) |
| Context governance + compaction | `agent/context_governance.py` | Keep requests inside the context window (**layer 6**). Its tool-result offloading was ported in layer 3, the adjacent-user merge in layer 5, compaction before a turn in layer 6; the rest is phase 2 |
| Streaming, reasoning events | hooks, `IncrementalThinkExtractor` | Live output in UIs |
| Provider-state continuation | `ProviderConversationStateController` | OpenAI Responses API server-side state |
| Usage estimation + per-round usage | `_usage_or_estimate`, `round_usages` | Cost tracking when providers don't report usage |
| Arrearage message | `is_arrearage_response` | Friendly "out of credit" message |

## Try it

`nanoclaw ask` now runs through the runner. It has no tools until layer 3, so
it behaves like before, but every reply goes through the full loop
*(since layer 5 `ask` goes through `AgentLoop`, which calls the runner)*:

```bash
uv run nanoclaw ask "hello"
```

## Tests

- `tests/toy_tools.py`: `echo`, `add`, `boom` (raises), `fail` (returns an
  error), `silent` (empty output), `big` (long output).
- `tests/test_tools_registry.py`: definitions sorted, lookup, every error
  message above, the retry hint added once.
- `tests/test_runner.py`, with the scripted `FakeProvider`:
  - plain answer; one tool then an answer, with the exact history; several
    calls in one reply in order; several iterations
  - what the provider is sent on each request (messages and tool definitions)
  - each kind of tool failure reported back, and the model recovering
  - truncation and the empty-output marker
  - tool calls ignored in cut-off replies
  - empty-reply retry, finalization, fallback, and the counter resetting after tools
  - model errors, before and after tools
  - iteration limit: finalization success, and the fallback for three kinds of failure
  - usage summed; `initial_messages` unchanged; every call answered exactly once

## Review against nanobot (before layer 3)

After layer 2 we re-read nanobot's `runner.py` (including `_request_model`),
`tools/execution.py`, `tools/registry.py`, `tools/base.py`,
`providers/base.py` and `providers/openai_compat_provider.py` line by line
against our code. These differences were fixed:

| # | Difference | Risk if left | Fix | nanobot |
|---|---|---|---|---|
| 1 | Nameless tool calls were run as "unknown tool" and **saved to history** | The API rejects the saved call, breaking every later request in the conversation | Drop them; retry once with a note; then a no-tools request | `runner.py:1012–1124` |
| 2 | Malformed tool-call arguments were **replayed** as invalid JSON | Strict providers reject the request | Rewrite replayed arguments as a JSON object (`{}` if unparseable) | `providers/openai_compat_provider.py:701`, `providers/base.py:169` |
| 3 | Two user messages in a row were sent (e.g. the "please answer" prompt) | vLLM, Ollama, Azure and others reject them | `enforce_role_alternation`, plus dropping unknown message keys | `providers/base.py:929`, `:1144` |
| 4 | `<think>` blocks from local models stayed in the answer | The user sees the model's internal reasoning; a thinking-only reply wasn't treated as empty | `strip_think` on every response | `utils/helpers.py:167`, `:291` |
| 5 | Exception text differed from what nanobot's runner sends; empty `refusal` replies were retried; two message texts differed slightly | The model saw different wording than nanobot | `Error: RuntimeError: ...`, `refusal` treated as final, exact nanobot texts | `tools/execution.py:185`, `runner.py:589`, `tools/base.py:99`, `tools/registry.py:135` |

Each fix has tests, and removing any one of them makes tests fail.
