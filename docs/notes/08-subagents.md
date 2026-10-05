# 08 — Subagents

Reference: nanobot at commit `d3d70f2`. Paths are relative to its `nanobot/` package.

## The concept

The main agent can hand a task to a **child agent** (a subagent) with the
`spawn` tool. The child starts fresh: a short system prompt of its own, the
task as its only message, and its own set of tools. It works through the task
alone and its final answer goes back to the main agent.

This keeps big side jobs (research, a long multi-file change) out of the main
conversation, and several can run at the same time.

```
user ──▶ main agent ──spawn(task)──▶ child agent (own prompt, own tools)
             ▲                              │
             └──────── final answer ◀───────┘
```

## Two modes

| `spawn(..., wait=...)` | What happens | nanobot |
|---|---|---|
| `wait=true` | The main agent waits. The child's final answer is the tool result, in the same turn | `SubagentManager.run_inline` (`agent/subagent.py:313`) |
| `wait=false` (default) | The tool returns at once: `Subagent [label] started (id: 1a2b3c4d). I'll notify you when it completes.` The child runs in the background. When it finishes, its report becomes a **follow-up turn**: the main agent reads it and tells the user in 1–2 sentences | `SubagentManager.spawn` (`:249`) |

### The follow-up turn (background mode)

The child's result is wrapped in a report (`templates/agent/subagent_announce.md`):

```
[Subagent 'Weather' completed successfully]

Task: Check the weather

Result:
Sunny, 20°C.

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs.
```

Then, as in nanobot's system turn for a subagent (`agent/loop.py:2066`, `_persist_subagent_followup` `:2454`):

1. The report is **saved as an assistant message** with `injected_event:
   "subagent_result"`, `subagent_task_id` and `sender_id: "subagent"`. Each
   task's report is saved once; a second copy is not saved again (the model
   still answers it, as in nanobot).
2. The model gets the conversation so far, then the report as **new user
   input**, without the time block.
3. The model's reply is saved and shown to the user.

Later turns replay the report as an assistant message.

## The child agent

| Part | What | nanobot |
|---|---|---|
| System prompt | `templates/agent/subagent_system.md`: "You are a subagent… Stay focused on the assigned task", the untrusted-content rule, the history log path, the skills list | `_build_subagent_prompt` (`:572`) |
| Messages | System prompt + the task as the user message (no time block, no conversation history) | `:434` |
| Tools | The standard 7: `read_file`, `write_file`, `edit_file`, `list_dir`, `exec`, `web_fetch`, `web_search`. **No `spawn`**, so no subagents inside subagents | `_build_tools` (`:227`, scope `subagent`) |
| Model settings | The main agent's model, max tokens and temperature; `temperature` in the `spawn` call overrides it | `with_generation_overrides` |
| Iteration limit | Same as the main agent (20 in NanoClaw; see differences), re-read at the start of every turn so a changed limit applies (nanobot `_sync_subagent_runtime_limits`, `loop.py:517`). When it runs out there is **no extra summary request**; the result is `Task completed but no final response was generated.` | `finalize_on_max_iterations=False`, `max_iterations_message` (`:471`) |
| Results | Model error: the error text (`Sorry, I encountered an error calling the AI model.`), marked as an error. Exception: `Error: <message>`. Empty: `Task completed but no final response was generated.` | `:491`, `:510` |
| Large tool results | Saved in the conversation's tool-results folder, as for the main agent | `session_key` passed to the runner |
| Running at once | At most **4**; the others wait (`queued` → `initializing` → `done`/`error`) | `asyncio.Semaphore`, `max_concurrent_subagents` (`config/schema.py:131`) |
| Task ID / label | First 8 characters of a UUID; label = `label`, or the task's first 30 characters + `...` | `:267`, `:268` |

## How the main agent's tools know the conversation

`spawn` needs the turn's session and model settings. As in nanobot, the loop
binds a **request context** (a Python `ContextVar`) around each run, and
`spawn` reads it (`agent/tools/context.py`). Called outside a turn, `spawn`
returns `Error: spawn requires an active model runtime`.

## `nanoclaw ask` and background subagents

`nanoclaw ask` sends one message and exits (one-shot). A background child
finishes *later*, so its result needs somewhere to go:

| | nanobot one-shot (`nanobot agent -m`) | nanobot chat / server | NanoClaw `ask` |
|---|---|---|---|
| Background result | Lost: running children are cancelled at exit (`aclose()`, `cli/agent.py:192`) | Delivered as a follow-up turn when it arrives | `ask` **waits** for running children, then runs each report as a follow-up turn and prints the reply |

What you see:

```
$ uv run nanoclaw ask "Research X in the background"
I've started researching X.              ← the main agent's reply
[subagent 'Research X' finished]         ← (stderr)
X turns out to be ...                    ← the follow-up turn's reply
```

Children started before an error in the main turn still report. When the
layer 10 chat arrives, reports will appear as they finish, as in nanobot.

## Runner options added

| Option | Default | Used by | nanobot |
|---|---|---|---|
| `finalize_on_max_iterations` | `True` (ask for a summary with tools off) | Subagents set `False` | `runner.py:112`, `:795` |
| `max_iterations_message` | `None` (the standard message) | Subagents set the "Task completed…" text | `runner.py:101`, `_max_iterations_fallback` `:1249` |

nanobot's third option, `error_message=None`, is not needed: with it nanobot
falls back to its default error text, which is what NanoClaw's runner already
returns.

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why | Later |
|---|---|---|---|---|---|
| 1 | Background results in one-shot mode | `agent -m` cancels running children at exit; results lost | `ask` waits for them and runs each report as a follow-up turn | Otherwise background `spawn` would do nothing useful until the layer 10 chat. Chosen with the user (option A) | Layer 10 chat delivers them as they finish |
| 2 | Iteration limit | 200 (`max_tool_iterations`), for the main agent and children | 20, for both | NanoClaw's limit since layer 2. Chosen with the user: keep 20 for now | To be updated later |
| 3 | A report arriving **during** a main turn | Added to that running turn (mid-turn injection, `loop.py:1095`) | Waits until the turn has finished, then becomes its own follow-up turn | NanoClaw has no mid-turn injection (layer 2 skipped it) | Phase 2: Mid-turn injections |
| 4 | How the report travels | `InboundMessage` on the message bus (channel `system`, sender `subagent`) | `SubagentResult` on the manager's `results` queue, read by the loop | NanoClaw has no bus | Phase 2: Message bus + channels |
| 5 | Child tools | Also `apply_patch`, `rg`, `exec_session`, `list_exec_sessions`, `run_cli_app` | The 7 tools NanoClaw has | Those tools aren't ported (layer 3) | With those tools |
| 6 | Line in the child prompt | Also "Tools like 'read_file' and 'web_fetch' can return native image content…" | Not included | The same image line left out in layer 4 (NanoClaw's tools don't return images) | — |
| 7 | Several `spawn` calls in one model reply | Run at the same time (`spawn` is `concurrency_safe`, and the main loop runs such tools in parallel) | Run one after another | NanoClaw's runner runs tools one at a time (parallel tools were skipped in layer 2). The results, requests and saved conversation are the same (checked); only the waiting time differs | Phase 2 (parallel tools) |
| 8 | Compaction on a **repeated** report | The summary marker goes one message too early, so a summarized message is replayed again ([upstream bug 5](../UPSTREAM_BUGS.md#5-summary-marker-one-message-too-early-on-a-repeated-subagent-report)) | Marker after everything that was summarized | Found in the second review; NanoClaw records the boundary before saving the report, so it was already correct. Kept, with a regression test | — |

Everything else matches nanobot (checked by running both; see below).

## What we left out (for later phases)

| nanobot feature | Where | What it does | Later phase |
|---|---|---|---|
| Live status reporting | `_SubagentHook`, `SubagentStatus.iteration/tool_events/usage`, `runtime_statuses`, the `my` tool, web UI | Shows each child's iteration, tool calls and token use while it runs. NanoClaw keeps the phase (`queued`, `initializing`, `done`, `error`) only | **Phase 2: Hooks / Web UI** |
| Logging of child tool calls | `_SubagentHook.before_execute_tools` | Debug log line per child tool call | **Phase 2: Hooks, Logging** |
| Compaction during a child's run | `consolidate_history`, `consolidate_provider_compaction` (`:450`) | Summarizes a child's own messages if its run outgrows the context window (session key `subagent:<id>`, not saved) | **Phase 2: Context governance** (with mid-turn compaction) |
| Project-folder scope | `workspace_scope`, `bind_workspace_scope` | Runs the child in the main agent's project folder, with workspace restriction | **Layer 9** (restriction) / **Phase 2: Separate project folder** |
| Shared background shell sessions | `ExecSessionManager`, `exec_session` tools | Long-running shell sessions shared by children; closed with them | **Phase 2** (with background `exec`) |
| Stop commands | `cancel_by_session` called by `/stop`; `aclose()` | Cancel a conversation's children. NanoClaw has `cancel_by_session()` and `close()`, with no command yet | **Layer 10** (chat commands) |
| Usage attribution | `llm_usage_source` | Counts a child's tokens under the conversation that started it | **Phase 2: Usage accounting** |
| Provider conversation state for reports | `_SUBAGENT_PROVIDER_TASK_META`, `provider_state` | Adds the report to server-side conversation state | **Phase 2: Provider conversation state** |
| Old constructor arguments | `provider=`/`model=`, `set_provider()` | Deprecated compatibility paths | Not needed |

## Files

| File | Contents | nanobot |
|---|---|---|
| `nanoclaw/agent/subagent.py` | `SubagentManager` (`spawn`, `run_inline`, the 4-slot queue, child tools and prompt, report), `SubagentStatus`, `SubagentResult` | `agent/subagent.py` |
| `nanoclaw/agent/tools/spawn.py` | `SpawnTool` | `agent/tools/spawn.py` |
| `nanoclaw/agent/tools/context.py` | `RequestContext`, `bind_request_context`, `current_request_context` | `agent/tools/context.py` (request part) |
| `nanoclaw/agent/runner.py` | `finalize_on_max_iterations`, `max_iterations_message`, `_max_iterations_fallback` | `agent/runner.py` |
| `nanoclaw/agent/loop.py` | `spawn` in the main tools; request context around each run; `process_subagent_result`, `_persist_subagent_followup`, `next_subagent_result`, `close` | `agent/loop.py` (`_build_turn`, `_persist_subagent_followup`, `aclose`) |
| `nanoclaw/__main__.py` | `ask` waits for background children and prints each follow-up reply | `cli/agent.py` (different: see difference #1) |
| `nanoclaw/templates/agent/subagent_system.md`, `subagent_announce.md` | The child's prompt and the report | same files (one word renamed: "NanoClaw's agent workspace") |

## Tests

`tests/test_subagents.py` (24 tests):

- the `spawn` schema matches nanobot's, and `spawn` errors outside a turn;
- the main agent has `spawn`; children and Dream don't;
- `wait=true`: the child's request (prompt, task, tools, temperature) and its answer as the tool result; temperature override; a child model error becomes an error result;
- the child's system prompt; out of iterations with no summary request;
- background: the reply text, the label rules, the report text, the follow-up turn (request, saved rows), replay in later turns, a report saved only once, failures and exceptions, the empty-answer fallback;
- at most 4 at once (the rest queued); `cancel_by_session` and `close`; waiting for running children;
- the runner's two new options;
- `nanoclaw ask` waits for a background child and prints both replies, also after an error;
- (second review) children follow a changed iteration limit; compaction on a repeated report puts the summary marker in the right place.

One existing test was updated: the main agent's tool list now includes `spawn`.

Every rule was checked by breaking it on purpose and confirming a test fails
(45 rules, plus 2 from the second review; two tests were added after this check found gaps: `wait=true` sends
no report, and `spawn` refuses a context without model settings). Two changes
cannot fail because they change nothing: the fallback texts for a child error
without error text and for an empty answer. NanoClaw's runner, like nanobot's,
always fills in both, so the fallbacks are kept as in nanobot.

## Review against nanobot

Checked by **running both projects** with scripted models (separate scripts
for the main agent and the children, because they run at the same time):

| Check | Result |
|---|---|
| `wait=true`, for a normal answer, a child model error and a child that runs out of iterations: the child's system prompt, every child request, the number of requests, the tool result the main agent sees, the saved conversation | Identical (child prompt: apart from difference #6) |
| Temperature override reaching the child | Identical |
| Background, for a long task (label cut at 30), a labelled task and a failing child: the `spawn` reply, the report text, the follow-up request, the saved conversation, the next user turn's request | Identical |
| The same report delivered twice: saved conversation and the second request | Identical |
| Tools offered | Main agent: ours are a subset of nanobot's 21 (with `spawn`); child: ours are a subset of nanobot's 12 (difference #5) |

33 checks, all identical once the differences above are allowed for. No new
nanobot bugs were found.

### Second review (after the build)

A second pass over nanobot's code, then more side-by-side runs for the paths
the first comparison didn't cover:

| Check | Result |
|---|---|
| Child request's `max_tokens` and temperature (no override) | Identical |
| A report whose task mentions `$weather` (the skill is loaded into the follow-up turn in both) | Identical request and saved conversation |
| A child whose tool output is too large: what the child sees, and where the file is saved | Identical |
| Two `wait=true` spawns in one reply | Identical requests and saved conversation (nanobot runs them at the same time: difference #7) |
| A report arriving after a crashed turn (the unanswered message gets the error reply first) | Identical request, saved conversation and flag |
| A report whose turn must compact first: archive request, follow-up request, summary, saved conversation, history journal | Identical (apart from the summary's timestamp) |
| The same, for a **repeated** report | **Different:** nanobot places the summary marker one message too early, so the next turn replays an already-summarized reply. Recorded as [upstream bug 5](../UPSTREAM_BUGS.md#5-summary-marker-one-message-too-early-on-a-repeated-subagent-report); NanoClaw's placement is correct (difference #8) |

Corrections made:

1. **Iteration limit re-read every turn.** nanobot copies the loop's limit to
   the subagents before each turn (`_sync_subagent_runtime_limits`); NanoClaw
   only set it once, so changing `loop.max_iterations` later didn't reach the
   children. Fixed, with a test.
2. **Regression test for the repeated-report boundary** (difference #8).
3. Recorded difference #7 (parallel `spawn` calls), which reading the code
   showed and the runs confirmed has no effect on the results.

Both new rules were checked by breaking them on purpose (caught).

