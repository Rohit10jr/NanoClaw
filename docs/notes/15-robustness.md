# 15 — Robustness: crash recovery and context governance

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

Two ways a long turn could go wrong until now:

- **The process dies mid-turn** (killed, crashed, the laptop lid closed).
  Layer 14 saved a checkpoint after every tool round for `/stop`; nothing read
  it back after a crash.
- **A single turn outgrows the context window.** Phase 1 checked the size once,
  before the turn's first request (layer 6 difference #1). A long tool loop
  could still send a request that is too big, and the model returned an error.

nanobot handles both. The checkpoint is restored at the start of the next
turn, and a **context governor** runs before every model request: it repairs
the request, measures it, and compacts it if needed. This layer ports both.

```
next turn ─▶ _restore_turn ─▶ restore_runtime_checkpoint (crash) ─▶ user-only tail ─▶ commands ─▶ runner
runner ─▶ before every request: ContextGovernor.prepare_request
             repair copy ─▶ budget tool results ─▶ measure ─▶ compact H (keep Δ) ─▶ fit unread reads ─▶ send
```

## Crash recovery (15.1)

| Piece | What it does | nanobot |
|---|---|---|
| `restore_runtime_checkpoint` | Rebuilds the killed turn from its latest checkpoint: the assistant message, the finished tool results, and "Error: Task interrupted before this tool finished." for calls still running. Rows already at the end of the session are not added twice. Malformed rows are left out; a checkpoint that isn't a dict is ignored until the next save clears it | `session/recovery.py:278` |
| `runtime_checkpoint_is_well_formed` | Strict check of a checkpoint's phase and rows (`awaiting_tools`, `tools_completed`, `final_response`) | `session/recovery.py:216` |
| `discard_runtime_checkpoint` | Drops a checkpoint without restoring it | `session/recovery.py:376` |
| Restore stage of a turn | Checkpoint first, then a user message left without a reply, before any command runs (so `/history` after a crash already shows the restored turn) | `agent/loop.py:2134` |

What you see: kill `nanoclaw chat` or `nanoclaw ask` while a tool runs, start
it again, and the next turn begins with the interrupted turn in its history;
the model sees which tool finished and which didn't.

## Context governance (15.2)

`nanoclaw/agent/context_governance.py`, called from `AgentRunner` before every
model request (between tool calls, for retries, for the final summary request
and inside subagents). nanobot: `agent/context_governance.py`.

| Step | What | nanobot |
|---|---|---|
| Repair the copy sent to the model | Remove placeholders and nameless tool calls, drop tool results without a call (or answering one twice), add an error result for a call whose result is missing, re-apply the tool-result budget, merge adjacent user messages (not the summary continuation or hidden subagent reports). The saved transcript is not changed | `:262-456` |
| Measure | The provider's `context_tokens` when the previous request was exactly this one, or a prefix of it and the provider counted more (nanobot D8); otherwise the estimate | `request_pressure`, `:506` |
| H and Δ | The last request as sent is the accepted history H; the next request is H plus the raw messages added since (Δ) | `ContextCompactionState`, `:138` |
| Compact | Summarize H; the request becomes system prompt + summary, "Continue the active task…" when Δ has no user message, then Δ. Event `ContextCompactionEvent(reason="context_pressure")` | `_compact_request_history`, `:623` |
| Still too big | The largest unread `read_file` results in Δ become a notice to read a smaller range, also in the saved turn (nanobot D9); otherwise `ContextWindowExceededError` | `_fit_pending_file_reads`, `:458` |
| Save | The summary returns as a `SessionSummaryCheckpoint`; `_save_turn` commits it where H ended, also between two of the turn's messages. Subagents and temporary turns never journal or commit (`persist=False`) | `prepare_request`, `:716`; `session/summary.py:30` |

`AgentLoop._compact_if_needed` (phase 1's once-per-turn check) is gone. Layer 6
differences #1 and #4 are closed; layer 8's "compaction during a child's run"
is done.

**The bug 4 switch** (`NANOCLAW_COMPACTION=fixed`) works unchanged: the
summarizer is still `Consolidator.summarize_transcript`, so `fixed` summarizes
an oversized archive request in parts, mid-turn too. One case it can't split:
accepted history that is a single user message plus tool steps (no second user
message to split at); it falls back to its keep-the-newest raw dump.

## How it was built

Two independent tasks in parallel, each by a sub-agent in its own worktree,
merged without conflicts.

| Task | Checked against nanobot |
|---|---|
| 15.1 Crash recovery | 21 checkpoint cases (each phase, malformed rows, overlap, non-dict) through both `restore_runtime_checkpoint` and well-formedness checks: byte-identical; a real `nanoclaw ask` process killed mid-tool, then the next run restores the turn |
| 15.2 Context governance | 109 checks through both governors (repair helpers, merging, budgets, pressure with and without provider counts, prefix rule, compaction with and without a user message in Δ, read fitting, overflow error, event reasons, saved boundaries): 0 differences in `nanobot` mode (re-run by the main session) |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Checkpoint storage | A database row | The JSONL session file, rewritten each tool round | JSONL sessions (decision 2) |
| 2 | Provider conversation state, provider-native compaction (Responses API) | Yes | No | NanoClaw has neither yet (layer 20) |
| 3 | Preparation in a worker thread | Yes | Inline | Simpler; the same result |
| 4 | Usage after a fallback provider or native compaction (nanobot D10) | Not a baseline | Always the next baseline | No fallback providers yet (layer 20) |

Upstream bug 7 (a cancelled turn keeps only its latest tool round) is still
present in both: the crash path restores the same single checkpoint.

## Tests

Layer 15 adds 62 tests (1,225 → 1,287): `tests/test_recovery.py` (39, adapted
from nanobot's recovery and loop-restore tests, plus a real subprocess killed
mid-tool) and `tests/test_context_governance.py` (+23); two `test_compaction.py`
tests now expect compaction inside the runner.

### Mutation checks

MUTATION_RESULTS
