# 14 — Message bus, session workers, `/stop` and injections

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

Until now each turn was a direct function call: the chat called the loop,
waited for the reply, and only then read your next line. nanobot instead runs
the agent as a **long-running loop on a message bus**: input goes on the bus
as messages, one **worker per session** processes them in order, and replies
and events come back on the bus. That gives three things the direct call
couldn't:

- **`/stop` cancels a running turn**, and the work it finished is kept.
- **Messages typed while the agent works** are folded into the running turn
  instead of waiting for it to end.
- **Subagent reports** arrive on the bus like any message, so a turn can wait
  for its children and take their reports in.

The same pipeline later serves a gateway, a web UI or a chat app.

```
chat / channels ──InboundMessage──▶ MessageBus ──▶ AgentLoop.run()
                                                   ├─ priority commands (/stop, /status): handled at once
                                                   ├─ session busy → queued → injected into the running turn
                                                   └─ session idle → worker → turn (TurnDelivery)
chat / channels ◀──OutboundMessage + events (stream deltas, progress, final reply)──┘
```

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `InboundMessage`, `OutboundMessage` | Messages on the bus; the session key is `channel:chat_id` unless overridden | `bus/events.py` |
| `MessageBus` | Inbound and outbound queues, typed events, local subscribers | `bus/queue.py` |
| `CommandRouter`, builtin commands | The 8 slash commands, independent of the terminal: priority (`/stop`, `/status`, run at once), exact (`/compact` on the session worker), prefix (`/history n`); handlers return a reply instead of printing. Command specs (title, icon, lifecycle) as nanobot serves them to its web UI | `command/router.py`, `command/builtin.py` |
| Runner injections | Pending input is drained before every model call and after a final answer (the answer is kept and the turn goes on, stream left open); a turn about to end can wait for its running subagents | `agent/runner.py` `injection_callback`, `terminal_injection_callback` |
| Checkpoints and cancellation guards | The runner reports its progress at three points of a tool round; guards stop a dependency that swallows cancellation from starting more model work | `agent/runner.py:64`, checkpoint callback |
| `AgentLoop.run()` | Consumes the bus: priority commands inline; a busy session's input queued and injected (commands act as barriers); one worker per session under the session lock; a turn waits up to 300 s for its subagents | `agent/loop.py:1096-1819` |
| `cancel_active_tasks` | Cancels the session's worker and subagents, drops queued input, returns a count; the cancelled turn is restored from its latest checkpoint and saved (always, even without a checkpoint) | `agent/loop.py:1700-1745` (fix `421bd6b`) |
| `TurnDelivery` | Routes a turn's stream, progress and final reply to its chat, with a stream id per segment; started/running/completed/idle events | `agent/turn_delivery.py` |
| `restore_runtime_checkpoint`, `finish_turn` | Rebuild a cancelled turn: assistant message, finished tool results, "Error: Task interrupted before this tool finished." for pending calls | `session/recovery.py` |
| Subagent reports on the bus | Reported as a system message on the bus and processed as a follow-up turn, or injected into a running turn | `agent/subagent.py` |

## What you see in `nanoclaw chat`

| Situation | Behaviour |
|---|---|
| Normal turn | Same as layer 13 (streamed reply) |
| `/stop` while the agent works | "Stopped 1 task(s)." The turn stops at once; finished tool results stay in history, an unfinished tool call is saved as interrupted |
| Typing a message while the agent works | It is folded into the same turn; the model answers both together |
| A subagent finishes while the turn waits for it | Its report is folded into the turn |
| A subagent finishes after the turn | A follow-up turn reports it |
| Status line while working | `nanoclaw is working… type /stop to cancel` |

`nanoclaw ask` still makes one direct call (`process_direct`, as nanobot's
`agent -m`), now through the same turn pipeline.

## How it was built

Phase A in parallel: the bus and command router (14.1), the runner's
injections, checkpoints and cancellation guards (14.2). Phase B: the
bus-driven loop, workers, cancellation, `TurnDelivery`, subagent reports on
the bus and the chat over the bus (14.3).

| Task | Checked against nanobot |
|---|---|
| 14.1 Bus and router | 30 inputs through both routers: identical; the 8 command specs identical; chat output byte-identical before and after |
| 14.2 Runner | 15 scripted scenarios (injections at start, mid-tool, after answers, errors, empty replies, terminal waits, max iterations): byte-identical results |
| 14.3 Loop | 12 scenarios through both `AgentLoop.run()` (single turn, queued message injected, `/stop` in one and two tool rounds, subagent report during and after a turn, `/stop` while waiting for a subagent, `/status` while running, two sessions at once, `/history`, a failing turn, `/compact` as a barrier): identical outbound messages and saved sessions, except nanobot's extra web-search lines in `/status`. `nanoclaw chat` run in a real terminal for `/stop`, typing during a turn and Ctrl+C |

## Upstream bug found

**Bug 7:** a turn cancelled with `/stop` keeps only its latest tool round;
earlier finished rounds of the same turn are lost, because the checkpoint is
replaced each round. Reproduced with nanobot's own `AgentLoop.run()`; NanoClaw
behaves the same. See `UPSTREAM_BUGS.md`.

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Prompt while a turn runs | Classic chat waits for the reply and drops keys typed meanwhile (no `/stop` mid-turn) | The prompt stays open: `/stop` and new messages work during a turn; output is printed above the prompt without live redraw | Makes cancellation and injection usable in the terminal |
| 2 | `/new` on an idle session | Cancels its own worker and leaves it marked as cancelling | Doesn't cancel its own worker | Avoids a stuck state |
| 3 | `process_direct` | Returns the reply | Returns the run result; events go to the caller | Used by `ask`, `compact`, `dream` and tests |
| 4 | Checkpoint storage | A separate database row | Rewrites the session file each tool round | JSONL sessions (decision 2) |
| 5 | Command replies | Some texts mention "🐈 nanobot", web search usage | "nanoclaw", no web search lines | Names; no search usage tracking |
| 6 | Left out of the bus | Runtime admission, delivery ids, outbound recorder | — | Not needed without channels and the web UI |

User-visible changes that match nanobot: command replies on an idle session
are saved as hidden rows; an unexpected error shows "Sorry, I encountered an
error."; `/history` hides injected reports; layer 8 differences #3 and #4
(reports only after a turn, no wait for children) are gone.

## Left for later

Crash recovery and "Continue" after a restart (layer 15), idle
auto-compaction (layer 21; the loop's idle tick is in place), a global
concurrency limit, goals and cron turns.

Status (layer 21): idle auto-compaction is done (task 21.2,
`nanoclaw/agent/autocompact.py`; the idle tick calls
`_check_expired_sessions_if_due`), and `nanoclaw gateway` runs `AgentLoop.run`
as a long-running process. Cron turns come with tasks 21.1/21.4; the global
concurrency limit and goals are still open.

## Tests and mutation checks

Layer 14 adds 114 tests (1,111 → 1,225): `test_bus.py`,
`test_command_router.py`, `test_runner_injections.py`, `test_loop_bus.py`,
`test_cancellation.py` and additions to the chat and subagent tests, adapted
from nanobot's bus, router, runner-injection and loop tests.

90 rules were broken on purpose: 78 caught at once; of the 12 survivors, 10
were real gaps, closed with 11 new tests (e.g. `/status` counting a finished
task, `/new` from the bus taking the session lock, the worker continuing after
a failed message, a cancelled turn written to disk, one stream id per
segment); 2 are equivalent (re-queue order that can't differ on one event
loop; a deadline check that ends in the same empty result). Result: 88 of 90
caught, 2 equivalent.
