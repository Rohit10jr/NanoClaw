# 21 — Runtime: gateway, cron, heartbeat, scheduled Dream, memory history

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

Until now NanoClaw only ran while you used it: a `chat` session or one `ask`.
nanobot is also a **long-running process**, the **gateway**, that keeps the
agent alive, runs **scheduled jobs** and is what channels (Telegram, the web
UI) connect to. This layer adds the gateway and what runs inside it:

- **Cron**: the agent can schedule reminders and recurring tasks for itself
  (`cron` tool); due jobs come back as turns in the chat that created them.
- **Scheduled Dream**: memory consolidation every 2 hours instead of only on
  `/dream`.
- **Heartbeat**: every 30 minutes the agent reads `HEARTBEAT.md`; if it lists
  active tasks, it works on them and a small model call decides whether the
  result is worth sending to you.
- **Idle compaction**: conversations idle for 15 minutes are summarized in
  the background, so the next message starts small.
- **Memory history**: the memory files live in a small git repository; every
  Dream run is a commit you can inspect and undo.

```
nanoclaw gateway ── AgentLoop.run() (bus, session workers, idle tick → AutoCompact)
                 ├─ CronService ── user jobs → run_bound_cron_job → cron turn in the job's session
                 │               ├─ "dream"     → Dream → commit_dream_changes (GitStore)
                 │               └─ "heartbeat" → HEARTBEAT.md tasks → turn in session "heartbeat"
                 │                                → evaluate_response → deliver to the latest channel chat
                 └─ GET /health
```

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `CronService` | Jobs in `workspace/cron/jobs.json` (atomic writes, corrupt-file backup, external edits merged through an action log); schedules `at`, `every`, `cron` with time zones (croniter); one timer; run history (20); one-shot jobs disabled or deleted; protected system jobs | `cron/service.py`, `cron/types.py` |
| `CronTool` (`cron`) | `add`/`list`/`remove`/`update`/`enable`/`disable`/`history`; exactly one of `every_seconds`, `cron_expr`, `at`; jobs bound to the current chat; no `add` inside a cron turn. Main agent only | `agent/tools/cron.py` |
| Cron turns | `run_bound_cron_job` renders `cron_reminder.md` and submits it to the job's session; a busy session runs it after its other input; never injected into a running turn; saved as "Scheduled cron job triggered: …" | `cron/bound_runner.py`, `agent/cron_turns.py` |
| Scheduling prompt section | The tool contract's `## Scheduling` section when the cron tool is registered (layer 11 difference #1 closed) | `templates/agent/tool_contract.md` |
| `nanoclaw gateway` | Foreground process: loop, MCP, cron, `/health` (`gateway.host`/`port`, default `127.0.0.1:18790`), SIGINT/SIGTERM graceful shutdown (a second signal forces), banner. With no channels it warns and keeps running | `cli/gateway.py`, `cli/gateway_runtime.py` |
| `AutoCompact` | On the loop's 1 s idle tick, at most every `idleCompactCheckIntervalSeconds` (60): sessions idle `idleCompactAfterMinutes` (15) with unarchived messages are summarized; Dream, `subagent:`, busy and archiving sessions skipped; re-checked under the session lock; the summary is used by the next turn | `agent/autocompact.py`, `agent/loop.py` |
| `GitStore` | dulwich repository in the workspace tracking `SOUL.md`, `USER.md`, `memory/MEMORY.md`; Dream commits (`dream: manual run`, `dream: periodic memory consolidation`); log, diff, revert | `utils/gitstore.py` |
| `/dream-log`, `/dream-restore` | A Dream change as a diff; list versions; undo one as a new commit | `command/builtin.py` |
| Prompt overrides | `/dream-prompt` and `/evaluator-prompt` copy a built-in prompt to `prompts/*.md` in the workspace, where it can be edited | `utils/workspace_prompts.py` |
| Scheduled Dream | Protected job `dream`, every `dream.intervalH` (2) hours or a legacy `dream.cron`; commits after each run | `cli/gateway_runtime.py:660-722` |
| Heartbeat | Protected job `heartbeat` every `gateway.heartbeat.intervalS` (1800 s); only lines under `## Active Tasks` in `HEARTBEAT.md` count; target is the latest non-CLI chat of an enabled channel (none while only the terminal exists); `evaluate_response` decides whether to deliver (fail-closed); the delivery is recorded in the target chat | `cli/gateway_runtime.py:244-800`, `utils/evaluator.py` |

```json
{
  "gateway": { "port": 18790, "heartbeat": { "enabled": true, "intervalS": 1800 } },
  "agents": { "defaults": { "idleCompactAfterMinutes": 15, "dream": { "intervalH": 2 } } }
}
```

## How it was built

Phase A in parallel: cron (21.1), the gateway with idle compaction (21.2), the
memory history and prompt overrides (21.3). Phase B: scheduled Dream and
heartbeat as system jobs in the gateway (21.4). Integration: three conflicts
(dependencies, the loop, the CLI), all keep-both; `uv.lock` regenerated with
uv. 21.4 also restored nanobot's full `AGENTS.md` template (cron and
heartbeat sections); so that older workspaces' unedited copies don't start
appearing in the prompt, the earlier template's hash still counts as unedited.

| Task | Checked against nanobot |
|---|---|
| 21.1 Cron | 104 cases (next runs incl. DST in three cities, resume rule, tool results and errors, store after edit sequences, job execution outcomes, bound-runner messages, prompt texts): identical |
| 21.2 Gateway, idle compaction | Idle compaction 27/27 identical; `/health` JSON 4/4; real gateway subprocesses stopped with SIGTERM and SIGINT |
| 21.3 Memory history | 9 scenarios through real loops and command routers: 208 identical, 72 expected (product name, shorter prompts README, layer 6 difference #9) |
| 21.4 System jobs | Through both real gateway startups with a scripted model: HEARTBEAT.md parsing, target selection, evaluator requests and decisions, system-job registration, scheduled Dream and heartbeat runs: 1,061 identical, 5 expected (product name in the first commit) |

## Behaviour copied from nanobot worth knowing

- Every Dream run commits, even without a memory change: Dream's cursor file
  is tracked (by name, though `.gitignore` excludes it).
- A workspace inside another git repository gets no memory history, and Dream
  then always reports "no memory changes".
- dulwich reads your global git config: `commit.gpgsign` it can't satisfy
  makes Dream commits fail.
- Jobs created in `chat` or `ask` only fire while the gateway runs.

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Background gateway, OS services, update admission | Yes | Foreground only | Later |
| 2 | `/health` `instance_id` | Hash of the update-profile path | Hash of config path and workspace | No updater |
| 3 | Gateway crash after startup | Exit 0 | Exit 1 | Clearer for scripts |
| 4 | Web UI branches (cron metadata, archived chats, recovery) | Yes | No | Layer 23 |
| 5 | Heartbeat's message tool switched off | Yes | No message tool; same effect | — |
| 6 | Session generations when recording a delivery | Checked | Recorded under the session lock | JSONL sessions |
| 7 | Product name in commits, templates and prompt texts | nanobot | nanoclaw | — |

## Tests

Layer 21 adds 564 tests (2,655 → 3,219 passed): `tests/cron/` (nanobot's cron
tests), `test_cron_loop.py`, `test_gateway.py` (incl. real subprocesses),
`test_autocompact.py`, `test_autocompact_unit.py`, `test_gitstore.py`,
`test_dream_history.py`, `test_gateway_jobs.py`, `test_layer21_gaps.py`.

### Mutation checks

420 rules were broken on purpose:

| Area | Rules | Caught at once | Gaps closed with tests | Equivalent | Result |
|---|---|---|---|---|---|
| Cron, cron tool, cron turns, Scheduling section | 185 | 121 | 60 | 4 | 181 caught |
| Gateway, settings | 85 | 66 | 19 | 0 | 85 caught |
| Idle compaction | 38 | 32 | 3 | 3 | 35 caught |
| Memory history, prompt overrides, templates | 68 | 61 | 7 | 0 | 68 caught |
| Heartbeat, evaluator | 44 | 33 | 11 | 0 | 44 caught |

69 new tests (`test_layer21_gaps.py`) close the gaps, for example: a cron
schedule's time zone (Tokyo, New York, the 2026-03-08 clock change); a stale
external edit for a job moved to another chat ignored; run history keeping the
newest 20; a successful run clearing the last error; deferred cron turns in
order and never injected into a running turn; an unknown time zone in `update`
returned as a tool error; scheduled bus events drained before the gateway exits
and signal handlers restored; IPv6 health URLs; the heartbeat running as
session `heartbeat` with the target's chat and MCP connected; the evaluator
failing closed on a missing decision; the diff cut at 6,000 characters.

Equivalent (7): a boundary where both sides give the same next run; enabled
checks repeated before a job runs (and every disable also clears its next run);
deferred cron turns that only exist while their session queue does; and
summary reads in temporary turns that always find the same value.

