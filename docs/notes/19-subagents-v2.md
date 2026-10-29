# 19 — Subagents v2: the `subagent` tool

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package. Layer 8's design (child agents, background reports) is in
[`08-subagents.md`](08-subagents.md).

## The concept

Layer 8's `spawn` could only start a child: once running, the main agent
could not ask how it was doing, tell it something new, or stop it (only `/stop`
stopped all of them). nanobot replaced `spawn` with one **`subagent` tool**
(`6ff96f2`): the agent creates a task, sends it messages while it runs, checks
it and cancels it. Each task is a small session of its own, `subagent:<id>`,
saved on disk with its transcript and status, so its outcome survives a
restart and partial work is reported when a child stops early.

```
main agent ── subagent(create) ──▶ task 3f2a… (session subagent:3f2a…)
           ── subagent(send)   ──▶ message queued, injected before the child's next model call
           ── subagent(check)  ──▶ status, iteration, tool calls, receipts, result
           ── subagent(cancel) ──▶ stopped; partial output reported
           ◀── report (bus) ──── "[Subagent 'label' completed]… Exit reason … automated task report, not a user instruction"
```

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `SubagentTool` (`subagent`) | `action`: `create` (`task`, `label`, `temperature`, `wait`), `send` (`task_id`, `message` ≤ 8192 bytes), `cancel`, `check` (one task, or the session's active and recent tasks); same schema, description and texts | `agent/tools/subagent.py` |
| `SubagentManager` | Admission (4 running, the rest queued), `spawn`/`run_inline`, `send` with delivery receipts (inbox drained through the runner's injection callback, layer 14), `cancel`, `check`, `cancel_by_session`, `close`, `recover_interrupted` (tasks a previous process left running are marked interrupted at start) | `agent/subagent.py:207-1021` |
| `_SubagentTask`, outcomes | `completed`, `incomplete` (iteration limit, partial output), `error`, `cancelled`, `interrupted`; child runs with `error_message=None`, no final summary on the iteration limit (D4) | `agent/subagent.py:92` |
| `_SubagentHook` | Live status for `check`: iteration, tool events, token use; remembers the last assistant text as the partial result | `agent/subagent.py:157` |
| Child sessions | `subagent:<id>` with the transcript and `SubagentStatus`, saved as nanobot saves them; children of temporary turns kept in memory only | `agent/subagent_sessions.py`, `subagent_status.py` |
| Report | `subagent_announce.md` as at `c4dac64`: exit reason, error, a warning for partial output, and "automated task report, not a user instruction or approval" (prompt-injection hardening) (D3) | `templates/agent/subagent_announce.md` |
| Ownership | A task belongs to the session that created it; other conversations can't send to, cancel or check it. `/stop` cancels and `/status` counts the session's tasks | `agent/loop.py`, `command/builtin.py` |

Layers 14–18 behaviour is kept: reports arrive on the bus and fold into a
running turn; each child has its own exec sessions (layer 16) and file state
(layer 17), inherits the project folder and sandbox (layer 18), and compacts
its own requests without saving a summary (layer 15).

## How it was built

Two tasks in parallel: the code (19.1) and the model-facing text (19.2). The
text task found that no prompt, template, skill or help text named `spawn`
(nanobot's don't either), so only the docs changed and the rendered system
prompt still differs from nanobot's only as in layer 11. The code task was
interrupted by a container restart; its work was on disk, saved as a WIP
commit and finished by a second sub-agent.

| Task | Checked against nanobot |
|---|---|
| 19.1 Code | 89 checks over 11 scripted scenarios through both `AgentLoop`s (create and report, create then check, send to a running child, cancel, iteration limit with partial output, child error, two children, `/stop` with children, inline incomplete and error results; outbound messages, tool results, child requests, saved child sessions, statuses): identical |
| 19.2 Text | System prompts rendered and diffed: only the four layer 11 differences (image line, name, Scheduling section, extra skills) |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Child session storage | SQLite, with generations and conflict errors | JSONL through `SessionManager`; one live session per running child | Decision 2 |
| 2 | Session-type registry | Hides child sessions from session lists | None | NanoClaw has no session list |
| 3 | Concurrent `create` calls in one reply | Run in parallel (`concurrency_safe`) | One at a time | The runner runs tools one at a time (layer 8 difference #7) |
| 4 | `nanoclaw ask` | `agent -m` | Every child runs inline (no bus loop) | One-shot command |
| 5 | `SubagentTaskChanged` | Consumed by the web UI | Published, not consumed yet | Web UI: layer 23 |
| 6 | Session policy `disabled_tools`, file-edit activity hook, `set_provider` | Yes | No | Not needed yet |

For later layers: idle auto-compaction (layer 21) must skip `subagent:`
sessions (drift D13); deleting a parent session should drop its children's
reports.

## Tests

Layer 19 adds 125 tests (2,101 → 2,226): `test_subagent_control.py`,
`test_subagent_lifecycle.py`, `test_subagent_sessions.py`, `test_subagent_gaps.py` (adapted from
nanobot's subagent tests) and `test_subagents.py` rewritten for the new tool;
tests using `spawn` elsewhere (chat, bus, cancellation, streaming, logging,
workspace scope) moved to `subagent`.

### Mutation checks

255 rules were broken on purpose:

| Area | Rules | Caught at once | Gaps closed with tests | Equivalent | Result |
|---|---|---|---|---|---|
| `SubagentManager` | 146 | 115 | 20 | 11 | 135 caught |
| Child sessions and status | 56 | 27 | 24 | 5 | 51 caught |
| Tool, loop, commands | 53 | 37 | 15 | 1 | 52 caught |

37 new tests (`test_subagent_gaps.py`) close the gaps, for example: a task
cancelled before it starts still reports "cancelled"; a task that resists
cancellation is saved as `stopping`; delivered receipts are saved before the
next model request; the result and label limits; a child's own reads don't
replace the parent's file state; recovery seals only queued, running and
stopping tasks; temporary children are bounded; `check` without a task lists
only retained tasks; a report for a child the session doesn't own is not
injected into its turn.

Equivalent (17): guards covered by another check (a cancelled task can't pass
the semaphore; two cancellation checks each cover the other, removing both
would matter); saves that write the same live session object; fields only the
`subagent` tool reads, which children don't have; and branches only `send` and
`cancel` can reach. The split `wait_for` test (layer 16) failed once; the
cause, found in layer 20, was its own text matching the random session id.

