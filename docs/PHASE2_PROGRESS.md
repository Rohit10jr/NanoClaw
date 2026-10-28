# Phase 2 progress

Checklist for phase 2 (plan: [`ROADMAP.md`](ROADMAP.md), "Phase 2 — the
production backend"). nanobot reference: `c4dac64`. Each layer: tasks →
sub-agents in parallel for independent tasks (each in its own worktree) →
review of every task against nanobot → integration → full test suite →
comparison and mutation checks → docs → PR.

Legend: ⏳ in progress · ✅ done and verified · ⛔ blocked

## Layer 11 — catch up with upstream (`p2/layer-11`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 11.1 System prompt | `agent/context.py`, `agent/skills.py`, `templates/agent/{identity,platform_policy}.md`, 5 skill descriptions, `nanoclaw skills` | sub-agent (worktree) | Both projects' prompts rendered and diffed: only deliberate differences | ✅ |
| 11.2 Iteration-limit notice, subagent report | `agent/loop.py`, `runtime_context.py`, `session/manager.py`, `agent/subagent.py`, `subagent_announce.md` | sub-agent (worktree) | nanobot code run on the same scenarios; exact strings asserted | ✅ |
| 11.3 File, shell, DNS fixes | `tools/filesystem.py`, `tools/shell.py`, `security/network.py` | sub-agent (worktree) | Fail-before/pass-after per fix; nanobot's 53 regression tests pass on the same inputs | ✅ |
| 11.4 Schema, error hint, web | `tools/schema.py`, `tools/registry.py`, `tools/web.py`, ddgs pin | sub-agent (worktree) | 7,922 cases identical to nanobot | ✅ |
| 11.5 OpenAI reasoning models | `providers/openai_compat.py` | sub-agent (worktree) | Request settings identical in 17 cases | ✅ |
| 11.6 Slash text goes to the model | `cli/chat.py` | sub-agent (worktree) | nanobot's tests adapted; 22 fail on old code | ✅ |
| 11.7 Windows test fixes | tests only | sub-agent (worktree, after 11.1–11.6) | Linux suite unchanged; simulated no-console and encoding checks; not run on Windows | ✅ |
| Upstream bugs re-check | read-only | sub-agent | Bug 1 fixed upstream; 2–6 reproduced at `c4dac64` | ✅ |
| Integration | merges | main session | One conflict (old hint text in a new test), fixed; 784 passed | ✅ |
| Mutation checks | 25 rules | main session | 25/25 caught | ✅ |
| Docs | note 11, CODE_MAP, ROADMAP, README, WINDOWS, UPSTREAM_BUGS, stale lines | main session | — | ✅ |
| PR | `p2/layer-11` → `main` | main session | PR #3 | ✅ |

## Layer 12 — foundations (`p2/layer-12`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 12.1 Config file | new `nanoclaw/config/`, `__main__.py`, `providers/factory.py`, iteration defaults (200), `nanoclaw config` | sub-agent (worktree) | Same nanobot-style config loaded through both: identical shared values; nanobot's full default config loads; `config init` output loads in nanobot | ✅ |
| 12.2 Logging | loguru, `cli/log_control.py`, log calls in memory/session/subagent/tool results/web | sub-agent (worktree) | Texts checked against nanobot lines | ✅ |
| 12.3 Retries and usage | `providers/base.py`, `openai_compat.py`, `runner.py`, memory model calls, `LLMUsage` | sub-agent (worktree) | 15 SDK errors: same decision and wait as nanobot | ✅ |
| 12.4 Remaining log calls | `runner.py`, `loop.py`, `violations.py`, `utils/tokens.py` | sub-agent (worktree, after integration) | Diff reviewed: only logging plus a count in one return value | ✅ |
| Integration | retry mode from config, `--logs`, provider retry logs | main session | 3 merges without conflicts | ✅ |
| Mutation checks | 23 rules | main session | 22 caught, 1 equivalent (5 gaps closed with new tests) | ✅ |
| Docs | note 12, CODE_MAP, README, ROADMAP | main session | — | ✅ |
| PR | `p2/layer-12` → `main` | main session | PR #4 | ✅ |

## Layer 13 — hooks, events, streaming (`p2/layer-13`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 13.1 Provider streaming | `providers/base.py`, `openai_compat.py`, `fake.py` | sub-agent (worktree), phase A | 20 chunk sequences: 19 identical to nanobot, 1 deliberate; retry rule read against nanobot | ✅ |
| 13.2 Hooks and events | `agent/hook.py`, `events.py`, `bus/outbound_events.py`, `agent/progress_hook.py`, `agent/turn_hooks.py`, think extraction | sub-agent (worktree), phase A | Byte-identical events for the same callback sequences | ✅ |
| 13.3 Terminal stream renderer | `cli/stream.py` | sub-agent (worktree), phase A | 44/56 outputs identical, 12 differ only by escaping (re-run by main session) | ✅ |
| 13.4 Wiring | runner, loop, memory, chat, ask, channel settings | sub-agent (worktree), phase B | Identical event sequences for scripted turns; identical chat-turn output; real terminal run | ✅ |
| Mutation checks | 83 rules | sub-agent (worktree) | 81 caught, 2 equivalent (8 tests added for 10 gaps) | ✅ |
| Docs | note 13, CODE_MAP, README, notes 02/10, UPSTREAM_BUGS (bug 6 in reasoning lines) | main session | — | ✅ |
| PR | `p2/layer-13` → `main` | main session | PR #5 | ✅ |

## Layer 14 — bus, session workers, `/stop`, injections (`p2/layer-14`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 14.1 Bus and command router | `bus/events.py`, `bus/queue.py`, `command/router.py`, `command/builtin.py`; chat uses the router | sub-agent (worktree), phase A | 30 router inputs identical; 8 specs identical; chat output byte-identical | ✅ |
| 14.2 Runner injections and cancellation | `agent/runner.py` | sub-agent (worktree), phase A | 15 scenarios byte-identical to nanobot | ✅ |
| 14.3 Bus-driven loop | loop `run()`, workers, `TurnDelivery`, cancellation, recovery restore, subagent reports on the bus, chat over the bus | sub-agent (worktree), phase B | 12 scenarios through both `AgentLoop.run()`: 11 identical, `/status` differs only by web-search lines (re-run by main session); real terminal runs | ✅ |
| Upstream bug 7 | cancelled turn loses earlier rounds | main session | Confirmed in nanobot's own run | ✅ |
| Mutation checks | 90 rules | sub-agent (worktree) | 88 caught, 2 equivalent (11 tests added) | ✅ |
| Docs | note 14, CODE_MAP, README, notes 08, UPSTREAM_BUGS | main session | — | ✅ |
| PR | `p2/layer-14` → `main` | main session | PR #6 | ✅ |

## Layer 15 — robustness (`p2/layer-15`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 15.1 Crash recovery | `session/recovery.py`, loop restore stage | sub-agent (worktree), parallel | 21 checkpoint cases byte-identical to nanobot; real process killed mid-tool, next run restores the turn | ✅ |
| 15.2 Context governance | `agent/context_governance.py`, runner call site, loop (replaces `_compact_if_needed`), memory, subagents | sub-agent (worktree), parallel | 109 checks, 0 differences (re-run by main session); own mutation run 28/29, 1 equivalent | ✅ |
| Integration | merges | main session | No conflicts; 1,287 passed | ✅ |
| Mutation checks | 96 rules (+29 by 15.2) | sub-agent (worktree) | 90 caught, 6 equivalent (22 tests added); 1,309 passed | ✅ |
| Docs | note 15, CODE_MAP, README, ROADMAP | main session | — | ✅ |
| PR | `p2/layer-15` → `main` | main session | PR #7 | ✅ |

## Layer 16 — shell: exec sessions, Windows PowerShell (`p2/layer-16`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 16.1 Exec sessions | new `agent/tools/exec_session.py`, `shell_output.py`; `shell.py` `exec` through sessions (POSIX) | sub-agent (worktree), phase A | 44 scenarios through both: 41 identical, 3 deliberate (guard always on); checked by main session | ✅ |
| 16.2 Exec settings, Windows paths in violations | `config/schema.py` `tools.exec`, `tools/violations.py` | sub-agent (worktree), phase A | 16 config files loaded through both: identical; nanobot has the same Windows-path gap (NanoClaw fix) | ✅ |
| 16.3 Wiring | loop (`/new`, shutdown, pending work), subagents, config → tools, tool hints, chat | sub-agent (worktree), phase B | 21 loop-level checks through both `AgentLoop`s with real processes: identical; `ask` leaves no process behind | ✅ |
| 16.4 Windows shell | PowerShell/cmd, Windows environment, Job Object + `taskkill` tree kill | sub-agent (worktree), phase C | 486 Windows cases identical (monkeypatched); not yet run on Windows | ✅ |
| Integration | merges | main session | No conflicts; 1,596 passed, 6 Windows-only skipped | ✅ |
| Mutation checks | 193 rules | sub-agent (worktree) | 192 caught, 1 equivalent (38 tests added); 1,634 passed | ✅ |
| Docs | note 16, CODE_MAP, README, ROADMAP, WINDOWS, notes 03/08 | main session | — | ✅ |
| PR | `p2/layer-16` → `main` | main session | PR #8 | ✅ |

## Layer 17 — more tools: `apply_patch`, `rg`, file state, MCP (`p2/layer-17`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 17.1 `apply_patch`, edit options | new `agent/tools/apply_patch.py`; `filesystem.py` descriptions, `edit_file` `line_hint`/`expected_replacements` | sub-agent (worktree), parallel | 78 cases identical (results and file bytes); re-run after integration | ✅ |
| 17.2 `rg` | new `agent/tools/rg.py` (on exec sessions), registration when ripgrep is installed | sub-agent (worktree), parallel | 45 cases identical after the guard fix in integration | ✅ |
| 17.3 File state | new `agent/tools/file_state.py`, `read_file` `force`, runner read context | sub-agent (worktree), parallel | 19 runner scenarios identical; re-run after integration | ✅ |
| 17.4 MCP | new `agent/tools/mcp.py` (stdio + streamable HTTP, tools only), `mcpServers` config, loop lifecycle, registry order | sub-agent (worktree), parallel | Definitions 10/10, calls 19/20 identical (images: deliberate) | ✅ |
| Integration | merges, `apply_patch` → file state, `rg` guard rule, a racy test fixed | main session | 3 small conflicts; 1,907 passed, 6 skipped | ✅ |
| Mutation checks | 274 rules | sub-agent (worktree) | 264 caught, 10 equivalent (79 tests added); 1,986 passed | ✅ |
| Docs | note 17, CODE_MAP, README, ROADMAP, notes 03/06/07/08/09 | main session | — | ✅ |
| PR | `p2/layer-17` → `main` | main session | PR #9 | ✅ |

## Layer 18 — isolation: bubblewrap sandbox, separate project folder (`p2/layer-18`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 18.1 Sandbox | new `agent/tools/sandbox.py` (bubblewrap), `exec`/`rg` wrapping, `tools.exec.sandbox` forcing restriction, bind roots in the guard | sub-agent (worktree), parallel | 62 cases: 60 identical, 2 deliberate (no seatbelt); bubblewrap installed, live tests run | ✅ |
| 18.2 Project folder | workspace scope, `--project`, tools resolving against the project, `# Current Project` prompt section, `AGENTS.md` from the project, subagents | sub-agent (worktree), parallel | 564 checks identical | ✅ |
| Integration | merge, sandbox restriction under a project scope (gap found and fixed), sandbox line in the runtime block | main session | 1 conflict; comparisons re-run; 2,075 passed, 6 skipped | ✅ |
| Mutation checks | 214 rules | sub-agent (worktree) | 206 caught, 8 equivalent (26 tests added); 2,101 passed | ✅ |
| Docs | note 18, CODE_MAP, README, ROADMAP, notes 03/04/08/09 | main session | — | ✅ |
| PR | `p2/layer-18` → `main` | main session | PR #10 | ✅ |

## Layer 19 — subagents v2: the `subagent` tool (`p2/layer-19`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 19.1 Subagent manager v2 | `agent/subagent.py` (tasks, outcomes, status hook, partial results, send/cancel/check, inbox via injections), child sessions `subagent:<id>`, `subagent_status`, report template | sub-agent (worktree); interrupted by a container restart, finished by a second sub-agent from the saved work | 89 checks through both `AgentLoop`s identical | ✅ |
| 19.2 The `subagent` tool | 19.1 also takes the tool and the tests; 19.2 the model-facing text (prompts, templates, skills, help) and docs | sub-agent (worktree), in parallel with 19.1 | No model-facing text named `spawn`; prompts differ from nanobot only as in layer 11 | ✅ |
| Integration | merges | main session | No conflicts; 2,189 passed, 6 skipped | ✅ |
| Mutation checks | 255 rules | sub-agent (worktree) | 238 caught, 17 equivalent (37 tests added); 2,226 passed | ✅ |
| Docs | note 19, CODE_MAP, README, ROADMAP, note 08 | main session | — | ✅ |
| PR | `p2/layer-19` → `main` | main session | PR #11 | ✅ |

## Layer 20 — providers: registry, presets, `/model`, Anthropic, fallback (`p2/layer-20`) ✅

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 20.1 Registry, presets, `/model` | `providers/registry.py`, `factory.py`, model presets and runtime, `--preset`, `/model`, config | sub-agent (worktree), phase A | 63 routing cases: 60 identical, 3 deliberate; `/model` 16/16; re-run after integration | ✅ |
| 20.2 Anthropic adapter | new `providers/anthropic_provider.py`, `anthropic` dependency | sub-agent (worktree), phase A | 134 request/response/stream/error comparisons: 2 deliberate (malformed replay args), 0 unexpected | ✅ |
| 20.3 Fallback models | `providers/fallback_provider.py`, `fallbackModels` | sub-agent (worktree), phase B | 72 scenarios identical; re-run after merging | ✅ |
| Integration | merges, factory builds the real Anthropic adapter | main session | No conflicts; comparisons re-run; 2,596 passed, 6 skipped | ✅ |
| Mutation checks | 309 rules | sub-agent (worktree); interrupted by a container restart, finished by a second sub-agent | 300 caught, 9 equivalent (59 tests added); 2,655 passed | ✅ |
| Docs | note 20, CODE_MAP, README, ROADMAP, notes 01/12 | main session | — | ✅ |
| PR | `p2/layer-20` → `main` | main session | PR #12 | ✅ |

## Blockers and open items

| Item | Status |
|---|---|
| Real-model tests need API keys as environment secrets | Open; FakeProvider only until then |
| Windows shell (16.4) not yet run on Windows | Needs a run of `uv run pytest` on Windows by the user |
| PRs #2–#4 merged; #5–#11 (layers 13–19) open; later layers stacked on them | — |
| Rewrite earlier commits to the user's author | After phase 2 (user decision) |
