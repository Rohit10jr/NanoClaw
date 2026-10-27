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

## Layer 13 — hooks, events, streaming (`p2/layer-13`) ⏳

Phase A (parallel, standalone pieces using nanobot's names), then phase B (wiring).

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 13.1 Provider streaming | `providers/base.py` (`chat_stream`, `chat_stream_with_retry`), `openai_compat.py` (SSE, `_parse_chunks`), `fake.py` | sub-agent (worktree) | | ⏳ |
| 13.2 Hooks and events | new `agent/hook.py`, `events.py`, `bus/outbound_events.py`, `agent/progress_hook.py`, `agent/turn_hooks.py`, think extractor | sub-agent (worktree) | | ⏳ |
| 13.3 Terminal stream renderer | new `cli/stream.py` | sub-agent (worktree) | | ⏳ |
| 13.4 Wiring | runner (hooks, streaming), loop (turn hook, events), chat/ask (renderer), retry and compaction events | after A | | |
| Comparison, mutation checks, docs, PR | | main session | | |

## Blockers and open items

| Item | Status |
|---|---|
| Real-model tests need API keys as environment secrets | Open; FakeProvider only until then |
| PR #2 (docs) not merged; layer 11 branches from it | Open |
