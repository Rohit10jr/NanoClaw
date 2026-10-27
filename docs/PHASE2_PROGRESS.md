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

## Layer 12 — foundations (`p2/layer-12`) ⏳

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 12.1 Config file | new `nanoclaw/config/`, `__main__.py`, `providers/factory.py`, iteration defaults (200), `nanoclaw config` | sub-agent (worktree) | | ⏳ |
| 12.2 Logging | loguru, `cli/log_control.py`, log calls in memory/session/subagent/skills/tools | sub-agent (worktree) | | ⏳ |
| 12.3 Retries and usage | `providers/base.py`, `openai_compat.py`, `runner.py`, memory model calls, `LLMUsage` | sub-agent (worktree) | | ⏳ |
| Integration | wire `providerRetryMode` and `--logs`; merges | main session | | |
| Comparison, mutation checks, docs, PR | | main session | | |

## Blockers and open items

| Item | Status |
|---|---|
| Real-model tests need API keys as environment secrets | Open; FakeProvider only until then |
| PR #2 (docs) not merged; layer 11 branches from it | Open |
