# Phase 2 progress

Checklist for phase 2 (plan: [`ROADMAP.md`](ROADMAP.md), "Phase 2 — the
production backend"). nanobot reference: `c4dac64`. Each layer: tasks →
sub-agents in parallel for independent tasks (each in its own worktree) →
review of every task against nanobot → integration → full test suite →
comparison and mutation checks → docs → PR.

Legend: ⏳ in progress · ✅ done and verified · ⛔ blocked

## Layer 11 — catch up with upstream (`p2/layer-11`)

| Task | Scope (files) | Sub-agent | Verification | Status |
|---|---|---|---|---|
| 11.1 System prompt | `agent/context.py`, `agent/skills.py`, `templates/agent/{identity,platform_policy}.md`, built-in skill descriptions | | | ⏳ |
| 11.2 Loop and subagent report | `agent/loop.py` (iteration-limit notice), `agent/subagent.py`, `templates/agent/subagent_announce.md` | | | ⏳ |
| 11.3 File, shell and network fixes | `tools/filesystem.py`, `tools/shell.py` (guard), `security/network.py` | | | ⏳ |
| 11.4 Schema, error hint, web | `tools/schema.py`, `tools/registry.py`, `tools/web.py`, `pyproject.toml` (ddgs) | | | ⏳ |
| 11.5 OpenAI reasoning models | `providers/openai_compat.py` | | | ⏳ |
| 11.6 Slash text goes to the model | `cli/chat.py` | | | ⏳ |
| 11.7 Windows test fixes | tests only (after 11.1–11.6) | | | |
| 11.8 Docs, comparison, mutation checks, PR | docs | main session | | |

## Blockers and open items

| Item | Status |
|---|---|
| Real-model tests need API keys as environment secrets | Open; FakeProvider only until then |
| PR #2 (docs) not merged; layer 11 branches from it | Open |
