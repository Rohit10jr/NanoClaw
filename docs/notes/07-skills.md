# 07 — Skills

Reference: nanobot at commit `d3d70f2`. Paths are relative to its `nanobot/` package.

## The concept

A **skill** is a folder with a `SKILL.md` file: a short header (name,
description, optional requirements) and step-by-step instructions, sometimes
with helper scripts. For example, `weather/SKILL.md` explains how to get a
forecast with `curl wttr.in/...`.

```markdown
---
name: weather
description: Get current weather and forecasts (no API key required).
metadata: {"nanobot":{"emoji":"🌤️","requires":{"bins":["curl"]}}}
---

# Weather
...
```

**Skills are loaded only when needed.** The system prompt lists each skill's
name, description and path, about one line per skill. When a request matches,
the model reads the full file with `read_file`. Twenty skills cost twenty lines
of prompt, not twenty documents.

## Where skills come from

| Source | Folder | Shown as |
|---|---|---|
| Your skills | `<workspace>/skills/<name>/SKILL.md` (created empty on first run) | "Workspace skills" |
| Built-in skills | `nanoclaw/skills/<name>/SKILL.md` | "Built-in skills" |

A workspace skill with the same name as a built-in one **replaces** it, so you
can customise a built-in skill by copying it into `skills/`.

Both groups are shown under the root `skills`, so the model asks for
`skills/weather/SKILL.md`. That file isn't in the workspace for a built-in
skill, so `read_file` looks for a missing `skills/<name>/...` file in the
built-in folder (nanobot `_builtin_skill_read_path`). It never leaves that
folder (`skills/../agent/...` is refused).

## What the model sees

At the end of the system prompt (after memory, before a conversation summary):

```
# Skills

The following skills extend your capabilities. Each group lists one root and relative SKILL.md paths; join them when using `read_file`.

### Workspace skills (`skills`)
- **release** — How we release.  `release/SKILL.md`

### Built-in skills (`skills`)
- **weather** — Get current weather and forecasts (no API key required).  `weather/SKILL.md`
- **summarize** — Summarize or extract text/transcripts ... (unavailable: CLI: summarize)  `summarize/SKILL.md`
```

| Feature | How | nanobot |
|---|---|---|
| **Requirements** | `metadata.nanobot.requires.bins` (programs) and `.env` (environment variables). A skill with something missing is still listed, marked `(unavailable: CLI: gh, ENV: TOKEN)` | `skills.py:266–345` |
| **Always-on skills** | `always: true` (top level or in `metadata`) puts the whole skill in every prompt under `# Active Skills`; it is not listed again under `# Skills` | `skills.py:350`, `context.py:132` |
| **`$skill-name` in a message** | Writing `$weather` loads that skill's full text for this turn, appended to your message as runtime context (`[Active Skills — instructions for this user turn]`). It is saved with the message, like the time block | `skills.py:165, 182`, `context.py:323`, `loop.py:762` |

`$` must start a word: `US$weather` doesn't count. Unknown or unavailable
skills, and always-on ones, are ignored.

## Built-in skills

Copied from nanobot; `nanobot` in the text renamed (the `"nanobot"` key in
`metadata` is kept, so skills work in both projects):

| Skill | Needs | What it does |
|---|---|---|
| `skill-creator` (with `scripts/`) | — | How to write a good skill; scripts to create, validate and package one. Dream points to it |
| `memory` | — | How to search the history journal `memory/history.jsonl` |
| `weather` | `curl` | Forecasts from wttr.in / Open-Meteo |
| `github` | `gh` | Issues, PRs and CI with the GitHub CLI |
| `tmux` | `tmux` | Driving interactive programs in tmux (socket variable renamed `NANOCLAW_TMUX_SOCKET_DIR`) |
| `summarize` | `summarize` | Summarizing URLs, files and videos |

Left out (they need features NanoClaw doesn't have): `cron` (cron tool),
`image-generation` (image tool), `my` (nanobot's self-inspection tool),
`update-setup` (nanobot's upgrade flow), `clawhub` (installs third-party skills
from a public registry; see "What we left out").

## Dream writes skills

Layer 6 had removed the skills part of Dream. It is back, as in nanobot:

- Dream's instructions include the "Skills" rules: create a skill only when a
  workflow has appeared at least twice, merge into an existing skill when one
  overlaps, follow the skill-creator format (path given absolutely).
- Dream may now write under `skills/` (besides `MEMORY.md`, `SOUL.md`,
  `USER.md`) and read the built-in skills. Errors for other paths use
  nanobot's exact text: `Error: Path notes.txt is outside allowed directory
  <workspace>/skills (this is a hard policy boundary, ...)`.

## CLI

```bash
uv run nanoclaw skills          # list all skills with descriptions (nanobot's /skill)
uv run nanoclaw ask 'What is the $weather in Paris?'   # load a skill for this turn
```

(Single quotes keep the shell from expanding `$weather`.)

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why | Later |
|---|---|---|---|---|---|
| 1 | Built-in skills | 11 | 6 (see above) | The other 5 need features we don't have. Chosen with the user | Some with later layers / phase 2 |
| 2 | Metadata key | Reads `nanobot`, else `openclaw` | Also reads `nanoclaw` first | Lets NanoClaw-specific skills say so; nanobot skills work unchanged | — |
| 3 | Dream's tools | Also `apply_patch` | No `apply_patch` | Not ported in layer 3 | — |
| 4 | Two lines in the bundled skills | `skill-creator` recommends nanobot's `grep` tool (`output_mode`, `head_limit`); `tmux` says "prefer exec background mode" | `skill-creator` suggests shell `grep -l` / `-c` / `-F` and `read_file` paging; `tmux` says "prefer plain exec" | NanoClaw has neither the `grep` tool nor background `exec` (layer 3); same approach as the templates in layer 4. Found in the second review | Phase 2 if those tools are added |

Everything else matches nanobot exactly, including the order of skills (the
order the folders are listed by the file system, unsorted, as in nanobot).

## What we left out

| nanobot feature | Where | What it does | Later phase |
|---|---|---|---|
| Plugin skills | `agent/plugins.py`, `list_skills` | Skills bundled in installable plugins ("Agent Plugin skills" group) | **Phase 2: Plugins** |
| CLI-app skill aliases | `apps/cli`, `_skill_aliases` | Old skill names mapped to installed CLI apps | **Phase 2** (with CLI apps) |
| Disabling skills | `disabled_skills` config, web UI toggles | Hide chosen skills | **Phase 2: configuration / Web UI** |
| ClawHub | `skills/clawhub` | Search and install community skills | **Phase 2**, after layer 9 (installing third-party code is a safety question) |
| Separate project folder | `build_skills_summary(workspace=)` | Absolute skill roots when the project folder differs from the agent's home | **Phase 2: Separate project folder** |

## Files

| File | Contents |
|---|---|
| `nanoclaw/agent/skills.py` | `SkillsLoader`, `parse_skill_metadata`, `valid_skill_metadata`, `BUILTIN_SKILLS_DIR` |
| `nanoclaw/skills/` | The 6 built-in skills and a README |
| `nanoclaw/templates/agent/skills_section.md` | The `# Skills` section text |
| `nanoclaw/agent/context.py` | `# Active Skills` and `# Skills` sections; `$skill` block on the current message |
| `nanoclaw/agent/loop.py` | Adds the `$skill` block before saving the user message |
| `nanoclaw/agent/tools/filesystem.py` | `_builtin_skill_read_path` fallback in `read_file` |
| `nanoclaw/agent/memory.py`, `templates/agent/dream.md` | Dream's skills rules, `skills/` writes, built-in reads, nanobot's boundary error text |
| `nanoclaw/utils/templates.py` | Creates `skills/` on first run |
| `nanoclaw/__main__.py` | `nanoclaw skills` |
| `pyproject.toml` | Adds `pyyaml` (nanobot parses the header with `yaml.safe_load`) |

## Tests

`tests/test_skills.py` (29 tests): header parsing and validation; discovery,
precedence and ignored folders; requirements (programs, environment variables,
three metadata forms, wrong shapes); description fallback; loading for the
prompt; always-on skills; `$skill` detection and its block; the skills summary;
the 6 bundled skills are valid; `read_file`'s built-in fallback (workspace copy
wins, no escaping); prompt sections; the `$skill` block on the message and in a
saved turn; Dream's instructions and its absolute-path read of skill-creator;
the `skills/` folder; the CLI command. Existing tests were updated for the new
`# Skills` section, the `skills/` folder and Dream's error text.

Every rule was checked by breaking it on purpose and confirming a test fails
(26 rules). One more, the `is_absolute()` check in the `read_file` fallback,
cannot fail on Linux (an absolute path never starts with `skills`); it is kept
as in nanobot.

## Review against nanobot

Checked by **running both projects**, with nanobot pointed at the same built-in
skills folder so the logic is compared rather than the skill choice:

| Check | Result |
|---|---|
| `list_skills` (filtered and not), always-on skills, summary, `load_skills_for_context` | Identical |
| `$skill` blocks for 6 messages (several skills, repeats, `US$x`, unknown, unavailable, always-on) | Identical |
| Availability and description of 6 skills (missing program/env, JSON-string metadata, wrong shapes, no header) | Identical |
| Header parsing (5 cases) and validation (4 cases) | Identical |
| **Whole system prompt with no exclusions** (memory, always-on skill, workspace and built-in skills, summary) | Identical except the two differences recorded in layer 4 (image line, emoji in `SOUL.md`). The comparisons since layer 4 no longer need to skip the `# Skills` section |
| Current message with `$skill` (content and marker) | Identical |
| `read_file` of `skills/...` paths (built-in, workspace, escape, script, missing) | Identical |
| A turn with `$skill`: request, system prompt skills sections, saved session | Identical |
| Dream: prompt; writing a new skill; refused writes; built-in and outside reads; writing `MEMORY.md` | Identical (including error texts); tool set differs only by `apply_patch` |
| Skill listing (`/skill` vs `nanoclaw skills`) | Identical |

The earlier layers' comparisons were re-run with the skills sections included:
all match (apart from a timestamp and the workspace path, which always differ
between two runs).

No new nanobot bugs were found. (The built-in skills' `skills/` root looked
like one until we found nanobot's `read_file` fallback, now ported.)

### Second review (after the build)

| Check | How | Result |
|---|---|---|
| Copied skill files | `diff -r` of each of the 6 skills against nanobot's | Only the intended renames (`nanobot` → `NanoClaw`, the tmux socket variable); script permissions kept |
| Tools the skill texts mention | Searched the 6 `SKILL.md` files for tool names | Two lines named nanobot features NanoClaw lacks (`grep` tool, background `exec`); adapted and recorded as difference #4 |
| Loader logic on random skill folders | 300 random trees (odd names, missing or broken headers, CRLF, `always` variants, metadata as mapping / JSON string / broken / wrong shapes, met and unmet requirements, workspace overrides, stray files), comparing the list, the unfiltered list, always-on skills, summary, full loading, 18 `$skill` messages, and the description and availability of every name | **0 differences** |
| skill-creator scripts | Ran `init_skill.py`, `quick_validate.py` and `package_skill.py` | Work (the validator rightly rejects the template's unfilled `[TODO: ...]` description) |
| A `$skill` turn, then `/compact` | Both projects, same built-in folder | Archive request, saved session and journal identical |
| Dream with `$weather` in a journal entry | Both projects | Same request (the skill is loaded into Dream's turn in both) |

Correction made: the two skill lines in difference #4. No code changes were needed.
