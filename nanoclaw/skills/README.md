# NanoClaw built-in skills

Skills bundled with NanoClaw, copied from nanobot's `nanobot/skills/` (which
adapted them from [OpenClaw](https://github.com/openclaw/openclaw)'s skill
system).

## Skill format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (`name`, `description`, optional `metadata`)
- Markdown instructions for the agent

Two lines are adapted to NanoClaw's tools: `skill-creator` suggests shell
`grep` instead of nanobot's `grep` tool, and `tmux` drops "exec background
mode" (NanoClaw's `exec` has none).

The `metadata` field keeps nanobot's `{"nanobot": {...}}` form (for example
`"requires": {"bins": ["curl"]}`), so skills work in both projects.

A skill in the workspace's `skills/` folder with the same name replaces the
built-in one.

## Available skills

| Skill | Description |
|-------|-------------|
| `github` | Interact with GitHub using the `gh` CLI |
| `memory` | Search past conversations in the history log |
| `skill-creator` | Create new skills |
| `summarize` | Summarize URLs, files, and YouTube videos (needs the `summarize` CLI) |
| `tmux` | Remote-control tmux sessions |
| `weather` | Get weather info using wttr.in and Open-Meteo |

nanobot also bundles `cron`, `image-generation`, `my`, `update-setup` and
`clawhub`; they need features NanoClaw doesn't have (see `docs/notes/07-skills.md`).
