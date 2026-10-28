# 25.2 — Agent Plugins v1

Reference: nanobot at commit `c4dac64` (`nanobot/agent/plugins.py`,
`tests/agent/test_agent_plugins.py`). Paths are relative to its `nanobot/`
package.

## The concept

An **Agent Plugin** is a folder you drop into `<workspace>/plugins/<name>/`
that brings skills and MCP servers together, in the
[Agent Plugins v1](https://agent-plugins.org) layout:

```
<workspace>/plugins/demo/
├── plugin.json            {"$schema": ".../1.0.0/plugin.schema.json", "name": "demo", ...}
├── skills/<skill>/SKILL.md  ordinary skills (name must match the folder)
└── mcp.json               {"$schema": ".../1.0.0/mcp.schema.json", "mcpServers": {...}}  stdio only
```

Nothing in it is used until you enable it:

```bash
nanoclaw plugins list                 # what is installed, enabled or not, its skills and servers
nanoclaw plugins enable demo          # shows fingerprint, permissions, skills, server commands; asks y/N
nanoclaw plugins enable demo --yes    # no question (scripts)
nanoclaw plugins disable demo
```

Once enabled, its skills appear in the prompt under `### Agent Plugin skills
(`plugins`)` (between workspace and built-in skills; a workspace skill of the
same name wins, a plugin skill beats a built-in one), `read_file` may read
the skill folders even when the tools are restricted to a project, and its
MCP servers start with the next `chat`, `ask` or `gateway`, named `<plugin>`
(one server) or `<plugin>--<server>` (several). A server in your own
`tools.mcpServers` with the same name wins.

## Trust model

Enabling a plugin is a decision to trust that exact package: its skills
become instructions to the agent and its MCP servers run as local programs
with your permissions (any command on `PATH`, or a file inside the package).
`enable` therefore shows what you are approving and records it as a
**fingerprint**: a SHA-256 over every path in the package, every symlink's
target text and every file's bytes. The approval is a marker file
`<config dir>/plugin-data/<sha256(workspace)[:12]>/<name>/enabled` (mode 0600,
directories 0700) holding `{"fingerprint", "root"}`; it lives outside the
workspace, so the agent cannot write it while its tools are restricted
(`--restrict`; unrestricted tools and `exec` can write anywhere you can, this
marker included), and a symlinked `plugin-data` that leads outside the config
folder is refused. Every use
re-computes the fingerprint: if any byte, file or link in the package
changed, or the package moved, the marker is deleted and the plugin is off
until you enable it again (the CLI also refuses if the package changed
between the review it printed and the write). Skill text is taken from the
same bytes that were hashed, so a skill swapped during a check is never
shown to the model. Paths are resolved and must stay inside the package
(`plugins/` inside the workspace, skills inside the package, `./command`
and `cwd` inside the package or its data folder), so a symlink pointing
outside is ignored; two packages claiming the same name are both ignored.
What the fingerprint cannot cover: programs the servers start from `PATH`
(e.g. `python`), files outside the package that a symlink points to (the
link text is hashed, not the target), and a change made after a check and
before a server process starts. Those are part of what you trust when you
enable.

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `nanoclaw/agent/plugins.py` | Discovery, manifest validation, fingerprints, markers, skill documents, `read_file` skill dirs (cached, re-verified per package), MCP server translation | `agent/plugins.py` (ported as is) |
| `SkillsLoader._discover_skills`, `build_skills_summary` | Order workspace → plugin → built-in; "Agent Plugin skills" group | `agent/skills.py:123`, `:281` |
| `_FsTool._resolve_read` | Enabled plugin skill folders join the restricted read roots | `agent/tools/filesystem.py:145` |
| `nanoclaw.agent.tools.mcp.configured_servers` | `tools.mcpServers` + plugin servers, for gateway, chat and ask | `agent/tools/mcp.py:1409` |
| `nanoclaw/cli/plugins.py` | `nanoclaw plugins list/enable/disable` | web UI Apps page (`webui/mcp_presets_api.py`, `plugin-<name>` rows) |

## Differences from nanobot

1. **Enabling from the CLI.** nanobot enables plugins only from its web UI
   Apps page, through the MCP presets API (`/api/settings/mcp-presets`),
   which NanoClaw does not have; porting it would mean porting that whole
   API, so the web UI endpoints are left out. The CLI shows the review and
   asks for confirmation.
2. **No hot reload.** nanobot's web UI reloads MCP connections after a
   toggle; NanoClaw resolves plugin servers when `chat`, `ask` or `gateway`
   starts. Skills change from the next turn.
3. **Skill text without the snapshot.** Until the skills snapshot (task 25.1)
   is merged, every skills lookup re-verifies plugin packages, so a package
   changed in the middle of building a prompt drops its skill from that
   prompt instead of keeping the approved text; the replacement is never
   shown either way.
