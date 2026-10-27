# NanoClaw

A minimal personal AI agent, rebuilt layer by layer to understand how
[nanobot](https://github.com/HKUDS/nanobot) works.

nanobot is a framework plus agent of about 130k lines, most of it chat-app
integrations, a web UI and many model providers. NanoClaw rebuilds only the
**agent core**: the loop, provider interface, tools, context, sessions, memory,
skills, subagents, safety guards and a terminal chat. It keeps nanobot's
concepts and class names, so the two codebases can be read side by side, and
leaves the production features (multi-channel routing, hooks, plugins,
recovery, governance) for later.

## Status

| # | Layer | Status |
|---|---|---|
| 0 | Architecture recon + project skeleton | done |
| 1 | Provider interface (OpenAI-compatible) | done |
| 2 | Agent loop (runner) | done |
| 3 | Tools (files, shell, web) | done |
| 4 | Context / system prompt | done |
| 5 | Sessions + agent loop | done |
| 6 | Memory + compaction | done |
| 7 | Skills | done |
| 8 | Subagents | done |
| 9 | Safety (workspace limits, shell guards) | done |
| 10 | Terminal chat (CLI) | done |
| 11 | Catch up with upstream nanobot (`c4dac64`) | done |
| 12 | Foundations: config file, logging, retries, token usage | done |
| 13 | Hooks, events, streaming replies | done |
| 14 | Message bus, session workers, `/stop`, injections | done |
| 15 | Crash recovery, compaction before every model request | done |
| 16 | Shell: background exec sessions, Windows PowerShell | done |
| 17 | More tools: `apply_patch`, `rg`, unchanged-file reads, MCP servers | done |

Phase 1 (the core agent) is complete. Phase 2 builds the production backend
(layers 11–25; progress in [`docs/PHASE2_PROGRESS.md`](docs/PHASE2_PROGRESS.md)).

- [`docs/ROADMAP.md`](docs/ROADMAP.md): every layer in build order, plus the
  production features we'll review afterwards.
- [`docs/CODE_MAP.md`](docs/CODE_MAP.md): everything built so far, mapped
  symbol by symbol to the nanobot code it comes from.
- [`docs/notes/`](docs/notes/): one note per layer explaining the concept, how
  nanobot does it, what we built and what we left out.
- [`docs/UPSTREAM_BUGS.md`](docs/UPSTREAM_BUGS.md): bugs found in nanobot while
  porting it, with reproductions and suggested fixes, for issues/PRs upstream.

## Development

```bash
uv sync
uv run pytest
uv run nanoclaw --version
```

## Working on a project folder

By default the agent works in its workspace (`~/.nanoclaw/workspace`), which
also holds its own files. `nanoclaw chat --project DIR` / `nanoclaw ask
--project DIR "..."` (or `NANOCLAW_PROJECT=DIR`) point it at a code folder
instead, as nanobot's WebUI project picker does: relative paths in the file
tools, `exec`'s and `rg`'s default directory and `AGENTS.md` come from the
project, while memory, skills, `SOUL.md` and `USER.md` stay in the workspace
(Dream only ever edits the workspace). With `--restrict` the tools are kept
inside the project, though they can still read the agent's skills and
`memory/history.jsonl`. `nanoclaw prompt --project DIR` shows the system
prompt with its `# Current Project` section. The choice lasts for one run; it
is not saved with the session.

## Configuration

NanoClaw talks to any OpenAI-compatible API (OpenAI, OpenRouter, DeepSeek,
Groq, Ollama, vLLM, ...):

```bash
export NANOCLAW_API_KEY=sk-...
export NANOCLAW_MODEL=gpt-4o-mini                  # default
export NANOCLAW_API_BASE=https://openrouter.ai/api/v1   # omit for OpenAI
uv run nanoclaw chat                      # chat in the terminal (type exit or Ctrl+C to quit)
uv run nanoclaw ask "My name is Rohit"
uv run nanoclaw ask "What's my name?"     # the conversation continues
uv run nanoclaw ask --new "hello"         # forget it and start over
uv run nanoclaw prompt        # show the system prompt (no key needed)
```

Or put the settings in a config file, `~/.nanoclaw/config.json`, with
nanobot's field names (a nanobot `config.json` loads too; sections NanoClaw
doesn't use are ignored). Environment variables override the file, and flags
override both:

```bash
uv run nanoclaw config init      # write a default config.json (or add missing settings)
uv run nanoclaw config show      # the settings a run would use, API keys masked
uv run nanoclaw config path      # where the file is (-c FILE or NANOCLAW_CONFIG to use another)
```

```json
{
  "agents": { "defaults": { "model": "openai/gpt-4o-mini", "maxToolIterations": 200 } },
  "providers": { "openrouter": { "apiKey": "${OPENROUTER_API_KEY}" } }
}
```

Failed model requests are retried (rate limits, timeouts, server errors; 1, 2,
4 seconds or the server's Retry-After); quota and authentication errors stop
at once. `"providerRetryMode": "persistent"` keeps retrying. Add `--logs` to
`chat`, `ask`, `compact` or `dream` to see what the agent does internally.

Conversations are saved in `~/.nanoclaw/sessions/`, outside the workspace, so
the agent's file tools can't edit them. If the process is killed in the middle
of a turn, the next turn starts with the interrupted turn restored (finished
tool results kept, unfinished calls marked as interrupted). Use `--session NAME` to keep several
separate conversations (default `cli:direct`).

Replies stream word by word, with the model's reasoning shown as dim `✻` lines
(as in nanobot; turn lines off with `channels.sendProgress`, `sendToolHints`,
`showReasoning` in the config). `nanoclaw ask` keeps stdout for the reply only.

In the chat, slash commands: `/new` (start over), `/compact`, `/stop` (cancel
the running turn and background subagents; finished work is kept), `/status`,
`/history [n]`, `/dream`, `/skill`, `/help`. While the agent works you see what
it does (`↳ $ ls`, `↳ read notes.txt`), and you can keep typing: a new message
is folded into the running turn, as in nanobot.

Memory, as in nanobot:

```bash
uv run nanoclaw compact     # summarize the conversation so far (also automatic, before any model request that would be too long)
uv run nanoclaw dream       # fold the summaries into long-term memory (memory/MEMORY.md, USER.md, SOUL.md)
export NANOCLAW_CONTEXT_WINDOW=128000   # your model's context window in tokens (default 128000)
export NANOCLAW_COMPACTION=nanobot      # or "fixed": NanoClaw's fix for upstream bug 4 (A/B testing it:
                                        # uv run python scripts/compare_compaction.py --dry-run)
```

Skills, as in nanobot: Markdown procedures in `skills/<name>/SKILL.md`
(your workspace) or built in (`github`, `memory`, `skill-creator`,
`summarize`, `tmux`, `weather`). The agent sees a one-line list and reads a
skill when it needs it; `$name` in a message loads one for that turn.

```bash
uv run nanoclaw skills                                   # list skills
uv run nanoclaw ask 'What is the $weather in Paris?'     # use a skill now
```

Summaries are kept in a journal, `memory/history.jsonl` in the workspace;
`nanoclaw dream` reads the new entries and updates the memory files, which the
agent sees in every system prompt.

Subagents, as in nanobot: the agent can hand a task to a child agent with the
`spawn` tool. With `wait=true` it waits for the answer; otherwise the child
works in the background (up to 4 at once) and the agent tells you the result
when it's done. `nanoclaw ask` waits for background children before exiting.

Safety, as in nanobot: `web_fetch` and shell commands can't reach private or
internal network addresses (`localhost`, `192.168.x.x`, cloud metadata, ...),
and a short deny list blocks commands like `rm -rf`. With `--restrict` the file
tools and shell commands also stay inside the workspace:

```bash
uv run nanoclaw ask --restrict "clean up the notes"   # or NANOCLAW_RESTRICT_TO_WORKSPACE=1
export NANOCLAW_SSRF_WHITELIST=100.64.0.0/10          # private ranges you trust (e.g. Tailscale)
```

These are guards, not a sandbox: run the agent somewhere you don't mind it
working, or, on Linux, set `"tools": {"exec": {"sandbox": "bwrap"}}` in the
config to run every shell command inside bubblewrap, where only the workspace
is writable and nothing outside it but system folders is visible (this also
turns on the workspace restriction).

Shell commands, as in nanobot: a command still running after a second keeps
running in the background (up to an hour by default) and the agent checks on
it, sends it input or stops it. `/new`, `/stop` and leaving the chat stop the
commands it started. On Windows commands run in PowerShell (or `cmd`). Limits
and allowed commands go in the config file under `tools.exec` (`maxRuntime`,
`idleTimeout`, `allowPatterns`, `denyPatterns`, `enable`, ...).

MCP servers, as in nanobot, go under `tools.mcpServers`: a `command` (stdio)
or a `url` (streamable HTTP; SSE for URLs ending in `/sse`). Their tools appear
as `mcp_<server>_<tool>`; `enabledTools` limits which, `toolTimeout` (30 s)
bounds each call. Private-network URLs are blocked unless listed in
`tools.ssrfWhitelist`. OAuth servers and `useProxy` are not supported.

```json
{ "tools": { "mcpServers": {
  "fs": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] },
  "remote": { "url": "https://mcp.example.com/mcp", "headers": { "Authorization": "Bearer ${MCP_TOKEN}" } }
} } }
```

The agent's personality and instructions live in editable Markdown files in
the workspace (`~/.nanoclaw/workspace` by default, change with `-w`):
`SOUL.md` (personality), `USER.md` (about you), `AGENTS.md` (workspace rules).
They are created on first run and never overwritten.
