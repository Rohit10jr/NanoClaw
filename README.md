# NanoClaw

A minimal personal AI agent, rebuilt layer by layer to understand how
[nanobot](https://github.com/HKUDS/nanobot) works.

nanobot is a framework plus agent of about 130k lines, most of it chat-app
integrations, a web UI and many model providers. NanoClaw rebuilds only the
**agent core**: the loop, provider interface, tools, context, sessions, memory,
skills and subagents. It keeps nanobot's concepts and class names, so the two
codebases can be read side by side, and leaves the production features
(multi-channel routing, hooks, plugins, recovery, governance) for later.

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
| 7 | Skills | planned |
| 8 | Subagents | planned |
| 9 | Safety (workspace limits, shell guards) | planned |
| 10 | Terminal chat (CLI) | planned |

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

## Configuration

NanoClaw talks to any OpenAI-compatible API (OpenAI, OpenRouter, DeepSeek,
Groq, Ollama, vLLM, ...):

```bash
export NANOCLAW_API_KEY=sk-...
export NANOCLAW_MODEL=gpt-4o-mini                  # default
export NANOCLAW_API_BASE=https://openrouter.ai/api/v1   # omit for OpenAI
uv run nanoclaw ask "My name is Rohit"
uv run nanoclaw ask "What's my name?"     # the conversation continues
uv run nanoclaw ask --new "hello"         # forget it and start over
uv run nanoclaw prompt        # show the system prompt (no key needed)
```

Conversations are saved in `~/.nanoclaw/sessions/`, outside the workspace, so
the agent's file tools can't edit them. Use `--session NAME` to keep several
separate conversations (default `cli:direct`).

Memory, as in nanobot:

```bash
uv run nanoclaw compact     # summarize the conversation so far (also automatic when it gets too long)
uv run nanoclaw dream       # fold the summaries into long-term memory (memory/MEMORY.md, USER.md, SOUL.md)
export NANOCLAW_CONTEXT_WINDOW=128000   # your model's context window in tokens (default 128000)
```

Summaries are kept in a journal, `memory/history.jsonl` in the workspace;
`nanoclaw dream` reads the new entries and updates the memory files, which the
agent sees in every system prompt.

The agent's personality and instructions live in editable Markdown files in
the workspace (`~/.nanoclaw/workspace` by default, change with `-w`):
`SOUL.md` (personality), `USER.md` (about you), `AGENTS.md` (workspace rules).
They are created on first run and never overwritten.
