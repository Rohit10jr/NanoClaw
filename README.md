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
| 1 | Provider interface (OpenAI-compatible) | planned |
| 2 | Agent loop (runner) | planned |
| 3 | Tools (files, shell, web) | planned |
| 4 | Context / system prompt | planned |
| 5 | Sessions | planned |
| 6 | Memory + compaction | planned |
| 7 | Skills | planned |
| 8 | Subagents | planned |
| 9 | Safety (workspace limits, shell guards) | planned |
| 10 | Terminal chat (CLI) | planned |

Each layer has a note in [`docs/notes/`](docs/notes/) explaining the concept,
how nanobot does it, what we built and what we left out.

## Development

```bash
uv sync
uv run pytest
uv run nanoclaw --version
```
