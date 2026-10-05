# 00 — Architecture map

Reference: nanobot at commit `d3d70f2` (2026-10-01). File paths below are
relative to the nanobot repo root.

## What nanobot is

nanobot is a personal AI assistant: you chat with it from a terminal, a web UI
or a chat app (Telegram, Slack, Discord, …), and it answers by calling a
language model that can use tools such as reading files, running shell commands
and fetching web pages. It remembers past conversations and can learn new
procedures from "skill" files.

### Where the code is (non-test Python, approx. lines)

| Area | Lines | Rebuild? |
|---|---|---|
| `channels/` — 20+ chat app integrations | 30k | No (maybe one later) |
| `agent/` — loop, runner, tools, memory, skills, subagents | 24k | **Yes — this is the core** |
| `webui/` — browser UI backend | 20k | No |
| `providers/` — model backends | 17k | One adapter |
| `cli/` — commands, onboarding wizard | 9k | A small REPL |
| `utils/`, `session/`, `cron/`, `config/`, `bus/`, `security/`, … | ~20k | Parts of `session/`, `bus/`, `security/` |

The agent core's *ideas* fit in a few thousand lines; most of the 24k is
production hardening (retries, recovery, streaming, usage accounting, hooks).

## The core flow

```
Channel ──InboundMessage──▶ MessageBus ──▶ AgentLoop ──▶ AgentRunner ⇄ Provider (LLM)
   ▲                                          │              ⇅
   └──────OutboundMessage── MessageBus ◀──────┘            Tools
                                              │
                         reads/writes: Session, Memory, Skills, Templates
```

(Adapted from `docs/architecture.md`.)

One turn, step by step:

1. A **channel** receives a user message and publishes an `InboundMessage` to
   the **message bus** (`bus/events.py`, `bus/queue.py` — two `asyncio.Queue`s,
   inbound and outbound).
2. The **AgentLoop** (`agent/loop.py`) picks the message up, works out its
   **session key** (e.g. `telegram:12345`), loads that session's history and
   asks the **ContextBuilder** (`agent/context.py`) to assemble the system prompt
   plus recent messages.
3. It hands that to the **AgentRunner** (`agent/runner.py`), which talks to the
   model.
4. The runner calls the **provider** (`providers/base.py` `LLMProvider.chat`).
   The model replies with text, tool calls, or both.
5. If there are tool calls, the runner executes them through the
   **ToolRegistry** (`agent/tools/registry.py`), appends the results as `tool`
   messages and calls the model again. This repeats until the model answers
   without tool calls or `max_iterations` is reached
   (`runner.py`: `for iteration in range(spec.max_iterations)`).
6. Back in the loop, the new messages are saved to the **session**
   (`session/manager.py`, JSONL files) and the reply is published as an
   `OutboundMessage`, which the channel sends to the user.

### Loop vs runner — the key split

| | `AgentLoop` (channel-facing) | `AgentRunner` (model-facing) |
|---|---|---|
| Owns | which session, which workspace, building context, saving history, publishing replies | calling the provider, running tools, feeding results back, stop conditions |
| Input | `InboundMessage` | `AgentRunSpec` (messages, tools, model, limits) |
| Output | `OutboundMessage` | `AgentRunResult` (final text, all messages, tools used, stop reason) |

nanobot's own guidance: problems with routing/sessions/delivery start in
`loop.py`; problems with model calls/tools/iteration limits start in
`runner.py`. We keep the same split.

## State on disk

| Store | nanobot location | Purpose |
|---|---|---|
| Config | `~/.nanobot/config.json` | providers, model, tools, channels |
| Workspace | `~/.nanobot/workspace/` | the agent's home: identity, memory, skills |
| Identity | `<workspace>/SOUL.md`, `USER.md` | who the agent is, who the user is — injected into the prompt |
| Project instructions | `<project>/AGENTS.md` | per-project rules |
| Sessions | `~/.nanobot/sessions/<workspace-id>/*.jsonl` | recent conversation, replayed each turn |
| Long-term memory | `<workspace>/memory/MEMORY.md`, `history.jsonl` | facts that outlive the session; "Dream" periodically consolidates history into memory |
| Skills | `<workspace>/skills/`, built-in `nanobot/skills/` | Markdown procedures loaded on demand |

## Design rules worth copying (from `.agent/design.md`)

- **Core stays small; extend at the edges.** New capabilities go into tools,
  channels, skills or MCP servers, not into the loop/runner.
- **Prefer duplication over premature abstraction** for channels and providers.
- **Type dynamic boundaries at the edge.** Parse raw API payloads into typed
  objects once; don't pass raw dicts through the core.
- **Explicit over magical.** Config in one schema; provider selection traceable.

## NanoClaw plan

We rebuild the right-hand side of the diagram first and replace channels + bus
with a direct call (`process_direct` in nanobot does the same for the CLI).

Layers 0–7 are built; current status is in [`../ROADMAP.md`](../ROADMAP.md).

| Layer | nanobot reference | NanoClaw module |
|---|---|---|
| 1 Provider | `providers/base.py`, `openai_compat_provider.py` | `nanoclaw/providers/` |
| 2 Runner | `agent/runner.py` | `nanoclaw/agent/runner.py` |
| 3 Tools | `agent/tools/base.py`, `registry.py`, `filesystem.py`, `shell.py`, `web.py` | `nanoclaw/agent/tools/` |
| 4 Context | `agent/context.py`, `templates/` | `nanoclaw/agent/context.py` |
| 5 Sessions | `session/manager.py`, `agent/loop.py` | `nanoclaw/session/`, `nanoclaw/agent/loop.py` |
| 6 Memory | `agent/memory.py`, `autocompact.py` | `nanoclaw/agent/memory.py` |
| 7 Skills | `agent/skills.py` | `nanoclaw/agent/skills.py` |
| 8 Subagents | `agent/subagent.py`, `tools/spawn.py` | `nanoclaw/agent/subagent.py` |
| 9 Safety | `security/`, `tools/sandbox.py` | `nanoclaw/security/` |
| 10 CLI | `cli/` | `nanoclaw/cli.py` |

## Deferred (review after the core works)

- Message bus + multi-channel routing (`bus/`, `channels/`)
- Hooks and plugins (`agent/hook.py`, `agent/plugins.py`)
- Recovery, checkpoints, injections (`session/recovery.py`, runner checkpoint/injection callbacks)
- Context governance, usage accounting (`agent/context_governance.py`, `llm_usage/`)
- Streaming, provider retries, model presets
- MCP, cron, web UI, image/audio
