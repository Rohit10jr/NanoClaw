# 13 — Hooks, events and streaming

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

Until now the agent produced its reply in one piece, and the only live signal
was a progress callback before tools ran. nanobot instead lets **hooks**
watch every step of a run, turns what they see into **events**, and
**streams** the reply word by word. Every live signal a UI shows (reply
deltas, reasoning, tool starts and ends, retry waits, compaction) is one of
these events. This layer brings that machinery to NanoClaw, and the terminal
chat now streams like nanobot's classic chat.

```
provider.chat_stream_with_retry ──deltas──▶ runner ──hook calls──▶ AgentProgressHook ──events──▶ EventSink
                                                                                                 │
                                                       StreamRenderer (chat) / stdout+stderr (ask)◀┘
```

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `LLMProvider.chat_stream`, `safe_chat_stream`, `chat_stream_with_retry` | Stream a reply as content, reasoning and tool-call deltas. The default `chat_stream` calls `chat()` and sends the whole text as one delta. Retries follow layer 12's rules, but **never after text has been shown**; a stalled stream (timeout) starts a new segment (`on_stream_recover`) and retries | `providers/base.py:50, 1085, 1147, 1230, 1580-1645` |
| OpenAI-compatible streaming | `stream=True` with usage; each chunk waited for at most 90 s (`NANOCLAW_STREAM_IDLE_TIMEOUT_S`); `_parse_chunks` assembles the same response as the non-streamed path (tool-call pieces by index, duplicate IDs, legacy `function_call`, cached tokens); a stream that ends without a finish reason is a retryable connection error | `providers/openai_compat_provider.py:1556-1683, 1839-1968` |
| `AgentHook`, `CompositeHook`, hook contexts | 15 callbacks around a run (`before_run`, `before_iteration`, `on_stream`, `on_stream_end(resuming)`, `before_execute_tools`, per-tool calls, `emit_reasoning`, `after_iteration`, `finalize_content`, …). The composite calls each hook in order and isolates failures | `agent/hook.py` |
| Events | `ProgressEvent` (text, tool hints, tool start/end payloads, reasoning), `StreamDeltaEvent`, `StreamEndEvent(resuming)`, `ContextCompactionEvent(reason)`, `RetryWaitEvent`, … and the `EventSink` they go to | `events.py`, `bus/outbound_events.py` |
| `AgentProgressHook`, `build_agent_turn_hook` | Turns hook calls into events; always the first hook of a turn | `agent/progress_hook.py`, `agent/turn_hooks.py` |
| `IncrementalThinkExtractor`, `strip_think` | Pull `<think>` reasoning out of a stream, even when a tag is split across chunks | `utils/helpers.py:173, 261` |
| `StreamRenderer` | Draws the streamed reply in the terminal (live Markdown, the spinner, progress and reasoning lines) | `cli/stream.py`, `cli/terminal.py:276-337` |

## What you see

| Command | Behaviour |
|---|---|
| `nanoclaw chat` | Replies stream word by word. The model's reasoning appears as dim `✻` lines, tool hints as `↳` lines, retry waits as `↳ Model request failed, retry in 1s (attempt 1).`, compaction as `Compressing context…` / `Context compacted.`. Subagent follow-ups stream too |
| `nanoclaw ask` | stdout still holds only the reply, so it can be piped. When stdout is a terminal, the reply streams there as plain text; otherwise it is printed once at the end. Tool hints, reasoning, retry waits and compaction go to stderr |
| Settings | `channels.sendProgress`, `channels.sendToolHints`, `channels.showReasoning` (nanobot's names, default true) switch those lines off |

## How the runner uses the hooks

Every request now goes through `chat_stream_with_retry`, as in nanobot.
Deltas reach the hook only when it wants streaming. Per iteration, in
nanobot's order (`agent/runner.py:310-1244`): `before_iteration` → stream →
reasoning (`emit_reasoning` / `emit_reasoning_end`) → `on_stream_end(resuming=True)`
before tools → `before_execute_tools` → `before_execute_tool` /
`after_execute_tool` / `on_execute_tool_error` per tool → `after_iteration`.
`finalize_content` replaces the runner's own `<think>` stripping. Each tool
call also produces a tool event (`name`, `ok`/`error`, detail), as in nanobot's
`tools/execution.py`. The old `on_progress` callback is still accepted: it is
wrapped into a progress hook.

## How it was built

Phase A, three standalone pieces in parallel with nanobot's names: provider
streaming (13.1), hooks and events (13.2), the terminal renderer (13.3). Phase
B (13.4) wired them into the runner, loop, chat and ask.

| Task | Checked against nanobot |
|---|---|
| 13.1 Provider streaming | 20 chunk sequences through both `_parse_chunks`: 19 identical, 1 deliberate (no `<tool_call>` text rescue, as in the non-streamed path); the retry rule read against `base.py:1621-1645` |
| 13.2 Hooks and events | The same callback sequences through both progress hooks: byte-identical events; the think extractor identical on split tags |
| 13.3 Renderer | 14 call sequences × 4 terminal modes: 44 identical outputs, 12 different only where NanoClaw escapes brackets |
| 13.4 Wiring | Scripted turns (`<think>`, text before tools, failing tool, retries, errors; streaming and not) through both runners: identical event sequences, final answers, tool events and saved messages (leaving out two events only nanobot's web UI uses); one chat turn's terminal output identical; `nanoclaw chat` run in a real terminal |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Brackets in progress and reasoning lines | Printed as markup (`[/x]` crashes the turn; upstream bug 6, also in reasoning lines) | Escaped | Bug fix |
| 2 | `nanoclaw ask` output | `agent -m` sends everything to stdout through a renderer | stdout = the reply only; the rest on stderr | Keep `ask` pipeable (layer 10 decision) |
| 3 | Runner with no hook | A bare `AgentHook` | A non-streaming progress hook | Subagents keep `<think>` stripped |
| 4 | Subagent follow-ups in chat | Streamed into the previous turn's renderer through `patch_stdout` | Streamed while the prompt is hidden | Cleaner output |
| 5 | Stream timing (TTFT) and the call observer | Recorded | Not recorded | No usage store (layer 12) |
| 6 | `RetryStatusEvent`, `ResponseSourceEvent`, `TurnEndEvent` | Emitted for the web UI | Not emitted | Web UI backend (layer 23) |
| 7 | Live subagent status (`_SubagentHook`), file-edit activity hook | Yes | No | Layer 19 |

## Tests

Layer 13 adds 188 tests (923 → 1,111): `test_streaming_provider.py`,
`test_hooks.py`, `test_events.py`, `test_stream_renderer.py`,
`test_streaming_turn.py`, adapted from nanobot's streaming, hook, renderer and
CLI tests.

### Mutation checks

83 rules across the providers, hooks, progress hook, runner, loop, memory,
renderer, chat, `ask` and the channel settings were broken on purpose. 71 were
caught at once. Of the 12 survivors, 10 were real test gaps, closed with 8 new
tests (e.g. a recovered stream segment that showed nothing can still be
retried; an error after partly streamed text is printed and the text not
repeated; `showReasoning` off hides finished sentences). 2 are equivalent: a
shortcut in the think extractor that returns the same result, and a
`renderer.close()` that always comes after the final stream end, so it does
nothing. Result: 81 of 83 caught, 2 equivalent.
