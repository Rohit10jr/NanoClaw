# 22 — Session history for the web UI

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

nanobot's web UI shows more than the conversation the model sees: reasoning,
tool calls as they start and end, compaction notices, how each turn ended and
what it used. It receives these live as **wire events** over a websocket, and
when a chat is reopened it reads them back from a **per-chat display history**
(`GET /api/sessions/<key>/webui-thread`). The agent's own session file can't
serve that: it holds what the model needs, not what the screen showed.

This layer builds both halves without a server: turning NanoClaw's bus events
into nanobot's exact wire JSON, and storing and replaying the persisted part
per chat. Layer 23 adds the server, the websocket channel and the frontend.

```
bus (OutboundMessage, stream/progress/turn events)
      └─ WebUIOutboundProjector.project ──▶ WireFrame(payload → websocket, layer 23;
                                                       record → WebUITranscriptRecorder ──▶ ~/.nanoclaw/webui/<ws>/<chat>.jsonl)
GET webui-thread ── build_webui_thread_payload ◀── display history (or backfill from the session JSONL)
GET /api/sessions ── webui_sessions_payload ◀── sessions + display history
```

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| Encoders (`webui/outbound_wire.py`) | `message` (with `kind`, `tool_events` scrubbed of inline binary), `delta`, `stream_end`, `reasoning_delta`/`reasoning_end`, `file_edit`, `turn_end` (latency, usage, round usages, outcome), `retry_status`, `context_compaction`, `recovery_state`, `goal_status`, `session_updated`, `turn_model_updated`, `error` | `webui/outbound_wire.py` |
| `WebUIOutboundProjector` (`webui/outbound_projection.py`) | Per chat: channel settings filters, one-shot reasoning as delta + end, stream and reasoning buffers so the completed `stream_end` record carries the full text, `turn_seq` numbering within a turn, audience (chat / all / web UI). No socket | `channels/manager.py` `_send_once`, `channels/websocket/runtime.py`, `webui/outbound_projection.py` |
| Persisted events | Every `message` and `file_edit`, completed `stream_end` and `reasoning_end`, `turn_end`, `context_compaction` (not `started`), and the user's message; never deltas, retry or goal status, errors | `channels/websocket/runtime.py:1147-1200` |
| `lifecycle_outbound` | Turn lifecycle on the bus → `goal_status` running/idle, `turn_model_updated`, `turn_end` | `session/webui_turns.py` `WebuiTurnCoordinator` |
| New events | `RetryStatusEvent` from the provider retry loop and the fallback chain (waiting, recovered, cleared, exhausted); `TurnRuntimeAdmitted`; `TurnCompleted` with runtime, usage, round usages and failure kind (layer 13 difference #6 closed) | `providers/base.py`, `providers/fallback_provider.py`, `agent/loop.py:2250` |
| `WebUIHistoryStore` (`webui/history.py`) | One append-only JSONL file per chat next to the sessions: `~/.nanoclaw/webui/<workspace-id>/<chat>.jsonl`, nanobot's record format, mode 0600, no symlinks, bad lines skipped, parsed files cached | the pre-SQLite `webui/*.jsonl` files, `webui/history.py` |
| `build_webui_thread_payload` (`webui/transcript.py`) | `{schemaVersion: 3, projection: "events", events, completed_turn_ids, active_turn_id, has_pending_tool_calls, page, revision}` with paging (before/after cursors, oversized turns), prompt index, trace details, revisions for ETags | `webui/transcript.py` |
| Backfill | A chat with no display history is projected from its session file (text, reasoning, tool calls, file edits; hidden and compaction rows skipped) | `transcript.py` `build_session_thread_response` |
| `webui_sessions_payload` (`webui/session_list_index.py`) | `/api/sessions` rows for `websocket:` chats, newest first, title and preview; `delete_webui_session` removes a chat, its subagent sessions and their histories; generated titles after the first turn (`session/webui_turns.py`) | `webui/session_list_index.py` |

## How it was built

Two tasks in parallel: the wire projection (22.1) and the display history
(22.2). At integration, one file both created; 22.2's private copies of
`project_tool_events` and the metadata keys were replaced by 22.1's; a test
runs a real websocket turn, stores its projected records and reads them back
through `webui-thread` and the session list.

| Task | Checked against nanobot |
|---|---|
| 22.1 Wire projection | 46 scripted sequences through nanobot's real channel dispatcher, websocket channel, projector and turn coordinator: 149 frames and 66 persisted records identical; 13 retry cases through both retry loops (36 retry events) identical; re-run after integration |
| 22.2 Display history | 36 cases, 48 checks (full `webui-thread` payloads incl. paging, prompt index and trace details; backfill; session list rows): identical; re-run after integration |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | Display history storage | SQLite tables | One JSONL file per chat in nanobot's older record format | Decision 2 |
| 2 | A chat without display history | 404 (history imported into SQLite at startup) | Projected from its session file on read | Terminal chats and older chats still show their past |
| 3 | Sidebar title, preview, time | Stored when saved | Computed on read (cached) | No SQLite |
| 4 | Session generations | Checked on append | None | JSONL sessions |
| 5 | Where the first `retry_status: cleared` comes from | The provider | The runner, only for web UI turns | NanoClaw's provider has no request context |
| 6 | `ResponseSourceEvent` | Emitted per call | Not emitted (passed through when present) | Same reason |
| 7 | Error reply in a web UI chat | None; the failed `turn_end` shows it | Same (changed in this layer; the terminal keeps its error reply) | — |

## Tests

Layer 22 adds 278 tests (3,219 → 3,497): `tests/webui/` (nanobot's transcript,
prompt index, session identity and session list tests, the history store),
`test_webui_outbound_projection.py` (nanobot's wire tests, the frontend's
fixture file, real websocket turns, the end-to-end replay),
`test_turn_delivery.py`, `test_layer22_gaps.py`, and additions to the retry and
fallback tests.

### Mutation checks

444 rules were broken on purpose:

| Area | Rules | Caught at once | Gaps closed with tests | Equivalent | Result |
|---|---|---|---|---|---|
| Wire encoders, projector | 145 | 109 | 34 | 2 | 143 caught |
| Retry status, turn lifecycle, usage | 74 | 68 | 6 | 0 | 74 caught |
| History store, thread projection, backfill | 144 | 73 | 66 | 5 | 139 caught |
| Session list, titles, deletion | 81 | 41 | 39 | 1 | 80 caught |

75 new tests (`test_layer22_gaps.py`) close the gaps, for example: base64
scrubbing in tool events (case, leading spaces, the marker window); the
projector forgetting a chat's buffers; a fallback that fails without retries
not sending a second "cleared"; usage of the summary requests counted per
round; every history validation rule, symlink refusal (also a symlink swapped
in after the check), the 8 MiB limit and the read cache; page limits and
cursors; the first fork marker as boundary; title generation from the first
visible exchange and never over a title the user edited.

Equivalent (8): an empty turn id that every caller already treats as missing;
outcomes NanoClaw never produces; cache entries that can never be reused; and
values rejected earlier or records that never carry the field.

