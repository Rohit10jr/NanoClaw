# 23 — Web UI backend

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

nanobot's web UI is a React app that talks to the gateway: it fetches
`/webui/bootstrap` for tokens, opens a websocket for chat (send messages,
receive streamed replies, tool activity, turn ends), and calls a few REST
endpoints for the chat list, a chat's history and the slash commands. This
layer gives `nanoclaw gateway` that backend, so **nanobot's own UI, built
unchanged, works against NanoClaw**. Layer 22 already turns NanoClaw's events
into the UI's wire format and keeps each chat's display history; this layer
adds the server, the websocket channel and the endpoints.

```
browser (nanobot web UI) ── GET / , /assets/* ─────────────▶ static files (channels.websocket.webuiDist)
                         ── GET /webui/bootstrap ───────────▶ tokens, ws path, capabilities
                         ── WS ?token= ──▶ WebSocketChannel ──InboundMessage──▶ bus ──▶ AgentLoop
                                              ◀── WebUIOutboundProjector ◀── bus outbound/events
                                              └─ WebUITranscriptRecorder ──▶ per-chat display history
                         ── GET /api/sessions, …/webui-thread, /api/commands, … ──▶ rest_api
```

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `ChannelManager`, `BaseChannel` | Starts and stops channels with the gateway, dispatches outbound bus messages to them (retries: `channels.sendMaxRetries`), reports their state in `/health` | `channels/manager.py`, `channels/base.py` |
| `WebSocketChannel` | aiohttp on one port (default `127.0.0.1:8765`): websocket, bootstrap, REST, static files. Inbound envelopes: `message` (validation, `message_accepted`, dedupe of `(chat_id, turn_id)` for 10 minutes, `user_message` to other subscribers), `new_chat`, `attach`, `webui_request` (`session.delete`, `sidebar.update`). Outbound: layer 22's projector, frames sent per audience, persisted records written to the display history; an active-turn registry; title generation after a turn | `channels/websocket/runtime.py`, `webui/inbound_commands.py`, `session/webui_turns.py` |
| Bootstrap and tokens | `/webui/bootstrap`: `ws_path`, a one-time websocket token, an API token with a TTL, limits, capabilities. Loopback without a secret by default; `tokenIssueSecret` (sent as `X-Nanobot-Auth`) for remote access; `0.0.0.0` requires a secret | `webui/gateway_tokens.py`, `channels/websocket/ws_http.py:729` |
| Static files | The built UI at the origin root, SPA fallback, `index.html` no-cache, hashed assets immutable, `.gz` siblings; unknown `/api/*` is a 404, never the page | `ws_http.py:2001` `_serve_static` |
| REST (`webui/rest_api.py`) | `/api/sessions`, `…/webui-thread` (ETag, 304, paging), `…/prompts`, `…/trace-detail`, `…/context`, `/api/commands`, `/api/webui/sidebar-state`, read-only `/api/settings` (model, presets, providers), `/api/workspaces`, `/api/settings/pairing` (empty), `/api/webui/skills`; bearer auth on every route | `channels/websocket/ws_http.py` |

```json
{ "channels": { "websocket": { "enabled": true, "webuiDist": "/path/to/nanobot/web/dist" } } }
```

`nanoclaw gateway`, then open http://127.0.0.1:8765. NanoClaw ships no
frontend build: build nanobot's `webui/` (bun) and point `webuiDist` (or
`NANOCLAW_WEBUI_DIST`) at its `nanobot/web/dist`.

## How it was built

Phase A in parallel: the server and websocket channel (23.1), the REST
endpoints (23.2), and nanobot's UI built unchanged in the scratchpad (23.3;
not in the repository). Reading the UI's source corrected two research
assumptions: the browser never checks `webui.core.v1`, and it calls
`/api/settings/pairing` and `/api/webui/skills` on load. Phase B (23.4): the
real UI in Chromium (Playwright) against `nanoclaw gateway`.

Found and fixed with the real UI: the composer refused to send ("Choose
model") because `/api/settings` listed no providers; titles weren't generated
when the provider had no model name; the chat details popover couldn't load a
chat's automations (task 23.5).

| Task | Checked against nanobot |
|---|---|
| 23.1 Server, websocket channel | 71 observations through both real channels (envelopes, bootstrap with and without secret, loopback vs remote, static files, outbound frames and history records): 64 identical, 7 deliberate (capabilities, no session generations, no upload field) |
| 23.2 REST | 48 requests through nanobot's real HTTP handler: 41 identical, 7 explained (304 content type, NanoClaw's token estimate, commands NanoClaw lacks, fixed workspace, workspace path); provider rows: the 20 both list identical |
| 23.4 End to end | 9 scenarios in the real UI: Markdown streaming; reasoning and tool rows live and after reload; two chats and switching; delete; `/stop` and the slash menu; a failed model call; `/status` and `/help`; reload during a streaming turn; the bootstrap secret prompt. All pass. `tests/e2e/` runs them when `NANOCLAW_E2E_WEBUI_DIST` points at a build |

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | `channels.websocket.enabled` default | On | Off | No bundled frontend |
| 2 | Attachments, media, voice transcription | Yes | "Attachments are not supported"; the UI keeps the draft | Later |
| 3 | Settings | Editable (providers, models, channels, …) | Read-only | Edit `config.json` |
| 4 | Recovery coordinator ("Continue" after a failed turn) | Yes | No; the failure notice is live only | Not ported |
| 5 | Temporary chats, forks, per-chat project picker, pairing, subagent panel, `!` shell | Yes | No | Not ported; the UI hides or degrades them |
| 6 | Capabilities advertised | Several | `webui.core.v1` | Only the core is implemented |
| 7 | Turn registry | Module globals | On the channel | Clearer ownership |

## Tests

Layer 23 adds 126 tests (3,497 → 3,623): `test_websocket_channel.py`,
`test_channel_manager.py`, `test_webui_gateway_e2e.py` (in-process gateway),
`tests/webui/test_rest_api.py`, and `tests/e2e/` (the browser scenarios,
skipped unless a UI build is given).

### Mutation checks

MUTATION_RESULTS
