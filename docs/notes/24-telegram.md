# 24 — Telegram and pairing

Reference: nanobot at commit `c4dac64`. Paths are relative to its `nanobot/`
package.

## The concept

A **chat app channel**: the agent runs in `nanoclaw gateway` and you talk to
it from Telegram, on your phone. Messages from Telegram become inbound bus
messages like the terminal's or the web UI's; replies go back as Telegram
messages, streamed by editing the message as the text grows. Heartbeat and
cron results (layer 21) now have somewhere to go: the latest Telegram chat.

Who may talk to the bot: the user ids in `allowFrom`, or anyone you approve
through **pairing**. An unknown person who writes to the bot in private gets a
short code; you approve it with `/pairing approve <code>` (terminal or web UI),
and from then on they're allowed.

```
Telegram ── getUpdates (polling) ──▶ TelegramChannel ── is_allowed? ──▶ InboundMessage ──▶ bus ──▶ AgentLoop
              │                            └─ unknown DM ──▶ pairing code ──▶ /pairing approve
Telegram ◀── sendMessage / editMessageText (HTML, ≤ 4096 chars) ◀── ChannelManager (filters) ◀── bus
```

## The pieces

| Piece | What it does | nanobot |
|---|---|---|
| `TelegramChannel` | python-telegram-bot polling (or webhook): ordered per-chat updates, polling watchdog, `RetryAfter`; Markdown → Telegram HTML (code, tables, links, escaping); splitting at 4096 characters; streaming as plain text then edited into HTML every `streamEditInterval` (0.6 s); typing indicator and 👀 reaction; `/start`, `/help`, forwarded slash commands; group policy (`mention` by default) with mention detection; one session per forum topic; photos and documents downloaded and passed as paths, voice as `[voice: <path>]` | `channels/telegram/runtime.py` |
| `nanoclaw channels validate telegram` | Checks the token with `getMe`, never printing it; `nanoclaw channels status` | `channels/telegram/validation.py` (web UI only in nanobot) |
| Manager filters for chat apps | For channels that don't use the web UI projector: reasoning, progress, tool hints and retry waits filtered by the channel's settings, each event sent through its own method; rejected deliveries never retried | `channels/manager.py` `_send_once` |
| Pairing store | Codes per channel and sender with expiry, approve, deny, revoke, delivery records; `~/.nanoclaw/pairing.json` | `pairing/store.py` |
| Allow check | `"*"`, then `allowFrom`, then approved pairs; an unapproved private sender gets a code and no turn runs | `channels/base.py` |
| `/pairing` | List, approve, deny, revoke; in the slash menu | `command/builtin.py` |
| Web UI pairing | `/api/settings/pairing` with the real requests; approve/deny from Settings; `pairing_changed` frames | `webui/settings_*.py` |

```json
{
  "channels": {
    "telegram": { "enabled": true, "token": "${TELEGRAM_BOT_TOKEN}", "allowFrom": ["123456789"] }
  }
}
```

Get a token from @BotFather, put your numeric user id in `allowFrom` (or leave
it empty and pair), run `nanoclaw channels validate telegram`, then
`nanoclaw gateway`, and message the bot. In groups, @mention it (privacy mode
may need turning off in BotFather for it to see every message).

## How it was built

Two tasks in parallel: the channel (24.1) and pairing (24.2); the channel used
a stand-in for pairing until 24.2 was merged, then the real module.

| Task | Checked against nanobot |
|---|---|
| 24.1 Telegram | 100 cases through both channels: Markdown→HTML (20), streaming previews (20), splitting (13), group policy and mentions (11), topic session keys (4), `allowFrom` (10), inbound messages on the bus (7), Bot API call sequences with a recording fake bot and clock (15): identical; re-run after integration. nanobot's 142 Telegram tests all ported. In-process gateway tests with a fake bot: allowed user → turn → 3 HTML chunks, streaming edits, pairing code for an unknown sender, group stranger ignored, heartbeat delivered to the Telegram chat |
| 24.2 Pairing | 29 cases (store sequences, `/pairing` replies, web UI payloads and errors): identical; re-run after integration. Flow tests: unknown sender → code → approved from the terminal and from a real web UI socket → allowed |

The web UI browser test (layer 23) was re-run after the manager changes.

## Differences from nanobot (deliberate)

| # | Difference | nanobot | NanoClaw | Why |
|---|---|---|---|---|
| 1 | `useProxy` | Shared network proxy | Ignored; `HTTPS_PROXY` applies | No shared proxy setting |
| 2 | Voice messages | Transcribed when a provider is set | Always `[voice: <path>]` (nanobot's text without a provider) | No transcription |
| 3 | Token check | Web UI channel settings | `nanoclaw channels validate telegram` | No settings editing in the web UI |
| 4 | Commands menu | Includes `/restart`, `/goal`, `/trigger` | Without them | NanoClaw lacks them |
| 5 | Product name | nanobot | nanoclaw (`/start`, pairing reply) | — |

A live test needs a bot token; until then everything runs with mocked bots.

## Tests

Layer 24 adds 290 tests (3,774 → 4,064): `test_telegram_channel.py` (nanobot's
142 test functions), `test_telegram_validation.py`, `test_telegram_gateway.py`,
`test_channel_manager_dispatch.py`, `tests/pairing/`.

### Mutation checks

MUTATION_RESULTS
