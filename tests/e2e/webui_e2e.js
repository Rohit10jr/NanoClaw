// End-to-end test of nanobot's real web UI against `nanoclaw gateway`.
//
// Usage: node webui_e2e.js BASE_URL OUT_DIR STATE_ROOT [SECRET]
//
// BASE_URL serves the UI from a gateway started by webui_gateway.py (its
// scripted RoutingProvider answers by keyword). Scenarios drive the UI with
// Playwright, save screenshots under OUT_DIR and assert on the DOM and on the
// WebSocket frames / REST responses. With SECRET, only the bootstrap-secret
// scenario runs (the gateway must be started with that tokenIssueSecret).
// Prints one JSON line per scenario ({"scenario", "ok", "detail"}) and exits
// non-zero when any scenario fails.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const [BASE, OUT, STATE_ROOT, SECRET] = process.argv.slice(2);
fs.mkdirSync(OUT, { recursive: true });
const results = [];

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

async function openPage(browser, url = BASE + '/') {
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
  const rec = { frames: [], errors: [], responses: [] };
  page.on('pageerror', (e) => rec.errors.push(e.message));
  page.on('websocket', (ws) => {
    ws.on('framesent', (f) => rec.frames.push({ dir: '>', data: parse(f.payload) }));
    ws.on('framereceived', (f) => rec.frames.push({ dir: '<', data: parse(f.payload) }));
  });
  rec.threads = [];
  page.on('response', (r) => {
    rec.responses.push({ url: r.url(), status: r.status() });
    if (r.url().includes('/webui-thread') && r.status() === 200) {
      r.json().then((body) => rec.threads.push({ url: r.url(), body })).catch(() => {});
    }
  });
  page.rec = rec;
  await page.goto(url, { waitUntil: 'networkidle' });
  return page;
}

function parse(payload) {
  try {
    return JSON.parse(String(payload));
  } catch {
    return { raw: String(payload) };
  }
}

function received(page, event, chatId) {
  return page.rec.frames
    .filter((f) => f.dir === '<' && f.data.event === event && (!chatId || f.data.chat_id === chatId))
    .map((f) => f.data);
}

async function waitFrame(page, pred, timeout = 15000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const hit = page.rec.frames.find((f) => f.dir === '<' && pred(f.data));
    if (hit) return hit.data;
    await page.waitForTimeout(100);
  }
  throw new Error('timed out waiting for a frame');
}

async function composer(page) {
  const box = page.locator('textarea').first();
  await box.waitFor({ timeout: 15000 });
  return box;
}

async function send(page, text) {
  const box = await composer(page);
  await box.fill(text);
  await box.press('Enter');
}

async function sendCommand(page, text) {
  // The slash menu takes the first Enter (it completes the command); a second
  // Enter sends it when the composer still holds the text.
  const box = await composer(page);
  await box.fill(text);
  await page.waitForTimeout(400);
  await box.press('Enter');
  await page.waitForTimeout(400);
  if ((await box.inputValue()).trim()) await box.press('Enter');
}

function sessionFileName(chatId) {
  return Buffer.from(`websocket:${chatId}`).toString('base64url');
}

async function activeChatId(page) {
  const pane = page.locator('[data-active="true"][data-testid^="workbench-pane-websocket:"]').first();
  await pane.waitFor({ timeout: 15000 });
  const id = await pane.getAttribute('data-testid');
  return id.replace('workbench-pane-websocket:', '');
}

async function mainText(page) {
  return page.locator('main').innerText();
}

async function sidebarTitles(page) {
  // Sidebar rows carry a "Topic actions for <title>" menu button.
  const labels = await page.$$eval('button[aria-label^="Topic actions for "]', (bs) =>
    bs.map((b) => b.getAttribute('aria-label').replace('Topic actions for ', '')),
  );
  return labels;
}

async function apiGet(page, urlPath) {
  return page.evaluate(async (p) => {
    const boot = await (await fetch('/webui/bootstrap', { headers: secretHeader() })).json();
    function secretHeader() {
      const s = new URLSearchParams(location.hash.replace(/^#\??/, '')).get('bootstrapSecret');
      return s ? { 'X-Nanobot-Auth': s } : {};
    }
    const r = await fetch(p, { headers: { Authorization: `Bearer ${boot.api_token}` } });
    return { status: r.status, body: await r.json().catch(() => null) };
  }, urlPath);
}

function findFiles(root, pred, acc = []) {
  if (!fs.existsSync(root)) return acc;
  for (const name of fs.readdirSync(root)) {
    const p = path.join(root, name);
    const st = fs.statSync(p);
    if (st.isDirectory()) findFiles(p, pred, acc);
    else if (pred(p)) acc.push(p);
  }
  return acc;
}

async function scenario(name, browser, fn) {
  let page;
  try {
    const detail = await fn(async (url) => (page = await openPage(browser, url)));
    results.push({ scenario: name, ok: true, detail: detail || '' });
  } catch (e) {
    if (page) await page.screenshot({ path: path.join(OUT, `${name}_FAILED.png`) }).catch(() => {});
    results.push({ scenario: name, ok: false, detail: String(e && e.stack ? e.stack : e).slice(0, 800) });
  } finally {
    if (page) {
      if (process.env.E2E_DUMP_FRAMES) {
        fs.writeFileSync(path.join(OUT, `${name}_frames.json`), JSON.stringify(page.rec, null, 1));
      }
      await page.close().catch(() => {});
    }
  }
  console.log(JSON.stringify(results[results.length - 1]));
}

const shot = (page, name) => page.screenshot({ path: path.join(OUT, name + '.png') });

async function s1_markdown(open) {
  const page = await open();
  await send(page, 'Show me markdown please');
  await page.getByText('second item').first().waitFor({ timeout: 15000 });
  const chatId = await activeChatId(page);
  await waitFrame(page, (d) => d.event === 'turn_end' && d.chat_id === chatId);
  const deltas = received(page, 'delta', chatId);
  assert(deltas.length >= 5, `expected several deltas, got ${deltas.length}`);
  const main = page.locator('main');
  assert((await main.locator('h1, h2, h3').filter({ hasText: 'Release notes' }).count()) >= 1, 'no heading');
  assert((await main.locator('pre, code').filter({ hasText: "print('hello from nanoclaw')" }).count()) >= 1, 'no code block');
  assert((await main.locator('li').filter({ hasText: 'first item' }).count()) >= 1, 'no list');
  await shot(page, 's1_markdown');
  return `${deltas.length} deltas; heading, code block and list rendered`;
}

async function s2_reasoning_tools(open) {
  const page = await open();
  await send(page, 'Please inspect the notes file');
  // While the turn runs the activity cluster is open: reasoning + tool rows.
  await page.getByText('Let me read').first().waitFor({ timeout: 20000 });
  const chatId = await activeChatId(page);
  await waitFrame(page, (d) => d.chat_id === chatId && d.event === 'message' && d.kind === 'progress');
  await page.waitForTimeout(300);
  const live = await mainText(page);
  await shot(page, 's2_reasoning_tools_live');
  const reasoningLive = await reasoningVisible(page);
  const toolLive = /notes\.txt/i.test(live);
  await page.getByText('launch code is 42').first().waitFor({ timeout: 20000 });
  await waitFrame(page, (d) => d.event === 'turn_end' && d.chat_id === chatId);
  assert(received(page, 'reasoning_delta', chatId).length > 0, 'no reasoning_delta frame');
  await page.reload({ waitUntil: 'networkidle' });
  await page.getByText('launch code is 42').first().waitFor({ timeout: 15000 });
  await page.waitForTimeout(800);
  // A completed turn folds its activity behind the answer's "Worked for" action.
  await expandCompletedActivity(page);
  const after = await mainText(page);
  await shot(page, 's2_reasoning_tools_reload');
  const reasoningAfter = await reasoningVisible(page);
  const toolAfter = /notes\.txt/i.test(after);
  assert(!/<think>/.test(live) && !/<think>/.test(after), 'raw <think> tag shown');
  assert(reasoningLive, `reasoning block not shown live:\n${live}`);
  assert(toolLive, `tool activity not shown live:\n${live}`);
  assert(reasoningAfter, `reasoning block not shown after reload:\n${after}`);
  assert(toolAfter, `tool activity not shown after reload:\n${after}`);
  return 'reasoning + tool rows shown live and after reload';
}

async function expandCompletedActivity(page) {
  // Each answer block's "..." menu holds a "Worked for ..." toggle for the
  // activity folded above it; open every one.
  const triggers = page.locator('main [data-message-block-menu-trigger]');
  const n = await triggers.count();
  let opened = 0;
  for (let i = 0; i < n; i++) {
    await triggers.nth(i).locator("xpath=ancestor::*[@data-thread-display-unit][1]").hover().catch(() => {});
    await triggers.nth(i).click({ force: true }).catch(() => {});
    await page.waitForTimeout(200);
    const item = page.locator('[data-message-block-menu]').getByText(/^Worked/).first();
    if (await item.count()) {
      await item.click();
      opened += 1;
      await page.waitForTimeout(300);
    } else {
      await page.keyboard.press('Escape');
    }
  }
  assert(opened > 0, 'no "Worked for" activity action on the completed turn');
}

async function reasoningVisible(page) {
  // Live, reasoning sits in a collapsed "Worked"/"Thinking" row; open it.
  const toggles = page.locator('main button[aria-expanded="false"]', { hasText: /Worked|Thought|Thinking|Reason/i });
  const n = await toggles.count();
  for (let i = 0; i < n; i++) await toggles.nth(i).click().catch(() => {});
  await page.waitForTimeout(300);
  const hits = page.getByText('I should read notes.txt first');
  const count = await hits.count();
  for (let i = 0; i < count; i++) if (await hits.nth(i).isVisible()) return true;
  return false;
}

async function s3_two_chats(open) {
  const page = await open();
  await send(page, 'Alpha question here');
  await page.getByText('Echo: Alpha question here').first().waitFor({ timeout: 15000 });
  const chatA = await activeChatId(page);
  await waitFrame(page, (d) => d.event === 'session_updated' && d.chat_id === chatA && d.scope === 'metadata' && received(page, 'turn_end', chatA).length > 0);
  await page.getByRole('button', { name: 'New topic' }).click();
  await page.waitForTimeout(500);
  await send(page, 'Bravo question here');
  await page.getByText('Echo: Bravo question here').first().waitFor({ timeout: 15000 });
  const chatB = await activeChatId(page);
  assert(chatA !== chatB, 'new topic did not create a second chat');
  await page.getByText('Bravo question here topic').first().waitFor({ timeout: 10000 });
  await page.getByText('Alpha question here topic').first().waitFor({ timeout: 10000 });
  const titles = await sidebarTitles(page);
  const ia = titles.indexOf('Alpha question here topic');
  const ib = titles.indexOf('Bravo question here topic');
  assert(ia >= 0 && ib >= 0 && ib < ia, `sidebar order wrong: ${JSON.stringify(titles)}`);
  await shot(page, 's3_two_chats');
  // Switch to A: its history loads, B's does not show.
  await page.locator('aside, nav').getByText('Alpha question here topic').first().click();
  await page.getByText('Echo: Alpha question here').first().waitFor({ timeout: 10000 });
  assert((await activeChatId(page)) === chatA, 'switching did not activate chat A');
  assert(!(await mainText(page)).includes('Echo: Bravo'), 'chat A shows chat B history');
  await shot(page, 's3_switch_alpha');
  await page.locator('aside, nav').getByText('Bravo question here topic').first().click();
  await page.getByText('Echo: Bravo question here').first().waitFor({ timeout: 10000 });
  assert(!(await mainText(page)).includes('Echo: Alpha'), 'chat B shows chat A history');
  await shot(page, 's3_switch_bravo');
  return `sidebar: ${JSON.stringify(titles)}`;
}

async function s4_delete(open) {
  const page = await open();
  await page.getByRole('button', { name: 'New topic' }).click();
  await page.waitForTimeout(300);
  await send(page, 'Charlie doomed chat');
  await page.getByText('Echo: Charlie doomed chat').first().waitFor({ timeout: 15000 });
  const chatId = await activeChatId(page);
  await page.getByText('Charlie doomed chat topic').first().waitFor({ timeout: 10000 });
  const sessionFiles = () => findFiles(STATE_ROOT, (p) => p.includes(chatId) || p.includes(sessionFileName(chatId)));
  assert(sessionFiles().length > 0, 'no session files for the chat before delete');
  await page.getByRole('button', { name: 'Topic actions for Charlie doomed chat topic' }).click();
  await page.getByRole('menuitem', { name: /Delete/ }).click();
  await page.getByRole('alertdialog').getByRole('button', { name: 'Delete' }).click()
    .catch(async () => page.getByRole('dialog').getByRole('button', { name: 'Delete' }).click());
  const resp = await waitFrame(page, (d) => d.event === 'webui_response' && d.ok !== undefined &&
    page.rec.frames.some((f) => f.dir === '>' && f.data.request_id === d.request_id && /delete/.test(f.data.action || '')));
  assert(resp.ok === true, `session.delete failed: ${JSON.stringify(resp)}`);
  await page.waitForTimeout(800);
  assert(!(await sidebarTitles(page)).includes('Charlie doomed chat topic'), 'chat still in sidebar');
  await shot(page, 's4_deleted');
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);
  const titles = await sidebarTitles(page);
  assert(!titles.includes('Charlie doomed chat topic'), `chat back after reload: ${JSON.stringify(titles)}`);
  const left = sessionFiles();
  assert(left.length === 0, `files left: ${JSON.stringify(left)}`);
  const sessions = await apiGet(page, '/api/sessions');
  assert(!JSON.stringify(sessions.body).includes(chatId), 'REST /api/sessions still lists the chat');
  await shot(page, 's4_deleted_reload');
  return `delete request ok; files removed; sidebar after reload ${JSON.stringify(titles)}`;
}

async function s4b_delete_running(open) {
  // Deleting a chat whose turn is still streaming: the turn is discarded and
  // its end must not bring the session back.
  const page = await open();
  await page.getByRole('button', { name: 'New topic' }).click();
  await page.waitForTimeout(300);
  await send(page, 'Delta slowstream doomed');
  const chatId = await activeChatId(page);
  await page.getByText('word2').first().waitFor({ timeout: 15000 });
  await page.waitForTimeout(800);
  const label = (await sidebarTitles(page)).find((t) => /Delta slowstream doomed/.test(t));
  assert(label, `running chat not in sidebar: ${JSON.stringify(await sidebarTitles(page))}`);
  await page.getByRole('button', { name: `Topic actions for ${label}` }).click();
  await page.getByRole('menuitem', { name: /Delete/ }).click();
  await page.getByRole('alertdialog').getByRole('button', { name: 'Delete' }).click()
    .catch(async () => page.getByRole('dialog').getByRole('button', { name: 'Delete' }).click());
  // Let the scripted stream (about 6 s) run past its end.
  await page.waitForTimeout(8000);
  await shot(page, 's4b_deleted_running');
  const files = findFiles(STATE_ROOT, (p) => p.includes(chatId) || p.includes(sessionFileName(chatId)));
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);
  const titles = await sidebarTitles(page);
  await shot(page, 's4b_deleted_running_reload');
  assert(files.length === 0, `files re-created after the turn ended: ${JSON.stringify(files)}`);
  assert(!titles.some((t) => /Delta slowstream/.test(t)), `chat back after reload: ${JSON.stringify(titles)}`);
  return `deleted mid-turn; nothing re-created; sidebar ${JSON.stringify(titles)}`;
}

async function s3b_switch_while_streaming(open) {
  // A turn streaming in chat A must not leak into chat B while B is shown,
  // and A shows its full reply when switched back.
  const page = await open();
  await page.getByRole('button', { name: 'New topic' }).click();
  await page.waitForTimeout(300);
  await send(page, 'Echo slowstream in A');
  const chatA = await activeChatId(page);
  await page.getByText('word2').first().waitFor({ timeout: 15000 });
  await page.getByRole('button', { name: 'New topic' }).click();
  await page.waitForTimeout(300);
  await send(page, 'Foxtrot in B');
  await page.getByText('Echo: Foxtrot in B').first().waitFor({ timeout: 15000 });
  const chatB = await activeChatId(page);
  assert(chatA !== chatB, 'no second chat');
  await page.waitForTimeout(1500);
  const bText = await mainText(page);
  await shot(page, 's3b_chat_b_while_a_streams');
  assert(!/word\d/.test(bText), `chat A's stream leaked into B:\n${bText}`);
  await waitFrame(page, (d) => d.event === 'turn_end' && d.chat_id === chatA, 15000);
  const label = (await sidebarTitles(page)).find((t) => /Echo slowstream/.test(t)) || 'Echo slowstream in A';
  await page.locator('aside, nav').getByText(label).first().click();
  await page.getByText('done-slow.').first().waitFor({ timeout: 10000 });
  const aText = await mainText(page);
  await shot(page, 's3b_chat_a_complete');
  assert((aText.match(/word1 /g) || []).length === 1, `chat A reply duplicated or missing:\n${aText}`);
  assert(!aText.includes('Foxtrot'), 'chat B content in chat A');
  return 'no cross-chat leak; A complete on return';
}

async function s5_stop_and_slash(open) {
  const page = await open();
  await page.getByRole('button', { name: 'New topic' }).click();
  await page.waitForTimeout(300);
  await send(page, 'Run something sleepy');
  const chatId = await activeChatId(page);
  await waitFrame(page, (d) => d.chat_id === chatId && d.event === 'message' && d.kind === 'tool_hint', 15000);
  await page.getByText('sleep 8').first().waitFor({ timeout: 5000 });
  await page.waitForTimeout(500);
  await shot(page, 's5_running');
  // While the turn runs, "/" offers /stop first.
  let box = await composer(page);
  await box.fill('/');
  await page.waitForTimeout(800);
  const runningMenu = await slashMenuCommands(page);
  await shot(page, 's5_slash_menu_running');
  assert(runningMenu[0] === '/stop', `running slash menu: ${JSON.stringify(runningMenu)}`);
  await box.fill('');
  const t0 = Date.now();
  await sendCommand(page, '/stop');
  // Like nanobot, a cancelled chat turn ends with goal_status idle (no
  // turn_end: loop.py publishes a completion only for session-queue commands)
  // and the /stop reply.
  await waitFrame(page, (d) => d.event === 'goal_status' && d.chat_id === chatId && d.status === 'idle', 10000);
  const elapsed = Date.now() - t0;
  await page.getByText('Stopped 1 task(s).').first().waitFor({ timeout: 5000 });
  await page.waitForTimeout(1000);
  const text = await mainText(page);
  await shot(page, 's5_stopped');
  assert(elapsed < 4500, `turn did not stop promptly (${elapsed} ms)`);
  assert(!text.includes('Slept well.'), 'the tool finished instead of being stopped');
  // The composer is idle again (no Stop control) and the next message works.
  await send(page, 'After stop hello');
  await page.getByText('Echo: After stop hello').first().waitFor({ timeout: 15000 });
  // Idle slash menu: NanoClaw's commands from /api/commands (the UI shows 8
  // at a time and hides /stop while idle); filtering reaches the rest.
  box = await composer(page);
  await box.fill('/');
  await page.waitForTimeout(800);
  const idleMenu = await slashMenuCommands(page);
  await shot(page, 's5_slash_menu');
  for (const cmd of ['/new', '/status', '/compact']) assert(idleMenu.includes(cmd), `idle slash menu lacks ${cmd}: ${JSON.stringify(idleMenu)}`);
  assert(!idleMenu.includes('/stop'), 'idle slash menu shows /stop');
  for (const [query, cmd] of [['/hel', '/help'], ['/dream-l', '/dream-log']]) {
    await box.fill(query);
    await page.waitForTimeout(500);
    const filtered = await slashMenuCommands(page);
    assert(filtered.includes(cmd), `slash menu for ${query} lacks ${cmd}: ${JSON.stringify(filtered)}`);
  }
  await box.fill('');
  return `idle ${elapsed} ms after /stop; UI: ${text.split('\n').filter((l) => /stop/i.test(l)).join(' | ')}`;
}

async function slashMenuCommands(page) {
  // Each palette row ends with its command text (e.g. "/new", "/model [preset]").
  const text = await page.locator('body').innerText();
  return [...text.matchAll(/^(\/[a-z][a-z-]*)(?: \[[^\]]*\])?$/gm)].map((m) => m[1]);
}

async function s6_failed_turn(open) {
  const page = await open();
  await page.getByRole('button', { name: 'New topic' }).click();
  await page.waitForTimeout(300);
  await send(page, 'Please explode now');
  const chatId = await activeChatId(page);
  const end = await waitFrame(page, (d) => d.event === 'turn_end' && d.chat_id === chatId, 15000);
  await page.waitForTimeout(1000);
  const text = await mainText(page);
  await shot(page, 's6_failed');
  // nanobot's failed-turn state: turn_end carries the outcome, no assistant
  // message (no "Error: ..." bubble), the UI shows its failure notice.
  assert(end.outcome === 'failed' && end.failure_kind === 'model', `turn_end: ${JSON.stringify(end)}`);
  const messages = received(page, 'message', chatId);
  assert(messages.length === 0, `assistant message sent: ${JSON.stringify(messages)}`);
  assert(received(page, 'delta', chatId).length === 0, 'error text streamed');
  assert(/Model provider request failed/.test(text), `no failure notice:\n${text}`);
  assert(!/scripted provider failure/.test(text), 'raw exception text shown');
  // After a reload the thread replays the failed turn without an error
  // bubble. (The notice itself is live-only UI state; nanobot's persistent
  // "Continue" offer comes from its WebUI recovery coordinator, which
  // NanoClaw does not port.)
  await page.reload({ waitUntil: 'networkidle' });
  await page.getByText('Please explode now').first().waitFor({ timeout: 10000 });
  await page.waitForTimeout(800);
  const replay = await mainText(page);
  assert(!/scripted provider failure|Error:/.test(replay), `error text after reload:\n${replay}`);
  await shot(page, 's6_failed_reload');
  await send(page, 'Recovered hello');
  await page.getByText('Echo: Recovered hello').first().waitFor({ timeout: 15000 });
  await shot(page, 's6_next_message');
  return `turn_end outcome=${end.outcome} failure_kind=${end.failure_kind}; notice shown live, no error bubble after reload; next message answered`;
}

async function s7_commands(open) {
  const page = await open();
  await page.getByRole('button', { name: 'New topic' }).click();
  await page.waitForTimeout(300);
  await send(page, 'Hello before commands');
  await page.getByText('Echo: Hello before commands').first().waitFor({ timeout: 15000 });
  await page.waitForTimeout(500);
  await sendCommand(page, '/status');
  await page.waitForTimeout(2000);
  let text = await mainText(page);
  await shot(page, 's7_status');
  assert(/model|tokens|context|session/i.test(text.split('/status').pop() || ''), `/status reply not rendered:\n${text}`);
  await sendCommand(page, '/help');
  await page.waitForTimeout(2000);
  text = await mainText(page);
  await shot(page, 's7_help');
  assert(/nanoclaw commands:[\s\S]*\/stop/.test(text), `/help reply not rendered:\n${text}`);
  return 'status and help rendered';
}

async function s8_reload_midstream(open) {
  const page = await open();
  await page.getByRole('button', { name: 'New topic' }).click();
  await page.waitForTimeout(300);
  await send(page, 'Give me a slowstream answer');
  const chatId = await activeChatId(page);
  await page.getByText('word3').first().waitFor({ timeout: 15000 });
  const mark = page.rec.frames.length;
  const threadMark = page.rec.threads.length;
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(700);
  // Find the running chat again (the UI may restore it as the active pane).
  if (!(await page.locator(`[data-testid="workbench-pane-websocket:${chatId}"]`).count())) {
    const label = (await sidebarTitles(page)).find((t) => /^Give me a/.test(t));
    if (label) await page.locator('aside, nav').getByText(label).first().click().catch(() => {});
  }
  // The UI's own webui-thread fetch for the chat (or, if it reused a cached
  // thread, the same REST call made directly) names the running turn.
  let restActive = null;
  for (let i = 0; i < 20 && !restActive; i++) {
    const hit = page.rec.threads.slice(threadMark).find((t) => t.url.includes(chatId));
    if (hit) restActive = hit.body.active_turn_id || '';
    else await page.waitForTimeout(100);
  }
  if (restActive === null) {
    const direct = await apiGet(page, `/api/sessions/websocket%3A${chatId}/webui-thread?limit=40&direction=latest`);
    restActive = (direct.body && direct.body.active_turn_id) || '';
  }
  await shot(page, 's8_after_reload_running');
  const runningText = await mainText(page);
  // After the reload: the thread REST payload names the active turn and the
  // re-attached socket reports it running.
  const afterReload = () => page.rec.frames.slice(mark).filter((f) => f.dir === '<' && f.data.chat_id === chatId).map((f) => f.data);
  const running = afterReload().some((d) => d.event === 'goal_status' && d.status === 'running');
  assert(restActive, 'webui-thread after reload has no active_turn_id');
  await page.getByText('done-slow.').first().waitFor({ timeout: 20000 });
  await page.waitForTimeout(800);
  await shot(page, 's8_completed');
  const finalText = await mainText(page);
  assert(running, 'no running state after reload');
  assert((finalText.match(/word1 /g) || []).length === 1, `reply duplicated after resume:\n${finalText}`);
  assert(afterReload().some((d) => d.event === 'turn_end'), 'no turn_end after reload');
  return `active_turn_id ${restActive.slice(0, 8)} after reload, goal_status running, completed; UI before completion: ${runningText.replace(/\n+/g, ' | ').slice(0, 200)}`;
}

async function s9_secret(open) {
  let page = await open();
  await page.waitForTimeout(1500);
  const text = await page.locator('body').innerText();
  await shot(page, 's9_secret_prompt');
  const boot = page.rec.responses.find((r) => r.url.includes('/webui/bootstrap'));
  assert(boot && boot.status === 401, `bootstrap without secret: ${JSON.stringify(boot)}`);
  assert((await page.locator('input[type="password"]').count()) > 0 || /secret|password/i.test(text), `no secret prompt:\n${text}`);
  // A wrong secret is rejected; typing the right one connects.
  const input = page.locator('input').first();
  await input.fill('wrong-secret');
  await input.press('Enter');
  await page.waitForTimeout(1200);
  assert((await page.locator('textarea').count()) === 0, 'wrong secret accepted');
  await input.fill(SECRET);
  await input.press('Enter');
  await composer(page);
  await shot(page, 's9_secret_typed');
  await page.close();
  page = await open(`${BASE}/#?bootstrapSecret=${encodeURIComponent(SECRET)}`);
  await send(page, 'Secret hello');
  await page.getByText('Echo: Secret hello').first().waitFor({ timeout: 15000 });
  await shot(page, 's9_with_secret');
  return 'secret prompt without secret (401); works with #?bootstrapSecret';
}

(async () => {
  const browser = await chromium.launch();
  const only = process.env.E2E_ONLY ? process.env.E2E_ONLY.split(',') : null;
  const all = SECRET
    ? [['s9_secret', s9_secret]]
    : [
        ['s1_markdown', s1_markdown],
        ['s2_reasoning_tools', s2_reasoning_tools],
        ['s3_two_chats', s3_two_chats],
        ['s3b_switch_while_streaming', s3b_switch_while_streaming],
        ['s4_delete', s4_delete],
        ['s4b_delete_running', s4b_delete_running],
        ['s5_stop_and_slash', s5_stop_and_slash],
        ['s6_failed_turn', s6_failed_turn],
        ['s7_commands', s7_commands],
        ['s8_reload_midstream', s8_reload_midstream],
      ];
  for (const [name, fn] of all) {
    if (only && !only.includes(name)) continue;
    await scenario(name, browser, fn);
  }
  await browser.close();
  process.exit(results.every((r) => r.ok) ? 0 : 1);
})();
