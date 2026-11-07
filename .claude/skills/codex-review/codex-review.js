#!/usr/bin/env node
// Cross-model review helper: Claude writes a plan, Codex reviews it in ONE
// persistent Codex thread until both agree, then the plan gets a marker.
//
// Usage (run from anywhere inside the project):
//   node codex-review.js start  <plan.md> [--new]   first review round (new Codex thread)
//   node codex-review.js reply  <plan.md> <msg.md>  send Claude's response, same thread ("-" = stdin)
//   node codex-review.js stamp  <plan.md>           write the APPROVED marker (only after Codex approved this exact text)
//   node codex-review.js status <plan.md>
//
// The marker is bound to a sha256 of the plan body, so editing the plan after
// approval invalidates it and the Stop hook will ask for another review.

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawnSync } = require('child_process');

const MARKER_RE = /^<!-- codex-review: .* -->[ \t]*$/gm;
const VALID_MARKER_RE = /^<!-- codex-review: APPROVED thread=(\S+) sha256=([0-9a-f]{64}) -->$/;

function projectRoot() {
  if (process.env.CLAUDE_PROJECT_DIR) return path.resolve(process.env.CLAUDE_PROJECT_DIR);
  return path.resolve(__dirname, '..', '..', '..');
}

function planBody(text) {
  return text.replace(/\r\n/g, '\n').replace(MARKER_RE, '').trimEnd();
}

function bodyHash(text) {
  return crypto.createHash('sha256').update(planBody(text), 'utf8').digest('hex');
}

// Returns { ok: true } or { ok: false, why }
function checkMarker(file) {
  if (!fs.existsSync(file)) return { ok: true, missing: true };
  const text = fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
  const lines = text.trimEnd().split('\n');
  const last = lines[lines.length - 1].trim();
  const m = last.match(VALID_MARKER_RE);
  if (!m) {
    const anywhere = lines.some((l) => VALID_MARKER_RE.test(l.trim()));
    return { ok: false, why: anywhere ? 'plan was edited after approval (marker is no longer last)' : 'no approval marker at end of file' };
  }
  if (m[2] !== bodyHash(text)) return { ok: false, why: 'plan was edited after approval (hash mismatch)' };
  return { ok: true };
}

function stateDir() {
  return path.join(projectRoot(), '.claude', 'codex-review-state');
}

function stateFile(plan) {
  const rel = path.relative(projectRoot(), plan).replace(/[\\/:]+/g, '__');
  return path.join(stateDir(), rel + '.json');
}

function loadState(plan) {
  const f = stateFile(plan);
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : null;
}

function saveState(plan, st) {
  fs.mkdirSync(stateDir(), { recursive: true });
  fs.writeFileSync(stateFile(plan), JSON.stringify(st, null, 2));
}

function parseVerdict(msg) {
  const m = msg.match(/VERDICT:\s*(APPROVED|CHANGES_REQUESTED)/gi);
  if (!m) return 'UNKNOWN';
  return /CHANGES/i.test(m[m.length - 1]) ? 'CHANGES_REQUESTED' : 'APPROVED';
}

function runCodex(args, prompt) {
  const root = projectRoot();
  const outFile = path.join(stateDir(), `last-${process.pid}.txt`);
  fs.mkdirSync(stateDir(), { recursive: true });
  // prompt goes through stdin ("-") to avoid shell quoting of long Chinese text
  const fullArgs = [...args, '--json', '-o', `"${outFile}"`, '-'];
  // single command string: codex is an npm .cmd shim on Windows, so it needs a shell
  const r = spawnSync(['codex', ...fullArgs].join(' '), {
    cwd: root,
    input: prompt,
    encoding: 'utf8',
    shell: true,
    maxBuffer: 64 * 1024 * 1024,
  });
  let threadId = null;
  for (const line of (r.stdout || '').split('\n')) {
    try {
      const ev = JSON.parse(line);
      if (ev.type === 'thread.started') threadId = ev.thread_id;
      if (ev.type === 'turn.failed' || ev.type === 'error') {
        console.error('[codex error]', JSON.stringify(ev).slice(0, 800));
      }
    } catch (_) { /* non-JSON line */ }
  }
  let msg = '';
  if (fs.existsSync(outFile)) {
    msg = fs.readFileSync(outFile, 'utf8');
    fs.unlinkSync(outFile);
  }
  if (r.status !== 0 || !msg) {
    console.error(`codex exited with status ${r.status}`);
    console.error((r.stderr || '').split('\n').filter((l) => /error/i.test(l)).slice(-5).join('\n'));
    process.exit(2);
  }
  return { threadId, msg };
}

const REVIEWER_BRIEF = (relPlan) => `你是這個專案的實作計劃審稿人（reviewer）。作者是另一個 AI（Claude）。
請閱讀計劃檔：${relPlan}
需要時可以自行讀取專案裡的程式碼與文件來核對計劃是否正確。你是唯讀的，不要修改任何檔案。

審稿原則：
1. 第一輪就把真正重要的問題一次找出來：會導致 bug、錯誤結果、遺漏的 corner case、與現有程式碼不符、無法驗證的步驟。
2. 不要為了挑而挑。風格偏好、可有可無的優化、過度設計，不要列為必改問題（可以標成「建議，非必要」）。
3. 每個問題給編號（R1、R2…），說明：問題、為什麼重要、建議的改法。
4. 之後的回合，作者會逐條回覆：已修改，或提出反駁理由。對每一條你都必須明確表態：
   「接受」（被說服或確認已修正）或「堅持」（並清楚說明你堅持的是什麼、為什麼）。
   不要因為想結束就放水，也不要在後續回合發明與先前無關的新小毛病；只有在修改本身引入新問題時才提出新問題。
5. 只有在所有必改問題都已解決或你已被說服時，才判定通過。

回覆的最後一行必須是以下其中之一（單獨一行）：
VERDICT: APPROVED
VERDICT: CHANGES_REQUESTED`;

function resolvePlan(p) {
  if (!p) usage();
  const abs = path.resolve(p);
  if (!fs.existsSync(abs)) {
    console.error(`plan not found: ${abs}`);
    process.exit(1);
  }
  return abs;
}

function usage() {
  console.error('usage: codex-review.js start|reply|stamp|status <plan.md> [msg.md|-] [--new]');
  process.exit(1);
}

function main() {
  const [cmd, planArg, extra] = process.argv.slice(2);
  if (!cmd) usage();
  const plan = resolvePlan(planArg);
  const relPlan = path.relative(projectRoot(), plan).replace(/\\/g, '/');
  const st = loadState(plan);

  if (cmd === 'start') {
    if (st && st.threadId && !process.argv.includes('--new')) {
      console.error(`A Codex thread already exists for this plan (${st.threadId}).`);
      console.error('Use "reply" to continue the SAME conversation, or "start --new" only for a genuinely new plan.');
      process.exit(1);
    }
    const text = fs.readFileSync(plan, 'utf8');
    const { threadId, msg } = runCodex(['exec', '-s', 'read-only'], REVIEWER_BRIEF(relPlan));
    const verdict = parseVerdict(msg);
    saveState(plan, { plan: relPlan, threadId, rounds: 1, verdict, verdictHash: bodyHash(text) });
    console.log(msg.trim());
    console.log(`\n[codex-review] round 1, thread ${threadId}, verdict ${verdict}`);
    return;
  }

  if (cmd === 'reply') {
    if (!st || !st.threadId) {
      console.error('No Codex thread for this plan yet. Run "start" first.');
      process.exit(1);
    }
    if (!extra) usage();
    const reply = extra === '-' ? fs.readFileSync(0, 'utf8') : fs.readFileSync(path.resolve(extra), 'utf8');
    const text = fs.readFileSync(plan, 'utf8');
    const prompt = `作者（Claude）的回覆如下。計劃檔 ${relPlan} 已依回覆更新，請重新讀取最新內容後，對每一條先前的問題明確表態「接受」或「堅持」。\n\n${reply}\n\n記得最後一行寫 VERDICT: APPROVED 或 VERDICT: CHANGES_REQUESTED。`;
    const { msg } = runCodex(['exec', 'resume', '-c', 'sandbox_mode="read-only"', st.threadId], prompt);
    const verdict = parseVerdict(msg);
    saveState(plan, { ...st, rounds: st.rounds + 1, verdict, verdictHash: bodyHash(text) });
    console.log(msg.trim());
    console.log(`\n[codex-review] round ${st.rounds + 1}, thread ${st.threadId}, verdict ${verdict}`);
    return;
  }

  if (cmd === 'stamp') {
    const text = fs.readFileSync(plan, 'utf8');
    if (!st || st.verdict !== 'APPROVED') {
      console.error('Codex has not approved this plan. Keep reviewing (reply) until VERDICT: APPROVED.');
      process.exit(1);
    }
    if (st.verdictHash !== bodyHash(text)) {
      console.error('The plan changed after Codex approved it. Send the change to Codex with "reply" first.');
      process.exit(1);
    }
    const eol = text.includes('\r\n') ? '\r\n' : '\n';
    const body = planBody(text);
    const marker = `<!-- codex-review: APPROVED thread=${st.threadId} sha256=${bodyHash(text)} -->`;
    fs.writeFileSync(plan, (body + '\n\n' + marker + '\n').replace(/\n/g, eol));
    console.log(`[codex-review] stamped ${relPlan}`);
    return;
  }

  if (cmd === 'status') {
    console.log(JSON.stringify({ state: st, marker: checkMarker(plan) }, null, 2));
    return;
  }

  usage();
}

module.exports = { checkMarker, bodyHash, projectRoot };

if (require.main === module) main();
