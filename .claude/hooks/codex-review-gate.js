#!/usr/bin/env node
// Stop hook: the "gatekeeper" of cross model review.
// It does not review anything. It only checks that every plan under docs/plans/
// that was written or edited in this session ends with a valid Codex APPROVED
// marker (see .claude/skills/codex-review). If not, it blocks Claude from
// stopping and tells it to run the codex-review skill.
//
// Escape hatch: set CODEX_REVIEW_GATE=off in the environment.

const fs = require('fs');
const path = require('path');
const { checkMarker } = require('../skills/codex-review/codex-review.js');

const WRITE_TOOLS = new Set(['Write', 'Edit', 'MultiEdit', 'NotebookEdit']);
const MAX_BLOCKS_WITHOUT_PROGRESS = 2;

function readStdin() {
  try {
    return JSON.parse(fs.readFileSync(0, 'utf8') || '{}');
  } catch (_) {
    return {};
  }
}

function plansTouched(transcriptPath, root) {
  const plansDir = path.join(root, 'docs', 'plans').toLowerCase() + path.sep;
  const found = new Set();
  if (!transcriptPath || !fs.existsSync(transcriptPath)) return found;
  for (const line of fs.readFileSync(transcriptPath, 'utf8').split('\n')) {
    if (!line.includes('tool_use')) continue;
    let entry;
    try { entry = JSON.parse(line); } catch (_) { continue; }
    const content = entry && entry.message && entry.message.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (block.type !== 'tool_use' || !WRITE_TOOLS.has(block.name)) continue;
      const p = block.input && (block.input.file_path || block.input.notebook_path);
      if (!p) continue;
      const abs = path.resolve(root, p);
      const lower = abs.toLowerCase();
      if (lower.startsWith(plansDir) && lower.endsWith('.md') && path.basename(lower) !== 'readme.md') found.add(abs);
    }
  }
  return found;
}

function main() {
  if ((process.env.CODEX_REVIEW_GATE || '').toLowerCase() === 'off') return;
  const input = readStdin();
  const root = path.resolve(process.env.CLAUDE_PROJECT_DIR || input.cwd || process.cwd());

  const pending = [];
  for (const plan of plansTouched(input.transcript_path, root)) {
    const r = checkMarker(plan);
    if (!r.ok) pending.push({ rel: path.relative(root, plan).replace(/\\/g, '/'), why: r.why, mtime: fs.statSync(plan).mtimeMs });
  }

  // Loop guard: if Claude keeps stopping with the exact same unreviewed state,
  // let it stop after a couple of tries and warn the user instead of looping forever.
  const guardDir = path.join(root, '.claude', 'codex-review-state');
  const guardFile = path.join(guardDir, `gate-${input.session_id || 'unknown'}.json`);
  if (pending.length === 0) {
    if (fs.existsSync(guardFile)) fs.unlinkSync(guardFile);
    return;
  }
  const signature = JSON.stringify(pending.map((p) => [p.rel, p.why, p.mtime]));
  let guard = { signature: '', count: 0 };
  try { guard = JSON.parse(fs.readFileSync(guardFile, 'utf8')); } catch (_) { /* first block */ }
  guard.count = guard.signature === signature ? guard.count + 1 : 1;
  guard.signature = signature;
  fs.mkdirSync(guardDir, { recursive: true });
  fs.writeFileSync(guardFile, JSON.stringify(guard));

  const list = pending.map((p) => `- ${p.rel}（${p.why}）`).join('\n');
  if (guard.count > MAX_BLOCKS_WITHOUT_PROGRESS) {
    process.stdout.write(JSON.stringify({
      systemMessage: `codex-review gate：以下計劃仍未通過 Codex 審查，已連續攔截 ${MAX_BLOCKS_WITHOUT_PROGRESS} 次沒有進展，先放行。\n${list}`,
    }));
    return;
  }

  process.stdout.write(JSON.stringify({
    decision: 'block',
    reason:
      `以下實作計劃尚未通過 Codex 交叉審查，還不能收工：\n${list}\n\n` +
      '請使用 codex-review skill（.claude/skills/codex-review/SKILL.md）：用 start 開始審查（已有 thread 就用 reply 接續同一個對話），' +
      '逐條修改或反駁，直到 Codex 回覆 VERDICT: APPROVED，再用 stamp 蓋上 marker。' +
      '如果 Codex CLI 無法執行，或雙方在同一爭議卡住需要使用者裁決，請直接向使用者說明原因，不要自行蓋章。',
  }));
}

main();
