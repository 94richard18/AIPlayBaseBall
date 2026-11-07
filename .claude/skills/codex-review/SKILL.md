---
name: codex-review
description: 讓 Codex 審查 docs/plans/ 底下的實作計劃，Claude 與 Codex 在同一個 Codex 對話裡來回討論直到達成共識，通過後在計劃結尾蓋上 APPROVED marker。Stop hook 擋下「計劃沒有 marker」的收工時，或使用者要求 codex review / 交叉審查計劃時使用。
---

# Codex 交叉審查（cross model review）

你（Claude）是計劃的作者，Codex 是審稿人。目標是**真正的共識**，不是跑滿幾輪。

所有指令都用這支輔助腳本（路徑相對於專案根目錄）：

```
node .claude/skills/codex-review/codex-review.js <command> <plan.md> ...
```

Codex 每輪可能要跑好幾分鐘，Bash 呼叫請設 `timeout: 600000`。

## 流程

1. **開始審查**（每份計劃只做一次，會開一個新的 Codex 對話並記住 thread id）：
   `node .claude/skills/codex-review/codex-review.js start docs/plans/<plan>.md`
   - 如果回報這份計劃已經有 thread，**不要**加 `--new`，改用 `reply` 接續同一個對話。只有計劃內容換成完全不同的主題時才用 `--new`。

2. **逐條處理 Codex 的意見**（R1、R2…）：
   - 同意 → 直接修改計劃檔。
   - 不同意 → 不要改，準備具體理由（引用程式碼、數據、使用者已做的決定）說服 Codex。
   - 不要為了過關假裝問題不存在，也不要為了討好而接受會造成過度設計的建議。

3. **回覆 Codex**（永遠接續同一個對話）：把回覆寫成一個暫存檔，逐條對應編號，例如：
   ```
   R1 已修改：在 3-2 節加上……
   R2 不同意：……（理由）
   ```
   然後執行：
   `node .claude/skills/codex-review/codex-review.js reply docs/plans/<plan>.md <回覆檔>`
   暫存檔放在系統暫存目錄或 scratchpad，不要放進專案。

4. 重複 2–3，**沒有輪數上限**，直到最後一行是 `VERDICT: APPROVED`。
   - 如果兩邊卡在同一個爭議超過三輪仍無法說服對方，停下來把雙方立場整理給使用者裁決，不要自行蓋章。

5. **蓋章**（只有 Codex 對「目前這份內容」判定 APPROVED 才會成功）：
   `node .claude/skills/codex-review/codex-review.js stamp docs/plans/<plan>.md`
   - 蓋章後**不要再修改計劃**；任何修改都會讓 marker 失效（內容 hash 不符），Stop hook 會要求重審。若真的要改，改完用 `reply` 告訴 Codex 改了什麼，取得 APPROVED 後再 `stamp`。

6. 向使用者簡短回報：審了幾輪、Codex 提出哪些重點、哪些已修改、哪些經討論後保留。

## 注意

- 審查期間不要開始實作，等蓋章後再進入實作。
- Codex 以唯讀模式執行，它可以自己讀專案的程式碼，不需要把程式碼貼給它。
- `status` 指令可以看目前 thread、輪數、最後判定與 marker 狀態。
- 若 Codex CLI 失敗（未登入、模型錯誤、網路），把錯誤訊息回報給使用者，不要自行蓋章繞過。
