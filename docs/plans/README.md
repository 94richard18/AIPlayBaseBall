# 實作計劃

這個資料夾放實作計劃（`*.md`）。在 Claude Code 裡新增或修改這裡的計劃後，Stop hook（`.claude/hooks/codex-review-gate.js`）會要求先經過 Codex 交叉審查（`.claude/skills/codex-review/`）才能收工。

- 審查通過的計劃，檔案最後一行會有 `<!-- codex-review: APPROVED thread=... sha256=... -->`。
- marker 綁定計劃內容的 hash；通過後再修改計劃，marker 就會失效，需要重新審查。
- 暫時關閉檢查：啟動 Claude Code 前設定環境變數 `CODEX_REVIEW_GATE=off`。

本檔（README.md）只是說明，不會被 hook 檢查。
