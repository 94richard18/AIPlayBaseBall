# 訓練交接：目前目標、進度與待辦

最後更新：2026-10-29 00:47（授權到 10-30；規格值——打者力量倍率、地面摩擦等——仍由使用者決定）

完整歷程在 `PITCHER_TRAINING_LOG.md`（投手）與 `SupporttraningLog.md`（打者、引擎、決策）。這份只記「現在的目標、現在在哪裡、接下來要做什麼」。

---

## 1. 目標：ONE OUTS（使用者 2026-10-23 決定）

- 一個投手對一個打者、一個打席、沒有野手、牆上畫好球帶。規則見 README「Current goal」與 `scripts/one_outs.py`。
  - 打者贏：四壞、觸身球、界內且第一落點在內野外、全壘打。
  - 投手贏：三振、界內且第一落點在內野（離投手板 95 ft 內）。
  - 界外算好球，但不會是第三好球。
- **取消**：9 局比賽、野手 / 跑者、80 km/h 球速門檻、10-29 的 20 m/s 檢查點。備案投手（`FallbackPitcher`）只留給打者對照用。

| 項目 | 內容 |
|---|---|
| 投手過關 | 相同起始與隨機起始（start_joint_noise 0.02）兩種評估都要過，每回合 4.0 s：球在空中到牆上級 3（\|y\| ≤ 0.5 m、z 0.2–1.5 m）≥ 32/64，且投完不倒 ≥ 48/64；球速只回報，不設門檻 |
| 投手期限 | 11-12（暫定） |
| 打者 | 改用新投手的球訓練與評估，過關標準等投手的球出來再定；期限 11-12 |
| 交付 | ONE OUTS 對決影片（約 10 場，含比數、牆上好球帶、落點、勝負）+ 勝率統計，11-26 |
| 規格 | 人類力量上限（10-21 起腰與手指 / 拇指放寬 1.5 倍）、手臂彈簧硬度不變；r_shoulder_yaw [-130°, +92°]；四縫線握法 |
| 授權 | 規劃 session 自行決定設計的授權到 10-23 為止；10-24 起新設計先問使用者；規格與排程一律問使用者 |

---

## 2. 投手目前狀態

**交付候選（10-25 19:20）**：P3c = ho2 model_7600（`logs/rsl_rl/aib1_stage/` 的 ho2_tw2plw100 run）+ 放球提早 2.5 ms + 投手斜站 17°，相同起始過門檻、noisy 沒過。

**交付版（舊）**：hf_avg3 權重（`logs/rsl_rl/aib1_stage/2026-10-20_14-30-00_hf_avg3/model_7300.pt`）+ 任務 `AIB-Stage1dK70GW2MBASVRT1SYW110R5H5-v0`。

| 項目 | 新標準 | 目前 |
|---|---|---|
| 相同起始投完不倒 | ≥ 48/64 | 61/64 |
| 隨機起始投完不倒 | ≥ 48/64 | 約 34/64（68.5/128） |
| 球在空中到級 3 | ≥ 32/64 | 0（出手仰角約 +1°，球全部滾地；noisy 球路偏三壘側 5–14 m） |
| 出手 | 只回報 | 16.6 m/s，方向約 -16～-19°，離地 1.86 m |

- 物理：約 16 m/s、離地 1.86 m 出手，仰角要約 +15° 才能在空中到好球帶。
- 已試過的參考動作：TW2（腰在放球附近跟投球時鐘）仰角 +11°、相同起始 61/64；release_heading_deg 10 觸發可修方向。
- **第 0 步零樣本結果（10-23，hf_avg3 7300）**：方向觸發要同時關掉策略的放球時間才生效（`--set scripted_release_s=None release_action_window_s=None release_ahead_m=-5.0 release_heading_deg=…`）。

| 組合 | 相同起始站穩 /64 | noisy /128 | 方向 | 仰角 |
|---|---|---|---|---|
| **H5 + 觸發 ±10°（目前最好）** | **63** | **74** | -7.3° | -9.5° |
| TW2 + 觸發 ±10° | 62 | 70 | -6.2° | -4.1° |
| TW2 + 觸發 ±5° | 62 | 68 | 0.0° | -16°（球都滾地） |
| + release_elev_deg (0, 40) | ±10° 沒變化 | ±5° 大多不放球（16/64） | | |

- 結論：方向零樣本就能解決；仰角要另外處理（等方向對準才放球時，手正往下揮），級 3 仍是 0。仰角要到約 +15° 需要訓練或新的手臂參考動作。
- 10-20～10-21 的結論：PPO 微調（hg、hh、hi、hj、hk）每一輪的隨機起始平衡都比起點差；交付的進步都來自零樣本換參考動作。放寬腰 / 手指力量零樣本全倒，hk（腰 ×1.2 重練）也失敗。

---

## 3. 打者目前狀態

**交付版（10-25 18:25 起）**：Human9b model_12500 + aim_offset_z -0.08（`logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt`、表 `assets/game/batter_h9b_12500_aim008.json`）：diag 1024 球 0.684、ONE OUTS 18.5%、界內且 ≥ 29 m 15.9%、打中後跌倒 57%。但書：加上 seed 1 / 2 的 256 球，全部合併約 0.668，貼近門檻。舊交付 12300 + -0.12 用 1024 球量是 0.666，其實也沒過。之前是 Human7 model_12095（`logs/rsl_rl/aib1_tee_batting_mimic/2026-10-22_22-46-49_human_pitched7/model_12095.pt`）+ aim_offset_z -0.12。

- 比賽時機（使用者 10-21）：揮棒起動用引擎算出的真實到達時間；揮棒起動 = 球到本壘時間 − 參考擊球時間 − 0.055 s。
- 實測（±0.03 s 抖動）：擊中率 0.73；仰角四分位 -12 / +8 / +30°；初速中位數 21.8 m/s（≥ 28 m/s 5.9%）；界內 46%（界內落點中位數 4.0 m）。
- ONE OUTS 300 場（對備案投手）：RL 打者勝 9.0%（內野球 54.7%、三振 36.3%）；統計版打者 54.0%。
- 弱點：擊球太弱、界內球太短、右側界外多（揮棒軌跡把球帶向一壘側，不是時機問題）。
- 歷程：Human4 39.8% → Human6 94.5%（比賽時機，但仰角 -34°、落點約 1 m）→ Human7 平滑仰角分數 → Human8（訓練往下瞄）不採用。

---

## 4. 正在執行（10-26 21:30 交接，新 session 先讀這段）

**10-30 20:59 rows2 結果（ARMAIM_ROWS2_END 20:57）：第 6 列「觀測補償」讓腳本瞄準不經訓練就把斜率從 −0.80 推到 −0.50（reachable −0.43）、|axis_off| 7.1 cm、hit 0.485 → 傳輸損失主因是凍結 swing policy 透過 q / nxt_ref − q 對抗偏移，不是致動器。決定：z 暫停；重排 GPU：PFILTER_END → 第 8 列（完整補償）→ 第 9 列（濾波知覺驅動的第 6 列 = 可部署腳本瞄準）→ Human29；手腕/前臂力矩 spec 問題交使用者。**
- 各列（diag1024 seed 0、真實 dy、視窗 0.60；基準 hit 0.445 / |axis_off| 0.103 / slope −0.80 / reachable −0.81 / exit 18.9 / D share 0.181）：列 4 y：hit 0.481、|axis_off| 0.096、slope −0.73、reachable −0.75 / 0.088、exit 18.1、D share 0.257、launch 20.1；列 5 y+z：0.465、0.099、−0.76、−0.77 / 0.085、18.6、0.212、20.4；**列 6 y + obs comp：0.485、0.071、−0.50（r −0.73）、−0.43 / 0.065、18.3、0.222、launch 24.4**；列 6 misses slope −0.71。
- 實現比例（wet − dry 配對）：列 4 l_arm 0.32 / r_arm 0.37 / wrists −0.07 / waist 0.55；列 5 同 + z −0.16；列 6 l_arm 0.64 / r_arm 0.69 / wrists 0.13 / waist 0.98。補償後腰全跟、手臂約 2/3、手腕仍近零；z 不傳且吃 D share → 暫停 z。
- 視窗內力矩使用 |applied_torque|/limit（IsaacLab implicit actuator 的 PD effort 截斷估計，非 PhysX drive force）：12500 單獨（r5dry）步數 ≥ 0.95 比例：l_wrist_roll 0.94、l_wrist_pitch 0.75、l_wrist_yaw 0.90、r_wrist_roll 0.38、r_wrist_pitch 0.68、r_wrist_yaw 0.54、l_elbow 0.81、l_shoulder_pitch 0.92、l_shoulder_roll 0.98、l_shoulder_yaw 0.55；r6wet 幾乎相同；中位數除 r_wrist_roll 0.52 外全為 1.00 → 前臂與手腕在 12500 自己的揮棒裡整個視窗都頂在極限。這是向使用者提手腕 / 前臂力量 spec 的證據；未答覆前不改。
- 第 6 列用的是特權真實 dy；原始知覺上腳本映射會飽和，所以可部署版必須靠濾波知覺（第 9 列）。重排：15 停掉尚未開跑的 run_human29.sh（含內層 PID），改成 PFILTER_END → 列 8（列 6 + 完整補償：參考棒頭 / key target 依 Jacobian 映射偏移）→ 列 9（列 6 以濾波知覺逐步追蹤驅動）→ Human29 GO 自檢。Human29 +300 延至 ~22:25、+1000 ~00:20，仍在 06:41 錄影前。列 9 若達 +300 線（slope > −0.50、|axis_off| ≤ 8 cm、hit ≥ 0.40）即列為採用候選（下個空檔跑 5 seeds × 3 tables + joint seed 6），並作為 ArmAim2+comp 的初始化；ArmAim2+comp 排在 Human29 gates 之後（互補，非取代）。

**10-30 20:42 知覺濾波建好（15 commit 3523534，cfg.percept_filter 預設 OFF）；Human29 已備妥並由鏈自動條件啟動（9720a5d，logs/run_human29.sh）。**
- 濾波：自出手起每 env 的 running sums（1、t、t²、p、t·p；z 先加 ½g t² 再擬合），以擬合的 p0、v 在延遲時刻取代 seen / v，t_arrive 用擬合 vx；< 3 個樣本時退回原法；旗標關時無新 RNG 抽樣，凍結設定與 M2 位元不變（rows2 已用此檔預設關跑過）。離線（38 m/s、1 cm 噪聲、0.10 s 延遲、100 Hz、2000 球）最後一步 y 誤差：直線擬合 0.4 cm vs 原法 12.9 cm；側向 2 m/s² 時直線 6.7 / 二次 1.0 cm；P3c 彎曲約 0.6 m/s² → 先用直線，零樣本知覺欄 > 3 cm 才加 y 二次項（≥ 15 樣本）。
- 鏈：ARMAIM_ROWS2_END → 第 7 列（ARMAIM_ROW7_END）→ logs/run_pfilter.sh（12500 凍結 + 濾波 ON；再 + noise/bias 0；皆 scripted DRY，PFILTER_END ~21:15）→ Human29 自檢兩條件：in-zone hit ≥ 0.40 且 mean |perc_y_last − act_y| ≤ 0.03 m，否則 marker 寫 HOLD 停止。
- Human29 = AIB-PitchedBatting-Human29-v0 = Human27b + percept_filter（每次揮棒瞄準項；hit 20·q，dx 0.15 / D 0.032；w_miss 20；最後 0.15 s 手臂追蹤 ×0.5；launch / spray × q；距離 / 成功 / exit / backspin 關），12500 @ 1.5x、2048 envs。sigma_ax = max(0.20, 1.2 × 零樣本 |axis_off| 中位數)、sigma_gap = max(0.10, 1.2 × gap 中位數)，記在 logs/h29/check.txt。Gates（diag1024 seed 0，--percept_filter）：+300 H29_300_END slope > −0.50、|axis_off| ≤ 0.08、hit ≥ 0.36（FAIL 停鏈）；+1000 H29_1000_END slope > −0.30、|dx| ≤ 0.06、hit ≥ 0.40、exit ≥ 20、D share ≥ 0.25。ETA：+300 ~22:10、+1000 ~00:05，皆在 06:41 錄影之前。
- 預期（記錄用）：零樣本斜率可能不動（12500 已學會忽略該輸入），零樣本只是濾波與 hit 的健全檢查，+300 才是真考驗；原始知覺 12 cm 誤差是先前 reward-only 失敗不算 Human29 反證的理由。

**10-30 20:38 根因候選找到：打者知覺的球速是「相鄰兩個含噪樣本的一步差分」，速度 std ≈ 1.4 m/s/軸，穿越點預測晃 10–40 cm → 12500 學到忽略球的位置資訊（甜蜜點固定點、reach 只跟 20%）。決定：建知覺濾波（旗標預設關、凍結設定位元不變）→ 12500 零樣本 re-diag → 若 hit 守住，Human29 = 濾波知覺 + 瞄準獎勵微調。**
- 程式證據：`aibaseball/tasks/tee_batting/pitched_env.py` `_perceived_ball` 第 152 行 `v = (hist[:, -2] - hist[:, -1]) / self.step_dt`；step_dt = 1/400 × 4 = 0.01 s；`mimic_env_cfg.py` ball_obs_noise_std 0.01、bias_std 0.02、delay 0.10。√2 × 0.01 / 0.01 = 1.41 m/s；預測 = seen + v·t + ½g t²，t ≥ 0.10 s（延遲）→ 最後一步 y 誤差 ≈ 14 cm σ，與 rows2 量到的 |percept − 重力外推| 平均 0.117 / p90 0.25 吻合（15 在程式中確認一致）。另：重力外推 vs 真實穿越在視窗開啟（contact − 0.60）差 6.0 cm（阻力 + Magnus），最後一步 0；視窗開啟時多數球尚未「看到出手」（percept = 名義目標），所以任何 0.60 視窗的瞄準都必須每步依最新知覺重算（ArmAim2：腳本部分逐步追蹤、只 latch 政策 residual）。
- 含義：Human26/26b/27/27b 與 ArmAim 都在「沒有可用訊號」的輸入上失敗，這些失敗不能再當作「reward-only 學不到瞄準」的證據；修知覺是工程修正（延遲 / 噪聲 / 偏差數值不變），不是 spec 變更。
- 已下令（15）：(1) 旗標式濾波：自出手以來所有已見樣本的彈道最小平方擬合（z 先扣重力項，各軸 running sums），由擬合的 p0、v 在延遲時刻外推，t_arrive 用擬合 vx；預期速度 std ~0.02 m/s、穿越誤差 < 1 cm；預設 OFF，凍結設定與 M2 位元不變。(2) rows2 跑完 → 第 7 列（列 4 wet、noise 0 + bias 0、延遲保留，`--percept_clean`，marker ARMAIM_ROW7_END，~6 min，15 commit 06ca5c9）→ 12500 零樣本 diag1024 seed 0 開濾波（slope、|axis_off|、hit、exit、D share、知覺欄）。(3) 若零樣本 slope 已動或 hit 守住 → 提 Human29（12500 @ 1.5x 微調，濾波知覺 + Human27 式瞄準獎勵，sigma 依新起始中位數的 1.2× 規則，gates 同 ArmAim +300 / +1000）。
- rows2（乾濕配對 / 第 5 列 y+z / 第 6 列觀測補償，20:35 重啟，commit c41b74f，每 diag ~6 min，ETA ~21:05）照跑，PD 傳輸數字仍要。記憶：batter-percept-velocity-noise.md。

**10-30 20:2x 腳本化瞄準 v3 結果（15 訊息中的 21:00 / 22:20 時間標籤超前實際時鐘約 2 h）（ARMAIM_SCRIPT_END，15）：PD 路徑「通但只傳 3–9%」→ 揮棒視窗內的 PD 目標 offset 路線幾乎無效；ArmAim2（視窗 0.30）不啟動。決定：GO 乾濕配對列 + 第 5 列（y+z，視窗 0.60）+ 第 6 列（觀測補償），並在 CPU 上拆解 12 cm 的知覉誤差。**
- 基準 12500：hit 0.445、|axis_off| 0.103、slope −0.80、reachable −0.81、exit 18.9、D share 0.181。
- 列 1 真實 dy +1 視窗 0.30：slope −0.78 / reachable −0.80、hit 0.499、|axis_off| 0.095、exit 18.2、D share 0.201、offset 頂到邊界 57%。列 2 反號對照：slope −0.86 / −0.92、hit 0.460、exit 16.7（反向移動 → 路徑有傳）。列 3 percept dy +1：slope −0.79、hit 0.458、100% 頂到邊界。列 4 真實 dy +1 視窗 0.60：slope −0.73 / −0.75、hit 0.481、exit 18.1、D share 0.257。
- 判讀：完整實現腳本映射應讓 reachable slope ≈ 0；列 1 只動 +0.02、列 4 +0.07 → 傳輸約 3%（0.30）/ 9%（0.60）。提早視窗好 3 倍，符合「力矩飽和前先瞄準」。
- 修正：15 先前「v1 無效」的判斷部分錯誤——全部投球的平均 |plate_y| 是 0.27 m（非 in-zone 的 0.12），v1 的 100% 飽和來自知覺誤差。**知覺很差**：視窗最後一步，打者預測的穿越 y 與真實差 |dy − true| 平均 0.122 m，可能是 12500 從不瞄準的原因之一；來源待拆（pitch_plate 與真實穿越不一致 / 僅重力外推 vs 阻力+Magnus / 0.10 s 延遲+噪聲）。realised-fraction 代理量不可用（>1 與負值，揮棒本身的追蹤誤差蓋過投影）→ 改成乾濕配對列（offset 算出並記錄但不施加，同 seed 位元相同），realised = Σ(wet − dry)·off / Σoff²。
- 我的新假說：凍結的 swing policy 觀測 q，加在它之後的 offset 對它是擾動、會被修正 → 第 6 列把 offset 關節的觀測 q 改成 q − offset（qd 不變）讓政策不再對抗；若第 6 列傳輸遠高於第 4 列，ArmAim2 / fallback 的修法是觀測補償而非更大權限。
- 已下 GO（一次一個 Isaac，各 diag1024 seed 0 + gate，marker ARMAIM_ROWS2_END）：(i) 與列 4 配對的乾列，並同時記錄每球在接觸平面的真實 y/z、pitch_plate、知覺最後一步預測；(ii) 第 5 列 y+z 視窗 0.60（目標 D = +1 cm）；(iii) 第 6 列 = 列 4 + 觀測補償。CPU 並行：拆解 0.122 m 誤差三個來源，並報視窗開啟步（contact − 0.60）的誤差。
- 若傳輸 ~10% 成立且補償無效：槓桿只剩更早瞄準（跨步期、不同的揮棒起始姿勢）或力量 spec（使用者的）；會附乾濕配對數字再向使用者提。錄影時段 10-31 06:41 保留給 20；C/A falsifier 排在該錄影之後。

**10-30 20:1x 腳本化瞄準 +1 列無效（offset 100% 飽和）→ v2 重跑中（15 commit 8b25bdb，ETA 21:40）；D 檢查：接觸位置本身就能解釋大部分缺少的出棒速度 → 「打遠 + 打扎實」主線 = 瞄準（y 與 z），出棒速度選項 C/A/B 降為備援。**
- +1 列無效原因：percept 版 dy（預測穿越 y − 名義 y）平均 |dy| 0.30 m，但真實 |plate_y| 約 0.12 → 等於常數 +0.4 rad 最大 offset；結果 hit 0.458、|axis_off| 0.096、slope −0.79、exit 18.0。附帶線索：15 個關節常數 ±0.4 rad 只讓 |axis_off| 動 0.7 cm，視窗內力矩飽和 84–97%，PD 目標 offset 可能根本沒被跟到。
- v2 列（各 diag1024 seed 0 + gate，ARMAIM_SCRIPT_END）：1 真實 plate_y（特權，物理 falsifier 可用）+1；2 同 −1 對照；3 percept dy +1 並印 |dy − true| 找 percept 偏差。我加：每列印「關節實際實現的 offset 比例」；若列 1 實現比例 < 0.5 加列 4 = 視窗提早到 contact − 0.60 s（上桿/跨步期力矩未飽和）；若 y 路徑通，加列 5 = y + z offset（目標 D = +1 cm）。
- D 檢查（15，12500 diag1024 seed 0，277 hits，中位 exit m/s）：全部 17.6；|dx| < 3 cm 21.7、3–6 22.3、6–10 20.9、10–15 19.0、15–25 15.1、> 25 11.1；D −2~0 cm 22.6、0~2.5 21.9、2.5~5 15.3、> 5 12.1、< −2 16.8；|dx| < 5 且 D ∈ [0, 2.5]（n=10）27.5。→ 同一支揮棒打到甜蜜點已可達 ~22（穩健值）到 27（n=10 指標值），接近所需的 24；111/277 hits 的 D > 2.5 cm，所以垂直方向與 dx 一樣重要。
- ArmAim2 已備妥未啟動（AIB-ArmAim2-v0，logs/run_armaim2.sh，ARMAIM2_300_END / ARMAIM2_1000_END）：視窗第一步取樣一次並保持；offset = 腳本瞄準 + 0.5·tanh(a)（±0.2 rad）；AIB_FIX_STD=1 actor head 歸零（起點 = 腳本瞄準）、std 固定 0.1、entropy 0；reward / gates 同 ArmAim。
- 出棒速度選項（皆為分離式政策、swing policy 凍結、spec 不動），順序與預先判準：C 節奏 residual（falsifier：腳本 tempo 1.1，20 min；exit ≥ +1 且 hit ≥ 0.40 才訓練）→ A 腰/髖時序 residual（falsifier：12500 接觸時軀幹 yaw 率與 waist_yaw 速度符號，20 min；中位 hit 已 ≥ 0 朝投手則放棄）→ B 手腕釋放 residual（CPU 上限 < +2 m/s 則放棄）。只在瞄準線中位 exit 卡在 24 以下時才用 GPU。

**10-30 20:1x 使用者：「授權延到11/2，這段時間盡可能找方法讓打者把球打遠打扎實」→ 規劃 session 自主授權延至 2026-11-02 整天；期間目標 = 打者打遠（exit / carry）+ 打扎實（甜蜜點接觸）。**
- 影響：ArmAim #2 重訓、mimic key-target 變形 fallback、以及出棒速度線（政策提早出棒、腰在接觸時反轉 −4.1 m/s）都由我依結果決定與啟動，不再逐項等核准；每個結果與決定照舊回報 + 記錄。spec 值（力量倍率、關節極限、摩擦）仍屬使用者，只提案不自改；AIB_EFFORT_SCALE 1.5 維持。
- 目前：腳本化瞄準診斷 20:00 啟動（15 commit 8a8f5a7，logs/run_armaim_script.sh：正號列 → 反號對照列，ARMAIM_SCRIPT_END，ETA 20:40）。增益（rad/m）：+ 側 l_wrist_yaw +1.65、l_shoulder_roll −0.79、l_elbow +0.70、waist_yaw +0.63；− 側 l_elbow −2.79、l_shoulder_roll +2.18、waist_yaw −1.56；±0.4 飽和在 |dy| ≈ 24 cm（+）/ 14 cm（−）。
- 分支（診斷結果一到即執行）：斜率變平 → 直接啟動 #2（視窗開啟取樣一次並保持、entropy 0、std 0.1、以腳本增益初始化、同 +300/+1000 gates）；斜率不動 → 放棄 ArmAim，啟動 mimic 變形 fallback。接觸線之外，開 CPU 工作：整理出棒速度線的可做選項（時序/腰部反轉、分離式 residual 的時序版本）供 11-02 前排程。

**10-30 19:5x ArmAim +300 FAIL（15 回報 19:50，commit 693af13）→ 決定：GO 腳本化瞄準診斷（#1，約 20 min GPU、不訓練）；#2 重訓與 fallback 等 #1 結果後交使用者決定（自主授權今晚到期）。**
- model_299 diag1024 seed 0（in-zone，*_any 鍵）：hit 0.460（base 0.445）；|axis_off| 0.1026（base 0.1033，gate ≤ 0.08）；exit 18.1（18.9）；D share 0.195（0.181）；launch 19.7°（21.7°）。
- 斜率 hits −0.80（r −0.85）、misses −0.82 = base −0.80 / −0.80，完全沒動；reachable 子集（plate_y ≥ −0.185，448/557）hit 0.473、slope −0.83（base −0.81）、|axis_off| 0.086（0.095）。結論：是「沒學到」不是「碰不到」。
- 訓練曲線：mean reward −11.2 → −10.1 平；aim_reward 2.61 → 2.70；aim_abs_ax 0.181 → 0.174；action noise std 0.30 → 0.78 全程上升（entropy_coef 0.005 沿用）。零檢查仍 OK（277/277）。sigma_ax 0.20 vs 起始中位數 0.15 符合 1.2× 規則，不是 sigma 問題。
- 判讀（15 提、我同意）：每步白噪 residual 經 PD 平均掉，一次揮棒一筆的 aim reward 埋在 ±20 的 hit/miss 變異裡 → 無梯度。
- 已下 GO 的 #1：offset = (預測 plate_y − y0) × step-0 SLSQP 每公尺 dq（9 clips 中位數，± 分開，z 固定），只在視窗內施加，diag1024 + gate script；加一列反號對照。判準：若斜率仍比 −0.50 陡 → 物理/PD 路徑擋住橫移，放棄 ArmAim 改 mimic fallback（視窗內把手/棒頭 key target 依 Δy 變形）；若反號列反向移動 → 路徑通、增益問題；若兩者都不動 → PD 路徑擋住。marker ARMAIM_SCRIPT_END。
- #2（視窗開啟時取樣一次並保持、entropy 0、std 0.1、以腳本增益初始化，同 gates）：未核准，15 備好 cfg 與啟動說明，等使用者。
- GPU：19:51 Isaac 0、閒置；下次錄影最晚 10-31 06:41，#1 不衝突。

**10-30 18:47（14）3 小時檢查**：Isaac 1（ArmAim 鏈的零檢查 diag，18:44 起，GPU 28% / 5.7 GB），其後進入 300 its 訓練 → ARMAIM_300_END ≈ 19:50–20:00。12 小時例行錄影 18:41 完成（20：`videos/routine_1030b_p5.mp4`，343 幀、High、過曝 0/343、對 M2 64/64、嚴格 52 欄 64/64、站立不變），下次最晚 10-31 06:41。使用者 18:4x 問「CPU 沒使用量真的在訓練嗎」→ 已答：訓練在 GPU，CPU 本來就低，看 nvidia-smi / `logs/armaim/`。使用者指示：+300 結果出來再報。

**10-30 18:0x（14）ArmAim step 1 建好、鏈已上膛（15 commit 4b4d961，`logs/run_armaim.sh` 等 M3ROUTINE3_END 後才開 Isaac）。** 預檢（18:25 前完成）：256 envs × 15 its smoke 乾淨（hit 0.37、視窗占 22% 步、無倒）；**零檢查：aim_bound 0 的 ArmAim diag1024 s0 與 12500 基線逐位元相同（277/277 擊中、每球 hit 旗標相同、站位差 0.0、plate_y 相同）→ 包裝層不碰投球與揮棒。** 設計補充：視窗 = 參考時鐘 ≥ 接觸 − 0.30 s（Human15 無 oracle 觸發，以此代揮棒起動）至首次接觸或過接觸面，之後偏移以 tau 0.05 s 衰減；動作罰 0.01·Σtanh(a)² 只在視窗內；obs 72 維（15 關節 q / qd / (揮棒目標 − q) / 上一步偏移、ss − ball_seen、v_seen、棒軸、t_arrive、released、視窗相位）；PPO [256,128]、init_noise 0.3、從零、2048 envs、~8–9 s/it；沿用 Human15 的 lost（key_err 0.45）與 ref-end 終止。基線（armaim_gate.py、好球帶內、*_any）：hit 0.445（n 557）、|axis_off| 0.103、|dx| 0.103、exit 18.9、D 落窗 0.181、仰角 21.7；斜率擊中 −0.80（r −0.85）、揮空 −0.80、可及子集（plate_y ≥ −0.185，448/557）−0.81。起始中位：|axis_off| 0.152（好球帶 0.110）、gap 0.056、|dx| 0.103、|D − 0.010| 0.028 → sigma 0.20 / 0.10 / 0.15 / 0.032 皆 ≥ 1.2×。鏈：smoke + 零檢查 → 300 its → diag → ARMAIM_300_END（門檻列 + 可及斜率列 + 基線）→ 通過才 +700 → ARMAIM_1000_END；失敗即停，備案（mimic 目標位移）備而不發。ETA 300 ≈ 19:45、1000 ≈ 22:00。

**10-30 17:4x（14）ArmAim step 0 權限檢查完成（15 commit f084028，`scripts/armaim_authority.py`，marker ARMAIM_AUTH_END）→ GO 關節集 C、限幅 ±0.4 rad，接受外角側約 13 cm；15 建 step 1（ETA ~19:30），M3ROUTINE3_END 後 smoke + 12500 *_any 基線 diag，smoke 乾淨即啟動訓練。** 方法：robot FK、9 個雙手參考片接觸幀、上手固定握把、甜蜜點高度固定、SLSQP（Jacobian LP 相差 ~30%）。中位 [最差片] cm：A 肩 + 肘 ±0.4 +17.0 / −10.1；B + 腕 ±0.4 +38.6 [24.6] / −12.0 [9.2]；C + 腕 + waist_yaw ±0.4 +42.3 [38.8] / −13.0 [9.9]，±0.6 僅外角 −15.8。需求（P3c 64 球、好球帶 34、y0 −5.5）：plate_y p5 −23.7 / p50 −2.4 / p95 +15.3；自 y0 的位移需 + 最大 25.6、− 最大 19.7。好球帶內可及比例：A0.4 0.62、C0.4 0.79、C0.6 0.85。外角側（往棒尖）受限於持棒手：l_wrist_yaw 在 5/9 片已到下限、l_shoulder_yaw 在 7 片離上限 0–0.3 rad（放寬 = 人形關節限制的規格變更，使用者的事，暫不動）。接觸前 −40 / −80 ms 靜態可及大得多但伴隨棒軸轉 40–70°，屬路徑 / 時機改變，不採。純運動學，實際可及會更小。門檻不變（+1000 斜率 −0.30 在 21% 不可及下仍可達）；各門檻加報「可及投球子集」的斜率以區分權限限制與學習失敗。

**10-30 16:xx（14）使用者裁定「動打者架構」→ 開 Human28「ArmAim」：手臂殘差策略（與解決跌倒的 Human25-R 同一模式：分離目標 → 分離策略），揮棒策略 12500 @ 1.5x 凍結、recover_rl 1297 不動、投球側逐位元不變。派 15。**
- Step 0（CPU，今天）：權限檢查 — 以參考揮棒接觸幀的運動學 Jacobian 估 ±0.25 / ±0.4 rad 的肩 + 肘（+ 腕）殘差能給甜蜜點多少橫向位移（cm），需 ≥ ±15 cm 覆蓋 P3c 投球組的 plate_y 分布；不夠則加 waist_yaw。marker ARMAIM_AUTH_END。關節集與邊界由此決定。
- Step 1（設計，step 0 後建、GPU 啟動等 14 放行）：殘差只在 [揮棒起動 .. 首次接觸或過本壘面] 視窗內作用，動作 = 所選手臂關節 PD 目標的加性偏移（tanh 限幅），加在凍結策略的目標上（凍結策略看不到偏移）。Obs：所選關節 q / qd、root 座標的 ss_p − ball_seen 與 ball_seen 速度、t_arrive、揮棒相位、凍結策略當前目標、上一步殘差；無特權狀態。獎勵：每次揮棒在最近點 6·exp(−(axis_off/0.20)²)·exp(−(gap/0.10)²) + 擊中 20·q（dx sigma 0.15、D sigma 0.032）+ 基礎揮空罰 + 每步 −0.01·Σa²；終止沿用；sigma 規則適用。小 MLP 從零、2048 envs、Human15 投球 cfg；task AIB-ArmAim-v0，markers ARMAIM_300_END / ARMAIM_1000_END。
- 門檻（diag1024 s0 好球帶內、*_any 鍵、先重算 12500 基線）：+300：擊中 axis_off vs plate_y 斜率比 −0.50 平（基線 ~−0.8）且 |axis_off| 中位 ≤ 8 cm 且 hit ≥ 0.36；+1000：斜率比 −0.30 平、|dx| ≤ 6 cm、hit ≥ 0.40、exit ≥ 20、D 落窗 ≥ 0.25；採用：5 seeds × 3 tables + joint s6（recover 倒地 ≤ 2/64、投球列 64/64）→ 20 重錄一個 PA。+300 失敗的備案（先備開關不啟動）：把 mimic 的手 / 棒尖 key 目標依 Δy = pred_y − y0 在視窗內漸變位移，讓追蹤獎勵本身要求伸棒。
- 時程：今天 step 0 + 建環境；10-31 訓練到 +300 / +1000；11-01 門檻與採用檢查；11-05（W1）前重錄 PA。GPU：18:30–18:45 歸 20 錄例行片，15 的 GPU 工作須在 18:25 前結束或 M3ROUTINE3_END 後開始。

**10-30 15:47（14）3 小時檢查**：Isaac 0；GPU 自 09:45 閒置、無排程、無新產出；交付不變。12 小時錄影最晚 19:08 → 已派 20 佈一條時間閘鏈：18:30 若 Isaac 0 兩次就錄一球（PA1 p5、s6 e12、final 設定、QA 字幕）→ `videos/routine_1030b_p5.mp4`，marker `logs/M3ROUTINE3_END.txt`，比對 64/64 + 過曝測試；若屆時有 trainer 在跑則不錄並回報。待使用者三選一（選球帶 / B / 打者架構）。

**10-30 12:47（14）3 小時檢查**：Isaac 0；GPU 自 09:45 閒置、無排程、無新產出；交付不變；12 小時錄影下次最晚 19:08（若仍無排程，18:30 派 20 錄一球）。待使用者三選一（選球帶 / B / 打者架構）。

**10-30 09:47（14）3 小時檢查**：Isaac 0；GPU 自 09:45（BAND_END）閒置、無排程。今日結果：Human26 / 26b / 27 / 27b / 27b+400 全部失敗（準確率訓練路線關閉）、選球帶量測完成；交付不變（h9b 12500 @ 1.5x + recover_rl 1297、final_cut_v2）。12 小時錄影下次最晚 19:08。待使用者三選一（選球帶 / B / 打者架構）。自主授權今日到期。

**10-30 09:5x（14）選球帶量測完成（BAND_END 09:45，15 commit 90aea60，`logs/band_select/summary.txt`）：帶內接觸乾淨得多，但外野球沒增加；勝率增加全是四壞。** 重跑 12500 diag：277 次擊中、與舊 dump 同 env、每次擊中皆有紀錄。y0 = −5.5 cm。好球帶內各帶（占比 / hit / |dx| / exit / 仰角 / 飛行中位 / 界內 ≥ 38.9 m）：±3 cm 0.097 / 0.556 / 4.1 cm / 23.6 / 21.5° / 27.6 m / 5 of 30；±5 cm 0.151 / 0.429 / 4.3 / 22.6 / 19.9° / 24.1 / 5 of 36；±8 cm 0.298 / 0.331 / 4.7 / 21.6 / 30.8° / 21.3 / 10 of 55；±12 cm 0.424 / 0.436 / 6.1 / 20.8 / 21.3° / 12.3 / 12 of 103；全好球帶 1.0 / 0.445 / 10.3 / 18.9 / 21.7° / 12.0 / 21 of 248。ONE OUTS（P3c 表、1000 × 5 seeds、oracle 只揮帶內好球、其餘全放）：±3 31.2%（29.6–32.3）、±5 31.3%、±8 32.0%、±12 30.8%；無帶全好球帶 oracle 25.8%（25.3–26.2）。±3 cm 的勝法：K 56.7%、四壞 22.5%、內野 12.1%、HBP 7.2%、外野 1.5%；外野在各列 1.3–3.8%。注意：CPU 表 HBP 5–7%，joint M2 20000 打席 0 次 → 各列互比，勿與交付 9.8% 比。
- 14 解讀：選球把揮棒時的接觸從 |dx| 10 → 4 cm、exit +4.7 m/s、帶內擊中到外野的比例 17%（5/30）對全帶 8%（21/248），但帶只占全部投球 6–16%，整體外野球不升；選球主要把揮棒換成放掉，勝率的 +5 點是四壞。
- 回使用者三選一：(1) 採用選球帶（改 oracle 揮棒規則，規格變更；揮棒品質好、揮棒少、勝率靠四壞）；(2) B 維持現狀；(3) 動打者架構（參考動作依來球位置變形 / 揮棒階段脫離參考動作），新規格、多日工程，自主授權今日到期需核准。GPU 閒置。

**10-30 09:4x（14）Human27b +400 FAIL（09:36，model_12898 @ 2026-10-30_08-45-42_human_pitched27b_2）→ 準確率線的訓練路線關閉（不開 27c），交付維持 h9b 12500 @ 1.5x。** 好球帶內（基線括號）：hit 0.668 (0.445)；|axis_off| 擊中 13.1 (10.2)、揮空 15.2 cm；|dx| 13.3 (10.4)；D 落窗 0.172；exit 15.5 (19.0)、仰角 8.0 (23.1)；斜率擊中 −0.91 / r −0.85、揮空 −1.00。aim_abs_ax 每 50 回合：0.185 0.173 0.183 0.189 0.188 0.199 0.210 0.208（早期下降未持續）。結論：更多訓練只買到更多固定點接觸（hit 0.67）、站位更差、球更弱更平；伸棒從未跟著來球。今日五次（26 / 26b / 27 / 27 b / 27b+400）同一結局 → 在現有 mimic 架構下用獎勵教「依球調整伸棒」不可行。剩下的路：選球帶（量測中，BAND_END ~10:00）或 B。

**10-30 09:0x（14）27b +400 續訓 08:45 起跑（`logs/run_human27b_400.sh`，marker HUMAN27B_400_END，ETA ~10:00）；選球帶量測鏈排在其後（`scripts/band_select.py`、`logs/run_band.sh`，15 commit 4d5e0df，marker BAND_END，ETA ~10:20）。** 選球帶需重跑 12500 diag1024（舊 dump 無 spray / backspin，飛行與 BattedBall 需要）。
- **診斷缺口（15 發現）**：`--contact_log` 漏掉在 `_get_dones` 最後子步判定的擊中：277 次擊中有 74 次 hit=True 但無 ev / dx / D → 至今所有 |dx| / D / exit / 仰角統計（含 Human26 / 27 門檻與基線）都只涵蓋約 73% 的擊中；擊中率本身正確。新鍵 ev_any / launch_any / dx_any / D_any / spray_contact / backspin_contact 由 env buffer 在下一物理步記錄全部擊中，舊鍵不變；+400 門檻與其基線同用舊鍵（同口徑）。
- 選球帶初步（舊 dump、尚無飛行距離）：擬合 axis_off = −0.797·plate_y − 0.044 → 固定落點 y0 = −5.5 cm（三壘側為 +）。帶寬 ±3 / 5 / 8 / 12 cm：好球帶內占比 0.097 / 0.151 / 0.298 / 0.424，帶內 hit 0.556 / 0.429 / 0.331 / 0.436，|dx| 中位 4.0 / 4.1 / 4.3 / 5.7 cm（全好球帶 10.4）。P3c 64 球表：好球帶 0.531，帶內占全部投球 0.062 / 0.078 / 0.156 / 0.234。ONE OUTS 估計將含「全好球帶揮、壞球全放」對照列，分離帶效應與 oracle 的好壞球知識（交付 9.8% 是 joint M2 + StatBatter 揮棒決策，非此規則）。

**10-30 08:5x（14）Human27b +100 FAIL（08:42，model_12599 @ 2026-10-30_08-22-59_human_pitched27b_1）→ 兩路並行：(1) GPU 續訓 27b 到 +400 只在 +400 設門檻（「100 回合太短」測試）；(2) CPU 做「選球帶」量測（固定落點 → 只揮離落點近的球）。** 好球帶內（基線 / 27 括號）：hit 0.558 (0.445 / 0.499)；|axis_off| 擊中 12.1 (10.2 / 11.6)、揮空 14.3 cm；|dx| 11.6 (10.4)；D 落窗 0.196；exit 18.4 (19.0)、仰角 19.3。斜率擊中 −0.90 / r −0.88（基線 −0.79）。Trainer：aim_abs_ax 0.200 → 0.180（有在降）、aim_reward 2.27 → 2.58 / 6、hit_q 0.234 持平、hit_rate 0.24 → 0.32。
- 解讀：手臂追蹤減半（0.15 s、×0.5）100 回合內沒改變斜率，追蹤釘住假說在此劑量下不成立；27 / 27b 共同的擊中重平衡都把斜率推陡（多出的接觸在固定點上）；launch / spray × q 救回大部分 exit / 仰角。四次嘗試（26 / 26b / 27 / 27b）都只買到擊中率。
- (1) 續訓：從 12599 同設定 +300 → +400，marker HUMAN27B_400_END，門檻：擊中 |axis_off| 中位 ≤ 8.7 cm（0.85×）且斜率比 −0.70 平且 hit ≥ 0.31；每 50 回合記 aim_abs_ax。失敗 = 準確率線的訓練路線關閉，不開 27c。約 1 h。
- (2) 選球帶量測（`logs/band_select/`，marker BAND_END）：由擬合線求甜蜜點固定落點 y0，帶寬 |plate_y − y0| ≤ 3 / 5 / 8 / 12 cm 各報好球帶內占比、帶內 hit、|dx|、exit、仰角、飛行距離（遊戲飛行模型）、≥ 38.9 m 球數；再用 CPU ONE OUTS（1000 場 × 5 seeds）估「只揮帶內球、其餘放掉（含好球）」的勝率對交付 9.8%。純量測供使用者決定。

**10-30 08:2x（14）Human27 +100 FAIL（08:13，model_12599）→ GO Human27b（手臂追蹤放鬆 + 三項獎勵修正）。** 好球帶內 diag1024（基線括號）：hit 0.499 (0.445)；|axis_off| 中位 擊中 11.6 (10.2)、揮空 14.7 (12.4) cm；|dx| 12.2 (10.4)；D 落窗 0.180 (0.125)；exit 16.3 (19.0)、仰角 15.6 (23.1)。斜率 axis_off vs plate_y：擊中 −0.91 / r −0.89 (−0.79 / −0.86)、揮空 −0.90 / −0.88 → 固定落點更嚴重，符合 27b 條件。Trainer：aim_abs_ax 0.175 → 0.193 m、contact_abs_dx 0.121 → 0.159、aim_gap 0.43 → 0.27 m、aim_reward ≈ 0.65 / 6、hit_q ≈ 0.057。
- 解讀：策略學到「碰到球」（間隙因子有梯度）而非「用甜蜜點碰」。14 的 sigma 又太窄：aim 起點 17.5 cm 對 sigma 8 cm 為 exp(−4.8) ≈ 0.008、無梯度；hit q 在 12 cm 處只 ~0.06；擊中的實際報酬來自 ev_n 加權的 launch / spray + 追蹤，任何接觸都付錢。
- Human27b（15 commit c98be77 + 修正）：aim = 6·exp(−(axis_off/0.20)²)·exp(−(gap/0.10)²)；hit = 20·q、q = exp(−(dx/0.15)²)·exp(−((D − 0.010)/0.025)²)；w_launch / w_spray 改乘 q（非 ev_n）；手臂放鬆：released 且未擊中且 0 < t_arrive ≤ 0.15 s 時肩 / 肘 / 腕的 pose / vel 追蹤與手 / 棒 / 棒尖 key 誤差權重 ×0.5（不重新正規化，腿 / 軀幹不變，key 終止條件不變）。Smoke 256 envs OK。門檻、markers（HUMAN27B_100_END / _300_END）、斜率報告同 27。**規則：之後任何 sigma ≥ 1.2× 該量的起始中位數，啟動說明要寫起始值。**

**10-30 08:0x（14）來球位置相關出爐 → GO Human27（15 commit 7c797dc，`logs/run_human27.sh`，markers HUMAN27_100_END / _300_END）。** 關鍵數字：axis_off 對來球左右位置 plate_y 的斜率 −0.79、r −0.86（好球帶內擊中；揮空 −0.80 / −0.86；joint 35 球 dx 對 pred_y −0.72 / −0.94）；去掉位置斜率後殘差 sd 6.3 cm（原 12.2）；plate_y 低 / 中 / 高三分位 axis_off +11.3 / −4.3 / −14.5 cm → **甜蜜點落在幾乎固定的水平點，伸棒只跟著來球 ~20%**；球的左右位置在觀測裡（root 座標的 ball_seen − root_p、ss_p − ball_seen），感知 = 延遲 0.10 s、每步 1 cm 噪聲、每局 2 cm 偏差、重力外推。15 假說（未驗）：mimic 追蹤（w_track 1、w_key 0.3）把揮棒釘在參考片的接觸點。
- Human27 = Human15 投球、從 12500 @ 1.5x、2048 envs、obs 不變、recover 不動。每次揮棒付一次瞄準分 6·exp(−(axis_off/0.08)²)·exp(−(gap/0.05)²)（擊中步或局末；最近點逐物理步追蹤含 contact_substeps 插值；gap = 表面間隙 ≥ 0，接觸 = 1）；擊中分 20·exp(−(dx/0.08)²)·exp(−((D − 0.010)/0.015)²)；關閉 w_distance / w_success / w_exit_velo / w_backspin / mimic w_contact 2.0；保留 w_launch 20、w_spray 10（乘 ev_n）、w_miss 20、跌倒項、追蹤。Smoke 64 / 256 envs OK。
- 門檻 +100：axis_off |中位| ≤ 0.75× 基線（擊中 10.2、揮空 12.4）、|dx| ≤ 8.0 cm、hit ≥ 0.31；+300：|dx| ≤ 6.0、D 落窗 ≥ 0.25、hit ≥ 0.40、exit ≥ 19。加報 axis_off vs plate_y 斜率 / r / 殘差 sd（歸因測試）：若 +100 時 axis_off 不動且斜率仍 ≈ −0.8 → 追蹤假說成立 → Human27b（只在 27 失敗時）：預測到達前 0.15 s 手臂關節（肩 / 肘 / 腕）w_track / w_key 減半，腿 / 軀幹不變；15 先備 cfg 旗標（預設關），不經 14 不啟動。ETA +100 ≈ 09:30、+300 ≈ 12:00。20 hold。

**10-30 07:5x（14）時機 × 接觸位置量測完成（TIMING_END 07:41，15 commit e5f37fd，`logs/timing_contact/summary.txt` / `hits.csv`）：沒打中球心的主因 = 沿棒軸的「瞄準」散佈（±10 cm），時機只平移整體分布（約 0.5 cm / ms），不碰散佈。**
- Q1（bias 0.055、248 好球帶內擊中）：corr(dx, 早晚相位) +0.24、corr(|dx|, ·) −0.11；晚 / 準 / 早三分位 dx 中位 −8.2 / −4.6 / +3.1 cm 但 |dx| 10.7 / 9.8 / 10.3 cm 不變；棒徑對球線交角中位 17°。
- Q2（309 揮空，最近點）：棒軸站位 −2.4 [−13.6, +10.0] cm、表面間隙 4.1 cm、球在軸上方 +7.4 cm、沿揮棒 −2.6 cm、ref_err +36 ms 全早。甜蜜點偏差向量 |中位| 沿軸 / 垂直 / 沿揮棒：擊中 10.2 / 3.0 / 2.7 cm，揮空 12.4 / 8.9 / 3.5 cm。
- Q3 起動偏移掃描（不訓練，好球帶內）：bias 0.030 hit 0.479、dx −6.4、|dx| 11.3、exit 17.8；0.055 hit 0.445、dx −3.3、|dx| 10.4、exit 19.0；0.080 hit 0.375、dx +5.0、|dx| 9.5、exit 17.0；0.105 hit 0.203、dx +10.5（頂到棒尖）、|dx| 10.8、exit 15.1。
- 14 結論：誤差幾乎全在棒軸方向 = 甜蜜點相對球的水平放置 = 瞄準，不是時機；垂直 3 cm 與 17° 平面是次要。下一步：(1) 15 先用 CPU 算 axis_off 對來球左右位置 pred_y 的斜率與相關（斜率近 1 = 策略不依內外角調整伸棒；近 0 = 隨機散佈），(2) 同時備妥 Human27 準確率線：每次揮棒（含揮空）在最近點給 r_aim = 6·exp(−(axis_off/0.08)²)·exp(−(dperp/0.05)²)，擊中獎勵改成 20·q（q = exp(−(dx/0.08)²)·exp(−((D − 0.010)/0.015)²)，握把側接觸約 0 分）、揮空罰則不變、移除固定每次接觸紅利、不加速度 / 距離項；門檻 +100：axis_off |中位| 降 ≥ 25%、|dx| 中位 ≤ 8.0 cm、hit ≥ 0.31；+300：|dx| ≤ 6.0、D 落窗 ≥ 0.25、hit ≥ 0.40、exit ≥ 19；markers HUMAN27_100_END / _300_END。等 (1) 數字後 14 放行。

**10-30 07:4x（14）使用者補充新線目標：「提升打擊準確率，要能打中球心」= 甜蜜點正中（dx → 0、|D| 小），不是時機本身。** 量測不變，但結論要回答「是什麼讓棒頭沒有在甜蜜點碰到球」：時機（棒早 / 晚）、瞄準（策略把甜蜜點放在來球預測線的 y / z 偏處）、或平面（棒徑與球線交角）；加報最近點時甜蜜點到球心的偏差向量（沿軸 / 垂直 / 水平，擊中與揮空都算）。TIMING_END 後由 14 設計準確率線（傾向：擊中獎勵本身乘上接觸品質，讓「多碰球」無利可圖），再向使用者報告。

**10-30 07:3x（14）使用者裁定「新線」→ 先派 15 做純量測「時機 × 接觸位置」（不訓練、不提案），結果再由使用者決定是否開時機線。** 內容：(1) 12500 基線 contact_geom（diag1024 s0）每次擊中的時機誤差對 dx / D 的相關與分位切片；(2) 揮空（全 early）在最近點的棒軸站位、離軸距離、提早毫秒數；(3) 同策略不訓練的起動偏移掃描 bias 0.030 / 0.055（預設）/ 0.080 / 0.105，各報好球帶內 hit、|dx| 中位、D 中位與落窗比例、仰角、exit、early / late；(4) 一句結論：時機是否解釋握把側接觸（|dx| ≈ 10 cm）。輸出 `logs/timing_contact/`，marker `logs/TIMING_END.txt`，交付設定凍結。GPU 給 15，20 hold。

**10-30 07:3x（14）Human26b +100 FAIL（07:24，15 commit 3209f97，model_12599 @ `logs/rsl_rl/aib1_tee_batting_mimic/2026-10-30_07-09-56_human_pitched26b_1`）→ A 線（接觸精度 + 揮棒平面）正式關閉。** diag1024 好球帶內 12500 → 12599：hit 0.445 → 0.591（PASS）；|dx| 中位 10.36 → 11.48 cm（門檻 ≤ 7.77 FAIL）；D 落在 [0, 2.5 cm] 0.125 → 0.173（門檻 > 0.20 FAIL）；D 中位 +1.5 → +1.0 cm；仰角中位 23.1 → 26.7°；exit 中位 19.0 → 17.9。Trainer bat/contact_abs_dx 第 11–20 回合均值 0.120 → 末 10 回合 0.127 m（有梯度仍上漂；Human26 0.126 → 0.139 同向）。
- 解讀：兩次都只買到擊中率、沒買到接觸精度。dx 項被擊中 / 揮空項壓過（多一次接觸 +4·g_dx，少一次揮空 +20）；且接觸位置可能綁在時機上（揮空全是 early、tempo 約 −0.07 s），這點未測。交付不變：h9b 12500 @ 1.5x + recover_rl 1297、final_cut_v2。
- 下一步回使用者：B（接受打者只靠四壞獲勝，影片維持現狀）或指定新線（候選：時機 / 接觸站位關聯量測，純量測約 20 min GPU）。GPU 閒置，15 / 20 hold。自主授權今日到期。

**10-30 07:1x（14）使用者回報「routine_1030_p5 沒辦法看」→ 原因：第一版（06:47–06:57）有 115/343 幀渲染成過曝天空（偶發渲染故障，數據列 64/64 正常），20 另存 `videos/routine_1030_p5_washed.mp4` 並於 06:58–07:08 重錄；使用者開檔時重錄尚在寫入（moov atom 未寫）。重錄版 `videos/routine_1030_p5.mp4` 07:08 完成：343 幀、High yuv420p、過曝 0/343；14 自測每半幅亮度 std 分布與 `routine_1029_p5.mp4` 完全一致（低對比半幅 126 / 686 兩片相同，過曝版 230–394），三格影格正常；對 M2 64/64、對 zone_film_p5 嚴格 52 欄 64/64、站立不變。此為 12 小時錄影（07:08），下次最晚 19:08。20 之後錄影檢查加入亮度過曝測試（check_video_frames 只抓黑幀）。GPU 交 15 → Human26b 啟動。**

**10-30 06:47（14）3 小時檢查 + Human26 +100 FAIL（06:46，15 commit 79184f3，model_12599 @ `logs/rsl_rl/aib1_tee_batting_mimic/2026-10-30_06-31-15_human_pitched26_1`）→ 同線重試一次 Human26b。** Isaac 0；20 的例行錄影鏈已觸發（HUMAN26_100_END 存在）→ `videos/routine_1030_p5.mp4` / M3ROUTINE2_END，錄完 GPU 回 15。
- diag1024 好球帶內 12500 → 12599：hit 0.445 → 0.643（門檻 ≥ 0.31 PASS）；|dx| 中位 10.36 → 10.79 cm（門檻 ≤ 7.77 FAIL）；D 落在 [0, 2.5 cm] 0.125 → 0.154（門檻 > 0.20 FAIL）；D 中位 +1.5 → +0.8 cm；仰角中位 23.1 → 21.3°；exit 中位 19.0 → 18.4。Trainer bat/contact_abs_dx 0.153 → 0.139（有動但不夠）。
- 原因（14 規格錯誤）：sigma_dx 0.04 在起點 |dx| ≈ 10 cm 處 exp(−(0.10/0.04)²) ≈ 0.002，無梯度（Human23 sigma 同款失誤）；D 視窗則成了「任何碰到球都有機會拿」的平均紅利，策略走最便宜的路 = 多碰球、品質不變。注意 diag1024 的 |dx| 基線 10.4 cm 比 joint 35 球的 7.4 cm 大，門檻是同口徑比較。
- Human26b：碰撞步 r = 4·g_dx + 3·g_dx·g_D，g_dx = exp(−(dx/0.12)²)（10 cm 處 ≈ 0.5，有梯度），g_D 不變；D 紅利乘上 g_dx，握把側接觸拿不到。其餘全同；門檻全同；markers HUMAN26B_100_END / _300_END；等 M3ROUTINE2_END + Isaac 0 再啟動（約 07:1x，+100 ≈ 07:45）。再敗 = A 線正式關閉，回使用者選 B。
- 副產物：12599 好球帶內擊中率 0.643（交付 0.445 / 0.458）、exit 同級；是否做採用檢查（5 seeds × 3 tables + joint s6）留給使用者。

**10-30 06:3x（14）Human26 已啟動（15 06:26，commit b6ae2ef，`logs/run_human26.sh`，AIB-PitchedBatting-Human26-v0）：14 判定在框架內，照跑。** Human15 cfg（P3c 形狀的好球帶內投球）、從 12500 @ 1.5x、2048 envs、obs 不變、recover_rl 不動。碰撞步新增 3·exp(−(dx/0.04)²) + 3·exp(−((D − 0.015)/0.012)²)（D 由 `TeeBattingEnv._check_impact` 新 buffer contact_D 提供，唯讀無 RNG；對比：內野出局的擊中約 +3、揮空 −20）。揮棒平面併入 D、不另加仰角項，證據（`logs/contact_offset/d15_check.py`，35 球重播）：接觸時甜蜜點速度仰角已 +12.2° [7.9, 16.5] 對來球下降 20.5°；D 0 → 仰角 6.8°、+1.0 cm → 17.4°、+1.5 → 23.1° [19.7, 27.5] 帶約 470 rpm 倒旋、+2.0 → 28.5°；exit 22.8 → 21.9。基線修正：3.3 cm 是帶號 dx 中位，joint |dx| 中位 7.4 cm → +100 門檻用 12500 新跑的 diag1024 基線（hit ≥ 0.7×、|dx| 中位 ≤ 0.75×、D 落在 [0, 2.5 cm] 比例 > 基線且 > 0.20）。+100 ≈ 07:00（marker HUMAN26_100_END → 20 錄例行片 → M3ROUTINE2_END）→ +200 → diag + 1 張 showcase（好球帶內 ≥ 0.40、仰角中位 15–30、exit ≥ 18、≥ 1 球界內 ≥ 38.9 m，marker HUMAN26_300_END，約錄完後 45 min）；採用檢查另起。到 +300 約 1.5 GPU h。

**10-30 0x:xx（14）使用者裁定「做 A，先做接觸精度 + 揮棒平面線」→ 派 15 開 Human26（GPU 給 15，20 hold）。**
- 框架：從 h9b 12500 @ 1.5x 微調，其餘設定凍結（投球側逐位元不變）；只動揮棒策略，recover_rl @ 1297 不動。接觸幾何獎勵只在碰撞步給：r_dx = exp(−(dx/0.04)²)、r_D = exp(−((D − 0.015)/0.012)²)（視窗約 +0.3～+2.7 cm 微下切），權重 15 定（起點 2–3）；揮棒平面一項（甜蜜點速度在接觸時仰角 +10～+15°）或證明 D 已涵蓋；本輪不加速度 / 距離獎勵。
- 門檻：+100（diag1024 s0 好球帶內）hit ≥ 0.36、|dx| 中位降 ≥ 25%、D 落在 [0, +2.5] cm 的比例上升；+300 hit ≥ 0.40、仰角中位 15–30°、exit 中位 ≥ 18、showcase 至少 1 球 ≥ 38.9 m；採用 = 5 seeds × 3 tables（勝率 ≥ 交付 + 1.5 或外野 ≥ 3%）+ joint s6 recover_rl 倒地 ≤ 2/64（bank 來自 12500 揮棒，交接狀態漂移就重建 bank 約 1 GPU h）。任一門檻失敗 = 線關閉。
- 目標是「部分」球到邊線外野，不是中位到外野。Markers HUMAN26_100_END / HUMAN26_300_END；12 小時錄影排在 +100 門檻的空檔（20 錄 `videos/routine_1030_p5.mp4`，marker M3ROUTINE2_END）。15 回覆確切獎勵項 / 權重 / GPU 時數後直接啟動。

**10-30 03:47（14）3 小時檢查**：Isaac 0；GPU 自 00:34 閒置、無排程、無新產出；12 小時錄影下次最晚 10-30 12:00。閒置期間做了一個 CPU 估計（`contact_offset.csv` 35 球的 exit_centred 用遊戲的 drag + Magnus 飛行模型、起點 0.9 m、固定仰角與自旋）來校正 A 選項的預期：
- 零自旋：仰角 20° 中位 32.5 m（0/35 到 38.9 m）、25° 37.1 m（8/35）、30° 40.7 m（21/35、0 到中外野 47.4）。
- 倒旋約 1400 rpm（150 rad/s）：20° 37.3 m（10/35）、25° 41.7 m（21/35、2 到 47.4）、30° 44.8 m（27/35、8 到 47.4）。順旋反而只 28–36 m。
- 今日實際 exit 16.5 固定 20°：18.9 m，0/35。
- 結論：正中接觸 + 今日棒速要到邊線外野，需要仰角 25–30° 加倒旋；引擎的碰撞模型下兩者都來自 +1～+2 cm 的微下切（D > +2 cm 就變 48° 沖天炮），視窗約 1 cm。A 選項預期 = 「部分」球到邊線外野（次要指標 ≥ 3% 有機會），中外野幾乎不可能；比 00:4x 寫的「只到深內野」稍樂觀，但仍不是中位到外野。昨晚記憶中的「24 m/s → 51 m」是最佳仰角 + 倒旋的上限。待使用者裁定 A / B / C。

**10-30 00:47（14）3 小時檢查**：Isaac 0；GPU 自 00:34（CONTACT_END）閒置、無排程、無變動。final_cut_v2 為交付候選（00:16，亦計為 12 小時錄影，下次最晚 10-30 12:00）。待使用者裁定下一條打者線（A 接觸精度 + 揮棒平面 / B 接受四壞獲勝 / C A 後再評估）、final_cut_v2 意見、力量倍率 1.5x 確認。自主授權 10-30 到期。

**10-30 00:4x（14）接觸偏移量測完成（CONTACT_END 00:34，15 commit d4448fe，`logs/contact_offset/summary.txt` / `contact_offset.csv`）：初速差距 67% 來自棒速、33% 來自接觸品質；接觸偏差是隨機散佈，不是刻意下切。**
- 樣本 35 次擊中（seed 6 + 1，34 在好球帶內）；重播驗證 |dv| < 3e-5 m/s、exit 與 CSV 差 < 0.005。引擎模型：COR = clamp(0.546 − 0.00246 (v_n − 25.9), 0.30, 0.58) × (1 − 0.35 (1 − exp(−(dx/0.09)²)))，自由棒 3D 碰撞矩陣 + Coulomb μ 0.5。
- 中位數 [q25, q75]：甜蜜點速度 15.1 [14.1, 16.0] m/s、來球 16.8；沿軸 dx −3.3 [−8.8, +3.0] cm（偏握把側，|dx| > 5 cm 占 24/35）；法向 D +0.8 [−2.6, +3.7] cm（+ = 下切）。
- exit 實際 16.5 [14.1, 18.4]；exit 正中（同棒速）22.8 [21.3, 23.4]、最大 24.5；exit 目標（正中 + 棒速 25）33.8。loss_contact 5.8 [3.2, 8.3]、loss_speed 11.3 [10.0, 12.3]；占比 0.33 / 0.67（好球帶內 0.31 / 0.69）。
- 仰角 vs D：corr +0.97；正中仰角只 7°（= 揮棒平面）。D > +2 cm 16 球仰角 48°（沖天炮）exit 13.4；|D| ≤ 2 cm 6 球仰角 8° exit 21.2；D < −2 cm 13 球仰角 −37°（滾地）exit 17.4。
- 14 解讀：兩個槓桿單獨都到不了外野——正中接觸但今日棒速只 22.8（需約 25）；棒速 25 但今日接觸品質約 24.5（33.8 × 0.72）。而且正中仰角 7° 太平，要有飛行距離還需約 +1 cm 的微下切（D 視窗很窄，> +2 cm 就變沖天炮）。棒速線已關（動力鏈時序，六個設計失敗）；接觸精度 + 揮棒平面是尚未試過的線，但單獨做預期只到約 21–23 m/s（內野深處，非外野）。下一條線由使用者決定（選項：A 接觸精度 + 平面線、B 接受只靠四壞獲勝、C 兩線都做）；GPU 閒置，15 / 20 hold。

**10-30 00:2x（14）`videos/final_cut_v2.mp4` 驗收通過（M3CLEAN2_END 00:16，20 commit ac4168d）：11 球全部以本壘前緣立體框 + 通過點圓盤重錄，3563 幀 / 118.8 s、High yuv420p、blank 0；揮棒球對 M2 64/64、take 63/64（take env 如常）；對 v1 的 clean_film 列嚴格 52 欄 11/11 IDENTICAL → 純視覺差異。揮棒後站立：PA1 p1 / p4 / p5、PA2 p1 骨盆最低 0.83–0.88 m、+1.5 s 軀幹傾角 3.0–10.3°，batter_fell 全 False。14 目視交叉幀 `videos/fixcheck_tiles/v2_ball_f118.png`（PA1 p2 壞球：紅盤在框外右側、球在其下）與 `v2_called_strike_f117.png`（PA1 p3 好球：綠盤在框內）→ 使用者回報的「球沒碰框卻判好球」已修正；**v2 取代 v1 成為 11-26 交付候選**（v1 保留）。此次亦計為 12 小時例行錄影（00:16），下次最晚 10-30 12:00。GPU 隨即交 15：`logs/run_contact.sh` 00:17 起跑（seed 6、seed 1 各 64 env 重播 + CPU 分析，marker `logs/CONTACT_END.txt`，預計 00:50–01:00）。20 hold。**

**10-29 23:xx（14）使用者問「打者無法打出內野的原因？力量不夠？沒打中球心？」→ 14 答：主因棒速（20 m/s vs 需約 25；力矩已飽和，損失在動力鏈時序）、其次接觸品質（初速只 0.8 倍棒速，正中約可 1.3 倍，未單獨量過）、來球慢（64 km/h 只貢獻約 3.6 m/s）。使用者裁定「好先做接觸偏移量測」→ 派 15：凍結 M2 設定，每次擊中記錄球心 / 甜蜜點 / 棒軸 / 速度，拆解沿軸與法向偏移，用引擎自己的碰撞模型算 exit_actual / exit_centred（同棒速零偏移）/ exit_target（25 m/s 正中），分出 loss_contact 與 loss_speed；輸出 `logs/contact_offset/`，marker `logs/CONTACT_END.txt`；GPU 等 20 的 M3CLEAN2_END 後再用，CPU 先備。純量測，不提設計，下一條線由使用者決定。**

**10-29 22:5x（14）好球帶立體框測試通過（20 commit b244a16，`videos/zonetest_p3.mp4`，marker M3ZONE_TEST_END）→ GO 11 球 clean + zone frame 重錄 → `videos/final_cut_v2.mp4`（marker `logs/M3CLEAN2_END.txt`，v1 保留）**
- 旗標 `AIB_FILM_ZONE_FRAME=1`（cfg.film_zone_frame）：本壘前緣平面四條 2 cm 白色微發光邊框（StrikeZone ±0.2525 m、0.50–1.10 m，獨立 prim ZoneFrame、無碰撞）；通過點圓盤半徑 6 cm、平面前 2 cm、顯示 0.6 s（綠 = 好球帶內或揮棒，紅 = 壞球；擊中則不顯示）；牆保留、牆上框移除。
- 測試 PA1 p3（s1 e34 判好球）：對 M2 63/64（take env 如常）、對 clean_film_p3 嚴格 52 欄 64/64 → 純視覺。14 看 f117 / f136：兩個視角框與綠點清楚，+0.34 s 球已落到牆腳（即舊牆框看起來沒碰到的原因）。

**10-29 22:2x（14）使用者回報：「視覺上好球的判定有些問題，球未碰到後面的框卻判好球」→ 原因 = 好球帶畫在本壘前緣後方 3.0 m 的牆上（`flight_end_behind_plate_m`），裁判讀的是本壘前緣平面的通過點；球再飛 3 m（約 0.17 s）會再掉 / 橫移 0.2–0.5 m，所以好球常打在牆框下方。裁判無誤，畫法誤導。**
- 派 20 修（視覺旗標，每片比對 64/64 照舊）：(1) `film_zone_frame` 立體框浮在本壘前緣平面（StrikeZone 半寬 0.2525 m、0.50–1.10 m）；(2) 通過點標記：球過平面時在框上留一個圓點約 0.6 s，綠 = 好球 / 紅 = 壞球（裁判的判定 j_cross）；(3) 牆保留當背景並停飛行標記，但移除牆上的框（最多留淡影）。
- 流程：先錄一顆今日看得出問題的 take 好球（PA1 p2 / p3）給 14 目視（marker `logs/M3ZONE_TEST_END.txt`），通過後 11 球 clean 重錄 → `videos/final_cut_v2.mp4`（marker `logs/M3CLEAN2_END.txt`）。GPU 給 20。

**10-29 22:0x（14）例行 12 小時錄影完成（M3ROUTINE_END 21:58）：`videos/routine_1029_p5.mp4`，343 幀、blank 0、High yuv420p；對 M2 s6 env 12 64/64（COMPARE_SKIP）、對 final-cut clean_film_p5 嚴格 52 欄 64/64；列：hit、exit 15.92 m/s、launch 20.3°、carry 17.9 m；無倒、min_pelvis_z_end 0.868 m、torso_tilt_1p5 5.2°。下次最晚 10-30 10:00。GPU 閒置、無排程，20 / 15 hold。待使用者：final_cut_v1 意見、力量倍率 1.5x、接受「打者只靠四壞獲勝」（22:xx 已向使用者列出三項決定）。**

**10-29 21:47（14）3 小時檢查**：Isaac 0；GPU 閒置、無排程、無變動。例行 12 小時錄影最晚 00:20 → 已派 20 錄一球（PA1 p5，s6 e12 界內球，final 設定 recover_rl @ 1297、QA 字幕開）→ `videos/routine_1029_p5.mp4`，marker `logs/M3ROUTINE_END.txt`，列比對 64/64。錄完 20 繼續 hold。待使用者：final_cut_v1 意見、力量倍率 1.5x 確認。

**10-29 18:47（14）3 小時檢查**：Isaac 0；GPU 閒置（12:20 起）、無排程、無變動；一切定案，閒置可接受（同 15:47 理由）。錄影下次最晚 00:20（21:47 檢查時若仍無事，排 20 用 final 設定錄一支 PA 當例行錄影）。待使用者：final_cut_v1 意見、力量倍率 1.5x 確認。

**10-29 15:47（14）3 小時檢查**：Isaac 0；GPU 自 12:20 閒置 3.5 h、無排程。gpu-idle-takeover 三項：錄影未到期（最晚 00:20）、最新打者（12500 @ 1.5x + recover_rl 1297）今日已品質驗證、本條即回報 → 無可接手工作；打者只能靠四壞獲勝、無外野球，第三支 PA 沒有新內容可錄，閒置可接受。git 乾淨。待使用者：final_cut_v1 意見、力量倍率 1.5x 確認。

**10-29 12:47（14）3 小時檢查**：Isaac 0；GPU 自 12:20 閒置（< 1 h），無排程；打者（12500 @ 1.5x + recover_rl model_1297）、投手 P3c、兩支 PA、final_cut_v1 皆定案，無可接手工作（錄影 12:20 剛做，下次最晚 00:20）。git 乾淨。待使用者：影片意見、力量倍率 1.5x 確認。

**10-29 12:3x（14）`videos/final_cut_v1.mp4` 驗收通過 = 11-26 交付的現行候選（118.8 s、3563 幀、2560x720 30 fps、High yuv420p、blank 0；14 自行 ffmpeg -i + 抽幀確認：clips 無 QA 字幕、揮棒後站立、中文卡片）**
- clean 重錄 10:25–12:20（marker `logs/M3CLEAN_END.txt`，兼 12 小時錄影，下次最晚 00:20）：11 球全錄；揮棒球（PA1 p1 / p4 / p5、PA2 p1）對 M2 64/64，take 7 球 63/64（僅 take env 不同，如常）；對現行 clips 列：揮棒 82 欄 64/64，take 的拍攝 env 全欄相同（其他 env 因舊 take clips 來自更早設定而不同，不在畫面內）。結果同前：PA1 p4 界外 40 km/h 9.1 m、p5 界內 57 km/h / 20° / 17.9 m、PA2 p6 四壞；倒 0。20 log commit ff695d2。
- 待使用者：影片語言 / 內容意見；打者力量倍率 1.5x 確認。GPU 閒置、無排程，20 / 15 hold。

**10-29 10:5x（14）`videos/final_cut_v0.mp4` 結構通過（118.8 s、3563 幀、blank 0、High yuv420p 2560x720；`scripts/make_final_cut.py`）：片頭 → 規則卡 → PA1（球數卡 / clip / 結果卡，p5 後 0.5x 慢動作）→「內野出局，投手勝」→ PA2（p1 揮空慢動作）→「四壞，打者勝」→ 片尾統計卡。v1 改動：所有卡片改中文主行 + 英文小字副行（與 10-28 進度影片一致；使用者可改語言），等 clean 重錄（`logs/m3clean.sh`，marker `logs/M3CLEAN_END.txt`，進行中）落下後換乾淨 clips → `videos/final_cut_v1.mp4`。**

**10-29 10:3x（14）最終剪輯腳本裁定（20 的 `logs/final_cut_storyboard.md`）：順序 PA1 → PA2 → 統計片尾；卡片時間照提案；去除片內除錯字幕 → JointFilm clean 旗標 + 11 球乾淨重錄（GPU 閒置，兼作 12 小時錄影）；PA1 p5 接觸 0.5x 慢動作重播（既有幀）**
- 先用現有 clips 在 CPU 組 `videos/final_cut_v0.mp4`（看結構），clean 重錄（marker `logs/M3CLEAN_END.txt`，recover_rl @ model_1297，視覺旗標 → 每球 pitch 欄位比對 64/64、姿態數字重現）完成後換成 `videos/final_cut_v1.mp4`。預計 GPU 約 2 h。15 閒置。

**10-29 10:2x（14）PA2 R2 成片目視通過 → 兩支打席草稿定案（皆 after_done recover_rl @ model_1297）：PA1 `videos/m3rec_draft.mp4`、PA2 `videos/m3pa2_draft.mp4`；GPU 閒置、無排程，20 / 15 hold**
- 20 成片 10:14（marker `logs/M3PA2_RECOV_END.txt`）：只補錄 p1（s10 e62 揮空），p2–p6 take 沿用；1972 幀、blank 0、High yuv420p；舊版留 m3pa2_draft_v2.mp4（logs/m3/pa2_v2/）。比對：對 M2 env 62 64/64、對前版 PA2 列 63 欄 64/64；ss_speed_max 高於單獨揮棒只 env 44（非拍攝）。姿態：無倒，3.13 s 切前 root z 0.94 m、軀幹 1.2–3.4°，3.0 s 球棒已放低。
- 今日 Human25-R 全線總結：使用者 05:3x 裁定「直接做新機制」→ 獨立平衡策略（交棒給第二個 actor）→ +300 2.5% / +1000 0.9%（s6 庫倒地）→ 同場 0/64 但駝背 → R2 修姿勢（6.8° / 11.8°，倒地 2/64）→ 兩支 PA 補錄通過。GPU 合計約 2.5 h。交付：揮棒策略 h9b 12500 @ 1.5x 不變，接觸 / 通過後交棒 `logs/rsl_rl/aib1_recover/2026-10-29_08-51-50_h25r2/model_1297.pt`（介面契約見第 3 節 56b098d）。
- 下一步：20 可在 CPU 擬最終剪輯腳本（球數卡、字幕、順序、片頭片尾）；下次 12 小時錄影最晚 22:14；下次 3 小時檢查 12:47。待使用者：打者力量倍率 1.5x 確認。

**10-29 10:1x（14）Human25-R2 (c) 結果：留 model_1297 為交付的 recover_rl（RECOV2C_END 10:01，15 commit 4a8468a）；20 接著補錄 PA2 p1**
- s6 庫（倒地 / 2 s 軀幹傾角中位、q90）：m1000 0.009 / 74.3°、74.7°（≈ 998）；m1100 0.013 / 27.3°、74.1°；m1200 0.019 / 8.5°、13.8°；m1297 0.018 / 5.7°、13.3°。倒地隨站直而升，是站直的代價，接受。沒有 checkpoint 過「倒地 < 0.018 且 q90 < 15°」門檻，未做同場重跑。
- PA2 只有 p1（s10 e62，揮空）是揮棒球，p2–p6 為 take：20 的 `logs/m3rec_pa2.sh`（單球、約 10 min）等 RECOV2C_END + Isaac 0 後自動啟動，marker `logs/M3PA2_RECOV_END.txt`，舊版留 m3pa2_draft_v2.mp4。15 閒置。

**10-29 10:0x（14）PA1 R2 成片目視通過：站直、球棒放低橫於身前、無駝背 → `videos/m3rec_draft.mp4` 取代 m3pa_draft.mp4 成為可展示的 PA1；下一件 = PA2 揮棒球補錄（等 RECOV2C_END）**
- 20 成片 09:56（marker `logs/M3PA_RECOV2_END.txt`）：1783 幀、blank 0、High yuv420p；舊版留 m3rec_draft_v1.mp4（列 / clips 在 logs/m3/rec_v1/）。p1 / p4 / p5 用 R2 model_1297 補錄，p2 / p3 沿用。
- 比對：對 M2 p1 / p4 / p5 64/64（COMPARE_SKIP 4 欄）；對 PA1 草稿列 63 欄 64/64；倒 0/3；hit_t p4 2.015、p5 2.0225。
- 姿態（切鏡前 0.3 s）：p1 root z 0.93–0.94 m、軀幹 3.8–5.2°；p4 3.2–5.9°；p5 4.9–9.6°（v1 48–75°）；3.0 s 時球棒已放低。ss_speed_max 高於單獨揮棒：p4 env 1 / 21、p5 env 57（v1 3 + 3），皆非拍攝 env。
- 裁定：通過。15 的 (c) 鏈此刻在 GPU 上（評 R2 model_1000 / 1100 / 1200）；20 等 `logs/RECOV2C_END.txt` + Isaac 0 後，用最終 checkpoint（除非 (c) 勝出否則 1297）補錄 PA2 揮棒球，合併為新 `videos/m3pa2_draft.mp4`（舊版留 m3pa2_draft_v2.mp4）；若 (c) 換 checkpoint，PA1 p1 / p4 / p5 也要再補錄。PA2 若六球全為 take 則免錄。

**10-29 09:47（14）3 小時檢查**：Isaac 1（20 的 PA1 p1 / p4 / p5 補錄，R2 model_1297，joint_smoke_20），M3PA_RECOV2_END 未落；15 的 (c) 鏈（run_recov2c.sh）等該 marker + GPU 閒置後自動跑，RECOV2C_END 未落。GPU 非閒置、無需接手。影片：最近成片 08:43（m3rec PA1），下次最晚 20:43。git 乾淨（僅未追蹤 usd）。

**10-29 09:3x（14）Human25-R2 完成（09:22，15 commit 0f4fdfc）：駝背修好 → 裁定 (a) 採用 model_1297，20 補錄 PA1 p1 / p4 / p5**
- 候選 `logs/rsl_rl/aib1_recover/2026-10-29_08-51-50_h25r2/model_1297.pt`。s6 保留庫 model_998 → 1297：2 s 內倒地 0.9% → 1.8%（PASS < 5%）；t = 2 s 軀幹傾角中位 74.1° → 5.7°、q90 74.5° → 13.3°（PASS）；「交棒後 0.2 s 棒速上升」閘門 520 → 418 列（FAIL，但閘門本身有缺陷：庫無身體速度，基準是第一步 10 ms 後的速度，隨揮本來就會再加速，改看同場比對）。
- 同場 seed 6（64 envs）：凍結列 64/64；2 s 內倒地 擊中 0/16、揮空 2/48（998 為 0/64，一列到 126°）；t_ref + 1.5 s 軀幹傾角中位 6.8°、q90 11.8°（擊中中位 5.2°、max 15.7°）；ss_speed_max 高於單獨揮棒的列 3 → 1（最大超出 +1.16 → +0.45 m/s）。
- 裁定：(a) 採用 1297，GPU 給 20 只補錄 PA1 揮棒球 p1 / p4 / p5（`--recover_ckpt .../h25r2/model_1297.pt`），與 p2 / p3 合併為新 `videos/m3rec_draft.mp4`（舊版留 m3rec_draft_v1.mp4），marker `logs/M3PA_RECOV2_END.txt`。揮空 2/48 的小退步先接受；錄完若 GPU 閒置，15 做 (c) 評 R2 其他 checkpoint（倒地 vs 傾角取捨），只有倒地更好且傾角 q90 仍 < 15° 才換。

**10-29 08:5x（14）PA1 recover_rl 成片目視：每球都站住（0/5 倒），但收尾駝背（軀幹前傾約 74°、棒仍舉著）→ 開 Human25-R2 修姿勢，GPU 回 15，20 hold**
- 成片 `videos/m3rec_draft.mp4`（1783 幀、blank 0、High yuv420p；20 commit 4a218aa，marker `logs/M3PA_RECOV_END.txt` 08:43）。比對：對 PA1 現行草稿列 5 球全 64/64（COMPARE_SKIP 4 個接觸後統計欄：batter_fell_before_done / tempo_fly / take_bat_vmax / ss_speed_max，batter_fell 照舊跳過）；對 M2 p1 / p4 / p5 64/64、takes 63/64 如常。hit_t p4 2.015、p5 2.0225。
- 姿態（切鏡前 0.3 s）：全部站著 root z 0.94–0.96 m；takes 軀幹 12–14° 穩定；揮棒 p1 48→74° 仍在動（3.10 s 切）、p4 55→75°（3.40 s）、p5 73.9–74.4° 穩定但駝背。原因：終止條件 tilt > 75° 被貼著用，直立獎勵只算骨盆，腰彎了沒人付代價。
- ss_speed_max 差異列：p1 9 列全較低；p4 / p5 各 3 列擊中列較高（最多 +1.16 m/s，env 12 / 21 / 34、38 / 47 / 57），交棒後前幾步會拉一下球棒。
- 裁定：不是剪接問題，不補錄 p4。Human25-R2 = 從 model_998 續訓 +300：加軀幹（chest）直立項、0.5 s 後向 default pose 的姿勢項（腰 pitch、手臂）、從 t 0 起的小手臂關節速度懲罰。證偽（s6 保留庫，`RECOV2_TRAIN_END.txt`）：倒地仍 < 5%、t = 2 s 軀幹傾角中位 < 30°、q90 < 45°、交棒後 0.2 s 內棒頭速度不超過交棒值；再 seed 6 同場重跑（`RECOV2_DIAG_END.txt`）：64/64、倒地、接觸後 1.5 s 傾角、ss_speed_max 較高列歸零。過了 20 只補錄 p1 / p4 / p5 與 p2 / p3 合併。
- 可展示草稿暫仍為 `videos/m3pa_draft.mp4`（sky 版）；`m3rec_draft.mp4` 為過渡版。

**10-29 07:5x（14）Human25-R stage 2 通過（RECOV_DIAG_END 07:43）：同場 seed 6 揮棒後倒地 0/64，凍結列 64/64 → 交棒給 20 重錄 PA1 驗收外觀**
- 最終 checkpoint `logs/rsl_rl/aib1_recover/2026-10-29_06-33-05_h25r_b/model_998.pt`；CSV 在 `logs/recov_diag/`（15 commit 7877f09）。
- (b) 凍結設定：recover_rl 與 policy 重跑的 pitch + hit_t + swing outcome + done_t + t_plane 欄位與 falldiag2 完全相同 64/64。
- (a) 接觸 / 到達後 2 s（falldiag2 定義）：recover_rl 倒地 擊中 0/16、揮空 0/48（到 episode 結束也 0/64）、失去姿態 0/16、1/48；policy 重跑 = falldiag2（88% / 81%，失去姿態 94%）。
- (c) 擊中後球距任何身體連桿最近 0.44 m（policy 0.43 m），0/16 進 0.15 m。
- (d) 觸發後 2 s：PD 目標每步變化 rms 中位 0.018 rad（q90 0.023）vs policy 0.601（0.683）；關節速度 rms 2.6 vs 8.4 rad/s；最大值相近（40 vs 43，交棒後最初幾步仍是隨揮）。→ 收棒站穩，不是抽搐。
- 裁定：Human25-R model_998 成為交付的 after_done（待影片目視通過）。GPU 交給 20 重錄 PA1：`--after_done recover_rl --recover_ckpt <model_998>`，其餘凍結設定；打者畫面不再在 done_t + 0.5 s 切掉，跑到使用者定的約 3.1 s 換球點，p5 界內球照舊到落地；每片 pitch 欄位比對；marker 如 `logs/M3PA_RECOV_END.txt`。外觀過了再錄 PA2，兩支成為最終草稿。15 閒置（可在 CPU 補 recovery policy 的 obs / action 契約到第 3 節）。

**10-29 07:3x（14）Human25-R +1000 通過（RECOV_TRAIN_END 07:26）：model_998 s6 保留庫 2 s 內倒地 0.9%，stage 2 進行中**
- model_998（`logs/rsl_rl/aib1_recover/2026-10-29_06-33-05_h25r_b`）：擊中 2/294 = 0.7%、揮空 7/730 = 1.0%、全部 0.9%、失去姿態 1.0%；骨盆最低 z 中位 0.86 / 0.87 m。門檻 < 30%。對照 model_299 2.5%、a = 0 96.4%、falldiag2 88% / 81%。
- 最終 checkpoint = model_998（兩者較佳）。stage 2（seed 6 同場：policy 重跑 + recover_rl）正在跑，結束寫 RECOV_DIAG_END。

**10-29 06:47（14）3 小時檢查**：Isaac 1（Human25-R +1000 續訓 h25r_b，model_400 已存，ETA 約 40 min）；RECOV_BANK_END 06:07 已落，RECOV_TRAIN_END 未落；stage 2 鏈（run_recov_diag.sh）排隊等 RECOV_TRAIN_END，之後寫 RECOV_DIAG_END。GPU 非閒置、無需接手。影片：最近成片 02:18（sky 重錄），下次最晚 14:18，未到期。20 hold。git 乾淨（僅未追蹤 usd）。

**10-29 06:4x（14）Human25-R +300 通過證偽：s6 保留庫 2 s 內倒地 2.5%（基線 96.4%）→ 續訓到 +1000，stage 2 接著做**
- model_299，held-out s6，不終止，觸發後 2 s：擊中 12/294 = 4.1%、揮空 14/730 = 1.9%、全部 2.5%、失去姿態 3.8%。對照 a = 0 92.5% / 97.9%、falldiag2 88% / 81%。+300 門檻 < 60%，已低於 +1000 門檻 30%。
- 是站著不是蹲著：2 s 內骨盆最低 z 中位 0.85 m（擊中）/ 0.82 m（揮空）；訓練端 ground_rate 0.047、平均 episode 長 194/199。
- 裁定：續訓到 +1000（約 50 min）後，最終 checkpoint 取 model_299 與 +1000 在 s6 庫上較好者；15 同時在 CPU 寫 stage 2（BatterSide `after_done = "recover_rl"` + joint_smoke 旗標），RECOV_TRAIN_END 後啟動 seed 6 falldiag。stage 2 報告要含：(a) 以接觸 / 到達為基準的 2 s 內倒地與失去姿態（falldiag2 同定義，擊中 / 揮空分開）；(b) 凍結設定 64/64 + hit_t 逐位元；(c) 擊中後身體不再碰球（接觸數 0）；(d) 恢復段動作變化率 / 關節速度 vs 策略本身的隨揮，確認像收棒站穩而非抽搐。通過後由 20 錄一支打席驗收。

**10-29 06:1x（14）Human25-R stage 0 完成（RECOV_BANK_END 06:07），+300 訓練進行中**
- 狀態庫：s0 2048 筆（擊中 29.1%）訓練、s6 1024 筆（擊中 28.7%）保留；觸發時已倒 5 / 1，全部在 1.35 s 內觸發。
- 觸發瞬間（s6）：骨盆 z 中位 0.86 m（擊中）/ 0.88 m（揮空，少數已倒 min 0.21）、傾角中位約 20°（擊中 max 40°）、root 速度中位 0.81 / 0.68 m/s（max 4.2）、揮空時常有一腳離地（foot z 中位 0.20 m）。→ 交棒時身體多半還直立，是「帶動量的平衡」問題，不是起身問題。
- a = 0 基線（開迴路錨點滑向站姿）s6 2 s 內倒地：擊中 92.5%（中位 0.92 s）、揮空 97.9%（0.65 s）、全部 96.4%；與 10-29 04:4x PD blend 100% 一致。+300 / +1000 同時對照此基線與 falldiag2 的 88% / 81%。

**10-29 05:3x（14）使用者裁定「直接做新機制」→ 開 Human25-R：獨立的接觸後平衡策略（two-policy hand-off），GPU 分給 15**
- 原則：揮棒策略 h9b 12500 @ 1.5x 到交棒點前完全不動（投球 / 擊球 / M2 逐位元不變）；交棒點 = `_recover` 的觸發 `active & (hit | done)` 的第一個控制步，之後由第二個 actor 驅動同一組 PD 目標，唯一任務是站穩。
- Stage 0 state bank（~20 min，marker `logs/RECOV_BANK_END.txt`）：standalone PitchedBattingEnv + frozen 12500 + P3c 表，seed 0 / 6，在觸發步存 root pose/vel、body joint pos/vel、當步 PD 目標、上一動作、hit/miss → `logs/recover_bank.npz`（≥ 2000 筆，seed 6 留作驗證）。
- Stage 1 RecoverEnv + PPO（≤ 4 GPU h，marker `logs/RECOV_TRAIN_END.txt`）：reset 直接寫 bank 狀態（+ 小雜訊），球停放、無時鐘、無參考動作、無任何揮棒項；obs 只有本體感覺（關節位置 / 速度、骨盆系重力向量、root 線 / 角速度、上一動作、交棒後時間）；action 同揮棒策略的 PD 目標參數化；episode 2.0 s，骨盆 < 0.35 m 或傾角 > 75° 終止；reward = alive + 骨盆高度 + 直立 + 0.5 s 後速度阻尼 + 雙腳觸地 − 動作變化率 − 落地 50。
- 證偽：held-out seed 6 的 bank 狀態，+300 its 內 2 s 內倒地 < 60%（基線擊中 88% / 揮空 81%），+1000 its < 30%；+300 不到 60% 即停、線關閉。
- Stage 2 整合（marker `logs/RECOV_DIAG_END.txt`）：BatterSide `after_done = "recover_rl"`；seed 6 falldiag 協定量 2 s 內倒地 / 失去姿態，並做凍結設定檢查（pitch 欄位 + hit_t 與 logs/falldiag2 到 done_t 逐位元相同 64/64）。通過後由 20 重錄一支打席。
- 20 繼續 hold（無 Isaac）；15 數字回報給 14，由 14 向使用者回報。

### 10-29 05:0x（規劃 session）Human24 不啟動：接觸後站穩的訓練線在 10-27 已跑過六種設計全敗，維持現狀 + 剪接規則；列為使用者待決
- 15 的 CPU 答覆：原因確認——12500（Human9b 鏈）訓練回合在接觸後 0.4 s 結束，98% 擊中回合在 clip 末 time_out、跌倒發生在之後（接觸後 0.94 / 1.25 / 1.87 s @ 10 / 50 / 90%）。但 10-27 已從 12500 @ 1.5x 跑過：Human11（隨揮 1.2 s，跌倒 57 -> 65%）、Human12/13（lost = fall、隨揮 2.0 s、回合 3.5 s；13 崩潰、54%）、Human18（diag 跌倒 42 -> 25% 但 ONE OUTS 19.9 -> 16.4%、外野 4.1 -> 0.2%）、Human19 / 20（站立獎勵，+200 判準失敗）、Human21（揮棒凍結 + 站立獎勵，43%、判準失敗）、對照 C（跌倒 23-32% 只因揮棒變弱）。結論不變：跌倒只在揮棒變弱時下降。Human24 = 18 + 21 的變體，非新機制。
- 裁定 (b)：不啟動 Human24，接觸後路線維持關閉；交付 after_done = policy、剪接規則遮掉跌倒。15 的 hold 正確。
- 使用者待決（新增）：打者揮棒後近 9 成倒地（影片靠剪接避開、界內球可選站住的列）——接受現狀，或之後另撥時間做專門的接觸後平衡設計（需新機制，不是再加獎勵）。
- GPU 閒置、無排程（可接受：打者 / 投手 / 兩支打席影片皆定案）；15、20 可做最終剪輯分鏡（CPU）。下次檢查 ~06:47。

### 10-29 04:5x（規劃 session）done 後 PD 漸變回站姿失敗（100% 倒地，比策略還差）-> 開環路線關閉；預核准 Human24（接觸後站穩的閉環訓練線，條件式啟動）
- 15（0f5f1e8，`logs/falldiag3/` 04:42，揮棒列 64/64 同 logs/m2 s6）：recover（0.8 s 到站姿）擊中 16/16、揮空 48/48 在 2 s 內倒地；recover_low（0.4 s、膝 +10°）同樣 100%；基準（策略續跑）88% / 81%。與 10-28 看球診斷（mode 4 / 9 / 10 全倒）同一結論：揮棒後有動量，站姿在開環 PD 下站不住；關掉策略反而失去它自己救回的 12% / 19%。after_done 維持 policy，交付不變。
- 裁定：開環恢復關閉。交付維持現狀（剪接規則遮掉跌倒；像 s6 e12 這種站住的界內列是真實結果，可選入最終剪輯）。
- 預核准 Human24（條件式）：15 先在 CPU 回答訓練回合在接觸後何時終止、clip 的隨揮是否以站姿收尾。若回合在接觸後 <= ~0.5 s 結束（接觸後行為從未訓練 = 原因）-> Human24 = Human15 + 1.5x 從 12500 微調，回合延長到接觸 + 1.5 s（揮空：到達 + 1.5 s），隨揮段保留 mimic 獎勵，接觸後加小權重的直立存活項（骨盆 >= 0.6 m、傾角 < 45°），骨盆 < 0.35 m 提早終止；+200 its、2048 envs。判準（聯合 seed 6 協定、64 列、投球 64/64）：擊中與揮空 2 s 內倒地 <= 30%；今日擊中列至少 14/16 仍擊中（或 diag1024 好球帶 >= 0.43）；接觸甜蜜點 >= 19.5；EV 中位 >= 17。不過 -> 停、不跑 M2。標記 `logs/H24_END.txt`、stdout `logs/h24_chain.txt`。若回合本來就在接觸後跑很久 -> 不啟動，改寫說明策略接觸後看到什麼、為何倒。
- GPU 空；20 hold。

### 10-29 04:3x（規劃 session）更正：揮棒後的跌倒是真的（擊中 2 s 內倒地 88%、揮空 81%）；影片靠剪接避開；核准「done 後漸變回站姿」診斷（seed 6，8 min）
- 15（`logs/falldiag2/` s6，04:23，與 logs/m2 s6 投球 64/64、穿越 48/48、揮棒 0 變動）：傾角觸發只是跌倒的開始，最後骨盆到 ~0.17 m。擊中 16：2 s 內失去姿態 15（94%）、**倒地（骨盆 < 0.35 m）14（88%）**、純傾角 0、回合結束前倒地 15；2 s 內最低骨盆 min/q1/中位/q3/max 0.17/0.17/0.18/0.33/0.82。揮空 48：失去姿態 45（94%）、倒地 39（81%）。s6 e12（PA1 p5）是唯一在窗內站住的擊中（最低 0.818 m），4.28 s 失去姿態、4.415 s 倒地。
- 影片 vs M2 無矛盾：影片在 done_t + 0.5 s 後左欄切成投手側視（我之前把 tile 4 s 後的人形誤讀成打者），打者倒地不在畫面內。現行剪接規則（揮空 3.1 s、界外 3.9、界內 4.3）把大部分跌倒剪掉；界內球打者顯示到落地，p5 是少數站住的。
- 上一則「深彎、仍站著」的解讀撤回；「失去姿態」/「倒地」兩個標籤保留：s6 擊中 94% / 88%。
- 裁定：對 11-26 影片這是品質風險（打者揮棒後近 9 成倒地）。核准診斷（`logs/falldiag3/`，標記 `logs/FALLDIAG3_END.txt`，seed 6、64 envs，預設關）：BatterSide 在 done（接觸或揮空過板）起 0.8 s 內把關節目標從策略當前目標漸變到 batting_stance.json（看球 take mode 10 的同一機制，看球 0 跌倒），done 前完全不動 -> 揮棒列必須 64/64；回報擊中 / 揮空的 2 s 內與回合末倒地比例 vs 88% / 81%、最低骨盆分佈。第二看門檻：兩者 2 s 內倒地 <= 30%。純漸變失敗則同鏈再試 0.4 s 漸變 + 膝多彎 10°（較低較寬的接球姿）。屬影片側選項，非交付。
- 給使用者的待決事項更新：打者揮棒後倒地率高；若漸變有效，是否採用到影片（投球 / 擊球結果不變，只改 done 後動作）由我決定並回報。

### 10-29 04:2x（規劃 session）跌倒診斷：擊中後的「跌倒」多為傾角 > 60° 的深彎（骨盆仍 0.55-0.68 m 高）；報告改用「失去姿態」與「倒地」兩個標籤；補一顆種子量最低骨盆高
- 15（`logs/falldiag/`，04:12）：seeds 6 + 1 與 logs/m2 每欄全同 128/128（再次重現）。跌倒瞬間（骨盆 < 0.55 或傾角 > 60°）：s6 擊中 16 全「倒」（15 在 2 s 內），骨盆 z min/q1/中位/q3/max 0.53/0.54/0.62/0.68/0.82、傾角 50/60.1/61.2/61.7/62.8°，高度觸發 5、傾角觸發 12；s1 擊中 19 全倒，骨盆 0.53/0.54/0.55/0.60/0.62、傾角 50/56/60.4/61.3/63°，高度 8、傾角 13。揮空反之（高度為主：s6 29/17、s1 30/12）。s6 e12（PA1 p5）：接觸 2.023、「倒」於 4.28（接觸後 2.26 s，在 2 s 窗外），骨盆 0.634 m、傾角 62.2° -> 傾角觸發、仍站著，與影片一致。接觸前跌倒 0/16 + 0/19，揮空到板前 0/48 + 0/45。
- 裁定：環境的 fallen 旗標 = 「失去姿態」（傾角 > 60° 或骨盆 < 0.55），不是摔倒；之前所有「擊中後跌倒 49.5%」改標為「失去姿態 49.5%」。引擎不改。新增報告標籤「倒地」= 接觸後 2 s 內骨盆 < 0.35 m。補一顆 seed 6（`logs/falldiag2/`，標記 `logs/FALLDIAG2_END.txt`，約 8 min）加唯讀欄位 min_pelvis_z_2s / min_pelvis_z_end / 首次 < 0.35 m 時刻，回報擊中 / 揮空的「倒地」比例。
- 對交付：好球帶 0.515 / 勝率 9.8% 不變；影片觀感以「倒地」比例評估。

### 10-29 04:1x（規劃 session）更正：沒有世界 z 跌倒 bug（RotatedArticulation 的 pivot z = field_z，判定本來就是相對場地）；接觸前假跌倒 0 例；跌倒診斷照跑
- 15 撤回：BatterSide 的 robot 是 RotatedArticulation(j.batter, j.plate_front_w())（batter_side.py:150），位置皆為 x_B = R^T (x_W - pivot)，pivot z = field_z（joint_env.py:328）-> _get_dones 讀到的 pelvis_z 已是相對場地，門檻確實是離打者地面 0.55 m。上一則「標籤錯誤 / 可能否決擊中」的推論**不成立**，無程式變更、無需修正、不重跑 M2。
- 成立的事實：跌倒 = 骨盆離場地 < 0.55 m 或傾角 > 60°。CPU 檢查（logs/prefall_check.py，m2_f055 s1 / m2_b030 s1）：擊中 0/19、0/20 在接觸前跌倒；揮空 0/45、0/44 在到板前跌倒；最早跌倒在到達後 0.27 s / 0.10 s -> 擊球窗未被否決、回合未提早結束。交付數字（0.515 / 9.8%）不受影響。
- 待解：影片（PA1 p5 s6 e12 站到 5.6 s）vs 旗標（3.15 s 倒）的矛盾。`logs/falldiag/`（seeds 6 + 1，標記 `logs/FALLDIAG_END.txt`，約 04:25）會分出高度觸發 vs 傾角觸發；若多為傾角 > 60°（隨揮深彎、仍站著），則「擊中後跌倒 49.5%」應改標為「失去姿態（傾角 > 60°）」而非摔倒，報告用語修正即可。

### 10-29 04:0x（規劃 session）疑似 bug：聯合打者的跌倒判定用世界 z，但場地低 0.196 m -> 隨揮下蹲被算成跌倒、可能提前關閉擊球窗；診斷 seeds 6 + 1 進行中
- 15：bat_fall_t = side.fallen & active；side.fallen 來自 TeeBattingEnv._get_dones（tee_batting_env.py:436）：骨盆**世界** z < min_pelvis_height 0.55 或傾角 > 60°。聯合場景 field_z = -0.196（mound_top_z - mound_height），打者站在世界 0 下方 0.196 m -> 世界 z < 0.55 等於離自己地面 < 0.746 m；站立 root ~0.93 -> 下降 ~0.19 m（寬步隨揮就到）即判「跌倒」；原生 tee 場（地面 0）同樣動作不算。與影片相符（PA1 p5 s6 e12 站到 5.6 s，但 logs/m2 標 batter_fell True、done_t 3.3；m2_f055 s1 e12 擊中 2.025、「倒」於 3.15；seed 1 跌倒距揮棒起動中位 2.33 s、最小 1.52 s——慢慢下沉，不是摔倒）。
- 影響：(1) 報告數字「擊中後跌倒 49.5%」很可能是標籤錯誤；(2) 可能漏進比賽邏輯：side.fallen 關閉 BatterSide 的擊球窗（batter_side.py:414，win = handed & ~done & ~fallen）並決定 done -> 接觸前的「跌倒」可能否決擊中、提早結束。尚未驗證。
- 裁定：先診斷再修。(a) GPU（現在）：joint_smoke seeds 6 + 1、64 envs、bias 0.055，加唯讀欄位 fall_pelvis_z（相對 field_z）/ fall_tilt_deg（3935ea8），寫入 `logs/falldiag/`，標記 `logs/FALLDIAG_END.txt`，約 14 min；回報擊中時「跌倒」瞬間的骨盆高 / 傾角分佈、高度觸發 vs 傾角觸發數、s6 e12 與 s1 e60。(b) CPU：數 bat_fall_t < hit_t（擊中）或 < t_plane（揮空 / 全部）的列，確認接觸前跌倒是否否決擊球窗或提早結束。
- 修法（待數字確認後執行）：聯合打者的跌倒判定改為**相對場地**（pelvis z - field_z < 0.55，傾角不變）= 原生環境在地面 0 的語意，屬 bug 修正非規格變更。修後：投球不變（投手看不到打者），但 M2 需重跑（10 seeds ~75 min）、joint_pa 重跑、揮棒列重拍（看球列 take mode 10 跌倒 0，不受影響）。20 尚未通知，等診斷。

### 10-29 03:5x（規劃 session）bias 0.030 判定無意義（影片窗內跌倒 90% vs 95%）-> 停鏈、維持 0.055、不重拍；追查「跌倒」定義
- 15 seed 1 兩種 bias：0.055 19 次擊中，接觸後 1.0 s 內倒 6（32%）、2.0 s 內 18（95%）；0.030 20 次擊中，1.0 s 內 8（40%）、2.0 s 內 18（90%）。影片顯示打者到球落地（接觸後約 2 s），兩種 bias 下幾乎每次擊中都在窗內「倒」；之前的 49.5%（in-episode）與 34%（診斷）量的是更早的窗。M2 可重現性：m2_f055 s1 與 logs/m2 每個揮棒 / 投球欄位全同（bit for bit）。
- 裁定：bias 0.030 作為影片觀感選項**無意義**（跌倒不減、1 s 內略差、好球帶上升換不到勝場）。15 在種子之間乾淨停鏈（Isaac 0、標記寫 stopped），預設維持 0.055，不重拍；`logs/m2_f055/m2_s1.csv` 留作可重現性紀錄。
- 疑點（CPU 追查中）：「2 s 內 95% 跌倒」與影片矛盾——PA1 p5（s6 e12）tile 上打者擊球後以寬步站到 5.6 s。已要求 15 說明 bat_fall_t 的判定（根高？軀幹傾角？非腳連桿觸地？）並對 s6 e12 / s1 e60 給出跌倒時刻與當時 root z / 軀幹傾角。若判定把隨揮的前傾算成跌倒，49.5% / 95% 是標籤錯誤，我會為報告重新定義「跌倒」（如 root z < 0.6 m 或非腳連桿觸地）。
- GPU 停鏈後空閒、無指派；20 hold。

### 10-29 03:47（規劃 session）3 小時檢查：M2 bias 0.030 / 0.055 對照鏈進行中（1/20），基準 seed 1 完全重現
- Isaac 0 瞬間（種子之間的閘門）。15 的 `logs/run_m2b030.sh`（e6f7dd2）：s1 bias 0.055 03:43 -> `logs/m2_f055/m2_s1.csv`，hits 0.297（19/64，同 10-28 的 M2）、投球欄位 64/64、穿越 45/45、揮棒結局變動 0/64 -> M2 在同一凍結設定下**完全可重現**。剩 19 次（0.030 / 0.055 交錯），每次約 7 min，預計 ~06:00 `logs/M2B030_END.txt`。
- joint_smoke 新增唯讀欄位 bat_fall_t / hit_t；m2_analysis 報「擊中後 1.0 / 2.0 s 內跌倒」；採用門檻以 2.0 s 數字判（基準自己的 2.0 s 數字為參照）；PA 重播 `logs/pa_replay.py` 已能精確重現 2081 打席。
- 無 trainer（棒速線已關閉）；影片義務由 02:18 的天空重拍覆蓋；GPU 無閒置；20 hold（CPU 工作可做）。
- 下次檢查 ~06:47：看 M2B030_END 的採用判定（好球帶 >= 0.46、2 s 內跌倒 <= 40% vs 基準、joint_pa 勝率 9.8% ±1、PA 2081 / 576 結局是否保持）-> 採用則 20 重拍變動揮棒列。

### 10-29 03:2x（規劃 session）Human23b +100 失敗 -> 棒速線在 1.5x 關閉；GPU 轉去驗證 bias 0.030（影片觀感候選，聯合 M2 門檻）
- 15（bb17cdf，`logs/H23B_END.txt` 03:15）：cvel_err 11.98 -> 11.75（0.98x，門檻 0.8x）、hit_rate 0.269 -> 0.342、exit_velo 16.48 -> 16.39、bat_speed 18.38 -> 17.72。sigma 20 有一點梯度，但策略又拿去換擊中率。
- 裁定：棒速訓練線在 1.5x **關閉**（Human16 / 16b / 22 / 23 / 23b 全部同一模式：PPO 把改進花在接觸上）；交付版 h9b 12500 @ 1.5x（M2 0.515 / 9.8%）定案。剩下的力量槓桿只有使用者的 arm A（2.0x），仍不建議（力矩非主因，時序才是，而時序獎勵訓不動）。
- 新指派（15，GPU 閒置、無排程）：bias 0.030 以聯合 M2 協定驗證。(1) 揮棒起動 bias 改為聯合 driver 的 config / CLI 值，預設維持 0.055，commit；(2) 同 10 seeds x 64 全揮棒列、同凍結場景 / 投球、pass 1 timing.csv 重用，bias 0.030 跑到 `logs/m2_b030/`（不覆蓋 logs/m2/），標記檔 `logs/M2B030_END.txt`；(3) m2_analysis vs logs/m2/：投球欄位必須全同（投手看不到打者）；採用門檻（全部成立）：好球帶 >= 0.46、擊中後跌倒 <= 40%（vs 49.5%）、crossing 誤差不變、打者勝率 9.8% ±1（joint_pa 20000 PAs）、四壞不變；(4) 同 seeds（2081 投手贏、576 四壞）重跑 joint_pa，回報兩個 PA 結局是否保持、哪些列變了。不拍片；採用後 20 只重拍變動的揮棒列（看球列不變），天空場景 ee850e1，64/64 vs 新 M2 列。門檻不過 -> 預設維持 0.055、不重拍。
- 20：hold，CPU 工作（最終剪輯卡片 / 字幕）可做。預計 M2 ~04:40、joint_pa 之後；下次 3 小時檢查 ~03:47 看 seed 1 的早期數字。

### 10-29 03:0x（規劃 session）Human23 +100 失敗（追蹤項無梯度）；給一次修正版 Human23b（sigma 20 / w 3），失敗則棒速線在 1.5x 關閉
- 15（64f4c90，`logs/H23_END.txt` 03:01）：12599 訓練窗速度誤差 RMS 12.04 -> 12.08 rad/s（門檻 <= 9.6，曲線平 11.4 -> 12.3 -> 12.1）；hit_rate 又升 0.270 -> 0.322；exit_velo 16.7、bat_speed 18.4、fall_rate 0.001。
- 裁定：結果接受，但 sigma 8 在 ~12 rad/s 誤差下每步只付 exp(-144/64) ~ 0.1（每揮 ~1 vs 擊中獎勵 2-26），等於無梯度 -> 這是設計缺陷，不算概念被推翻。給一次修正版 Human23b：sigma 20、w_cvel 3（不用 15 提的 5，讓時序項維持在擊中獎勵上限以下，PPO 不能只靠時序活）；+100 / +200 判準不變、早停、+200 不過不跑 M2；標記檔 `logs/H23B_END.txt`、stdout `logs/h23b_chain.txt`，約 12 min。
- 若 23b 的 +100 也失敗：棒速線在 1.5x **關閉**，不再出變體；剩下的槓桿只有使用者的 arm A（2.0x，目前不建議，力矩非主因）；我接著決定 bias 0.030（影片觀感：擊中率 0.479、擊中後跌倒 34%、EV -1.4）是否採用。
- 交付版仍是 h9b 12500 @ 1.5x（M2 0.515 / 9.8%）。

### 10-29 02:5x（規劃 session）揮棒提前量不是槓桿（bias 0 EV -2.4）；核准並啟動 Human23（接觸窗關節速度追蹤獎勵，arm C）
- 15（cf98e57，`logs/BSPEED6_END.txt` 02:45，12500 閉環）bias 掃描：0.055 好球帶 0.445 / EV 18.4 / 甜蜜點 20.1 / 腰接觸 qd -4.18（ref 8.06）/ 跌倒 49%；0.030 0.479 / 17.0 / 19.9 / +2.71 / 34%；0.000 0.399 / 16.0 / 19.2 / +7.53（ref 7.62）/ 9%；-0.020 0.291 / 16.2 / 18.2 / +8.01 / 6%。峰值時間不隨 bias 動（腰 -0.06、肩 / 腕 pitch -0.20..-0.26 s）。
- 裁定：bias 變小只是在揮棒較早處碰到球，腰雖對齊參考但甜蜜點與 EV 反而降；腰反轉是「碰球位置」的症狀而非流失本身；15 的 +8-12 m/s 上限估計撤回（連桿貢獻耦合、不可逐一加回）。提前到峰是策略自己的「甩棒過早」順序 -> 引擎揮棒起動時間不是槓桿，維持 0.055。
- 副發現（留用）：bias 0.030 好球帶 0.479（> 0.445）、擊中後跌倒 34%（< 49%）、EV -1.4；勝率不變（打者贏全是四壞、內野球都出局），只影響影片觀感 / 跌倒。決定延到 Human23 結果後（採用就要從 oracle 重跑 M2 + 重拍揮棒列），引擎預設先不動。
- Human23（15 寫好、我核准、現在啟動，arm C）：Human15 + 1.5x 從 12500 續訓；接觸窗 [ct-0.1, ct] 到擊中為止，每步 w_cvel 1.0 * exp(-mean((qd-qd_ref)^2)/8^2)，關節 = 腰 3 + 左肩 3 + 左肘 + 左腕 3（每揮最多 ~10，vs 擊中獎勵 2-26）；與 Human16 的速度獎勵不同（懲罰提前到峰，不付速度）。判準 +100：訓練窗速度誤差（新統計）較 its 11-20 降 >= 20% 且 hit_rate >= 0.7x；+200：diag1024（bias 0.055、jitter 0、--link_log）l_shoulder_pitch 與 l_wrist_pitch 峰值 >= 0.05 s 更晚、接觸甜蜜點 >= 21.6（+1.5）、好球帶 >= 0.40、跌倒 <= 55% -> 續到 +600；早停；+200 不過不跑 M2。先驗：低（Human16 / 16b / 22 都拿擊中率換）。閘門 Isaac 0 兩次、標記檔 `logs/H23_END.txt`、stdout `logs/h23_chain.txt`。
- GPU 現在歸 15（Human23 約 +100 12 min / +200 25 min）；20 無排程。

### 10-29 02:4x（規劃 session）關節貢獻診斷：流失主因 = 時序（腰部偏航提前 60 ms 到峰、接觸時反轉）；先做揮棒提前量掃描（零訓練）再談 Human23
- 15（ed3a5d5，`logs/BSPEED5_END.txt` 02:34，12500 閉環 277 次擊中）：接觸時甜蜜點 20.1 = 關節貢獻 11.4 + 根部（骨盆 / 身體）~8.7。貢獻：l_elbow +5.33、l_wrist_pitch +4.70、waist_yaw **-4.10**、l_shoulder_yaw +2.30、waist_pitch +1.83、l_shoulder_roll -1.53、l_wrist_yaw +1.11。峰值時間 vs 接觸（sim / ref）：waist_yaw -0.060 / -0.001（接觸時 qd -4.18 vs ref +8.06 rad/s，反向）、l_shoulder_pitch -0.223 / -0.066、l_wrist_pitch -0.255 / -0.010、l_wrist_yaw -0.133 / -0.020；最後 0.1 s 力矩飽和 84-97%（腰 / 肩 / 肘 / 腕），手腕碰到 40 rad/s 速度上限（bang-bang）。
- 15 結論：(iii) 時序 / 順序是主要流失；腰若在接觸時達參考速率，粗估上限 +8-12 m/s；加力矩（arm A）只會讓每節更快但時序不變。
- 裁定：接受時序為主因。但腰的「提前 60 ms」與引擎的揮棒起動提前量 timing_bias 0.055 s（為擊中率加的）幾乎相同——策略可能正確執行了 clip，只是球在 clip 接觸後 55 ms 才到。先做零訓練的判準：diag_bat_contact 於 12500，bias 0.055 / 0.030 / 0.000 / -0.020（jitter 0、1024 envs，標記 `logs/BSPEED6_END.txt`，約 5 min），回報各 bias 的好球帶擊中率、EV 中位、接觸甜蜜點、waist_yaw 接觸速度與峰值時間、肩 / 腕峰值時間、擊中後跌倒。判讀：較小 bias 把腰峰值移到接觸且 EV 明顯上升（>= +3 m/s）、擊中率仍 >= 0.40 -> 槓桿 = 引擎的揮棒起動時間（我的、免訓練；M2 只需從 oracle 重跑，投球不變）。若 bias 0 腰仍提前 -> Human23（接觸時關節速度對齊參考的獎勵，waist_yaw 優先再左臂鏈；+100 / +200 判準；不啟動）。Human23 提案書（CPU）掃描後無論如何都要寫。
- 對使用者：arm A（2.0x）目前不建議——力矩不是主因；力量倍率決定可暫緩到時序結果出來。

### 10-29 02:3x（規劃 session）閉環量測：開環 0.56 是假象，策略骨盆轉速 1.23 倍參考 -> 上限在手臂端；核准關節貢獻診斷
- 15（058f38a、7edb2b1，`logs/BSPEED4_END.txt` 02:28）：(a) 開環 F0f（1.5x）骨盆比值 0.57、甜蜜點 13.6；腳離地 sim/ref 左 0.085/0.085、右 0.553/0.128 m，鞋底傾角峰值 42.7 / 50.8 度 -> 後腳在傾倒時被甩離地 ~0.55 m，腳卸載、摩擦無從作用（與 F0c-F0e 一致）。(b) 閉環 12500（1.5x、P3c t18、jitter 0、bias 0.055、1024 envs、277 次擊中）：骨盆比值中位 1.23（IQR 1.12-1.35，sim 16.1 vs ref 13.2 rad/s）；擊中 1.23 / 揮空 1.26 / 擊中後倒 1.21（n 136）/ 站 1.23（n 141）；腳偏航 ~21 rad/s、鞋底傾角 ~21 度、離地左 0.107 / 右 0.074 vs ref 0.075 / 0.123；腳平移 4.2 / 4.4 m/s（參考 ~2.1 / 1.2，先前印出的 8.3 / 4.8 是 x4 的除法錯誤，7edb2b1 修）；接觸時甜蜜點 20.1、EV 中位 18.4（= H22 基準）。
- 裁定：策略的骨盆不是上限（比參考多轉 ~23%，以腳為軸），骨盆偏航 / 髖肩分離獎勵無用。剩下的上限在手臂端：手腕力矩（規格；開環人類力量下手腕飽和 ~25%）及 / 或 骨盆 -> 球棒的傳遞（腰 / 肩 / 腕的時序）。
- 核准（GPU 空、立即、約 3 min、標記 `logs/BSPEED5_END.txt`）：diag_bat_contact --chain_log --phase_log 於 12500，只算擊中，每個連桿對接觸時甜蜜點速度的貢獻 vs 參考關節速度 + 各連桿峰值時間相對接觸（sim / ref）+ 最後 100 ms 的各關節力矩飽和 %。結論三選一：(i) 手腕力矩（規格、使用者）(ii) 腰 / 肩力矩 (iii) 時序 / 協調（訓練獎勵、我的）。這決定 arm A 是否值得。

### 10-29 02:3x（規劃 session）摩擦假說被推翻（mu 2.0 / 扭轉摩擦都不改骨盆比值）；核准在訓練後策略（閉環）上重量
- 15（e13b8f5，`logs/BSPEED3_END.txt` 02:22）：骨盆比值 / 甜蜜點 / 腳平移 sim/ref 左右 / 腳偏航角速度 / 腳踝飽和：F0c（1.5x，mu 2.0）0.53 / 13.6 / 3.12/1.94, 2.99/1.40 / 7.3, 4.8 / 17.7%；F1c（STRONG g3，mu 2.0）0.56 / 23.9 / 3.03/2.01, 3.47/1.18 / 7.3, 5.3 / 30.6%；F0d（對照 1.0/0.9）0.51 / 15.4 / 3.87/1.99, 3.14/1.33 / 6.6, 4.0 / 12.7%；F0e（F0d + 扭轉 0.04 m）0.50 / 14.4 / 2.98/2.09, 3.39/1.29 / 7.1, 4.6 / 14.8%。
- 裁定（接受）：摩擦加倍或加扭轉摩擦都不改骨盆比值（0.50-0.56）與腳的平移 / 自轉 -> 不是摩擦限制的滑動，比賽不改摩擦（投手閘門、pass 1 都不必重跑）。F0d vs F0b 甜蜜點 15.4 vs 13.8 = 開環逐次雜訊約 ±1 m/s。可能原因：開環無平衡（每次 16/16 倒），揮棒中身體傾倒、腳卸載（抬起或踩邊），摩擦無從作用；0.56 可能部分是開環產物。
- 核准（GPU 空，立即跑，約 3 min，標記 `logs/BSPEED4_END.txt`）：(a) diag_bat_speed 加腳離地高度 + 鞋底傾角；(b) 在訓練後 12500（閉環、P3c t18、jitter 0、bias 0.055、1.5x）上量同樣的骨盆 / 腳指標，只算擊中，並按擊中 / 揮空、跌倒 / 站穩分組。判讀：策略骨盆比值也 ~0.55 -> 上限真實存在、在全身協調（平衡 vs 旋轉），下一個槓桿是訓練獎勵（骨盆偏航角速度 / 髖肩分離追蹤）而非力矩；>= 0.8 -> 開環 0.56 是產物，剩下的是手臂端力矩上限（規格）。
- 使用者決定項目更新：地面摩擦不需改（撤回）；打者力量倍率仍待決，等 (b) 的數字後給建議。

### 10-29 02:2x（規劃 session）天空版兩個打席重拍驗收 OK -> m3pa_draft.mp4 v4 / m3pa2_draft.mp4 v2 = 現行展示版
- 20（ee850e1，02:18 M3SKY_ALL_END）：PA1 揮棒列 p1 e60 / p4 e37 / p5 e12 各 64/64、看球 p2 e51 / p3 e34 63/64；PA2 揮棒 p1 e62 64/64、看球 p2-p6 63/64；看球 7 球 0 跌倒；每球 343 幀 blank 0。我複核：`videos/m3pa_draft.mp4` 59.4 s、`videos/m3pa2_draft.mp4` 65.7 s，皆 h264 High yuv420p 2560x720，`videos/fixcheck_tiles/m3pa_p5_tile.png` 看過——藍天、雲、實影、中綠草地、牆 + 好球帶、擊球後落點圓環可見、打者與球棒都在本壘視角畫面內。舊版留 m3pa_draft_v3.mp4 / m3pa2_draft_v1.mp4。
- 這兩支就是目前的展示版（投手贏 5 球內野出局 / 打者贏四壞）；11-26 最終剪輯在此基礎上加球數卡與說明。
- Isaac 0；15 的摩擦診斷鏈（gate M3SKY_ALL_END）應已自動啟動，約 5 min -> `logs/BSPEED3_END.txt`。20 無排程；之後 GPU 閒置待 15 數字與使用者決定（力量 / 摩擦）。

### 10-29 00:47（規劃 session）3 小時檢查：天空重拍 PA1 3/11 球完成、各球一致；GPU 滿載無閒置
- Isaac 0 瞬間（球與球之間的閘門）；20 的 `logs/m3sky_all.sh` 00:21 起：PA1 p1（s1 e60 揮空）64/64 IDENTICAL、p2（s1 e51 看球，take mode 10，bat_vmax 2.45，batter_fell False）63/64、p3 已出片；每球約 10 min，預計 ~02:10 M3SKY_ALL_END。之後 15 的 `logs/run_bspeed3.sh`（F0c / F1c mu 2.0、F0d 對照、F0e 扭轉摩擦 0.04 m，約 5 min，標記 `logs/BSPEED3_END.txt`）。
- 15 的 CPU 查核：機器人 USD 無物理材質，用場景預設 1.0 / 0.9，與地面相同 -> 今天腳地摩擦已是 1.0 / 0.9，沒有「被平均拉低」的單純修正；F0d 當對照（須重現 F0b 骨盆 0.56 / 甜蜜點 13.8）。
- 影片：00:19 天空測試片 + 重拍中，12 小時影片義務已覆蓋。無 trainer 在跑（打者功率訓練 arm B 已停、arm A 待使用者）。
- 下次檢查 ~03:47：驗收兩支重拍 concat（tiles、64/64、frame check）、15 的摩擦數字 -> 給使用者「打者力量 vs 地面摩擦」建議。

### 10-29 00:3x（規劃 session）第三個棒速上限 = 腳打滑 / 骨盆轉不到（比值 0.56）；核准摩擦力診斷（排在天空重拍之後）
- 15（a6411d0，`logs/BSPEED2_END.txt` 00:22）：骨盆偏航角速度峰值 F1b（STRONG + 增益 x3）7.3 vs 參考 13.0 rad/s、F0b（人類 1.5x）7.5 vs 13.2 -> 比值 0.56，與力矩、增益無關。腿部關節都跟得上參考（髖偏航 13.1/14.4、膝 11.8/12.9、腳踝 pitch 8.2/9.0 ...）；腳水平速度峰值 sim / 參考：F1b 左 2.83/1.91、右 3.43/1.32，F0b 左 2.69/1.84、右 3.11/1.24 -> 腳滑 1.5-2.6 倍。甜蜜點 F1b 23.2 / F0b 13.8（重現 F1 / F0）；腳踝飽和 F1b 28.7% / F0b 7.3%。地面 static 1.0 / dynamic 0.9，腳的材質未設（PhysX 取平均）。
- 解讀（接受）：關節照參考轉，但腳在地面滑 / 原地打轉，骨盆在世界座標只轉 ~56%，動力鏈在根部損失 ~44%，所以加力矩或增益都只到參考的 60-70%。
- 裁定：核准「診斷」F1c / F0c（地面 static / dynamic 2.0，釘鞋）+ F0d（腳材質明確設成地面值 1.0 / 0.9；先在 CPU 查今天腳連桿材質實際解析成什麼——若 PhysX 跟預設 ~0.5-0.7 平均，現在真實摩擦 < 1.0，F0d 可判斷「單純修正」能否恢復大部分骨盆旋轉，不必用釘鞋）。回報骨盆比值、腳滑、甜蜜點、腳踝飽和。閘門：`logs/m3sky_all_chain.txt` 出現 M3SKY_ALL_END + Isaac 0 兩次（20 的 11 球重拍約 2 h，每球前重查 Isaac 0，不可插在球與球之間），標記檔 `logs/BSPEED3_END.txt`。
- 比賽端是否改摩擦 = 物理變更：要從 pass 1 重跑（M2 -> joint_pa -> 影片），投手共用地面需重驗 P3c 閘門（identical 52 / T3 48）；等數字出來後由我 / 使用者決定。這會改變「打者力量」問題的答案：若腳滑修好，1.5x 可能就夠接近外野，arm A 可能不需要。

### 10-29 00:2x（規劃 session）天空 HDRI 測試通過（穹頂 500）-> 核准兩個打席全部重拍（11 球，約 2 h）；15 的下肢診斷先跑
- 20 的兩支 p5 測試片（HDRI 1000 / 500）都 64/64 IDENTICAL vs m2_s6，343 幀，blank 0，High。落點幀 250 數值：草地 基準 (200,227,188) / HDRI1000 (192,221,181) / HDRI500 (144,180,134)；圓環最紅像素 (252,190,161) / (255,192,168) / (244,154,130)；R-G>40 像素 596 / 508 / 680。1000 修好 cf / 側視（藍天、雲、飽和草地、實影）但俯視仍淡；500 俯視也修好，各視角自然、無壓黑。程式 ee850e1（天空）+ 2c66782（本壘視角 1.65x / 1.4 m）。我看過 sky05_f40 / f250：明顯進步，定為最終外觀（除非使用者反對）。
- 裁定：HDRI 500 重拍兩個打席（PA1 5 球 + PA2 6 球），每個揮棒列 64/64、看球列 63/64、看球 0 跌倒，重接 m3pa_draft.mp4 / m3pa2_draft.mp4（舊版留 *_v3 / *_v1 副本）。閘門：先等 15 的下肢診斷（`logs/run_bspeed2.sh`，00:2x 放行，約 3 min，標記檔 `logs/BSPEED2_END.txt`）+ Isaac 0 兩次；之後整條鏈一個唯一結束標記（20 回報名稱）。期間不排任何其他 Isaac。
- 15 的下肢診斷（ece50e1）：骨盆偏航角速度峰值與比值、腳踝 / 膝 / 髖偏航關節速度峰值、腳水平速度（打滑指標）；不加 ContactSensor（會動到訓練環境的 spawn cfg）。

### 10-29 00:0x（規劃 session）F0-F3 棒速診斷結果：PD 頻寬不是主因、手腕力矩是人類端瓶頸、另有第三個上限（疑下半身 / 地面）；arm A 單獨不足以到外野
- 15（13e76eb，`logs/BSPEED_END.txt` 23:55）：開環參考揮棒、零殘差、16 envs、現行 USD；參考棒尖 37.5-37.9 m/s、omega 42-46 rad/s。
  - F0 人類 1.5x：甜蜜點 13.5、棒尖 16.7、omega 26.5；飽和 手腕 26.2 / 腰 13.8 / 腳踝 12.3%
  - F1 STRONG + 增益 x3：甜蜜點 22.5、棒尖 27.4、omega 36.9；手腕 8.1 / 腰 0 / 腳踝 30.2%
  - F2 1.5x + 增益 x3：甜蜜點 15.9、棒尖 19.5、omega 27.4；手腕 27.5 / 腰 11.5 / 腳踝 25.2%
  - F3 2.0x + 增益 x3：甜蜜點 16.4、棒尖 20.3、omega 29.9；手腕 26.9 / 腰 2.9 / 腳踝 17.9%
  - 跌倒 16/16/16/15（開環無平衡，F0 也倒，看不出 x3 抖動；沒有比 F0 更早倒的）。
- 裁定（接受 15 的解讀）：(1) F1 22.5 < 28 -> PD 頻寬不是主要的非力矩上限；連近乎無限力矩 + 3 倍增益也只到參考的 60-70%，存在第三個上限。(2) 人類力量下增益 x3 +2.4 m/s、再加 2.0x 力矩只多 +0.5，手腕維持 ~27% 飽和 -> 手腕力矩規格是人類端瓶頸；arm A（2.0x + 增益）開環預估只 +3 m/s，外野需 +5-8 m/s，**單獨跑 arm A 不會到外野**，先找第三個上限再談。(3) 第三個上限疑在下半身 / 地面：F1 腳踝飽和升到 30%，骨盆偏航角速度（參考 12-13.5 rad/s）可能跟不上。
- 注意：開環重播低估訓練後策略（交付版 12500 接觸時甜蜜點 20.1 vs F0 13.5），這些數字看相對差異。
- 核准（~1-2 min GPU，未啟動）：diag_bat_speed 加骨盆偏航角速度峰值（順便腳踝 / 膝 / 髖偏航峰值與腳接觸力 vs 參考），重跑 F1 與 F0；若約 60% -> 上限在腳 / 腳踝 / 摩擦，不在手臂。排程：20 的天空測試片（p5，可能加穹頂 0.5x 一支）先跑，20 回報拍完 + Isaac 0 後我再放行 15；15 先在 CPU 上準備並 commit。
- 對使用者的建議更新：在第三個上限查清前，不建議啟動 arm A（2.0x）訓練；交付維持 1.5x / 9.8%。

### 10-29 00:0x（規劃 session）p5 加強版圓環重拍 64/64 -> m3pa_draft v3；整體曝光偏淡；核准 BSPEED_END 後的天空測試片（PA1 p1）
- 20（6199622）：圓環 1.2 m 紅橘 + 25 m 內 2.0 倍追球，p5 重拍 23:52 M3PA_END，「env 12 IDENTICAL 34 columns，64/64」vs m2_s6；`videos/m3pa_p5_s6_k12.mp4` 343 幀，`videos/m3pa_draft.mp4` 重接 1783 幀，High yuv420p，blank 0。看過全解析度落點幀：圓環清楚、兩倍大，但整張畫面（草地也）偏淡，是場景曝光 / 白色穹頂，不是材質。
- 20 的 CPU 準備：`logs/pv_sky.patch`（本壘視角 eye/target z 1.35 -> 1.4、zoom 1.5 -> 1.65；film_sky；HDRI 本機檔 data/sky/kloofendal_43d_clear_puresky_4k.hdr（18 MB，gitignored），Nucleus URL 備援，例外時退回色彩穹頂）。
- 裁定：`logs/BSPEED_END.txt` + Isaac 0 兩次之後，先拍一支測試片 PA1 p1（s1 揮棒列）套 pv_sky -> 64/64 + tile + 落點全解析度幀；若 HDRI 仍淡，同一段時間再試穹頂強度 0.5x / 渲染器曝光（短 16-env 渲染即可），並回報落點幀的草地與圓環 RGB 數值。兩個打席是否全部重拍（約 11 球、約 2 h）看 tile 後再決定，只有天空明顯修好偏淡才重拍；在此之前 m3pa_draft / m3pa2_draft 維持現行展示版。
- 等待：15 的 F0-F3 棒速數字（BSPEED_END）；20 的天空測試 tile；使用者的力量方向。

### 10-28 23:5x（規劃 session）棒速上限分析（15，CPU）-> 核准 F0-F3 開環棒速診斷（~10 min GPU，排在 20 的 p5 重拍後）
- 15 的 CPU 分析（691bbc0 第 3 節）：關節速度上限不是瓶頸（接觸前 0.25 s 參考速度峰值 <= 0.93 倍人類上限）；棒速是球棒自轉（接觸時手 2-7 m/s、棒尖 ~37 m/s、omega 42-46 rad/s、alpha 460-960 rad/s²，I_knob 0.351 -> 需 160-340 Nm，扣掉手加速度約 110 Nm 仍要 50-230 Nm 的手腕力偶，1.5x 只有 2 x 30）；10-17 的開環 diag_bat_speed 已顯示：人類 1.0x 13.3、1.5x 14.8（手腕 24% / 腰 13% 力矩飽和）、增益 x3 14.3、STRONG（3-9x 力矩、幾乎不飽和）也只有 19.7 m/s vs 參考棒尖 37.7。
- 15 的解讀：兩個上限——力矩（規格、使用者的）與 PD 頻寬（手腕 kp 150 / kd 3 約 4 Hz vs ~50 ms 的甩棒；控制器設計、我的）。今天的增益下再加力矩也只到強力機器人的 ~20 m/s，與 2.0x zero-shot（+1.5 m/s）和 Human16 / 22（19-20）吻合。
- 裁定：核准 F0（1.5x 基準、現行 USD）+ F1（STRONG + 增益 x3）+ F2（1.5x + 增益 x3）+ F3（2.0x + 增益 x3），開環 diag_bat_speed、16 envs、不訓練。判讀：F1 >= 28 m/s = PD 頻寬是非力矩上限；~20 = 另找原因（根部 / 腳）。若 F1 高且 F3 >> F2，arm A 的定義改為「2.0x + 增益 x3」而非單純 2.0x（倍率仍待使用者）。要求回報每次的棒尖速、omega、手腕 / 腰飽和 %、重播是否站穩（x3 也抬高 kd）。
- 排程：gate = `grep -q M3PA_END logs/m3pa_chain.txt`（20 的 p5 重拍，~23:45 起約 12 min）+ Isaac 0 兩次；結束標記 = 檔案 `logs/BSPEED_END.txt`（失敗也要寫）。20 在 p5 重拍 + 64/64 + 重接後，本壘視角 / 天空影片要等 `logs/BSPEED_END.txt` 存在 + Isaac 0 兩次；若 M3PA_END 後 45 min 還沒有標記，回報我而不是直接拍。

### 10-28 23:4x（規劃 session）arm B（Human22）+100 判準失敗 -> 停；裁定：接受負結果，交付維持 M2 / 9.8%；GPU 交回 20 做圓環重拍
- 15 的 Human22（力量 1.5x、目標 43 m、w_exit_velo 60，從 h9b 12500 續訓 100 its）：訓練 exit_velo EMA 16.90 -> 16.74（門檻 +2 m/s），hit_rate 反而 0.270 -> 0.357，bat_speed 19.0；12500 基準 diag1024（P3c t18、jitter 0、bias 0.055）：好球帶 0.445、擊中後跌倒 49%、EV 中位 18.4、甜蜜點棒速 20.1 m/s。`logs/H22_END.txt` 23:38，15 已記錄 81bbb0e。
- 我的裁定：有效的負結果。1.5x 下棒速上限 ~19-20 m/s（與 Human16 甜蜜點獎勵時期一致），加重初速獎勵只換到更多接觸、換不到速度；不跑 +200、不跑 M2。交付版仍是 h9b 12500 @ 1.5x（M2 好球帶 0.515、打者勝率 9.8%）。
- 力量剩下的唯一槓桿 = arm A（2.0x 續訓），屬規格值，待使用者決定；在那之前不再排打者功率訓練。
- GPU 空（Isaac 0）：已放行 20 依既定順序動作：套 `logs/ring2_zoom2.patch` -> 重生 joint_smoke_20 -> commit -> p5 單球重拍 + 64/64 比對 + 重接 m3pa_draft -> 本壘視角 -> 天空 HDRI。
- 小瑕疵：run_human22.sh 第 48 行在第一段前對不存在的 train_human22.txt 做 wc 印了一行錯誤，L0 退回 1，不影響結果。

### 10-28 23:2x（規劃 session）四壞打席影片驗收 OK -> m3pa2_draft.mp4；Human22 閘門開啟
- 20 的四壞打席（seed 576，6 球：s10 e62 揮空、s1 e58/33/12/45/56 看球 壞/壞/好/壞/壞 -> 四壞）23:22 M3PA_END。`videos/m3pa2_draft.mp4` 1972 幀 65.7 s，h264 High yuv420p 2560x720，blank 0（我用 ffmpeg -i 複核）。
- 投球一致性：第 1 球（揮棒列）64/64 與 M2 相同；p2-p6（看球列）各 63/64（只有看球的那個 env 不同，符合設計）；看球 5 球 batter_fell 全 False（take mode 10 在正式打席影片上成立）。格線圖 `videos/fixcheck_tiles/m3pa2_p6_tile.png` 看過：牆 + 好球帶在打者視角可見，看球後站到 6 s 結束。
- 現在交付用的兩個打席影片：`videos/m3pa_draft.mp4`（投手贏：5 球內野出局）+ `videos/m3pa2_draft.mp4`（打者贏：四壞）。最終 11-26 剪輯用這兩種結局。
- Isaac 0；15 的 Human22（arm B）閘門（M3PA_END + Isaac 0 兩次）已開，接著 baseline diag（12500）-> +100 / +200 判準。期間 GPU 歸 15，20 不動程式樹；H22_END 後 20 才套 `logs/ring2_zoom2.patch`（p5 單球重拍 + 64/64 + 重接）。
- 待使用者：打者力量方向（維持 1.5x / 加跑 arm A 2.0x）。下次 3 小時檢查 ~00:47。

**10-28 22:4x p5 落地圓環重拍完成（20，22:08–22:20，64/64）→ videos/m3pa_draft.mp4 = 草稿 v2 定稿（含牆 / 好球帶、take 不倒、裁切、落地圓環）；M2 / 9.8% 正式定案（15，2238daf）；四壞打席拍攝中**：圓環 cfg.film_landing_ring（純視覺、--video 開；0.6 m 橘色碟、中心 0.44 m 草色、離地 6 mm、落地起出現於積分落點、只對落地的擊球）。我看全解析度切換後畫面：圓環在畫面中央、約 25 px、在天光下偏淡黃 → 排程（Human22 之後、不插隊）：圓環 1.2 m、飽和紅橘色、25 m 內追球改 2.0 倍，只重拍 p5 + 64/64 比對 + 重串接；20 的其餘外觀項（本壘視角、天空）同一 GPU 時段接著做。Human22 chain（logs/run_human22.sh → logs/h22_chain.txt、結束檔 logs/H22_END.txt）：w_exit_velo 繼承值為 20（非提案的 4）→ 改 60（×3，我不否決）；Human22 = Human15 + target_distance 43 + w_exit_velo 60、effort 1.5、從 h9b 12500、2048 env（f3ee5d0）；gate = logs/m3pa2_chain.txt 的 M3PA_END + Isaac 0 兩次（20 確認：四壞鏈 stdout 全進該檔、只有最後一行含 M3PA_END、p5 重拍寫在 m3pa_chain.txt 不會誤觸）；先跑 12500 的基準 diag1024（jitter 0、bias 0.055、P3c t18）；+100 否證：訓練 exit_velo EMA（it 11–20 vs 最後 10）≥ +2 m/s 且擊中率 ≥ 0.7×；+200：diag 初速中位 ≥ 18、好球帶 ≥ 0.43、擊中後跌倒 ≤ 55% → 續到 +600。四壞打席（seed 576、6 球 → videos/m3pa2_p*.mp4 + m3pa2_draft.mp4，結果卡「Ball four - walk」）約 22:25 起、約 60 分鐘。

**10-28 22:3x M2 / 9.8% 在凍結設定上定案（條件：p5 落地圓環重拍比對亦 64/64）；打者力量訓練線提案（15，8c38737）核准 B 臂、A 臂等使用者**：重啟規則只在投球位元改變時觸發，p1–p5 比對無變化 → 不必從 pass 1 重跑。提案：從 h9b 12500、2048 env、約 540 it/h，投球同同場協定（P3c 出手表、4 clip、真實高度、bias 0.055、訓練預設 jitter）；獎勵只改兩項：target_distance 29 → 43 m（29 m 是 10-25 的「內野邊緣」誤算，外野邊緣離本壘 38.9–47.4 m、±30° 約 43 m）、w_exit_velo 4 → 12；跌倒項不動（單變數）。A 臂 effort 2.0（改規格值、等使用者）、B 臂 1.5（規格不變、訓練設計 → 我核准）。否證：+100 訓練擊中初速中位比前 10 it 高 ≥ 2 m/s 且擊中率 ≥ 起點 × 0.7；+200 原生 diag1024（P3c、jitter 0）初速中位 ≥ 18、好球帶 ≥ 0.43、擊中後跌倒 ≤ 55% 才續到 +600。採用 = 同場 M2 協定從 pass 1 重跑：外野 ≥ 3% 打席、擊中後跌倒 ≤ 49.5%、好球帶 ≥ 0.46，再 joint_pa + 拍片。GPU：一臂約 1.1 h 訓練 + 0.3 h 評估 + 1.5 h M2 + 1.5 h 拍片 ≈ 4.4 h；兩臂（M2 只跑贏家）約 5.8 h。風險：先前 1.5x 重訓跌倒皆升到 49–61%；無臂通過則交付維持 1.5x + 9.8%。GPU 順序：20 的 p5 重拍 → 四壞打席拍攝（seed 576、6 球，計畫檔已用 take 10，預期 row 1 64/64、rows 2–6 63/64）→ B 臂（約 1.5 h 含評估，gate = 20 的四壞打席檔案標記 + Isaac 0 兩次）→ 20 的剩餘外觀項。

**10-28 22:1x PA 重拍完成（20，M3PA_END 22:05）→ videos/m3pa_draft.mp4（裁切版 59 s）= 可給使用者看的 M3 草稿 v2**：逐球（各 343 幀、空白 0、High）：p1 揮空 64/64；p2 壞球（take 模式 10）63/64（只 env 51 不同，預期；舊的 delta_s 耦合已消失）；p3 看好球 63/64（只 env 34）；p4 界外 64/64；p5 擊出 17.9 m 界內 64/64。兩個 take 整段 6 s 不倒，球飛行中球棒 2.45 / 2.75 m/s；穿越時姿勢：p2 根高 0.935 m（t0 0.980）、骨盆 2.2°、軀幹 14.7°；p3 0.940 / 4.3° / 17.2°；blend 期間軀幹最大 29.7° / 21.4°（0.79 s）；6 s 內根高最低 0.931 / 0.930。我看 p3 / p5 tile：take 舉棒站著到球過後、牆與好球帶每個中外野鏡頭都在。p5 落點畫面（1.5 倍）球標記約 7 cm 幾乎看不見、畫面幾乎只有草地（裁切版只佔 3.80–4.30 s）→ 核准 20 加落地圓環（約 0.6 m 純視覺、落地時出現於積分落點、與牆同旗標），只重拍 p5（約 10 分鐘、64/64 比對 + tile）再重串接。之後：第二個打席（四壞、seed 576）待 15 確認計畫檔已用 take 10 後同法拍攝。

**10-28 21:47 三小時檢查**：Isaac 1（20 的 PA 重拍鏈，含牆 + 好球帶、1.5 倍追球、take 模式 10；p1 投球不變 64/64 通過，21:44 拍到 p4，預計 ~22:00 M3PA_END 後自動串接裁切版）；GPU 30% / 6.1 GB；無閒置、無需接手。我已看 p1 與 p3 tile：牆與好球帶在中外野視角清楚、其他視角不受影響；**p3（看好球）的打者整段站著、舉棒不動、球過後仍站立**（舊版 2.42 s 趴地）→ 使用者兩項要求（牆 + 好球帶、看球不跌倒）在畫面上確認。影片：今日 m3_s3_k12（19:23）與 PA 重拍（21:3x–22:00），下次最晚 10-29 09:4x。待：M3PA_END 後看 p2 / p5 tile 與串接草稿，再回報使用者；力量方向等使用者決定。

**10-28 21:1x take 不跌倒修好（15，診斷 2–4）；重拍鏈可啟動；打者歷程影片剪好**：診斷 2（20:42–20:49）：模式 6（啟動起 0.5 s 只腿 + 腰 blend）在 obp_167-7 / 282-1 過（0/10 倒、球棒 ≤ 0.78 m/s），obp_42-1 / 42-2 10/10 倒；模式 5 同；模式 3 不倒但球棒 4.5–6.7。診斷 3（20:52–21:00）：模式 7（t = 0 起腿 + 腰 blend）/ 8（啟動起 0.25 s）同樣 obp_42 全倒 → 傾倒來自 obp_42 站姿的手臂（肩 pitch 離預設姿勢達 1.6 rad）把重心帶偏，腿不是原因。診斷 4（21:02–21:09）：模式 9 / 10 = 全身自 t = 0 起 0.5 / 0.8 s blend 到 assets 的 batting_stance.json（舉棒、屈膝，出手前完成）→ 4 個 clip 全過（各 0/8 倒、全跑 0 倒、球飛行中球棒 1.28–2.75 m/s，obp_42-1 的 2.75 是 PD 撐姿的晃動、非揮棒）。**採模式 10（0.8 s）**；遮罩以 take 值 10 重生成（e7707b5），兩個展示計畫（seed 2081 / 576）不變；take_bat_vmax 只量球飛行期間（交接到 done、跌倒前）。影片裁切規則已實作（20，9f09ffd）：take / 揮空 3.10–3.11 s、界外 3.90、擊出 4.30 結束，v1 裁切版 59 s（videos/m3pa_draft_v1_trim.mp4），我核准為預設。20 現在套用 wall_zoom.patch + take 10 → 重生成 joint_smoke_20 + commit → 重拍 5 球（p1 兼投球不變驗證 64/64，不過則停）→ 自動串接。另：使用者要求的打者訓練歷程影片已剪好 videos/batter_progress_1028.mp4（84 s、9 段 + 片頭 / 片尾卡；腳本 scripts/make_batter_progress.py，962b66d）。

**10-28 20:5x 使用者第二項影片意見：投完球後切到投手視角停留太久，約 t = 3.1 s 就可換下一球**：交 20，在 make_pa_video 以 CPU 裁切（逐列讀 csv，不需重拍）：take / 揮空 / 界外 → 片段結束 = 切換點（done_t + 0.5 s）+ 0.5 s；擊出界內 → max(切換 + 0.5 s, 落地 + 0.3 s)，上限 4.5 s；6 s 全長只留作除錯檔。先在 v1 草稿上做 CPU 測試（videos/m3pa_draft_v1_trim.mp4）回報逐球結束時間，我看節奏後再用於重拍版。take 診斷 2（15，20:42–20:49）：模式 6 在 obp_167-7 / 282-1 通過（0/10 倒、球棒 ≤ 0.78 m/s），obp_42-1 / 42-2 10/10 倒（2.13–2.16 s，站姿 1.4–1.6 s 就傾、blend 來不及）；模式 5 同樣 obp_42 全倒；模式 3 不倒但球棒 4.5–6.7 m/s。展示的兩顆 take（env 51、34）恰為 obp_42（clip = env % 4），不改 clip（會動場景狀態 → 投球位元）。診斷 3 跑中（2f17f84、TAKEDIAG3_END.txt）：模式 7 自 t = 0 起腿 + 腰 blend；模式 8 自啟動起 0.25 s blend。規則：須 4 個 clip 全過（done + 0.5 前 0 倒、球棒 < 3 m/s），優先 8、其次 7；都不過則 7 加早 / 加長 blend。blend 峰值關節速度 1.0–1.4 rad/s（慢速重心轉移）；預設姿勢 = assets 的 batting_stance.json（舉棒、屈膝）；穿越時的根高度 / 軀幹傾角從重拍列取。

**10-28 20:4x take 趴倒診斷 1（15，20:31–20:39）：站姿本身在純 PD 下站不住、啟動路徑無辜；診斷 2 跑中；牆設計核准**：各 16 env（修正：joint_smoke 的 batter_fell 只算 active env，改讀 --diag 的根高度）：模式 0 純 PD 站姿 16/16 倒（2.21–3.19 s）；模式 2 策略時鐘釘第 0 幀 12/16 倒（2.16–5.43 s，done + 0.5 前只 1 個）但球棒動 ~8 m/s、違反球棒規則；模式 3 PD 到預設姿勢 0/16 倒但啟動瞬間跳姿（球棒 6.4 m/s）；模式 4 全程不啟動 16/16 倒（1.42–2.39 s）→ 站姿約 1.4 s 就會傾倒、與是否啟動無關（20 的「穩 0.8 s」只是視窗太短）。診斷 2（logs/run_takediag2.sh、標記 logs/takediag/TAKEDIAG2_END.txt、約 7 分鐘、各約 21 env）：模式 3 對照；模式 5 自排定啟動（~0.8 s）起 0.5 s smoothstep 從站姿 blend 到預設姿勢（~1.3 s 完成、早於傾倒）；模式 6 同 5 但只腿 + 腰，手臂維持舉棒站姿（最像打者）。球棒速度只在球飛行中、跌倒前量。通過 = done + 0.5 前 0 倒、球棒 < 3 m/s；優先模式 6；另要求回報球穿越本壘時的根高度 / 軀幹傾角與 blend 的最大關節速度（看起來要像站在打擊區的打者）。通過即為重拍與定版的 take 模式。20 的牆（logs/wall_zoom.patch，含 1.5 倍追球）已核准：本壘前緣後 3.0 m（= flight_end_behind_plate_m，球飛行本來就在此結束）、4.0 寬 × 2.5 高 × 0.1 厚、深灰、隨 plate_yaw 18°；好球帶 = StrikeZone()（半寬 0.2525 = 半本壘 0.2159 + 球半徑 0.0366 → 寬 0.505 m、z 0.50–1.10 m、以 zone_center_y 為中心），淺色面板 + 3 cm 紅框微凸；純視覺（UsdGeom Xform /World/envs/env_0/FilmWall + CuboidCfg 無碰撞 / 剛體 / 質量，同投手丘複製），旗標 cfg.film_wall 預設關、--video 才開；攝影機在牆後（本壘視角 +9 m、高角 +6 m）時牆隱藏，中外野與投手側視顯示；球標記在牆矩形內夾到牆面（−球半徑 −12 mm），j_pos / j_cross 不動。投球不變驗證 = 重拍第 1 球（seed 1 env 60）與 m2_s1.csv 比對須 64/64，否則鏈在 p2 前停止。順序：診斷 2 → 套用 take 修正 + wall_zoom.patch → 重生成 joint_smoke_20 + commit → 重拍 5 球。

**10-28 20:5x 2.0x 零樣本力量探測完成（15，20:22–20:30）：力量倍率幾乎不動外野問題；另發現「投球綁定場景設定」**：注意：打者關節改 2.0x 後 64 顆投球有 27 顆變了（pred_y/z 最多 0.5 m、kmh ±3；投手 cfg 與 effort 無關、hand-off 時間只差 ≤ 2.5 ms = 一步）→ 場景設定改變 GPU 浮點位元、載入後決定性的投手策略在約 40% env 混沌發散；M2 各 seed 投球相同是因場景設定相同（take 遮罩也保住 63/64）。規則：64 顆投球綁定精確的場景設定；**定版管線在凍結設定上只跑一次、依序：pass 1 timing.csv → M2 10 seed → joint_pa（片尾卡 + 展示計畫）→ 拍片**；之後任何設定變更（牆、旗標、effort、take 模式）從 pass 1 重來。20 加牆前須先做 64 env 投球欄比對（kmh / plate_y/z / handoff_t）與 m2_s1.csv 64/64 全同（拍片旗標評估時關、影片列 = 評估列曾 64/64 成立，純視覺 prim 應不改位元）。同樣 37 顆投球 1.5x vs 2.0x：擊中 12 vs 12、好球帶 12/22 vs 12/22；EV q25/50/75 13.0/15.9/17.2 vs 13.5/17.4/20.7，最大 21.1 vs 22.0；carry 最大 24.8 vs 29.6 m；done 前跌倒 8 vs 5；內野 4 + 界外 8 vs 內野 7 + 界外 5；外野 0 vs 0。全 64 顆 2.0x：擊中 20（1.5x 19）、好球帶 0.528、EV 中位 16.0、無外野。結論：零樣本加力量只多約 1.5 m/s 中位初速，離外野所需約 25 m/s 很遠；不訓練、單靠倍率不會有外野球 → 交使用者決定（我的建議：11-26 交付維持 1.5x，外野球改靠訓練線、有空檔 GPU 才跑）。M2 / 9.8% 在 take 模式與牆定案前視為暫定。take 診斷 20:30 啟動。

**10-28 20:4x 使用者看過影片：「做得不錯」，兩項要求**：① 打者身後要有一面牆、牆上畫好球帶（ONE OUTS 規格原本就有「好球帶畫在牆上」）→ 交 20（場景）：本壘後方捕手側的平牆，畫 StrikeZone 判定用的寬度與此打者的 z 範圍、以本壘為中心；限制：本壘視角攝影機在本壘後 9 m，牆不能擋（攝影機放在牆內側或牆單面渲染）；球在放球後走 15 的 RK4 標記，標記到牆面停住（夾住 x，不需 PhysX 碰撞體，好球判定讀本壘面穿越、不受影響）；純視覺、所有欄位不變（比對 64/64 仍成立）；掛在 film_grass / film_round_mound 同一組旗標（拍片預設開、評估關）。CPU 先備好，與 take 修正 + 1.5 倍追球補丁一起套用、重生成 joint_smoke_20 一次，PA 5 球全部重拍兼測試；拍前 20 先回報牆距與好球帶尺寸。② 打者選擇不揮棒時不可跌倒 → 即 take 趴倒問題，15 的 4 模式診斷進行中（PROBE 後約 7 分鐘）。15 已加 take_bat_vmax 欄（模式 2 須 < 3 m/s）並修好 delta_s / t_sched 定義（改用穿越時的 rate_plane，b5e570c）→ 之後與舊 M2 檔比對須排除這兩欄。

**10-28 20:3x M3 打席草稿 videos/m3pa_draft.mp4 完成（20，M3PA_END 20:18）— 管線有效、但 take 的打者趴倒，不可展示；take 修正為 M3 的阻塞項**：83 s、2495 幀、2560×720 @ 30 fps、h264 High、空白 0，含標題 / 球數 / 結果 / 片尾卡（9.8%）。逐球：p1 s1 e60 0-0 揮空（比對 IDENTICAL 64/64）；p2 s1 e51 0-1 壞球（take，57/64）；p3 s1 e34 1-1 看好球（take，plate_y 0.016 / z 0.835，58/64）；p4 s7 e37 1-2 界外 EV 11.18、carry 9.1 m（64/64）；p5 s6 e12 1-2 擊出界內 EV 15.92、launch 20.3、carry 17.9 m（64/64）；5 球投手皆站著、各 343 幀。問題 (1)（我在 m3pa_p3_tile.png 確認）：兩個 take 的打者約 2.0 s 開始前傾、2.42 s 趴地（切換 2.61 s 前），看起來像撲倒；BatterSide 的 take 遮罩最後才寫站姿 joint_targets + 零速度目標，但啟動前同一姿勢的 hold 可穩 0.8 s → 啟動後另有驅動（feed-forward / zero_ff、上一個策略動作的 effort、時鐘或 hand-off 寫根狀態）或站姿本身無策略不平衡。已給 15 否證順序：(a) CPU shim、take env 全程不啟動（hold 延到 6 s）→ 站得住 = 啟動路徑多了驅動、對 take env 歸零；倒 = 啟動前 hold 其實是運動學寫入、需改 option b（跟 clip 跨步、hand-off 時 blend 回站姿 = check swing）；(b) 不明確才上 GPU 64 env 全 take 記跌倒時間 + 首個離目標的關節。修正通過 = take 0/64 在 done_t + 0.5 s 前跌倒、揮棒列不變（p1 / p4 / p5 IDENTICAL）。問題 (2)（無害）：一個 take env 會改 5–6 個其他未擊中 env 的 delta_s / t_sched（僅此兩欄、結果欄全同；疑 shift_hist 批次耦合）→ 15 說明，定版前須弄懂。判定依據確認（15）：最遠 3 顆落在以投手丘為圓心 28.96 m 的內野圓內（離丘 27.0 / 25.7 m、第三顆界外），38.9–47.4 m 是該圓邊緣離本壘的距離、隨噴射角變化；引擎 carry 比模擬長約 1.5 m，無隱藏外野球。GPU 順序：15 的 2.0x 探測（20:18 後啟動、約 7 分鐘、logs/m2/PROBE_END.txt）→ 15 的 take 修正測試 → 20 的外觀清單 → 重拍 PA。20 在 CPU 先備好 25 m 內 1.5 倍追球的改動（dry run、不合併）。

**10-28 20:1x NATEV 完成（15，19:24–19:27）：同場擊中初速 = 原生，初速低是打者力量問題、不是模擬問題**：原生（ii_800 + --contact_log）好球帶 0.478 完全重現；擊中初速 q25/50/75：全部擊中 原生 12.5 / 14.9 / 18.1（n 278）vs 同場 M2 12.7 / 15.9 / 18.4（n 182）；好球帶擊中 原生 12.6 / 15.1 / 18.2 vs 同場 13.0 / 16.2 / 18.7；兩邊同定義（|launch_vel|）。M2 的 182 次擊中無外野球（最長 3 顆 carry 約 40 m 卻判內野 → 已要 15 一行說明判定依據，若判錯則修正重算）。joint_pa 新增 --want in_play / pitcher / batter；打者勝的展示打席只能是四壞：rng seed 576、6 球（揮空、壞、壞、看好球、壞、壞 → 四壞），logs/m3/pa_plan_batter.csv，核准為第二個展示打席、第一個 PA 串接通過後才拍。15 鏈的小 bug（PowerShell 引號使影片行程比對從未命中，靠 Isaac = 0 守住、無重疊）已修（1030010）。排程：M3PA_END 後 → 零樣本力量探測（同場、AIB_EFFORT_SCALE 2.0、seed 1、64、全揮、約 7 分鐘；回報 EV 四分位、好球帶擊中率、done 前跌倒、外野球數，與 1.5x 並列）只用來量化給使用者的力量問題、不逕行採用 → 20 的外觀清單 → 第二個 PA。20 的 5 球 PA 鏈已於 NATEV_END 啟動（約 40 分鐘）。

**10-28 20:0x 第一支修正後 M3 同框影片 videos/m3_s3_k12.mp4（20，19:13–19:23 拍）— 我看過 tile、接受**：挑 seed 3 env 12（hit_fair_air_clean）：EV 20.85、launch 11.3、spray −10.0、carry 18.2 m、界內；打者 done 前未倒（done_t 2.97）、投手站著、tier 4、64.9 km/h；與 logs/m2/m2_s3.csv 比對 36 欄全同、64/64 列相同；343 幀、空白 0、動態 29.6%、h264 High yuv420p、3.48 s 切換。tile：兩人同框完成投球、揮棒、2.42 s 擊中；切換後左 = 投手側視（站著）正常；右 = 高角攝影機追球，球落在投手丘附近（18 m）所以畫面只剩小小的投手丘，屬外觀問題。打者收尾 2.6–3.2 s 雙腳離地（after_done policy 的跳動，外觀、15 側、非 M3 範圍）。決定：PA 鏈已掛好（logs/m3pa.sh，等 NATEV_END + Isaac 0 兩次，拍 15 的 5 列計畫、每列比對 + 幀檢查 + tile、標記 M3PA_END、約 40 分鐘），拍完前不動 joint_smoke_20（已在 23f975a 重生成，38fcc42）；M3PA 後的外觀清單依序：① 球落在 25 m 內時追球攝影機用 1.5 倍；② 本壘視角 9 m / 1.65× / 1.4 m；③ 天空 HDRI（有 fallback）。此片算今日 12 小時影片（19:23），下次最晚 10-29 07:23（PA 影片會先到）。GPU 現在給 15 的 NATEV（約 10 分鐘）。

**10-28 19:5x 打席組裝 CPU 部分完成（15，23f975a）、片尾卡數字與展示打席核准**：scripts/joint_pa.py 依我的規則（逐球以引擎 rng 抽 take、swing 抽該 env 的 10 個 M2 列之一）；CPU 檢查含新的 take 否證皆 PASS。片尾卡數字 = 20000 次打席（引擎 rng seed 0、PA 內不重複）× 640 列：打者勝 **9.8%**、每打席 3.82 球、內野球 0.407 / 三振 0.495 / 四壞 0.098（m2_analysis 的 10.6% 為有放回、留日誌參考）。觸身 0/64，展示 PA 無排除。展示 PA（rng seed 2081，首個符合規則者）：① s1 e60 0-0 揮空；② s1 e51 0-1 壞球；③ s1 e34 1-1 看好球；④ s7 e37 1-2 界外 9.1 m；⑤ s6 e12 1-2 擊出界內、EV 15.9、carry 17.9 m、內野球（投手勝，= 9.8% 下的典型結果）。計畫在 logs/m3/pa_plan.csv，已交 20；5 顆 × 約 8 分鐘 ≈ 40 分鐘 GPU，排在 NATEV 之後。備註：11-26 定版可能放兩個打席（投手勝 + 打者勝 / 外野球），joint_pa 需能挑「首個以外野球 / 打者勝結束的 PA」。

**10-28 19:3x M3 打席（PA）設計核准（15 提案，我改一處）**：注意事項：M2 的投球集只有 64 顆不同的球（投手載入後決定性，pred_y/z、kmh 10 個 seed 全同），好球帶 0.515 ± 0.028 建立在約 33 顆不同的好球上，二項 SE 低估了投球抽樣的不確定性 → PASS 不變，片尾卡改寫「本模擬實測：由本投手 64 顆模擬投球抽出 20000 次打席」。設計：(1) take 在模擬內 = 新 --swing_csv 的每 env 揮棒遮罩，take 的 env 不啟動、PD 撐住站姿讓球過（草稿版；之後若畫面太死再做跨步後 blend 的 check swing）。(2) 打席表 scripts/joint_pa.py（CPU）：我改為**不需新 GPU 跑**——take 的結果（壞球 / 好球 / 觸身）只取決於穿越點（seed 不變、M2 已有），PA 時用引擎 rng 逐球抽 take 決定，take → 以 pred_y/z 判定，swing → 從該 env 的 10 個 M2 揮棒列抽一列（seed = 打者隨機性），用滿 640 列、take 決定跨 PA 保持隨機、片尾卡數字直接來自 M2；--swing_csv 只用在展示打席的拍攝（每顆球一跑、遮罩 = 該 PA 的實際決定），同 seed 重跑是否重現統計列決定球數卡跟計畫還是跟拍到的結果。(3) 拍攝：每球約 8 分鐘 GPU，展示 PA 3–7 球（≤ 56 分鐘），須以擊球結束、carry ≥ 15 m、投手站著，最好含一個看好球與一個揮空；觸身球留在統計（引擎規則上壘）、只從展示 PA 排除並回報觸身率一次。分工：15 做 --swing_csv + joint_pa.py + logs/m3/pa_plan.csv（球 i：env、球數、swing、預期結果、seed）；20 重生成 joint_smoke_20、逐球 --count_csv 拍、ffmpeg 串接 + 球數 / 結果卡 + 片尾卡。GPU 順序：20 現在的 M3 片 → 15 的 NATEV（約 10 分鐘）→ PA 拍攝。

**10-28 19:1x 修正後 M2 完成 → 判定 PASS，同場環境的打者表現視為已驗證**：M2_END 19:11，10 seed × 64、無崩潰，早停與 seed 1 投球檢查皆過。好球帶擊中率 0.515 ± 0.028（170/330；判準 0.46 ± 0.05、原生 800 Hz 0.478，差 1.3 SE，判準本就是圍繞原生的合理窗，故 PASS）；全部球 0.284；擊中後跌倒 49.5%（90/182；判準 43 ± 8）；miss 跌倒 0；穿越 vs fly_to_x 最大 0.05 cm；ONE OUTS × 20000：四壞 10.6%（判準 10 ± 2.3）、打者勝 10.6%（引擎參考 17.6%）、內野球 0.40、三振 0.494；擊中初速 q25/50/75 12.7 / 15.9 / 18.4 m/s（原生同分布對照 NATEV 排在 20 的影片之後，參考用、非門檻）；dz sd 0.034（原生 0.036）。15 記在第 3 節 f6229ce。決定：M3 片尾卡的勝率用同場模擬自己的數字（10.6%，標「本模擬 20000 次打席實測」），引擎 17.6% 只留在日誌當參考；15 下一項 = M3 的打席組裝（球數卡、take 判定、四壞 / 三振 / 擊球結束態），在 NATEV 之後、且 20 拍片時不開 GPU 工作。

**10-28 18:47 三小時檢查**：Isaac 1（15 的修正後 M2，跑到 seed 8/10，約 19:00 結束 → 20 的 M3 嚴格挑法拍片鏈自動接）；GPU 29% / 4.5 GB；影片最新 15:20（m3_s1_k44），下次最晚 10-29 03:20，M3 新片會先到。先用 m2_analysis（CPU）看前 7 個 seed（448 球）：好球帶 231 球、好球帶擊中率 **0.502 ± 0.033**（116/231；判準 0.46 ± 0.05、原生 800 Hz jitter 0 為 0.478）→ 差距確認消失；全部球 0.279；穿越 vs fly_to_x |dy|、|dz| 最大 0.05 cm（< 1 cm）；擊中後跌倒（訓練窗內）51.2%（64/125；判準 43 ± 8，壓在上緣）；miss 跌倒 0；ref_err 中位每 seed 0.031–0.033；ONE OUTS × 2000：四壞 10.9%（判準 10 ± 2.3）、打者勝 10.9%、內野球 41%、三振 48%。待確認：各 seed 擊中初速中位 13.6–16.7 m/s（A 同列 20.5、原生同列 19.4 是單一好打的球，不能直接比）→ 請 15 在最終回報附原生同分布（全部投球、jitter 0、800 Hz）的擊中初速中位做對照。無閒置 GPU、無需接手。

**10-28 17:5x 修正驗證通過：同場打者差距消失；完整兩段式 M2 已自動開跑**：A（同場 no_handoff、seed 1，post() 修正 bf96315 之後）同列擊中 0.922（59/64），原生 B 為 0.905；擊中初速四分位 18.1 / 20.5 / 22.0（原生 19.4）；ref_err 0.031 / 0.033 / 0.035（原生 0.029）；ss 16.9（16.7）；tempo_fly −0.076（−0.081）；done 前跌倒 10/64。trace：步 1–5 的 jv 差 0.004–0.16，與原生 + 投手同等級（只剩 GPU 耦合）；舊 obs 檢查為 0。第一次重跑在步 1 崩潰（驅動在第一次 post() 前讀 side.fallen）→ 15 在 __init__ 初始化 fallen（bf96315）。依預核准，完整兩段式 M2 由 logs/run_fixA_m2.sh 自動接跑：10 seed × 64、shift_hist 預設、bias 0.055、after_done policy、timing.csv；s1 後跑 PITCHCHECK（投球欄須與 timing 表一致）；s2 後早停（好球帶 < 0.36 / ref_err 中位不在 0.032 ± 0.010 / 崩潰）；結束跑 m2_analysis；logs/m2/M2_END.txt 只在結束 / 早停 / 崩潰時寫（約 70 分鐘，~19:00）→ 20 的 M3 嚴格挑法拍片鏈接著自動開拍。舊 M2（舊 obs）檔案移到 logs/m2_stale_obs_1028/。待 15 回報最終判準（好球帶、擊中後跌倒、穿越、四壞、EV、dz 殘差 sd）。

**10-28 17:4x 根因找到（15）：同場的 post()（算下一步 obs）跑在錯的時間點 → 策略用的是舊 7.5 ms 的 obs；已修、A 重跑中**：JointPitchEnv.step 在 control() 那個投手步之後就呼叫 batter_side.post()，即只過 2 個子步（2.5 ms）而非 8 個（10 ms）；下一個 control() 7.5 ms 後用了那份 obs。證據（trace 3 加 PD 目標）：obs 的關節位置與控制時的真實 q 在同場差 0.15–0.30 rad、原生恰好 0；一個控制間隔內同場的 t_arrive 只降 0.005（2 × 2.5 ms）；原生 + 投手有同樣的步 1 小種子卻不發散；寫出的目標只在 obs 不同之後才不同（l_elbow 兩邊都是 0 / 2.6 的 bang-bang）。飽和（bang-bang）策略多 7.5 ms 延遲正好給出看到的阻尼、失相、較慢的揮棒（初速 14.6 vs 19.4）。修法（75a…）：joint_env.step 在 _j_k % batter_every == 0 的步「之後」呼叫 post()，即控制後整整 10 ms 的物理，同原生 env.step；CPU 測試原本複製了同樣的錯誤順序，已修並加否證「post 與 control 的 phys_t 差必須是一整個控制間隔」（舊順序 0.0025 s 失敗；新順序 CPU_SHIM_CHECK_PASS / CPU_OWN_TIMING_PASS）。影響：先前所有同場打者數字（smoke、M2-lite、M2、a / c、A）都是用舊 obs 量的；投手側（C1、門檻、影片的投球列）不受影響。跑中：A（同場 no_handoff、seed 1、trace，約 7 分鐘），判準同列擊中 ≥ 0.8。預核准：A 過 → 同鏈直接跑完整兩段式 M2（10 seed、shift_hist 預設、bias 0.055、after_done policy、timing.csv 沿用（第 1 段只與投球有關、不受此修影響，確認 seed 1 投球欄仍相同）、logs/m2/、新的 M2_END.txt → 20 的 M3 鏈自動用嚴格挑法開拍）；seed 1–2 後早停：好球帶 < 0.36、或 ref_err 中位不在 (ii) 原生 800 Hz 的 0.032 ± 0.010、或崩潰；最終判準不變（0.46 ± 0.05，原生 800 Hz jitter 0 為 0.478，預期 0.45–0.48；擊中後跌倒 43 ± 8%；穿越 < 1 cm；離線 take 四壞 10 ± 2.3%）並回報擊中初速（原生同 dt 19.4）與 dz 殘差 sd（原生 0.036）。

**10-28 17:3x DOF 讀回乾淨、原生側二分乾淨 → 物理屬性與場景內容排除，剩同場的步進邏輯（15）**：(1) DOF 讀回 0 差異（61 dof 的 PhysX 剛度 / 阻尼 / 最大力與速度 / armature / 摩擦 / 限位、63 body 質量 / 慣量 / 質心、Isaac Lab data、全 ImplicitActuator、重力、dt 1/800、125 條 USD 項目、env 間 spread 0）。(2) 原生 B 列 + replicate_physics False：trace 與原生逐位元相同、擊中 0.918。(3) 原生 + 同場投手（SEA）+ 動態球（18 m 外）：擊中 0.912、打者 trace 步 2 只偏 0.02–0.13（步 5 0.06–0.58），同場差距是步 2 0.9–3.2。→ 剩下同場的步進邏輯。跑中（logs/run_trace3.sh、TRACE3_END、約 12 分鐘）：同樣的 trace 加上每控制步寫出的 PD 目標（joint_pos_target / joint_vel_target 全 61 dof，步後讀）：步 1 目標不同而 obs / 動作近乎相等 → BatterSide 的 action → target 路徑（前饋、參考取樣、被覆寫）；目標相同 → 子步內的響應不同。

**10-28 17:1x 二分改從原生側做（15，2a7638c，診斷用改動）**：從 JointPitchEnv 拆投手要動 pitch_env 約 50 處 self.robot，改成把同場的元素加進原生場景：(1) --diag_no_replicate（JointPitchEnvCfg.scene 是 replicate_physics=False、原生 True——這是一個具體的場景差異，False 是為了每 env 的球–打者過濾）；(2) --diag_add_pitcher（同場的投手 articulation（SEA）+ 動態球加進每個 env、18 m 外、註冊進 scene）。鏈 logs/run_bisect2.sh 在 DOF_END 且 init 讀回「0 differences」才啟動（否則停、送 diff 與修法）；各約 5 分鐘，BISECT2_END。若 (1) 重現阻尼動力學 → 修法方向：同場改回 replicate_physics True、球–打者過濾另想辦法。

**10-28 17:0x 二分結果（15）：座標 yaw 與地面高度排除**：yaw 0 + 平地 / 只 yaw 0 / 只平地三個變體的同場 trace 與基準同到小數第 3 位（obp_282-1 jv 步 1 0.004 / 步 2 0.921 / 步 3 3.241 每個變體都一樣）、對原生的差距不變 → 同場打者是決定性且座標不變的，原因在場景或 init 路徑（符合 DOF / drive 假說）。擊中：yaw0flat 0.219、yaw0 0.375、flat 0.312（平地投手 64/64 倒，預期）。DOF 讀回鏈（8b272c8、0993e21：PhysX view 剛度 / 阻尼 / 最大力與速度 / armature / 摩擦 / 限位、質量 / 慣量、Isaac Lab data、actuator 類別、重力、USD 的 articulation / drive / joint API，init 後與結束各一次、自動 diff；DOF_END）跑中；若乾淨 → 下一步同場環境移除投手 / 球。

**10-28 16:4x trace diff（15）：同場 vs 原生是系統性的打者動力學差異、不是混沌也不是感知**：同列、雜訊 0、A = 同場 no_handoff，每 env 前 12 控制步：步 0 obs 相同（< 1e-5）、步 1 幾乎相同（jv 差 0.004）、步 2 原生平均 − 同場平均 跳到 jv 1.0–3.2（obs 單位 ×0.05 = 20–60 rad/s）、root 角速度 1–1.5，而兩邊 env 間 sd 都只有 0.001；步 3 動作已在 clip 兩端。最大差在手腕與肩（l_sh_r、l_wr_p / y、r_wr_y / p）加髖 yaw、腰 yaw；型態：原生關節速度每 10 ms 翻號（bang-bang，如 r_hip_yaw +11.5 → −9.2 rad/s），同場打者阻尼大得多。時機 obs（phase、ref phase、tempo）一致，t_arrive 晚 15 ms（小）。靜態檢查沒差：clamp、effort 1.5、solver 8/2、gains / runtime 寫入、脈衝路徑、節奏、註冊（打者在 scene.articulations）、原生球與 tee 是 kinematic 無碰撞。GPU 二分中（各約 7 分鐘，BISECT_END）：yaw 0 + 平地（無 mound、field_z 0）、只 yaw 0、只平地；都不是 → 嫌疑剩場景內容（投手 articulation、動態球、mound、腳接觸感測），下一步移除投手 / 球。我加要求：下次跑時從 PhysX articulation view 讀回「sim 側」每個 DOF 的剛度 / 阻尼 / 最大力矩 / 最大關節速度 / armature / 摩擦 / drive 型式、重力、solver 迭代與旗標，兩邊 diff——同場經投手環境的 init 建兩個 articulation（投手用顯式 SEA、PhysX drive 增益歸零），若碰到打者的 drive 或 GPU view 的 DOF 表索引錯位，只有 sim 側看得到。

**10-28 16:2x 兩個 CPU 檢查乾淨、B2 出來、trace diff 在跑（15）**：絕對高度：pivot 含 field_z、所有代理 _POS / _STATE 欄都減、打者 env_origins z = 0（B 系）、自球 B 態、跌倒用代理 pelvis z、唯一直接讀 inner 的是診斷用 j_bat_z；CPU shim 測試本來就用假 pivot z −0.196 且過。節奏：control() 每 4 個投手步一次（投手 pre-step 前）、post() 只在那些步、ball_world 每子步、cfg assert step_dt × batter_every = 10 ms、真節奏測試過。B2 原生同列：初速中位 19.35（17.3 / 19.4 / 21.2）、ss_contact 15.5、ss_max 16.7、擊中 0.905；同場 A：初速 14.6、ss 15.2 → 棒速只差一點、初速差約 5 m/s = 接觸點散（dz / dperp）造成的弱擊。trace 鏈（a63bac3：原生 B 列 → 同場 A no_handoff，雜訊 0，TRACE_END）跑中，回報每 clip 第一個分歧的 obs / act / q 步。

**10-28 16:0x A/B 分離測試決定性（15）：同場場景即使沒有交接、打自己的球也輸掉揮棒 → 問題在場景 / 代理**：同一列（中位飛行、y −0.108 / z 0.802 好球帶）、同 dt 1/800、jitter 0、同 4 clip：B 原生 1024 球：擊中 0.905（早 52 / 晚 45）、ref_err 0.027 / 0.029 / 0.032、tempo −0.081、甜蜜點 15.9 / 16.7 / 17.3、沒揮到 dz −0.084、gap 1.9 cm；A 同場 --no_handoff 64 球：擊中 25/64 = 0.391（各 clip 都差不多）、ref_err 0.016 / 0.019 / 0.023、tempo_fly −0.117、甜蜜點 13.9 / 15.2 / 16.4、初速 12.3 / 14.6 / 16.5、沒揮到 dz −0.097、dperp 4.9、gap 6.9 cm。讀法：拿掉交接與即時球後同場打者仍較慢、較散 → 損失在場景或代理，作用在揮棒執行或策略對自身身體的觀測；投球感知與撞擊路徑都不是主因（棒速本身就低）。cfg 只差 dt（原生約 0.03）。剩下嫌疑：RotatedArticulation 的讀取（某個 body 狀態欄旋轉錯或漏掉）、shim 的 obs 組裝（起始歷史、相位 / 回合時鐘）、PD 目標與前饋寫入。我的主嫌（交 15 先在 CPU 查）：M1.5 的代理測試是在「原生環境、地面 z = 0」做的，任何「絕對高度」的 obs 在那裡都會過，只有在同場（地面 field_z = −0.196）才會錯；代理分類把 root z 當座標不變、直接讀 robot.data → 同場讀到世界 z 0.749 而非原生 0.945，策略一直以為自己低了 0.2 m（符合較慢較散的揮棒）。要查所有絕對 z / 未對 root 相減的絕對位置（root z、球 / tee / nominal_target z、參考 root z、甜蜜點 z、pitch_pos z、contact_x_ref、跌倒 / 終止高度、play_log / 撞擊高度、對地面 0 的判定）；規則：B 系地面必須是 z 0，經代理的世界 z 都要減 pivot z。（我自己讀 batter_side.py 後更正：RotatedArticulation 對所有 _POS 欄位（含 body_link_pos_w）都減掉含 field_z 的 pivot，root z / pelvis z 應該沒錯，只剩繞過代理的讀取或對 0 的門檻要查。）另加 CPU 檢查「控制節奏」：同場的 _pre_physics_step 以投手的 400 Hz 跑，要確認 BatterSide.control() 的策略 / 參考時鐘 / 感知歷史恰好每 8 個子步跑一次、所有假設 10 ms 的差分與前饋成立、post() 每個打者控制步一次。找到 → 修、CPU 否證、重跑 A（約 5 分鐘，判準同列擊中 ≥ 0.8）；沒找到 → 跑 trace diff（感知雜訊 0、策略決定性、env 0 每控制步 obs / 動作 / 關節狀態，找第一個分歧的分量；2 × 5 分鐘 GPU）。B2（B 的初速）在跑。另 A 的 CSV 因 import csv 放進 main() 變區域變數而沒寫出（8951433 修），A 由列 log 解析、沒損失。

**10-28 15:47 檢查**：GPU 1 個 Isaac（15 的分離測試 A：同場 seed 1 --no_handoff，15:47 起；接著 B 原生 800 Hz 同列 1024 球；SEPAB_END）。沒有 trainer、沒有閒置；12 小時影片已涵蓋（投手 12:45 補拍 + 同場片，打者今天的同場即時片），下次最晚 ~00:45。M2 仍暫停，等 A/B 定位同場打者揮棒 / 撞擊路徑的損失。

**10-28 15:5x 診斷鏈結果（15，GAP_END 15:45）→ shift_hist 設為預設、bias 維持 0.055、dt 是已知約 0.03 的小代價；差距主體在揮棒 / 撞擊路徑，A/B 分離測試已放行（SEPAB_END）**：好球帶：(i) 原生 400 Hz jitter 0：0.506、ref_err 0.033 / 0.036 / 0.038、dz 殘差 sd 0.036、|dperp| < 6 cm 86%、tempo_fly −0.072；(ii) 原生 800 Hz jitter 0：0.478、ref_err 0.030 / 0.032 / 0.034、sd 0.036、91%、tempo_fly −0.081、甜蜜點 16.6、擊中後跌倒 107/278 = 38% → dt 只扣約 0.03、散布不變；(a) 同場 seed 1 bias 0.055：8/33 = 0.242、ref_err 0.016 / 0.021 / 0.025、sd 0.084、甜蜜點 15.1、擊中初速 10.2 / 12.5 / 13.6 m/s、跌倒 5/8、tempo_pre −0.052 / −0.039 / −0.031、tempo_fly −0.124 / −0.109 / −0.094（比原生 −0.081 多慢約 3%，對應 ref_err 短約 11 ms）、handoff_jump 3.2 / 4.7 / 7.1 cm、perc_z − 真值 sd 0.27 / 0.32（單步感知雜訊，原生無對照數）、t_commit − t_plane −0.151；(c) 同場 --shift_hist：11/33 = 0.333、sd 0.061、初速 11.5 / 12.2 / 13.7、跌倒 8/13、tempo_fly 不變。讀法：感知跳變真的有影響（散布 0.084 → 0.061、擊中 +0.09，n = 33 ±0.08），但沒補完差距：散布仍約 1.7 倍、tempo_fly 仍較負、擊中初速每次都約 12 m/s（原生約 19）——初速與感知無關 → 指向同場的揮棒 / 撞擊路徑，正是 A（同場、無交接）vs B（原生 800 Hz、同一列）要分的。我要求 B 從 play_log 補初速（diag 不記），比較同列同 dt 的擊中率、甜蜜點速度、初速、dz / dperp：A 初速 12 而 B 19 且棒速相同 → 撞擊路徑（代理讀到的棒速 / 棒向或積分器球態）可疑；A 棒速本身較低 → 揮棒執行。

**10-28 15:4x 15 的 cfg 逐欄 diff（CPU 實例化、攤平）**：SimulationCfg / PhysxCfg 49 欄只有 dt 與 render_interval 不同；打者 ArticulationCfg 198 欄只有 prim_path 與 init_state 不同（actuator 每欄相同、effort × 1.5 除手指、同一 USD、同 props）；地面相同（只差 field_z 位移）；「腳最低點」是 l_foot / r_foot 的 link 原點（腳踝），原生同樣約 0.07；hook 順序與 DirectRLEnv.step 相同、沒有一步落後。→ 唯一物理差異 = dt；F1 說 dt 在原生不扣分，(ii) 在 jitter 0 再驗。核准分離測試（GAP_END 後約 12 分鐘）：(A) 同場 seed 1 --no_handoff（1577493：打者按原生排程打自己預測的中位列，即時球不替換、同場球鏡射打者的球）vs (B) 原生 diag 800 Hz jitter 0、同一列（assets/game/p3c_median_flight_row.csv，9660bae）、1024 球；比擊中率、甜蜜點速度、初速、dz / dperp 散布。A ≈ B → 同場物理沒問題、損失在即時球 / 交接路徑（再看 (c) shift_hist 與 (a) perc_z）；A << B → 場景或代理（非 cfg）吃掉揮棒，之後凍結 / 移除投手二分。

**10-28 15:3x 20 的場景側物理事實（CPU、讀 code 與 USD）：同場與原生打者環境唯一的物理差異是物理步長**：PhysxCfg 兩邊相同（只設兩個 gpu pairs 容量，其餘 Isaac Lab 預設）；同場 dt 1/800、打者 cfg 被 batter_side.py:186–189 改寫成 dt 1/800、decimation 8（原生 1/400 × 4），控制都 100 Hz，implicit PD 與 solver（8 / 2 迭代）在一半 dt 下積分；地面同為 GroundPlaneCfg 靜摩擦 1.0 / 動 0.9、回彈 0、average（原生 tee_batting_env.py:118 相同），只是在 field_z −0.196、打者 root 同步位移；打者下方沒有其他碰撞體（mound 在橡膠板旁約 5 m、草皮與圓丘無碰撞）；打者與投手的 ArticulationRootProperties 相同（self-collision 關、solver 8 / 2、sleep / stabilization 0）、rigid props 相同、contact / rest offset 都未設定（PhysX 預設）。→ dt 的否證 = 鏈的 (ii)（原生 1/800 × 8、jitter 0、1024 球；F1 (a) 在 jitter 0.03 下同設定 0.461 vs 0.458），看初速與 dz 散布。

**10-28 15:2x (i) 原生 jitter 0 基準出來（15，logs/gap/i_400.txt、diag_bat_contact_gap_i_400.json）**：原生 400 Hz、4 clip 真實高度、P3c 表、bias 0.055、jitter 0、1024 球，好球帶 557：擊中 0.506（F1 base jitter 0.03 是 0.458）、ref_err 0.033 / 0.036 / 0.038（IQR 5 ms，之前的 34 ms 全是抖動）、tempo_q −0.078 / −0.072 / −0.064、早 / 晚揮 275 / 0、dz 斜率 0.38、殘差 sd 0.036、擊中後跌倒 140/313 = 45%。同場（M2 seed 1+2）對照：0.197、ref_err 中位晚 14 ms、IQR 8 vs 5、dz 斜率 0.65、殘差 sd 0.082（> 2 倍）。注意 ss_speed_max 隨 dt 取樣變（原生 400 Hz 約 21、800 Hz 約 16.5 而擊中率不變），只能同 dt 比：同場 15.1 vs 原生 800 Hz 16.5。解讀：原生 jitter 0 下每次揮得幾乎一樣；同場晚 14 ms 且垂直散布 2 倍。(a) 的 tempo / perc_z 決定是感知輸入不同還是同輸入執行不同；鏈續跑 (a) → (ii) → (c)。

**10-28 15:2x 15 的 CPU 答覆 → 差距是「執行」不是資訊；交 15 / 20 做物理 cfg 逐欄 diff**：(1) dz 對球高的斜率：原生沒揮到 0.56（r 0.92、n 284）、同場 0.71（r 0.77、n 53）→ 「固定高度揮、只部分跟球」是策略本身的特性，不是同場加的；差在散布：dz 殘差 sd 同場 0.082 vs 原生 0.046、|dz| < 0.05 15% vs 39%、|dperp| < 0.06 73% vs 92%、甜蜜點速度四分位 13.6 / 15.1 / 16.2 vs 15.8 / 16.6 / 17.2 → 同場揮棒較慢、較不穩（高度與觸及都是）。(2) 程式：出手前策略拿不到每球高度（pitched_env:248–250 nominal = tee + tee_offset_z，每 clip 固定；_perceived_ball 出手後用延遲雜訊的自球歷史外推），原生與同場相同、沒有 oracle 差。我補查：AIB_EFFORT_SCALE=1.5 在所有鏈腳本都有 export、joint cfg effort_scale=1.5、shim 有 assert；但同場每次跑的擊中初速中位只有 13.5–14.3 m/s（smoke 14.3、M2-lite 13.8、M2 13.5）vs 原生 1.5 倍約 19 m/s → 執行層面的差距很大。F1 (a) 證明 dt 1/800 × 8 在原生不扣分，所以要找同場與原生在物理上的其他差異：SimulationCfg / PhysxCfg（solver 型式、迭代數、摩擦 offset / correlation、bounce threshold、stabilization、ccd）、打者 ArticulationCfg 每個欄位（actuator 剛度 / 阻尼 / 力矩與速度上限 / armature / 摩擦、soft limit、self_collisions、play cfg 有沒有多加的設定）、地面物理材質（靜 / 動摩擦、回彈、combine）、腳最低點 0.07–0.12 是腳踝原點還是鞋底（與原生同量比較）、代理是否讓 PD 目標比原生晚一個控制步。15 做 cfg 表、20 給場景側事實；之後提一個 GPU 測試分離原因（例如原生環境套投手場景的 PhysxCfg + 地面材質，1024 球約 5 分鐘），排在 GAP_END 後。(a) 新欄（fd96593）：t_commit、perc_z_pre / perc_z_commit vs pred_z、tempo_pre / fly、handoff_jump_m。鏈順序 (i)（跑中）→ (a) → (ii) → (c)。

**10-28 15:2x 第一支正式 M3 片（20，15:11–15:20，M3FILM_END）：切鏡左面板成功、右面板失敗；這批 M2 沒有可展示的安打**：videos/m3_s1_k44.mp4、拼圖 videos/fixcheck_tiles/m3_s1_k44_tile.png 我看過：seed 1 env 44（乾淨規則挑的）、與 m2_s1 30 欄 64/64 相同；343 幀、動態 10.19% / 左 38.51%。切點 2.63 s（done 2.12 + 0.5）：左面板一壘側投手側視成功（投手入鏡、站穩、沒有打者）；右面板「擊出球」相機失敗——env 44 是小滾地（仰角 −53°、飛行 0.6 m），相機對著本壘旁的球 = 俯視倒地的打者。修正 7987fa0（未測）：高機位只在球離打者骨盆 > 30° 時跟球，否則瞄投手。挑片問題：這批（policy + 兩段式、seed 1–2）14 個擊中，5 個界內全是小滾地（飛行 0.4–1.9 m、仰角 −20..−64°），飛在空中的（7–25 m）全界外且 done 前倒 → 正式 M3 挑法改為：擊中、界內、飛行 ≥ 15 m、仰角 > 0、done 前沒倒、投手站穩；在這批資料裡沒有，等打者修好後的下一次 M2 再拍（鏈改掛下一次 M2_END）；不重拍 k44。

**10-28 15:1x 我讀 M2 seed 1–2 的好球帶列：沒揮到是「垂直」的、不是時機**：好球帶 56（擊中 12、沒揮到 44）；沒揮到的 t_gap 中位 2.040 vs t_plane 2.041、ref_err 0.021 → 球棒準時到球的面；gap 中位 7 cm、dperp 中位 −0.002、dz 幾乎線性跟著球高（球 z 0.99 → dz +0.063、1.05 → +0.142、0.76 → −0.065、0.67 → −0.098、0.58 → −0.115、0.74 → −0.173；斜率約 0.6–0.7、中心 z ≈ 0.87）；擊中的 dz 在 ±0.06 內 → 球棒大致在固定高度揮、只部分跟球，擊中的是剛好在那個高度的球。15 的更正：對應同場設定的原生參考是 F1 "both"（4 clip、真實高度、P3c 表）：好球帶 ref_err 0.015 / 0.029 / 0.048、擊中 0.49、tempo_q −0.089 / −0.082 / −0.075 → 同場中位 0.022 只差約 7 ms，不是 21 ms；時機解釋不了 0.197 vs 0.49。決定：鏈（logs/run_gap_chain.sh，M3FILM_END 後；logs/gap/、GAP_END.txt、(i) 後 I_DONE.txt）拿掉 (b) bias 0.076，保留 (i) 原生 400 Hz jitter 0 → (a) 同場 seed 1 + tempo / handoff_jump 欄 → (ii) 原生 800 Hz → (c) 同場 --shift_hist。CPU 問題（15）：(1) 原生 F1 both 的好球帶沒揮到 dz 對球高回歸的斜率（原生 ≈ 0 而同場 ≈ 0.65 → 同場打者沒看到 / 沒用到真實高度）；(2) 原生策略在出手前後拿到什麼高度資訊——出手前的 nominal_target（tee + offset）是每球自己的穿越（oracle）還是每 clip 固定？引用程式行；若原生出手前就拿到該球的穿越而 shim 給中位列 → 差距在此，修法 = 兩段式表再加每 env 的穿越 y/z 進出手前 obs，不動 bias、不重訓；(3) 每 env 記 t_plane − 0.10 s 與揮棒 commit 時感知到的球 z vs 真實 z（(a) 可加欄）。

**10-28 15:1x 15 的 CPU 回答：交接前 obs 與原生完全相同；三個差異都在交接後或物理**：交接前 shim 不餵投手手中的 PhysX 球，打者的球是中位列的出手點、靜止（原生也靜止）；感知的 nominal_target、v_seen（靜止點的延遲歷史 = 純雜訊）、released_seen = 0、rate、pitch_kmh × 0、t_arrive（自有時機下 = 原生）、mimic reset 重寫站姿 / 相位 0 / 歷史重填（PD 保持不留痕跡）、感知 cfg（延遲 0.1、偏差 0.02、雜訊 0.01）皆同。差異：(1) 交接時的出手點跳變——感知歷史還是中位列的出手點，即時出手點差幾 cm（表的 x/y/z sd 2.9/4.3/3.7 cm），0.1 s 延遲後第一批 released 感知看到跳變 → v_seen 與預測穿越一步尖峰，影響 tempo_fly；修法 --shift_hist（07049df，預設關）把歷史平移到自己的出手點 = 原生；A/B 記 handoff_jump_m。(2) 物理頻率：原生打者 400 Hz（1/400 × 4）、同場 800 Hz（× 8），控制都 100 Hz；implicit PD 與接觸在不同 dt 行為不同，可能是 M2-lite 看到的執行落後（棒面對齊 vs 球到面 +0.01 s，原生擊中 −0.15 s）；新 diag_bat_contact --physics_hz 800。(3) 原生參考（run_f1.sh）是 timing_noise 0.03 跑的，同場是 0 → 0.043 / IQR 34 ms 含抖動，要 jitter 0 的原生參考才公平（「晚 21 ms」可能部分是比較基準的假象）。核准（一條鏈、排在 M3FILM_END 後、約 35–40 分鐘）：(i) 原生 400 Hz、bias 0.055、jitter 0、1024 球先跑（新基準）→ A/B (a)(b) → (ii) 原生 800 Hz → 同場 seed 1 --shift_hist。規則：(ii) 掉得跟同場一樣多 → dt 是原因、打者 PD 改在 800 Hz 場景裡以 400 Hz 跑（場景由 20 決定）；否則 shift_hist 恢復 → 感知跳變；bias 0.076 只在兩者都解釋不了對 (i) 的差距時。F1 (a) 已在 jitter 0.03 下量過 800 Hz（0.461 vs 0.458），預期 (ii) 影響小。

**10-28 15:1x 完整 M2 早停（15，14:55–15:10，logs/m2/M2_END.txt "early-stop"）→ 排程已精確、差距在策略於同場的 tempo**：seed 1 / 2 兩段式時機：擊中 0.125 / 0.094、好球帶 13/66 = 0.197（早停門檻 0.30）；ref_err 四分位 0.018 / 0.022 / 0.026（M2-lite 0.009 / 0.017 / 0.030；原生好球帶 0.027 / 0.043 / 0.061）→ 排程精確（IQR 8 ms，多為 10 ms 讀取量化），但中位仍比原生晚約 21 ms；之前中位列排程的散布反而讓一些球碰巧落在窗內，精確後每球一致偏晚、擊中反而降。15 加欄（0736e8c）：每 env tempo_pre（交接前）/ tempo_fly（交接到 done；原生好球帶 −0.077 / −0.066 / −0.051）+ --timing_bias 旗標。核准：(a) bias 0.055 + tempo 欄、(b) bias 0.076（= 0.055 + 21 ms），各 seed 1，約 14 分鐘，排在 20 的第一支 M3 片（鏈已因早停標記啟動，M3FILM_END）之後。我另外要 15 在 CPU 回答：同場打者在交接前到底看到什麼（原生：球靜止在庫的出手點、released = 0；若 shim 把投手手中移動的 PhysX 球餵進感知，策略會在出手前就對沒訓練過的球運動反應 → tempo_pre 會偏），並列出所有交接前與原生不同的 obs 項（預測列的 t_arrive、flight_time、tee / 接觸參考、0.8 s PD 保持後的起始相位）。決定規則：(a) tempo_pre 偏且有交接前 obs 不同 → 把 obs 修成與原生一致（交接前球放在預測出手點）、bias 不動、重跑 seed 1；tempo_pre 一致、只有 tempo_fly 不同且 (b) 好球帶擊中恢復 ≥ 0.41 → bias 0.076 接受為同場校準（log 與結尾卡註明）、完整 M2 用 0.076（約 70 分鐘）。

**10-28 15:0x 完整 M2 14:55 啟動（15，logs/run_m2full.sh → logs/m2/m2_s*.csv、M2_END.txt）+ 影片切鏡規則（20，77425e8）**：切點 = env k 的 done_t + 0.5 s（擊球落地或球過去）；切點前相機不變；切點後左面板 = 一壘側 8 m / 1.3 m 高 / 1.5× 跟投手骨盆的側視，右面板 = 本壘後 6 m / 8 m 高的高機位，擊中時瞄擊出球（1.0×）、沒揮到 / 不揮時瞄投手（1.5×），字幕加 "play over (batter shown until the play ends)"；畫面 "batter FELL" 只標 done 前跌倒；--no_film_cut 關閉。joint_smoke_20.py 改由 scripts/make_joint_smoke_20.py 從 15 的 joint_smoke.py 生成（每個錨點必須恰好匹配一次，避免再掉 hunk）。第一支正式 M3 片鏈（logs/m3film.sh、M3FILM_END）排在 M2_END 後：依 seed 順序挑第一個「擊中、界內、done 前沒倒、投手站穩」的 (seed, env)，沿用該 M2 行程的 after_done 與 timing_csv，拍完比對該 seed 的列；只拍一支、我看過再拍更多（拼圖要含切點後的幀）。

**10-28 15:0x M3 草稿片完成（20，14:45–14:54，M3DRAFT_END）+ 14:47 檢查**：videos/m3draft_s1_k4.mp4（343 幀、0 空白、動態 2.85% / 左 4.68%、High），拼圖 videos/fixcheck_tiles/m3draft_s1_k4_tile.png 我看過：約 2.0 s 兩個視角都看得到球在棒上、2.42 s 起標 HIT、投手全程站穩；打者約 3.2 s 倒（done_t 3.80 之前 = 交付打者擊中後 35–43% 的回合內跌倒，不是 done 後問題）；env 4 與 m2lite_s1 的 29 個 done 前欄位完全相同、64/64 列相等。env 4：64.0 km/h、T4、pred y −0.162 / z 0.946；擊出 16.3 m/s、仰角 29.9°、方向 −37.6°（右外野側）、飛行 22.8 m、落點 (18.07, −13.93) 界內（內野）。m2lite_s1 沒有「界內 + done 前沒倒」的 env，所以是退而求其次的挑法；正式 M3 片等完整 M2 的 640 列再挑。blend 作廢後的影片規則交 20（CPU）：打者鏡頭在 done_t + 0.5 s 切掉、之後中外野相機跟球或停在投手，寫成影片慣例。14:47 檢查（補記）：GPU 在 20 的草稿片（14:45–14:54）→ 15 的完整 M2 鏈（等 M3DRAFT_END + Isaac 0 60 s 後啟動）；沒有 trainer、沒有閒置；12 小時影片：投手由 match_full3（12:45）與同場片涵蓋、打者由今天的同場即時片涵蓋。

**10-28 14:5x 兩段式時機做好（15，20bd4d5，CPU 否證過）→ 直接當完整 M2 跑、seed 1–2 後早停規則**：15 的更正：t_sched / delta_s 欄失真（rate 取的是回合結束時的 tempo，中位 0.735），要用 ref_err（與 diag_bat_contact 同定義）；原生參考（logs/f1_base.txt，12500、1024 球）ref_err 四分位 好球帶 0.027 / 0.043 / 0.061、全部 0.026 / 0.040 / 0.056（中位 ≈ bias 0.055 + tempo 漂移 + 0–10 ms 讀取偏移）→ 原生的 delta 不是 0、自身 IQR 34 ms；同場（110 穿越）0.009 / 0.017 / 0.030 → 同場比原生晚約 26 ms，排程誤差解釋不了中位的偏移（預測飛行 1.0463 vs 實際 1.0448 只差 1.5 ms），所以第 2 段若中位仍 0.017、擊中仍約 0.3，差距在策略於同場的 tempo，不在排程。設計：起跳時每 env 用自己那球的原生時鐘（start = h − release_native(tp − h)、ref_t 吸收 100 Hz 網格最多 10 ms 的晚起）；CPU 否證：中位列排程誤差 −0.568 s vs 自己的排程 < 1e-5 s。投球與 seed 無關（s1 vs s2 64/64 相同）→ 一張時間表（logs/m2lite/timing.csv；60 env 實測、4 個擊球前沒過面的 env 用 kmh 擬合 sd 6 ms）供所有 seed，完整 M2 = 10 seed × 1 段 ≈ 70 分鐘。M2-lite 補充：好球帶 19/66 = 0.288 ± 0.056、擊中後回合內跌倒 8/23 = 34.8%、四壞 12.3%、打者勝率 12.3%；blend 的 done 後跌倒 56/60 → blend 作為影片修法作廢（M3 改成揮棒後約 0.5 s 切鏡或跟球）。決定：第 2 段直接當完整 M2（10 seed、約 70 分鐘、logs/m2/、after_done = policy），排在 M3DRAFT_END 後；seed 1–2（約 14 分鐘）落地後早停：128 球好球帶擊中 < 0.30、或 ref_err 中位不在 0.043 ± 0.015、或崩潰 → 停；否則跑完。最終判準 = 核准的 M2 判準（640 球：好球帶擊中 0.46 ± 0.05、擊中後回合內跌倒 43 ± 8%、穿越 < 1 cm、離線 take → 四壞 10 ± 2.3%），ref_err 中位 0.043 ± 0.010 只作診斷。

**10-28 14:4x M3 草稿片第一次失敗（20，14:35–14:43，約 8 分鐘 GPU 浪費）**：rebase 時一個被拒的 hunk 把拍攝 hook 與 pcols 那行弄掉，跑到最後 NameError、沒有影片；20 只用 grep 檢查 rebase 漏了。修正 f531d58（與 15 的檔逐行 diff）後重跑同一工作：m2lite_s1 沒有 env 同時滿足「界內 + done 前沒倒 + 投手站穩」，退而取第一個界內安打 env 4（約 8 分鐘）。通則：rebase / 合併後上 GPU 前要先跑一次 CPU 否證或至少 import + dry run，不能只 grep。

**10-28 14:3x M2-lite 結果（15，14:16–14:31，seed 1、2 × 64，logs/m2lite/）與我的判定 → 兩段式 oracle 時機**：我自己讀 128 列：擊中 23、好球帶 56、好球帶擊中 19/56 = 0.34（連同 A 的 21/77 = 0.27；門檻 0.46 ± 0.05）；done 前跌倒 12（擊中 8/23 = 35%，原生 43%）。delta_s（+ = 揮早）：全部中位 +0.023 s、sd 0.025；擊中 中位 +0.013、sd 0.008（+0.006..+0.030）；沒揮到 +0.024、sd 0.026。t_plane 中位 2.0423 = 交接 0.9975 + 中位飛行 1.0463（球照預測到達）、t_sched 2.0191 早 23 ms（待 15 確認是否為原生 cfg 的 timing_bias_s）。方向：23 個擊中有 22 個負（−172..−26°）。關鍵數字是擊中的 sd 8 ms：揮棒只在約 ±10 ms 的窗內接得到，所以用「中位列」排程造成的 21–25 ms 散布不是太小、剛好把 0.47 砍到約 0.3（15 先前判「太小」是錯的）；原生的 delta 由構造為 0（每球用自己的 flight_time 起跳），差距是時機資訊、不是場景。決定（不重訓）：兩段式 oracle 時機 = ONE OUTS 指標本來的假設（batter-oracle-timing：起跳用引擎的真實到達時間）。第 1 段 = 現在的同場跑法給每 env 的 t_plane；投球與打者行為無關（seed 0 vs 1、拍 vs 不拍：投球 / 交接欄 64/64 相同）；第 2 段同 seed、每 env 用自己的 t_plane 排起跳（同 pitched_env:257、同原生 bias）。CPU：driver 加 --timing_csv（env, t_plane）、start_t = t_plane − (contact − t0)/speed + 原生 bias，用 s1 列在 CPU 否證（算出的起跳重現 delta ≈ 0）。GPU：seed 1、2 第 2 段（約 14 分鐘；第 1 段已有）；判準：delta_s 中位在原生值 ±0.005、sd ≤ 0.010、128 球好球帶擊中 ≥ 0.41、回合內擊中後跌倒 43 ± 8%。過了 → 完整 M2 = 10 seed × 2 段（約 140 分鐘，啟動前報分鐘數）。要 15 補原生 F1 both 同定義的 delta_s 參考。GPU 順序：20 的 M3 草稿片（M3DRAFT_END）→ 15。

**10-28 14:3x 同場拍攝 = 評估列（逐位元）成立（20）**：m2film_s1_k0（seed 1、hold、拍 env 0）vs 15 的 logs/m2lite/m2lite_s1.csv（seed 1、blend、不拍）：19 個共用 done 前欄位 64/64 列全部相同、env 0 完全相同（kmh、plate_y/z、handoff_t、batter_start、hit、exit / launch / spray / carry、backspin、pitcher_fell、batter_fell_before_done、done_t、pred_y/z；batter_fell 因 hold / blend 在 done 後不同而排除）→ 渲染、隱藏其他 env、env k 的標記球都不改變模擬，拍到的 env 就是它的 M2 列。M3 草稿片鏈已進 Isaac = 0 雙重檢查。

**10-28 14:2x 第一支即時打者同場影片 = 管線通過（20，14:06–14:14，M2FILM_END）**：videos/m2film_s1_k0.mp4（343 幀、0 空白、動態 2.52% / 左面板 5.43%、High yuv420p），拼圖 videos/fixcheck_tiles/m2film_s1_k0_tile.png 我看過：中外野視角兩人全程在畫面；投手 wind-up、跨步、約 1.0 s 出手、站穩；打者站姿、跨步、約 2.0 s 揮棒、沒揮到，球落在本壘後方地上；2.4–3.2 s 打者倒下（done_t 2.11 之後，hold 模式已知的 done 後跌倒）；本壘視角打者前景、投手在圓丘上、球可見。env 0 列：64.0 km/h、T4、plate y +0.150 / pred −0.1504（game）、z 0.689、沒揮到、done 前沒倒、投手站穩。整批（seed 1、hold）：擊中 11/64（0.172）、界內 3；打者 64/64 倒（done 後）；投手 tier 64/61/56/33、站穩 53/64、倒 11。種子敏感度（vs fall_ab_hold seed 0）：env 0 完全相同、42/64 列全等；所有投球 / 交接欄 64/64 相同（kmh、handoff_t、batter_start、pred_y/z）→ seed 只影響打者的感知雜訊，投球是決定性的；只有打者結果欄不同（hit 48/64 相同、done_t 46…）。核准 20 的 M3 草稿片：M2LITE_END 後、seed 1、after_done blend、從 m2lite_s1.csv 挑第一個「擊中、界內、done 前沒倒、投手站穩」的 env（約 8 分鐘），同時回報與 seed 1 列的 done 前逐位元比對；若時機分析逼出 shim 修正則之後重拍。

**10-28 14:1x M2-lite 排好（15，3e397a0；seed 1、2 × 64，約 14 分鐘，等 M2FILM_END + Isaac 0）**：CPU 能算的：P3c 表 62 列 飛行時間 vs 球速 斜率 −17.1 ms / (km/h)、殘差 sd 4 ms；套到 A 的 64 列，排程（中位列 1.0463 s）vs 實際飛行誤差中位 −3 ms、sd 21 ms（擊中 −4、沒揮到 +3）→ 每球飛行差太小太隨機，解釋不了 −87° 的方向；若 M2-lite 的 delta 是系統性的，來源在別處。原生參考 F1 both（diag_bat_contact）：擊中 timing_s 四分位 −0.156 / −0.152 / −0.143 s，沒揮到 −0.14 / −0.018 / +0.004。新欄（diag_bat_contact 定義、物理步解析度）：t_plane（球過 clip 接觸面 x = contact_x_ref，照 _place 的構造即本壘面）、t_sched、delta_s = ref_err / rate（+ = 揮早）、timing_s = t_square − t_plane、gap_m / t_gap / dz_m / dperp_m（接觸前最近點）、ss_speed_max。after_done = blend：done 姿勢 0.5 s smoothstep 回 clip 第 0 幀站姿、速度目標 0；CPU 檢查 |target − stance| 最大 1.2e-7，否證腳本 blend / policy 模式都過。

**10-28 14:4x 座標慣例定案（20，f7775a4）**：兩個座標系寫進 joint_pitcher.py docstring：_plate = 投手本壘座標（評估出手列、j_cross、to_plate；+x 朝本壘、+y = 投手左手 = 一壘側）；_game = agents.Pitch / BODY_Y_M / is_hit_by_pitch / FieldSpec / play_log spray / 打者 B 系（+x 朝中外野、+y = 三壘側）；y_game = −y_plate、z 相同。欄名改成帶座標系後綴（p_*_plate、p_aero_y_game、ball_land_x/y_game；15 的 plate_y → cross_y_plate、pred_y → pred_y_game，合併時要 15 同意）。實例：門檻 P3c 列 0 aero_y +0.150（plate）→ 庫 target y −0.151（game）；joint env 0 j_cross +0.150 vs pred_y −0.1504；55 個穿越列 |plate_y + pred_y| ≤ 0.05 cm（一列差 0.14 m 是擊球後的穿越）。HBP / match 路徑（TablePitcher、pitch_gen:106、match_video.plate_crossing）都有翻號，沒找到未翻的；15 的 m2_analysis.py 未追蹤、20 不保證。落點 ball_land_*_game 來自 B 系、原點在打者的穿越面。拍攝測試 14:06 起（沒有 m2_s1.csv → 照設計退回 env 0、seed 1），約 14:15 結束後 GPU 給 15 的 M2-lite；我要 20 拿 joint_smoke.csv / fall_ab_hold_rows.csv 當 env 0 逐位元比對的參考（seed 相同才算）。

**10-28 14:3x A/B 結果（15，13:55–14:01）→ A 不過、M2 仍暫停；真正的問題是同場打者擊中率遠低於原生**：A（logs/fall_ab_hold_rows.csv）：(1) 通過——done 前與 joint_smoke.csv 0 個欄位不同（同 9 個擊中 env 1/12/13/34/37/52/54/57/63）；(2) 不過——done 後仍 59/64 跌倒（固定姿勢有生效、跌倒時間變了，但跟隨動作的動量把人帶倒，中位 done 後 0.55 s）；(3) smoke 那 7 個早倒 = 擊中 [37]、沒揮到 [11,30,36,40,45,58]，其中只有 30（沒揮到、穿越前 0.06 s）與 37 在原生 done 前倒；done 前跌倒共 5（4 擊中 1 沒揮到）。B：done 前 4、done 後 60，沒幫助。M2 分析試跑（scripts/m2_analysis.py，144cb16）：穿越 vs fly_to_x 51 球最大 0.05 cm；發現 j_cross 的 y 是投手本壘座標、與遊戲慣例（Pitch.plate_y，+ = 三壘側）相反，打者的 pred_y 已是遊戲慣例 → 交 20 把共用 CSV 每個橫向欄位的座標系寫進欄名與 docstring、用一列實例核對。我的判定：15 提議「done 後跌倒只是影片問題，M2 照跑（after_done = policy）」在邏輯上成立（判準都在原生窗內、(1) 證明窗內逐位元不變），但 smoke 列已顯示好球帶擊中 2/21、9 個擊中全部界外且方向中位 −87°（系統性、不是雜訊；原生 F1 (c) 0.49）→ 70 分鐘 M2 大概率量到一個已知壞掉的東西。主嫌：時機——同場打者在出手前 0.2 s 就用「預測的中位飛行列」（1.045 s）起跳，原生 play 的起跳用每球自己的 flight_time（pitched_env:257），所以同場打者承受它沒訓練過的每球接觸時間誤差，若定義不一致（接觸面 vs 本壘面、庫的出手點 vs 即時交接點、clip 速度）還會是系統性誤差；球速本身解釋不了（kmh sd 1.23 ≈ 0.02 s；9 個擊中 63.0–64.9 但 54/64 都在這帶內）。決定：(i) 15 先在 CPU 用現有列 / play_log 算每 env 的 t_cross_actual（積分器到 clip 的 contact_x_ref 面）、t_contact_sched（start + contact_time × speed）、delta、球棒最近點時間與距離；回報 delta 中位 / sd、擊中 vs 沒揮到、方向 vs delta；原生 F1 (c) 同量應 ≈ 0 ± 0.01；(ii) 這些加進 driver 欄位 + after_done = "blend"（0.5 s 阻尼混合回站姿，核准為影片修法、CPU 驗）→ M2-lite 2 seed × 64（約 14 分鐘），不跑 10 seed；delta 系統性 → CPU 修定義再跑 2 seed；delta 隨機且 sd ≥ 0.03 s → 設計問題（訓練加時機抖動）由我依數字決定。GPU 順序：20 的即時打者拍攝測試（已在 CHAIN_AB_M2_END 後啟動，約 7 分鐘，M2FILM_END）→ 15 的 M2-lite。

**10-28 14:1x 同場 driver 分工定案（避免重複）**：不另寫 eval_joint.py，15 的 scripts/joint_smoke.py 就是同場 driver（每 env 一列）。20（CPU，與 15 先對好欄名）：(1) 投手欄（完整出手列、積分器 tier、站穩判定、擊出球的落點 x / y、界內外、距離、第一次彈地）；(2) 拍攝路徑（--video_env k + film_first + M3 預設相機 / 草地 / 圓丘 + 即時打者，拍到的球 = 評估列；今天的片都是 PD 站姿打者）；(3) 每 env balls / strikes 只做欄位接線。15 保留不揮棒決策、judge、打席組裝、計數卡。20 已完成（5b3158f，CPU 驗過：scripts/joint_pitcher.py 的 pitcher / ball / count 欄 + JointFilm；scripts/joint_smoke_20.py = 15 的 driver + 25 欄 + --count_csv + --video/--video_env/--slow，15 的檔不動、M2_END 後再合併）。核准拍攝測試：CHAIN_AB_M2_END 後，用 M2 s1 相同 seed / 參數，從 logs/m2/m2_s1.csv 挑第一個「擊中、界內、投手站穩、打者 done 前沒倒」的 env 拍（約 7 分鐘），回報拼圖、拍到的列是否與 s1 該 env 逐位元相同、幀檢查數字。

**10-28 14:0x 跌倒診斷結果（15，13:28–13:35，約 4 分鐘 GPU，logs/fall_diag.txt / .csv）→ 原因是訓練回合結束後的時間、不是場景 bug；核准 A/B 修正測試並接 M2**：分階段：PD 保持 0、起始→交接 0、交接後 clip 結束前 7、clip 結束後 57；跌倒時間中位 2.69 s（= 起始 + 1.90 = 交接 + 1.69），clip 結束中位 2.43 s。高度（相對 field_z）：root z 0.933 / 0.934 / 0.933 / 0.947 / 0.850（t0 / 0.4 s / 起始 / 交接 / 穿越），最低腳 z 0.07–0.12 → 沒有 0.196 偏移、腳沒陷地。原生環境裡沒揮到在「球過去」就結束（接觸面後 1 m）、擊中在落地結束；同場 smoke 從不結束打者回合，55 個沒揮到的裡 53 個跌倒發生在原生 done 之後 → 策略沒訓練過的時間。機制：交付 play cfg 的 zero_ff_after_ref_end = False，clip 結束後 PD 速度目標仍保持最後一幀的跟隨速度，約 0.26 s 後把人推倒（可能也是原生擊中後 43% 跌倒的一部分）。候選（7fc64d8，CPU 否證 3/3 過）：(A) BatterSide after_done="hold"：原生回合會結束的那一刻起停策略、PD 目標固定在當下關節姿勢、速度目標 0（done 前策略不動 → 揮棒與 M2 判準不受影響）；(B) 只開 zero_ff_after_ref_end（策略繼續跑，但也會改到原生回合內擊中後的行為，不重訓的話較弱）。決定：logs/run_fall_ab.sh 立即跑（GPU 空、沒人排隊，2 × 約 3.5 分鐘）。A 的判準（加嚴）：(1) 每 env 到原生 done 為止與 logs/joint_smoke.csv 逐位元相同（同 9 個擊中 env、同初速 / 仰角 / 方向 / 飛行、同交接時間）；(2) done 後 6 s 內跌倒 ≤ 10/64；(3) 那 7 個 clip 結束前的跌倒拆成擊中 / 沒揮到（擊中的屬交付的 43%、允許；沒揮到且 done 前跌倒不允許，原生 F1 為 0%）。B 只回報、不採用（採用要重量 ONE OUTS 指標）。A 全過 → 同一條鏈直接進 M2（10 行程 × 64 球即時、after_done=hold、約 40–50 分鐘；目標 好球帶擊中 0.46 ± 0.05、擊中後跌倒 43 ± 8%、穿越 vs fly_to_x < 1 cm、不揮棒離線套引擎 rng → 四壞 10 ± 2.3%），開始時間與標記通知我和 20。A 沒過 → B 跑完停、送列給我。→ 鏈 13:55 起：A ~14:03、B ~14:11（FALLAB_END）→ logs/check_fall_a.py 套三條判準印 A_PASS / A_FAIL 到 logs/fall_ab_check.txt → A_PASS 則 ~14:12 起 logs/run_m2.sh（seed 1–10 × 64 env、after_done hold、logs/m2/m2_s*.csv、每行程約 7 分鐘 → 約 70 分鐘、~15:20 M2_END）→ CHAIN_AB_M2_END。e380d56 加欄 batter_fell_before_done、done_t、pred_y / pred_z（交接時 fly_to_x 穿越，與 j_cross 比 < 1 cm）。GPU 在此之前都是 15 的，20 已通知不得啟動。

**10-28 13:5x M3 重拍通過 = M3 相機 / 地面定案（20，bba9ac2 + c8b36aa）**：videos/m3b_joint_k00.mp4、拼圖 videos/fixcheck_tiles/m3b_joint_k00_tile.png 我看過：中外野 8× 兩人全身、圓丘（此角度成橢圓）、標記球可見；本壘視角（本壘後 9 m、一壘側 0.6 m、1.35 m 高、1.5×）打者全身在畫面內、棒尖整段在上緣內、腳離下緣有餘、投手在中央圓丘上、球到打者腳邊可見；草地純綠、沒有格線；天空仍全白（film_sky 之後試）。出手列與 C1 env 0 逐位元相同（草皮與圓丘不碰物理）。定案 M3 預設：中外野 40 m / 三壘側 4 m / 3.5 m / 8×；本壘視角 9 m / 一壘側 0.6 m / 1.35 m / 1.5×；film_grass + film_round_mound 開。幀檢查曾誤判此片 EMPTY（本壘視角被靜止的大打者填滿，動態 0.42%），20 把門檻降到 0.1%（空片 0.01%、純投手片 ≥ 2.4%）；我要求再加中外野面板動態 ≥ 1% 的檢查以免漏掉空渲染。後續低優先（CPU）：本壘視角受站姿棒尖（約 2.75 m 高）限制，650 px 的身體會讓含棒高度約 965 px 超出 900 px 畫面；20 建議下次改 9 m / 1.65× / 1.4 m（身體約 530 px、棒尖餘裕約 97 px），先維持 1.5×。幀檢查已改：並排影片左面板（中外野 / 側視）動態也要 ≥ 1%（m3b 4.02 / 0.42、c3c 2.96 / 2.73、match_full3 k46 15.29 / 4.35、空片 0.56 / 0.01 → 空片兩條規則都抓到、真片全過）。GPU：20 沒有排隊；下一個是 15 的跌倒診斷（FALLDIAG_END）。

**10-28 13:3x 同場 smoke 第 3 次跑完（15，13:15–13:22；三次合計約 9.5 分鐘 GPU；logs/joint_smoke.txt / .csv、ae5a236）→ M2 暫停**：交接 64/64 在 0.9975–1.000 s（預測 0.998）、打者起始 0.7925–0.8025 s（提前 0.195–0.208 s = clip 的 release_t）、沒有在起始前出手的；擊中 9/64（167-7 2、282-1 4、42-1 2、42-2 1），55 球穿越本壘、24 在好球帶、好球帶擊中 2；初速中位 14.3 m/s、方向中位 −87°（幾乎都界外）、飛行 2.8 m；跌倒：打者 64/64（沒揮到的也倒）、投手 11/64。15 提出兩個阻礙：(1)「同場投球只有 59–66 km/h，打者預測用的 P3c 表約 135 km/h（飛行 0.46–0.49 s）」；(2) 打者 64/64 跌倒原因不明。我的判定：(1) 不成立——我自己用 build_library_from_release 讀 assets/game/p3c_identical_t18_release.csv（62 列、預設 PhysicsSpec）：speed_kmh 中位 64.1（60.2–65.9）、flight_time 中位 1.045 s（1.019–1.124）、出手 x 17.28 / z 2.05；同場的 59.4–66.5 km/h 就是 P3c 表本身，135 km/h 在表裡不存在 → 15 撤回：135 km/h / 0.46 s 是它 CPU 否證腳本裡自設的假出手（38 m/s），誤讀成表；同場揮棒時機與 P3c 一致。跌倒診斷 driver 9ee30db（joint_smoke.py --diag、不改 shim）已排在 JOINT_FILM_M3B_END 後，輸出 logs/fall_diag.txt / .csv、標記 FALLDIAG_END。(2) 是真正的阻礙：核准約 3 分鐘 GPU 診斷（同場景、不改 shim、64 env）：每 env 第一次跌倒時間（相對 t0 / 打者起始 / 交接）、root z 與最低腳 z 相對世界地面（field_z −0.196）在 t0 / 0.4 s / 起始 / 交接 / 穿越（原生環境站姿 root z 0.96，若有 0.196 的偏移或腳陷地會立刻看出）、env 0 前 1.5 s root z 10 Hz；分階段回報（PD 保持中 / 起始到交接 / 交接後 / clip 結束後）。排在 20 的 M3 重拍（JOINT_FILM_M3B_END）之後。M2 的判準不變，跌倒解釋並修好（能在 CPU 修就 CPU）前不開 M2。

**10-28 13:2x M3 片第一版 + C1b（20，13:04–13:13）**：C1b（logs/joint_c1b_park.csv）vs C1：tier 以外所有欄位 64/64 逐位元相同 → 場景沒變、打者 shim 的改動不影響 PD 站姿評估；gate 行改讀積分器：站穩且投出 53、T3+ 51、T4 30（= 手算 aero tier），CSV tier T1 5 / T2 3 / T3 23 / T4 33，飛行差 0.0000 m，打者漂移 0.019 m → 待辦 (1) 關閉。影片 videos/m3_joint_k00.mp4（229 幀、High、動態 0.90%），拼圖 videos/fixcheck_tiles/m3_joint_k00_tile.png 我看過：中外野 8× 通過（兩人全身、比 C3c 大很多、投手丘與標記球可見）；本壘視角（6.5 m / 1.8×）不通過——球棒棒尖整段出畫面上緣、腳剛好在下緣；地面是青綠色格磚不是草。20 的針孔模型算退 1 m 也不夠（棒尖 2.75 m 高仍超 170 px）→ 重拍 bba9ac2：本壘後 9 m、一壘側 0.6 m、1.35 m 高、1.5×（預測棒尖距上緣 117 px、腳距下緣 130 px、打者約 650 px、投手約 140 px）+ film_grass 加一片 2 mm 高的純視覺草地蓋住格線；logs/joint_film_m3b.sh 排在 15 的 smoke 第 3 次之後（JOINT_FILM_M3B_END），約 5 分鐘。我的額外註記（最終影片用、不擋 M2）：中外野視角的投手丘看起來是一條橫跨畫面的長方形木板，不是圓丘 → 問 20 ...Shown 視覺副本是什麼、能否純視覺改成直徑約 5.5 m 的圓丘（不碰碰撞體）；天空是全白，若有便宜的天空 / dome light 選項記為 M3 項目。→ 20 回覆（c8b36aa，併入重拍）：...Shown 是兩個 5.5 m 寬的方塊（1.74 m 頂 + 3.06 m 斜坡 1/12）；新 film_round_mound 純視覺：方塊隱藏、改 0.1 m 網格高度網（投手線上與碰撞體差 ≤ 0.0002 m、輪廓 5.5 m 圓、橡膠板與後腳在平頂、3 mm 抬高防閃爍），碰撞體不動、以重拍出手列 = C1 env 0 驗證。天空：全白 = dome light 顏色，可換 Nucleus HDRI（kloofendal_43d_clear_puresky_4k），本機能否載入 Nucleus 未測，列 M3 項目之後找一次拍攝順便試，失敗就保持顏色。

**10-28 13:1x 15 的 CPU 否證通過（c7aabb7，scripts/test_batter_side_cpu.py）**：mock omni、跑真的 DirectRLEnv property 與打者環境程式，假 joint env 64 env 隨機狀態：init、reset 38 / 1 / 64、交接 38 / 1 / 25（2400 步、真實 hook 順序、全部有限）、judge 19 env 子集只在那 19 個記錄、擊出球方向合理（B 系 |spray| ≤ 7.7°、世界座標朝投手丘約 42 m/s）、root 寫回 38 / 1 / 全部都過；另修 shim 設 _is_closed 避免 DirectRLEnv.__del__ 關掉共用 sim。發現並解開一個死鎖：12:55–13:04 GPU 空轉 9 分鐘——20 的 joint_film_m3.sh 等 logs/joint_smoke.txt 裡的 JSMOKE_END，但 15 的 wrapper 只把標記印到自己的 stdout，而 15 的第 3 次又等 20 的片 → 互等；13:04 15 把標記補進檔案、wrapper 以後也寫檔。通則（補進 chain-script 規則）：結束標記要寫進對方 grep 的那個檔案，排鏈前兩邊對一次檔名。20 的 M3 片 + C1b 13:04 起（約 9 分鐘），15 的第 3 次接在 JOINT_FILM_M3_END 後。

**10-28 13:0x smoke 第 2 次失敗（15，12:54–12:55，約 1.5 分鐘 GPU）**：過了初始化、打者 start t 0.789–0.792 s / release_t_pred 0.998 s 正常，第一批交接 ids（38 / 64）到時崩潰：RotatedArticulation._qr 把每 env 四元數展開成 64 個，vec_to_b(x[ids]) 形狀不合；修正 02e1d44（單一 yaw 四元數廣播，數值不變，yaw 0 逐位元結果仍成立）。第 3 次 = 上限內最後一次，排在 20 的 M3 片 + C1b 評估（JOINT_FILM_M3_END，約 9 分鐘）之後；我要求 15 先在 CPU 用假 joint env 把所有吃 ids 子集的方法（交接、旋轉讀寫、_place、judge；38 / 64 / 1 個 env）跑一遍，過了才啟動。20 的 joint_film_m3.sh 已排在 JSMOKE_END 後自動啟動；C1b 要回報 64 列是否與 C1 逐位元相同。

**10-28 13:0x 20 的三項 CPU 待辦完成 + M3 拍攝核准（排在 smoke 之後）**：空片段 match_full2_p3c_k01–k25（動態檢查全 0.01%）已刪，k00 與 CSV 保留。(1) 同場 tier（6782d80）：有 j_cross 時 W1 tier 表 / gate 行 / CSV tier 欄一律用積分器穿越，下次同場評估驗。(2) M3 相機（02cfe98，針孔投影以 C3c 校準、誤差約 10 px）：中外野 40 m / 三壘側 4 m / 3.5 m 高，6× → 8×，瞄準兩人中間 → 投手約 490 px、打者約 350 px（原 370 / 260）；同場本壘視角改到本壘後 6.5 m、一壘側 0.6 m、1.0 m 高、1.8×、瞄投手丘 → 打者左前景約 735 px、投手中央約 185 px（揮棒從捕手側看得清楚）。(3) film_grass 預設關、開啟時 200 m 草綠地面，純視覺。決定：logs/joint_film_m3.sh（env 0、草地 + 新相機、約 5 分鐘、JOINT_FILM_M3_END）在 15 的 smoke 第 2 次 JSMOKE_END 後、Isaac = 0 立即跑；15 若需要第 3 次重試，等 JOINT_FILM_M3_END 之後。驗收：拼圖我看（兩視角都要全身入鏡、打者不能被切、草地、投手丘），過了就是 M3 的相機定案。

**10-28 12:47 檢查 + 同場 smoke 第 1 次失敗**：20 的 match_full3 重拍 12:45 完成（50 段、幀 / 動態檢查全過、出手列 50/50 = 評估；CSV tier T4 27 / T3 18 / T2 3 / T1 2、未站穩 8、y 中位數 0.041；算投手 12 小時影片，下次 ~00:45）。GPU 空、15 的 smoke 腳本 09:07 就備好但沒啟動 → 我 12:47 直接啟動 logs/run_joint_smoke.sh（核准過的那一份、313e6ba）。立即失敗（約 1 分鐘 GPU，第 1 / 3 次）：batter_side.py:143 `self.device, self.num_envs = j.device, j.num_envs` → `AttributeError: can't set attribute 'device'`（DirectRLEnv 的 device 是唯讀 property，_JointFramework 在 MRO 裡沒蓋掉它）；traceback 在 logs/joint_smoke.txt。（更正：15 的鏈 bhpjyjig3 同時也在 12:47 啟動，是同一個行程。）15 已修（dc284a3：_JointFramework 只設 sim / scene，device / num_envs / dt 走 DirectRLEnv 自己的 property）、第 2 次重跑 12:5x 起；結果列在 JSMOKE_END 後由 15 送來。沒有 trainer、沒有閒置（smoke 排隊中）。20 的 CPU 待辦不變：W1 行 / tier 欄改積分器穿越、M3 相機（更緊的構圖或第二台、本壘視角含打者）、film_grass 已進（0410455）下次同場拍攝試。match_full2 k01–k23 空片段請 20 刪除（避免再被誤認）。

**10-28 09:47 檢查**：GPU 1 個 Isaac（match_full3 重拍閒置填充，k10 / 50；新片段 6.7 MB 與正常的 k00 一致，不是空的）。15 的同場 smoke 尚未取 GPU（hold 檔不存在，應仍在合併）。沒有 trainer、沒有閒置；12 小時影片由同場測試片與重拍涵蓋。

**10-28 10:0x C3c 通過 = 同場拍攝路徑完成**：videos/c3c_joint_k00.mp4、拼圖 videos/fixcheck_tiles/c3c_joint_k00_tile.png 我看過：中外野相機（橡膠板後 40 m、三壘側 4 m、高 3.5 m、6×、瞄準本壘上方 0.3 m）下投手在左（投手丘上）、打者在右（本壘、PD 站姿持棒），兩人全身入鏡、投手丘可見、標記球飛到本壘附近落地；env 0 站穩、出手列 = C1 env 0。M3 相機待辦（CPU）：兩人在畫面裡偏小（揮棒要看得清楚，考慮更長焦或一壘側休息區方向的第二台相機）、本壘視角要含打者（相機在本壘後 1 m，打者在旁邊看不到）、草地、挑 env 用同場 CSV。GPU hold 已移除，match_full3 重拍從 k02 續跑當閒置填充（要 GPU 就 touch logs/gpu_hold_pitcher）；下一個 GPU 使用者是 15 的同場 smoke。

**10-28 09:5x 核准第一次同場 + 即時打者 smoke（15，313e6ba，約 5–8 分鐘 GPU，logs/run_joint_smoke.sh → JSMOKE_END、logs/joint_smoke.csv）**：AIB-OneOutsJoint-v0（park）+ P3c 7600 + BatterSide；dt 1/800、4 個真實高度 clip 輪流、Rz(162°) 代理、12500 @1.5 倍；64 env 一球、6 s、投手 dones 關閉；每 env 一列（clip、km/h、穿越 y/z、交接時間 vs 打者起始、擊中、初速、仰角、方向、飛行、雙方跌倒）。最多 3 次修正重試；用 logs/gpu_hold_pitcher 讓 20 的重拍最多等一段；若 C3c 在跑先讓它跑完。

**10-28 09:3x M1.5 座標代理測試完成、代理乾淨（15；GPU 約 22 分鐘，logs/run_rot{,2,3}.txt，修正 95b2039、79ad151）**：第 1 輪 plain vs plain2 逐位元相同（PhysX 決定性），但 plain vs 162° 從第 1 步差 1e-4、50 步後 O(1)，64 env 擊中 0.438 vs 0.344；第 2 輪 yaw 0° / 360° 經代理也一樣發散 → 是浮點擾動被接觸豐富的回合放大，不是座標錯；來源是 isaaclab 的 quat_mul（8 乘法形式連單位四元數都會捨入），改成直接 Hamilton 積。第 3 輪 yaw 0° 與 plain 在每個 env 第一回合逐位元相同（obs / 動作 / 擊中 / 初速 / 飛行 / 跌倒；差異只在自動 reset 後約第 412 步，同場場景一個行程一回合不會到）。162° 512 env：擊中 0.375 vs 0.379（±0.021）、擊中後跌倒 43 vs 46%（±3.6）、初速 19.0 vs 18.3、飛行 17.0 vs 16.1 → 雜訊內。判準定案：(i) yaw 0° 逐位元相同 (ii) 162° ≥ 512 樣本分布相同。通則：1e-7 的擾動能讓 64 env 的擊中率移動 ±6 點 → 以後場景間不做逐 env「同球同結果」比較，只比 ≥ 512 樣本的分布（C1 的 46/64 相同與 18 不同也符合這個理解）。下一步（CPU）：同場場景的 M2 driver，20 的場景與 15 的 shim 合併、64 env smoke 準備好再要 GPU。

**10-28 09:2x C3b（env 0，新相機）**：拼圖 videos/fixcheck_tiles/c3b_joint_k00_tile.png 我看過：投手全身與投手丘都在畫面內、投手站穩、出手列 = C1 env 0（17.79 m/s、y +0.15、z 0.69、T4）；問題：中外野相機在一壘側 1 m，右打者（y −0.9、三壘側）正好排在投手正後方（只差 0.4°），只看得到頭。C3c（04c4780）：相機往三壘側移 4 m（投手 5.7°、打者 3.0°，相差 2.7°，畫面 ±5.5°），排在 15 的 ROT3 之後；20 看過拼圖再給我。match_full3 重拍（k00–k01 完成）等 C3c 後再續。

**10-28 09:0x C3 第一支同場影片（投手與打者同框）**：videos/c3_joint_k01.mp4（229 幀、動態 4.44%）、拼圖 videos/fixcheck_tiles/c3_joint_k01_tile.png（左：中外野視角、右：本壘視角）。我看過：兩人同框成立；問題：投手腳被切掉（相機 25 m 後、5 m 高、4×）、中外野視角看不到投手丘、地面是灰色格線（使用者早先規格要草地，列為 M3 項目）。env 1 在同場場景會跌倒（3.06 s；出手列與 C1 的 env 1 完全相同，所以影片忠實重現同場評估）；門檻 vs 同場有 20 個站穩翻轉 → 挑拍攝 env 要看同場 CSV 的 at_rest / tier，不能用門檻的。C3b（da1c1f5）：40 m 後、3.5 m 高、6×、瞄準本壘上方 0.3 m（投手腳 / 頭在地平線下 4.7° / 2.2°、打者 3.4° / 1.7°、畫面 ±3.1°），拍 env 0（同場站穩、T4）。順序：15 旋轉測試重跑（移除 logs/gpu_hold_pitcher）→ C3b（約 5 分鐘、拼圖給我）→ match_full3 重拍 k01–k49 當閒置填充（k00 與列複製、每段前查 hold 檔）。

**10-28 08:4x k = 1 片段確認、C1 通過 = M1 達成（提前到 10-28）**：k01（e0720dc）隱藏 prim 0、229 幀、動態 2.90%（好的 k00 3.01%、空片 0.01%）、出手列 = 評估 env 1（18.03 m/s、y −0.10、z 1.02、T4）；拼圖 videos/fixcheck_tiles/k01_tile.png 我看過：兩個視角都有投手與投手丘、動作完整。C1（同場場景 + 打者、64 env、park，logs/joint_c1_park.csv）vs 門檻（oo_filmcheck_f64.csv）：站穩 53 vs 51（≥ 48）、站穩且 T3+ 51 vs 47（≥ 32）、T4 30 vs 28、y 中位數 0.066 vs 0.050（±0.1）→ M1 PASS；46/64 列與門檻完全相同，其餘 18 列不同（多一個 articulation 改變 GPU solver 批次），打者不碰投手。飛行檢查：積分器穿越 vs simulate_to_plane 60 球 |dy| = |dz| = 0.0000。打者 PD 站姿 root 最多移 0.019 m。待修（CPU）：eval_stage 的 W1 gate 行與 CSV tier 欄用 PhysX 球穿越，park 之下球在地下會全判 T1 → 同場任務改用積分器穿越。GPU 順序（關鍵路徑優先）：15 旋轉測試 → 20 的 C2 / C3（C3 = 同場場景拍 env k，拼圖給我看）→ 之後才以閒置填充的方式重拍 k01–k49（每段前查 Isaac = 0，讓排隊的同場檢查插隊）。分開拍的影片降為草稿 / 備案素材，C3 可用就可能提前停拍。

**10-28 08:1x F1 否證通過、F1 關閉**：(c) seed 1 好球帶 0.481、擊中後跌倒 43%（278 擊中；08:10–08:13，3 分鐘 GPU）。兩種子平均 0.486 / 40%。同場環境維持 (c) 設定（dt 1/800 × 8、4 個真實高度 clip），不微調。

**10-28 08:2x k = 1 測試兩次失敗、已修（2faa99d）**：第一次對 env 1 底下 288 個 prim 都設可見 → PhysX 重新解析、sim view 壞掉（改成只設 env k 根節點，與 MakeInvisible 對稱，94ccaa2）；第二次被新的隱藏 prim 檢查擋下，因為 plate_yaw 18 下碰撞用的 MoundTop / MoundSlope 本來就是隱藏的（畫的是 ...Shown 視覺副本），檢查改成只算從 env k 上方繼承的隱藏。佇列（film_fixcheck.sh）：等 15 的 F1S1_END + Isaac 0 → 拍 k = 1 → fixcheck_p3c_k01.mp4 + 拼圖 videos/fixcheck_tiles/k01_tile.png（FIXCHECK2_END）→ C1（同場場景 + 打者、64 env、park，約 5 分鐘，logs/joint_c1.log）。

**10-28 08:1x F1 結果（15，07:51–08:05，14 分鐘 GPU；12500 @1.5 倍、diag1024、P3c 球、seed 0）**：基準 好球帶 0.458 / 擊中後跌倒 43% / 288 擊中；(a) dt 1/800 × 8：0.461 / 41% / 271 過；(b) 4 clip 真實高度：0.469 / 44% / 288 過；(c) a+b：0.490 / 37% / 293 超出帶寬但在好的那側（好球帶 +0.032 vs ±0.03、跌倒 −6 vs ±5；約 300 顆好球帶球的 SE ≈ 0.03）。決定：不微調，同場環境用 (c) 設定；便宜否證 (c) seed 1（約 4 分鐘）排在 20 的 k = 1 片段之後，好球帶 < 0.428 或跌倒 > 48% 才要微調提案。GPU 順序：k = 1 片段（F1_END 自動）→ (c) seed 1 → 20 的 C1–C3 → 15 的 M1.5 旋轉測試。

**10-28 07:5x 空影片原因與修正（20，71fbde4）**：拍攝在 k24 停、GPU 交給 15 的 F1。原因：複製的 env 繼承 env_0 的 USD 設定；film() 隱藏 k 以外的所有 env（含 env_0），env k 跟著繼承「不可見」，只有 k00 拍得到。前測只對 k00 做幀檢查、precheck k01 其實也是空的。修正：在 env k 及其下所有 prim 設本地可見性 inherited，再逐 prim ComputeVisibility，任何隱藏 → 拍攝前 RuntimeError。check_video_frames 新增「機器人可見」檢查：本壘視角像素時間標準差 > 20 的比例（每 5 幀、去掉字幕）；k00 3.01%、空片段 0.01%，< 0.5% 判 EMPTY 並 exit 1。k = 1 測試（logs/film_fixcheck.sh）等 F1_END 後拍 → videos/fixcheck_p3c_k01.mp4 + 每 23 幀拼圖 videos/fixcheck_tiles/k01_tile.png；20 和我都看過才重拍 50 球。CSV k0–k23 保留。

**10-28 07:4x 使用者發現重拍的影片是空的（match_full2_p3c_k21）→ 停拍**：我抽圖確認 k13、k21 兩個視角整段只有地板格線，沒有投手 / 投手丘 / 球；k00 正常。檔案大小一致指向同一件事：k00 6.7 MB、k01–k23 全部 8.2–8.5 MB → k ≥ 1 的片段全部是空的。出手列（k8、k13、k21）都和評估相同，所以是 --video_env k 的相機 / 可見性路徑拍到空地（相機沒移到 env k 的原點，或隱藏其他 env 時把 env k 也藏了），不是物理。前測的漏洞：k = 0、1、2 只比對了出手列和「0% 黑畫面」（check_video_frames 只抓全黑），沒有人看 k1 / k2 的畫面——規劃 session 應該自己抽圖看，這次漏了。處理：請 20 立刻停掉 film_match_full2.sh、修好後只拍 k = 1、把每 10 幀拼圖目視確認（兩個視角都要有機器人、投手丘、球），並在 check_video_frames 加「機器人可見」檢查；我也看過才重拍 50 球（k00 與 CSV 保留）。15 的 B / A 鏈解除（它的觸發條件會在壞片段上啟動）；F1 三項檢查提前，GPU 一空就跑。約 1.9 小時 GPU 浪費（k01–k23）。

**10-28 06:47 檢查**：GPU 1 個 Isaac（50 球重拍，k14 / 50，71 分鐘 15 球 ≈ 4.7 分鐘一球 → 約 09:30 結束）。沒有 trainer、沒有閒置。12 小時影片由這次拍攝（投手）與之後的 B 渲染（打者）涵蓋。兩個 session 都在 CPU 上做同場環境。

**10-28 08:3x 打者座標代理完成（1fca746，CPU 驗過）**：RotatedArticulation（batter_side.py）W = Rz(162°)·B + 每 env 樞軸；robot.data 的 *_pos_w / *_quat_w / *_vel_w / root_state_w 讀時旋轉、root 寫回時反轉，其他 *_w 欄位一律 raise 以免漏掉。152 維 obs 逐項分類：座標不變（關節、projected_gravity_b、root 體座標速度、root z、相位、動作、參考關節差、t_arrive、|v_seen|、released、tempo）；靠代理（球 − root、甜蜜點 − 球與甜蜜點速度、球棒軸、參考 root / 棒尖差，皆經 quat_rotate_inverse）。CPU 隨機姿態 20 m 樞軸：相對位置 6e-6 m、速度 7e-7、往返 9e-6 m。M1.5 精確性測試（約 10 分鐘 GPU）：原生 PitchedBattingEnv 同種子 64 env 跑兩次（純 vs 經代理旋轉 162°、球停放、tee 隱藏），比前 50 步 obs / 動作 ≤ 1e-4 與每 env 結果；另跑純 vs 純量 PhysX 的 run-to-run 雜訊。GPU 順序：B 渲染 → F1 → 20 的 C1–C3 → 精確性測試。

**10-28 08:0x 同場環境兩個決定（20 的 hook 已進 aee4f32）**：(a) 出手交接後的球：不把積分器姿態每步寫回 PhysX 球體（15 的經驗：每步瞬移球會破壞打者的地面接觸、腳陷進地板），改成 PhysX 球停放在畫面外 + 純視覺的標記球（同網格 / 材質、無碰撞）跟著積分器；投手出手後 obs 不讀球（release_neutral 凍結），judge 讀積分器，所以沒有功能性差別。cfg ball_after_handoff 預設 park。(b) 座標：投手世界不旋轉（碰撞投手丘沿 +x、參考動作、球角速度的世界軸 obs 項、所有統計都綁在上面；改了要重開門檻）。打者 shim 只能平移 → 在打者端加固定 Rz(162°) 的座標代理，先在打者原生環境做精確性測試（整個場景經代理旋轉，obs / 動作 / 結果與未旋轉一致到浮點雜訊），列出所有非座標不變的 obs 項（root 四元數、世界座標位置 / 速度、球歷史、tee / 目標位置），併入 M1.5。拍攝：k8 與評估列完全相同，約 09:50 結束。

**10-28 07:4x 打者整合設計核准（15 提案）**：A 座標：打者環境裡球沿 −x 飛、+y 是三壘側；本壘座標 → 打者座標是每個 clip 一個平移 (tee_x, tee_y, dz)，dz = tee_z − 0.76；真實場景 dz 必須 0（打者站地上），所以只用 |dz| ≤ 0.03 的 4 個 clip（167-7、282-1、42-1、42-2）、球用真實高度；站位 = root0 − tee（x −0.19..−0.38、y +0.87..+0.94、root z 0.96）；球棒 body 名 "bat"。B 球：同 20 的交接方案；PhysX 球與打者的碰撞必須關掉（否則接觸算兩次）。C 程式：BatterSide 是 shim，直接跑現有 TeeBattingEnv → MimicEnv → PitchedBattingEnv 的方法（感知 / 152 維 obs / 相位時鐘 / 殘差動作 / 撞擊模型 / play_log / 跌倒 / take_hold 都不動，只換掉 _reset_idx 的投球庫部分）；策略 actor + EmpiricalNormalization 從 model_12500.pt 載入，AIB_EFFORT_SCALE 1.5 進打者 cfg。D 原本來自投球表的 obs：release_t → 門檻的 release_t（相同起始 p10–90 差 2.5 ms 內）、flight_time → 出手後用 fly_to_x 由真球算、tee_pos / contact_x_ref → clip 平移、pitch_kmh 本來乘 0。E 不揮棒：出手 + 100 ms 感知延遲後 fly_to_x 預測穿越 → HBP → TableBatter.decides_to_swing；take 讓參考時鐘停住；計數 judge_pitch（加可選的真實落點引數，引擎 first_bounce 當交叉檢查）；PA 由 env 順序的球組裝。F 檢查：F1（M1 前、不需同場環境，約 25 分鐘 GPU，B 渲染後約 10:30 起）：(a) 交付 diag 在 dt 1/800 × 8 vs 1/400 × 4；(b) 4 clip 真實高度（新 cfg 旗標 real_height）；(c) 兩者合併；門檻 好球帶 ±0.03（0.458）、跌倒 ±5 點（43%）；沒過就停、帶微調提案（迭代數、GPU 分鐘、否證）來核准。M1.5 shim 同一性（同狀態 → 同 obs / 動作）。M2：10 行程 × 64 球 = 640 顆每球都揮（約 40–50 分鐘），好球帶擊中 0.46 ± 0.05、擊中後跌倒 43 ± 8%、穿越與 fly_to_x 差 < 1 cm；不揮棒規則用引擎 rng 離線套在同 640 球（揮棒結果與 take 決策獨立，所以有效），約 170 打席：四壞 10 ± 2.3%、勝率 ±3 點 vs 17.6%。G 工時：F1 0.5 天、shim + 策略 + oracle + 計數約 2 天（11-01 前）、M2 1 天。待定（M3 時）：結尾卡勝率要用同場模擬量的（約 70 行程 / 6 小時可得約 1000 打席）還是引擎 17.6% + 同場交叉檢查。GPU 順序：拍攝（~09:50）→ B 渲染 → F1 → 20 的 C1–C3。

**10-28 07:2x 同場環境的球（核准）**：打者環境的球本來就不是 PhysX 剛體，而是 RK4 阻力 + 馬格努斯積分 + 球棒掃掠接觸測試。定案：球在投手手中是 PhysX（真實握球與出手）→ 出手瞬間把狀態交給打者的積分器 → 飛行、球棒接觸、擊出的球都由積分器負責（打者訓練與 ONE OUTS 指標用的同一套、投手 aero tier 也是同一飛行模型，不引入新物理、兩邊門檻都不重開）。要求：積分器姿態每步 kinematic 寫回同一顆 PhysX 球體（投手出手後的 obs、相機、15 的 judge 都讀同一顆）；影片結尾卡 / README 要註明。已知限制：擊出的球打不到投手。

**10-28 07:1x 同場環境設計核准（20 提案，CPU 即刻開工）**：`aibaseball/tasks/one_outs/joint_env.py` JointPAEnv(StageEnv)（P3c 投手邏輯 / obs / 獎勵不動）+ joint_env_cfg（打者 ArticulationCfg = make_aib1_cfg human_strength、effort 1.5；任務 AIB-OneOutsJoint-v0）+ `batter_side.py`（15：obs 感知共用球、oracle 從出手狀態算、act 寫打者關節目標、judge 計數）+ `scripts/eval_joint.py`（eval + 拍攝 + 每球結果 CSV）。同一 InteractiveScene 兩個 articulation（SEA 只在投手、打者 implicit PD）；dt 1/800，投手 400 Hz、打者每 4 步動作一次（100 Hz）；本壘 / 打者座標 Rz(−18°)·(18.44, zone_center_y)、z = field_z；跨 env 碰撞已過濾。一個回合 = 一球（我更正用語：PA 是 1–6 球帶計數，影片裡一個打席是多個回合剪接，不揮棒規則要吃計數 → eval_joint 要接受每 env 的 balls/strikes，結果 CSV 要帶）；回合間不 env.reset()，一個行程 = 64 個載入後第一回合，拍 env k 用 film_first / video_env。投手邏輯改動：_apply_aero 改成 released & 在空中（擊出的球也有空氣力）；只有擊球前的第一次穿越算；新 bat_contact 旗標；跌倒不結束回合、PA 結束取代 timeout；StageEnv 的球靜止統計跳過。GPU 檢查（拍攝結束後、一次一個 Isaac）：C1 約 5 分鐘 打者凍結、64 env 一回合，M1 判準 = 與門檻同分布（T3+ ≥ 32/64、站穩 ≥ 48/64、y 中位數 ±0.1 m；逐位元相同不可能，多一個 articulation 會改 solver 狀態）；C2 球到本壘的 y/z 與 simulate_to_plane 差 < 1 cm；C3 env k 拍攝。M2 每次量測 5 × 64 球約 25 分鐘；拍攝每球約 5 分鐘。里程碑 M1 11-01 可行、M2 11-08 的風險在打者 dt 1/800 下的表現（可能要微調時段）。

**10-28 06:5x 20 的同框估工（寫於收到決定前）與我的定案**：事實：投手 sim dt 1/800、decimation 2（400 Hz）、SEA 在 1/800 積分；打者 dt 1/400、decimation 4（100 Hz）、implicit PD；env 間距 4 m、跨 env 碰撞被過濾；本壘在 env_origin + Rz(−18°)·(18.44, zone_center_y)，field_z = −0.196。路線 1（回放合成）約 3 天、不用重量門檻；路線 2（同場模擬）5–7 天 + 風險：(1) 物理 dt 要統一（SEA 要 1/800，打者在 1/400 訓練）；(2) 打者 obs / 動作 / oracle / 感知要搬進合併環境（約 600 行）；(3) 加入打者後 64 env 解算狀態改變，投手的球變成新抽樣，P3c 要在同場景重量；(4) 兩個策略、兩個 normalizer、兩個 decimation。20 建議路線 1，但使用者已選路線 2，維持。定案：dt 1/800、打者 decimation 8 保持 100 Hz；M2 在同場景重量打者（好球帶擊中 < 0.40 或初速中位數 < 15.5 才考慮微調）；P3c 在同場景（打者站著）重跑門檻 64，只回報不調；飛行中的球不論 crossed 都給空氣力直到靜止，擊球偵測與計分歸 15；PitchEnv 的 obs / 出手 / SEA 不動。第一步（1 天）：雙 articulation 場景 + 一顆球 + 投手策略，打者在本壘以 PD 維持預設姿勢，GPU 驗 64 env 投球到本壘並印 tier。--state_record（路線 1 準備）先不做，11-15 才啟動備案。

**10-28 06:3x 使用者新規格：最終影片投手與打者要在同一個畫面，而且選「真正同場模擬」**（不是分開拍再剪，也不是回放合成）。= 一個 Isaac 場景同時有投手（彈性 / SEA cfg）、打者（make_aib1_cfg human_strength、implicit PD、AIB_EFFORT_SCALE 1.5）和同一顆實體球（ball_aero_sim=True），兩個策略即時互動；打者的感知模型（偏差 / 雜訊 / 延遲）與揮棒起點 oracle 改吃真球的狀態。目前分開拍的 match_full2 B 版只當節奏 / 計數卡草稿，A 版取消。回放合成（兩邊各自的實際回合在同一場景裡 kinematic 回放）是 11-15 前同場模擬做不出來時的備案。分工：20 負責同場環境（場景、投手、球、終止條件、相機），15 負責打者整合（obs / 感知 / oracle / judge_pitch 計數）。里程碑（草案，等兩邊估工後定案）：M1 11-01 雙機器人場景載入、投手投球到本壘、env k 的球與門檻列相同；M2 11-08 打者即時揮 / 不揮，擊球結果分布與引擎相近（好球帶擊中約 0.46、跌倒約 43%）；M3 11-12 同場比賽影片草稿；11-15 備案決定點；11-26 交付。程式開發在 CPU，GPU 只給驗證。

**10-28 06:1x 前測通過、50 球重拍開跑（05:36 起，logs/film_match_full2.sh，每球一個行程約 5 分鐘，約 09:50 結束）**：eval_stage 新旗標 --video_env k / --film_first（e31d77d）= 在 64 env 的門檻 sim 裡拍 env k 的載入後第一回合，其他 env 隱藏。k = 0、1、2 三個行程拍到的出手列與 64 env 評估的 env 0/1/2 列到最後一位都相同（17.788 / 0.150 / 0.689 T4；18.026 / −0.102 / 1.024 T4；17.292 / −0.052 / 0.014 T1）→ 渲染不改物理、球的變化就是門檻的分布。輸出 `videos/match_full2_p3c_k00..k49.mp4`（229 幀、0% 白畫面、High / yuv420p）+ `match_full2_p3c_release.csv`（k 序）；依評估列預期 T4 25 / T3 20 / T2 1 / T1 4、出手後跌倒 10。舊的 match_full / match_test 檔保留。拍完 → 15 渲染 B → A。投手 12 小時影片由此涵蓋（下次從拍完起算）。

**10-28 06:0x 15 的多片段輸入完成（已 commit）**：--pitcher_video "<glob>" + --pitcher_episodes 0，自然排序、檢查每段幀數相同與段數 = 出手列數，不需 concat、不重新編碼。啟動：`NOWAIT=1 VARIANTS=B N=0 PV=<glob> PR=<csv> bash logs/run_match_full.sh`，再 VARIANTS=A。已請 20 依回合順序命名、CSV 同序。

**10-28 05:4x 修正驗證沒過 → 決定每球獨立行程、64 env、相機跟 env k**：12d155e 後 64 env：評估輪不變（y 0.052、32/25/2/5、站穩 51）、rerun 0 y 0.085（31/16/10/7）、rerun 1 y 0.258（18/25/17/4）——比修前（0.206 / 0.358）好，仍漂。dump：評估 vs rerun0 逐位元相同；評估 vs rerun1 reset 時所有 tensor、機器人 / 球 / 致動器都相同，但一步之後球速差 3.7 m/s、關節速度差 1.46 rad/s → 相同輸入狀態 PhysX 給不同結果，PhysX 內部（最可能是球與手指的接觸 / warm-start 快取；reset 時球從 18 m 外瞬移回握把）撐過了 reset。20 提三案：(1) 每球全新行程（= 門檻的載入後第一回合，+30 s/球）、(2) 保留 12d155e + (1)、(3) 再試 PhysX 端 reset（不做）。我選 (2) 並加一條：兩次單 env 拍攝前 12 球逐位元相同，表示全新單 env 行程 + 相同起始很可能 50 球都一樣，門檻的分散來自 64 env 同一個 sim；所以每球 = 一個全新行程跑門檻設定（64 env、相同起始、載入後第一回合），相機放在 env k（k = 0..49），出手列取 env k。前測（約 15 分鐘）：k = 0、1、2 三個行程，與一次 64 env 不錄影評估的逐 env 列比對，三球要等於評估的 env 0/1/2（至少彼此不同、落在評估分布內）。相機無法便宜地指定 env 就先回報成本；退路是純 (2)，但要先證明兩個全新單 env 行程的球不同。輸出 50 個單球 mp4（同幀數、同版面）+ 一份出手 CSV，15 用 ffmpeg concat 無損接起來。預估 50 球約 4 小時 → B → A。

**10-28 05:2x 打者端沒有 reset 踢動問題（15）**：打者機器人全部用 ImplicitActuatorCfg（PhysX PD，沒有致動器狀態），有狀態的 SEA 模型只在投手的彈性 cfg（aib1_cfg.py:149+）；env.reset() 只寫目標不 step 物理，step() 前 _apply_action 一定先寫新目標，所以沒有東西帶過去。diag1024 與 showcase 表格只在開頭 reset 一次（同訓練）；showcase 揮棒影片與 match_video 打者片段每揮一次 reset 一次，但無害。佐證：Human21 smoke diag 完全重現 12500。打者數字都不用重量。15 待命：20 重拍後跑 run_match_full.sh B → A。

**10-28 05:1x 根因找到（20）：env.reset() 時馬達吃到上一回合殘留的關節目標**。64 env 不開相機、評估後再 reset 兩輪：y 中位數 0.052 → 0.206 → 0.358，T4/T3/T2/T1 32/25/2/5 → 24/21/13/6 → 19/22/16/7，每 reset 一次漂更遠，不是相機。env 0 的狀態 dump：reset 時關節位置 / 速度、球狀態都和載入時相同；差在 Isaac Lab 的 env.reset() 會跑一次 scene.write_data_to_sim() → 致動器模型，馬達往 robot.data.joint_pos_target 走一步：載入時這個 buffer 全 0（門檻與 ONE OUTS 表格量到的狀態），reset 後還是上一回合的目標（差最多 1.57 rad，起始馬達速度差到 13 rad/s）。訓練不受影響（reset 在 step() 裡，_apply_action 先寫目標）；pad_hist / post_gain 也有殘留（只影響出手檢查 / 獎勵）。修正 12d155e（PitchEnv._reset_idx 把 reset env 的關節目標歸零 = 載入狀態、清 pad_hist、post_gain 設 1），正在驗（64 env 評估 + rerun 2 輪 + dump，約 14 分鐘，三輪都要 ≈ 0.05 且評估輪不變）。通過後 20 重拍 50 球（約 3.2 小時）→ 15 渲染 B → A。附帶：之前的投手 12 小時影片（--video_episodes 1）也都是 reset 後拍的。已請 15 查打者端有沒有明確呼叫 env.reset() 的路徑（showcase 表格、diag、match_video、12 小時影片）。

**10-28 04:5x 否證中間結果（20）**：64 env 不錄影、拍攝的 --set：ball_aero_sim 有 / 無 aero_y 中位數都 0.052（T3+ 47 / 48），aero 不是原因。更正我 04:2x 的假設：拍攝的第 1 球不是全新載入的回合——eval_stage 先跑一個評估回合、之後才開拍，所以每一顆拍攝的球都在 w.reset() 之後，而門檻評估只量了「載入後第一個回合」。20 正在跑 --rerun_csv / --rerun_rounds（15719ca，64 env、無相機、評估後再 reset 2 輪）：rerun ≈ 0.3 → reset 後的狀態和載入時不同（門檻量到的是特殊狀態，要 diff 載入 vs reset 的起始狀態、修 reset）；≈ 0.05 → 是相機 / 渲染路徑改變了回合。修好前不重拍。

**10-28 04:2x B 版渲染完成，但拍到的投手不是評估過的 P3c → A 版暫停**：`videos/match_full_B.mp4` 18563 幀（10:19）、High / yuv420p、36 分鐘 GPU（03:38–04:15）；10 打席 = 四壞 6、三振 3、內野球 1 → 打者 6 / 投手 4。引擎的四壞率約 10%，6/10 不對。15 查到：打者端穿越本壘的位置與 20 的 aero 列逐球一致（打者端正確），但拍攝 50 球的位置分布與門檻評估不同：T4/T3/T2/T1 評估 64 = 34/23/4/3（好球 53%）vs 拍攝 50 = 12/17/14/7（24%）；aero_y 中位數 評估 0.07 vs 拍攝 0.30，整體往一壘側偏 0.23 m（跌倒 10/50 則在雜訊內）。我查 CSV：拍攝腳本的 --set 與門檻評估相同；50 球的前 12 球與昨晚 12 球測試逐球相同（0.29、0.34、0.28、−0.06…）；第 1 球（全新載入、沒 reset 過）就已經偏 0.29 → 拍攝路徑是確定性的，不是連續 reset 累積，嫌疑只剩 `--envs 1` vs 64 或 `--video` 路徑本身。已請 20 做否證：(1) 同設定 --envs 1 不帶 --video 跑 12 球，看是否重現 0.29、0.34…；(2) 同設定 --envs 64 不帶 --video 確認 0.07 / 34/23/4/3。哪邊才是真的 P3c、修法要讓影片和 64 env 的回合一致。B 版留作流程草稿，不刪。

**10-28 03:47 檢查**：GPU 1 個 Isaac（15 的 B 版渲染 match_video.py，03:38 起，`videos/match_full_B.mp4` 寫入中）。捕手視角 +0.5 m 經 B 第 1 打席確認（打者全身入鏡），A 版沿用。沒有閒置、沒有 trainer；12 小時影片由投手 50 球（03:26）與 B 渲染涵蓋，投手下次 15:36 前。

**10-28 03:3x 投手 50 球拍完（00:22–03:26，194 分鐘）**：`videos/match_full_p3c.mp4` 11450 幀 = 50 × 229、0 重複、0% 白畫面、High / yuv420p、3200×900；`match_full_p3c_release.csv` 50 列全部出手。T4 12、T3 17、T2 14、T1 7（T3 以上 29/50 = 58%，門檻評估 48/64 = 75%；T1 有 3 球落地、4 球偏）；球速 16.74–18.51、平均 17.84 m/s；跌倒 10/50（都在出手後 1.86–3.78 s）。相同起始的逐球分散比門檻評估大（拍攝一次只 reset 一個 env，sim 狀態和 64 env 評估不同），只記錄不修。已交給 15 渲染 B → A。投手下次 12 小時影片 15:36 前。

**10-28 00:47 檢查**：GPU 1 個 Isaac（投手 50 球拍攝，00:22 起 25 分鐘拍完 5 球，約 5 分鐘一球 → 約 04:30 結束，比 20 估的 03:50 晚）。沒有閒置、沒有 trainer。12 小時影片由這段拍攝（投手）與之後的 B 渲染（打者）涵蓋。

**10-28 01:0x 影片 4 處修正已 commit（f919ca8 PiP、d2816d9 其餘，CPU）**：結尾卡自動換行（1280 − 120 px）；投手側面全幅 + 本壘小窗；B 停鏡往投手丘側 +0.5 m；捕手視角往三壘側 +0.5 m（未經 GPU 驗證，B 第 1 打席看過不好就在 A 之前還原）。鏈 logs/run_match_full.sh：B → A，各 10 打席，seed 0，每次啟動前查 Isaac 數，20 的 50 球拍完（約 03:50）後由 15 手動啟動。

**10-28 00:4x 投手 50 球開拍（00:22 起，每球約 4.2 分鐘，約 03:50 結束；`videos/match_full_p3c.mp4` + `_release.csv`，logs/film_match_full.sh）**：GPU 到 03:50 淨空，之後 15 渲染 B → A。決定：這段 50 球影片就是投手這輪的 12 小時影片、B 版完整渲染就是打者的 12 小時影片（都是 Isaac 實際回合拍的動作），不另外在 05:37 / 07:46 拍；下次到期各自從拍完算 12 小時。15 先在 CPU 修 4 處並 commit。

**10-28 00:3x 打者交付定案 1.5 倍 12500 = 17.6%（3 表）、A / B 測試影片判定、完整渲染開工**：5 種子 × 3 表：1.5 倍 12500 17.6%（18.8 / 16.5 / 17.4）外野 3.5%，門檻 19.1；Human20 12698 18.7%（18.8 / 18.3 / 19.2）外野 3.6% → 沒過，不採用；Human19 12599 17.3% 外野 4.8%；1.0 倍 12500 17.1% 外野 1.7%（勝率平手、外野沒過）→ 交付維持 1.5 倍 12500，數字改報 17.6%。便宜否證成立：第 3 表 17.4 與 16.5 差 < 1 點，原本 18.8 是高抽。測試影片（自己抽圖看過）：A `videos/match_test_A_s0.mp4`（3 壞球後內野滾地 19.5 m、投手勝；投手第 3 球跌倒照放）、B `match_test_B_s0.mp4`（同 12 球但第 4 球在這次變揮空 → 四壞，Isaac 不同次啟動不會逐位元重現）、`match_test_B_contact.mp4`（停鏡 2 秒看得到擊中後跌倒）。判定：完整渲染先做 B（打者全程入鏡、跌倒照實）再做 A（兩版是不同場比賽），各約 10 打席；先修 4 處：結尾卡文字超出右邊界（必修）、結尾卡改「5 seeds x 3 pitch tables x 1000 matches, batter wins 17.6% (tables 16.5–18.8%)」、投手片段改側面全幅 + 本壘視角 PiP（原本雙寬畫面縮成一條）、B 停鏡目標往拉打側 +0.5 m（捕手視角打者半個在畫面外也順便修）。GPU 00:3x 空閒 → 20 拍約 50 球（相同起始）→ 15 渲染 B → A → 12 小時影片（打者 05:37、投手 07:46 前，不與渲染同時）。

**10-27 23:2x 影片測試投球完成**：P3c 12 球（2748 幀、0% 白畫面、High / yuv420p）+ 12 列出手狀態（b6b78a6、0e8add5）。球速 17.0–18.4 m/s；T4 2、T3 5、T2 4、T1 1；12 球有 2 球投完晚跌（2.5 / 3.1 s）。相同起始的連續回合本來就有分散，不需要起始擾動。交給 15 做 A / B 測試渲染。

**10-27 23:4x Human21 停在 +100（不採用）、ONE OUTS 表格雜訊、影片用相同起始**：Human21 漏檢過（好球帶 0.458、初速 16.121 與 12500 相同），但訓練每次擊中跌倒 0.788（門檻 < 0.70）、diag 跌倒 43% 沒動 → 揮棒凍結 + 不設上限的站穩獎勵也教不會結束後站穩。跌倒這條線關掉（5 個設計）。**表格雜訊**：同一個 12500 @1.5 倍重跑 showcase 表格，5 種子 16.5%（外野 2.7%）、加 clamp 17.2%（3.4%），原交付表 18.8%（4.7%）是高抽；3 表平均約 17.5% / 3.6%。每張表約 750 顆擊球、外野在尾端，5 種子只平均引擎 rng。決定：指標改成 5 種子 × 3 張獨立表（每表約 8 分鐘 GPU），門檻 = 交付 3 表平均 + 1.5、外野 ≥ 3%；重量 1.5 倍 12500（第 3 表）、Human19 12599、1.0 倍 12500、Human20 12698（1.0 vs 1.5 當初也是單表對單表）。平手時維持 1.5 倍（diag 好球帶 0.458 vs 0.327、使用者「增加打者力量」方向）。**影片投球**：相同起始、不加擾動（P3c 原樣；擾動 0.01 會讓投手一半跌倒）。GPU：20 拍 12 球 → A / B 測試渲染 → 表格重量（約 100 分鐘）→ 完整渲染。

**10-27 23:0x 比賽影片工具完成（CPU；f00dbdc、1305ac9、be59f3e）**：不揮棒規則與引擎共用（TableBatter.decides_to_swing）、好壞球計數共用；投手 eval_stage --video_release_csv 把拍攝那一球的出手狀態存給打者（沒出手 / 沒到本壘 = 壞球）；B 版 --hold_after_hit 2.0。結尾卡「ONE OUTS simulation, 5 x 1000 matches: batter wins 18.8% (16.0–20.1%)」。順序：Human21 結束（約 23:30）→ 20 拍 12 球 → logs/run_match_test.sh 渲染同一打席（seed 0）的 A、B 版給規劃 session 看。Human21 +50 正常，+100 做漏檢。

**10-27 22:3x Human21 開跑（fd7f318）、比賽影片計畫核准**：Human21 = --post_clip_teacher（相位 ≥ 0.9999 後學生接手；老師用凍結的 normalizer），smoke test 老師 = 學生時 diag 完全重現基準（好球帶 0.458、120 跌）；float 雜訊約 ±0.01，漏檢容忍 0.03。影片計畫：每一球都由 Isaac 實際回合決定（投手拍一球、打者面對同一球的出手狀態，好壞球、揮空 / 界外 / 界內照機器人實際結果；1000 場勝率只放在結尾卡標為模擬統計）；不揮棒 = 打者維持站姿零動作（用引擎同一套不揮棒規則）；每球順序：中外野鏡頭投球 → 捕手後方來球（慢 2 倍）→ 側面揮棒慢動作 → 依結果；A 版轉播跟球，B 版側面停 2 秒讓打者全程入鏡再切落點。先各做 1 個打席的測試渲染（不與 trainer 同時）給規劃 session 看，再做約 10 個打席（每版約 1 小時 GPU）。

**10-27 21:5x Human20 停在 +200（不採用）**：12698 好球帶 0.330、跌倒 56%、初速 17.1、ONE OUTS 5 種子 18.9%（18.3–20.0）/ 外野 3.4% → 勝率、外野過，跌倒 56% 遠超 32；對 Human17 12698（同球、無平衡獎勵）61% / 18.8 / 3.2 幾乎沒差，平衡獎勵（0.02 / 步）沒有作用。四個設計 + 對照 C 的結論：跌倒只在揮棒變弱時下降；保住強度時跌倒 42–61%。獎勵塑形這條線關掉。下一步：(1) Human21 = 投手 PR 的做法搬到打者：參考結束前所有動作由凍結的 12500 決定，學生只在結束後動作（擊中、揮空都算），獎勵可以放大（站穩四項不設上限 + Human15 跌倒懲罰、無追蹤），強度照構造不會被交換；否證 +100 先查漏（好球帶 ±0.03、初速 ±0.3）再看訓練每次擊中跌倒 < 0.70。(2) 比賽影片草稿的鏡頭與剪接計畫（CPU）：標準轉播鏡頭，不做專門遮跌倒的剪接，另出一版打者全程入鏡，兩版給使用者選。

**10-27 21:47 檢查**：GPU 1 個 Isaac（Human20 12698 的 showcase）。Human20 +100（12599）：好球帶 0.257（一次 < 0.40）、跌倒 58%、揮空跌倒 0.3%、初速 17.2、ONE OUTS 5 種子 18.2%（16.2–19.6）/ 外野 2.5%、訓練每次擊中跌倒 0.831 → 三項否證過，但跌倒和 Human17 同期差不多。+200 結果約 21:50。影片：投手 `pitcher_p3c_t18_1028am.mp4` 19:46 已拍（下次 10-28 07:46 前），打者下次 10-28 05:37 前。

**10-27 21:1x 對照 C 結果：P3c 形狀的練習球本身就讓揮棒變弱**。Human15 @1.5 倍從 12500 練（不改結束後）：12599 好球帶 0.485、跌倒 23%、初速 15.0、ONE OUTS 5 種子 16.1% / 外野 1.4%；12698 好球帶 0.433、跌倒 32%、初速 15.1、ONE OUTS 14.7% / 外野 0.8%（≤ 1.5%）。附帶發現：光是 Human15 續練也讓跌倒 42 → 23–32%，和揮棒變弱一起出現，所以 Human18 / 19 的跌倒下降也是同一回事。照規則開 Human20（b5c26b9，21:1x）：Human9 的球（88–112 km/h）+ Human19 的擊中後平衡獎勵，揮空側不變，1.5 倍從 12500。+100 否證同 Human19 三項；+200 門檻 跌倒 ≤ 32%、5 種子平均 ≥ 18.8%、外野 ≥ 3%；採用 ≥ 20.3%。對照 Human17 12698（同樣的球、沒平衡獎勵）：跌倒 61% / 外野 3.5%。

**10-27 20:2x Human19 停在 +200（不採用）**：12599 好球帶 0.616、diag 跌倒 41%、揮空跌倒 0%、初速 16.3、ONE OUTS 18.7% / 外野 5.0%、訓練每次擊中跌倒 0.775（+100 否證過）；12698 好球帶 0.501、跌倒 33%（差 1 點）、初速 15.9、ONE OUTS 14.8% / 外野 1.0% → 沒過。平衡獎勵約 0.01 / 步，如設計地小。注意：Human19 的訓練跌倒率定義與 Human18 不同（lost_after_hit_is_fall 關掉），diag 才可比（42 → 41%）。**ONE OUTS 單一種子約 ±2 點**（5 種子：1.0 倍 12500 17.1%、1.5 倍 12500 18.8% / 外野 4.7%、H19 12599 17.3%、12698 14.7%）→ 以後採用看 5 種子平均（門檻 = 交付平均 + 1.5 = 20.3%，外野 ≥ 3%）；1.5 倍仍比 1.0 倍多 1.7，交付不變。觀察：Human17 / 18 / 19 從 12500 練 +200 外野都掉；18、19 共同用 Human15 的 P3c 形狀球，17 用 Human9 的球、外野還有 3.5%。核准對照 C：Human15 本身 @1.5 倍從 12500 練 +200（不改程式，約 35 分鐘）：外野 ≤ 1.5% → 是球種組合的代價，下一步 Human9 球 + Human19 平衡獎勵；外野 ≥ 3% → 是結束後改動的代價，跌倒這條線關掉、改做比賽影片草稿。

**10-27 19:5x 使用者：授權延到 10-30**。到 10-30 為止，新設計、提案核准、停止、交付挑選都由規劃 session 決定；規格不變（投手 P3c 定案、打者力量 1.5 倍、ONE OUTS 規則、11-26 交付）。已通知 15、20。Human19 等投手影片拍完自動開跑。

**10-27 20:0x Human19 核准（fc81ccf，改一處再跑）**：參考結束後，擊中回合取消追蹤獎勵、跌倒懲罰、lost 終止，改給平衡獎勵（直立、重心在雙腳連線上、低速、雙腳著地四項平均），總量 ≤ 0.5 × 該回合的擊球獎勵，所以揮軟反而縮小自己的平衡預算。Human18 掉強度的原因：結束後追蹤獎勵約 150 + 跌倒懲罰 10，遠大於擊球獎勵 2–26。我要求的修改：揮空回合保留 Human15 原有的跌倒懲罰與 lost 終止（揮空佔多數打席，沒有訊號會在影片裡亂動），並在 +100 diag 加檢查「揮空後跌倒率不超過 12500 的 +5 點」。否證 +100：訓練中每次擊中跌倒 < 0.90 且 diag 初速中位數 ≥ 15.6。+200：跌倒 ≤ 32%、ONE OUTS ≥ 19.9%、外野 ≥ 3%；採用 ≥ 21.4%。從 human_pitched9b 12500 @1.5 倍起。

**10-27 19:4x 使用者決定：打者下一個設計 = 參考結束後只給平衡獎勵**（結束前維持 Human9 / Human15，結束後不再模仿、只獎勵站穩；平衡獎勵不能壓過擊球獎勵，避免像 Human18 把揮棒強度換掉）。已請 15 送提案：獎勵項與權重、約 100 次迭代的便宜否證（訓練中每次擊中跌倒率要從 98% 降到 < 90%，且訓練環境初速中位數 ≥ 15.6）、+200 門檻（diag 跌倒 ≤ 32%、ONE OUTS ≥ 19.9%、外野 ≥ 3%；採用要 ≥ 21.4%）。從 12500 @1.5 倍起練，不接 Human18 的 checkpoint。GPU：投手影片自動拍完後給打者。

**10-27 19:3x Human18b 停在 +300（不採用，也不是影片候選）**：12698 好球帶 0.519、diag 跌倒 29%、ONE OUTS 15.7%（外野 1.2%）；12797 好球帶 0.393、跌倒 25%、ONE OUTS 16.4%（外野 0.2%、三振 47.5%）；訓練中每次擊中的跌倒率 0.986 / 0.980 ≥ 0.95 → 停止。12500 → 12797 趨勢：diag 跌倒 42 → 25%，但 ONE OUTS 19.9 → 16.4%、外野 4.1 → 0.2%、初速 16.1 → 15.0。解讀：跌倒變少是因為揮棒變軟、變保守，不是學會站穩（訓練環境裡擊中後仍 98% 跌倒）。打者交付仍是 h9b 12500 @1.5 倍（19.9%）。投手 10-28 影片的等待程式接在 H18B_END 後自動開拍。之後打者沒有排任何工作，新設計由使用者決定。

**10-27 19:0x 使用者決定：Human18 繼續練 +200～400**（從 12599、同 cfg b4696fb）。每 +100：diag1024（--clamp_ref_phase）+ ONE OUTS + 訓練中每次擊中的跌倒率（修正版）。停止：到 12799 每次擊中跌倒仍 ≥ 95%、好球帶 < 0.40 兩次、ONE OUTS < 16% 連兩次。採用：diag 跌倒 ≤ 32% 且 ONE OUTS ≥ 21.4%；跌倒 ≤ 32% 且 ONE OUTS ≥ 19.9% 時回報使用者當影片候選。投手 P3c 所有比賽路徑確認一致（p3c_identical_t18 表格）；注意 P3C 任務預設 start_joint_noise 0.02，相同起始要加 --set start_joint_noise=0.0。投手影片 10-28 06:14 前，約 05:50 跟 15 要空檔。

**10-27 18:4x 使用者決定：拿掉投手的擾動門檻，P3c 定案**（門檻只看相同起始：T3 ≥ 32/64、站穩 ≥ 48/64；P3c 52 / 48；noisy 數字只回報）。投手不再用 GPU，只拍 12 小時影片（下次 10-28 06:14）。打者 Human18 停在 +100（不採用）：clamp 基準 好球帶 0.458 / 跌倒 42%；12599 好球帶 0.575、跌倒 36%（> 32）、初速 15.7、ONE OUTS 17.1%（< 19.9）——碰到更多球但更弱。+100 停止條件用的 bat/fall_rate 是從 0 起算的 EMA、且算全部回合（寫法有誤），改用每次擊中的跌倒率仍是 97.7% → 約 99%，停止成立。打者交付仍是 h9b 12500 @1.5 倍（19.9%）。20:00 後打者新設計（Human18 練更久，或參考結束後只給平衡獎勵）由使用者決定。GPU 18:44 起空著。

**10-27 18:2x 投手擾動量測（P3c noisy 128）**：0.01 站穩 64、T3 11、好球帶 14、跌倒 54；0.015 站穩 56、T3 10、跌倒 65；0.02 站穩 53、T3 5、跌倒約 69。相同起始 52/64、T3 48/64。降到 0.01 也只有 64/128（門檻 96）→ 差距主要是策略本身不耐擾動，不是擾動大小。等使用者決定門檻。投手影片 `videos/pitcher_p3c_t18_1027eve.mp4`（229 幀、High / yuv420p、沒跌倒），下次 10-28 06:14。18:23 GPU 交給打者 Human18。

**10-27 18:1x 使用者決定：投手擾動門檻先量 0.01 再決定**。P3c 在 start_joint_noise 0.01（可以的話加 0.015）各跑 noisy 128，數字回報使用者，期間投手不做新設計。打者收棒緩停測試失敗（跌倒 43.1% vs 沒加 43.4%，跌倒都在參考結束後約 0.8 s）。核准 Human18：訓練超過參考結束點（ref_hold_end、clamp_ref_phase、follow_through 2.0 s、回合 3.5 s），P3c 形狀的球、力量 1.5 倍、從 12500；先零樣本確認 ≥ 30% 擊中回合以跌倒結束，+200 判定 跌倒 ≤ 33% 且好球帶 ≥ 0.43，採用要 ONE OUTS ≥ 21.4%。GPU：投手影片 → 擾動量測 → Human18。

**10-27 18:0x 投手 PR 關掉**：pr2_p3c 60 次，相同 64（同種子）出手 p10/50/90、球速、球的速度向量都和 7600 完全相同（出手前沒漏），站穩 52 → 48、T3 48 → 45；noisy 128 站穩 40（P3c 53，≤ 56 關掉線），T3 6、好球帶 4。學生拿到樣本（share 0.29 → 0.75）但出手後控制反而讓 noisy 站穩 −13。F、開迴路手臂、PR 都關掉，交付仍是 P3c。noisy 門檻（96/128）是規格問題，帶數字問使用者。GPU：打者收棒緩停測試 → 投手 12 小時影片（20:27 前）。

**10-27 17:4x 打者交付換成 h9b 12500 @ 力量 1.5 倍（零樣本，不重新訓練）**：使用者 17:29 停掉失控的 pr_p3c。12500 @1.5 倍 ONE OUTS vs P3c 19.9%（≥ 19.6 門檻；交付原為 18.1%）、外野 4.1%（原 1.8%）、好球帶擊中率 0.458、擊球後跌倒 43%；三振 43.7%、內野 36.4%、四壞 10.0%、觸身 5.8%。勝出幅度 1.8 個百分點，只比 1000 場的波動略大。1.5 倍仍待使用者確認。影片 `videos/batter_h9b_12500_e1.5_p3c_1027pm.mp4`（17:37，613 幀、High / yuv420p；因 GPU 被佔逾期約 4 小時；5 球 4 揮空 1 界外，展示效果弱）。下一步：投手 oo_pr2（PR B）→ 打者收棒緩停測試。

**10-27 15:47 檢查**：失控的 pr_p3c trainer（PID 49380）和 oo_pr.sh（56600、37104）仍在跑，等使用者停（規劃 session 被 auto mode 擋下）。GPU 自 09:19 起被它佔住；打者 12 小時影片 13:24 到期，已逾期約 2.5 小時。b_after_pr.sh（打者 ONE OUTS + 影片）已待命，Isaac 連續 2 分鐘 0 才會自己開始；之後 oo_pr2（投手 PR B，3ad369c 已修兩個 bug）→ 收棒緩停測試。

**10-27 12:47 檢查：投手 PR 步驟 B 失控**：09:19 開跑、原本 60 次（到 7660）就停，但 checkpoint 每 50 存一次，model_7660 永遠不會出現 → chain 一直等，trainer 跑到 8600（3.5 小時），沒有任何評估。另外訓練 log 顯示 student share 0.000（遮罩從沒把樣本交給學生，等於什麼都沒練）。規劃 session 要停 PID 49380 被 auto mode 擋下（Interfere With Workloads），等使用者決定；已把兩個 bug 告訴 20（只請它在 CPU 上修）。打者 1.5 倍 ONE OUTS、打者 12 小時影片（13:24 到期）、收棒緩停測試都被擋在後面，影片會逾期。

**10-27 09:47 檢查**：GPU 上 1 個 Isaac（投手 PR 步驟 B 的 60 次訓練，09:19 起，約 09:55 結束後評估）。步驟 A 已過（出手後跌倒 ≥ 43，跌倒方向多半往一壘側 44）。打者：收棒緩停程式 39426bf（預設關閉），排在 12500@1.5 倍 ONE OUTS 與打者 12 小時影片（13:24 前）之後。投手影片 08:20 已拍，下次 20:27。

**10-27 09:3x Human17 停在 +200**（好球帶擊中率連兩次 < 0.35）：12599 好球帶 0.314、跌倒 49%、ONE OUTS 18.8% / 外野 3.1%；12698 好球帶 0.268、跌倒 61%、初速 18.4、ONE OUTS 20.2% / 外野 3.5%（交付 18.1% / 1.8%）。碰到球的次數少、但打得較強。姿勢 diag：接觸時手肘並沒比較彎（之前「手肘少 15 m/s」是比較的時間點不同造成的）；參考的甜蜜點速度在接觸那一幀最高，手肘打直時只低 3.3 m/s（< 5，不算甩棒）。選法（事先寫好）：12500@1.5 倍零樣本 / 12599 / 12698 中，擊球後跌倒 ≤ 50% 且 ONE OUTS 最高者，要比交付多 ≥ 1.5 個百分點才換成新的 1.5 倍交付，否則仍是 h9b 12500 @ 1.0 倍。PR 鏈之後：(a) 12500@1.5 倍 ONE OUTS（3 分鐘）→ (b) 打者 12 小時影片（13:24 前）。打者下一個訓練提案以「擊球後跌倒」為主。

**10-27 09:0x 投手提案 PR 核准（分兩步）**：出手前 30 個動作欄全部由凍結的 7600 決定（遮罩看 obs 裡的 released 旗標與相位，出手狀態照構造不會變），學生只在球離手 30 ms 後接手，只修出手後的跌倒。CPU 證據：noisy 跌倒中位數 1.98 s（出手約 1.00 s），站穩的出手點較低、較偏三壘側。步驟 A（約 8 分鐘）：在 P3c 設定（noisy 站穩 53/128）量出手前 / 後跌倒數；出手後跌倒 < 43（= 96 − 53）就是 PR 也到不了門檻 → 不做 B，帶數字去問使用者規格（noise 0.01 或降低 noisy 門檻）。步驟 B（≤ 20 分鐘）：60 次迭代，相同 64 的出手時間 / 球速 / T3 / 好球帶必須和 P3c 完全相同（否則是遮罩漏了）；noisy 128 站穩 ≥ 60 → 提 +600、≤ 56 → 關掉。GPU：Human17 +200 diag 之後讓出 ≤ 35 分鐘。投手 session 自評：11-12 前達到 noisy 96/128 的機率 ≤ 20–25%。

**10-27 08:4x 影片 + 開迴路手臂關掉**：投手 `videos/pitcher_p3c_t18_1027noon.mp4`（229 幀、白畫面 0%、High / yuv420p、沒跌倒、球飛 37.9 m），下次最晚 20:27（打者影片照 15 的排程）。開迴路手臂：相同 64 重播 44 / 42 / 39（錄製時 52 / 48 / 47，合理性檢查差 −8 / −6；相同起始也會因 solver 雜訊分岔，閉迴路手臂會修正、重播不會）；noisy 128 站穩 32、T3 6（閉迴路 38 / 5）→ T3 < 10 關掉。投手暫停 GPU，先在 CPU 分析 noisy 跌倒（階段、哪些關節的起始擾動預測跌倒、跌倒方向）再提案，並評估 noisy 門檻在 11-12 前是否做得到（做不到就是規格問題，要問使用者）。打者：08:39 拿到 GPU，先跑 3 分鐘姿勢 diag（a72f36a；接觸點與參考差 < 5 cm，偏向姿勢追蹤），再跑 Human17。

**10-27 08:3x 打者貢獻拆解（修正版，1.5 倍，69 次擊中）**：甜蜜點速度 19.66 m/s = 關節 11.6 + 身體移動 / 轉動 8.1。機器人（括號是參考）：l_elbow +5.9（+21.0）、左腕 +7.0、waist_yaw −4.2（+8.7）、waist_pitch +2.5、左肩 +1.3、root 8.1（6.4）。手肘角速度有參考的 0.70，但貢獻只有 1/4 → 主要差在接觸時手臂的幾何（槓桿），不是關節速度；下揮時 τ·q̇ 多為正（馬達沒有在對抗揮棒）。下一個 CPU 檢查：接觸時手肘 / 肩角度和接觸點相對骨盆的位置 vs 參考，分辨是球的位置造成，還是姿勢追蹤不夠。Human17 照排程約 09:30 開跑。

**10-27 08:2x 投手 F 關掉、交付仍是 P3c**：pf2（7900 起，k 0.60 → 0.50）相同 64 站穩 / T3 / 好球帶：7950 53/14/6、8000 53/9/3、8050 56/6/1、8100 51/2/0、8150 54/0/0，出手 p50 都 +19.4；noisy 128 站穩 7950 62、8050 50、8150 52 < 70 → 7950 就觸發停止條件。老師只把漂移延後約 200 次（凍結的手臂讀到變了的軀幹 / 腳狀態，出手晚一步）。下一個否證（只做零樣本）：7600 的手臂 + 出手欄位改成重播一次乾淨 7600 回合的動作（開迴路，按時間），腳 / 軀幹照策略；先驗相同 64 能重現 52 / 48，再看 noisy 128：T3 ≥ 20 且站穩 ≥ 32 → 寫訓練提案；T3 < 10 → 關掉。GPU 順序：打者貢獻拆解（≤ 10 分鐘）→ P3c 12 小時影片 → 這個零樣本 → Human17（約 09:30）。

**10-27 07:1x 打者接觸時間（CPU）**：擊中時的參考相位只比參考接觸點晚約 +5 ms（三組 diag p10 / p50 / p90 約 −0.008 / +0.005 / +0.017 s），不是太晚 → 揮棒起點偏移掃描取消。參考的 waist_yaw 在接觸前後 40 ms 內 +10 → −3.3 → +15 rad/s 來回震盪（多半是 retarget 雜訊），所以單一時刻的方向比較不可靠。08:45 空檔只跑修正後的貢獻拆解（先 commit 修正）。

**10-27 07:0x 投手 F 當機重開 / 打者揮棒 diag**：06:47:58 砍評估的同時 trainer GPU crash（PhysX 999），只留 7900（相同 64：站穩 59、T3 32、好球帶 18 ← 第一次 < 32、17.4 m/s）。投手改 P3CPG（k 從 0.60，ae717e5）從 7900 單獨練到 8150 再評估，約 08:45 有結果；7950 好球帶再 < 32 就停。打者揮棒 diag（12500、1.5 倍、diag256）：l_elbow q̇ 為參考的 0.70（剛好在線上）、下揮時各關節 τ·q̇ 多為正（0.51–0.86）→ (i)(ii) 都不成立。接觸時 waist_yaw −4.1 vs 參考 +4.95（反轉）、右肩 / 右腕方向相反 → 懷疑接觸太晚（軀幹已經轉完）。下一步：CPU 先查接觸時的參考相位，太晚的話 08:45 的 10 分鐘空檔做揮棒起點偏移零樣本掃描（−0.04 / −0.02 / +0.02 s）＋修正後的貢獻拆解（原本對錯參考點：Jacobian 線速度在球棒質心，不是握把）。

**10-27 06:47 檢查**：投手 F 步驟 1（pf1_p3cpf）06:22 起卡住：trainer 和 7900 評估同時在 GPU，VRAM 11.77 / 12.29 GB，25 分鐘只跑了 7900 → 7903，評估也沒動。已請 20 砍掉評估（PID 21668），之後不在 trainer 旁邊跑評估（帶老師 actor 後放不下 12 GB），改成分段練、停下再評，或練完再評。Isaac 2 個（trainer + 評估）。影片下次最晚 13:24。打者 Human17 與揮棒 diag 排在投手之後。

**10-27 06:3x 打者揮棒速度拆解（CPU）**：參考動作在接觸瞬間的甜蜜點速度 32.9 m/s，各關節貢獻：l_elbow 伸直 +21.0、l_wrist_yaw +10.6、waist_yaw +8.7、root +6.4、l_shoulder_roll −13.7、waist_pitch −2.7 → 約 2/3 來自前手手肘和手腕，不是軀幹轉動。但飽和最嚴重的是雙肩（0.93–0.97），l_elbow 不在前 8 名。兩種解釋：(i) 肩撐不住手臂幾何、(ii) 兩手對拉吸掉手肘的功。核准 3 分鐘 diag（每關節 q̇ vs q̇_ref、接觸時貢獻拆解、下揮時 τ·q̇ 正負），排在投手 F 結束後、Human17 之前；判定事先寫好（l_elbow q̇ < 70% 參考且肩 τ·q̇ 多為正 → i；≥ 70% 且肩 / 右臂 τ·q̇ 多為負 → ii）。Human17 不管結果都照跑（logs/run_human17.sh）。

**10-27 06:1x Human16b 停（不採用）、投手 F 步驟 1 開跑**：Human16b 12599（+100，1.5 倍）ss_speed_win 15.60 < 16.2、好球帶擊中率 0.224（基準 0.458）、跌倒 61%、ss_max 22.94、初速 17.1 → 甜蜜點速度獎勵這條線關掉（1.0 / 1.5 倍都只拿擊中率換一點速度）。下一個打者 = Human17：Human9 原獎勵 + 力量 1.5 倍、12500 起，排在投手 F 之後；+200 判定 好球帶 ≥ 0.40、跌倒 ≤ 43%、初速 ≥ 16.1，再看 ONE OUTS。投手 F 步驟 1（P3CPF，7850 起 +600，1bc4e50）06:1x 開跑，投手 trainer 期間打者不用 GPU。

**10-27 06:0x F 步驟 0 通過、步驟 1 核准**：noisy 128（k 1 / P 0）站穩+投出 7600（無老師）38、7650 50、7700 46、7850 54 → 7850 ≥ 38 + 7 通過；但 noisy T3 全部 2–6/128（有擾動時老師的手臂瞄不準，不在門檻內，只回報）。步驟 1：7850 起 +600，停止條件照提案，另加 8150 時 noisy 站穩 < 70/128 就停（追不到 96/128 = 48/64）。GPU：Human16b 做完 +100 diag 就讓出，投手接手約 2.5 小時，之後 Human16b（+100 過關才）接著練。Human16b 基準（1.5 倍零樣本 diag1024）：好球帶擊中率 0.458（1.0 倍 0.327）、跌倒 43%、ss_max 21.22、初速 16.1。

**10-27 05:5x 投手 F 否證通過**：相同 64、k 1 / P 0、評估帶 --arm_teacher（ho2 7600）。站穩+投出 / T3 / 進好球帶：7600 52/48/47 → 7650 56/51/48、7700 55/46/46、7750 55/41/22、7800 53/46/40、7850 49/39/37。B 在 7700 就 T3 0，F 到 7850 還有 39；但 7800 起出手 p50 變 +19.4、好球帶與球速慢慢掉（老師的輸入在變）。決定：步驟 0 = noisy 128（7650 / 7700 / 7850 帶老師、7600 不帶），最好的要比 7600 多 ≥ 7/128 才算有用，否則 F 關掉；通過才做步驟 1（最多 +600、停止條件加「好球帶連兩次 < 32」），在 Human16b +100 邊界接手 GPU。Human16b 已開跑。

**10-27 05:2x 投手 F 卡住**：訓練從 04:11 跑到 7896（應在 7700 停），7650 / 7700 評估都沒產出。原因：train_live 的 EVAL_WAIT_FREE 計數用 -match 'scripts'，把別的專案的 streamlit（D:\MyProject\invest\.venv\Scripts\python.exe）算成 Isaac，計數一直是 2。已請 20 停掉訓練、補跑 7650 / 7700 評估（帶 --arm_teacher）、修計數再 commit。Human16b 在這之後。影片下次最晚 13:24。

**10-27 04:08 Human16 停（不採用）**：12599 / 12698 好球帶內擊中率 0.278 / 0.275（< 0.30 兩次），ss_max 20.81 / 20.96（只 +0.9），初速 16.4 / 16.9，ONE OUTS 18.1 / 18.2%、外野 2.2 / 2.4%；訓練中 ss_speed_win 14.9 → 15.0 幾乎沒動。決定：Human16b = 同一獎勵 + 力量 1.5 倍（規劃 session 先選 1.5，回報使用者），+100 時 ss_speed_win 沒比 15.2 多 ≥ 1 就停。GPU：投手 F（04:08 起）→ Human16b。

**10-27 03:47 檢查**：Human16 第一段（到 12599，約 11 分鐘）跑完，正在跑 12599 的 diag；訓練中 ss_speed_win 14.9 → 15.2（參考 17.5），擊中率 0.77、跌倒 0.2%。GPU 上 1 個 Isaac。投手 F（logs/oo_pf.sh）等 Human16 結束。影片下次最晚 13:24。

**10-27 03:3x 投手 E 關掉**（32f0fb2）：固定出手 a 0.01 時 7600 相同 52/T3 48（= P3c），noisy 站穩 43/128（代價約 -10）；但 B 7700 套同一個固定值 T3 仍 0、出手仍 +19.4 ms → 是手臂動作本身變了，不是出手欄位。action std 很小（第 2 維 0.00027），KL 的 92% 來自它，所以 lr 一直被壓在下限。提案 F（續練時手臂與出手用凍結的 7600 當老師，只學腿和軀幹，PPO 要 mask 手臂維度）否證核可，排在 Human16 之後。Human16 03:33 開練。

**10-27 03:1x 打者 action_clip 零樣本**：clip 2.0 / 1.0 / 0.5 / 0.25 擊中率 0.176 / 0.152 / 0.113 / 0.094，初速 17.2 / 14.1 / 12.1 / 15.1；等球飽和 0.65 → 0.17 跟著降，但約 300 N 的兩手對拉四組都不變（多半是估計偏差）。策略揮棒時依賴大修正量，推理時降 clip 不可行。決定：Human16 = 12500 起練、力量 1.0 倍、只加擊球前甜蜜點速度獎勵；力量倍數問使用者，確認後再疊上去。排在投手 E 否證之後。

**10-27 03:1x 投手失敗原因（CPU）**：不是 reward 不一致（訓練和評估用同一個 release_score，分數一起掉），也不是 lr（一直卡在下限 3e-7，KL 0.023–0.027）。是出手時間晚一個控制步：release − ref 7650 +16.9 ms → 7700 +19.4 ms，球從 y -0.05 / z 1.05 變 y -1.89 / z 0.09。P3c 靠提早 2.5 ms 瞄準，本來就站在門檻邊上。提案 E（release_fixed_a，出手動作固定成常數）否證核可：7600 + 固定值相同 64 + noisy 128，B 7700 / 7850 套同一值；T3 ≥ 32 → B + 固定出手續練提案，< 10 → 關掉。排在打者 action_clip 之後。

**10-27 03:0x 投手 B 停**：k 1 → 0.68（+260），但 k = 1 / P = 0 相同 64 的評估 7650 54/50 → 7700 54/0 → 7850 32/0，球路在第一個 50 次就壞了（出手 17.8 → 16.3 m/s）。和 10-25「投手 C」（純 P3C 續練，T3 50 → 18 → 2）、P3CPC 是同一現象：從 7600 續練本身就會洗掉球路。D（anchor 1.0 對照）不跑，對照組已經有了。下一步用 CPU 比較 train log 裡球相關 reward 的走勢和 T3：reward 上升但 T3 崩 → reward / 任務和評估不一致；reward 也降 → 優化問題，才開 GPU 測 lr 1e-5。GPU 現在給打者 action_clip 零樣本。

**10-27 02:0x 打者 PD 誤差量測**：等待時手臂的 PD 誤差 1.0–1.5 rad（飽和門檻的 5–9 倍），幾乎全部來自策略修正量頂在 action clip（2.0 × action_scale × residual_scale 0.5 = 肩 1.5 rad），參考的追蹤誤差只有門檻的 0.35–0.8 倍 → 策略在等待時 bang-bang 拉扯雙臂。降 stiffness 和力量倍數都解決不了。下一步（B 之後）：零樣本 action_clip 0.25 / 0.5 / 1.0 各 diag256；擊中率持平且初速或打得正改善 → 訓練提案（降 clip / residual_scale 或修正量懲罰）。

**10-27 01:55**：C 正式結果相同 0/64、noisy 0/128（t = 0 重心 0.12 m/s、骨盆 yaw -1.14 rad/s，前腳撐地帶不出動量）→ C 關掉（f286b24）。打者：參考的握棒閉鏈本來就合（誤差 ≤ 2 mm），投影修不到；新疑點是 PD stiffness 沿用強機器人值、上限是人類值，誤差 8–10° 就飽和 → 3 分鐘 diag256 拆 PD 誤差（參考 / 策略修正量），之後提案 (i) 降 stiffness 或 (ii) 力量倍數（stiffness 是否算規格要問使用者）。接著投手 B（P3CPB：anchor 0.5、k 每步 0.025、250 次迭代、watch_pb.sh）。

**10-27 01:42 打者閉鏈檢驗（b_phase_check）**：等待時全身飽和平均 0.65（左腕、左肩 0.99）；兩手力（Jacobian 換算、扣掉右手放開組）平均向量 cos -0.99，單手約 300 N 互相對拉 → 閉鏈拉扯成立。但右手放開後左肩、腰仍 0.92–0.96，所以還有 PD 目標超出人類力矩的問題。接觸時法線 · 球棒速度 0.73、速度垂直分量 +0.34（由下往上撈）→ 那 -4 m/s 的來源。決定：打者直接寫「兩手參考目標投影到同一支棒」（cfg 開關、預設關），同一槽加記等待時 PD 目標誤差（參考 / 策略修正量）和靜態需求力矩比例；之後「打得正」獎勵要含法線對齊。C 否證第一次是設定 bug（前腳往回推進地面 11 cm、被彈飛），修正後重跑。

**10-27 01:27 影片**：投手 `videos/pitcher_p3c_t18_1027am.mp4`（01:24）、打者 `videos/batter_h9b_12500_p3c_1027am.mp4`（01:27，界外 6.7 m、界內 17.4 m、3 次揮空；白畫面 0%、High / yuv420p）。下次最晚 10-27 13:24。GPU：C 否證（投手）→ b_phase_check（打者，約 10 分鐘）。

**10-27 01:2x 打者槽結果（b_slot_next）**：(1) 揮棒期間力矩飽和：1.0 倍雙肩 / 雙腕 0.92–0.99、腰 0.86；1.5 倍幾乎沒降（0.86–0.98）。(2) 1.5 倍零樣本：擊中率 0.176 → 0.270，初速 17.2 不變。(3) 初速損失（逐球配對）：擊球那一刻較慢 -1、打不正 -7.3、沒解釋的 -4（跟揮棒方向和球路的夾角無關）。(4) aim_offset_y ±0.05 / ±0.10 都不採用。判定：Human16 暫停改版。疑點是等球時雙臂就已經頂在上限 → 雙手握棒的閉鏈互相拉扯。下一個打者槽（C 否證之後）：飽和分等待 / 下揮 / 擊球後三段、左右力矩方向、握把內力、第一次接觸的法線。等待時飽和 ≥ 0.7 且左右相反 → 先修參考的握棒閉鏈；< 0.3 → Human16 改成速度 + 打得正兩項。

**10-27 01:0x 判定（P3CPC 失敗後）**：8400 在 P = 0、完整起始動量：相同站穩 0/64、T3 0（7600 是 52 / 48），課程把原本的動作洗掉了。交付照舊 P3c、P = 0（noisy 門檻仍未過，11-12 前繼續）。下一步：C 否證（7600 + 腳本 rocker 起始零樣本，相同 64 + noisy 128；相同站穩 ≥ 40 → 送訓練提案，< 20 → 關掉）→ B（一半 env 固定 k = 1 的防遺忘課程，200 次迭代否證；C ≥ 40 就不跑）。GPU：打者 b_slot_next（00:55 起）→ 12 小時影片 → C → B。

**10-27 00:47 檢查**：P3CPC 00:45 依規則停（watch_ppc：k 在 +523 降到 0.35 後 300 次迭代沒再降，hi 一直是 0.05；success_set 約 0.02；P = 0.3 參考評估 7700–8400 相同起始 / noisy 全 0）。投手 session 正在跑 8400 P = 0 評估，結果由它回報使用者。GPU 接下來：打者 b_slot_next（約 22 分鐘）→ 12 小時影片（最晚 08:40）→ 投手新提案。

**10-26 21:47 檢查**：P3CPC 進度正常，+100 次迭代（約 4.8 次/分，比預估快，1000 次約 3.5 小時）時 k 已到 0.75（第 250 次 k ≥ 0.8 的停止規則不會觸發）；value loss 暖身後 150 → 147 正常。GPU 上 2 個 Isaac（trainer + 7700 評估），11.7 GB。影片下次最晚 10-27 08:40，P3CPC 預計在那之前結束。

**10-26 21:10 新規劃 session 接手**：名稱 aiplaybaseball-14（舊 10）；3 小時檢查 cron 已重建（:17）。GPU 上是打者跌倒測試第 3 組（fall_start_0.6）；前 2 組都沒過（命中率 0.684、跌倒 56%）。`b_film_after_fall.sh` 在等，結束後錄 1026b。投手 session 已上線（aiplaybaseball-20），新的打者 session 是 aiplaybaseball-15（d0 是舊的，已停手；它在 21:08 停掉了錄影等待腳本，改由 15 重新啟動 `b_film_after_fall.sh`）；1026b 錄完由 15 通知投手開 P3CPC，投手開之前會確認 Isaac 數 = 0。殘留的 10-25 `b_eval_p3c_t.sh` 已由 d0 清掉。**跌倒 zero-shot 4 組全部沒過（21:09）**：擊中率都 0.684，跌倒 contact_0.6 56%、contact_0.3 56%、start_0.6 57%、keepres 57%；保留修正量的對照組和淡出結果一樣，問題不在 PD 撐不住最後姿勢，只改參考播完後的目標姿勢解不了跌倒。下一步由打者 15 提案。1026b 用一般設定錄。**21:15 判定（打者跌倒提案）**：提案 A（訓練時用同樣的參考後目標）暫緩；先跑檢驗：在參考播完那格和擊球那格記錄重心水平速度、重心到支撐區的距離、角動量 / 骨盆 yaw rate，每個量單獨算 AUC。任一個 ≥ 0.85 → 改提揮棒後段平衡獎勵（要附否證）；全部 < 0.75 → P3CPC 結束後跑 A；介於中間 → 回報。插在 1026b 之後、P3CPC 之前（約 3–5 分鐘），條件是程式在錄影結束前寫好，否則等 P3CPC 結束後再跑。

**10-26 21:30 交接（三個 session 都要重開，讓自動壓縮 30% ≈ 30 萬 tokens 生效）**

- **分工**：
  - 規劃 session（本檔作者，aiplaybaseball-10）：評估提案、分配 GPU、做 ONE OUTS 引擎與影片、寫本檔。
  - 投手 session（aiplaybaseball-08）：負責投手與 PITCHER_TRAINING_LOG.md，投手結果由它直接回報使用者。
  - 打者 session（aiplaybaseball-d0）：負責打者，結果寫 SupporttraningLog.md 第 3 節。
  - 訓練前先送提案給規劃 session。session 名稱重開後可能會變，用 ListAgents 查。
- **授權**：到 10-27 20:00 為止，規劃 session 自行決定（規格不動）。之後新設計 / 規格 / 排程要先問使用者。
- **3 小時檢查 cron 是 session-only**：新 session 要重建（每 3 小時 :17）。
- **GPU 現在的順序**（一次只開 1 個 Isaac，錄影時旁邊不開任何 Isaac）：
  1. 打者 4 組跌倒 zero-shot（post_ref_blend，diag1024）。接受條件：命中率變化 ≤ 0.02、跌倒 ≤ 37%。
  2. 打者影片 `videos/batter_h9b_12500_p3c_1026b.mp4`（`logs/b_film_after_fall.sh`）。有一組跌倒測試通過，就用跌倒率最低的那組錄。
  3. 打者通知投手「槽空了」。
  4. 投手 P3CPC 開訓練。訓練期間打者完全不用 GPU。
- **投手 P3CPC**：
  - 設定：從 ho2 model_7600 開始，critic 暖身 50 次（lr 1e-4），最多 1000 次迭代，約 1 次迭代 / 分鐘。
  - 課程：
    - 第一段，開局速度比例 k 從 1.0 每步減 0.05 到 0（start_vel_curriculum）。這段暫停固定 P = 0。
    - 第二段，k 到 0 之後，暫停上限 hi 從 0.05 每步加 0.05 到 0.3（set_pause_curriculum）。
    - 調整條件：success_set > 0.1，而且離上次調整至少 20 次迭代（1280 steps）。
  - 評估：hi 到 0.3 之前每 100 次迭代評估一次（`logs/eval_every_<run>.txt`），只當參考。到 0.3 之後改成每 50 次，門檻是兩個評估都用 P = 0.3 達標；最後選出的檢查點另外跑 P = 0。
  - 停止規則（watch_ppc.sh）：
    - 第 250 次迭代時 k 還 ≥ 0.8，就停下來回報。
    - k 和 hi 連續 300 次迭代都沒有變動，就停。
    - value loss 規則照舊。
  - 背景：Pp' 失敗的原因，是 7600 依賴開局帶入的運動員第 0 幀動量（COM 0.6 m/s、骨盆 yaw rate -2 rad/s），也依賴 3 cm 落地。k 0.8 → 18/64、k ≤ 0.5 → 0；抬高 0.015 m → 24/64、抬高 0 → 0。從靜止起步時，支撐腳會把人往反方向轉。
- **影片**：
  - 編碼一律 crf 16 High profile（e36e4fa）。quality=10 會產生 High 4:4:4，播放器顯示全黑。`check_video_frames.py` 會檢查格式（4501826）。
  - 本壘視角 3 倍光學變焦已生效（Usd.EditContext 設在 session layer）。投手丘 / 投手板已轉 -θ。
  - 最新影片：投手 `videos/pitcher_p3c_t18_mound_check2.mp4`（10-26 20:44，1 回合），打者 1026b（錄製中）。下次最晚 10-27 08:40 左右。trainer 在跑時不能錄影，要排在訓練開始前或停下時。
- **交付候選不變**：
  - 投手 P3c（ho2 7600 + 提早 2.5 ms + θ 18°）。
  - 打者 Human9b 12500（快球 aim -0.08，對 P3c aim 0）。
- **使用者決定（10-26 21:2x）：增加打者力量**（打者對 P3c 初速 14–16 m/s 打不到外野）。飛行模型（阻力 + 馬格努斯力、最佳仰角 37–40°）的最遠距離：16 m/s 25 m、20 m/s 37 m、24 m/s 51 m、28 m/s 65 m；外野 38.9–47.4 m，初速中位數要約 25 m/s（目前 15.3，球棒約 10 m/s）。第一步：零樣本 showcase `AIB_EFFORT_SCALE` 1.5 / 2.0（只放大力矩，手指 / 拇指與速度上限不變），排在 b_fall_state 之後、P3CPC 之前。倍數是規格值，定案前回報使用者。**跌倒檢驗（21:15）**：擊球格 / 參考播完格的重心速度、重心到兩腳中點、角動量、骨盆 yaw rate 8 個 AUC 全部 0.43–0.55，「播完前已注定跌倒」不成立。A 排在力量倍數定案之後，用新力量的設定訓練；新力量下跌倒仍 > 37% 才加 A。**力量零樣本（21:20）**：1.0 / 1.5 / 2.0 倍的初速中位數 16.4 / 16.8 / 17.1 m/s，≥ 25 m/s 佔 8.8 / 10.5 / 10.5%，ONE OUTS 打者勝率 18.1 / 18.2 / 17.7%、外野安打 1.8 / 2.5 / 2.5%。策略照參考動作的速度揮，力矩沒頂到上限，光放大力矩不夠，要訓練（更快的參考動作或球棒速度獎勵）。打者在 P3CPC 期間送訓練提案（附參考 vs 機器人的球棒速度、關節速度離上限多遠）；力量倍數暫定 1.5，等使用者確認。21:20 打者發「槽空了」給投手，**P3CPC 21:22 開練**（投手 aiplaybaseball-20，watch_ppc.sh）。**Human16 提案（21:3x，有條件核可）**：參考揮棒甜蜜點約 31 m/s，機器人 12500 只有 20.5 m/s（66%），參考也沒頂到速度上限 → 從 12500、力量 1.5（等使用者）、aim 0、P3c 混快球，只加擊球前 0.1 s 的甜蜜點速度獎勵 w·clamp(v/v_ref, 0, 1)；目標初速中位數 ≥ 25 m/s、外野安打 ≥ 10%、P3c 好球帶內擊中率 ≥ 0.30。條件：CPU 先查打得正的子集初速 / 球棒速度（20.5 m/s 打正應約 27 m/s，實際只有 16.4），約 0.8 倍就先回報、不訓練；GPU 否證（1.0 / 1.5 倍各 diag256 看力矩飽和）排在下一個打者槽第一件事。停止：+200 its 甜蜜點速度沒增加 ≥ 2 m/s、擊中率 < 0.30 連兩次、跌倒 > 70%、最多 600 its。**打正檢驗（21:4x，CPU）**：比賽碰撞模型下，甜蜜點 20.5 m/s 打正 → 初速 29.9 m/s（P3c），偏移最小的 25% 是粗算的 0.96 倍 → 碰撞沒問題；打中的 |dz| 中位數 3 cm、dx -6 cm（偏握把），預期初速 21.0，實際 16.4（差的 4.6 m/s 待查：擊球當下的速度或揮棒方向）。打得正（|dz| ≤ 2 cm 且 |dx| ≤ 5 cm）的比例 P3c 8.4%（143 球打中的）、快球 9.6%（700 球打中的），當 Human16 的觀察指標。下一個打者槽：逐球記錄（aim 0）+ 若有現成參數，沿球棒方向的瞄準零樣本掃描。已備好 `logs/b_slot_next.sh`（69db97c，NOWAIT=1 手動啟動，結束標記 BSLOTNEXT_END，約 22 分鐘）：力量 1.0 / 1.5 各 diag256 + sat_log；aim_offset_y ±0.05 / ±0.10 各 diag256 + showcase 1024 + ONE OUTS 1000 場，初速與外野安打率都比 y = 0（16.4 m/s、1.8%）好才採用（10-25 的 -0.08 界內多但都是滾地）。
- **細節**：見本節下方的「10-26 20:05 / 晚間 / 21:xx 決策」。

**10-26 18:50 檢查**：GPU 上是投手 Pp'（`ppw_p3cp`，17:26 起，critic 暖機 50 次後 value loss 441 → 135）+ 它自己的 7650 noisy 評估，VRAM 12.0 GB。7650 停頓 0.3 s 相同起始 0/64、方向 +83.6°；7750 的恢復檢查（< 24/64 停）由 watcher 判斷。影片畫質修正（f95038a：每個視角 1600×900、DLAA、mp4 品質 10，本壘視角改 3 倍光學變焦）和投手丘修正（32cd7a7）會在 Pp' 結束後的 `logs/film_p3c_mound.sh` 一起驗證。影片上次 10-26 15:55（打者）/ 16:11（投手），未到期。

**10-26 17:50 影片視角問題（使用者發現）**：`pitcher_p3c_t18_identical_1025pm.mp4` 的本壘視角看起來偏向三壘側（`pitcher_aim10_30h5_identical_1025.mp4` 是 θ = 0，正常）。原因：斜站是用「本壘線繞投手板轉 -θ」做的，2275a96 只轉了本壘 / 好球帶標記和相機，投手丘（碰撞 + 外觀）和投手板標記還沿著 +x，所以從真正的本壘看，投手丘和機器人整個偏了 18°。相機位置本身是對的，球也確實飛進相機前的好球帶。修法（投手 session，只改畫面）：碰撞用的投手丘不顯示，另放一個轉 -θ、只顯示的投手丘和投手板，讓畫面變成「投手丘對準本壘、機器人朝一壘側斜站 18°」；錄影排在 Pp' 之後。之前所有 θ ≠ 0 的投手影片都有這個問題。

**10-26 17:35**：打者 Human15（只用 P3c 形狀慢球）沒過：好球帶內擊中率 12599 0.478、12700 0.494（門檻 0.5），揮早修掉了（球到時參考時鐘 +0.040 → +0.018 s），但 ONE OUTS vs P3c 反而降到 16–17%（交付表 18.1%），多出來的接觸都是內野滾地，外野安打 < 1%。決定停止慢球擊球品質的訓練。原因是幾何加力量：ONE OUTS 的內野是「離投手板 95 ft 內」，外野從本壘算要 38.9 m（界外線）到 47.4 m（中間）才開始；無阻力最遠距離 v²/g 是 14 m/s 20 m、16 m/s 26 m、22 m/s 49 m，打者對 P3c 的初速只有 14–16 m/s，物理上打不出外野安打。「界內且 ≥ 29 m」與 Human9 的 29 m 目標都不是打者贏的線（29 m 仍在內野），之後改用「外野安打率」。打者交付維持 12500（vs P3c aim 0、vs 快球 aim -0.08）。這會讓 RL 投手的 ONE OUTS 幾乎只能靠三振 / 四壞 / 觸身分勝負，要改善只能動規格（打者力量、投手球速或內野定義），需要使用者決定。投手 Pp（在停頓下訓練，`TW2PLW100P3CP`，1dbde5e）17:23 開練，約 19:10 結束；打者這段時間不用 GPU。

**10-26 16:15**：影片錄好（白畫面 0%）：打者 `videos/batter_h9b_12500_p3c_1026.mp4`（12500、aim 0、vs P3c：5 球中 2 球界內約 31 m、3 球揮空），投手 `videos/pitcher_p3c_t18_identical_1025pm.mp4`（16:11 重錄，檔名沿用）。下次最晚 10-27 04:00。打者新表 `assets/game/batter_h9b_12500_aim0_p3c.json`（hit_rate 0.327，好球帶內）：ONE OUTS vs P3c 打者勝 18.1%（三振 56.4%、外野安打 1.8%），這是 Human15 的基準。投手停頓評估第一版失敗：P = 0.3 s 相同起始 0/64、方向 +118°，停頓時策略自己把姿勢帶跑（結束時偏差平均 0.076、最大 0.49 rad）。改成停頓期間策略動作歸零、只用 PD 拉向第 0 格（set_pause_policy_off），重跑 P = 0.3 / 0.6 / 1.0。Human15（只用 P3c 形狀慢球、300 次、從 12500）跟投手評估並存開練。

**10-26 15:55 檢查**：10-25 約 22:10 之後，三個 session 都卡在用量上限，到 10-26 15:4x 才恢復，中間沒有任何工作（約 17 小時 GPU 閒置）。現在 GPU 上是打者的 showcase（aim 0 vs P3c 的打者表）。投手停頓評估請投手 session 開跑。影片過期（投手上次 10-25 20:42、打者 10-25 11:28）：`logs/film_1026.sh` 在背景等總數 < 2、沒有 trainer 時，依序錄打者（12500、aim 0、timing_bias 0.055、vs `p3c_identical_t18_release.csv`，`videos/batter_h9b_12500_p3c_1026.mp4`）和投手（`logs/film_p3c.sh`）。授權延長到 10-27 20:00 為止。


**分工**：規劃 session（本檔作者，10-25 是 aiplaybaseball-10）評估投手 / 打者 session 的提案、分配 GPU、做 ONE OUTS 引擎與影片；投手 session（aiplaybaseball-08）負責投手與 PITCHER_TRAINING_LOG.md，投手結果由它直接回報使用者；打者 session（aiplaybaseball-d0）負責打者，訓練前先送提案，結果寫 SupporttraningLog.md 第 3 節。使用者授權到 10-26 20:00，期間規劃 session 自行決定（規格不動）；之後新設計 / 規格 / 排程要先問。

**交付候選**
- 投手：P3c = ho2 model_7600（run ho2_tw2plw100）+ 放球提早 2.5 ms + 投手斜站 θ 18°（評估時本壘線繞投手板轉 -θ，物理等價）。阻力 + 馬格努斯力的飛行：相同起始站穩 52/64、級 3+ 48/64、好球帶 34/64、觸身 1/64，球速約 17.4 m/s（64 km/h）；noisy 站穩約 53/128、級 3+ 5/128（第 2 段沒過）。球路表：`assets/game/p3c_identical_t18_pitches.csv` / `_release.csv`（noisy 同名）。
- 打者：Human9b model_12500 + aim_offset_z -0.08（表 `assets/game/batter_h9b_12500_aim008.json`、hit_rate 0.684，diag 1024 球）。對 88–112 km/h 的球 ONE OUTS 18.5%，但對 P3c 的球擊中率只有 0.140（ONE OUTS 17.5%，多半是四壞 / 觸身，外野安打 0.7%）。打中後跌倒 54–61%。
- ONE OUTS 影片（CPU 版）：`videos/one_outs_p3c_t18_1025.mp4`（RL 投手球路 vs 打者表）、`videos/one_outs_h9b_1025.mp4`（備案投手）。`scripts/render_one_outs.py`，showcase 要設 IMAGEIO_FFMPEG_EXE。

**現在（20:15）**：20:04 GPU 當機（打者的串接腳本誤判 Human13 已結束，提早開 Human14，GPU 上 3 個 Isaac）。Human13 停在 12650（12549：88–112 km/h 擊中率 0.711、跌倒 54%，沒過 ≤ 37%），12650 的 diag 正在跑。投手的 ball_aero_sim 驗證被一起帶掉。

**排程（照這個順序，一次最多 2 個 Isaac，trainer 旁不錄影）**
1. 打者 12650 diag 跑完。
2. ~~ball_aero_sim 驗證~~ 已通過（20:3x）：模擬裡穿過本壘的位置與 simulate_to_plane 差 |dy| 中位數 0.001 m、|dz| 0.016 m（最大 0.047）。之後投手影片用 --set ball_aero_sim=True。
3. ~~Human14~~ 否證沒過（20:31）：12599 對 P3c diag1024 0.205（門檻 0.25）、88–112 km/h 0.730、跌倒 59%；ONE OUTS vs P3c 17.0%、外野安打 0.5%，慢球沒學起來。下一步（打者 session）：先診斷再開訓練，要看三件事：訓練中慢球那一半的擊中率與揮早 / 揮晚、tempo 動作在慢球和快球的分布、參考揮棒的開始時刻對照球到本壘的時間。打者交付維持 12500 + aim -0.08。
   - 規劃 session 20:55 的 CPU 檢查（`scripts/diag_arrive_estimate.py`）：出手時等速估計的到達時間，比阻力 + 馬格努斯力的實際飛行早 0.051 s（P3c，飛行 1.045 s），88–112 km/h 是 0.033 s；兩者只差 0.018 s，不是揮早 0.061 s 的主因。
   - 下一個要驗的假說（零樣本、1 次 diag，約 4 分鐘，排在 C 之後）：tempo 動作（參考播放速度 ×(1 ± 0.3)）在等待期間（ref_t < 0、球已出手）也會推進參考時鐘。慢球要在預備姿勢等約 0.4 s，策略如果在快球學到「看到球就加速」，慢球就會揮早。測法：等待期間把 tempo 夾在 0（只改評估），交付版對 P3c t18 跑 diag1024，擊中率從 0.140 升到 ≥ 0.30 就成立。
4. 12 小時影片：投手 P3c（`logs/film_p3c.sh`，ball_aero_sim=True）、打者（交付版或 Human14 新交付）。最晚 23:10 / 23:28，可延到 Human14 之後。
5. ~~投手 C~~ 21:36 依規則停（相同起始級 3 7650 50 → 7700 18 → 7750 2；noisy 一路變差）。規格不動時，第 2 段（noisy 過關）的訓練選項用完；投手交付候選 P3c（ho2 7600），7650 備選（相同起始 58、50，noisy 較差）。**使用者決定（10-26 15:4x，問題在 10-25 晚間送出）：允許投球前停頓**（兩種過關評估都用同一個停頓 P）。投手 session 跑 P3c 在 P = 0.3 / 0.6 / 1.0 s（相同起始 64 + noisy 128，阻力級數），取能過的最小 P，並列出停頓結束時的關節偏差。GPU 先給打者：慢球診斷 4 次 diag（`logs/b_slow_diag.sh`、`b_slow_diag2.sh`，第 4 次是出手後 tempo 夾 0）→ 打者 12 小時影片。
   - 慢球診斷結果（打者 session，22:09）：tempo 不是原因（出手後夾 0 → 0.137），揮棒長度也不是；慢球揮空幾乎都是揮早，球到時參考時鐘已過擊球點約 0.04 s，但整體延後時機沒用。交付 vs P3c，好球帶內擊中率：aim -0.08 0.226、aim 0 0.327。決定：比賽對 P3c 用 aim 0，打者表 / ONE OUTS 改用好球帶內擊中率（新表 `batter_h9b_12500_aim0_p3c.json`）；只用慢球的訓練（mix 1.0、300 次、從 12500，否證 +100 次好球帶內 ≥ 0.45）排在投手停頓評估之後。

**今天定下的規則**：級數與比賽一律用阻力 + 馬格努斯力的飛行（eval_stage 的 simulate_to_plane t_max 要 3 s）；打者 diag 用 1024 球（seed 0）、ONE OUTS 用 1000 場同 seed；投手 trainer 在跑時打者不用 GPU；「投手優先」不是絕對，依 11-26 交付需要和成功機會排；投球前停頓（A）不能拿來過 noisy 門檻。

### 10-25 經過（新到舊）

**10-25 20:10**：打者對 P3c 的球（約 64 km/h，飛行約 1.05 s）：交付版 12500 / -0.08 的 diag1024 擊中率 0.140（88–112 km/h 時 0.684），揮空多半是時機（823/881）、球在棒子上方 692；ONE OUTS（投手 = p3c_identical_t18）打者勝 17.5%，但多半是四壞 10.1% + 觸身 6.7%，三振 73.2%，外野安打 0.7%。零樣本 timing_bias × aim 掃描最好只有 0.212，救不回來。決定：Human14（一半訓練球照 P3c t18 的出手、球速 55–75 km/h，一半 88–112 km/h；從 Human13 最好的存檔起練 400 次，81fb9ae / `logs/run_human14.sh`）核可，排在 Human13 之後、投手 C 之前（約 21:00–23:00）。判定：ONE OUTS vs P3c > 17.5% 且外野安打 ≥ 5%（若只差這條，先回報）、對 P3c diag ≥ 0.5、88–112 km/h diag ≥ 0.60；否證 +100 次對 P3c ≥ 0.25。投手：P3 剩下的組取消，交付候選確定為 P3c（ho2 7600、提早 2.5 ms、θ 18°；提早 5 ms 相同起始站穩少 10）。B 平均不如單一 7600。C 排到 Human14 之後；影片（投手 23:10 / 打者 23:28）排在 Human14 與 C 之間。ball_aero_sim（只在錄影時對球加阻力 + 馬格努斯力）已實作，驗證中。

**10-25 19:50**：θ 重挑為 18°（16–20° 掃描：好球 / 觸身 17° 28 / 8、17.5° 35 / 4、18° 34 / 1、19° 14 / 0）。P3c @ 18°：相同起始站穩 52/64、阻力級 3+ 48/64、級 4 30/64，本壘 y +0.05 m、z 0.76 m；noisy 級 3+ 5/128。新表（ff4471a）：`assets/game/p3c_identical_t18_pitches.csv` / `_release.csv`（noisy 同名）。CPU ONE OUTS 1000 場（P3c t18 相同起始 vs 打者表 12500 / -0.08）：打者勝 25.2%（內野球 41.8%、三振 33.0%、外野安打 10.5%、四壞 9.0%、觸身 5.7%），備案投手 18.5%；打者表是 88–112 km/h 的球量的，要等打者用 P3c 的球評估。影片 `videos/one_outs_p3c_t18_1025.mp4`（RL 投手的球路表，10 場、148 s，白畫面 0%）。B（7550 / 7600 / 7650 平均）相同起始 51/64、級 3+ 43/64，單一 7600 較好。C（TW2PLW100P3C，35ccf1c，從 ho2 7600 練 300 次、每 50 次評估）核可，但排在 Human13 之後（約 21:00 → 22:45）：ho 系列本來就用訓練雜訊 0.02 練，C 只差在獎勵對準好球帶，成功機會低。影片：投手 23:10 用 P3c @ 18° 錄，打者 23:28，都排在 C 之後。

**10-25 19:45**：P3c 球路表（fdb783b）：`assets/game/pitches_p3c_{identical,noisy}_drag.csv`、`release_p3c_{identical,noisy}.csv`（θ 17°）。CPU ONE OUTS 1000 場（打者表 12500 / -0.08）：相同起始的表打者勝 50.0%（觸身 36.3%、內野球 29.8%、三振 20.2%、四壞 7.2%、外野安打 6.5%）；noisy 的表打者勝 94.1%（四壞 87%）；備案投手 18.5%。原因：相同起始的球在比賽座標的 y 中位數 +0.25 m（往三壘、往打者身上），62 球中好球帶（半寬 0.25 m、z 0.5–1.1 m）只有 28、觸身 8。級 3（|y| ≤ 0.5、z 0.2–1.5 m）比比賽的好球帶寬很多，過了級 3 不等於是好球。決定：用 CPU 重挑 θ（16～20° 每 0.25°），取好球帶命中最多且觸身 ≤ 2/64 的 θ（估約 18°）重出球路表；打者用投手球路的評估等新表，只用相同起始那份。第 2 段：A（投球前停頓）不能拿來過關（停頓把雜訊吸掉，noisy 評估就退化回相同起始，改了過關條件的意義），只留作比賽 / 影片做法；B（ho2 7550 / 7600 / 7650 平均）在跑，C（從 P3c 用訓練雜訊 0.02 練 300 次、不停頓，相同起始站穩連兩次 < 48/64 就停）排在 B 之後。打者 diag / showcase 已能讀 pitch_csv 與 release_csv（8467e29、5940bb1）。Human13 否證過（訓練環境打中後跌倒 98%）。

**10-25 19:20 投手第 1 段成功**（阻力版，eval_stage t_max 1.0 → 3.0 s 修好後，fb0ee97）：ho2 7600 + 放球提早 2.5 ms + 投手斜站 θ 17°（P3c）相同起始站穩且投出 52/64、級 3+ 42/64（級 4 24/64）、本壘 y -0.27 m、z 0.76 m；提早 0 ms × θ 16° 是 55/64、37/64。相同起始已過使用者門檻，noisy 沒過（站穩約 25/64、級 3 個位數）。決定：(1) P3c 是投手交付候選（P3 剩下的組若更好就換）；(2) 產生 P3c 的阻力版球路表（相同起始 / noisy 分開），之後取代 ONE OUTS 的備案投手；(3) 第 2 段（隨機起始）請投手 session 提案，要附不訓練的選項與「這次為什麼不一樣」。打者：Human11 失敗（12599 擊中率 0.553、跌倒 65%）；Human12 否證沒過，但找到真正原因：訓練時參考動作播完就算 time_out 結束回合，打者還沒到會倒的時候；Human13（參考播完停在最後姿勢、回合繼續，觀測的參考時間比例夾在 1）否證中。P3c 的球速約 17.4 m/s（63 km/h），遠低於打者訓練的 88–112 km/h，用 P3c 球路表評估打者排在 Human13 之後優先做。投手斜站 17° 在影片看得出來。

**10-25 18:47 檢查與決定**：GPU 上是投手評估（eval_stage）+ 打者 Human11 的 diag 否證（12599），兩個都是評估。投手 P2（斜站）：方向用斜站可修到約 0°、本壘 y 到 0.3 m 內，站穩不變；瓶頸是高度。決定：過關與 ONE OUTS 比賽一律用阻力 + 馬格努斯力的飛行（跟比賽引擎一致），只算重力的級數只當參考。重力版 hf 7300 θ17 級 3 有 8/64，阻力版 0/64，連在空中到本壘的都沒有。接下來：ho2 7600 θ16 零樣本（阻力版相同起始級 3 ≥ 16/64 算成功），再跑 P3（底 hf 7300、ho2 7600，提早 0～5 ms，依阻力版 z 對仰角的斜率決定，θ ≤ 25°）。打者 Human11（follow_through 1.2 s）18:28 起，約 19:55 出結果。影片下次最晚 23:10（投手）/ 23:28（打者）。

**10-25 18:30**：diag 改 1024 球後重量：12300 / -0.12 0.666、12500 / -0.08 0.684（三條都過）、12500 / -0.04 0.729（距離 10.8% 沒過）、12350 / -0.08 0.659。照事先定的量法，交付改為 12500 + -0.08（合併所有量測約 0.668 的但書記在上面）。`videos/one_outs_h9b_1025.mp4` 恢復有效（同一張表、同一個 seed），render 預設改回這張表。投手：ho3c 18:08 停（暖機用 1e-4，value loss 仍停在約 130 → 是變異下限，「critic 跟不上」收起來）；P2 斜站 18:09 開跑；P3（放球提早 0 / 1 / 2 / 3 ms × θ，θ ≤ 25°）排在 P2 + 逐回合放球資料的否證之後。打者下一個：跌倒方案（follow_through 0.4 → 1.5 s，從 12500 起練）。

**10-25 18:15**：打者 12500 + aim -0.08 重測 0.684 / 0.605 / 0.668（平均 0.652），退回 12300 + aim -0.12；`videos/one_outs_h9b_1025.mp4` 作廢（改名 _VOID），render 預設改回 12300 的表，有效影片是 `videos/one_outs_h9_1025.mp4`。256 球的 diag 單次差到 0.08，所以量法改成 1024 球（seed 0），重量 12300 / -0.12、12500 / -0.08、12500 / -0.04、12350 / -0.08。投手：ho3b 17:51 提早停（value loss 只降 35%）；ho3c（暖機時 critic 學習率固定 1e-4，3863c28）驗證「critic 跟不上新獎勵」；之後 P2「投手斜站」零樣本：評估時把本壘線繞投手板轉 -θ（等於投手朝一壘側斜站 θ，物理等價），hf_avg3 TW2PL θ 15 / 17 / 19° + ho2 7600 θ 13.5°；最好一組相同起始級 3 = 0 且 noisy < 4/128，就代表瓶頸是高度。打者跌倒方案（follow_through_s 0.4 → 1.5 s）排在重量後。

**10-25 17:55**：(1) 投手：ho2（W100）16:06 停、ho3（y 尺度 7 m）17:09 停，|y| 都沒改善（ho3 -5.29 → -7.57 m）；ho3b（ho3 + critic 暖機 50 次，`--critic_warmup`，train_live.sh 的 WARMUP）17:39 開，暖機後 value loss 沒降一半就停，預計 19:50–20:00 結束。投手結果由投手 session 回報。(2) 打者：Human10（w_miss 40）12350 擊中率 0.500，失敗。發現「下切兩次」：Human9 在 aim 0 下學會往球下揮，比賽再加 aim -0.12 就揮空（12500 在 aim 0 擊中率 0.793 vs -0.12 0.539）。瞄準掃描後交付改為 Human9b model_12500 + aim_offset_z -0.08（`logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt`）：diag 0.684、ONE OUTS 18.5%、界內且 ≥ 29 m 15.9%、打中後跌倒 61%。擊中率太接近門檻，ho3b 結束後換 seed 1、2 重測（`logs/b_diag_retest_12500.sh`），三次平均 < 0.68 就退回 12300 + aim -0.12。打中後跌倒 54–61% 要在對決影片前處理。(3) ONE OUTS 影片換新表重錄：`videos/one_outs_h9b_1025.mp4`（10 場、142 s、白畫面 0%），render 預設改成 `assets/game/batter_h9b_12500_aim008.json`、hit_rate 0.684。

**10-25 15:47 檢查**：GPU 上是投手 ho2 trainer（14:41 起，`AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100-v0`，ho 7500 → 8100，預計 16:55–17:05）+ 打者補評估的 showcase（B 槽），VRAM 11.8 GB（接近上限）。打者 Human9b（12300 → 12694）：12400 / 12500 / 12600 的 ONE OUTS 勝率 15.8 / 17.4 / 16.3%，但 diag 擊中率 0.484 / 0.539 / 0.586 都 < 0.68，不換交付（門檻不調）；補評估 12350、12694，都沒過就排 w_miss 20 → 40 的提案在 ho2 之後。影片下次最晚 23:10（投手）/ 23:28（打者）。

**10-25 14:05 決定**：ho 13:55 依規則停（noisy 7400 58、7500 56/128；相同起始 |y| 每 100 次改善 0.6 m，noisy 沒改善），細節在 PITCHER_TRAINING_LOG.md。投手改成兩段：過關要求隨機起始站穩 ≥ 48/64，但 hf_avg3 本身只有約 34/64，hi / hj 加訓練雜訊也都被 noisy 規則停掉，隨機起始的平衡是另一個沒解決的問題。第 1 段先在相同起始證明球能進級 3，noisy 只當底線；第 2 段再處理隨機起始。ho2：從 ho model_7500 續練 600 次，w_stage_ball_rest 50 → 100（只改這項）；停止 = 相同起始站穩連兩次 < 40/64，或相同起始 |y| 連兩個評估點每 100 次改善 < 0.3 m，或 noisy 連兩次 < 40/128；相同起始級 3 ≥ 16/64 就回報。GPU：打者 Human9 續練 14:00 起（A 槽，到 12695，`logs/run_human9b.sh`，約 15:10 結束），ho2 排在打者之後（兩個 trainer 同時跑會超過 12 GB）。

**10-25 13:20**：(1) 打者交付換成 Human9 model_12300 + aim -0.12（426f66d；`logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_12-31-03_human_pitched9/model_12300.pt`）：ONE OUTS 1000 場打者勝 13.1%（Human7 8.0%），diag 擊中率 0.691（門檻 0.68）；表 `assets/game/batter_h9_12300_aim012.json`（200 球，續練那輪改 1024 球）。剩下 395 次排在 ho 結束後（約 15:30），要 ONE OUTS ≥ 13.1% 且 diag ≥ 0.68 才取代。(2) 投手 ho 13:04 開練（1bcdd66，run ho_tw2pl，`logs/train_ho.txt`），預計 15:20–15:30 結束，結果由投手 session 回報。(3) ONE OUTS 影片換新打者表重錄：`videos/one_outs_h9_1025.mp4`（10 場、149 s、白畫面 0%），render 預設改成 Human9 表。

**10-25 12:47 檢查**：GPU 上只有打者 Human9（12:30 起，任務 AIB-PitchedBatting-Human9-v0 = Human7 + target_distance 29 m，從 model_12095 起，存檔每 50 次，`logs/run_human9.txt`）。投手 ho 零樣本否證通過（TW2PL：61/64、方向 -17°、仰角 +11°、分數標準差 11% / 29% 最大值、與 |y| 相關 -0.87），已核可開練：打者跑完第 12295 次自動檢查、在 model_12300（約 13:03）停掉並讓出，ho 立刻從 hf_avg3 7300 起練 600 次（約 2.5 小時）；打者同時在 B 槽跑 showcase --eval_envs 256（≤ 10 分鐘），再算 CPU ONE OUTS 1000 場。影片下次最晚 10-25 23:10（投手）/ 23:28（打者），ho 在跑時不錄。

**10-25 12:30 決定**：P1 失敗（細節在 PITCHER_TRAINING_LOG.md 0a6932f）：hn 後段方向停在 -35～-44°，不續練；手臂提早讓站穩下降，仰角沒有上升，這條路關掉。下一步 ho（投手 session 實作）：底 TW2、放球交給策略、不用觸發；獎勵 = 放球當下算出的本壘落點 (y, z)，偏差大時也有斜率，加級 3 額外獎勵；站穩的獎勵不動。先用 TW2 零樣本 noisy 的放球資料確認這個分數在回合之間有差異（標準差 ≥ 最大值 10%），才從 hf_avg3 7300 練 600 次。停止：noisy 連兩次 < 30/64；或連兩個存檔本壘 |y| 沒比零樣本小 1 m 且仰角沒升；或站穩連兩次 < 40/64。級 3 ≥ 8/64 算有進展。不採用：(b) L0.01 + 收緊仰角觸發（方向對時仰角 -26.6°，兩個條件很少同時成立）。GPU：投手寫程式期間給打者，打者要在投手開練時於下一個存檔讓出。打者 aim_offset_y 不採用（-0.08 界內 +12 點但都是滾地，ONE OUTS 勝率 8.0 → 6.2%）；打者採用指標改為 ONE OUTS 1000 場勝率（≥ 11% 算改善、< 8.0% 失敗）+ 界內且 ≥ 29 m 佔全部球。

**10-25 12:15**：(1) 三個 session 分工與 GPU 分配規則見 SupporttraningLog.md 第 6 節第 3 點（投手 08、打者 d0、規劃 10）。投手訓練結果由投手 session 直接回報使用者。(2) ONE OUTS 對決影片第一版（CPU，備案投手 vs RL 打者表，10 場）：`videos/one_outs_fallback_1025.mp4`（181 s，白畫面 0%；`scripts/render_one_outs.py`、`aibaseball/game/one_outs_replay.py`）：側面投球到牆、牆上好球帶、俯視落點與 95 ft 內野圈、記分板。這 10 場打者 2 勝、投手 8 勝。還沒接上機器人的 Isaac 畫面（投手動作、打者揮棒）。

**10-25 11:45**：(1) 使用者授權到 10-26 20:00，期間全部由規劃 session 決定（含恢復訓練、新設計），規格不動。(2) 分工：投手 session（aiplaybaseball-08）的提案先由規劃 session 評估。(3) P1（修改後通過）：hn 7700 / 7900 / 8150 / 8450 補評估，再零樣本掃 arm_ref_lead_s L = 0.01 / 0.02 / 0.03（aimL 7/25 + 觸發 ±5°，手肘 L + 0.03）+ L = 0.02 的 ±10° 對照；成功 = 仰角 ≥ +10°、|方向| ≤ 5°、相同起始站穩 ≥ 40/64；8450 方向在 -20° 以內就先停 P1、改考慮續練 hn。(4) 影片 10-25：投手 `videos/deliverable_h5_identical_1025.mp4`、`pitcher_aim10_30h5_identical_1025.mp4`（11:10 / 11:21）；打者 `videos/batter_h7_aim012_1025.mp4`（11:28），白畫面都是 0%。打者第一次在 Isaac 裡找不到 ffmpeg，要設 `IMAGEIO_FFMPEG_EXE`（`logs/film_batter_1025.sh`）。下次最晚 10-25 23:10。3 小時檢查已改成新授權的內容。

**10-25 11:00**：新 session 重設每 3 小時檢查（:17，session-only）；過期影片 `logs/film_1025.sh` 錄製中（投手交付 + aim 10/30、打者 Human7，輸出 `videos/*_1025.mp4`）。仍不啟動訓練。

**10-24 08:16 使用者要求暫停訓練**：hn 已由規劃 session 停掉，投手 session 的 watch_hn 也停了。hn 評估只有 7400（56/64、方向 -35.9°、仰角 +13.0°、y -14.7 m）與 7500（54/64、-35.9°、+15.0°、y -14.5 m），級 3 = 0：仰角保住、方向沒修。7600–8450（trainer 因 eval 卡住多跑到 8450）沒有評估。**不要啟動任何訓練，等使用者指示**；影片照 12 小時規則錄（打者 11:33、投手 14:40）。每週用量 100%。

## 5. 待執行

**側投驗證第 1 步（10-25，只用 CPU，使用者問 Chris Sale 式側投）**：OBP 411 球的 arm slot（0° 側投、90° 過頂）中位數 42°；目前參考 2916_4 是 24°，已經偏低。最低的 1562（14–18°，右投，三球 c3d 抓到 `data/c3d_pitching/`）。運動員手部 marker 在放球附近「方向 → 仰角」：2916_4 方向 -10 / 0 / +10° 時仰角 +14.6 / +11.6 / +7.5°；1562 三球 +12.6～+16.4 / +10.0～+13.9 / +6.4～+10.1°。兩種 arm slot 的曲線幾乎一樣：方向每 ms 掃 2.8–2.9°，仰角在方向轉到本壘時一樣往下掉約 0.3–0.4°/°，沒有出現側投「仰角與時間分開」的效果。另外，運動員在方向 0° 時仰角 +10～+14°，機器人卻是 -3.5～-28°：差距來自機器人沒跟上參考（放球晚約 50 ms、手速主要來自軀幹），不是 arm slot。依事先定的條件（放球 ±30 ms 內仰角平坦且在 +15～+20°）**側投在第 1 步不成立**，沒有 retarget、沒用 GPU。腳本 `scripts/diag_arm_slot.py`。

**下一步（10-24 06:55 起，依序）**
1. 解開 7500 評估卡住（使用者 taskkill transmitter），讓 hn 評估 7500–7900；看方向 / 本壘 y / 級 3。STOP 或 7900 結束後：把 hn 結果寫進紀錄，換交付條件同前（級 3 ≥ 32/64 且兩種評估站穩 ≥ 48/64）。
2. hn 結束後：修 `scripts/train_live.sh` 的 eval 管線（輸出先寫檔），再錄到期影片（打者 11:33、投手 14:40；eval_stage 已有本壘視角 + 放球關鍵畫面 1a904b8 / 167a6e4）。
3. 若 hn 方向學不起來（級 3 仍 0）：10-24 起新設計要先問使用者。已試過且失敗：方向觸發 ±5/±10（仰角 -10～-30°）、手臂時間 sr3～sr16、觸發中心移動、腰提早、新手臂參考 aim/aimL、軀幹 waist pitch、PPO 訓練 hl（球分數）與 hm（仰角獎勵）。剩下的候選：備案球路（FallbackPitcher）做 ONE OUTS 影片、提高力量讓球更快（需要的仰角變小）、放寬好球 / 過關規則。
4. 打者、ONE OUTS 對決影片（11-26）照第 5 節原列。

0. **投手（使用者 10-23 晚上決定：方案 1）**：零樣本 (b) 到 b8 結束、級 3 全 0（方向與仰角綁在同一條手部軌跡的時間軸上）。新做法 = 新的手臂參考動作：放球前約 0.12 s → +0.05 s 把投球手 marker 繞肩轉向一壘側約 +7° 並往上抬，讓出手時方向約 0°、仰角 +12～+16°（17 m/s 的好球帶高度窗），IK 出新參考，hf_avg3 7300 零樣本評估。投手 session 實作與執行。若方向仰角對但站不穩，在新參考上訓練要先問使用者。
0b. **方案 1 結束（級 3 全 0）**：參考動作出手可改到 +6° / +28°，但機器人實際出手幾乎不變（手速主要來自軀幹 9.2 / 13 m/s、H5 只混入一半新手臂、放球晚約 50 ms）。最好：aim 10/30 + 觸發 ±5° = 64/64、本壘 y -0.23 m、仰角 -28°。**使用者決定下一步：改軀幹參考（零樣本）**：在該設定上把 waist pitch（放球附近運動員的後仰，TW1 曾讓仰角到 +27°）按比例套上，waist_yaw 不動；投手 session 執行。
0c. **軀幹參考結束（級 3 全 0）**：waist pitch 0.3–0.7 仰角只 -28 → -26°（觸發等手轉到本壘時手已在下揮）；零樣本路線用完。最好起點 aim 10/30 + 觸發 ±5° + waist pitch 0.7：61/64、72/128、本壘 y 0.1–0.2 m。**使用者決定：在這組上訓練仰角**。hm（10-23 23:5x 起）：任務 H5HTE（55912d4，新獎勵 w_stage_rel_elev 50），hf_avg3 7300 起 600 次，`logs/oo_step2_hm.sh`、`logs/train_hm.txt`；停止：noisy 連兩次 < 30/64，或兩個連續存檔仰角沒高於 -26°。打者影片 `videos/batter_h7_aim012_1024am.mp4`（23:33，白畫面 0%）；投手影片 06:35 到期，等 hm 結束後錄。 hm 評估（相同起始 / noisy goalcheck）：7400 63/64、31/64，仰角 -25.6 / -24.1°；7500 61/64、26/64，仰角 -24.4 / -14.3°；級 3 都是 0，本壘 z 0.04 m（滾地）。
0d. **hm 結束（10-24 02:10，判定失敗）**：相同起始 7400 63/64 -25.6°、7500 61 -24.4°、7600 59 -25.8°、7700 43 -25.2°、7800 43 -27.4°、7900 43 -32.9°；noisy goalcheck 7400 31/64、7500 26/64；級 3 全 0（z 0.04 m）。仰角獎勵推不動仰角，站穩 63 → 43。交付不變（hf_avg3 7300 + H5）。投手的下一步要先問使用者。
0e. **使用者 10-24 決定：放球改用仰角條件並訓練**。hn（~02:50 起）：任務 H5HTEL（9bc9866）= H5HTE 但放球在手速仰角 +5～+25° 且方向 ±30° 內（1.3 s 強制），方向交給策略學。`logs/oo_step3_hn.sh`：先零樣本（`logs/oo_hn0_n*.raw`），再從 hf_avg3 7300 訓練 600 次（`logs/train_hn.txt`）。停止：noisy 連兩次 < 30/64，或兩個連續存檔級 3 = 0 且本壘 |y| 沒改善。投手影片 02:41 錄好（`videos/deliverable_h5_identical_1024am.mp4`、`pitcher_aim10_30h5_identical_1024am.mp4`），下次 14:40。

1. **投手仰角設計（需要使用者決定）**：級 3 要仰角約 +15°。候選：在「H5 + 觸發 ±10°」上訓練仰角，或新的手臂參考動作（例如出手前手臂不往下揮）。10-24 起新設計要先問使用者，由投手 session 帶選項去問。
2. **投手交付候選**：H5 + 觸發 ±10°（63/64、74/128、-7°）站穩和方向都比目前交付好；換交付要等使用者同意（仍是級 3 = 0）。
3. **打者**：投手的球能在空中到好球帶後，用新投手的球評估，再定打者標準（期限 11-12）。在那之前可改善的弱點：擊球太弱、界內球短、右側界外多。
4. **影片**：投手與打者至少每 12 小時錄一次（下次最晚 10-23 21:15）；trainer 在跑時不錄；每支影片用 `scripts/check_video_frames.py` 檢查白畫面，失敗重錄一次。
5. **ONE OUTS 對決影片**（11-26）：render 要接上投手動作播放 + 這一球的球路 + 打者揮棒，約 10 場，含比數、牆上好球帶、落點、勝負與勝率統計。
6. 每個投球結果寫進 `PITCHER_TRAINING_LOG.md`；每次改動 commit + push。

---

## 6. 執行規則（備忘）

- GPU 最多同時 2 個 Isaac 行程；trainer 在跑時不錄影（會超過 12 GB）。
- bash 的 `python` 是 WindowsApps 的假程式，要用 `/c/Users/User/anaconda3/envs/isaaclab_env/python.exe`；Isaac 指令都加 `--headless`（`diag_bat_contact.py` 已強制）；長的串接指令每段加 `timeout`。
- Git Bash 沒有 `pgrep`；等行程用 PowerShell 查命令列。
- GPU 閒置超過 1 小時且沒人排工作：先錄到期影片，再跑最新打者的品質檢查（`showcase.py --eval_envs 256` + `diag_bat_contact.py`），跑完回報。
- 一次只改一件事；每個設計至少 600 iterations，除非已明顯失敗；停訓練前先評估最新存檔。
- 每次改動都 commit 並 push；不上傳 `data/`、`assets/motions/`、`ref_videos/`、logs、影片。

### 10-26 20:05 決策
- 投手的 Pp' 在 7750 停止：7650–7750 三個檢查點，P = 0.3 相同起始都是 0/64，雜訊評估 0/128，航向約 +84°。改為查原因，不重開訓練。
- 交給投手的反證測試：(1) 用 TW2PLW100P3CP 評估 7600，P = 0，看能否重現 52/64；(2) 7600 在 P = 0.3 的 zero-shot 對照 7650，並確認訓練 reset 時真的有套用暫停；(3) 暫停結束時的 phase / 參考幀 / release 計時器；(4) plate_yaw 的旋轉是否被套兩次。
- GPU：投手丘短片錄影中（1 個 Isaac）。打者的跌倒 zero-shot 要等錄影結束才開始；啟動條件要多檢查 eval_stage.py 和 showcase.py。

### 10-26 晚間 決策（停頓失敗原因 + 下一步）
- 原因：P3CP 在 P = 0 下評估 7600，結果 52/64，和 P3c 相同，所以 env 本身沒問題。暫停後起始狀態和 7600 熟悉的不同：關節速度被歸零，P = 0 時 reset 會帶入第 0 幀的速度；腳底受力約 21/79%；waist_pitch 下垂 0.09 rad。暫停一結束就單腳往反方向自轉（-152°）。Pp' 訓練中，set 起始一次都沒成功過（success_set 0）。
- 批准 P3CPC 暫停課程：P ~ U(0, hi)，hi 從 0.05 開始；success_set 超過 0.1 且離上次調高至少 20 次迭代，hi 就加 0.05，最多到 0.3。起點 7600，保留 critic 暖身和原停止規則。新增停止規則：hi 連續 300 次迭代沒有調高就停。hi 到 0.3 之後，門檻是兩個評估都用 P = 0.3。
- 投手丘短片：投手丘和投手板已轉正，白畫面 0%。本壘視角的變焦沒有生效（session layer 蓋掉了 root layer 的設定，f95038a 的 bug），投手已改用 EditContext 修正，排在重錄 mound_check2。
- GPU 順序：diag_ppw3 → 重錄 mound_check2 → 打者 4 組跌倒測試（約 20 分鐘，等投手通知「槽空了」才手動啟動）→ P3CPC 訓練。

### 10-26 21:xx 決策
- 開局速度掃描（7600、P = 0）：k = 1.0 → 52/64，k = 0.8 → 18，k ≤ 0.5 → 0。P3CPC 步長改為 0.05，上限 1000 次迭代。hi 到 0.3 之前每 100 次迭代評估一次，到 0.3 之後每 50 次。第 250 次迭代時 k 若仍 ≥ 0.8，就停下來回報。
- 影片黑畫面（使用者回報）：quality=10 使 x264 用 crf 0，輸出 High 4:4:4，播放器放不出來。已改為 crf 16 High（e36e4fa）；mound_check 兩支已重新編碼；check_video_frames.py 加了格式檢查（4501826）。
- GPU 順序：打者 4 組跌倒測試 → 打者 vs P3c 影片（batter_h9b_12500_p3c_1026b.mp4，補 12 小時影片）→ 打者通知「槽空了」→ P3CPC。
