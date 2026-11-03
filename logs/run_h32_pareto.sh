#!/bin/bash
# run_h32_pareto.sh (14, 11-03 19:1x): form-vs-tables Pareto survey of the H32 line (1.0x, no saim), evaluation only.
# 1. Form diag rows (11-02 protocol: jitter 0, 1024 envs, --form_log; logs/h32_form.py row) per ckpt. 14496 / 14995 / 15494 / 15993
#    reuse the 11-02 jsons (diag_bat_contact_form_<I>, identical protocol); new: 14700, 15200, 15700. 12500 kept as the start row.
#    H31 (12500 -> 14496) was trained at 1.5x + saim (history) and is not on the 1.0x line, so it is not surveyed.
#    -> logs/h33/h32_form_sweep.txt, marker H32_FORM_SWEEP_END.
# 2. Game-scene tables (both columns, run_h33_tables.sh) on the up to 3 lowest-pose ckpts with hit >= 0.85 and fair carry median
#    >= 24 m (14: 15494 at 24.8 gets tabled), lowest pose first; stop at the first that passes 16.93 / 16.76 with OF >= 3 (= the Pareto candidate).
#    15993's tables are the 11-02 ones (23.25 / 27.69, OF 15.6 / 20.2), not re-run.
# 3. Joint s6 (R6b, 64 envs) of the candidate -> logs/h33/joint_s6_<I>_h32.csv, marker H32_PARETO_JOINT_END. idle() + nofilm throughout.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE AIB_TIMING_NOISE AIB_TEMPO_SCRIPT
B=logs/rsl_rl/aib1_tee_batting_mimic; O=logs/h33
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
S=$O/h32_form_sweep.txt; M=logs/H32_FORM_SWEEP_END.txt; MJ=logs/H32_PARETO_JOINT_END.txt; rm -f $M $MJ; MSG="aborted"; ST=1
trap '[ $ST -eq 1 ] && echo "H32_FORM_SWEEP_END $(date +%H:%M) $MSG" > $M; echo "H32_PARETO_JOINT_END $(date +%H:%M) $MSG" > $MJ; cat $MJ' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
fv() { echo "$2" | sed -n "s/.*| $1 \([-+0-9.na]*\)[ \/].*/\1/p"; }
hitf() { echo "$1" | sed -n 's/.*| n [0-9]* hit \([0-9.]*\) .*/\1/p'; }
declare -A DIR=([12500]=2026-10-25_13-56-51_human_pitched9b [14496]=2026-11-01_11-42-57_human_pitched_h31_4 [14700]=2026-11-01_20-56-20_human_pitched_h32_1 \
  [14995]=2026-11-01_20-56-20_human_pitched_h32_1 [15200]=2026-11-02_00-24-42_human_pitched_h32_2 [15494]=2026-11-02_00-24-42_human_pitched_h32_2 \
  [15700]=2026-11-02_04-02-58_human_pitched_h32_3 [15993]=2026-11-02_04-02-58_human_pitched_h32_3)
LIST="12500 14496 14700 14995 15200 15494 15700 15993"
echo "$(date +%H:%M) H32 Pareto survey (jitter 0, 1024 envs, 11-02 form protocol; 12500 = start, 14496 = H32 start = last H31 1.5x ckpt)" > $S
echo "ckpt | hit | exit | carry | launch | pose_err mean/q90 | key_err mean/q90 | after contact | split" >> $S
for I in $LIST; do
  T=form_$I
  if [ ! -f logs/diag_bat_contact_$T.json ]; then
    MSG="form diag $I"; idle
    timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $B/${DIR[$I]}/model_$I.pt --percept_filter --envs 1024 --seed 0 --tag $T \
      --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
      --physics_dt 0.00125 --contact_log --form_log > $O/diag_$T.txt 2>&1
    if grep -q Traceback $O/diag_$T.txt || [ ! -f logs/diag_bat_contact_$T.json ]; then MSG="form diag $I crash or no json"; exit 1; fi
  fi
  $P logs/h32_form.py logs/diag_bat_contact_$T.json $I logs/h32/form_$I.csv 2>&1 | tail -1 >> $S
done
CAND=$(tail -n +3 $S | while read -r L; do I=${L%% *}; [ "$I" = 12500 ] && continue; h=$(hitf "$L"); c=$(fv carry "$L"); p=$(fv pose "$L")
  awk -v h="$h" -v c="$c" 'BEGIN{exit !(h ~ /^[0-9.]+$/ && c ~ /^[0-9.]+$/ && h + 0 >= 0.85 && c + 0 >= 24)}' && echo "$p $I"; done | sort -n | head -3 | awk '{print $2}' | tr '\n' ' ')
echo "$(date +%H:%M) table candidates (hit >= 0.85, fair carry >= 24, lowest pose first): ${CAND:-none}" >> $S
ST=0; echo "H32_FORM_SWEEP_END $(date +%H:%M) $(tail -n +3 $S | tr '\n' '#' | cut -c1-2500)" > $M
PAR=""; TR=""
for I in $CAND; do
  if [ $I = 15993 ]; then TM="11-02 tables model_15993 | jitter 0.03: 23.25% / OF 15.6% | jitter 0: 27.69% / OF 20.2%"
  else MSG="tables $I"; idle; bash logs/run_h33_tables.sh $B/${DIR[$I]}/model_$I.pt $I form_$I > logs/h33_tables_$I.out 2>&1; TM=$(cut -c1-300 logs/H33_TABLES_${I}_END.txt); fi
  TR="$TR [$I] $TM"; echo "$(date +%H:%M) $I $TM" >> $S
  Wa=$(echo "$TM" | sed -n 's/.*jitter 0.03: \([0-9.]*\)% \/ OF \([0-9.]*\)%.*/\1/p'); Oa=$(echo "$TM" | sed -n 's/.*jitter 0.03: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
  Wb=$(echo "$TM" | sed -n 's/.*jitter 0: \([0-9.]*\)% .*/\1/p'); Ob=$(echo "$TM" | sed -n 's/.*jitter 0: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
  if awk -v a="$Wa" -v b="$Wb" -v oa="$Oa" -v ob="$Ob" 'BEGIN{n = "^[0-9.]+$"; exit !(a ~ n && b ~ n && oa ~ n && ob ~ n && a + 0 >= 16.93 && b + 0 >= 16.76 && oa + 0 >= 3 && ob + 0 >= 3)}'; then PAR=$I; break; fi
done
[ -n "$PAR" ] || { MSG="no Pareto candidate (candidates: ${CAND:-none};$TR)"; exit 0; }
MSG="joint s6 $PAR"; idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $B/${DIR[$PAR]}/model_$PAR.pt --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
  --csv $O/joint_s6_${PAR}_h32.csv > $O/joint_s6_${PAR}_h32.txt 2>&1
if [ -f $O/joint_s6_${PAR}_h32.csv ] && ! grep -q Traceback $O/joint_s6_${PAR}_h32.txt; then JS=$($P logs/film_pick.py $O/joint_s6_${PAR}_h32.csv $O/film_pick_${PAR}_h32.txt | head -1); else JS=crashed; fi
MSG="Pareto candidate model_$PAR (${DIR[$PAR]}) | $(grep "^$PAR |" $S | cut -c1-200) |$TR | joint s6 (R6b): $JS | csv $O/joint_s6_${PAR}_h32.csv"
