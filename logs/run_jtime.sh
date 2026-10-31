#!/bin/bash
# JTIME (planning 14, 10-31 18:56) re-scoped by 15's CPU audit: the joint schedule is right (live t_plane = timing.csv
# t_plane, 0.0 ms per clip; ref clock at the plane delta_s +0.040 on every clip vs native ref_err 0.047), but the joint
# batter swings slow and over the ball (ss_max ~16 vs native 22, dz -0.06..-0.08 vs +0.02). The joint runs 800 Hz physics,
# the native diag / training 400 Hz (10-28: the ss reading and the hit rate move with dt). Falsifier, after RECOV4_END
# (newer than this chain's start): native diag of model_13298 + row 9 saim, phase 2.6, jitter 0, P3c releases, real heights,
#  N8_<clip> = each joint clip alone at 800 Hz (256 envs)  -> per-clip native-800 reference for the joint rows;
#  N4      = the 4 joint clips together at 400 Hz (1024)    -> the same set at the training dt.
#  14 (19:22) additions: plain 12500 (no saim) on the 4 clips at 800 Hz and at 400 Hz (is 13298 more dt-sensitive than
#  12500?); ss_max / dz split by hit / miss.
# Native 800 ~ joint (low hit, launch < 0) -> the physics dt is the cause; native 800 ~ native 400 -> the joint scene.
# -> logs/JTIME_END.txt (EXIT trap). One Isaac at a time.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
H=$(ls -td $B/*human_pitched30c_long1 | head -1)/model_13298.pt
H0=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
O=logs/jtime; mkdir -p $O
M=logs/JTIME_END.txt; rm -f $M; MSG="aborted"; OUT=""
touch $O/chain.stamp
trap 'echo "JTIME_END $(date +%H:%M) $MSG$OUT" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
row() {  # $1 tag, $2 envs, $3 ckpt, $4 aim flags (or ""), rest: extra flags
  local TG=$1 N=$2 CK=$3 AX=$4; shift 4
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $CK $AX --envs $N --seed 0 --tag $TG --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --contact_log "$@" > $O/diag_$TG.txt 2>&1
  if grep -q Traceback $O/diag_$TG.txt || [ ! -f logs/diag_bat_contact_$TG.json ]; then OUT="$OUT | $TG: crash or no json"; return; fi
  OUT="$OUT | $TG: $($P - <<PY
import json, numpy as np
P = {k: np.array(v, dtype=float) for k, v in json.load(open("logs/diag_bat_contact_$TG.json"))["per_pitch"].items()}
z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10); h = z & (P["hit"] > 0)
m = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")
print(f"in-zone n {z.sum()} hit {h.sum() / max(z.sum(), 1):.3f}; hits: exit med {m(P['ev_contact'][h]):.1f} launch med {m(P['launch_contact'][h]):.1f} "
      f"fair {(np.abs(P['spray_contact'][h]) <= 45).mean() if h.any() else float('nan'):.2f}; all in-zone: ss_max med {m(P['ss_speed_max'][z]):.1f} "
      f"dz med {m(P['dz'][z]):+.3f} (hit {m(P['dz'][h]):+.3f} / miss {m(P['dz'][z & (P['hit'] == 0)]):+.3f}), ss_max hit {m(P['ss_speed_max'][h]):.1f} / miss {m(P['ss_speed_max'][z & (P['hit'] == 0)]):.1f}; t_square - t_plane med {m((P['t_square'] - P['t_plane'])[z]):+.3f} ref_err med {m(P['ref_err'][z]):.3f}")
PY
)"
}
until [ -f logs/RECOV4_END.txt ] && [ logs/RECOV4_END.txt -nt $O/chain.stamp ]; do sleep 60; done
MSG="13298 + saim, phase 2.6, j0, real heights (joint ph26 s6 / s7: hit 0.36 / 0.45, ss_max 16.1 / 15.9, dz -0.082 / -0.060)"
for c in obp_167-7 obp_282-1 obp_42-1 obp_42-2; do row N8_${c} 256 $H "$SAIM" --clips $c --physics_dt 0.00125; done
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
row N4_400 1024 $H "$SAIM" --clips $C4
row P8_12500 1024 $H0 "" --clips $C4 --physics_dt 0.00125
row P4_12500 1024 $H0 "" --clips $C4
