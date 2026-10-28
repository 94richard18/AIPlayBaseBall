#!/bin/bash
# Human23 (decision session 10-29): AIB-PitchedBatting-Human23-v0 (Human15 P3c pitches + contact-window joint-velocity
# tracking, w_cvel 1.0 / sigma 8 rad/s over waist + left arm) at AIB_EFFORT_SCALE 1.5, from the delivery 12500, 2048 envs,
# 100-it chunks to +600.
# Falsifiers / early stop:
#   +100: train mimic/cvel_err (EMA, window RMS rad/s) last 10 its <= 0.8 x its 11-20, and bat/hit_rate last 10 >= 0.7 x its 11-20;
#   +200: diag1024 vs P3c t18 (jitter 0, bias 0.055, aim 0, --link_log): l_shoulder_pitch and l_wrist_pitch peaks >= 0.05 s
#         later than 12500's (-0.223 / -0.255 s), sweet spot at contact >= 21.6 m/s, in-zone >= 0.40, falls after a hit <= 55%.
#   Pass -> to +600 (diag after each chunk, reported). No M2 unless +200 passes.
# Gate: Isaac 0 twice 60 s apart. End marker: the file logs/H23_END.txt ("H23_END HH:MM ..."), also on failure.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
export AIB_EFFORT_SCALE=1.5
TASK=AIB-PitchedBatting-Human23-v0
C0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
R=assets/game/p3c_identical_t18_release.csv
TL=logs/train_human23.txt
M=logs/H23_END.txt
rm -f $M
MSG="aborted"
trap 'echo "H23_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
end() { MSG="$1"; exit 0; }
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(cnt)" -lt 1 ]; do sleep 20; done; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
d() {  # $1 ckpt $2 tag
    idle
    timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --envs 1024 --seed 0 --tag h23_$2 --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log --chain_log --phase_log --link_log > logs/h23_$2_contact.txt 2>&1
}
stats() { $P -c "
import json; S = json.load(open('logs/diag_bat_contact_h23_$1.json'))['summary']; h = S['hits']; c = S['contact_log']; L = S['link_log']['rows']
print(S['by_speed']['slow in zone']['hit_rate'], round(h['fell'] / max(h['n'], 1) * 100), c['ss_contact'][1], c['ev'][1],
      L['l_shoulder_pitch']['t_pk_sim'], L['l_wrist_pitch']['t_pk_sim'], L['waist_yaw']['t_pk_sim'])"; }
# one stat's trend over a chunk's log lines: "its 11-20 / last 10 / n" (EMAs restart from 0 in each train.py run)
trend() { tail -n +$1 $TL | grep "$2:" | grep -o "[0-9.]*$" | $P -c "
import sys; v=[float(x) for x in sys.stdin.read().split()]
w=v[10:20]; print(f'{sum(w)/max(len(w),1):.4f} {sum(v[-10:])/max(len(v[-10:]),1):.4f} {len(v)}')"; }

until [ "$(cnt)" -lt 1 ] && sleep 60 && [ "$(cnt)" -lt 1 ]; do sleep 20; done
date +"%H:%M Human23: gate open (Isaac 0 twice 60 s apart)"

RUN=$(dirname $C0); RUN=$(basename $RUN); CK=$(basename $C0); LIST=""
for k in 1 2 3 4 5 6; do
    idle; L0=$(( $(cat $TL 2>/dev/null | wc -l) + 1 ))
    $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 100 --run_name human_pitched23_$k --resume True --load_run $RUN --checkpoint $CK >> $TL 2>&1
    D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched23_$k 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
    [ -n "$I" ] || end "chunk $k: no checkpoint, stop | ckpts:$LIST"
    RUN=$(basename $D); CK=model_$I.pt; LIST="$LIST $D/$CK"
    read V0 V1 NV <<< "$(trend $L0 mimic/cvel_err)"; read H0 H1 NH <<< "$(trend $L0 bat/hit_rate)"
    TR="train cvel_err $V0 -> $V1 hit_rate $H0 -> $H1 (n $NV)"
    if [ $k -eq 1 ]; then
        echo "$(date +%H:%M) $I (+100): $TR"
        $P -c "import sys; sys.exit(0 if $NV >= 20 and $V1 <= 0.8 * $V0 and $H1 >= 0.7 * $H0 else 1)" || end "stop at +100: falsifier failed ($TR) | ckpts:$LIST"
        echo "+100 falsifier passed"
        continue
    fi
    d $D/$CK $I; read Z F SS EV TS TW TY <<< "$(stats $I)"
    echo "$(date +%H:%M) $I (+$((k*100))): in-zone $Z fall after hit $F% ss contact $SS EV med $EV | peak t l_sh_pitch $TS l_wr_pitch $TW waist_yaw $TY | $TR"
    [ -n "$Z" ] || end "diag failed at $I: stop | ckpts:$LIST"
    if [ $k -eq 2 ]; then
        $P -c "import sys; sys.exit(0 if $TS >= -0.173 and $TW >= -0.205 and $SS >= 21.6 and $Z >= 0.40 and $F <= 55 else 1)" || end "stop at +200: falsifier failed (in-zone $Z fall $F% ss $SS peaks $TS / $TW) | ckpts:$LIST"
        echo "+200 falsifier passed: continue to +600"
    fi
done
end "done to +600 | ckpts:$LIST | isaac after: $(cnt)"
