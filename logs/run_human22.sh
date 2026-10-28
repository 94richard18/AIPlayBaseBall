#!/bin/bash
# Human22 = arm B power line (decision session 10-28): AIB-PitchedBatting-Human22-v0 (Human15 P3c-shaped pitches, target
# 43 m, w_exit_velo 20 -> 60) at AIB_EFFORT_SCALE 1.5, from the delivery 12500, 2048 envs, 100-it chunks to +600.
# Gate: 20's walk-PA chain end (grep -q M3PA_END logs/m3pa2_chain.txt) + Isaac 0 twice 60 s apart.
# Falsifiers / early stop:
#   +100: train bat/exit_velo (EMA mean on hits) last 10 its >= its 11-20 + 2 m/s and bat/hit_rate last 10 >= 0.7 x its 11-20;
#   +200: diag1024 vs P3c t18 (timing jitter 0, bias 0.055, aim 0): EV median on hits >= 18, in-zone >= 0.43, falls
#         after a hit <= 55%. Pass -> continue to +600 (diag after each chunk, reported). No M2 unless +200 passes.
# Own end marker: the file logs/H22_END.txt ("H22_END HH:MM ...").
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
export AIB_EFFORT_SCALE=1.5
TASK=AIB-PitchedBatting-Human22-v0
C0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
R=assets/game/p3c_identical_t18_release.csv
TL=logs/train_human22.txt
M=logs/H22_END.txt
rm -f $M
end() { echo "H22_END $(date +%H:%M) $1" | tee $M; exit 0; }
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(cnt)" -lt 1 ]; do sleep 20; done; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
d() {  # $1 ckpt $2 tag
    idle
    timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --envs 1024 --seed 0 --tag h22_$2 --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > logs/h22_$2_contact.txt 2>&1
}
stats() { $P -c "
import json; s=json.load(open('logs/diag_bat_contact_h22_$1.json'))['summary']; h=s['hits']; c=s.get('contact_log') or {}
m=lambda v: v[1] if v else float('nan')
print(s['by_speed']['slow in zone']['hit_rate'], round(h['fell']/max(h['n'],1)*100), m(c.get('ev')))"; }
# one stat's trend over a chunk's log lines: "its 11-20 / last 10 / n" (the stats are EMAs that restart from 0 in each
# train.py run: ~7 its of warm-up, so the start window skips the first 10)
trend() { tail -n +$1 $TL | grep "bat/$2:" | grep -o "[0-9.]*$" | $P -c "
import sys; v=[float(x) for x in sys.stdin.read().split()]
w=v[10:20]; print(f'{sum(w)/max(len(w),1):.4f} {sum(v[-10:])/max(len(v[-10:]),1):.4f} {len(v)}')"; }

date +"%H:%M Human22: waiting for M3PA_END in logs/m3pa2_chain.txt"
until grep -q M3PA_END logs/m3pa2_chain.txt 2>/dev/null; do sleep 60; done
until [ "$(cnt)" -lt 1 ] && sleep 60 && [ "$(cnt)" -lt 1 ]; do sleep 20; done
date +"%H:%M gate open (M3PA_END + Isaac 0 twice 60 s apart)"

d $C0 base12500; read BZ BF BEV <<< "$(stats base12500)"
echo "$(date +%H:%M) baseline 12500: in-zone $BZ fall after hit $BF% EV med $BEV"

RUN=$(dirname $C0); RUN=$(basename $RUN); CK=$(basename $C0); LIST=""
for k in 1 2 3 4 5 6; do
    idle; L0=$(( $(wc -l < $TL 2>/dev/null || echo 0) + 1 ))
    $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 100 --run_name human_pitched22_$k --resume True --load_run $RUN --checkpoint $CK >> $TL 2>&1
    D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched22_$k 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
    [ -n "$I" ] || end "chunk $k: no checkpoint, stop | ckpts:$LIST"
    RUN=$(basename $D); CK=model_$I.pt; LIST="$LIST $D/$CK"
    read E0 E1 NE <<< "$(trend $L0 exit_velo)"; read H0 H1 NH <<< "$(trend $L0 hit_rate)"
    TR="train exit_velo $E0 -> $E1 hit_rate $H0 -> $H1 (n $NE)"
    if [ $k -eq 1 ]; then
        echo "$(date +%H:%M) $I (+100): $TR"
        $P -c "import sys; sys.exit(0 if $E1 >= $E0 + 2 and $H1 >= 0.7 * $H0 else 1)" || end "stop at +100: falsifier failed ($TR) | ckpts:$LIST"
        echo "+100 falsifier passed"
        continue
    fi
    d $D/$CK $I; read Z F EV <<< "$(stats $I)"
    echo "$(date +%H:%M) $I (+$((k*100))): in-zone $Z fall after hit $F% EV med $EV | $TR"
    [ -n "$Z" ] || end "diag failed at $I: stop | ckpts:$LIST"
    if [ $k -eq 2 ]; then
        $P -c "import sys; sys.exit(0 if float('$EV') >= 18 and float('$Z') >= 0.43 and $F <= 55 else 1)" || end "stop at +200: falsifier failed (EV $EV in-zone $Z fall $F%) | ckpts:$LIST"
        echo "+200 falsifier passed: continue to +600"
    fi
done
end "done to +600 | ckpts:$LIST | isaac after: $(cnt)"
