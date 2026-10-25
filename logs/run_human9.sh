#!/bin/bash
# Human9 (ONE OUTS target 29 m + train at aim_offset_z -0.12) from Human7 model_12095: 600 its, early stop at +200,
# then the ONE OUTS gate (decision session 10-25): CPU 1000 matches, same seed, measured hit rate; base 8.0%, improve >= 11%, fail < 8.0%.
# NOT started automatically - needs the decision session's go-ahead and a slot (pitcher first).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export IMAGEIO_FFMPEG_EXE='C:\Users\User\anaconda3\envs\isaaclab_env\lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe'
export SAVE_EVERY=50  # decision session 10-25: yield both slots at the next checkpoint when the pitcher starts training
date +"%H:%M start Human9"
$P -u scripts/train.py --task AIB-PitchedBatting-Human9-v0 --headless --num_envs 2048 --max_iterations 600 --run_name human_pitched9 --resume True --load_run 2026-10-22_22-46-49_human_pitched7 --checkpoint model_12095.pt > logs/train_human9.txt 2>&1 &
TP=$!
# early stop: at iteration 12295 (+200, ~26 min) the training success rate (fair and >= 29 m, of hits) must be >= 0.08 (Human7 ~0.06)
while kill -0 $TP 2>/dev/null; do
    it=$(grep -o "Learning iteration [0-9]*" logs/train_human9.txt | tail -1 | grep -o "[0-9]*$")
    if [ -n "$it" ] && [ "$it" -ge 12295 ]; then
        sr=$(grep "bat/success_rate" logs/train_human9.txt | tail -1 | grep -o "[0-9.]*$")
        if $P -c "import sys; sys.exit(0 if float('$sr') < 0.08 else 1)"; then echo "early stop at $it: success_rate $sr < 0.08"; kill $TP; fi
        break
    fi
    sleep 60
done
wait $TP
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched9 | head -1); C=$D/$(ls $D | grep model | sort -t_ -k2 -n | tail -1)
date +"%H:%M Human9 ended $C"
grep "bat/hit_rate\|bat/success_rate\|bat/fair_rate\|bat/carry:" logs/train_human9.txt | tail -4 | tr -s ' '
$P -u scripts/diag_bat_contact.py --headless --checkpoint $C --tag human9 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/h9_contact.txt 2>&1
HR=$(grep -o "hit rate [0-9.]*" logs/h9_contact.txt | grep -o "[0-9.]*$"); echo "hit rate $HR"
timeout 420 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $C --eval_envs 256 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/b_aimy_h9.txt 2>&1
$P scripts/sum_aimy.py h9
$P - <<'PY'
import re, json
PAT = re.compile(r"\[play\] env (\d+): carry\s+([\d.]+) m \| EV\s+([\d.]+) m/s.*?\| LA\s+([-\d.]+) deg \| spray\s+([-+\d.]+) deg \| spin\s+([\d.]+) rpm \(backspin\s+([-+\d.]+)\)")
seen, sw = set(), []
for m in PAT.finditer(open("logs/b_aimy_h9.txt", errors="ignore").read()):
    g = m.groups(); e = int(g[0])
    if e in seen: continue
    seen.add(e); sw.append(dict(env=e, carry_m=float(g[1]), exit_velo_mps=float(g[2]), launch_deg=float(g[3]), spray_deg=float(g[4]), spin_rpm=float(g[5]), backspin_rpm=float(g[6])))
json.dump(dict(source="logs/b_aimy_h9.txt", swings=sw), open("logs/b_aimy_table_h9.json", "w"))
PY
$P scripts/one_outs.py --matches 1000 --batter_table logs/b_aimy_table_h9.json --hit_rate $HR --show 0 2>&1 | tail -2
date +"%H:%M done"
