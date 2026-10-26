#!/bin/bash
# Decision session 10-26 21:00: after the 4 fall runs (same GPU slot, before the pitcher's P3CPC), film batter 12500 vs P3c
# (film_1026.sh batter part, crf 16 from e36e4fa). Uses the passing fall variant with the lowest fall rate, else the plain setting.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export IMAGEIO_FFMPEG_EXE='C:\Users\User\anaconda3\envs\isaaclab_env\lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe'
until grep -q '^FALLZSALL_END' logs/b_fall_zeroshot_all.txt; do sleep 30; done
pick=$($P - <<'PY'
import json
flags = {"fall_contact_0.6": "--post_ref_pose contact --post_ref_blend 0.6", "fall_contact_0.3": "--post_ref_pose contact --post_ref_blend 0.3",
         "fall_start_0.6": "--post_ref_pose start --post_ref_blend 0.6",
         "fall_contact_0.6_keepres": "--post_ref_pose contact --post_ref_blend 0.6 --post_ref_keep_residual"}
best = None
for tag in flags:
    try:
        S = json.load(open(f"logs/diag_bat_contact_{tag}.json"))["summary"]; s = S["hits"]
    except Exception:
        continue
    hit, fall = S["hit_rate"], s["fell"] / max(s["n"], 1)
    if abs(hit - 0.684) <= 0.02 and fall <= 0.37 and (best is None or fall < best[1]):
        best = (tag, fall)
print(f"{best[0]}|{flags[best[0]]}" if best else "none|")
PY
)
tag=${pick%%|*}; extra=${pick#*|}
echo "$(date +%H:%M) film setting: $tag $extra"
OUT=videos/batter_h9b_12500_p3c_1026b.mp4
for try in 1 2; do
    timeout 1800 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt --swings 5 --all --release_csv assets/game/p3c_identical_t18_release.csv --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0.0 $extra --out $OUT 2>&1 | grep -E "^\[pitched\]|Error"
    [ -s $OUT ] && $P scripts/check_video_frames.py $OUT && break
done
echo "BFILM_END $(date +%H:%M)"
