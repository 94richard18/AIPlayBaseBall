"""11-26 final cut (planning 14, 10-29 storyboard): the two showcase plate appearances joined from their per-pitch joint
films - title card, rule card, then per PA a title card, per pitch a count card / the clip (make_pa_video's trim) / a
result card, a 0.5x replay around contact (or the plate crossing on a miss) after the pitches listed in --replay, the PA
result card; a stats end card. No simulation: the clips are the filmed rows. Re-encoded once (libx264 crf 16, High,
yuv420p).

  python scripts/make_final_cut.py --out videos/final_cut_v0.mp4 \
      --pa logs/m3/pa_plan_recov.csv "videos/m3rec_p{pitch}_s{seed}_k{env}.mp4" "logs/m3/rec_film_p{pitch}.csv" \
      --pa logs/m3/pa_plan_batter_recov.csv "videos/m3pa2_p{pitch}_s{seed}_k{env}.mp4" "logs/m3/pa2_film_p{pitch}.csv" \
      --replay 1:5 2:1
"""
import argparse
import csv
import math

import cv2
import imageio.v2 as imageio

from make_pa_video import card, clip_end, result_text

TITLE = ["ONE OUTS", "pitcher vs batter, both learned by reinforcement learning",
         "one joint physics simulation - no animation"]
RULE = ["One out decides it", "strikeout or out: the pitcher wins", "walk or hit: the batter wins"]
PA_RESULT = {"out": "Strikeout - the pitcher wins", "out_in_play": "Infield out - the pitcher wins","walk": "Ball four - the batter wins", "hit": "Hit - the batter wins"}
END_NOTE = "after each swing the batter keeps its balance with a separately learned recovery policy"
REPLAY_BEFORE_S, REPLAY_AFTER_S, REPLAY_SLOW = 0.35, 0.35, 2  # joint time around the moment; each frame shown twice


def pa_outcome(plan, rows):
    last, row = plan[-1], rows[-1]
    if last["expected"] == "ball" and last["balls_before"] == "3":
        return "walk"
    if last["expected"] == "in_play" and row and row.get("ball_fair") == "True" and float(row["carry_m"]) >= 38.9:
        return "hit"  # outfield (memory: 29 m is still infield)
    return "out_in_play" if last["expected"] == "in_play" else "out"


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pa", nargs=3, action="append", required=True, metavar=("PLAN", "CLIPS", "ROWS"))
    a.add_argument("--replay", nargs="*", default=[], help="PA:pitch (1-based PA) to replay at 0.5x")
    a.add_argument("--end", default="logs/m3pa_endcard.txt")
    a.add_argument("--out", default="videos/final_cut_v0.mp4")
    a.add_argument("--frame_dt", type=float, default=0.0175)
    o = a.parse_args()
    replay = {tuple(int(x) for x in s.split(":")) for s in o.replay}
    pas = []
    for plan_p, clips, rows_p in o.pa:
        plan = list(csv.DictReader(open(plan_p, newline="")))
        rows = [list(csv.DictReader(open(rows_p.format(**p), newline="")))[int(p["env"])] for p in plan]
        pas.append((plan, clips, rows))
    first = imageio.get_reader(pas[0][1].format(**pas[0][0][0]))
    fps = first.get_meta_data()["fps"]
    h, w = first.get_data(0).shape[:2]
    first.close()
    wr = imageio.get_writer(o.out, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                            ffmpeg_params=["-crf", "16", "-profile:v", "high", "-preset", "slow"])
    n_sec = {"s": 0.0}

    def hold(img, sec):
        for _ in range(int(round(sec * fps))):
            wr.append_data(img)
        n_sec["s"] += sec

    hold(card(w, h, TITLE, [72, 40, 40]), 3.0)
    hold(card(w, h, RULE, [60, 44, 44]), 2.5)
    for i, (plan, clips, rows) in enumerate(pas, 1):
        hold(card(w, h, [f"Plate appearance {i}"], [72]), 1.5)
        for p, row in zip(plan, rows):
            hold(card(w, h, [f"Pitch {p['pitch']}", f"count {p['balls_before']}-{p['strikes_before']}"], [72, 52]), 1.0)
            end = clip_end(p, row)
            rd = imageio.get_reader(clips.format(**p))  # streamed: a 343-frame clip is ~1.9 GB decoded
            n_live = 0
            for k, f in enumerate(rd):
                if k * o.frame_dt > end + 1e-6:
                    break
                wr.append_data(f)
                n_live += 1
            rd.close()
            n_sec["s"] += n_live / fps
            hold(card(w, h, [result_text(p, row)], [60]), 1.5)
            if (i, int(p["pitch"])) in replay:
                t0 = float(row.get("hit_t", "nan"))
                if math.isnan(t0):
                    t0 = float(row["t_plane"])
                k0, k1 = int((t0 - REPLAY_BEFORE_S) / o.frame_dt), int(math.ceil((t0 + REPLAY_AFTER_S) / o.frame_dt))
                rd = imageio.get_reader(clips.format(**p))
                seg = [f for k, f in enumerate(rd) if max(k0, 0) <= k <= k1]
                rd.close()
                for f in seg:
                    f = f.copy()
                    cv2.putText(f, "replay 0.5x", (w // 2 - 110, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
                    cv2.putText(f, "replay 0.5x", (w // 2 - 110, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
                    for _ in range(REPLAY_SLOW):
                        wr.append_data(f)
                n_sec["s"] += (k1 + 1 - max(k0, 0)) * REPLAY_SLOW / fps
                print(f"PA{i} pitch {p['pitch']}: replay around t {t0:.3f} s, frames {max(k0, 0)}-{k1}")
            print(f"PA{i} pitch {p['pitch']}: {n_live} frames to t {end:.2f} s, {result_text(p, row)}")
        hold(card(w, h, [PA_RESULT[pa_outcome(plan, rows)]], [64]), 2.0)
    hold(card(w, h, open(o.end, encoding="utf-8").read().strip().split("|") + [END_NOTE], [44, 44, 32]), 4.0)
    wr.close()
    print(f"wrote {o.out} ({w}x{h} @ {fps} fps, {n_sec['s']:.1f} s)")


if __name__ == "__main__":
    main()
