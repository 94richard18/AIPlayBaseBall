"""11-26 final cut (planning 14, 10-29 storyboard): the two showcase plate appearances joined from their per-pitch joint
films - title card, rule card, then per PA a title card, per pitch a count card / the clip (make_pa_video's trim) / a
result card, a 0.5x replay around contact (or the plate crossing on a miss) after the pitches listed in --replay, the PA
result card; a stats end card. Cards are Chinese with the English as a smaller second line (planning 14 10-29: the user
reads Chinese; the 10-28 progress video's msjh / msjhbd fonts). No simulation: the clips are the filmed rows. Re-encoded
once (libx264 crf 16, High, yuv420p).

  python scripts/make_final_cut.py --out videos/final_cut_v1.mp4 \
      --pa logs/m3/pa_plan_recov.csv "videos/m3clean_p{pitch}_s{seed}_k{env}.mp4" "logs/m3/clean_film_p{pitch}.csv" \
      --pa logs/m3/pa_plan_batter_recov.csv "videos/m3clean2_p{pitch}_s{seed}_k{env}.mp4" "logs/m3/clean2_film_p{pitch}.csv" \
      --replay 1:5 2:1
"""
import argparse
import csv
import math

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_pa_video import clip_end, result_text

ZH, ZH_B = "C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msjhbd.ttc"
BG, FG, FG_EN = (16, 20, 28), (235, 238, 242), (160, 168, 180)
TITLE = [("ONE OUTS", None, 80), ("投手與打者都由強化學習訓練", "pitcher vs batter, both learned by reinforcement learning", 44),
         ("同一個物理模擬，沒有預錄動畫", "one joint physics simulation - no animation", 44)]
RULE = [("一個出局定勝負", "One out decides it", 64), ("三振或出局：投手勝", "strikeout or out: the pitcher wins", 46),
        ("保送或安打：打者勝", "walk or hit: the batter wins", 46)]
RESULT_ZH = {"swinging_strike": "揮棒落空，好球", "called_strike": "未揮棒，好球", "ball": "壞球", "foul": "界外球",
             "in_play": "擊出界內球"}
PA_RESULT = {"out": ("三振：投手勝", "Strikeout - the pitcher wins"),
             "out_in_play": ("內野出局：投手勝", "Infield out - the pitcher wins"),
             "walk": ("四壞保送：打者勝", "Ball four - the batter wins"), "hit": ("安打：打者勝", "Hit - the batter wins")}
END = [("在這個模擬中：由這位投手的 64 球模擬抽出 20000 個打席（每球 10 個打者種子）", None, 40),
       ("打者勝率（保送、外野安打或全壘打）9.8%", None, 52),
       ("打者揮棒後，由另一個學習出來的恢復策略保持平衡", "after each swing the batter keeps its balance with a separately learned recovery policy", 36)]
REPLAY_ZH = "慢動作 0.5x"
REPLAY_BEFORE_S, REPLAY_AFTER_S, REPLAY_SLOW = 0.35, 0.35, 2  # joint time around the moment; each frame shown twice


def card(w, h, items):
    """items: (main, english or None, main size) - the main line bold, the English 0.55x below it in grey."""
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    lines = []
    for main, en, size in items:
        lines.append((main, ImageFont.truetype(ZH_B, size), FG))
        if en:
            lines.append((en, ImageFont.truetype(ZH, max(int(size * 0.55), 22)), FG_EN))
    y = (h - sum(f.size * 1.4 for _, f, _ in lines)) / 2
    for text, f, col in lines:
        d.text(((w - d.textlength(text, font=f)) / 2, y), text, font=f, fill=col)
        y += f.size * 1.4
    return np.asarray(im)


def result_zh(p, row):
    r = RESULT_ZH[p["expected"]]
    if p["expected"] == "ball" and p["balls_before"] == "3":
        r = "四壞球，保送"
    elif p["expected"] in ("called_strike", "swinging_strike") and p["strikes_before"] == "2":
        r += "，三振"
    if p["expected"] in ("foul", "in_play") and row:
        r += f"（擊球初速 {float(row['exit_mps']) * 3.6:.0f} km/h，仰角 {float(row['launch_deg']):.0f}°，飛行 {float(row['carry_m']):.1f} m）"
    return r


def label(f, text, font):
    """Bottom-centre caption on a film frame (the replay)."""
    im = Image.fromarray(f)
    d = ImageDraw.Draw(im)
    x, y = (im.width - d.textlength(text, font=font)) / 2, im.height - font.size - 24
    d.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0))
    return np.asarray(im)


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
    a.add_argument("--out", default="videos/final_cut_v1.mp4")
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
    rfont = ImageFont.truetype(ZH_B, 44)

    def hold(img, sec):
        for _ in range(int(round(sec * fps))):
            wr.append_data(img)
        n_sec["s"] += sec

    hold(card(w, h, TITLE), 3.0)
    hold(card(w, h, RULE), 2.5)
    for i, (plan, clips, rows) in enumerate(pas, 1):
        hold(card(w, h, [(f"第 {i} 個打席", f"Plate appearance {i}", 76)]), 1.5)
        for p, row in zip(plan, rows):
            b, s = p["balls_before"], p["strikes_before"]
            hold(card(w, h, [(f"第 {p['pitch']} 球", f"Pitch {p['pitch']}", 76),
                             (f"球數 {b}-{s}（{b} 壞 {s} 好）", f"count {b}-{s}", 54)]), 1.0)
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
            hold(card(w, h, [(result_zh(p, row), result_text(p, row), 56)]), 1.5)
            if (i, int(p["pitch"])) in replay:
                t0 = float(row.get("hit_t", "nan"))
                if math.isnan(t0):
                    t0 = float(row["t_plane"])
                k0, k1 = int((t0 - REPLAY_BEFORE_S) / o.frame_dt), int(math.ceil((t0 + REPLAY_AFTER_S) / o.frame_dt))
                rd = imageio.get_reader(clips.format(**p))
                seg = [f for k, f in enumerate(rd) if max(k0, 0) <= k <= k1]
                rd.close()
                for f in seg:
                    f = label(f, REPLAY_ZH, rfont)
                    for _ in range(REPLAY_SLOW):
                        wr.append_data(f)
                n_sec["s"] += len(seg) * REPLAY_SLOW / fps
                print(f"PA{i} pitch {p['pitch']}: replay around t {t0:.3f} s, frames {max(k0, 0)}-{k1}")
            print(f"PA{i} pitch {p['pitch']}: {n_live} frames to t {end:.2f} s, {result_text(p, row)}")
        zh, en = PA_RESULT[pa_outcome(plan, rows)]
        hold(card(w, h, [(zh, en, 68)]), 2.0)
    en = open("logs/m3pa_endcard.txt", encoding="utf-8").read().strip().split("|")
    hold(card(w, h, [(END[0][0], en[0], END[0][2]), (END[1][0], en[1], END[1][2]), END[2]]), 4.0)
    wr.close()
    print(f"wrote {o.out} ({w}x{h} @ {fps} fps, {n_sec['s']:.1f} s)")


if __name__ == "__main__":
    main()
