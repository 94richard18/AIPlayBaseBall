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
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_pa_video import LAND_TAIL_S, clip_end, result_text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aibaseball.game.agents import StrikeZone  # noqa: E402
from aibaseball.game.field import MOUND_DIST, FieldSpec  # noqa: E402
from aibaseball.game.one_outs import INFIELD_RADIUS_M  # noqa: E402

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
    """Judged from the last film's simulated ball (planning 14 11-06): a fair first landing beyond the 95 ft infield
    circle around the rubber is a hit (one_outs.judge_pitch's rule; the old carry >= 38.9 m stand-in is gone)."""
    last, row = plan[-1], rows[-1]
    if last["expected"] == "ball" and last["balls_before"] == "3":
        return "walk"
    if last["expected"] == "in_play" and row:
        x, y = float(row["ball_land_x_game"]), float(row["ball_land_y_game"])
        if FieldSpec().is_fair(x, y) and math.hypot(x - MOUND_DIST, y) > INFIELD_RADIUS_M and math.hypot(x, y) > MOUND_DIST:
            return "hit"
    return "out_in_play" if last["expected"] == "in_play" else "out"


def film_check(p, row):
    """The film's own ball against the plan's call: a take is a strike iff the simulated crossing is in the zone."""
    if p["expected"] in ("ball", "called_strike"):
        y, z = float(row["plate_y"]), float(row["plate_z"])
        got = "called_strike" if StrikeZone().contains(y, z) else "ball"
        return got == p["expected"], f"take crossing y {y:+.3f} z {z:.3f} -> {got}"
    if p["expected"] == "in_play":
        return row.get("hit") == "True" and row.get("ball_fair") == "True", f"hit {row.get('hit')} fair {row.get('ball_fair')}"
    return True, ""


def scoreboard(f, font, match, n_match, wins, b, s, kmh):
    """20 11-06 band (planning 14): match k / N, pitcher-batter wins, the count before the pitch, the pitch speed;
    bottom-left of the left (pitcher-side) view, over the mound dirt. No innings / outs / runs (ONE OUTS)."""
    im = Image.fromarray(f)
    d = ImageDraw.Draw(im, "RGBA")
    text = f"第 {match} / {n_match} 場    投手 {wins[0]} - {wins[1]} 打者    球數 {b} 壞 {s} 好    球速 {kmh:.0f} km/h"
    w = d.textlength(text, font=font)
    x0, y0 = 16, im.height - font.size - 34
    d.rectangle([x0, y0, x0 + w + 28, y0 + font.size + 20], fill=(10, 14, 22, 190))
    d.text((x0 + 14, y0 + 8), text, font=font, fill=(240, 242, 245))
    return np.asarray(im)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pa", nargs=3, action="append", required=True, metavar=("PLAN", "CLIPS", "ROWS"))
    a.add_argument("--replay", nargs="*", default=[], help="PA:pitch (1-based PA) to replay at 0.5x")
    a.add_argument("--out", default="videos/final_cut_v1.mp4")
    a.add_argument("--frame_dt", type=float, default=0.0175)
    a.add_argument("--end_card", default="logs/m3pa_endcard.txt", help="'none': no stats end card (its numbers are M2-era)")
    a.add_argument("--max_end", type=float, default=None,
                   help="in-play clips run to the landing (done_t) + 0.3 s, capped here (planning 14 11-06: the path view's landing "
                        "ring; the 4.5 s cap stays for the other pitches); default = make_pa_video's clip_end")
    a.add_argument("--take_tail", type=float, default=None, help="take clips end at done_t (or t_plane) + this (14 11-06: 0.2)")
    a.add_argument("--comment", default=None, help="mp4 comment metadata (e.g. that takes are kinematic)")
    a.add_argument("--match_cards", action="store_true", help="'match k / N' cards + the scoreboard band (20 11-06)")
    a.add_argument("--final_tally", action="store_true", help="closing card: the W-L over all N matches (14 11-06: final video)")
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
                            ffmpeg_params=["-crf", "16", "-profile:v", "high", "-preset", "slow"]
                            + (["-metadata", f"comment={o.comment}"] if o.comment else []))
    n_sec = {"s": 0.0}
    rfont = ImageFont.truetype(ZH_B, 44)
    sfont = ImageFont.truetype(ZH_B, 30)
    wins = [0, 0]  # pitcher, batter
    bad = []

    def hold(img, sec):
        for _ in range(int(round(sec * fps))):
            wr.append_data(img)
        n_sec["s"] += sec

    hold(card(w, h, TITLE), 3.0)
    hold(card(w, h, RULE), 2.5)
    for i, (plan, clips, rows) in enumerate(pas, 1):
        if o.match_cards:
            hold(card(w, h, [(f"第 {i} / {len(pas)} 場", f"Match {i} / {len(pas)}", 76),
                             (f"投手 {wins[0]} - {wins[1]} 打者", f"pitcher {wins[0]} - {wins[1]} batter", 48)]), 1.5)
        else:
            hold(card(w, h, [(f"第 {i} 個打席", f"Plate appearance {i}", 76)]), 1.5)
        for p, row in zip(plan, rows):
            ok, why = film_check(p, row)
            print(f"PA{i} pitch {p['pitch']}: film {'MATCHES' if ok else 'DIFFERS FROM'} the plan ({p['expected']}; {why})")
            if not ok:
                bad.append(f"PA{i} pitch {p['pitch']}")
            b, s = p["balls_before"], p["strikes_before"]
            hold(card(w, h, [(f"第 {p['pitch']} 球", f"Pitch {p['pitch']}", 76),
                             (f"球數 {b}-{s}（{b} 壞 {s} 好）", f"count {b}-{s}", 54)]), 1.0)
            end = clip_end(p, row)
            if o.max_end is not None and p["expected"] == "in_play":
                end = min(float(row["done_t"]) + LAND_TAIL_S, o.max_end)
            if o.take_tail is not None and p["expected"] in ("ball", "called_strike"):
                # planning 14 11-06: take mode 11 holds the stance kinematically and R6b snaps on release after the pitch,
                # so a take ends with the ball at the catcher and the call card follows
                t_d = float(row["done_t"])
                end = (t_d if t_d == t_d else float(row["t_plane"])) + o.take_tail
            rd = imageio.get_reader(clips.format(**p))  # streamed: a 343-frame clip is ~1.9 GB decoded
            n_live = 0
            for k, f in enumerate(rd):
                if k * o.frame_dt > end + 1e-6:
                    break
                if o.match_cards:
                    f = scoreboard(f, sfont, i, len(pas), wins, b, s, float(row["kmh"]))
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
        res = pa_outcome(plan, rows)
        wins[res in ("walk", "hit")] += 1
        zh, en = PA_RESULT[res]
        lines = [(zh, en, 68)]
        if o.match_cards:
            lines.append((f"投手 {wins[0]} - {wins[1]} 打者", f"pitcher {wins[0]} - {wins[1]} batter", 48))
        hold(card(w, h, lines), 2.0)
        print(f"PA{i} result: {res} (pitcher {wins[0]} - {wins[1]} batter)")
    if o.final_tally:
        win = ("投手勝", "the pitcher wins") if wins[0] > wins[1] else ("打者勝", "the batter wins") if wins[1] > wins[0] else ("平手", "a tie")
        hold(card(w, h, [("最終比分", f"Final score - {len(pas)} matches", 76),
                         (f"投手 {wins[0]} - {wins[1]} 打者", f"pitcher {wins[0]} - {wins[1]} batter", 60),
                         (win[0], win[1], 48)]), 4.0)
    if o.end_card != "none":
        en = open(o.end_card, encoding="utf-8").read().strip().split("|")
        hold(card(w, h, [(END[0][0], en[0], END[0][2]), (END[1][0], en[1], END[1][2]), END[2]]), 4.0)
    wr.close()
    print(f"wrote {o.out} ({w}x{h} @ {fps} fps, {n_sec['s']:.1f} s); plan check: {'all films match' if not bad else 'DIFFER ' + ' '.join(bad)}")


if __name__ == "__main__":
    main()
