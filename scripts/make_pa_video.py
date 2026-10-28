"""M3 plate appearance video (planning 14, 10-28): the per-pitch joint films of logs/m3/pa_plan.csv joined in order, a count
card before each pitch, a result card after it (from the filmed row, not the plan) and an end card. Cards are rendered at
the clip size; the output is re-encoded once (libx264 crf 16, yuv420p, High) so every segment has the same stream format.

  python scripts/make_pa_video.py --plan logs/m3/pa_plan.csv --clips "videos/m3pa_p{pitch}_s{seed}_k{env}.mp4" \
      --rows "logs/m3/pa_film_p{pitch}.csv" --end "measured in this simulation: ..." --out videos/m3pa_draft.mp4
"""
import argparse
import csv

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RESULT = {"swinging_strike": "Swinging strike", "called_strike": "Called strike", "ball": "Ball", "foul": "Foul ball",
          "in_play": "In play"}
FONT = "C:/Windows/Fonts/arial.ttf"


def card(w, h, lines, sizes):
    im = Image.new("RGB", (w, h), (16, 20, 28))
    d = ImageDraw.Draw(im)
    fonts = [ImageFont.truetype(FONT, s) for s in sizes]
    tot = sum(f.size * 1.35 for f in fonts)
    y = (h - tot) / 2
    for line, f in zip(lines, fonts):
        x = (w - d.textlength(line, font=f)) / 2
        d.text((x, y), line, font=f, fill=(235, 238, 242))
        y += f.size * 1.35
    return np.asarray(im)


def result_text(p, row):
    r = RESULT[p["expected"]]
    if p["expected"] in ("foul", "in_play") and row:
        r += f"  -  exit {float(row['exit_mps']) * 3.6:.0f} km/h, launch {float(row['launch_deg']):.0f} deg, carry {float(row['carry_m']):.1f} m"
    return r


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--plan", default="logs/m3/pa_plan.csv")
    a.add_argument("--clips", default="videos/m3pa_p{pitch}_s{seed}_k{env}.mp4")
    a.add_argument("--rows", default="logs/m3/pa_film_p{pitch}.csv")
    a.add_argument("--title", default="ONE OUTS - one plate appearance|pitcher and batter learned by RL, one joint physics simulation")
    a.add_argument("--end", required=True, help="end card lines separated by |")
    a.add_argument("--card_s", type=float, default=2.0)
    a.add_argument("--out", default="videos/m3pa_draft.mp4")
    o = a.parse_args()
    plan = list(csv.DictReader(open(o.plan, newline="")))
    first = imageio.get_reader(o.clips.format(**plan[0]))
    fps = first.get_meta_data()["fps"]
    h, w = first.get_data(0).shape[:2]
    first.close()
    n_card = int(round(o.card_s * fps))
    wr = imageio.get_writer(o.out, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                            ffmpeg_params=["-crf", "16", "-profile:v", "high", "-preset", "slow"])

    def hold(img, n=n_card):
        for _ in range(n):
            wr.append_data(img)

    t = o.title.split("|")
    hold(card(w, h, t, [64] + [40] * (len(t) - 1)))
    for p in plan:
        hold(card(w, h, [f"Pitch {p['pitch']}", f"count {p['balls_before']}-{p['strikes_before']}"], [72, 52]))
        rows = list(csv.DictReader(open(o.rows.format(**p), newline="")))
        row = rows[int(p["env"])]
        rd = imageio.get_reader(o.clips.format(**p))
        k = 0
        for fr in rd:
            wr.append_data(fr)
            k += 1
        rd.close()
        hold(card(w, h, [result_text(p, row)], [60]))
        print(f"pitch {p['pitch']}: {k} frames, {result_text(p, row)}")
    e = o.end.split("|")
    hold(card(w, h, e, [48] * len(e)), 2 * n_card)
    wr.close()
    print(f"wrote {o.out} ({w}x{h} @ {fps} fps)")


if __name__ == "__main__":
    main()
