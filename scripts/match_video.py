"""ONE OUTS match video: plate appearances of the RL pitcher against the RL batter, every pitch a real Isaac clip.

Per pitch (14's approved plan, 2026-10-27):
  1. the pitcher clip k and its own release state (row k of --pitcher_release, eval_stage --release_csv format);
  2. the plate crossing of that release (drag + Magnus, build_library_from_release); a release that never reaches
     the plate is a ball;
  3. hit by pitch -> the batter wins (the engine's body box);
  4. take / swing: TableBatter.decides_to_swing with the PA's rng (the engine's zone probabilities and draws);
  5. the batter clip faces exactly that pitch (showcase --release_csv <one row>, --take for a take);
  6. the count: judge_pitch (the engine's rules) on the Isaac outcome - a miss, or the batted ball's first bounce.
The end card's win rate is the engine's 5 x 1000-match simulation, labelled as such.

--dry_run: no Isaac - the swing outcome is drawn as TableBatter.swing would (hit rate, then a table row); tests the
logic and the compositor on CPU (placeholder frames when a clip is missing).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402

from aibaseball.game import TableBatter  # noqa: E402
from aibaseball.game.agents import BattedBall, Pitch, StrikeZone  # noqa: E402
from aibaseball.game.field import FieldSpec  # noqa: E402
from aibaseball.game.fielding import FieldingSpec  # noqa: E402
from aibaseball.game.one_outs import is_hit_by_pitch, judge_pitch  # noqa: E402
from aibaseball.physics import PhysicsSpec  # noqa: E402
from aibaseball.physics.grass import GrassSpec  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--pitcher_video", required=True,
                    help="eval_stage --video file of --pitcher_episodes episodes (equal frame counts), or a glob of one clip per pitch")
parser.add_argument("--pitcher_episodes", type=int, default=0, help="> 0: split --pitcher_video into this many equal clips")
parser.add_argument("--pitcher_release", required=True,
                    help="their release rows (row k = episode k; eval_stage --video_release_csv, not --release_csv)")
parser.add_argument("--checkpoint", default="logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt")
parser.add_argument("--batter_table", default="logs/b_table_e15_12500.json", help="dry run: the delivery's batted balls")
parser.add_argument("--hit_rate", type=float, default=0.458, help="dry run: the delivery's in-zone hit rate")
parser.add_argument("--pitcher_layout", choices=["pip", "side_by_side"], default="pip",
                    help="pitcher film (side view | plate view): pip = side view full frame, plate view inset bottom right")
parser.add_argument("--variant", choices=["A", "B"], default="A", help="A: ball-follow after contact; B: batter in frame 2 s")
parser.add_argument("--seed", type=int, default=0, help="the PA rng (take decisions)")
parser.add_argument("--pas", type=int, default=1, help="plate appearances (pitcher clips are used in order)")
parser.add_argument("--end_card", default="ONE OUTS simulation, 5 x 1000 matches: batter wins 18.8% (16.0-20.1%)")
parser.add_argument("--work", default="videos/match_work")
parser.add_argument("--out", default=None)
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

P = sys.executable
SHOW = ["scripts/showcase.py", "--headless", "--task", "human_pitched", "--swings", "1", "--all",
        "--timing_noise", "0", "--timing_bias", "0.055", "--aim_offset_z", "0.0"]
FPS, SIZE = 30, (1280, 720)


def plate_crossing(row: dict, path: str):
    """(speed m/s, engine plate_y (+ third base), plate_z) of one release row, or None if it never reaches the plate."""
    import torch

    from aibaseball.physics.pitch_gen import build_library_from_release

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    try:
        lib = build_library_from_release(path, PhysicsSpec())
    except AssertionError:
        return None
    with torch.no_grad():  # the library's frame: +y the third-base side, as the engine's plate_y
        return float(lib.v0[0].norm()), float(lib.target[0, 1]), float(lib.target[0, 2])


def batter_clip(release_path: str, take: bool, tag: str):
    """Runs the Isaac batter on one pitch; returns (clip path, per-pitch result dict)."""
    out, res = os.path.join(args.work, f"{tag}.mp4"), os.path.join(args.work, f"{tag}.json")
    cmd = [P, "-u", *SHOW, "--checkpoint", args.checkpoint, "--release_csv", release_path, "--out", out,
           "--result_json", res] + (["--take"] if take else []) + (["--hold_after_hit", "2.0"] if args.variant == "B" else [])
    env = dict(os.environ, AIB_EFFORT_SCALE=os.environ.get("AIB_EFFORT_SCALE", "1.5"))
    subprocess.run(cmd, check=True, env=env, timeout=1800)
    with open(res, encoding="utf-8") as f:
        return out, json.load(f)[0]


def fit(fr):
    """Letterboxed into SIZE (the pitcher film is the side view and the plate view side by side)."""
    h, w = fr.shape[:2]
    s = min(SIZE[0] / w, SIZE[1] / h)
    img = cv2.resize(fr, (int(w * s), int(h * s)))
    out = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    y0, x0 = (SIZE[1] - img.shape[0]) // 2, (SIZE[0] - img.shape[1]) // 2
    out[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return out


def pip(fr):
    """eval_stage film frame = side view | plate view (equal halves) -> side view with the plate view inset."""
    h, w = fr.shape[:2]
    if args.pitcher_layout != "pip" or w < 2 * h:
        return fr
    side, plate = fr[:, : w // 2], fr[:, w // 2:]
    out = cv2.resize(side, SIZE)
    iw, ih = SIZE[0] // 3, SIZE[1] // 3
    x0, y0 = SIZE[0] - iw - 16, SIZE[1] - ih - 16
    out[y0 - 3:y0 + ih + 3, x0 - 3:x0 + iw + 3] = 255
    out[y0:y0 + ih, x0:x0 + iw] = cv2.resize(plate, (iw, ih))
    return out


def resampled(frames, fps):
    """Source frames at fps -> FPS (frames repeated / dropped by time)."""
    n = 0
    for i, fr in enumerate(frames):
        while n < (i + 1) * FPS / fps - 1e-6:
            yield fr
            n += 1


def pitcher_clips():
    """One (frames list, fps) per pitch."""
    if args.pitcher_episodes > 0:
        r = imageio.get_reader(args.pitcher_video)
        fps, frames = r.get_meta_data()["fps"], [np.asarray(f)[..., :3] for f in r]
        n = len(frames) // args.pitcher_episodes
        assert n * args.pitcher_episodes == len(frames), f"{len(frames)} frames do not split into {args.pitcher_episodes} episodes"
        return [([pip(f) for f in frames[i * n:(i + 1) * n]], fps) for i in range(args.pitcher_episodes)]
    out = []
    for c in sorted(glob.glob(args.pitcher_video)):
        r = imageio.get_reader(c)
        out.append(([np.asarray(f)[..., :3] for f in r], r.get_meta_data()["fps"]))
    return out


def frames_of(src, n_placeholder: int, text: str):
    """src: (frames, fps), a clip path, or None (placeholder)."""
    if isinstance(src, str) and os.path.exists(src):
        r = imageio.get_reader(src)
        src = ([np.asarray(f)[..., :3] for f in r], r.get_meta_data()["fps"])
    if isinstance(src, tuple):
        for fr in resampled(src[0], src[1]):
            yield fit(fr)
        return
    blank = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8)
    cv2.putText(blank, text, (60, SIZE[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 230), 2, cv2.LINE_AA)
    for _ in range(n_placeholder):
        yield blank


def banner(fr, lines):
    fr = fr.copy()
    cv2.rectangle(fr, (0, 0), (SIZE[0], 44 * len(lines) + 16), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(fr, s, (24, 44 + 44 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return fr


def card(lines, seconds=3.0):
    fr = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(fr, s, (60, 260 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return [fr] * int(seconds * FPS)


def main():
    os.makedirs(args.work, exist_ok=True)
    clips = pitcher_clips()
    with open(args.pitcher_release, newline="") as f:
        rows = list(csv.DictReader(f))
    assert args.dry_run or len(clips) == len(rows), f"{len(clips)} pitcher clips vs {len(rows)} release rows"
    zone, fld, fs, grass, phys = StrikeZone(), FieldSpec(), FieldingSpec(), GrassSpec(), PhysicsSpec()
    tb = TableBatter.from_json(args.batter_table, args.hit_rate)
    out = args.out or f"videos/match_{args.variant}_s{args.seed}.mp4"
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8)
    log, k = [], 0
    for pa in range(args.pas):
        rng = random.Random(args.seed * 1000 + pa)
        balls = strikes = 0
        res = None
        while res is None:
            assert k < len(rows), "ran out of pitcher clips"
            rel = os.path.join(args.work, f"pa{pa}_p{k}_release.csv")
            cross = plate_crossing(rows[k], rel)
            p = Pitch(cross[0], cross[1], cross[2]) if cross else Pitch(float(rows[k]["speed"]), 2.0, 0.0)
            ev = dict(pa=pa, pitch=k, balls=balls, strikes=strikes, plate_y=p.plate_y, plate_z=p.plate_z,
                      kmh=p.speed_mps * 3.6, reached=cross is not None)
            if is_hit_by_pitch(p.plate_y, p.plate_z):
                kind, ball, res, extra, clip = "hit_by_pitch", None, ("batter", "hit_by_pitch"), {}, None
                if not args.dry_run:
                    clip, _ = batter_clip(rel, True, f"pa{pa}_p{k}")
            else:
                swing = cross is not None and tb.decides_to_swing(p, zone, rng)
                clip, ball = None, None
                if args.dry_run:  # the rest of TableBatter.swing's draws
                    in_zone = zone.contains(p.plate_y, p.plate_z)
                    kind = "take" if not swing else (
                        "miss" if rng.random() >= tb.hit_rate * (1.0 if in_zone else tb.out_zone_contact_scale) else "contact")
                    if kind == "contact":
                        ball = BattedBall(*rng.choice(tb.rows))
                elif cross is None:
                    kind = "take"  # the ball never reached the plate: no batter clip
                else:
                    clip, r = batter_clip(rel, not swing, f"pa{pa}_p{k}")
                    ev["isaac"] = r
                    if not swing:
                        kind = "take"
                    elif r["outcome"] in ("FAIR", "FOUL"):
                        b = r["ball"]
                        kind, ball = "contact", BattedBall(b["exit_velo_mps"], b["launch_deg"], b["spray_deg"], b["backspin_rpm"])
                    else:
                        kind = "miss"
                balls, strikes, res, extra = judge_pitch(balls, strikes, p, kind, ball, zone, fld, fs, grass, phys)
            ev.update(kind=kind, after=f"{balls}-{strikes}", end=extra.get("end"),
                      land=extra.get("land"), ball=ball.__dict__ if ball else None)
            log.append(ev)
            print(f"[match] PA {pa + 1} pitch {k + 1}: {ev['kmh']:.0f} km/h y {p.plate_y:+.2f} z {p.plate_z:.2f} "
                  f"{kind} -> {balls}-{strikes}" + (f" {res[1]}" if res else ""), flush=True)
            head = [f"PA {pa + 1}   pitch {k + 1}   count {ev['balls']}-{ev['strikes']}   {ev['kmh']:.0f} km/h"]
            for fr in frames_of(clips[k] if k < len(clips) else None, 2 * FPS, f"pitcher clip {k + 1}"):
                w.append_data(banner(fr, head))
            tail = head + [f"{kind.replace('_', ' ')}  ->  " + (res[1].replace("_", " ") if res else f"{balls}-{strikes}")]
            for fr in frames_of(clip, 2 * FPS, f"batter: {kind}"):
                w.append_data(banner(fr, tail))
            k += 1
        for fr in card([f"PA {pa + 1}: {res[0]} wins ({res[1].replace('_', ' ')})"]):
            w.append_data(fr)
        log.append(dict(pa=pa, winner=res[0], how=res[1]))
    for fr in card(["This video: " + ", ".join(f"PA {e['pa'] + 1} {e['winner']}" for e in log if "winner" in e),
                    args.end_card], 5.0):
        w.append_data(fr)
    w.close()
    with open(out.replace(".mp4", ".json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(f"[match] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
