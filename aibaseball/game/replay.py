"""Top-view replay of a game's event log (one-month plan, W4 fallback / engine check): field, ball (size = height),
fielders, runners, strike-zone inset and a scoreboard, rendered with matplotlib into an mp4.

The final W4 video renders the robots in Isaac Sim from the same log; this one needs only the CPU.
"""

from __future__ import annotations

import math

import numpy as np

from .agents import StrikeZone
from .field import ALIGNMENT, BASE_PATH, BASES, FOUL_ANGLE_DEG, FieldSpec
from .fielding import FieldingSpec

PITCH_S = 0.6  # screen time per pitch not put in play
AFTER_PLAY_S = 0.8  # pause after a ball in play


def _base_path_point(d: float) -> tuple[float, float]:
    """Point d metres along the base path from home (home -> 1B -> 2B -> 3B -> home)."""
    d = min(max(d, 0.0), 4 * BASE_PATH)
    k = min(int(d // BASE_PATH), 3)
    f = (d - k * BASE_PATH) / BASE_PATH
    (x0, y0), (x1, y1) = BASES[k], BASES[(k + 1) % 4]
    return x0 + f * (x1 - x0), y0 + f * (y1 - y0)


def _covered(fs: FieldingSpec, pos: str, t: float) -> float:
    """Distance a fielder has run t seconds after contact (reaction, then accelerate to top speed)."""
    t = t - fs.reaction_s
    if t <= 0.0:
        return 0.0
    v, a = fs.speed_mps[pos], fs.accel_mps2
    return 0.5 * a * t * t if t <= v / a else v * v / (2.0 * a) + v * (t - v / a)


def _ball_at(path: list, t: float):
    if t >= path[-1][0]:
        return path[-1][1:4]
    i = int(np.searchsorted([s[0] for s in path], t))
    if i == 0:
        return path[0][1:4]
    (t0, *p0), (t1, *p1) = path[i - 1][:4], path[i][:4]
    f = (t - t0) / max(t1 - t0, 1e-9)
    return tuple(a + f * (b - a) for a, b in zip(p0, p1))


def play_frames(ev: dict, fps: int, fs: FieldingSpec):
    """Per frame: (ball (x, y, z) or None, fielder positions dict, runner positions list, throw segment or None)."""
    play = ev["play"]
    if play is None:
        return [None] * max(1, round(PITCH_S * fps))
    end = play.path[-1][0] if play.t_fielded is None else play.t_fielded + 1.5
    end = min(end, 9.0)
    target = play.at[:2] if play.at else None
    starts = [i + 1 for i, b in enumerate(ev["bases"]) if b]
    dests = [d for d in play.runner_to if d is not None]
    frames = []
    for k in range(round((end + AFTER_PLAY_S) * fps)):
        t = k / fps
        ball = _ball_at(play.path, t) if (play.t_fielded is None or t <= play.t_fielded) else None
        throw = None
        fielders = dict(ALIGNMENT)
        if play.fielder and target:
            fx, fy = ALIGNMENT[play.fielder]
            d = math.hypot(target[0] - fx, target[1] - fy)
            s = min(1.0, _covered(fs, play.fielder, t) / d) if d > 1e-6 else 1.0
            fielders[play.fielder] = (fx + s * (target[0] - fx), fy + s * (target[1] - fy))
            if play.kind == "ground_out" and t > play.t_fielded + fs.transfer_s:
                b1 = BASES[1]
                f = min(1.0, (t - play.t_fielded - fs.transfer_s) * fs.throw_mps / math.hypot(b1[0] - target[0], b1[1] - target[1]))
                throw = (target[0] + f * (b1[0] - target[0]), target[1] + f * (b1[1] - target[1]))
        runners = []
        run = max(0.0, t - fs.batter_start_s) * fs.runner_mps
        batter_to = play.batter_to if play.batter_to else (1 if play.kind == "ground_out" else 0)
        if batter_to:
            runners.append(_base_path_point(min(run, batter_to * BASE_PATH)))
        for b0, b1 in zip(starts, dests):
            runners.append(_base_path_point(min(b0 * BASE_PATH + max(0.0, t - 0.3) * fs.runner_mps, b1 * BASE_PATH)))
        frames.append((ball, fielders, runners, throw))
    return frames


def render(log: list, out_path: str, fps: int = 15, field: FieldSpec | None = None, zone: StrikeZone | None = None,
           fs: FieldingSpec | None = None, max_events: int | None = None) -> int:
    """Write the replay of `log` (Game.log) to out_path (mp4). Returns the number of frames."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field, zone, fs = field or FieldSpec(), zone or StrikeZone(), fs or FieldingSpec()
    fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 0.78, 1.0])
    ax.set_xlim(-76, 76)
    ax.set_ylim(-6, 126)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#1d5c2e")
    # field: fence, foul lines, infield, bases (screen: +X field = up, +Y field (third-base side) = left)
    a = np.linspace(-FOUL_ANGLE_DEG, FOUL_ANGLE_DEG, 91)
    r = [field.fence_distance(v) for v in a]
    ax.plot(-np.array(r) * np.sin(np.radians(a)), np.array(r) * np.cos(np.radians(a)), color="#d8c9a3", lw=3)
    for s in (-1, 1):
        L = field.fence_distance(FOUL_ANGLE_DEG)
        ax.plot([0, -s * L * math.sin(math.radians(FOUL_ANGLE_DEG))], [0, L * math.cos(math.radians(FOUL_ANGLE_DEG))], color="white", lw=1)
    ax.fill([-b[1] for b in BASES], [b[0] for b in BASES], color="#a07d4f", alpha=0.6)
    ax.scatter([-b[1] for b in BASES], [b[0] for b in BASES], marker="s", color="white", s=30, zorder=3)
    fielder_dots = ax.scatter([], [], s=60, color="#2b6cd6", edgecolor="white", zorder=4)
    runner_dots = ax.scatter([], [], s=60, color="#e04a3a", edgecolor="white", zorder=4)
    ball_dot = ax.scatter([], [], s=20, color="white", edgecolor="black", zorder=5)
    throw_dot = ax.scatter([], [], s=20, color="yellow", edgecolor="black", zorder=5)
    # strike-zone inset and scoreboard
    zx = fig.add_axes([0.80, 0.55, 0.18, 0.30])
    zx.set_xlim(-0.6, 0.6)
    zx.set_ylim(0.0, 1.6)
    zx.set_aspect("equal")
    zx.set_xticks([])
    zx.set_yticks([])
    zx.add_patch(plt.Rectangle((-zone.half_width_m, zone.bottom_m), 2 * zone.half_width_m, zone.top_m - zone.bottom_m, fill=False, color="black"))
    pitch_dot = zx.scatter([], [], s=40, color="#e04a3a")
    board = fig.text(0.80, 0.50, "", va="top", ha="left", family="monospace", fontsize=11, color="white")

    events = log if max_events is None else log[:max_events]
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    n = 0
    for ev in events:
        p = ev["pitch"]
        pitch_dot.set_offsets([[-p.plate_y, p.plate_z]])  # catcher's view: third-base side on the left
        bases = "".join(b if on else "-" for b, on in zip("123", ev["bases"]))
        half = "Top" if ev["top"] else "Bot"
        score = ev.get("score", ("", ""))
        res = ev["result"]
        if ev["play"] is not None and ev["play"].kind not in ("foul",):
            res = f"{res} ({ev['play'].fielder or '-'})"
        board.set_text(f"{half} {ev['inning']}  Out {ev['outs']}\nB {ev['balls']}  S {ev['strikes']}  [{bases}]\n"
                       f"Away {score[0]}  Home {score[1]}\n{p.speed_mps * 3.6:.0f} km/h  {ev['swing']}\n{res}")
        for fr in play_frames(ev, fps, fs):
            if fr is None:
                fielders, runners, ball, throw = ALIGNMENT, [], None, None
                runners = [_base_path_point(i * BASE_PATH) for i, on in enumerate(ev["bases"], 1) if on]
            else:
                ball, fielders, runners, throw = fr
            fielder_dots.set_offsets([[-y, x] for x, y in fielders.values()])
            runner_dots.set_offsets([[-y, x] for x, y in runners] or np.empty((0, 2)))
            if ball is None:
                ball_dot.set_offsets(np.empty((0, 2)))
            else:
                ball_dot.set_offsets([[-ball[1], ball[0]]])
                ball_dot.set_sizes([20 + 8 * ball[2]])
            throw_dot.set_offsets([[-throw[1], throw[0]]] if throw else np.empty((0, 2)))
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            n += 1
    writer.close()
    plt.close(fig)
    return n
