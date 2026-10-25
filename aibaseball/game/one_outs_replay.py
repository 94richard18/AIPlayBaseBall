"""ONE OUTS match replay (CPU, matplotlib -> mp4): side view of each pitch to the wall, the wall's painted strike zone
from the catcher's side, a top view of the batted ball and its first bounce against the 95 ft infield circle, and a
scoreboard (match, count, speed, swing, result, win tally).

The deliverable (11-26) puts the robots' Isaac clips next to this; this part needs only the engine's match logs.
"""

from __future__ import annotations

import math

import numpy as np

from ..physics import PhysicsSpec
from ..physics.grass import GrassSpec
from .agents import StrikeZone
from .field import FOUL_ANGLE_DEG, MOUND_DIST, FieldSpec
from .fielding import FieldingSpec, ball_path
from .one_outs import INFIELD_RADIUS_M

RELEASE = (MOUND_DIST - 1.7, 1.86)  # (x, z): the deliverable pitcher's release (1.86 m high, ~1.7 m stride)
SLOW = 2.0  # the pitch is shown at half speed
HOLD_S = 0.6  # pause after each pitch
TITLE_S = 1.2  # match title card / final result card
MAX_HIT_S = 6.0
HOW_ZH = {"walk": "四壞保送", "hit_by_pitch": "觸身球", "outfield_hit": "外野安打", "home_run": "全壘打",
          "strikeout": "三振", "infield_ball": "內野球（出局）"}
SWING_ZH = {"take": "不揮", "miss": "揮空", "contact": "擊中", "hit_by_pitch": "觸身"}


def _pitch_path(p, n: int):
    """n points (x, y, z) from the release to the plate, drag-free, ending at (plate_y, plate_z)."""
    x0, z0 = RELEASE
    T = x0 / max(p.speed_mps * 0.93, 1.0)  # mean speed ~7% under the release speed (drag)
    vz = (p.plate_z - z0 + 0.5 * 9.81 * T * T) / T
    t = np.linspace(0.0, T, n)
    return np.stack([x0 * (1 - t / T), p.plate_y * t / T, z0 + vz * t - 0.5 * 9.81 * t * t], 1), T


def render_matches(matches: list, out_path: str, fps: int = 30, zone: StrikeZone | None = None,
                   field: FieldSpec | None = None) -> int:
    """matches: list of OneOutsResult. Writes out_path, returns the frame count."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    zone, field = zone or StrikeZone(), field or FieldSpec()
    fs, grass, phys = FieldingSpec(), GrassSpec(), PhysicsSpec()

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#16351f")
    # side view: pitcher on the left, the wall on the right
    side = fig.add_axes([0.03, 0.56, 0.62, 0.40])
    side.set_xlim(-0.5, MOUND_DIST + 1.5)
    side.set_ylim(0.0, 2.6)
    side.set_facecolor("#24502f")
    side.set_xticks([0, MOUND_DIST])
    side.set_xticklabels(["投手板", "本壘 / 牆"], color="white")
    side.set_yticks([0.5, 1.0, 1.5, 2.0])
    side.tick_params(colors="white", labelsize=8)
    side.add_patch(plt.Rectangle((MOUND_DIST + 0.3, 0.0), 0.25, 2.6, color="#7a6a55"))
    side.add_patch(plt.Rectangle((MOUND_DIST + 0.25, zone.bottom_m), 0.05, zone.top_m - zone.bottom_m, color="white"))
    side.set_title("側面：投球到牆（慢動作 1/2）", color="white", fontsize=10)
    (side_trail,) = side.plot([], [], color="#ffd966", lw=1.5)
    side_ball = side.scatter([], [], s=40, color="white", edgecolor="black", zorder=5)
    # the wall from the catcher's side (third-base side on the left)
    wall = fig.add_axes([0.69, 0.52, 0.28, 0.44])
    wall.set_xlim(-1.0, 1.0)
    wall.set_ylim(0.0, 1.8)
    wall.set_aspect("equal")
    wall.set_facecolor("#7a6a55")
    wall.set_xticks([])
    wall.set_yticks([])
    wall.add_patch(plt.Rectangle((-zone.half_width_m, zone.bottom_m), 2 * zone.half_width_m, zone.top_m - zone.bottom_m,
                                 fill=False, color="white", lw=2))
    wall.set_title("牆上好球帶（捕手視角）", color="white", fontsize=10)
    old_dots = wall.scatter([], [], s=40)
    cur_dot = wall.scatter([], [], s=90, color="#ff3b30", edgecolor="white", zorder=5)
    # top view of the batted ball
    top = fig.add_axes([0.03, 0.04, 0.42, 0.46])
    top.set_xlim(-80, 80)
    top.set_ylim(-8, 125)
    top.set_aspect("equal")
    top.axis("off")
    a = np.linspace(-FOUL_ANGLE_DEG, FOUL_ANGLE_DEG, 91)
    r = np.array([field.fence_distance(v) for v in a])
    top.plot(-r * np.sin(np.radians(a)), r * np.cos(np.radians(a)), color="#d8c9a3", lw=2)
    for s in (-1, 1):
        L = field.fence_distance(FOUL_ANGLE_DEG)
        top.plot([0, -s * L * math.sin(math.radians(FOUL_ANGLE_DEG))], [0, L * math.cos(math.radians(FOUL_ANGLE_DEG))],
                 color="white", lw=1)
    c = np.linspace(0, 2 * math.pi, 120)
    top.plot(INFIELD_RADIUS_M * np.sin(c), MOUND_DIST + INFIELD_RADIUS_M * np.cos(c), color="#e0b070", lw=1.2, ls="--")
    top.text(0, MOUND_DIST + INFIELD_RADIUS_M + 2, "內野 95 ft（落在裡面 = 投手勝）", color="#e0b070", ha="center", fontsize=8)
    (top_trail,) = top.plot([], [], color="#ffd966", lw=1.5)
    top_ball = top.scatter([], [], s=30, color="white", edgecolor="black", zorder=5)
    land_mark = top.scatter([], [], s=120, marker="x", color="#ff3b30", zorder=6)
    board = fig.text(0.50, 0.46, "", va="top", ha="left", fontsize=15, color="white", linespacing=1.6)
    banner = fig.text(0.73, 0.18, "", ha="center", va="center", fontsize=40, color="white", weight="bold",
                      bbox=dict(facecolor="#000000", alpha=0.0, edgecolor="none"))

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    n = 0

    def emit(k=1):
        nonlocal n
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        for _ in range(k):
            writer.append_data(img)
            n += 1

    def clear():
        for art in (side_trail, top_trail):
            art.set_data([], [])
        for art in (side_ball, top_ball, land_mark, cur_dot):
            art.set_offsets(np.empty((0, 2)))

    wins = {"batter": 0, "pitcher": 0}
    for mi, m in enumerate(matches, 1):
        clear()
        old_dots.set_offsets(np.empty((0, 2)))
        board.set_text("")
        banner.set_text(f"第 {mi} 場")
        banner.get_bbox_patch().set_alpha(0.6)
        emit(round(TITLE_S * fps))
        banner.set_text("")
        banner.get_bbox_patch().set_alpha(0.0)
        old_xy, old_c = [], []
        for ev in m.pitches:
            clear()
            p = ev["pitch"]
            count = f"{ev['balls']} 壞 {ev['strikes']} 好"
            head = f"第 {mi} / {len(matches)} 場　打者 {wins['batter']} : {wins['pitcher']} 投手\n{count}　{p.speed_mps * 3.6:.0f} km/h"
            board.set_text(head)
            _, T = _pitch_path(p, 2)
            k = max(2, round(T * SLOW * fps))
            path, T = _pitch_path(p, k)
            for i in range(k):
                u = MOUND_DIST - path[: i + 1, 0]
                side_trail.set_data(u, path[: i + 1, 2])
                side_ball.set_offsets([[u[-1], path[i, 2]]])
                emit()
            in_zone = zone.contains(p.plate_y, p.plate_z)
            cur_dot.set_offsets([[-p.plate_y, p.plate_z]])
            sw = ev["swing"]
            line = f"{head}\n{'好球帶內' if in_zone else '好球帶外'}　{SWING_ZH.get(sw, sw)}"
            if sw == "contact":
                hp, _ = ball_path(ev["ball"], field, grass, fs, phys)
                hp = [s for s in hp if s[0] <= MAX_HIT_S]
                fair = ev.get("land") is not None and field.is_fair(*ev["land"])
                dist = math.hypot(*ev["land"]) if ev.get("land") else float("nan")
                tag = "全壘打" if ev.get("end") == "home_run" else ("界內" if fair else "界外")
                board.set_text(f"{line}\n初速 {ev['ball'].exit_speed_mps * 3.6:.0f} km/h　仰角 {ev['ball'].launch_deg:+.0f}°\n"
                               f"{tag}　第一落點 {dist:.0f} m")
                t0 = 0.0
                xs, ys = [], []
                for s in hp:
                    xs.append(-s[2])
                    ys.append(s[1])
                    if s[0] - t0 >= 1.0 / fps or s is hp[-1]:
                        t0 = s[0]
                        top_trail.set_data(xs, ys)
                        top_ball.set_offsets([[xs[-1], ys[-1]]])
                        emit()
                if ev.get("land"):
                    land_mark.set_offsets([[-ev["land"][1], ev["land"][0]]])
            else:
                board.set_text(line)
            emit(round(HOLD_S * fps))
            old_xy.append([-p.plate_y, p.plate_z])
            old_c.append("#ffffff" if in_zone else "#9bb7d4")
            old_dots.set_offsets(old_xy)
            old_dots.set_color(old_c)
        wins[m.winner] += 1
        banner.set_text(f"{'打者勝' if m.winner == 'batter' else '投手勝'}：{HOW_ZH.get(m.how, m.how)}")
        banner.get_bbox_patch().set_alpha(0.6)
        board.set_text(f"第 {mi} / {len(matches)} 場結束\n打者 {wins['batter']} : {wins['pitcher']} 投手")
        emit(round(1.8 * TITLE_S * fps))
        banner.set_text("")
        banner.get_bbox_patch().set_alpha(0.0)
    writer.close()
    plt.close(fig)
    return n
