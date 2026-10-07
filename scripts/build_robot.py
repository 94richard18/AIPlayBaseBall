"""Generate the AIB-1 humanoid URDF, solve the batting stance, and render a preview image.

Usage (any python with numpy/scipy/matplotlib, Isaac Sim not required):
    python scripts/build_robot.py [--arm_masses human]
Outputs to assets/aib1/ (human arm masses: assets/aib1_human/): aib1.urdf, meshes/bat.obj, batting_stance.json, preview.png
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aibaseball.physics.bat_model import BatSpec  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid, summary  # noqa: E402
from aibaseball.robot.stance import solve_stance  # noqa: E402
from aibaseball.robot.urdf_writer import write_urdf  # noqa: E402

OUT = os.path.join(ROOT, "assets", "aib1")


def _shape_lines(shape, R, p):
    """Line segments approximating a shape in world coordinates."""
    Rs, ps = R @ shape.rot, p + R @ shape.pos
    if shape.kind == "cylinder":
        h = shape.size[1] / 2
        return [(ps - Rs[:, 2] * h, ps + Rs[:, 2] * h, shape.size[0])]
    if shape.kind == "box":
        sx, sy, sz = np.array(shape.size) / 2
        c = [np.array([a, b, d]) for a in (-sx, sx) for b in (-sy, sy) for d in (-sz, sz)]
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
        return [(ps + Rs @ c[i], ps + Rs @ c[j], 0.004) for i, j in edges]
    if shape.kind == "sphere":
        return [(ps, ps, shape.size[0])]
    return []


def render_preview(robot, q, root_pos, path, bat=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = robot.fk(q, root_pos=root_pos)
    fig = plt.figure(figsize=(15, 6))
    views = [("front (from the plate)", 0, 0), ("side (from the pitcher)", 0, 90), ("3/4", 20, 40)]
    for k, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for name, link in robot.links.items():
            R, p = P[name]
            for s in link.shapes:
                for a, b, r in _shape_lines(s, R, p):
                    col = "saddlebrown" if name == "bat" else ("tab:blue" if name.startswith("l_") else
                                                               "tab:red" if name.startswith("r_") else "gray")
                    if np.allclose(a, b):
                        ax.scatter(*a, s=(r * 900) ** 2 / 10, c=col, alpha=0.4)
                    else:
                        ax.plot(*zip(a, b), c=col, lw=max(1.0, r * 120), alpha=0.8)
        if bat is not None:
            Rb, pb = P["bat"]
            ax.scatter(*(pb + Rb[:, 0] * bat.sweet_spot_x), c="gold", s=40)
        ax.set_box_aspect((1, 1, 2))
        ax.set_xlim(-0.5, 0.5), ax.set_ylim(-0.5, 0.5), ax.set_zlim(0, 2.0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def render_hand(robot, q, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = robot.fk(q)
    fig = plt.figure(figsize=(10, 5))
    for k, side in enumerate(("r", "l")):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        R0, p0 = P[f"{side}_hand"]
        for name, link in robot.links.items():
            if not (name.startswith(side + "_") and any(t in name for t in ("hand", "index", "middle", "ring", "pinky", "thumb")))\
                    and name != "bat":
                continue
            R, p = P[name]
            for s in link.shapes:
                for a, b, r in _shape_lines(s, R, p):
                    a, b = R0.T @ (a - p0), R0.T @ (b - p0)
                    col = "saddlebrown" if name == "bat" else ("tab:orange" if "thumb" in name else "tab:gray")
                    ax.plot(*zip(a, b), c=col, lw=max(1.0, r * 250))
        ax.set_xlim(-0.12, 0.12), ax.set_ylim(-0.12, 0.12), ax.set_zlim(-0.22, 0.02)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{side} hand (grip pose, hand frame)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm_masses", choices=["robot", "human"], default="robot")
    out = OUT if parser.parse_args().arm_masses == "robot" else OUT + "_human"
    arm_masses = "robot" if out == OUT else "human"
    bat = BatSpec()
    robot = build_humanoid(with_bat=True, bat=bat, arm_masses=arm_masses)
    print(summary(robot))
    print("bat:", {k: round(v, 4) for k, v in bat.mass_properties().items()})
    urdf = write_urdf(robot, out, bat=bat)
    print("URDF ->", urdf)

    stance = solve_stance(robot, bat)
    stance.save(os.path.join(out, "batting_stance.json"))
    print(f"stance: cost {stance.residual:.4f}, top-hand gap {stance.info['top_hand_gap_m']*1000:.2f} mm, "
          f"limits hit: {stance.info['at_limit']}")

    root = np.array([0, 0, stance.root_height])
    render_preview(robot, stance.joint_pos, root, os.path.join(out, "preview.png"), bat)
    render_hand(robot, stance.joint_pos, os.path.join(out, "preview_hands.png"))
    print("preview ->", os.path.join(out, "preview.png"))


if __name__ == "__main__":
    main()
