"""Build the pitcher variant of AIB-1 (no bat): URDF + four-seam grip + USD.

    python scripts/build_pitcher.py --headless
Outputs to assets/aib1_pitcher/: aib1_pitcher.urdf, aib1_pitcher.usd, grip.json, default_pose.json
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from aibaseball.robot.ball_grip import solve_grip  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid, summary  # noqa: E402
from aibaseball.robot.urdf_writer import write_urdf  # noqa: E402

OUT = os.path.join(ROOT, "assets", "aib1_pitcher")


def main():
    robot = build_humanoid(with_bat=False, name="aib1_pitcher", arm_masses="human")  # human arm anthropometrics
    print("[pitcher]", summary(robot), flush=True)
    urdf = write_urdf(robot, OUT)
    grip = solve_grip(robot)
    grip.save(os.path.join(OUT, "grip.json"))
    print(f"[pitcher] grip tip gaps (mm) {grip.tip_gaps_mm}, palm gap {grip.palm_gap_mm:.1f} mm", flush=True)
    # default pose: standing, throwing hand in the grip, glove hand relaxed
    pose = {j.name: 0.0 for j in robot.actuated}
    pose.update(grip.joint_pos)
    for j in robot.actuated:
        pose[j.name] = min(max(pose[j.name], j.lower + 1e-3), j.upper - 1e-3)
    with open(os.path.join(OUT, "default_pose.json"), "w", encoding="utf-8") as f:
        json.dump(dict(root_height=0.03 + 0.45 + 0.45 + 0.08, joint_pos=pose), f, indent=2)
    cfg = UrdfConverterCfg(
        asset_path=urdf, usd_dir=OUT, usd_file_name="aib1_pitcher.usd", fix_base=False, merge_fixed_joints=False,
        self_collision=False, replace_cylinders_with_capsules=True, force_usd_conversion=True, make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0), target_type="position"),
    )
    print("[pitcher] USD:", UrdfConverter(cfg).usd_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
