"""Convert assets/aib1/aib1.urdf to USD and close the two-hand grip with a loop joint.

The bat is welded to the bottom (left) hand inside the articulation. The top (right) hand is
attached to the handle by a spherical joint that is excluded from the articulation tree
(PhysX loop closure), so both arms transmit force to the bat like a real two-hand grip.

Usage:
    python scripts/convert_robot_usd.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset_dir", type=str, default="assets/aib1", help="e.g. assets/aib1_human")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pxr import Gf, Usd, UsdPhysics  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from aibaseball.robot.stance import Stance  # noqa: E402

ASSET_DIR = os.path.join(ROOT, args.asset_dir)


def find_prim(stage, name):
    for prim in stage.Traverse():
        if prim.GetName() == name and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    raise KeyError(name)


def main():
    cfg = UrdfConverterCfg(
        asset_path=os.path.join(ASSET_DIR, "aib1.urdf"),
        usd_dir=ASSET_DIR,
        usd_file_name="aib1.usd",
        fix_base=False,
        merge_fixed_joints=False,  # keep the bat as its own rigid body
        self_collision=False,
        replace_cylinders_with_capsules=True,
        force_usd_conversion=True,
        make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
            target_type="position",
        ),
    )
    usd_path = UrdfConverter(cfg).usd_path
    print("USD:", usd_path)

    stance = Stance.load(os.path.join(ASSET_DIR, "batting_stance.json"))
    stage = Usd.Stage.Open(usd_path)
    hand = find_prim(stage, "r_hand")
    bat = find_prim(stage, "bat")
    root = stage.GetDefaultPrim()
    joint_path = root.GetPath().AppendChild("top_hand_grip")
    joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([hand.GetPath()])
    joint.CreateBody1Rel().SetTargets([bat.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*stance.top_hand_local))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*stance.bat_top_local))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateExcludeFromArticulationAttr().Set(True)
    stage.GetRootLayer().Save()

    n_bodies = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    n_joints = sum(1 for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint))
    print(f"rigid bodies: {n_bodies}, revolute joints: {n_joints}")
    print("hand prim:", hand.GetPath(), " bat prim:", bat.GetPath(), " loop joint:", joint_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
    finally:
        sys.stdout.flush()
        app.close()
