"""Hold the batting stance under PD control and check stability, loop-grip gap and sim speed.

    python scripts/test_stance.py --headless --num_envs 1024
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=2.0)
parser.add_argument("--dt", type=float, default=1 / 400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from aibaseball.robot.aib1_cfg import make_aib1_cfg  # noqa: E402
from aibaseball.robot.stance import Stance  # noqa: E402
from aibaseball.robot.aib1_cfg import STANCE_PATH  # noqa: E402


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9)
        ),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot: ArticulationCfg = make_aib1_cfg()


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=3.0))
    sim.reset()
    robot = scene["robot"]
    print(f"[test] bodies={robot.num_bodies} joints={robot.num_joints}", flush=True)
    stance = Stance.load(STANCE_PATH)
    hand_id = robot.find_bodies("r_hand")[0][0]
    bat_id = robot.find_bodies("bat")[0][0]
    pelvis_id = robot.find_bodies("pelvis")[0][0]
    top_local = torch.tensor(stance.top_hand_local, device=sim.device).expand(args.num_envs, 3)
    bat_local = torch.tensor(stance.bat_top_local, device=sim.device).expand(args.num_envs, 3)

    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.set_joint_position_target(robot.data.default_joint_pos)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)
    steps = int(args.seconds / args.dt)
    t0 = time.time()
    for i in range(steps):
        scene.write_data_to_sim()
        sim.step(render=not args.headless)
        scene.update(args.dt)
        if i % int(0.25 / args.dt) == 0 or i == steps - 1:
            pos = robot.data.body_link_pos_w
            quat = robot.data.body_link_quat_w
            p_hand = pos[:, hand_id] + quat_apply(quat[:, hand_id], top_local)
            p_bat = pos[:, bat_id] + quat_apply(quat[:, bat_id], bat_local)
            gap = (p_hand - p_bat).norm(dim=-1)
            h = pos[:, pelvis_id, 2]
            qerr = (robot.data.joint_pos - robot.data.default_joint_pos).abs().max(dim=-1).values
            print(f"[test] t={i*args.dt:5.2f}s pelvis z {h.mean():.3f} (min {h.min():.3f}) "
                  f"grip gap {gap.mean()*1000:.1f} mm (max {gap.max()*1000:.1f}) "
                  f"max|q-q0| {qerr.max():.3f} max|qd| {robot.data.joint_vel.abs().max():.2f}", flush=True)
    wall = time.time() - t0
    print(f"[test] {steps} steps x {args.num_envs} envs in {wall:.1f}s -> "
          f"{steps*args.num_envs/wall:,.0f} env-physics-steps/s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
