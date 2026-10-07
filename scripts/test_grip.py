"""Physical grip test: can the AIB-1 fingers hold a baseball by contact friction through a fast arm motion,
and what happens when they open? Root fixed, so only the arm/hand matters.

    python scripts/test_grip.py --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=1 / 400)
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--squeeze", type=float, default=0.0, help="extra flexion (rad) beyond contact = grip force")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from aibaseball.physics.constants import BallSpec  # noqa: E402
from aibaseball.robot.aib1_cfg import PITCHER_GRIP, make_pitcher_cfg  # noqa: E402
from aibaseball.robot.ball_grip import BallGrip  # noqa: E402

BALL = BallSpec()


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = make_pitcher_cfg(fix_root=True)
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL.radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, max_linear_velocity=1000.0,
                                                         max_angular_velocity=10000.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=args.friction,
                                                            dynamic_friction=args.friction * 0.9, restitution=0.3),
        ),
    )


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=args.dt, device=args.device,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=args.friction, dynamic_friction=args.friction * 0.9)))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    robot, ball = scene["robot"], scene["ball"]
    grip = BallGrip.load(PITCHER_GRIP)
    hand = robot.find_bodies("r_hand")[0][0]
    c_local = torch.tensor(grip.ball_center_hand, device=sim.device).unsqueeze(0)
    q0 = robot.data.default_joint_pos.clone()
    sp = robot.find_joints("r_shoulder_pitch")[0][0]
    sy = robot.find_joints("r_shoulder_yaw")[0][0]
    el = robot.find_joints("r_elbow")[0][0]
    fingers = robot.find_joints("r_(index|middle|ring|pinky|thumb)_.*")[0]
    # arm out in front, elbow bent (a throwing-like position)
    q0[:, sp], q0[:, el] = -1.2, 1.2
    q_hold = q0.clone()
    for name in ("r_index_mcp", "r_index_pip", "r_middle_mcp", "r_middle_pip", "r_thumb_cmc_flex", "r_thumb_mcp",
                 "r_ring_mcp", "r_ring_pip"):
        q_hold[:, robot.find_joints(name)[0][0]] += args.squeeze
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    robot.set_joint_position_target(q0)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)

    def place_ball():
        p = robot.data.body_link_pos_w[:, hand] + quat_apply(robot.data.body_link_quat_w[:, hand], c_local)
        ball.write_root_state_to_sim(torch.cat([p, torch.tensor([[1.0, 0, 0, 0]], device=sim.device),
                                                torch.zeros(1, 6, device=sim.device)], -1))

    place_ball()

    def gap():
        p = robot.data.body_link_pos_w[:, hand] + quat_apply(robot.data.body_link_quat_w[:, hand], c_local)
        return (ball.data.root_pos_w - p).norm().item()

    t = 0.0
    phase_log = []
    steps = int(2.4 / args.dt)
    released = False
    for i in range(steps):
        tgt = q_hold.clone() if t > 0.05 else q0.clone()
        if t > 0.5:  # fast whip: shoulder internal rotation + elbow extension, ~1 s
            s = min(1.0, (t - 0.5) / 0.25)
            tgt[:, sy] = 1.5 * math.sin(math.pi * s - math.pi / 2)
            tgt[:, el] = 1.2 - 1.1 * s
            tgt[:, sp] = -1.2 - 0.6 * s
        if t > 0.74 and not released:  # open the fingers near the end of the whip
            released = True
        if released:
            tgt[:, fingers] = 0.0
        robot.set_joint_position_target(tgt)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(args.dt)
        t += args.dt
        if i % int(0.05 / args.dt) == 0 or (0.70 < t < 0.80):
            v = ball.data.root_lin_vel_w[0]
            w = ball.data.root_ang_vel_w[0]
            hv = robot.data.body_link_lin_vel_w[0, hand]
            print(f"[grip] t={t:4.2f}s ball-to-grip {gap()*1000:6.1f} mm | ball speed {v.norm():5.1f} m/s, "
                  f"hand speed {hv.norm():5.1f} m/s, spin {w.norm()*60/(2*math.pi):6.0f} rpm | released {released}",
                  flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
