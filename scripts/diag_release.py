"""Physics sanity check of the release: ball speed vs. hand / fingertip speeds around release.

A real release cannot make the ball faster than the fingertips pushing it (ball speed ~ fingertip speed).
If the ball is much faster, the contact solver is ejecting it (squeeze/penetration artefact).

    python scripts/diag_release.py --headless --checkpoint <model.pt> [--dt 0.0025]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--dt", type=float, default=1 / 400)
parser.add_argument("--squeeze", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402
from aibaseball.tasks.pitching.pitch_env_cfg import PitchPlayEnvCfg  # noqa: E402

TIPS = ["r_index_distal", "r_middle_distal", "r_ring_middle", "r_thumb_distal"]


def main():
    cfg = PitchPlayEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.dt = args.dt
    if args.squeeze is not None:
        cfg.grip_squeeze = args.squeeze
    env = gym.make("AIB-Pitch-Play-v0", cfg=cfg)
    w = RslRlVecEnvWrapper(env)
    agent = PitchPPORunnerCfg()
    runner = OnPolicyRunner(w, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    rob = base.robot
    tip_ids = [rob.find_bodies(n)[0][0] for n in TIPS]
    hand = base.hand_id
    from aibaseball.robot.ball_grip import TIPS as GRIP_TIPS
    from isaaclab.utils.math import quat_apply
    globals()["quat_apply"] = quat_apply
    globals()["PADS"] = torch.tensor([GRIP_TIPS[k][1] for k in ("index", "middle", "ring")], dtype=torch.float32,
                                     device=base.device)
    # wrap the physics step to log every sub-step
    hist = []
    orig = base._apply_action

    def logged():
        orig()
        d = rob.data
        # pad (contact point) velocity = link velocity + omega x (R r_pad); index/middle/ring pads from the grip model
        q = d.body_link_quat_w[0, tip_ids[:3]]
        r = quat_apply(q, PADS)
        v_pad = d.body_link_lin_vel_w[0, tip_ids[:3]] + torch.cross(d.body_link_ang_vel_w[0, tip_ids[:3]], r, dim=-1)
        tips = torch.cat([v_pad.norm(dim=-1), d.body_link_lin_vel_w[0, tip_ids[3:]].norm(dim=-1)])
        hist.append(dict(t=float(base._ref_time()[0]), ball=float(base.ball.data.root_lin_vel_w[0].norm()),
                         hand=float(d.body_link_lin_vel_w[0, hand].norm()), tip_max=float(tips.max()),
                         gap=float((base.ball.data.root_pos_w[0] - base._grip_point()[0]).norm()),
                         released=bool(base.released[0])))

    base._apply_action = logged
    with torch.inference_mode():
        obs, _ = w.reset()
        for _ in range(int(1.6 / base.step_dt)):
            obs, _, _, _ = w.step(policy(obs))
            if base.released[0] and len(hist) > 5 and hist[-1]["gap"] > 0.5:
                break
    i_rel = next(i for i, h in enumerate(hist) if h["gap"] > 0.02)
    print("[rel]   t(s)  gap(mm)  ball(m/s) hand(m/s) fingertip_max(m/s)", flush=True)
    for h in hist[max(0, i_rel - 12): i_rel + 8]:
        print(f"[rel] {h['t']:6.3f} {h['gap']*1000:7.1f} {h['ball']:9.1f} {h['hand']:9.1f} {h['tip_max']:12.1f}", flush=True)
    peak_tip = max(h["tip_max"] for h in hist[: i_rel + 1])
    peak_hand = max(h["hand"] for h in hist[: i_rel + 1])
    ball_out = max(h["ball"] for h in hist[i_rel: i_rel + 8])
    print(f"[rel] peak hand {peak_hand:.1f} m/s, peak fingertip {peak_tip:.1f} m/s (before release); ball after "
          f"release {ball_out:.1f} m/s ({ball_out*3.6:.0f} km/h); ratio ball/fingertip = {ball_out/peak_tip:.2f}",
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
