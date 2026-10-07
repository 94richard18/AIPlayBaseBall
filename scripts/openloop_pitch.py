"""Open-loop reference pitch (zero residual, fingers squeezed in the grip): how long does the ball stay in the hand?

    python scripts/openloop_pitch.py --headless [--squeeze 0.3] [--fix_root]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--squeeze", type=float, default=0.3)
parser.add_argument("--speed", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.pitching.pitch_env_cfg import PitchEnvCfg  # noqa: E402


def main():
    cfg = PitchEnvCfg()
    cfg.scene.num_envs = 2
    cfg.rsi_prob = 0.0
    cfg.grip_squeeze = args.squeeze
    cfg.speed = args.speed
    cfg.max_key_err = 10.0
    cfg.min_release_speed = -1.0
    env = gym.make("AIB-Pitch-v0", cfg=cfg).unwrapped
    env.reset()
    act = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    for k in range(int(env.ref.duration / env.speed / env.step_dt)):
        env.step(act)
        t = env._ref_time()[0].item()
        gap = (env.ball.data.root_pos_w - env._grip_point()).norm(dim=-1)[0].item()
        hv = env.robot.data.body_link_lin_vel_w[0, env.hand_id].norm().item()
        bv = env.ball.data.root_lin_vel_w[0].norm().item()
        if k % 5 == 0 or env.new_release[0] or abs(t - env.release_ref) < 0.03:
            print(f"[ol] t={t:5.3f} (release ref {env.release_ref:.3f}) gap {gap*1000:6.1f} mm | hand {hv:5.1f} m/s "
                  f"ball {bv:5.1f} m/s | released {bool(env.released[0])} dropped {bool(env.dropped[0])} "
                  f"pelvis z {env.robot.data.root_link_pos_w[0,2]:.2f}", flush=True)
        if env.reset_buf[0]:
            print("[ol] episode ended", flush=True)
            break


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
