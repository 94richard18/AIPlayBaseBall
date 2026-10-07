"""Open-loop replay of the reference (zero residual) in physics: can the PD + feed-forward swing hit the ball?

    python scripts/openloop_mimic.py --headless [--ff 1.0] [--speed 1.0]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ff", type=float, default=1.0)
parser.add_argument("--speed", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.tee_batting.mimic_env_cfg import TeeBattingMimicEnvCfg  # noqa: E402


def main():
    cfg = TeeBattingMimicEnvCfg()
    cfg.scene.num_envs = 4
    cfg.rsi_prob = 0.0
    cfg.vel_feedforward = args.ff
    cfg.speed = args.speed
    cfg.max_key_err = 10.0
    env = gym.make("AIB-TeeBatting-Mimic-v0", cfg=cfg).unwrapped
    env.reset()
    act = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    best = 9.0
    for k in range(int(env.ref.duration / env.speed / env.step_dt)):
        env.step(act)
        knob, axis, cm, v_cm, w, _ = env._bat_state()
        ss, ssv = env._sweet_spot(knob, axis, cm, v_cm, w)
        d = (ss - env.tee_pos).norm(dim=-1)[0].item()
        best = min(best, d) if not env.hit[0] else best
        t = env._ref_time()[0].item()
        if k % 10 == 0:
            r = env._ref(env._ref_time())
            dq = (env.robot.data.joint_pos[0, env.body_joint_ids] - r["joint_pos"][0])
            worst = dq.abs().topk(4)
            from isaaclab.utils.math import quat_error_magnitude
            print(f"[ol]   root pos err {(env.robot.data.root_link_pos_w[0]-r['root_pos'][0]).norm()*100:5.1f} cm, "
                  f"root rot err {quat_error_magnitude(env.robot.data.root_link_quat_w[:1], r['root_quat'][:1]).item():.2f} rad, "
                  f"worst joints " + ", ".join(f"{env.body_joint_names[i]} {dq[i]:+.2f}" for i in worst.indices.tolist())
                  + f" | done={bool(env.reset_buf[0])} fallen={bool(env.fallen[0])}", flush=True)
        if abs(t - env.ref.contact_time) < 0.06 or k % 20 == 0:
            r = env._ref(env._ref_time())
            tip = knob + axis * env.bat_geom.length
            print(f"[ol] t={t:5.3f}s ss->ball {d*100:6.1f} cm  ss speed {ssv[0].norm():5.1f} m/s  bat tip err "
                  f"{(tip[0]-r['key_pos'][0,-1]).norm()*100:5.1f} cm  hit {bool(env.hit[0])}  pelvis z "
                  f"{env.robot.data.root_link_pos_w[0,2]:.2f}", flush=True)
    print(f"[ol] min sweet-spot distance before contact {best*100:.1f} cm; hit={bool(env.hit[0])}; "
          f"launch speed {env.launch_vel[0].norm():.1f} m/s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
