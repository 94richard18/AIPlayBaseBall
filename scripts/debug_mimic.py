"""Kinematic check: put the robot exactly in the reference state around contact and measure, in the sim,
the sweet-spot-to-ball distance and the key-body errors (validates frames/conventions of the reference).

    python scripts/debug_mimic.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.tee_batting.mimic_env_cfg import TeeBattingMimicPlayEnvCfg  # noqa: E402


def main():
    cfg = TeeBattingMimicPlayEnvCfg()
    cfg.scene.num_envs = 8
    env = gym.make("AIB-TeeBatting-Mimic-Play-v0", cfg=cfg).unwrapped
    env.reset()
    ref = env.ref
    ids = torch.arange(env.num_envs, device=env.device)
    offsets = torch.linspace(-0.04, 0.02, env.num_envs, device=env.device)
    # place each env at a time around contact (kinematic: write state, forward, read)
    env.cfg.rsi_prob = 0.0
    env._reset_idx(ids)
    t = ref.contact_time + offsets
    env.t0[:] = t
    r = ref.sample(t)
    origins = env.scene.env_origins
    root = torch.cat([r["root_pos"] + origins, r["root_quat"], torch.zeros_like(r["root_lin_vel"]),
                      torch.zeros_like(r["root_ang_vel"])], -1)
    env.robot.write_root_state_to_sim(root)
    jp = env.robot.data.default_joint_pos.clone()
    jp[:, env.body_joint_ids] = r["joint_pos"][:, env.ref_cols]
    env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)
    knob, axis, cm, v_cm, w, _ = env._bat_state()
    ss, _ = env._sweet_spot(knob, axis, cm, v_cm, w)
    key = torch.cat([env.robot.data.body_link_pos_w[:, env.key_body_ids], (knob + axis * env.bat_geom.length).unsqueeze(1)], 1)
    ref_key = r["key_pos"] + origins.unsqueeze(1)
    print("[debug] tee (rel):", (env.tee_pos[0] - origins[0]).tolist(), flush=True)
    for i in range(env.num_envs):
        err = (key[i] - ref_key[i]).norm(dim=-1)
        print(f"[debug] t-contact {offsets[i]*1000:+5.0f} ms | sweet spot -> ball {((ss[i]-env.tee_pos[i]).norm()*100):5.1f} cm | "
              f"key err (cm) " + " ".join(f"{e*100:4.1f}" for e in err.tolist()), flush=True)
    print("[debug] key order:", ["l_hand", "r_hand", "l_foot", "r_foot", "torso", "bat", "bat_tip"], flush=True)
    print("[debug] sim bat axis:", axis[-1].tolist(), " ref bat dir:",
          torch.nn.functional.normalize(ref_key[-1, -1] - ref_key[-1, -2], dim=-1).tolist(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
