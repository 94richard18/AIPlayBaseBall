"""Record real release states of a pitching policy (start states for the follow-through balance skill).

    python scripts/collect_release_states.py --headless --checkpoint <pitch model.pt> [--motion <npz>]
Writes data/pitch_release_states.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--motion", type=str, default=None, help="reference the pitching policy was trained with")
parser.add_argument("--envs", type=int, default=1024)
parser.add_argument("--states", type=int, default=4096)
parser.add_argument("--min_kmh", type=float, default=0.0, help="keep only releases at least this fast")
parser.add_argument("--out", type=str, default="data/pitch_release_states.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.pitching import pitch_env_cfg as C  # noqa: E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402


def main():
    cfg = C.PitchElasticEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    cfg.recovery_buffer_size = args.states
    if args.motion:
        cfg.motion_file = os.path.abspath(args.motion)
    env = gym.make("AIB-PitchElastic-v0", cfg=cfg)
    base = env.unwrapped
    w = RslRlVecEnvWrapper(env)
    ag = PitchPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    pol = r.get_inference_policy(device=base.device)
    keep = args.min_kmh / 3.6
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(20000):
            obs, _, _, _ = w.step(pol(obs))
            if base.rb_count >= args.states and (keep <= 0 or (base.rb_speed[: base.rb_count] >= keep).sum() >= args.states // 2):
                break
            if k % 400 == 0:
                print(f"[collect] step {k}: {base.rb_count} states, mean {base.rb_speed[: base.rb_count].mean() * 3.6:.1f} km/h",
                      flush=True)
        if keep > 0:  # drop slow releases
            m = base.rb_speed[: base.rb_count] >= keep
            idx = m.nonzero(as_tuple=False).squeeze(-1)
            for buf in (base.rb_root, base.rb_qpos, base.rb_qvel, base.rb_t, base.rb_speed):
                buf[: len(idx)] = buf[idx].clone()
            for p, v in base.rb_sea or []:
                p[: len(idx)] = p[idx].clone()
                v[: len(idx)] = v[idx].clone()
            base.rb_count = len(idx)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    base.save_release_states(args.out)
    s = base.rb_speed[: base.rb_count] * 3.6
    print(f"[collect] wrote {args.out}: {base.rb_count} states, release {s.mean():.1f} km/h "
          f"(min {s.min():.1f}, max {s.max():.1f})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
