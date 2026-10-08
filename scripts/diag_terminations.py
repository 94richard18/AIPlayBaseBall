"""Why do pitching episodes end? First-episode termination reason and time per env (training dones, no override).

    python scripts/diag_terminations.py --headless --checkpoint <model.pt> [--cfg stand_first] [--envs 64]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--cfg", choices=["elastic", "stand_first", "speed"], default="stand_first")
parser.add_argument("--envs", type=int, default=64)
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
    cfg = {"elastic": C.PitchElasticEnvCfg, "stand_first": C.PitchStandFirstEnvCfg, "speed": C.PitchSpeedEnvCfg}[args.cfg]()
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    env = gym.make("AIB-PitchElastic-v0", cfg=cfg)
    base = env.unwrapped
    reason = {}
    orig = base._get_dones

    def record():
        term, trunc = orig()
        t = (base.episode_length_buf.float() + 1) * base.step_dt
        for i in (term | trunc).nonzero(as_tuple=False).squeeze(-1).tolist():
            if i in reason:
                continue
            why = ("fallen" if base.fallen[i] else "lost tracking" if base._lost[i]
                   else "dropped" if base.dropped[i] and base.released[i]
                   else "follow-through done" if base.released[i] and term[i] else "time/ref end")
            reason[i] = (why, float(t[i]), bool(base.released[i]), float(base._key_err[i]))
        return term, trunc
    base._get_dones = record
    w = RslRlVecEnvWrapper(env)
    ag = PitchPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    with torch.inference_mode():
        obs, _ = w.reset()
        for _ in range(int(4.0 / base.step_dt)):
            obs, _, _, _ = w.step(pol(obs))
            if len(reason) == base.num_envs:
                break
    counts = {}
    for why, t, rel, ke in reason.values():
        counts.setdefault(why, []).append(t)
    for why, ts in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        ts = sorted(ts)
        print(f"[term] {why:20s} {len(ts):3d}/{base.num_envs}  time median {ts[len(ts) // 2]:.2f} s  "
              f"(min {ts[0]:.2f}, max {ts[-1]:.2f})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
