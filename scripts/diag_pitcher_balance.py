"""When does the pitcher fall? Pelvis height / tilt over a full pitch + 2 s after release.

    python scripts/diag_pitcher_balance.py --headless --checkpoint <model.pt> [--elastic] [--train_cfg]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--elastic", action="store_true")
parser.add_argument("--train_cfg", action="store_true", help="training config instead of play config")
parser.add_argument("--envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402
from aibaseball.tasks.pitching import pitch_env_cfg as C  # noqa: E402


def main():
    if args.elastic:
        cfg, tid = (C.PitchElasticEnvCfg(), "AIB-PitchElastic-v0") if args.train_cfg else (C.PitchElasticPlayEnvCfg(), "AIB-PitchElastic-Play-v0")
    else:
        cfg, tid = (C.PitchEnvCfg(), "AIB-Pitch-v0") if args.train_cfg else (C.PitchPlayEnvCfg(), "AIB-Pitch-Play-v0")
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    cfg.episode_length_s = 4.0
    if hasattr(cfg, "terminate_on_cross"):
        cfg.terminate_on_cross = False
    env = gym.make(tid, cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():  # observe the whole follow-through (training ends 0.8 s after release)
        orig()
        return torch.zeros_like(base.released), base.episode_length_buf >= base.max_episode_length - 1
    base._get_dones = keep_running
    w = RslRlVecEnvWrapper(env)
    ag = PitchPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    pol = r.get_inference_policy(device=base.device)
    n = base.num_envs
    t_rel = torch.full((n,), -1.0, device=base.device)
    t_fall = torch.full((n,), -1.0, device=base.device)
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(int(3.5 / base.step_dt)):
            obs, _, _, _ = w.step(pol(obs))
            t = base.episode_length_buf.float() * base.step_dt
            d = base.robot.data
            z = d.body_link_pos_w[:, base.pelvis_id, 2]
            tilt = torch.rad2deg(torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1, 1)))
            newly_rel = base.released & (t_rel < 0)
            t_rel[newly_rel] = t[newly_rel]
            fallen = (z < 0.5) | (tilt > 60)
            newly = fallen & (t_fall < 0)
            t_fall[newly] = t[newly]
            if k % int(0.25 / base.step_dt) == 0:
                print(f"[pb] t={t[0]:.2f}s pelvis z {z.mean():.2f} (min {z.min():.2f}) tilt {tilt.mean():3.0f} deg "
                      f"(max {tilt.max():3.0f}) released {int(base.released.sum())}/{n} fallen {int(fallen.sum())}/{n}", flush=True)
    for i in range(n):
        dt = (t_fall[i] - t_rel[i]).item() if t_fall[i] >= 0 and t_rel[i] >= 0 else None
        print(f"[pb] env {i:2d}: release {t_rel[i]:.2f}s  fall {'%.2fs' % t_fall[i] if t_fall[i] >= 0 else 'never'}"
              + (f"  ({dt:+.2f}s after release)" if dt is not None else ""), flush=True)
    falls = (t_fall >= 0)
    after = falls & (t_rel >= 0)
    rel_dt = (t_fall - t_rel)[after]
    print(f"[pb] SUMMARY: fell {int(falls.sum())}/{n}; time after release: "
          + (f"min {rel_dt.min():.2f}s, median {rel_dt.median():.2f}s" if len(rel_dt) else "-"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
