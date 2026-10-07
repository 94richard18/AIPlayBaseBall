"""Are recovery starts faithful? Fill the release-state buffer from real pitches, then start every env from it
and compare the fall timing / state with the natural follow-through.

    python scripts/diag_recovery.py --headless --checkpoint <model.pt>
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
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
from aibaseball.tasks.pitching.pitch_env_cfg import PitchElasticEnvCfg  # noqa: E402


def main():
    cfg = PitchElasticEnvCfg()
    cfg.scene.num_envs = 32
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    cfg.recovery_min_states = 1
    env = gym.make("AIB-PitchElastic-v0", cfg=cfg)
    base = env.unwrapped
    w = RslRlVecEnvWrapper(env)
    ag = PitchPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    pol = r.get_inference_policy(device=base.device)
    n = base.num_envs
    with torch.inference_mode():
        obs, _ = w.reset()
        # phase 1: natural pitches, record time from release to fall
        t_rel = torch.full((n,), -1.0, device=base.device)
        t_fall = torch.full((n,), -1.0, device=base.device)
        for k in range(int(2.0 / base.step_dt)):
            obs, _, done, _ = w.step(pol(obs))
            t = k * base.step_dt
            nr = base.new_release & (t_rel < 0)
            t_rel[nr] = t
            nf = base.fallen & (t_fall < 0) & (t_rel >= 0)
            t_fall[nf] = t
        ok = (t_rel >= 0) & (t_fall >= 0)
        print(f"[rec] natural: {int((t_rel >= 0).sum())}/{n} released, fall {(t_fall - t_rel)[ok].mean():.3f}s after release "
              f"(n={int(ok.sum())}); buffer {base.rb_count} states", flush=True)
        # phase 2: every env starts from a stored release state
        base.cfg.recovery_prob = 1.0
        ids = torch.arange(n, device=base.device)
        base._reset_idx(ids)
        base.scene.write_data_to_sim()
        base.sim.forward()
        obs = base._get_observations()["policy"]
        d = base.robot.data
        print(f"[rec] after restore: recovery_ep {int(base.recovery_ep.sum())}/{n}, pelvis z {d.root_link_pos_w[:, 2].mean():.2f}, "
              f"root speed {d.root_lin_vel_w.norm(dim=-1).mean():.2f} m/s", flush=True)
        t_fall = torch.full((n,), -1.0, device=base.device)
        for k in range(int(1.0 / base.step_dt)):
            act = pol(base.obs_buf["policy"] if hasattr(base, "obs_buf") and base.obs_buf else obs)
            _, _, _, _ = w.step(act)
            obs = base.obs_buf["policy"]
            t = (k + 1) * base.step_dt
            nf = base.fallen & (t_fall < 0)
            t_fall[nf] = t
            if k in (0, 4, 20, 40, 80):
                print(f"[rec] t={t:.3f}s pelvis z {d.root_link_pos_w[:, 2].mean():.2f} tilt "
                      f"{torch.rad2deg(torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1, 1))).mean():.0f} deg "
                      f"fallen {int((t_fall >= 0).sum())}/{n}", flush=True)
        f = t_fall >= 0
        print(f"[rec] recovery starts: fell {int(f.sum())}/{n}, mean time to fall {t_fall[f].mean():.3f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
