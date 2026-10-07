"""When does the batter lose balance? Pelvis height / tilt over the whole pitch, episodes kept running
after the pitch passes (in training they end there, so post-swing balance is never trained).

    python scripts/diag_batter_balance.py --headless --checkpoint <model.pt>
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
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
from aibaseball.tasks.tee_batting.agents.rsl_rl_ppo_cfg import TeeBattingMimicPPORunnerCfg  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env_cfg import PitchedBattingEnvCfg  # noqa: E402


def main():
    cfg = PitchedBattingEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.pitch_speed_range_start = cfg.pitch_speed_range_final = (75.0, 85.0)
    cfg.timing_noise_s = 0.0
    cfg.pitch_ramp_steps = 0
    cfg.episode_length_s = 3.0
    env = gym.make("AIB-PitchedBatting-v0", cfg=cfg)
    base = env.unwrapped
    base._get_dones_orig = base._get_dones

    def no_early_end():  # keep simulating after a miss / after contact to see the follow-through
        term, to = base._get_dones_orig()
        return base.fallen.clone(), to
    base._get_dones = no_early_end
    w = RslRlVecEnvWrapper(env)
    ag = TeeBattingMimicPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    pol = r.get_inference_policy(device=base.device)
    n = base.num_envs
    fall_t = torch.full((n,), -1.0, device=base.device)
    contact_ref = base.ref.contact_time
    pass_t = torch.full((n,), -1.0, device=base.device)
    min_h = torch.full((n,), 9.0, device=base.device)
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(int(2.8 / base.step_dt)):
            obs, _, done, _ = w.step(pol(obs))
            d = base.robot.data
            h = d.root_link_pos_w[:, 2]
            tilt = torch.rad2deg(torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1, 1)))
            min_h = torch.minimum(min_h, h)
            t = base.phys_t
            passed = (base.pitch_pos[:, 0] < base.contact_x_ref) & (pass_t < 0)
            pass_t[passed] = t[passed]
            newly = (base.fallen) & (fall_t < 0)
            fall_t[newly] = t[newly]
            if k % 20 == 0:
                print(f"[bal] t={t[0]:.2f}s ref_t={base.ref_t[0]:.2f} (contact {contact_ref[base.clip[0]]:.2f}) pelvis z "
                      f"{h.mean():.2f} (min {h.min():.2f}) tilt {tilt.mean():4.0f} deg (max {tilt.max():3.0f}) fallen "
                      f"{int(base.fallen.sum())}/{n} hit {int(base.hit.sum())}", flush=True)
            if bool(done.any()):
                pass
    for i in range(n):
        rel = (fall_t[i] - pass_t[i]).item() if fall_t[i] >= 0 and pass_t[i] >= 0 else None
        print(f"[bal] env {i:2d}: hit {bool(base.hit[i])}  fell {'yes' if fall_t[i] >= 0 else 'no '} "
              f"at {fall_t[i]:.2f}s, ball passed at {pass_t[i]:.2f}s"
              + (f" -> fell {rel:+.2f}s after the ball passed" if rel is not None else "")
              + f" | min pelvis z {min_h[i]:.2f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
