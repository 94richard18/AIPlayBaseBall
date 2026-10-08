"""Footing check of a pitching policy: planted feet vs the athlete's schedule, COM error, capture point, slip.

    python scripts/diag_footing.py --headless --checkpoint <pitch model.pt> [--envs 16]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--flat", action="store_true", help="no mound (contact-report test)")
parser.add_argument("--all_bodies", action="store_true", help="contact sensors on every body: who carries the weight?")
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
    cfg = C.PitchStandFirstEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    if args.flat:
        cfg.mound_height = 0.0
    if args.all_bodies:
        cfg.contact_bodies = ".*"
    env = gym.make("AIB-PitchStandFirst-v0", cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():
        orig()
        return torch.zeros_like(base.released), base.episode_length_buf >= base.max_episode_length - 1
    base._get_dones = keep_running
    w = RslRlVecEnvWrapper(env)
    ag = PitchPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    c = base.cfg
    print("[ft]  t   | planted robot L R (ref L R) | COM err | COM v robot / ref (x) | capture out | slip | released")
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(int(1.6 / base.step_dt)):
            obs, _, _, _ = w.step(pol(obs))
            if k % int(0.05 / base.step_dt):
                continue
            t = base._ref_time()
            com, com_v = base._com()
            rc, rv, rcon = base._ref_centroid(t)
            f = base.feet.data.net_forces_w_history[:, :, base.foot_sensor_ids].norm(dim=-1).max(1).values
            pl = (f > c.contact_force_n).float().mean(0)
            if args.all_bodies and t[0] < 0.4:
                h = base.feet.data.net_forces_w_history[0].norm(dim=-1).max(0).values  # env 0, max over substeps
                top = torch.argsort(h, descending=True)[:4].tolist()
                print(f"[all] t {t[0]:.2f} " + ", ".join(f"{base.feet.body_names[i]} {h[i]:.0f} N" for i in top), flush=True)
            if t[0] < 0.4:
                fz = base.robot.data.body_link_pos_w[:4, base.foot_ids, 2]
                print(f"[dbg] t {t[0]:.2f} force L,R env0-3 {f[:4].round().tolist()}  foot link z {fz.round(decimals=3).tolist()}  "
                      f"sensor bodies {base.feet.body_names}", flush=True)
            print(f"[ft] {t[0]:.2f} | {pl[0]:.2f} {pl[1]:.2f} ({int(rcon[0, 0])} {int(rcon[0, 1])}) | "
                  f"{(com - rc).norm(dim=-1).mean():.2f} m | {com_v[:, 0].mean():+.2f} / {rv[0, 0]:+.2f} | "
                  f"{base.stats['capture_out_m']:.2f} | {base.stats['foot_slip_mps']:.2f} | {int(base.released.sum())}",
                  flush=True)
    print("[ft] stats:", {k: round(float(base.stats[k]), 3) for k in ("lead_foot_err_m", "com_err_m", "contact_match",
                                                                     "foot_slip_mps", "capture_out_m")}, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
