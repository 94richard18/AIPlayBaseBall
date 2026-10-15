"""Record a pitching policy's PD targets (body joints q_target / qd_target, finger targets) per control step from the
set position, with the reference time, the release time and the ball at the release. The arm + ball stage replays the
upper-body targets as its reference (StageEnv upper_ref_path) - the targets, not the joint angles, so the same
actuators reproduce the same (lagging) arm motion.

    python scripts/record_upper_targets.py --headless --task AIB-PitchElastic-v0 \
        --checkpoint logs/rsl_rl/aib1_pitch/2026-10-07_09-13-50_elastic_balance3/model_11499.pt \
        --out logs/stage_states/elastic_11499_targets.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-PitchElastic-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=1.6)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():
        term, _ = orig()
        z = torch.zeros_like(term)
        return z, z
    base._get_dones = keep_running
    w = RslRlVecEnvWrapper(env)
    ag = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    T = int(round(args.seconds / base.step_dt))
    t_ref, qt, qdt, ft = [], [], [], []
    with torch.inference_mode():
        obs, _ = w.reset()
        for _ in range(T):
            obs, _, _, _ = w.step(pol(obs))
            t_ref.append(float(base._ref_time()[0]))
            qt.append(base.q_target[:, base.body_ids].median(0).values.cpu().numpy())
            qdt.append(base.qd_target[:, base.body_ids].median(0).values.cpu().numpy())
            ft.append(base.q_target[:, base.finger_ids].median(0).values.cpu().numpy())
        rel = base.released.cpu().numpy()
        rel_t = (base.release_step.float() * base.step_dt).cpu().numpy()
        v = base.rel_vel.cpu().numpy()
    print(f"[record] released {rel.sum()}/{len(rel)} at {np.median(rel_t[rel]):.3f} s, ball velocity (median) "
          f"{np.median(v[rel], 0).round(1)} m/s = {np.median(np.linalg.norm(v[rel], axis=-1)) * 3.6:.1f} km/h", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, t_ref=np.array(t_ref), q_target=np.array(qt), qd_target=np.array(qdt), finger_target=np.array(ft),
             body_names=np.array(base.body_names), release_t=float(np.median(rel_t[rel])) if rel.any() else -1.0,
             checkpoint=args.checkpoint)
    print(f"[record] wrote {args.out} ({len(t_ref)} steps)", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
