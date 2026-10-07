"""Which joints limit the pitch? Torque saturation and joint-speed use in the 0.25 s before release.

    python scripts/diag_pitch_sat.py --headless --checkpoint <model.pt>
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402
from aibaseball.tasks.pitching.pitch_env_cfg import PitchPlayEnvCfg  # noqa: E402


def main():
    cfg = PitchPlayEnvCfg()
    cfg.scene.num_envs = 8
    env = gym.make("AIB-Pitch-Play-v0", cfg=cfg)
    w = RslRlVecEnvWrapper(env)
    agent = PitchPPORunnerCfg()
    runner = OnPolicyRunner(w, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    rob = base.robot
    ids = base.body_ids
    names = base.body_names
    eff = rob.data.joint_effort_limits[0, ids]
    vlim = rob.data.joint_vel_limits[0, ids]
    sat = torch.zeros(len(ids), device=base.device)
    vmax = torch.zeros(len(ids), device=base.device)
    n = 0
    with torch.inference_mode():
        obs, _ = w.reset()
        for _ in range(int(1.5 / base.step_dt)):
            obs, _, _, _ = w.step(policy(obs))
            t = base._ref_time()
            win = (~base.released) & (t > base.release_ref - 0.25)
            if win.any():
                tau = rob.data.applied_torque[:, ids][win].abs()
                sat += (tau >= 0.98 * eff).float().sum(0)
                vmax = torch.maximum(vmax, (rob.data.joint_vel[:, ids][win].abs() / vlim).max(0).values)
                n += int(win.sum())
            if base.released.all():
                break
    print(f"[sat] release speed mean {base.rel_vel[base.released].norm(dim=-1).mean()*3.6:.1f} km/h", flush=True)
    order = (sat / max(n, 1)).argsort(descending=True)
    print("[sat] joint            torque-saturated  peak|qd|/limit  limit(Nm, rad/s)", flush=True)
    for i in order[:14].tolist():
        print(f"[sat] {names[i]:18s} {sat[i]/max(n,1):10.2f} {vmax[i]:14.2f}     {eff[i]:.0f}, {vlim[i]:.0f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
