"""Run the latest mimic policy and report actuator saturation / velocity tracking around contact.

    python scripts/diag_policy.py --headless [--checkpoint path]
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--speed", type=float, default=1.15)
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
from aibaseball.tasks.tee_batting.agents.rsl_rl_ppo_cfg import TeeBattingMimicPPORunnerCfg  # noqa: E402
from aibaseball.tasks.tee_batting.mimic_env_cfg import TeeBattingMimicPowerEnvCfg  # noqa: E402


def main():
    ckpt = args.checkpoint or max(glob.glob(os.path.join(ROOT, "logs/rsl_rl/aib1_tee_batting_mimic/*/model_*.pt")),
                                  key=os.path.getmtime)
    print("[diag] checkpoint", ckpt, flush=True)
    cfg = TeeBattingMimicPowerEnvCfg()
    cfg.scene.num_envs = 4
    cfg.rsi_prob = 0.0
    cfg.speed = args.speed
    cfg.speed_ramp_steps = 0
    env = gym.make("AIB-TeeBatting-MimicPower-v0", cfg=cfg)
    w = RslRlVecEnvWrapper(env)
    agent = TeeBattingMimicPPORunnerCfg()
    runner = OnPolicyRunner(w, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    obs, _ = w.get_observations()
    rob = base.robot
    ids = base.body_joint_ids
    eff = rob.data.joint_effort_limits[0, ids] if hasattr(rob.data, "joint_effort_limits") else None
    names = base.body_joint_names
    sat_acc = torch.zeros(len(names), device=base.device)
    nsteps = 0
    for k in range(200):
        with torch.inference_mode():
            obs, _, _, _ = w.step(policy(obs))
        t = base._ref_time()[0].item()
        knob, axis, cm, v_cm, wv, _ = base._bat_state()
        ss, ssv = base._sweet_spot(knob, axis, cm, v_cm, wv)
        if base.ref.contact_time - 0.15 < t < base.ref.contact_time + 0.01:
            tau = rob.data.applied_torque[0, ids]
            comp = rob.data.computed_torque[0, ids]
            ratio = tau.abs() / eff if eff is not None else tau.abs()
            sat_acc += (comp.abs() > tau.abs() + 1e-3).float()
            vlim = rob.data.joint_vel_limits[0, ids] if hasattr(rob.data, "joint_vel_limits") else None
            if vlim is not None:
                vr = rob.data.joint_vel[0, ids].abs() / vlim
                vel_acc = vr if "vel_acc" not in dir() else torch.maximum(vel_acc, vr)
            nsteps += 1
            r = base._ref(base._ref_time())
            dqd = rob.data.joint_vel[0, ids] - r["joint_vel"][0]
            top = ratio.topk(6)
            print(f"[diag] t-c {t-base.ref.contact_time:+.3f}s ss speed {ssv[0].norm():5.1f} (ref "
                  f"{torch.linalg.norm((r['key_pos'][0,-1]-r['key_pos'][0,-2]))*0:.0f}) hit {bool(base.hit[0])} | sat: "
                  + ", ".join(f"{names[i]} {ratio[i]:.2f}" for i in top.indices.tolist())
                  + " | worst vel lag: " + ", ".join(f"{names[i]} {dqd[i]:+.1f}" for i in dqd.abs().topk(3).indices.tolist()),
                  flush=True)
        if base.hit[0] and base.ref.contact_time + 0.01 < t:
            break
    print("[diag] fraction of steps saturated (pre-contact):",
          ", ".join(f"{names[i]} {sat_acc[i]/max(nsteps,1):.2f}" for i in sat_acc.topk(8).indices.tolist()), flush=True)
    if "vel_acc" in dir():
        print("[diag] peak |qd|/limit (pre-contact):", ", ".join(f"{names[i]} {vel_acc[i]:.2f}" for i in vel_acc.topk(8).indices.tolist()), flush=True)
    print(f"[diag] launch speed {base.launch_vel[0].norm():.1f} m/s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
