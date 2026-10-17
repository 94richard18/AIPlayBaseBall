"""Is the human-strength batter's bat speed limited by its joints? Open-loop reference swing (zero residual) in physics,
per actuator group: share of the swing at the torque limit (PD demand >= 98% of effort) and at the speed limit.

    python scripts/diag_bat_speed.py --headless [--strength human|strong] [--effort_scale 1.0] [--gain_scale 1.0] [--ff 1.0]

Prints the sweet-spot speed (peak before contact, and at the reference contact time) next to the reference's own
(kinematic) bat speed, so the gap can be split into "joints saturate" vs "PD tracking lags".
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--strength", choices=("human", "strong"), default="human")
parser.add_argument("--effort_scale", type=float, default=1.0, help="multiplies every group's torque limit (not fingers)")
parser.add_argument("--ff", type=float, default=1.0, help="velocity feed-forward of the reference")
parser.add_argument("--gain_scale", type=float, default=1.0, help="multiplies PD stiffness and damping (not the torque limits)")
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--window", type=float, default=0.3, help="s before the reference contact counted as the swing")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.robot.aib1_cfg import make_aib1_cfg  # noqa: E402
from aibaseball.tasks.tee_batting.mimic_env_cfg import TeeBattingHumanPlayEnvCfg  # noqa: E402


def main():
    cfg = TeeBattingHumanPlayEnvCfg()
    cfg.robot = make_aib1_cfg(human_strength=args.strength == "human", effort_scale=args.effort_scale)
    for name, a in cfg.robot.actuators.items():  # controller gains only: the torque / speed limits stay as chosen
        if name not in ("finger", "thumb"):
            a.stiffness *= args.gain_scale
            a.damping *= args.gain_scale
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    cfg.vel_feedforward = args.ff
    cfg.max_key_err = 10.0
    env = gym.make("AIB-TeeBatting-Human-Play-v0", cfg=cfg).unwrapped
    env.reset()
    n, dev = env.num_envs, env.device
    act = torch.zeros(n, cfg.action_space, device=dev)
    contact_t = env.ref.contact_time[env.clip]
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    peak = torch.zeros(n, device=dev)
    at_contact = torch.zeros(n, device=dev)
    ref_peak = torch.zeros(n, device=dev)
    prev_tip = None
    groups = {name: a for name, a in env.robot.actuators.items() if name not in ("finger", "thumb")}
    sat_t = {g: torch.zeros((), device=dev) for g in groups}
    sat_v = {g: torch.zeros((), device=dev) for g in groups}
    steps = torch.zeros((), device=dev)
    for _ in range(int((contact_t.max().item() + 0.2) / env.speed / env.step_dt)):
        t = env._ref_time()
        r = env._ref(t)
        tip = r["key_pos"][:, -1]
        if prev_tip is not None:
            ref_v = (tip - prev_tip).norm(dim=-1) / env.step_dt
            ref_peak = torch.where(alive & (t <= contact_t), torch.maximum(ref_peak, ref_v), ref_peak)
        prev_tip = tip
        env.step(act)
        alive &= ~env.reset_buf
        knob, axis, cm, v_cm, w, _ = env._bat_state()
        _, ssv = env._sweet_spot(knob, axis, cm, v_cm, w)
        s = ssv.norm(dim=-1)
        t = env._ref_time()
        pre = alive & (t <= contact_t) & ~env.hit
        peak = torch.where(pre, torch.maximum(peak, s), peak)
        near = alive & ((t - contact_t).abs() < 0.5 * env.step_dt * env.speed + 1e-6)
        at_contact = torch.where(near, s, at_contact)
        swing = alive & (t > contact_t - args.window) & (t <= contact_t)
        if swing.any():
            d = env.robot.data
            steps += swing.sum()
            for g, a in groups.items():
                ids = a.joint_indices
                tau = d.computed_torque[:, ids].abs()
                vel = d.joint_vel[:, ids].abs()
                eff = getattr(a, "effort_limit_sim", a.effort_limit)
                vlim = getattr(a, "velocity_limit_sim", a.velocity_limit)
                hit_t = (tau >= 0.98 * eff).any(-1)
                hit_v = (vel >= 0.95 * vlim).any(-1)
                sat_t[g] += (hit_t & swing).sum()
                sat_v[g] += (hit_v & swing).sum()
    ok = alive.sum().item()
    print(f"== diag_bat_speed strength={args.strength} effort_scale={args.effort_scale} gain_scale={args.gain_scale} ff={args.ff} "
          f"({ok}/{n} envs swung to the contact time without a reset)")
    m = alive if ok else torch.ones_like(alive)
    print(f"sweet spot peak before contact {peak[m].mean():.1f} m/s (min {peak[m].min():.1f}, max {peak[m].max():.1f}); "
          f"at the reference contact time {at_contact[m].mean():.1f} m/s")
    print(f"reference bat tip peak (kinematic) {ref_peak[m].mean():.1f} m/s")
    print("swing window share at the limit (any joint of the group): torque / speed")
    for g in groups:
        print(f"  {g:9s} {100 * (sat_t[g] / steps.clamp_min(1)).item():5.1f}% / {100 * (sat_v[g] / steps.clamp_min(1)).item():5.1f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
