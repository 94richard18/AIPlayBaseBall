"""Why does the pitcher fall right after release? Centre of mass, capture point and foot support around release.

    python scripts/diag_pitcher_com.py --headless --checkpoint <model.pt> [--elastic]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--elastic", action="store_true")
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--gain", type=float, default=None, help="override post_release_residual_gain")
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
    cfg, tid = (C.PitchElasticEnvCfg(), "AIB-PitchElastic-v0") if args.elastic else (C.PitchEnvCfg(), "AIB-Pitch-v0")
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    cfg.episode_length_s = 4.0
    if args.gain is not None:
        cfg.post_release_residual_gain = args.gain
    env = gym.make(tid, cfg=cfg)
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
    pol = r.get_inference_policy(device=base.device)
    n, dev = base.num_envs, base.device
    rob = base.robot
    mass = rob.root_physx_view.get_masses().to(dev)  # (n, bodies)
    M = mass.sum(-1, keepdim=True)
    lf, rf = rob.find_bodies("l_foot")[0][0], rob.find_bodies("r_foot")[0][0]
    print(f"[com] total mass {M[0].item():.1f} kg, playback speed {base.speed}, ref length {None}", flush=True)
    leg_names = [nm for nm in rob.joint_names if any(k in nm for k in ("hip", "knee", "ankle"))]
    leg_ids = [rob.joint_names.index(nm) for nm in leg_names]
    effort = rob.root_physx_view.get_dof_max_forces().to(dev)[:, leg_ids]
    sat = torch.zeros(len(leg_ids), device=dev)  # post-release steps at the torque limit, per leg joint
    sat_n = 0
    t_rel = torch.full((n,), -1.0, device=dev)
    t_fall = torch.full((n,), -1.0, device=dev)
    rows = []  # per step: t, ref t, com, com vel, feet, tilt, pelvis z
    with torch.inference_mode():
        obs, _ = w.reset()
        o = base.scene.env_origins
        for k in range(int(2.5 / base.step_dt)):
            obs, _, _, _ = w.step(pol(obs))
            d = rob.data
            t = base.episode_length_buf.float() * base.step_dt
            com = (d.body_com_pos_w * mass.unsqueeze(-1)).sum(1) / M - o
            vcom = (d.body_com_lin_vel_w * mass.unsqueeze(-1)).sum(1) / M
            feet = d.body_link_pos_w[:, [lf, rf]] - o.unsqueeze(1)
            z = d.body_link_pos_w[:, base.pelvis_id, 2]
            tilt = torch.rad2deg(torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1, 1)))
            newly = base.released & (t_rel < 0)
            t_rel[newly] = t[newly]
            fallen = (z < 0.5) | (tilt > 60)
            nf = fallen & (t_fall < 0)
            t_fall[nf] = t[nf]
            post = base.released & (t_fall < 0)
            if post.any():
                tau = d.applied_torque[:, leg_ids][post].abs()
                sat += (tau >= 0.98 * effort[post]).float().sum(0)
                sat_n += int(post.sum())
            rows.append((t.clone(), base._ref_time().clone(), com.clone(), vcom.clone(), feet.clone(), tilt.clone(), z.clone()))
    g = 9.81
    print("[com] per env at release (frame: +x toward plate). cp = capture point = com + v*sqrt(h/g)", flush=True)
    for dt_off in (-0.10, 0.0, 0.05, 0.10, 0.15, 0.20):
        acc = []
        for i in range(n):
            if t_rel[i] < 0:
                continue
            ts = torch.stack([r_[0][i] for r_ in rows])
            k = int(torch.argmin((ts - (t_rel[i] + dt_off)).abs()))
            t_, tr, com, v, feet, tilt, pz = (x[i] for x in rows[k])
            cp = com[:2] + v[:2] * (com[2].clamp_min(0.1) / g) ** 0.5
            acc.append(torch.cat([tr.view(1), com, v, cp, feet[0], feet[1], tilt.view(1), pz.view(1)]))
        a = torch.stack(acc).mean(0)
        print(f"[com] release{dt_off:+.2f}s ref_t {a[0]:.3f} | com x {a[1]:.2f} y {a[2]:.2f} z {a[3]:.2f} | v {a[4]:+.2f} {a[5]:+.2f} {a[6]:+.2f} "
              f"| cp x {a[7]:.2f} y {a[8]:.2f} | Lfoot x {a[9]:.2f} y {a[10]:.2f} z {a[11]:.2f} | Rfoot x {a[12]:.2f} y {a[13]:.2f} z {a[14]:.2f} "
              f"| tilt {a[15]:.0f} pelvis z {a[16]:.2f}", flush=True)
    frac = sat / max(sat_n, 1)
    order = torch.argsort(frac, descending=True)[:8]
    print("[com] leg torque at limit after release (share of steps): "
          + ", ".join(f"{leg_names[i]} {frac[i]:.0%}" for i in order.tolist()), flush=True)
    rel = (t_fall - t_rel)[(t_fall >= 0) & (t_rel >= 0)]
    print(f"[com] fell {int((t_fall >= 0).sum())}/{n}, median {rel.median() if len(rel) else float('nan'):.2f}s after release", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
