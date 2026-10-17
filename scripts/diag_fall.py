"""Why do noisy-start pitches fall at ~1.8 s? Falls vs successes over the post-release window, one simulator launch.

    python scripts/diag_fall.py --headless --task AIB-Stage1dN70-v0 --checkpoint <model.pt> [--envs 64]
        [--set start_joint_noise=0.02 start_noise_ramp_steps=0]

Per window (1.0-1.4 / 1.4-1.8 / 1.8-2.2 s), for the episodes that fell in 1.5-2.2 s and the ones standing at the end:
foot contact share, foot slip (horizontal foot speed while in contact), COM distance to the support (segment between
the planted feet / the one planted foot), COM speed, and the joints most often at their torque limit (|tau| >= 95%).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1dN70-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--set", nargs="*", default=[], help="env cfg overrides, key=value")
parser.add_argument("--sat", type=float, default=0.95, help="torque saturation threshold (fraction of the limit)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402

WINDOWS = ((0.85, 1.0), (1.0, 1.4), (1.4, 1.8), (1.8, 2.2))


def say(*a):
    print(*a, flush=True)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    for kv in args.set:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        setattr(cfg, k, v.lower() in ("1", "true") if isinstance(old, bool) else type(old)(ast.literal_eval(v)))
    for k, v in (("landed_start_prob", 0.0), ("rsi_prob", 0.0), ("assist_force_n", 0.0)):
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():  # record falls, never end an episode early (as eval_stage)
        orig()
        z = torch.zeros_like(base.fallen)
        return z, z
    base._get_dones = keep_running
    w = RslRlVecEnvWrapper(env)
    ag = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    T = int(round(cfg.episode_length_s / base.step_dt)) - 2
    n, dt = base.num_envs, base.step_dt
    names = list(base.body_names)  # aligned with base.body_ids
    knee = base.robot.find_joints("l_knee")[0][0]  # lead knee (left leg leads)
    lim = base.robot.root_physx_view.get_dof_max_forces().to(base.device)[0][base.body_ids].clamp_min(1.0)
    nw = len(WINDOWS)
    # per env, per window sums: steps, lead/back contact, lead/back slip (sum, count), COM-support distance, COM speed
    acc = {k: np.zeros((n, nw)) for k in ("steps", "c0", "c1", "s0", "s1", "sd", "sv", "kq", "kv", "fx", "fz")}
    sat = np.zeros((n, nw, len(names)))
    fell_at = np.full(n, np.nan)
    cause = np.full(n, "", dtype=object)  # which fall condition fired first (pelvis / head / <body> touch / flight)
    rb_names = list(base.robot.body_names)
    head_rel = np.full((n, 2), np.nan)  # head - lead foot (x toward the plate, y toward first base) at the fall
    bland = np.full((n, 4), np.nan)  # back foot's first touchdown after 1.0 s: time, back foot - COM y, - lead foot y / x
    b_air = np.zeros(n, dtype=bool)  # back foot off the ground since 1.0 s
    at18 = np.full((n, 5), np.nan)  # at 1.8 s: COM - lead foot x / y, back foot - lead foot x, COM to support, planted
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            obs, _, _, _ = w.step(pol(obs))
            t = (k + 1) * dt
            f = base.fallen.cpu().numpy()
            new = np.isnan(fell_at) & f
            if new.any():
                d0 = base.robot.data
                pz = (d0.body_link_pos_w[:, base.pelvis_id, 2] < cfg.fallen_pelvis_z).cpu().numpy()
                hz = (base._head_z() < cfg.fallen_head_z).cpu().numpy()
                tb = np.full(n, -1)
                if cfg.fallen_touch_z > 0:
                    pp = d0.body_link_pos_w[:, base.touch_ids]
                    x = pp[..., 0] - base.scene.env_origins[:, 0:1]
                    gr = torch.where(x < cfg.mound_slope_start_x, torch.full_like(x, cfg.mound_top_z),
                                     (cfg.mound_top_z - (x - cfg.mound_slope_start_x) * cfg.mound_slope).clamp_min(base.field_z))
                    h = (pp[..., 2] - gr).cpu().numpy()
                    tb = np.where((h < cfg.fallen_touch_z).any(-1), h.argmin(-1), -1)
                for e in np.nonzero(new)[0]:
                    cs = (["pelvis"] if pz[e] else []) + (["head"] if hz[e] else []) +                          ([rb_names[base.touch_ids[tb[e]]] + " touch"] if tb[e] >= 0 else [])
                    cause[e] = "+".join(cs) or "flight"
            if new.any():
                tq = d0.body_link_quat_w[:, base.torso_id]
                hd = d0.body_link_pos_w[:, base.torso_id] + quat_apply(tq, torch.tensor([0.0, 0.0, 0.62], device=base.device).expand(n, 3))
                hr = (hd[:, :2] - d0.body_link_pos_w[:, base.foot_ids[0], :2]).cpu().numpy()
                head_rel[new] = hr[new]
            fell_at[new] = t
            if t >= 1.0:
                d2 = base.robot.data
                bc = (base.feet.data.net_forces_w_history[:, :, base.foot_sensor_ids[1]].norm(dim=-1).max(1).values
                      > cfg.contact_force_n).cpu().numpy()
                land = b_air & bc & np.isnan(bland[:, 0]) & np.isnan(fell_at)
                if land.any():
                    c2, _ = base._com()
                    bp, lp = d2.body_link_pos_w[:, base.foot_ids[1], :2], d2.body_link_pos_w[:, base.foot_ids[0], :2]
                    v = torch.stack([bp[:, 1] - c2[:, 1], bp[:, 1] - lp[:, 1], bp[:, 0] - lp[:, 0]], -1).cpu().numpy()
                    bland[land, 0] = t
                    bland[land, 1:] = v[land]
                b_air |= ~bc
            if abs(t - 1.8) < dt / 2:
                d1 = base.robot.data
                c1, _ = base._com()
                lf, bf = d1.body_link_pos_w[:, base.foot_ids[0], :2], d1.body_link_pos_w[:, base.foot_ids[1], :2]
                fz = base.feet.data.net_forces_w_history[:, :, base.foot_sensor_ids].norm(dim=-1).max(1).values > cfg.contact_force_n
                ab = bf - lf
                sg = (((c1[:, :2] - lf) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0.0, 1.0)
                dseg = (c1[:, :2] - (lf + sg.unsqueeze(-1) * ab)).norm(dim=-1)
                at18[:] = torch.stack([c1[:, 0] - lf[:, 0], c1[:, 1] - lf[:, 1], bf[:, 0] - lf[:, 0], dseg,
                                       fz[:, 0].float() * 1 + fz[:, 1].float() * 2], -1).cpu().numpy()
            wi = [i for i, (a, b) in enumerate(WINDOWS) if a <= t < b]
            if not wi:
                continue
            i = wi[0]
            d = base.robot.data
            forces = base.feet.data.net_forces_w_history[:, :, base.foot_sensor_ids].norm(dim=-1).max(1).values
            planted = forces > cfg.contact_force_n
            feet = d.body_link_pos_w[:, base.foot_ids]
            fv = d.body_link_lin_vel_w[:, base.foot_ids, :2].norm(dim=-1)
            com, com_v = base._com()
            p, a, b = com[:, :2], feet[:, 0, :2], feet[:, 1, :2]
            ab = b - a
            s = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0.0, 1.0)
            d_seg = (p - (a + s.unsqueeze(-1) * ab)).norm(dim=-1)
            dist = torch.where(planted[:, 0] & planted[:, 1], d_seg,
                               torch.where(planted[:, 0], (p - a).norm(dim=-1),
                                           torch.where(planted[:, 1], (p - b).norm(dim=-1),
                                                       torch.full_like(d_seg, float("nan")))))
            pl, fvn = planted.cpu().numpy(), fv.cpu().numpy()
            acc["steps"][:, i] += 1
            acc["c0"][:, i] += pl[:, 0]
            acc["c1"][:, i] += pl[:, 1]
            acc["s0"][:, i] += np.where(pl[:, 0], fvn[:, 0], 0.0)
            acc["s1"][:, i] += np.where(pl[:, 1], fvn[:, 1], 0.0)
            acc["sd"][:, i] += np.nan_to_num(dist.cpu().numpy(), nan=0.0)
            acc["sv"][:, i] += com_v.norm(dim=-1).cpu().numpy()
            acc["kq"][:, i] += d.joint_pos[:, knee].cpu().numpy()
            acc["kv"][:, i] += d.joint_vel[:, knee].cpu().numpy()
            f0 = base.feet.data.net_forces_w[:, base.foot_sensor_ids[0]]  # lead foot ground reaction
            acc["fx"][:, i] += f0[:, :2].norm(dim=-1).cpu().numpy()
            acc["fz"][:, i] += f0[:, 2].abs().cpu().numpy()
            tq = (d.applied_torque[:, base.body_ids].abs() / lim).cpu().numpy()
            sat[:, i] += tq >= args.sat
        ok = base._stand_state()[4].cpu().numpy() & np.isnan(fell_at)
    late = (fell_at >= 1.5) & (fell_at < 2.2)
    say(f"== diag_fall: {args.checkpoint} | {n} episodes, {args.task}, set {' '.join(args.set) or '-'}")
    say(f"  fell {int((~np.isnan(fell_at)).sum())} (in 1.5-2.2 s: {int(late.sum())}), standing at the end {int(ok.sum())}")
    say(f"  fall times: {np.round(np.sort(fell_at[~np.isnan(fell_at)]), 2).tolist()}")
    for lo, hi in ((0.0, 1.5), (1.5, 2.6), (2.6, 9.0)):
        m = (fell_at >= lo) & (fell_at < hi)
        if m.any():
            u, k = np.unique(cause[m], return_counts=True)
            say(f"  fall cause {lo:.1f}-{hi:.1f} s ({int(m.sum())}): " + ", ".join(f"{a} {b}" for a, b in sorted(zip(u, k), key=lambda z: -z[1])))
    lt = (fell_at >= 1.9) & (fell_at < 2.9)
    for gname, g in (("fell 1.9-2.9 s", lt), ("standing", ok)):
        if not g.any():
            continue
        say(f"  -- {gname} ({int(g.sum())}): at 1.8 s COM - lead foot x {np.median(at18[g, 0]):+.2f} y {np.median(at18[g, 1]):+.2f} m"
            f" (each: {' '.join(f'{v:+.2f}/{w:+.2f}' for v, w in at18[g, :2])}); back foot - lead foot x median "
            f"{np.median(at18[g, 2]):+.2f} m (stepped through in {int((at18[g, 2] > 0).sum())}); COM to the feet segment "
            f"{np.median(at18[g, 3]):.2f} m; planted lead only {int((at18[g, 4] == 1).sum())}, back only "
            f"{int((at18[g, 4] == 2).sum())}, both {int((at18[g, 4] == 3).sum())}, none {int((at18[g, 4] == 0).sum())}")
        bl = bland[g]
        hit = ~np.isnan(bl[:, 0])
        say(f"     back foot touchdown after 1.0 s: {int(hit.sum())}/{int(g.sum())}"
            + (f", median at {np.median(bl[hit, 0]):.2f} s; back foot - COM y {np.median(bl[hit, 1]):+.2f} m (1B side of the COM "
               f"in {int((bl[hit, 1] > 0).sum())}), - lead foot y {np.median(bl[hit, 2]):+.2f} x {np.median(bl[hit, 3]):+.2f} m"
               if hit.any() else ""))
        wi = [i for i, (a, b) in enumerate(WINDOWS) if a == 1.4][0]
        st = acc["steps"][g, wi].clip(min=1)
        share = (sat[g, wi] / st[:, None]).mean(0)
        say("     at torque limit 1.4-1.8 s: " + ", ".join(f"{nm} {share[names.index(nm)]:.2f}" for nm in
            ("l_hip_roll", "l_ankle_roll", "l_hip_yaw", "l_hip_pitch", "l_ankle_pitch", "r_hip_roll") if nm in names)
            + " | top: " + ", ".join(f"{names[j]} {share[j]:.2f}" for j in np.argsort(-share)[:4]))
        if gname.startswith("fell"):
            hr = head_rel[g]
            say(f"     head - lead foot at the fall (x toward the plate, y toward first base): median x {np.median(hr[:, 0]):+.2f} "
                f"y {np.median(hr[:, 1]):+.2f} m; forward (x > |y|) {int(((hr[:, 0] > np.abs(hr[:, 1]))).sum())}, "
                f"1B side (y > |x|) {int((hr[:, 1] > np.abs(hr[:, 0])).sum())}, 3B side {int((-hr[:, 1] > np.abs(hr[:, 0])).sum())}, "
                f"backward {int((-hr[:, 0] > np.abs(hr[:, 1])).sum())}")
    for gname, g in (("fell 1.5-2.2 s", late), ("standing", ok)):
        if not g.any():
            continue
        say(f"  -- {gname} ({int(g.sum())} episodes)")
        for i, (a, b) in enumerate(WINDOWS):
            st = acc["steps"][g, i].clip(min=1)
            c0, c1 = acc["c0"][g, i], acc["c1"][g, i]
            say(f"    {a:.1f}-{b:.1f} s: contact lead {np.mean(c0 / st):.2f} back {np.mean(c1 / st):.2f} | slip lead "
                f"{np.sum(acc['s0'][g, i]) / max(c0.sum(), 1):.2f} back {np.sum(acc['s1'][g, i]) / max(c1.sum(), 1):.2f} m/s"
                f" | COM to support {np.mean(acc['sd'][g, i] / st):.3f} m, COM speed {np.mean(acc['sv'][g, i] / st):.2f} m/s")
            say(f"      lead knee {np.mean(acc['kq'][g, i] / st):+.2f} rad, {np.mean(acc['kv'][g, i] / st):+.1f} rad/s | "
                f"lead-foot force horizontal {np.mean(acc['fx'][g, i] / st):.0f} N, vertical {np.mean(acc['fz'][g, i] / st):.0f} N")
            share = (sat[g, i] / st[:, None]).mean(0)
            top = np.argsort(-share)[:6]
            say("      at torque limit: " + ", ".join(f"{names[j]} {share[j]:.2f}" for j in top if share[j] > 0.005))


if __name__ == "__main__":
    main()
    app.close()
