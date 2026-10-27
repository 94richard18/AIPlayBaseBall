"""How do noisy-start throws fall after the release? (peer request 2026-10-20: the faster +35 / +40 ms elbow-lead
runs trade noisy balance for speed - before the next lever, how do they fall)

    python scripts/diag_fall_release.py --headless --task <task> --checkpoint <model.pt> [--envs 128]
        [--set start_joint_noise=0.02 start_noise_ramp_steps=0]

Per outcome group (fell / standing at rest / standing not at rest): fall time after the release (histogram), fall
direction (trunk lean at the fall: toward the plate / back / first base / third base), which foot unloads first after
the release, and at release + 0.3 s: COM distance outside the support (segment between the planted feet, or the one
planted foot) and its direction, whole-body angular momentum about the COM (translational part, world axes), torso
angular velocity, lead-knee angle. Non-fallers: which at-rest sub-check fails.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=128)
parser.add_argument("--set", nargs="*", default=["start_joint_noise=0.02", "start_noise_ramp_steps=0"])
parser.add_argument("--per_env", type=str, default=None, help="save a per-env .npz (start noise, outcome, fall time / direction)")
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


def say(*a):
    print(*a, flush=True)


def seg_dist(p, a, b):
    """Distance from p to segment a-b (xy), and the offset vector p - closest point."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0.0, 1.0)[..., None]
    off = p - (a + t * ab)
    return np.linalg.norm(off, axis=-1), off


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    for kv in args.set:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        if isinstance(old, bool):
            v = v.lower() in ("1", "true")
        elif old is None:
            try:
                v = None if v == "None" else ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
        else:
            v = type(old)(v)
        setattr(cfg, k, v)
    for k, v in (("landed_start_prob", 0.0), ("rsi_prob", 0.0), ("assist_force_n", 0.0)):
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():
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
    n, dt, dev, rob = base.num_envs, base.step_dt, base.device, base.robot
    knee = rob.find_joints("l_knee")[0][0]  # lead knee (left leg leads)
    masses = rob.root_physx_view.get_masses().to(dev)  # (n, bodies)
    up_local = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    rel_t = np.full(n, np.nan)
    fell_t = np.full(n, np.nan)
    fall_dir = np.full((n, 2), np.nan)  # trunk lean at the fall (x toward the plate, y toward first base)
    unload = np.full((n, 2), np.nan)  # first time after the release each foot (lead, back) is unloaded
    at03 = {k: np.full((n, d), np.nan) for k, d in (("sup", 1), ("off", 2), ("L", 3), ("w", 3), ("knee", 1), ("planted", 2))}
    k03 = int(round(0.3 / dt))
    rel_k = np.full(n, -1)
    b_lift = np.full(n, np.nan)  # back foot: first unloaded time after the release, then first re-loaded (touchdown)
    b_td = np.full((n, 3), np.nan)  # touchdown time after the release, back foot - lead foot x / y
    yaw_s = {dtt: np.full((n, 2), np.nan) for dtt in (0.0, 0.2, 0.4, 0.6)}  # pelvis / torso yaw (deg) after the release
    bq45 = np.full((n, len(base.back_leg_cols)), np.nan)  # back-leg joint angles at release + 0.45 s
    b_tdyaw = np.full(n, np.nan)  # back-foot yaw - lead-foot yaw at the back-foot touchdown (deg)
    b_tdyawabs = np.full(n, np.nan)
    pel = {dtt: np.full((n, 7), np.nan) for dtt in (0.1, 0.25, 0.4)}  # pelvis pos - lead foot, quat (IK test poses)
    leadp = np.full((n, 3), np.nan)
    prev_rel = np.zeros(n, dtype=bool)
    with torch.inference_mode():
        obs, _ = w.reset()
        q_start = rob.data.joint_pos.cpu().numpy().copy()  # the start noise per env: deviation from the cross-env mean
        for k in range(T):
            obs, _, _, _ = w.step(pol(obs))
            t = (k + 1) * dt
            d = rob.data
            rel = base.released.cpu().numpy()
            newr = rel & ~prev_rel
            rel_t[newr], rel_k[newr] = t, k
            prev_rel = rel
            f = base.fallen.cpu().numpy()
            nf = np.isnan(fell_t) & f
            if nf.any():
                lean = quat_apply(d.body_link_quat_w[:, base.torso_id], up_local)[:, :2].cpu().numpy()
                fall_dir[nf] = lean[nf]
                fell_t[nf] = t
            forces = base.feet.data.net_forces_w_history[:, :, base.foot_sensor_ids].norm(dim=-1).max(1).values
            planted = (forces > cfg.contact_force_n).cpu().numpy()  # (n, 2) lead, back
            after = ~np.isnan(rel_t) & (t > rel_t)
            for j in range(2):
                m = after & np.isnan(unload[:, j]) & ~planted[:, j]
                unload[m, j] = t - rel_t[m]
            m = after & ~np.isnan(b_lift) & np.isnan(b_td[:, 0]) & planted[:, 1]
            if m.any():
                fp = d.body_link_pos_w[:, base.foot_ids, :2].cpu().numpy()
                def yawq(q):
                    qw, qx, qy, qz = q.cpu().numpy().T
                    return np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
                fy = yawq(d.body_link_quat_w[:, base.foot_ids[1]]) - yawq(d.body_link_quat_w[:, base.foot_ids[0]])
                b_tdyaw[m] = ((fy + 180) % 360 - 180)[m]
                b_tdyawabs[m] = yawq(d.body_link_quat_w[:, base.foot_ids[1]])[m]
                b_td[m, 0] = t - rel_t[m]
                b_td[m, 1:] = (fp[:, 1] - fp[:, 0])[m]
            b_lift[after & np.isnan(b_lift) & ~planted[:, 1]] = 1.0
            for dtt, arr in pel.items():
                hp = (rel_k >= 0) & ((k - rel_k) == int(round(dtt / dt)))
                if hp.any():
                    lf = d.body_link_pos_w[:, base.foot_ids[0]]
                    arr[hp] = torch.cat([d.root_link_pos_w - lf, d.root_link_quat_w], -1).cpu().numpy()[hp]
                    if dtt == 0.4:
                        leadp[hp] = (lf - base.scene.env_origins).cpu().numpy()[hp]
            h45 = (rel_k >= 0) & ((k - rel_k) == int(round(0.45 / dt)))
            if h45.any():
                bq45[h45] = d.joint_pos[:, base.body_ids[base.back_leg_cols]].cpu().numpy()[h45]
            for dtt, arr in yaw_s.items():
                hy = (rel_k >= 0) & ((k - rel_k) == int(round(dtt / dt)))
                if hy.any():
                    def yaw(q):
                        qw, qx, qy, qz = q.cpu().numpy().T
                        return np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
                    arr[hy, 0] = yaw(d.root_link_quat_w)[hy]
                    arr[hy, 1] = yaw(d.body_link_quat_w[:, base.torso_id])[hy]
            hit = rel_k >= 0
            hit &= (k - rel_k) == k03
            if hit.any():
                com, comv = base._com()
                feet = d.body_link_pos_w[:, base.foot_ids, :2].cpu().numpy()
                c = com[:, :2].cpu().numpy()
                both_d, both_off = seg_dist(c, feet[:, 0], feet[:, 1])
                one = np.where(planted[:, :1], feet[:, 0], feet[:, 1])
                one_off = c - one
                sup_d = np.where(planted.all(-1), both_d,
                                 np.where(planted.any(-1), np.linalg.norm(one_off, axis=-1), np.nan))
                off = np.where(planted.all(-1)[:, None], both_off, one_off)
                rr = d.body_link_pos_w - com.unsqueeze(1)
                vv = d.body_link_lin_vel_w - comv.unsqueeze(1)
                L = (masses.unsqueeze(-1) * torch.cross(rr, vv, dim=-1)).sum(1).cpu().numpy()
                at03["sup"][hit, 0] = sup_d[hit]
                at03["off"][hit] = off[hit]
                at03["L"][hit] = L[hit]
                at03["w"][hit] = d.body_link_ang_vel_w[:, base.torso_id].cpu().numpy()[hit]
                at03["knee"][hit, 0] = np.degrees(d.joint_pos[:, knee].cpu().numpy())[hit]
                at03["planted"][hit] = planted[hit]
        both, speed, tilt, head_z, ok = (x.cpu().numpy() for x in base._stand_state())
    fell = ~np.isnan(fell_t)
    up = ~fell
    rest = ok & up
    groups = (("fell", fell), ("at rest", rest), ("up, not at rest", up & ~rest))
    say(f"== diag_fall_release: {args.checkpoint} | {args.task}, {n} envs, set {args.set}")
    say(f"  outcome: fell {fell.sum()}, at rest {rest.sum()}, up not at rest {(up & ~rest).sum()}; released "
        f"{int((~np.isnan(rel_t)).sum())}, release time median {np.nanmedian(rel_t):.3f} s")
    dtf = fell_t - rel_t
    edges = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, np.inf]
    hist = np.histogram(dtf[fell & ~np.isnan(dtf)], bins=edges)[0]
    say("  fall time after the release (s): " + ", ".join(
        f"{'<0' if a == -np.inf else f'{a:.1f}'}-{'' if b == np.inf else f'{b:.1f}'} {h}" for a, b, h in zip(edges[:-1], edges[1:], hist))
        + f"; median {np.nanmedian(dtf[fell]):.2f} s")
    fd = fall_dir[fell]
    if len(fd):
        ax = np.where(np.abs(fd[:, 0]) >= np.abs(fd[:, 1]), np.where(fd[:, 0] > 0, "toward the plate", "back"),
                      np.where(fd[:, 1] > 0, "toward 1B", "toward 3B"))
        say("  fall direction (trunk lean at the fall): " + ", ".join(f"{a} {int((ax == a).sum())}" for a in
                                                                      ("toward the plate", "back", "toward 1B", "toward 3B"))
            + f"; median lean x {np.median(fd[:, 0]):+.2f}, y {np.median(fd[:, 1]):+.2f} (sin of the tilt)")
    for lab, m in groups:
        if not m.any():
            continue
        u = unload[m]
        first = np.where(np.isnan(u[:, 0]) & np.isnan(u[:, 1]), "none",
                         np.where(np.nan_to_num(u[:, 0], nan=9) <= np.nan_to_num(u[:, 1], nan=9), "lead", "back"))
        say(f"  [{lab}, {m.sum()}] first foot to unload after the release: lead {int((first == 'lead').sum())}, back "
            f"{int((first == 'back').sum())}, none {int((first == 'none').sum())}; median time lead {np.nanmedian(u[:, 0]):.2f} s, "
            f"back {np.nanmedian(u[:, 1]):.2f} s")
        a = {k: v[m] for k, v in at03.items()}
        say(f"      at release + 0.3 s: COM outside the support median {np.nanmedian(a['sup']) * 100:.1f} cm (offset x "
            f"{np.nanmedian(a['off'][:, 0]) * 100:+.1f} y {np.nanmedian(a['off'][:, 1]) * 100:+.1f} cm), planted lead / back "
            f"{np.nanmean(a['planted'][:, 0]) * 100:.0f}% / {np.nanmean(a['planted'][:, 1]) * 100:.0f}%; angular momentum "
            f"about the COM (x, y, z) median {np.nanmedian(a['L'][:, 0]):+.1f} {np.nanmedian(a['L'][:, 1]):+.1f} "
            f"{np.nanmedian(a['L'][:, 2]):+.1f} kg m2/s; torso angular velocity {np.nanmedian(a['w'][:, 0]):+.1f} "
            f"{np.nanmedian(a['w'][:, 1]):+.1f} {np.nanmedian(a['w'][:, 2]):+.1f} rad/s; lead knee {np.nanmedian(a['knee']):.0f} deg")
    for lab, m in groups:
        if not m.any():
            continue
        td = b_td[m]
        has = ~np.isnan(td[:, 0])
        bf = (td[:, 0] < (fell_t - rel_t)[m]) if lab == "fell" else has
        say(f"  [{lab}] back foot touchdown after the release: {int(has.sum())}/{m.sum()}"
            + (f", time median {np.nanmedian(td[:, 0]):.2f} s, back - lead foot x {np.nanmedian(td[:, 1]):+.2f} y "
               f"{np.nanmedian(td[:, 2]):+.2f} m" if has.any() else "")
            + (f"; before the fall {int((bf & has).sum())}" if lab == "fell" else "")
            + "; pelvis / torso yaw (deg) at +0.0/+0.2/+0.4/+0.6 s: "
            + " ".join(f"{np.nanmedian(yaw_s[x][m, 0]):+.0f}/{np.nanmedian(yaw_s[x][m, 1]):+.0f}" for x in (0.0, 0.2, 0.4, 0.6)))
    say(f"  back-foot yaw - lead-foot yaw at the back-foot touchdown (deg), median: " + ", ".join(
        f"{lab} {np.nanmedian(b_tdyaw[m]):+.0f} (abs {np.nanmedian(b_tdyawabs[m]):+.0f})" for lab, m in groups if m.any()))
    np.savez(os.path.splitext(args.checkpoint)[0].replace(os.sep, "_").replace("/", "_")[-60:] + "_falldiag.npz" if False else
             "logs/falldiag_poses.npz", rest=rest, fell=fell, b_td=b_td, b_tdyaw=b_tdyaw, b_tdyawabs=b_tdyawabs,
             pel01=pel[0.1], pel025=pel[0.25], pel04=pel[0.4], leadp=leadp, bq45=bq45)
    if args.per_env:  # per-env table for offline analysis (planning 10-27: which start-noise joints predict a fall)
        np.savez(args.per_env, joint_names=np.array(rob.joint_names), q_noise=q_start - q_start.mean(0, keepdims=True),
                 rest=rest, fell=fell, up=up, rel_t=rel_t, fell_t=fell_t, fall_dir=fall_dir, unload=unload,
                 off03=at03["off"], L03=at03["L"])
        say(f"  per-env table -> {args.per_env}")
    if rest.any():
        say(f"  at rest: back-leg joints {[base.body_names[i] for i in base.back_leg_cols]} at release + 0.45 s, median "
            f"{np.round(np.nanmedian(bq45[rest], 0), 3).tolist()}")
    nr = up & ~rest
    if nr.any():
        say(f"  up but not at rest ({nr.sum()}): feet not both down {int((~both[nr]).sum())}, COM speed >= "
            f"{cfg.stand_max_com_v} {int((speed[nr] >= cfg.stand_max_com_v).sum())}, trunk tilt >= {cfg.stand_max_tilt_deg:.0f} "
            f"{int((tilt[nr] >= cfg.stand_max_tilt_deg).sum())}, head <= {cfg.stand_min_head_z} "
            f"{int((head_z[nr] <= cfg.stand_min_head_z).sum())}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        say(traceback.format_exc())
        raise
    finally:
        app.close()
