"""Grip diagnosis: why does the ball leave the hand early (S2: waist on the throw clock, 57/64 slipped out at -50 deg)?

    python scripts/diag_grip.py --headless --task AIB-Stage1dK70GW2MBASVRT1S2-v0 --checkpoint <model.pt> [--envs 16]

Per env, over the last window_s before its release (ball centre > release_distance from the grip point):
ball acceleration relative to the hand and the force the hand has to put on the ball (m (a - g)) in the hand frame,
finger joint torques against their effort limits, finger tracking error, and each pad's gap to the ball surface (index,
middle, ring pads; thumb distal origin) - which one opens first. No contact sensor on the ball: the forces are the ball's
own momentum balance, which is all the fingers / palm together.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1dK70GW2MBASVRT1S2-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--set", nargs="*", default=[], help="env cfg overrides, key=value")
parser.add_argument("--window_s", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_rotate_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.robot.ball_grip import FINGER_JOINTS  # noqa: E402

PADS = ("index", "middle", "ring", "thumb")


def say(*a):
    print(*a, flush=True)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    for kv in args.set:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        if isinstance(old, bool):
            v = v.lower() in ("1", "true")
        elif old is None or v == "None":
            import ast
            try:
                v = None if v == "None" else ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass  # a plain string (e.g. grip_file=grip_v2.json)
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
    n, dev, rob, ball = base.num_envs, base.device, base.robot, base.ball
    fid = base.finger_ids
    eff = rob.data.joint_effort_limits[0, fid] if hasattr(rob.data, "joint_effort_limits") else \
        rob.root_physx_view.get_dof_max_forces()[0, fid].to(dev)
    m_ball = float(base.phys.ball.mass)
    rad = float(base.phys.ball.radius)
    g = torch.tensor([0.0, 0.0, -9.81], device=dev)
    # pad gaps from the real collision cylinders (shape 0 of each contact link), not the grip solver's pad points
    from aibaseball.robot.humanoid import build_humanoid
    hk = build_humanoid(with_bat=False, name="aib1_pitcher")
    seg_ids, seg_a, seg_b, seg_r = [], [], [], []
    for ln in ("r_index_distal", "r_middle_distal", "r_ring_middle", "r_thumb_distal"):
        sh = hk.links[ln].shapes[0]
        ax = np.asarray(sh.rot)[:, 2] * sh.size[1] / 2
        seg_ids.append(rob.find_bodies(ln)[0][0])
        seg_a.append(np.asarray(sh.pos) - ax)
        seg_b.append(np.asarray(sh.pos) + ax)
        seg_r.append(sh.size[0])
    seg_a = torch.tensor(np.array(seg_a), dtype=torch.float32, device=dev).unsqueeze(0).expand(n, 4, 3)
    seg_b = torch.tensor(np.array(seg_b), dtype=torch.float32, device=dev).unsqueeze(0).expand(n, 4, 3)
    seg_r = torch.tensor(seg_r, device=dev)
    T = int(round(1.3 / base.step_dt))
    rec = {k: [] for k in ("tr", "gap", "f_hand", "a_rel", "tau", "qerr", "pads", "w_hand", "ball_hand", "palm_dn")}
    from aibaseball.robot.humanoid import R_PALM_RIGHT
    volar = torch.tensor(R_PALM_RIGHT @ np.array([0.0, 0.0, -1.0]), dtype=torch.float32, device=dev).expand(n, 3)
    rel_at = np.full(n, -1)
    prev_rel = np.zeros(n, dtype=bool)
    v_prev = None
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            obs, _, _, _ = w.step(pol(obs))
            d = rob.data
            hq = d.body_link_quat_w[:, base.hand_id]
            vb = ball.data.root_lin_vel_w
            a_b = (vb - v_prev) / base.step_dt if v_prev is not None else torch.zeros_like(vb)
            v_prev = vb.clone()
            f_w = m_ball * (a_b - g)  # net non-gravity force on the ball = what the hand pushes with
            hand_v = d.body_link_lin_vel_w[:, base.hand_id]
            rec["tr"].append(base._ref_time().cpu().numpy())
            rec["gap"].append((ball.data.root_pos_w - base._grip_point()).norm(dim=-1).cpu().numpy())
            rec["f_hand"].append(quat_rotate_inverse(hq, f_w).cpu().numpy())
            rec["a_rel"].append(quat_rotate_inverse(hq, vb - hand_v).cpu().numpy())
            rec["tau"].append((d.applied_torque[:, fid].abs() / eff).cpu().numpy())
            rec["qerr"].append((d.joint_pos[:, fid] - base.q_target[:, fid]).cpu().numpy())
            lp, lq = d.body_link_pos_w[:, seg_ids], d.body_link_quat_w[:, seg_ids]
            pa, pb = lp + quat_apply(lq, seg_a), lp + quat_apply(lq, seg_b)
            cb = ball.data.root_pos_w.unsqueeze(1)
            tt = (((cb - pa) * (pb - pa)).sum(-1) / ((pb - pa) ** 2).sum(-1)).clamp(0.0, 1.0).unsqueeze(-1)
            rec["pads"].append(((cb - (pa + tt * (pb - pa))).norm(dim=-1) - seg_r - rad).cpu().numpy())
            rec["w_hand"].append(d.body_link_ang_vel_w[:, base.hand_id].norm(dim=-1).cpu().numpy())
            rec["palm_dn"].append(-quat_apply(hq, volar)[:, 2].cpu().numpy())  # cos(palm normal, down): 1 = palm down
            rec["ball_hand"].append(quat_rotate_inverse(hq, ball.data.root_pos_w - d.body_link_pos_w[:, base.hand_id]).cpu().numpy())
            rel = base.released.cpu().numpy()
            rel_at[rel & ~prev_rel] = k
            prev_rel = rel
    R = {k: np.stack(v, 1) for k, v in rec.items()}
    W = int(round(args.window_s / base.step_dt))
    ok = rel_at >= W
    say(f"== diag_grip: {args.checkpoint} | {args.task}, {n} envs; released {int((rel_at >= 0).sum())}, analysed "
        f"{int(ok.sum())}; window {args.window_s * 1000:.0f} ms before the release; ball {m_ball * 1000:.0f} g, "
        f"friction ball 1.0 / 0.9 (static / dynamic, robot default material {cfg.sim.physics_material.static_friction} / "
        f"{cfg.sim.physics_material.dynamic_friction}), finger effort limits {[round(float(x), 1) for x in eff]} Nm")
    tr = np.array([R["tr"][i, k] for i, k in enumerate(rel_at) if k >= 0])
    say(f"  release reference time p10/50/90 {np.percentile(tr, 10):.3f}/{np.median(tr):.3f}/{np.percentile(tr, 90):.3f} s "
        f"(release_ref {float(base.release_ref):.3f})")
    idx = np.array([np.arange(k - W, k + 1) for k in rel_at[ok]])
    rows = np.nonzero(ok)[0][:, None]
    f = R["f_hand"][rows, idx]  # (m, W+1, 3) hand frame
    fm = np.linalg.norm(f, axis=-1)
    say(f"  force the hand puts on the ball (m (a - g)), last {args.window_s * 1000:.0f} ms: median {np.median(fm):.0f} N, "
        f"peak (median over envs) {np.median(fm.max(1)):.0f} N; hand-frame components at the peak (median) "
        f"x {np.median(f[np.arange(len(f)), fm.argmax(1), 0]):+.0f} y {np.median(f[np.arange(len(f)), fm.argmax(1), 1]):+.0f} "
        f"z {np.median(f[np.arange(len(f)), fm.argmax(1), 2]):+.0f} N")
    say(f"  hand angular speed at the peak (median) {np.median(R['w_hand'][rows, idx].max(1)):.1f} rad/s; ball speed relative "
        f"to the hand at the release (median) {np.median(np.linalg.norm(R['a_rel'][rows, idx][:, -1], axis=-1)):.2f} m/s")
    tau = R["tau"][rows, idx]
    say("  finger joints, last window: |torque| / effort limit median / max, % steps >= 0.98, tracking error median (deg):")
    for j, nm in enumerate(FINGER_JOINTS):
        e = np.degrees(R["qerr"][rows, idx][:, :, j])
        say(f"    {nm:15s} {np.median(tau[:, :, j]):.2f} / {tau[:, :, j].max():.2f}, {np.mean(tau[:, :, j] >= 0.98) * 100:4.0f}%, "
            f"{np.median(e):+6.1f}")
    pads = R["pads"][rows, idx]  # (m, W+1, 4) gap pad -> ball surface
    say("  finger cylinder surface gap to the ball surface (mm, < 0 = pressed in), median at -W / -W/2 / release, and how often each pad opens first (> 3 mm):")
    first = []
    for i in range(len(pads)):
        t_open = [np.argmax(pads[i, :, p] > 0.003) if (pads[i, :, p] > 0.003).any() else 10**6 for p in range(4)]
        first.append(int(np.argmin(t_open)) if min(t_open) < 10**6 else -1)
    first = np.array(first)
    for p, nm in enumerate(PADS):
        say(f"    {nm:7s} {np.median(pads[:, 0, p]) * 1000:6.1f} / {np.median(pads[:, W // 2, p]) * 1000:6.1f} / "
            f"{np.median(pads[:, -1, p]) * 1000:6.1f}   first to open in {int((first == p).sum())}/{len(first)}")
    say(f"    none opened > 3 mm before the release in {int((first == -1).sum())}/{len(first)}")
    # early releases (windup slip-outs, before 0.9 s reference time) vs the rest: same window lines, plus the palm
    rel_tr = np.array([R["tr"][i, k] if k >= 0 else np.nan for i, k in enumerate(rel_at)])
    early = ok & (rel_tr < 0.9)
    ti = FINGER_JOINTS.index("thumb_cmc_flex")
    for lab, m in (("early (< 0.9 s)", early), ("on time", ok & ~early)):
        if not m.any():
            continue
        rr = np.nonzero(m)[0][:, None]
        ii = np.array([np.arange(k - W, k + 1) for k in rel_at[m]])
        fmm = np.linalg.norm(R["f_hand"][rr, ii], axis=-1)
        pd = R["palm_dn"][rr, ii]
        say(f"  [{lab}, {int(m.sum())}] release ref time median {np.median(rel_tr[m]):.2f} s; last window: force on the ball "
            f"median {np.median(fmm):.0f} N, thumb_cmc_flex at limit {np.mean(R['tau'][rr, ii][:, :, ti] >= 0.98) * 100:.0f}%; "
            f"palm facing down (cos to gravity) at the release median {np.median(pd[:, -1]):+.2f}, window mean {np.mean(pd):+.2f}")
    # the hold before the throw (reference time 0.2-0.5 s, not released): where the ball actually sits in the hand
    hold = (R["tr"] >= 0.2) & (R["tr"] < 0.5)
    for i, k in enumerate(rel_at):
        if k >= 0:
            hold[i, k:] = False
    bh = R["ball_hand"][hold]
    ref = base.ball_local[0].cpu().numpy()
    off = np.median(bh, 0) - ref
    say(f"  hold (ref time 0.2-0.5 s): ball centre in the r_hand frame median {np.round(np.median(bh, 0), 4).tolist()} vs grip "
        f"{np.round(ref, 4).tolist()} -> offset {np.round(off * 1000, 1).tolist()} mm (|{np.linalg.norm(off) * 1000:.1f}| mm); pad gaps "
        "(mm) median: " + ", ".join(f"{nm} {np.median(R['pads'][hold][:, p]) * 1000:.1f}" for p, nm in enumerate(PADS))
        + "; thumb_cmc_flex at limit " + f"{np.mean(R['tau'][hold][:, FINGER_JOINTS.index('thumb_cmc_flex')] >= 0.98) * 100:.0f}%")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        say(traceback.format_exc())
        raise
    finally:
        app.close()
