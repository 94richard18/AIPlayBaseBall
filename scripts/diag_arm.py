"""Throwing-arm diagnosis for a stage-1 throw policy (stage_env.StageEnv): why the arm adds so little hand speed.

    python scripts/diag_arm.py --headless --task AIB-Stage1dK70GW2MBASVR-v0 --checkpoint <model.pt> [--envs 64]
        [--set start_joint_noise=0.02 start_noise_ramp_steps=0]

1. Hand speed split (pelvis + trunk + arm, along the hand's velocity) of the reference itself, kinematic (forward
   kinematics of the posed robot, central differences): the athlete's motion (scale 1) and the task's PD reference
   (throw_upper_scale), at the reference release and at the robot's median release offset.
2. Right arm (+ waist) joints over the last window_s before each env's release: reference (athlete, PD base) vs robot
   angle and angular velocity, peak timing, SEA motor torque / motor speed / joint speed at their limits, spring
   deflection.
3. Raw policy action on those joints near the release (residual scale saturated at |a| ~ 1, or unused at ~ 0).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1dK70GW2MBASVR-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--set", nargs="*", default=[], help="env cfg overrides, key=value")
parser.add_argument("--window_s", type=float, default=0.15, help="time before the release to analyse")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_mul, quat_conjugate  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402

JOINTS = ["r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_roll", "r_wrist_pitch",
          "r_wrist_yaw", "waist_yaw", "waist_pitch"]
D = 180.0 / np.pi


def say(*a):
    print(*a, flush=True)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    if hasattr(cfg, "landed_start_prob"):
        cfg.landed_start_prob = 0.0
    for kv in args.set:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        if isinstance(old, bool):
            v = v.lower() in ("1", "true")
        elif old is None or v == "None":
            import ast
            v = None if v == "None" else ast.literal_eval(v)
        else:
            v = type(old)(v)
        setattr(cfg, k, v)
    if hasattr(cfg, "rsi_prob"):
        cfg.rsi_prob = 0.0
    if hasattr(cfg, "assist_force_n"):
        cfg.assist_force_n = 0.0
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
    n, dev, rob = base.num_envs, base.device, base.robot
    jnames = list(rob.joint_names)
    jidx = [jnames.index(j) for j in JOINTS]  # robot joint indices
    bcol = [base.body_names.index(j) for j in JOINTS]  # action / body columns
    nj = len(JOINTS)

    # SEA limit flags per physics sub-step (motor torque at the muscle limit, motor at its speed limit)
    seas = base._seas()
    sat_t = torch.zeros(n, nj, device=dev)  # sub-steps at the limit this env step
    sat_v = torch.zeros(n, nj, device=dev)
    sub_n = [0]
    j2c = {}  # robot joint index -> column in JOINTS
    for c_i, j in enumerate(jidx):
        j2c[j] = c_i
    for a in seas:
        cols = [(k, j2c[int(j)]) for k, j in enumerate(a.joint_indices) if int(j) in j2c] \
            if not isinstance(a.joint_indices, slice) else []
        if not cols:
            continue
        orig_compute = a.compute

        def compute(action, joint_pos, joint_vel, a=a, cols=cols, orig_compute=orig_compute):
            c = a.cfg
            if not a.needs_sync.any():
                qd_des = action.joint_velocities if action.joint_velocities is not None else torch.zeros_like(
                    action.joint_positions)
                tau = c.motor_kp * (action.joint_positions - a.motor_pos) + c.motor_kd * (qd_des - a.motor_vel)
                for k, ci in cols:
                    sat_t[:, ci] += (tau[:, k].abs() >= c.motor_effort).float()
            out = orig_compute(action, joint_pos, joint_vel)
            for k, ci in cols:
                sat_v[:, ci] += (a.motor_vel[:, k].abs() >= c.motor_velocity * 0.999).float()
                defl[:, ci] = a.deflection[:, k]
                mlim[ci] = (c.motor_effort, c.motor_velocity)
            return out
        a.compute = compute
    defl = torch.zeros(n, nj, device=dev)
    mlim = {}
    orig_apply = base._apply_action

    def apply():
        sub_n[0] += 1
        orig_apply()
    base._apply_action = apply
    vlim = rob.data.joint_vel_limits[0, jidx].cpu().numpy() if hasattr(rob.data, "joint_vel_limits") else \
        rob.root_physx_view.get_dof_max_velocities()[0, jidx].cpu().numpy()

    T = int(round(1.3 / base.step_dt))  # through the release (~1.0 s reference time)
    rec = {k: [] for k in ("tr", "q", "qd", "base", "based", "ath", "athd", "act", "satt", "satv", "defl")}
    released_at = np.full(n, -1)
    prev = np.zeros(n, dtype=bool)
    sp = torch.as_tensor(base.speed, device=dev)
    ascale = (base.action_scale[bcol] if torch.is_tensor(base.action_scale) else torch.tensor(base.action_scale)[bcol]).to(dev)
    res_arm = cfg.arm_residual_scale if cfg.arm_residual_scale is not None else cfg.residual_scale
    resc = torch.tensor([res_arm if any(k in j for k in ("shoulder", "elbow", "wrist")) else cfg.residual_scale
                         for j in JOINTS], device=dev)
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            act = pol(obs)
            sat_t.zero_(); sat_v.zero_(); sub_n[0] = 0
            obs, _, _, _ = w.step(act)
            d = rob.data
            tr = base._ref_time() - sp * base.step_dt  # the reference time this step's targets were sampled for (approx)
            ath = base.ref.sample(tr)
            a_cols = base.ref_cols[bcol]
            ns = max(sub_n[0], 1)
            qt = base.q_target[:, jidx]
            bpos = qt - base.actions[:, bcol] * ascale * resc  # PD base (reference) = target minus the residual
            rec["tr"].append(tr.cpu().numpy())
            rec["q"].append(d.joint_pos[:, jidx].cpu().numpy())
            rec["qd"].append(d.joint_vel[:, jidx].cpu().numpy())
            rec["base"].append(bpos.cpu().numpy())
            rec["based"].append((base.qd_target[:, jidx] / max(cfg.vel_feedforward, 1e-6)).cpu().numpy())
            rec["ath"].append(ath["joint_pos"][:, a_cols].cpu().numpy())
            rec["athd"].append((ath["joint_vel"][:, a_cols] * sp).cpu().numpy())
            rec["act"].append(act[:, bcol].cpu().numpy())
            rec["satt"].append((sat_t / ns).cpu().numpy())
            rec["satv"].append((sat_v / ns).cpu().numpy())
            rec["defl"].append(defl.cpu().numpy())
            rel = base.released.cpu().numpy()
            released_at[rel & ~prev] = k
            prev = rel
        rel_off = (base.rel_ref_t - base.release_ref).cpu().numpy()
        relm = base.released.cpu().numpy()
    R = {k: np.stack(v, 1) for k, v in rec.items()}  # (n, T, ...)
    W = int(round(args.window_s / base.step_dt))
    ok = (released_at >= W)
    say(f"== diag_arm: {args.checkpoint} | {args.task}, {n} envs, set {args.set}; released {int(relm.sum())}/{n}, "
        f"analysed {int(ok.sum())}; window {args.window_s * 1000:.0f} ms before the release, step {base.step_dt * 1000:.1f} ms")
    say(f"  release - release_ref median {np.median(rel_off[relm]) * 1000:+.1f} ms; throw_upper_scale "
        f"{getattr(cfg, 'throw_upper_scale', None)}, arm residual scale {res_arm}, action scale "
        f"{[round(float(x), 2) for x in ascale.cpu()]}")
    idx = np.array([np.arange(k - W, k + 1) for k in released_at[ok]])  # (m, W+1)
    rows = np.nonzero(ok)[0][:, None]

    def win(key):
        return R[key][rows, idx]  # (m, W+1, nj)

    q, qd, bq, bqd, aq, aqd = (win(k) for k in ("q", "qd", "base", "based", "ath", "athd"))
    act, st, sv, df = win("act"), win("satt"), win("satv"), win("defl")
    tt = (np.arange(-W, 1) * base.step_dt * 1000)
    say("  per joint, last window before the release (medians over envs; deg, deg/s; peak = max |vel| in the window, "
        "t = its time before the release in ms):")
    say("    joint              |  athlete peak vel  t |  PD base peak vel  t |  robot peak vel  t | robot-base angle "
        "rms / at rel | vel at rel athl/base/robot | motor torque lim / motor vel lim / joint vel lim % | deflection "
        "med/max deg | spring Nm max | action |a| med, >0.9 %, mean a at rel")
    for c_i, j in enumerate(JOINTS):
        def peak(v):
            a = np.abs(v[:, :, c_i])
            ii = a.argmax(1)
            return np.median(a.max(1)) * D, np.median(tt[ii])
        pa, ta = peak(aqd)
        pb, tb = peak(bqd)
        pr, tr_ = peak(qd)
        err = (q[:, :, c_i] - bq[:, :, c_i]) * D
        rms = np.median(np.sqrt((err ** 2).mean(1)))
        e_rel = np.median(err[:, -1])
        vr = [np.median(x[:, -1, c_i]) * D for x in (aqd, bqd, qd)]
        jl = np.mean(np.abs(qd[:, :, c_i]) >= vlim[c_i] * 0.98) * 100
        ml = mlim.get(c_i)
        a_abs = np.abs(act[:, :, c_i])
        k_s = None
        for a in seas:
            if not isinstance(a.joint_indices, slice) and jidx[c_i] in [int(x) for x in a.joint_indices]:
                k_s = a.cfg.spring_stiffness
        dmax = np.median(np.abs(df[:, :, c_i]).max(1)) * D
        say(f"    {j:18s} | {pa:8.0f} {ta:+5.0f} | {pb:8.0f} {tb:+5.0f} | {pr:8.0f} {tr_:+5.0f} | {rms:6.1f} / {e_rel:+6.1f} "
            f"| {vr[0]:+7.0f} / {vr[1]:+7.0f} / {vr[2]:+7.0f} | "
            + (f"{np.mean(st[:, :, c_i]) * 100:5.1f} / {np.mean(sv[:, :, c_i]) * 100:5.1f} / {jl:5.1f}" if ml else
               f"  -   /   -   / {jl:5.1f}")
            + (f" | {np.median(np.abs(df[:, :, c_i])) * D:5.1f} / {dmax:5.1f} | {dmax / D * k_s:5.0f}" if k_s else " |   -   |   -  ")
            + f" | {np.median(a_abs):.2f}, {np.mean(a_abs > 0.9) * 100:4.1f}, {np.mean(act[:, -10:, c_i]):+.2f}")
    say("  SEA gains (motor_kp / motor_kd / spring Nm/rad): " + ", ".join(
        f"{a.cfg.joint_names_expr[0]} {a.cfg.motor_kp:.0f}/{a.cfg.motor_kd:.1f}/{a.cfg.spring_stiffness:.0f}" for a in seas))
    if mlim:
        say("  SEA motor limits (Nm, rad/s): " + ", ".join(f"{JOINTS[c]} {v[0]:.0f}/{v[1]:.0f}" for c, v in sorted(mlim.items()))
            + "; joint speed limits (rad/s): " + ", ".join(f"{j} {v:.0f}" for j, v in zip(JOINTS, vlim)))
    # last 50 ms: limit use where it matters most
    k5 = int(round(0.05 / base.step_dt))
    say("  last 50 ms: motor torque at limit % / motor speed at limit % per joint: " + ", ".join(
        f"{j} {np.mean(st[:, -k5:, c]) * 100:.0f}/{np.mean(sv[:, -k5:, c]) * 100:.0f}" for c, j in enumerate(JOINTS)))

    # 1. kinematic hand speed split of the reference
    def fk_split(t, scaled):
        h = 0.0025
        pos = []
        for dt_ in (-h, 0.0, h):
            tt_ = t + dt_
            rr = base._ref(tt_) if scaled else None
            raw = base.ref.sample(tt_)
            jp = rob.data.joint_pos.clone()
            jp[:, base.body_ids] = rr["joint_pos"] if scaled else raw["joint_pos"][:, base.ref_cols]
            root = torch.cat([raw["root_pos"] + base.scene.env_origins, raw["root_quat"]], -1)
            rob.write_root_pose_to_sim(root)
            rob.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            base.sim.forward()
            base.scene.update(dt=0.0)
            dd = rob.data
            pos.append((dd.body_link_pos_w[:, base.hand_id].clone(), dd.body_link_pos_w[:, base.torso_id].clone(),
                        dd.body_link_quat_w[:, base.torso_id].clone(), dd.root_link_pos_w.clone()))
        (h0, t0, q0, p0), (h1, t1, q1, p1), (h2, t2, q2, p2) = pos
        vh, vtc, vp = (h2 - h0) / (2 * h) * sp, (t2 - t0) / (2 * h) * sp, (p2 - p0) / (2 * h) * sp
        dq = quat_mul(q2, quat_conjugate(q0))
        dq = torch.where(dq[:, :1] < 0, -dq, dq)
        ang = 2 * torch.atan2(dq[:, 1:].norm(dim=-1), dq[:, 0])
        om = dq[:, 1:] / dq[:, 1:].norm(dim=-1, keepdim=True).clamp_min(1e-9) * (ang / (2 * h) * sp).unsqueeze(-1)
        vt = vtc + torch.cross(om, h1 - t1, dim=-1)
        u = vh / vh.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        parts = torch.stack([(vp * u).sum(-1), ((vt - vp) * u).sum(-1), ((vh - vt) * u).sum(-1)], -1)
        return parts[0].cpu().numpy(), float(vh[0].norm())

    with torch.inference_mode():
        off = float(np.median(rel_off[relm])) if relm.any() else 0.0
        say("  reference wrist (r_hand link) speed at the release, kinematic (pelvis + trunk + arm = along the hand's velocity; |v|):")
        for lab, scaled in (("athlete (scale 1.0)", False), (f"PD reference (scale {getattr(cfg, 'throw_upper_scale', 1)})", True)):
            for tl, dt_ in (("release_ref", 0.0), (f"release_ref {off * 1000:+.1f} ms", off)):
                t = torch.full((n,), float(base.release_ref) + dt_, device=dev)
                p, vn = fk_split(t, scaled)
                say(f"    {lab:26s} at {tl:22s}: pelvis {p[0]:+.2f} + trunk {p[1]:+.2f} + arm {p[2]:+.2f} = "
                    f"{p.sum():.2f} m/s (|v| {vn:.2f})")
            best = None
            for dt_ in np.arange(-0.06, 0.061, 0.0025):
                t = torch.full((n,), float(base.release_ref) + dt_, device=dev)
                p, vn = fk_split(t, scaled)
                if best is None or vn > best[1]:
                    best = (dt_, vn, p)
            say(f"    {lab:26s} peak |v| within +-60 ms: {best[1]:.2f} m/s at {best[0] * 1000:+.1f} ms (pelvis "
                f"{best[2][0]:+.2f} + trunk {best[2][1]:+.2f} + arm {best[2][2]:+.2f})")
        rk = base.ref.sample(torch.tensor([float(base.release_ref) - 0.001, float(base.release_ref) + 0.001], device=dev))
        say(f"  athlete ball key point speed at release_ref (mocap space): "
            f"{float((rk['key_pos'][1, -1] - rk['key_pos'][0, -1]).norm() / 0.002):.2f} m/s")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        say(traceback.format_exc())
        raise
    finally:
        app.close()
