"""Kinetic-chain timing diagnosis for a stage throw policy (stage_env.StageEnv), plan docs/plans/fastball_restart.md 5-15.

    python scripts/diag_chain.py --headless --task AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3C-v0 \
        --checkpoint <model.pt> [--arm_teacher <model.pt>] [--envs 128] [--set start_joint_noise=0.02 start_noise_ramp_steps=0]
    python scripts/diag_chain.py --ref_only --task ...   # tool check on the retargeted reference only (no simulator run)

Kinematic peak timing (not an energy-transfer analysis; nothing here diagnoses a cause). Same functions as
aibaseball/analysis/pitch_report.py (diag_pitch) for the robot and the retargeted reference:
  * five speed curves as chain_series (pelvis yaw rate, trunk yaw rate, shoulder internal rotation, elbow extension,
    hand speed), but SIGNED: each curve's direction is the sign of the retargeted reference at its own peak, the robot
    uses the same sign (hand speed is a magnitude);
  * peak window: release - 450 ms .. release (no follow-through); a peak within 5 ms of a window edge, or a signed
    peak <= 0, is unreadable for that env and curve;
  * arm peak = the earlier of shoulder IR and elbow extension (both must be readable);
  * handoff window = [pelvis peak, trunk peak]; empty (trunk not after pelvis) -> handoff saturation unreadable;
  * waist yaw / pitch saturation = share of physics sub-steps with motor torque OR motor speed at its limit, in the
    handoff window and in the 450 ms window;
  * per-env class: (C1) order pelvis < trunk < arm with both gaps >= 10 ms AND handoff waist yaw or pitch saturation
    >= 0.3 ("consistent with the waist-limit hypothesis"); (C2) trunk peak < 10 ms after the pelvis peak or before it
    ("timing difference observed"); (C3) other readable; unreadable listed apart. 10 ms / 0.3 are empirical
    descriptive thresholds (no causal or error guarantee). n_valid < 96 (0.75 n) -> EVIDENCE INSUFFICIENT.
  * reported only (no thresholds): hip-shoulder separation at lead-foot strike and its max in the window, peak ratios
    trunk / pelvis and arm / trunk, peak times relative to the release.
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3C-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--arm_teacher", type=str, default=None, help="checkpoint whose frozen actor drives the arm + release columns")
parser.add_argument("--envs", type=int, default=128)
parser.add_argument("--set", nargs="*", default=[], help="env cfg overrides, key=value")
parser.add_argument("--ref_only", action="store_true", help="tool check: the retargeted reference only")
parser.add_argument("--dump", type=str, default=None, help="save per-env results (.npz)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.analysis import pitch_report as PR  # noqa: E402

LO, HI, EDGE, GAP, SAT = -0.45, 0.0, 0.005, 0.010, 0.3
NAMES = ["pelvis", "trunk", "shoulder IR", "elbow ext", "hand"]
UNITS = ["deg/s", "deg/s", "deg/s", "deg/s", "m/s"]


def say(*a):
    print(*a, flush=True)


def signed_chain(R_pel, R_tor, qd_sh_yaw, qd_elbow, hand_v, dt):
    """chain_series without the absolute value (same filters)."""
    s = np.stack([PR.ang_vel_from_rot(R_pel, dt)[:, 2] * 57.3, PR.ang_vel_from_rot(R_tor, dt)[:, 2] * 57.3,
                  qd_sh_yaw * 57.3, qd_elbow * 57.3, hand_v], 1)
    return PR.smooth(s, dt, 0.02)


def reference_signed(cfg):
    """Retargeted reference: signed chain curves, separation series, release, strike (as PR.reference_profile)."""
    from aibaseball.mocap.centroidal import mound_z, reference_centroidal
    from aibaseball.robot.ball_grip import FINGER_JOINTS, BallGrip
    from aibaseball.robot.humanoid import build_humanoid

    d = np.load(cfg.motion_file, allow_pickle=True)
    t = d["time"]
    dt = float(t[1] - t[0])
    names = [str(n) for n in d["joint_names"]]
    robot = build_humanoid(with_bat=False, name="aib1_pitcher", arm_masses="human")
    grip = BallGrip.load(os.path.join(os.path.dirname(PR.__file__), "..", "..", "assets", "aib1_pitcher", "grip.json"))
    fingers = {f"l_{n}": 0.2 for n in FINGER_JOINTS} | grip.joint_pos
    Rz = Rotation.from_euler("z", math.radians(cfg.ref_yaw_deg))
    Rr = Rz * Rotation.from_quat(np.concatenate([d["root_quat"][:, 1:], d["root_quat"][:, :1]], -1))
    root = Rz.apply(d["root_pos"])
    T = len(t)
    Rp, Rt, hand = np.zeros((T, 3, 3)), np.zeros((T, 3, 3)), np.zeros((T, 3))
    for f in range(T):
        P = robot.fk(dict(zip(names, d["joint_pos"][f])) | fingers, root_pos=root[f], root_rot=Rr[f].as_matrix())
        Rp[f], Rt[f], hand[f] = P["pelvis"][0], P["torso"][0], P["r_hand"][1]
    hand_v = np.linalg.norm(PR.smooth(hand, dt, 0.03, deriv=1), axis=-1)
    qd = d["joint_vel"]
    s = signed_chain(Rp, Rt, qd[:, names.index("r_shoulder_yaw")], qd[:, names.index("r_elbow")], hand_v, dt)
    sep = (PR.yaw_deg(Rp) - PR.yaw_deg(Rt) + 180) % 360 - 180
    cen = reference_centroidal(cfg.motion_file, lambda x: mound_z(x, cfg.mound_top_z, cfg.mound_height, cfg.mound_slope,
                                                                   cfg.mound_slope_start_x))
    strike = float(t[np.argmax(cen["contact"][:, 0] & (t > 0.3))])
    return dict(t=t, dt=dt, s=s, sep=sep, release=float(d["contact_time"]), strike=strike)


def peaks(t, s, t_rel, sign):
    """Per curve: (time rel. release, signed value, readable) in [t_rel + LO, t_rel + HI]."""
    m = (t >= t_rel + LO - 1e-9) & (t <= t_rel + HI + 1e-9)
    out = []
    for j in range(s.shape[1]):
        v = s[:, j] * sign[j]
        i = int(np.argmax(np.where(m, v, -np.inf)))
        tp = t[i] - t_rel
        ok = bool(m.any()) and v[i] > 0 and (tp - LO) > EDGE and (HI - tp) > EDGE
        out.append((tp, v[i], ok))
    return out


def classify(pk, sat_hand):
    """pk: peaks() of one env; sat_hand: (yaw, pitch) handoff saturation or None. -> C1 / C2 / C3 / unreadable"""
    (tp, _, okp), (tt, _, okt) = pk[0], pk[1]
    if not (okp and okt):
        return "unreadable"
    if tt - tp < GAP:
        return "C2"
    if not (pk[2][2] and pk[3][2]) or sat_hand is None:
        return "unreadable"
    ta = min(pk[2][0], pk[3][0])
    if ta - tt >= GAP and max(sat_hand) >= SAT:
        return "C1"
    return "C3"


def ref_report(cfg):
    ref = reference_signed(cfg)
    sign = np.where(np.array([ref["s"][np.argmax(np.abs(np.where(
        (ref["t"] >= ref["release"] + LO) & (ref["t"] <= ref["release"] + HI), ref["s"][:, j], 0))), j]
        for j in range(5)]) >= 0, 1.0, -1.0)
    sign[4] = 1.0
    pk = peaks(ref["t"], ref["s"], ref["release"], sign)
    # wiring check: the unsigned curve of pitch_report (abs before the filter) in the same window
    prof = PR.reference_profile(cfg)
    chk = PR.chain_peaks(prof["t"], prof["chain"], prof["release"], LO, HI)
    say(f"[ref] retargeted reference ({os.path.basename(cfg.motion_file)}, ref_yaw {cfg.ref_yaw_deg}), release "
        f"{ref['release']:.3f} s, lead-foot strike {ref['strike'] - ref['release']:+.3f} s rel. release; signs {sign.tolist()}")
    for j in range(5):
        say(f"[ref]   {NAMES[j]:12s} peak {pk[j][0] * 1000:+6.0f} ms  {pk[j][1]:8.1f} {UNITS[j]:5s} readable {pk[j][2]} | "
            f"pitch_report |.| {chk[j][0] * 1000:+6.0f} ms {chk[j][1]:8.1f}")
    tp, tt = pk[0][0], pk[1][0]
    ta = min(pk[2][0], pk[3][0])
    say(f"[ref]   order pelvis < trunk < arm: {tp < tt < ta} (gaps {(tt - tp) * 1000:+.0f} / {(ta - tt) * 1000:+.0f} ms); "
        f"separation at strike {PR.at(ref['t'], ref['sep'], ref['strike']):+.1f} deg, max |.| in window "
        f"{np.max(np.abs(ref['sep'][(ref['t'] >= ref['release'] + LO) & (ref['t'] <= ref['release'])])):.1f} deg; "
        f"ratios trunk/pelvis {pk[1][1] / pk[0][1]:.2f}, arm(IR)/trunk {pk[2][1] / pk[1][1]:.2f}")
    return ref, sign, pk


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
    ref, sign, rpk = ref_report(cfg)
    if args.ref_only:
        return
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

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
    if args.arm_teacher:
        from aibaseball.rl.arm_teacher import attach_arm_teacher

        attach_arm_teacher(r.alg, args.arm_teacher, list(base._arm_cols) + [29 + base.cfg.release_action_col], base.device)
    pol = r.get_inference_policy(device=base.device)
    n, dev, rob = base.num_envs, base.device, base.robot
    jn = list(rob.joint_names)
    j_sh, j_el, j_kn = jn.index("r_shoulder_yaw"), jn.index("r_elbow"), jn.index("r_knee")
    WJ = [jn.index("waist_yaw"), jn.index("waist_pitch")]

    # waist SEA limit flags per physics sub-step (as diag_arm: motor torque at its limit OR motor speed at its limit)
    sat = torch.zeros(n, 2, device=dev)
    sub_n = [0]
    hooked = set()
    for a in base._seas():
        if isinstance(a.joint_indices, slice):
            continue
        cols = [(k, WJ.index(int(j))) for k, j in enumerate(a.joint_indices) if int(j) in WJ]
        if not cols:
            continue
        hooked |= {c for _, c in cols}
        orig_compute = a.compute

        def compute(action, joint_pos, joint_vel, a=a, cols=cols, orig_compute=orig_compute):
            c = a.cfg
            tf = {}
            if not a.needs_sync.any():
                qd_des = action.joint_velocities if action.joint_velocities is not None else torch.zeros_like(
                    action.joint_positions)
                tau = c.motor_kp * (action.joint_positions - a.motor_pos) + c.motor_kd * (qd_des - a.motor_vel)
                for k, ci in cols:
                    tf[ci] = tau[:, k].abs() >= c.motor_effort
            out = orig_compute(action, joint_pos, joint_vel)
            for k, ci in cols:
                vf = a.motor_vel[:, k].abs() >= c.motor_velocity * 0.999
                sat[:, ci] += (vf | tf[ci]).float() if ci in tf else vf.float()
            return out
        a.compute = compute
    if hooked != {0, 1}:
        say(f"[chain] WARNING: waist SEA hooks on columns {sorted(hooked)} only")
    orig_apply = base._apply_action

    def apply():
        sub_n[0] += 1
        orig_apply()
    base._apply_action = apply

    T = int(round(1.3 / base.step_dt))
    dt = base.step_dt
    rec = {k: [] for k in ("Rp", "Rt", "hand", "sh", "el", "sat")}
    released_at = np.full(n, -1)
    fell_at = np.full(n, -1)
    strike_at = np.full(n, -1)
    prev = np.zeros(n, dtype=bool)

    def qmat(q):  # (n, 4) wxyz -> (n, 3, 3)
        q = q.cpu().numpy()
        return Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], -1)).as_matrix()

    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            act = pol(obs)
            sat.zero_(); sub_n[0] = 0
            obs, _, _, _ = w.step(act)
            d = rob.data
            rec["Rp"].append(qmat(d.body_link_quat_w[:, base.pelvis_id]))
            rec["Rt"].append(qmat(d.body_link_quat_w[:, base.torso_id]))
            rec["hand"].append(d.body_link_pos_w[:, base.hand_id].cpu().numpy())
            rec["sh"].append(d.joint_vel[:, j_sh].cpu().numpy())
            rec["el"].append(d.joint_vel[:, j_el].cpu().numpy())
            rec["sat"].append((sat / max(sub_n[0], 1)).cpu().numpy())
            rel = base.released.cpu().numpy()
            released_at[rel & ~prev] = k
            prev = rel
            fl = base.fallen.cpu().numpy()
            fell_at[fl & (fell_at < 0)] = k
            if hasattr(base, "s_landed"):
                ld = base.s_landed.cpu().numpy()
                strike_at[ld & (strike_at < 0)] = k
    R = {k: np.stack(v, 1) for k, v in rec.items()}  # (n, T, ...)
    t = (np.arange(T) + 1) * dt
    W = int(round(-LO / dt))
    not_rel = released_at < 0
    short = (released_at >= 0) & (released_at < W + 5)
    fell_pre = (released_at >= 0) & (fell_at >= 0) & (fell_at <= released_at)
    valid = ~not_rel & ~short & ~fell_pre
    say(f"== diag_chain: {args.checkpoint} | {args.task}, teacher {args.arm_teacher}, {n} envs, set {args.set}; step "
        f"{dt * 1000:.1f} ms")
    say(f"[chain] n_valid {int(valid.sum())}/{n} (excluded: not released {int(not_rel.sum())}, window incomplete "
        f"{int(short.sum())}, fell before the release {int(fell_pre.sum())}); "
        + ("EVIDENCE INSUFFICIENT (n_valid < 0.75 n)" if valid.sum() < 0.75 * n else "evidence ok"))
    rows = []
    for i in np.nonzero(valid)[0]:
        hand_v = np.linalg.norm(PR.smooth(R["hand"][i], dt, 0.03, deriv=1), axis=-1)
        s = signed_chain(R["Rp"][i], R["Rt"][i], R["sh"][i], R["el"][i], hand_v, dt)
        tr = t[released_at[i]]
        pk = peaks(t, s, tr, sign)
        win = (t >= tr + LO - 1e-9) & (t <= tr + 1e-9)
        sat_w = R["sat"][i][win].mean(0)
        sh_ = None
        if pk[0][2] and pk[1][2] and pk[1][0] > pk[0][0]:
            hw = (t >= tr + pk[0][0] - 1e-9) & (t <= tr + pk[1][0] + 1e-9)
            sh_ = R["sat"][i][hw].mean(0)
        sep = (PR.yaw_deg(R["Rp"][i]) - PR.yaw_deg(R["Rt"][i]) + 180) % 360 - 180
        sep_st = sep[strike_at[i]] if 0 <= strike_at[i] <= released_at[i] else np.nan
        rows.append(dict(env=i, pk=pk, cls=classify(pk, sh_), sat_w=sat_w, sat_h=sh_, sep_st=sep_st,
                         sep_max=np.max(np.abs(sep[win]))))
    m = len(rows)
    if not m:
        return

    def q3(x):
        x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
        return (f"{np.median(x):8.1f} [{np.percentile(x, 25):7.1f}, {np.percentile(x, 75):7.1f}] (n {len(x)})"
                if len(x) else "      -")
    say("[chain] per curve over valid envs (readable only): peak time ms rel. release median [p25, p75] | signed peak "
        "value | unreadable count | reference")
    for j in range(5):
        ok = [r_["pk"][j] for r_ in rows if r_["pk"][j][2]]
        say(f"[chain]   {NAMES[j]:12s} t {q3([p[0] * 1000 for p in ok])} | {q3([p[1] for p in ok])} {UNITS[j]} | "
            f"unreadable {m - len(ok)} | ref {rpk[j][0] * 1000:+.0f} ms {rpk[j][1]:.1f}")
    both = [r_ for r_ in rows if r_["pk"][0][2] and r_["pk"][1][2]]
    say(f"[chain]   gap trunk - pelvis ms {q3([(r_['pk'][1][0] - r_['pk'][0][0]) * 1000 for r_ in both])} | ref "
        f"{(rpk[1][0] - rpk[0][0]) * 1000:+.0f}")
    arm = [r_ for r_ in both if r_["pk"][2][2] and r_["pk"][3][2]]
    say(f"[chain]   gap arm - trunk ms  {q3([(min(r_['pk'][2][0], r_['pk'][3][0]) - r_['pk'][1][0]) * 1000 for r_ in arm])} | "
        f"ref {(min(rpk[2][0], rpk[3][0]) - rpk[1][0]) * 1000:+.0f}")
    say(f"[chain]   ratio trunk / pelvis {q3([r_['pk'][1][1] / r_['pk'][0][1] for r_ in both])} | ref {rpk[1][1] / rpk[0][1]:.2f}")
    ir = [r_ for r_ in rows if r_["pk"][1][2] and r_["pk"][2][2]]
    say(f"[chain]   ratio IR / trunk     {q3([r_['pk'][2][1] / r_['pk'][1][1] for r_ in ir])} | ref {rpk[2][1] / rpk[1][1]:.2f}")
    say(f"[chain]   separation at lead-foot strike deg {q3([r_['sep_st'] for r_ in rows])} | max |.| in window "
        f"{q3([r_['sep_max'] for r_ in rows])}")
    for c, lab in enumerate(("waist_yaw", "waist_pitch")):
        say(f"[chain]   {lab} saturation 450 ms window {q3([r_['sat_w'][c] for r_ in rows])} | handoff window "
            f"{q3([r_['sat_h'][c] if r_['sat_h'] is not None else None for r_ in rows])}")
    cnt = {k: sum(r_["cls"] == k for r_ in rows) for k in ("C1", "C2", "C3", "unreadable")}
    say(f"[chain] classes over {m} valid envs: " + ", ".join(f"{k} {v} ({v / m:.0%})" for k, v in cnt.items())
        + "  (C1 consistent with the waist-limit hypothesis, C2 timing difference observed; descriptive only)")
    if args.dump:
        np.savez(args.dump, env=[r_["env"] for r_ in rows], cls=[r_["cls"] for r_ in rows],
                 pk=np.array([[p[:2] for p in r_["pk"]] for r_ in rows]),
                 ok=np.array([[p[2] for p in r_["pk"]] for r_ in rows]),
                 sat_w=np.array([r_["sat_w"] for r_ in rows]),
                 sat_h=np.array([r_["sat_h"] if r_["sat_h"] is not None else [np.nan, np.nan] for r_ in rows]),
                 sep_st=[r_["sep_st"] for r_ in rows], sep_max=[r_["sep_max"] for r_ in rows])


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        say(traceback.format_exc())
        raise
    finally:
        app.close()
