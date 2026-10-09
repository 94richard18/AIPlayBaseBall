"""All pitching diagnostics in one simulator run (robot vs athlete).

One pass of full pitches records every body / joint / contact / ball state, then prints:
  1. outcome      release time / speed, strikes, falls, tracking loss (termination reasons)
  2. footing      feet on the ground vs the athlete's schedule, COM error, capture point, slip (per phase)
  3. foot strike  first lead-foot contact: impact force, descent speed, bounce, knee torque flips
  4. mechanics    hip-shoulder separation, arm launch position, pelvis opening, pivot knee, trunk tilt (strike / release)
  5. chain        peak-speed timing of pelvis, trunk, shoulder IR, elbow, hand (velocity principle)
  6. energy       segment energies (legs+pelvis, trunk, arm, ball), energy flow into the arm, joint powers,
                  ground reaction force from the COM (braking / vertical, body weights)
  7. saturation   share of time joints sit at their torque limit (stride, strike -> release, after release)

    python scripts/diag_pitch.py --headless --checkpoint <pitch model.pt> [--envs 16] [--sections 1,4,6]
    what-ifs: --zero (no policy, reference PD only), --ref_after_release, --leg_scale 2.0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--cfg", choices=["elastic", "stand_first", "speed"], default="stand_first")
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=2.7, help="simulated time; must cover release + 1.5 s")
parser.add_argument("--sections", type=str, default="1,2,3,4,5,6,7,8")
parser.add_argument("--zero", action="store_true", help="what-if: no policy, pure reference PD tracking")
parser.add_argument("--ref_after_release", action="store_true", help="what-if: zero residual actions after the release")
parser.add_argument("--leg_scale", type=float, default=1.0, help="what-if: scale hip/knee/ankle torque limits")
parser.add_argument("--leg_stiffness", type=float, default=1.0, help="what-if: scale hip/knee/ankle PD stiffness")
parser.add_argument("--depen", type=float, default=None, help="what-if: PhysX max depenetration velocity (m/s, robot)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.analysis import pitch_report as PR  # noqa: E402
from aibaseball.tasks.pitching import pitch_env_cfg as C  # noqa: E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402

SECTIONS = {int(s) for s in args.sections.split(",")}


def say(*a):
    print(*a, flush=True)


def ext(a, fn):
    """min / max of a window, NaN when the window is empty (e.g. a pitch released before its foot strike)."""
    return float(fn(a)) if a.size else float("nan")


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def main():
    cfg = {"elastic": C.PitchElasticEnvCfg, "stand_first": C.PitchStandFirstEnvCfg, "speed": C.PitchSpeedEnvCfg}[args.cfg]()
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    if args.leg_scale != 1.0:
        for g in ("hip", "knee", "ankle"):
            cfg.robot.actuators[g].effort_limit_sim *= args.leg_scale
    if args.leg_stiffness != 1.0:
        for g in ("hip", "knee", "ankle"):
            cfg.robot.actuators[g].stiffness *= args.leg_stiffness
    if args.depen is not None:
        cfg.robot.spawn.rigid_props.max_depenetration_velocity = args.depen
    env = gym.make("AIB-PitchElastic-v0", cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():  # observe everything; record when training would have ended the episode
        orig()
        return torch.zeros_like(base.released), base.episode_length_buf >= base.max_episode_length - 1
    base._get_dones = keep_running
    w = RslRlVecEnvWrapper(env)
    ag = PitchPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    rob = base.robot
    names = list(rob.body_names)
    jn = list(rob.joint_names)
    n, B, J = base.num_envs, len(names), len(jn)
    dt = base.step_dt
    T = int(args.seconds / dt)
    f32 = np.float32
    rec = dict(p=np.zeros((T, n, B, 3), f32), q=np.zeros((T, n, B, 4), f32), v=np.zeros((T, n, B, 3), f32),
               w=np.zeros((T, n, B, 3), f32), lp=np.zeros((T, n, B, 3), f32), lq=np.zeros((T, n, B, 4), f32),
               jq=np.zeros((T, n, J), f32), jv=np.zeros((T, n, J), f32), tau=np.zeros((T, n, J), f32),
               ball_p=np.zeros((T, n, 3), f32), ball_v=np.zeros((T, n, 3), f32), feet_f=np.zeros((T, n, 2), f32),
               rel=np.zeros((T, n), bool), fallen=np.zeros((T, n), bool), lost=np.zeros((T, n), bool),
               ref_t=np.zeros((T, n), f32), ref_contact=np.zeros((T, n, 2), bool), com_err=np.zeros((T, n), f32))
    effort = rob.root_physx_view.get_dof_max_forces().cpu().numpy()[0]
    mass = rob.root_physx_view.get_masses().cpu().numpy()[0]
    inertia = rob.root_physx_view.get_inertias().cpu().numpy()[0].reshape(B, 3, 3)
    o = base.scene.env_origins
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            act = pol(obs)
            if args.zero:
                act = torch.zeros_like(act)
            elif args.ref_after_release:
                act = torch.where(base.released.unsqueeze(-1), torch.zeros_like(act), act)
            obs, _, _, _ = w.step(act)
            d = rob.data
            rec["p"][k] = (d.body_com_pos_w - o.unsqueeze(1)).cpu().numpy()
            rec["q"][k] = d.body_com_quat_w.cpu().numpy()
            rec["v"][k] = d.body_com_lin_vel_w.cpu().numpy()
            rec["w"][k] = d.body_ang_vel_w.cpu().numpy()
            rec["lp"][k] = (d.body_link_pos_w - o.unsqueeze(1)).cpu().numpy()
            rec["lq"][k] = d.body_link_quat_w.cpu().numpy()
            rec["jq"][k], rec["jv"][k] = d.joint_pos.cpu().numpy(), d.joint_vel.cpu().numpy()
            rec["tau"][k] = d.applied_torque.cpu().numpy()
            rec["ball_p"][k] = (base.ball.data.root_pos_w - o).cpu().numpy()
            rec["ball_v"][k] = base.ball.data.root_lin_vel_w.cpu().numpy()
            rec["feet_f"][k] = base.feet.data.net_forces_w_history[:, :, base.foot_sensor_ids].norm(dim=-1).max(1).values.cpu().numpy()
            rec["rel"][k], rec["fallen"][k], rec["lost"][k] = (base.released.cpu().numpy(), base.fallen.cpu().numpy(),
                                                               base._lost.cpu().numpy())
            tr = base._ref_time()
            rc, _, rcon = base._ref_centroid(tr)
            com, _ = base._com()
            rec["ref_t"][k], rec["ref_contact"][k] = tr.cpu().numpy(), rcon.cpu().numpy()
            rec["com_err"][k] = (com - rc).norm(dim=-1).cpu().numpy()
    t = (np.arange(T) + 1) * dt
    ref = PR.reference_profile(cfg)
    bid = {nm: i for i, nm in enumerate(names)}
    first = lambda mask: np.where(mask.any(0), t[np.argmax(mask, 0)], np.nan)  # noqa: E731
    t_rel = first(rec["rel"])
    t_fall = first(rec["fallen"])
    t_lost = first(rec["lost"] & ~rec["rel"])
    lead_on = rec["feet_f"][:, :, 0] > cfg.contact_force_n
    t_strike = first(lead_on & (t[:, None] > 0.6))
    i_rel = [int(np.argmax(rec["rel"][:, i])) if rec["rel"][:, i].any() else None for i in range(n)]
    ball_kmh = [float(np.linalg.norm(base.rel_vel[i].cpu().numpy())) * 3.6 if i_rel[i] is not None else None for i in range(n)]
    out_strike = [bool(base._release_outcome(torch.tensor([i], device=base.device))["strike"][0]) if i_rel[i] is not None
                  else False for i in range(n)]
    tag = " [what-if: " + ", ".join(k for k in ("zero", "ref_after_release") if getattr(args, k)) + \
          (f" leg torque x{args.leg_scale}" if args.leg_scale != 1.0 else "") +           (f" leg stiffness x{args.leg_stiffness}" if args.leg_stiffness != 1.0 else "") +           (f" depenetration {args.depen} m/s" if args.depen is not None else "") + "]"         if (args.zero or args.ref_after_release or args.leg_scale != 1.0 or args.leg_stiffness != 1.0
            or args.depen is not None) else ""
    say(f"== diag_pitch: {args.checkpoint}{tag} | {n} full pitches, cfg {args.cfg}")

    # ---------------------------------------------------------------- 1 outcome
    if 1 in SECTIONS:
        say("\n[1] OUTCOME")
        say(f"  release: athlete {ref['release']:.3f} s | robot median {med(t_rel):.3f} s ({int(np.isfinite(t_rel).sum())}/{n} released), "
            f"{med(ball_kmh):.1f} km/h (min {min([b for b in ball_kmh if b] or [0]):.0f}, max {max([b for b in ball_kmh if b] or [0]):.0f}), "
            f"strikes {sum(out_strike)}/{n}")
        # "up 1.5 s after the release" only counts pitches observed that long (the first version simulated 1.6 s and
        # counted pitches that had not fallen ~0.5 s after the release as standing)
        seen = [i for i in range(n) if np.isfinite(t_rel[i]) and t_rel[i] + 1.5 <= t[-1]]
        up = [np.isnan(t_fall[i]) or t_fall[i] - t_rel[i] >= 1.5 for i in seen]
        after = [t_fall[i] - t_rel[i] for i in range(n) if np.isfinite(t_rel[i]) and np.isfinite(t_fall[i])]
        short = int(np.isfinite(t_rel).sum()) - len(seen)
        say(f"  falls: {int(np.isfinite(t_fall).sum())}/{n}; still up 1.5 s after the release: {sum(up)}/{len(up)}"
            + (f" ({short} released too late to be observed 1.5 s - use a longer --seconds)" if short else "")
            + (f"; fall median {med(after):+.2f} s after the release" if after else ""))
        say(f"  tracking lost before the release (training would end the episode): {int(np.isfinite(t_lost).sum())}/{n}"
            + (f", median at {med(t_lost):.2f} s" if np.isfinite(t_lost).any() else ""))

    # ---------------------------------------------------------------- 2 footing
    if 2 in SECTIONS:
        say("\n[2] FOOTING (share of envs; feet on the ground vs the athlete's schedule)")
        feet_on = rec["feet_f"] > cfg.contact_force_n
        phases = [("leg lift / stride (0.1 s - strike)", lambda i: (t > 0.1) & (t < t_strike[i])),
                  ("strike -> release", lambda i: (t >= t_strike[i]) & (t < t_rel[i])),
                  ("after the release (0.5 s)", lambda i: (t >= t_rel[i]) & (t < t_rel[i] + 0.5))]
        for label, sel in phases:
            match, cerr, slip = [], [], []
            for i in range(n):
                if not (np.isfinite(t_strike[i]) and np.isfinite(t_rel[i])):
                    continue
                m = sel(i)
                if not m.any():
                    continue
                match.append((feet_on[m, i] == rec["ref_contact"][m, i]).mean())
                cerr.append(rec["com_err"][m, i].mean())
                fv = np.linalg.norm(rec["v"][m, i][:, [bid["l_foot"], bid["r_foot"]], :2], axis=-1)
                slip.append((fv * feet_on[m, i]).sum(-1).mean())
            say(f"  {label:36s} contact match {med(match) * 100:5.1f}% | COM error {med(cerr):.2f} m | planted-foot slip {med(slip):.2f} m/s")
        say(f"  lead-foot strike: athlete {ref['strike']:.3f} s | robot median {med(t_strike):.3f} s")
        # after the release, foot by foot: is the lead foot still planted, when does the back foot come down?
        lead_held, back_land, back_land_ref = [], [], []
        rc = rec["ref_contact"]
        for i in range(n):
            if not np.isfinite(t_rel[i]):
                continue
            m = (t >= t_rel[i]) & (t < t_rel[i] + 0.5)
            ref_lead = m & rc[:, i, 0]
            if ref_lead.any():
                lead_held.append(feet_on[ref_lead, i, 0].mean())
            later = t > t_rel[i] + 0.05
            on = later & feet_on[:, i, 1]
            back_land.append(t[np.argmax(on)] - t_rel[i] if on.any() else np.nan)
            on_ref = later & rc[:, i, 1]
            back_land_ref.append(t[np.argmax(on_ref)] - t_rel[i] if on_ref.any() else np.nan)
        say(f"  after the release: lead foot planted while the athlete's is {med(lead_held) * 100:.0f}% of the time; "
            f"back foot lands {med(back_land):+.2f} s after the release (reference {med(back_land_ref):+.2f} s, "
            f"{int(np.isfinite(back_land).sum())}/{len(back_land)} land before the episode ends)")

    # ---------------------------------------------------------------- 3 foot strike
    if 3 in SECTIONS:
        say("\n[3] LEAD-FOOT STRIKE (first contact after 0.6 s)")
        jk, ja = jn.index("l_knee"), jn.index("l_ankle_pitch")
        imp, vz, bounce, flips, ank_sat = [], [], [], [], []
        for i in range(n):
            if not np.isfinite(t_strike[i]):
                continue
            k0 = int(np.argmax(t >= t_strike[i]))
            win = slice(k0, min(T, k0 + int(0.15 / dt)))
            imp.append(rec["feet_f"][win, i, 0].max())
            vz.append(rec["v"][max(0, k0 - 2), i, bid["l_foot"], 2])
            bounce.append(rec["v"][win, i, bid["l_foot"], 2].max())
            s = np.sign(rec["tau"][win, i, jk])
            flips.append(int((np.abs(np.diff(s)) > 1).sum()))
            ank_sat.append((np.abs(rec["tau"][win, i, ja]) >= 0.98 * effort[ja]).mean())
        bw = mass.sum() * PR.G
        say(f"  descent speed at contact {med(vz):+.2f} m/s | peak contact force {med(imp):.0f} N ({med(imp) / bw:.1f} body weights)")
        say(f"  lead foot max upward speed in the next 0.15 s {med(bounce):+.2f} m/s (bounce if > 0.3) | "
            f"knee torque sign flips {med(flips):.0f} | ankle at its torque limit {med(ank_sat) * 100:.0f}% of the time")

    # ---------------------------------------------------------------- 4 mechanics / 5 chain
    def body_rot(key, i):
        q = rec["lq"][:, i, bid[key]]
        return Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], -1)).as_matrix()

    series = {}
    for i in range(n):
        if not (np.isfinite(t_strike[i]) and np.isfinite(t_rel[i])):
            continue
        Rp, Rt = body_rot("pelvis", i), body_rot("torso", i)
        sh, hand = rec["lp"][:, i, bid["r_shoulder_pitch_link"]], rec["lp"][:, i, bid["r_hand"]]
        series[i] = dict(mech=PR.mechanics_series(Rp, Rt, sh, hand, rec["jq"][:, i, jn.index("r_knee")]),
                         chain=PR.chain_series(Rp, Rt, rec["jv"][:, i, jn.index("r_shoulder_yaw")],
                                               rec["jv"][:, i, jn.index("r_elbow")],
                                               np.linalg.norm(rec["v"][:, i, bid["r_hand"]], axis=-1), dt))
    if 4 in SECTIONS and series:
        say("\n[4] MECHANICS                                              athlete |  robot (median)")
        for label, ev_ref, ev_rob in (("FOOT STRIKE", ref["strike"], t_strike), ("RELEASE", ref["release"], t_rel)):
            say(f"  --- at {label}")
            for key in ref["mech"]:
                a = PR.at(ref["t"], ref["mech"][key], ev_ref)
                b = med([PR.at(t, s["mech"][key], ev_rob[i]) for i, s in series.items()])
                say(f"  {key:54s} {a:+7.2f} | {b:+7.2f}")
        sep_r = ref["mech"]["hip-shoulder separation (deg)"]
        m = (ref["t"] > 0.5) & (ref["t"] < ref["release"] + 0.05)
        j = int(np.argmax(np.abs(sep_r) * m))
        pk = [(abs(s["mech"]["hip-shoulder separation (deg)"][int(np.argmax(np.abs(s["mech"]["hip-shoulder separation (deg)"])
                * ((t > 0.5) & (t < t_rel[i] + 0.05))))]),
               t[int(np.argmax(np.abs(s["mech"]["hip-shoulder separation (deg)"]) * ((t > 0.5) & (t < t_rel[i] + 0.05))))]
               - t_strike[i]) for i, s in series.items()]
        say(f"  max hip-shoulder separation: athlete {abs(sep_r[j]):.0f} deg at {(ref['t'][j] - ref['strike']) * 1000:+.0f} ms "
            f"from strike | robot {med([p[0] for p in pk]):.0f} deg at {med([p[1] for p in pk]) * 1000:+.0f} ms")
    if 5 in SECTIONS and series:
        say("\n[5] KINETIC CHAIN (peak speed, time vs each one's release)   athlete            | robot (median)")
        rp = PR.chain_peaks(ref["t"], ref["chain"], ref["release"])
        bp = [PR.chain_peaks(t, s["chain"], t_rel[i]) for i, s in series.items()]
        for j, nm in enumerate(PR.SEGMENTS):
            say(f"  {nm:24s} {rp[j][0] * 1000:+5.0f} ms {rp[j][1]:6.0f} {PR.SEG_UNITS[j]:5s} | "
                f"{med([p[j][0] for p in bp]) * 1000:+5.0f} ms {med([p[j][1] for p in bp]):6.0f}")
        say("  order athlete: " + " -> ".join(PR.SEGMENTS[i] for i in np.argsort([p[0] for p in rp])))
        say("  order robot:   " + " -> ".join(PR.SEGMENTS[i] for i in np.argsort([med([p[j][0] for p in bp]) for j in range(5)])))

    # ---------------------------------------------------------------- 6 energy
    if 6 in SECTIONS and series:
        say("\n[6] ENERGY FLOW (same method for both: segment kinetic + potential energy, robot masses)")
        arm_internal = [jn.index(x) for x in jn if x.startswith("r_") and any(k in x for k in ("elbow", "wrist", "index", "middle",
                                                                                                   "ring", "pinky", "thumb"))]
        res = []
        for i in series:
            q = rec["q"][:, i]
            R = Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1).reshape(-1, 4)).as_matrix().reshape(T, B, 3, 3)
            Ek, Ep = PR.segment_energy(mass, inertia, R, rec["p"][:, i], rec["v"][:, i], rec["w"][:, i])
            bv = rec["ball_v"][:, i]
            ball_e = 0.5 * PR.BALL_MASS * (bv ** 2).sum(-1) + PR.BALL_MASS * PR.G * rec["ball_p"][:, i, 2]
            internal = (rec["tau"][:, i, arm_internal] * rec["jv"][:, i, arm_internal]).sum(-1)
            E, rate, flow = PR.energy_flow(t, names, Ek, Ep, ball_e, rec["rel"][:, i], dt, internal)
            com = (rec["p"][:, i] * mass[None, :, None]).sum(1) / mass.sum()
            grf = PR.grf_from_com(PR.smooth(com, dt, 0.08, deriv=2), mass.sum())
            res.append((i, E, rate, flow, grf))

        def group_peaks(tt, E, rel, strike):
            m = (tt > strike - 0.3) & (tt <= rel + 0.02)
            if not m.any():
                return [(np.nan, np.nan)] * E.shape[1]
            return [(E[m, j].max() - E[0, j], tt[m][np.argmax(E[m, j])] - rel) for j in range(E.shape[1])]

        rg = group_peaks(ref["t"], ref["E"], ref["release"], ref["strike"])
        bg = [group_peaks(t, E, t_rel[i], t_strike[i]) for i, E, *_ in res]
        say("  energy peak above the set position (J), time vs release:       athlete          | robot (median)")
        for j, g in enumerate(PR.GROUPS):
            say(f"  {g:14s} {rg[j][0]:7.0f} J at {rg[j][1] * 1000:+5.0f} ms | {med([b[j][0] for b in bg]):7.0f} J at "
                f"{med([b[j][1] for b in bg]) * 1000:+5.0f} ms")
        mr = (ref["t"] > ref["strike"] - 0.2) & (ref["t"] <= ref["release"])
        ar = ref["arm_rate"]
        say(f"  arm + ball energy rate, peak (W): athlete {ar[mr].max():.0f} at {(ref['t'][mr][np.argmax(ar[mr])] - ref['release']) * 1000:+.0f} ms"
            f" | robot {med([ext(rate[(t > t_strike[i] - 0.2) & (t <= t_rel[i])], np.max) for i, E, rate, flow, g in res]):.0f}")
        say(f"  of it through the shoulder from the trunk (robot; minus elbow/wrist/finger muscle power), peak (W): "
            f"{med([ext(flow[(t > t_strike[i] - 0.2) & (t <= t_rel[i])], np.max) for i, E, rate, flow, g in res]):.0f}; energy in, stride -> release: "
            f"{med([np.clip(flow[(t > t_strike[i] - 0.2) & (t <= t_rel[i])], 0, None).sum() * dt for i, E, rate, flow, g in res]):.0f} J")
        ball_ke_ref = 0.5 * PR.BALL_MASS * PR.at(ref["t"], ref["ball_speed"], ref["release"]) ** 2
        say(f"  ball kinetic energy at release: athlete {ball_ke_ref:.0f} J | robot {med([0.5 * PR.BALL_MASS * (b / 3.6) ** 2 for b in ball_kmh if b]):.0f} J")
        mm = (ref["t"] >= ref["strike"]) & (ref["t"] <= ref["release"])
        say(f"  ground reaction (whole body, from the COM; literature: lead foot alone ~0.75 BW braking at max ER):")
        say(f"    braking (toward the rubber) peak, strike -> release: athlete {-ref['grf'][mm, 0].min():.2f} BW | robot "
            f"{med([-ext(g[(t >= t_strike[i]) & (t <= t_rel[i]), 0], np.min) for i, E, rate, flow, g in res]):.2f} BW")
        say(f"    vertical peak, strike -> release:                    athlete {ref['grf'][mm, 2].max():.2f} BW | robot "
            f"{med([ext(g[(t >= t_strike[i]) & (t <= t_rel[i]), 2], np.max) for i, E, rate, flow, g in res]):.2f} BW")
        # joint (muscle) power: who generates / absorbs energy before the release
        groups = {"pivot hip (r)": ["r_hip_pitch", "r_hip_roll", "r_hip_yaw"], "pivot knee (r)": ["r_knee"],
                  "lead hip (l)": ["l_hip_pitch", "l_hip_roll", "l_hip_yaw"], "lead knee (l)": ["l_knee"],
                  "waist": ["waist_yaw", "waist_roll", "waist_pitch"], "throwing shoulder": ["r_shoulder_pitch",
                  "r_shoulder_roll", "r_shoulder_yaw"], "throwing elbow": ["r_elbow"]}
        say("  joint muscle power, stride -> release (robot, median): +work generated / -work absorbed, peak power")
        for label, js in groups.items():
            ix = [jn.index(x) for x in js]
            pos, neg, pk = [], [], []
            for i in series:
                m = (t > t_strike[i] - 0.3) & (t <= t_rel[i])
                pw = (rec["tau"][m, i][:, ix] * rec["jv"][m, i][:, ix]).sum(-1)
                pos.append(np.clip(pw, 0, None).sum() * dt)
                neg.append(np.clip(pw, None, 0).sum() * dt)
                pk.append(ext(pw, np.max))
            say(f"    {label:18s} +{med(pos):6.0f} J  {med(neg):+7.0f} J  peak {med(pk):6.0f} W")

    # ---------------------------------------------------------------- 7 saturation
    if 8 in SECTIONS and series:
        say("\n[8] AFTER THE RELEASE (median over envs; robot | athlete), time from each one's release")
        from aibaseball.mocap.centroidal import mound_z

        gfn = lambda x: mound_z(x, cfg.mound_top_z, cfg.mound_height, cfg.mound_slope, cfg.mound_slope_start_x)  # noqa: E731
        feet_on = rec["feet_f"] > cfg.contact_force_n
        rob_s = {}
        for i in series:
            q = rec["q"][:, i]
            R = Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1).reshape(-1, 4)).as_matrix().reshape(T, B, 3, 3)
            com = (rec["p"][:, i] * mass[None, :, None]).sum(1) / mass.sum()
            com_v = (rec["v"][:, i] * mass[None, :, None]).sum(1) / mass.sum()
            Rt = body_rot("torso", i)
            rob_s[i] = PR.after_release_series(
                t, com, com_v, rec["lp"][:, i][:, [bid["l_foot"], bid["r_foot"]]], feet_on[:, i], gfn,
                rec["lp"][:, i, bid["pelvis"], 2], rec["lp"][:, i, bid["torso"], 2] + Rt[:, 2, 2] * 0.62,
                np.degrees(np.arccos(np.clip(Rt[:, 2, 2], -1, 1))),
                PR.angular_momentum(mass, inertia, R, rec["p"][:, i], rec["v"][:, i], rec["w"][:, i]),
                rec["jq"][:, i, jn.index("l_knee")], rec["jq"][:, i, jn.index("l_hip_pitch")])
        offsets = np.arange(0.0, 0.501, 0.05)
        keys = list(next(iter(rob_s.values())).keys())
        for block in (keys[:8], keys[8:]):
            say("  dt(s) " + "".join(f"| {k[:26]:>26s} " for k in block))
            for dto in offsets:
                cells = []
                for k in block:
                    rv = med([PR.at(t, s[k], t_rel[i] + dto) for i, s in rob_s.items()])
                    av = PR.at(ref["t"], ref["after"][k], ref["release"] + dto)
                    cells.append(f"| {rv:+8.2f} | {av:+8.2f}   ")
                say(f"  {dto:+.2f} " + "".join(f"{c:>29s}" for c in cells))
        fall = [t_fall[i] - t_rel[i] for i in series if np.isfinite(t_fall[i])]
        say(f"  falls (pelvis < {cfg.fallen_pelvis_z} m or head < {cfg.fallen_head_z} m): median {med(fall):+.2f} s after the release")

    if 7 in SECTIONS and series:
        say("\n[7] TORQUE AT LIMIT (share of steps, median over envs)")
        groups = {"pivot leg": "r_(hip|knee|ankle)", "lead leg": "l_(hip|knee|ankle)", "waist": "waist_",
                  "throwing shoulder": "r_shoulder", "throwing elbow": "r_elbow"}
        import re

        phases = [("stride", lambda i: (t > 0.45) & (t < t_strike[i])), ("strike->release", lambda i: (t >= t_strike[i]) & (t < t_rel[i])),
                  ("after release", lambda i: (t >= t_rel[i]) & (t < t_rel[i] + 0.3))]
        say("  " + " " * 18 + "".join(f"{p[0]:>18s}" for p in phases))
        for label, rx in groups.items():
            ix = [k for k, x in enumerate(jn) if re.match(rx, x)]
            row = []
            for _, sel in phases:
                vals = []
                for i in series:
                    m = sel(i)
                    if m.any():
                        vals.append((np.abs(rec["tau"][m, i][:, ix]) >= 0.98 * effort[ix]).mean())
                row.append(med(vals))
            say(f"  {label:18s}" + "".join(f"{v * 100:17.0f}%" for v in row))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
