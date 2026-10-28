"""Joint ONE OUTS scene smoke run (batter session, M2 prep 10-28): P3c pitcher + the live RL batter in one scene.

One process = one pitch per env (the first-after-load reset; no env is reset again: the pitcher's dones are disabled,
as in eval_stage). Prints one row per env: clip, pitch speed, plate crossing (plate frame), swing outcome (hit, exit
speed, launch, spray, carry), falls of both players, and the batter's start / hand-off timing.

    python scripts/joint_smoke.py --headless --pitcher <P3c model.pt> --batter <model_12500.pt> --envs 64 --csv logs/joint_smoke.csv
"""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pitcher", type=str, required=True)
parser.add_argument("--batter", type=str, required=True)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=6.0, help="joint episode length (release ~1 s + flight + batted ball)")
parser.add_argument("--release_csv", type=str, default="assets/game/p3c_identical_t18_release.csv",
                    help="the batter's predicted pitch before the hand-off (the P3c gate table)")
parser.add_argument("--csv", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--after_done", choices=("policy", "hold", "blend"), default="policy",
                    help="the batter once its training episode would have ended (BatterSide after_done)")
parser.add_argument("--zero_ff", action="store_true", help="cfg.zero_ff_after_ref_end: no velocity feed-forward past the clip end")
parser.add_argument("--timing_csv", type=str, default=None,
                    help="two-pass oracle timing: per env handoff_t, t_plane of this seed's pitch (scripts/make_timing_csv.py)")
parser.add_argument("--dof_only", action="store_true", help="exit right after the --dof_dump readout")
parser.add_argument("--dof_dump", type=str, default=None, help="json: the batter's DOF / body properties as PhysX reports them")
parser.add_argument("--trace", type=str, default=None,
                    help="npz: per env (first per clip) obs / action / body joints per control step from its start; perception noise 0")
parser.add_argument("--diag_yaw", type=float, default=None, help="diagnostic: batter frame yaw (deg) instead of 162")
parser.add_argument("--diag_flat", action="store_true", help="diagnostic: no mound (field_z 0)")
parser.add_argument("--no_handoff", action="store_true", help="control: the batter plays its own predicted pitch (native)")
parser.add_argument("--no_shift_hist", action="store_true", help="BatterSide shift_hist off: the percept sees the release-point jump")
parser.add_argument("--timing_bias", type=float, default=None, help="batter cfg.timing_bias_s (default 0.055)")
parser.add_argument("--diag", type=str, default=None, help="fall diag CSV: per env first fall time + root / foot heights by phase")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.one_outs.batter_side import JOINT_CLIPS, BatterSide  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env_cfg import PitchedBattingHumanPlayEnvCfg  # noqa: E402

TASK = "AIB-OneOutsJoint-v0"


def main():
    assert os.environ.get("AIB_EFFORT_SCALE") == "1.5"
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    cfg.episode_length_s = args.seconds
    cfg.ball_after_handoff = "park"
    if args.diag_flat:
        cfg.mound_height = 0.0
    if hasattr(cfg, "landed_start_prob"):
        cfg.landed_start_prob = 0.0
    if hasattr(cfg, "rsi_prob"):
        cfg.rsi_prob = 0.0
    if hasattr(cfg, "assist_force_n"):
        cfg.assist_force_n = 0.0
    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():  # one pitch per env: record falls, never reset (a reset would restart the batter too)
        orig()
        z = torch.zeros_like(base.fallen)
        return z, z
    base._get_dones = keep_running

    bcfg = PitchedBattingHumanPlayEnvCfg()
    bcfg.pitch_release_csv = args.release_csv
    bcfg.scene.num_envs = args.envs
    bcfg.zero_ff_after_ref_end = args.zero_ff
    if args.trace:  # deterministic percept and placement, as the native trace (diag_bat_contact --trace)
        for k in ("ball_obs_noise_std", "ball_obs_bias_std", "tee_xy_noise", "tee_height_noise", "joint_pos_noise"):
            setattr(bcfg, k, 0.0)
    own = None
    if args.timing_csv:
        tr = sorted(csv.DictReader(open(args.timing_csv)), key=lambda r: int(r["env"]))
        assert [int(r["env"]) for r in tr] == list(range(args.envs)), args.timing_csv
        own = [[float(r["handoff_t"]), float(r["t_plane"])] for r in tr]
    side = BatterSide(base, bcfg, args.batter, after_done=args.after_done, own_timing=own,
                      timing_bias=args.timing_bias, shift_hist=not args.no_shift_hist,
                      no_handoff=args.no_handoff)
    if args.diag_yaw is not None:  # before the first reset places the batter
        from isaaclab.utils.math import quat_from_euler_xyz
        z = torch.zeros(base.num_envs, device=base.device)
        side.robot.q_r = quat_from_euler_xyz(z, z, torch.full_like(z, args.diag_yaw * torch.pi / 180.0))
    if args.dof_dump:
        from dof_readout import dump
        dump(base.batter, base.sim, args.dof_dump, "joint")
        if args.dof_only:
            env.close()
            return
    side.record_swing()  # timing / geometry columns (diag_bat_contact's definitions)
    if args.trace:
        side.trace = []
    print(f"[joint] batter after_done {args.after_done}, zero_ff_after_ref_end {args.zero_ff}, "
          f"timing {'own pitch (' + args.timing_csv + ')' if own else 'median-flight row'}, bias {side.cfg.timing_bias_s}, shift_hist {not args.no_shift_hist}, no_handoff {args.no_handoff}", flush=True)
    base.batter_side = side
    w = RslRlVecEnvWrapper(env)  # its reset = the first-after-load reset, now with the batter's own
    ag = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.pitcher)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    obs = w.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    n = base.num_envs
    T = int(round(cfg.episode_length_s / base.step_dt)) - 2
    p_fell = torch.zeros(n, dtype=torch.bool, device=base.device)
    b_fell = torch.zeros_like(p_fell)
    b_fell_pre = torch.zeros_like(p_fell)
    hit = torch.zeros_like(p_fell)
    start_at = torch.full((n,), float("nan"), device=base.device)
    nan = lambda: torch.full((n,), float("nan"), device=base.device)  # noqa: E731
    fall_t, cross_t = nan(), nan()
    feet = base.batter.find_bodies(["l_foot", "r_foot"])[0]
    gz = float(base.field_z)
    marks = {}  # phase -> (root z, lowest foot z) above the world ground, per env
    env0 = []

    def heights():
        d = base.batter.data
        return d.root_pos_w[:, 2] - gz, d.body_link_pos_w[:, feet, 2].min(-1).values - gz

    def mark(name, m):
        if m.any():
            rz, fz = heights()
            old = marks.setdefault(name, (nan(), nan()))
            old[0][m], old[1][m] = rz[m], fz[m]
    mark("t0", torch.ones(n, dtype=torch.bool, device=base.device))
    print(f"[joint] {n} envs, {T} pitcher steps ({cfg.episode_length_s} s), batter start t "
          f"{side.start_t.min():.3f}..{side.start_t.max():.3f} s, release_t_pred {base.release_t_pred[0]:.3f} s", flush=True)
    with torch.inference_mode():
        for _ in range(T):
            was = side.active.clone()
            was_done = side.done.clone()
            obs, _, _, _ = w.step(pol(obs))
            now = base.episode_length_buf.float() * base.step_dt
            start_at = torch.where(side.active & ~was, now, start_at)
            if args.diag:
                mark("t0.4", torch.full((n,), abs(float(now[0]) - 0.4) < 0.5 * base.step_dt, device=base.device))
                mark("start", side.active & ~was)
                mark("handoff", torch.isnan(marks.get("handoff", (nan(),))[0]) & side.handed)
                crossed = torch.isnan(cross_t) & ~torch.isnan(base.j_cross[:, 0])
                mark("cross", crossed)
                cross_t = torch.where(crossed, now, cross_t)
                fell = torch.isnan(fall_t) & side.fallen
                fall_t = torch.where(fell, now, fall_t)
                if float(now[0]) <= 1.5 + 1e-6 and int(round(float(now[0]) / base.step_dt)) % 40 == 0:
                    env0.append((round(float(now[0]), 3), round(float(heights()[0][0]), 3)))
            p_fell |= base.fallen
            b_fell |= side.fallen & side.active
            b_fell_pre |= side.fallen & side.active & ~was_done  # inside the training episode (native window)
            hit |= side.hit
    if args.dof_dump:  # again after the run (anything written to the sim later)
        dump(base.batter, base.sim, args.dof_dump.replace(".json", "_end.json"), "joint end")
    if args.trace:
        import numpy as np
        names_t = [os.path.splitext(os.path.basename(f))[0] for f in side.cfg.motion_files]
        tr, out_t = side.trace, {}
        for e in range(n):
            c = names_t[int(side.clip[e])]
            if c in out_t:
                continue
            ks = [i for i, s in enumerate(tr) if bool(s["active"][e])][:150]
            out_t[c] = {f: np.stack([tr[i][f][e].cpu().numpy() for i in ks]) for f in ("obs", "act", "q", "k")}
        act = np.stack([s["active"].cpu().numpy() for s in tr])  # every env's first 12 active steps (n, 12, D)
        all_t = {"all__obs": np.stack([np.stack([tr[i]["obs"][e].cpu().numpy() for i in np.flatnonzero(act[:, e])[:12]])
                                       for e in range(n)]),
                 "all__clip": np.array([names_t[int(c)] for c in side.clip])}
        np.savez(args.trace, **all_t, **{f"{c}__{f}": v for c, d in out_t.items() for f, v in d.items()})
        print(f"[joint] trace -> {args.trace}: {sorted(out_t)} x {[len(d['k']) for d in out_t.values()]} steps", flush=True)
    rows = {}
    for rec in side.play_log:  # first batted ball per env
        rows.setdefault(int(rec["env"]), rec)
    out = []
    names = list(JOINT_CLIPS)
    R = side.rec

    def swing_cols(e):  # + delta: the swing reference reaches contact before the ball does (early)
        t_pl = R["t_plane"][e]
        return dict(t_plane=t_pl, ref_err=R["ref_err"][e], delta_s=R["ref_err"][e] / side.rate[e],
                    t_sched=t_pl - R["ref_err"][e] / side.rate[e], timing_s=R["t_square"][e] - t_pl,
                    gap_m=R["gap"][e], t_gap=R["t_gap"][e], dz_m=R["dz"][e], dperp_m=R["dperp"][e],
                    ss_speed_max=R["ss_speed_max"][e],
                    tempo_pre=R["tempo_pre"][e] / R["n_pre"][e].clamp_min(1.0),  # mean tempo action before the hand-off
                    tempo_fly=R["tempo_fly"][e] / R["n_fly"][e].clamp_min(1.0),
                    t_commit=R["t_commit"][e], perc_z_pre=perc_pre[e], perc_z_commit=perc_commit[e])  # hand-off -> done (diag_bat_contact tempo_q)
    if R["perc"]:  # perceived crossing z at the last control step before t_plane - 0.10 s and before the commit
        pt = torch.stack([p[0] for p in R["perc"]], 0)  # (steps, envs)
        pz = torch.stack([p[1] for p in R["perc"]], 0)

        def perc_at(tq):
            k = ((pt <= tq.unsqueeze(0)).sum(0) - 1).clamp_min(0)
            v = pz.gather(0, k.unsqueeze(0)).squeeze(0)
            return torch.where(torch.isfinite(tq), v, torch.full_like(v, float("nan")))
        tp_ref = side.own[:, 1] if side.own is not None else R["t_plane"]
        perc_pre, perc_commit = perc_at(tp_ref - 0.10), perc_at(R["t_commit"])
    else:
        perc_pre = perc_commit = torch.full((n,), float("nan"), device=base.device)
    for e in range(n):
        rec = rows.get(e, {})
        jc = base.j_cross[e].tolist()  # (y - zone centre, z above the field); NaN when the bat met it first
        out.append(dict(
            env=e, clip=names[e % len(names)], kmh=round(float(side.pitch_kmh[e]), 1),
            plate_y=round(jc[0], 3), plate_z=round(jc[1], 3),
            handoff_t=round(float(base.handoff_t[e]), 4), batter_start=round(float(start_at[e]), 4),
            handed=bool(side.handed[e]), hit=bool(hit[e]), exit_mps=round(rec.get("exit_velo_mps", float("nan")), 2),
            launch_deg=round(rec.get("launch_deg", float("nan")), 1), spray_deg=round(rec.get("spray_deg", float("nan")), 1),
            carry_m=round(rec.get("carry_m", float("nan")), 1), batter_fell=bool(b_fell[e]), pitcher_fell=bool(p_fell[e]),
            backspin_rpm=round(rec.get("backspin_rpm", float("nan")), 0),
            batter_fell_before_done=bool(b_fell_pre[e]), done_t=round(float(side.done_t[e]), 4),
            pred_y=round(float(side.pitch_plate[e, 0]), 4), pred_z=round(float(side.pitch_plate[e, 1]), 4),
            flight_pred=round(float(side.flight_time[e]), 4),
            handoff_jump_m=round(float(side.handoff_jump[e]), 4),  # live release vs the predicted pre-release ball  # fly_to_x at the hand-off: release to the contact plane
            **{k: round(float(v), 4) for k, v in swing_cols(e).items()}))
    for o in out:
        print("[joint] " + " ".join(f"{k}={v}" for k, v in o.items()), flush=True)
    late = sum(1 for o in out if o["handed"] and not (o["batter_start"] <= o["handoff_t"]))
    hr = sum(o["hit"] for o in out) / n
    print(f"[joint] summary: handed {sum(o['handed'] for o in out)}/{n}, hits {hr:.3f}, batter falls "
          f"{sum(o['batter_fell'] for o in out)}, pitcher falls {sum(o['pitcher_fell'] for o in out)}, released before "
          f"the batter's start {late}", flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(out[0]))
            wr.writeheader()
            wr.writerows(out)
    if args.diag:
        clip_end = side.start_t + (side.ref.duration[side.clip] - side.t0) / side.speed  # joint time of the ref's end
        rows, ph = [], {"hold": 0, "start_to_handoff": 0, "after_handoff": 0, "after_clip_end": 0, "never": 0}
        win = {"before_done": 0, "after_done": 0}  # the training episode's end: pitch passed / ball landed / fallen
        for e in range(n):
            ft, st, ho, ce = (float(fall_t[e]), float(start_at[e]), float(base.handoff_t[e]), float(clip_end[e]))
            p = ("never" if ft != ft else "hold" if not ft >= st else "start_to_handoff" if ft < ho
                 else "after_handoff" if ft < ce else "after_clip_end")
            ph[p] += 1
            dt_ = float(side.done_t[e])
            if ft == ft:
                win["before_done" if not ft > dt_ else "after_done"] += 1
            row = dict(env=e, clip=names[e % len(names)], fall_t=round(ft, 4), batter_start=round(st, 4), handoff_t=round(ho, 4),
                       clip_end=round(ce, 4), phase=p, fall_minus_start=round(ft - st, 4), fall_minus_handoff=round(ft - ho, 4),
                       hit=bool(hit[e]), done_t=round(dt_, 4))
            for k, (rz, fz) in marks.items():
                row[f"root_z_{k}"], row[f"foot_z_{k}"] = round(float(rz[e]), 3), round(float(fz[e]), 3)
            rows.append(row)
        with open(args.diag, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"[diag] falls by phase: {ph}; vs the training episode's end (done_t median "
              f"{side.done_t.nanmedian():.3f} s): {win}", flush=True)
        for k, (rz, fz) in marks.items():
            print(f"[diag] {k}: root z {rz.nanmedian():.3f} ({rz.nan_to_num(9).min():.3f}..{rz.nan_to_num(-9).max():.3f}), "
                  f"lowest foot z {fz.nanmedian():.3f} ({fz.nan_to_num(9).min():.3f}..{fz.nan_to_num(-9).max():.3f}) m above field_z {gz}", flush=True)
        ok = ~torch.isnan(fall_t)
        if ok.any():
            print(f"[diag] fall t median {fall_t[ok].median():.3f} s, minus start {(fall_t - start_at)[ok].median():.3f}, "
                  f"minus hand-off {(fall_t - base.handoff_t)[ok].median():.3f}, clip end median {clip_end.median():.3f}", flush=True)
        print(f"[diag] env 0 root z (10 Hz, first 1.5 s): {env0}", flush=True)
    print("JOINT_SMOKE_END", flush=True)


main()
app.close()
