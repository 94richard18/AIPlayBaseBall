"""Evaluate a stage-1 step policy (stage_env.StageEnv): N full episodes, falls, standing at the end, key frames to
look at and a side-view video, in one simulator launch.

    python scripts/eval_stage.py --headless --task AIB-Stage1a-v0 (or 1b) --checkpoint <model.pt> [--envs 64]
        [--video videos/stage1a_iter_150.mp4] [--stochastic]
Prints "[goal] k/N" = episodes standing at rest at the end (read by train_cycles.sh).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1a-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--stochastic", action="store_true", help="sample the action noise (as in training)")
parser.add_argument("--video", type=str, default=None, help="side-view video of env 0 (+ key frames .png next to it)")
parser.add_argument("--video_episodes", type=int, default=2)
parser.add_argument("--pitch_csv", type=str, default=None,
                    help="arm + ball step: one row per episode 'speed_mps,plate_y,plate_z,crossed' for the game engine")
parser.add_argument("--set", nargs="*", default=[], help="what-if env cfg overrides, key=value")
parser.add_argument("--rsi", type=float, default=None, help="what-if: every episode starts at this reference time")
parser.add_argument("--assist", action="store_true", help="keep the training assist force (full strength)")
parser.add_argument("--zero", action="store_true", help="what-if: zero actions (PD on the set position only)")
parser.add_argument("--trace", action="store_true", help="print env 0 every 0.1 s (feet, pelvis, trunk)")
parser.add_argument("--slow", type=float, default=2.0, help="video slow-motion factor")
parser.add_argument("--gate_min_speed", type=float, default=10.0,
                    help="throw tasks: a release slower than this (m/s) is a drop and fails the [goal] count")
parser.add_argument("--gate_min_release_t", type=float, default=0.9,
                    help="throw tasks: a release before this reference time (s; athlete ~0.98 s) is a drop")
parser.add_argument("--ball_depen", type=float, default=None, help="what-if: ball max_depenetration_velocity (m/s)")
parser.add_argument("--ball_pos_iters", type=int, default=None, help="what-if: ball solver position iterations")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402


def say(*a):
    print(*a, flush=True)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    if args.ball_depen is not None:
        cfg.ball.spawn.rigid_props.max_depenetration_velocity = args.ball_depen
    if args.ball_pos_iters is not None:
        cfg.ball.spawn.rigid_props.solver_position_iteration_count = args.ball_pos_iters
    cfg.scene.num_envs = args.envs
    if hasattr(cfg, "landed_start_prob"):
        cfg.landed_start_prob = 0.0  # evaluate whole episodes from the set position
    for kv in args.set:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        if isinstance(old, bool):
            v = v.lower() in ("1", "true")
        elif old is None or v == "None" or isinstance(old, (tuple, list, dict)):
            import ast
            try:
                v = None if v == "None" else ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass  # a plain string (e.g. grip_file=grip_v2.json)
        else:
            v = type(old)(v)
        setattr(cfg, k, v)
    if args.rsi is not None:
        cfg.rsi_prob, cfg.rsi_min_s, cfg.rsi_margin_before_release = 1.0, args.rsi, 10.0  # what-if: start at this time
    elif hasattr(cfg, "rsi_prob"):
        cfg.rsi_prob = 0.0  # judged from the set position
    if hasattr(cfg, "assist_force_n") and not args.assist:
        cfg.assist_force_n = 0.0  # judged on full physics (the training assist starts at full strength in a new env)
    if args.video:
        cfg.viewer.resolution = (1280, 720)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():  # record falls, never end an episode early
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
    if args.stochastic:
        pol = lambda x: r.alg.actor_critic.act(r.obs_normalizer(x))  # noqa: E731
    T = int(round(cfg.episode_length_s / base.step_dt)) - 2
    n = base.num_envs
    fell_at = np.full(n, np.nan)
    com_dy18 = np.full(n, np.nan)
    com_dytd = np.full(n, np.nan)  # COM y - lead foot y 0.1 s after the robot's touchdown (stage 1b), and its y speed
    com_vytd = np.full(n, np.nan)  # COM y - lead foot y at 1.8 s (y + toward first base)
    air_cross = np.zeros(n, dtype=bool)  # ball crossed the plate before touching the ground (W1 tier T2+)
    grounded = np.zeros(n, dtype=bool)  # ball flight ended on the ground before any plate crossing
    prev_cr = np.zeros(n, dtype=bool)
    arm_abs, arm_n = np.zeros(n), np.zeros(n)
    rel_split = np.full((n, 3), np.nan)  # hand speed at the release: pelvis / trunk / arm parts along its direction
    grp_abs, grp_n = np.zeros(3), 0.0  # |a| over 0.9-1.0 s: legs (0-11) / waist (12-14) / arms (15-28)
    rel_yaw = np.full(n, np.nan)  # pelvis (root) yaw at the release step, deg (0 = facing +x)
    rel_jv = np.full((n, 2), np.nan)  # right elbow / shoulder yaw (internal rotation) velocity at the release, deg/s
    prev_rel = np.zeros(n, dtype=bool)
    lift_max = np.zeros(n)  # lead foot's highest point over its start in the lift window (0.2-0.95 s)
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            act = pol(obs)
            if args.zero:
                act = torch.zeros_like(act)
            if hasattr(base, "_arm_cols"):  # arm residual size over the throw (0.9-1.0 s reference time), raw action units
                tr_now = base._ref_time()
                win = ((tr_now >= 0.9) & (tr_now < 1.0)).float()
                arm_abs += (act[:, base._arm_cols].abs().mean(-1) * win).cpu().numpy()
                arm_n += win.cpu().numpy()
                if win.any():
                    aa = act[:, :29].abs()
                    for gi, (g0, g1) in enumerate(((0, 12), (12, 15), (15, 29))):
                        grp_abs[gi] += float((aa[:, g0:g1].mean(-1) * win).sum())
                    grp_n += float(win.sum())
            obs, _, _, _ = w.step(act)
            f = base.fallen.cpu().numpy()
            if args.trace and k % 40 == 0:
                dd = base.robot.data
                o = base.scene.env_origins[0]
                fz = base.feet.data.net_forces_w_history[0, :, base.foot_sensor_ids].norm(dim=-1).max(0).values
                fp = dd.body_link_pos_w[0, base.foot_ids] - o
                pel = dd.root_link_pos_w[0] - o
                b, sp, tl, hz, _ = base._stand_state()
                qerr = (dd.joint_pos[0, base.body_ids] - base.q_target[0, base.body_ids]).abs()
                say(f"t {(k + 1) * base.step_dt:.2f} feet N {fz[0]:.0f}/{fz[1]:.0f} z {fp[0, 2]:.3f}/{fp[1, 2]:.3f} "
                    f"x {fp[0, 0]:+.2f}/{fp[1, 0]:+.2f} | pelvis {pel[0]:+.2f} {pel[1]:+.2f} {pel[2]:.2f} | tilt {tl[0]:.0f} "
                    f"| com v {sp[0]:.2f} | max q err {qerr.max():.2f} rad ({base.body_names[int(qerr.argmax())]})")
            fell_at[np.isnan(fell_at) & f] = (k + 1) * base.step_dt
            if hasattr(base, "released"):
                rel_now = base.released.cpu().numpy()
                if (rel_now & ~prev_rel).any():
                    qw, qx, qy, qz = base.robot.data.root_quat_w.cpu().numpy().T
                    yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
                    rel_yaw = np.where(rel_now & ~prev_rel, yaw, rel_yaw)
                    # hand speed at the release, split along the hand's velocity: pelvis + trunk (torso rigid motion
                    # relative to the pelvis, at the hand) + arm (the rest)
                    dd = base.robot.data
                    vh = dd.body_link_lin_vel_w[:, base.hand_id]
                    vp = dd.root_link_lin_vel_w
                    vt = dd.body_link_lin_vel_w[:, base.torso_id] + torch.cross(
                        dd.body_link_ang_vel_w[:, base.torso_id],
                        dd.body_link_pos_w[:, base.hand_id] - dd.body_link_pos_w[:, base.torso_id], dim=-1)
                    u = vh / vh.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    parts = torch.stack([(vp * u).sum(-1), ((vt - vp) * u).sum(-1), ((vh - vt) * u).sum(-1)], -1)
                    new_rel = torch.as_tensor(rel_now & ~prev_rel, device=base.device)
                    rel_split[new_rel.cpu().numpy()] = parts[new_rel].cpu().numpy()
                    jn = list(base.robot.joint_names)
                    if "r_elbow" in jn:
                        jv = dd.joint_vel[:, [jn.index("r_elbow"), jn.index("r_shoulder_yaw")]].cpu().numpy() * 180.0 / np.pi
                        rel_jv[rel_now & ~prev_rel] = jv[rel_now & ~prev_rel]
                prev_rel = rel_now
            if hasattr(base, "s_crossed"):  # first plate crossing this step, the flight not ended on the ground before
                cr_now = base.s_crossed.cpu().numpy()
                air_cross |= cr_now & ~prev_cr & ~grounded
                grounded |= base.s_ball_done.cpu().numpy() & ~cr_now
                prev_cr = cr_now
            if hasattr(base, "s_land_step"):
                at = (base.s_landed & ((base.episode_length_buf - base.s_land_step) == int(round(0.1 / base.step_dt)))).cpu().numpy()
                if at.any():
                    cm, cv = base._com()
                    dy = (cm[:, 1] - base.robot.data.body_link_pos_w[:, base.foot_ids[0], 1]).cpu().numpy()
                    com_dytd[at] = dy[at]
                    com_vytd[at] = cv[:, 1].cpu().numpy()[at]
            if abs((k + 1) * base.step_dt - 1.8) < base.step_dt / 2:
                cm, _ = base._com()
                com_dy18 = (cm[:, 1] - base.robot.data.body_link_pos_w[:, base.foot_ids[0], 1]).cpu().numpy()
                up18 = np.isnan(fell_at)
            if 0.2 <= (k + 1) * base.step_dt <= 0.95:
                lz = base.robot.data.body_link_pos_w[:, base.foot_ids[0], 2] - base.s_feet0[:, 0, 2]
                lift_max = np.maximum(lift_max, lz.cpu().numpy())
        both, speed, tilt, head_z, ok = (x.cpu().numpy() for x in base._stand_state())
        dd = base.robot.data  # movement toward the plate (+x) since the first step: lead foot, back foot, pelvis
        dx = torch.cat([dd.body_link_pos_w[:, base.foot_ids, 0] - base.s_feet0[..., 0],
                        (dd.root_link_pos_w[:, 0] - base.s_pelvis0[:, 0]).unsqueeze(-1)], -1).cpu().numpy()
        lift_stage = cfg.stage == "1b1"
        if lift_stage:
            lifted = (base.s_lift_t >= cfg.lift_hold_min_s).cpu().numpy()
            slid = base.s_slid.cpu().numpy()
            done_lift = base.lift_done().cpu().numpy()
            lift_t = base.s_lift_t.cpu().numpy()
            ok = ok & done_lift
        if cfg.stage == "1c":
            ok = ok & base.stride_ok().cpu().numpy()
        stride_stage = cfg.stage == "1b"
        if stride_stage:  # success also needs the stride (the robot's lead foot landed >= lead_stride_min_m ahead)
            landed = base.s_landed.cpu().numpy()
            ok = ok & landed
    fell = ~np.isnan(fell_at)
    ok = ok & ~fell
    mode = "with training action noise" if args.stochastic else "noise-free"
    say(f"== eval_stage: {args.checkpoint} | {n} episodes of {cfg.episode_length_s:.1f} s, {args.task}, {mode}")
    say(f"  falls: {fell.sum()}/{n}" + (f", median at {np.nanmedian(fell_at):.2f} s" if fell.any() else ""))
    up = ~fell
    say(f"  at rest standing at the end (feet down, COM < {cfg.stand_max_com_v} m/s, trunk < {cfg.stand_max_tilt_deg:.0f} "
        f"deg, head > {cfg.stand_min_head_z} m): {ok.sum()}/{n} (failed: fell {fell.sum()}, feet {(up & ~both).sum()}, "
        f"COM speed {(up & (speed >= cfg.stand_max_com_v)).sum()}, trunk tilt {(up & (tilt >= cfg.stand_max_tilt_deg)).sum()}, "
        f"head {(up & (head_z <= cfg.stand_min_head_z)).sum()})")
    down = (head_z < 1.0) | (tilt > 60.0)  # face-down on the hands etc.: not "fallen" by the env's thresholds
    say(f"  down at the end (head < 1.0 m or trunk tilt > 60 deg at {cfg.episode_length_s:.1f} s): {int(down.sum())}/{n} "
        f"(of them not flagged as falls {int((down & ~fell).sum())})")
    if stride_stage:
        say(f"  stride landed (lead foot >= {cfg.lead_stride_min_m} m ahead): {landed.sum()}/{n}, not fallen without a stride "
            f"{(up & ~landed).sum()}")
    if getattr(cfg, "throw_upper", False):
        rel = base.released.cpu().numpy(); cr = base.s_crossed.cpu().numpy(); zone = base.in_zone().cpu().numpy()
        v = base.rel_vel.norm(dim=-1).cpu().numpy(); mx = base.s_ball_maxx.cpu().numpy(); yz = base.s_cross_yz.cpu().numpy()
        say(f"  ball: released {rel.sum()}/{n}, release speed median {np.median(v[rel]) if rel.any() else 0:.1f} m/s; "
            f"crossed the plate {cr.sum()}/{n}, in the zone {zone.sum()}/{n}; farthest x median {np.median(mx):.1f} m"
            + (f"; at the plate median y {np.median(yz[cr, 0]):+.2f} m, z {np.median(yz[cr, 1]):.2f} m" if cr.any() else ""))
        if rel.any():
            vv = base.rel_vel.cpu().numpy()[rel]; rp = (base.rel_pos - base.scene.env_origins).cpu().numpy()[rel]
            say(f"  ball at release (median): velocity ({np.median(vv[:, 0]):+.1f}, {np.median(vv[:, 1]):+.1f}, "
                f"{np.median(vv[:, 2]):+.1f}) m/s, position ({np.median(rp[:, 0]):+.2f}, {np.median(rp[:, 1]):+.2f}, "
                f"{np.median(rp[:, 2]):.2f}) m, at {np.median(base.rel_ref_t.cpu().numpy()[rel]):.2f} s")
        if args.pitch_csv:  # release speed; y + toward FIRST base (the pitcher's left, facing +x), z above the plate ground
            os.makedirs(os.path.dirname(os.path.abspath(args.pitch_csv)), exist_ok=True)
            with open(args.pitch_csv, "w", encoding="utf-8") as f:
                f.write("speed_mps,plate_y,plate_z,crossed" + chr(10))
                for i in range(n):
                    if rel[i]:
                        f.write(f"{v[i]:.3f},{yz[i, 0] - cfg.zone_center_y:.4f},{yz[i, 1]:.4f},{int(cr[i])}" + chr(10))
            say(f"  pitch table: {args.pitch_csv} ({int(rel.sum())} rows)")
        # a dropped ball is not a throw: A25/A50 "passed" with the ball slipping out at 0.65 s (1-2 m/s), fe A75 from
        # 3550 on dropped it at 0.83 s - keep-best must not pick a checkpoint that stands because it never throws
        thrown = rel & (v >= args.gate_min_speed) & (base.rel_ref_t.cpu().numpy() >= args.gate_min_release_t)
        rt = base.rel_ref_t.cpu().numpy()
        say(f"  not thrown: early release (< {args.gate_min_release_t:.2f} s) {int((rel & (rt < args.gate_min_release_t)).sum())}, "
            f"slow (< {args.gate_min_speed:.0f} m/s, on time) {int((rel & (rt >= args.gate_min_release_t) & (v < args.gate_min_speed)).sum())}, "
            f"not released {int((~rel).sum())}; early releases at reference time median "
            + (f"{np.median(rt[rel & (rt < args.gate_min_release_t)]):.2f} s" if (rel & (rt < args.gate_min_release_t)).any() else "-"))
        say(f"  thrown (release >= {args.gate_min_speed:.0f} m/s at >= {args.gate_min_release_t:.2f} s): {thrown.sum()}/{n}; "
            f"at rest and thrown: {(ok & thrown).sum()}/{n}")
        # W1 pitch tiers (user 2026-10-18), the highest each pitch reaches: T0 not thrown; T1 release heading within
        # +-10 deg of the plate line (+x); T2 crosses the plate in the air, |y| <= 1.0 m; T3 |y| <= 0.5 m, z 0.2-1.5 m;
        # T4 strike |y| <= 0.2525 m, z 0.50-1.10 m (aibaseball/game/agents.py StrikeZone). Gate: at rest and thrown
        # >= 60/64, at rest and T3+ on the same pitch >= 32/64 (both evals, 4.0 s episodes)
        vv_all = base.rel_vel.cpu().numpy()
        head = np.degrees(np.arctan2(vv_all[:, 1], vv_all[:, 0]))
        py, pz = yz[:, 0] - cfg.zone_center_y, yz[:, 1]
        t1 = thrown & (np.abs(head) <= 10.0)
        t2 = t1 & cr & air_cross & (np.abs(py) <= 1.0)
        t3 = t2 & (np.abs(py) <= 0.5) & (pz >= 0.2) & (pz <= 1.5)
        t4 = t3 & (np.abs(py) <= 0.2159 + 0.0366) & (pz >= 0.50) & (pz <= 1.10)
        tier = t1.astype(int) + t2 + t3 + t4  # nested: 0..4 (thrown but off-line counts as T0)
        rest = ok  # at rest (stand state, not fallen), before the throw condition
        say("  W1 pitch tiers (highest per pitch) x at rest:  T0 not thrown / T0 off-line (> 10 deg) / T1 / T2 / T3 / T4")
        for lab, m in (("at rest", rest), ("not at rest", ~rest)):
            say(f"    {lab:11s} " + " / ".join(f"{int(x):3d}" for x in [(~thrown & m).sum(), (thrown & ~t1 & m).sum()]
                                              + [((tier == t) & m).sum() for t in range(1, 5)]))
        say(f"  release speed median {np.median(v[rel]) if rel.any() else 0:.1f} m/s, heading median "
            f"{np.median(head[rel]) if rel.any() else 0:+.1f} deg (+ toward 1B); crossed in the air {int((cr & air_cross).sum())}/{n}")
        drop = base.s_drop.cpu().numpy() if hasattr(base, "s_drop") else np.zeros(n, dtype=bool)
        if arm_n.sum() > 0:
            say(f"  arm action |a| mean over 0.9-1.0 s {(arm_abs.sum() / arm_n.sum()):.3f} (freeze_arms {cfg.freeze_arms}, "
                f"arm residual scale {getattr(cfg, 'arm_residual_scale', None)}); ball dropped {int(drop.sum())}/{n}")
        if getattr(cfg, "release_action_window_s", None) is not None and rel.any():
            dt_rel = (base.rel_ref_t - base.release_ref).cpu().numpy()[rel] * 1000.0
            fo = base.s_rel_forced.cpu().numpy()
            say(f"  release time - release_ref p10/50/90 {np.percentile(dt_rel, 10):+.1f}/{np.median(dt_rel):+.1f}/"
                f"{np.percentile(dt_rel, 90):+.1f} ms (window +-{cfg.release_action_window_s * 1000:.0f} ms); forced at the "
                f"window end {int(fo.sum())}/{n}")
        if grp_n > 0:
            g = grp_abs / grp_n
            say(f"  |a| mean over 0.9-1.0 s by group: legs {g[0]:.3f}, waist {g[1]:.3f}, arms {g[2]:.3f} (raw policy output)")
        rs = rel & ~np.isnan(rel_split[:, 0])
        if rs.any():
            m = np.median(rel_split[rs], 0)
            say(f"  wrist (r_hand link) speed at the release along its direction (median): pelvis {m[0]:.2f} + trunk {m[1]:.2f} + arm "
                f"{m[2]:.2f} = {m.sum():.2f} m/s ({rs.sum()} releases)")
        rj = rel & ~np.isnan(rel_jv[:, 0])
        if rj.any():
            say(f"  at the release (median): elbow velocity {np.median(rel_jv[rj, 0]):+.0f} deg/s (+ extending), shoulder "
                f"yaw (IR) {np.median(rel_jv[rj, 1]):+.0f} deg/s; elbow extending in {int((rel_jv[rj, 0] >= 0).sum())}/{int(rj.sum())}")
        ry = rel & ~np.isnan(rel_yaw)
        if ry.sum() > 2:
            cc = np.corrcoef(head[ry], rel_yaw[ry])[0, 1]
            sl = np.polyfit(rel_yaw[ry], head[ry], 1)[0]
            say(f"  release heading p10/50/90 {np.percentile(head[ry], 10):+.1f}/{np.median(head[ry]):+.1f}/"
                f"{np.percentile(head[ry], 90):+.1f} deg; pelvis yaw at release p10/50/90 "
                f"{np.percentile(rel_yaw[ry], 10):+.1f}/{np.median(rel_yaw[ry]):+.1f}/{np.percentile(rel_yaw[ry], 90):+.1f} deg; "
                f"corr {cc:+.2f}, slope {sl:+.2f} deg heading per deg yaw ({ry.sum()} releases)")
        if rel.any():
            elev = np.degrees(np.arctan2(vv_all[:, 2], np.hypot(vv_all[:, 0], vv_all[:, 1])))
            rz = (base.rel_pos - base.scene.env_origins).cpu().numpy()[:, 2] - base.field_z
            sc = base.release_score().cpu().numpy() if hasattr(base, "release_score") else np.zeros(n)
            say(f"  release elevation median {np.median(elev[rel]):+.1f} deg, z median {np.median(rz[rel]):.2f} m above the "
                f"plate ground; release score (gravity-only, 0..1) median {np.median(sc[rel]):.2f}, at rest "
                + (f"{np.median(sc[rel & rest]):.2f}" if (rel & rest).any() else "-"))
        say(f"  W1 gate: at rest and thrown {int((rest & thrown).sum())}/{n} (>= 60/64), at rest and T3+ "
            f"{int((rest & (tier >= 3)).sum())}/{n} (>= 32/64), at rest and T4 {int((rest & (tier >= 4)).sum())}/{n}")
        if rel.any():
            # aero flight (drag + Magnus) from each release state to the plate plane - the eval episodes fly the ball
            # gravity-only (play_mode off), so this is the realistic plate speed / location. W1 gate (user 2026-10-19):
            # at rest and crossing the plate plane in the air at >= 80 km/h (22.2 m/s) >= 60/64, the same pitch T3
            # >= 32/64, both evals
            from aibaseball.physics.ball_flight import simulate_to_plane
            pl = (base.rel_pos - base.scene.env_origins).cpu().double()
            cr_a, v_a, _, re_a = simulate_to_plane(pl, base.rel_vel.cpu().double(), base.rel_omega.cpu().double(), base.phys,
                                                   cfg.plate_distance, dt=0.005, ground_z=base.field_z)
            v_a, re_a = v_a.numpy(), re_a.numpy() & rel
            ya, za = cr_a[:, 1].numpy() - cfg.zone_center_y, cr_a[:, 2].numpy() - base.field_z
            fast = re_a & (v_a >= 80.0 / 3.6)
            t3a = thrown & (np.abs(head) <= 10.0) & re_a & (np.abs(ya) <= 0.5) & (za >= 0.2) & (za <= 1.5)
            say(f"  aero flight (drag + Magnus from the release state): reached the plate plane in the air "
                f"{int(re_a.sum())}/{n}; speed there median "
                + (f"{np.median(v_a[re_a]):.1f} m/s ({np.median(v_a[re_a]) * 3.6:.0f} km/h), y {np.median(ya[re_a]):+.2f} m, "
                   f"z {np.median(za[re_a]):.2f} m" if re_a.any() else "-")
                + f" (release median {np.median(v[rel]):.1f} m/s, spin median {np.median(base.rel_omega.norm(dim=-1).cpu().numpy()[rel]) * 60 / (2 * np.pi):.0f} rpm)")
            say(f"  W1 gate (aero, >= 80 km/h = 22.2 m/s over the plate): 1. at rest and in the air and fast "
                f"{int((rest & fast).sum())}/{n} (>= 60/64); 2. and T3 (aero location) {int((rest & fast & t3a).sum())}/{n} "
                f"(>= 32/64); at rest and aero T3 any speed {int((rest & t3a).sum())}/{n}")
        ok = ok & thrown
    if lift_stage:
        say(f"  leg lift (lead foot >= {cfg.lift_min_m} m up on the back leg alone >= {cfg.lift_hold_min_s} s): "
            f"{lifted.sum()}/{n}, median {np.median(lift_t):.2f} s; back foot moved > {cfg.back_slide_max_m} m: "
            f"{slid.sum()}/{n}; lift done with the feet back home: {done_lift.sum()}/{n}")
    if not np.isnan(com_dytd).all():
        say(f"  COM y - lead foot y 0.1 s after the touchdown (+ toward 1B), median {np.nanmedian(com_dytd):+.2f} m, COM y speed "
            f"{np.nanmedian(com_vytd):+.2f} m/s ({int((~np.isnan(com_dytd)).sum())} landed)")
    if not np.isnan(com_dy18).all():
        say(f"  COM y - lead foot y at 1.8 s (+ toward 1B), median: all {np.median(com_dy18):+.2f} m, not fallen by then "
            + (f"{np.median(com_dy18[up18]):+.2f} m ({int(up18.sum())})" if up18.any() else "-"))
    say(f"  lead foot lift (highest over its start, 0.2-0.95 s): median {np.median(lift_max):.2f} m, max "
        f"{lift_max.max():.2f} m, >= 0.15 m in {(lift_max >= 0.15).sum()}/{n}")
    say(f"  toward the plate (+x) since the start (median): lead foot {np.median(dx[:, 0]):+.2f} m, back foot "
        f"{np.median(dx[:, 1]):+.2f} m, pelvis {np.median(dx[:, 2]):+.2f} m (min pelvis {dx[:, 2].min():+.2f} m)")
    if up.any():
        say(f"  not fallen, at the end (median): trunk tilt {np.median(tilt[up]):.1f} deg, COM speed {np.median(speed[up]):.3f} "
            f"m/s, head {np.median(head_z[up]):.2f} m, both feet down {both[up].mean() * 100:.0f}%")
    if not args.stochastic:
        say(f"[goal] {ok.sum()}/{n}")
    if args.video:  # (--zero: the video too - it filmed the loaded policy, a fall the statistics never had)
        film(w, base, (lambda x: torch.zeros(x.shape[0], cfg.action_space, device=x.device)) if args.zero else pol, T)


def film(w, base, pol, T, fps=30):
    """Side view of env 0 (others hidden); key frames at 0, 1/3, 2/3 and the end of episode 1 side by side (.png)."""
    import cv2
    import imageio
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for i in range(1, base.num_envs):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}")).MakeInvisible()

    def grab():
        for _ in range(6):
            img = base.render()
            if img is not None and img.size and img[..., :3].max() > 0:
                return np.ascontiguousarray(img[..., :3])
        return None

    every = max(1, int(round(1.0 / (fps * args.slow * base.step_dt))))
    keys = {0, T // 3, 2 * T // 3, T - 1}
    frames, kf = [], []
    for e in range(args.video_episodes):
        with torch.inference_mode():
            obs, _ = w.reset()
        fell = None
        for k in range(T):
            with torch.inference_mode():
                obs, _, _, _ = w.step(pol(obs))
                obs = obs.clone()
            t = (k + 1) * base.step_dt
            if fell is None and bool(base.fallen[0]):
                fell = t
            if k % every and not (e == 0 and k in keys):
                continue
            o = base.scene.env_origins[0].cpu().numpy()  # follow the pelvis along the ground (1b walks off the view)
            pel = base.robot.data.root_link_pos_w[0].cpu().numpy()
            o = o + [pel[0] - o[0] - 0.3, pel[1] - o[1], 0.0]
            base.sim.set_camera_view(eye=(o + [0.3, -4.0, 0.9]).tolist(), target=(o + [0.3, 0.0, 0.7]).tolist())
            img = grab()
            if img is None:
                continue
            img = img.copy()
            txt = f"episode {e + 1}  t = {t:.2f} s" + (f"  FELL at {fell:.2f} s" if fell is not None else "")
            cv2.putText(img, txt, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, txt, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            if e == 0 and k in keys:
                kf.append(cv2.resize(img, (640, 360)))
            if k % every == 0:
                frames.append(img)
        # StageEnv never runs the release check (_get_dones is its own): measure the ball's distance to the grip
        gap = float((base.ball.data.root_pos_w[0] - base._grip_point()[0]).norm())
        rel = "ball held" if gap < base.cfg.release_distance else f"ball out of the hand at the end ({gap:.2f} m from the grip)"
        say(f"[video] episode {e + 1}: fell at {fell}, {rel}")
    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    imageio.mimwrite(args.video, frames, fps=fps, quality=8, macro_block_size=1)
    if kf:
        png = os.path.splitext(args.video)[0] + "_keyframes.png"
        imageio.imwrite(png, np.concatenate([np.concatenate(kf[:2], 1), np.concatenate(kf[2:4], 1)], 0)
                        if len(kf) == 4 else np.concatenate(kf, 1))
        say(f"[video] key frames {png}")
    say(f"[video] wrote {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        say(traceback.format_exc())  # app.close() ends the process first and Kit holds stderr - the error was silent
        raise
    finally:
        app.close()
