"""Why does the pitched-ball batter miss? Per-pitch bat-ball geometry and timing, no video.

For every pitch (one episode per env, training randomisation, full pitch speed range) it records, at each physics
step with 8 interpolated sub-steps (the bat and ball move ~15 cm per physics step):
  * gap_m      closest ball-to-bat-surface gap before contact (<= 0 means contact)
  * x_on_bat   where along the bat (from the knob) the closest point was; sweet spot for reference
  * dz_m       ball centre minus closest bat-axis point, vertical (+ = ball above the bat)
  * dperp_m    the same offset, horizontal (along the bat's swing direction: + = ball ahead of the bat)
  * timing_s   time the bat is square to the pitch line (|axis_x| smallest while the sweet spot is fast)
               minus time the ball crosses the contact plane (+ = late)
and summarises misses vs hits: which error dominates (timing, height, reach).

    python scripts/diag_bat_contact.py --headless --checkpoint <model.pt> --envs 256
Output: one summary line per group + logs/diag_bat_contact_<tag>.json
"""

import argparse
import json
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=256)
parser.add_argument("--tag", type=str, default="human3")
parser.add_argument("--timing_bias", type=float, default=None, help="override cfg.timing_bias_s (s, + = pitch later)")
parser.add_argument("--tempo_zero_after_release", action="store_true", help="cfg.tempo_zero_after_release: no tempo correction once the ball is in the air")
parser.add_argument("--post_ref_blend", type=float, default=None, help="cfg.post_ref_blend_s: after the clip, blend to a balanced pose over this many s")
parser.add_argument("--post_ref_pose", type=str, default=None, help="cfg.post_ref_pose: contact | start")
parser.add_argument("--post_ref_keep_residual", action="store_true", help="cfg.post_ref_keep_residual")
parser.add_argument("--mix_frac", type=float, default=None, help="cfg.pitch_mix_frac")
parser.add_argument("--mix_speed", type=float, nargs=2, default=None, help="cfg.pitch_mix_speed_kmh")
parser.add_argument("--mix_release_csv", type=str, default=None, help="training mix: cfg.pitch_mix_release_csv (half the pitches shaped like these releases)")
parser.add_argument("--release_csv", type=str, default=None, help="pitches from the RL pitcher's eval_stage --release_csv (own release state each)")
parser.add_argument("--pitch_csv", type=str, default=None, help="pitches from the RL pitcher's eval_stage --pitch_csv table (speed_mps, plate_y, plate_z, crossed)")
parser.add_argument("--phase_horizon", type=float, default=None, help="cfg.phase_horizon_s (s): phase input scale, e.g. the training episode 2.6")
parser.add_argument("--aim_offset_z", type=float, default=None, help="override cfg.aim_offset_z (m, < 0 = aim under the ball)")
parser.add_argument("--timing_noise", type=float, default=None, help="override cfg.timing_noise_s (+- arrival offset, s)")
parser.add_argument("--dof_only", action="store_true", help="exit right after the --dof_dump readout")
parser.add_argument("--dof_dump", type=str, default=None, help="json: the batter's DOF / body properties as PhysX reports them")
parser.add_argument("--diag_no_replicate", action="store_true", help="diagnostic: scene.replicate_physics False (as the joint scene)")
parser.add_argument("--diag_add_pitcher", action="store_true",
                    help="diagnostic: the joint scene's pitcher articulation + dynamic ball in every env, 18 m away")
parser.add_argument("--trace", type=str, default=None,
                    help="npz: first env per clip, obs / action / body joints per control step (150); perception noise 0")
parser.add_argument("--seed", type=int, default=None, help="env seed (repeat measurements with different pitches)")
parser.add_argument("--post_clip_teacher", type=str, default=None, help="Human21: frozen actor until the clip end, the checkpoint after")
parser.add_argument("--clamp_ref_phase", action="store_true", help="cfg.clamp_ref_phase: reference-time input held at 1 past the clip")
parser.add_argument("--fall_state", action="store_true",
                    help="record whole-body balance state at contact and at the clip end; AUC of each for falls after a hit")
parser.add_argument("--contact_log", action="store_true",
                    help="per hit, at the contact step: sweet-spot speed, contact station dx, swing-vs-pitch angle, exit speed")
parser.add_argument("--body_log", action="store_true",
                    help="lower body vs the reference over the last --body_window s of reference time before contact: pelvis yaw "
                         "rate, foot translation speed, foot yaw rate, foot lift, sole tilt (peaks); split by hit / miss, fell / stood")
parser.add_argument("--body_window", type=float, default=0.3)
parser.add_argument("--link_log", action="store_true",
                    help="upper-body joints (waist / shoulders / elbows / wrists), hits: |joint speed| peak before contact and its time "
                         "vs contact, robot vs reference, and torque saturation over the last 0.1 s of reference time before contact")
parser.add_argument("--sat_log", action="store_true",
                    help="per body joint: fraction of swing steps (ball in flight, before contact) with the torque at its limit")
parser.add_argument("--phase_log", action="store_true",
                    help="torque saturation split into wait / downswing / after contact, l-r torque signs while waiting, "
                         "contact normal vs sweet-spot velocity")
parser.add_argument("--chain_log", action="store_true",
                    help="(implies --phase_log) per hit at contact: each body joint's share of the sweet-spot velocity "
                         "(bat Jacobian x joint velocity), robot vs reference joint velocity; downswing joint power sign")
parser.add_argument("--release_top_hand", action="store_true",
                    help="control run: a copy of the robot USD with the top-hand loop joint (top_hand_grip) disabled")
parser.add_argument("--action_clip", type=float, default=None, help="override cfg.action_clip (zero-shot: limit the policy's residual)")
parser.add_argument("--aim_offset_y", type=float, default=None, help="override cfg.aim_offset_y (m, world y)")
parser.add_argument("--zero_ff_after_ref_end", action="store_true", help="cfg.zero_ff_after_ref_end: no velocity feed-forward on the held pose")
parser.add_argument("--physics_dt", type=float, default=None,
                    help="joint ONE OUTS env check: physics dt (s); decimation is set so the policy keeps its 100 Hz control")
parser.add_argument("--clips", type=str, nargs="+", default=None, help="restrict the MotionLib to these clips (e.g. obp_42-1)")
parser.add_argument("--real_height", action="store_true", help="cfg.real_height: pitch heights not re-centred on the clip's contact")
parser.add_argument("--settle_tail", type=float, default=None, help="cfg.settle_tail_s: the reference decelerates to a stop after the clip")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.phase_log = args.phase_log or args.chain_log
args.headless = True  # diagnostics only: a GUI window also keeps the app alive after the run
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.tee_batting.agents.rsl_rl_ppo_cfg import TeeBattingMimicPPORunnerCfg  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env_cfg import (  # noqa: E402
    PitchedBattingHuman3EnvCfg,
    PitchedBattingHumanPlayEnvCfg,
)

N_SUB = 8
FAST_SS = 10.0  # m/s: sweet-spot speed that counts as "swinging" for the square-bat time


def main():
    cfg = PitchedBattingHumanPlayEnvCfg()
    train = PitchedBattingHuman3EnvCfg()
    for k in ("tee_xy_noise", "tee_height_noise", "joint_pos_noise", "ball_obs_bias_std", "ball_obs_noise_std",
              "ball_obs_delay_s", "pitch_speed_range_start", "pitch_speed_range_final", "timing_noise_s",
              "w_miss", "w_tempo"):
        setattr(cfg, k, getattr(train, k))
    cfg.pitch_ramp_steps = 0  # full speed range and timing noise
    if args.trace:  # deterministic percept and placement (the joint trace sets the same)
        for k in ("ball_obs_noise_std", "ball_obs_bias_std", "tee_xy_noise", "tee_height_noise", "joint_pos_noise"):
            setattr(cfg, k, 0.0)
    if args.timing_bias is not None:
        cfg.timing_bias_s = args.timing_bias
    if args.aim_offset_z is not None:
        cfg.aim_offset_z = args.aim_offset_z
    if args.aim_offset_y is not None:
        cfg.aim_offset_y = args.aim_offset_y
    if args.action_clip is not None:
        cfg.action_clip = args.action_clip
    if args.phase_horizon is not None:
        cfg.phase_horizon_s = args.phase_horizon
    if args.pitch_csv:
        cfg.pitch_table_csv = args.pitch_csv
    if args.release_csv:
        cfg.pitch_release_csv = args.release_csv
    if args.mix_release_csv:
        cfg.pitch_mix_release_csv = args.mix_release_csv
    if args.mix_frac is not None:
        cfg.pitch_mix_frac = args.mix_frac
    if args.mix_speed is not None:
        cfg.pitch_mix_speed_kmh = tuple(args.mix_speed)
    cfg.tempo_zero_after_release = args.tempo_zero_after_release
    if args.post_ref_blend is not None:
        cfg.post_ref_blend_s = args.post_ref_blend
    if args.post_ref_pose:
        cfg.post_ref_pose = args.post_ref_pose
    cfg.post_ref_keep_residual = args.post_ref_keep_residual
    if args.timing_noise is not None:
        cfg.timing_noise_s = args.timing_noise
    if hasattr(cfg, "tee_noise_ramp_steps"):
        cfg.tee_noise_ramp_steps = 0
    cfg.scene.num_envs = args.envs
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.clamp_ref_phase = args.clamp_ref_phase
    cfg.zero_ff_after_ref_end = args.zero_ff_after_ref_end
    if args.settle_tail is not None:
        cfg.settle_tail_s = args.settle_tail
    if args.physics_dt is not None:
        ctrl = cfg.sim.dt * cfg.decimation
        cfg.sim.dt, cfg.decimation = args.physics_dt, int(round(ctrl / args.physics_dt))
        cfg.sim.render_interval = cfg.decimation
        print(f"[diag] physics dt {cfg.sim.dt:.6f} s, decimation {cfg.decimation} (control {cfg.sim.dt * cfg.decimation:.4f} s)", flush=True)
    if args.clips:
        import os as _os
        cfg.motion_files = [f for f in cfg.motion_files if _os.path.splitext(_os.path.basename(f))[0] in args.clips]
        assert len(cfg.motion_files) == len(args.clips), f"clips not found: {args.clips}"
    cfg.real_height = args.real_height
    if args.release_top_hand:
        from pxr import Usd, UsdPhysics  # noqa: E402

        src = cfg.robot.spawn.usd_path
        stage = Usd.Stage.Open(src)
        jp = [p for p in stage.Traverse() if p.GetName() == "top_hand_grip"]
        assert len(jp) == 1, "top_hand_grip not found"
        UsdPhysics.Joint(jp[0]).GetJointEnabledAttr().Set(False)
        dst = os.path.join(os.path.dirname(src), "aib1_diag_no_top_hand.usd")  # same folder: relative references still resolve
        stage.GetRootLayer().Export(dst)
        cfg.robot.spawn.usd_path = dst
        print(f"[bat] top hand released: {dst}", flush=True)
    if args.diag_no_replicate:
        cfg.scene.replicate_physics = False
    if args.diag_add_pitcher:  # bisect the joint-vs-native batter dynamics (10-28): the pitcher's presence alone
        from isaaclab.assets import Articulation, RigidObject
        from aibaseball.tasks.one_outs.joint_env_cfg import JointPitchEnvCfg
        from aibaseball.tasks.tee_batting.tee_batting_env import TeeBattingEnv
        jc = JointPitchEnvCfg()
        p0 = jc.robot.init_state.pos
        pcfg = jc.robot.replace(prim_path="/World/envs/env_.*/Pitcher",
                                init_state=jc.robot.init_state.replace(pos=(p0[0] - 18.0, p0[1], p0[2])))
        bcfg_p = jc.ball.replace(prim_path="/World/envs/env_.*/PBall",
                                 init_state=jc.ball.init_state.replace(pos=(-18.0, 1.0, 0.2)))
        _orig_setup = TeeBattingEnv._setup_scene

        def _setup_with_pitcher(self):
            self.diag_pitcher, self.diag_pball = Articulation(pcfg), RigidObject(bcfg_p)  # spawned in env_0 before the clone
            _orig_setup(self)
            self.scene.articulations["diag_pitcher"] = self.diag_pitcher
            self.scene.rigid_objects["diag_pball"] = self.diag_pball

        TeeBattingEnv._setup_scene = _setup_with_pitcher
        print(f"[diag] + pitcher ({type(pcfg.actuators).__name__}, {len(pcfg.actuators)} groups) and dynamic ball at x -18", flush=True)
    env = gym.make("AIB-PitchedBatting-Human-Play-v0", cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env)
    agent = TeeBattingMimicPPORunnerCfg()
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    if args.post_clip_teacher:  # Human21: as train.py --post_clip_teacher
        from aibaseball.rl.arm_teacher import attach_post_release_student, post_clip_rows

        attach_post_release_student(runner.alg, args.post_clip_teacher, post_clip_rows(env.unwrapped, runner.obs_normalizer),
                                    env.unwrapped.device, runner.obs_normalizer)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    if args.dof_dump:
        from dof_readout import dump
        dump(base.robot, base.sim, args.dof_dump, "native")
        if args.dof_only:
            env.close()
            return
    n, dev, geom, r_ball = base.num_envs, base.device, base.bat_geom, base.phys.ball.radius

    inf = torch.full((n,), float("inf"), device=dev)
    rec = dict(gap=inf.clone(), x_on_bat=torch.zeros(n, device=dev), dz=torch.zeros(n, device=dev),
               dperp=torch.zeros(n, device=dev), t_gap=torch.zeros(n, device=dev),
               t_plane=torch.full((n,), float("nan"), device=dev), sq_ax=torch.ones(n, device=dev),
               t_square=torch.full((n,), float("nan"), device=dev), ss_speed_max=torch.zeros(n, device=dev),
               hit=torch.zeros(n, dtype=torch.bool, device=dev), fell=torch.zeros(n, dtype=torch.bool, device=dev),
               t_fall=torch.full((n,), float("nan"), device=dev), t_hit=torch.full((n,), float("nan"), device=dev),
               tempo_sum=torch.zeros(n, device=dev), tempo_n=torch.zeros(n, device=dev),
               ref_err=torch.full((n,), float("nan"), device=dev))
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    if args.sat_log:
        jids = base.body_joint_ids
        sat_n = torch.zeros(len(jids), device=dev)
        sat_steps = torch.zeros((), device=dev)
    if args.phase_log:
        pj = base.body_joint_ids
        pnames = list(base.body_joint_names)
        PH = ("wait", "down", "after")
        ph_sat = {k: torch.zeros(len(pj), device=dev) for k in PH}
        ph_n = {k: torch.zeros((), device=dev) for k in PH}
        pairs = [(pnames.index(nm), pnames.index("r_" + nm[2:]), nm[2:]) for nm in pnames
                 if nm.startswith("l_") and ("r_" + nm[2:]) in pnames]
        pair_opp = torch.zeros(len(pairs), device=dev)
        rob = base.robot
        arm = {sd: [i for i, nm in enumerate(pnames) if re.match(rf"{sd}_(shoulder|elbow|wrist)", nm)] for sd in "lr"}
        hand_body = {sd: rob.find_bodies(f"{sd}_hand")[0][0] for sd in "lr"}
        fixed = rob.is_fixed_base
        wrench = dict(fl=[], fr=[])
        pd_err = dict(ref=[], res=[], tot=[])  # wait-phase PD target error split: reference part / policy residual part
        pair_tl = torch.zeros(len(pairs), device=dev)
        pair_tr = torch.zeros(len(pairs), device=dev)
        swing_on = torch.zeros(n, dtype=torch.bool, device=dev)
        t_hit_phys = torch.full((n,), float("inf"), device=dev)
        for k in ("n_dot_v", "vz_frac", "normal_z"):
            rec[k] = torch.full((n,), float("nan"), device=dev)
    if args.chain_log:
        nj = len(pj)
        chain = {k: torch.full((n, nj), float("nan"), device=dev) for k in ("cc", "qd", "qdr", "q", "qr")}
        # contact geometry (world frame, m): robot / reference sweet spot and the ball minus the pelvis, pelvis drift from the reference
        pose = {k: torch.full((n, 3), float("nan"), device=dev) for k in ("ss_rel", "ss_rel_ref", "ball_rel", "root_drift")}
        chain_chk = {k: torch.full((n,), float("nan"), device=dev) for k in ("v", "joints", "res_link", "res_com")}
        pw_pos, pw_neg = torch.zeros(nj, device=dev), torch.zeros(nj, device=dev)
    if args.contact_log:
        for k in ("ss_contact", "dx_contact", "swing_pitch_deg", "ev_contact", "ref_phase_hit", "D_contact", "launch_contact",
                  "ev_any", "launch_any", "dx_any", "D_any", "spray_contact", "backspin_contact"):
            rec[k] = torch.full((n,), float("nan"), device=dev)
    if args.fall_state:
        rob = base.robot
        mass = rob.root_physx_view.get_masses().to(dev)  # (n, bodies)
        foot_ids = rob.find_bodies(".*foot.*")[0] or rob.find_bodies(".*ankle_roll.*")[0]
        FS = ("com_v", "com_foot", "L_orb", "yaw_rate")
        for k in FS:
            for at in ("hit", "end"):
                rec[f"{k}_{at}"] = torch.full((n,), float("nan"), device=dev)

        def balance_state():
            d = rob.data
            m = mass.unsqueeze(-1)
            com = (d.body_com_pos_w * m).sum(1) / m.sum(1)
            vcom = (d.body_com_lin_vel_w * m).sum(1) / m.sum(1)
            # orbital angular momentum about the whole-body CoM, per kg
            L = (torch.cross(d.body_com_pos_w - com.unsqueeze(1), d.body_com_lin_vel_w - vcom.unsqueeze(1), dim=-1) * m).sum(1)
            feet = d.body_pos_w[:, foot_ids].mean(1)
            return dict(com_v=vcom[:, :2].norm(dim=-1), com_foot=(com[:, :2] - feet[:, :2]).norm(dim=-1),
                        L_orb=L.norm(dim=-1) / m.sum(1).squeeze(-1), yaw_rate=d.root_ang_vel_w[:, 2].abs())
    if args.body_log:
        from isaaclab.utils.math import quat_apply

        from aibaseball.mocap.motion import KEY_BODIES

        rob_b = base.robot
        bl_feet = [rob_b.find_bodies(f"{sd}_foot")[0][0] for sd in "lr"]
        bl_fk = [KEY_BODIES.index(f"{sd}_foot") for sd in "lr"]
        BL = ["yaw_sim", "yaw_ref"] + [f"{k}_{sd}" for k in ("foot_v_sim", "foot_v_ref", "foot_yaw", "lift_sim", "lift_ref", "tilt_deg")
                                       for sd in "lr"]
        for k in BL:
            rec[k] = torch.zeros(n, device=dev)
        bl_prev = {}
        bl_z0 = {}
    if args.link_log:
        lk_ids = [i for i, nm in enumerate(base.body_joint_names) if re.match(r"waist|(l|r)_(shoulder|elbow|wrist)", nm)]
        lk_names = [base.body_joint_names[i] for i in lk_ids]
        lk_j = [base.body_joint_ids[i] for i in lk_ids]
        nl = len(lk_ids)
        lk = {k: torch.zeros(n, nl, device=dev) for k in ("pk", "pk_ref", "sat", "sat_n")}
        lk.update({k: torch.full((n, nl), float("nan"), device=dev) for k in ("t_pk", "t_pk_ref")})
    prev = {}
    past_end = torch.full((n,), float("nan"), device=dev)  # reference clock minus the clip's end, at the last physics step
    rec["fall_past_end"] = torch.full((n,), float("nan"), device=dev)
    orig_apply = base._apply_action

    def hooked_apply():
        knob, axis, cm, v_cm, w, _ = base._bat_state()
        ball = base.pitch_pos.clone()
        t = base.phys_t.clone()
        live = ~finished & ~base.hit
        if prev:
            for f in torch.linspace(0.0, 1.0, N_SUB + 1, device=dev)[1:]:
                k = prev["knob"] + (knob - prev["knob"]) * f
                a = prev["axis"] + (axis - prev["axis"]) * f
                a = a / a.norm(dim=-1, keepdim=True)
                b = prev["ball"] + (ball - prev["ball"]) * f
                x = ((b - k) * a).sum(-1).clamp(0.0, geom.length)
                p = k + a * x.unsqueeze(-1)
                off = b - p
                gap = off.norm(dim=-1) - geom.radius_at(x) - r_ball
                better = live & (gap < rec["gap"])
                # swing direction: horizontal, perpendicular to the bat axis, along the sweet spot's velocity
                _, v_ss = base._sweet_spot(knob, axis, cm, v_cm, w)
                dperp_dir = v_ss - (v_ss * a).sum(-1, keepdim=True) * a
                dperp_dir[:, 2] = 0.0
                dperp_dir = dperp_dir / dperp_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                rec["gap"] = torch.where(better, gap, rec["gap"])
                rec["x_on_bat"] = torch.where(better, x, rec["x_on_bat"])
                rec["dz"] = torch.where(better, off[:, 2], rec["dz"])
                rec["dperp"] = torch.where(better, (off * dperp_dir).sum(-1), rec["dperp"])
                rec["t_gap"] = torch.where(better, prev["t"] + (t - prev["t"]) * f, rec["t_gap"])
            # ball crosses the contact plane (moving -x)
            xr = base.contact_x_ref
            cross = live & (prev["ball"][:, 0] >= xr) & (ball[:, 0] < xr) & torch.isnan(rec["t_plane"])
            frac = (prev["ball"][:, 0] - xr) / (prev["ball"][:, 0] - ball[:, 0]).clamp_min(1e-6)
            rec["t_plane"] = torch.where(cross, prev["t"] + (t - prev["t"]) * frac, rec["t_plane"])
            # reference clock when the ball crosses: + = the swing reference is already past its contact frame (early)
            rec["ref_err"] = torch.where(cross, base.ref_t - base.ref.contact_time[base.clip], rec["ref_err"])
            # tempo action while the ball is in flight, before contact
            fl = live & (t >= base.release_t)
            rec["tempo_sum"] += torch.where(fl, base.actions[:, -1].clamp(-1.0, 1.0), torch.zeros_like(t))
            rec["tempo_n"] += fl.float()
        _, v_ss = base._sweet_spot(knob, axis, cm, v_cm, w)
        sp = v_ss.norm(dim=-1)
        rec["ss_speed_max"] = torch.where(live, torch.maximum(rec["ss_speed_max"], sp), rec["ss_speed_max"])
        sq = live & (sp > FAST_SS) & (axis[:, 0].abs() < rec["sq_ax"])
        rec["sq_ax"] = torch.where(sq, axis[:, 0].abs(), rec["sq_ax"])
        rec["t_square"] = torch.where(sq, t, rec["t_square"])
        prev.update(knob=knob.clone(), axis=axis.clone(), ball=ball, t=t)
        past_end.copy_(torch.where(~finished, base.ref_t - base.ref.duration[base.clip], past_end))
        if args.body_log:
            d = rob_b.data
            rr_b = base._ref(base.ref_t)
            ct = base.ref.contact_time[base.clip]
            win = ~finished & (base.ref_t > ct - args.body_window) & (base.ref_t <= ct)
            fz = d.body_link_pos_w[:, bl_feet, 2] - base.scene.env_origins[:, 2:3]
            fz_r = rr_b["key_pos"][:, bl_fk, 2] - base.scene.env_origins[:, 2:3]
            if not bl_z0:  # standing heights at the first step (foot frame above the ground), robot and reference
                bl_z0.update(sim=fz.clone(), ref=fz_r.clone())
            up = quat_apply(d.body_link_quat_w[:, bl_feet].reshape(-1, 4),
                            torch.tensor([0.0, 0.0, 1.0], device=dev).expand(2 * n, 3)).reshape(n, 2, 3)
            vals = dict(yaw_sim=d.root_ang_vel_w[:, 2].abs(),
                        yaw_ref=(rr_b["root_ang_vel"][:, 2] * base.rate / base.speed).abs())
            fv = d.body_link_lin_vel_w[:, bl_feet, :2].norm(dim=-1)
            if bl_prev:
                # the reference clock moves once per control step (decimation physics steps): the change over one control step
                fv_r = (rr_b["key_pos"][:, bl_fk, :2] - bl_prev["kp"]).norm(dim=-1) / base.step_dt
            else:
                fv_r = torch.zeros_like(fv)
            bl_prev["kp"] = rr_b["key_pos"][:, bl_fk, :2].clone()
            for i, sd in enumerate("lr"):
                vals[f"foot_v_sim_{sd}"] = fv[:, i]
                vals[f"foot_v_ref_{sd}"] = fv_r[:, i]
                vals[f"foot_yaw_{sd}"] = d.body_link_ang_vel_w[:, bl_feet[i], 2].abs()
                vals[f"lift_sim_{sd}"] = fz[:, i] - bl_z0["sim"][:, i]
                vals[f"lift_ref_{sd}"] = fz_r[:, i] - bl_z0["ref"][:, i]
                vals[f"tilt_deg_{sd}"] = torch.rad2deg(torch.acos(up[:, i, 2].clamp(-1.0, 1.0)))
            for k, v in vals.items():
                rec[k] = torch.where(win, torch.maximum(rec[k], v), rec[k])
        if args.link_log:
            d = base.robot.data
            ct = base.ref.contact_time[base.clip]
            rr_l = base._ref(base.ref_t)
            qd_l = d.joint_vel[:, lk_j].abs()
            qdr_l = (rr_l["joint_vel"][:, lk_ids] * (base.rate / base.speed).unsqueeze(-1)).abs()
            win_l = (~finished & ~base.hit & (base.ref_t > ct - 0.3)).unsqueeze(-1)
            up = win_l & (qd_l > lk["pk"])
            lk["pk"] = torch.where(up, qd_l, lk["pk"])
            lk["t_pk"] = torch.where(up, t.unsqueeze(-1).expand(-1, nl), lk["t_pk"])  # physics time; minus the hit time later
            win_r = (~finished & (base.ref_t > ct - 0.3) & (base.ref_t <= ct + 0.02)).unsqueeze(-1)
            up_r = win_r & (qdr_l > lk["pk_ref"])
            lk["pk_ref"] = torch.where(up_r, qdr_l, lk["pk_ref"])
            lk["t_pk_ref"] = torch.where(up_r, (base.ref_t - ct).unsqueeze(-1).expand(-1, nl), lk["t_pk_ref"])
            last = (~finished & ~base.hit & (base.ref_t > ct - 0.1) & (base.ref_t <= ct)).unsqueeze(-1)
            sat_l = d.computed_torque[:, lk_j].abs() > d.applied_torque[:, lk_j].abs() + 1e-3
            lk["sat"] += (sat_l & last).float()
            lk["sat_n"] += last.float().expand(-1, nl)
        if args.contact_log or args.phase_log:
            was_hit, v_in, ref_t_pre = base.hit.clone(), base.pitch_vel.clone(), base.ref_t.clone()
        if args.chain_log:
            # sweet-spot velocity split by joint, bat state before this physics step (the one the hit test uses)
            d = rob.data
            J = rob.root_physx_view.get_jacobians()[:, base.bat_id - (1 if fixed else 0)]  # (n, 6, 6 + dofs)
            cols = pj + (0 if fixed else 6)
            p_ss = knob + axis * geom.sweet_spot_x
            qd_all = d.joint_vel[:, pj]
            Jl, Ja = J[:, :3][:, :, cols], J[:, 3:][:, :, cols]
            # PhysX gives the linear rows at the bat's centre of mass (10-27 check: residual 0.17 m/s at the COM, 14.9 at the knob)
            cm_b = d.body_com_pos_w[:, base.bat_id]
            cc_l = Jl + torch.cross(Ja, (p_ss - cm_b).unsqueeze(-1).expand_as(Ja), dim=1)
            v_j = (cc_l * qd_all.unsqueeze(1))  # (n, 3, nj)
            vh = v_ss / sp.unsqueeze(-1).clamp_min(1e-6)
            cc_now = (v_j * vh.unsqueeze(-1)).sum(1)
            if not fixed:  # check: the floating-base columns with the root twist close the sum (two origin conventions)
                tw = torch.cat([d.root_link_lin_vel_w, d.root_link_ang_vel_w], -1).unsqueeze(-1)
                Jb = J[:, :, :6]
                vb = (Jb[:, :3] @ tw).squeeze(-1) + torch.cross((Jb[:, 3:] @ tw).squeeze(-1), p_ss - knob, dim=-1)
                res_link = (vb + v_j.sum(-1) - v_ss).norm(dim=-1)
                cc_c = Jl + torch.cross(Ja, (p_ss - cm_b).unsqueeze(-1).expand_as(Ja), dim=1)
                vb_c = (Jb[:, :3] @ tw).squeeze(-1) + torch.cross((Jb[:, 3:] @ tw).squeeze(-1), p_ss - cm_b, dim=-1)
                res_com = (vb_c + (cc_c * qd_all.unsqueeze(1)).sum(-1) - v_ss).norm(dim=-1)
            else:
                res_link = res_com = torch.zeros_like(sp)
            rr = base._ref(base.ref_t)
            qdr_now = rr["joint_vel"] * (base.rate / base.speed).unsqueeze(-1)
            ch_pre = dict(cc=cc_now, qd=qd_all.clone(), qdr=qdr_now, q=d.joint_pos[:, pj].clone(), qr=rr["joint_pos"],
                          v=sp, joints=cc_now.sum(-1), res_link=res_link, res_com=res_com)
            rp = d.root_link_pos_w
            pose_pre = dict(ss_rel=p_ss - rp, ss_rel_ref=base._sweet_spot_of(rr["key_pos"]) - rr["root_pos"],
                            ball_rel=ball - rp, root_drift=rp - rr["root_pos"])
        orig_apply()
        if args.phase_log:
            nh2 = base.hit & ~was_hit & ~finished
            swing_on.logical_or_(live & (sp > 5.0) & (t >= base.release_t))
            t_hit_phys.copy_(torch.where(nh2, t, t_hit_phys))
            # contact normal: ball centre at first contact minus the closest bat-axis point (bat at the start of the step)
            xc = ((base.launch_pos - knob) * axis).sum(-1).clamp(0.0, geom.length)
            nrm = base.launch_pos - (knob + axis * xc.unsqueeze(-1))
            nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            vhat = v_ss / sp.unsqueeze(-1).clamp_min(1e-6)
            for k, val in (("n_dot_v", (nrm * vhat).sum(-1)), ("vz_frac", vhat[:, 2]), ("normal_z", nrm[:, 2])):
                rec[k] = torch.where(nh2, val, rec[k])
            if args.chain_log:
                for k in chain:
                    chain[k].copy_(torch.where(nh2.unsqueeze(-1), ch_pre[k], chain[k]))
                for k in chain_chk:
                    chain_chk[k].copy_(torch.where(nh2, ch_pre[k], chain_chk[k]))
                for k in pose:
                    pose[k].copy_(torch.where(nh2.unsqueeze(-1), pose_pre[k], pose[k]))
        if args.contact_log:
            nh = base.hit & ~was_hit & ~finished
            # angle between the sweet spot's velocity and the reversed pitch direction (0 = swinging straight back up the pitch)
            cosang = (v_ss * -v_in).sum(-1) / (sp * v_in.norm(dim=-1)).clamp_min(1e-6)
            for k, val in (("ss_contact", sp), ("dx_contact", base.contact_x - geom.sweet_spot_x),
                           ("swing_pitch_deg", torch.rad2deg(torch.acos(cosang.clamp(-1.0, 1.0)))),
                           ("ev_contact", base.launch_vel.norm(dim=-1)),
                           ("D_contact", base.contact_D),  # Human26: normal offset (+ = undercut, contact_offset.py's D)
                           ("launch_contact", torch.rad2deg(torch.atan2(base.launch_vel[:, 2], base.launch_vel[:, :2].norm(dim=-1)))),
                           # reference clock at the hit step minus the clip's contact time (+ = the reference is past contact)
                           ("ref_phase_hit", ref_t_pre - base.ref.contact_time[base.clip])):
                rec[k] = torch.where(nh, val, rec[k])
            # every hit, also the ones made in _get_dones' last-sub-step impact check (missed by the keys above: ~27% of
            # the 12500 hits): read from the env's stored launch / contact buffers on the next physics step. The game's
            # BattedBall inputs: spray = initial azimuth (+ = +Y, left field), backspin about v_h x z (the env's), rpm.
            na = base.hit & torch.isnan(rec["ev_any"]) & ~finished
            lv = base.launch_vel
            vh = torch.nn.functional.normalize(lv * torch.tensor([1.0, 1.0, 0.0], device=dev), dim=-1)
            back = torch.cross(vh, torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(lv), dim=-1)
            for k, val in (("ev_any", lv.norm(dim=-1)), ("launch_any", torch.rad2deg(torch.atan2(lv[:, 2], lv[:, :2].norm(dim=-1)))),
                           ("dx_any", base.contact_x - geom.sweet_spot_x), ("D_any", base.contact_D),
                           ("spray_contact", torch.rad2deg(torch.atan2(lv[:, 1], lv[:, 0]))),
                           ("backspin_contact", (base.launch_omega * back).sum(-1) * (30.0 / np.pi))):
                rec[k] = torch.where(na, val, rec[k])
        rec["hit"] |= base.hit & ~finished

    base._apply_action = hooked_apply
    obs, _ = wrapped.reset()
    # the pitch of each env's (first) episode: read now - finished envs are reset to a new pitch before the end
    kmh0, plate0 = base.pitch_kmh.clone(), base.pitch_plate.clone()
    clip_first = base.clip.clone()  # the first episode's clip (the trace keeps that episode)
    prev.clear()
    trace = [] if args.trace else None
    for _ in range(int(cfg.episode_length_s / base.step_dt) + 5):
        with torch.inference_mode():
            act = policy(obs)
            if trace is not None and len(trace) < 150:
                trace.append(dict(obs=obs.clone(), act=act.clone(), q=base.robot.data.joint_pos[:, base.body_joint_ids].clone(),
                                  k=base.episode_length_buf.clone()))
            obs, _, dones, _ = wrapped.step(act)
        if trace is not None and len(trace) <= 150 and "tp" not in trace[-1]:  # the targets that step wrote (all dofs)
            trace[-1]["tp"] = base.robot.data.joint_pos_target.clone()
            trace[-1]["tv"] = base.robot.data.joint_vel_target.clone()
        t_now = (base.episode_length_buf.float() * base.step_dt)
        new_hit = base.hit & ~finished & torch.isnan(rec["t_hit"])
        rec["t_hit"] = torch.where(new_hit, t_now, rec["t_hit"])
        new_fall = base.fallen & ~finished & ~rec["fell"]
        rec["t_fall"] = torch.where(new_fall, t_now, rec["t_fall"])  # time into the episode of the first fall
        rec["fall_past_end"] = torch.where(new_fall, past_end, rec["fall_past_end"])  # < 0 in the clip, then hold / tail
        rec["fell"] |= base.fallen & ~finished
        if args.fall_state:
            bs = balance_state()
            at_end = ~finished & (base.ref_t >= base.ref.duration[base.clip]) & torch.isnan(rec["com_v_end"])
            for k in FS:
                rec[f"{k}_hit"] = torch.where(new_hit, bs[k], rec[f"{k}_hit"])
                rec[f"{k}_end"] = torch.where(at_end, bs[k], rec[f"{k}_end"])
        if args.phase_log:
            d = base.robot.data
            tq_c, tq_a = d.computed_torque[:, pj], d.applied_torque[:, pj]
            sat = tq_c.abs() > tq_a.abs() + 1e-3
            released = ~finished & (base.phys_t >= base.release_t)
            masks = dict(wait=released & ~swing_on & ~base.hit, down=released & swing_on & ~base.hit,
                         after=~finished & base.hit & (base.phys_t - t_hit_phys < 0.3))
            for k in PH:
                ph_sat[k] += (sat & masks[k].unsqueeze(-1)).float().sum(0)
                ph_n[k] += masks[k].float().sum()
            if args.chain_log:  # downswing joint power: + = the motor drives the motion, - = it brakes / absorbs
                pw = tq_a * d.joint_vel[:, pj] * masks["down"].float().unsqueeze(-1)
                pw_pos += pw.clamp_min(0.0).sum(0)
                pw_neg += (-pw).clamp_min(0.0).sum(0)
            w = masks["wait"].float()
            if masks["wait"].any() and base.episode_length_buf[0] % 5 == 0:
                wi2 = masks["wait"].nonzero(as_tuple=False).squeeze(-1)
                qn = d.joint_pos[wi2][:, pj]
                q_ref = base._ref(base.ref_t)["joint_pos"][wi2]
                res_part = base.actions[wi2, :-1] * base.action_scale * base.cfg.residual_scale
                pd_err["ref"].append((q_ref - qn).abs().cpu())
                pd_err["res"].append(res_part.abs().cpu())
                pd_err["tot"].append((base.joint_targets[wi2][:, pj] - qn).abs().cpu())
            if masks["wait"].any() and base.episode_length_buf[0] % 5 == 0:  # every 5th control step is enough
                J = rob.root_physx_view.get_jacobians()
                wi = masks["wait"].nonzero(as_tuple=False).squeeze(-1)
                for sd in "lr":
                    cols = pj[arm[sd]] + (0 if fixed else 6)
                    Jh = J[wi][:, hand_body[sd] - (1 if fixed else 0)][:, :, cols]  # (m, 6, k)
                    tau = tq_a[wi][:, arm[sd]]  # (m, k): tau = J^T F  ->  least-squares F (6)
                    F = torch.linalg.lstsq(Jh.transpose(1, 2), tau.unsqueeze(-1)).solution.squeeze(-1)
                    wrench["f" + sd].append(F[:, :3].cpu())
            for i, (a, b, _) in enumerate(pairs):
                pair_opp[i] += ((tq_a[:, a] * tq_a[:, b] < 0).float() * w).sum()
                pair_tl[i] += (tq_a[:, a] * w).sum()
                pair_tr[i] += (tq_a[:, b] * w).sum()
        if args.sat_log:
            d = base.robot.data
            sw = ~finished & ~base.hit & (base.phys_t >= base.release_t)  # swinging: ball in flight, no contact yet
            sat = (d.computed_torque[:, jids].abs() > d.applied_torque[:, jids].abs() + 1e-3)
            sat_n += (sat & sw.unsqueeze(-1)).float().sum(0)
            sat_steps += sw.float().sum()
        finished |= dones.bool()
        if finished.all():
            break

    r = {k: v.cpu().numpy() for k, v in rec.items()}
    r["timing"] = r["t_square"] - r["t_plane"]
    r["kmh"] = kmh0.cpu().numpy()
    r["plate_y"], r["plate_z"] = (v.cpu().numpy() for v in plate0.unbind(-1))
    ss = float(geom.sweet_spot_x)

    def summary(name, m):
        if m.sum() == 0:
            print(f"[bat] {name}: 0 pitches", flush=True)
            return {}
        q = lambda a: np.nanpercentile(a[m], [25, 50, 75]).round(3).tolist()  # noqa: E731
        s = dict(n=int(m.sum()), gap_m=q(r["gap"]), dz_m=q(r["dz"]), dperp_m=q(r["dperp"]),
                 x_minus_sweet_m=q(r["x_on_bat"] - ss), timing_s=q(r["timing"]),
                 ss_speed_max=q(r["ss_speed_max"]), fell=int(r["fell"][m].sum()))
        mf = m & r["fell"]
        if mf.any():  # when the falls happen: into the episode, and after contact (hits only)
            s["fall_t_s"] = np.nanpercentile(r["t_fall"][mf], [10, 25, 50, 75, 90]).round(2).tolist()
            s["fall_after_hit_s"] = np.nanpercentile((r["t_fall"] - r["t_hit"])[mf], [10, 25, 50, 75, 90]).round(2).tolist()
            pe, tail = r["fall_past_end"][mf], base.cfg.settle_tail_s
            s["fall_phase"] = dict(in_clip=int((pe < 0).sum()), in_tail=int(((pe >= 0) & (pe < tail)).sum()),
                                   after_tail=int((pe >= tail).sum()), past_end_s=np.nanpercentile(pe, [10, 50, 90]).round(2).tolist())
        print(f"[bat] {name}: " + json.dumps(s), flush=True)
        return s

    done = finished.cpu().numpy()
    if base.ref.tail_stats:  # what the settle tail asks for beyond the clip's last frame
        ts = base.ref.tail_stats
        print("[bat] settle_tail: " + json.dumps(dict(s=base.cfg.settle_tail_s, tau=base.cfg.settle_tau,
              root_disp_max_m=round(max(x["root_disp_m"] for x in ts), 3), root_turn_max_rad=round(max(x["root_turn_rad"] for x in ts), 3),
              joint_exc_max_rad=round(max(x["joint_exc_rad"] for x in ts), 3),
              joint_exc_by_clip=[(x["joint_exc_name"], round(x["joint_exc_rad"], 2)) for x in ts])), flush=True)
    out = dict(checkpoint=args.checkpoint, sweet_spot_x=ss, pitches=int(done.sum()), hit_rate=float(r["hit"][done].mean()))
    print(f"[bat] pitches {out['pitches']}  hit rate {out['hit_rate']:.3f}  (quartiles 25/50/75)", flush=True)
    out["hits"] = summary("hits", done & r["hit"])
    miss = done & ~r["hit"]
    out["misses"] = summary("misses", miss)
    # dominant error per miss: the largest of |timing| * swing speed, |dz|, |dperp| (all in metres)
    late = np.abs(r["timing"]) * np.nan_to_num(r["ss_speed_max"])
    errs = np.stack([np.nan_to_num(late, nan=9.0), np.abs(r["dz"]), np.abs(r["dperp"])], -1)
    dom = np.argmax(errs, -1)
    out["miss_dominant"] = {name: int((miss & (dom == i)).sum()) for i, name in enumerate(("timing", "height", "reach"))}
    out["miss_late"] = int((miss & (r["timing"] > 0)).sum())
    out["miss_ball_above"] = int((miss & (r["dz"] > 0)).sum())
    print("[bat] miss dominant error: " + json.dumps(out["miss_dominant"])
          + f"  late {out['miss_late']}/{int(miss.sum())}  ball above bat {out['miss_ball_above']}/{int(miss.sum())}",
          flush=True)
    # by pitch speed (mixed libraries / the RL pitcher's slow pitches): hit rate, early / late misses, tempo, reference clock
    tempo = r["tempo_sum"] / np.maximum(r["tempo_n"], 1)
    out["by_speed"] = {}
    zone = (np.abs(r["plate_y"]) <= 0.2525) & (r["plate_z"] >= 0.50) & (r["plate_z"] <= 1.10)  # StrikeZone
    groups = [("slow < 80 km/h", done & (r["kmh"] < 80)), ("fast >= 80 km/h", done & (r["kmh"] >= 80))]
    if args.release_csv or args.pitch_csv or args.mix_release_csv:  # pitches with a real plate location
        groups += [("slow in zone", done & (r["kmh"] < 80) & zone), ("slow out of zone", done & (r["kmh"] < 80) & ~zone)]
    for name, m in groups:
        if not m.any():
            continue
        mm = m & ~r["hit"]
        g = dict(n=int(m.sum()), hit_rate=round(float(r["hit"][m].mean()), 3), misses=int(mm.sum()),
                 early=int((mm & (r["timing"] < 0)).sum()), late=int((mm & (r["timing"] > 0)).sum()),
                 tempo_q=np.nanpercentile(tempo[m], [25, 50, 75]).round(3).tolist(),
                 ref_err_s_q=np.nanpercentile(r["ref_err"][m], [25, 50, 75]).round(3).tolist(),
                 kmh_q=np.nanpercentile(r["kmh"][m], [10, 50, 90]).round(1).tolist())
        out["by_speed"][name] = g
        print(f"[bat] {name}: " + json.dumps(g), flush=True)
    if args.fall_state:
        hm = r["hit"].astype(bool)
        fall_auc = {}
        for k in FS:
            for at in ("hit", "end"):
                x, y = r[f"{k}_{at}"][hm], r["fell"][hm].astype(bool)
                ok = ~np.isnan(x)
                x, y = x[ok], y[ok]
                if y.all() or not y.any():
                    continue
                rk = x.argsort().argsort() + 1.0  # rank AUC (fallers have the larger value -> > 0.5)
                auc = (rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
                fall_auc[f"{k}_{at}"] = dict(auc=round(float(auc), 3), n=int(ok.sum()),
                                             fell_med=round(float(np.median(x[y])), 3),
                                             stood_med=round(float(np.median(x[~y])), 3))
        out["fall_auc"] = fall_auc
        print("[bat] fall_auc: " + json.dumps(fall_auc), flush=True)
    if args.sat_log:
        frac = (sat_n / sat_steps.clamp_min(1)).cpu().numpy()
        names = [base.body_joint_names[i] for i in range(len(jids))]
        top = sorted(zip(frac.tolist(), names), reverse=True)[:10]
        out["torque_sat"] = {nm: round(f, 3) for f, nm in top}
        print("[bat] torque_sat (frac of swing steps, top 10): " + json.dumps(out["torque_sat"]), flush=True)
    if args.phase_log:
        ps = {}
        for k in PH:
            f = (ph_sat[k] / ph_n[k].clamp_min(1)).cpu().numpy()
            top = sorted(zip(f.tolist(), pnames), reverse=True)[:8]
            ps[k] = dict(steps=int(ph_n[k]), mean_all=round(float(f.mean()), 3), top={nm: round(v, 3) for v, nm in top})
        nw = float(ph_n["wait"].clamp_min(1))
        ps["wait_lr_pairs"] = {nm: dict(opp_sign_frac=round(float(pair_opp[i]) / nw, 3), tau_l=round(float(pair_tl[i]) / nw, 1),
                                        tau_r=round(float(pair_tr[i]) / nw, 1)) for i, (_, _, nm) in enumerate(pairs)}
        if pd_err["ref"]:
            ref_e, res_e, tot_e = (torch.cat(pd_err[k]).median(0).values.numpy() for k in ("ref", "res", "tot"))
            show = sorted(range(len(pnames)), key=lambda i: -tot_e[i])[:10]
            ps["wait_pd_err_rad"] = {pnames[i]: dict(total=round(float(tot_e[i]), 3), ref_minus_q=round(float(ref_e[i]), 3),
                                                     residual=round(float(res_e[i]), 3)) for i in show}
        if wrench["fl"]:
            fl, fr = torch.cat(wrench["fl"]).numpy(), torch.cat(wrench["fr"]).numpy()
            nl, nr = np.linalg.norm(fl, axis=1), np.linalg.norm(fr, axis=1)
            cos = (fl * fr).sum(1) / np.maximum(nl * nr, 1e-6)
            net = np.linalg.norm(fl + fr, axis=1) / np.maximum((nl + nr) / 2, 1e-6)
            qq = lambda v: [round(float(x), 2) for x in np.percentile(v, [25, 50, 75])]
            # force each arm exerts on its hand (world, N, least-squares from the arm torques): a fight = similar size,
            # opposite direction (cos ~ -1), net force much smaller than either
            ps["wait_hand_force"] = dict(samples=len(nl), F_l_N=qq(nl), F_r_N=qq(nr), cos_lr=qq(cos), net_over_mean=qq(net),
                                          F_l_mean=[round(float(x), 1) for x in fl.mean(0)], F_r_mean=[round(float(x), 1) for x in fr.mean(0)])
        hm = r["hit"].astype(bool)
        q = lambda v: [round(float(x), 3) for x in np.nanpercentile(v, [25, 50, 75])] if np.isfinite(v).any() else None
        ps["contact"] = dict(n=int(hm.sum()), n_dot_v=q(r["n_dot_v"][hm]), vz_frac=q(r["vz_frac"][hm]), normal_z=q(r["normal_z"][hm]))
        out["phase_log"] = ps
        for k, v in ps.items():
            print(f"[bat] phase_log {k}: " + json.dumps(v), flush=True)
    if args.chain_log:
        hm = r["hit"].astype(bool)
        cc, qd, qdr = (chain[k].cpu().numpy()[hm] for k in ("cc", "qd", "qdr"))
        med = lambda v: round(float(np.nanmedian(v)), 2) if np.isfinite(v).any() else None
        cmed = {nm: med(cc[:, i]) for i, nm in enumerate(pnames)}
        top = sorted((nm for nm in pnames if cmed[nm] is not None), key=lambda nm: -abs(cmed[nm]))[:12]
        groups = {"legs": [i for i, nm in enumerate(pnames) if re.search("hip|knee|ankle", nm)],
                  "waist": [i for i, nm in enumerate(pnames) if nm.startswith("waist")]}
        for sd in "lr":
            for part in ("shoulder", "elbow", "wrist"):
                groups[f"{sd}_{part}"] = [i for i, nm in enumerate(pnames) if nm.startswith(f"{sd}_{part}")]
        chk = {k: v.cpu().numpy()[hm] for k, v in chain_chk.items()}
        key = ("l_elbow", "l_wrist_yaw", "waist_yaw", "l_shoulder_roll", "l_shoulder_pitch", "l_shoulder_yaw",
               "r_elbow", "r_shoulder_roll", "r_shoulder_pitch", "r_wrist_yaw")
        qrat = {}
        for nm in key:
            i = pnames.index(nm)
            qrat[nm] = dict(qd=med(qd[:, i]), qd_ref=med(qdr[:, i]), ratio=med(qd[:, i] / np.where(np.abs(qdr[:, i]) > 0.5, qdr[:, i], np.nan)))
        pp, pn = pw_pos.cpu().numpy(), pw_neg.cpu().numpy()
        nd = max(float(ph_n["down"]), 1.0)
        arm_ids = [i for i, nm in enumerate(pnames) if re.match(r"(l|r)_(shoulder|elbow|wrist)|waist", nm)]
        power = {pnames[i]: dict(pos_frac=round(float(pp[i] / max(pp[i] + pn[i], 1e-6)), 3), mean_W=round(float((pp[i] - pn[i]) / nd), 1))
                 for i in arm_ids}
        out["chain_log"] = dict(n=int(hm.sum()), v_ss=med(chk["v"]), joints_sum=med(chk["joints"]),
                                residual_link=med(chk["res_link"]), residual_com=med(chk["res_com"]),
                                contrib_top={nm: cmed[nm] for nm in top},
                                contrib_groups={g: med(cc[:, ids].sum(1)) for g, ids in groups.items()},
                                qd_vs_ref=qrat, down_power=power)
        qa, qra = chain["q"].cpu().numpy()[hm], chain["qr"].cpu().numpy()[hm]
        pq = lambda v: [round(float(x), 3) for x in np.nanpercentile(v, [10, 50, 90])] if np.isfinite(v).any() else None
        out["chain_log"]["pose_q"] = {nm: dict(q=pq(qa[:, pnames.index(nm)]), q_ref=pq(qra[:, pnames.index(nm)]),
                                               diff=pq(qa[:, pnames.index(nm)] - qra[:, pnames.index(nm)]))
                                      for nm in ("l_elbow", "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "r_elbow",
                                                 "waist_yaw", "l_wrist_yaw", "l_wrist_pitch")}
        po = {k: v.cpu().numpy()[hm] for k, v in pose.items()}
        po["ss_minus_ref"] = po["ss_rel"] - po["ss_rel_ref"]
        out["chain_log"]["pose_rel_xyz"] = {k: [pq(v[:, a]) for a in range(3)] for k, v in po.items()}
        # lever: sweet-spot distance from the pelvis, robot minus reference (- = the bat is closer to the body)
        out["chain_log"]["pose_ss_dist_minus_ref"] = pq(np.linalg.norm(po["ss_rel"], axis=1) - np.linalg.norm(po["ss_rel_ref"], axis=1))
        for k, v in out["chain_log"].items():
            print(f"[bat] chain_log {k}: " + json.dumps(v), flush=True)
    if args.contact_log:
        hm = r["hit"].astype(bool)
        dzh, dxh = np.abs(r["dz"][hm]), np.abs(r["dx_contact"][hm])
        q = lambda v: [round(float(x), 3) for x in np.nanpercentile(v, [25, 50, 75])] if len(v) else None
        out["contact_log"] = dict(n=int(hm.sum()), ss_contact=q(r["ss_contact"][hm]), ss_max=q(r["ss_speed_max"][hm]),
                                  dx=q(r["dx_contact"][hm]), abs_dz=q(dzh), swing_pitch_deg=q(r["swing_pitch_deg"][hm]),
                                  ev=q(r["ev_contact"][hm]), ref_phase_hit=q(r["ref_phase_hit"][hm]), squared_up=round(float(((dzh <= 0.02) & (dxh <= 0.05)).mean()), 3) if hm.any() else None)
        print("[bat] contact_log: " + json.dumps(out["contact_log"]), flush=True)
        zone_c = (np.abs(r["plate_y"]) <= 0.2525) & (r["plate_z"] >= 0.50) & (r["plate_z"] <= 1.10)  # StrikeZone
        for nm, m in (("hits", hm), ("in-zone hits", hm & zone_c)):  # Human26 falsifier quantities
            Dm = r["D_contact"][m]
            out.setdefault("contact_geom", {})[nm] = dict(
                n=int(m.sum()), abs_dx_med=round(float(np.nanmedian(np.abs(r["dx_contact"][m]))), 4) if m.any() else None,
                D=q(Dm), D_in_0_25=round(float(((Dm >= 0) & (Dm <= 0.025)).mean()), 3) if m.any() else None,
                launch=q(r["launch_contact"][m]), ev=q(r["ev_contact"][m]))
            print(f"[bat] contact_geom {nm}: " + json.dumps(out["contact_geom"][nm]), flush=True)
    if args.body_log:
        q3 = lambda v: [round(float(x), 3) for x in np.nanpercentile(v, [25, 50, 75])] if len(v) else None
        hb, fb = r["hit"].astype(bool), r["fell"].astype(bool)
        ratio = r["yaw_sim"] / np.maximum(r["yaw_ref"], 1e-6)
        groups = dict(hits=hb, misses=~hb, hit_fell=hb & fb, hit_stood=hb & ~fb, miss_fell=~hb & fb, miss_stood=~hb & ~fb)
        out["body_log"] = {g: dict(n=int(m.sum()), pelvis_ratio=q3(ratio[m]), yaw_sim=q3(r["yaw_sim"][m]), yaw_ref=q3(r["yaw_ref"][m]),
                                   **{f"{k}_{sd}": q3(r[f"{k}_{sd}"][m]) for k in ("foot_v_sim", "foot_v_ref", "foot_yaw", "lift_sim",
                                                                                 "lift_ref", "tilt_deg") for sd in "lr"})
                           for g, m in groups.items()}
        for g, v in out["body_log"].items():
            print(f"[bat] body_log {g}: " + json.dumps(v), flush=True)
    if args.link_log:
        hm = r["hit"].astype(bool)
        L = {k: v.cpu().numpy()[hm] for k, v in lk.items()}
        th = t_hit_phys.cpu().numpy()[hm] if args.phase_log else np.full(int(hm.sum()), np.nan)
        md = lambda v: round(float(np.nanmedian(v)), 3) if np.isfinite(v).any() else None
        rows = {}
        for i, nm in enumerate(lk_names):
            rows[nm] = dict(sim_pk=md(L["pk"][:, i]), ref_pk=md(L["pk_ref"][:, i]),
                            ratio=md(L["pk"][:, i] / np.maximum(L["pk_ref"][:, i], 1e-6)),
                            t_pk_sim=md(L["t_pk"][:, i] - th), t_pk_ref=md(L["t_pk_ref"][:, i]),
                            sat_last100ms_pct=round(float(100 * L["sat"][:, i].sum() / max(L["sat_n"][:, i].sum(), 1)), 1))
        out["link_log"] = dict(n=int(hm.sum()), rows=rows)
        print(f"[bat] link_log (hits {int(hm.sum())}; peaks before contact, times in s vs contact; sat over the last 0.1 s of reference time)")
        print(f"[bat] link | sim peak | ref peak | ratio | t peak sim / ref | sat %")
        for nm, v in rows.items():
            print(f"[bat] {nm:18s} | {v['sim_pk']} | {v['ref_pk']} | {v['ratio']} | {v['t_pk_sim']} / {v['t_pk_ref']} | {v['sat_last100ms_pct']}")
    path = os.path.join(ROOT, "logs", f"diag_bat_contact_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(summary=out, per_pitch={k: v.tolist() for k, v in r.items()}), f)
    print(f"[bat] wrote {path}", flush=True)
    if args.dof_dump:  # again after the run (anything written to the sim later)
        dump(base.robot, base.sim, args.dof_dump.replace(".json", "_end.json"), "native end")
    if trace:  # first env per clip, steps while its first episode runs
        names_t = [os.path.splitext(os.path.basename(f))[0] for f in base.cfg.motion_files]
        out_t = {}
        for e in range(n):
            c = names_t[int(clip_first[e])]
            if c in out_t:
                continue
            ks = [0]
            while len(ks) < len(trace) and int(trace[len(ks)]["k"][e]) > int(trace[len(ks) - 1]["k"][e]):
                ks.append(len(ks))  # until its first reset
            out_t[c] = {f: np.stack([trace[i][f][e].cpu().numpy() for i in ks]) for f in ("obs", "act", "q", "k")}
        all_t = {"all__obs": np.stack([s["obs"].cpu().numpy() for s in trace[:12]], 1),  # (n, 12, D): every env's first 12 steps
                 "all__tp": np.stack([s["tp"].cpu().numpy() for s in trace[:12]], 1),
                 "all__tv": np.stack([s["tv"].cpu().numpy() for s in trace[:12]], 1),
                 "all__clip": np.array([names_t[int(c)] for c in clip_first])}
        np.savez(args.trace, **all_t, **{f"{c}__{f}": v for c, d in out_t.items() for f, v in d.items()})
        print(f"[bat] trace -> {args.trace}: {sorted(out_t)} x {[len(d['k']) for d in out_t.values()]} steps", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
