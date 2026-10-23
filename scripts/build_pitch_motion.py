"""Retarget OBP fastballs onto the AIB-1 pitcher and prepare imitation references.

    python scripts/build_pitch_motion.py [data/c3d_pitching/<trial>.c3d ...]
Outputs assets/motions/pitch_<session_pitch>.npz
"""

import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aibaseball.mocap.follow_through import append_follow_through  # noqa: E402
from aibaseball.mocap.motion import prepare  # noqa: E402
from aibaseball.mocap.obp import load_trial  # noqa: E402
from aibaseball.mocap.retarget import retarget, save  # noqa: E402
from aibaseball.robot.ball_grip import FINGER_JOINTS, BallGrip  # noqa: E402
from aibaseball.robot.humanoid import HUMAN_ACTUATOR_GROUPS, build_humanoid  # noqa: E402

GRIP = os.path.join(ROOT, "assets", "aib1_pitcher", "grip.json")
# leg IK: strong temporal smoothness + human joint-speed caps (without them the IK flipped between solutions at
# joint limits around release: 0.6 rad hip-yaw jumps in one frame, ~50 rad/s leg joint speeds)
# Applied only from KEEP_UNTIL_S on: smoothing the whole clip also bent the trunk path before the release (the waist lost
# its forward flexion and the release speed collapsed), so the clean plain-IK frames before that are kept.
LEG_SMOOTH = {"hip": 0.6, "knee": 0.6, "ankle": 0.6, "waist": 0.3}
LEG_MAX_VEL = {k: HUMAN_ACTUATOR_GROUPS[k].velocity for k in ("hip", "knee", "ankle", "waist")}
# planted feet flat (pivot foot on the rubber, lead foot after landing): the markers do not show foot roll and the
# pivot foot rolled 28 deg onto its edge, so the robot stood on the sole's edge and chattered
FLAT = dict(flat_weight=8.0, flat_both=True)
KEEP_UNTIL_S = 0.96  # the first IK flip (waist / lead hip) is at the release frame, 0.98 s
FOLLOW_THROUGH_S = 1.1
# throwing-hand aim warp (ONE OUTS 2026-10-23, user plan 1): AIM_WARP="yaw_deg:lift_deg" rotates the right elbow / wrist /
# hand markers about the right shoulder by yaw (about +z, + toward first base) and lift (about the horizontal axis
# across the hand's path, + up), smoothstep in over release + AIM_TIMES[0..1], held, out over AIM_TIMES[2..3]: on the
# robot the hand lines up with the plate only once it is already swinging down (heading 0 at -3.5 deg elevation,
# +12 deg elevation at heading -7), so the reference's release direction itself is turned
AIM_KEYS = ("r_elbow", "r_wrist", "r_wrist_radial", "r_wrist_ulnar", "r_hand")
# AIM_TIMES="a0:a1:a2:a3" overrides; the first try (out over +0.02 -> +0.05) put the robot's lagging release in the
# warp's way back down (-26 to -31 deg)
AIM_TIMES = tuple(float(x) for x in os.environ.get("AIM_TIMES", "-0.12:-0.06:0.02:0.05").split(":"))


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def aim_warp(trial, yaw_deg, lift_deg, times=AIM_TIMES, elbow=True):
    """Rotate the throwing arm's markers about the shoulder around the release (in place)."""
    t = np.arange(trial.T) / trial.rate - trial.contact / trial.rate
    a0, a1, a2, a3 = times
    w = _smooth((t - a0) / (a1 - a0)) * (1.0 - _smooth((t - a2) / (a3 - a2)))
    hand = trial.lm["r_hand"]
    v = np.gradient(hand, 1.0 / trial.rate, axis=0)[trial.contact]
    across = np.cross([0.0, 0.0, 1.0], v)  # horizontal, across the hand's path
    across /= np.linalg.norm(across)
    sh = trial.lm["r_shoulder"]
    keys = AIM_KEYS if elbow else AIM_KEYS[1:]
    for i in range(trial.T):
        if w[i] <= 0.0:
            continue
        R = (Rotation.from_rotvec(np.deg2rad(yaw_deg * w[i]) * np.array([0.0, 0.0, 1.0]))
             * Rotation.from_rotvec(-np.deg2rad(lift_deg * w[i]) * across))
        for k in keys:
            trial.lm[k][i] = sh[i] + R.apply(trial.lm[k][i] - sh[i])


def hand_dir(p, i, rate):
    """Heading (deg, from atan2(vy, vx)) and elevation (deg) of a point path's velocity at frame i."""
    v = np.gradient(p, 1.0 / rate, axis=0)[i]
    return np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2]))), np.linalg.norm(v)


def main(paths):
    sel = {r["file"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "c3d_pitching", "selected.csv")))}
    robot = build_humanoid(with_bat=False, name="aib1_pitcher")
    # widened joint ranges for the retarget, e.g. JOINT_LIMITS="r_shoulder_yaw:-2.27:1.6" (user 2026-10-20: the layback was
    # clipped at -91.6 deg for the last 0.15 s before the release); OUT_SUFFIX names the output npz
    limits = {}
    for spec in filter(None, os.environ.get("JOINT_LIMITS", "").split(",")):
        jn, lo_, hi_ = spec.split(":")
        limits[jn] = (float(lo_), float(hi_))
        robot.joint(jn).lower, robot.joint(jn).upper = float(lo_), float(hi_)
        print(f"[pitch] joint range override {jn}: [{float(lo_):.3f}, {float(hi_):.3f}] rad", flush=True)
    grip = BallGrip.load(GRIP)
    fingers = {f"l_{n}": 0.2 for n in FINGER_JOINTS} | grip.joint_pos  # glove hand lightly curled
    fingers |= {"l_thumb_cmc_rot": 0.5, "l_index_abd": 0.0, "l_middle_abd": 0.0}
    ball_local = np.array(grip.ball_center_hand)
    for path in paths:
        trial = load_trial(path, body_cutoff=20.0, bat_cutoff=30.0)
        h0 = hand_dir(trial.lm["r_hand"], trial.contact, trial.rate)
        if os.environ.get("AIM_WARP"):
            yaw_, lift_ = (float(x) for x in os.environ["AIM_WARP"].split(":"))
            aim_warp(trial, yaw_, lift_)
            h1 = hand_dir(trial.lm["r_hand"], trial.contact, trial.rate)
            print(f"[pitch] aim warp yaw {yaw_:+.1f} lift {lift_:+.1f} deg: athlete hand at release heading "
                  f"{h0[0]:+.1f} -> {h1[0]:+.1f}, elevation {h0[1]:+.1f} -> {h1[1]:+.1f} deg, {h0[2]:.1f} -> {h1[2]:.1f} m/s", flush=True)
        if os.environ.get("AIM_ONLY"):  # marker check only, no retarget
            continue
        tag = sel.get(os.path.basename(path), {}).get("session_pitch", trial.name).replace("_", "-")
        out = os.path.join(ROOT, "assets", "motions", f"pitch_{tag}{os.environ.get('OUT_SUFFIX', '')}.npz")
        t_rel = trial.contact / trial.rate
        print(f"[pitch] {trial.name}: {trial.exit_velo_mph:.1f} mph, athlete {trial.height:.2f} m, "
              f"release ~{t_rel:.2f} s, peak hand marker {trial.bat_speed_mph.max():.1f} mph", flush=True)
        r0 = retarget(trial, t_before=t_rel, t_after=0.14, rate=120.0, robot=robot, verbose=False,
                      finger_pose=fingers, ball_local=ball_local, **FLAT)
        plain = dict(time=r0.time, root_pos=r0.root_pos, root_quat=r0.root_quat, joint_pos=r0.joint_pos)
        rv = Rotation.from_quat(np.concatenate([plain["root_quat"][:, 1:], plain["root_quat"][:, :1]], -1)).as_rotvec()
        keep = np.concatenate([plain["root_pos"], rv, plain["joint_pos"]], 1)[plain["time"] <= KEEP_UNTIL_S]
        res = retarget(trial, t_before=t_rel, t_after=0.14, rate=120.0, robot=robot, verbose=False,
                       finger_pose=fingers, ball_local=ball_local, smooth=LEG_SMOOTH, max_joint_vel=LEG_MAX_VEL, keep=keep,
                       **FLAT)
        res = append_follow_through(res, robot, fingers, duration=FOLLOW_THROUGH_S, balance_pitch_deg=35.0, balance_waist=0.2)
        save(res, out)
        d = prepare(out, out, finger_pose=fingers, ball_local=ball_local, limits=limits)
        ball = d["key_pos"][:, -1]
        v = np.linalg.norm(np.gradient(ball, 1 / 120, axis=0), axis=1)
        ir = int(res.contact_time * 120)
        bh = hand_dir(ball, ir, 120.0)
        print(f"[pitch]  robot reference ball at release: heading {bh[0]:+.1f} deg, elevation {bh[1]:+.1f} deg, "
              f"{bh[2]:.1f} m/s; 1 frame earlier / later heading {hand_dir(ball, ir - 1, 120.0)[0]:+.1f} / "
              f"{hand_dir(ball, ir + 1, 120.0)[0]:+.1f}, elevation {hand_dir(ball, ir - 1, 120.0)[1]:+.1f} / "
              f"{hand_dir(ball, ir + 1, 120.0)[1]:+.1f}", flush=True)
        over = [f"{n} {np.abs(d['joint_vel'][:, i]).max():.0f}>{robot.joint(str(n)).velocity:.0f}"
                for i, n in enumerate(d["joint_names"]) if np.abs(d["joint_vel"][:, i]).max() > robot.joint(str(n)).velocity]
        print(f"[pitch]  -> {out}\n         keypoint RMS {res.keypoint_err.mean()*100:.1f} cm (max {res.keypoint_err.max()*100:.1f}), "
              f"ball speed at release {v[int(res.contact_time*120)]:.1f} m/s (peak {v.max():.1f}), ball@release {res.tee_pos.round(2)}\n"
              f"         joints over velocity limit: {over or 'none'}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or [os.path.join(ROOT, "data", "c3d_pitching", "001596_002916_73_191_012_FF_944.c3d")]
    main(args)
