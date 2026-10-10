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


def main(paths):
    sel = {r["file"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "c3d_pitching", "selected.csv")))}
    robot = build_humanoid(with_bat=False, name="aib1_pitcher")
    grip = BallGrip.load(GRIP)
    fingers = {f"l_{n}": 0.2 for n in FINGER_JOINTS} | grip.joint_pos  # glove hand lightly curled
    fingers |= {"l_thumb_cmc_rot": 0.5, "l_index_abd": 0.0, "l_middle_abd": 0.0}
    ball_local = np.array(grip.ball_center_hand)
    for path in paths:
        trial = load_trial(path, body_cutoff=20.0, bat_cutoff=30.0)
        tag = sel.get(os.path.basename(path), {}).get("session_pitch", trial.name).replace("_", "-")
        out = os.path.join(ROOT, "assets", "motions", f"pitch_{tag}.npz")
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
        res = append_follow_through(res, robot, fingers, duration=FOLLOW_THROUGH_S)
        save(res, out)
        d = prepare(out, out, finger_pose=fingers, ball_local=ball_local)
        ball = d["key_pos"][:, -1]
        v = np.linalg.norm(np.gradient(ball, 1 / 120, axis=0), axis=1)
        over = [f"{n} {np.abs(d['joint_vel'][:, i]).max():.0f}>{robot.joint(str(n)).velocity:.0f}"
                for i, n in enumerate(d["joint_names"]) if np.abs(d["joint_vel"][:, i]).max() > robot.joint(str(n)).velocity]
        print(f"[pitch]  -> {out}\n         keypoint RMS {res.keypoint_err.mean()*100:.1f} cm (max {res.keypoint_err.max()*100:.1f}), "
              f"ball speed at release {v[int(res.contact_time*120)]:.1f} m/s (peak {v.max():.1f}), ball@release {res.tee_pos.round(2)}\n"
              f"         joints over velocity limit: {over or 'none'}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or [os.path.join(ROOT, "data", "c3d_pitching", "001596_002916_73_191_012_FF_944.c3d")]
    main(args)
