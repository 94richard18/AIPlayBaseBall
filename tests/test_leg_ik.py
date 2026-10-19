"""aibaseball.robot.leg_ik: FK matches the kinematic model; the catch-step target is reachable from the pelvis poses the
gz 7300 standers / fallers had after the release (logs/falldiag_poses.npz from diag_fall_release.py), within 1 cm."""

import math
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from aibaseball.robot.humanoid import build_humanoid
from aibaseball.robot.leg_ik import LegIK

TARGET, YAW = (-0.10, -0.52), 79.0


def test_fk_matches_model():
    rb, ik = build_humanoid(with_bat=False, name="aib1_pitcher"), LegIK()
    rng = np.random.default_rng(0)
    q = rng.uniform(ik.lo.numpy(), ik.hi.numpy(), (20, 6))
    R0, p0 = Rotation.random(20, random_state=1).as_matrix(), rng.normal(size=(20, 3))
    R, p = ik.fk(torch.tensor(q, dtype=torch.float32), torch.tensor(R0, dtype=torch.float32), torch.tensor(p0, dtype=torch.float32))
    for i in range(20):
        Rr, pr = rb.fk(dict(zip(ik.names, q[i])), root_pos=p0[i], root_rot=R0[i])["r_foot"]
        assert np.allclose(pr, p[i].numpy(), atol=1e-5) and np.allclose(Rr, R[i].numpy(), atol=1e-5)


def test_catch_target_reachable(path="logs/falldiag_poses.npz"):
    if not os.path.exists(path):
        return
    d, ik = np.load(path), LegIK()
    lead = d["leadp"]
    for key in ("pel025", "pel04"):
        pel = d[key]
        ok = ~np.isnan(pel[:, 0]) & ~np.isnan(lead[:, 0])
        n = int(ok.sum())
        p0 = torch.tensor(lead[ok] + pel[ok, :3], dtype=torch.float32)  # pelvis pos (lead foot local frame)
        qw = pel[ok, 3:]
        R0 = torch.tensor(Rotation.from_quat(np.c_[qw[:, 1:], qw[:, :1]]).as_matrix(), dtype=torch.float32)
        tgt = torch.tensor(lead[ok], dtype=torch.float32)
        tgt[:, 0] += TARGET[0]
        tgt[:, 1] += TARGET[1]
        lyaw = 0.0  # back-foot yaw relative to the lead foot's; lead foot yaw is not in the file - test the relative pose
        R_t = torch.tensor(Rotation.from_euler("z", math.radians(YAW + lyaw)).as_matrix(), dtype=torch.float32).expand(n, 3, 3)
        q, pe, re = ik.solve(R0, p0, R_t, tgt, torch.zeros(n, 6), iters=20)
        lim = ((q <= ik.lo + 1e-3) | (q >= ik.hi - 1e-3)).any(-1)
        print(f"{key}: {n} poses, pos err p50/p90/max {np.percentile(pe.numpy() * 1000, [50, 90]).round(1)} / "
              f"{pe.max().item() * 1000:.1f} mm, rot err max {math.degrees(re.max().item()):.1f} deg, at a joint limit {int(lim.sum())}")
        assert (q >= ik.lo - 1e-6).all() and (q <= ik.hi + 1e-6).all()
