"""Batched 6-DOF leg IK (torch) on the AIB-1 kinematic model: hip yaw / roll / pitch, knee, ankle pitch / roll.

Used by the follow-through catch step (b1'): every step, per env, the back-leg joint angles that put the foot link at a
world pose given the current pelvis pose. Damped Gauss-Newton on position (m) + orientation (rotation vector, weighted)
with a finite-difference Jacobian, warm-started, clamped to the joint limits.
"""

from __future__ import annotations

import torch

from .humanoid import build_humanoid


def _rot_axis(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation about a unit axis (3,) by ang (...,)."""
    x, y, z = axis
    c, s = torch.cos(ang), torch.sin(ang)
    C = 1.0 - c
    return torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1),
    ], -2)


def rotvec_of(R: torch.Tensor) -> torch.Tensor:
    """Rotation vector of (..., 3, 3) rotations (angles < pi)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    ang = torch.acos(((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return w * (ang / (2.0 * torch.sin(ang).clamp_min(1e-6))).unsqueeze(-1)


class LegIK:
    def __init__(self, side: str = "r", device="cpu", dtype=torch.float32):
        robot = build_humanoid(with_bat=False, name="aib1_pitcher")
        self.names = [f"{side}_{n}" for n in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")]
        js = [robot.joint(n) for n in self.names]
        assert js[0].parent == robot.root, js[0].parent
        for a, b in zip(js[:-1], js[1:]):
            assert b.parent == a.child, (a.name, b.name)
        f = lambda v: torch.as_tensor(v, dtype=dtype, device=device)  # noqa: E731
        self.pos = [f(j.pos) for j in js]
        self.rot = [f(j.rot) for j in js]
        self.axis = [f(j.axis) for j in js]
        self.lo = f([j.lower for j in js])
        self.hi = f([j.upper for j in js])

    def fk(self, q: torch.Tensor, R0: torch.Tensor, p0: torch.Tensor):
        """Foot link pose (R, p) for joint angles q (n, 6) and pelvis pose R0 (n, 3, 3), p0 (n, 3)."""
        R, p = R0, p0
        for i in range(6):
            p = p + (R @ self.pos[i].unsqueeze(-1)).squeeze(-1)
            R = R @ self.rot[i] @ _rot_axis(self.axis[i], q[:, i])
        return R, p

    def solve(self, R0, p0, R_t, p_t, q0, iters: int = 8, w_rot: float = 0.3, damping: float = 1e-3, eps: float = 1e-4,
              max_step: float = 0.25):
        """Joint angles (n, 6) putting the foot at (R_t, p_t); also returns the final position error (m) and rotation error (rad)."""
        q = q0.clamp(self.lo, self.hi).clone()

        def resid(qq):
            R, p = self.fk(qq, R0, p0)
            return torch.cat([p - p_t, w_rot * rotvec_of(R_t.transpose(-1, -2) @ R)], -1)

        for _ in range(iters):
            r = resid(q)
            J = torch.stack([(resid(q + eps * torch.eye(6, dtype=q.dtype, device=q.device)[j]) - r) / eps for j in range(6)], -1)
            JT = J.transpose(-1, -2)
            dq = torch.linalg.solve(JT @ J + damping * torch.eye(6, dtype=q.dtype, device=q.device), -(JT @ r.unsqueeze(-1)))
            dq = dq.squeeze(-1)
            dq = dq * (max_step / dq.abs().amax(-1, keepdim=True).clamp_min(max_step))  # trust region per iteration
            q = (q + dq).clamp(self.lo, self.hi)
        R, p = self.fk(q, R0, p0)
        return q, (p - p_t).norm(dim=-1), rotvec_of(R_t.transpose(-1, -2) @ R).norm(dim=-1)
