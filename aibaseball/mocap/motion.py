"""Reference motion (retargeted OBP swing) for imitation: offline preparation + torch sampler."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from ..physics.bat_model import BatSpec
from ..robot.humanoid import build_humanoid
from ..robot.stance import GRIP_FINGERS

# bodies tracked in the imitation reward (link origins) + the bat tip
KEY_BODIES = ["l_hand", "r_hand", "l_foot", "r_foot", "torso", "bat"]
# pitching: the last key point is the ball (held in the right hand) instead of the bat tip
PITCH_KEY_BODIES = ["l_hand", "r_hand", "l_foot", "r_foot", "torso"]


def prepare(npz_in: str, npz_out: str, finger_pose: dict | None = None, ball_local=None):
    """Add velocities and key-body trajectories (from the robot's FK) to a retarget result.

    Pitching (`ball_local` given): robot without bat, key points = PITCH_KEY_BODIES + ball centre.
    """
    pitching = ball_local is not None
    d = dict(np.load(npz_in))
    t = d["time"]
    dt = float(t[1] - t[0])
    robot0 = build_humanoid()
    lo = np.array([robot0.joint(str(n)).lower for n in d["joint_names"]])
    hi = np.array([robot0.joint(str(n)).upper for n in d["joint_names"]])
    q = np.clip(d["joint_pos"], lo + 1e-3, hi - 1e-3)
    d["joint_pos"] = q
    win = 9 if len(t) > 20 else 5
    qd = savgol_filter(q, win, 3, deriv=1, delta=dt, axis=0)
    p = d["root_pos"]
    v = savgol_filter(p, win, 3, deriv=1, delta=dt, axis=0)
    quat = d["root_quat"]
    R = Rotation.from_quat(np.concatenate([quat[:, 1:], quat[:, :1]], -1))
    w = np.zeros_like(p)
    w[1:-1] = (R[2:] * R[:-2].inv()).as_rotvec() / (2 * dt)
    w[0], w[-1] = w[1], w[-2]
    w = savgol_filter(w, win, 3, axis=0)

    bat = BatSpec()
    robot = build_humanoid(with_bat=not pitching, bat=bat)
    if finger_pose is not None:
        fingers = finger_pose
    else:
        fingers = {f"{s}_{k}": val for s in ("l", "r") for k, val in GRIP_FINGERS.items()}
    names = list(d["joint_names"])
    bodies = PITCH_KEY_BODIES if pitching else KEY_BODIES
    key = np.zeros((len(t), len(bodies) + 1, 3))
    for f in range(len(t)):
        P = robot.fk(dict(zip(names, q[f])) | fingers, root_pos=p[f], root_rot=R[f].as_matrix())
        for i, b in enumerate(bodies):
            key[f, i] = P[b][1]
        if pitching:
            Rh, ph = P["r_hand"]
            key[f, -1] = ph + Rh @ np.asarray(ball_local)
        else:
            Rb, pb = P["bat"]
            key[f, -1] = pb + Rb[:, 0] * bat.length
    last = "ball" if pitching else "bat_tip"
    d.update(joint_vel=qd, root_lin_vel=v, root_ang_vel=w, key_pos=key, key_names=np.array(bodies + [last]))
    np.savez(npz_out, **d)
    return d


class MotionRef:
    """Batched linear interpolation of the reference at arbitrary times (seconds)."""

    def __init__(self, path: str, device):
        d = np.load(path)
        f = lambda k: torch.tensor(d[k], dtype=torch.float32, device=device)  # noqa: E731
        self.dt = float(d["time"][1] - d["time"][0])
        self.n = len(d["time"])
        self.duration = float(d["time"][-1])
        self.contact_time = float(d["contact_time"])
        self.tee_pos = f("tee_pos")
        self.joint_names = [str(n) for n in d["joint_names"]]
        self.root_pos, self.root_quat = f("root_pos"), f("root_quat")
        self.root_lin_vel, self.root_ang_vel = f("root_lin_vel"), f("root_ang_vel")
        self.joint_pos, self.joint_vel = f("joint_pos"), f("joint_vel")
        self.key_pos = f("key_pos")

    def rotate_z(self, angle: float):
        """Rotate the whole reference about the vertical axis through the origin (aiming the delivery)."""
        if angle == 0.0:
            return
        c, s = math.cos(angle), math.sin(angle)
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=self.root_pos.device)
        self.root_pos = self.root_pos @ R.T
        self.root_lin_vel = self.root_lin_vel @ R.T
        self.root_ang_vel = self.root_ang_vel @ R.T
        self.key_pos = self.key_pos @ R.T
        self.tee_pos = self.tee_pos @ R.T
        qz = torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)], device=self.root_pos.device)
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = self.root_quat.unbind(-1)
        self.root_quat = torch.stack([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2], -1)

    def _idx(self, t: torch.Tensor):
        x = (t / self.dt).clamp(0, self.n - 1 - 1e-4)
        i0 = x.floor().long()
        a = (x - i0).unsqueeze(-1)
        return i0, i0 + 1, a

    def sample(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        i0, i1, a = self._idx(t)
        lerp = lambda v: v[i0] * (1 - a.view(-1, *[1] * (v.dim() - 1))) + v[i1] * a.view(-1, *[1] * (v.dim() - 1))  # noqa: E731
        q0, q1 = self.root_quat[i0], self.root_quat[i1]
        q1 = torch.where(((q0 * q1).sum(-1, keepdim=True) < 0), -q1, q1)
        quat = torch.nn.functional.normalize(q0 * (1 - a) + q1 * a, dim=-1)
        return dict(root_pos=lerp(self.root_pos), root_quat=quat, root_lin_vel=lerp(self.root_lin_vel),
                    root_ang_vel=lerp(self.root_ang_vel), joint_pos=lerp(self.joint_pos), joint_vel=lerp(self.joint_vel),
                    key_pos=lerp(self.key_pos))


class MotionLib:
    """Several reference clips (same joint set), sampled per environment: sample(t, clip_ids)."""

    def __init__(self, paths: list[str], device):
        clips = [np.load(p) for p in paths]
        self.names = [str(c["source"]) for c in clips]
        self.joint_names = [str(n) for n in clips[0]["joint_names"]]
        for c in clips:
            assert [str(n) for n in c["joint_names"]] == self.joint_names
        self.dt = float(clips[0]["time"][1] - clips[0]["time"][0])
        self.num_clips = len(clips)
        n_frames = [len(c["time"]) for c in clips]
        T = max(n_frames)

        def stack(key):
            out = []
            for c in clips:
                a = c[key]
                pad = np.repeat(a[-1:], T - len(a), axis=0)  # hold the last frame
                out.append(np.concatenate([a, pad], 0))
            return torch.tensor(np.stack(out), dtype=torch.float32, device=device)

        self.root_pos, self.root_quat = stack("root_pos"), stack("root_quat")
        self.root_lin_vel, self.root_ang_vel = stack("root_lin_vel"), stack("root_ang_vel")
        self.joint_pos, self.joint_vel = stack("joint_pos"), stack("joint_vel")
        self.key_pos = stack("key_pos")
        f = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
        self.n_frames = torch.tensor(n_frames, device=device)
        self.duration = f([float(c["time"][-1]) for c in clips])
        self.contact_time = f([float(c["contact_time"]) for c in clips])
        self.tee_pos = torch.stack([f(c["tee_pos"]) for c in clips])

    def sample(self, t: torch.Tensor, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.minimum((t / self.dt).clamp_min(0), (self.n_frames[clip] - 1).float() - 1e-4)
        i0 = x.floor().long()
        i1 = i0 + 1
        a = (x - i0).unsqueeze(-1)

        def lerp(v):
            v0, v1 = v[clip, i0], v[clip, i1]
            aa = a.view(-1, *[1] * (v0.dim() - 1))
            return v0 * (1 - aa) + v1 * aa

        q0, q1 = self.root_quat[clip, i0], self.root_quat[clip, i1]
        q1 = torch.where(((q0 * q1).sum(-1, keepdim=True) < 0), -q1, q1)
        quat = torch.nn.functional.normalize(q0 * (1 - a) + q1 * a, dim=-1)
        return dict(root_pos=lerp(self.root_pos), root_quat=quat, root_lin_vel=lerp(self.root_lin_vel),
                    root_ang_vel=lerp(self.root_ang_vel), joint_pos=lerp(self.joint_pos), joint_vel=lerp(self.joint_vel),
                    key_pos=lerp(self.key_pos))
