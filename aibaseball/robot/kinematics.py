"""Minimal kinematic-tree model used to author the robot (URDF) and solve stance IK offline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- rotations
def rot_axis(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rpy_to_mat(rpy) -> np.ndarray:
    r, p, y = rpy
    return rot_axis([0, 0, 1], y) @ rot_axis([0, 1, 0], p) @ rot_axis([1, 0, 0], r)


def mat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def frame_from_z(direction) -> np.ndarray:
    """Rotation whose local +Z is `direction` (for cylinders along a segment)."""
    z = np.asarray(direction, float)
    z = z / np.linalg.norm(z)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def mat_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return (w, x, y, z)


# ----------------------------------------------------------------------------- tree model
@dataclass
class Shape:
    kind: str  # "box" | "cylinder" | "sphere" | "mesh"
    size: tuple  # box: (sx,sy,sz); cylinder: (radius, length); sphere: (radius,); mesh: (filename, scale)
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    collision: bool = True
    visual: bool = True
    color: str = "body"


@dataclass
class Link:
    name: str
    mass: float
    com: np.ndarray
    inertia: np.ndarray  # 3x3 about COM, link frame
    shapes: list[Shape] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    pos: np.ndarray  # child frame origin in parent frame (at q = 0)
    rot: np.ndarray  # child frame orientation in parent frame (at q = 0)
    type: str = "revolute"  # "revolute" | "fixed"
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))  # in child frame
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0
    group: str = ""  # actuator group name


@dataclass
class Robot:
    name: str
    root: str
    links: dict[str, Link] = field(default_factory=dict)
    joints: list[Joint] = field(default_factory=list)

    def add_link(self, link: Link) -> Link:
        assert link.name not in self.links, link.name
        self.links[link.name] = link
        return link

    def add_joint(self, joint: Joint) -> Joint:
        assert joint.parent in self.links and joint.child in self.links, (joint.parent, joint.child)
        self.joints.append(joint)
        return joint

    @property
    def actuated(self) -> list[Joint]:
        return [j for j in self.joints if j.type != "fixed"]

    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)

    def fk(self, q: dict[str, float], root_pos=np.zeros(3), root_rot=np.eye(3)) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World pose (R, p) of every link."""
        poses = {self.root: (np.asarray(root_rot, float), np.asarray(root_pos, float))}
        # joints are added parent-before-child
        for j in self.joints:
            Rp, pp = poses[j.parent]
            R = Rp @ j.rot
            p = pp + Rp @ j.pos
            if j.type == "revolute":
                R = R @ rot_axis(j.axis, q.get(j.name, 0.0))
            poses[j.child] = (R, p)
        return poses


# ----------------------------------------------------------------------------- inertia helpers
def inertia_box(m, sx, sy, sz):
    return np.diag([m * (sy**2 + sz**2) / 12, m * (sx**2 + sz**2) / 12, m * (sx**2 + sy**2) / 12])


def inertia_cylinder_z(m, r, h):
    i_t = m * (3 * r**2 + h**2) / 12
    return np.diag([i_t, i_t, 0.5 * m * r**2])


def rotate_inertia(I, R):
    return R @ I @ R.T


def mirror_y(v) -> np.ndarray:
    v = np.array(v, float)
    v[1] = -v[1]
    return v


def mirror_axis(a) -> np.ndarray:
    """A rotation axis under reflection y -> -y maps to (-ax, ay, -az) with the same angle sense."""
    a = np.array(a, float)
    return np.array([-a[0], a[1], -a[2]])


def mirror_rot(R) -> np.ndarray:
    M = np.diag([1.0, -1.0, 1.0])
    return M @ R @ M
