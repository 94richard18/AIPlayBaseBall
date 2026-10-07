"""Custom 1.85 m / ~88 kg humanoid with two 16-DOF five-finger hands ("AIB-1").

Joint count: legs 2x6, waist 3, arms 2x7, hands 2x16  = 61 DOF.
Link count with the optional bat: 63 (PhysX articulations allow at most 64 links).

Frames: robot base frame x forward, y left, z up. Every side-specific part is authored for
the RIGHT side (y < 0) and mirrored (y -> -y) for the left side.

Hand design (palm frame, right hand): x distal, y toward the thumb, z dorsal.
  index / middle : abduction, MCP, PIP, DIP            (4 DOF each)
  ring / pinky   : MCP, PIP (DIP rigidly coupled, pre-bent)  (2 DOF each)
  thumb          : CMC opposition roll, CMC flexion, MCP, IP (4 DOF)
The pitching task needs independent index/middle/thumb control (grip, release, spin);
ring and pinky only support the ball, so they get fewer joints to stay within the link budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics.bat_model import BatSpec
from .kinematics import (
    Joint,
    Link,
    Robot,
    Shape,
    frame_from_z,
    inertia_box,
    inertia_cylinder_z,
    mirror_axis,
    mirror_rot,
    mirror_y,
    rot_axis,
    rotate_inertia,
)

X, Y, Z = np.eye(3)
I3 = np.eye(3)

# Palm frame of the RIGHT hand expressed in the hand-link frame (hand link frame == robot frame
# at q = 0, arm hanging down, palm facing the body midline, thumb forward).
R_PALM_RIGHT = np.stack([-Z, X, -Y], axis=1)

# Point where a bat handle sits in a closed fist (palm frame), and the handle direction (toward thumb).
GRIP_POINT_PALM = np.array([0.080, 0.0, -0.027])
GRIP_DIR_PALM = np.array([0.0, 1.0, 0.0])


@dataclass
class ActuatorGroup:
    effort: float
    velocity: float
    stiffness: float
    damping: float
    armature: float


# Strong electric-humanoid class actuators (Nm, rad/s). Joint ranges/speeds cover a retargeted
# elite human swing (OBP); arms/wrists/waist are faster than a human to reach the required
# ~43 m/s sweet-spot bat speed (human elite ~36-38 m/s). Torques were doubled after the imitation
# diagnostics showed saturation of waist/hips/shoulders/wrists already at the human swing speed, and
# raised again (torque x1.5, speed x1.4) when the 1.15x-speed swing saturated torque and joint speed.
# Fingers: a fastball at release needs ~280 N centripetal grip on the ball -> 12-15 Nm finger joints.
ACTUATOR_GROUPS: dict[str, ActuatorGroup] = {
    "hip": ActuatorGroup(800.0, 35.0, 2000.0, 50.0, 0.05),
    "knee": ActuatorGroup(700.0, 35.0, 2000.0, 50.0, 0.05),
    "ankle": ActuatorGroup(300.0, 30.0, 1000.0, 25.0, 0.02),
    "waist": ActuatorGroup(900.0, 60.0, 1500.0, 40.0, 0.05),
    "shoulder": ActuatorGroup(540.0, 60.0, 600.0, 15.0, 0.02),
    "elbow": ActuatorGroup(420.0, 65.0, 400.0, 10.0, 0.015),
    "wrist": ActuatorGroup(180.0, 70.0, 150.0, 3.0, 0.005),
    "finger": ActuatorGroup(12.0, 20.0, 30.0, 0.5, 0.0005),
    "thumb": ActuatorGroup(15.0, 20.0, 40.0, 0.6, 0.0005),
}

# Human-level limits (peak values reported in pitching / strength biomechanics) for the pitcher:
# hip extension ~300 Nm, knee ~250 Nm, plantarflexion ~200 Nm, trunk axial rotation ~200 Nm,
# shoulder (internal rotation torque in pitching ~70-100 Nm, IR velocity up to ~7000 deg/s),
# elbow ~80 Nm / ~2500 deg/s, wrist ~20 Nm, fingertip forces ~50-100 N.
# Constant-torque PD up to the speed limit is an upper bound of muscle (no force-velocity drop).
HUMAN_ACTUATOR_GROUPS: dict[str, ActuatorGroup] = {
    "hip": ActuatorGroup(300.0, 20.0, 2000.0, 50.0, 0.05),
    "knee": ActuatorGroup(250.0, 25.0, 2000.0, 50.0, 0.05),
    "ankle": ActuatorGroup(200.0, 20.0, 1000.0, 25.0, 0.02),
    "waist": ActuatorGroup(200.0, 25.0, 1500.0, 40.0, 0.05),
    "shoulder": ActuatorGroup(100.0, 120.0, 600.0, 15.0, 0.02),
    "elbow": ActuatorGroup(80.0, 50.0, 400.0, 10.0, 0.015),
    "wrist": ActuatorGroup(20.0, 40.0, 150.0, 3.0, 0.005),
    "finger": ActuatorGroup(5.0, 20.0, 30.0, 0.5, 0.0005),
    "thumb": ActuatorGroup(7.0, 20.0, 40.0, 0.6, 0.0005),
}


def _cyl_between(p0, p1, radius, color="body", collision=True) -> Shape:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    return Shape("cylinder", (radius, float(np.linalg.norm(d))), pos=(p0 + p1) / 2, rot=frame_from_z(d), color=color,
                 collision=collision)


def _seg_link(name, mass, p0, p1, radius, color="body", extra: list[Shape] | None = None) -> Link:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    I = rotate_inertia(inertia_cylinder_z(mass, radius, float(np.linalg.norm(d))), frame_from_z(d))
    shapes = [_cyl_between(p0, p1, radius, color)] + (extra or [])
    return Link(name, mass, (p0 + p1) / 2, I, shapes)


def _point_link(name, mass, radius=0.02, collision=False) -> Link:
    I = np.eye(3) * (0.4 * mass * radius**2)
    return Link(name, mass, np.zeros(3), I, [Shape("sphere", (radius,), collision=collision, color="joint")])


class _SideBuilder:
    """Adds parts authored for the right side, mirrored when side == 'l'."""

    def __init__(self, robot: Robot, side: str):
        self.robot, self.side = robot, side
        self.mirror = side == "l"

    def n(self, name):
        return f"{self.side}_{name}"

    def p(self, v):
        return mirror_y(v) if self.mirror else np.array(v, float)

    def a(self, v):
        return mirror_axis(v) if self.mirror else np.array(v, float)

    def R(self, R):
        return mirror_rot(R) if self.mirror else np.array(R, float)

    def link(self, link: Link) -> Link:
        if self.mirror:
            M = np.diag([1.0, -1.0, 1.0])
            link.com = mirror_y(link.com)
            link.inertia = M @ link.inertia @ M
            for s in link.shapes:
                s.pos = mirror_y(s.pos)
                s.rot = mirror_rot(s.rot)
        link.name = self.n(link.name)
        return self.robot.add_link(link)

    def joint(self, name, parent, child, pos, axis, lower, upper, group, rot=I3, joint_type="revolute"):
        g = ACTUATOR_GROUPS.get(group)
        # Mirroring flips the sense of abduction/roll/yaw-type motions only through the axis;
        # limits stay in anatomical terms (same numbers on both sides).
        return self.robot.add_joint(
            Joint(
                name=self.n(name),
                parent=parent,
                child=self.n(child) if not child.startswith(("l_", "r_")) else child,
                pos=self.p(pos),
                rot=self.R(rot),
                type=joint_type,
                axis=self.a(axis),
                lower=lower,
                upper=upper,
                effort=g.effort if g else 0.0,
                velocity=g.velocity if g else 0.0,
                group=group,
            )
        )


# ----------------------------------------------------------------------------- body parts
def _add_leg(robot: Robot, side: str):
    b = _SideBuilder(robot, side)
    b.link(_point_link("hip_yaw_link", 1.5, 0.05))
    b.joint("hip_yaw", "pelvis", "hip_yaw_link", [0, -0.10, -0.03], Z, -1.0, 1.0, "hip")
    b.link(_point_link("hip_roll_link", 1.5, 0.05))
    b.joint("hip_roll", b.n("hip_yaw_link"), "hip_roll_link", [0, 0, 0], X, -0.8, 0.4, "hip")
    b.link(_seg_link("thigh", 9.0, [0, 0, -0.04], [0, 0, -0.43], 0.07))
    b.joint("hip_pitch", b.n("hip_roll_link"), "thigh", [0, 0, 0], Y, -2.2, 0.8, "hip")
    b.link(_seg_link("shank", 4.5, [0, 0, -0.03], [0, 0, -0.42], 0.05))
    b.joint("knee", b.n("thigh"), "shank", [0, 0, -0.45], Y, 0.0, 2.4, "knee")
    b.link(_point_link("ankle_pitch_link", 0.5, 0.035))
    b.joint("ankle_pitch", b.n("shank"), "ankle_pitch_link", [0, 0, -0.45], Y, -1.0, 0.9, "ankle")
    foot_size = (0.26, 0.11, 0.05)
    foot_c = np.array([0.04, 0.0, -0.055])
    b.link(Link("foot", 1.5, foot_c, inertia_box(1.5, *foot_size),
                [Shape("box", foot_size, pos=foot_c, color="shoe")]))
    b.joint("ankle_roll", b.n("ankle_pitch_link"), "foot", [0, 0, 0], X, -0.5, 0.5, "ankle")


# Arm segment masses (kg). "robot": original actuator-housing masses; "human": de Leva (1996) proportions
# for an 86.6 kg male (upper arm 2.7 %, forearm 1.6 %, hand 0.6 % of body mass) - used for the pitcher.
ARM_MASSES = {
    "robot": dict(shoulder_pitch=1.0, shoulder_roll=0.5, upper_arm=3.0, forearm=1.2, forearm_distal=0.8, wrist=0.1, hand=0.40),
    "human": dict(shoulder_pitch=0.2, shoulder_roll=0.15, upper_arm=2.0, forearm=0.8, forearm_distal=0.55, wrist=0.05, hand=0.33),
}


def _add_arm(robot: Robot, side: str, masses: str = "robot"):
    b = _SideBuilder(robot, side)
    m = ARM_MASSES[masses]
    b.link(_point_link("shoulder_pitch_link", m["shoulder_pitch"], 0.055, collision=False))
    b.joint("shoulder_pitch", "torso", "shoulder_pitch_link", [0, -0.20, 0.40], Y, -3.1, 1.2, "shoulder")
    b.link(_point_link("shoulder_roll_link", m["shoulder_roll"], 0.05))
    b.joint("shoulder_roll", b.n("shoulder_pitch_link"), "shoulder_roll_link", [0, 0, 0], X, -2.8, 1.5, "shoulder")
    b.link(_seg_link("upper_arm", m["upper_arm"], [0, 0, -0.05], [0, 0, -0.28], 0.045))
    b.joint("shoulder_yaw", b.n("shoulder_roll_link"), "upper_arm", [0, 0, 0], Z, -1.6, 1.6, "shoulder")
    b.link(_seg_link("forearm", m["forearm"], [0, 0, -0.03], [0, 0, -0.13], 0.04))
    b.joint("elbow", b.n("upper_arm"), "forearm", [0, 0, -0.30], -Y, 0.0, 2.6, "elbow")
    b.link(_seg_link("forearm_distal", m["forearm_distal"], [0, 0, 0.0], [0, 0, -0.12], 0.035))
    b.joint("wrist_roll", b.n("forearm"), "forearm_distal", [0, 0, -0.13], Z, -1.6, 1.6, "wrist")
    b.link(_point_link("wrist_pitch_link", m["wrist"], 0.025))
    b.joint("wrist_pitch", b.n("forearm_distal"), "wrist_pitch_link", [0, 0, -0.14], X, -1.2, 1.2, "wrist")
    _add_hand(b, m["hand"])


def _palm(v):
    """Palm-frame vector -> hand-link frame (right-hand authoring)."""
    return R_PALM_RIGHT @ np.asarray(v, float)


def _add_hand(b: _SideBuilder, palm_mass: float = 0.40):
    palm_size = np.array([0.095, 0.084, 0.030])
    palm_c = _palm([0.0475, 0.0, 0.0])
    b.link(Link("hand", palm_mass, palm_c, rotate_inertia(inertia_box(palm_mass, *palm_size), R_PALM_RIGHT),
                [Shape("box", tuple(palm_size), pos=palm_c, rot=R_PALM_RIGHT, color="hand")]))
    b.joint("wrist_yaw", b.n("wrist_pitch_link"), "hand", [0, 0, 0], -Y, -0.9, 0.8, "wrist")
    hand = b.n("hand")

    def seg(name, mass, length, radius, bend=0.0, tail: tuple[float, float] | None = None) -> Link:
        """Finger segment along palm +x from the joint; optional rigid distal 'tail' (length, radius) pre-bent."""
        p1 = _palm([length, 0, 0])
        link = _seg_link(name, mass, [0, 0, 0], p1, radius, color="hand")
        if tail:
            d = rot_axis(_palm(Y), bend) @ _palm(X)
            link.shapes.append(_cyl_between(p1, p1 + d * tail[0], tail[1], color="hand"))
        return link

    fingers = {  # name: (base in palm frame, (prox, mid, dist) lengths, radius, full 4-DOF?)
        "index": ([0.095, 0.027, 0.0], (0.044, 0.025, 0.022), 0.0090, True),
        "middle": ([0.095, 0.008, 0.0], (0.048, 0.029, 0.024), 0.0092, True),
        "ring": ([0.093, -0.011, 0.0], (0.045, 0.027, 0.023), 0.0088, False),
        "pinky": ([0.087, -0.029, 0.0], (0.035, 0.021, 0.020), 0.0078, False),
    }
    flex = _palm(Y)  # positive = curl toward the palm (-z_palm)
    abd = _palm(Z)
    for f, (base, (l1, l2, l3), r, full) in fingers.items():
        parent = hand
        if full:
            b.link(_point_link(f"{f}_knuckle", 0.010, 0.008))
            b.joint(f"{f}_abd", parent, f"{f}_knuckle", _palm(base), abd, -0.35, 0.35, "finger")
            parent = b.n(f"{f}_knuckle")
            b.link(seg(f"{f}_proximal", 0.020, l1, r))
            b.joint(f"{f}_mcp", parent, f"{f}_proximal", [0, 0, 0], flex, -0.3, 1.6, "finger")
            b.link(seg(f"{f}_middle", 0.012, l2, r * 0.95))
            b.joint(f"{f}_pip", b.n(f"{f}_proximal"), f"{f}_middle", _palm([l1, 0, 0]), flex, 0.0, 1.9, "finger")
            b.link(seg(f"{f}_distal", 0.008, l3, r * 0.9))
            b.joint(f"{f}_dip", b.n(f"{f}_middle"), f"{f}_distal", _palm([l2, 0, 0]), flex, 0.0, 1.4, "finger")
        else:
            b.link(seg(f"{f}_proximal", 0.020, l1, r))
            b.joint(f"{f}_mcp", parent, f"{f}_proximal", _palm(base), flex, -0.3, 1.6, "finger")
            b.link(seg(f"{f}_middle", 0.018, l2, r * 0.95, bend=0.35, tail=(l3, r * 0.9)))
            b.joint(f"{f}_pip", b.n(f"{f}_proximal"), f"{f}_middle", _palm([l1, 0, 0]), flex, 0.0, 1.9, "finger")

    # thumb: base frame pointing distal-lateral and slightly volar
    R_t = rot_axis(_palm(Z), 0.9) @ rot_axis(_palm(X), -0.5)
    t_x = R_t @ _palm(X)  # thumb long axis
    t_flex = R_t @ _palm(-Z)  # flex toward palm centre in the thumb plane
    base = _palm([0.020, 0.032, -0.012])
    b.link(_point_link("thumb_base", 0.010, 0.010))
    b.joint("thumb_cmc_rot", hand, "thumb_base", base, t_x, -0.3, 1.5, "thumb")

    def tseg(name, mass, length, radius):
        return _seg_link(name, mass, [0, 0, 0], t_x * length, radius, color="hand")

    b.link(tseg("thumb_metacarpal", 0.025, 0.045, 0.0110))
    b.joint("thumb_cmc_flex", b.n("thumb_base"), "thumb_metacarpal", [0, 0, 0], t_flex, -0.4, 1.0, "thumb")
    b.link(tseg("thumb_proximal", 0.015, 0.034, 0.0100))
    b.joint("thumb_mcp", b.n("thumb_metacarpal"), "thumb_proximal", t_x * 0.045, t_flex, -0.2, 1.0, "thumb")
    b.link(tseg("thumb_distal", 0.010, 0.028, 0.0095))
    b.joint("thumb_ip", b.n("thumb_proximal"), "thumb_distal", t_x * 0.034, t_flex, -0.2, 1.4, "thumb")


def palm_grip_frame(side: str) -> tuple[np.ndarray, np.ndarray]:
    """(grip point, handle direction toward the barrel) in the hand-link frame of the given side."""
    p = _palm(GRIP_POINT_PALM)
    d = _palm(GRIP_DIR_PALM)
    if side == "l":
        return mirror_y(p), mirror_y(d)
    return p, d


def _add_bat(robot: Robot, bat: BatSpec, bottom_hand: str = "l"):
    """Bat rigidly held by the bottom hand (knob side). The top hand is closed with a loop joint in USD."""
    props = bat.mass_properties()
    grip_p, grip_d = palm_grip_frame(bottom_hand)
    # bat frame: +x along the bat (knob -> barrel); choose y,z orthonormal
    R_bat = frame_from_z(grip_d) @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])  # columns: x=grip_d
    assert np.allclose(R_bat[:, 0], grip_d / np.linalg.norm(grip_d))
    origin = grip_p - bat.bottom_hand_x * R_bat[:, 0]
    I_bat = np.diag([props["inertia_axial"], props["inertia_transverse"], props["inertia_transverse"]])
    shapes = [Shape("mesh", ("meshes/bat.obj", 1.0), collision=False, color="bat")]
    for x0, x1, r in bat.collision_capsules(6):
        shapes.append(_cyl_between([x0, 0, 0], [x1, 0, 0], r, color="bat", collision=True))
        shapes[-1].visual = False
    robot.add_link(Link("bat", bat.mass, np.array([props["cm_x"], 0, 0]), I_bat, shapes))
    robot.add_joint(Joint("bat_grip", f"{bottom_hand}_hand", "bat", origin, R_bat, type="fixed"))


def build_humanoid(with_bat: bool = True, bat: BatSpec | None = None, name: str = "aib1",
                   arm_masses: str = "robot") -> Robot:
    robot = Robot(name, root="pelvis")
    pelvis_size = (0.20, 0.30, 0.16)
    robot.add_link(Link("pelvis", 11.0, np.array([0, 0, 0.03]), inertia_box(11.0, *pelvis_size),
                        [Shape("box", pelvis_size, pos=np.array([0, 0, 0.03]))]))
    for s in ("l", "r"):
        _add_leg(robot, s)

    robot.add_link(_point_link("waist_yaw_link", 1.0, 0.05))
    robot.add_joint(Joint("waist_yaw", "pelvis", "waist_yaw_link", np.array([0, 0, 0.12]), I3, axis=Z,
                          lower=-1.4, upper=1.4, **_g("waist")))
    robot.add_link(_point_link("waist_roll_link", 1.0, 0.05))
    robot.add_joint(Joint("waist_roll", "waist_yaw_link", "waist_roll_link", np.zeros(3), I3, axis=X,
                          lower=-0.7, upper=0.7, **_g("waist")))
    torso_box = (0.22, 0.36, 0.38)
    torso_c = np.array([0, 0, 0.22])
    torso = Link("torso", 30.0, np.array([0, 0, 0.27]), inertia_box(30.0, 0.22, 0.36, 0.60), [
        Shape("box", torso_box, pos=torso_c),
        Shape("cylinder", (0.045, 0.10), pos=np.array([0, 0, 0.47])),
        Shape("sphere", (0.10,), pos=np.array([0, 0, 0.62]), color="head"),
    ])
    robot.add_link(torso)
    robot.add_joint(Joint("waist_pitch", "waist_roll_link", "torso", np.zeros(3), I3, axis=Y,
                          lower=-0.7, upper=1.2, **_g("waist")))
    for s in ("l", "r"):
        _add_arm(robot, s, arm_masses)
    if with_bat:
        _add_bat(robot, bat or BatSpec(), bottom_hand="l")
    return robot


def _g(group: str) -> dict:
    g = ACTUATOR_GROUPS[group]
    return dict(effort=g.effort, velocity=g.velocity, group=group)


# Standing height of the pelvis frame with straight legs: hip offset + thigh + shank + ankle-to-sole.
PELVIS_HEIGHT_STRAIGHT = 0.03 + 0.45 + 0.45 + 0.08


def summary(robot: Robot) -> str:
    total = sum(l.mass for l in robot.links.values())
    n_dof = len(robot.actuated)
    return f"{robot.name}: {len(robot.links)} links, {n_dof} DOF, mass {total:.1f} kg"


if __name__ == "__main__":
    r = build_humanoid()
    print(summary(r))
