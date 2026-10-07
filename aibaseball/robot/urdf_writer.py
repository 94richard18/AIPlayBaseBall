"""Write a `Robot` tree to URDF (+ bat mesh as OBJ)."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from ..physics.bat_model import BatSpec
from .kinematics import Robot, mat_to_rpy

COLORS = {
    "body": (0.82, 0.84, 0.88, 1.0),
    "joint": (0.25, 0.27, 0.30, 1.0),
    "hand": (0.30, 0.32, 0.36, 1.0),
    "shoe": (0.10, 0.10, 0.12, 1.0),
    "head": (0.15, 0.45, 0.85, 1.0),
    "bat": (0.78, 0.60, 0.38, 1.0),
}


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _origin(parent, pos, rot):
    ET.SubElement(parent, "origin", xyz=_fmt(pos), rpy=_fmt(mat_to_rpy(rot)))


def _geometry(parent, shape):
    g = ET.SubElement(parent, "geometry")
    if shape.kind == "box":
        ET.SubElement(g, "box", size=_fmt(shape.size))
    elif shape.kind == "cylinder":
        ET.SubElement(g, "cylinder", radius=f"{shape.size[0]:.6g}", length=f"{shape.size[1]:.6g}")
    elif shape.kind == "sphere":
        ET.SubElement(g, "sphere", radius=f"{shape.size[0]:.6g}")
    elif shape.kind == "mesh":
        ET.SubElement(g, "mesh", filename=shape.size[0], scale=_fmt([shape.size[1]] * 3))
    else:
        raise ValueError(shape.kind)


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def write_urdf(robot: Robot, out_dir: str, bat: BatSpec | None = None) -> str:
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    if bat is not None:
        write_obj(os.path.join(out_dir, "meshes", "bat.obj"), *bat.mesh())

    root = ET.Element("robot", name=robot.name)
    for name, rgba in COLORS.items():
        m = ET.SubElement(root, "material", name=name)
        ET.SubElement(m, "color", rgba=_fmt(rgba))

    for link in robot.links.values():
        el = ET.SubElement(root, "link", name=link.name)
        inert = ET.SubElement(el, "inertial")
        ET.SubElement(inert, "origin", xyz=_fmt(link.com), rpy="0 0 0")
        ET.SubElement(inert, "mass", value=f"{link.mass:.6g}")
        I = link.inertia
        ET.SubElement(inert, "inertia", ixx=f"{I[0,0]:.6g}", ixy=f"{I[0,1]:.6g}", ixz=f"{I[0,2]:.6g}",
                      iyy=f"{I[1,1]:.6g}", iyz=f"{I[1,2]:.6g}", izz=f"{I[2,2]:.6g}")
        for s in link.shapes:
            if s.visual:
                v = ET.SubElement(el, "visual")
                _origin(v, s.pos, s.rot)
                _geometry(v, s)
                ET.SubElement(v, "material", name=s.color)
            if s.collision and s.kind != "mesh":
                c = ET.SubElement(el, "collision")
                _origin(c, s.pos, s.rot)
                _geometry(c, s)

    for j in robot.joints:
        el = ET.SubElement(root, "joint", name=j.name, type=j.type)
        _origin(el, j.pos, j.rot)
        ET.SubElement(el, "parent", link=j.parent)
        ET.SubElement(el, "child", link=j.child)
        if j.type == "revolute":
            ET.SubElement(el, "axis", xyz=_fmt(j.axis / np.linalg.norm(j.axis)))
            ET.SubElement(el, "limit", lower=f"{j.lower:.6g}", upper=f"{j.upper:.6g}",
                          effort=f"{j.effort:.6g}", velocity=f"{j.velocity:.6g}")

    ET.indent(root)
    path = os.path.join(out_dir, f"{robot.name}.urdf")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path
