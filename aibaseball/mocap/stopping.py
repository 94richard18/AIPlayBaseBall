"""Can a body state be brought to rest at all? Centroidal LP (a necessary condition: no joint limits, no kinematics).

State: COM xy, COM velocity xy, centroidal angular momentum Lx, Ly; COM height fixed (Fz = m g).
Controls per step: horizontal ground force (friction pyramid mu) and the centre of pressure inside the support polygon.
  m dv/dt = F,   dL/dt = (cop - com) x (F, m g)
Rest: v = 0, L = 0, COM over the support at the end. `theta_max` bounds the whole-body pitch / roll rotation
theta = integral L / I (how far the body may tip while it absorbs its spin).
Athlete (pitch_2916-4) at the release: stoppable within 30 deg; the robot at 17950 (lead foot ~0.38 m toward the
throwing-arm side): 2/16.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

G = 9.81
SOLE = [(x, y) for x in (-0.09, 0.17) for y in (-0.055, 0.055)]  # foot box corners (ankle frame), bottom 8 cm under it
def hull(pts):
    pts = sorted(map(tuple, pts))
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def halfplanes(poly, shrink=0.0):
    """A p <= b for a CCW convex polygon."""
    A, b = [], []
    for i in range(len(poly)):
        a, c = poly[i], poly[(i + 1) % len(poly)]
        e = c - a
        n = np.array([e[1], -e[0]]) / np.linalg.norm(e)  # outward for CCW
        A.append(n)
        b.append(n @ a - shrink)
    return np.array(A), np.array(b)


def sole_xy(R, p):
    return np.array([(p + R @ np.array([x, y, -0.08]))[:2] for x, y in SOLE])


def solve(c0, v0, L0, h, m, I, supports, T=1.5, dt=0.01, mu=0.8, theta_max=None):
    """supports: list of (t_from, polygon). Returns (feasible, min terminal violation)."""
    N = int(T / dt)
    nv = 4 * N + 1  # fx fy px py per step + terminal slack s
    rows_ub, rhs_ub = [], []
    v_const = v0[:2].copy()
    c = [np.zeros(nv), np.zeros(nv)]
    c_const = c0[:2].copy()
    L = [np.zeros(nv), np.zeros(nv)]
    L_const = L0[:2].copy()
    th = [np.zeros(nv), np.zeros(nv)]
    th_const = np.zeros(2)
    vrow = [np.zeros(nv), np.zeros(nv)]
    for k in range(N):
        fx, fy, px, py = 4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3
        # torque about the COM (using c_k)
        tx = np.zeros(nv); tx[py] += m * G; tx -= m * G * c[1]; tx[fy] += h  # (py - cy) m g + h fy
        tx_c = -m * G * c_const[1]
        ty = np.zeros(nv); ty[px] -= m * G; ty += m * G * c[0]; ty[fx] -= h  # -(px - cx) m g - h fx
        ty_c = m * G * c_const[0]
        L = [L[0] + dt * tx, L[1] + dt * ty]
        L_const = L_const + dt * np.array([tx_c, ty_c])
        th = [th[0] + dt * L[0] / I[0], th[1] + dt * L[1] / I[1]]
        th_const = th_const + dt * L_const / I
        for j, f in enumerate((fx, fy)):
            vrow[j] = vrow[j].copy(); vrow[j][f] += dt / m
        v_const_k = v_const
        c = [c[j] + dt * vrow[j] for j in range(2)]
        c_const = c_const + dt * v_const_k
        # support at this step
        t = k * dt
        poly = [P for t0, P in supports if t >= t0][-1]
        A, b = halfplanes(poly)
        for a_, b_ in zip(A, b):
            r = np.zeros(nv); r[px], r[py] = a_; rows_ub.append(r); rhs_ub.append(b_)
        lim = mu * m * G / np.sqrt(2)
        for f in (fx, fy):
            for s in (1, -1):
                r = np.zeros(nv); r[f] = s; rows_ub.append(r); rhs_ub.append(lim)
        if theta_max is not None:
            for j in range(2):
                for s in (1, -1):
                    rows_ub.append(s * th[j]); rhs_ub.append(theta_max - s * th_const[j])
    # terminal: |v|, |L| <= s ; COM inside final support (shrunk 3 cm) within s
    S = nv - 1
    for row, const, scale in ((vrow[0], v_const[0], 1.0), (vrow[1], v_const[1], 1.0),
                              (L[0], L_const[0], 0.05), (L[1], L_const[1], 0.05)):
        for s in (1, -1):
            r = s * row * scale; r = r.copy(); r[S] -= 1; rows_ub.append(r); rhs_ub.append(-s * const * scale)
    A, b = halfplanes(supports[-1][1], 0.03)
    for a_, b_ in zip(A, b):
        r = a_[0] * c[0] + a_[1] * c[1]; r = r.copy(); r[S] -= 1; rows_ub.append(r); rhs_ub.append(b_ - a_ @ c_const)
    cost = np.zeros(nv); cost[S] = 1.0
    bounds = [(None, None)] * (nv - 1) + [(0, None)]
    res = linprog(cost, A_ub=np.array(rows_ub), b_ub=np.array(rhs_ub), bounds=bounds, method="highs")
    if res.status != 0:
        return False, np.inf
    return res.x[S] < 1e-4, res.x[S]


def state_from_links(m, pc, vc, Rl, wl, Iloc):
    M = m.sum()
    C = (m[:, None] * pc).sum(0) / M
    V = (m[:, None] * vc).sum(0) / M
    L = np.zeros(3)
    Icom = np.zeros((3, 3))
    for i in range(len(m)):
        Iw = Rl[i] @ Iloc[i] @ Rl[i].T
        r = pc[i] - C
        L += m[i] * np.cross(r, vc[i] - V) + Iw @ wl[i]
        Icom += Iw + m[i] * (r @ r * np.eye(3) - np.outer(r, r))
    return C, V, L, Icom
