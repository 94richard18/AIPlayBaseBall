"""Human26 design check (CPU): the 35 contact_offset hits replayed at the sweet spot with normal offset D = 0 / +1 / +1.5 /
+2 cm along e (the same construction as contact_offset.py), plus the sweet-spot velocity elevation at contact.
Question: does D alone set the launch (fold the swing plane into r_D), or does the bat path's elevation matter?"""
import os, sys
import numpy as np, torch
sys.path.insert(0, ".")
from aibaseball.physics import BatGeometry, BatSpec, PhysicsSpec, resolve_impact
from aibaseball.physics.impact import ContactQuery
spec, geom = PhysicsSpec(), BatGeometry(BatSpec(), "cpu", dtype=torch.float64); rb = spec.ball.radius
T = lambda a: torch.as_tensor(a, dtype=torch.float64)
unit = lambda v: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
D_all = {}
for s in ("s6", "s1"):
    z = np.load(f"logs/contact_offset/{s}.npz"); env = z["env"].astype(int)
    f = np.array([np.flatnonzero(env == e)[0] for e in np.unique(env)])
    for k in z.files:
        D_all.setdefault(k, []).append(z[k][f])
d = {k: T(np.concatenate(v)) for k, v in D_all.items()}
cm, ax, vcm, w, bv, bw = d["bat_cm"], d["bat_axis"], d["bat_vel_cm"], d["bat_omega"], d["ball_vel_in"], d["ball_omega_in"]
knob = cm - geom.cm_x * ax
xs = torch.full((len(cm),), geom.sweet_spot_x, dtype=torch.float64)
a = knob + xs.unsqueeze(-1) * ax
vb = vcm + torch.cross(w, a - cm, dim=-1)
vr = bv - vb
u = -unit(vr - (vr * ax).sum(-1, keepdim=True) * ax)
e = unit(torch.cross(ax, u, dim=-1)); e = torch.where(e[:, 2:3] < 0, -e, e)
R = geom.radius_at(xs) + rb
elev = torch.rad2deg(torch.asin(vb[:, 2] / vb.norm(dim=-1)))  # sweet-spot velocity elevation (+ = upward)
la = lambda v: torch.rad2deg(torch.asin(v[:, 2] / v.norm(dim=-1)))
for D in (0.0, 0.01, 0.015, 0.02):
    DD = torch.full_like(xs, D).clamp(max=0.999 * float(R.min()))
    n = unit(u * (R ** 2 - DD ** 2).sqrt().unsqueeze(-1) + e * DD.unsqueeze(-1))
    bc = a + R.unsqueeze(-1) * n
    r = resolve_impact(geom, spec, ContactQuery(torch.ones(len(xs), dtype=torch.bool), torch.zeros_like(xs), xs, n, bc - rb * n),
                       bc, bv, bw, cm, ax, vcm, w)
    v, om = r.ball_vel, r.ball_omega
    vh = unit(torch.cat([v[:, :2], torch.zeros_like(v[:, :1])], -1))
    back = (om * torch.cross(vh, torch.tensor([0., 0., 1.], dtype=torch.float64).expand_as(vh), dim=-1)).sum(-1) * 60 / 6.2832
    L = la(v).numpy()
    print(f"D {D*100:+.1f} cm: exit med {np.median(v.norm(dim=-1).numpy()):.1f}  launch med {np.median(L):.1f} [{np.quantile(L,.25):.1f}, {np.quantile(L,.75):.1f}]  "
          f"backspin med {np.median(back.numpy()):.0f} rpm  corr(launch, elev) {np.corrcoef(L, elev.numpy())[0,1]:+.2f}")
E = elev.numpy()
print(f"sweet-spot velocity elevation at contact: med {np.median(E):.1f} [{np.quantile(E,.25):.1f}, {np.quantile(E,.75):.1f}] deg; "
      f"pitch descent med {np.median(torch.rad2deg(torch.asin(-bv[:, 2] / bv.norm(dim=-1))).numpy()):.1f} deg")
