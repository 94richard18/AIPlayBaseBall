# CPU: exit speed of the game's impact model vs contact offset (bat pivoting about the knob, horizontal, ball head-on)
import sys, torch
sys.path.insert(0, r"D:\MyProject\AIPlayBaseBall")
from aibaseball.physics.bat_model import BatSpec
from aibaseball.physics.constants import PhysicsSpec
from aibaseball.physics.impact import BatGeometry, ContactQuery, resolve_impact
spec, bat = PhysicsSpec(), BatSpec()
g = BatGeometry(bat, "cpu")
r = spec.ball.circumference / 6.283185
def ev(v_ss, v_pitch, dx=0.0, dz=0.0):
    x = torch.tensor([g.sweet_spot_x + dx])
    w = v_ss / g.sweet_spot_x                          # rad/s about the knob (vertical axis)
    axis = torch.tensor([[0.0, 1.0, 0.0]])             # bat along +y, sweet spot moves +x
    omega = torch.tensor([[0.0, 0.0, -w]])             # (0,0,-w) x (0,y,0) = (+w y, 0, 0)
    cm = torch.tensor([[0.0, g.cm_x, 0.0]]); v_cm = torch.tensor([[w * g.cm_x, 0.0, 0.0]])
    R = float(g.radius_at(x)) + r
    dz_ = max(min(dz, R * 0.999), -R * 0.999)
    n = torch.tensor([[(R**2 - dz_**2) ** 0.5, 0.0, dz_]]) / R   # bat axis point -> ball centre
    axis_pt = torch.tensor([[0.0, float(x), 0.0]])
    ball_c = axis_pt + n * R
    cq = ContactQuery(hit=torch.tensor([True]), frac=torch.zeros(1), x_on_bat=x, normal=n, point=ball_c - r * n)
    res = resolve_impact(g, spec, cq, ball_c, torch.tensor([[-v_pitch, 0.0, 0.0]]), torch.zeros(1, 3), cm, axis, v_cm, omega)
    return float(res.ball_vel.norm()), float(res.cor)
for vp, name in ((18.0, "P3c 64 km/h"), (28.0, "fast 100 km/h")):
    e0, c0 = ev(20.5, vp)
    print(f"{name}: v_ss 20.5 head-on at sweet spot -> EV {e0:.1f} m/s (COR {c0:.3f}; 0.2*vp+1.2*vbat = {0.2*vp+1.2*20.5:.1f}); v_ss 31 -> {ev(31.0, vp)[0]:.1f}")
    print("   dz cm :", "  ".join(f"{d:+d}:{ev(20.5, vp, dz=d/100)[0]:.1f}" for d in (0, 1, 2, 3, 4, 5, 6)))
    print("   dx cm :", "  ".join(f"{d:+d}:{ev(20.5, vp, dx=d/100)[0]:.1f}" for d in (-20, -10, -5, 0, 5)))
