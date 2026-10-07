"""Sanity tests for the physics core. Run: python tests/test_physics.py (or pytest)."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aibaseball.physics import (  # noqa: E402
    MPH,
    BatGeometry,
    BatSpec,
    PhysicsSpec,
    detect_contact,
    launch_state,
    resolve_impact,
    simulate_flight,
)

T = torch.float64
SPEC = PhysicsSpec()


def swing_hit(sweet_speed: float, undercut: float, x_hit: float | None = None):
    """Bat along +Y rotating about Z (pivot 0.35 m behind the knob); ball on the bat path.

    undercut > 0: ball center above the bat axis (bat strikes the lower half of the ball).
    """
    bat = BatSpec()
    geom = BatGeometry(bat, "cpu", T)
    x_hit = bat.sweet_spot_x if x_hit is None else x_hit
    pivot_off = 0.35
    w = sweet_speed / (bat.sweet_spot_x + pivot_off)
    knob = torch.tensor([[0.0, 0.0, 0.0]], dtype=T)
    axis = torch.tensor([[0.0, 1.0, 0.0]], dtype=T)
    omega_bat = torch.tensor([[0.0, 0.0, -w]], dtype=T)
    cm = knob + geom.cm_x * axis
    v_cm = torch.tensor([[w * (geom.cm_x + pivot_off), 0.0, 0.0]], dtype=T)

    rr = geom.radius_at(torch.tensor([x_hit], dtype=T)).item() + SPEC.ball.radius - 1e-3
    dz = undercut
    dx = math.sqrt(max(rr**2 - dz**2, 0.0))
    ball_c = torch.tensor([[dx, x_hit, dz]], dtype=T)
    zero = torch.zeros(1, 3, dtype=T)
    c = detect_contact(geom, ball_c, SPEC.ball.radius, knob, axis, knob, axis)
    assert bool(c.hit[0]), "contact not detected"
    return resolve_impact(geom, SPEC, c, ball_c, zero, zero, cm, axis, v_cm, omega_bat)


def test_bat_properties():
    p = BatSpec().mass_properties()
    print(f"bat: density {p['density']:.0f} kg/m3, CM {p['cm_x']*100:.1f} cm from knob, "
          f"I_cm {p['inertia_transverse']:.4f}, I_6in {p['inertia_6in']:.4f} kg m^2")
    assert 550 < p["density"] < 850  # maple ~ 700
    assert 0.55 < p["cm_x"] < 0.62  # balance point ~ 22-24 in
    assert 0.17 < p["inertia_6in"] < 0.26  # typical 34" wood


def test_flight_reference():
    v, w = launch_state(torch.tensor([105 * MPH], dtype=T), torch.tensor([28.0], dtype=T), torch.tensor([2200.0], dtype=T))
    r = simulate_flight(torch.tensor([[0, 0, 0.9]], dtype=T), v, w, SPEC, dt=0.002)
    carry_ft = r.carry.item() / 0.3048
    print(f"105 mph / 28 deg / 2200 rpm -> {carry_ft:.0f} ft")
    assert 400 < carry_ft < 445


def test_square_hit_collision_efficiency():
    res = swing_hit(44.0, 0.0)
    ev = res.ball_vel.norm().item()
    q = ev / 44.0 - 1.0
    print(f"square sweet-spot hit at 44 m/s: EV {ev:.1f} m/s ({ev/MPH:.1f} mph), q = {q:.3f}, COR {res.cor.item():.3f}")
    assert 0.15 < q < 0.26  # literature: q ~ 0.2 for wood bats


def test_undercut_backspin():
    res = swing_hit(44.0, 0.025)
    v, w = res.ball_vel[0], res.ball_omega[0]
    la = math.degrees(math.atan2(v[2], v[:2].norm()))
    rpm = -w[1].item() * 60 / (2 * math.pi)  # backspin for +X travel is -Y
    print(f"undercut 2.5 cm: EV {v.norm():.1f} m/s, LA {la:.1f} deg, backspin {rpm:.0f} rpm")
    assert la > 15 and rpm > 1000


def test_off_sweet_spot_loses_speed():
    ev_ss = swing_hit(44.0, 0.0).ball_vel.norm().item()
    ev_in = swing_hit(44.0, 0.0, x_hit=0.50).ball_vel.norm().item()
    print(f"sweet spot EV {ev_ss:.1f} vs 'jammed' (x=0.50 m) EV {ev_in:.1f}")
    assert ev_in < ev_ss


def test_swept_detection_no_tunnel():
    """Bat moves 10 cm in one step through a ball: must be detected."""
    bat = BatSpec()
    geom = BatGeometry(bat, "cpu", T)
    axis = torch.tensor([[0.0, 1.0, 0.0]], dtype=T)
    ball = torch.tensor([[0.0, bat.sweet_spot_x, 0.0]], dtype=T)
    c = detect_contact(geom, ball, SPEC.ball.radius, torch.tensor([[-0.08, 0, 0]], dtype=T), axis,
                       torch.tensor([[0.02, 0, 0]], dtype=T), axis)
    assert bool(c.hit[0]) and c.frac[0] < 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  [ok] {name}")
