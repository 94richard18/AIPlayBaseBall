"""Ball-on-grass model checks. Run: python tests/game/test_grass.py (or pytest)."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aibaseball.physics import BallSpec  # noqa: E402
from aibaseball.physics.grass import GrassSpec, bounce, ground_path, restitution  # noqa: E402

BALL, GRASS, G = BallSpec(), GrassSpec(), 9.80665


def energy(v, w):
    return 0.5 * BALL.mass * sum(c * c for c in v) + 0.5 * BALL.inertia * sum(c * c for c in w)


def test_drop_rebounds_to_e_squared():
    vz = -math.sqrt(2.0 * G * 1.0)  # dropped from 1 m
    v, _ = bounce((0.0, 0.0, vz), (0.0, 0.0, 0.0), GRASS, BALL)
    assert abs(v[2] ** 2 / (2.0 * G) - restitution(vz, GRASS) ** 2) < 1e-9


def test_bounce_never_adds_energy():
    for v, w in [((30.0, 0.0, -8.0), (0.0, 0.0, 0.0)), ((30.0, 5.0, -8.0), (0.0, 200.0, 0.0)),
                 ((20.0, 0.0, -3.0), (0.0, -250.0, 0.0)), ((5.0, -2.0, -12.0), (100.0, 50.0, 30.0))]:
        v2, w2 = bounce(v, w, GRASS, BALL)
        assert energy(v2, w2) <= energy(v, w) + 1e-9


def test_topspin_skips_forward_backspin_checks():
    r = BALL.radius
    v = (25.0, 0.0, -6.0)
    top, _ = bounce(v, (0.0, 2.0 * 25.0 / r, 0.0), GRASS, BALL)  # omega_y > 0 = topspin for +x travel
    back, _ = bounce(v, (0.0, -25.0 / r, 0.0), GRASS, BALL)
    flat, _ = bounce(v, (0.0, 0.0, 0.0), GRASS, BALL)
    assert top[0] > v[0] > flat[0] >= back[0]  # flat and backspin both slide through the bounce: same friction cap


def test_rolling_distance():
    v0 = 10.0
    path = ground_path(0.0, 0.0, 0.0, (v0, 0.0, -0.5), (0.0, v0 / BALL.radius, 0.0), GRASS, BALL, G, 0.01)
    expect = v0 * v0 / (2.0 * GRASS.mu_roll * G)
    assert abs(path[-1][1] - expect) / expect < 0.05, (path[-1][1], expect)


def test_stop_callback():
    path = ground_path(0.0, 0.0, 0.0, (20.0, 0.0, -5.0), (0.0, 0.0, 0.0), GRASS, BALL, G, 0.01, stop=lambda x, y: x > 15.0)
    assert 15.0 < path[-1][1] < 15.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
