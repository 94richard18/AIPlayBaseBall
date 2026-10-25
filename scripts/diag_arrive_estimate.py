"""Batter perception check (10-25): the constant-velocity arrival estimate at release vs the true drag + Magnus flight, P3c pitches vs 88-112 km/h.

    python scripts/diag_arrive_estimate.py
"""
import os, sys, torch
ROOT = r"D:\MyProject\AIPlayBaseBall"; sys.path.insert(0, ROOT)
from aibaseball.physics import PhysicsSpec
from aibaseball.physics.pitch_gen import build_library, build_library_from_release
spec = PhysicsSpec()
def report(name, lib):
    # constant-velocity estimate at release: distance along x / |vx0|, vs the true (drag + Magnus) flight time
    d = (lib.release[:, 0] - lib.target[:, 0]).abs()
    est = d / lib.v0[:, 0].abs()
    T = lib.flight_time
    err = T - est  # + = the ball arrives later than the batter's estimate -> the batter swings early
    q = torch.tensor([0.1, 0.5, 0.9])
    print(f"{name}: speed {lib.speed_kmh.median():.0f} km/h, flight {T.median():.3f} s, late vs const-v estimate p10/50/90 {torch.quantile(err, q).numpy().round(3)} s")
report("P3c t18", build_library_from_release(os.path.join(ROOT, "assets/game/p3c_identical_t18_release.csv"), spec))
report("88-112 km/h", build_library(512, spec, speed_kmh=(88.0, 112.0)))
