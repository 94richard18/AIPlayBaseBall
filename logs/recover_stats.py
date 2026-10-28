"""Fall diag 3 (planning 14, 10-29): after_done recover run vs logs/m2 s6. Swing rows must not move (pitch + hit / exit /
launch / spray / carry / backspin; batter_fell is what the recovery changes). 'On the ground' = pelvis < 0.35 m above
field_z within 2 s of contact (hits) / of the plate arrival (misses); 'lost posture' = the env's fallen flag.
Last line: PASS / FAIL (bar: on the ground within 2 s <= 30% on hits and on misses)."""
import csv, sys
A = list(csv.DictReader(open("logs/m2/m2_s6.csv"))); B = list(csv.DictReader(open(sys.argv[1])))
cols = ["clip", "kmh", "handoff_t", "handed", "pred_y", "pred_z", "flight_pred", "handoff_jump_m", "pitcher_fell",
        "hit", "exit_mps", "launch_deg", "spray_deg", "carry_m", "backspin_rpm"]
same = sum(all(a[c] == b[c] for c in cols) for a, b in zip(A, B))
cr = [(a, b) for a, b in zip(A, B) if a["hit"] != "True" and b["hit"] != "True"]
crs = sum(all(a[c] == b[c] for c in ("plate_y", "plate_z", "t_plane")) for a, b in cr)
print(f"REC rows identical (pitch + swing) {same}/{len(A)}; crossing {crs}/{len(cr)}")
f = lambda x: float(x) if x not in ("", "nan") else float("nan")
ok = same == len(A)
for lab, rows in (("hits", [r for r in B if r["hit"] == "True"]), ("misses", [r for r in B if r["hit"] != "True"])):
    n = len(rows); m2 = sorted(f(r["min_pelvis_z_2s"]) for r in rows)
    g2 = sum(x < 0.35 for x in m2); ge = sum(f(r["min_pelvis_z_end"]) < 0.35 for r in rows)
    lp = sum(f(r["bat_fall_t"]) - f(r["t_ref"]) <= 2.0 for r in rows)
    q = "/".join(f"{m2[int(k * (n - 1))]:.2f}" for k in (0, .25, .5, .75, 1))
    print(f"REC {lab} n {n}: on the ground within 2 s {g2} ({g2 / n:.0%}), by episode end {ge} ({ge / n:.0%}); "
          f"lost posture within 2 s {lp} ({lp / n:.0%}); min pelvis z in 2 s min/q1/med/q3/max {q}")
    ok &= g2 / n <= 0.30
print("REC PASS" if ok else "REC FAIL")
