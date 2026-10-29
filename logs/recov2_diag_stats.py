"""Human25-R2 stage 2 light (planning 14, 10-29): joint_smoke seed 6 after_done recover_rl with the R2 checkpoint vs
logs/falldiag2/fd_s6.csv (after_done policy = the swing alone) and logs/recov_diag/recover_rl.csv (model_998).
argv: the R2 recover_rl csv. Frozen pitch + hit_t + swing columns 64/64; on the ground within 2 s of contact / arrival;
torso tilt at t_ref + 1.5 s (R2 only: the column is new); rows whose ss_speed_max is above the swing-alone value."""
import csv, sys
F2 = list(csv.DictReader(open("logs/falldiag2/fd_s6.csv")))
R1 = list(csv.DictReader(open("logs/recov_diag/recover_rl.csv")))
R2 = list(csv.DictReader(open(sys.argv[1])))
f = lambda x: float(x) if x not in ("", "nan") else float("nan")
cols = ["clip", "kmh", "handoff_t", "handed", "pred_y", "pred_z", "flight_pred", "handoff_jump_m", "pitcher_fell",
        "hit", "hit_t", "exit_mps", "launch_deg", "spray_deg", "carry_m", "backspin_rpm", "t_plane", "done_t"]
same = sum(all(a[c] == b[c] for c in cols) for a, b in zip(F2, R2))
bad = sorted({c for a, b in zip(F2, R2) for c in cols if a[c] != b[c]})
print(f"RD2 R2 vs falldiag2: pitch + hit_t + swing columns identical {same}/{len(F2)}" + (f" (differ: {bad})" if bad else ""))
for nm, X in (("model_998", R1), ("R2", R2)):
    for lab, sel in (("hits", lambda r: r["hit"] == "True"), ("misses", lambda r: r["hit"] != "True")):
        rows = [r for r in X if sel(r)]
        g = sum(f(r["min_pelvis_z_2s"]) < 0.35 for r in rows)
        hi = [f(b["ss_speed_max"]) - f(a["ss_speed_max"]) for a, b in zip(F2, X) if sel(b)]
        n_hi = sum(d > 0.01 for d in hi)
        print(f"RD2 {nm} {lab} n {len(rows)}: on the ground within 2 s {g}; ss_speed_max above the swing alone (> 0.01 m/s) "
              f"{n_hi} (max {max(hi, default=float('nan')):+.2f} m/s)")
tt = sorted(f(r["torso_tilt_1p5_deg"]) for r in R2 if f(r["torso_tilt_1p5_deg"]) == f(r["torso_tilt_1p5_deg"]))
q = lambda p: tt[int(p * (len(tt) - 1))] if tt else float("nan")
for lab, sel in (("hits", "True"), ("misses", "False")):
    v = sorted(f(r["torso_tilt_1p5_deg"]) for r in R2 if (r["hit"] == "True") == (sel == "True"))
    print(f"RD2 R2 {lab}: torso tilt at t_ref + 1.5 s med {v[len(v) // 2] if v else float('nan'):.1f} max {max(v, default=float('nan')):.1f} deg")
g = sum(f(r["min_pelvis_z_2s"]) < 0.35 for r in R2)
hi = sum(f(b["ss_speed_max"]) - f(a["ss_speed_max"]) > 0.01 for a, b in zip(F2, R2))
print(f"RD2 RESULT R2 frozen {same}/64, on the ground {g}/64, torso tilt at +1.5 s med {q(0.5):.1f} q90 {q(0.9):.1f} deg "
      f"(n {len(tt)}), ss_speed_max above the swing alone {hi}/64")
