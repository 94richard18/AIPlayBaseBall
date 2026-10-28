"""Human25-R stage 2 (planning 14, 10-29): joint_smoke seed 6 after_done recover_rl vs logs/falldiag2/fd_s6.csv (after_done
policy). argv: recover_rl csv, policy rerun csv (same seed, this code: the follow-through baseline for the stats).
(b) frozen config: pitch columns + hit_t (+ the swing outcome) bit for bit vs falldiag2, 64/64; (a) on the ground (pelvis
< 0.35 m above field_z) / lost posture (fallen flag) within 2 s of contact (hits) / plate arrival (misses); (c) batted
ball within 0.15 m of a batter link origin from hit_t + 0.05 s; (d) over the 2 s from the hand-off trigger: body PD
target change per step (rms rad), body joint speed rms / max (rad/s), recover_rl vs the policy's own follow-through."""
import csv, sys
F2 = list(csv.DictReader(open("logs/falldiag2/fd_s6.csv")))
R, P = (list(csv.DictReader(open(p))) for p in sys.argv[1:3])
f = lambda x: float(x) if x not in ("", "nan") else float("nan")
cols = ["clip", "kmh", "handoff_t", "handed", "pred_y", "pred_z", "flight_pred", "handoff_jump_m", "pitcher_fell",
        "hit", "hit_t", "exit_mps", "launch_deg", "spray_deg", "carry_m", "backspin_rpm", "t_plane", "done_t"]
for nm, X in (("recover_rl", R), ("policy rerun", P)):
    same = sum(all(a[c] == b[c] for c in cols) for a, b in zip(F2, X))
    bad = sorted({c for a, b in zip(F2, X) for c in cols if a[c] != b[c]})
    print(f"RD {nm} vs falldiag2: pitch + hit_t + swing columns identical {same}/{len(F2)}" + (f" (differ: {bad})" if bad else ""))
for nm, X in (("falldiag2", F2), ("policy rerun", P), ("recover_rl", R)):
    for lab, rows in (("hits", [r for r in X if r["hit"] == "True"]), ("misses", [r for r in X if r["hit"] != "True"])):
        n = len(rows)
        g = sum(f(r["min_pelvis_z_2s"]) < 0.35 for r in rows)
        lp = sum(f(r["bat_fall_t"]) - f(r["t_ref"]) <= 2.0 for r in rows)
        ge = sum(f(r["min_pelvis_z_end"]) < 0.35 for r in rows)
        print(f"RD {nm} {lab} n {n}: on the ground within 2 s {g} ({g / max(n, 1):.0%}), lost posture within 2 s {lp} "
              f"({lp / max(n, 1):.0%}), on the ground by episode end {ge} ({ge / max(n, 1):.0%})")
for nm, X in (("policy rerun", P), ("recover_rl", R)):
    h = [r for r in X if r["hit"] == "True"]
    near = sum(f(r["ball_body_min_m"]) < 0.15 for r in h)
    mn = min((f(r["ball_body_min_m"]) for r in h), default=float("nan"))
    st = {k: sorted(f(r[k]) for r in X if f(r["pt_t"]) == f(r["pt_t"])) for k in ("pt_tgt_rate", "pt_jv_rms", "pt_jv_max")}
    med = lambda v: v[len(v) // 2] if v else float("nan")
    q9 = lambda v: v[int(0.9 * (len(v) - 1))] if v else float("nan")
    print(f"RD {nm}: hits with the ball within 0.15 m of a body link after hit_t + 0.05 s {near}/{len(h)} (closest {mn:.2f} m); "
          + "; ".join(f"{k} med {med(v):.3f} q90 {q9(v):.3f}" for k, v in st.items()))
g = sum(f(r["min_pelvis_z_2s"]) < 0.35 for r in R) / len(R)
print(f"RD RESULT recover_rl on the ground within 2 s {g:.3f} (all 64)")
