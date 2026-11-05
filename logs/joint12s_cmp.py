"""Joint s6 at 12 s vs the 6.5 s reference (14, 11-06): the hit / NaN-exit envs' batted ball (exit, launch, spray, carry,
backspin) + the landing time, and whether the other envs' per-pitch columns are unchanged.

    python logs/joint12s_cmp.py <ref csv 6.5 s> <test csv 12 s>"""
import csv
import math
import sys

ref = {int(r["env"]): r for r in csv.DictReader(open(sys.argv[1]))}
tst = {int(r["env"]): r for r in csv.DictReader(open(sys.argv[2]))}
nan = lambda v: v in ("", "nan")  # noqa: E731
miss = [e for e, r in ref.items() if r["hit"] == "True" and nan(r["exit_mps"])]
out = []
for e in miss:
    t = tst[e]
    if nan(t["exit_mps"]):
        out.append(f"e{e}: still no record (done_t {t['done_t']})")
        continue
    fair = abs(float(t["spray_deg"])) <= 45.0
    out.append(f"e{e}: EV {float(t['exit_mps']):.1f} LA {float(t['launch_deg']):+.1f} spray {float(t['spray_deg']):+.1f} "
               f"carry {float(t['carry_m']):.1f} m done_t {t['done_t']} {'fair' if fair else 'foul'}")
cols = ("hit", "exit_mps", "launch_deg", "spray_deg", "carry_m", "batter_fell_before_done", "pitcher_fell", "hit_t", "gap_m")
diff = []
for e, r in ref.items():
    if e in miss:
        continue
    d = [c for c in cols if r[c] != tst[e][c] and not (nan(r[c]) and nan(tst[e][c]))]
    if d:
        diff.append(f"e{e} {','.join(d)}")
pop = [s for s in out if "LA +" in s and float(s.split("LA ")[1].split()[0]) >= 45]
print(f"{len(miss)} hit / NaN-exit envs at 6.5 s: " + " | ".join(out)
      + f" || launch >= 45 deg: {len(pop)}/{len(miss)} || other envs vs 6.5 s ({', '.join(cols)}): "
      + ("unchanged" if not diff else f"{len(diff)} differ: " + "; ".join(diff[:12])))
