"""H31 / H33 training curve from an rsl_rl train log: every N its, ss_ca / ss_ca_lt_lo / hit_rate / track / fall_rate.
Usage: h31_curve.py train.txt [every=50] [upto_it]  (upto_it: print only the value at that iteration, for the +50 check)"""
import re
import sys

txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
every = int(sys.argv[2]) if len(sys.argv) > 2 else 50
keys = {"ss_ca": "bat/ss_ca", "lt_lo": "bat/ss_ca_lt_lo", "hit": "bat/hit_rate", "track": "mimic/track_reward",
        "fall": "bat/fall_rate", "eh": "bat/exit_hinge",
        "fc": "bat/fair_carry", "lhi": "bat/launch_hi", "llo": "bat/launch_lo", "of": "bat/outfield", "pose": "mimic/form_pose_swing", "f": "mimic/form_f", "la": "bat/launch_angle"}
rows = []
for blk in re.split(r"Learning iteration ", txt)[1:]:
    it = int(blk.split("/")[0])
    v = {k: (re.search(re.escape(n) + r":\s*([-0-9.]+)", blk) or [None, "nan"])[1] for k, n in keys.items()}
    rows.append((it, v))
if not rows:
    print("no iterations"); sys.exit(1)
if len(sys.argv) > 3:
    tgt = int(sys.argv[3])
    it, v = min(rows, key=lambda r: abs(r[0] - tgt))
    print(f"it {it} " + " ".join(f"{k} {float(x):.3f}" for k, x in v.items())); sys.exit(0)
i0 = rows[0][0]
sel = [r for r in rows if (r[0] - i0) % every == 0] + ([rows[-1]] if (rows[-1][0] - i0) % every else [])
print("curve: " + " | ".join(f"{it}: " + " ".join(f"{k} {float(x):.2f}" for k, x in v.items()) for it, v in sel))
