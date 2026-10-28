"""M2 at bias 0.030 (logs/m2_b030) vs M2 (logs/m2), per seed: pitch columns must be bit-identical (pitches do not see the
batter); the crossing columns (plate_y / plate_z / t_plane) only where neither run hit the ball. Then the swing rows that changed."""
import csv, glob, os, sys
A_DIR, B_DIR = "logs/m2", sys.argv[1] if len(sys.argv) > 1 else "logs/m2_b030"
PITCH = ["clip", "kmh", "handoff_t", "handed", "pred_y", "pred_z", "flight_pred", "handoff_jump_m", "pitcher_fell", "backspin_rpm"]
CROSS = ["plate_y", "plate_z", "t_plane"]
SWING = ["hit", "exit_mps", "launch_deg", "spray_deg", "carry_m", "batter_fell"]
tot = eq = cr = crn = rows = 0; changed = []
for fb in sorted(glob.glob(f"{B_DIR}/m2_s*.csv"), key=lambda f: int(f.split("_s")[-1][:-4])):
    s = int(fb.split("_s")[-1][:-4]); A = list(csv.DictReader(open(f"{A_DIR}/m2_s{s}.csv"))); B = list(csv.DictReader(open(fb)))
    for a, b in zip(A, B):
        rows += 1; tot += 1; eq += all(a[c] == b[c] for c in PITCH)
        if a["hit"] != "True" and b["hit"] != "True":
            crn += 1; cr += all(a[c] == b[c] for c in CROSS)
        if any(a[c] != b[c] for c in SWING): changed.append((s, int(a["env"])))
print(f"M2BCMP pitch columns identical {eq}/{tot} rows; crossing identical {cr}/{crn} rows (neither hit); swing outcome changed {len(changed)}/{rows}")
print("changed (seed, env):", " ".join(f"{s}:{e}" for s, e in changed))
