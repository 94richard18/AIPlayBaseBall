"""csv_cmp.py <a.csv> <b.csv>: strict row compare of two joint_smoke csvs (same envs): identical cells, per-column max |delta|."""
import csv
import sys

A, B = (list(csv.DictReader(open(p))) for p in sys.argv[1:3])
assert len(A) == len(B), (len(A), len(B))
diff, same, bad = {}, 0, set()
for a, b in zip(A, B):
    for k in a:
        if a[k] == b.get(k):
            same += 1
            continue
        bad.add(int(a["env"]))
        try:
            d = abs(float(a[k]) - float(b[k]))
            diff[k] = max(diff.get(k, 0.0), d if d == d else float("inf"))
        except ValueError:
            diff[k] = float("inf")  # text / bool flip
top = sorted(diff.items(), key=lambda x: -x[1])[:8]
print(f"{'STRICT MATCH' if not diff else 'DIFFER'}: {same} identical cells, {len(bad)}/{len(A)} envs differ; max |delta| "
      + (", ".join(f"{k} {v:.3g}" for k, v in top) if top else "none"))
