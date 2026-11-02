import csv, sys
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
T = lambda x: x == 'True'
tot_g = tot_n = 0; tilts = []; out = []
for s in sys.argv[1:]:
    b = list(csv.DictReader(open(f"logs/h32/joint_s{s}_15993.csv")))
    hits = [x for x in b if T(x['hit'])]; fair = [x for x in hits if abs(f(x['spray_deg'])) <= 45]
    of = sum(f(x['carry_m']) >= 38.9 for x in fair)
    fell = sum(T(x['batter_fell']) for x in b); bd = sum(T(x['batter_fell_before_done']) for x in b)
    g = sum(f(x['min_pelvis_z_2s']) < 0.35 and not T(x['batter_fell_before_done']) for x in b)
    t = sum(f(x['torso_tilt_1p5_deg']) <= 30 for x in b)
    tot_g += g; tot_n += len(b); tilts.append(t)
    out.append(f"s{s} n {len(b)}: hits {len(hits)} fair {len(fair)} outfield {of} fell before {bd} after {fell - bd} after-done ground {g} tilt ok {t}")
out.append(f"TOTAL {len(tilts)} seeds: after-done ground {tot_g}/{tot_n} ({100 * tot_g / tot_n:.1f}%), tilt ok mean {sum(tilts) / len(tilts):.1f}")
print(" | ".join(out))
