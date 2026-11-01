"""joint csv -> joint row + film env pick (14, 11:58 rule: 2 envs, no fall, fair, highest carry, one fly + one line drive).
Usage: film_pick.py joint.csv out_envs_file"""
import csv, statistics as st, sys
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
b = list(csv.DictReader(open(sys.argv[1])))
T = lambda x: x == 'True'
hits = [x for x in b if T(x['hit'])]
fair = [x for x in hits if abs(f(x['spray_deg'])) <= 45]
car = [f(x['carry_m']) for x in fair]
fell = sum(T(x['batter_fell']) for x in b); bd = sum(T(x['batter_fell_before_done']) for x in b)
ground = sum(f(x['min_pelvis_z_2s']) < 0.35 for x in b); tilt = sum(f(x['torso_tilt_1p5_deg']) <= 30 for x in b)
q75 = lambda v: (lambda w: st.quantiles(w, n=4)[2] if len(w) > 1 else float('nan'))([x for x in v if x == x])
med = lambda v: (lambda w: st.median(w) if w else float('nan'))([x for x in v if x == x])  # NaN = ball still in flight at the end
print(f"joint s6 n {len(b)}: hits {len(hits)}, fair {len(fair)}, carry fair median {med(car):.1f} max {max([c for c in car if c == c], default=float('nan')):.1f}, "
      f"outfield (fair, carry >= 38.9) {sum(c >= 38.9 for c in car)}, exit {med([f(x['exit_mps']) for x in hits]):.1f}, "
      f"launch {med([f(x['launch_deg']) for x in hits]):.1f} (q75 {q75([f(x['launch_deg']) for x in hits]):.1f}, >= 45 deg {sum(f(x['launch_deg']) >= 45 for x in hits)}, carry nan {sum(f(x['carry_m']) != f(x['carry_m']) for x in hits)}), fell {fell} (before done {bd}, after {fell - bd}), ground {ground}, tilt ok {tilt}")
c = sorted([x for x in fair if not T(x['batter_fell'])], key=lambda x: -f(x['carry_m']))
d = lambda x: f"env {x['env']} carry {f(x['carry_m']):.1f} launch {f(x['launch_deg']):.1f} exit {f(x['exit_mps']):.1f} spray {f(x['spray_deg']):.1f}"
print(f"no-fall fair hits {len(c)}: " + " | ".join(d(x) for x in c[:6]))
if len(c) < 2:
    pick = None
else:
    fly = [x for x in c if f(x['launch_deg']) >= 25]; line = [x for x in c if 10 <= f(x['launch_deg']) < 25]
    pick = [fly[0], line[0]] if fly and line else c[:2]
open(sys.argv[2], 'w').write(("FALLBACK" if pick is None else " ".join(x['env'] for x in pick)) + "\n")
print("FILM_ENVS -> " + ("FALLBACK" if pick is None else " ; ".join(d(x) for x in pick)))
