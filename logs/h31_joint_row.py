"""64-row joint s6 summary for the film compare: hits, fair, carry median / max (fair), outfield (fair carry >= 38.9 m),
exit / launch medians, ss_max, recover (pelvis under 0.35 m within 2 s, batter fell). Usage: h31_joint_row.py joint.csv"""
import csv
import sys

import numpy as np

f = lambda x: float(x) if x not in ("", "nan") else float("nan")  # noqa: E731
b = list(csv.DictReader(open(sys.argv[1])))
g = lambda k: np.array([f(x[k]) for x in b])  # noqa: E731
hit = np.array([x["hit"] == "True" for x in b])
fair = hit & (np.abs(g("spray_deg")) <= 45)
cy = g("carry_m")
med = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")  # noqa: E731
mx = lambda a: float(np.nanmax(a)) if np.isfinite(a).any() else float("nan")  # noqa: E731
print(f"hits {hit.sum()}/{len(b)}, fair {fair.sum()}, carry fair med {med(cy[fair]):.1f} max {mx(cy[fair]):.1f}, "
      f"outfield {(fair & (cy >= 38.9)).sum()}, exit med {med(g('exit_mps')[hit]):.1f}, launch med {med(g('launch_deg')[hit]):.1f}, "
      f"ss_max med {med(g('ss_speed_max')):.1f}, recover: ground {(g('min_pelvis_z_2s') < 0.35).sum()}, "
      f"fell {sum(x['batter_fell'] == 'True' for x in b)}")
