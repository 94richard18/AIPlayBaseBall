"""tempo_script sweep row (14, 11-03 14:2x): diag json -> hit, ss_contact, fair exit / launch medians, pose / key (vs the reference
played at the diag's tempo_script, same ref_t), top-3 torque_sat. Usage: tempo_row.py <tag>"""
import json
import sys

import numpy as np

d = json.load(open(f"logs/diag_bat_contact_{sys.argv[1]}.json"))
P = {k: np.array(v, float) for k, v in d["per_pitch"].items()}
h = P["hit"] > 0
f = h & (np.abs(P["spray_contact"]) <= 45)
md = lambda x, m: np.nanmedian(x[m]) if m.any() else float("nan")
ts = d.get("torque_sat", {})
top = ", ".join(f"{k} {v}" for k, v in list(ts.items())[:3])
print(f"hit {h.mean():.3f} ss_contact {md(P['ss_contact'], h):.1f} ss_max {md(P['ss_speed_max'], h):.1f} fair exit {md(P['ev_contact'], f):.1f} "
      f"fair launch {md(P['launch_contact'], f):+.1f} pose {np.nanmean(P['form_pose_sw'][h]):.3f} key {np.nanmean(P['form_key_sw'][h]):.3f} "
      f"fell {np.nanmean(P['fell']):.3f} torque_sat top3 [{top}]")
