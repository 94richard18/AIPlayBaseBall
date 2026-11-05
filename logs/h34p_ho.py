"""h34p_ho.py <ref tag> <test tag> <lead tag 000|010|025>: H34P honest pose (14, 11-05) = joint pose error from the pin hand-over
(release - lead) to contact only, mean / q90 over hit episodes (as h32_form), ref vs test. One line."""
import json
import sys

import numpy as np


def pose(tag, w):
    P = json.load(open(f"logs/diag_bat_contact_{tag}.json"))["per_pitch"]
    h, x = np.array(P["hit"], float) > 0, np.array(P[f"form_pose_ho{w}"], float)
    return f"{np.nanmean(x[h]):.3f}/{np.nanpercentile(x[h], 90):.3f}"


print(f"pose hand-over (release - {int(sys.argv[3]) / 100:.2f} s) -> contact: {sys.argv[1]} {pose(sys.argv[1], sys.argv[3])} | {sys.argv[2]} {pose(sys.argv[2], sys.argv[3])}")
