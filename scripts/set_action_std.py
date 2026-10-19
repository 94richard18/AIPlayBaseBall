"""Reset the policy's action noise std on some action columns of a checkpoint (exploration restart), keeping the
weights, normalizers and optimizer state.

    python scripts/set_action_std.py <in.pt> <out.pt> <c0-c1=std> [<c0-c1=std> ...]

Columns follow PitchEnv.body_names (find_joints(BODY_JOINT_EXPR, preserve_order=True): hips 0-5, knees 6-7, ankles 8-11,
waist 12-14, shoulders 15-20, elbows 21-22, wrists 23-28), then the 16 finger columns (29-44).
"""

import sys

import torch

src, dst, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
d = torch.load(src, map_location="cpu", weights_only=False)
std = d["model_state_dict"]["std"]
before = std.clone()
for s in specs:
    cols, v = s.split("=")
    c0, c1 = (int(x) for x in cols.split("-"))
    std[c0:c1 + 1] = float(v)
torch.save(d, dst)
print(f"[std] {src} -> {dst}")
for i in range(len(std)):
    if std[i] != before[i]:
        print(f"  col {i:2d}: {before[i]:.4f} -> {std[i]:.4f}")
