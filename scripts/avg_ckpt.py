"""Average the policy weights (and observation normalizers) of several rsl_rl checkpoints - a cheap test of whether a
moving average of the weights is steadier than any single checkpoint (ft 3825 -> 3850: identical eval 44 -> 2 while
training success held at 0.28).
    python scripts/avg_ckpt.py <out.pt> <a.pt> <b.pt> ...
"""
import sys

import torch

out, srcs = sys.argv[1], sys.argv[2:]
cs = [torch.load(p, map_location="cpu", weights_only=False) for p in srcs]
avg = cs[-1]
for key in ("model_state_dict", "obs_norm_state_dict", "critic_obs_norm_state_dict"):
    for k, v in avg[key].items():
        if torch.is_tensor(v) and v.is_floating_point() and k != "count":
            avg[key][k] = torch.stack([c[key][k] for c in cs]).mean(0)
avg["infos"] = {"averaged": srcs}
torch.save(avg, out)
print(f"[avg] {out} = mean of {len(srcs)}: {', '.join(srcs)}")
