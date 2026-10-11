"""Zero the critic's output layer of an rsl_rl checkpoint (and its Adam moments), keep the actor.

After a reward redesign the value scale changes completely (reward v2: outcomes of +-100..200 per episode instead of
hundreds of small per-step imitation points). A zero value head starts from "no expectation" instead of confidently
wrong values that would give the first PPO updates garbage advantages.

    python scripts/reset_critic_head.py in.pt out.pt
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("critic.") and k.endswith(".weight")})
last = f"critic.{idx[-1]}"
names = list(sd.keys())
for suffix in ("weight", "bias"):
    key = f"{last}.{suffix}"
    sd[key] = torch.zeros_like(sd[key])
    opt = ck.get("optimizer_state_dict")
    if opt is not None:  # parameters are in model_state_dict order
        st = opt["state"].get(names.index(key))
        if st is not None:
            st["exp_avg"].zero_()
            st["exp_avg_sq"].zero_()
torch.save(ck, dst)
print(f"zeroed {last} ({tuple(sd[last + '.weight'].shape)}) -> {dst}")
