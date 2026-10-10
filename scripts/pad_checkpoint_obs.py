"""Warm-start a checkpoint for a longer observation: new inputs appended at the end get zero first-layer weights
(actor and critic, so the networks start out exactly as before), zero Adam moments, and a neutral normalizer
(mean 0, std 1).

    python scripts/pad_checkpoint_obs.py <in.pt> <out.pt> <new_obs_dim>
"""
import sys

import torch

src, dst, new = sys.argv[1], sys.argv[2], int(sys.argv[3])
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
old = sd["actor.0.weight"].shape[1]
assert new > old, (old, new)
pad = new - old


def pad_cols(w):
    return torch.cat([w, torch.zeros(*w.shape[:-1], pad, dtype=w.dtype)], -1)


for k in ("actor.0.weight", "critic.0.weight"):
    sd[k] = pad_cols(sd[k])
for st in ck["optimizer_state_dict"]["state"].values():
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st and st[k].dim() == 2 and st[k].shape[1] == old:
            st[k] = pad_cols(st[k])
for key in ("obs_norm_state_dict", "critic_obs_norm_state_dict"):
    if ck.get(key):
        n = ck[key]
        n["_mean"] = pad_cols(n["_mean"])
        n["_var"] = torch.cat([n["_var"], torch.ones(1, pad)], -1)
        n["_std"] = torch.cat([n["_std"], torch.ones(1, pad)], -1)
torch.save(ck, dst)
print(f"{src}: obs {old} -> {new} -> {dst}")
