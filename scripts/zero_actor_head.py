"""Zero the actor's output layer of an rsl_rl checkpoint (weights and bias, and their Adam moments): the policy mean is
exactly 0 - the PD reference alone (zero actions) - while the hidden layers, the critic and the observation normalizers
carry over. A fresh residual on a new reference (K70G: zero actions stride 64/64) without a blind critic.
    python scripts/zero_actor_head.py <out.pt> <in.pt> [first:last]
With a column range (action indices, last included) only those rows of the output layer are zeroed - e.g. 15:28 the
arms (shoulder / elbow / wrist) after training with freeze_arms, whose rows got no meaningful signal.
"""
import os
import sys

import torch

out, src = sys.argv[1], sys.argv[2]
rows = slice(None)
if len(sys.argv) > 3:
    a, b = (int(x) for x in sys.argv[3].split(":"))
    rows = slice(a, b + 1)
c = torch.load(src, map_location="cpu", weights_only=False)
sd = c["model_state_dict"]
names = list(sd.keys())  # all parameters, in model.parameters() order (the optimizer's param ids)
last = max(int(k.split(".")[1]) for k in names if k.startswith("actor."))
heads = [f"actor.{last}.weight", f"actor.{last}.bias"]
st = c["optimizer_state_dict"]["state"]
for h in heads:
    sd[h][rows].zero_()
    i = names.index(h)
    if i in st:
        for m in ("exp_avg", "exp_avg_sq"):
            st[i][m][rows].zero_()
c["infos"] = {"zero_actor_head": src, "rows": str(rows)}
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
torch.save(c, out)
print(f"[zero_actor_head] {out}: {', '.join(heads)} rows {rows} zeroed (from {src}), std {sd['std'].mean():.3f}")
