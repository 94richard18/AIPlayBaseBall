"""Grow an rsl_rl checkpoint to more observations / actions (new entries appended at the end).

New input columns get zero weights (the old policy ignores them until it learns to use them), new
action rows output zero mean with `--new_std` exploration, the observation normaliser starts at
mean 0 / std 1 for the new entries, and the Adam moments are zero-padded so training can resume.

    python scripts/expand_checkpoint.py IN.pt OUT.pt --obs 198 [--act 46 --new_std 0.3] [--obs_insert 110:1]
--obs_insert POS:COUNT inserts COUNT new observation columns before column POS (of the old layout);
any remaining growth is appended at the end.
"""

import argparse

import torch


def pad_to(t: torch.Tensor, shape, fill: float = 0.0) -> torch.Tensor:
    if tuple(t.shape) == tuple(shape):
        return t
    out = torch.full(shape, fill, dtype=t.dtype)
    out[tuple(slice(0, s) for s in t.shape)] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--obs", type=int, required=True)
    ap.add_argument("--act", type=int, default=None)
    ap.add_argument("--new_std", type=float, default=0.3)
    ap.add_argument("--obs_insert", type=str, default=None)
    a = ap.parse_args()
    d = torch.load(a.inp, map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    old_obs = sd["actor.0.weight"].shape[1]
    old_act = sd["std"].shape[0]
    n_act = a.act or old_act
    last = max(int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight"))
    new_shapes = {}
    for k, v in sd.items():
        shape = tuple(v.shape)
        if k in ("actor.0.weight", "critic.0.weight"):
            shape = (shape[0], a.obs)
        elif k == f"actor.{last}.weight":
            shape = (n_act, shape[1])
        elif k == f"actor.{last}.bias":
            shape = (n_act,)
        elif k == "std":
            shape = (n_act,)
        new_shapes[k] = shape
    # column map for observations: old column j -> new column col_map[j]
    ins_pos, ins_cnt = (map(int, a.obs_insert.split(":")) if a.obs_insert else (old_obs, 0))
    col_map = torch.tensor([j if j < ins_pos else j + ins_cnt for j in range(old_obs)])
    assert old_obs + ins_cnt <= a.obs

    def remap_cols(t, n_new, fill):
        out = torch.full((*t.shape[:-1], n_new), fill, dtype=t.dtype)
        out[..., col_map] = t
        return out

    for k, v in list(sd.items()):
        if k in ("actor.0.weight", "critic.0.weight"):
            sd[k] = remap_cols(v, a.obs, 0.0)
            continue
        fill = a.new_std if k == "std" else 0.0
        sd[k] = pad_to(v, new_shapes[k], fill)
    # Adam state follows the parameter order (= state_dict order for ActorCritic)
    opt = d.get("optimizer_state_dict")
    if opt is not None:
        keys = list(sd.keys())
        for i, st in opt["state"].items():
            for m in ("exp_avg", "exp_avg_sq"):
                if m in st:
                    if keys[i] in ("actor.0.weight", "critic.0.weight"):
                        st[m] = remap_cols(st[m], a.obs, 0.0)
                    else:
                        st[m] = pad_to(st[m], new_shapes[keys[i]])
    for nk in ("obs_norm_state_dict", "critic_obs_norm_state_dict"):
        if nk in d and d[nk]:
            n = d[nk]
            n["_mean"] = remap_cols(n["_mean"], a.obs, 0.0)
            n["_var"] = remap_cols(n["_var"], a.obs, 1.0)
            n["_std"] = remap_cols(n["_std"], a.obs, 1.0)
    torch.save(d, a.out)
    print(f"expanded obs {old_obs}->{a.obs}, act {old_act}->{n_act}: {a.out}")


if __name__ == "__main__":
    main()
