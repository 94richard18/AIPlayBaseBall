"""Arm teacher (F, planning session 2026-10-27): the arm (shoulder / elbow / wrist) and release action columns come from a
frozen copy of a checkpoint's actor; the trainable actor only drives the rest.

B / P3CPC / 10-25 C all lost the pitch within 50 iterations while the actor weights moved 0.016 %: the ball left one 2.5 ms
step later and slower (17.8 -> 16.3 m/s) even with the release command held (E), so the arm motion itself drifted.

The teacher's mean replaces those columns inside the action distribution (no gradient), and their std entries are held at
the loaded values, so the rollout and the PPO update see the same mean / std there: their log-prob ratio is exactly 1 and
they add nothing to the KL, the entropy gradient or the surrogate gradient. The critic is untouched. Nothing new is stored
in the checkpoint: evaluate a student checkpoint with the same teacher (eval_stage --arm_teacher)."""

from __future__ import annotations

import copy

import torch
from torch.distributions import Normal


def attach_arm_teacher(alg, ckpt_path: str, cols: list[int], device) -> torch.Tensor:
    """Patch alg.actor_critic (rsl_rl ActorCritic) in place; returns the bool mask of teacher columns."""
    ac = alg.actor_critic
    teacher = copy.deepcopy(ac.actor)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    teacher.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    n = ac.std.numel() if ac.noise_std_type == "scalar" else ac.log_std.numel()
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[cols] = True
    object.__setattr__(ac, "_arm_teacher", teacher)  # not a submodule: stays out of the state dict / optimizer

    def mean_of(obs):
        m = ac.actor(obs)
        with torch.no_grad():
            t = teacher(obs)
        return torch.where(mask, t, m)

    def update_distribution(obs):
        mean = mean_of(obs)
        std = (ac.std if ac.noise_std_type == "scalar" else torch.exp(ac.log_std)).expand_as(mean)
        std = torch.where(mask, std.detach(), std)
        ac.distribution = Normal(mean, std)

    ac.update_distribution = update_distribution
    ac.act_inference = mean_of
    # the restored Adam moments could still move the held std entries with zero gradient: put them back after updates
    sp = ac.std if ac.noise_std_type == "scalar" else ac.log_std
    held = sp.detach().clone()
    update0 = alg.update

    def update(*a, **k):
        out = update0(*a, **k)
        with torch.no_grad():
            sp[mask] = held[mask]
        return out

    alg.update = update
    print(f"[INFO]: arm teacher {ckpt_path}: {int(mask.sum())} action columns from the frozen actor ({cols})", flush=True)
    return mask
