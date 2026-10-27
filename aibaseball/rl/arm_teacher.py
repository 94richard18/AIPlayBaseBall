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


def post_release_rows(env, normalizer, margin_s: float = 0.03):
    """PR (planning session 2026-10-27): rows (samples) the student may act on - the ball has left the hand for sure.

    The released flag and the phase are read from the (normalized) policy observation, so the same rule holds in the
    rollout, the PPO update and the eval: released and reference time >= release_ref + window - shift + margin (the
    latest forced finger-open, plus the ~17 ms the ball still rolls off the fingers). Before that, every action column
    comes from the frozen teacher (the deliverable's actor), so the release state cannot change."""
    c = env.cfg
    i_rel = 2 * len(env.body_ids) + 2 * len(env.finger_ids) + 19  # PitchEnv._get_observations: released, then phase
    i_ph = i_rel + 1
    t_min = float(env.release_ref) + c.release_action_window_s - c.release_action_shift_s + margin_s

    def rows(obs):
        m, s = normalizer._mean[0], normalizer._std[0] + normalizer.eps
        rel = obs[:, i_rel] * s[i_rel] + m[i_rel]
        ph = (obs[:, i_ph] * s[i_ph] + m[i_ph]) * c.phase_duration_s
        return ((rel > 0.5) & (ph >= t_min)).unsqueeze(-1)

    rows.t_min, rows.i_rel = t_min, i_rel
    return rows


def attach_post_release_student(alg, ckpt_path: str, rows, device):
    """PR: all action columns from a frozen copy of the checkpoint's actor except on student rows (rows(obs) True), where
    the trainable actor acts. Teacher rows have a detached mean and std: log-prob ratio 1, no gradient, KL 0."""
    ac = alg.actor_critic
    teacher = copy.deepcopy(ac.actor)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    teacher.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    object.__setattr__(ac, "_arm_teacher", teacher)
    object.__setattr__(ac, "_pr_rows", rows)

    def mean_of(obs):
        with torch.no_grad():
            t = teacher(obs)
        return torch.where(rows(obs), ac.actor(obs), t)

    def update_distribution(obs):
        mean = mean_of(obs)
        std = (ac.std if ac.noise_std_type == "scalar" else torch.exp(ac.log_std)).expand_as(mean)
        std = torch.where(rows(obs), std, std.detach())
        ac.distribution = Normal(mean, std)

    ac.update_distribution = update_distribution
    ac.act_inference = mean_of
    print(f"[INFO]: post-release student, teacher {ckpt_path}: student acts once released and ref t >= {rows.t_min:.4f} s "
          f"(obs released index {rows.i_rel})", flush=True)
