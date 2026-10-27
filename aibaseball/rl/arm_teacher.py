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

    The phase is read from the (normalized) policy observation, so the same rule holds in the rollout, the PPO update and
    the eval: reference time >= release_ref + window - shift + margin (the latest forced finger-open, plus the ~17 ms the
    ball still rolls off the fingers), so every env still running has released. The released flag is not used:
    release_neutral (Stage1dF on) freezes the ball slice of the observation, the flag included, at its held value 0 -
    the first PR run (pr_p3c, 2026-10-27) gave the student no sample in 1000 iterations. Before that time, every action
    column comes from the frozen teacher (the deliverable's actor), so the release state cannot change."""
    c = env.cfg
    i_rel = 2 * len(env.body_ids) + 2 * len(env.finger_ids) + 19  # PitchEnv._get_observations: released, then phase
    i_ph = i_rel + 1
    t_min = float(env.release_ref) + c.release_action_window_s - c.release_action_shift_s + margin_s

    def rows(obs):
        m, s = normalizer._mean[0, i_ph], normalizer._std[0, i_ph] + normalizer.eps
        ph = (obs[:, i_ph] * s + m) * c.phase_duration_s
        return (ph >= t_min).unsqueeze(-1)

    rows.t_min, rows.i_rel = t_min, i_rel
    return rows


def attach_post_release_student(alg, ckpt_path: str, rows, device, normalizer=None):
    """PR: all action columns from a frozen copy of the checkpoint's actor except on student rows (rows(obs) True), where
    the trainable actor acts. Teacher rows have a detached mean and std: log-prob ratio 1, no gradient, KL 0.
    normalizer (batter PC): the live observation normalizer; the teacher then sees the observation under a frozen copy of
    it (its statistics at attach time = the checkpoint's), so the running statistics cannot move the teacher's actions."""
    ac = alg.actor_critic
    teacher = copy.deepcopy(ac.actor)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    teacher.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    object.__setattr__(ac, "_arm_teacher", teacher)
    object.__setattr__(ac, "_pr_rows", rows)

    frozen = None
    if normalizer is not None:
        frozen = copy.deepcopy(normalizer).eval()
        frozen.until = 0  # never updates

    def teacher_in(obs):
        if frozen is None:
            return obs
        raw = obs * (normalizer._std + normalizer.eps) + normalizer._mean
        return (raw - frozen._mean) / (frozen._std + frozen.eps)

    def mean_of(obs):
        with torch.no_grad():
            t = teacher(teacher_in(obs))
        return torch.where(rows(obs), ac.actor(obs), t)

    def update_distribution(obs):
        mean = mean_of(obs)
        std = (ac.std if ac.noise_std_type == "scalar" else torch.exp(ac.log_std)).expand_as(mean)
        std = torch.where(rows(obs), std, std.detach())
        ac.distribution = Normal(mean, std)

    ac.update_distribution = update_distribution
    ac.act_inference = mean_of
    print(f"[INFO]: post-release / post-clip student, teacher {ckpt_path}: student acts once ref t (phase) >= {rows.t_min:.4f} "
          f"(obs index {rows.i_rel})", flush=True)


def post_clip_rows(env, normalizer, phase_min: float = 0.9999):
    """Batter PC (decision session 2026-10-27, Human21): rows the student may act on - the reference clip has ended.

    The reference-time input t / duration (TeeBattingMimicEnv, held at 1 past the clip with clamp_ref_phase) is read from
    the normalized policy observation, so the same rule holds in the rollout, the PPO update and the evals. Before the
    clip end every action column comes from the frozen teacher, so the swing cannot change."""
    i_ph = env.phase_obs_index

    def rows(obs):
        m, s = normalizer._mean[0, i_ph], normalizer._std[0, i_ph] + normalizer.eps
        return ((obs[:, i_ph] * s + m) >= phase_min).unsqueeze(-1)

    rows.t_min, rows.i_rel = phase_min, i_ph
    return rows
