"""P3c anchor (pitcher session 20, user-approved 11-07, planning 14 GO): PPO loss + lambda * mean((mu_theta(s) - mu_P3c(s))^2)
over every rollout state of each mini-batch and all action columns, mu_P3c = the mean action of a frozen copy of a
checkpoint's actor (P3c = ho2 model_7600).

Every resume from 7600 drifted with no reward term at all (fbctl, control B with critic warm-up: release -1.2..-1.6 m/s,
T3+stand 48 -> 0 in 100-200 iterations), so any new term is masked by the drift; the anchor holds the whole policy near P3c
while a term moves it.

No rsl_rl change: the PPO symmetry hook (use_mirror_loss without data augmentation) calls the augmentation function twice
per mini-batch - first with the observations (we return [obs; obs] and keep obs), then with the actor's means on the
originals (we return [means; teacher(obs)]); rsl_rl then adds mirror_loss_coeff * MSE(actor(obs), teacher(obs)) and logs
it as "Symmetry loss" (the unweighted MSE). The stored observations are already normalized; the anchor needs the
checkpoint's normalizer to be frozen (count >= until), which train.py checks."""

from __future__ import annotations

import copy

import torch


def attach_anchor(alg, ckpt_path: str, coef: float, device) -> torch.nn.Module:
    """Set alg.symmetry so that PPO adds coef x the anchor MSE; returns the frozen anchor actor."""
    teacher = copy.deepcopy(alg.actor_critic.actor)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    teacher.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    keep = {}

    def augment(obs=None, actions=None, env=None, is_critic=False):
        if obs is not None:  # first call: the mini-batch observations
            keep["obs"] = obs
            return torch.cat([obs, obs], dim=0), None
        with torch.no_grad():  # second call: the actor's means on the originals -> the anchor's means as the target
            target = teacher(keep.pop("obs"))
        return None, torch.cat([actions, target], dim=0)

    alg.symmetry = dict(use_data_augmentation=False, use_mirror_loss=True, mirror_loss_coeff=coef,
                        data_augmentation_func=augment, _env=None)
    print(f"[INFO]: P3c anchor {ckpt_path}: loss + {coef:g} x mean((mu - mu_anchor)^2) over all states and columns", flush=True)
    return teacher
