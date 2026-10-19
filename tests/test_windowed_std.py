"""aibaseball.rl.windowed_std: the windowed noise std is used consistently for sampling and log-probs (CPU only)."""

import math

import torch
from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from torch.distributions import Normal

from aibaseball.rl.windowed_std import WindowedStd

N_OBS, N_ACT, PHASE_COL = 216, 45, 110
ARM = [16, 18, 20, 22, 24, 26, 28]
PD, REL = 1.1167, 0.9806
LO, HI = (REL - 0.3) / PD, REL / PD


def make(k=2.5, seed=0):
    torch.manual_seed(seed)
    ac = ActorCritic(N_OBS, N_OBS, N_ACT, actor_hidden_dims=[64], critic_hidden_dims=[64], init_noise_std=0.03)
    norm = EmpiricalNormalization(shape=[N_OBS], until=1e8)
    raw = torch.randn(4096, N_OBS)
    raw[:, PHASE_COL] = torch.rand(4096) * 2.0  # phase 0..2 (t / phase_duration)
    norm.train()
    norm(raw)  # fit the statistics
    ws = WindowedStd(ac, norm, PHASE_COL, LO, HI, ARM, k)
    return ac, norm, ws, raw


def test_log_prob_matches_sampling_distribution():
    k = 2.5
    ac, norm, ws, raw = make(k)
    norm.eval()
    obs = norm(raw)
    a = ac.act(obs)
    lp = ac.get_actions_log_prob(a)
    # reference: the policy's own mean, std x k on the arm columns inside the window, x 1 elsewhere
    mean = ac.actor(obs)
    inside = (raw[:, PHASE_COL] >= LO) & (raw[:, PHASE_COL] <= HI)
    std = ac.std.expand_as(mean).clone()
    std[:, ARM] = torch.where(inside.unsqueeze(-1), std[:, ARM] * k, std[:, ARM])
    ref = Normal(mean, std).log_prob(a).sum(-1)
    assert torch.allclose(lp, ref, atol=1e-4)
    # the samples themselves: arm columns spread k x wider inside the window
    z = (a - mean) / ac.std
    r_in = z[inside][:, ARM].std() / z[inside][:, [0, 1, 2]].std()
    r_out = z[~inside][:, ARM].std() / z[~inside][:, [0, 1, 2]].std()
    assert abs(r_in - k) < 0.15 and abs(r_out - 1.0) < 0.1, (r_in, r_out)


def test_update_recomputes_the_same_log_prob():
    """PPO recomputes log-probs on the stored (normalized) obs in the update: same snapshot -> same masks -> same values,
    even after the normalizer moved during the rollout."""
    ac, norm, ws, raw = make()
    norm.train()
    obs = norm(raw)  # rollout: stats move while observations are normalized
    with torch.no_grad():
        a = ac.act(obs)
        lp_rollout = ac.get_actions_log_prob(a)
    norm(torch.randn(4096, N_OBS) * 3 + 1)  # more drift before the update
    ac.act(obs)  # the update's recompute on the stored obs (snapshot not refreshed yet)
    assert torch.allclose(ac.get_actions_log_prob(a), lp_rollout, atol=1e-5)
    ws.snapshot()  # after the update: the next rollout uses new stats
    assert not math.isnan(float(ws.s))


def test_inference_is_unchanged():
    ac, norm, ws, raw = make()
    norm.eval()
    obs = norm(raw)
    assert torch.allclose(ac.act_inference(obs), ac.actor(obs))
