"""Human25-R falsifier: the recovery policy on held-out bank states (default the seed-6 bank), one 2 s episode per
state, no termination. Per hit / miss: on the ground within 2 s (pelvis < 0.35 m above the field), lost posture within
2 s (the env's fallen flag: pelvis < 0.55 m or tilt > 60 deg), tilted over (> 75 deg). --zero: a = 0 (the open-loop
anchor blend, the baseline).

    AIB_EFFORT_SCALE=1.5 python scripts/recover_eval.py --headless --checkpoint <model.pt> [--bank logs/recover_bank_s6.npz]
Last line: REVAL <tag> ground <all> hits <h> misses <m>
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--zero", action="store_true")
parser.add_argument("--bank", type=str, default="logs/recover_bank_s6.npz")
parser.add_argument("--tag", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.recover.recover_env_cfg import RecoverEvalEnvCfg  # noqa: E402


def load_actor(path, dev):
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck["model_state_dict"]
    n_obs, n_act, n_cr = sd["actor.0.weight"].shape[1], sd["actor.6.weight"].shape[0], sd["critic.0.weight"].shape[1]
    ac = ActorCritic(n_obs, n_cr, n_act, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                     activation="elu", init_noise_std=0.3).to(dev)
    ac.load_state_dict(sd)
    ac.eval()
    norm = EmpiricalNormalization(shape=[n_obs], until=1.0e8).to(dev)
    norm.load_state_dict(ck["obs_norm_state_dict"])
    norm.eval()
    return lambda o: ac.act_inference(norm(o))


def main():
    assert os.environ.get("AIB_EFFORT_SCALE") == "1.5"
    cfg = RecoverEvalEnvCfg()
    cfg.bank_path = args.bank
    cfg.scene.num_envs = int(np.load(args.bank)["hit"].shape[0])
    env = gym.make("AIB-Recover-Eval-v0", cfg=cfg)
    base = env.unwrapped
    n, dev = base.num_envs, base.device
    pol = None if args.zero else load_actor(args.checkpoint, dev)
    obs, _ = env.reset()
    hit = base.bank_hit[base.idx].clone()
    min_pz = torch.full((n,), 9.0, device=dev)
    lost = torch.zeros(n, dtype=torch.bool, device=dev)
    over = torch.zeros(n, dtype=torch.bool, device=dev)
    t_ground = torch.full((n,), float("nan"), device=dev)
    for k in range(int(base.max_episode_length) - 1):  # 1.99 s: the time-out step resets after it
        with torch.inference_mode():
            act = torch.zeros(n, base.cfg.action_space, device=dev) if pol is None else pol(obs["policy"])
        obs, *_ = env.step(act)
        pz, tilt = base.pz, base.tilt  # this step's state, read in _get_dones before any reset
        min_pz = torch.minimum(min_pz, pz)
        lost |= (pz < 0.55) | (tilt > 60.0)
        over |= tilt > 75.0
        t_ground = torch.where(torch.isnan(t_ground) & (pz < 0.35), torch.full_like(pz, (k + 1) * base.step_dt), t_ground)
    g = min_pz < 0.35
    tag = args.tag or ("zero" if args.zero else os.path.basename(args.checkpoint))

    def pct(m, sel):
        return f"{int((m & sel).sum())}/{int(sel.sum())} ({(m & sel).sum().item() / max(int(sel.sum()), 1):.1%})"

    q = lambda x: " / ".join(f"{v:.2f}" for v in np.quantile(x.cpu().numpy(), [0.0, 0.25, 0.5, 0.75, 1.0])) if len(x) else "-"  # noqa: E731
    for nm, sel in (("hits", hit), ("misses", ~hit)):
        print(f"[reval] {tag} {nm}: on the ground within 2 s {pct(g, sel)}, lost posture {pct(lost, sel)}, tilted > 75 deg "
              f"{pct(over, sel)}; min pelvis z {q(min_pz[sel])}; t on the ground med "
              f"{np.nanmedian(t_ground[sel].cpu().numpy()) if (g & sel).any() else float('nan'):.2f} s", flush=True)
    print(f"REVAL {tag} ground {g.float().mean().item():.3f} hits {g[hit].float().mean().item():.3f} "
          f"misses {g[~hit].float().mean().item():.3f} lost {lost.float().mean().item():.3f}", flush=True)
    env.close()


main()
app.close()
