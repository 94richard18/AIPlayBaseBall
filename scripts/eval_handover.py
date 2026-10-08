"""Full pitches with two skills: the pitching policy until the release, the balance policy afterwards.

    python scripts/eval_handover.py --headless --pitch <pitch model.pt> --balance <balance model.pt> [--envs 32]
Prints release speed, strikes and whether the pitcher is still up 1.5 s after the release.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pitch", type=str, required=True)
parser.add_argument("--balance", type=str, required=True)
parser.add_argument("--envs", type=int, default=32)
parser.add_argument("--cfg", choices=["elastic", "stand_first", "speed"], default="elastic",
                    help="env config the pitching policy was trained with")
parser.add_argument("--hold_s", type=float, default=1.5, help="must stay up this long after the release")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.pitching import pitch_env_cfg as C  # noqa: E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchBalancePPORunnerCfg, PitchPPORunnerCfg  # noqa: E402


def main():
    cfg = {"elastic": C.PitchElasticEnvCfg, "stand_first": C.PitchStandFirstEnvCfg, "speed": C.PitchSpeedEnvCfg}[args.cfg]()
    single = os.path.abspath(args.pitch) == os.path.abspath(args.balance)  # one policy does the whole pitch
    bal_cfg = cfg if single else C.PitchBalanceEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    cfg.episode_length_s = 4.0
    cfg.follow_through_s = args.hold_s + 0.5
    cfg.post_release_residual_gain = bal_cfg.post_release_residual_gain  # the balance skill's action range
    env = gym.make("AIB-PitchElastic-v0", cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():  # only falls end an episode early; one pitch per env
        orig()
        return base.fallen.clone(), base.episode_length_buf >= base.max_episode_length - 1
    base._get_dones = keep_running
    w = RslRlVecEnvWrapper(env)
    policies = []
    for path, ag in ((args.pitch, PitchPPORunnerCfg()), (args.balance, PitchBalancePPORunnerCfg())):
        r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
        r.load(path)
        r.eval_mode()  # frozen observation normalisers (both policies see every step here)
        policies.append(r.get_inference_policy(device=base.device))
    pitch_pol, bal_pol = policies
    n, dev = base.num_envs, base.device
    t_rel = torch.full((n,), -1.0, device=dev)
    t_fall = torch.full((n,), -1.0, device=dev)
    speed = torch.zeros(n, device=dev)
    strike = torch.zeros(n, dtype=torch.bool, device=dev)
    done_once = torch.zeros(n, dtype=torch.bool, device=dev)
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(int(3.8 / base.step_dt)):
            rel = base.released.clone()
            act = torch.where(rel.unsqueeze(-1), bal_pol(obs), pitch_pol(obs))
            obs, _, dones, _ = w.step(act)
            t = torch.full((n,), (k + 1) * base.step_dt, device=dev)  # all envs start together; resets don't count
            new = base.released & (t_rel < 0) & ~done_once
            if new.any():
                t_rel[new] = t[new]
                speed[new] = base.rel_vel[new].norm(dim=-1)
                ids = new.nonzero(as_tuple=False).squeeze(-1)
                strike[ids] = base._release_outcome(ids)["strike"]
            fell = (dones > 0) & ~done_once & (t_rel >= 0)
            t_fall[fell] = t[fell]
            pre = (dones > 0) & ~done_once & (t_rel < 0)
            done_once |= dones > 0
            t_fall[pre] = 0.0  # fell (or dropped) before releasing
    up = (t_rel >= 0) & ((t_fall < 0) | (t_fall - t_rel >= args.hold_s))
    kmh = speed[t_rel >= 0] * 3.6
    print(f"[handover] released {int((t_rel >= 0).sum())}/{n}, release {kmh.mean():.1f} km/h "
          f"(min {kmh.min():.1f}, max {kmh.max():.1f}), strikes {int(strike.sum())}/{n}", flush=True)
    print(f"[handover] still up {args.hold_s:.1f} s after the release: {int(up.sum())}/{n}", flush=True)
    for i in range(n):
        if t_rel[i] >= 0 and t_fall[i] >= 0:
            print(f"[handover] env {i:2d}: {speed[i] * 3.6:5.1f} km/h, fell {t_fall[i] - t_rel[i]:+.2f} s after the release",
                  flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
