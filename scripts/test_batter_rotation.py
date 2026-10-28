"""M1.5 exactness test of the joint-scene batter frame proxy (batter session, 10-28).

Runs the delivery batter in its native env for one episode per env and records, per control step, the policy obs and
actions plus the per-env outcome (hit, exit speed, landing). With --rotate the batter articulation is wrapped in
RotatedArticulation (yaw 162 deg about each env origin), so the robot really stands rotated in the world while the env
code sees its usual frame. Compare two runs with --compare a.npz b.npz (plain vs plain gives PhysX run-to-run noise).

    python scripts/test_batter_rotation.py --headless --checkpoint <model.pt> --out logs/rot_plain.npz [--rotate]
    python scripts/test_batter_rotation.py --compare logs/rot_plain.npz logs/rot_rot.npz
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rotate", action="store_true")
parser.add_argument("--yaw", type=float, default=162.0)
parser.add_argument("--release_csv", type=str, default="assets/game/p3c_identical_t18_release.csv")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--compare", type=str, nargs=2, default=None)
parser.add_argument("--pre_steps", type=int, default=50, help="steps compared as 'before contact'")

if "--compare" in sys.argv:
    import numpy as np

    args = parser.parse_args()
    a, b = (np.load(p) for p in args.compare)
    k = args.pre_steps
    d_obs = np.abs(a["obs"] - b["obs"])
    d_act = np.abs(a["act"] - b["act"])
    first = [int(np.argmax(d_obs[:, e].max(-1) > 1e-3)) if (d_obs[:, e].max(-1) > 1e-3).any() else -1 for e in range(d_obs.shape[1])]
    same_hit = (a["hit"] == b["hit"]).mean()
    both = a["hit"] & b["hit"]
    dv = np.abs(a["exit"] - b["exit"])[both]
    dl = np.linalg.norm(a["land"] - b["land"], axis=-1)[both]
    print(f"pre-contact (first {k} steps): obs max |d| {d_obs[:k].max():.2e}  act max |d| {d_act[:k].max():.2e}")
    print(f"whole episode: obs max |d| {d_obs.max():.2e}  act max |d| {d_act.max():.2e}  first step obs |d| > 1e-3: "
          f"median {int(np.median([f for f in first if f >= 0])) if any(f >= 0 for f in first) else 'never'}")
    print(f"hit agree {same_hit:.3f} (hits {a['hit'].mean():.3f} / {b['hit'].mean():.3f})  both-hit exit |d| max "
          f"{dv.max() if len(dv) else 0:.3f} m/s  (carry m, spray deg) |d| max {dl.max() if len(dl) else 0:.2f}  "
          f"fell agree {(a['fell'] == b['fell']).mean():.3f}")
    print("ROT_CMP_END")
    sys.exit()

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.one_outs.batter_side import JOINT_CLIPS, RotatedArticulation  # noqa: E402
from aibaseball.tasks.tee_batting.agents.rsl_rl_ppo_cfg import TeeBattingMimicPPORunnerCfg  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env_cfg import PitchedBattingHumanPlayEnvCfg  # noqa: E402


def main():
    assert os.environ.get("AIB_EFFORT_SCALE") == "1.5"
    cfg = PitchedBattingHumanPlayEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    cfg.pitch_release_csv = args.release_csv
    cfg.motion_files = [f for f in cfg.motion_files if os.path.splitext(os.path.basename(f))[0] in JOINT_CLIPS]
    cfg.real_height = True
    cfg.timing_noise_s, cfg.timing_bias_s, cfg.aim_offset_z = 0.0, 0.055, 0.0
    cfg.ball_obs_noise_std, cfg.ball_obs_bias_std = 0.0, 0.0  # perception noise off: the two runs see the same ball
    env = gym.make("AIB-PitchedBatting-Human-Play-v0", cfg=cfg)
    base = env.unwrapped
    if args.rotate:  # before the wrapper's first reset: every root write already goes through the proxy
        o = base.scene.env_origins
        art = RotatedArticulation(base.robot, o.clone(), args.yaw)
        art.pivot = o - quat_apply(art.q_r, o)  # rotate each env about its own origin (W = R (B - o) + o)
        base.robot = art
    wrapped = RslRlVecEnvWrapper(env)
    agent = TeeBattingMimicPPORunnerCfg()
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=base.device)
    obs = wrapped.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    n, steps = base.num_envs, base.max_episode_length - 1
    O, A = [], []
    hit = torch.zeros(n, dtype=torch.bool, device=base.device)
    fell = torch.zeros_like(hit)
    over = torch.zeros_like(hit)
    exit_v = torch.zeros(n, device=base.device)
    land = torch.zeros(n, 2, device=base.device)
    base.play_log.clear()
    with torch.inference_mode():
        for _ in range(steps):
            act = policy(obs)
            raw = obs if torch.is_tensor(obs) else obs["policy"]  # the env's obs (no extra _get_observations call: it updates the percept history)
            O.append(raw.cpu().numpy())
            A.append(act.cpu().numpy())
            live = ~over
            obs, _, done, _ = wrapped.step(act)
            done = done.bool()
            hit |= base.hit & live & ~done  # a done env is already reset here: its hit flag is the new episode's
            fell |= base.fallen & live  # set in _get_dones before the reset
            over |= done  # the first episode only (a done env auto-resets)
    seen = set()
    for r in base.play_log:  # first record per env = its first episode's batted ball
        e = int(r["env"])
        if e in seen:
            continue
        seen.add(e)
        exit_v[e] = r["exit_velo_mps"]
        land[e] = torch.tensor([r["carry_m"], r["spray_deg"]])
    np.savez(args.out, obs=np.stack(O), act=np.stack(A), hit=hit.cpu().numpy(), fell=fell.cpu().numpy(),
             exit=exit_v.cpu().numpy(), land=land.cpu().numpy())
    print(f"[rot] {'rotated' if args.rotate else 'plain'} {len(O)} steps, hits {hit.float().mean():.3f}, "
          f"play_log {len(base.play_log)} -> {args.out}", flush=True)
    print("ROT_RUN_END", flush=True)


main()
env_close = getattr(app, "close", None)
if env_close:
    env_close()
