"""Joint ONE OUTS scene smoke run (batter session, M2 prep 10-28): P3c pitcher + the live RL batter in one scene.

One process = one pitch per env (the first-after-load reset; no env is reset again: the pitcher's dones are disabled,
as in eval_stage). Prints one row per env: clip, pitch speed, plate crossing (plate frame), swing outcome (hit, exit
speed, launch, spray, carry), falls of both players, and the batter's start / hand-off timing.

    python scripts/joint_smoke.py --headless --pitcher <P3c model.pt> --batter <model_12500.pt> --envs 64 --csv logs/joint_smoke.csv
"""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pitcher", type=str, required=True)
parser.add_argument("--batter", type=str, required=True)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=6.0, help="joint episode length (release ~1 s + flight + batted ball)")
parser.add_argument("--release_csv", type=str, default="assets/game/p3c_identical_t18_release.csv",
                    help="the batter's predicted pitch before the hand-off (the P3c gate table)")
parser.add_argument("--csv", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.one_outs.batter_side import JOINT_CLIPS, BatterSide  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env_cfg import PitchedBattingHumanPlayEnvCfg  # noqa: E402

TASK = "AIB-OneOutsJoint-v0"


def main():
    assert os.environ.get("AIB_EFFORT_SCALE") == "1.5"
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    cfg.episode_length_s = args.seconds
    cfg.ball_after_handoff = "park"
    if hasattr(cfg, "landed_start_prob"):
        cfg.landed_start_prob = 0.0
    if hasattr(cfg, "rsi_prob"):
        cfg.rsi_prob = 0.0
    if hasattr(cfg, "assist_force_n"):
        cfg.assist_force_n = 0.0
    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():  # one pitch per env: record falls, never reset (a reset would restart the batter too)
        orig()
        z = torch.zeros_like(base.fallen)
        return z, z
    base._get_dones = keep_running

    bcfg = PitchedBattingHumanPlayEnvCfg()
    bcfg.pitch_release_csv = args.release_csv
    bcfg.scene.num_envs = args.envs
    side = BatterSide(base, bcfg, args.batter)
    base.batter_side = side
    w = RslRlVecEnvWrapper(env)  # its reset = the first-after-load reset, now with the batter's own
    ag = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.pitcher)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    obs = w.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    n = base.num_envs
    T = int(round(cfg.episode_length_s / base.step_dt)) - 2
    p_fell = torch.zeros(n, dtype=torch.bool, device=base.device)
    b_fell = torch.zeros_like(p_fell)
    hit = torch.zeros_like(p_fell)
    start_at = torch.full((n,), float("nan"), device=base.device)
    print(f"[joint] {n} envs, {T} pitcher steps ({cfg.episode_length_s} s), batter start t "
          f"{side.start_t.min():.3f}..{side.start_t.max():.3f} s, release_t_pred {base.release_t_pred[0]:.3f} s", flush=True)
    with torch.inference_mode():
        for _ in range(T):
            was = side.active.clone()
            obs, _, _, _ = w.step(pol(obs))
            now = base.episode_length_buf.float() * base.step_dt
            start_at = torch.where(side.active & ~was, now, start_at)
            p_fell |= base.fallen
            b_fell |= side.fallen & side.active
            hit |= side.hit
    rows = {}
    for rec in side.play_log:  # first batted ball per env
        rows.setdefault(int(rec["env"]), rec)
    out = []
    names = list(JOINT_CLIPS)
    for e in range(n):
        rec = rows.get(e, {})
        jc = base.j_cross[e].tolist()  # (y - zone centre, z above the field); NaN when the bat met it first
        out.append(dict(
            env=e, clip=names[e % len(names)], kmh=round(float(side.pitch_kmh[e]), 1),
            plate_y=round(jc[0], 3), plate_z=round(jc[1], 3),
            handoff_t=round(float(base.handoff_t[e]), 4), batter_start=round(float(start_at[e]), 4),
            handed=bool(side.handed[e]), hit=bool(hit[e]), exit_mps=round(rec.get("exit_velo_mps", float("nan")), 2),
            launch_deg=round(rec.get("launch_deg", float("nan")), 1), spray_deg=round(rec.get("spray_deg", float("nan")), 1),
            carry_m=round(rec.get("carry_m", float("nan")), 1), batter_fell=bool(b_fell[e]), pitcher_fell=bool(p_fell[e])))
    for o in out:
        print("[joint] " + " ".join(f"{k}={v}" for k, v in o.items()), flush=True)
    late = sum(1 for o in out if o["handed"] and not (o["batter_start"] <= o["handoff_t"]))
    hr = sum(o["hit"] for o in out) / n
    print(f"[joint] summary: handed {sum(o['handed'] for o in out)}/{n}, hits {hr:.3f}, batter falls "
          f"{sum(o['batter_fell'] for o in out)}, pitcher falls {sum(o['pitcher_fell'] for o in out)}, released before "
          f"the batter's start {late}", flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(out[0]))
            wr.writeheader()
            wr.writerows(out)
    print("JOINT_SMOKE_END", flush=True)


main()
app.close()
