"""Evaluate a stage-1 step policy (stage_env.StageEnv): N full episodes, falls, standing at the end, key frames to
look at and a side-view video, in one simulator launch.

    python scripts/eval_stage.py --headless --task AIB-Stage1a-v0 --checkpoint <model.pt> [--envs 64]
        [--video videos/stage1a_iter_150.mp4] [--stochastic]
Prints "[goal] k/N" = episodes standing at rest at the end (read by train_cycles.sh).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1a-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--stochastic", action="store_true", help="sample the action noise (as in training)")
parser.add_argument("--video", type=str, default=None, help="side-view video of env 0 (+ key frames .png next to it)")
parser.add_argument("--video_episodes", type=int, default=2)
parser.add_argument("--slow", type=float, default=2.0, help="video slow-motion factor")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402


def say(*a):
    print(*a, flush=True)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    if args.video:
        cfg.viewer.resolution = (1280, 720)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():  # record falls, never end an episode early
        orig()
        z = torch.zeros_like(base.fallen)
        return z, z
    base._get_dones = keep_running
    w = RslRlVecEnvWrapper(env)
    ag = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    if args.stochastic:
        pol = lambda x: r.alg.actor_critic.act(r.obs_normalizer(x))  # noqa: E731
    T = int(round(cfg.episode_length_s / base.step_dt)) - 2
    n = base.num_envs
    fell_at = np.full(n, np.nan)
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            obs, _, _, _ = w.step(pol(obs))
            f = base.fallen.cpu().numpy()
            fell_at[np.isnan(fell_at) & f] = (k + 1) * base.step_dt
        both, speed, tilt, head_z, ok = (x.cpu().numpy() for x in base._stand_state())
    fell = ~np.isnan(fell_at)
    ok = ok & ~fell
    mode = "with training action noise" if args.stochastic else "noise-free"
    say(f"== eval_stage: {args.checkpoint} | {n} episodes of {cfg.episode_length_s:.1f} s, {args.task}, {mode}")
    say(f"  falls: {fell.sum()}/{n}" + (f", median at {np.nanmedian(fell_at):.2f} s" if fell.any() else ""))
    up = ~fell
    say(f"  at rest standing at the end (feet down, COM < {cfg.stand_max_com_v} m/s, trunk < {cfg.stand_max_tilt_deg:.0f} "
        f"deg, head > {cfg.stand_min_head_z} m): {ok.sum()}/{n} (failed: fell {fell.sum()}, feet {(up & ~both).sum()}, "
        f"COM speed {(up & (speed >= cfg.stand_max_com_v)).sum()}, trunk tilt {(up & (tilt >= cfg.stand_max_tilt_deg)).sum()}, "
        f"head {(up & (head_z <= cfg.stand_min_head_z)).sum()})")
    if up.any():
        say(f"  not fallen, at the end (median): trunk tilt {np.median(tilt[up]):.1f} deg, COM speed {np.median(speed[up]):.3f} "
            f"m/s, head {np.median(head_z[up]):.2f} m, both feet down {both[up].mean() * 100:.0f}%")
    if not args.stochastic:
        say(f"[goal] {ok.sum()}/{n}")
    if args.video:
        film(w, base, pol, T)


def film(w, base, pol, T, fps=30):
    """Side view of env 0 (others hidden); key frames at 0, 1/3, 2/3 and the end of episode 1 side by side (.png)."""
    import cv2
    import imageio
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for i in range(1, base.num_envs):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}")).MakeInvisible()

    def grab():
        for _ in range(6):
            img = base.render()
            if img is not None and img.size and img[..., :3].max() > 0:
                return np.ascontiguousarray(img[..., :3])
        return None

    every = max(1, int(round(1.0 / (fps * args.slow * base.step_dt))))
    keys = {0, T // 3, 2 * T // 3, T - 1}
    frames, kf = [], []
    for e in range(args.video_episodes):
        with torch.inference_mode():
            obs, _ = w.reset()
        fell = None
        for k in range(T):
            with torch.inference_mode():
                obs, _, _, _ = w.step(pol(obs))
                obs = obs.clone()
            t = (k + 1) * base.step_dt
            if fell is None and bool(base.fallen[0]):
                fell = t
            if k % every and not (e == 0 and k in keys):
                continue
            o = base.scene.env_origins[0].cpu().numpy()
            base.sim.set_camera_view(eye=(o + [0.3, -4.0, 0.9]).tolist(), target=(o + [0.3, 0.0, 0.7]).tolist())
            img = grab()
            if img is None:
                continue
            img = img.copy()
            txt = f"episode {e + 1}  t = {t:.2f} s" + (f"  FELL at {fell:.2f} s" if fell is not None else "")
            cv2.putText(img, txt, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, txt, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            if e == 0 and k in keys:
                kf.append(cv2.resize(img, (640, 360)))
            if k % every == 0:
                frames.append(img)
        say(f"[video] episode {e + 1}: fell at {fell}")
    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    imageio.mimwrite(args.video, frames, fps=fps, quality=8, macro_block_size=1)
    if kf:
        png = os.path.splitext(args.video)[0] + "_keyframes.png"
        imageio.imwrite(png, np.concatenate([np.concatenate(kf[:2], 1), np.concatenate(kf[2:4], 1)], 0)
                        if len(kf) == 4 else np.concatenate(kf, 1))
        say(f"[video] key frames {png}")
    say(f"[video] wrote {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
