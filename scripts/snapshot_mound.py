"""Side-view snapshots of the pitcher on the mound (set position, lead-foot landing, release, after the release).

    python scripts/snapshot_mound.py --headless [--checkpoint <pitch model.pt>] [--out videos/mound.png]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,
                    default="logs/rsl_rl/aib1_pitch/2026-10-07_19-35-41_recovery_step/model_12000.pt")
parser.add_argument("--out", type=str, default="videos/mound.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.pitching import pitch_env_cfg as C  # noqa: E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402


def grab(base):
    for _ in range(6):
        img = base.render()
        if img is not None and img.size and img[..., :3].max() > 0:
            return np.ascontiguousarray(img[..., :3])
    return None


def label(img, lines):
    img = img.copy()
    for i, t in enumerate(lines):
        y = 40 + 34 * i
        cv2.putText(img, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    cfg = C.PitchElasticPlayEnvCfg()
    cfg.scene.num_envs = 1
    cfg.terminate_on_cross = False
    cfg.episode_length_s = 3.0
    cfg.viewer.resolution = (960, 600)
    env = gym.make("AIB-PitchElastic-Play-v0", cfg=cfg, render_mode="rgb_array")
    w = RslRlVecEnvWrapper(env)
    ag = PitchPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=env.unwrapped.device)
    base, sim = env.unwrapped, env.unwrapped.sim
    o = base.scene.env_origins[0].cpu().numpy()
    eye, target = o + [1.0, -4.6, 0.55], o + [1.0, 0.0, 0.45]
    shots = {0.02: "set position on the rubber", 0.85: "lead foot should be planted (capture: 0.83 s)",
             1.12: "release (capture: 0.98 s)", 1.35: "0.25 s after the release"}
    frames = []
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(int(1.45 / base.step_dt)):
            obs, _, _, _ = w.step(pol(obs))
            obs = obs.clone()
            t = (k + 1) * base.step_dt
            sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            for ts, name in shots.items():
                if abs(t - ts) < 0.5 * base.step_dt:
                    img = grab(base)
                    if img is not None:
                        frames.append(label(img, [f"t = {t:.2f} s: {name}",
                                                  "mound: rubber 10 in above the field, 1:12 slope"]))
    grid = np.concatenate([np.concatenate(frames[0:2], 1), np.concatenate(frames[2:4], 1)], 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[snap] wrote {args.out} ({len(frames)} shots)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
