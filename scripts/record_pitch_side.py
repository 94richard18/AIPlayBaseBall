"""Side-view slow-motion video of full pitches (stride, foot strike, release, follow-through) on the mound.

    python scripts/record_pitch_side.py --headless --checkpoint <pitch model.pt> [--pitches 3] [--slow 4]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, default="videos/pitch_side.mp4")
parser.add_argument("--pitches", type=int, default=3)
parser.add_argument("--slow", type=float, default=4.0, help="slow-motion factor")
parser.add_argument("--cfg", choices=["elastic", "stand_first", "speed"], default="stand_first",
                    help="env config the policy was trained with")
parser.add_argument("--seconds", type=float, default=2.6, help="simulated time per pitch (falls come ~1 s after the release)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.pitching import pitch_env_cfg as C  # noqa: E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402

FPS = 30


def grab(base):
    for _ in range(6):
        img = base.render()
        if img is not None and img.size and img[..., :3].max() > 0:
            return np.ascontiguousarray(img[..., :3])
    return None


def put(img, lines):
    for i, (t, col) in enumerate(lines):
        y = 36 + 32 * i
        cv2.putText(img, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    return img


def main():
    cfg = {"elastic": C.PitchElasticEnvCfg, "stand_first": C.PitchStandFirstEnvCfg, "speed": C.PitchSpeedEnvCfg}[args.cfg]()
    cfg.play_mode = True  # ball flies with PhysX + aerodynamics, zone drawn
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    cfg.scene.num_envs = 1
    cfg.terminate_on_cross = False
    cfg.episode_length_s = args.seconds + 0.5
    cfg.viewer.resolution = (1280, 720)
    env = gym.make("AIB-PitchElastic-Play-v0", cfg=cfg, render_mode="rgb_array")
    base, sim = env.unwrapped, env.unwrapped.sim
    orig = base._get_dones

    def no_early_end():  # keep filming after a fall
        orig()
        return torch.zeros_like(base.released), base.episode_length_buf >= base.max_episode_length - 1
    base._get_dones = no_early_end
    w = RslRlVecEnvWrapper(env)
    ag = PitchPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    every = max(1, int(round(1.0 / (FPS * args.slow * base.step_dt))))
    name = os.path.basename(os.path.dirname(args.checkpoint)) + "/" + os.path.basename(args.checkpoint)
    frames = []
    for p in range(args.pitches):
        with torch.inference_mode():
            obs, _ = w.reset()
        t_rel, kmh, fell_at = None, None, None
        for k in range(int(args.seconds / base.step_dt)):
            with torch.inference_mode():
                obs, _, _, _ = w.step(pol(obs))
                obs = obs.clone()
            t = (k + 1) * base.step_dt
            if t_rel is None and bool(base.released[0]):
                t_rel, kmh = t, float(base.rel_vel[0].norm()) * 3.6
            if fell_at is None and bool(base.fallen[0]):
                fell_at = t
            if k % every:
                continue
            o = base.scene.env_origins[0].cpu().numpy()
            pel = base.robot.data.root_link_pos_w[0].cpu().numpy() - o
            cx = float(np.clip(pel[0], 0.3, 1.8))
            sim.set_camera_view(eye=(o + [cx, -4.2, 0.9]).tolist(), target=(o + [cx, 0.0, 0.6]).tolist())
            img = grab(base)
            if img is None:
                continue
            lines = [(f"pitch {p + 1}/{args.pitches}   t = {t:.2f} s   ({args.slow:.0f}x slow motion)   {name}", (255, 255, 255)),
                     ("athlete: lead foot down 0.88 s, release 0.98 s", (200, 200, 200))]
            if t_rel is not None:
                lines.append((f"released at {t_rel:.2f} s, {kmh:.0f} km/h", (120, 255, 120)))
            if fell_at is not None:
                lines.append((f"FELL at {fell_at:.2f} s", (255, 90, 90)))
            frames.append(put(img.copy(), lines))
        print(f"[side] pitch {p + 1}: release {t_rel}, {kmh} km/h, fell at {fell_at}", flush=True)
        frames += [frames[-1]] * FPS  # hold the last frame 1 s
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=FPS, quality=8, macro_block_size=1)
    print(f"[side] wrote {args.out} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
