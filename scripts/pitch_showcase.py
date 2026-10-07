"""Evaluate the pitching policy and/or record a video (catcher-view + side view of the flight).

    python scripts/pitch_showcase.py --headless --eval_envs 32          # statistics (full pitches from the set position)
    python scripts/pitch_showcase.py --headless --pitches 4             # video: videos/pitch_showcase.mp4 (best strike)
"""

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--out", type=str, default="videos/pitch_showcase.mp4")
parser.add_argument("--pitches", type=int, default=3)
parser.add_argument("--eval_envs", type=int, default=0)
parser.add_argument("--elastic", action="store_true", help="tendon-elastic throwing arm (AIB-PitchElastic)")
parser.add_argument("--dt", type=float, default=None, help="override physics dt (checkpoints trained at 1/400)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.eval_envs == 0:
    args.enable_cameras = True
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402
from aibaseball.tasks.pitching.pitch_env_cfg import PitchElasticPlayEnvCfg, PitchPlayEnvCfg  # noqa: E402

if args.elastic:
    PitchPlayEnvCfg = PitchElasticPlayEnvCfg  # noqa: F811
TASK = "AIB-PitchElastic-Play-v0" if args.elastic else "AIB-Pitch-Play-v0"


def newest_checkpoint():
    files = glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "aib1_pitch", "*", "model_*.pt"))
    return max(files, key=os.path.getmtime)


def load_policy(env, ckpt):
    agent = PitchPPORunnerCfg()
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(ckpt)
    return runner.get_inference_policy(device=env.unwrapped.device)


def text(img, lines, big=None):
    img = np.ascontiguousarray(img[..., :3])
    y = 40
    for t, col in lines:
        cv2.putText(img, t, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        y += 36
    if big:
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
        org = ((img.shape[1] - size[0]) // 2, img.shape[0] - 50)
        cv2.putText(img, big, org, cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(img, big, org, cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 220, 0), 3, cv2.LINE_AA)
    return img


_last = None


def grab(base):
    global _last
    for _ in range(4):
        img = base.render()
        if img is not None and img.size and img[..., :3].max() > 0:
            _last = img
            return img
    return _last


def run_eval(ckpt):
    cfg = PitchPlayEnvCfg()
    if args.dt:
        cfg.sim.dt = args.dt
    cfg.scene.num_envs = args.eval_envs
    env = gym.make(TASK, cfg=cfg)
    w = RslRlVecEnvWrapper(env)
    policy = load_policy(w, ckpt)
    base = env.unwrapped
    with torch.inference_mode():
        obs, _ = w.reset()
        for _ in range(int(cfg.episode_length_s / base.step_dt)):
            obs, _, _, _ = w.step(policy(obs))
            if len(base.play_log) >= args.eval_envs:
                break
    logs = base.play_log[: args.eval_envs]
    kmh = np.array([r["release_kmh"] for r in logs]) if logs else np.zeros(0)
    strike = np.array([r["strike"] for r in logs]) if logs else np.zeros(0, bool)
    ok = (kmh >= cfg.target_speed * 3.6 - 1e-3) & strike
    stats = dict(checkpoint=ckpt, pitches=args.eval_envs, released=len(logs),
                 release_kmh_mean=float(kmh.mean()) if len(kmh) else 0.0, strike_rate=float(strike.sum() / args.eval_envs),
                 success_rate=float(ok.sum() / args.eval_envs),
                 plate_y_mean=float(np.mean([r["plate_y"] for r in logs])) if logs else 0.0,
                 plate_z_mean=float(np.mean([r["plate_z"] for r in logs])) if logs else 0.0)
    os.makedirs(os.path.join(ROOT, "videos"), exist_ok=True)
    with open(os.path.join(ROOT, "videos", "pitch_stats.json"), "w", encoding="utf-8") as f:
        json.dump(dict(stats=stats, pitches=logs), f, indent=2)
    print("[eval]", json.dumps(stats, indent=2), flush=True)


def run_video(ckpt):
    cfg = PitchPlayEnvCfg()
    if args.dt:
        cfg.sim.dt = args.dt
    cfg.scene.num_envs = 1
    cfg.terminate_on_cross = False
    cfg.episode_length_s = 6.0
    cfg.viewer.resolution = (1280, 720)
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
    w = RslRlVecEnvWrapper(env)
    policy = load_policy(w, ckpt)
    base, sim = env.unwrapped, env.unwrapped.sim
    title = ("AIB-1 humanoid  |  pitching: 4-seam fastball held by finger contact  |  Isaac Sim 4.5 PhysX + drag/Magnus",
             (255, 255, 255))
    best = None
    for k in range(args.pitches):
        frames = []
        with torch.inference_mode():
            obs, _ = w.reset()
        n0 = len(base.play_log)
        steps, after = 0, 0
        while steps < 600:
            with torch.inference_mode():
                obs, _, _, _ = w.step(policy(obs))
                obs = obs.clone()
            steps += 1
            o = base.scene.env_origins[0].cpu().numpy()
            ball = base.ball.data.root_pos_w[0].cpu().numpy() - o
            rel = bool(base.released[0])
            if not rel:  # side view of the delivery, slow motion (100 Hz -> 30 fps)
                sim.set_camera_view(eye=(o + [1.0, -4.2, 1.4]).tolist(), target=(o + [0.6, 0.0, 1.1]).tolist())
                phase = "delivery (3.3x slow motion)"
            else:  # catcher's view: the ball comes toward the camera and crosses the zone
                after += 1
                sim.set_camera_view(eye=(o + [cfg.plate_distance + 2.5, 0.0, 1.0]).tolist(),
                                    target=(o + [cfg.plate_distance - 6.0, 0.0, 1.0]).tolist())
                phase = "catcher view (slow motion)"
            img = grab(base)
            lines = [title, (phase, (200, 200, 200))]
            if rel:
                v = float(base.rel_vel[0].norm()) * 3.6
                lines.append((f"release speed {v:5.1f} km/h   ball x {ball[0]:5.2f} m (plate at {cfg.plate_distance} m)",
                              (120, 220, 255)))
            frames.append(text(img, lines))
            if bool(base.crossed[0]) and after > 3:
                break
        if len(base.play_log) <= n0:
            print("[pitch-video] no release this pitch", flush=True)
            continue
        rec = base.play_log[-1]
        ok = rec["strike"] and rec["release_kmh"] >= cfg.target_speed * 3.6 - 1e-3
        verdict = f"{rec['release_kmh']:.1f} km/h  -  {'STRIKE' if rec['strike'] else 'BALL'}"
        for _ in range(60):  # hold on the zone with the crossing point
            frames.append(text(grab(base), [title, (f"plate crossing: y {rec['plate_y']:+.2f} m, z {rec['plate_z']:.2f} m"
                                                    f"   spin {rec['spin_rpm']:.0f} rpm", (120, 220, 255))], big=verdict))
        score = (1000 if ok else 0) + (500 if rec["strike"] else 0) + rec["release_kmh"]
        print(f"[pitch-video] pitch {k + 1}: {verdict}", flush=True)
        if best is None or score > best[0]:
            best = (score, frames, rec)
    if best is None:
        raise RuntimeError("no pitch released")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, best[1], fps=30, quality=8, macro_block_size=1)
    print(f"[pitch-video] wrote {args.out}: {json.dumps(best[2])}", flush=True)


if __name__ == "__main__":
    try:
        ck = args.checkpoint or newest_checkpoint()
        print("[pitch] checkpoint", ck, flush=True)
        run_eval(ck) if args.eval_envs else run_video(ck)
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
