"""Record a showcase video of the trained batter in Isaac Sim, and/or evaluate carry statistics.

    # video (slow-motion swing -> ball-tracking flight -> overview of the landing spot)
    python scripts/showcase.py --headless --enable_cameras
    # statistics over many swings (training randomization on), no video
    python scripts/showcase.py --headless --eval_envs 256

Output: videos/showcase.mp4 (+ videos/showcase_stats.json for --eval_envs)
"""

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="launch", choices=["pitched", "human_pitched", "human", "perception", "multi", "launch", "power", "mimic", "rl"],
                    help="mimic: OBP imitation policy, rl: pure-RL policy")
parser.add_argument("--checkpoint", type=str, default=None, help="model_*.pt (default: newest in logs/)")
parser.add_argument("--out", type=str, default="videos/showcase.mp4")
parser.add_argument("--swings", type=int, default=1)
parser.add_argument("--all", action="store_true", help="keep every swing in the video (not only the best)")
parser.add_argument("--pitch_kmh", type=float, nargs=2, default=None, help="pitched task: pitch speed range")
parser.add_argument("--timing_noise", type=float, default=None, help="pitched task: +- arrival-time offset (s)")
parser.add_argument("--timing_bias", type=float, default=None, help="pitched task: cfg.timing_bias_s (s, + = pitch later)")
parser.add_argument("--tee_offset_z", type=float, default=None, help="pitched task eval: cfg.tee_offset_z (m, + = pitch aimed above the sweet-spot path)")
parser.add_argument("--release_csv", type=str, default=None, help="pitches from the RL pitcher's eval_stage --release_csv (own release state each)")
parser.add_argument("--pitch_csv", type=str, default=None, help="pitches from the RL pitcher's eval_stage --pitch_csv table (speed_mps, plate_y, plate_z, crossed)")
parser.add_argument("--phase_horizon", type=float, default=None, help="cfg.phase_horizon_s (s): phase input scale, e.g. the training episode 2.6")
parser.add_argument("--aim_offset_z", type=float, default=None, help="batter: cfg.aim_offset_z (m, < 0 = aim under the ball)")
parser.add_argument("--aim_offset_y", type=float, default=None, help="batter: cfg.aim_offset_y (m, lateral aim offset, world y)")
parser.add_argument("--fixed_cam", action="store_true", help="pitched task: one fixed catcher's-view camera")
parser.add_argument("--eval_envs", type=int, default=0, help=">0: run statistics instead of a video")
parser.add_argument("--post_ref_blend", type=float, default=None, help="cfg.post_ref_blend_s: after the clip, blend to a balanced pose over this many s")
parser.add_argument("--post_ref_pose", type=str, default=None, help="cfg.post_ref_pose: contact | start")
parser.add_argument("--post_ref_keep_residual", action="store_true", help="cfg.post_ref_keep_residual")
parser.add_argument("--clamp_ref_phase", action="store_true", help="cfg.clamp_ref_phase: reference-time input held at 1 past the clip")
parser.add_argument("--settle_tail", type=float, default=None, help="cfg.settle_tail_s: the reference decelerates to a stop after the clip")
parser.add_argument("--zero_ff_after_ref_end", action="store_true", help="cfg.zero_ff_after_ref_end: no velocity feed-forward on the held pose")
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
from aibaseball.tasks.tee_batting.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    TeeBattingMimicPPORunnerCfg,
    TeeBattingPPORunnerCfg,
)
from aibaseball.tasks.tee_batting.mimic_env_cfg import (  # noqa: E402
    TeeBattingMimicEnvCfg,
    TeeBattingMimicLaunchEnvCfg,
    TeeBattingMimicLaunchPlayEnvCfg,
    TeeBattingMimicPlayEnvCfg,
    TeeBattingMimicPowerEnvCfg,
    TeeBattingMimicPowerPlayEnvCfg,
    TeeBattingMultiEnvCfg,
    TeeBattingMultiPlayEnvCfg,
    TeeBattingHumanEnvCfg,
    TeeBattingHumanPlayEnvCfg,
    TeeBattingPerceptionEnvCfg,
    TeeBattingPerceptionPlayEnvCfg,
)
from aibaseball.tasks.tee_batting.pitched_env_cfg import (  # noqa: E402
    PitchedBattingEnvCfg,
    PitchedBattingHuman3EnvCfg,
    PitchedBattingHumanPlayEnvCfg,
    PitchedBattingPlayEnvCfg,
)
from aibaseball.tasks.tee_batting.tee_batting_env_cfg import TeeBattingEnvCfg, TeeBattingPlayEnvCfg  # noqa: E402

TASKS = {
    "pitched": ("AIB-PitchedBatting-Play-v0", PitchedBattingPlayEnvCfg, PitchedBattingEnvCfg, TeeBattingMimicPPORunnerCfg),
    "human_pitched": ("AIB-PitchedBatting-Human-Play-v0", PitchedBattingHumanPlayEnvCfg, PitchedBattingHuman3EnvCfg,
                      TeeBattingMimicPPORunnerCfg),
    "human": ("AIB-TeeBatting-Human-Play-v0", TeeBattingHumanPlayEnvCfg, TeeBattingHumanEnvCfg, TeeBattingMimicPPORunnerCfg),
    "perception": ("AIB-TeeBatting-Perception-Play-v0", TeeBattingPerceptionPlayEnvCfg, TeeBattingPerceptionEnvCfg,
                   TeeBattingMimicPPORunnerCfg),
    "multi": ("AIB-TeeBatting-Multi-Play-v0", TeeBattingMultiPlayEnvCfg, TeeBattingMultiEnvCfg, TeeBattingMimicPPORunnerCfg),
    "launch": ("AIB-TeeBatting-MimicLaunch-Play-v0", TeeBattingMimicLaunchPlayEnvCfg, TeeBattingMimicLaunchEnvCfg,
               TeeBattingMimicPPORunnerCfg),
    "power": ("AIB-TeeBatting-MimicPower-Play-v0", TeeBattingMimicPowerPlayEnvCfg, TeeBattingMimicPowerEnvCfg,
              TeeBattingMimicPPORunnerCfg),
    "mimic": ("AIB-TeeBatting-Mimic-Play-v0", TeeBattingMimicPlayEnvCfg, TeeBattingMimicEnvCfg, TeeBattingMimicPPORunnerCfg),
    "rl": ("AIB-TeeBatting-Play-v0", TeeBattingPlayEnvCfg, TeeBattingEnvCfg, TeeBattingPPORunnerCfg),
}
TASK_ID, PLAY_CFG, TRAIN_CFG, AGENT_CFG = TASKS[args.task]

MPH = 0.44704


def newest_checkpoint() -> str:
    exp = AGENT_CFG().experiment_name
    files = glob.glob(os.path.join(ROOT, "logs", "rsl_rl", exp, "*", "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"no checkpoints under logs/rsl_rl/{exp}")
    return max(files, key=os.path.getmtime)


def load_policy(env, ckpt):
    agent_cfg = AGENT_CFG()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    return runner.get_inference_policy(device=env.unwrapped.device)


_last_frame = None


def grab(base) -> np.ndarray:
    """Render a frame; the RTX annotator occasionally returns an empty (black) buffer, so retry / reuse."""
    global _last_frame
    for _ in range(4):
        img = base.render()
        if img is not None and img.size and img[..., :3].max() > 0:
            _last_frame = img
            return img
    return _last_frame if _last_frame is not None else img


def overlay(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], big: str | None = None) -> np.ndarray:
    img = np.ascontiguousarray(img[..., :3])
    y = 40
    for text, color in lines:
        cv2.putText(img, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        y += 36
    if big:
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)[0]
        org = ((img.shape[1] - size[0]) // 2, img.shape[0] - 60)
        cv2.putText(img, big, org, cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(img, big, org, cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 220, 0), 3, cv2.LINE_AA)
    return img


def run_video(ckpt: str):
    cfg = PLAY_CFG()
    cfg.scene.num_envs = 1
    cfg.terminate_on_land = False
    cfg.episode_length_s = 30.0
    cfg.viewer.resolution = (1600, 900)  # sharper video (user 2026-10-26): larger frame + DLAA instead of the default DLSS
    if hasattr(cfg.sim, "render"):
        cfg.sim.render.antialiasing_mode = "DLAA"
    env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env)
    policy = load_policy(wrapped, ckpt)
    base = env.unwrapped
    sim = base.sim
    target = cfg.target_distance
    title = ("AIB-1 humanoid  |  tee batting  |  Isaac Sim 4.5 PhysX + drag/Magnus ball model", (255, 255, 255))
    results = []
    best = None  # (score, frames, rec): keep the longest swing, successful ones first

    for swing in range(args.swings):
        frames = []
        with torch.inference_mode():
            obs, _ = wrapped.reset()
        n_logged = len(base.play_log)
        post_hit = 0
        landed = False
        step = 0
        while not landed and step < 2000:
            with torch.inference_mode():
                obs, _, _, _ = wrapped.step(policy(obs))
                obs = obs.clone()
            step += 1
            hit = bool(base.hit[0])
            ball = base.ball_pos[0].cpu().numpy() if hit else base.tee_pos[0].cpu().numpy()
            if not hit or post_hit < 40:
                # slow-motion swing: every 100 Hz step becomes a frame (30 fps video -> 3.3x slow)
                tee = base.tee_pos[0].cpu().numpy()
                sim.set_camera_view(eye=(tee + [0.15, -2.9, 0.65]).tolist(), target=(tee + [-0.3, 0.6, 0.3]).tolist())
                post_hit += int(hit)
                keep, phase = True, "swing (3.3x slow motion)"
            else:
                # ball-tracking camera, roughly real time
                eye = ball + np.array([-14.0, -16.0, 5.0])
                eye[2] = max(eye[2], 2.0)
                sim.set_camera_view(eye=eye.tolist(), target=ball.tolist())
                keep, phase = step % 3 == 0, "ball flight (real time)"
            if not keep:
                continue
            img = grab(base)
            lines = [title, (phase, (200, 200, 200))]
            if hit:
                v = base.launch_vel[0]
                ev = float(v.norm())
                la = float(torch.rad2deg(torch.atan2(v[2], v[:2].norm())))
                rpm = float(base.launch_omega[0].norm()) * 60 / (2 * np.pi)
                d = float(np.linalg.norm(ball[:2] - base.launch_pos[0, :2].cpu().numpy()))
                lines += [(f"exit velocity {ev:5.1f} m/s ({ev / MPH:5.1f} mph)   launch {la:5.1f} deg   spin {rpm:5.0f} rpm",
                           (120, 220, 255)),
                          (f"distance {d:6.1f} m   height {ball[2]:5.1f} m   target {target} m", (255, 255, 255))]
            frames.append(overlay(img, lines))
            landed = len(base.play_log) > n_logged

        if not landed:
            print("[showcase] no landing recorded for this swing", flush=True)
            continue
        rec = base.play_log[-1]
        results.append(rec)
        ok = rec["carry_m"] >= target and abs(rec["spray_deg"]) <= cfg.fair_half_angle_deg
        verdict = f"CARRY {rec['carry_m']:.1f} m  -  {'TARGET REACHED' if ok else 'target ' + str(target) + ' m'}"
        land = base.ball_pos[0].cpu().numpy()
        # overview: hold ~3 s while the camera rises over the field
        for k in range(90):
            a = k / 89
            mid_x = 0.5 * max(land[0], target)
            eye = np.array([mid_x - 40.0, -60.0, 15.0]) * (1 - a) + np.array([mid_x - 60.0, -120.0, 85.0]) * a
            sim.set_camera_view(eye=eye.tolist(), target=[mid_x, 0.0, 0.0])
            img = grab(base)
            frames.append(overlay(img, [title, (f"landing spot (red) vs 150.3 m arc (yellow)", (200, 200, 200)),
                                        (f"exit velocity {rec['exit_velo_mph']:.1f} mph   launch {rec['launch_deg']:.1f} deg"
                                         f"   spray {rec['spray_deg']:+.1f} deg", (120, 220, 255))], big=verdict))
        score = rec["carry_m"] + (1000.0 if ok else 0.0)
        print(f"[showcase] swing {swing + 1}: carry {rec['carry_m']:.1f} m {'SUCCESS' if ok else ''}", flush=True)
        if best is None or score > best[0]:
            best = (score, frames, rec)

    if best is None:
        raise RuntimeError("no swing landed")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, best[1], fps=30, quality=None, macro_block_size=1, ffmpeg_params=["-crf", "16", "-profile:v", "high", "-preset", "slow"])
    print(f"[showcase] wrote {args.out} ({len(best[1])} frames), best carry {best[2]['carry_m']:.1f} m", flush=True)
    for r in results:
        print("[showcase]", json.dumps(r), flush=True)


def run_eval(ckpt: str):
    cfg = PLAY_CFG()
    # same randomization as training
    train = TRAIN_CFG()
    cfg.tee_xy_noise, cfg.tee_height_noise, cfg.joint_pos_noise = train.tee_xy_noise, train.tee_height_noise, train.joint_pos_noise
    for k in ("ball_obs_bias_std", "ball_obs_noise_std", "ball_obs_delay_s"):
        setattr(cfg, k, getattr(train, k))
    if hasattr(cfg, "tee_noise_ramp_steps"):
        cfg.tee_noise_ramp_steps = 0  # full randomisation range in evaluation
    if hasattr(cfg, "pitch_ramp_steps"):  # pitched tasks: training pitch speeds / timing noise, full range
        for k in ("pitch_speed_range_start", "pitch_speed_range_final", "timing_noise_s"):
            setattr(cfg, k, getattr(train, k))
        cfg.pitch_ramp_steps = 0
        if args.pitch_kmh:
            cfg.pitch_speed_range_start = cfg.pitch_speed_range_final = tuple(args.pitch_kmh)
        if args.timing_noise is not None:
            cfg.timing_noise_s = args.timing_noise
        if args.timing_bias is not None:
            cfg.timing_bias_s = args.timing_bias
        if args.tee_offset_z is not None:
            cfg.tee_offset_z = args.tee_offset_z
        if args.aim_offset_z is not None:
            cfg.aim_offset_z = args.aim_offset_z
        if args.phase_horizon is not None:
            cfg.phase_horizon_s = args.phase_horizon
        if args.pitch_csv:
            cfg.pitch_table_csv = args.pitch_csv
        if args.release_csv:
            cfg.pitch_release_csv = args.release_csv
        cfg.clamp_ref_phase = args.clamp_ref_phase
        cfg.zero_ff_after_ref_end = args.zero_ff_after_ref_end
        if args.settle_tail is not None:
            cfg.settle_tail_s = args.settle_tail
        if args.post_ref_blend is not None:
            cfg.post_ref_blend_s = args.post_ref_blend
        if args.post_ref_pose:
            cfg.post_ref_pose = args.post_ref_pose
        cfg.post_ref_keep_residual = args.post_ref_keep_residual
        if args.aim_offset_y is not None:
            cfg.aim_offset_y = args.aim_offset_y
    cfg.scene.num_envs = args.eval_envs
    env = gym.make(TASK_ID, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env)
    policy = load_policy(wrapped, ckpt)
    base = env.unwrapped
    obs, _ = wrapped.reset()
    for _ in range(int(cfg.episode_length_s / base.step_dt) + 5):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs))
        if len(base.play_log) >= args.eval_envs:
            break
    logs = base.play_log[: args.eval_envs]
    carry = np.array([r["carry_m"] for r in logs]) if logs else np.zeros(0)
    fair = np.array([abs(r["spray_deg"]) <= cfg.fair_half_angle_deg for r in logs]) if logs else np.zeros(0, bool)
    stats = dict(
        checkpoint=ckpt, swings=args.eval_envs, landed=len(logs),
        carry_mean=float(carry.mean()) if len(carry) else 0.0,
        carry_median=float(np.median(carry)) if len(carry) else 0.0,
        carry_max=float(carry.max()) if len(carry) else 0.0,
        success_rate=float(((carry >= cfg.target_distance) & fair).sum() / args.eval_envs),
        exit_velo_mph_mean=float(np.mean([r["exit_velo_mph"] for r in logs])) if logs else 0.0,
        launch_deg_mean=float(np.mean([r["launch_deg"] for r in logs])) if logs else 0.0,
        spin_rpm_mean=float(np.mean([r["spin_rpm"] for r in logs])) if logs else 0.0,
    )
    if logs:  # batted-ball quality over the balls that came off the bat
        ev = np.array([r["exit_velo_mps"] for r in logs])
        stats.update(exit_velo_mps_median=float(np.median(ev)), exit_velo_ge28=float((ev >= 28.0).mean()),
                     fair_frac=float(fair.mean()), fair_carry_median=float(np.median(carry[fair])) if fair.any() else 0.0,
                     launch_deg_quartiles=[float(x) for x in np.percentile([r["launch_deg"] for r in logs], [25, 50, 75])],
                     spray_deg_quartiles=[float(x) for x in np.percentile([r["spray_deg"] for r in logs], [25, 50, 75])])
    os.makedirs(os.path.join(ROOT, "videos"), exist_ok=True)
    with open(os.path.join(ROOT, "videos", "showcase_stats.json"), "w", encoding="utf-8") as f:
        json.dump(dict(stats=stats, swings=logs), f, indent=2)
    print("[eval]", json.dumps(stats, indent=2), flush=True)


def run_pitched_video(ckpt: str):
    """Every pitch -> one video: catcher view of the incoming pitch, slow-motion swing, flight / MISS."""
    cfg = PLAY_CFG()
    cfg.scene.num_envs = 1
    cfg.terminate_on_land = False
    cfg.episode_length_s = 30.0
    cfg.viewer.resolution = (1600, 900)  # sharper video (user 2026-10-26): larger frame + DLAA instead of the default DLSS
    if hasattr(cfg.sim, "render"):
        cfg.sim.render.antialiasing_mode = "DLAA"
    if args.pitch_kmh:
        cfg.pitch_speed_range_start = tuple(args.pitch_kmh)
        cfg.pitch_speed_range_final = tuple(args.pitch_kmh)
    if args.timing_noise is not None:
        cfg.timing_noise_s = args.timing_noise
    if args.timing_bias is not None:
        cfg.timing_bias_s = args.timing_bias
    if args.aim_offset_z is not None:
        cfg.aim_offset_z = args.aim_offset_z
    if args.phase_horizon is not None:
        cfg.phase_horizon_s = args.phase_horizon
    if args.pitch_csv:
        cfg.pitch_table_csv = args.pitch_csv
    if args.release_csv:
        cfg.pitch_release_csv = args.release_csv
    cfg.clamp_ref_phase = args.clamp_ref_phase
    cfg.zero_ff_after_ref_end = args.zero_ff_after_ref_end
    if args.settle_tail is not None:
        cfg.settle_tail_s = args.settle_tail
    if args.post_ref_blend is not None:
        cfg.post_ref_blend_s = args.post_ref_blend
    if args.post_ref_pose:
        cfg.post_ref_pose = args.post_ref_pose
    cfg.post_ref_keep_residual = args.post_ref_keep_residual
    if args.aim_offset_y is not None:
        cfg.aim_offset_y = args.aim_offset_y
    env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env)
    policy = load_policy(wrapped, ckpt)
    base, sim = env.unwrapped, env.unwrapped.sim
    title = ("AIB-1 vs pitched fastballs  |  perception: 2 cm bias, 1 cm noise, 100 ms delay  |  drag/Magnus",
             (255, 255, 255))
    frames, results = [], []
    for k in range(args.swings):
        with torch.inference_mode():
            obs, _ = wrapped.reset()
        n_logged = len(base.play_log)
        o = base.scene.env_origins[0].cpu().numpy()
        kmh = float(base.pitch_kmh[0])
        outcome, step, post = None, 0, 0
        while step < 1500:
            with torch.inference_mode():
                obs, _, dones, _ = wrapped.step(policy(obs))
                obs = obs.clone()
            step += 1
            if bool(dones[0]) and not bool(base.hit[0]):  # the pitch got past the batter
                outcome = "MISS"
                break
            hit = bool(base.hit[0])
            pitch = base.pitch_pos[0].cpu().numpy() - o
            tee = base.nominal_target[0].cpu().numpy() - o
            if args.fixed_cam:  # catcher's position behind the plate, looking at the pitcher
                sim.set_camera_view(eye=(o + [tee[0] - 3.6, -0.7, 1.5]).tolist(), target=(o + [tee[0] + 17.0, 0.4, 0.9]).tolist())
                post += int(hit)
                keep = (not hit or post < 40) or step % 3 == 0
                phase = "catcher view" + (" (3.3x slow motion)" if (not hit or post < 40) else " (real time)")
            elif not hit and pitch[0] > tee[0] + 3.0:  # pitch on its way: from behind the catcher
                sim.set_camera_view(eye=(o + [tee[0] - 3.5, tee[1] - 1.6, 1.7]).tolist(),
                                    target=(o + [tee[0] + 12.0, tee[1] + 0.3, 1.1]).tolist())
                keep, phase = step % 2 == 0, "pitch incoming (catcher view)"
            elif not hit or post < 40:  # the swing, side view, slow motion
                sim.set_camera_view(eye=(o + tee + [0.15, -2.9, 0.65]).tolist(), target=(o + tee + [-0.3, 0.6, 0.3]).tolist())
                post += int(hit)
                keep, phase = True, "swing (3.3x slow motion)"
            else:  # batted ball flight
                ball = base.ball_pos[0].cpu().numpy() - o
                eye = ball + np.array([-14.0, -16.0, 5.0])
                eye[2] = max(eye[2], 2.0)
                sim.set_camera_view(eye=(o + eye).tolist(), target=(o + ball).tolist())
                keep, phase = step % 3 == 0, "batted ball (real time)"
            if keep:
                lines = [title, (f"pitch {k + 1}/{args.swings}: {kmh:.0f} km/h fastball   -   {phase}", (200, 200, 200))]
                if hit:
                    v = base.launch_vel[0]
                    ev = float(v.norm())
                    la = float(torch.rad2deg(torch.atan2(v[2], v[:2].norm())))
                    lines.append((f"exit velocity {ev / MPH:5.1f} mph   launch {la:5.1f} deg", (120, 220, 255)))
                frames.append(overlay(grab(base), lines))
            if len(base.play_log) > n_logged:
                rec = base.play_log[-1]
                fair = abs(rec["spray_deg"]) <= cfg.fair_half_angle_deg
                outcome = f"{'FAIR' if fair else 'FOUL'}  {rec['carry_m']:.1f} m"
                results.append(rec)
                break
        outcome = outcome or "NO RESULT"
        print(f"[pitched] pitch {k + 1}: {kmh:.0f} km/h -> {outcome}", flush=True)
        for _ in range(45):  # 1.5 s result card
            frames.append(overlay(grab(base), [title, (f"pitch {k + 1}/{args.swings}: {kmh:.0f} km/h", (200, 200, 200))],
                                  big=outcome))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=30, quality=None, macro_block_size=1, ffmpeg_params=["-crf", "16", "-profile:v", "high", "-preset", "slow"])
    print(f"[pitched] wrote {args.out} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    try:
        ckpt = args.checkpoint or newest_checkpoint()
        print("[showcase] checkpoint:", ckpt, flush=True)
        if args.eval_envs > 0:
            run_eval(ckpt)
        elif args.task in ("pitched", "human_pitched"):
            run_pitched_video(ckpt)
        else:
            run_video(ckpt)
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
