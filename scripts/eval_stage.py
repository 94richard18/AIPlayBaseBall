"""Evaluate a stage-1 step policy (stage_env.StageEnv): N full episodes, falls, standing at the end, key frames to
look at and a side-view video, in one simulator launch.

    python scripts/eval_stage.py --headless --task AIB-Stage1a-v0 (or 1b) --checkpoint <model.pt> [--envs 64]
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
parser.add_argument("--zero", action="store_true", help="what-if: zero actions (PD on the set position only)")
parser.add_argument("--trace", action="store_true", help="print env 0 every 0.1 s (feet, pelvis, trunk)")
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
    if hasattr(cfg, "landed_start_prob"):
        cfg.landed_start_prob = 0.0  # evaluate whole episodes from the set position
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
            act = pol(obs)
            if args.zero:
                act = torch.zeros_like(act)
            obs, _, _, _ = w.step(act)
            f = base.fallen.cpu().numpy()
            if args.trace and k % 40 == 0:
                dd = base.robot.data
                o = base.scene.env_origins[0]
                fz = base.feet.data.net_forces_w_history[0, :, base.foot_sensor_ids].norm(dim=-1).max(0).values
                fp = dd.body_link_pos_w[0, base.foot_ids] - o
                pel = dd.root_link_pos_w[0] - o
                b, sp, tl, hz, _ = base._stand_state()
                qerr = (dd.joint_pos[0, base.body_ids] - base.q_target[0, base.body_ids]).abs()
                say(f"t {(k + 1) * base.step_dt:.2f} feet N {fz[0]:.0f}/{fz[1]:.0f} z {fp[0, 2]:.3f}/{fp[1, 2]:.3f} "
                    f"x {fp[0, 0]:+.2f}/{fp[1, 0]:+.2f} | pelvis {pel[0]:+.2f} {pel[1]:+.2f} {pel[2]:.2f} | tilt {tl[0]:.0f} "
                    f"| com v {sp[0]:.2f} | max q err {qerr.max():.2f} rad ({base.body_names[int(qerr.argmax())]})")
            fell_at[np.isnan(fell_at) & f] = (k + 1) * base.step_dt
        both, speed, tilt, head_z, ok = (x.cpu().numpy() for x in base._stand_state())
        dd = base.robot.data  # movement toward the plate (+x) since the first step: lead foot, back foot, pelvis
        dx = torch.cat([dd.body_link_pos_w[:, base.foot_ids, 0] - base.s_feet0[..., 0],
                        (dd.root_link_pos_w[:, 0] - base.s_pelvis0[:, 0]).unsqueeze(-1)], -1).cpu().numpy()
        lift_stage = cfg.stage == "1b1"
        if lift_stage:
            lifted = (base.s_lift_t >= cfg.lift_hold_min_s).cpu().numpy()
            slid = base.s_slid.cpu().numpy()
            done_lift = base.lift_done().cpu().numpy()
            lift_t = base.s_lift_t.cpu().numpy()
            ok = ok & done_lift
        stride_stage = cfg.stage == "1b"
        if stride_stage:  # success also needs the stride (the robot's lead foot landed >= lead_stride_min_m ahead)
            landed = base.s_landed.cpu().numpy()
            ok = ok & landed
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
    if stride_stage:
        say(f"  stride landed (lead foot >= {cfg.lead_stride_min_m} m ahead): {landed.sum()}/{n}, not fallen without a stride "
            f"{(up & ~landed).sum()}")
    if lift_stage:
        say(f"  leg lift (lead foot >= {cfg.lift_min_m} m up on the back leg alone >= {cfg.lift_hold_min_s} s): "
            f"{lifted.sum()}/{n}, median {np.median(lift_t):.2f} s; back foot moved > {cfg.back_slide_max_m} m: "
            f"{slid.sum()}/{n}; lift done with the feet back home: {done_lift.sum()}/{n}")
    say(f"  toward the plate (+x) since the start (median): lead foot {np.median(dx[:, 0]):+.2f} m, back foot "
        f"{np.median(dx[:, 1]):+.2f} m, pelvis {np.median(dx[:, 2]):+.2f} m (min pelvis {dx[:, 2].min():+.2f} m)")
    if up.any():
        say(f"  not fallen, at the end (median): trunk tilt {np.median(tilt[up]):.1f} deg, COM speed {np.median(speed[up]):.3f} "
            f"m/s, head {np.median(head_z[up]):.2f} m, both feet down {both[up].mean() * 100:.0f}%")
    if not args.stochastic:
        say(f"[goal] {ok.sum()}/{n}")
    if args.video:  # (--zero: the video too - it filmed the loaded policy, a fall the statistics never had)
        film(w, base, (lambda x: torch.zeros(x.shape[0], cfg.action_space, device=x.device)) if args.zero else pol, T)


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
            o = base.scene.env_origins[0].cpu().numpy()  # follow the pelvis along the ground (1b walks off the view)
            pel = base.robot.data.root_link_pos_w[0].cpu().numpy()
            o = o + [pel[0] - o[0] - 0.3, pel[1] - o[1], 0.0]
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
        # StageEnv never runs the release check (_get_dones is its own): measure the ball's distance to the grip
        gap = float((base.ball.data.root_pos_w[0] - base._grip_point()[0]).norm())
        rel = "ball held" if gap < base.cfg.release_distance else f"ball out of the hand at the end ({gap:.2f} m from the grip)"
        say(f"[video] episode {e + 1}: fell at {fell}, {rel}")
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
