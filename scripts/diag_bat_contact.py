"""Why does the pitched-ball batter miss? Per-pitch bat-ball geometry and timing, no video.

For every pitch (one episode per env, training randomisation, full pitch speed range) it records, at each physics
step with 8 interpolated sub-steps (the bat and ball move ~15 cm per physics step):
  * gap_m      closest ball-to-bat-surface gap before contact (<= 0 means contact)
  * x_on_bat   where along the bat (from the knob) the closest point was; sweet spot for reference
  * dz_m       ball centre minus closest bat-axis point, vertical (+ = ball above the bat)
  * dperp_m    the same offset, horizontal (along the bat's swing direction: + = ball ahead of the bat)
  * timing_s   time the bat is square to the pitch line (|axis_x| smallest while the sweet spot is fast)
               minus time the ball crosses the contact plane (+ = late)
and summarises misses vs hits: which error dominates (timing, height, reach).

    python scripts/diag_bat_contact.py --headless --checkpoint <model.pt> --envs 256
Output: one summary line per group + logs/diag_bat_contact_<tag>.json
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=256)
parser.add_argument("--tag", type=str, default="human3")
parser.add_argument("--timing_bias", type=float, default=None, help="override cfg.timing_bias_s (s, + = pitch later)")
parser.add_argument("--tempo_zero_after_release", action="store_true", help="cfg.tempo_zero_after_release: no tempo correction once the ball is in the air")
parser.add_argument("--post_ref_blend", type=float, default=None, help="cfg.post_ref_blend_s: after the clip, blend to a balanced pose over this many s")
parser.add_argument("--post_ref_pose", type=str, default=None, help="cfg.post_ref_pose: contact | start")
parser.add_argument("--mix_frac", type=float, default=None, help="cfg.pitch_mix_frac")
parser.add_argument("--mix_speed", type=float, nargs=2, default=None, help="cfg.pitch_mix_speed_kmh")
parser.add_argument("--mix_release_csv", type=str, default=None, help="training mix: cfg.pitch_mix_release_csv (half the pitches shaped like these releases)")
parser.add_argument("--release_csv", type=str, default=None, help="pitches from the RL pitcher's eval_stage --release_csv (own release state each)")
parser.add_argument("--pitch_csv", type=str, default=None, help="pitches from the RL pitcher's eval_stage --pitch_csv table (speed_mps, plate_y, plate_z, crossed)")
parser.add_argument("--phase_horizon", type=float, default=None, help="cfg.phase_horizon_s (s): phase input scale, e.g. the training episode 2.6")
parser.add_argument("--aim_offset_z", type=float, default=None, help="override cfg.aim_offset_z (m, < 0 = aim under the ball)")
parser.add_argument("--timing_noise", type=float, default=None, help="override cfg.timing_noise_s (+- arrival offset, s)")
parser.add_argument("--seed", type=int, default=None, help="env seed (repeat measurements with different pitches)")
parser.add_argument("--clamp_ref_phase", action="store_true", help="cfg.clamp_ref_phase: reference-time input held at 1 past the clip")
parser.add_argument("--zero_ff_after_ref_end", action="store_true", help="cfg.zero_ff_after_ref_end: no velocity feed-forward on the held pose")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # diagnostics only: a GUI window also keeps the app alive after the run
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.tee_batting.agents.rsl_rl_ppo_cfg import TeeBattingMimicPPORunnerCfg  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env_cfg import (  # noqa: E402
    PitchedBattingHuman3EnvCfg,
    PitchedBattingHumanPlayEnvCfg,
)

N_SUB = 8
FAST_SS = 10.0  # m/s: sweet-spot speed that counts as "swinging" for the square-bat time


def main():
    cfg = PitchedBattingHumanPlayEnvCfg()
    train = PitchedBattingHuman3EnvCfg()
    for k in ("tee_xy_noise", "tee_height_noise", "joint_pos_noise", "ball_obs_bias_std", "ball_obs_noise_std",
              "ball_obs_delay_s", "pitch_speed_range_start", "pitch_speed_range_final", "timing_noise_s",
              "w_miss", "w_tempo"):
        setattr(cfg, k, getattr(train, k))
    cfg.pitch_ramp_steps = 0  # full speed range and timing noise
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
    if args.mix_release_csv:
        cfg.pitch_mix_release_csv = args.mix_release_csv
    if args.mix_frac is not None:
        cfg.pitch_mix_frac = args.mix_frac
    if args.mix_speed is not None:
        cfg.pitch_mix_speed_kmh = tuple(args.mix_speed)
    cfg.tempo_zero_after_release = args.tempo_zero_after_release
    if args.post_ref_blend is not None:
        cfg.post_ref_blend_s = args.post_ref_blend
    if args.post_ref_pose:
        cfg.post_ref_pose = args.post_ref_pose
    if args.timing_noise is not None:
        cfg.timing_noise_s = args.timing_noise
    if hasattr(cfg, "tee_noise_ramp_steps"):
        cfg.tee_noise_ramp_steps = 0
    cfg.scene.num_envs = args.envs
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.clamp_ref_phase = args.clamp_ref_phase
    cfg.zero_ff_after_ref_end = args.zero_ff_after_ref_end
    env = gym.make("AIB-PitchedBatting-Human-Play-v0", cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env)
    agent = TeeBattingMimicPPORunnerCfg()
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    n, dev, geom, r_ball = base.num_envs, base.device, base.bat_geom, base.phys.ball.radius

    inf = torch.full((n,), float("inf"), device=dev)
    rec = dict(gap=inf.clone(), x_on_bat=torch.zeros(n, device=dev), dz=torch.zeros(n, device=dev),
               dperp=torch.zeros(n, device=dev), t_gap=torch.zeros(n, device=dev),
               t_plane=torch.full((n,), float("nan"), device=dev), sq_ax=torch.ones(n, device=dev),
               t_square=torch.full((n,), float("nan"), device=dev), ss_speed_max=torch.zeros(n, device=dev),
               hit=torch.zeros(n, dtype=torch.bool, device=dev), fell=torch.zeros(n, dtype=torch.bool, device=dev),
               t_fall=torch.full((n,), float("nan"), device=dev), t_hit=torch.full((n,), float("nan"), device=dev),
               tempo_sum=torch.zeros(n, device=dev), tempo_n=torch.zeros(n, device=dev),
               ref_err=torch.full((n,), float("nan"), device=dev))
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    prev = {}
    orig_apply = base._apply_action

    def hooked_apply():
        knob, axis, cm, v_cm, w, _ = base._bat_state()
        ball = base.pitch_pos.clone()
        t = base.phys_t.clone()
        live = ~finished & ~base.hit
        if prev:
            for f in torch.linspace(0.0, 1.0, N_SUB + 1, device=dev)[1:]:
                k = prev["knob"] + (knob - prev["knob"]) * f
                a = prev["axis"] + (axis - prev["axis"]) * f
                a = a / a.norm(dim=-1, keepdim=True)
                b = prev["ball"] + (ball - prev["ball"]) * f
                x = ((b - k) * a).sum(-1).clamp(0.0, geom.length)
                p = k + a * x.unsqueeze(-1)
                off = b - p
                gap = off.norm(dim=-1) - geom.radius_at(x) - r_ball
                better = live & (gap < rec["gap"])
                # swing direction: horizontal, perpendicular to the bat axis, along the sweet spot's velocity
                _, v_ss = base._sweet_spot(knob, axis, cm, v_cm, w)
                dperp_dir = v_ss - (v_ss * a).sum(-1, keepdim=True) * a
                dperp_dir[:, 2] = 0.0
                dperp_dir = dperp_dir / dperp_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                rec["gap"] = torch.where(better, gap, rec["gap"])
                rec["x_on_bat"] = torch.where(better, x, rec["x_on_bat"])
                rec["dz"] = torch.where(better, off[:, 2], rec["dz"])
                rec["dperp"] = torch.where(better, (off * dperp_dir).sum(-1), rec["dperp"])
                rec["t_gap"] = torch.where(better, prev["t"] + (t - prev["t"]) * f, rec["t_gap"])
            # ball crosses the contact plane (moving -x)
            xr = base.contact_x_ref
            cross = live & (prev["ball"][:, 0] >= xr) & (ball[:, 0] < xr) & torch.isnan(rec["t_plane"])
            frac = (prev["ball"][:, 0] - xr) / (prev["ball"][:, 0] - ball[:, 0]).clamp_min(1e-6)
            rec["t_plane"] = torch.where(cross, prev["t"] + (t - prev["t"]) * frac, rec["t_plane"])
            # reference clock when the ball crosses: + = the swing reference is already past its contact frame (early)
            rec["ref_err"] = torch.where(cross, base.ref_t - base.ref.contact_time[base.clip], rec["ref_err"])
            # tempo action while the ball is in flight, before contact
            fl = live & (t >= base.release_t)
            rec["tempo_sum"] += torch.where(fl, base.actions[:, -1].clamp(-1.0, 1.0), torch.zeros_like(t))
            rec["tempo_n"] += fl.float()
        _, v_ss = base._sweet_spot(knob, axis, cm, v_cm, w)
        sp = v_ss.norm(dim=-1)
        rec["ss_speed_max"] = torch.where(live, torch.maximum(rec["ss_speed_max"], sp), rec["ss_speed_max"])
        sq = live & (sp > FAST_SS) & (axis[:, 0].abs() < rec["sq_ax"])
        rec["sq_ax"] = torch.where(sq, axis[:, 0].abs(), rec["sq_ax"])
        rec["t_square"] = torch.where(sq, t, rec["t_square"])
        prev.update(knob=knob.clone(), axis=axis.clone(), ball=ball, t=t)
        orig_apply()
        rec["hit"] |= base.hit & ~finished

    base._apply_action = hooked_apply
    obs, _ = wrapped.reset()
    # the pitch of each env's (first) episode: read now - finished envs are reset to a new pitch before the end
    kmh0, plate0 = base.pitch_kmh.clone(), base.pitch_plate.clone()
    prev.clear()
    for _ in range(int(cfg.episode_length_s / base.step_dt) + 5):
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
        t_now = (base.episode_length_buf.float() * base.step_dt)
        new_hit = base.hit & ~finished & torch.isnan(rec["t_hit"])
        rec["t_hit"] = torch.where(new_hit, t_now, rec["t_hit"])
        new_fall = base.fallen & ~finished & ~rec["fell"]
        rec["t_fall"] = torch.where(new_fall, t_now, rec["t_fall"])  # time into the episode of the first fall
        rec["fell"] |= base.fallen & ~finished
        finished |= dones.bool()
        if finished.all():
            break

    r = {k: v.cpu().numpy() for k, v in rec.items()}
    r["timing"] = r["t_square"] - r["t_plane"]
    r["kmh"] = kmh0.cpu().numpy()
    r["plate_y"], r["plate_z"] = (v.cpu().numpy() for v in plate0.unbind(-1))
    ss = float(geom.sweet_spot_x)

    def summary(name, m):
        if m.sum() == 0:
            print(f"[bat] {name}: 0 pitches", flush=True)
            return {}
        q = lambda a: np.nanpercentile(a[m], [25, 50, 75]).round(3).tolist()  # noqa: E731
        s = dict(n=int(m.sum()), gap_m=q(r["gap"]), dz_m=q(r["dz"]), dperp_m=q(r["dperp"]),
                 x_minus_sweet_m=q(r["x_on_bat"] - ss), timing_s=q(r["timing"]),
                 ss_speed_max=q(r["ss_speed_max"]), fell=int(r["fell"][m].sum()))
        mf = m & r["fell"]
        if mf.any():  # when the falls happen: into the episode, and after contact (hits only)
            s["fall_t_s"] = np.nanpercentile(r["t_fall"][mf], [10, 25, 50, 75, 90]).round(2).tolist()
            s["fall_after_hit_s"] = np.nanpercentile((r["t_fall"] - r["t_hit"])[mf], [10, 25, 50, 75, 90]).round(2).tolist()
        print(f"[bat] {name}: " + json.dumps(s), flush=True)
        return s

    done = finished.cpu().numpy()
    out = dict(checkpoint=args.checkpoint, sweet_spot_x=ss, pitches=int(done.sum()), hit_rate=float(r["hit"][done].mean()))
    print(f"[bat] pitches {out['pitches']}  hit rate {out['hit_rate']:.3f}  (quartiles 25/50/75)", flush=True)
    out["hits"] = summary("hits", done & r["hit"])
    miss = done & ~r["hit"]
    out["misses"] = summary("misses", miss)
    # dominant error per miss: the largest of |timing| * swing speed, |dz|, |dperp| (all in metres)
    late = np.abs(r["timing"]) * np.nan_to_num(r["ss_speed_max"])
    errs = np.stack([np.nan_to_num(late, nan=9.0), np.abs(r["dz"]), np.abs(r["dperp"])], -1)
    dom = np.argmax(errs, -1)
    out["miss_dominant"] = {name: int((miss & (dom == i)).sum()) for i, name in enumerate(("timing", "height", "reach"))}
    out["miss_late"] = int((miss & (r["timing"] > 0)).sum())
    out["miss_ball_above"] = int((miss & (r["dz"] > 0)).sum())
    print("[bat] miss dominant error: " + json.dumps(out["miss_dominant"])
          + f"  late {out['miss_late']}/{int(miss.sum())}  ball above bat {out['miss_ball_above']}/{int(miss.sum())}",
          flush=True)
    # by pitch speed (mixed libraries / the RL pitcher's slow pitches): hit rate, early / late misses, tempo, reference clock
    tempo = r["tempo_sum"] / np.maximum(r["tempo_n"], 1)
    out["by_speed"] = {}
    zone = (np.abs(r["plate_y"]) <= 0.2525) & (r["plate_z"] >= 0.50) & (r["plate_z"] <= 1.10)  # StrikeZone
    groups = [("slow < 80 km/h", done & (r["kmh"] < 80)), ("fast >= 80 km/h", done & (r["kmh"] >= 80))]
    if args.release_csv or args.pitch_csv or args.mix_release_csv:  # pitches with a real plate location
        groups += [("slow in zone", done & (r["kmh"] < 80) & zone), ("slow out of zone", done & (r["kmh"] < 80) & ~zone)]
    for name, m in groups:
        if not m.any():
            continue
        mm = m & ~r["hit"]
        g = dict(n=int(m.sum()), hit_rate=round(float(r["hit"][m].mean()), 3), misses=int(mm.sum()),
                 early=int((mm & (r["timing"] < 0)).sum()), late=int((mm & (r["timing"] > 0)).sum()),
                 tempo_q=np.nanpercentile(tempo[m], [25, 50, 75]).round(3).tolist(),
                 ref_err_s_q=np.nanpercentile(r["ref_err"][m], [25, 50, 75]).round(3).tolist(),
                 kmh_q=np.nanpercentile(r["kmh"][m], [10, 50, 90]).round(1).tolist())
        out["by_speed"][name] = g
        print(f"[bat] {name}: " + json.dumps(g), flush=True)
    path = os.path.join(ROOT, "logs", f"diag_bat_contact_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(summary=out, per_pitch={k: v.tolist() for k, v in r.items()}), f)
    print(f"[bat] wrote {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
