"""Pitching mechanics checklist (robot vs athlete) at lead-foot strike and at release.

Hip-shoulder separation, throwing arm in the "launch" position at foot strike (hand above / behind the shoulder),
pelvis already turned to the target, pivot-knee bend and push-off extension, trunk tilt at release.

    python scripts/diag_mechanics.py --headless --checkpoint <pitch model.pt> [--envs 16]
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.mocap.centroidal import mound_z, reference_centroidal  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid  # noqa: E402
from aibaseball.tasks.pitching import pitch_env_cfg as C  # noqa: E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402

YAW = math.radians(-8.0)  # the reference is rotated by ref_yaw_deg in the env


def yaw_of(R):
    """Heading (deg) of a body's forward (x) axis projected on the ground."""
    f = R[..., :, 0]
    return np.degrees(np.arctan2(f[..., 1], f[..., 0]))


def metrics(R_pel, R_tor, p_sh, p_hand, p_el, knee_r, plate_dir):
    """Per-frame mechanics. Positions world (n,3), rotations (n,3,3)."""
    sep = (yaw_of(R_pel) - yaw_of(R_tor) + 180) % 360 - 180  # hip-shoulder separation
    pel_to_target = (yaw_of(R_pel) - plate_dir + 180) % 360 - 180
    hand_above = p_hand[:, 2] - p_sh[:, 2]  # m, hand above the throwing shoulder
    hand_behind = -(p_hand[:, 0] - p_sh[:, 0])  # m, hand behind the shoulder (away from the plate)
    up = R_tor[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))  # trunk tilt from vertical
    lateral = np.degrees(np.arctan2(-up[:, 1], up[:, 2]))  # toward the glove side (+) in the frontal plane
    return dict(sep=sep, pel_to_target=pel_to_target, hand_above=hand_above, hand_behind=hand_behind,
                trunk_tilt=tilt, trunk_lateral=lateral, knee_r=np.degrees(knee_r))


def reference(cfg):
    d = np.load(cfg.motion_file, allow_pickle=True)
    names = [str(n) for n in d["joint_names"]]
    robot = build_humanoid(with_bat=False, name="aib1_pitcher", arm_masses="human")
    Rz = Rotation.from_euler("z", YAW)
    Rr = Rz * Rotation.from_quat(np.concatenate([d["root_quat"][:, 1:], d["root_quat"][:, :1]], -1))
    root = Rz.apply(d["root_pos"])
    out = {k: [] for k in ("Rp", "Rt", "sh", "hand", "el")}
    for f in range(len(d["time"])):
        P = robot.fk(dict(zip(names, d["joint_pos"][f])), root_pos=root[f], root_rot=Rr[f].as_matrix())
        out["Rp"].append(P["pelvis"][0])
        out["Rt"].append(P["torso"][0])
        out["sh"].append(P["r_shoulder_pitch_link"][1])
        out["hand"].append(P["r_hand"][1])
        out["el"].append(P["r_forearm"][1])
    cen = reference_centroidal(cfg.motion_file, lambda x: mound_z(x, cfg.mound_top_z, cfg.mound_height, cfg.mound_slope,
                                                                   cfg.mound_slope_start_x))
    strike = float(d["time"][np.argmax(cen["contact"][:, 0] & (d["time"] > 0.3))])
    m = metrics(np.array(out["Rp"]), np.array(out["Rt"]), np.array(out["sh"]), np.array(out["hand"]),
                np.array(out["el"]), d["joint_pos"][:, names.index("r_knee")], 0.0)
    return d["time"], m, strike, float(d["contact_time"])


def main():
    cfg = C.PitchStandFirstEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.rsi_prob = 0.0
    cfg.recovery_prob = 0.0
    env = gym.make("AIB-PitchElastic-v0", cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():
        orig()
        return torch.zeros_like(base.released), base.episode_length_buf >= base.max_episode_length - 1
    base._get_dones = keep_running
    w = RslRlVecEnvWrapper(env)
    ag = PitchPPORunnerCfg()
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    rob = base.robot
    ids = {k: rob.find_bodies(b)[0][0] for k, b in (("Rp", "pelvis"), ("Rt", "torso"), ("sh", "r_shoulder_pitch_link"),
                                                     ("hand", "r_hand"), ("el", "r_forearm"))}
    kr = rob.joint_names.index("r_knee")
    n = base.num_envs
    T = int(1.5 / base.step_dt)
    rec = {k: np.zeros((T, n, 3, 3) if k in ("Rp", "Rt") else (T, n, 3)) for k in ids}
    knee = np.zeros((T, n))
    strike = np.full(n, np.nan)
    rel = np.full(n, np.nan)
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            obs, _, _, _ = w.step(pol(obs))
            d = rob.data
            o = base.scene.env_origins
            for key, b in ids.items():
                if key in ("Rp", "Rt"):
                    q = d.body_link_quat_w[:, b].cpu().numpy()
                    rec[key][k] = Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], -1)).as_matrix()
                else:
                    rec[key][k] = (d.body_link_pos_w[:, b] - o).cpu().numpy()
            knee[k] = d.joint_pos[:, kr].cpu().numpy()
            t = (k + 1) * base.step_dt
            f = base.feet.data.net_forces_w_history[:, :, base.foot_sensor_ids[0]].norm(dim=-1).max(1).values.cpu().numpy()
            new = (f > cfg.contact_force_n) & np.isnan(strike) & (t > 0.6)
            strike[new] = t
            newr = base.released.cpu().numpy() & np.isnan(rel)
            rel[newr] = t
    tt = (np.arange(T) + 1) * base.step_dt
    rt, rm, rstrike, rrel = reference(cfg)

    def at(times, series, t):
        return float(np.interp(t, times, series))

    rows = [("pelvis heading vs target (deg, 0 = facing the plate)", "pel_to_target"),
            ("hip-shoulder separation (deg)", "sep"),
            ("throwing hand above shoulder (m)", "hand_above"),
            ("throwing hand behind shoulder (m)", "hand_behind"),
            ("trunk tilt from vertical (deg)", "trunk_tilt"),
            ("trunk lateral tilt to glove side (deg)", "trunk_lateral"),
            ("pivot (back) knee flexion (deg)", "knee_r")]
    print(f"[mech] lead-foot strike: athlete {rstrike:.3f} s, robot median {np.nanmedian(strike):.3f} s | "
          f"release: athlete {rrel:.3f} s, robot median {np.nanmedian(rel):.3f} s")
    for label, ev_ref, ev_rob in (("FOOT STRIKE", rstrike, strike), ("RELEASE", rrel, rel)):
        print(f"[mech] --- at {label} ---")
        for name, key in rows:
            a = at(rt, rm[key], ev_ref)
            vals = []
            for i in range(n):
                if np.isnan(ev_rob[i]):
                    continue
                m = metrics(rec["Rp"][:, i], rec["Rt"][:, i], rec["sh"][:, i], rec["hand"][:, i], rec["el"][:, i],
                            knee[:, i], 0.0)
                vals.append(at(tt, m[key], ev_rob[i]))
            print(f"[mech] {name:52s} athlete {a:+7.2f} | robot {np.median(vals):+7.2f}", flush=True)
    # peak separation and when it happens relative to foot strike
    i_pk = np.argmax(np.abs(rm["sep"]) * ((rt > 0.5) & (rt < rrel + 0.05)))
    print(f"[mech] max hip-shoulder separation: athlete {abs(rm['sep'][i_pk]):.0f} deg at {(rt[i_pk] - rstrike) * 1000:+.0f} ms "
          "from foot strike", flush=True)
    seps, tms = [], []
    for i in range(n):
        if np.isnan(strike[i]) or np.isnan(rel[i]):
            continue
        m = metrics(rec["Rp"][:, i], rec["Rt"][:, i], rec["sh"][:, i], rec["hand"][:, i], rec["el"][:, i], knee[:, i], 0.0)
        mask = (tt > 0.5) & (tt < rel[i] + 0.05)
        j = np.argmax(np.abs(m["sep"]) * mask)
        seps.append(abs(m["sep"][j]))
        tms.append(tt[j] - strike[i])
    print(f"[mech] max hip-shoulder separation: robot   {np.median(seps):.0f} deg at {np.median(tms) * 1000:+.0f} ms from foot strike",
          flush=True)
    # pivot leg: deepest bend before the push-off and how far it extends
    m0 = (rt > 0.3) & (rt < rstrike)
    print(f"[mech] pivot knee: athlete deepest {rm['knee_r'][m0].max():.0f} deg, at foot strike {at(rt, rm['knee_r'], rstrike):.0f} deg",
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
