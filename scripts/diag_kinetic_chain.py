"""Kinetic-chain sequencing ("velocity principle"): when does each segment reach its peak speed?

Pelvis rotation, trunk rotation (about the vertical), shoulder internal rotation, elbow extension and hand speed;
for the reference (athlete) and for the robot (median over envs), with times relative to each one's release.

    python scripts/diag_kinetic_chain.py --headless --checkpoint <pitch model.pt> [--envs 16]
"""

import argparse
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
from scipy.signal import savgol_filter  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.robot.humanoid import build_humanoid  # noqa: E402
from aibaseball.tasks.pitching import pitch_env_cfg as C  # noqa: E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402

SEGMENTS = ["pelvis rotation", "trunk rotation", "shoulder int. rotation", "elbow extension", "hand speed"]
UNITS = ["deg/s", "deg/s", "deg/s", "deg/s", "m/s"]


def reference_series(cfg):
    d = np.load(cfg.motion_file, allow_pickle=True)
    t, dt = d["time"], float(d["time"][1] - d["time"][0])
    names = [str(n) for n in d["joint_names"]]
    robot = build_humanoid(with_bat=False, name="aib1_pitcher", arm_masses="human")
    Rr = Rotation.from_quat(np.concatenate([d["root_quat"][:, 1:], d["root_quat"][:, :1]], -1))
    torso_R, hand_p = [], []
    for f in range(len(t)):
        P = robot.fk(dict(zip(names, d["joint_pos"][f])), root_pos=d["root_pos"][f], root_rot=Rr[f].as_matrix())
        torso_R.append(P["torso"][0])
        hand_p.append(P["r_hand"][1])
    Rt = Rotation.from_matrix(np.array(torso_R))
    w_pel = np.zeros((len(t), 3))
    w_tor = np.zeros((len(t), 3))
    w_pel[1:-1] = (Rr[2:] * Rr[:-2].inv()).as_rotvec() / (2 * dt)
    w_tor[1:-1] = (Rt[2:] * Rt[:-2].inv()).as_rotvec() / (2 * dt)
    hand_v = np.linalg.norm(np.gradient(np.array(hand_p), dt, axis=0), axis=1)
    qd = d["joint_vel"]
    s = np.stack([np.abs(w_pel[:, 2]) * 57.3, np.abs(w_tor[:, 2]) * 57.3, np.abs(qd[:, names.index("r_shoulder_yaw")]) * 57.3,
                  np.abs(qd[:, names.index("r_elbow")]) * 57.3, hand_v], 1)
    return t, savgol_filter(s, 7, 2, axis=0), float(d["contact_time"])


def peaks(t, s, t_rel, lo=-0.45, hi=0.15):
    m = (t >= t_rel + lo) & (t <= t_rel + hi)
    out = []
    for j in range(s.shape[1]):
        i = np.argmax(np.where(m, s[:, j], -1))
        out.append((t[i] - t_rel, s[i, j]))
    return out


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
    sy, el = rob.joint_names.index("r_shoulder_yaw"), rob.joint_names.index("r_elbow")
    tor, hand = rob.find_bodies("torso")[0][0], rob.find_bodies("r_hand")[0][0]
    n = base.num_envs
    T = int(1.5 / base.step_dt)
    S = np.zeros((T, n, 5))
    t_rel = np.full(n, np.nan)
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            obs, _, _, _ = w.step(pol(obs))
            d = rob.data
            S[k] = torch.stack([d.root_ang_vel_w[:, 2].abs() * 57.3, d.body_ang_vel_w[:, tor, 2].abs() * 57.3,
                                d.joint_vel[:, sy].abs() * 57.3, d.joint_vel[:, el].abs() * 57.3,
                                d.body_lin_vel_w[:, hand].norm(dim=-1)], 1).cpu().numpy()
            new = base.released.cpu().numpy() & np.isnan(t_rel)
            t_rel[new] = (k + 1) * base.step_dt
    tt = (np.arange(T) + 1) * base.step_dt
    rt, rs, rrel = reference_series(cfg)
    ref = peaks(rt, rs, rrel)
    rob_pk = [peaks(tt, S[:, i], t_rel[i]) for i in range(n) if not np.isnan(t_rel[i])]
    print(f"[kc] release: athlete {rrel:.3f} s | robot median {np.nanmedian(t_rel):.3f} s ({len(rob_pk)}/{n} released)")
    print("[kc] segment                  | athlete: peak time vs release, peak       | robot (median): peak time, peak")
    for j, nm in enumerate(SEGMENTS):
        rt_ = np.median([p[j][0] for p in rob_pk])
        rv = np.median([p[j][1] for p in rob_pk])
        print(f"[kc] {nm:24s} | {ref[j][0] * 1000:+5.0f} ms  {ref[j][1]:7.0f} {UNITS[j]:5s}           | "
              f"{rt_ * 1000:+5.0f} ms  {rv:7.0f} {UNITS[j]}", flush=True)
    order_ref = np.argsort([p[0] for p in ref])
    order_rob = np.argsort([np.median([p[j][0] for p in rob_pk]) for j in range(5)])
    print("[kc] order athlete:", " -> ".join(SEGMENTS[i] for i in order_ref))
    print("[kc] order robot  :", " -> ".join(SEGMENTS[i] for i in order_rob))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
