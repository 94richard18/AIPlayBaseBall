"""Lead-foot strike at control-step resolution: height above the mound, vertical speed, contact force, ankle torque.

    python scripts/diag_foot_strike.py --headless --checkpoint <pitch model.pt> [--zero] [--t0 0.75 --t1 1.0]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--zero", action="store_true", help="no policy: pure PD tracking of the reference")
parser.add_argument("--t0", type=float, default=0.78)
parser.add_argument("--t1", type=float, default=1.00)
parser.add_argument("--env", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.mocap.centroidal import SOLE  # noqa: E402
from aibaseball.tasks.pitching import pitch_env_cfg as C  # noqa: E402
from aibaseball.tasks.pitching.agents.rsl_rl_ppo_cfg import PitchPPORunnerCfg  # noqa: E402


def main():
    cfg = C.PitchStandFirstEnvCfg()
    cfg.scene.num_envs = 4
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
    c, i = base.cfg, args.env
    lf = base.foot_ids[0]
    ank = [base.robot.joint_names.index(n) for n in ("l_ankle_pitch", "l_ankle_roll", "l_knee")]
    corners = torch.tensor([list(v) for v in SOLE], dtype=torch.float32, device=base.device)
    print("[fs]  t     | sole above mound | foot vz   | contact N (max over substeps) | ankle pitch tau / limit | knee tau | ref sole")
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(int(args.t1 / base.step_dt) + 1):
            act = pol(obs)
            obs, _, _, _ = w.step(torch.zeros_like(act) if args.zero else act)
            t = (k + 1) * base.step_dt
            if t < args.t0:
                continue
            d = base.robot.data
            o = base.scene.env_origins[i]
            q = d.body_link_quat_w[i, lf].expand(4, 4)
            pts = d.body_link_pos_w[i, lf] + quat_apply(q, corners) - o
            x0, top = c.mound_slope_start_x, c.mound_top_z
            g = torch.where(pts[:, 0] < x0, torch.full_like(pts[:, 0], top),
                            (top - (pts[:, 0] - x0) * c.mound_slope).clamp_min(top - c.mound_height))
            gap = float((pts[:, 2] - g).min())
            vz = float(d.body_link_lin_vel_w[i, lf, 2])
            f = float(base.feet.data.net_forces_w_history[i, :, base.foot_sensor_ids[0]].norm(dim=-1).max())
            tau = d.applied_torque[i, ank]
            lim = base.robot.root_physx_view.get_dof_max_forces()[i, ank[0]].item()
            rr = base._ref(base._ref_time())
            ref_z = float(rr["key_pos"][i, 2, 2] - o[2]) - 0.08
            print(f"[fs] {t:.4f} | {gap * 100:+6.1f} cm | {vz:+6.2f} m/s | {f:7.0f} | {tau[0]:+6.0f} / {lim:.0f} | {tau[2]:+6.0f} | "
                  f"ankle link ref z {ref_z + 0.08:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        app.close()
