"""Human25-R stage 0 (planning 14, 10-29): the state bank for the post-contact recovery policy.

The frozen delivery swing (h9b model_12500 at AIB_EFFORT_SCALE 1.5) bats in the standalone PitchedBattingEnv (play cfg,
P3c release table, the joint scene's four clips at real heights, the game's bias 0.055 / aim 0, diag-style timing noise
+-0.03 s for spread). At the hand-off trigger of BatterSide._recover - the end of the first control step with a hit or
the training episode's end (the pitch passed / time out) - every env's state is snapshot once:
root state (env-local position, quaternion, lin / ang vel, world axes), all joint positions / velocities, the PD position
targets applied in that step (the recovery anchor q0), the last swing action, hit label, fallen flag.

    AIB_EFFORT_SCALE=1.5 python scripts/recover_bank.py --headless --checkpoint <12500.pt> --envs 2048 --seed 0 --out logs/recover_bank_s0.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--release_csv", type=str, default="assets/game/p3c_identical_t18_release.csv")
parser.add_argument("--timing_noise", type=float, default=0.03)
parser.add_argument("--max_steps", type=int, default=600)
# Human25-R3 (planning 14, 10-30): the bank from the game's batter = the swing run inside AIB-ArmAim-Play-v0 with the
# scripted aim (the residual action is zero and ignored in scripted mode), as diag_bat_contact --armaim
parser.add_argument("--armaim", action="store_true", help="the swing inside the ArmAim play env (+ the flags below)")
parser.add_argument("--aim_scripted", type=str, default=None, help="--armaim: scripted-aim gain JSON")
parser.add_argument("--aim_bound", type=float, default=0.4)
parser.add_argument("--aim_window", type=float, default=0.60)
parser.add_argument("--aim_obs_comp", action="store_true")
parser.add_argument("--aim_obs_comp_full", action="store_true")
parser.add_argument("--phase_horizon", type=float, default=None, help="cfg.phase_horizon_s (s): the swing's phase input scale, training 2.6")
parser.add_argument("--percept_filter", action="store_true", help="cfg.percept_filter: ballistic LSQ crossing prediction")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.one_outs.batter_side import JOINT_CLIPS  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env_cfg import PitchedBattingHumanPlayEnvCfg  # noqa: E402


def main():
    assert os.environ.get("AIB_EFFORT_SCALE") == "1.5", "the delivery batter runs at AIB_EFFORT_SCALE 1.5"
    if args.armaim:
        from aibaseball.tasks.tee_batting.pitched_env_cfg import ArmAimPlayEnvCfg

        cfg = ArmAimPlayEnvCfg()
        cfg.swing_checkpoint = os.path.abspath(args.checkpoint)
        cfg.aim_scripted = os.path.abspath(args.aim_scripted) if args.aim_scripted else None
        cfg.aim_bound = args.aim_bound
        cfg.aim_window_s = args.aim_window
        cfg.aim_obs_comp = args.aim_obs_comp
        cfg.aim_obs_comp_full = args.aim_obs_comp_full
    else:
        cfg = PitchedBattingHumanPlayEnvCfg()
    cfg.percept_filter = args.percept_filter
    if args.phase_horizon is not None:
        cfg.phase_horizon_s = args.phase_horizon
    cfg.pitch_release_csv = args.release_csv
    cfg.motion_files = [f for f in cfg.motion_files if os.path.splitext(os.path.basename(f))[0] in JOINT_CLIPS]
    assert len(cfg.motion_files) == len(JOINT_CLIPS), cfg.motion_files
    cfg.real_height = True
    cfg.timing_noise_s = args.timing_noise
    cfg.timing_bias_s = 0.055
    cfg.aim_offset_z = 0.0
    cfg.pitch_ramp_steps = 0
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    env = gym.make("AIB-ArmAim-Play-v0" if args.armaim else "AIB-PitchedBatting-Human-Play-v0", cfg=cfg)
    base = env.unwrapped
    n, dev = base.num_envs, base.device

    from rsl_rl.modules import ActorCritic, EmpiricalNormalization

    if args.armaim:  # the env runs the frozen swing itself; the residual policy's action (ignored when scripted) is zero
        z_act = torch.zeros(n, env.unwrapped.single_action_space.shape[0], device=dev)
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    sd = ck["model_state_dict"]
    n_obs, n_act, n_cr = sd["actor.0.weight"].shape[1], sd["actor.6.weight"].shape[0], sd["critic.0.weight"].shape[1]
    ac = None if args.armaim else ActorCritic(n_obs, n_cr, n_act, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                     activation="elu", init_noise_std=0.3).to(dev)
    if ac is not None:
        ac.load_state_dict(sd)
        ac.eval()
        norm = EmpiricalNormalization(shape=[n_obs], until=1.0e8).to(dev)
        norm.load_state_dict(ck["obs_norm_state_dict"])
        norm.eval()

    rob = base.robot
    feet = [rob.find_bodies(f"{s}_foot")[0][0] for s in "lr"]
    snapped = torch.zeros(n, dtype=torch.bool, device=dev)
    bank = {k: [] for k in ("env", "root", "jp", "jv", "tgt", "vtgt", "act", "hit", "fallen", "pelvis_z", "tilt_deg",
                            "root_speed", "foot_z", "t")}
    armed = [False]

    def snap(ids):
        if len(ids) == 0:
            return
        d = rob.data
        root = d.root_state_w[ids].clone()
        root[:, :3] -= base.scene.env_origins[ids]
        g = d.projected_gravity_b[ids]
        for k, v in (("env", ids), ("root", root), ("jp", d.joint_pos[ids]), ("jv", d.joint_vel[ids]),
                     ("tgt", base.joint_targets[ids]), ("vtgt", base.joint_vel_targets[ids]), ("act", base.actions[ids]),
                     ("hit", base.hit[ids]), ("fallen", base.fallen[ids]),
                     ("pelvis_z", d.body_link_pos_w[ids, base.pelvis_id, 2] - base.scene.env_origins[ids, 2]),
                     ("tilt_deg", torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1.0, 1.0)))),
                     ("root_speed", d.root_lin_vel_w[ids].norm(dim=-1)),
                     ("foot_z", d.body_link_pos_w[ids][:, feet, 2] - base.scene.env_origins[ids, 2:3]),
                     ("t", base.phys_t[ids])):
            bank[k].append(v.detach().clone().cpu())
        snapped[ids] = True

    orig_reset = base._reset_idx

    def hooked_reset(env_ids):
        if armed[0] and env_ids is not None:  # the training episode ends here (pitch passed / time out): pre-reset state
            ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
            snap(ids[~snapped[ids]])
        orig_reset(env_ids)

    base._reset_idx = hooked_reset
    obs, _ = env.reset()
    armed[0] = True
    for step in range(args.max_steps):
        with torch.inference_mode():
            act = z_act if args.armaim else ac.act_inference(norm(obs["policy"]))
        obs, *_ = env.step(act)
        new = (base.hit & ~snapped).nonzero(as_tuple=False).squeeze(-1)
        snap(new)  # the control step with the contact (its end state)
        if bool(snapped.all()):
            break
    out = {k: torch.cat(v).numpy() for k, v in bank.items()}
    out["joint_names"] = np.array(rob.joint_names)
    out["body_joint_ids"] = base.body_joint_ids.cpu().numpy()
    out["default_joint_pos"] = rob.data.default_joint_pos[0].cpu().numpy()
    out["default_root_z"] = float(rob.data.default_root_state[0, 2])
    out["seed"] = args.seed
    np.savez(args.out, **out)
    h = out["hit"]
    q = lambda x: " / ".join(f"{v:.2f}" for v in np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0]))  # noqa: E731
    print(f"[bank] seed {args.seed}: {len(h)} states of {n} envs in {step + 1} steps; hits {int(h.sum())} "
          f"({h.mean():.3f}), misses {int((~h).sum())}; fallen at the trigger {int(out['fallen'].sum())}", flush=True)
    for nm, m in (("hits", h), ("misses", ~h)):
        if m.any():
            print(f"[bank] {nm}: pelvis z min/q1/med/q3/max {q(out['pelvis_z'][m])} m | tilt {q(out['tilt_deg'][m])} deg | "
                  f"root speed {q(out['root_speed'][m])} m/s | foot z max {q(out['foot_z'][m].max(-1))} m", flush=True)
    print(f"[bank] wrote {args.out}", flush=True)
    env.close()


main()
app.close()
