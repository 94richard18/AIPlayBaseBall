"""CPU falsifier for BatterSide (batter session, 10-28): the shim on a fake joint env, no Isaac app, no GPU.

omni / isaacsim / pxr / carb are mocked so the real DirectRLEnv (its read-only properties) and the real batter env code
import. A fake joint env (64 envs, random states) drives the joint env's hook order (reset, control at 100 Hz, ball_world
every physics step, post); the fake articulation checks the shape of every write against its env_ids. Hand-offs come
in env subsets (38 arbitrary envs, then 1, then the rest), as do resets (38, 1, all).

    python scripts/test_batter_side_cpu.py --batter <model_12500.pt>
    python scripts/test_batter_side_cpu.py --batter <model_12500.pt> --own_timing   # two-pass oracle timing check
"""

import argparse
import importlib.abc
import importlib.machinery
import os
import re
import sys
import types
from unittest import mock

PFX = ("omni", "isaacsim", "pxr", "carb", "usdrt")


class _MockModule(types.ModuleType):
    __path__ = []

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        m = mock.MagicMock(name=f"{self.__name__}.{n}")
        setattr(self, n, m)
        return m


class _MockFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in PFX:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        return _MockModule(spec.name)

    def exec_module(self, m):
        pass


sys.meta_path.insert(0, _MockFinder())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import random_orientation  # noqa: E402

from aibaseball.mocap.motion import KEY_BODIES  # noqa: E402
from aibaseball.tasks.one_outs.batter_side import JOINT_CLIPS, BatterSide  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env_cfg import PitchedBattingHumanPlayEnvCfg  # noqa: E402

N, SUB = 64, 38
G = torch.Generator().manual_seed(0)
torch.manual_seed(0)  # random_orientation draws from the global generator (unseeded, a fake bat could cross the pitch)


def rnd(*s):
    return torch.randn(*s, generator=G)


def check_ids(name, x, ids):
    k = N if ids is None else (len(ids) if not isinstance(ids, slice) else N)
    assert x.shape[0] == k, f"{name}: first dim {tuple(x.shape)} vs {k} env ids"


class FakeData:
    def __init__(self, nj, nb):
        q = random_orientation(N, "cpu")
        self.joint_pos, self.joint_vel = rnd(N, nj) * 0.1, rnd(N, nj) * 0.1
        self.default_joint_pos, self.default_joint_vel = torch.zeros(N, nj), torch.zeros(N, nj)
        self.soft_joint_pos_limits = torch.stack([-torch.ones(N, nj) * 2, torch.ones(N, nj) * 2], -1)
        self.joint_pos_limits = self.soft_joint_pos_limits.clone()
        self.default_root_state = torch.cat([torch.zeros(N, 2), torch.ones(N, 1), torch.tensor([[1.0, 0, 0, 0]]).expand(N, 4),
                                             torch.zeros(N, 6)], -1)
        for f in ("root_link_pos_w", "root_com_pos_w", "root_pos_w"):
            setattr(self, f, rnd(N, 3) + torch.tensor([18.0, -6.0, 1.0]))
        for f in ("root_link_quat_w", "root_com_quat_w", "root_quat_w"):
            setattr(self, f, q.clone())
        for f in ("root_lin_vel_w", "root_ang_vel_w", "root_com_lin_vel_w", "root_com_ang_vel_w", "root_link_lin_vel_w",
                  "root_link_ang_vel_w", "root_lin_vel_b", "root_ang_vel_b", "projected_gravity_b"):
            setattr(self, f, rnd(N, 3))
        for f in ("body_link_pos_w", "body_com_pos_w", "body_pos_w"):
            setattr(self, f, rnd(N, nb, 3) + torch.tensor([18.0, -6.0, 1.0]))
        for f in ("body_link_quat_w", "body_com_quat_w", "body_quat_w"):
            setattr(self, f, random_orientation(N * nb, "cpu").view(N, nb, 4))
        for f in ("body_lin_vel_w", "body_ang_vel_w", "body_com_lin_vel_w", "body_com_ang_vel_w", "body_link_lin_vel_w",
                  "body_link_ang_vel_w"):
            setattr(self, f, rnd(N, nb, 3))
        self.root_state_w = torch.cat([self.root_pos_w, self.root_quat_w, self.root_lin_vel_w, self.root_ang_vel_w], -1)
        self.root_link_state_w = self.root_state_w.clone()
        self.root_com_state_w = self.root_state_w.clone()
        self.applied_torque = torch.zeros(N, nj)
        self.computed_torque = torch.zeros(N, nj)

    def __getattr__(self, name):
        raise AttributeError(f"FakeData has no {name} (add it to the fake if the real Articulation has it)")


class FakeArticulation:
    def __init__(self, joint_names, body_names):
        self.joint_names, self.body_names = joint_names, body_names
        self.num_joints, self.num_bodies = len(joint_names), len(body_names)
        self.data = FakeData(self.num_joints, self.num_bodies)
        self.calls = {}
        self._ALL_INDICES = torch.arange(N, dtype=torch.long)

    def _rec(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def find_joints(self, exprs, joint_subset=None, preserve_order=False):
        exprs = [exprs] if isinstance(exprs, str) else exprs
        if preserve_order:
            ids = [i for e in exprs for i, n in enumerate(self.joint_names) if re.fullmatch(e, n)]
        else:
            ids = [i for i, n in enumerate(self.joint_names) if any(re.fullmatch(e, n) for e in exprs)]
        return ids, [self.joint_names[i] for i in ids]

    def find_bodies(self, exprs, preserve_order=False):
        exprs = [exprs] if isinstance(exprs, str) else exprs
        ids = [i for i, n in enumerate(self.body_names) if any(re.fullmatch(e, n) for e in exprs)]
        return ids, [self.body_names[i] for i in ids]

    def reset(self, env_ids=None):
        self._rec("reset")

    def write_root_state_to_sim(self, s, env_ids=None):
        self._rec("write_root_state_to_sim")
        check_ids("write_root_state_to_sim", s, env_ids)
        assert s.shape[-1] == 13 and torch.isfinite(s).all()

    def write_root_pose_to_sim(self, s, env_ids=None):
        self._rec("write_root_pose_to_sim")
        check_ids("write_root_pose_to_sim", s, env_ids)
        assert s.shape[-1] == 7

    def write_root_velocity_to_sim(self, s, env_ids=None):
        self._rec("write_root_velocity_to_sim")
        check_ids("write_root_velocity_to_sim", s, env_ids)

    def write_joint_state_to_sim(self, q, qd, joint_ids=None, env_ids=None):
        self._rec("write_joint_state_to_sim")
        check_ids("write_joint_state_to_sim q", q, env_ids)
        check_ids("write_joint_state_to_sim qd", qd, env_ids)

    def _target(self, name):
        def f(t, joint_ids=None, env_ids=None):
            self._rec(name)
            check_ids(name, t, env_ids)
            assert torch.isfinite(t).all(), name
        return f

    def __getattr__(self, name):
        if name.startswith("set_joint_") or name.startswith("write_joint_"):
            return self._target(name)
        if name in ("set_external_force_and_torque", "write_data_to_sim", "update"):
            return lambda *a, **k: self._rec(name)
        raise AttributeError(f"FakeArticulation has no {name}")


class FakeSim:
    device = "cpu"


class FakeJointCfg:
    batter_every = 4
    zone_center_y = 0.0


class FakeJoint:
    def __init__(self, batter):
        self.sim, self.cfg, self.batter = FakeSim(), FakeJointCfg(), batter
        self.device, self.num_envs, self.physics_dt, self.step_dt = "cpu", N, 1.0 / 800.0, 0.0025
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)
        self.release_t_pred = torch.full((N,), 0.9975)
        self.j_fly = torch.zeros(N, dtype=torch.bool)
        self.j_pos, self.j_vel, self.j_omega = torch.zeros(N, 3), torch.zeros(N, 3), torch.zeros(N, 3)

    def plate_front_w(self):
        import math
        c, s = math.cos(math.radians(-18.0)), math.sin(math.radians(-18.0))
        o = rnd(N, 3) * 10.0
        o[:, 2] = 0
        return o + torch.tensor([18.44 * c, 18.44 * s, -0.196])

    def release(self, ids):
        """Live release state near the mound (world): moving toward the plate along the -18 deg line."""
        import math
        d = torch.tensor([math.cos(math.radians(-18.0)), math.sin(math.radians(-18.0)), 0.0])
        self.j_pos[ids] = self.batter.inner_pivot[ids] - d * 17.0 + torch.tensor([0.0, 0.0, 1.9])
        self.j_vel[ids] = d * 38.0 + rnd(len(ids), 3) * 0.5
        self.j_omega[ids] = rnd(len(ids), 3) * 200.0
        self.j_fly[ids] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batter", required=True)
    ap.add_argument("--release_csv", default="assets/game/p3c_identical_t18_release.csv")
    ap.add_argument("--after_done", default="policy", choices=("policy", "hold", "blend"))
    ap.add_argument("--own_timing", action="store_true", help="the two-pass oracle timing falsifier only")
    a = ap.parse_args()
    os.environ["AIB_EFFORT_SCALE"] = "1.5"
    cfg = PitchedBattingHumanPlayEnvCfg()
    cfg.pitch_release_csv = a.release_csv
    cfg.scene.num_envs = N
    clip = [f for f in cfg.motion_files if os.path.splitext(os.path.basename(f))[0] in JOINT_CLIPS][0]
    jn = [str(x) for x in np.load(clip)["joint_names"]]
    bodies = ["pelvis"] + [b for b in KEY_BODIES if b != "pelvis"]
    art = FakeArticulation(jn, bodies)
    joint = FakeJoint(art)
    if a.own_timing:
        return own_timing_check(cfg, art, a.batter)
    take = torch.zeros(N, dtype=torch.bool)
    take[-8:] = True  # the last 8 envs take (ONE OUTS PA swing mask): stance targets the whole pitch, still handed / judged
    side = BatterSide(joint, cfg, a.batter, after_done=a.after_done, swing=~take)
    side.record_swing()
    art.inner_pivot = side.robot.pivot
    print(f"[cpu] init ok: {N} envs, clips {sorted(set(side.fixed_clip.tolist()))}, start_t pending", flush=True)
    perm = torch.randperm(N, generator=G)
    sub, one, rest = perm[:SUB].sort().values, perm[SUB:SUB + 1], perm[SUB + 1:].sort().values

    # resets: subset, single, all (the joint env's _reset_idx calls batter_side.reset(env_ids))
    for ids in (sub, one, torch.arange(N)):
        side.reset(ids)
    print(f"[cpu] reset ok (38 / 1 / 64), start_t {side.start_t.min():.3f}..{side.start_t.max():.3f} s", flush=True)

    # the hook order of the joint env's step: control at k % 4 == 0, ball_world every physics step, post on ctrl steps
    hand = {int(round(0.9975 / 0.0025)): sub, int(round(0.9975 / 0.0025)) + 3: one, int(round(0.9975 / 0.0025)) + 9: rest}
    steps = int(round(6.0 / 0.0025))  # the smoke run's episode: pitch, contact, batted-ball flight
    h0, bat = min(hand), side.bat_id
    swing = sub[:19]  # judge on an env subset: a bat barrel put in the live ball's path
    assert not take[swing].any(), "the bat-in-the-path subset must swing"
    stance = side._ref(torch.zeros(N))["joint_pos"]
    take_err, swing_dev = 0.0, 0.0
    q_up = torch.tensor([0.70710678, 0.0, -0.70710678, 0.0])  # bat x axis -> world +z (across the pitch line)
    for k in range(steps):
        ctrl = k % 4 == 0
        if ctrl:
            side.control()
            tq = side.joint_targets[:, side.body_joint_ids]
            take_err = max(take_err, float((tq[take] - stance[take]).abs().max()), float(side.joint_vel_targets[take].abs().max()))
            live = side.active & ~take & ~side.done
            if live.any():  # falsifier: the swinging envs' targets do leave the stance
                swing_dev = max(swing_dev, float((tq[live] - stance[live]).abs().max()))
        if k in hand:
            joint.release(hand[k])
        if k == h0 + 120:  # a still bat 0.25 m ahead on the ball's path, moving toward the mound
            u = joint.j_vel[swing] / joint.j_vel[swing].norm(dim=-1, keepdim=True)
            spot = joint.j_pos[swing] + u * 0.25
        if h0 + 120 <= k <= h0 + 140:
            d = art.data
            knob = spot - torch.tensor([0.0, 0.0, side.bat_geom.sweet_spot_x])
            for f in ("body_link_pos_w", "body_pos_w"):
                getattr(d, f)[swing, bat] = knob
            d.body_com_pos_w[swing, bat] = knob + torch.tensor([0.0, 0.0, side.bat_geom.cm_x])  # the bat's real CM
            for f in ("body_link_quat_w", "body_com_quat_w", "body_quat_w"):
                getattr(d, f)[swing, bat] = q_up
            d.body_com_lin_vel_w[swing, bat] = -joint.j_vel[swing] / joint.j_vel[swing].norm(dim=-1, keepdim=True) * 30.0
            d.body_com_ang_vel_w[swing, bat] = 0.0
        p, v, w = side.ball_world(joint.physics_dt)
        for nm, x in (("p", p), ("v", v), ("w", w)):
            assert x.shape == (N, 3), (nm, x.shape)
        if k == h0 + 140:  # the joint env's (world) ball after the hit flies back toward the mound (world -d)
            import math
            dw = torch.tensor([math.cos(math.radians(-18.0)), math.sin(math.radians(-18.0)), 0.0])
            hw = swing[side.hit[swing]]
            back = (v[hw] * dw).sum(-1)
            assert len(hw) and (back < -20.0).all(), back.tolist()
            print(f"[cpu] world ball after the hit ok: {len(hw)} envs, speed toward the mound "
                  f"{(-back).min():.1f}..{(-back).max():.1f} m/s", flush=True)
        m = joint.j_fly.unsqueeze(-1)
        joint.j_pos, joint.j_vel, joint.j_omega = (torch.where(m, p, joint.j_pos), torch.where(m, v, joint.j_vel),
                                                   torch.where(m, w, joint.j_omega))
        joint.episode_length_buf += 1
        if (k + 1) % 4 == 0:  # joint_env.step: post() after the interval's last pitcher step
            side.post()
    print(f"[cpu] take ok: {int(take.sum())} take envs |target - stance| max {take_err:.2e} (swinging envs reach {swing_dev:.2f} rad)", flush=True)
    assert take_err < 1e-6 and swing_dev > 0.1, (take_err, swing_dev)
    assert side.handed.all(), int(side.handed.sum())
    assert side.active.all(), int(side.active.sum())
    fin = torch.isfinite(side.pitch_pos).all() and torch.isfinite(side.tee_pos).all() and torch.isfinite(side.flight_time).all()
    print(f"[cpu] {steps} physics steps ok: handed {int(side.handed.sum())}/{N} (38 / 1 / 25), active {int(side.active.sum())}, "
          f"flight_time {side.flight_time.min():.3f}..{side.flight_time.max():.3f} s, pitch_kmh "
          f"{side.pitch_kmh.min():.1f}..{side.pitch_kmh.max():.1f}, finite {bool(fin)}, hits {int(side.hit.sum())}, "
          f"writes {art.calls}", flush=True)
    hit_ids = side.hit.nonzero(as_tuple=False).squeeze(-1)
    assert len(hit_ids) > 0 and set(hit_ids.tolist()) <= set(swing.tolist()), hit_ids.tolist()
    logged = sorted({int(r["env"]) for r in side.play_log})
    assert set(logged) <= set(swing.tolist()) and logged, logged
    r0 = next(r for r in side.play_log if int(r["env"]) == logged[0])
    print(f"[cpu] judge ok: hits {len(hit_ids)} (all in the 19-env swing subset), play_log envs {len(logged)}, first record "
          f"{ {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r0.items()} }", flush=True)
    spray = torch.tensor([float(r["spray_deg"]) for r in side.play_log])
    # the bat moves toward the mound = the batter's centre field (B +x): the batted balls go back up the middle
    assert spray.abs().max() < 45.0, spray.tolist()
    print(f"[cpu] batted-ball direction ok: |spray| max {spray.abs().max():.1f} deg (B +x = up the middle)", flush=True)

    # root write-back with subsets (the native reset writes the stance this way)
    st = side.robot.data.root_state_w
    for ids in (sub, one, None):
        side.robot.write_root_state_to_sim(st if ids is None else st[ids], ids)
        side.robot.write_root_pose_to_sim((st if ids is None else st[ids])[:, :7], ids)
    print("[cpu] root write-back ok (38 / 1 / all)", flush=True)
    R = side.rec
    miss = side.handed & ~side.hit
    assert torch.isfinite(R["t_plane"][miss]).all() and torch.isfinite(R["ref_err"][miss]).all()
    ft = R["t_plane"][miss] - side._joint.handoff_t[miss] if hasattr(side._joint, "handoff_t") else None
    print(f"[cpu] swing record ok: t_plane {R['t_plane'][miss].min():.3f}..{R['t_plane'][miss].max():.3f} s, ref_err median "
          f"{R['ref_err'][miss].median():+.3f}, gap median {R['gap'].median():.3f} m, done {int(side.done.sum())}/{N}", flush=True)
    if a.after_done == "blend":  # by done + BLEND_S the PD targets are the stance
        from aibaseball.tasks.one_outs.batter_side import BLEND_S
        late = side.done & (side._joint_time() - side.done_t > BLEND_S)
        side.control()
        stance = side._ref(torch.zeros(N))["joint_pos"]
        err = (side.joint_targets[late][:, side.body_joint_ids] - stance[late]).abs().max()
        print(f"[cpu] blend ok: {int(late.sum())} envs past done + {BLEND_S} s, |target - stance| max {err:.2e}", flush=True)
        assert late.any() and err < 1e-6
    print("CPU_SHIM_CHECK_PASS", flush=True)


def own_timing_check(cfg, art, ckpt):
    """Pass 1 schedules every env on the median-flight row; pass 2 on its own (hand-off, contact-plane time) from
    pass 1. The joint env's real cadence: one joint step = 2 physics steps (ball_world each), control / post every 4th
    step, hand-offs at the start of a joint step in env groups (different flights: random release speeds). The tempo
    action is zeroed (rate = speed), so with the native schedule the reference clock at the crossing must be exactly
    timing_bias_s * speed past contact (pitched_env._reset_idx: ball arrival = reference contact + bias)."""
    import copy

    def run(own, no_handoff=False):
        G.manual_seed(0)  # the same fake pitches (and bat states) in both passes
        torch.manual_seed(0)
        joint = FakeJoint(art)
        joint.handoff_t = torch.full((N,), float("nan"))
        side = BatterSide(joint, copy.deepcopy(cfg), ckpt, own_timing=own, no_handoff=no_handoff)
        side._actor.act_inference = lambda o: torch.zeros(o.shape[0], side._actor.actor[-1].out_features)
        side.record_swing()
        art.inner_pivot = side.robot.pivot
        side.reset(torch.arange(N))
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(1))
        j0 = int(round(0.9975 / joint.step_dt))
        hand = {j0: perm[:SUB], j0 + 3: perm[SUB:SUB + 1], j0 + 9: perm[SUB + 1:]}
        speeds = 30.0 + 15.0 * torch.rand(N, generator=torch.Generator().manual_seed(2))
        for j in range(int(round(2.6 / joint.step_dt))):
            ctrl = j % joint.cfg.batter_every == 0
            if ctrl:
                side.control()
                pt_ctrl = side.phys_t.clone()
            if j in hand:
                ids = hand[j]
                joint.release(ids)
                joint.j_vel[ids] = joint.j_vel[ids] / joint.j_vel[ids].norm(dim=-1, keepdim=True) * speeds[ids, None]
                joint.handoff_t[ids] = joint.episode_length_buf[ids].float() * joint.step_dt
            for _ in range(int(round(joint.step_dt / joint.physics_dt))):
                p, v, w = side.ball_world(joint.physics_dt)
                m = joint.j_fly.unsqueeze(-1)
                joint.j_pos, joint.j_vel, joint.j_omega = (torch.where(m, p, joint.j_pos), torch.where(m, v, joint.j_vel),
                                                           torch.where(m, w, joint.j_omega))
            joint.episode_length_buf += 1
            if (j + 1) % joint.cfg.batter_every == 0:  # joint_env.step: post() after the interval's last pitcher step
                live = side.active & (side.episode_length_buf > 0)
                gap = (side.phys_t - pt_ctrl)[live]  # the obs post() computes: a full control interval after control()
                assert not len(gap) or (gap - joint.step_dt * joint.cfg.batter_every).abs().max() < 1e-5, gap.unique()
                side.post()
        return side, joint

    s1, j1 = run(None)
    R1 = s1.rec
    ok1 = torch.isfinite(R1["t_plane"])
    assert ok1.sum() >= 48 and not s1.hit.any(), (int(ok1.sum()), int(s1.hit.sum()))  # the fake's random falls end some early
    own = torch.stack([j1.handoff_t, R1["t_plane"].nan_to_num(3.0)], -1)
    s2, j2 = run(own)
    R2 = s2.rec
    ok = ok1 & torch.isfinite(R2["t_plane"])
    assert ok.sum() >= 48 and (R2["t_plane"] - R1["t_plane"])[ok].abs().max() < 1e-5, "the pitch must not depend on the schedule"
    bias, spd = s2.cfg.timing_bias_s, s2.speed
    # ref_t moves in 100 Hz steps, advanced at the start of each control interval: at the crossing it reads the clock
    # of the interval's end (diag_bat_contact's ref_err has the same 0..10 ms offset); take it back to t_plane
    ctrl = j2.step_dt * j2.cfg.batter_every
    ahead = (torch.floor(R1["t_plane"] / ctrl + 1e-6) + 1.0) * ctrl - R1["t_plane"]
    e1 = (R1["ref_err"] / spd - ahead - bias)[ok]  # 0 = the native schedule
    e2 = (R2["ref_err"] / spd - ahead - bias)[ok]
    fl = (R1["t_plane"] - j1.handoff_t)[ok]
    print(f"[cpu] own timing ({int(ok.sum())} envs crossed in both passes): flights {fl.min():.3f}..{fl.max():.3f} s; ref_err / speed - bias: median row "
          f"{e1.median():+.4f} (sd {e1.std():.4f}, max |.| {e1.abs().max():.4f}) -> own pitch {e2.median():+.5f} "
          f"(max |.| {e2.abs().max():.5f}) s; start_t {s2.start_t.min():.3f}..{s2.start_t.max():.3f} s", flush=True)
    assert e2.abs().max() < 1e-4, e2.abs().max()
    assert e1.std() > 5 * e2.abs().max()
    s3, j3 = run(None, no_handoff=True)  # the predicted row on its own (native) schedule: exact without a hand-off
    R3 = s3.rec
    ok3 = torch.isfinite(R3["t_plane"])
    a3 = (torch.floor(R3["t_plane"] / ctrl + 1e-6) + 1.0) * ctrl - R3["t_plane"]
    e3 = (R3["ref_err"] / spd - a3 - bias)[ok3]
    print(f"[cpu] no_handoff: {int(ok3.sum())} envs crossed, ref_err / speed - bias max |.| {e3.abs().max():.5f} s, "
          f"flight {(R3['t_plane'] - s3.start_t - s3.release_t)[ok3].median():.3f} s (row {float(s3.lib.flight_time[0]):.3f})", flush=True)
    assert ok3.sum() >= 48 and e3.abs().max() < 1.5 * j3.physics_dt  # native too: the pitch leaves on the physics step after release_t
    print("CPU_OWN_TIMING_PASS", flush=True)


main()
