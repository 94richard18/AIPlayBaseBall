"""CPU check of SaimSwingEnv (Human30) against BatterSide's game-side scripted aim: same random percept / clock / hit
states through both offset rules for 60 control steps (parents' hooks stubbed) -> max |offset diff| must be 0, plus the
observation comp restores joint_pos. No Isaac.

    python scripts/test_saim_swing_cpu.py
"""
import os
import sys
import types
from unittest import mock

import importlib.abc
import importlib.machinery

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

import torch  # noqa: E402

from aibaseball.tasks.one_outs.batter_side import BatterSide  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env import PitchedBattingEnv  # noqa: E402
from aibaseball.tasks.tee_batting.pitched_env_cfg import PitchedBattingHuman30EnvCfg  # noqa: E402
from aibaseball.tasks.tee_batting.saim_swing_env import SaimSwingEnv  # noqa: E402

torch.manual_seed(0)
cfg = PitchedBattingHuman30EnvCfg()
n, nb = 64, 29
g = __import__("json").load(open(cfg.saim_gain))
na = len(cfg.aim_joints)


def common(o):
    o.aim_cols = torch.arange(na)
    o.aim_ids = torch.arange(na)
    o.q_lo, o.q_hi = -3 * torch.ones(nb), 3 * torch.ones(nb)
    o.res_off = torch.zeros(n, na)
    o.res_closed = torch.zeros(n, dtype=torch.bool)
    o.contact_x_ref = torch.zeros(n)
    o.ref = types.SimpleNamespace(contact_time=torch.full((4,), 1.0))
    o.clip = torch.randint(0, 4, (n,))
    o.nominal_target = torch.randn(n, 3) * 0.05


props = dict(num_envs=n, device="cpu", step_dt=0.02)  # (properties on the real envs)
A = type("A", (SaimSwingEnv,), props)
B = type("B", (BatterSide,), props)
a = A.__new__(A)
b = B.__new__(B)
common(a)
common(b)
a.cfg = cfg
a._gp, a._gm = torch.tensor(g["gain_plus"]), torch.tensor(g["gain_minus"])
a.track_stats = {k: torch.zeros(()) for k in ("saim_win_frac", "saim_abs_off", "saim_sat")}
b.nominal_target = a.nominal_target
b.clip = a.clip
b._saim = dict(gp=a._gp, gm=a._gm, bound=cfg.saim_bound, window=cfg.saim_window_s, full=True, tau=cfg.saim_tau_s, miss_blend=0.0)
b.saim_off_close = torch.zeros(n, na)
b.saim_t_close = torch.full((n,), float("nan"))
b.saim_sat = torch.zeros(n, dtype=torch.bool)
b._joint_time = lambda: torch.zeros(n)
hit = torch.zeros(n, dtype=torch.bool)
px = torch.full((n,), 15.0)
worst, peak = 0.0, 0.0
with mock.patch.object(PitchedBattingEnv, "_pre_physics_step", lambda self, act: None):
    for k in range(60):
        t = 0.3 + 0.02 * k
        hit = hit | (torch.rand(n) < 0.03)
        px = px - 0.6
        perc = torch.randn(n, 3) * 0.2
        tgt = torch.randn(n, nb) * 0.3
        for o in (a, b):
            o.ref_t = torch.full((n,), t)
            o.hit, o.pitch_pos = hit.clone(), torch.stack([px, torch.zeros(n), torch.zeros(n)], -1)
            o.joint_targets = tgt.clone()
        a._ball_seen = perc
        b._perc = perc
        a._pre_physics_step(torch.zeros(n, 30))
        b._saim_step()
        worst = max(worst, float((a.res_off - b.res_off).abs().max()), float((a.joint_targets - b.joint_targets).abs().max()))
        peak = max(peak, float(a.res_off.abs().mean()))
print(f"[saim_swing cpu] max |offset / target diff| vs BatterSide over 60 steps: {worst:.2e}; "
      f"peak mean |offset| {peak:.3f} rad")
assert worst == 0.0 and peak > 0.01
print("SAIM_SWING_CPU PASS")
