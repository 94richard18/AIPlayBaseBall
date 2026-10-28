"""Sim-side articulation readout (planning 14, 10-28): the DOF / body properties as PhysX reports them after init, not
the cfg, for diffing the native batter against the joint scene's. Import and call dump(art, sim, path) after the env is
built; diff two dumps with `python scripts/dof_readout.py a.json b.json`."""

import json
import sys


def dump(art, sim, path, label=""):
    try:
        _dump(art, sim, path, label)
    except Exception as e:  # noqa: BLE001  a diagnostic: never kill the run
        import traceback
        print(f"[dof] {label} readout failed: {e!r}", flush=True)
        traceback.print_exc()


def _dump(art, sim, path, label):
    import torch
    from pxr import PhysxSchema, UsdPhysics
    import omni.usd

    v = art.root_physx_view
    out = dict(label=label, joint_names=list(art.joint_names), body_names=list(art.body_names))
    getters = dict(stiffness="get_dof_stiffnesses", damping="get_dof_dampings", max_force="get_dof_max_forces",
                   max_vel="get_dof_max_velocities", armature="get_dof_armatures",
                   friction="get_dof_friction_coefficients", limits="get_dof_limits", mass="get_masses",
                   inertia="get_inertias", coms="get_coms")
    for k, g in getters.items():
        try:
            x = getattr(v, g)().float().cpu()
        except Exception as e:  # noqa: BLE001
            out[k] = f"n/a: {e}"
            continue
        out[k] = x[0].tolist()  # env 0
        out[k + "_env_spread"] = float((x - x[0:1]).abs().max())  # any env different from env 0
    out["isaaclab_data"] = {k: getattr(art.data, k)[0].float().cpu().tolist() for k in
                            ("joint_stiffness", "joint_damping", "joint_armature", "joint_effort_limits",
                             "joint_vel_limits", "default_joint_pos") if hasattr(art.data, k)}
    out["actuators"] = {n: type(a).__name__ for n, a in art.actuators.items()}
    try:
        out["gravity_sim"] = list(sim.physics_sim_view.get_gravity())
    except Exception as e:  # noqa: BLE001
        out["gravity_sim"] = f"n/a: {e}"
    out["gravity_cfg"] = list(sim.cfg.gravity)
    out["dt"] = sim.cfg.dt
    try:
        stage = omni.usd.get_context().get_stage()
        root = art.cfg.prim_path.replace(".*", "0")
        prims = [p for p in stage.Traverse() if str(p.GetPath()).startswith(root + "/") or str(p.GetPath()) == root]
        usd = {}
        for p in prims:
            if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
                a = PhysxSchema.PhysxArticulationAPI(p)
                usd["articulation " + str(p.GetPath())[len(root):]] = dict(
                    pos_iters=a.GetSolverPositionIterationCountAttr().Get(), vel_iters=a.GetSolverVelocityIterationCountAttr().Get(),
                    self_coll=a.GetEnabledSelfCollisionsAttr().Get(), enabled=a.GetArticulationEnabledAttr().Get(),
                    sleep=a.GetSleepThresholdAttr().Get(), stab=a.GetStabilizationThresholdAttr().Get())
            for d in ("angular", "linear"):
                if p.HasAPI(UsdPhysics.DriveAPI, d):
                    dr = UsdPhysics.DriveAPI(p, d)
                    usd["drive " + p.GetName()] = dict(type=dr.GetTypeAttr().Get(), kp=dr.GetStiffnessAttr().Get(),
                                                       kd=dr.GetDampingAttr().Get(), fmax=dr.GetMaxForceAttr().Get())
            if p.HasAPI(PhysxSchema.PhysxJointAPI):
                j = PhysxSchema.PhysxJointAPI(p)
                usd["physxjoint " + p.GetName()] = dict(armature=j.GetArmatureAttr().Get(), friction=j.GetJointFrictionAttr().Get(),
                                                        maxvel=j.GetMaxJointVelocityAttr().Get())
            if p.IsA(UsdPhysics.Joint) and not p.HasAPI(UsdPhysics.DriveAPI, "angular"):
                usd["joint " + p.GetName()] = dict(type=p.GetTypeName(),
                                                   excluded=UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get(),
                                                   enabled=UsdPhysics.Joint(p).GetJointEnabledAttr().Get())
        out["usd"] = {k: {kk: (vv if isinstance(vv, (int, float, bool, str, type(None))) else str(vv)) for kk, vv in d.items()}
                      for k, d in usd.items()}
    except Exception as e:  # noqa: BLE001  the PhysX numbers above are still written
        out["usd"] = f"n/a: {e!r}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"[dof] {label} -> {path}: {len(out['joint_names'])} dofs, actuators {out['actuators']}", flush=True)


def _diff(a, b, key="", tol=1e-6):
    n = 0
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ("label",):
                continue
            if k not in a or k not in b:
                print(f"  only in {'b' if k not in a else 'a'}: {key}/{k}")
                n += 1
            else:
                n += _diff(a[k], b[k], f"{key}/{k}", tol)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            n += _diff(x, y, f"{key}[{i}]", tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
            print(f"  {key}: {a} vs {b}")
            n += 1
    elif a != b:
        print(f"  {key}: {a} vs {b}")
        n += 1
    return n


if __name__ == "__main__":
    A, B = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
    n = _diff(A, B)
    print(f"[dof] {n} differences ({A['label']} vs {B['label']})")
