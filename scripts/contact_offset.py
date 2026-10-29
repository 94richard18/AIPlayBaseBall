"""Contact-offset measurement (planning 14, user 10-29 "打者無法打出內野的原因是力量不夠還是沒有打中球心"): where the
exit-velocity loss comes from - bat speed or contact quality. CPU only: reads joint_smoke --contact_dump npz + its CSV
per seed and replays the engine's own impact model (aibaseball/physics/impact.py resolve_impact) counterfactually.

Per first impact of each env:
  axial offset dx = contact station - sweet spot (m, + toward the tip)
  normal offset D = (ball centre - closest point on the bat axis) . e, e = unit(axis x u) signed e_z > 0, where
      u = -unit(component of the ball-bat relative velocity at contact normal to the axis) = the head-on normal.
      D > 0 = ball centre above the head-on line (the bat passes UNDER the ball centre = undercut in the bat-ball
      literature: backspin, higher launch); D < 0 = above it (overcut, topspin, grounders).
  exit_actual   = |ball velocity| from the recorded impact (checked against a replay and the CSV exit_mps)
  exit_centred  = same bat state (v_cm, omega), contact moved to the sweet spot, ball centre on the head-on normal (D 0)
  exit_target   = exit_centred with the bat's v_cm and omega scaled so the sweet-spot speed is 25 m/s
  loss_contact  = exit_centred - exit_actual;  loss_speed = exit_target - exit_centred

    python scripts/contact_offset.py --runs logs/contact_offset/s6 logs/contact_offset/s1 --out logs/contact_offset
(each run: <run>.npz + <run>.csv)
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aibaseball.game.agents import StrikeZone  # noqa: E402
from aibaseball.physics import BatGeometry, BatSpec, PhysicsSpec, resolve_impact  # noqa: E402
from aibaseball.physics.impact import ContactQuery  # noqa: E402

TARGET_SS = 25.0  # m/s sweet-spot speed (the outfield need, 10-26)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", default="logs/contact_offset")
args = ap.parse_args()

spec, geom = PhysicsSpec(), BatGeometry(BatSpec(), "cpu", dtype=torch.float64)
r_ball = spec.ball.radius
zone = StrikeZone()
T = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
unit = lambda v: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # noqa: E731


def impact(x, n, ball_c, d):
    cq = ContactQuery(hit=torch.ones(len(x), dtype=torch.bool), frac=torch.zeros(len(x), dtype=torch.float64),
                      x_on_bat=x, normal=n, point=ball_c - r_ball * n)
    return resolve_impact(geom, spec, cq, ball_c, d["bv"], d["bw"], d["cm"], d["axis"], d["vcm"], d["w"])


def head_on(d, x):
    """Bat-axis point at station x, bat velocity there, head-on unit normal u (bat -> ball side)."""
    a = d["knob"] + x.unsqueeze(-1) * d["axis"]
    vb = d["vcm"] + torch.cross(d["w"], a - d["cm"], dim=-1)
    vr = d["bv"] - vb
    vr_perp = vr - (vr * d["axis"]).sum(-1, keepdim=True) * d["axis"]
    return a, vb, -unit(vr_perp), vr


rows = []
for run in args.runs:
    z = np.load(run + ".npz")
    tab = {int(r["env"]): r for r in csv.DictReader(open(run + ".csv"))}
    env = z["env"].astype(int)
    first = np.array([np.flatnonzero(env == e)[0] for e in np.unique(env)])  # one impact per env (hit latches)
    d = {k: T(z[k][first]) for k in ("x_on_bat", "normal", "point", "ball_c", "bat_cm", "bat_axis", "bat_vel_cm",
                                      "bat_omega", "ball_vel", "ball_omega", "cor")}
    d.update(bv=T(z["ball_vel_in"][first]), bw=T(z["ball_omega_in"][first]), cm=d["bat_cm"], axis=d["bat_axis"],
             vcm=d["bat_vel_cm"], w=d["bat_omega"])
    d["knob"] = d["cm"] - geom.cm_x * d["axis"]
    x = d["x_on_bat"]
    replay = impact(x, d["normal"], d["ball_c"], d)  # sanity: the recorded inputs give the recorded output
    rep_err = (replay.ball_vel - d["ball_vel"]).norm(dim=-1).max().item()
    # geometry at the actual contact
    a, vb_c, u, vr = head_on(d, x)
    off = d["ball_c"] - a
    e = unit(torch.cross(d["axis"], u, dim=-1))
    e = torch.where(e[:, 2:3] < 0, -e, e)
    D = (off * e).sum(-1)
    dx = x - geom.sweet_spot_x
    p_ss = d["knob"] + geom.sweet_spot_x * d["axis"]
    v_ss = d["vcm"] + torch.cross(d["w"], p_ss - d["cm"], dim=-1)
    rel_angle = torch.rad2deg(torch.acos((d["normal"] * u).sum(-1).clamp(-1, 1)))  # contact normal vs head-on
    # centred: sweet spot, head-on
    xs = torch.full_like(x, geom.sweet_spot_x)
    a_s, _, u_s, _ = head_on(d, xs)
    R_s = geom.radius_at(xs) + r_ball
    cen = impact(xs, u_s, a_s + R_s.unsqueeze(-1) * u_s, d)
    # target: centred, bat scaled to a 25 m/s sweet spot
    k = (TARGET_SS / v_ss.norm(dim=-1)).unsqueeze(-1)
    dt = dict(d, vcm=d["vcm"] * k, w=d["w"] * k)
    a_t, _, u_t, _ = head_on(dt, xs)
    tgt = impact(xs, u_t, a_t + R_s.unsqueeze(-1) * u_t, dt)
    la = lambda v: torch.rad2deg(torch.asin((v[:, 2] / v.norm(dim=-1)).clamp(-1, 1)))  # noqa: E731
    for i, e_ in enumerate(np.unique(env)):
        r = tab[int(e_)]
        ex = float(d["ball_vel"][i].norm())
        rows.append(dict(
            run=os.path.basename(run), env=int(e_), t=round(float(z["t"][first[i]]), 4), csv_hit=r["hit"],
            csv_exit_mps=r["exit_mps"], pred_y=r["pred_y"], pred_z=r["pred_z"],
            # the pitch's plate location: the hand-off prediction (plate_y / z is the batted ball's crossing on hits;
            # pred_y = -plate_y on the misses, the zone is symmetric in y)
            in_zone=zone.contains(float(r["pred_y"]), float(r["pred_z"])),
            pitch_speed=round(float(d["bv"][i].norm()), 3), ss_speed=round(float(v_ss[i].norm()), 3),
            contact_bat_speed=round(float(vb_c[i].norm()), 3), rel_speed=round(float(vr[i].norm()), 3),
            normal_speed=round(float(z["normal_speed"][first[i]]), 3), rel_angle_deg=round(float(rel_angle[i]), 2),
            dx_m=round(float(dx[i]), 4), D_m=round(float(D[i]), 4), dz_world_m=round(float(off[i, 2]), 4),
            cor=round(float(d["cor"][i]), 4), exit_actual=round(ex, 3), launch_actual=round(float(la(d["ball_vel"])[i]), 2),
            exit_centred=round(float(cen.ball_vel[i].norm()), 3), launch_centred=round(float(la(cen.ball_vel)[i]), 2),
            exit_target=round(float(tgt.ball_vel[i].norm()), 3),
            loss_contact=round(float(cen.ball_vel[i].norm()) - ex, 3),
            loss_speed=round(float(tgt.ball_vel[i].norm() - cen.ball_vel[i].norm()), 3)))
    print(f"[contact] {run}: {len(first)} impacts, replay max |dv| {rep_err:.2e} m/s, csv hits "
          f"{sum(r['hit'] == 'True' for r in tab.values())}", flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "contact_offset.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

lines = [f"contact offset (engine resolve_impact replay; sweet spot x {geom.sweet_spot_x:.4f} m from the knob, ball r "
         f"{r_ball:.4f} m; COR = clamp({spec.impact.cor_ref} + ({spec.impact.cor_slope:.6f})(v_n - "
         f"{spec.impact.cor_ref_speed:.2f}), {spec.impact.cor_min}, {spec.impact.cor_max}) x (1 - "
         f"{spec.impact.vib_loss_max}(1 - exp(-(dx/{spec.impact.vib_width})^2))); impulse from the 3D free-bat collision "
         f"matrix K with Coulomb friction mu {spec.impact.friction}, tangential COR {spec.impact.tangential_cor})",
         f"exit_mps check: max |exit_actual - csv exit_mps| "
         f"{max([abs(r['exit_actual'] - float(r['csv_exit_mps'])) for r in rows if r['csv_exit_mps'] not in ('', 'nan')] or [float('nan')]):.3f} m/s"]
qs = lambda v: f"{np.median(v):.3f} [{np.quantile(v, 0.25):.3f}, {np.quantile(v, 0.75):.3f}]"  # noqa: E731
for nm, sel in (("hits", [True] * len(rows)), ("in-zone hits", [r["in_zone"] for r in rows])):
    R = [r for r, s in zip(rows, sel) if s]
    if not R:
        continue
    g = lambda k: np.array([r[k] for r in R], dtype=float)  # noqa: E731
    lines.append(f"{nm} n {len(R)}: median [q25, q75]")
    for k in ("pitch_speed", "ss_speed", "contact_bat_speed", "rel_speed", "rel_angle_deg", "dx_m", "D_m", "dz_world_m",
              "cor", "exit_actual", "exit_centred", "exit_target", "loss_contact", "loss_speed", "launch_actual",
              "launch_centred"):
        lines.append(f"  {k:18s} {qs(g(k))}")
    lc, ls = g("loss_contact"), g("loss_speed")
    lines.append(f"  share of the gap to the target (median of loss / (exit_target - exit_actual)): contact "
                 f"{np.median(lc / (lc + ls)):.2f}, speed {np.median(ls / (lc + ls)):.2f}")
    lines.append(f"  corr(launch_actual, D) {np.corrcoef(g('launch_actual'), g('D_m'))[0, 1]:+.2f}; corr(launch_actual, "
                 f"dz_world) {np.corrcoef(g('launch_actual'), g('dz_world_m'))[0, 1]:+.2f}; D > 0 (undercut) "
                 f"{int((g('D_m') > 0).sum())}/{len(R)}; |dx| > 0.05 m {int((np.abs(g('dx_m')) > 0.05).sum())}/{len(R)}")
open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
