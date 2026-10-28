"""Joint ONE OUTS run helpers (pitcher session, M2 columns 10-28): per-env pitcher release columns, the batted ball's
landing point / fair flag, the count, and the film path (eval_stage's M3 joint cameras) for scripts/joint_smoke.py.

Lateral frames (planning 14, 10-28: every column with a left / right sign names its frame):
  _plate = the pitcher's plate frame (eval_stage release rows, j_cross, joint_env to_plate): rubber at the origin, +x
           toward the plate (plate_distance at its front), +y = the pitcher's left = the FIRST-base side, z up.
  _game  = the game convention (agents.Pitch.plate_y, one_outs.BODY_Y_M / is_hit_by_pitch, field.FieldSpec, play_log
           spray_deg, the batter frame B of batter_side): +x from the plate toward centre field, +y = the THIRD-base side
           (a right-handed batter's side), spray + = toward left field.
  The 180 deg turn between them: y_game = -y_plate (x_game = plate_distance - x_plate; vectors flip x and y), as
  pitch_gen.build_library_from_release, agents.TablePitcher and match_video.plate_crossing do. Vertical columns are
  the same in both. Worked row (10-28): P3c gate table row 0 aero_y +0.150 (plate) -> library / engine plate_y -0.151;
  joint env 0 (fall_ab_hold) j_cross y +0.150 (plate), the batter's pred_y -0.1504 (game), z 0.689 / 0.6892.

Columns appended after the batter session's (agreed 10-28, frame suffixes 10-28):
  p_released, p_release_t, p_speed, p_heading_plate, p_elev, p_rel_x_plate, p_rel_y_plate, p_rel_z (above the field),
  p_vx_plate, p_vy_plate, p_vz, p_wx_plate, p_wy_plate, p_wz, p_spin_rpm, p_aero_y_plate, p_aero_y_game, p_aero_z, p_tier,
  p_standing - the gate's release row (eval_stage film_release_row), tier from the drag + Magnus flight of the release
  state (aero y / z written also when it hit the ground first, as in the gate rows);
  ball_land_x_game, ball_land_y_game, ball_dist_m, ball_fair - from play_log's carry / spray (frame B: origin at the
  batter's crossing plane, not the plate centre - a few dm), NaN / False without a batted ball;
  count_balls, count_strikes - from --count_csv (env,balls,strikes), 0 / 0 by default (plumbing only)."""

import csv
import math
import os

import numpy as np
import torch

GATE_MIN_SPEED, GATE_MIN_RELEASE_T = 10.0, 0.9  # eval_stage --gate_min_speed / --gate_min_release_t defaults
PITCH_COLS = ("p_released", "p_release_t", "p_speed", "p_heading_plate", "p_elev", "p_rel_x_plate", "p_rel_y_plate", "p_rel_z",
              "p_vx_plate", "p_vy_plate", "p_vz", "p_wx_plate", "p_wy_plate", "p_wz", "p_spin_rpm", "p_aero_y_plate",
              "p_aero_y_game", "p_aero_z", "p_tier", "p_standing")
BALL_COLS = ("ball_land_x_game", "ball_land_y_game", "ball_dist_m", "ball_fair")
COUNT_COLS = ("count_balls", "count_strikes")


def pitcher_columns(base, p_fell):
    """One dict per env: the release row of the gate (thrown = released, >= 10 m/s, reference time >= 0.9 s; tiers t1
    |heading| <= 10, t2 reached the plate and |y| <= 1, t3 |y| <= 0.5 and z 0.2..1.5, t4 the zone with a ball radius).
    Call at the end of the run: no env was reset, so the release buffers still hold the pitch. p_standing = the stand
    criteria at the end and no fall in the run (eval_stage at_rest)."""
    from aibaseball.physics.ball_flight import simulate_to_plane
    cfg, n = base.cfg, base.num_envs
    pl = base.to_plate(base.rel_pos - base.scene.env_origins).cpu().double()
    vp, wp = base.to_plate(base.rel_vel).cpu().double(), base.to_plate(base.rel_omega).cpu().double()
    cr, _, _, re = simulate_to_plane(pl, vp, wp, base.phys, cfg.plate_distance, dt=0.005, t_max=3.0, ground_z=base.field_z)
    ok = base._stand_state()[-1] & ~p_fell
    rel, rt = base.released.cpu().numpy(), base.rel_ref_t.cpu().numpy()
    ok, cr, re = ok.cpu().numpy(), cr.numpy(), re.numpy()
    pl, vp, wp = pl.numpy(), vp.numpy(), wp.numpy()
    out = []
    for e in range(n):
        r = bool(rel[e])
        v = float(np.linalg.norm(vp[e]))
        head = math.degrees(math.atan2(vp[e, 1], vp[e, 0]))
        el = math.degrees(math.atan2(vp[e, 2], math.hypot(vp[e, 0], vp[e, 1])))
        ya, za, reached = float(cr[e, 1]) - cfg.zone_center_y, float(cr[e, 2]) - base.field_z, bool(re[e]) and r
        thrown = r and v >= GATE_MIN_SPEED and float(rt[e]) >= GATE_MIN_RELEASE_T
        t1 = thrown and abs(head) <= 10.0
        t2 = t1 and reached and abs(ya) <= 1.0
        t3 = t2 and abs(ya) <= 0.5 and 0.2 <= za <= 1.5
        t4 = t3 and abs(ya) <= 0.2159 + 0.0366 and 0.50 <= za <= 1.10
        out.append(dict(
            p_released=r, p_release_t=round(float(rt[e]), 4), p_speed=round(v, 3), p_heading_plate=round(head, 2), p_elev=round(el, 2),
            p_rel_x_plate=round(float(pl[e, 0]), 3), p_rel_y_plate=round(float(pl[e, 1]), 3),
            p_rel_z=round(float(pl[e, 2]) - base.field_z, 3),
            p_vx_plate=round(float(vp[e, 0]), 4), p_vy_plate=round(float(vp[e, 1]), 4), p_vz=round(float(vp[e, 2]), 4),
            p_wx_plate=round(float(wp[e, 0]), 3), p_wy_plate=round(float(wp[e, 1]), 3), p_wz=round(float(wp[e, 2]), 3),
            p_spin_rpm=round(float(np.linalg.norm(wp[e])) * 60.0 / (2.0 * math.pi), 0),
            p_aero_y_plate=round(ya, 3), p_aero_y_game=round(-ya, 3), p_aero_z=round(za, 3),  # as the gate rows: written also when it hit the ground first
            p_tier=int(t1) + int(t2) + int(t3) + int(t4), p_standing=bool(ok[e])))
    return out


def ball_columns(rec):
    """The first batted ball's landing point from play_log's carry and spray (game frame) and FieldSpec.is_fair."""
    from aibaseball.game.field import FieldSpec
    carry, spray = rec.get("carry_m", float("nan")), rec.get("spray_deg", float("nan"))
    if not rec or carry != carry or spray != spray:
        return dict(ball_land_x_game=float("nan"), ball_land_y_game=float("nan"), ball_dist_m=float("nan"), ball_fair=False)
    x, y = carry * math.cos(math.radians(spray)), carry * math.sin(math.radians(spray))
    return dict(ball_land_x_game=round(x, 2), ball_land_y_game=round(y, 2), ball_dist_m=round(carry, 1), ball_fair=FieldSpec().is_fair(x, y))


def count_columns(path, n):
    """--count_csv env,balls,strikes -> per env count (plumbing: nothing in the scene reads it yet)."""
    c = [dict(count_balls=0, count_strikes=0) for _ in range(n)]
    if path:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                e = int(r["env"])
                if 0 <= e < n:
                    c[e] = dict(count_balls=int(r["balls"]), count_strikes=int(r["strikes"]))
    return c


class JointFilm:
    """Film env k of the joint run as it plays (the eval row of that env is the filmed pitch: one process = one pitch
    per env, no reset). Same cameras as eval_stage film() in joint mode (M3 defaults 10-28): left a centre-field view
    with both players, right the view from behind the plate; the flight marker of env k only; others hidden. Frames
    every step_dt * slow * fps; mp4 crf 16 High profile.

    Film convention (planning 14, 10-28: the batter falls after its training episode ends - 56/60 under blend - and the
    film must not show it): the batter is shown until the play ends, cut = the batter's done_t + CUT_AFTER_S (done = the
    batted ball landed, or the pitch passed). From the cut both cameras leave the batter: left a side view of the
    pitcher from the first-base side (the batter ~65 deg off its axis), right a high camera behind the plate (8 m up)
    aimed at the batted ball (hit) or at the pitcher (miss / take) - the batter below it, ~30 deg outside the half
    field of view. The end card / README says so. M3 s1 k44 (10-28): a dribbler landed 0.6 m from the plate and the
    ball camera looked straight down at the fallen batter, so the ball is followed only while it is more than
    MIN_BATTER_DEG away from the batter's pelvis as seen from the camera; otherwise the pitcher."""
    CUT_AFTER_S = 0.5
    SIDE_EYE, SIDE_ZOOM = (0.5, 8.0, 1.3), 1.5  # plate frame: x from the rubber, y off the zone centre (+ first base), z
    HIGH_EYE, HIGH_ZOOM_BALL, HIGH_ZOOM_P = (6.0, 0.0, 8.0), 1.0, 1.5  # x behind the plate front
    NEAR_BALL_M, HIGH_ZOOM_NEAR = 25.0, 2.0  # planning 14 10-28: a ball within this of the plate gets the closer zoom
    MIN_BATTER_DEG = 30.0  # the zoom-1 half field of view is ~18 deg (vertical) / ~30 deg (horizontal)
    CF_BACK, CF_SIDE, CF_UP, CF_ZOOM = 40.0, -4.0, 3.5, 8.0
    CF_T = (0.42, -0.45, 0.75)
    PV_J_EYE, PV_J_TGT, PV_J_ZOOM = (9.0, 0.6, 1.4), (-10.0, -0.3, 1.4), 1.65  # 10-28: batter 482 px, bat tip 97 px from the top

    def __init__(self, base, path, env, slow=2.0, fps=30):
        import imageio
        import omni.usd
        from pxr import Usd, UsdGeom
        self.base, self.path, self.E = base, path, env
        stage = omni.usd.get_context().get_stage()
        self.stage, self.Usd, self.UsdGeom = stage, Usd, UsdGeom
        self.cam = UsdGeom.Camera(stage.GetPrimAtPath("/OmniverseKit_Persp"))
        self.focal0 = self.cam.GetFocalLengthAttr().Get() if self.cam else None
        E = env
        for i in range(base.num_envs):
            if i != E:
                UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}")).MakeInvisible()
        # cloned envs inherit env_0's opinions: a local "inherited" opinion on the env k root (eval_stage film())
        root = stage.GetPrimAtPath(f"/World/envs/env_{E}")
        UsdGeom.Imageable(root).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)

        def own_hidden(p):
            while p != root:
                a = UsdGeom.Imageable(p).GetVisibilityAttr() if UsdGeom.Imageable(p) else None
                if a and a.Get() == UsdGeom.Tokens.invisible:
                    return True
                p = p.GetParent()
            return False
        hidden = [str(p.GetPath()) for p in Usd.PrimRange(root)
                  if UsdGeom.Imageable(p) and UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible
                  and not own_hidden(p)]
        print(f"[video] env {E}: {sum(1 for _ in Usd.PrimRange(root))} prims, hidden {len(hidden)} {hidden[:3]}", flush=True)
        if hidden:
            raise RuntimeError(f"env {E} prims hidden: {hidden[:5]}")
        base.marker_envs = [E]
        self.every = max(1, int(round(1.0 / (fps * slow * base.step_dt))))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.vw = imageio.get_writer(path, fps=fps, quality=None, macro_block_size=1,
                                     ffmpeg_params=["-crf", "16", "-profile:v", "high", "-preset", "slow"])
        self.k, self.n, self.rep, self.last = 0, 0, 0, None
        g = base.scene.env_origins[E].cpu().numpy() + [0.0, 0.0, base.field_z]
        th = -np.radians(base.cfg.plate_yaw_deg)
        rz = lambda x, y: [np.cos(th) * x - np.sin(th) * y, np.sin(th) * x + np.cos(th) * y]  # noqa: E731
        zc, pd = base.cfg.zone_center_y, base.cfg.plate_distance
        self.cf = ((g + np.array(rz(-self.CF_BACK, zc + self.CF_SIDE) + [self.CF_UP])).tolist(),
                   (g + np.array(rz(self.CF_T[0] * pd, zc + self.CF_T[1]) + [self.CF_T[2]])).tolist())
        self.pv = ((g + np.array(rz(pd + self.PV_J_EYE[0], zc + self.PV_J_EYE[1]) + [self.PV_J_EYE[2]])).tolist(),
                   (g + np.array(rz(pd + self.PV_J_TGT[0], zc + self.PV_J_TGT[1]) + [self.PV_J_TGT[2]])).tolist())
        self.side_eye = (g + np.array(rz(self.SIDE_EYE[0], zc + self.SIDE_EYE[1]) + [self.SIDE_EYE[2]])).tolist()
        self.high_eye = (g + np.array(rz(pd + self.HIGH_EYE[0], zc + self.HIGH_EYE[1]) + [self.HIGH_EYE[2]])).tolist()
        self.plate_xy = g[:2] + np.array(rz(pd, zc))
        self.cut_at = None  # joint time of the first frame after the cut
        # film_wall: hidden for a camera behind it (the plate view, the high camera) - its back would fill the frame
        self.wall = stage.GetPrimAtPath(f"/World/envs/env_{E}/FilmWall") if getattr(base.cfg, "film_wall", False) else None
        self.wall_x, self.g, self.th = pd + base.cfg.flight_end_behind_plate_m, g, th

    def _focal(self, z):
        if self.focal0:
            with self.Usd.EditContext(self.stage, self.stage.GetSessionLayer()):  # the viewport camera lives there
                self.cam.GetFocalLengthAttr().Set(self.focal0 * z)

    def _view(self, which, hit=False):
        b, E = self.base, self.E
        if which in ("cf", "pv"):
            (eye, tgt), z = (self.cf, self.CF_ZOOM) if which == "cf" else (self.pv, self.PV_J_ZOOM)
        else:  # after the cut: the pitcher's pelvis, or the batted ball (the joint env's flight marker position)
            pel = (b.robot.data.root_link_pos_w[E].cpu().numpy() + [0.0, 0.0, -0.1]).tolist()
            if which == "side":
                eye, tgt, z = self.side_eye, pel, self.SIDE_ZOOM
            elif hit and self._off_batter(self.high_eye, b.j_pos[E].cpu().numpy()) > self.MIN_BATTER_DEG:
                ball = b.j_pos[E].cpu().numpy()
                eye, tgt, z = self.high_eye, ball.tolist(), self.ball_zoom(np.linalg.norm(ball[:2] - self.plate_xy))
            else:
                eye, tgt, z = self.high_eye, pel, self.HIGH_ZOOM_P
        b.sim.set_camera_view(eye=eye, target=tgt)
        self._focal(z)
        if self.wall and self.wall.IsValid():
            d = np.asarray(eye, dtype=float)[:2] - self.g[:2]
            behind = np.cos(-self.th) * d[0] - np.sin(-self.th) * d[1] > self.wall_x  # eye in the plate frame
            vis = self.UsdGeom.Imageable(self.wall).GetVisibilityAttr()
            vis.Set(self.UsdGeom.Tokens.invisible if behind else self.UsdGeom.Tokens.inherited)

    @classmethod
    def ball_zoom(cls, dist_m):
        """Zoom of the high camera on the batted ball: closer for a ball within NEAR_BALL_M of the plate (m3_s3_k12: an
        18 m ball near the mound read small at zoom 1)."""
        return cls.HIGH_ZOOM_NEAR if dist_m <= cls.NEAR_BALL_M else cls.HIGH_ZOOM_BALL

    def _off_batter(self, eye, tgt):
        """Angle (deg) between the camera's aim at tgt and the batter's pelvis, both seen from eye."""
        e = np.asarray(eye, dtype=float)
        a, c = np.asarray(tgt, dtype=float) - e, self.base.batter.data.root_pos_w[self.E].cpu().numpy().astype(float) - e
        return float(np.degrees(np.arccos(np.clip(a @ c / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-9), -1.0, 1.0))))

    def _grab(self):
        for _ in range(6):
            img = self.base.render()
            if img is not None and img.size and img[..., :3].max() > 0:
                return np.ascontiguousarray(img[..., :3]).copy()
        return None

    def step(self, t, txt="", done_t=float("nan"), hit=False):
        """After each env step (t = joint time): a frame every self.every steps; a failed grab repeats the last frame.
        done_t: the batter's done time of env k (NaN while it runs) - the cut is CUT_AFTER_S after it (None: no cut)."""
        import cv2
        k, self.k = self.k, self.k + 1
        self._pose(t)
        if k % self.every:
            return
        cut = done_t is not None and done_t == done_t and t >= done_t + self.CUT_AFTER_S
        if cut and self.cut_at is None:
            self.cut_at = t
        left, right = ("side", "high") if cut else ("cf", "pv")
        self._view(left, hit)
        self.base.render()  # one stale render after the move
        img = self._grab()
        if img is None:
            if self.last is not None:
                self.vw.append_data(self.last)
                self.n, self.rep = self.n + 1, self.rep + 1
            return
        self._view(right, hit)
        self.base.render()
        pv = self._grab()
        ball_view = cut and hit and self._off_batter(self.high_eye, self.base.j_pos[self.E].cpu().numpy()) > self.MIN_BATTER_DEG
        labels = ("from home plate",) if not cut else ("the batted ball" if ball_view else "from behind home plate",)
        if cut:
            txt = (txt + "  " if txt else "") + "play over (batter shown until the play ends)"
        for im, s in ((img, f"t = {t:.2f} s" + (f"  {txt}" if txt else "")), (pv, labels[0])):
            if im is not None:
                cv2.putText(im, s, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(im, s, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        self.last = np.concatenate([img, np.zeros_like(img) if pv is None else pv], 1)
        self.vw.append_data(self.last)
        self.n += 1

    def _pose(self, t):
        """Batter posture of env k every step (planning 14 10-28, takes): root height above the field and the tilt from
        vertical of the pelvis (root) and torso body z axes, written next to the video at close()."""
        bat = self.base.batter
        if not hasattr(self, "poses"):
            self.poses, self.torso_id = [], (bat.body_names.index("torso") if "torso" in bat.body_names else None)

        def tilt(q):  # body z axis vs world z: 1 - 2 (x^2 + y^2) for q = (w, x, y, z)
            return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0))))
        z = float(bat.data.root_pos_w[self.E, 2]) - self.base.field_z
        tp = tilt(bat.data.root_quat_w[self.E].cpu().numpy())
        tt = tilt(bat.data.body_quat_w[self.E, self.torso_id].cpu().numpy()) if self.torso_id is not None else float("nan")
        self.poses.append((round(t, 4), round(z, 4), round(tp, 2), round(tt, 2)))

    def close(self):
        self.vw.close()
        if getattr(self, "poses", None):
            out = os.path.splitext(self.path)[0] + "_batter_pose.csv"
            with open(out, "w", newline="") as f:
                f.write("t,root_z,pelvis_tilt_deg,torso_tilt_deg\n" + "".join(",".join(map(str, r)) + "\n" for r in self.poses))
        self._focal(1.0)
        print(f"[video] wrote {self.path} ({self.n} frames, {self.rep} repeated after a failed grab), env {self.E}, cut at "
              f"{'none' if self.cut_at is None else f'{self.cut_at:.2f} s'}", flush=True)
