# AIPlayBaseBall: a humanoid that hits and pitches in Isaac Sim

AIB-1 is a custom 1.85 m humanoid with two 16-DOF five-finger hands. It learns baseball skills in
Isaac Sim / Isaac Lab with physics-based ball models (drag, Magnus lift, bat-ball impulse, finger-contact grip).

| Task | Goal | Status |
|---|---|---|
| Tee batting | carry ≥ 150.3 m | ✅ 158.8 m on 16/16 swings (strong actuators) |
| Tee batting A: random tee position | adapt the swing to where the ball is | ✅ contact on every swing, average 97 m |
| Tee batting B: perception noise | 2 cm bias, 1 cm noise, 100 ms latency | ✅ contact on every swing, average 84 m |
| Batting D: pitched fastballs | hit 100–150 km/h pitches | 🔄 makes contact, mostly weak ground balls |
| Pitching | ≥ 120 km/h into a 65 × 95 cm zone at 18.44 m | 🔄 114.6 km/h, 47% strikes (human-level strength, tendon-elastic arm); falls in the follow-through |

## Environment

| Item | Version |
|---|---|
| Isaac Sim | 4.5 (pip) |
| Isaac Lab | 2.0.2 |
| RL | rsl_rl 2.2.4 (PPO) |
| GPU used | RTX 3080 Ti 12 GB |

Install Isaac Sim 4.5 and Isaac Lab 2.0.x, then run the scripts below with the Isaac Lab Python
environment (`$PY`).

## Physics (`aibaseball/physics/`)

| Part | Model |
|---|---|
| Ball | MLB spec: 145.3 g, 23.18 cm circumference |
| Bat | 34 in / 32 oz maple; mass properties integrated from the profile |
| Flight | gravity + drag + Magnus + spin decay (A. M. Nathan's fits), RK4, indoor air 1.194 kg/m³ |
| Bat-ball contact | swept test of moving bat and ball (no tunnelling at 150 km/h) + 3D rigid-body impulse with speed-dependent COR, sweet-spot vibration loss and friction (backspin from undercut) |
| Pitches | analytic library aimed with the same flight model (`pitch_gen.py`) |
| Grip | the ball is a free rigid body held only by finger contact; a momentum audit caps the release speed at the finger-pad speed (removes contact-solver ejection) |

`python tests/test_physics.py` checks flight distances against Statcast-like references and the
collision efficiency of a wood bat.

## Robot (`aibaseball/robot/`)

* 61 DOF / 63 links (fits PhysX's 64-link articulation limit): legs 2×6, waist 3, arms 2×7, hands 2×16.
* Batter: strong actuators (needed: 150 m off a tee needs ~96 mph bat speed, beyond elite humans).
* Pitcher: human-level joint torques and speeds, human arm masses (de Leva), and a tendon-like
  series elastic throwing arm (`sea.py`).

```powershell
& $PY scripts/build_robot.py                    # batter URDF + stance IK + previews
& $PY scripts/convert_robot_usd.py --headless   # batter USD (+ two-hand loop-closure grip)
& $PY scripts/build_pitcher.py --headless       # pitcher URDF/USD + four-seam grip
```

## Reference motions (not in this repository)

Swings and pitches are imitated from the Driveline **OpenBiomechanics Project** (OBP) motion-capture data
(<https://github.com/drivelineresearch/openbiomechanics>), licensed CC BY-NC-SA 4.0 with additional
restrictions for professional sports organizations. The raw data and the motions retargeted from it are
**not** included here; regenerate them locally:

```powershell
& $PY scripts/obp_fetch.py --top 12 --side R        # fastest right-handed swings -> data/c3d/
& $PY scripts/build_motion.py --all --no-preview    # retarget swings -> assets/motions/obp_*.npz
& $PY scripts/build_pitch_motion.py                 # retarget a fastball -> assets/motions/pitch_*.npz
```

(`build_pitch_motion.py` expects the fastest right-handed fastball C3D in `data/c3d_pitching/`.)

## Training (`aibaseball/tasks/`)

```powershell
& $PY scripts/train.py --task <TASK> --headless --num_envs 2048
```

| Task id | What |
|---|---|
| `AIB-TeeBatting-v0` | pure RL tee batting (reward exploits documented in the code) |
| `AIB-TeeBatting-Mimic-v0` / `-MimicPower-v0` / `-MimicLaunch-v0` | imitation of an OBP swing, then power / launch-angle stages |
| `AIB-TeeBatting-Multi-v0` | step A: 9 swings + random tee position |
| `AIB-TeeBatting-Perception-v0` | step B: perception bias / noise / latency |
| `AIB-PitchedBatting-v0` | step D: pitched balls, swing-tempo action |
| `AIB-Pitch-v0` / `AIB-PitchElastic-v0` | pitching (human strength / tendon-elastic arm) |

Evaluation and videos:

```powershell
& $PY scripts/showcase.py --headless --task launch --eval_envs 16          # tee batting statistics
& $PY scripts/showcase.py --headless --task pitched --swings 5 --fixed_cam  # batting vs pitches video
& $PY scripts/pitch_showcase.py --headless --elastic --pitches 4            # pitching video
```

Diagnostics used during development: `diag_policy.py`, `diag_release.py`, `diag_pitch_sat.py`,
`diag_batter_balance.py`, `diag_pitcher_balance.py`.

`scripts/train.py`, `scripts/play.py` and `scripts/cli_args.py` are adapted from Isaac Lab (BSD-3-Clause).
