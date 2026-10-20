# AIPlayBaseBall: a humanoid that hits and pitches in Isaac Sim

AIB-1 is a custom 1.85 m humanoid with two 16-DOF five-finger hands. It learns baseball skills in
Isaac Sim / Isaac Lab with physics-based ball models (drag, Magnus lift, bat-ball impulse, finger-contact grip).

| Task | Goal | Status |
|---|---|---|
| Tee batting | carry ≥ 150.3 m | ✅ 158.8 m on 16/16 swings (strong actuators) |
| Tee batting A: random tee position | adapt the swing to where the ball is | ✅ contact on every swing, average 97 m |
| Tee batting B: perception noise | 2 cm bias, 1 cm noise, 100 ms latency | ✅ contact on every swing, average 84 m |
| Batting D: pitched fastballs | hit 100–150 km/h pitches | 🔄 makes contact, mostly weak ground balls |
| Human-strength batter | swing with human joint torques / speeds | 🔄 ~18.5 m/s sweet-spot speed; next: contact quality (sweet-spot hits) |
| Pitching, stage 1 (stand through the throw) | stride, throw, release, stop and stand at rest | 🔄 stands at rest after the throw on 56/64 (identical starts) and 34/64 (random starts); release ~14.9 m/s, ~19° off line; the W1 gate needs ~24 m/s at release, so no pitch reaches the plate in the air yet |
| Game engine | 9 innings, scripted fielders / runners, replay video | ✅ CPU engine + top-view replay (`aibaseball/game/`) |

The earlier 114.6 km/h / 47 % strikes pitching result (`AIB-PitchElastic-v0`) fell in the follow-through and
could not be reproduced. Pitching restarted as small steps with "stand at rest after the throw" as the criterion;
the full history is in [`PITCHER_TRAINING_LOG.md`](PITCHER_TRAINING_LOG.md) (Traditional Chinese).

Video of the pitcher in training (2026-10-18): <https://youtu.be/swZP6mBEq0c>

## Current plan (one month, started 2026-10-15; moved one week on 2026-10-20)

Reduced scope on one PC: only the pitcher and the batter are physics RL; fielders, catcher and runners are
scripted, and the game is played back from a recording.

| Gate | Due | Content |
|---|---|---|
| W1 pitcher | 2026-11-05 (check on 10-29: release < 20 m/s → the game uses the pitch-library fallback) | both evals (identical and random starts, 4.0 s episodes): at rest with the ball crossing the plate in the air at ≥ 80 km/h (drag included) ≥ 60/64, and at rest + ≥ 80 km/h + ball tier 3 (catchable) on the same pitch ≥ 32/64; human-strength joints and the elastic arm stay as they are |
| W2 batter | 2026-11-12 | human-strength batter vs 80–100 km/h pitches (pitch library until the RL pitcher is ready), grass bounce / roll model |
| W3 engine | 2026-11-12 | rules, scripted fielders / runners, 9 innings unattended |
| W4 video | 2026-11-26 | integration, broadcast camera, full-game video |

Ball tiers, highest reached per pitch: T0 not thrown or > 10° off line, T1 release heading within ±10° of the
plate line, T2 reaches the plate in the air within 1.0 m sideways, T3 catchable (|y| ≤ 0.5 m, z 0.2–1.5 m),
T4 strike.

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
| Grip | the ball is a free rigid body held only by finger contact (four-seam grip); a momentum audit caps the release speed at the finger-pad speed (removes contact-solver ejection) |
| Grass | bounce with spin and friction, then sliding and rolling to rest (`grass.py`, after R. Cross and A. M. Nathan) |

`python tests/test_physics.py` checks flight distances against Statcast-like references and the
collision efficiency of a wood bat.

## Robot (`aibaseball/robot/`)

* 61 DOF / 63 links (fits PhysX's 64-link articulation limit): legs 2×6, waist 3, arms 2×7, hands 2×16.
* Batter: strong actuators (needed: 150 m off a tee needs ~96 mph bat speed, beyond elite humans), or
  human-level limits (`make_aib1_cfg(human_strength=True)`), used for the game.
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
| `AIB-TeeBatting-Human-v0` / `AIB-PitchedBatting-Human-v0` / `-Human2-v0` | the same with the human-strength batter |
| `AIB-Pitch-v0` / `AIB-PitchElastic-v0` | pitching (human strength / tendon-elastic arm) |
| `AIB-Stage1a-v0` … `AIB-Stage1d*-v0` | pitching in steps: 1a stand still → 1b stride and stop → 1d stride + scripted throw, stand at rest |
| `AIB-Stage1dK70GW2M-v0` | current base: PD-reference lower body + full RL residual, leg PD ×1.5, waist controller ×2, 4.0 s |
| `AIB-Stage1dK70GW2MH-v0` | + release when the hand's heading is within ±10° of the plate line |
| `AIB-Stage1dK70GW2MB-v0` | + ball reward (predicted plate crossing) for pitches that end at rest — in training |

Evaluation and videos:

```powershell
& $PY scripts/showcase.py --headless --task launch --eval_envs 16          # tee batting statistics
& $PY scripts/showcase.py --headless --task pitched --swings 5 --fixed_cam  # batting vs pitches video
& $PY scripts/pitch_showcase.py --headless --elastic --pitches 4            # pitching video
```

Pitching diagnostics, all in one simulator run (robot vs the athlete reference):

```powershell
& $PY scripts/diag_pitch.py --headless --checkpoint <model.pt>   # outcome, footing, foot strike, mechanics,
                                                                 # kinetic chain, energy flow / joint power, torque limits
& $PY scripts/analyze_ref_dynamics.py                            # is the reference itself dynamically possible?
```

Other diagnostics: `diag_policy.py`, `diag_batter_balance.py`, `diag_bat_speed.py` (batter), `diag_release.py`,
`diag_recovery.py`, `diag_fall.py`, `diag_release_window.py` (pitcher).

Pitching stage training and evaluation:

```bash
bash scripts/train_live.sh <task> <checkpoint.pt> <max_iterations> <run_name> [every=50] [goal=60]  # one run, each new checkpoint evaluated
bash scripts/train_select.sh <task> <checkpoint.pt> <best> <rounds> <prefix>                       # keep-best rounds on the random-start eval
```

```powershell
& $PY scripts/eval_stage.py --headless --task <TASK> --checkpoint <model.pt> --envs 64          # identical starts: at rest, falls, ball tiers
& $PY scripts/eval_stage.py --headless --task <TASK> --checkpoint <model.pt> --set start_joint_noise=0.02  # random starts
& $PY scripts/avg_ckpt.py <out.pt> <a.pt> <b.pt> ...                                           # average checkpoint weights
```

## Game engine (`aibaseball/game/`, CPU only)

Rules, count and base state, scripted fielders who run at the ball (fly balls on the flight model, grounders on
the grass model) and base running against the throw. The pitcher and the batter are stand-ins until the RL players
are plugged in; `--pitches` replays the RL pitcher's evaluated pitches (CSV from `eval_stage.py --pitch_csv`).

```powershell
python scripts/sim_game.py --games 10 [--pitches pitches.csv]   # line scores and outcome counts
python scripts/render_game.py --seed 0 --out videos/game_seed0.mp4  # top-view replay video
```

`scripts/train.py`, `scripts/play.py` and `scripts/cli_args.py` are adapted from Isaac Lab (BSD-3-Clause).
