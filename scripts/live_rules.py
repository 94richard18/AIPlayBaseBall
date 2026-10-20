"""Watch a train_live.sh run and apply the stop rules, so nobody has to poll it (token saving, 2026-10-20).

    python scripts/live_rules.py --run hi_syw110r5h5 --from_it 7500 [--noisy_lt 30] [--noisy_any_lt 20]
                                 [--speed_lt 15.5] [--win_noisy 34.25] [--win_speed 16.6] [--no_kill]

Reads logs/eval_<run>_<it>.txt (identical starts) and logs/eval_<run>_<it>_noise.txt (start noise 0.02) as
train_live.sh writes them. Rules (hh / hi, 2026-10-20), counted from --from_it:
  balance: two consecutive noisy scores (/64) < --noisy_lt, or any one < --noisy_any_lt
  speed:   two consecutive identical-start release speed medians < --speed_lt
A checkpoint with noisy >= --win_noisy and identical speed >= --win_speed is reported as a replacement candidate.
On a stop rule it kills the run's processes (trainer, train_live.sh, its evals) unless --no_kill. It exits when a
rule fires or the trainer is gone, and prints one line per checkpoint plus one verdict line - run it in the
background and read the output once.
"""
import argparse
import glob
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--from_it", type=int, default=0)
p.add_argument("--noisy_lt", type=float, default=30)
p.add_argument("--noisy_any_lt", type=float, default=20)
p.add_argument("--speed_lt", type=float, default=15.5)
p.add_argument("--win_noisy", type=float, default=34.25)
p.add_argument("--win_speed", type=float, default=16.6)
p.add_argument("--poll", type=int, default=60, help="seconds between file checks")
p.add_argument("--no_kill", action="store_true")
p.add_argument("--once", action="store_true",
               help="one pass over the eval files, then exit (no verdict -> exit code 2). train_live.sh's EVAL_WAIT_FREE "
                    "counts python.exe processes, so a resident watcher blocks every eval (hi, 2026-10-20) - call --once "
                    "from a bash loop (scripts/live_rules_loop.sh) instead")
a = p.parse_args()


def read(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None


def goal(text):
    m = re.search(r"^\[goal\] (\d+)/(\d+)", text, re.M)
    return int(m.group(1)) * 64 / int(m.group(2)) if m else None


def speed(text):
    m = re.search(r"release speed median ([\d.]+) m/s, heading", text)
    return float(m.group(1)) if m else None


def run_pids():
    """Process ids whose command line names the run (trainer, train_live.sh, evals of its checkpoints)."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '%s' -and "
          "$_.CommandLine -notmatch 'live_rules' } | ForEach-Object { $_.ProcessId }" % re.escape(a.run))
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.isdigit()]


def trainer_alive():
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train\\.py' -and "
          "$_.CommandLine -match '%s' } | Measure-Object | ForEach-Object { $_.Count }" % re.escape(a.run))
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


seen = {}  # it -> (identical /64, noisy /64, identical speed)
verdict = None
while verdict is None:
    for f in sorted(glob.glob(os.path.join(ROOT, "logs", f"eval_{a.run}_*_noise.txt"))):
        m = re.search(r"_(\d+)_noise\.txt$", f)
        it = int(m.group(1))
        if it in seen:
            continue
        noisy_t, ident_t = read(f), read(f.replace("_noise.txt", ".txt"))
        if not noisy_t or not ident_t or goal(noisy_t) is None or goal(ident_t) is None:
            continue  # still being written
        seen[it] = (goal(ident_t), goal(noisy_t), speed(ident_t))
        i, n, s = seen[it]
        win = " CANDIDATE" if n >= a.win_noisy and s is not None and s >= a.win_speed else ""
        print(f"[rules] {it}: identical {i:.0f}/64 noisy {n:.2f}/64 speed {s} m/s{win}", flush=True)
        its = [k for k in sorted(seen) if a.from_it <= k <= it]
        if its:
            n_last = [seen[k][1] for k in its[-2:]]
            s_last = [seen[k][2] for k in its[-2:]]
            if n_last[-1] < a.noisy_any_lt:
                verdict = f"balance stop at {it}: noisy {n_last[-1]:.2f} < {a.noisy_any_lt}"
            elif len(its) >= 2 and all(n < a.noisy_lt for n in n_last):
                verdict = f"balance stop at {it}: noisy {n_last[0]:.2f}, {n_last[1]:.2f} < {a.noisy_lt} twice"
            elif len(its) >= 2 and all(s is not None and s < a.speed_lt for s in s_last):
                verdict = f"speed stop at {it}: identical speed {s_last[0]}, {s_last[1]} < {a.speed_lt} twice"
        if verdict:
            break
    if verdict is None and a.once:
        break
    if verdict is None and not trainer_alive():
        verdict = "trainer ended (max iterations or crashed) - no rule fired"
        break
    if verdict is None:
        time.sleep(a.poll)

if verdict is None:  # --once, nothing decided yet
    raise SystemExit(2)
newest = sorted(glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "*", f"*_{a.run}", "model_*.pt")),
                key=lambda f: int(re.search(r"model_(\d+)\.pt$", f).group(1)))
if verdict.startswith(("balance", "speed")) and not a.no_kill:
    pids = run_pids()
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    verdict += f" - killed {len(pids)} processes"
    with open(os.path.join(ROOT, "logs", f"chain_{a.run}.txt"), "a", encoding="utf-8") as fh:
        fh.write(f"[rules] {verdict}\n")
cands = [it for it, (i, n, s) in seen.items() if n >= a.win_noisy and s is not None and s >= a.win_speed]
print(f"[rules] {verdict}; newest checkpoint {os.path.basename(newest[-1]) if newest else '-'} "
      f"(last evaluated {max(seen) if seen else '-'}); candidates {cands or 'none'}", flush=True)
