"""Replay the two showcase PA plans (engine rng seeds 2081 pitcher-win, 576 walk) on two M2 row sets (logs/m2 vs
logs/m2_b030) with joint_pa's play_pa, pitch by pitch: which pitches keep their outcome."""
import csv, glob, random, re, sys
D2 = sys.argv[1] if len(sys.argv) > 1 else "logs/m2_b030"
src = open("scripts/joint_pa.py", encoding="utf-8").read().rsplit("\nmain()", 1)[0]
sys.argv = ["joint_pa"]; g = {"__file__": "scripts/joint_pa.py", "__name__": "pa"}; exec(compile(src, "joint_pa.py", "exec"), g)
def load(d):
    envs = {}
    for p in sorted(glob.glob(f"{d}/m2_s*.csv"), key=lambda p: int(re.search(r"_s(\d+)\.csv", p).group(1))):
        s = int(re.search(r"_s(\d+)\.csv", p).group(1))
        for r in csv.DictReader(open(p)):
            if g["b"](r, "handed"): r["seed"] = s; envs.setdefault(int(r["env"]), []).append(r)
    return envs
E = {d: load(d) for d in ("logs/m2", D2)}
zone = g["StrikeZone"]()
for pa in (2081, 576):
    out = {}
    for d, envs in E.items():
        log, (win, how) = g["play_pa"](envs, random.Random(pa), zone); out[d] = (log, win, how)
    (la, wa, ha), (lb, wb, hb) = out.values()
    print(f"PA {pa}: 0.055 -> {wa} by {ha} ({len(la)} pitches) | 0.030 -> {wb} by {hb} ({len(lb)} pitches) | {'SAME END' if (wa, ha) == (wb, hb) else 'END CHANGED'}")
    for i in range(max(len(la), len(lb))):
        x = la[i] if i < len(la) else None; y = lb[i] if i < len(lb) else None
        fx = lambda z: "-" if z is None else f"s{z['row']['seed']} e{z['env']} {z['balls_before']}-{z['strikes_before']} {z['expected']}" + (f" EV {g['f'](z['row']['exit_mps']):.1f} carry {g['f'](z['row']['carry_m']):.1f}" if z['expected'] in ('in_play', 'foul') else "") + (" fell_before_done" if z['swing'] and g['b'](z['row'], 'batter_fell_before_done') else "")
        print(f"  {i + 1}. {fx(x):58s} | {fx(y)}{'' if fx(x) == fx(y) else '   <- changed'}")
