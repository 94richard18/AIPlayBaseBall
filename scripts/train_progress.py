"""Progress and time left of a train_cycles.sh chain, from the training log and the cycle reports.

    python scripts/train_progress.py --run balance --start 19300 --cycles 4 [--step 150] [--eval_min 8]

Prints a Markdown summary and writes it to logs/progress_<run>.md.
"""
import argparse
import datetime as dt
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--start", type=int, required=True)
p.add_argument("--cycles", type=int, required=True)
p.add_argument("--step", type=int, default=150)
p.add_argument("--eval_min", type=float, default=8.0, help="report + video per cycle (min)")
p.add_argument("--log", default=os.path.join(ROOT, "logs_pitch_stand.txt"))
a = p.parse_args()

with open(a.log, encoding="utf-8", errors="ignore") as f:
    tail = f.readlines()[-6000:]
its = [int(m.group(1)) for ln in tail if (m := re.search(r"Learning iteration (\d+)/", ln))]
secs = [float(m.group(1)) + float(m.group(2)) for ln in tail
        if (m := re.search(r"collection: ([\d.]+)s, learning ([\d.]+)s", ln))]
sps = [int(m.group(1)) for ln in tail if (m := re.search(r"Computation: (\d+) steps/s", ln))]
it = its[-1] if its else a.start
sec_it = sum(secs[-20:]) / len(secs[-20:]) if secs else 20.0
end = a.start + a.cycles * a.step
now = dt.datetime.now()

rows, t = [], now
for k in range(a.cycles):
    stop = a.start + (k + 1) * a.step
    rep = os.path.join(ROOT, "logs", f"cycle_{a.run}_{stop}.txt")
    up = fall = None
    if os.path.exists(rep):
        txt = open(rep, encoding="utf-8", errors="ignore").read()
        m = re.search(r"still up 1.5 s after the release: (\d+)/(\d+); fall median ([+\-\d.]+) s", txt)
        if m:
            up, fall = f"{m.group(1)}/{m.group(2)}", m.group(3)
    if up is not None:
        rows.append((k + 1, stop, "done", "-", up, fall))
        continue
    left_it = max(0, stop - max(it, stop - a.step))
    t = t + dt.timedelta(seconds=left_it * sec_it + a.eval_min * 60)
    state = "training" if it >= stop - a.step else "waiting"
    if it >= stop:
        state = "report / video"
    rows.append((k + 1, stop, state, t.strftime("%H:%M"), "-", "-"))

left = t - now
done_it = max(0, it - a.start)
out = [f"# Training progress: {a.run} ({now:%Y-%m-%d %H:%M})", "",
       f"- iteration **{it}** of {a.start} -> {end} ({done_it}/{end - a.start}, {100 * done_it / (end - a.start):.0f}%)",
       f"- speed: {sps[-1] if sps else '?'} steps/s, {sec_it:.1f} s per iteration (last 20)",
       f"- time left: **{int(left.total_seconds() // 3600)} h {int(left.total_seconds() % 3600 // 60)} min**, "
       f"finish around **{t:%H:%M}** (report + video {a.eval_min:.0f} min per cycle included)", "",
       "| cycle | to iteration | state | expected | still up 1.5 s | fall median (s) |",
       "|---|---|---|---|---|---|"]
out += [f"| {c} | {s} | {st} | {e} | {u} | {fm} |" for c, s, st, e, u, fm in rows]
text = "\n".join(out) + "\n"
print(text)
with open(os.path.join(ROOT, "logs", f"progress_{a.run}.md"), "w", encoding="utf-8") as f:
    f.write(text)
