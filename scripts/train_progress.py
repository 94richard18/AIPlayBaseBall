"""Progress and time left of a train_cycles.sh chain, from the training log and the cycle reports.

    python scripts/train_progress.py --run balance --start 19300 --cycles 4 [--step 150] [--eval_min 8]
    python scripts/train_progress.py --auto --oneline   # the chain train_cycles.sh started last, one line

Prints a Markdown summary and writes it to logs/progress_<run>.md (--oneline: one status line, nothing written).
"""
import argparse
import datetime as dt
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--run")
p.add_argument("--start", type=int)
p.add_argument("--cycles", type=int)
p.add_argument("--step", type=int, default=150)
p.add_argument("--auto", action="store_true", help="run / start / cycles / step from logs/current_chain.txt")
p.add_argument("--oneline", action="store_true", help="one status line (for the status-line mod)")
p.add_argument("--eval_min", type=float, default=8.0, help="report + video per cycle (min)")
p.add_argument("--log", default=os.path.join(ROOT, "logs_pitch_stand.txt"))
a = p.parse_args()
if a.auto:
    try:
        _, a.run, start, cycles, step = open(os.path.join(ROOT, "logs", "current_chain.txt")).read().split()
        a.start, a.cycles, a.step = int(start), int(cycles), int(step)
    except (OSError, ValueError):
        print("沒有訓練記錄" if a.oneline else "no logs/current_chain.txt")
        raise SystemExit(0)

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
    stop_it = a.start + (k + 1) * a.step
    rep = os.path.join(ROOT, "logs", f"cycle_{a.run}_{stop_it}.txt")
    up = fall = None
    lat = stop = "-"
    if os.path.exists(rep):
        txt = open(rep, encoding="utf-8", errors="ignore").read()
        m = re.search(r"still up 1.5 s after the release: (\d+)/(\d+); fall median ([+\-\d.]+) s", txt)
        if m:
            up, fall = f"{m.group(1)}/{m.group(2)}", m.group(3)
        m = re.search(r"capture point sideways from the robot's own lead foot: \|offset\| median ([\d.]+) m", txt)
        lat = m.group(1) if m else "-"
        m = re.search(r"stoppable .*lead foot alone (\d+)/(\d+) \| both feet where they are (\d+)/", txt)
        stop = f"{m.group(1)} / {m.group(3)} of {m.group(2)}" if m else "-"
    if up is not None:
        rows.append((k + 1, stop_it, "done", "-", up, fall, lat, stop))
        continue
    left_it = max(0, stop_it - max(it, stop_it - a.step))
    t = t + dt.timedelta(seconds=left_it * sec_it + a.eval_min * 60)
    state = "training" if it >= stop_it - a.step else "waiting"
    if it >= stop_it:
        state = "report / video"
    rows.append((k + 1, stop_it, state, t.strftime("%H:%M"), "-", "-", "-", "-"))

left = t - now
if a.oneline:
    done_all = all(r[2] == "done" for r in rows)
    idle = (now - dt.datetime.fromtimestamp(os.path.getmtime(a.log))).total_seconds() > 300
    nxt = next((r for r in rows if r[2] != "done"), None)
    last = next((r for r in reversed(rows) if r[2] == "done"), None)
    res = f" | 上一份 {last[1]}：站穩 {last[4]}" if last else ""
    if done_all or (idle and nxt is not None and nxt[2] != "report / video"):
        print(f"訓練 {a.run}：{'已完成' if done_all else '沒有在跑'}（{it}）{res}")
    else:
        pct = 100 * max(0, it - a.start) / (end - a.start)
        what = "評估錄影中" if nxt[2] == "report / video" else f"{it}/{nxt[1]}"
        print(f"訓練 {a.run}：{what}（全程 {pct:.0f}%）| 下一份報告 {nxt[1]} 約 {nxt[3]} | 全部完成約 {t:%H:%M}{res}")
    raise SystemExit(0)
done_it = max(0, it - a.start)
out = [f"# Training progress: {a.run} ({now:%Y-%m-%d %H:%M})", "",
       f"- iteration **{it}** of {a.start} -> {end} ({done_it}/{end - a.start}, {100 * done_it / (end - a.start):.0f}%)",
       f"- speed: {sps[-1] if sps else '?'} steps/s, {sec_it:.1f} s per iteration (last 20)",
       f"- time left: **{int(left.total_seconds() // 3600)} h {int(left.total_seconds() % 3600 // 60)} min**, "
       f"finish around **{t:%H:%M}** (report + video {a.eval_min:.0f} min per cycle included)", "",
       "| cycle | to iteration | state | expected | still up 1.5 s | fall median (s) | capture point sideways from lead foot (m) "
       "| stoppable at release (lead foot / both feet) |",
       "|---|---|---|---|---|---|---|---|"]
out += [f"| {c} | {s} | {st} | {e} | {u} | {fm} | {la} | {sp} |" for c, s, st, e, u, fm, la, sp in rows]
text = "\n".join(out) + "\n"
print(text)
with open(os.path.join(ROOT, "logs", f"progress_{a.run}.md"), "w", encoding="utf-8") as f:
    f.write(text)
