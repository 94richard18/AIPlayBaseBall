"""Flag blank (washed-out white) frames in a recorded video - the 10-22 batter video had catcher-view and ball-flight
frames that were plain white apart from the text overlay.

    python scripts/check_video_frames.py videos/a.mp4 [videos/b.mp4 ...] [--max_frac 0.02]

A frame is blank when the image below the 60-pixel caption band is almost uniformly bright: more than 97% of its
pixels above 225 in every channel. Also checks the stream is plain 8-bit 4:2:0 (High profile, yuv420p): crf 0 gave
High 4:4:4, which most players show black (10-26 mound_check2). Prints one line per video; exit code 1 if any video
exceeds --max_frac or is not yuv420p.

Empty-scene check (match_full2 10-28: clips for env k >= 1 showed only the floor grid, no robot): the share of pixels
whose grey value varies over the clip (std > 20 over every 5th frame, caption band excluded) in the fixed plate view
(right half of a side-by-side film, else the whole frame). The good k00 clip gives 3.0 %, the empty k01 / k21 0.01 %;
below --min_motion (0.1 %) the video is flagged EMPTY. The joint film (m3b) gives 0.42 %: its plate view is filled by
the batter standing still with the pitcher small behind - 0.5 % flagged it by mistake.
"""

import argparse
import subprocess
import sys

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("videos", nargs="+")
parser.add_argument("--max_frac", type=float, default=0.02, help="allowed fraction of blank frames")
parser.add_argument("--min_motion", type=float, default=0.1, help="% of plate-view pixels that must vary")
parser.add_argument("--caption_px", type=int, default=60, help="top band with the text overlay, ignored")
args = parser.parse_args()

bad = False
for path in args.videos:
    cap = cv2.VideoCapture(path)
    n, blank, first, grey = 0, 0, [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        body = frame[args.caption_px:]
        if (body.min(axis=-1) > 225).mean() > 0.97:
            blank += 1
            if len(first) < 5:
                first.append(n)
        if n % 5 == 0:
            grey.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[args.caption_px + 20:])
        n += 1
    g = np.stack(grey).astype(np.float32) if grey else np.zeros((1, 1, 1), np.float32)
    w = g.shape[2]
    motion = 100 * float((g[:, :, w // 2:] if w > 2.5 * (g.shape[1] + args.caption_px + 20) else g).std(0).__gt__(20).mean())
    empty = motion < args.min_motion
    bad |= empty
    frac = blank / max(n, 1)
    bad |= frac > args.max_frac
    import imageio_ffmpeg
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    vid = next((ln.split("Video:", 1)[1].strip() for ln in info.splitlines() if "Video:" in ln), "?")
    fmt_ok = "yuv420p" in vid and "4:4:4" not in vid
    bad |= not fmt_ok
    print(f"[frames] {path}: {n} frames, blank {blank} ({100 * frac:.1f}%)" + (f", first at {first}" if first else "")
          + f" | {vid.split(',')[0]}, {'yuv420p' if fmt_ok else 'NOT PLAYABLE: ' + vid[:60]}"
          + f" | motion {motion:.2f}%" + (" EMPTY" if empty else ""))
sys.exit(1 if bad else 0)
