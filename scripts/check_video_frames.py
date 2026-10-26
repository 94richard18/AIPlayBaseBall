"""Flag blank (washed-out white) frames in a recorded video - the 10-22 batter video had catcher-view and ball-flight
frames that were plain white apart from the text overlay.

    python scripts/check_video_frames.py videos/a.mp4 [videos/b.mp4 ...] [--max_frac 0.02]

A frame is blank when the image below the 60-pixel caption band is almost uniformly bright: more than 97% of its
pixels above 225 in every channel. Also checks the stream is plain 8-bit 4:2:0 (High profile, yuv420p): crf 0 gave
High 4:4:4, which most players show black (10-26 mound_check2). Prints one line per video; exit code 1 if any video
exceeds --max_frac or is not yuv420p.
"""

import argparse
import subprocess
import sys

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("videos", nargs="+")
parser.add_argument("--max_frac", type=float, default=0.02, help="allowed fraction of blank frames")
parser.add_argument("--caption_px", type=int, default=60, help="top band with the text overlay, ignored")
args = parser.parse_args()

bad = False
for path in args.videos:
    cap = cv2.VideoCapture(path)
    n, blank, first = 0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        body = frame[args.caption_px:]
        if (body.min(axis=-1) > 225).mean() > 0.97:
            blank += 1
            if len(first) < 5:
                first.append(n)
        n += 1
    frac = blank / max(n, 1)
    bad |= frac > args.max_frac
    import imageio_ffmpeg
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    vid = next((ln.split("Video:", 1)[1].strip() for ln in info.splitlines() if "Video:" in ln), "?")
    fmt_ok = "yuv420p" in vid and "4:4:4" not in vid
    bad |= not fmt_ok
    print(f"[frames] {path}: {n} frames, blank {blank} ({100 * frac:.1f}%)" + (f", first at {first}" if first else "")
          + f" | {vid.split(',')[0]}, {'yuv420p' if fmt_ok else 'NOT PLAYABLE: ' + vid[:60]}")
sys.exit(1 if bad else 0)
