"""Flag blank (washed-out white) frames in a recorded video - the 10-22 batter video had catcher-view and ball-flight
frames that were plain white apart from the text overlay.

    python scripts/check_video_frames.py videos/a.mp4 [videos/b.mp4 ...] [--max_frac 0.02]

A frame is blank when the image below the 60-pixel caption band is almost uniformly bright: more than 97% of its
pixels above 225 in every channel. Prints one line per video; exit code 1 if any video exceeds --max_frac.
"""

import argparse
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
    print(f"[frames] {path}: {n} frames, blank {blank} ({100 * frac:.1f}%)" + (f", first at {first}" if first else ""))
sys.exit(1 if bad else 0)
