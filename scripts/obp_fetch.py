"""Fetch selected files out of a remote OBP release zip with HTTP Range requests (no full download).

    python scripts/obp_fetch.py --list                      # list C3D names in hitting_c3d.zip
    python scripts/obp_fetch.py --top 12 --side R           # fastest right-handed swings -> data/c3d/

OBP data license: CC BY-NC-SA 4.0 + OBP restrictions (see data/obp/LICENSE-DATA.md).
"""

import argparse
import csv
import os
import struct
import urllib.request
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/hitting_c3d.zip"


def resolve(url: str) -> str:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return r.geturl()


def get_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def total_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def central_directory(url: str):
    size = total_size(url)
    tail = get_range(url, max(0, size - 70000), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # zip64
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = get_range(url, cd_off, cd_off + cd_size - 1)
    entries, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):  # zip64 extended info
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                vals, r = [], q + 4
                for flag in (usize == 0xFFFFFFFF, csize == 0xFFFFFFFF, off == 0xFFFFFFFF):
                    if flag:
                        vals.append(struct.unpack("<Q", extra[r:r + 8])[0])
                        r += 8
                it = iter(vals)
                if usize == 0xFFFFFFFF:
                    usize = next(it)
                if csize == 0xFFFFFFFF:
                    csize = next(it)
                if off == 0xFFFFFFFF:
                    off = next(it)
            q += 4 + hlen
        entries[name] = (method, csize, usize, off)
        p += 46 + nlen + elen + clen
    return entries


def fetch(url, entry) -> bytes:
    method, csize, _, off = entry
    head = get_range(url, off, off + 29)
    nlen, elen = struct.unpack("<HH", head[26:30])
    data = get_range(url, off + 30 + nlen + elen, off + 30 + nlen + elen + csize - 1)
    return zlib.decompress(data, -15) if method == 8 else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--side", default="R")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "c3d"))
    args = ap.parse_args()

    url = resolve(URL)
    entries = central_directory(url)
    c3ds = {n: e for n, e in entries.items() if n.lower().endswith(".c3d")}
    print(f"{len(c3ds)} c3d files in archive")
    if args.list:
        for n in sorted(c3ds)[:50]:
            print(" ", n)
        return

    poi = {r["session_swing"]: r for r in csv.DictReader(open(
        os.path.join(ROOT, "data", "obp", "baseball_hitting", "data", "poi", "poi_metrics.csv"), encoding="utf-8"))}
    meta = {r["session_swing"]: r for r in csv.DictReader(open(
        os.path.join(ROOT, "data", "obp", "baseball_hitting", "data", "metadata.csv"), encoding="utf-8"))}

    def speed(ss):
        try:
            return float(poi[ss]["bat_speed_mph_contact_x"])
        except (KeyError, ValueError):
            return -1.0

    cands = [ss for ss, m in meta.items() if m["hitter_side"] == args.side and speed(ss) > 0]
    cands.sort(key=speed, reverse=True)
    os.makedirs(args.out, exist_ok=True)
    picked = []
    for ss in cands:
        m = meta[ss]
        user, session = m["user"], m["session"]
        # session_swing "S_k" is the k-th swing file (by swing number) of that session
        k = int(ss.split("_")[1])
        files = sorted((n for n in c3ds if os.path.basename(n).startswith(f"{user}_{session}_")
                        and not n.endswith("_model.c3d")), key=lambda n: int(os.path.basename(n).split("_")[5]))
        if k > len(files):
            continue
        name = files[k - 1]
        dst = os.path.join(args.out, os.path.basename(name))
        if not os.path.exists(dst):
            with open(dst, "wb") as f:
                f.write(fetch(url, c3ds[name]))
        picked.append((ss, speed(ss), os.path.basename(name)))
        print(f"  {ss}: bat speed at contact {speed(ss):.1f} mph -> {os.path.basename(name)}", flush=True)
        if len(picked) >= args.top:
            break
    # static calibration trials of the picked athletes (model building / segment lengths)
    with open(os.path.join(args.out, "selected.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session_swing", "bat_speed_contact_mph", "file"])
        w.writerows(picked)


if __name__ == "__main__":
    main()
