"""yaml_diff.py <ref env.yaml> <new env.yaml> [allowed key ...]: flat diff of two rsl_rl params/env.yaml files (H32P, 11-03).
Prints 'OK ...' or 'DIFF ...'; a key present in both with a different value is unexpected unless allowed; keys only in the new
file are listed (new cfg fields) and unexpected unless allowed or equal to 0 / False / None (off defaults)."""
import sys

import yaml


def flat(d, p=""):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{p}{k}."))
        return out
    return {p[:-1]: d}


def load(f):
    return flat(yaml.unsafe_load(open(f, encoding="utf-8")))


a, b = load(sys.argv[1]), load(sys.argv[2])
allow = set(sys.argv[3:])
leaf = lambda k: k.rsplit(".", 1)[-1]
chg = [k for k in a.keys() & b.keys() if a[k] != b[k] and leaf(k) not in allow]
new = [k for k in b.keys() - a.keys() if leaf(k) not in allow and b[k] not in (0, 0.0, False, None)]
gone = sorted(a.keys() - b.keys())
note = " ".join(f"{leaf(k)}={b[k]}" for k in sorted(b.keys() - a.keys()))
if chg or new:
    print("DIFF " + " ".join(f"{k}: {a.get(k)} -> {b[k]}" for k in sorted(chg + new)) + f" | new keys: {note}")
else:
    print(f"OK (new keys: {note}; removed: {' '.join(gone) or '-'})")
