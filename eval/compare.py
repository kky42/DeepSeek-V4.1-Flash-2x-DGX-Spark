"""Paired comparison of quality_eval.py result files that asked the same questions (same subset digest).

  python3 eval/compare.py local.json deepseek.json [deepinfra.json ...]     # the second file is the reference

Per task and thinking mode: accuracy with a 95% Wilson interval, then for each file against the reference the share of
questions with the same answer, the discordant counts (this right / reference wrong, and the reverse) and an exact
two-sided McNemar p-value on them. Standard library only.
"""
import json
import math
import sys
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


runs = [(Path(f).stem, json.load(open(f))) for f in sys.argv[1:]]
digests = {name: r["subset_digest"] for name, r in runs}
if len(set(digests.values())) != 1:
    sys.exit(f"different question subsets: {digests}")
rows = {name: {(x["task"], x["mode"], x["index"]): x for x in r["rows"]} for name, r in runs}
ref = runs[1][0] if len(runs) > 1 else runs[0][0]
groups = sorted({(t, m) for t, m, _ in rows[runs[0][0]]})

print(f"subset digest {next(iter(digests.values()))}; reference = {ref}\n")
for task, mode in groups:
    keys = sorted(k for k in rows[runs[0][0]] if k[:2] == (task, mode))
    print(f"== {task}, thinking {mode} (n={len(keys)})")
    for name, _ in runs:
        rs = [rows[name][k] for k in keys]
        k = sum(r["ok"] for r in rs)
        lo, hi = wilson(k, len(rs))
        unparsed = sum(r["pred"] is None for r in rs)
        errors = sum(bool(r.get("error")) for r in rs)
        cut = sum(r.get("finish") == "length" for r in rs)
        toks = [r.get("completion_tokens") or 0 for r in rs]
        line = (f"  {name:10} {100 * k / len(rs):5.1f}%  [{lo:4.1f}-{hi:4.1f}]  unparsed {unparsed}  length-cut {cut}  "
                f"errors {errors}  mean tokens {sum(toks) / len(toks):6.0f}")
        if name != ref:
            same = sum(rows[name][x]["pred"] == rows[ref][x]["pred"] for x in keys)
            b = sum(rows[name][x]["ok"] and not rows[ref][x]["ok"] for x in keys)
            c = sum(rows[ref][x]["ok"] and not rows[name][x]["ok"] for x in keys)
            line += f"  | vs {ref}: same answer {100 * same / len(keys):5.1f}%, +{b}/-{c}, McNemar p={mcnemar_exact(b, c):.2f}"
        print(line)
    print()
