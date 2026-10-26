# ======================= PATHS AND SETTINGS (edit if needed) =======================
VAL_CSV = "rescore_validation.csv"
TEST_CSV = "rescore_test.csv"
MAX_CLEAN_FLAGGED = 0.01      # a rule may wrongly flag at most this share of clean VALIDATION pages
# ====================================================================================
"""Pick the page rule and its cut-off on validation only, then report it once on test."""
import csv
import sys
from collections import Counter

import numpy as np

RULES = ["p999", "peak3", "peak9", "blob5", "blob3"]


def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r["tampered"] = r["tampered"] == "1"
        for k in RULES:
            r[k] = float(r[k])
    return rows


def auc(pos, neg):
    """Chance a tampered page scores above a clean one; ties count half (many pages score exactly 0)."""
    if not pos or not neg:
        return float("nan")
    s = np.asarray(list(pos) + list(neg), dtype=float)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(counts) - (counts - 1) / 2)[inv]          # average rank for tied values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rates(rows, k, t):
    tam = [r for r in rows if r["tampered"]]; cln = [r for r in rows if not r["tampered"]]
    return sum(r[k] >= t for r in tam) / len(tam), sum(r[k] >= t for r in cln) / len(cln)


val, test = load(VAL_CSV), load(TEST_CSV)
print(f"validation {len(val)} pages, test {len(test)} pages", flush=True)
chosen = {}
print(f"\n{'rule':>6} | {'val AUC':>7} | {'cut-off (val)':>13} | {'val caught':>10} | {'val clean flagged':>17} || "
      f"{'TEST caught':>11} | {'TEST clean flagged':>18} | {'TEST AUC':>8}")
for k in RULES:
    pos = [r[k] for r in val if r["tampered"]]; neg = [r[k] for r in val if not r["tampered"]]
    best = None
    for t in sorted(set(pos + neg)) + [max(pos + neg) + 1]:   # the last cut-off flags nothing
        c, fa = rates(val, k, t)
        if fa <= MAX_CLEAN_FLAGGED and (best is None or c > best[1]):
            best = (t, c, fa)
    if k == "p999":            # the old rule as it ran, at 0.5
        best = (0.5, *rates(val, k, 0.5))
    t = best[0]; chosen[k] = t
    tc, tf = rates(test, k, t)
    tpos = [r[k] for r in test if r["tampered"]]; tneg = [r[k] for r in test if not r["tampered"]]
    print(f"{k:>6} | {auc(pos, neg):7.4f} | {t:13.4g} | {best[1]:10.1%} | {best[2]:17.2%} || "
          f"{tc:11.1%} | {tf:18.2%} | {auc(tpos, tneg):8.4f}")

for k in RULES:
    t = chosen[k]
    tot, miss = Counter(), Counter()
    for r in test:
        if r["tampered"]:
            tot[r["edit"]] += 1; miss[r["edit"]] += r[k] < t
    print(f"\nTEST misses by edit with {k} >= {t:.4g}: " + ", ".join(f"{e} {miss[e]}/{tot[e]}" for e in sorted(tot)))
