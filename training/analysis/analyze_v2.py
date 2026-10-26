# ======================= PATHS AND SETTINGS (edit if needed) =======================
FOLDER = "results"                 # folder with the per-page score CSVs
MODELS = ["ep3", "ep4"]            # v2 epochs to compare
V1_FILES = ("v1_validation_scores.csv", "v1_test_scores.csv")   # first fine-tune, same scoring (optional)
MODEL_FILES = "v2_{model}_{split}_scores.csv"   # Kaggle names them rescore_{model}_{split}.csv
RULE = "blob3"                     # page rule used by the app: largest marked blob (prob >= 0.3), in pixels
MAX_CLEAN_FLAGGED = 0.01           # cut-off must wrongly flag at most this share of clean VALIDATION pages
# ====================================================================================
"""Choose the epoch and its cut-off on the full validation split only; then report each once on test."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r["tampered"] = r["tampered"] == "1"
        r[RULE] = float(r[RULE])
        r["pixel_f1"] = float(r["pixel_f1"]) if r["pixel_f1"] else None
    return rows


def auc(pos, neg):
    """Chance a tampered page scores above a clean one; ties count half."""
    s = np.asarray(list(pos) + list(neg), dtype=float)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(counts) - (counts - 1) / 2)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rates(rows, t):
    tam = [r for r in rows if r["tampered"]]; cln = [r for r in rows if not r["tampered"]]
    ins = [r for r in tam if r["edit"] == "insertion"]
    return (sum(r[RULE] >= t for r in tam) / len(tam), sum(r[RULE] >= t for r in cln) / len(cln),
            sum(r[RULE] >= t for r in ins) / len(ins) if ins else float("nan"))


def pick(rows):
    vals = sorted({r[RULE] for r in rows})
    best = None
    for t in vals + [vals[-1] + 1]:
        c, fa, _ = rates(rows, t)
        # most tampered caught; on a tie keep the HIGHER cut-off (fewer specks can trip it)
        if fa <= MAX_CLEAN_FLAGGED and (best is None or c > best[1] or (c == best[1] and t > best[0])):
            best = (t, c, fa)
    return best


def summary(name, val, test):
    t, vc, vfa = pick(val)
    tc, tfa, tins = rates(test, t)
    tam = [r for r in test if r["tampered"]]
    f1 = np.mean([r["pixel_f1"] for r in tam])
    a = auc([r[RULE] for r in tam], [r[RULE] for r in test if not r["tampered"]])
    n_clean = sum(not r["tampered"] for r in val)
    print(f"{name:>5} | cut-off {t:>6.0f} px | VAL caught {vc:6.1%} clean flagged {vfa:5.2%} ({round(vfa * n_clean)}/{n_clean}) || "
          f"TEST caught {tc:6.1%} clean flagged {tfa:5.2%} insertions {tins:5.1%} | pixel F1 {f1:.3f} | AUC {a:.3f}")
    return {"name": name, "cut": t, "val_caught": vc, "val_fa": vfa, "test": (tc, tfa, tins), "f1": f1, "auc": a}


folder = Path(FOLDER)
print(f"rule: {RULE}; cut-off chosen on validation with at most {MAX_CLEAN_FLAGGED:.0%} of clean pages flagged\n")
res = []
if all((folder / f).exists() for f in V1_FILES):
    res.append(summary("v1", load(folder / V1_FILES[0]), load(folder / V1_FILES[1])))
v2 = []
for m in MODELS:
    v2.append(summary(m, load(folder / MODEL_FILES.format(model=m, split="validation")), load(folder / MODEL_FILES.format(model=m, split="test"))))
# the epoch is chosen on VALIDATION only: most caught, then fewest clean flagged, then the higher cut-off
chosen = max(v2, key=lambda s: (s["val_caught"], -s["val_fa"], s["cut"]))
print(f"\nchosen on validation: {chosen['name']} with blob >= {chosen['cut']:.0f} px")
test = load(folder / MODEL_FILES.format(model=chosen["name"], split="test"))
tot, miss = Counter(), Counter()
for r in test:
    if r["tampered"]:
        tot[r["edit"]] += 1; miss[r["edit"]] += r[RULE] < chosen["cut"]
print("TEST misses by edit: " + ", ".join(f"{e} {miss[e]}/{tot[e]}" for e in sorted(tot)))
fa = sorted((r[RULE], r["doc_type"]) for r in test if not r["tampered"] and r[RULE] >= chosen["cut"])
print(f"TEST clean pages flagged ({len(fa)}): blob sizes {[int(a) for a, _ in fa]}")
print("   by document:", dict(Counter(d for _, d in fa)))
