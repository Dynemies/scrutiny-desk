# ======================= PATHS AND SETTINGS (edit if needed) =======================
SCORES_SEARCH = "/kaggle"     # searched for scores_ours_test_finetuned.csv (this run's /kaggle/working or an added output)
DATA_ROOT = "/kaggle/input/datasets/vamsikrishnasai05/data-genx"   # for tamper/test/index.csv (edit type of each page)
THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
# ====================================================================================
"""Which tampered test pages does the fine-tuned FFDN miss, and what does a lower threshold cost in false alarms?"""
import csv
from collections import Counter, defaultdict
from pathlib import Path
from tqdm.auto import tqdm

print("looking for the score file ...", flush=True)
scores_p = max(Path(SCORES_SEARCH).rglob("scores_ours_test_finetuned.csv"), key=lambda p: p.stat().st_mtime)
index_p = next(p for p in Path(DATA_ROOT).rglob("index.csv") if p.parent.name == "test" and p.parent.parent.name == "tamper")
print("scores:", scores_p, "\nindex :", index_p)

meta = {Path(r["image"]).name: r for r in csv.DictReader(open(index_p))}
pages = []
for r in tqdm(list(csv.DictReader(open(scores_p))), desc="matching pages", unit="page"):
    m = meta[Path(r["image"]).name]
    pages.append(dict(tampered=r["tampered"] == "1", score=float(r["score"]),
                      f1=float(r["pixel_f1"]) if r["pixel_f1"] else None,
                      edit=m["edit"] or "-", doc=m["doc_type"], field=m["field"] or "-"))
tam = [p for p in pages if p["tampered"]]
cln = [p for p in pages if not p["tampered"]]
print(f"\n{len(pages)} test pages: {len(tam)} tampered, {len(cln)} clean")

print("\n--- threshold trade-off (image score = 99.9th percentile of the heatmap) ---")
print(f"{'threshold':>9} | {'tampered caught':>15} | {'clean wrongly flagged':>21}")
for t in THRESHOLDS:
    caught = sum(p["score"] >= t for p in tam)
    fa = sum(p["score"] >= t for p in cln)
    print(f"{t:>9} | {caught:>5}/{len(tam)} = {caught / len(tam):5.1%} | {fa:>5}/{len(cln)} = {fa / len(cln):6.2%}")


def table(key, t=0.5):
    tot, miss, f1 = Counter(), Counter(), defaultdict(list)
    for p in tam:
        tot[p[key]] += 1
        miss[p[key]] += p["score"] < t
        f1[p[key]].append(p["f1"])
    print(f"\n--- missed at threshold {t}, by {key} ---")
    print(f"{key:>22} | {'pages':>5} | {'missed':>6} | {'miss rate':>9} | {'mean pixel F1':>13}")
    for k in sorted(tot, key=lambda k: -miss[k] / tot[k]):
        print(f"{k:>22} | {tot[k]:>5} | {miss[k]:>6} | {miss[k] / tot[k]:>9.1%} | {sum(f1[k]) / len(f1[k]):>13.3f}")


table("edit")
table("doc")
table("field")
print("\nclean pages with the highest scores (closest to a false alarm):")
for p in sorted(cln, key=lambda p: -p["score"])[:5]:
    print(f"   {p['doc']:>22} score {p['score']:.3f}")
