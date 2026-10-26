# ======================= PATHS (edit if needed) =======================
SOURCE_DIR = "../samples/test_pages"   # genuine/, tampered/, tampered/true_masks/, answer_key.csv
SAMPLES_DIR = "./demo_data/lab"        # what the Tamper lab shows as "Test pages"
# ======================================================================
"""Runs the tamper model once on the bundled test pages, so the Tamper lab opens instantly.
Needs the model (downloaded automatically on first run, about 357 MB)."""
import csv
import shutil
import sys
import time
from pathlib import Path

from tqdm import tqdm

import lab
from tamper import model as M

src, out = Path(SOURCE_DIR), Path(SAMPLES_DIR)
rows = list(csv.DictReader(open(src / "answer_key.csv")))
shutil.rmtree(out, ignore_errors=True)
(out / "pages").mkdir(parents=True); (out / "truth").mkdir()
M.ensure_weights(lambda f, t: print(f"\r{t}", end="", flush=True)); print()
M.load_model()
table = []
for r in tqdm(rows, desc="checking test pages", unit="page", file=sys.stdout):
    pid = Path(r["file"]).stem
    data = (src / r["file"]).read_bytes()
    jpg, how = M.to_jpeg_bytes(data, r["file"])
    (out / "pages" / f"{pid}.jpg").write_bytes(jpg)
    t0 = time.time()
    prob, rgb = M.predict(jpg)
    v = M.verdict(prob); v.update({"seconds": round(time.time() - t0, 1), "input": how, "size": [rgb.shape[1], rgb.shape[0]]})
    lab._save_result(out, pid, prob, v)
    tm = ""
    if r["true_mask"]:
        tm = f"truth/{pid}_mask.png"; shutil.copy(src / r["true_mask"], out / tm)
    table.append({"id": pid, "name": pid.replace("_", " ").capitalize(), "truth": r["truth"], "edit": r["edit"], "field": r["field"],
                  "doc_type": r["doc_type"], "true_mask": tm})
    print(f"  {pid}: {'EDIT FOUND' if v['flagged'] else 'no edit'} ({v['area']} px, {v['seconds']} s)", flush=True)
with open(out / "samples.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
tam = [t for t, r in zip(table, rows) if t["truth"] == "tampered"]
print("done:", out)
