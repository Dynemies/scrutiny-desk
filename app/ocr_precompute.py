# ======================= PATHS (edit if needed) =======================
APPS_DIR = "./demo_data/applications"     # generator output: <split>/<APP id>/pages/<doc>.jpg
OCR_DIR = "./demo_data/ocr"               # OCR lines are cached here, one JSON per page
# =======================================================================
"""Runs the OCR model once on every demo page and caches the text lines, so the desk opens instantly.
Uploaded pages are read live by the same code (tamper/ocr.py)."""
import json
from pathlib import Path

from tqdm import tqdm

from tamper.ocr import ocr_lines

HERE = Path(__file__).resolve().parent
apps, out = HERE / APPS_DIR, HERE / OCR_DIR
pages = sorted(apps.glob("*/*/pages/*.jpg"))
for p in tqdm(pages, desc="OCR on demo pages", unit="page"):
    app_id, doc = p.parent.parent.name, p.stem
    dst = out / app_id / f"{doc}.json"
    if dst.exists():
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(ocr_lines(p.read_bytes()), indent=0))
print("done:", len(pages), "pages")
