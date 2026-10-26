# ======================= PATHS (edit if needed) =======================
DATA_DIR = "./demo_data"          # applications/, records/, pki/
# =====================================================================
"""Run every check once and cache the results, so demo mode opens instantly."""
import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

import engine

root = Path(DATA_DIR)
shutil.rmtree(root / "results", ignore_errors=True)
(root / "page_hashes.json").unlink(missing_ok=True)
store = engine.Store(DATA_DIR)
(root / "results").mkdir()
for app_id in tqdm(list(store.apps), desc="checking", unit="app", file=sys.stdout):
    r = engine.to_jsonable(engine.check_application(store, app_id))
    (root / "results" / f"{app_id}.json").write_text(json.dumps(r))
    print(f"  {app_id}: {engine.STATUS_LABEL[r['status']]}", flush=True)
