# ======================= PATHS (edit if needed) =======================
DATA_DIR = "./demo_data"          # generator output with answer keys
SPLITS = ["test"]                 # which splits to score ("train", "validation", "test")
# =====================================================================
"""Score the check engine against the generator's answer keys.

A planted problem counts as caught when the engine flags the same document
(any status other than verified) or, for intake, raises an application finding.
False alarms: documents the answer key marks verified that the engine flags.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

import engine

store = engine.Store(DATA_DIR)
caught, total = Counter(), Counter()
fa, clean_docs, clean_apps, clean_app_fa = 0, 0, 0, 0
apps = [a for a, p in store.apps.items() if p.parent.name in SPLITS]
for app_id in tqdm(apps, desc="scoring", unit="app", file=sys.stdout):
    ak = json.loads((store.apps[app_id] / "answer_key.json").read_text())
    r = engine.check_application(store, app_id)
    for p in ak["problems"]:
        t = p["type"]
        total[t] += 1
        if t == "intake_exceeds_approved":
            caught[t] += any(f["check"] == "Intake" for f in r["application_findings"])
        else:
            caught[t] += r["documents"][p["document"]]["status"] != "verified"
    for d, want in ak["expected_status"].items():
        if want == "verified":
            clean_docs += 1
            fa += r["documents"][d]["status"] != "verified"
    if ak["application_clean"]:
        clean_apps += 1
        clean_app_fa += r["status"] != "verified"

print(f"\n{len(apps)} applications ({', '.join(SPLITS)})")
print(f"{'problem':28} {'caught':>8} {'planted':>8} {'recall':>8}")
for t in sorted(total, key=lambda t: -total[t]):
    print(f"{t:28} {caught[t]:>8} {total[t]:>8} {caught[t] / total[t]:>8.0%}")
print(f"false alarms on clean documents: {fa} of {clean_docs} ({fa / max(clean_docs, 1):.1%})")
print(f"clean applications wrongly flagged: {clean_app_fa} of {clean_apps}")
