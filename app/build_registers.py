# ======================= PATHS (edit if needed) =======================
DATA_DIR = "./demo_data"                       # applications/<split>/<APP>/answer_key.json, records/
OUT_FILE = "./demo_data/registers/registers.json"
# =======================================================================
"""Builds SIMULATED issuer registers for the demo: what each issuing authority would hold for every document
the 14 demo applications carry (e-stamp, sub-registrar, fire NOC, NBA, AICTE approvals, payment ledger).

A register holds the document as the issuer issued it, so an edit made after issue (a repainted digit) differs
from the register. Which registers are online is set per state below; this coverage is invented for the
prototype and says nothing about real portals."""
import json
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
COVERAGE = {   # register -> {"online": [...states], "request": [...states]}; any other state = source unavailable
    "estamp": {"online": "*"},
    "payment_ledger": {"online": "*"},
    "aicte_approvals": {"online": "*"},
    "nba": {"online": "*"},
    "fire_noc": {"online": ["Kerala", "Tamil Nadu", "Karnataka", "Maharashtra", "Gujarat"]},
    "sub_registrar": {"online": ["Kerala", "Tamil Nadu", "Karnataka", "Maharashtra", "Gujarat"],
                      "request": ["Odisha", "Uttar Pradesh"]},
}

rules = json.loads((HERE / "rules" / "aicte_eoa.json").read_text())
regs = {r: {"entries": {}, "coverage": COVERAGE[r]} for r in COVERAGE}
for d, spec in rules["documents"].items():
    regs[spec["issuer"]["register"]]["name"] = spec["issuer"]["name"]

keys = sorted((HERE / DATA_DIR).glob("applications/*/*/answer_key.json"))
for ak_path in tqdm(keys, desc="registers", unit="application"):
    ak = json.loads(ak_path.read_text())
    for d, info in ak["documents"].items():
        fields = dict(info.get("fields_printed", {}))
        t = info.get("tamper")
        if t and t.get("field") in fields and "printed_before" in t:      # the issuer issued the value before the edit
            fields[t["field"]] = t["printed_before"]
        spec = rules["documents"][d]["issuer"]
        key = fields.get(spec["key"])
        if key:
            regs[spec["register"]]["entries"].setdefault(key, {"fields": fields, "issued_to": ak["application_id"]})   # first issue wins; a later copy is a re-use

out = HERE / OUT_FILE
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"_about": "SIMULATED registers for the synthetic demo data. Not real issuer data.", "registers": regs}, indent=1))
print({r: len(v["entries"]) for r, v in regs.items()})
