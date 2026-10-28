# ======================= PATHS (edit if needed) =======================
REGISTERS_FILE = "./demo_data/registers/registers.json"     # SIMULATED issuer registers (build_registers.py)
# =======================================================================
"""Verification with the issuing authority: looks up each document's reference number in the issuer's register
and compares what the issuer holds with what the page shows.

In this prototype the registers are simulated from the synthetic data (see build_registers.py). A real deployment
replaces lookup() with calls to the authorised source (issuer API, portal, DigiLocker, QR payload), keeping the same
result shape: source, time, status, and the field-by-field comparison."""
import datetime as dt
import json
from functools import lru_cache
from pathlib import Path

from reader import edits, fold, squash

HERE = Path(__file__).resolve().parent
CODE_FIELDS = {"aicte_id", "estamp_cert_no", "registration_no", "noc_no", "certificate_no", "ref_no", "transaction_id",
               "notary_reg_no", "survey_no"}
NAME_FIELDS = {"institution_name", "lessee_name", "lessor_name", "deponent_name", "address"}


@lru_cache(maxsize=1)
def registers():
    p = HERE / REGISTERS_FILE
    return json.loads(p.read_text())["registers"] if p.exists() else {}


def _same(field, a, b, parse_date):
    if field in CODE_FIELDS:
        return fold(a) == fold(b)
    da, db = parse_date(a), parse_date(b)
    if da and db:
        return da == db
    if field in NAME_FIELDS:      # one or two misread letters in a long name are not a register mismatch
        return edits(squash(a), squash(b)) <= 2
    return squash(a) == squash(b)


def lookup(register, key):
    reg = registers().get(register, {})
    for k, v in reg.get("entries", {}).items():
        if fold(k) == fold(key):
            return v
    return None


def verify(doc_spec, fields, state, parse_date):
    """Returns (verification record, finding or None)."""
    iss = doc_spec["issuer"]
    reg = registers().get(iss["register"], {})
    cov = reg.get("coverage", {})
    now = dt.datetime.now().isoformat(timespec="seconds")
    rec = {"source": iss["name"] + " (simulated)", "register": iss["register"], "queried_at": now,
           "key_field": iss["key"], "key": fields.get(iss["key"]), "compared": []}

    def mode():
        for m in ("online", "request"):
            v = cov.get(m)
            if v == "*" or (isinstance(v, list) and state in v):
                return m
        return "unavailable"

    m = mode()
    if not reg or m == "unavailable":
        rec["result"] = "source_unavailable"
        return rec, {"status": "source_unavailable", "check": "Issuer",
                     "text": f"{iss['name']} is not available online for {state or 'this state'}. Confirm with the issuing office by hand."}
    if m == "request":
        rec["result"] = "verification_pending"
        return rec, {"status": "verification_pending", "check": "Issuer",
                     "text": f"Request sent to {iss['name']} ({state}) at {now[11:16]}; awaiting reply."}
    if not rec["key"]:
        rec["result"] = "manual_review_required"
        return rec, {"status": "manual_review_required", "check": "Issuer",
                     "text": f"{doc_spec['labels'].get(iss['key'], iss['key'])} could not be read, so the issuer could not be asked."}
    hit = lookup(iss["register"], rec["key"])
    if hit is None:
        rec["result"] = "mismatch_detected"
        return rec, {"status": "mismatch_detected", "check": "Issuer", "field": iss["key"],
                     "text": f"{doc_spec['labels'].get(iss['key'], iss['key'])} {rec['key']} is not in the {iss['name']}."}
    bad = None
    for f in iss["compare"]:
        held, shown = hit["fields"].get(f), fields.get(f)
        ok = shown is not None and held is not None and _same(f, shown, held, parse_date)
        rec["compared"].append({"field": f, "document": shown, "issuer": held, "match": ok})
        if not ok and bad is None and shown is not None:
            bad = (f, shown, held)
    if bad:
        f, shown, held = bad
        rec["result"] = "mismatch_detected"
        return rec, {"status": "mismatch_detected", "check": "Issuer", "field": f,
                     "text": f"The issuer's register differs from the page: {doc_spec['labels'].get(f, f)}.",
                     "doc_value": shown, "record_value": held}
    rec["result"] = "verified"
    return rec, None
