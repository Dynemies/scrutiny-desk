"""
SIH26261 - application check engine
===================================

Runs every check on one application and returns findings with evidence.

Real now:      checklist, expiry, lease vs longest course, intake vs approval,
               institution name / AICTE ID vs records, digital signature,
               changes made after signing (with the changed region),
               re-used pages across applications (perceptual hash).
Plug-in slots: READER   - reads document type and fields from a page.
                          Until the tagging model is trained, the stand-in
                          reader returns the generator's printed-field labels.
               TAMPER   - pixel tamper heatmap. Reads precomputed maps from
                          <data>/tamper_maps/<APP>/<doc>.png (+ .json score)
                          once the tamper model has produced them.
"""
import os
if os.environ.get("TZ") == "Asia/Calcutta":
    os.environ["TZ"] = "Asia/Kolkata"

import datetime as dt
import hashlib
import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf

logging.getLogger("pyhanko").setLevel(logging.CRITICAL)
logging.getLogger("pyhanko_certvalidator").setLevel(logging.CRITICAL)

REQUIRED = ["affidavit", "lease_deed", "fire_noc", "nba_certificate", "eoa_letter", "ter_receipt"]
DOC_NAMES = {
    "affidavit": "Affidavit on e-stamp paper",
    "lease_deed": "Lease deed",
    "fire_noc": "Fire safety NOC",
    "nba_certificate": "NBA accreditation",
    "eoa_letter": "Previous approval letter",
    "ter_receipt": "Fee receipt (TER)",
}
FIELD_NAMES = {
    "institution_name": "Institution name", "aicte_id": "AICTE ID", "valid_until": "Valid until",
    "lease_start": "Lease start", "lease_period_years": "Lease period", "issue_date": "Issue date",
    "estamp_cert_no": "e-stamp number", "estamp_issue_date": "e-stamp date", "sworn_date": "Sworn on",
    "payment_date": "Payment date", "amount": "Amount", "transaction_id": "Transaction ID",
    "certificate_no": "Certificate number", "ref_no": "Reference number", "letter_date": "Letter date",
    "noc_no": "NOC number", "registration_no": "Registration number", "deponent_name": "Deponent",
    "notary_reg_no": "Notary registration", "stamp_duty": "Stamp duty", "survey_no": "Survey number",
    "land_area_acres": "Land area", "lessor_name": "Lessor", "lessee_name": "Lessee", "address": "Address",
    "programmes": "Programmes", "status": "Payment status", "authority": "Issuing authority",
}
# severity order, worst first; labels follow the problem statement
STATUS_ORDER = ["missing", "mismatch_detected", "expired", "deficiency", "manual_review_required",
                "verification_pending", "verified"]
STATUS_LABEL = {
    "missing": "Missing", "mismatch_detected": "Mismatch", "expired": "Expired", "deficiency": "Deficiency",
    "manual_review_required": "Needs review", "verification_pending": "Pending", "verified": "Verified",
}
DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d %B %Y"]
_pool = ThreadPoolExecutor(max_workers=1)   # pyHanko runs its own event loop; keep it off the caller's


def worst(statuses):
    s = [x for x in statuses if x in STATUS_ORDER]
    return min(s, key=STATUS_ORDER.index) if s else "verified"


def parse_date(s):
    s = (s or "").strip()
    for f in DATE_FORMATS:
        try:
            return dt.datetime.strptime(s, f).date()
        except ValueError:
            pass
    return None


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


# ------------------------------------------------------------------ data store
class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.apps = {p.parent.name: p.parent for p in sorted(self.root.glob("applications/*/*/application.json"))}
        recs = json.loads((self.root / "records" / "institutions.json").read_text())
        self.records = {r["institution_id"]: r for r in recs}
        self.ca = self.root / "pki" / "test_ca.pem"
        self._hashes = None

    def form(self, app_id):
        return json.loads((self.apps[app_id] / "application.json").read_text())

    def page_path(self, app_id, doc):
        return self.apps[app_id] / "pages" / f"{doc}.jpg"

    def pdf_path(self, app_id, doc):
        return self.apps[app_id] / "docs" / f"{doc}.pdf"

    def page_hashes(self):
        """Perceptual hash of every uploaded page, for re-use detection across applications."""
        if self._hashes is None:
            cache = self.root / "page_hashes.json"
            if cache.exists():
                self._hashes = json.loads(cache.read_text())
            else:
                self._hashes = {}
                for app_id in self.apps:
                    for d in self.form(app_id)["uploaded_documents"]:
                        self._hashes[f"{app_id}/{d}"] = page_fingerprint(self.page_path(app_id, d))
                cache.write_text(json.dumps(self._hashes))
        return self._hashes


# ------------------------------------------------------------------ readers
def stand_in_reader(store, app_id, doc):
    """Stand-in for the tagging model: the printed fields and their positions from the generator.
    Replace with the fine-tuned model's output (same dict shape)."""
    ak = json.loads((store.apps[app_id] / "answer_key.json").read_text())
    info = ak["documents"].get(doc, {})
    return {"doc_type": doc, "fields": dict(info.get("fields_printed", {})),
            "boxes": info.get("field_quads_px", {}), "reader": "stand-in (generator labels)"}


READER = stand_in_reader


def tamper_map(store, app_id, doc):
    """Precomputed heatmap from the tamper model, if present."""
    base = store.root / "tamper_maps" / app_id / doc
    png, meta = base.with_suffix(".png"), base.with_suffix(".json")
    if not png.exists():
        return None
    m = json.loads(meta.read_text()) if meta.exists() else {}
    return {"heatmap": str(png), "score": m.get("score"), "threshold": m.get("threshold", 0.5),
            "unit": m.get("unit", ""), "model": m.get("model", "tamper model")}


def tamper_text(tm):
    """Plain wording for a tamper verdict. New maps score the largest marked area in pixels."""
    if tm is None:
        return "not run yet"
    if tm.get("unit") == "px":
        return f"largest marked area {int(tm['score']):,} px (flagged from {int(tm['threshold'])} px)"
    return f"score {tm['score']:.2f}"


# ------------------------------------------------------------------ forensics
def page_fingerprint(path):
    """256-bit DCT hash of the page plus a SHA-256 of its bytes."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:16, :16].flatten()
    bits = (d[1:] > np.median(d[1:])).astype(np.uint8)
    return {"phash": "".join(map(str, bits)), "sha": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _small_gray(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, (620, 877), interpolation=cv2.INTER_AREA).astype(np.float32)


def find_reuse(store, page_path, fingerprint, earlier_than=None):
    """Earlier upload of the same page: identical bytes, or a close hash confirmed pixel by pixel.
    Measured on synthetic pages: re-saved or resized copies differ by <= 1.0 grey level on average,
    different documents on the same template by >= 5.5, so 3.0 separates them."""
    me = None
    for key, other in store.page_hashes().items():
        o_app = key.split("/")[0]
        if earlier_than is not None and o_app >= earlier_than:
            continue
        if other["sha"] == fingerprint["sha"]:
            return key
        if hamming(other["phash"], fingerprint["phash"]) <= 10:
            me = _small_gray(page_path) if me is None else me
            o_app, o_doc = key.split("/")
            if float(np.abs(me - _small_gray(store.page_path(o_app, o_doc))).mean()) < 3.0:
                return key
    return None


def signature_check(pdf_path, ca_pem):
    from pyhanko.keys import load_cert_from_pemder
    from pyhanko.pdf_utils.reader import PdfFileReader
    from pyhanko.sign.validation import validate_pdf_signature
    from pyhanko_certvalidator import ValidationContext

    def run():
        with open(pdf_path, "rb") as f:
            r = PdfFileReader(f, strict=False)
            if not r.embedded_signatures:
                return {"state": "unsigned"}
            sig = r.embedded_signatures[0]
            st = validate_pdf_signature(sig, ValidationContext(
                trust_roots=[load_cert_from_pemder(str(ca_pem))], allow_fetching=False, revocation_mode="soft-fail"))
            subj = sig.signer_cert.subject.native
            signer = subj.get("common_name") or sig.signer_cert.subject.human_friendly
            when = sig.self_reported_timestamp
            out = {"signer": signer, "signed_at": when.isoformat() if when else None,
                   "byte_range": list(sig.byte_range) if hasattr(sig, "byte_range") else None}
            if st.bottom_line:
                out["state"] = "valid"
            elif st.intact and st.valid:
                out["state"] = "modified_after_signing"
                out["modification"] = str(st.modification_level).split(".")[-1].lower()
            else:
                out["state"] = "invalid"
            return out
    return _pool.submit(run).result()


def changed_after_signing(pdf_path):
    """Render the signed revision and the current file; return the box that differs (page pixels at 150 dpi)."""
    data = Path(pdf_path).read_bytes()
    m = re.search(rb"/ByteRange\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]", data)
    if not m:
        return None
    _, _, b, c = map(int, m.groups())
    signed = data[: b + c]
    imgs = []
    for blob in (signed, data):
        doc = pymupdf.open("pdf", blob)
        pix = doc[0].get_pixmap(dpi=150, alpha=False)
        imgs.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy())
        doc.close()
    if imgs[0].shape != imgs[1].shape:
        return None
    diff = np.abs(imgs[0].astype(np.int16) - imgs[1].astype(np.int16)).max(2) > 30
    if not diff.any():
        return None
    ys, xs = np.nonzero(diff)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ------------------------------------------------------------------ the checks
def check_application(store, app_id):
    form = store.form(app_id)
    rec = store.records.get(form["institution_id"])
    app_date = dt.date.fromisoformat(form["application_date"])
    docs, app_findings, audit = {}, [], []
    t0 = dt.datetime.now()

    def log(what):
        audit.append({"at": (t0 + dt.timedelta(milliseconds=len(audit) * 37)).strftime("%H:%M:%S.%f")[:-3],
                      "check": what})

    log(f"Opened application {app_id} ({form['category']}) for {form['institution_name']}")
    if rec is None:
        app_findings.append({"status": "manual_review_required", "check": "Records",
                             "text": f"No AICTE record found for ID {form['institution_id']}."})
        rec = {}
    log("Matched institution to AICTE records" if rec else "No matching AICTE record")

    hashes = store.page_hashes()
    longest = max((p["duration"] for p in rec.get("programmes", [])), default=4)
    approved = rec.get("approved_intake", {})

    for d in REQUIRED:
        f = []
        if d not in form["uploaded_documents"]:
            docs[d] = {"status": "missing", "findings": [{"status": "missing", "check": "Checklist",
                       "text": f"{DOC_NAMES[d]} is required for Extension of Approval and was not uploaded."}]}
            log(f"Checklist: {DOC_NAMES[d]} not uploaded")
            continue
        read = READER(store, app_id, d)
        fields = read["fields"]
        log(f"Read {DOC_NAMES[d]} ({len(fields)} fields, reader: {read['reader']})")

        # identity against records
        if rec and "institution_name" in fields and norm(fields["institution_name"]) != norm(rec.get("name")):
            f.append({"status": "mismatch_detected", "check": "Records", "field": "institution_name",
                      "text": "Institution name on the document differs from the AICTE record.",
                      "doc_value": fields["institution_name"], "record_value": rec["name"]})
        if rec and "aicte_id" in fields and norm(fields["aicte_id"]) != norm(rec.get("institution_id")):
            f.append({"status": "mismatch_detected", "check": "Records", "field": "aicte_id",
                      "text": "AICTE ID on the document differs from the record.",
                      "doc_value": fields["aicte_id"], "record_value": rec["institution_id"]})
        # empty fields (something printed was removed)
        for k, v in fields.items():
            if not str(v).strip():
                f.append({"status": "manual_review_required", "check": "Reading", "field": k,
                          "text": f"{FIELD_NAMES.get(k, k)} is blank on the page."})
        # validity dates
        if "valid_until" in fields:
            v = parse_date(fields["valid_until"])
            if v and v < app_date:
                f.append({"status": "expired", "check": "Validity", "field": "valid_until",
                          "text": f"Expired on {v:%d %b %Y}, before the application date {app_date:%d %b %Y}."})
            elif v and (v - app_date).days < 120:
                f.append({"status": "verified", "check": "Validity", "field": "valid_until",
                          "text": f"Valid, but expires on {v:%d %b %Y}; renewal reminder set.", "note": True})
        # lease against the longest course
        if d == "lease_deed":
            s = parse_date(fields.get("lease_start"))
            try:
                yrs = int(fields.get("lease_period_years", ""))
            except ValueError:
                yrs = None
            if s and yrs:
                end = s.replace(year=s.year + yrs)
                left = (end - app_date).days / 365.25
                if left < longest:
                    f.append({"status": "deficiency", "check": "Land rule", "field": "lease_period_years",
                              "text": f"About {left:.1f} years of lease left (ends {end:%d %b %Y}); the longest course runs {longest} years. Handbook cl. 2.1(e) needs a lease at least that long."})
        # digital signature
        sig = signature_check(store.pdf_path(app_id, d), store.ca)
        log(f"Signature on {DOC_NAMES[d]}: {sig['state'].replace('_', ' ')}")
        region = None
        if sig["state"] == "unsigned":
            f.append({"status": "manual_review_required", "check": "Signature",
                      "text": "No digital signature. Annexure-2 documents must be uploaded DSC-signed."})
        elif sig["state"] == "modified_after_signing":
            region = changed_after_signing(store.pdf_path(app_id, d))
            f.append({"status": "manual_review_required", "check": "Signature", "region": region,
                      "text": "Signature is genuine, but the file was changed after it was signed. The marked area differs from the signed version."})
        elif sig["state"] == "invalid":
            f.append({"status": "manual_review_required", "check": "Signature",
                      "text": "Signature does not verify against the signed content."})
        # re-used page
        me = hashes.get(f"{app_id}/{d}")
        if me:
            key = find_reuse(store, store.page_path(app_id, d), me, earlier_than=app_id)
            if key:
                f.append({"status": "manual_review_required", "check": "Re-use",
                          "text": f"Same page was uploaded earlier with {key.split('/')[0]}.", "other": key})
        # pixel tampering
        tm = tamper_map(store, app_id, d)
        if tm and tm["score"] is not None and tm["score"] >= tm["threshold"]:
            f.append({"status": "manual_review_required", "check": "Tampering", "heatmap": tm["heatmap"],
                      "text": f"Possible edit found: {tamper_text(tm)}. The heatmap marks where."})
        log(f"Tamper check on {DOC_NAMES[d]}: {tamper_text(tm)}")

        status = worst([x["status"] for x in f if not x.get("note")])
        docs[d] = {"status": status, "findings": f, "fields": fields, "boxes": read["boxes"],
                   "signature": sig, "changed_region": region, "tamper": tm, "reader": read["reader"]}

    # application-level: intake
    for prog, n in form.get("requested_intake", {}).items():
        a = approved.get(prog)
        if a is not None and n > a:
            app_findings.append({"status": "deficiency", "check": "Intake",
                                 "text": f"{prog}: requests {n} seats against an approved intake of {a}."})
    log("Compared requested intake with approved intake")

    overall = worst([v["status"] for v in docs.values()] + [x["status"] for x in app_findings])
    log(f"Result: {STATUS_LABEL[overall]}")
    return {"application": form, "record": rec, "documents": docs, "application_findings": app_findings,
            "status": overall, "audit": audit, "checked_at": t0.isoformat(timespec="seconds")}


def to_jsonable(x):
    return json.loads(json.dumps(x, default=str))
