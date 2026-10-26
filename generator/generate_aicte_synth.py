#!/usr/bin/env python3
"""
SIH26261 - synthetic AICTE application generator
================================================

Makes fake but realistic Extension-of-Approval application files, where every
field, every planted problem and every edited pixel is known in advance.

For each fictional institution it writes:
  * scanned-style pages (JPEG) of six document types, some with Hindi lines
  * the uploaded PDF for each page, DSC-signed with a test certificate
  * planted problems: missing / expired / mismatched / unsigned / duplicate
    documents, lease shorter than the longest course, intake above approval,
    edits made after signing, and pixel tampering (5 edit types) with masks
  * an answer key per application (what is wrong, where, expected status)

Plus three ready-made training sets:
  donut/<split>/metadata.jsonl   page image -> document type + printed fields
  tamper/<split>/index.csv       authentic and tampered pages + pixel masks
  records/institutions.json      the mock "AICTE records" the checks compare to

Every page carries a small "synthetic sample" footer. All names, numbers and
institutions are invented; no real logos or emblems are drawn.

Run:   python generate_aicte_synth.py
Needs: pip install pymupdf pyhanko faker tqdm opencv-python-headless pillow numpy cryptography
"""

# =====================  PATHS AND SETTINGS - edit these  =====================
OUT_DIR = "./aicte_synth_out"        # everything is written here (wiped first)
FONT_DIR = "./fonts"                  # folder with the Noto .ttf files shipped with this script
N_APPLICATIONS = 2000                 # one application per fictional institution
START_INDEX = 0                       # first application number; for a second batch set e.g. 2000
SEED = 2026                           # each batch uses SEED + START_INDEX, so batches never repeat
PROGRESS_EVERY = 20                   # also print a plain progress line every N applications
DPI = 150                             # page resolution (A4 at 150 dpi = 1240 x 1754 px)
APPLICATION_WINDOW = ("2026-01-10", "2026-03-31")   # dates applications are filed
SPLIT = (0.8, 0.1, 0.1)               # train / validation / test, by institution
TEST_ONLY_LAYOUT = 2                  # this layout variant appears only in the test split
P_CLEAN_APPLICATION = 0.30            # share of applications with no planted problem
MAX_PROBLEMS_PER_APP = 3
P_EXTRA_TAMPER_COPY = 0.6             # chance each clean page also gets a tampered twin for the tamper set
PROBLEM_WEIGHTS = {                   # relative frequency of each planted problem
    "missing_document": 1.0,
    "expired_document": 1.2,
    "lease_insufficient": 0.6,
    "name_mismatch": 1.0,
    "id_mismatch": 0.8,
    "intake_exceeds_approved": 0.8,
    "unsigned_document": 0.8,
    "post_sign_edit": 0.8,
    "duplicate_document": 0.6,
    "tampered_image": 1.6,
}
# ===========================================================================

import os
if os.environ.get("TZ") == "Asia/Calcutta":          # tzlocal (used by pyHanko) rejects the old name
    os.environ["TZ"] = "Asia/Kolkata"

import csv
import datetime as dt
import io
import json
import logging
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from pyhanko.keys import load_cert_from_pemder
from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
from pyhanko.pdf_utils.reader import PdfFileReader
from pyhanko.sign import signers
from pyhanko.sign.validation import validate_pdf_signature
from pyhanko_certvalidator import ValidationContext

logging.getLogger("pyhanko").setLevel(logging.CRITICAL)
logging.getLogger("pyhanko_certvalidator").setLevel(logging.CRITICAL)

import sys
import time
rng = random.Random(SEED + START_INDEX)
np_rng = np.random.default_rng(SEED + START_INDEX)
fake = Faker("en_IN")
fake.seed_instance(SEED + START_INDEX)

PT2PX = DPI / 72.0
SEARCH_FLAGS = pymupdf.TEXT_DEHYPHENATE   # leaves out TEXT_PRESERVE_LIGATURES, so "ﬃ" is searched as "ffi"
A4 = pymupdf.paper_rect("a4")
FOOTER = "Synthetic sample generated for model training (SIH26261). Not a real document."

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------
STATES = [  # state, code, AICTE-style region, cities
    ("Tamil Nadu", "TN", "Southern", ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli", "Vellore"]),
    ("Andhra Pradesh", "AP", "South-Central", ["Tirupati", "Guntur", "Nellore", "Kurnool", "Kakinada", "Anantapur"]),
    ("Telangana", "TS", "South-Central", ["Hyderabad", "Warangal", "Karimnagar", "Khammam", "Nizamabad"]),
    ("Karnataka", "KA", "South-West", ["Bengaluru", "Mysuru", "Mangaluru", "Belagavi", "Tumakuru"]),
    ("Kerala", "KL", "South-West", ["Kochi", "Thrissur", "Kozhikode", "Kollam", "Palakkad"]),
    ("Maharashtra", "MH", "Western", ["Pune", "Nagpur", "Nashik", "Aurangabad", "Kolhapur"]),
    ("Gujarat", "GJ", "Western", ["Ahmedabad", "Vadodara", "Surat", "Rajkot", "Anand"]),
    ("Madhya Pradesh", "MP", "Central", ["Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain"]),
    ("Uttar Pradesh", "UP", "Northern", ["Lucknow", "Kanpur", "Noida", "Varanasi", "Prayagraj"]),
    ("Punjab", "PB", "North-West", ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Mohali"]),
    ("Haryana", "HR", "North-West", ["Gurugram", "Faridabad", "Rohtak", "Hisar", "Karnal"]),
    ("West Bengal", "WB", "Eastern", ["Kolkata", "Durgapur", "Siliguri", "Asansol", "Howrah"]),
    ("Odisha", "OD", "Eastern", ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur"]),
    ("Rajasthan", "RJ", "North-West", ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"]),
]
PROGRAMMES = [  # name, duration in years, possible intakes
    ("B.Tech Computer Science and Engineering", 4, [60, 120, 180, 240]),
    ("B.Tech Electronics and Communication Engineering", 4, [60, 120, 180]),
    ("B.Tech Mechanical Engineering", 4, [30, 60, 120]),
    ("B.Tech Civil Engineering", 4, [30, 60, 120]),
    ("B.Tech Electrical and Electronics Engineering", 4, [30, 60, 120]),
    ("B.Tech Artificial Intelligence and Data Science", 4, [60, 120]),
    ("M.Tech VLSI Design", 2, [18, 24]),
    ("M.Tech Structural Engineering", 2, [18, 24]),
    ("MBA", 2, [60, 120, 180]),
    ("MCA", 2, [60, 120]),
    ("Diploma in Mechanical Engineering", 3, [60, 120]),
]
NAME_PREFIX = ["Sri", "Shri", "Sant", "Vidya", "Gyan", "Prerana", "Navodaya", "Sanjeevani", "Adarsh", "Pragati",
               "Siddhartha", "Vivekananda", "Sarada", "Kalpana", "Bharati", "Jnana", "Aditya", "Sahyadri"]
NAME_KIND = ["Institute of Technology", "College of Engineering", "Institute of Engineering and Technology",
             "College of Engineering and Technology", "Institute of Technology and Management"]
DOC_TYPES = ["affidavit", "lease_deed", "fire_noc", "nba_certificate", "eoa_letter", "ter_receipt"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "long"]

# Problem -> expected document / application status (labels follow the brief)
STATUS_FOR = {
    "missing_document": "missing", "expired_document": "expired", "lease_insufficient": "deficiency",
    "name_mismatch": "mismatch_detected", "id_mismatch": "mismatch_detected",
    "intake_exceeds_approved": "deficiency", "unsigned_document": "manual_review_required",
    "post_sign_edit": "manual_review_required", "duplicate_document": "manual_review_required",
    "tampered_image": "manual_review_required",
}


def fmt_date(d, style):
    if style == "long":
        return f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}"
    return d.strftime(style)


def fmt_inr(n):
    s = str(int(n))
    if len(s) <= 3:
        return "Rs. " + s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "Rs. " + ",".join(groups + [tail])


def rand_date(a, b):
    return a + dt.timedelta(days=rng.randint(0, (b - a).days))


def digits(n):
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


# --------------------------------------------------------------------------
# Mock institutions and the "AICTE records"
# --------------------------------------------------------------------------
def make_institution(i, app_date):
    state, code, region, cities = rng.choice(STATES)
    city = rng.choice(cities)
    surname = fake.last_name()
    name = f"{rng.choice(NAME_PREFIX)} {surname} {rng.choice(NAME_KIND)}, {city}"
    progs = rng.sample(PROGRAMMES, rng.randint(3, 6))
    programmes = [{"programme": p, "duration": dur, "approved_intake": rng.choice(intakes)} for p, dur, intakes in progs]
    longest = max(p["duration"] for p in programmes)
    while True:   # an unplanted lease must outlast the longest course, or the answer key would miss a real deficiency
        lease_start = dt.date(rng.randint(1996, 2018), rng.randint(1, 12), rng.randint(1, 28))
        lease_years = rng.choice([30, 33, 50, 99])
        if (dt.date(lease_start.year + lease_years, lease_start.month, lease_start.day) - app_date).days / 365.25 >= longest + 1:
            break
    return {
        "institution_id": f"1-{rng.randint(10**9, 10**10 - 1)}",
        "name": name,
        "society": f"{surname} Educational {rng.choice(['Trust', 'Society'])}",
        "address": f"{rng.randint(1, 450)}, {fake.street_name()}, {city}, {state} - {rng.randint(500000, 799999)}",
        "city": city, "state": state, "state_code": code, "region": region,
        "principal": fake.name(),
        "programmes": programmes,
        "longest_course_years": longest,
        "lease": {"start": lease_start.isoformat(), "years": lease_years,
                  "survey_no": f"{rng.randint(10, 999)}/{rng.randint(1, 9)}{rng.choice('ABC')}",
                  "area_acres": round(rng.uniform(5.0, 30.0), 2),
                  "registration_no": f"{rng.randint(1000, 9999)}/{lease_start.year}",
                  "lessor": fake.name()},
        "prev_eoa_ref": f"F.No.{region}/{rng.randint(1, 9)}-{digits(10)}/{app_date.year - 1}/EOA",
        "prev_eoa_date": dt.date(app_date.year - 1, rng.randint(5, 7), rng.randint(1, 28)).isoformat(),
    }


# --------------------------------------------------------------------------
# Test DSC certificates (self-made CA; never use for anything real)
# --------------------------------------------------------------------------
def _name(cn, org=None):
    attrs = [x509.NameAttribute(NameOID.COMMON_NAME, cn)]
    if org:
        attrs.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, org[:64]))
    return x509.Name(attrs)


def make_ca(pki_dir):
    key = ec.generate_private_key(ec.SECP256R1())
    t0 = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(_name("SIH26261 Test DSC CA"))
            .issuer_name(_name("SIH26261 Test DSC CA")).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(t0)
            .not_valid_after(t0 + dt.timedelta(days=3650 * 2))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
            .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, False, False), True)
            .sign(key, hashes.SHA256()))
    ca_pem = pki_dir / "test_ca.pem"
    ca_pem.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return key, cert, ca_pem


def make_signer(pki_dir, ca_key, ca_cert, ca_pem, inst):
    key = ec.generate_private_key(ec.SECP256R1())
    t0 = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(_name(inst["principal"], inst["name"]))
            .issuer_name(ca_cert.subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(t0)
            .not_valid_after(t0 + dt.timedelta(days=365 * 5))
            .add_extension(x509.KeyUsage(True, True, False, False, False, False, False, False, False), True)
            .sign(ca_key, hashes.SHA256()))
    kp = pki_dir / f"{inst['institution_id']}_key.pem"
    cp = pki_dir / f"{inst['institution_id']}_cert.pem"
    kp.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                     serialization.NoEncryption()))
    cp.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return signers.SimpleSigner.load(str(kp), str(cp), ca_chain_files=(str(ca_pem),))


# --------------------------------------------------------------------------
# Document content: printed fields for each document type
# --------------------------------------------------------------------------
# field kinds drive tampering (what a plausible replacement looks like)
KINDS = {
    "estamp_cert_no": "id", "estamp_issue_date": "date", "stamp_duty": "amount", "institution_name": "name",
    "aicte_id": "id", "deponent_name": "person", "notary_reg_no": "id", "sworn_date": "date",
    "lessor_name": "person", "lessee_name": "name", "survey_no": "id", "land_area_acres": "number",
    "lease_start": "date", "lease_period_years": "number", "registration_no": "id",
    "noc_no": "id", "issue_date": "date", "valid_until": "date", "authority": "text", "address": "text",
    "certificate_no": "id", "programmes": "text", "ref_no": "id", "letter_date": "date",
    "transaction_id": "id", "amount": "amount", "payment_date": "date", "status": "text",
}
TAMPERABLE = {"id", "date", "amount", "number", "person"}
FIELD_FACT = {"estamp_issue_date": "issue_date", "sworn_date": "sworn_date", "lease_start": "lease_start",
              "issue_date": "issue_date", "valid_until": "valid_until", "letter_date": "letter_date",
              "payment_date": "payment_date"}   # printed date field -> its ISO fact


def doc_fields(doc_type, inst, app_date, datefmt, override):
    """Return (printed fields dict, true ISO facts dict). `override` holds planted changes."""
    nm = override.get("institution_name", inst["name"])
    aid = override.get("aicte_id", inst["institution_id"])
    f, true = {}, {}
    if doc_type == "affidavit":
        issue = app_date - dt.timedelta(days=rng.randint(3, 25))
        sworn = issue + dt.timedelta(days=rng.randint(0, 2))
        f = {"estamp_cert_no": f"IN-{inst['state_code']}{digits(14)}{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}",
             "estamp_issue_date": fmt_date(issue, datefmt), "stamp_duty": "Rs. 100",
             "institution_name": nm, "aicte_id": aid, "deponent_name": inst["principal"],
             "notary_reg_no": f"{inst['state_code']}/NOT/{rng.randint(1000, 9999)}/{rng.randint(2010, 2024)}",
             "sworn_date": fmt_date(sworn, datefmt)}
        true = {"issue_date": issue.isoformat(), "sworn_date": sworn.isoformat()}
    elif doc_type == "lease_deed":
        L = inst["lease"]
        start = dt.date.fromisoformat(override.get("lease_start", L["start"]))
        years = override.get("lease_years", L["years"])
        f = {"lessor_name": L["lessor"], "lessee_name": inst["society"], "institution_name": nm,
             "survey_no": L["survey_no"], "land_area_acres": f"{L['area_acres']:.2f}",
             "lease_start": fmt_date(start, datefmt), "lease_period_years": str(years),
             "registration_no": L["registration_no"]}
        end = dt.date(start.year + years, start.month, start.day)
        true = {"lease_start": start.isoformat(), "lease_years": years, "lease_end": end.isoformat(),
                "remaining_years": round((end - app_date).days / 365.25, 2)}
    elif doc_type == "fire_noc":
        issue = dt.date.fromisoformat(override["fire_issue"]) if "fire_issue" in override else \
            app_date - dt.timedelta(days=rng.randint(60, 500))
        valid = dt.date.fromisoformat(override["fire_valid"]) if "fire_valid" in override else \
            app_date + dt.timedelta(days=rng.randint(200, 900))
        f = {"noc_no": f"FRS/{inst['state_code']}/{issue.year}/{digits(5)}", "institution_name": nm,
             "address": inst["address"], "issue_date": fmt_date(issue, datefmt),
             "valid_until": fmt_date(valid, datefmt),
             "authority": f"Directorate of Fire and Rescue Services, {inst['state']}"}
        true = {"issue_date": issue.isoformat(), "valid_until": valid.isoformat()}
    elif doc_type == "nba_certificate":
        progs = [p["programme"] for p in inst["programmes"] if p["programme"].startswith("B.Tech")][:3] or \
                [inst["programmes"][0]["programme"]]
        valid = dt.date.fromisoformat(override["nba_valid"]) if "nba_valid" in override else \
            dt.date(app_date.year + rng.randint(1, 3), 6, 30)
        f = {"certificate_no": f"NBA/ACCR/{rng.randint(1000, 9999)}/{app_date.year - rng.randint(1, 3)}",
             "institution_name": nm, "programmes": "; ".join(progs), "valid_until": fmt_date(valid, datefmt)}
        true = {"valid_until": valid.isoformat(), "programmes": progs}
    elif doc_type == "eoa_letter":
        d = dt.date.fromisoformat(inst["prev_eoa_date"])
        f = {"ref_no": inst["prev_eoa_ref"], "letter_date": fmt_date(d, datefmt), "institution_name": nm,
             "aicte_id": aid}
        true = {"letter_date": d.isoformat(),
                "approved_intake": {p["programme"]: p["approved_intake"] for p in inst["programmes"]}}
    elif doc_type == "ter_receipt":
        pay = app_date - dt.timedelta(days=rng.randint(1, 20))
        amt = 50000 + 15000 * len(inst["programmes"])
        f = {"transaction_id": f"TXN{digits(12)}", "institution_name": nm, "aicte_id": aid,
             "amount": fmt_inr(amt), "payment_date": fmt_date(pay, datefmt), "status": "SUCCESS"}
        true = {"payment_date": pay.isoformat(), "amount": amt}
    return f, true


# --------------------------------------------------------------------------
# HTML layouts (three variants) and vector rendering
# --------------------------------------------------------------------------
TITLES = {
    "affidavit": ("AFFIDAVIT", "शपथ पत्र"),
    "lease_deed": ("REGISTERED LEASE DEED (EXTRACT)", "पट्टा विलेख"),
    "fire_noc": ("FIRE SAFETY NO OBJECTION CERTIFICATE", "अग्नि सुरक्षा अनापत्ति प्रमाण पत्र"),
    "nba_certificate": ("CERTIFICATE OF ACCREDITATION", None),
    "eoa_letter": ("EXTENSION OF APPROVAL", "अनुमोदन का विस्तार"),
    "ter_receipt": ("PAYMENT RECEIPT - TER CHARGES", None),
}
ISSUERS = {
    "affidavit": None,
    "lease_deed": "Office of the Sub-Registrar",
    "fire_noc": None,  # filled from the authority field
    "nba_certificate": "National Board of Accreditation (synthetic sample)",
    "eoa_letter": "All India Council for Technical Education (synthetic sample)",
    "ter_receipt": "AICTE Online Payment Gateway (synthetic sample)",
}
LABELS = {
    "estamp_cert_no": "Certificate No.", "estamp_issue_date": "Certificate Issued Date", "stamp_duty": "Stamp Duty Amount",
    "institution_name": "Name of the Institution", "aicte_id": "AICTE Permanent ID", "deponent_name": "Deponent",
    "notary_reg_no": "Notary Registration No.", "sworn_date": "Sworn on", "lessor_name": "Lessor",
    "lessee_name": "Lessee", "survey_no": "Survey No.", "land_area_acres": "Land Area (acres)",
    "lease_start": "Lease Commencement", "lease_period_years": "Lease Period (years)",
    "registration_no": "Registration No.", "noc_no": "NOC No.", "address": "Address of Premises",
    "issue_date": "Date of Issue", "valid_until": "Valid Until", "authority": "Issuing Authority",
    "certificate_no": "Certificate No.", "programmes": "Programmes Accredited", "ref_no": "Ref. No.",
    "letter_date": "Date", "transaction_id": "Transaction ID", "amount": "Amount Paid",
    "payment_date": "Payment Date", "status": "Payment Status",
}
BODY = {
    "affidavit": ["I, the deponent named above, Principal of the institution, do hereby solemnly affirm that all "
                  "information furnished in the application for Extension of Approval is true and correct, and that "
                  "the institution complies with the norms laid down in the Approval Process Handbook.",
                  "HI:मैं, उपर्युक्त नामित अभिसाक्षी, सत्यनिष्ठा से प्रतिज्ञान करता हूँ कि आवेदन में दी गई सभी जानकारी सत्य एवं सही है।"],
    "lease_deed": ["This extract certifies that the lease described below has been registered in this office and "
                   "is in force for the land on which the institution is situated."],
    "fire_noc": ["Certified that the premises described below have been inspected and the fire prevention and "
                 "fire safety measures were found adequate on the date of inspection.",
                 "HI:यह प्रमाणित किया जाता है कि उपर्युक्त परिसर का निरीक्षण किया गया है।"],
    "nba_certificate": ["This is to certify that the programmes listed below offered by the institution have been "
                        "accredited for the period indicated."],
    "eoa_letter": ["With reference to the application of the institution, Extension of Approval is granted for "
                   "the academic year with the approved intake shown below, subject to the conditions of the "
                   "Approval Process Handbook."],
    "ter_receipt": ["Received the Technical Education Regulatory charges as per the details below."],
}
CONDITIONS = {
    "affidavit": ["I further affirm that no part of the land or building is used for any purpose other than "
                  "technical education, and that the institution shall abide by all regulations notified by the Council.",
                  "Verified at the place and on the date stated below that the contents of this affidavit are true "
                  "to the best of my knowledge and belief and nothing material has been concealed."],
    "lease_deed": ["The lessee shall use the land only for running educational institutions.",
                   "The lease shall not be transferred or sub-let without the written consent of the lessor.",
                   "Stamp duty and registration fees have been paid as per the applicable schedule."],
    "fire_noc": ["This certificate is valid only for the premises and the date range stated above.",
                 "Fire extinguishers, hydrants and alarms shall be maintained in working order at all times.",
                 "Any change in the use or structure of the building requires a fresh certificate.",
                 "The certificate must be renewed before the date of expiry."],
    "nba_certificate": ["Accreditation applies only to the programmes named above and only for the period stated.",
                        "The institution shall inform the Board of any major change affecting the programmes."],
    "eoa_letter": ["The institution shall display this approval on its website and notice board.",
                   "Admissions beyond the approved intake are not permitted.",
                   "The approval is liable to be withdrawn if any information furnished is found to be false.",
                   "All undertakings given in the affidavit shall be complied with within the stipulated time."],
    "ter_receipt": ["This is a system-generated receipt for the payment recorded above.",
                    "Please quote the transaction ID in all correspondence."],
}
SIGN_LABEL = {"affidavit": "Signature of Deponent", "lease_deed": "Sub-Registrar", "fire_noc": "Fire Officer",
              "nba_certificate": "Member Secretary", "eoa_letter": "Regional Officer",
              "ter_receipt": "Authorised Signatory"}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html(doc_type, fields, true, inst, layout):
    title, title_hi = TITLES[doc_type]
    issuer = ISSUERS[doc_type] or (fields.get("authority") if doc_type == "fire_noc" else None)
    fb = "<b>" if layout in (0, 2) else ""
    fbe = "</b>" if fb else ""

    def val(k):
        return f"{fb}{esc(fields[k])}{fbe}"

    def para(t):
        if t.startswith("HI:"):
            return f"<p class='hi'>{esc(t[3:])}</p>"
        return f"<p>{esc(t)}</p>"

    keys = [k for k in fields if not (doc_type == "fire_noc" and k == "authority")]
    out = []
    if doc_type == "affidavit":  # e-stamp block on top
        out.append("<table class='estamp'><tr><td colspan='2' class='c'><span class='hi'>गैर न्यायिक</span> "
                   "NON JUDICIAL &nbsp; e-Stamp</td></tr>"
                   f"<tr><td>Certificate No.</td><td>{val('estamp_cert_no')}</td></tr>"
                   f"<tr><td>Certificate Issued Date</td><td>{val('estamp_issue_date')}</td></tr>"
                   f"<tr><td>Stamp Duty Amount</td><td>{val('stamp_duty')}</td></tr>"
                   f"<tr><td>Purchased by</td><td>{esc(inst['society'])}</td></tr></table>")
        keys = [k for k in keys if not k.startswith("estamp") and k != "stamp_duty"]
    if issuer:
        out.append(f"<p class='issuer'>{esc(issuer)}</p>")
    h = esc(title) + (f" / <span class='hi'>{esc(title_hi)}</span>" if title_hi and layout != 1 else "")
    out.append(f"<h2>{h}</h2>")
    if title_hi and layout == 1:
        out.append(f"<p class='hi c'>{esc(title_hi)}</p>")
    for t in BODY[doc_type]:
        if t.startswith("HI:") and layout == 1:
            continue
        out.append(para(t))
    if layout == 0:
        for k in keys:
            out.append(f"<p>{esc(LABELS[k])}: {val(k)}</p>")
    else:
        cls = "kv" if layout == 1 else "kv2"
        rows = "".join(f"<tr><td class='lab'>{esc(LABELS[k] if layout == 1 else LABELS[k].upper())}</td>"
                       f"<td>{val(k)}</td></tr>" for k in keys)
        out.append(f"<table class='{cls}'>{rows}</table>")
    if doc_type == "eoa_letter":
        rows = "".join(f"<tr><td>{esc(p)}</td><td>{n}</td></tr>" for p, n in true["approved_intake"].items())
        out.append(f"<table class='kv'><tr><td><b>Programme</b></td><td><b>Approved Intake</b></td></tr>{rows}</table>")
    if layout == 1:
        out.append("<p><b>Conditions</b></p><ol>" + "".join(f"<li>{esc(c)}</li>" for c in CONDITIONS[doc_type]) + "</ol>")
    else:
        out.append("<p><b>Conditions:</b></p>" + "".join(f"<p>{n}. {esc(c)}</p>" for n, c in enumerate(CONDITIONS[doc_type], 1)))
    out.append("<p>&nbsp;</p><p>&nbsp;</p><p>&nbsp;</p><p>&nbsp;</p>")
    out.append(f"<p class='sig'>{esc(SIGN_LABEL[doc_type])}</p>")
    return "\n".join(out)


def css_for(layout):
    lat = "notoserif" if layout in (0, 2) else "notosans"
    dev = "notoserifdev" if layout in (0, 2) else "notosansdev"
    size = {0: 11, 1: 10.5, 2: 10}[layout]
    faces = "".join(
        f"@font-face {{font-family: {fam}; src: url({fn}-Regular.ttf);}}"
        f"@font-face {{font-family: {fam}; font-weight: bold; src: url({fn}-Bold.ttf);}}"
        for fam, fn in [("notoserif", "NotoSerif"), ("notosans", "NotoSans"),
                        ("notoserifdev", "NotoSerifDevanagari"), ("notosansdev", "NotoSansDevanagari")])
    h2 = {0: "text-align:center;", 1: "text-align:left;", 2: "text-align:center; text-decoration:underline;"}[layout]
    return (faces + f"body {{font-family: {lat}; font-size: {size}pt; line-height: 1.45;}}"
            f".hi {{font-family: {dev};}} h2 {{font-size: {size + 3}pt; {h2} margin: 6pt 0 8pt 0;}}"
            ".c {text-align:center;} .issuer {text-align:center; font-weight:bold; margin:0 0 4pt 0;}"
            ".sig {text-align:right;} p {margin: 0 0 6pt 0; text-align: justify;}"
            "table {border-collapse: collapse; margin: 6pt 0 8pt 0;}"
            "table.kv td {border: 0.6pt solid #444; padding: 3pt 6pt;}"
            "table.kv2 td {padding: 3pt 8pt;} td.lab {width: 170pt;}"
            "table.estamp {width: 100%; background-color: #eef3e6;} table.estamp td {border: 0.6pt solid #6a7f4f; padding: 2pt 6pt;}")


def render_vector(doc_type, fields, true, inst, layout):
    html = build_html(doc_type, fields, true, inst, layout)
    story = pymupdf.Story(html=html, user_css=css_for(layout), archive=pymupdf.Archive(FONT_DIR))
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    margin = {0: 60, 1: 54, 2: 70}[layout]
    where = A4 + (margin, margin + 10, -margin, -margin - 20)
    more = 1
    while more:
        dev = writer.begin_page(A4)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    doc = pymupdf.open("pdf", buf.getvalue())
    page = doc[0]
    if layout == 2:
        page.draw_rect(A4 + (40, 40, -40, -40), color=(0.15, 0.15, 0.15), width=1.2)
    decorate(page, doc_type, inst)
    page.insert_text((60, A4.height - 26), FOOTER, fontsize=6.5, color=(0.45, 0.45, 0.45))
    return doc, page


def decorate(page, doc_type, inst):
    """Draw a seal and an ink signature above the signature label."""
    hits = page.search_for(SIGN_LABEL[doc_type], flags=SEARCH_FLAGS)
    if not hits:
        return
    r = hits[-1]
    # signature scribble
    x0, y = r.x0 - 10, r.y0 - 12
    ink = (0.08, 0.14, 0.55)
    pts = [pymupdf.Point(x0 + i * rng.uniform(9, 15), y + rng.uniform(-9, 5)) for i in range(7)]
    for a, b in zip(pts[:-1], pts[1:]):
        c1 = pymupdf.Point(a.x + rng.uniform(2, 8), a.y + rng.uniform(-10, 10))
        c2 = pymupdf.Point(b.x - rng.uniform(2, 8), b.y + rng.uniform(-10, 10))
        page.draw_bezier(a, c1, c2, b, color=ink, width=rng.uniform(0.8, 1.3))
    # seal
    cx, cy = r.x0 - 120, r.y0 - 22
    col = rng.choice([(0.45, 0.12, 0.55), (0.12, 0.2, 0.6), (0.6, 0.1, 0.15)])
    page.draw_circle((cx, cy), 32, color=col, width=1.4)
    page.draw_circle((cx, cy), 25, color=col, width=0.6)
    label = {"affidavit": "NOTARY", "lease_deed": "SUB-REGISTRAR", "fire_noc": "FIRE SERVICES",
             "nba_certificate": "ACCREDITED", "eoa_letter": "REGIONAL\nOFFICE", "ter_receipt": "PAID"}[doc_type]
    for fs in (5.5, 5.0, 4.5, 4.0):   # insert_textbox draws nothing when the text does not fit
        if page.insert_textbox(pymupdf.Rect(cx - 24, cy - 12, cx + 24, cy + 14), f"{label}\n{inst['state_code']}",
                               fontsize=fs, color=col, align=1) >= 0:
            break


def rasterize(page):
    pix = page.get_pixmap(dpi=DPI, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy()


def field_boxes(page, fields):
    """Pixel boxes of each printed field value (only values found exactly once on one line)."""
    boxes = {}
    for k, v in fields.items():
        s = str(v)
        if len(s) < 3:
            continue
        hits = page.search_for(s, flags=SEARCH_FLAGS)
        if len(hits) == 1:
            r = hits[0]
            boxes[k] = [r.x0 * PT2PX, r.y0 * PT2PX, r.x1 * PT2PX, r.y1 * PT2PX]
    return boxes


# --------------------------------------------------------------------------
# Scan simulation and JPEG with varied quantization tables
# --------------------------------------------------------------------------
STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
STD_CHROMA = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
              47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32


def qtables(quality, jitter):
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    out = []
    for base in (STD_LUMA, STD_CHROMA):
        t = []
        for q in base:
            v = (q * scale + 50) / 100
            if jitter:
                v *= rng.uniform(0.85, 1.15)
            t.append(int(min(255, max(1, round(v)))))
        out.append(t)
    return out


def jpeg_bytes(img, quality=None, jitter=None):
    quality = quality or rng.randint(70, 96)
    jitter = rng.random() < 0.5 if jitter is None else jitter
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", qtables=qtables(quality, jitter), subsampling=rng.choice([0, 2]))
    return buf.getvalue(), quality


def jpeg_roundtrip(img):
    b, q = jpeg_bytes(img)
    return np.array(Image.open(io.BytesIO(b)).convert("RGB")), q


def scan_sim(img, boxes):
    h, w = img.shape[:2]
    angle = rng.uniform(-1.2, 1.2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += rng.uniform(-6, 6)
    M[1, 2] += rng.uniform(-6, 6)
    base = rng.randint(240, 252)
    paper = np.array([base, base - rng.randint(0, 3), base - rng.randint(2, 10)], np.float32)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=paper.tolist())
    out = out.astype(np.float32) * (paper / 255.0)                          # paper tint
    out = out * rng.uniform(0.95, 1.03) + rng.uniform(-6, 4)                # brightness / contrast
    out += np_rng.normal(0, rng.uniform(0.5, 4.0), out.shape)               # sensor noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 0.8))
    quads = {}
    for k, (x0, y0, x1, y1) in boxes.items():
        pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]) @ M.T
        quads[k] = pts.tolist()
    return out, quads, angle, M


def quad_bbox(q, pad=0, shape=None):
    q = np.array(q)
    x0, y0 = q.min(0) - pad
    x1, y1 = q.max(0) + pad
    if shape is not None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(shape[1] - 1, x1), min(shape[0] - 1, y1)
    return int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))


# --------------------------------------------------------------------------
# Tampering: the five FSTS edit types
# --------------------------------------------------------------------------
def font_path(layout, bold):
    fam = "NotoSerif" if layout in (0, 2) else "NotoSans"
    return str(Path(FONT_DIR) / f"{fam}-{'Bold' if bold else 'Regular'}.ttf")


def font_pt(layout):
    return {0: 11, 1: 10.5, 2: 10}[layout]


def local_colors(img, bbox):
    x0, y0, x1, y1 = bbox
    pad = 6
    ring = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].reshape(-1, 3).astype(np.float32)
    if ring.size == 0:
        return np.array([245, 244, 238], np.float32), np.array([35, 35, 35], np.float32), 1.0
    lum = ring.mean(1)
    bright = ring[lum >= np.percentile(lum, 60)]          # >= so a flat background is never empty
    if len(bright) == 0:
        bright = ring
    paper = np.median(bright, 0)
    ink_px = ring[lum <= np.percentile(lum, 8)]
    ink = np.median(ink_px, 0) if len(ink_px) and ink_px.mean() < paper.mean() - 40 else np.array([35, 35, 35], np.float32)
    noise = float(bright.std()) if len(bright) > 1 else 1.0
    return paper, ink, noise


def clear_quad(img, quad, paper, noise):
    assert np.isfinite(paper).all() and np.isfinite(noise), "paper colour estimate failed"
    poly = np.array(quad, np.int32)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly], 1)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    fill = np.clip(paper + np_rng.normal(0, max(0.5, min(noise, 6)), img.shape), 0, 255).astype(np.uint8)
    img[m > 0] = fill[m > 0]
    return m


def draw_text_rotated(img, text, anchor_xy, angle, ttf, size_px, ink):
    """Draw `text` with its left baseline at anchor_xy, rotated like the page. Returns changed-pixel mask."""
    font = ImageFont.truetype(ttf, size_px)
    w = int(font.getlength(text)) + 40
    hgt = size_px * 3
    canvas = Image.new("L", (w, hgt), 0)
    ImageDraw.Draw(canvas).text((20, hgt // 2), text, fill=255, font=font, anchor="ls")
    canvas = canvas.rotate(angle, resample=Image.BICUBIC, center=(20, hgt // 2))
    alpha = np.array(canvas).astype(np.float32) / 255.0
    ax, ay = int(round(anchor_xy[0])) - 20, int(round(anchor_xy[1])) - hgt // 2
    H, W = img.shape[:2]
    xs, ys = max(0, ax), max(0, ay)
    xe, ye = min(W, ax + w), min(H, ay + hgt)
    if xe <= xs or ye <= ys:
        return np.zeros(img.shape[:2], np.uint8)
    a = alpha[ys - ay:ye - ay, xs - ax:xe - ax, None]
    region = img[ys:ye, xs:xe].astype(np.float32)
    img[ys:ye, xs:xe] = np.clip(region * (1 - a) + ink * a, 0, 255).astype(np.uint8)
    m = np.zeros(img.shape[:2], np.uint8)
    m[ys:ye, xs:xe] = (a[:, :, 0] > 0.08).astype(np.uint8)
    return cv2.dilate(m, np.ones((3, 3), np.uint8))


def new_value(kind, old, true_iso, datefmt):
    if kind == "date":
        try:
            d = dt.date.fromisoformat(true_iso) if true_iso else None
        except ValueError:
            d = None
        d = d or dt.date(2026, 1, 1)
        d2 = d.replace(year=d.year + rng.choice([1, 2, 3])) if rng.random() < 0.7 else \
            d + dt.timedelta(days=rng.randint(20, 300))
        return fmt_date(d2, datefmt)
    if kind == "person":
        return fake.name()
    chars = list(old)
    idx = [i for i, c in enumerate(chars) if c.isdigit()]
    if not idx:
        return None
    for i in rng.sample(idx, min(len(idx), rng.randint(1, 2))):
        chars[i] = str((int(chars[i]) + rng.randint(1, 9)) % 10)
    return "".join(chars)


def baseline_point(box_pt_y0, box_x0_px, layout, M):
    asc = 1.069 * font_pt(layout) * PT2PX     # Noto ascender, in pixels
    p = np.array([box_x0_px, box_pt_y0 + asc, 1.0]) @ M.T
    return p


def tamper(img, ctx, kind_pool, donor):
    """Apply one edit to one field. Returns (img, mask, info) or None if nothing suitable."""
    quads, raw_boxes, fields, true, layout, angle, M, datefmt = ctx
    cands = [k for k in quads if KINDS.get(k) in TAMPERABLE]
    if not cands:
        return None
    k = rng.choice(cands)
    kind = KINDS[k]
    op = rng.choice(["replacement", "replacement", "insertion", "removal", "copy_move", "splicing"])
    if op == "insertion" and kind not in ("id", "amount", "number"):
        op = "replacement"
    def similar(s):   # a forger copies a value of the same kind and roughly the same length
        return s != k and KINDS.get(s) == kind and 0.8 <= len(str(fields[s])) / max(1, len(str(fields[k]))) <= 1.25
    if op in ("copy_move",):
        same = [s for s in quads if similar(s)]
        if not same:
            op = "replacement"
    if op == "splicing" and not (donor and any(KINDS.get(s) == kind for s in donor["quads"])):
        op = "replacement"
    img = img.copy()
    bbox = quad_bbox(quads[k], pad=2, shape=img.shape)
    paper, ink, noise = local_colors(img, bbox)
    bold = layout in (0, 2)
    size_px = int(round(font_pt(layout) * PT2PX))
    old = fields[k]
    true_iso = true.get(FIELD_FACT.get(k, ""))
    mask = np.zeros(img.shape[:2], np.uint8)
    shown = None
    if op == "replacement":
        shown = new_value(kind, old, true_iso, datefmt)
        if not shown or shown == old:
            return None
        mask |= clear_quad(img, quads[k], paper, noise)
        anchor = baseline_point(raw_boxes[k][1], raw_boxes[k][0], layout, M)
        mask |= draw_text_rotated(img, shown, anchor, angle, font_path(layout, bold), size_px, ink)
    elif op == "insertion":
        extra = str(rng.randint(0, 9))
        shown = old + extra
        anchor = baseline_point(raw_boxes[k][1], raw_boxes[k][2], layout, M)
        mask |= draw_text_rotated(img, extra, anchor, angle, font_path(layout, bold), size_px, ink)
    elif op == "removal":
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.array(quads[k], np.int32)], 255)
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        img = cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)
        mask |= (m > 0).astype(np.uint8)
        shown = ""
    else:  # copy_move or splicing: paste another value of the same kind over this one
        if op == "copy_move":
            s = rng.choice([s for s in quads if similar(s)])
            src_img, src_q, shown = img, quads[s], fields[s]
        else:   # prefer the same field from another institution's document, as a forger would
            s = k if k in donor["quads"] else rng.choice([s for s in donor["quads"] if KINDS.get(s) == kind])
            src_img, src_q, shown = donor["img"], donor["quads"][s], donor["fields"][s]
        sx0, sy0, sx1, sy1 = quad_bbox(src_q, pad=2, shape=src_img.shape)
        patch = src_img[sy0:sy1, sx0:sx1].copy()
        x0, y0, x1, y1 = bbox
        tw = max(x1 - x0, sx1 - sx0)
        th = y1 - y0
        x1 = min(img.shape[1] - 1, x0 + tw)
        patch = cv2.resize(patch, (x1 - x0, th), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        plum = patch.mean(2)
        src_paper = np.median(patch[plum >= np.percentile(plum, 60)], 0)
        patch = np.clip(patch * (paper / np.maximum(src_paper, 1)), 0, 255).astype(np.uint8)  # match paper tone
        mask |= clear_quad(img, quads[k], paper, noise)
        img[y0:y1, x0:x1] = patch
        mask[y0:y1, x0:x1] = 1
    if rng.random() < 0.5:   # post-processing on the edited region, as forgers often do
        x0, y0, x1, y1 = quad_bbox(np.argwhere(mask > 0)[:, ::-1], pad=3, shape=img.shape)
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (3, 3), rng.uniform(0.3, 0.7))
    return img, mask, {"field": k, "edit": op, "printed_before": old, "shown_after": shown}


# --------------------------------------------------------------------------
# PDF wrapping, signing, post-signing edits, validation
# --------------------------------------------------------------------------
def wrap_pdf(jpg, w, h):
    doc = pymupdf.open()
    page = doc.new_page(width=w / PT2PX, height=h / PT2PX)
    page.insert_image(page.rect, stream=jpg)
    return doc.tobytes(garbage=3, deflate=True)


def sign_pdf(pdf_bytes, signer):
    w = IncrementalPdfFileWriter(io.BytesIO(pdf_bytes), strict=False)
    meta = signers.PdfSignatureMetadata(field_name="DSC", reason="Uploaded for AICTE approval")
    return signers.sign_pdf(w, meta, signer=signer).getvalue()


def _incremental_save(pdf_bytes, quad, new_text):
    """Cover a field on the signed page and write a new value, saved as an incremental update
    (a 'shadow'-style edit: the signature still verifies cryptographically, the content changed).
    PyMuPDF writes incremental updates only to the file it opened, so this goes through a temp file."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
        t.write(pdf_bytes)
        path = t.name
    doc = pymupdf.open(path)
    page = doc[0]
    x0, y0, x1, y1 = quad_bbox(quad, pad=2)
    r = pymupdf.Rect(x0 / PT2PX, y0 / PT2PX, x1 / PT2PX + 25, y1 / PT2PX)
    page.draw_rect(r, color=None, fill=(0.97, 0.97, 0.95))
    page.insert_text((r.x0 + 1, r.y1 - 3), new_text, fontsize=10.5, fontname="tiro")
    doc.saveIncr()
    doc.close()
    data = Path(path).read_bytes()
    os.unlink(path)
    return data


def check_signature(pdf_path, ca_pem):
    with open(pdf_path, "rb") as f:
        r = PdfFileReader(f, strict=False)
        if not r.embedded_signatures:
            return "unsigned"
        vc = ValidationContext(trust_roots=[load_cert_from_pemder(str(ca_pem))], allow_fetching=False,
                               revocation_mode="soft-fail")
        st = validate_pdf_signature(r.embedded_signatures[0], vc)
        if st.bottom_line:
            return "valid"
        if st.intact and st.valid:
            return "modified_after_signing"
        return "invalid"


# --------------------------------------------------------------------------
# Main assembly
# --------------------------------------------------------------------------
def choose_problems():
    if rng.random() < P_CLEAN_APPLICATION:
        return []
    n = rng.choices([1, 2, 3][:MAX_PROBLEMS_PER_APP], weights=[0.55, 0.3, 0.15][:MAX_PROBLEMS_PER_APP])[0]
    types, weights = list(PROBLEM_WEIGHTS), list(PROBLEM_WEIGHTS.values())
    chosen = []
    while len(chosen) < n:
        t = rng.choices(types, weights=weights)[0]
        if t not in chosen:
            chosen.append(t)
    return chosen


def main():
    out = Path(OUT_DIR)
    if out.exists():
        shutil.rmtree(out)
    for sub in ["pki", "records", "applications"] + [f"donut/{s}" for s in ("train", "validation", "test")] + \
            [f"tamper/{s}/{d}" for s in ("train", "validation", "test") for d in ("images", "masks")]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    ca_key, ca_cert, ca_pem = make_ca(out / "pki")

    a0, a1 = (dt.date.fromisoformat(x) for x in APPLICATION_WINDOW)
    n_tr = int(N_APPLICATIONS * SPLIT[0])
    n_va = int(N_APPLICATIONS * SPLIT[1])
    splits = ["train"] * n_tr + ["validation"] * n_va + ["test"] * (N_APPLICATIONS - n_tr - n_va)
    rng.shuffle(splits)

    records, donut_rows = [], {s: [] for s in ("train", "validation", "test")}
    tamper_rows = {s: [] for s in ("train", "validation", "test")}
    pool = {}            # doc_type -> last finished page (for duplicates and splicing donors)
    stats = {"pages": 0, "tamper_pages": 0, "problems": {}}

    bar = tqdm(range(START_INDEX, START_INDEX + N_APPLICATIONS), desc="applications", unit="app",
               file=sys.stdout, mininterval=2.0)
    t_start = time.time()
    for n_done, i in enumerate(bar):
        if n_done and n_done % PROGRESS_EVERY == 0:   # plain line, visible even where the bar does not redraw
            el = time.time() - t_start
            eta = el / n_done * (N_APPLICATIONS - n_done)
            print(f"[progress] {n_done}/{N_APPLICATIONS} applications | {stats['pages']} pages | "
                  f"{stats['tamper_pages']} tampered | elapsed {el / 60:.1f} min | about {eta / 60:.1f} min left",
                  flush=True)
        split = splits[i - START_INDEX]
        app_date = rand_date(a0, a1)
        inst = make_institution(i, app_date)
        signer = make_signer(out / "pki", ca_key, ca_cert, ca_pem, inst)
        app_id = f"APP{i:05d}"
        app_dir = out / "applications" / split / app_id
        (app_dir / "docs").mkdir(parents=True)
        (app_dir / "pages").mkdir()
        (app_dir / "masks").mkdir()

        problems = choose_problems()
        planned, override = {}, {d: {} for d in DOC_TYPES}
        free_docs = DOC_TYPES[:]
        rng.shuffle(free_docs)

        def take(options):
            for d in free_docs:
                if d in options:
                    free_docs.remove(d)
                    return d
            return None

        answer = []
        missing = None
        requested = {p["programme"]: p["approved_intake"] for p in inst["programmes"]}
        for p in problems:
            if p == "missing_document":
                missing = take(["fire_noc", "nba_certificate", "lease_deed", "ter_receipt", "affidavit"])
                if missing:
                    answer.append({"type": p, "document": missing, "detail": "required document not uploaded"})
            elif p == "expired_document":
                d = take(["fire_noc", "nba_certificate"])
                if d == "fire_noc":
                    v = app_date - dt.timedelta(days=rng.randint(10, 400))
                    override[d]["fire_valid"] = v.isoformat()
                    override[d]["fire_issue"] = (v - dt.timedelta(days=rng.randint(365, 1000))).isoformat()
                    answer.append({"type": p, "document": d, "field": "valid_until", "detail": f"expired on {v}"})
                elif d == "nba_certificate":
                    v = dt.date(app_date.year - 1, 6, 30)
                    override[d]["nba_valid"] = v.isoformat()
                    answer.append({"type": p, "document": d, "field": "valid_until", "detail": f"expired on {v}"})
            elif p == "lease_insufficient":
                d = take(["lease_deed"])
                if d:
                    yrs = rng.choice([30, 33])
                    remain = rng.uniform(0.5, inst["longest_course_years"] - 0.2)
                    start = app_date - dt.timedelta(days=int((yrs - remain) * 365.25))
                    override[d]["lease_start"] = dt.date(start.year, start.month, min(start.day, 28)).isoformat()
                    override[d]["lease_years"] = yrs
                    answer.append({"type": p, "document": d, "field": "lease_period_years",
                                   "detail": f"about {remain:.1f} years of lease left; longest course is "
                                             f"{inst['longest_course_years']} years"})
            elif p == "name_mismatch":
                d = take(DOC_TYPES)
                if d:
                    nm = inst["name"]
                    variants = [nm.replace(" of ", " of "), nm.replace("Institute", "Instt."),
                                nm.replace("Engineering", "Engg."), nm.split(",")[0],
                                nm.replace(nm.split()[1], nm.split()[1][:-1] + nm.split()[1][-1] * 2, 1)]
                    v = rng.choice([x for x in variants if x != nm] or [nm + " (Autonomous)"])
                    override[d]["institution_name"] = v
                    answer.append({"type": p, "document": d, "field": "institution_name",
                                   "detail": f"'{v}' does not match the AICTE record '{nm}'"})
            elif p == "id_mismatch":
                d = take(["affidavit", "eoa_letter", "ter_receipt"])
                if d:
                    old = inst["institution_id"]
                    j = rng.randint(2, len(old) - 1)
                    v = old[:j] + str((int(old[j]) + rng.randint(1, 9)) % 10) + old[j + 1:]
                    override[d]["aicte_id"] = v
                    answer.append({"type": p, "document": d, "field": "aicte_id",
                                   "detail": f"{v} does not match the AICTE record {old}"})
            elif p == "intake_exceeds_approved":
                prog = rng.choice(inst["programmes"])
                requested[prog["programme"]] = prog["approved_intake"] + rng.choice([30, 60])
                answer.append({"type": p, "document": "application_form", "field": prog["programme"],
                               "detail": f"requests {requested[prog['programme']]} against approved "
                                         f"{prog['approved_intake']}"})
            else:  # render-time problems
                d = take(DOC_TYPES if p != "duplicate_document" else [x for x in DOC_TYPES if x in pool])
                if d:
                    planned[d] = p

        documents = {}
        uploaded = [d for d in DOC_TYPES if d != missing]
        for d in uploaded:
            layout = TEST_ONLY_LAYOUT if (split == "test" and rng.random() < 0.5) else rng.choice(
                [x for x in (0, 1, 2) if x != TEST_ONLY_LAYOUT])
            datefmt = rng.choice(DATE_FORMATS)
            fields, true = doc_fields(d, inst, app_date, datefmt, override[d])
            page_jpg = app_dir / "pages" / f"{d}.jpg"
            pdf_path = app_dir / "docs" / f"{d}.pdf"
            info = {"layout": layout, "date_format": datefmt, "fields_printed": fields, "facts": true}

            if planned.get(d) == "duplicate_document":
                src = pool[d]
                shutil.copy(src["jpg"], page_jpg)
                shutil.copy(src["pdf"], pdf_path)
                info.update({"duplicate_of": src["app_id"], "fields_printed": src["fields"], "signed": src["signed"]})
                answer.append({"type": "duplicate_document", "document": d,
                               "detail": f"same page as uploaded with {src['app_id']}"})
                documents[d] = info
                continue

            vdoc, vpage = render_vector(d, fields, true, inst, layout)
            raw_boxes = field_boxes(vpage, fields)
            raster = rasterize(vpage)
            vdoc.close()
            scanned, quads, angle, M = scan_sim(raster, raw_boxes)
            first, q1 = jpeg_roundtrip(scanned)           # the original scan
            img, mask, tinfo = first, None, None
            ctx = (quads, raw_boxes, fields, true, layout, angle, M, datefmt)
            if planned.get(d) == "tampered_image":
                res = tamper(first, ctx, None, pool.get(d))
                if res:
                    img, mask, tinfo = res
                    answer.append({"type": "tampered_image", "document": d, "field": tinfo["field"],
                                   "detail": f"{tinfo['edit']}: '{tinfo['printed_before']}' shown as "
                                             f"'{tinfo['shown_after']}'", "mask": f"masks/{d}.png"})
                    info["fields_printed"] = dict(fields, **{tinfo["field"]: tinfo["shown_after"]})
                    info["tamper"] = tinfo
            jpg, q2 = jpeg_bytes(img)                     # the file the institution uploads
            page_jpg.write_bytes(jpg)
            h, w = img.shape[:2]
            if mask is not None:
                cv2.imwrite(str(app_dir / "masks" / f"{d}.png"), mask * 255)
            pdf = wrap_pdf(jpg, w, h)
            signed = planned.get(d) != "unsigned_document"
            if signed:
                pdf = sign_pdf(pdf, signer)
            else:
                answer.append({"type": "unsigned_document", "document": d, "detail": "no digital signature"})
            if planned.get(d) == "post_sign_edit":
                editable = [k for k in quads if KINDS.get(k) in TAMPERABLE]
                if editable:
                    k = rng.choice(editable)
                    nv = new_value(KINDS[k], fields[k], true.get(FIELD_FACT.get(k, "")), datefmt)
                    if nv:
                        pdf = _incremental_save(pdf, quads[k], nv)
                        info["post_sign_edit"] = {"field": k, "printed_before": fields[k], "shown_after": nv}
                        answer.append({"type": "post_sign_edit", "document": d, "field": k,
                                       "detail": f"'{fields[k]}' covered with '{nv}' after signing"})
            pdf_path.write_bytes(pdf)
            info.update({"signed": signed, "jpeg_quality": [q1, q2], "field_quads_px": quads})
            documents[d] = info
            stats["pages"] += 1

            clean = not planned.get(d) and not any(a.get("document") == d for a in answer)
            if clean:
                gt = {"gt_parse": {"doc_type": d, **fields}}
                name = f"{app_id}_{d}.jpg"
                shutil.copy(page_jpg, out / "donut" / split / name)
                donut_rows[split].append({"file_name": name, "ground_truth": json.dumps(gt, ensure_ascii=False)})
                # tamper set: the authentic page, and sometimes a tampered twin
                tdir = out / "tamper" / split
                shutil.copy(page_jpg, tdir / "images" / name)
                tamper_rows[split].append([f"images/{name}", "", 0, "", "", app_id, d])
                if rng.random() < P_EXTRA_TAMPER_COPY:
                    res = tamper(first, ctx, None, pool.get(d))
                    if res:
                        t_img, t_mask, t_info = res
                        tname = f"{app_id}_{d}_t.jpg"
                        (tdir / "images" / tname).write_bytes(jpeg_bytes(t_img)[0])
                        cv2.imwrite(str(tdir / "masks" / tname.replace(".jpg", ".png")), t_mask * 255)
                        tamper_rows[split].append([f"images/{tname}", f"masks/{tname.replace('.jpg', '.png')}",
                                                   1, t_info["edit"], t_info["field"], app_id, d])
                        stats["tamper_pages"] += 1
            pool[d] = {"jpg": page_jpg, "pdf": pdf_path, "app_id": app_id, "fields": info["fields_printed"],
                       "signed": signed, "img": first, "quads": quads}

        app_findings = [a for a in answer if a["document"] == "application_form"]
        statuses = {d: "verified" for d in documents}
        if missing:
            statuses[missing] = "missing"
        for a in answer:
            if a["document"] in statuses and a["type"] != "missing_document":
                statuses[a["document"]] = STATUS_FOR[a["type"]]
            stats["problems"][a["type"]] = stats["problems"].get(a["type"], 0) + 1

        application = {"application_id": app_id, "institution_id": inst["institution_id"],
                       "institution_name": inst["name"], "address": inst["address"], "region": inst["region"],
                       "application_date": app_date.isoformat(), "category": "Extension of Approval",
                       "requested_intake": requested, "uploaded_documents": uploaded}
        (app_dir / "application.json").write_text(json.dumps(application, indent=2, ensure_ascii=False))
        (app_dir / "answer_key.json").write_text(json.dumps(
            {"application_id": app_id, "split": split, "problems": answer, "expected_status": statuses,
             "application_status": "deficiency" if app_findings else ("manual_review_required" if any(
                 v != "verified" for v in statuses.values()) else "verified"),
             "application_clean": not answer, "documents": documents}, indent=2, ensure_ascii=False))
        rec = {k: v for k, v in inst.items()}
        rec["approved_intake"] = {p["programme"]: p["approved_intake"] for p in inst["programmes"]}
        records.append(rec)
        bar.set_postfix(pages=stats["pages"], tampered=stats["tamper_pages"])

    (out / "records" / "institutions.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    for s in donut_rows:
        with open(out / "donut" / s / "metadata.jsonl", "w", encoding="utf-8") as f:
            for r in donut_rows[s]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / "tamper" / s / "index.csv", "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["image", "mask", "tampered", "edit", "field", "application", "doc_type"])
            wcsv.writerows(tamper_rows[s])
    stats["donut_pages"] = {s: len(v) for s, v in donut_rows.items()}
    stats["tamper_set"] = {s: len(v) for s, v in tamper_rows.items()}
    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    print(f"[progress] done: {N_APPLICATIONS} applications in {(time.time() - t_start) / 60:.1f} min", flush=True)
    print(json.dumps(stats, indent=2))
    self_check(out, ca_pem)


def self_check(out, ca_pem):
    """Re-open what was written and confirm labels agree with the files."""
    keys = sorted(out.glob("applications/*/*/answer_key.json"))
    sig_expect = {"valid": 0, "unsigned": 0, "modified_after_signing": 0, "invalid": 0}
    bad = []
    for kp in tqdm(keys, desc="self-check", unit="app", file=sys.stdout, mininterval=2.0):
        ak = json.loads(kp.read_text())
        app = kp.parent
        for d, info in ak["documents"].items():
            got = check_signature(app / "docs" / f"{d}.pdf", ca_pem)
            sig_expect[got] += 1
            if "duplicate_of" in info:
                continue
            want = "unsigned" if not info["signed"] else ("modified_after_signing" if "post_sign_edit" in info else "valid")
            if got != want:
                bad.append(f"{app.name}/{d}: signature {got}, expected {want}")
            doc = pymupdf.open(app / "docs" / f"{d}.pdf")
            xref = doc[0].get_images()[0][0]
            if doc.extract_image(xref)["image"] != (app / "pages" / f"{d}.jpg").read_bytes():
                bad.append(f"{app.name}/{d}: embedded JPEG differs from page file")
            doc.close()
            if "tamper" in info:
                m = cv2.imread(str(app / "masks" / f"{d}.png"), 0)
                if m is None or m.max() == 0:
                    bad.append(f"{app.name}/{d}: empty tamper mask")
    for s in ("train", "validation", "test"):
        with open(out / "tamper" / s / "index.csv") as f:
            for row in csv.DictReader(f):
                if row["tampered"] == "1":
                    m = cv2.imread(str(out / "tamper" / s / row["mask"]), 0)
                    if m is None or m.max() == 0:
                        bad.append(f"tamper/{s}/{row['image']}: empty mask")
    print("signature checks:", sig_expect)
    print("problems found by self-check:", len(bad))
    for b in bad[:20]:
        print("  ", b)


if __name__ == "__main__":
    main()
