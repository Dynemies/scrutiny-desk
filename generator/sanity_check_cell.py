# ======================= PATHS (edit if needed) =======================
DATA_ROOT = "/kaggle/input/datasets/vamsikrishnasai05/data-genx"   # the dataset made from the generator output
SIG_SAMPLE = 300          # PDFs whose signature is re-validated (all of them is slow; 300 takes about a minute)
IMG_SAMPLE = 400          # pages opened and checked for size and readability
SHOW_PICTURES = True      # draw a few pages and tamper overlays at the end
# =====================================================================
import os, sys, subprocess, json, csv, random, collections
from pathlib import Path
if os.environ.get("TZ") == "Asia/Calcutta":
    os.environ["TZ"] = "Asia/Kolkata"

try:
    import pymupdf, pyhanko  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymupdf", "pyhanko"])
import cv2
import numpy as np
from tqdm.auto import tqdm

random.seed(0)
results = []          # (check, PASS/FAIL/WARN, detail)


def report(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""), flush=True)


roots = sorted(p.parent for p in Path(DATA_ROOT).rglob("summary.json")
               if (p.parent / "applications").is_dir())
assert roots, f"no generator output (summary.json + applications/) found under {DATA_ROOT}"
OUT = max(roots, key=lambda r: sum(1 for _ in r.glob("applications/*/*/answer_key.json")))
print("checking", OUT, "" if len(roots) == 1 else f"(other outputs found: {[str(r) for r in roots if r != OUT]})")
summary = json.loads((OUT / "summary.json").read_text())
print(json.dumps(summary, indent=1))

# ---------------------------------------------------------------- 1. applications
DOC_TYPES = ["affidavit", "lease_deed", "fire_noc", "nba_certificate", "eoa_letter", "ter_receipt"]
keys = sorted(OUT.glob("applications/*/*/answer_key.json"))
split_of, inst_split, layouts, problems, statuses = {}, collections.defaultdict(set), collections.defaultdict(collections.Counter), collections.Counter(), collections.Counter()
missing_files, mask_missing, status_bad = [], [], []
all_docs = []
for kp in tqdm(keys, desc="applications", unit="app"):
    app = kp.parent
    ak = json.loads(kp.read_text())
    form = json.loads((app / "application.json").read_text())
    split = app.parent.name
    split_of[app.name] = split
    inst_split[form["institution_id"]].add(split)
    if ak["split"] != split:
        status_bad.append(f"{app.name}: answer key says {ak['split']}, folder says {split}")
    for d in form["uploaded_documents"]:
        for f in (app / "docs" / f"{d}.pdf", app / "pages" / f"{d}.jpg"):
            if not f.exists():
                missing_files.append(str(f.relative_to(OUT)))
        info = ak["documents"].get(d, {})
        layouts[split][info.get("layout")] += 1
        all_docs.append((app, d, info))
    missing_doc = [d for d in DOC_TYPES if d not in form["uploaded_documents"]]
    for p in ak["problems"]:
        problems[p["type"]] += 1
        if p.get("mask") and not (app / p["mask"]).exists():
            mask_missing.append(f"{app.name}/{p['mask']}")
        if p["type"] == "missing_document" and p["document"] not in missing_doc:
            status_bad.append(f"{app.name}: says {p['document']} missing but it was uploaded")
    statuses.update(ak["expected_status"].values())
    if bool(ak["problems"]) == ak["application_clean"]:
        status_bad.append(f"{app.name}: application_clean disagrees with the problem list")

n_apps = len(keys)
report("applications found", n_apps > 0, f"{n_apps} ({collections.Counter(split_of.values())})")
report("every uploaded PDF and page exists", not missing_files, f"{len(missing_files)} missing" + (f", e.g. {missing_files[:3]}" if missing_files else ""))
report("answer keys consistent", not status_bad, f"{len(status_bad)} issues" + (f", e.g. {status_bad[:3]}" if status_bad else ""))
report("tamper masks referenced by answer keys exist", not mask_missing, f"{len(mask_missing)} missing")
leak = [i for i, s in inst_split.items() if len(s) > 1]
report("no institution appears in two splits", not leak, f"{len(leak)} shared")
test_only = [s for s in ("train", "validation") if layouts[s].get(2)]
report("layout 2 only in the test split", not test_only, f"layouts by split: { {s: dict(c) for s, c in layouts.items()} }")
report("all ten problem types present", len(problems) == 10, dict(problems), warn=len(problems) >= 8)
print("expected document statuses:", dict(statuses))

# ---------------------------------------------------------------- 2. Donut set
donut_bad, donut_types, donut_count = [], collections.Counter(), {}
for s in ("train", "validation", "test"):
    meta = OUT / "donut" / s / "metadata.jsonl"
    rows = [json.loads(l) for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()] if meta.exists() else []
    donut_count[s] = len(rows)
    for r in tqdm(rows, desc=f"donut {s}", unit="page", leave=False):
        gt = json.loads(r["ground_truth"])["gt_parse"]
        donut_types[gt.get("doc_type")] += 1
        if not (OUT / "donut" / s / r["file_name"]).exists():
            donut_bad.append(f"{s}/{r['file_name']}: file missing")
        if gt.get("doc_type") not in DOC_TYPES or any(v in ("", None) for v in gt.values()):
            donut_bad.append(f"{s}/{r['file_name']}: bad label")
        app_id = r["file_name"].split("_")[0]
        if split_of.get(app_id) != s:
            donut_bad.append(f"{s}/{r['file_name']}: belongs to {split_of.get(app_id)}")
report("Donut labels valid, files present, splits clean", not donut_bad, f"{donut_count}" + (f"; {len(donut_bad)} issues, e.g. {donut_bad[:3]}" if donut_bad else ""))
report("Donut has every document type", len(donut_types) == 6, dict(donut_types))

# ---------------------------------------------------------------- 3. tamper set
t_bad, t_count, edits, ratios, areas = [], {}, collections.Counter(), [], []
for s in ("train", "validation", "test"):
    idx = OUT / "tamper" / s / "index.csv"
    rows = list(csv.DictReader(open(idx))) if idx.exists() else []
    t_count[s] = collections.Counter(r["tampered"] for r in rows)
    for r in tqdm(rows, desc=f"tamper {s}", unit="img", leave=False):
        img_p = OUT / "tamper" / s / r["image"]
        if not img_p.exists():
            t_bad.append(f"{s}/{r['image']}: missing")
            continue
        if split_of.get(r["application"]) != s:
            t_bad.append(f"{s}/{r['image']}: belongs to {split_of.get(r['application'])}")
        if r["tampered"] != "1":
            continue
        edits[r["edit"]] += 1
        m = cv2.imread(str(OUT / "tamper" / s / r["mask"]), 0)
        if m is None or m.max() == 0:
            t_bad.append(f"{s}/{r['mask']}: missing or empty")
            continue
        areas.append((m > 0).mean())
        if len(ratios) < 150:          # does the edit sit where the mask says? compare with the authentic twin
            img = cv2.imread(str(img_p))
            orig = cv2.imread(str(img_p).replace("_t.jpg", ".jpg"))
            if img is None or orig is None or img.shape[:2] != m.shape:
                t_bad.append(f"{s}/{r['image']}: unreadable or mask size differs")
                continue
            diff = np.abs(img.astype(np.int16) - orig.astype(np.int16)).mean(2)
            inside = diff[m > 0].mean()
            outside = diff[m == 0].mean()
            ratios.append(inside / max(outside, 0.5))
report("tamper images and masks present, splits clean", not t_bad, f"{ {s: dict(c) for s, c in t_count.items()} }" + (f"; {len(t_bad)} issues, e.g. {t_bad[:3]}" if t_bad else ""))
report("all five edit types present", len(edits) == 5, dict(edits), warn=len(edits) >= 4)
if areas:
    report("mask size plausible (0.001% to 2% of page)", 1e-5 < np.median(areas) < 0.02,
           f"median {100 * np.median(areas):.3f}% of page")
if ratios:
    share = float(np.mean(np.array(ratios) > 3))
    report("edits sit inside their masks", share > 0.9,
           f"{100 * share:.0f}% of checked pairs change at least 3x more inside the mask (median {np.median(ratios):.1f}x)")

# ---------------------------------------------------------------- 4. page images
sample = random.sample(all_docs, min(IMG_SAMPLE, len(all_docs)))
sizes, unreadable = collections.Counter(), []
for app, d, info in tqdm(sample, desc="page images", unit="img"):
    im = cv2.imread(str(app / "pages" / f"{d}.jpg"))
    if im is None:
        unreadable.append(f"{app.name}/{d}")
        continue
    sizes[im.shape[:2]] += 1
    if im.mean() < 150 or im.std() < 10:
        unreadable.append(f"{app.name}/{d}: looks blank or dark")
report("sampled pages open and look like documents", not unreadable, f"sizes {dict(sizes)}" + (f"; issues {unreadable[:3]}" if unreadable else ""))

# ---------------------------------------------------------------- 5. signatures and embedded pages
try:
    import pymupdf
    from pyhanko.keys import load_cert_from_pemder
    from pyhanko.pdf_utils.reader import PdfFileReader
    from pyhanko.sign.validation import validate_pdf_signature
    from pyhanko_certvalidator import ValidationContext
    import logging
    logging.getLogger("pyhanko").setLevel(logging.CRITICAL)
    logging.getLogger("pyhanko_certvalidator").setLevel(logging.CRITICAL)
    from concurrent.futures import ThreadPoolExecutor
    _pool = ThreadPoolExecutor(max_workers=1)
    root = load_cert_from_pemder(str(OUT / "pki" / "test_ca.pem"))
    want_sample = [x for x in all_docs if "duplicate_of" not in x[2]]
    special = [x for x in want_sample if not x[2].get("signed", True) or "post_sign_edit" in x[2]]
    pick = special + random.sample(want_sample, max(0, min(SIG_SAMPLE, len(want_sample)) - len(special)))
    sig_bad, sig_seen = [], collections.Counter()
    for app, d, info in tqdm(pick, desc="signatures", unit="pdf"):
        pdf = app / "docs" / f"{d}.pdf"
        with open(pdf, "rb") as f:
            r = PdfFileReader(f, strict=False)
            if not r.embedded_signatures:
                got = "unsigned"
            else:   # pyHanko starts its own event loop; a notebook already runs one, so validate in a thread
                st = _pool.submit(validate_pdf_signature, r.embedded_signatures[0], ValidationContext(
                    trust_roots=[root], allow_fetching=False, revocation_mode="soft-fail")).result()
                got = "valid" if st.bottom_line else ("modified_after_signing" if st.intact and st.valid else "invalid")
        want = "unsigned" if not info.get("signed", True) else ("modified_after_signing" if "post_sign_edit" in info else "valid")
        sig_seen[got] += 1
        if got != want:
            sig_bad.append(f"{app.name}/{d}: {got}, expected {want}")
        doc = pymupdf.open(pdf)
        if doc.extract_image(doc[0].get_images()[0][0])["image"] != (app / "pages" / f"{d}.jpg").read_bytes():
            sig_bad.append(f"{app.name}/{d}: embedded page differs from pages/{d}.jpg")
        doc.close()
    report("signature status matches the answer key", not sig_bad,
           f"{dict(sig_seen)} over {len(pick)} PDFs (all unsigned and post-sign-edit ones included)" + (f"; e.g. {sig_bad[:3]}" if sig_bad else ""))
except ImportError as e:
    report("signature check", False, f"skipped, could not import pyhanko/pymupdf ({e}); turn Internet on", warn=True)

# ---------------------------------------------------------------- 6. size
size_gb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e9
report("dataset size", True, f"{size_gb:.2f} GB, {n_apps} applications, {size_gb * 1000 / max(n_apps, 1):.1f} MB each")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 70)
for name, status, detail in results:
    print(f"{status:5}  {name}")
fails = [r for r in results if r[1] == "FAIL"]
print("=" * 70)
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED - send me the lines above")

# ---------------------------------------------------------------- pictures
if SHOW_PICTURES:
    import matplotlib.pyplot as plt
    pages = random.sample([x for x in all_docs if "duplicate_of" not in x[2]], 6)
    fig, ax = plt.subplots(1, 6, figsize=(24, 6))
    for a, (app, d, info) in zip(ax, pages):
        a.imshow(cv2.cvtColor(cv2.imread(str(app / "pages" / f"{d}.jpg")), cv2.COLOR_BGR2RGB))
        a.set_title(f"{app.name}\n{d} (layout {info.get('layout')})", fontsize=9)
        a.axis("off")
    plt.tight_layout()
    plt.show()
    rows = [r for r in csv.DictReader(open(OUT / "tamper" / "train" / "index.csv")) if r["tampered"] == "1"]
    by_edit = {}
    for r in rows:
        by_edit.setdefault(r["edit"], r)
    fig, ax = plt.subplots(len(by_edit), 2, figsize=(14, 2.2 * len(by_edit)))
    ax = np.atleast_2d(ax)
    for i, (e, r) in enumerate(sorted(by_edit.items())):
        img = cv2.imread(str(OUT / "tamper" / "train" / r["image"]))
        orig = cv2.imread(str(OUT / "tamper" / "train" / r["image"].replace("_t.jpg", ".jpg")))
        m = cv2.imread(str(OUT / "tamper" / "train" / r["mask"]), 0)
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = max(0, ys.min() - 25), ys.max() + 25, max(0, xs.min() - 90), xs.max() + 90
        over = img.copy()
        cnt, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnt, -1, (0, 0, 255), 2)
        ax[i, 0].imshow(cv2.cvtColor(orig[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        ax[i, 0].set_title(f"authentic ({r['field']})", fontsize=9)
        ax[i, 1].imshow(cv2.cvtColor(over[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        ax[i, 1].set_title(f"{e}: tampered, mask in red", fontsize=9)
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
    plt.tight_layout()
    plt.show()
