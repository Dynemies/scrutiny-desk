# ======================= PATHS (edit if needed) =======================
DATA_DIR = "./demo_data"          # generator output: applications/, records/, pki/ (+ tamper_maps/ later)
# =====================================================================
import datetime as dt
import html
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pymupdf
import streamlit as st
from PIL import Image, ImageDraw

import engine

st.set_page_config(page_title="Scrutiny desk", page_icon="📋", layout="wide", initial_sidebar_state="expanded")

STAMP_CLASS = {"verified": "ok", "missing": "flag", "mismatch_detected": "flag", "expired": "flag",
               "deficiency": "flag", "manual_review_required": "review", "verification_pending": "review"}
INK = {"ok": (30, 106, 80), "flag": (179, 38, 30), "review": (154, 91, 0)}

st.markdown("""
<style>
@font-face { font-family:'Martel'; font-weight:700; src:url('./app/static/Martel-Bold.ttf') format('truetype'); }
@font-face { font-family:'Martel'; font-weight:800; src:url('./app/static/Martel-ExtraBold.ttf') format('truetype'); }
@font-face { font-family:'Mukta'; font-weight:400; src:url('./app/static/Mukta-Regular.ttf') format('truetype'); }
@font-face { font-family:'Mukta'; font-weight:500; src:url('./app/static/Mukta-Medium.ttf') format('truetype'); }
@font-face { font-family:'Mukta'; font-weight:600; src:url('./app/static/Mukta-SemiBold.ttf') format('truetype'); }
:root { --paper:#FAFBF9; --file:#E4EAE2; --ink:#172A46; --muted:#5B6878; --rule:#CDD5CB;
        --ok:#1E6A50; --flag:#B3261E; --review:#9A5B00; }
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp td, .stApp th, .stApp button,
[data-testid="stMarkdownContainer"] p { font-family: 'Mukta', sans-serif; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.stApp { background: var(--paper); color: var(--ink); }
h1, h2, h3, .serif { font-family: 'Martel', serif !important; color: var(--ink); letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: var(--file); border-right: 1px solid var(--rule); }
[data-testid="stSidebar"] * { color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1380px; }
.stApp p.desk-title { font-family: 'Martel', serif; font-weight: 800; font-size: 1.35rem; line-height: 1.2; margin: 0; }
.stApp p.desk-sub { color: var(--muted); font-size: 0.92rem; margin: 2px 0 14px 0; }
.stApp p.inst { font-family: 'Martel', serif; font-weight: 800; font-size: 2.0rem; line-height: 1.25; margin: 0 0 6px 0; }
.stApp p.meta { color: var(--muted); font-size: 1.0rem; margin: 0; }
.stApp p.meta b { color: var(--ink); font-weight: 600; }
/* the stamp: the one loud thing on the page */
.stamp { display: inline-block; transform: rotate(-4deg); border: 3px double currentColor; border-radius: 6px;
         padding: 6px 18px 2px 18px; font-family: 'Martel', serif; font-weight: 800; font-size: 1.55rem;
         letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.88; margin-top: 8px; white-space: nowrap;
         mask-image: radial-gradient(circle at 30% 40%, #000 60%, rgba(0,0,0,.78) 100%); }
.stamp small { display:block; font-family:'Mukta',sans-serif; font-weight:500; font-size:.8rem;
               letter-spacing:0; text-transform:none; text-align:center; margin-top:-2px; }
.ok { color: var(--ok); } .flag { color: var(--flag); } .review { color: var(--review); }
.badge { display:inline-block; border:1.5px solid currentColor; border-radius:4px; padding:0 8px;
         font-weight:600; font-size:.85rem; white-space:nowrap; background: rgba(255,255,255,.6); }
table.register { width:100%; border-collapse:collapse; font-size:.97rem; line-height:1.45; }
table.register td { border-top:1px solid var(--rule); padding:10px 8px; vertical-align:top; }
table.register tr:last-child td { border-bottom:1px solid var(--rule); }
table.register td.doc { font-weight:600; width:38%; line-height:1.3; }
table.register td.doc .badge { margin-top:6px; }
table.register .why { color: var(--ink); }
table.register .quiet { color: var(--muted); }
.appnote { border-left:4px solid var(--flag); background:#fff; padding:10px 14px; margin:6px 0 16px 0; }
.finding { border-left:4px solid currentColor; padding:4px 0 4px 12px; margin:0 0 12px 0; }
.finding .t { color: var(--ink); }
.finding .c { font-weight:600; font-size:.85rem; }
.sigcard { background:#fff; border:1px solid var(--rule); border-radius:6px; padding:10px 14px; font-size:.95rem; }
.sigcard b { font-weight:600; }
.stApp p.caption { color: var(--muted); font-size:.88rem; margin-top:-6px; }
.bar { display:flex; height:22px; border-radius:4px; overflow:hidden; border:1px solid var(--rule); }
.bar span { display:block; height:100%; }
table.plain { width:100%; border-collapse:collapse; font-size:.95rem; }
table.plain th { text-align:left; font-weight:600; color:var(--muted); border-bottom:1px solid var(--rule); padding:6px 8px; }
table.plain td { border-bottom:1px solid var(--rule); padding:7px 8px; vertical-align:top; }
table.diff { border-collapse:collapse; margin:6px 0 0 0; font-size:.95rem; }
table.diff td { padding:2px 10px 2px 0; vertical-align:top; color: var(--ink); border:none; }
table.diff, table.diff tr { border:none; background:transparent; }
table.diff td.k { color: var(--muted); white-space:nowrap; }
mark.new { background: rgba(179,38,30,.16); color: var(--flag); font-weight:600; padding:0 1px; border-radius:2px; }
mark.gone { background: rgba(23,42,70,.10); color: var(--ink); font-weight:600; padding:0 1px; border-radius:2px; }
.stApp table.register td, .stApp table.plain td, .stApp table.plain th { border-left:none; border-right:none; }
.stApp table.plain tr:first-child th { border-top:none; }
.stApp h3 { font-size:1.4rem; margin-bottom:.4rem; }
.hbar { height:12px; background: var(--ink); opacity:.85; border-radius:2px; }
div[data-baseweb="tab-list"] button p { font-family:'Martel',serif; font-weight:600; font-size:1rem; }
div[role="radiogroup"] label p { font-size:.95rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Opening the application store")
def get_store():
    return engine.Store(DATA_DIR)


@st.cache_data(show_spinner=False)
def get_result(app_id):
    cache = Path(DATA_DIR) / "results" / f"{app_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = engine.to_jsonable(engine.check_application(get_store(), app_id))
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(r))
    return r


store = get_store()
app_ids = list(store.apps)
results = {}
with st.spinner("Checking the queue"):
    for a in app_ids:
        results[a] = get_result(a)


def diff_html(doc_value, record_value):
    """Two lines, document vs record, with the characters that differ marked."""
    import difflib
    sm = difflib.SequenceMatcher(None, record_value, doc_value, autojunk=False)
    d_out, r_out = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        r_seg, d_seg = html.escape(record_value[i1:i2]), html.escape(doc_value[j1:j2])
        if op == "equal":
            r_out.append(r_seg); d_out.append(d_seg)
        else:
            if r_seg:
                r_out.append(f'<mark class="gone">{r_seg}</mark>')
            if d_seg:
                d_out.append(f'<mark class="new">{d_seg}</mark>')
    return ('<table class="diff"><tr><td class="k">Document</td><td>' + "".join(d_out) + '</td></tr>'
            '<tr><td class="k">AICTE record</td><td>' + "".join(r_out) + '</td></tr></table>')


def stamp_html(status, small=None, big=True):
    cls = STAMP_CLASS[status]
    label = engine.STATUS_LABEL[status]
    if big:
        return f'<div class="stamp {cls}">{label}' + (f"<small>{html.escape(small)}</small>" if small else "") + "</div>"
    return f'<span class="badge {cls}">{label}</span>'


def short(name):
    return name.split(",")[0]


def n_flags(r):
    return sum(1 for d in r["documents"].values() for f in d["findings"] if not f.get("note")) + len(r["application_findings"])


# ------------------------------------------------------------------ sidebar: the queue
with st.sidebar:
    st.markdown('<p class="desk-title">Scrutiny desk</p>'
                '<p class="desk-sub">Extension of Approval applications, checked before an officer opens them.</p>',
                unsafe_allow_html=True)
    show = st.segmented_control("Show", ["Needs action", "Verified", "All"], default="Needs action",
                                label_visibility="collapsed") or "Needs action"
    pool = [a for a in app_ids if show == "All" or (show == "Verified") == (results[a]["status"] == "verified")]
    if not pool:
        st.info("Nothing in this list.")
        st.stop()
    dot = {"ok": ":green[●]", "flag": ":red[●]", "review": ":orange[●]"}
    chosen = st.radio(
        "Applications", pool, label_visibility="collapsed",
        format_func=lambda a: f"{dot[STAMP_CLASS[results[a]['status']]]} {short(results[a]['application']['institution_name'])}",
        captions=[f"{a}, {engine.STATUS_LABEL[results[a]['status']].lower()}" for a in pool], key="queue")
    st.caption(f"{len(pool)} of {len(app_ids)} applications")


tab_app, tab_queue, tab_file = st.tabs(["Application", "Queue overview", "Check a file"])

# ------------------------------------------------------------------ helpers for evidence images
def load_page(app_id, doc):
    return Image.open(store.page_path(app_id, doc)).convert("RGB")


def quad_box(q, pad=6):
    q = np.array(q)
    return [float(q[:, 0].min() - pad), float(q[:, 1].min() - pad), float(q[:, 0].max() + pad), float(q[:, 1].max() + pad)]


def marked_page(app_id, doc, info):
    img = load_page(app_id, doc)
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    boxes = []
    if info.get("tamper") and info["tamper"].get("heatmap"):
        hm = Image.open(info["tamper"]["heatmap"]).convert("L").resize(img.size)
        red = Image.new("RGBA", img.size, INK["flag"] + (0,))
        red.putalpha(hm.point(lambda v: int(v * 0.55)))
        over = Image.alpha_composite(over, red)
        dr = ImageDraw.Draw(over)
    for f in info["findings"]:
        col = INK[STAMP_CLASS[f["status"]]]
        if f.get("check") == "Tampering" and f.get("heatmap"):
            b = heat_box(f["heatmap"], img.size)
            if b:
                dr.rectangle(b, outline=col + (255,), width=4)
                boxes.append((b, f))
        if f.get("field") and f["field"] in info.get("boxes", {}):
            b = quad_box(info["boxes"][f["field"]])
            dr.rectangle(b, outline=col + (255,), width=4, fill=col + (38,))
            boxes.append((b, f))
        if f.get("region"):
            x0, y0, x1, y1 = f["region"]
            b = [x0 - 8, y0 - 8, x1 + 8, y1 + 8]
            dr.rectangle(b, outline=col + (255,), width=4, fill=col + (38,))
            boxes.append((b, f))
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB"), boxes


def heat_box(heatmap_path, size, prob=0.3):
    """Box around the largest area the tamper model marks (same rule as the verdict), padded a little."""
    import cv2
    hm = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if hm is None:
        return None
    hm = cv2.resize(hm, size, interpolation=cv2.INTER_NEAREST)
    n, _, stats, _ = cv2.connectedComponentsWithStats((hm >= prob * 255).astype("uint8"), connectivity=8)
    if n < 2:
        return None
    j = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    x, y, w, h = (int(v) for v in stats[j, :4])
    return [x - 8, y - 8, x + w + 8, y + h + 8]


def crop(img, b, wpad=140, hpad=60):
    x0, y0, x1, y1 = b
    return img.crop((max(0, x0 - wpad), max(0, y0 - hpad), min(img.width, x1 + wpad), min(img.height, y1 + hpad)))


def render_signed_revision(pdf_path):
    data = Path(pdf_path).read_bytes()
    import re
    m = re.search(rb"/ByteRange\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]", data)
    out = []
    for blob in ((data[: int(m.group(3)) + int(m.group(4))]) if m else data, data):
        d = pymupdf.open("pdf", blob)
        pix = d[0].get_pixmap(dpi=150, alpha=False)
        out.append(Image.frombytes("RGB", (pix.w, pix.h), pix.samples))
        d.close()
    return out


# ------------------------------------------------------------------ tab: one application
with tab_app:
    r = results[chosen]
    form, rec = r["application"], r["record"]
    left, right = st.columns([7, 3], vertical_alignment="top")
    with left:
        filed = dt.date.fromisoformat(form["application_date"])
        st.markdown(
            f'<p class="inst">{html.escape(form["institution_name"])}</p>'
            f'<p class="meta">AICTE ID <b>{form["institution_id"]}</b>&emsp;{form["region"]} region&emsp;'
            f'Filed <b>{filed:%d %b %Y}</b>&emsp;Application <b>{chosen}</b></p>',
            unsafe_allow_html=True)
    with right:
        k = n_flags(r)
        small = "no issues found" if k == 0 else f"{k} finding{'s' if k != 1 else ''} to review"
        st.markdown(f'<div style="text-align:right">{stamp_html(r["status"], small)}</div>', unsafe_allow_html=True)

    for f in r["application_findings"]:
        st.markdown(f'<div class="appnote"><b>{f["check"]}.</b> {html.escape(f["text"])}</div>', unsafe_allow_html=True)

    st.write("")
    reg, ev = st.columns([5, 7], gap="large")
    with reg:
        st.markdown("### Documents")
        rows = []
        for d in engine.REQUIRED:
            info = r["documents"][d]
            real = [f for f in info["findings"] if not f.get("note")]
            if real:
                why = html.escape(real[0]["text"]) + (f' <span class="quiet">and {len(real) - 1} more</span>' if len(real) > 1 else "")
            elif info["status"] == "missing":
                why = "Not uploaded."
            else:
                sig = info.get("signature", {}).get("state")
                why = '<span class="quiet">Fields match the record. Signature ' + ("valid" if sig == "valid" else str(sig)) + ".</span>"
                notes = [f for f in info["findings"] if f.get("note")]
                if notes:
                    why += f'<br><span class="quiet">{html.escape(notes[0]["text"])}</span>'
            rows.append(f'<tr><td class="doc">{engine.DOC_NAMES[d]}<br>{stamp_html(info["status"], big=False)}</td>'
                        f'<td class="why">{why}</td></tr>')
        st.markdown(f'<table class="register">{"".join(rows)}</table>', unsafe_allow_html=True)
        with st.expander("Audit trail"):
            st.markdown('<table class="plain"><tr><th>Time</th><th>Step</th></tr>' + "".join(
                f'<tr><td style="white-space:nowrap">{a["at"]}</td><td>{html.escape(a["check"])}</td></tr>' for a in r["audit"]) + "</table>",
                unsafe_allow_html=True)
            st.caption(f"Checked {r['checked_at'].replace('T', ' ')}. Findings are advisory; the officer decides.")

    with ev:
        st.markdown("### Evidence")
        uploaded = [d for d in engine.REQUIRED if r["documents"][d]["status"] != "missing"]
        flagged = [d for d in uploaded if r["documents"][d]["status"] != "verified"]
        doc = st.radio("Document", uploaded, index=uploaded.index(flagged[0]) if flagged else 0, horizontal=True,
                       format_func=lambda d: engine.DOC_NAMES[d], label_visibility="collapsed", key=f"doc_{chosen}")
        info = r["documents"][doc]
        page, boxes = marked_page(chosen, doc, info)
        pcol, fcol = st.columns([5, 6], gap="medium")
        with pcol:
            st.image(page, width="stretch")
            st.markdown('<p class="caption">Uploaded page. Marked areas carry a finding.</p>', unsafe_allow_html=True)
        with fcol:
            real = [f for f in info["findings"] if not f.get("note")]
            if not real:
                st.markdown(f'<div class="finding ok"><span class="c">All checks passed</span><br>'
                            f'<span class="t">Name and ID match the AICTE record, dates are in force and the digital signature verifies.</span></div>',
                            unsafe_allow_html=True)
            for f in info["findings"]:
                cls = STAMP_CLASS[f["status"]]
                extra = diff_html(f["doc_value"], f["record_value"]) if f.get("doc_value") else ""
                st.markdown(f'<div class="finding {cls}"><span class="c">{f["check"]}</span><br><span class="t">{html.escape(f["text"])}</span>{extra}</div>',
                            unsafe_allow_html=True)
                hit = [b for b, ff in boxes if ff is f]
                if hit and f["check"] != "Signature":
                    st.image(crop(page, hit[0]), width="stretch")
                if f["check"] == "Signature" and f.get("region"):
                    signed, now = render_signed_revision(store.pdf_path(chosen, doc))
                    x0, y0, x1, y1 = f["region"]
                    ImageDraw.Draw(now).rectangle([x0 - 6, y0 - 4, x1 + 6, y1 + 4], outline=INK["review"], width=3)
                    st.image(crop(signed, f["region"], 260, 18), caption="As signed", width="stretch")
                    st.image(crop(now, f["region"], 260, 18), caption="As uploaded, after the change", width="stretch")
                if f["check"] == "Re-use" and f.get("other"):
                    o_app, o_doc = f["other"].split("/")
                    st.image(load_page(o_app, o_doc).resize((310, 438)), caption=f"Earlier upload in {o_app}")
            sig = info.get("signature") or {}
            if sig:
                state = {"valid": "Valid", "unsigned": "Not signed", "modified_after_signing": "Valid, but changed after signing",
                         "invalid": "Does not verify"}[sig["state"]]
                st.markdown(f'<div class="sigcard"><b>Digital signature.</b> {state}'
                            + (f'<br>Signed by {html.escape(sig.get("signer") or "")}' if sig.get("signer") else "") + "</div>",
                            unsafe_allow_html=True)
            tm = info.get("tamper")
            st.markdown('<p class="caption" style="margin-top:10px">'
                        + ("Tamper model: " + (engine.tamper_text(tm) if tm else "not connected yet") + ". ")
                        + f'Fields read by: {info.get("reader", "")}.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ tab: queue overview
with tab_queue:
    total = len(app_ids)
    by = {}
    for a in app_ids:
        by.setdefault(results[a]["status"], []).append(a)
    need = total - len(by.get("verified", []))
    st.markdown(f'<p class="inst" style="font-size:1.6rem">{need} of {total} applications need an officer</p>'
                f'<p class="meta">{len(by.get("verified", []))} passed every check and can move to the next stage.</p>',
                unsafe_allow_html=True)
    st.write("")
    colors = {"ok": "var(--ok)", "flag": "var(--flag)", "review": "var(--review)"}
    order = [s for s in engine.STATUS_ORDER if s in by]
    segs = "".join(f'<span title="{engine.STATUS_LABEL[s]}" style="width:{100 * len(by[s]) / total:.2f}%;'
                   f'background:{colors[STAMP_CLASS[s]]};opacity:{0.55 + 0.45 * (s in ("verified", "missing", "mismatch_detected"))}"></span>' for s in order)
    st.markdown(f'<div class="bar">{segs}</div>', unsafe_allow_html=True)
    st.markdown(" &nbsp; ".join(f'{stamp_html(s, big=False)} {len(by[s])}' for s in order), unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([4, 7], gap="large")
    with c1:
        st.markdown("### What the checks found")
        counts = {}
        for a in app_ids:
            for d in results[a]["documents"].values():
                for f in d["findings"]:
                    if not f.get("note"):
                        counts[f["check"]] = counts.get(f["check"], 0) + 1
            for f in results[a]["application_findings"]:
                counts[f["check"]] = counts.get(f["check"], 0) + 1
        mx = max(counts.values()) if counts else 1
        label = {"Checklist": "Missing documents", "Records": "Name or ID differs from record", "Validity": "Expired certificates",
                 "Land rule": "Lease shorter than course", "Signature": "Signature problems", "Re-use": "Pages re-used",
                 "Intake": "Intake above approval", "Reading": "Blank fields", "Tampering": "Possible edits"}
        st.markdown('<table class="plain">' + "".join(
            f'<tr><td style="width:46%">{label.get(k, k)}</td><td><div class="hbar" style="width:{100 * v / mx:.0f}%"></div></td>'
            f'<td style="width:8%;text-align:right">{v}</td></tr>' for k, v in sorted(counts.items(), key=lambda x: -x[1])) + "</table>",
            unsafe_allow_html=True)
    with c2:
        st.markdown("### Waiting for an officer")
        rows = []
        for a in sorted([a for a in app_ids if results[a]["status"] != "verified"],
                        key=lambda a: (engine.STATUS_ORDER.index(results[a]["status"]), a))[:30]:
            rr = results[a]
            allf = [f for d in rr["documents"].values() for f in d["findings"] if not f.get("note")] + rr["application_findings"]
            first = min(allf, key=lambda f: engine.STATUS_ORDER.index(f["status"]))["text"] if allf else ""
            rows.append(f'<tr><td style="white-space:nowrap">{a}</td><td>{html.escape(short(rr["application"]["institution_name"]))}</td>'
                        f'<td>{stamp_html(rr["status"], big=False)}</td><td class="quiet" style="color:var(--muted)">{html.escape(first[:110])}</td></tr>')
        st.markdown('<table class="plain"><tr><th>Application</th><th>Institution</th><th>Status</th><th>First finding</th></tr>'
                    + "".join(rows) + "</table>", unsafe_allow_html=True)

# ------------------------------------------------------------------ tab: check a single file
with tab_file:
    st.markdown("### Check a signed PDF")
    st.markdown('<p class="meta">Upload one document. The desk verifies its digital signature, looks for changes made after '
                'signing, and checks whether the same page was already submitted in another application.</p>', unsafe_allow_html=True)
    up = st.file_uploader("PDF document", type=["pdf"], label_visibility="collapsed")
    if up:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
            t.write(up.getvalue())
            path = t.name
        sig = engine.signature_check(path, store.ca)
        state = {"valid": ("verified", "Signature verifies and the file is unchanged since signing."),
                 "unsigned": ("manual_review_required", "No digital signature found."),
                 "modified_after_signing": ("manual_review_required", "Signature is genuine, but the file changed after signing."),
                 "invalid": ("manual_review_required", "Signature does not verify.")}[sig["state"]]
        a, b = st.columns([3, 5], gap="large")
        a.markdown(f'<div style="padding:6px 0 24px 4px">{stamp_html(state[0], "signature check")}</div>', unsafe_allow_html=True)
        b.markdown(f'<div class="finding {STAMP_CLASS[state[0]]}"><span class="c">Signature</span><br><span class="t">{state[1]}'
                   + (f' Signed by {html.escape(sig["signer"])}.' if sig.get("signer") else "") + "</span></div>", unsafe_allow_html=True)
        doc = pymupdf.open(path)
        imgs = doc[0].get_images()
        if imgs:
            raw = doc.extract_image(imgs[0][0])["image"]
            tmp = Path(tempfile.mkdtemp()) / "p.jpg"
            tmp.write_bytes(raw)
            fp = engine.page_fingerprint(tmp)
            hit = engine.find_reuse(store, tmp, fp)
            b.markdown(f'<div class="finding {"review" if hit else "ok"}"><span class="c">Re-use</span><br><span class="t">'
                       + (f"Same page is already on file in {hit.replace('/', ', ')}." if hit else "Not seen in any other application.")
                       + "</span></div>", unsafe_allow_html=True)
        if sig["state"] == "modified_after_signing":
            reg = engine.changed_after_signing(path)
            if reg:
                s_img, n_img = render_signed_revision(path)
                x0, y0, x1, y1 = reg
                ImageDraw.Draw(n_img).rectangle([x0 - 6, y0 - 4, x1 + 6, y1 + 4], outline=INK["review"], width=3)
                c, d2 = st.columns(2)
                c.image(crop(s_img, reg, 260, 18), caption="As signed", width="stretch")
                d2.image(crop(n_img, reg, 260, 18), caption="As uploaded, after the change", width="stretch")
        st.caption("Field reading and the tamper heatmap switch on here once the trained models are connected.")
