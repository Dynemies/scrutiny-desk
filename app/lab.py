# ======================= PATHS (edit if needed) =======================
SAMPLES_DIR = "./demo_data/lab"    # the 20 bundled test pages: pages/, truth/, results/, samples.csv
STORE_DIR = "./lab_store"          # pages you check yourself, with their results (kept between runs)
# ======================================================================
"""Tamper lab: drop scanned pages or PDFs, run the tamper model, and see what it marks:
verdict, heatmap, the model's mask, the true edit when it is known, and a close-up."""
import csv
import html
import io
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from tamper import model as M

HERE = Path(__file__).resolve().parent
SAMPLES_DIR = str(HERE / SAMPLES_DIR) if not Path(SAMPLES_DIR).is_absolute() else SAMPLES_DIR
STORE_DIR = str(HERE / STORE_DIR) if not Path(STORE_DIR).is_absolute() else STORE_DIR

EDIT_WORDS = {"replacement": "value replaced", "insertion": "character added", "removal": "value erased",
              "copy_move": "value copied from elsewhere on the page", "splicing": "value pasted from another document"}


def pretty(field):
    words = {"aicte": "AICTE", "id": "ID", "noc": "NOC", "nba": "NBA", "no": "number", "estamp": "e-stamp", "cert": "certificate"}
    return " ".join(words.get(w, w) for w in field.split("_")) + " field"


DOC_NAMES = {"affidavit": "Affidavit on e-stamp paper", "lease_deed": "Lease deed", "fire_noc": "Fire safety NOC",
             "nba_certificate": "NBA accreditation", "eoa_letter": "Previous approval letter", "ter_receipt": "Fee receipt (TER)"}

RED = np.array([200, 32, 28], np.float32)
GREEN = np.array([22, 128, 82], np.float32)
GOLD = np.array([226, 160, 20], np.float32)
AMBER = (176, 104, 14)

CSS = """
<style>
.lab-hero { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; margin:4px 0 14px 0; }
.lab-hero p.t { font-family:'Martel',serif; font-weight:800; font-size:2.0rem; line-height:1.15; margin:0; color:var(--ink); }
.lab-hero p.s { color:var(--muted); font-size:1.02rem; margin:4px 0 0 0; max-width:760px; }
.chip { display:inline-block; border-radius:999px; padding:2px 12px; font-weight:600; font-size:.85rem; white-space:nowrap; }
.chip.ready { background:#E3F1EA; color:var(--ok); } .chip.wait { background:#F6EEDC; color:var(--review); }
.chip.off { background:#F3E3E1; color:var(--flag); }
.tiles { display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin:6px 0 18px 0; }
.tile { background:#fff; border:1px solid var(--rule); border-radius:10px; padding:12px 16px; }
.tile b { display:block; font-family:'Martel',serif; font-weight:800; font-size:1.9rem; line-height:1.15; }
.tile span { color:var(--muted); font-size:.92rem; }
.card-cap { display:flex; justify-content:space-between; align-items:center; gap:6px; margin:-4px 0 2px 0; }
.card-cap .n { font-weight:600; font-size:.9rem; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pill { display:inline-block; border:1.5px solid currentColor; border-radius:4px; padding:0 7px; font-weight:700;
        font-size:.74rem; letter-spacing:.04em; white-space:nowrap; background:rgba(255,255,255,.7); }
.truth { font-size:.8rem; font-weight:600; }
.truth.good { color:var(--ok); } .truth.bad { color:var(--flag); }
.meter { position:relative; height:16px; background:#E9EDE7; border-radius:999px; margin:6px 0 4px 0; overflow:visible; }
.meter .fill { position:absolute; left:0; top:0; bottom:0; border-radius:999px; }
.meter .cut { position:absolute; top:-5px; bottom:-5px; width:3px; background:var(--ink); border-radius:2px; }
.meter-l { display:flex; justify-content:space-between; color:var(--muted); font-size:.85rem; }
table.facts { width:100%; border-collapse:collapse; font-size:.97rem; margin-top:6px; }
table.facts td { padding:6px 4px; border-bottom:1px solid var(--rule); vertical-align:top; border-left:none; border-right:none; }
table.facts td.k { color:var(--muted); width:46%; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:.88rem; color:var(--ink); margin:4px 0 0 0; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.zoomcap { color:var(--muted); font-size:.85rem; text-align:center; margin-top:-8px; }
.stApp p.side-h { font-family:'Martel',serif; font-weight:700; font-size:1.15rem; margin:0 0 6px 0; }
.stApp p.detail-h { font-family:'Martel',serif; font-weight:800; font-size:1.5rem; margin:0 0 6px 0; }
.stApp p.detail-h span { font-family:'Mukta',sans-serif; font-weight:500; font-size:1.05rem; color:var(--muted); }
.card-n { font-weight:600; font-size:.78rem; color:var(--ink); margin:-6px 0 2px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.card-cap { justify-content:flex-start; flex-wrap:wrap; margin:0 0 4px 0; gap:6px; }
</style>
"""


# ------------------------------------------------------------------ storage
def _read_rows(root):
    root = Path(root); rows = []
    if (root / "samples.csv").exists():
        for r in csv.DictReader(open(root / "samples.csv")):
            rows.append(r)
    if (root / "index.json").exists():
        rows += json.loads((root / "index.json").read_text())
    out = []
    for r in rows:
        pid = r["id"]
        res = root / "results" / f"{pid}.json"
        out.append({**r, "page": root / "pages" / f"{pid}.jpg", "prob": root / "results" / f"{pid}.png",
                    "result": json.loads(res.read_text()) if res.exists() else None,
                    "true_mask": (root / r["true_mask"]) if r.get("true_mask") else None})
    return out


def _save_result(root, pid, prob, v):
    (root / "results").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "results" / f"{pid}.png"), np.clip(prob * 255 + 0.5, 0, 255).astype(np.uint8))
    (root / "results" / f"{pid}.json").write_text(json.dumps(v))


@st.cache_data(show_spinner=False)
def _load(path, mtime, gray=False):
    if gray:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def load_rgb(p):
    return _load(str(p), Path(p).stat().st_mtime)


def load_prob(p):
    return _load(str(p), Path(p).stat().st_mtime, gray=True).astype(np.float32) / 255.0


def load_mask(p):
    return _load(str(p), Path(p).stat().st_mtime, gray=True) > 0


# ------------------------------------------------------------------ pictures
def heat_overlay(rgb, prob, strength=0.75):
    a = (np.clip((prob - 0.05) / 0.55, 0, 1) * strength)[..., None]
    out = rgb.astype(np.float32) * (1 - a) + RED * a
    return out.clip(0, 255).astype(np.uint8)


def heat_only(prob):
    """White paper, ink-coloured for low, red for high: a heatmap that reads on a slide or a screen."""
    p = np.clip(prob, 0, 1)[..., None]
    lo = np.array([250, 251, 249], np.float32); mid = np.array([240, 180, 60], np.float32)
    out = np.where(p < 0.5, lo * (1 - p * 2) + mid * (p * 2), mid * (2 - p * 2) + RED * (p * 2 - 1))
    return out.clip(0, 255).astype(np.uint8)


def mask_view(rgb, mask, color):
    out = (rgb.astype(np.float32) * 0.55 + 255 * 0.45)
    out[mask] = out[mask] * 0.25 + color * 0.75
    return out.clip(0, 255).astype(np.uint8)


def agreement(rgb, pred, truth):
    out = (rgb.astype(np.float32) * 0.5 + 255 * 0.5)
    for m, c in ((truth & ~pred, GREEN), (pred & ~truth, RED), (pred & truth, GOLD)):
        out[m] = out[m] * 0.2 + c * 0.8
    return out.clip(0, 255).astype(np.uint8)


def draw_box(img, box, pad=10):
    if box:
        x0, y0, x1, y1 = box
        cv2.rectangle(img, (max(0, x0 - pad), max(0, y0 - pad)), (x1 + pad, y1 + pad), AMBER, 5)
    return img


def crop(img, box, w=520, h=150):
    if not box:
        return None
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    hw, hh = max(w // 2, (x1 - x0) // 2 + 60), max(h // 2, (y1 - y0) // 2 + 40)
    H, W = img.shape[:2]
    return img[max(0, cy - hh):min(H, cy + hh), max(0, cx - hw):min(W, cx + hw)]


def f1(prob, truth, t=0.5):
    p = prob >= t
    return 2 * (p & truth).sum() / max(1, p.sum() + truth.sum())


@st.cache_data(show_spinner=False)
def thumb(page, prob, mtime):
    rgb = cv2.cvtColor(cv2.imread(page), cv2.COLOR_BGR2RGB)
    pr = cv2.imread(prob, cv2.IMREAD_GRAYSCALE)
    img = heat_overlay(rgb, pr.astype(np.float32) / 255) if pr is not None else rgb
    h, w = img.shape[:2]
    return cv2.resize(img, (300, int(300 * h / w)), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ status
def model_status():
    ok, why = M.dependencies_ok()
    if not ok:
        return "off", f"Model packages not installed ({why}). The 20 test pages still work."
    if M.weights_path().exists():
        return "ready", "Tamper model ready"
    return "wait", "Model downloads on first check (357 MB)"


def check_pages(files, root):
    root = Path(root); (root / "pages").mkdir(parents=True, exist_ok=True)
    idx_p = root / "index.json"
    index = json.loads(idx_p.read_text()) if idx_p.exists() else []
    bar = st.progress(0.0, text="Starting")
    if not M.weights_path().exists():
        M.ensure_weights(lambda f, t: bar.progress(min(f, 1.0), text=t))
    bar.progress(0.0, text="Loading the model")
    M.load_model()
    done = []
    for k, f in enumerate(files, 1):
        data = f.getvalue()
        pid = "u" + uuid.uuid4().hex[:8]
        try:
            jpg, how = M.to_jpeg_bytes(data, f.name)
        except Exception as e:
            st.warning(f"{f.name}: {e}"); continue
        (root / "pages" / f"{pid}.jpg").write_bytes(jpg)

        def prog(fr, t, k=k, n=len(files), name=f.name):
            bar.progress(min(1.0, ((k - 1) + fr) / n), text=f"{name} ({k} of {n}): {t}")
        t0 = time.time()
        prob, rgb = M.predict(jpg, prog)
        v = M.verdict(prob); v.update({"seconds": round(time.time() - t0, 1), "input": how, "size": [rgb.shape[1], rgb.shape[0]]})
        _save_result(root, pid, prob, v)
        index.append({"id": pid, "name": f.name, "truth": "", "edit": "", "field": "", "doc_type": "", "true_mask": "",
                      "added": time.strftime("%Y-%m-%d %H:%M")})
        idx_p.write_text(json.dumps(index, indent=1)); done.append(pid)
    bar.empty()
    return done


# ------------------------------------------------------------------ the tab
def render():
    st.markdown(CSS, unsafe_allow_html=True)
    state, msg = model_status()
    st.markdown(f'<div class="lab-hero"><div><p class="t">Tamper lab</p><p class="s">Drop scanned pages or PDFs. '
                f'The tamper model looks for anything repainted, pasted or erased, marks where, and says why it '
                f'flagged the page.</p></div><span class="chip {state}">{html.escape(msg)}</span></div>',
                unsafe_allow_html=True)

    up_col, which_col = st.columns([3, 2], gap="large")
    with up_col:
        files = st.file_uploader("Pages to check", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True,
                                 label_visibility="collapsed", key=f"lab_upload_{st.session_state.get('lab_round', 0)}")
        c1, c2 = st.columns([1, 1])
        if c1.button(f"Check {len(files)} page{'s' if len(files) != 1 else ''}" if files else "Check pages",
                     type="primary", disabled=not files or state == "off", width="stretch"):
            new = check_pages(files, STORE_DIR)
            if new:      # show the pages just checked, and empty the upload box
                st.session_state["lab_coll"] = "mine"; st.session_state["lab_sel"] = new[-1]
                st.session_state["lab_round"] = st.session_state.get("lab_round", 0) + 1
                st.rerun()
        if c2.button("Clear my pages", width="stretch",
                     disabled=not (Path(STORE_DIR) / "index.json").exists()):
            import shutil
            shutil.rmtree(STORE_DIR, ignore_errors=True); st.session_state.pop("lab_sel", None)
            st.session_state["lab_coll"] = "test"; st.rerun()
    samples, mine = _read_rows(SAMPLES_DIR), _read_rows(STORE_DIR)
    with which_col:
        names = {"test": f"Test pages ({len(samples)})", "mine": f"My pages ({len(mine)})"}
        st.session_state.setdefault("lab_coll", "test")
        pick = st.segmented_control("Collection", list(names), format_func=names.get, key="lab_coll",
                                    label_visibility="collapsed") or "test"
        st.caption("Test pages: 10 genuine and 10 edited pages from institutions the model never saw, "
                   "with the true edits known. Your own pages are kept until you clear them.")
    items = samples if pick == "test" else mine
    items = [i for i in items if i["result"]]
    if not items:
        st.info("Nothing checked here yet. Drop pages above and press Check." if pick == "mine"
                else "The test pages have not been scored yet.")
        return

    # scoreboard
    known = [i for i in items if i.get("truth") in ("tampered", "genuine")]
    flagged = [i for i in items if i["result"]["flagged"]]
    if known:
        tam = [i for i in known if i["truth"] == "tampered"]; gen = [i for i in known if i["truth"] == "genuine"]
        caught = sum(i["result"]["flagged"] for i in tam); fa = sum(i["result"]["flagged"] for i in gen)
        f1s = [f1(load_prob(i["prob"]), load_mask(i["true_mask"])) for i in tam if i["true_mask"]]
        tiles = [(f"{len(items)}", "pages checked"), (f"{caught} of {len(tam)}", "edited pages caught"),
                 (f"{fa} of {len(gen)}", "genuine pages wrongly flagged"),
                 (f"{np.mean(f1s):.2f}" if f1s else "–", "overlap with the true edit (pixel F1)")]
    else:
        secs = [i["result"].get("seconds") for i in items if i["result"].get("seconds")]
        tiles = [(f"{len(items)}", "pages checked"), (f"{len(flagged)}", "flagged for an officer"),
                 (f"{len(items) - len(flagged)}", "no edit found"), (f"{np.mean(secs):.0f} s" if secs else "–", "per page")]
    st.markdown('<div class="tiles">' + "".join(f'<div class="tile"><b>{a}</b><span>{b}</span></div>' for a, b in tiles)
                + "</div>", unsafe_allow_html=True)

    # a strip of pages, then the selected page at full width
    ids = [i["id"] for i in items]
    if st.session_state.get("lab_sel") not in ids:
        st.session_state["lab_sel"] = ids[0]
    per_row = 10
    for r0 in range(0, len(items), per_row):
        cols = st.columns(per_row, gap="small")
        for col, it in zip(cols, items[r0:r0 + per_row]):
            res = it["result"]; fl = res["flagged"]; sel = st.session_state["lab_sel"] == it["id"]
            with col:
                st.image(thumb(str(it["page"]), str(it["prob"]), it["prob"].stat().st_mtime), width="stretch")
                verdict = '<span class="pill flag">EDIT</span>' if fl else '<span class="pill ok">CLEAN</span>'
                truth = ""
                if it.get("truth") in ("tampered", "genuine"):
                    right = fl == (it["truth"] == "tampered")
                    truth = f'<span class="truth {"good" if right else "bad"}">{"✓" if right else "✗"}</span>'
                st.markdown(f'<div class="card-n">{html.escape(it["name"])}</div><div class="card-cap">{verdict}{truth}</div>',
                            unsafe_allow_html=True)
                if st.button("Shown" if sel else "View", key=f"open_{it['id']}", width="stretch",
                             type="primary" if sel else "secondary", help=f"Open {it['name']}"):
                    st.session_state["lab_sel"] = it["id"]; st.rerun()
    if known:
        st.caption("✓ the model got it right   ✗ it got it wrong (a missed edit or a false alarm)")
    st.divider()
    detail(next(i for i in items if i["id"] == st.session_state["lab_sel"]))


def detail(it):
    res = it["result"]; rgb = load_rgb(it["page"]); prob = load_prob(it["prob"])
    pred = prob >= M.BLOB_PROB
    truth = load_mask(it["true_mask"]) if it.get("true_mask") else None
    fl = res["flagged"]
    st.markdown(f'<p class="detail-h">{html.escape(it["name"])}'
                + (f' <span>· {html.escape(DOC_NAMES.get(it["doc_type"], it["doc_type"]))}</span>' if it.get("doc_type") else "")
                + "</p>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")
    with left:
        views = ["On page", "Heatmap", "Mask"] + (["Truth", "Compare"] if truth is not None else []) + ["Original"]
        view = st.segmented_control("View", views, default="On page", key=f"view_{it['id']}",
                                    label_visibility="collapsed") or "On page"
        if view == "On page":
            img = draw_box(heat_overlay(rgb, prob), res["box"] if fl else None)
        elif view == "Heatmap":
            img = heat_only(prob)
        elif view == "Mask":
            img = mask_view(rgb, pred, RED)
        elif view == "Truth":
            img = mask_view(rgb, truth, GREEN)
        elif view == "Compare":
            img = agreement(rgb, pred, truth)
        else:
            img = rgb
        st.image(img, width="stretch")
        if view == "Compare":
            st.markdown('<div class="legend"><span><i style="background:#E2A014"></i>both agree</span>'
                        '<span><i style="background:#168052"></i>true edit the model missed</span>'
                        '<span><i style="background:#C8201C"></i>marked, not edited</span></div>', unsafe_allow_html=True)
        elif view in ("On page", "Heatmap"):
            st.markdown('<div class="legend"><span><i style="background:#F0B43C"></i>some suspicion</span>'
                        '<span><i style="background:#C8201C"></i>strong: likely edited</span>'
                        + ('<span><i style="background:#B0680E"></i>box: largest marked area</span>' if fl else "") + "</div>",
                        unsafe_allow_html=True)
    with right:
        cls = "flag" if fl else "ok"
        word = "Edit found" if fl else "No edit found"
        sub = "flag for an officer" if fl else "nothing marked above the line"
        st.markdown(f'<div class="stamp {cls}">{word}<small>{sub}</small></div>', unsafe_allow_html=True)
        area, cut = res["area"], res["cutoff"]
        scale = max(cut * 4, area * 1.1, 1)
        st.markdown(f'<div style="margin-top:18px"><b>Largest marked area</b></div>'
                    f'<div class="meter"><div class="fill" style="width:{min(100, area / scale * 100):.1f}%;'
                    f'background:{"var(--flag)" if fl else "var(--ok)"}"></div>'
                    f'<div class="cut" style="left:{cut / scale * 100:.1f}%"></div></div>'
                    f'<div class="meter-l"><span>{area:,} px marked</span><span>flags at {cut} px</span></div>',
                    unsafe_allow_html=True)
        rows = [("Highest suspicion on the page", f"{res['peak']:.0%}"), ("Page", f"{res['size'][0]} × {res['size'][1]} px, {res.get('input', '')}")]
        if res.get("seconds"):
            rows.append(("Time", f"{res['seconds']:.0f} s"))
        if it.get("truth") in ("tampered", "genuine"):
            rows.append(("Truth", "edited" if it["truth"] == "tampered" else "genuine"))
            if it["truth"] == "tampered":
                rows.append(("What was changed", f"{EDIT_WORDS.get(it['edit'], it['edit'])} in the {pretty(it['field'])}"))
                rows.append(("Overlap with the true edit", f"{f1(prob, truth):.2f} (1.00 = exact)"))
            right_ = fl == (it["truth"] == "tampered")
            rows.append(("Model was", "right" if right_ else "wrong"))
        st.markdown('<table class="facts">' + "".join(f'<tr><td class="k">{k}</td><td>{html.escape(str(v))}</td></tr>' for k, v in rows)
                    + "</table>", unsafe_allow_html=True)
    box = res["box"] if fl else None
    if truth is not None and truth.any() and not fl:
        ys, xs = np.nonzero(truth); box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    if box:
        st.markdown('<p class="side-h" style="margin-top:14px">Close-up</p>', unsafe_allow_html=True)
        z1, z2 = st.columns(2, gap="medium")
        z1.image(crop(rgb, box, 760, 230), width="stretch")
        z1.markdown('<p class="zoomcap">as uploaded</p>', unsafe_allow_html=True)
        z2.image(crop(draw_box(heat_overlay(rgb, prob), box, pad=6), box, 760, 230), width="stretch")
        z2.markdown(f'<p class="zoomcap">{"what the model marked" if fl else "where the true edit is: the model did not mark it"}</p>',
                    unsafe_allow_html=True)
