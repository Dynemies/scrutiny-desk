# ======================= PATHS AND SETTINGS (edit if needed) =======================
RELEASES_API = "https://api.github.com/repos/Dynemies/scrutiny-desk/releases/latest"   # the model is the .pth file in the latest release
WEIGHTS_URL = "https://github.com/Dynemies/scrutiny-desk/releases/download/Bestpath/ffdn_v2_best.1.pth"   # used if the lookup fails
WEIGHTS_PATH = "models/ffdn_v2_best.pth"   # relative to the app folder; downloaded on first use
CUTOFF_PX = 143        # page is flagged when the largest marked area is at least this big (chosen on 1,555 validation pages)
BLOB_PROB = 0.3        # a pixel counts as marked at this probability
WINDOW, STRIDE = 512, 384
# ====================================================================================
"""Runs the fine-tuned FFDN tamper model on one page and returns the heatmap and the verdict.
Same preprocessing and page rule as training/finetune_ffdn.py and the Kaggle scoring cells."""
import io
import os
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
Q75 = np.array([8, 6, 5, 8, 12, 20, 26, 31, 6, 6, 7, 10, 13, 29, 30, 28, 7, 7, 8, 12, 20, 29, 35, 28,
                7, 9, 11, 15, 26, 44, 40, 31, 9, 11, 19, 28, 34, 55, 52, 39, 12, 18, 28, 32, 41, 52, 57, 46,
                25, 32, 39, 44, 52, 61, 60, 51, 36, 46, 48, 49, 56, 50, 52, 50], np.int64).reshape(8, 8)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def dependencies_ok():
    try:
        import torch, torchvision, jpegio, timm, efficientnet_pytorch  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, str(e)


def weights_path():
    return APP_DIR / WEIGHTS_PATH


def ensure_weights(progress=None):
    """Download the model once (about 357 MB). progress(fraction, text) is called while it downloads."""
    p = weights_path()
    if p.exists() and p.stat().st_size > 100_000_000:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    with urllib.request.urlopen(model_url(), timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0); done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk); done += len(chunk)
            if progress and total:
                progress(done / total, f"Downloading the model: {done / 1e6:.0f} of {total / 1e6:.0f} MB")
    tmp.rename(p)
    return p


def model_url():
    """Find the .pth file attached to the repository's latest release, whatever its name."""
    import json
    try:
        req = urllib.request.Request(RELEASES_API, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            assets = json.load(r).get("assets", [])
        for a in assets:
            if a["name"].endswith(".pth"):
                return a["browser_download_url"]
    except Exception:
        pass
    return WEIGHTS_URL


@lru_cache(maxsize=1)
def load_model():
    import torch
    from tamper.ffdn.ffdn import FFDN
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    m = FFDN()
    m.load_state_dict(torch.load(weights_path(), map_location="cpu"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), dev


def to_jpeg_bytes(data, name=""):
    """The model reads JPEG compression coefficients, so every page is fed as a JPEG.
    JPEG stays byte-identical; a PDF gives its embedded page image (or a 150-dpi render); other images are re-saved."""
    low = name.lower()
    if low.endswith(".pdf") or data[:4] == b"%PDF":
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf"); page = doc[0]
        imgs = page.get_images(full=True)
        if imgs:
            info = doc.extract_image(imgs[0][0])
            if info.get("ext") in ("jpg", "jpeg"):
                return info["image"], "page image embedded in the PDF"
        pix = page.get_pixmap(dpi=150)
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), "PDF rendered at 150 dpi"
    if data[:3] == b"\xff\xd8\xff":
        return data, "original JPEG"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image or PDF the desk can read")
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), "re-saved as JPEG"


def _jpeg_parts(jpg):
    import jpegio
    tmp = Path(f"/tmp/scrutiny_{os.getpid()}_{time.time_ns()}.jpg"); tmp.write_bytes(jpg)
    try:
        j = jpegio.read(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    dct = np.clip(np.abs(j.coef_arrays[0]), 0, 20).astype(np.int64)
    qt = j.quant_tables[0].astype(np.int64)
    return rgb, dct, (Q75 if qt.max() > 63 else qt)


def _starts(n):
    if n <= WINDOW:
        return [0]
    s = list(range(0, n - WINDOW + 1, STRIDE)); last = ((n - WINDOW) // 8) * 8
    return s if s[-1] == last else s + [last]


def predict(jpg, progress=None, batch=1):
    """Heatmap (probability that each pixel was edited), same size as the page.
    batch=1 is the fastest on CPU and peaks at about 1.6 GB of memory (batch=4: about 2.5 GB)."""
    import torch
    model, dev = load_model()
    rgb, dct, qt = _jpeg_parts(jpg)
    h, w = rgb.shape[:2]
    H, W = max(dct.shape[0], WINDOW), max(dct.shape[1], WINDOW)
    img = np.full((H, W, 3), 255, np.uint8); img[:h, :w] = rgb
    D = np.zeros((H, W), np.int64); D[:dct.shape[0], :dct.shape[1]] = dct
    x = torch.from_numpy((img.astype(np.float32) / 255.0 - MEAN) / STD).permute(2, 0, 1)
    Dt = torch.from_numpy(D)
    acc = torch.zeros((H, W)); cnt = torch.zeros((H, W))
    boxes = [(y, xx) for y in _starts(H) for xx in _starts(W)]
    qtt = torch.from_numpy(qt)[None, None]
    with torch.no_grad():
        for i in range(0, len(boxes), batch):
            bb = boxes[i:i + batch]
            im = torch.stack([x[:, y:y + WINDOW, xx:xx + WINDOW] for y, xx in bb]).to(dev)
            dc = torch.stack([Dt[y:y + WINDOW, xx:xx + WINDOW] for y, xx in bb])[:, None].to(dev)
            mk = torch.zeros((len(bb), 1, WINDOW, WINDOW), dtype=torch.long, device=dev)
            out = model(image=im, dct=dc, qt=qtt.expand(len(bb), 1, 8, 8).to(dev), mask=mk)["pred_mask"].float().cpu()[:, 0]
            for (y, xx), o in zip(bb, out):
                acc[y:y + WINDOW, xx:xx + WINDOW] += o; cnt[y:y + WINDOW, xx:xx + WINDOW] += 1
            if progress:
                done = min(i + batch, len(boxes))
                progress(done / len(boxes), f"Reading the page: part {done} of {len(boxes)}")
    return (acc / cnt.clamp(min=1)).numpy()[:h, :w].astype(np.float32), rgb


def verdict(prob):
    """The page rule used everywhere in the project: the largest area marked at >= BLOB_PROB, against CUTOFF_PX."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((prob >= BLOB_PROB).astype(np.uint8), connectivity=8)
    if n < 2:
        return {"area": 0, "flagged": False, "box": None, "peak": round(float(prob.max()), 4), "cutoff": CUTOFF_PX}
    j = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    x, y, w, h, a = (int(v) for v in stats[j])
    return {"area": a, "flagged": a >= CUTOFF_PX, "box": [x, y, x + w, y + h], "peak": round(float(prob.max()), 4),
            "cutoff": CUTOFF_PX}


def analyse(data, name="", progress=None):
    t0 = time.time()
    jpg, how = to_jpeg_bytes(data, name)
    prob, rgb = predict(jpg, progress)
    v = verdict(prob)
    v.update({"seconds": round(time.time() - t0, 1), "input": how, "size": [rgb.shape[1], rgb.shape[0]]})
    return prob, rgb, v
