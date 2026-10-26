# ======================= PATHS AND SETTINGS (edit if needed) =======================
DATA_ROOT = "/kaggle/input/datasets/vamsikrishnasai05/data-genx"     # our generator output (has tamper/test/index.csv)
WORK_DIR = "/kaggle/working/baseline"                                  # code, checkpoints, results
DOC_BENCH_FOLDER = "https://drive.google.com/drive/folders/1RZZxwYIX5e-lHKDw1CD45FwFC0QqJ7im"  # ForensicHub document checkpoints
RTM_FILE_ID = "11AHZ8ih_kDCFilGceevppcGkKR4vDJD2"                     # RTM dataset zip on Google Drive (from the RTM repo)
CHECKPOINT_PATH = ""        # set to a local .pth to skip the Drive download (e.g. if Drive quota is hit)
RTM_DIR = ""                # set to an unpacked RTM folder to skip its download
EVAL_OURS = True            # our synthetic tamper test split
EVAL_RTM = True             # RTM test split (real human edits)
LIMIT = 0                   # 0 = all images; e.g. 50 for a quick trial run
DEMO_PAGES_DIR = ""         # optional: the app's demo_data folder (uploaded as a dataset); heatmaps go to WORK_DIR/tamper_maps
WINDOW, STRIDE = 512, 384   # sliding window on full pages; both multiples of 8 so DCT blocks stay aligned
BATCH = 8
THRESHOLD = 0.5
# ====================================================================================
"""
No-training baseline: the public FFDN document-tampering checkpoint (ForensicHub
Document Benchmark, trained on DocTamper) scored on our synthetic tamper test split
and on RTM. Outputs per-image scores, pixel F1, image-level AUC and false alarms.
"""
import csv, glob, json, os, subprocess, sys, types, importlib.util, zipfile
from pathlib import Path

def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

try:
    import jpegio, efficientnet_pytorch, gdown, timm  # noqa: F401
except ImportError:
    sh([sys.executable, "-m", "pip", "install", "-q", "jpegio", "efficientnet_pytorch", "gdown", "timm"])

import cv2
import numpy as np
import torch
import torch.nn as nn
import jpegio
from tqdm.auto import tqdm

work = Path(WORK_DIR); work.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

# ------------------------------------------------------------------ 1. FFDN code (ForensicHub, loaded without the full framework)
hub = work / "ForensicHub"
if not hub.exists():
    sh(["git", "clone", "-q", "--depth", "1", "https://github.com/scu-zjz/ForensicHub.git", str(hub)])
ffdn_dir = hub / "ForensicHub" / "tasks" / "document" / "models" / "ffdn"
# FFDN imports only the registry decorator and BaseModel from the framework; stand in for those two.
fh = types.ModuleType("ForensicHub"); fh.__path__ = []
reg = types.ModuleType("ForensicHub.registry"); reg.register_model = lambda name: (lambda cls: cls)
core = types.ModuleType("ForensicHub.core"); core.__path__ = []
bm = types.ModuleType("ForensicHub.core.base_model"); bm.BaseModel = nn.Module
sys.modules.update({"ForensicHub": fh, "ForensicHub.registry": reg, "ForensicHub.core": core, "ForensicHub.core.base_model": bm})
spec = importlib.util.spec_from_file_location("ffdn_pkg", ffdn_dir / "__init__.py", submodule_search_locations=[str(ffdn_dir)])
pkg = importlib.util.module_from_spec(spec); sys.modules["ffdn_pkg"] = pkg; spec.loader.exec_module(pkg)
ffdn_mod = importlib.import_module("ffdn_pkg.ffdn")
ffdn_mod.FFDN.init_vph = lambda self, weight_path: None     # ImageNet backbone init not needed: the checkpoint holds all weights

# ------------------------------------------------------------------ 2. checkpoint
def fetch_checkpoint():
    if CHECKPOINT_PATH:
        return Path(CHECKPOINT_PATH)
    import gdown
    files = gdown.download_folder(DOC_BENCH_FOLDER, skip_download=True, quiet=True)
    print("Doc_Bench folder:")
    for f in files:
        print("   ", f.path)
    cand = [f for f in files if "ffdn" in f.path.lower() and f.path.lower().endswith((".pth", ".pt", ".ckpt"))]
    assert cand, "no FFDN checkpoint in the folder listing above - set CHECKPOINT_PATH by hand"
    f = sorted(cand, key=lambda f: f.path)[-1]
    out = work / "ckpt" / Path(f.path).name
    out.parent.mkdir(exist_ok=True)
    if not out.exists():
        gdown.download(id=f.id, output=str(out), quiet=False)
    return out


def load_ffdn(path):
    model = ffdn_mod.FFDN()
    ck = torch.load(path, map_location="cpu", weights_only=False)   # ForensicHub checkpoint (trusted source; see its issue #9)
    sd = ck
    for key in ("model", "state_dict", "model_state_dict", "net"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    own = model.state_dict()
    best, best_hit = None, -1
    for prefix in ("", "module.", "model.", "module.model.", "base_model.", "module.base_model."):
        cand = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        hit = sum(1 for k in cand if k in own and own[k].shape == cand[k].shape)
        if hit > best_hit:
            best, best_hit = cand, hit
    res = model.load_state_dict({k: v for k, v in best.items() if k in own and own[k].shape == v.shape}, strict=False)
    print(f"loaded {best_hit} of {len(own)} tensors; missing {len(res.missing_keys)}")
    if res.missing_keys:
        print("   first missing:", res.missing_keys[:8])
    assert best_hit >= 0.98 * len(own), "checkpoint does not match FFDN - stopping rather than scoring random weights"
    return model.to(dev).eval()


# ------------------------------------------------------------------ 3. inference on one JPEG
Q75 = np.array([8, 6, 5, 8, 12, 20, 26, 31, 6, 6, 7, 10, 13, 29, 30, 28, 7, 7, 8, 12, 20, 29, 35, 28,
                7, 9, 11, 15, 26, 44, 40, 31, 9, 11, 19, 28, 34, 55, 52, 39, 12, 18, 28, 32, 41, 52, 57, 46,
                25, 32, 39, 44, 52, 61, 60, 51, 36, 46, 48, 49, 56, 50, 52, 50], np.int64).reshape(8, 8)  # libjpeg quality 75
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_jpeg(path):
    """RGB pixels, luma DCT (|coef| clipped to 20, as FFDN was trained) and the luma quant table."""
    if not str(path).lower().endswith((".jpg", ".jpeg")):     # FFDN needs JPEG coefficients: re-encode losslessly-ish
        tmp = work / "tmp_reenc.jpg"
        cv2.imwrite(str(tmp), cv2.imread(str(path)), [cv2.IMWRITE_JPEG_QUALITY, 100]); path = tmp
    j = jpegio.read(str(path))
    dct = np.clip(np.abs(j.coef_arrays[0]), 0, 20).astype(np.int64)
    qt = j.quant_tables[0].astype(np.int64)
    if qt.max() > 63:          # FFDN's quant-table embedding has 64 entries; same fallback as ForensicHub's loader
        qt = Q75
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return rgb, dct, qt


def starts(n):
    if n <= WINDOW:
        return [0]
    s = list(range(0, n - WINDOW + 1, STRIDE))
    last = ((n - WINDOW) // 8) * 8
    return s if s[-1] == last else s + [last]


@torch.no_grad()
def predict(model, path):
    rgb, dct, qt = read_jpeg(path)
    h, w = rgb.shape[:2]
    H, W = max(dct.shape[0], WINDOW), max(dct.shape[1], WINDOW)      # DCT array is padded to whole blocks
    img = np.full((H, W, 3), 255, np.uint8); img[:h, :w] = rgb
    D = np.zeros((H, W), np.int64); D[:dct.shape[0], :dct.shape[1]] = dct
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    boxes = [(y, xx) for y in starts(H) for xx in starts(W)]
    qtt = torch.from_numpy(qt)[None, None]
    for i in range(0, len(boxes), BATCH):
        bb = boxes[i:i + BATCH]
        im = torch.from_numpy(np.stack([x[y:y + WINDOW, xx:xx + WINDOW] for y, xx in bb])).permute(0, 3, 1, 2).to(dev)
        dc = torch.from_numpy(np.stack([D[y:y + WINDOW, xx:xx + WINDOW] for y, xx in bb]))[:, None].to(dev)
        mk = torch.zeros((len(bb), 1, WINDOW, WINDOW), dtype=torch.long, device=dev)
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            out = model(image=im, dct=dc, qt=qtt.expand(len(bb), 1, 8, 8).to(dev), mask=mk)["pred_mask"]
        out = out.float().cpu().numpy()[:, 0]
        for (y, xx), o in zip(bb, out):
            acc[y:y + WINDOW, xx:xx + WINDOW] += o; cnt[y:y + WINDOW, xx:xx + WINDOW] += 1
    return (acc / np.maximum(cnt, 1))[:h, :w]


# ------------------------------------------------------------------ 4. metrics
_seen_mask_kinds = set()


def read_mask(path):
    """Tampered pixels as 0/1. Handles 0/255 grey masks and VOC-style palette masks (class index 1, 255 = ignore)."""
    from PIL import Image
    im = Image.open(path)
    a = np.array(im)
    if a.ndim == 3:
        a = a.max(2)
    kind = (im.mode, tuple(np.unique(a)[:6]))
    if im.mode not in _seen_mask_kinds:                 # print once per mask format, so the reading can be checked
        _seen_mask_kinds.add(im.mode)
        print(f"mask format {path.name}: mode {kind[0]}, values {kind[1]}", flush=True)
    t = (a > 0) & (a != 255) if im.mode == "P" else a > 0
    return t.astype(np.uint8)

def f1(prob, gt):
    p = prob >= THRESHOLD; g = gt > 0
    tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def auc(pos, neg):
    """Chance a tampered page scores above a clean one; ties count half (many pages score exactly 0)."""
    if not pos or not neg:
        return float("nan")
    s = np.asarray(list(pos) + list(neg), dtype=float)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(counts) - (counts - 1) / 2)[inv]          # average rank for tied values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(model, items, name):
    """items: list of (image_path, mask_path or None). Returns summary dict and writes per-image CSV."""
    if LIMIT:
        rnd = np.random.default_rng(0); idx = rnd.permutation(len(items))[:LIMIT]; items = [items[i] for i in idx]
    rows, pos, neg, f1s, fa = [], [], [], [], []
    for img_p, m_p in tqdm(items, desc=name, unit="img"):
        prob = predict(model, img_p)
        gt = np.zeros(prob.shape, np.uint8)
        if m_p:
            g = read_mask(m_p)
            gt = cv2.resize(g, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_NEAREST)
        tampered = gt.any()
        score = float(np.percentile(prob, 99.9))          # image score: near-max, robust to a few hot pixels
        (pos if tampered else neg).append(score)
        if tampered:
            f1s.append(f1(prob, gt))
        else:
            fa.append(float((prob >= THRESHOLD).mean()))
        rows.append([str(img_p), int(tampered), round(score, 4), round(f1s[-1], 4) if tampered else ""])
    with open(work / f"scores_{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["image", "tampered", "score", "pixel_f1"]] + rows)
    pred_t = [s >= THRESHOLD for s in pos]; pred_a = [s >= THRESHOLD for s in neg]
    out = {"set": name, "images": len(rows), "tampered": len(pos), "authentic": len(neg),
           "pixel_f1_mean": round(float(np.mean(f1s)), 4) if f1s else None,
           "image_auc": round(auc(pos, neg), 4),
           "tampered_flagged": round(float(np.mean(pred_t)), 4) if pos else None,
           "authentic_flagged": round(float(np.mean(pred_a)), 4) if neg else None,
           "authentic_pixel_false_alarm": round(float(np.mean(fa)), 6) if fa else None}
    print(json.dumps(out, indent=1), flush=True)
    return out


def our_items():
    root = next(p.parent for p in Path(DATA_ROOT).rglob("index.csv") if p.parent.name == "test" and p.parent.parent.name == "tamper")
    items = []
    for r in csv.DictReader(open(root / "index.csv")):
        items.append((root / r["image"], root / r["mask"] if r["tampered"] == "1" else None))
    return items


def rtm_items():
    base = Path(RTM_DIR) if RTM_DIR else work / "RTM"
    if not RTM_DIR and not any(base.rglob("test.txt")):
        import gdown
        z = work / "RTM.zip"
        if not z.exists():
            gdown.download(id=RTM_FILE_ID, output=str(z), quiet=False)
        with zipfile.ZipFile(z) as zf:
            for m in tqdm(zf.infolist(), desc="unzip RTM", unit="file"):
                zf.extract(m, base)
    test = next(base.rglob("test.txt"))
    imgs = {p.stem: p for p in test.parent.rglob("*") if p.parent.name == "JPEGImages"}
    masks = {p.stem: p for p in test.parent.rglob("*") if p.parent.name == "SegmentationClass"}
    names = [l.strip() for l in open(test) if l.strip()]
    print(f"RTM test list: {len(names)} names, {len(imgs)} images, {len(masks)} masks found")
    items = [(imgs[n], masks.get(n)) for n in names if n in imgs]
    assert items, "RTM layout not as documented (JPEGImages/, SegmentationClass/, test.txt) - check the unzip"
    return items


# ------------------------------------------------------------------ 5. run
if __name__ == "__main__" or True:
    model = load_ffdn(fetch_checkpoint())
    summary = []
    if EVAL_OURS:
        summary.append(evaluate(model, our_items(), "ours_test"))
    if EVAL_RTM:
        summary.append(evaluate(model, rtm_items(), "rtm_test"))
    if DEMO_PAGES_DIR:
        demo = Path(DEMO_PAGES_DIR)
        pages = sorted(demo.glob("applications/*/*/pages/*.jpg"))
        for p in tqdm(pages, desc="demo heatmaps", unit="page"):
            prob = predict(model, p)
            out = work / "tamper_maps" / p.parent.parent.name / p.stem      # inputs are read-only on Kaggle
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out.with_suffix(".png")), (prob * 255).astype(np.uint8))
            out.with_suffix(".json").write_text(json.dumps({"score": float(np.percentile(prob, 99.9)), "threshold": THRESHOLD,
                                                            "model": "FFDN (ForensicHub, DocTamper-trained, no fine-tuning)"}))
    if DEMO_PAGES_DIR:
        import shutil
        shutil.make_archive(str(work / "tamper_maps"), "zip", work, "tamper_maps")
        print("heatmaps for the app: download tamper_maps.zip and unzip it inside the app's demo_data folder")
    (work / "baseline_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nBASELINE DONE")
    for s in summary:
        print(f"{s['set']:10}  pixel F1 {s['pixel_f1_mean']}  image AUC {s['image_auc']}  "
              f"tampered flagged {s['tampered_flagged']}  authentic flagged {s['authentic_flagged']}")
