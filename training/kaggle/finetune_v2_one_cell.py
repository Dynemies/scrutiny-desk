# ======================= PATHS AND SETTINGS (edit if needed) =======================
DATA_ROOT = "/kaggle/input/datasets/vamsikrishnasai05/data-genx"   # generator output (tamper/train, validation, test)
START_SEARCH = "/kaggle/input"   # searched for <START_FOLDER_NAME>/best.pth = the first fine-tune (add its output as an input)
START_FOLDER_NAME = "finetune"
WORK_DIR = "/kaggle/working/finetune_v2"   # everything v2 writes: hotspots, ep1-4.pth, best.pth, CSVs, summary
RESUME_DIR = ""             # "" = fresh. To continue an interrupted v2 run: add its output as input and put "/kaggle/input" here
EPOCHS = 4
STEPS_PER_EPOCH = 600
BATCH = 8                   # 4 per T4
CROP = 512
LR = 3e-5                   # lower than the first run (1e-4): we refine a good model, not retrain it
P_TAMPER_CROP = 0.5         # crops centred on an edit (same as the first run)
P_HARDNEG = 0.25            # NEW: crops on clean pages, half at mined false spots, half on clean text
P_RECOMPRESS = 0.7
MINE_PAGES = 2000           # clean training pages scanned for false spots before training
MINE_MIN_AREA = 20          # ignore specks smaller than this (pixels)
VAL_PAGES = 400             # fixed validation pages per epoch: every insertion + a seeded random rest
MAX_CLEAN_FLAGGED = 0.01    # epoch choice: most tampered pages caught with at most 1% of clean pages flagged
TEST_LIMIT = 0              # 0 = all 1,607 test pages; e.g. 50 for a quick trial
MIN_FREE_RAM_GB = 3.0
MEM_LOG_EVERY = 100
PROGRESS_EVERY = 50
EVAL_BATCH = 8
WORKERS = 4
WINDOW, STRIDE, THRESHOLD = 512, 384, 0.5
SEED = 0
START_CHECKPOINT, DOC_BENCH_FOLDER = "", ""   # unused here (kept so the shared helpers load)
SCRIPT = "/kaggle/working/finetune_v2.py"   # this cell writes the training code here
BUDGET_H = 11.3             # stop starting new epochs so the final test fits in Kaggle's 12 hours
TEST_RESERVE_H = 0.75       # time kept for the final test (split over both GPUs)
# ====================================================================================
"""
ONE CELL: FFDN v2. Continues from the first fine-tune's best.pth and changes only:
  1. hard negatives - spots where the model wrongly fires on clean pages are mined first and trained on;
  2. the epoch is chosen with the rule the app uses (largest marked blob, <=1% clean pages flagged);
  3. every epoch is saved (ep1.pth ... ep4.pth); epoch 0 (the starting model) is scored too, so v2 must beat it.
Every phase runs in its own fresh Python process (no memory build-up). Run the cell again after any stop: it continues.
"""
import csv, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

SETTING_NAMES = ['DATA_ROOT', 'START_SEARCH', 'START_FOLDER_NAME', 'WORK_DIR', 'EPOCHS', 'STEPS_PER_EPOCH', 'BATCH', 'CROP', 'LR', 'P_TAMPER_CROP', 'P_HARDNEG', 'P_RECOMPRESS', 'MINE_PAGES', 'MINE_MIN_AREA', 'VAL_PAGES', 'MAX_CLEAN_FLAGGED', 'TEST_LIMIT', 'MIN_FREE_RAM_GB', 'MEM_LOG_EVERY', 'PROGRESS_EVERY', 'EVAL_BATCH', 'WORKERS', 'WINDOW', 'STRIDE', 'THRESHOLD', 'SEED', 'START_CHECKPOINT', 'DOC_BENCH_FOLDER']
WORKER_BODY = r'''"""
(Shared helpers from finetune_ffdn.py.) v2 continues fine-tuning FFDN (ForensicHub document checkpoint, DocTamper-trained) on the synthetic
AICTE tamper set, keeps the best epoch by validation score, then scores the test split
with exactly the same metrics as the baseline, so the two are comparable.
"""
import csv, io, json, os, random, subprocess, sys, time, types, importlib.util
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
import torch.nn.functional as F
import jpegio
from PIL import Image
from tqdm.auto import tqdm

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
work = Path(WORK_DIR); work.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
n_gpu = torch.cuda.device_count()
print("device:", dev, [torch.cuda.get_device_name(i) for i in range(n_gpu)], flush=True)
T0 = time.time()

# ------------------------------------------------------------------ 1. FFDN code
hub = work / "ForensicHub"
if not hub.exists():
    sh(["git", "clone", "-q", "--depth", "1", "https://github.com/scu-zjz/ForensicHub.git", str(hub)])
ffdn_dir = hub / "ForensicHub" / "tasks" / "document" / "models" / "ffdn"
fh = types.ModuleType("ForensicHub"); fh.__path__ = []
reg = types.ModuleType("ForensicHub.registry"); reg.register_model = lambda name: (lambda cls: cls)
core = types.ModuleType("ForensicHub.core"); core.__path__ = []
bm = types.ModuleType("ForensicHub.core.base_model"); bm.BaseModel = nn.Module
sys.modules.update({"ForensicHub": fh, "ForensicHub.registry": reg, "ForensicHub.core": core, "ForensicHub.core.base_model": bm})
spec = importlib.util.spec_from_file_location("ffdn_pkg", ffdn_dir / "__init__.py", submodule_search_locations=[str(ffdn_dir)])
pkg = importlib.util.module_from_spec(spec); sys.modules["ffdn_pkg"] = pkg; spec.loader.exec_module(pkg)
ffdn_mod = importlib.import_module("ffdn_pkg.ffdn")
ffdn_mod.FFDN.init_vph = lambda self, weight_path: None


def fetch_checkpoint():
    if START_CHECKPOINT:
        return Path(START_CHECKPOINT)
    out = work / "ckpt" / "ffdn_ckpt-8.pth"
    if out.exists():
        return out
    import gdown
    files = gdown.download_folder(DOC_BENCH_FOLDER, skip_download=True, quiet=True)
    f = next(f for f in files if "ffdn" in f.path.lower() and f.path.lower().endswith(".pth"))
    out.parent.mkdir(exist_ok=True)
    gdown.download(id=f.id, output=str(out), quiet=False)
    return out


def load_weights(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
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
    model.load_state_dict({k: v for k, v in best.items() if k in own and own[k].shape == v.shape}, strict=False)
    print(f"loaded {best_hit} of {len(own)} tensors from {Path(path).name}", flush=True)
    assert best_hit >= 0.98 * len(own), "checkpoint does not match FFDN - stopping"
    return ck


# ------------------------------------------------------------------ 2. JPEG helpers (same as the baseline)
Q75 = np.array([8, 6, 5, 8, 12, 20, 26, 31, 6, 6, 7, 10, 13, 29, 30, 28, 7, 7, 8, 12, 20, 29, 35, 28,
                7, 9, 11, 15, 26, 44, 40, 31, 9, 11, 19, 28, 34, 55, 52, 39, 12, 18, 28, 32, 41, 52, 57, 46,
                25, 32, 39, 44, 52, 61, 60, 51, 36, 46, 48, 49, 56, 50, 52, 50], np.int64).reshape(8, 8)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def jpeg_parts(src):
    """src: path or JPEG bytes. Returns RGB, |luma DCT| clipped to 20, luma quant table (fallback Q75)."""
    if isinstance(src, (bytes, bytearray)):
        j = jpegio.read(_tmpfile(src)); rgb = np.array(Image.open(io.BytesIO(src)).convert("RGB"))
    else:
        j = jpegio.read(str(src)); rgb = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
    dct = np.clip(np.abs(j.coef_arrays[0]), 0, 20).astype(np.int64)
    qt = j.quant_tables[0].astype(np.int64)
    if qt.max() > 63:
        qt = Q75
    return rgb, dct, qt


def _tmpfile(data):
    p = Path(f"/tmp/ffdn_{os.getpid()}.jpg"); p.write_bytes(data); return str(p)


def read_mask(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a.max(2)
    return (a > 0).astype(np.uint8)


# ------------------------------------------------------------------ 3. data
def split_items(split):
    root = next(p.parent for p in Path(DATA_ROOT).rglob("index.csv") if p.parent.name == split and p.parent.parent.name == "tamper")
    return [(root / r["image"], root / r["mask"] if r["tampered"] == "1" else None) for r in csv.DictReader(open(root / "index.csv"))]


class CropSet(torch.utils.data.Dataset):
    """Random 8-aligned crops. Half centred on an edit; some re-saved at a random JPEG quality on the same 8x8 grid."""
    def __init__(self, items, n):
        self.items, self.n = items, n
        self.tampered = [it for it in items if it[1] is not None]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rnd = random.Random((i * 7919 + int(time.time() * 1000)) % (2 ** 31))
        use_t = rnd.random() < P_TAMPER_CROP and self.tampered
        img_p, m_p = rnd.choice(self.tampered if use_t else self.items)
        rgb = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        mask = read_mask(m_p) if m_p else np.zeros((h, w), np.uint8)
        if use_t:
            ys, xs = np.nonzero(mask)
            cy, cx = int(ys.mean()), int(xs.mean())
            y0 = cy - rnd.randint(CROP // 6, CROP * 5 // 6); x0 = cx - rnd.randint(CROP // 6, CROP * 5 // 6)
        else:
            y0, x0 = rnd.randint(0, max(0, h - CROP)), rnd.randint(0, max(0, w - CROP))
        y0 = (min(max(0, y0), max(0, h - CROP)) // 8) * 8
        x0 = (min(max(0, x0), max(0, w - CROP)) // 8) * 8
        crop = np.full((CROP, CROP, 3), 255, np.uint8); mk = np.zeros((CROP, CROP), np.uint8)
        c = rgb[y0:y0 + CROP, x0:x0 + CROP]; crop[:c.shape[0], :c.shape[1]] = c
        m = mask[y0:y0 + CROP, x0:x0 + CROP]; mk[:m.shape[0], :m.shape[1]] = m
        if rnd.random() < P_RECOMPRESS:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, rnd.randint(70, 100)])
            crop, dct, qt = jpeg_parts(buf.tobytes())
        else:
            _, dct_full, qt = jpeg_parts(img_p)
            dct = np.zeros((CROP, CROP), np.int64)
            d = dct_full[y0:y0 + CROP, x0:x0 + CROP]; dct[:d.shape[0], :d.shape[1]] = d
        x = (crop.astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(dct)[None], torch.from_numpy(qt)[None],
                torch.from_numpy(mk.astype(np.int64))[None])


# ------------------------------------------------------------------ 4. sliding-window prediction and metrics (as the baseline)
def starts(n):
    if n <= WINDOW:
        return [0]
    s = list(range(0, n - WINDOW + 1, STRIDE)); last = ((n - WINDOW) // 8) * 8
    return s if s[-1] == last else s + [last]


@torch.no_grad()
def predict(model, path, bs=None):
    bs = bs or EVAL_BATCH
    rgb, dct, qt = jpeg_parts(path)
    h, w = rgb.shape[:2]
    H, W = max(dct.shape[0], WINDOW), max(dct.shape[1], WINDOW)
    img = np.full((H, W, 3), 255, np.uint8); img[:h, :w] = rgb
    D = np.zeros((H, W), np.int64); D[:dct.shape[0], :dct.shape[1]] = dct
    x = torch.from_numpy((img.astype(np.float32) / 255.0 - MEAN) / STD).permute(2, 0, 1)
    Dt = torch.from_numpy(D)
    acc = torch.zeros((H, W)); cnt = torch.zeros((H, W))
    boxes = [(y, xx) for y in starts(H) for xx in starts(W)]
    qtt = torch.from_numpy(qt)[None, None]
    for i in range(0, len(boxes), bs):
        bb = boxes[i:i + bs]
        im = torch.stack([x[:, y:y + WINDOW, xx:xx + WINDOW] for y, xx in bb]).to(dev)
        dc = torch.stack([Dt[y:y + WINDOW, xx:xx + WINDOW] for y, xx in bb])[:, None].to(dev)
        mk = torch.zeros((len(bb), 1, WINDOW, WINDOW), dtype=torch.long, device=dev)
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            out = model(image=im, dct=dc, qt=qtt.expand(len(bb), 1, 8, 8).to(dev), mask=mk)["pred_mask"]
        out = out.float().cpu()[:, 0]
        for (y, xx), o in zip(bb, out):
            acc[y:y + WINDOW, xx:xx + WINDOW] += o; cnt[y:y + WINDOW, xx:xx + WINDOW] += 1
    return (acc / cnt.clamp(min=1)).numpy()[:h, :w]


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


def evaluate(model, items, name, write_csv=True):
    model.eval()
    rows, pos, neg, f1s, fa = [], [], [], [], []
    for img_p, m_p in tqdm(items, desc=name, unit="img", leave=False):
        prob = predict(model, img_p)
        gt = read_mask(m_p) if m_p else np.zeros(prob.shape, np.uint8)
        tampered = bool(gt.any())
        score = float(np.percentile(prob, 99.9))
        (pos if tampered else neg).append(score)
        if tampered:
            f1s.append(f1(prob, gt))
        else:
            fa.append(float((prob >= THRESHOLD).mean()))
        rows.append([str(img_p), int(tampered), round(score, 4), round(f1s[-1], 4) if tampered else ""])
    if write_csv:
        with open(work / f"scores_{name}.csv", "w", newline="") as f:
            csv.writer(f).writerows([["image", "tampered", "score", "pixel_f1"]] + rows)
    out = {"set": name, "images": len(rows), "tampered": len(pos), "authentic": len(neg),
           "pixel_f1_mean": round(float(np.mean(f1s)), 4) if f1s else None,
           "image_auc": round(auc(pos, neg), 4),
           "tampered_flagged": round(float(np.mean([s >= THRESHOLD for s in pos])), 4) if pos else None,
           "authentic_flagged": round(float(np.mean([s >= THRESHOLD for s in neg])), 4) if neg else None,
           "authentic_pixel_false_alarm": round(float(np.mean(fa)), 6) if fa else None}
    return out



# ------------------------------------------------------------------ 5. v2: hard negatives, blob-rule selection, every epoch saved
def dice_loss(prob, gt):
    gt = gt.float(); inter = (prob * gt).sum((1, 2, 3)); den = prob.sum((1, 2, 3)) + gt.sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (den + 1)).mean()


class Wrapped(nn.Module):
    """FFDN returns its own cross-entropy; add Dice so the small edited regions are not drowned out by background."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, image, dct, qt, mask):
        o = self.m(image=image, dct=dct, qt=qt, mask=mask)
        loss = o["backward_loss"] + dice_loss(o["pred_mask"], mask)
        return loss[None], o["pred_mask"]


def split_meta(split):
    root = next(p.parent for p in Path(DATA_ROOT).rglob("index.csv") if p.parent.name == split and p.parent.parent.name == "tamper")
    return [{"img": root / r["image"], "mask": root / r["mask"] if r["tampered"] == "1" else None,
             "edit": r["edit"], "doc": r["doc_type"], "field": r["field"]} for r in csv.DictReader(open(root / "index.csv"))]


def val_set():
    """Fixed validation pages: every insertion (the hardest edit) plus a seeded random rest."""
    metas = split_meta("validation")
    ins = [m for m in metas if m["edit"] == "insertion"]
    rest = [m for m in metas if m["edit"] != "insertion"]
    k = max(0, min(len(rest), VAL_PAGES - len(ins)))
    return ins + random.Random(SEED).sample(rest, k)


def page_scores(prob):
    n, _, stats, _ = cv2.connectedComponentsWithStats((prob >= 0.3).astype(np.uint8), connectivity=8)
    return {"blob3": int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0,
            "p999": round(float(np.percentile(prob, 99.9)), 4),
            "peak9": round(float(cv2.blur(prob, (9, 9)).max()), 4)}


def pick_cutoff(rows, key="blob3"):
    """Best cut-off: most tampered pages caught while wrongly flagging at most MAX_CLEAN_FLAGGED of clean pages."""
    pos = [r[key] for r in rows if r["tampered"]]; neg = [r[key] for r in rows if not r["tampered"]]
    if not pos or not neg:
        return float("inf"), 0.0, 0.0
    best = (max(pos + neg) + 1, 0.0, 0.0)
    for t in sorted(set(pos + neg)):
        c = sum(p >= t for p in pos) / len(pos); fa = sum(q >= t for q in neg) / len(neg)
        if fa <= MAX_CLEAN_FLAGGED and c > best[1]:
            best = (t, c, fa)
    return best


def score_pages(model, metas, desc):
    model.eval(); rows = []
    for m in tqdm(metas, desc=desc, unit="page", leave=False):
        prob = predict(model, m["img"])
        gt = read_mask(m["mask"]) if m["mask"] else None
        rows.append({"image": m["img"].name, "tampered": int(gt is not None), "edit": m["edit"], "doc_type": m["doc"],
                     "field": m["field"], "pixel_f1": round(float(f1(prob, gt)), 4) if gt is not None else "", **page_scores(prob)})
    return rows


def write_rows(rows, path):
    cols = ["image", "tampered", "edit", "doc_type", "field", "pixel_f1", "blob3", "p999", "peak9"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def summarise(rows):
    cut, caught, fa = pick_cutoff(rows)
    tam = [r for r in rows if r["tampered"]]
    ins = [r for r in tam if r["edit"] == "insertion"]
    f1m = float(np.mean([r["pixel_f1"] for r in tam])) if tam else 0.0
    pos = [r["blob3"] for r in tam]; neg = [r["blob3"] for r in rows if not r["tampered"]]
    return {"pages": len(rows), "cutoff_px": cut, "caught": round(caught, 4), "clean_flagged": round(fa, 4),
            "insertion_caught": round(sum(r["blob3"] >= cut for r in ins) / len(ins), 4) if ins else None,
            "pixel_f1": round(f1m, 4), "auc_blob": round(auc(pos, neg), 4) if pos and neg else None,
            "select_score": round(caught + 0.001 * f1m, 5)}   # tampered pages caught at <=1% false alarms; pixel F1 breaks ties


def start_model_path():
    found = [p for p in Path(START_SEARCH).rglob("best.pth") if p.parent.name == START_FOLDER_NAME]
    assert found, (f"no {START_FOLDER_NAME}/best.pth under {START_SEARCH}: add the first fine-tuning notebook's output as an input")
    return max(found, key=lambda f: f.stat().st_mtime)


class CropSetV2(torch.utils.data.Dataset):
    """Crops: P_TAMPER_CROP centred on an edit; P_HARDNEG on clean pages, half at spots the old model wrongly fired on,
    half on clean ink (text the model must learn is untouched); the rest random. Some re-saved at a random JPEG quality."""
    def __init__(self, items, n, hot, clean):
        self.items, self.n, self.hot, self.clean = items, n, hot, clean
        self.tampered = [it for it in items if it[1] is not None]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rnd = random.Random((i * 7919 + int(time.time() * 1000)) % (2 ** 31))
        r = rnd.random(); centre = None
        if r < P_TAMPER_CROP and self.tampered:
            img_p, m_p = rnd.choice(self.tampered); centre = "mask"
        elif r < P_TAMPER_CROP + P_HARDNEG and self.clean:
            if self.hot and rnd.random() < 0.5:
                h = rnd.choice(self.hot); img_p, m_p, centre = Path(h["image"]), None, (h["cy"], h["cx"])
            else:
                img_p, m_p = rnd.choice(self.clean); centre = "ink"
        else:
            img_p, m_p = rnd.choice(self.items)
        rgb = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        mask = read_mask(m_p) if m_p else np.zeros((h, w), np.uint8)
        if centre == "mask":
            ys, xs = np.nonzero(mask); centre = (int(ys.mean()), int(xs.mean()))
        elif centre == "ink":
            ys, xs = np.nonzero(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < 110)
            if len(ys):
                k = rnd.randrange(len(ys)); centre = (int(ys[k]), int(xs[k]))
            else:
                centre = None
        if centre is not None:
            y0 = centre[0] - rnd.randint(CROP // 6, CROP * 5 // 6); x0 = centre[1] - rnd.randint(CROP // 6, CROP * 5 // 6)
        else:
            y0, x0 = rnd.randint(0, max(0, h - CROP)), rnd.randint(0, max(0, w - CROP))
        y0 = (min(max(0, y0), max(0, h - CROP)) // 8) * 8
        x0 = (min(max(0, x0), max(0, w - CROP)) // 8) * 8
        crop = np.full((CROP, CROP, 3), 255, np.uint8); mk = np.zeros((CROP, CROP), np.uint8)
        c = rgb[y0:y0 + CROP, x0:x0 + CROP]; crop[:c.shape[0], :c.shape[1]] = c
        m = mask[y0:y0 + CROP, x0:x0 + CROP]; mk[:m.shape[0], :m.shape[1]] = m
        if rnd.random() < P_RECOMPRESS:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, rnd.randint(70, 100)])
            crop, dct, qt = jpeg_parts(buf.tobytes())
        else:
            _, dct_full, qt = jpeg_parts(img_p)
            dct = np.zeros((CROP, CROP), np.int64)
            d = dct_full[y0:y0 + CROP, x0:x0 + CROP]; dct[:d.shape[0], :d.shape[1]] = d
        x = (crop.astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(dct)[None], torch.from_numpy(qt)[None],
                torch.from_numpy(mk.astype(np.int64))[None])


def load_state_dict_file(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


if __name__ == "__main__" or True:
    PHASE = os.environ.get("V2_PHASE", "")
    PART, NPARTS = int(os.environ.get("V2_PART", "0")), int(os.environ.get("V2_NPARTS", "1"))
    assert PHASE in ("mine", "ep0", "epoch", "final"), "run this through the one-cell runner"
    import gc
    try:
        import psutil
    except ImportError:
        psutil = None

    def mem_line():
        if psutil is None:
            return ""
        me = psutil.Process(); kids = 0
        for c in me.children(recursive=True):
            try:
                kids += c.memory_full_info().uss
            except Exception:
                pass
        vm = psutil.virtual_memory()
        return f"RAM main {me.memory_info().rss / 1e9:.1f} GB | workers {kids / 1e9:.1f} GB | free {vm.available / 1e9:.1f} of {vm.total / 1e9:.1f} GB"

    # ---------------- phase: find spots on clean training pages where the starting model wrongly fires
    if PHASE == "mine":
        model = load_state_dict_file(ffdn_mod.FFDN(), start_model_path()).to(dev).eval()
        clean = [m for m in split_meta("train") if m["mask"] is None]
        pages = random.Random(SEED).sample(clean, min(MINE_PAGES, len(clean)))[PART::NPARTS]
        hot, t0 = [], time.time()
        for i, m in enumerate(pages, 1):
            prob = predict(model, m["img"])
            n, _, stats, cent = cv2.connectedComponentsWithStats((prob >= 0.5).astype(np.uint8), connectivity=8)
            for j in range(1, n):
                if stats[j, cv2.CC_STAT_AREA] >= MINE_MIN_AREA:
                    hot.append({"image": str(m["img"]), "cy": int(cent[j][1]), "cx": int(cent[j][0]), "area": int(stats[j, cv2.CC_STAT_AREA])})
            if i % PROGRESS_EVERY == 0 or i == len(pages):
                el = time.time() - t0
                print(f"[mine {PART + 1}/{NPARTS}] {i}/{len(pages)} clean pages | {len(hot)} false spots | "
                      f"about {el / i * (len(pages) - i) / 60:.0f} min left", flush=True)
        (work / f"hotspots_part{PART}.json").write_text(json.dumps({"pages": len(pages), "hot": hot}))
        raise SystemExit(0)

    # ---------------- phase: score the starting model on the fixed validation pages (the bar v2 has to beat)
    if PHASE == "ep0":
        start = start_model_path()
        model = load_state_dict_file(ffdn_mod.FFDN(), start).to(dev)
        rows = score_pages(model, val_set(), "val (starting model)")
        write_rows(rows, work / "val_ep0.csv")
        s = summarise(rows)
        import shutil
        shutil.copy(start, work / "best.pth")
        (work / "history.json").write_text(json.dumps([{"epoch": 0, "train_loss": None, **s}], indent=1))
        print(f"[epoch 0 = starting model {start}] val caught {s['caught']:.1%} at blob >= {s['cutoff_px']} px "
              f"(clean flagged {s['clean_flagged']:.1%}) | insertions caught {s['insertion_caught']} | "
              f"pixel F1 {s['pixel_f1']} | AUC {s['auc_blob']}", flush=True)
        raise SystemExit(0)

    # ---------------- phase: final test, split across GPUs (the runner merges the parts)
    if PHASE == "final":
        model = load_state_dict_file(ffdn_mod.FFDN(), work / "best.pth").to(dev)
        metas = split_meta("test")
        if TEST_LIMIT:
            metas = random.Random(SEED).sample(metas, min(TEST_LIMIT, len(metas)))
        metas = metas[PART::NPARTS]
        rows, t0 = [], time.time()
        for i in range(0, len(metas), PROGRESS_EVERY):
            rows += score_pages(model, metas[i:i + PROGRESS_EVERY], f"test part {PART + 1}")
            el = time.time() - t0; done = min(i + PROGRESS_EVERY, len(metas))
            print(f"[test {PART + 1}/{NPARTS}] {done}/{len(metas)} pages | about {el / done * (len(metas) - done) / 60:.0f} min left", flush=True)
        write_rows(rows, work / f"test_part{PART}.csv")
        raise SystemExit(0)

    # ---------------- phase: one training epoch (a fresh process each time; resumes from last.pt)
    hist = json.loads((work / "history.json").read_text())
    best = max(h["select_score"] for h in hist)
    model = ffdn_mod.FFDN()
    state = None
    if (work / "last.pt").exists():
        state = torch.load(work / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        print(f"resumed from last.pt after epoch {state['epoch'] + 1}", flush=True)
    else:
        load_state_dict_file(model, start_model_path())
        print(f"epoch 1 starts from {start_model_path()}", flush=True)
    model = model.to(dev)
    net = Wrapped(model)
    if n_gpu > 1:
        net = nn.DataParallel(net)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * STEPS_PER_EPOCH, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    ep = 0
    if state:
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"]); scaler.load_state_dict(state["scaler"])
        ep = state["epoch"] + 1
        sched.total_steps = EPOCHS * STEPS_PER_EPOCH
        sched.last_epoch = min(ep * STEPS_PER_EPOCH, sched.total_steps - 1)
        sched._get_lr_called_within_step = True
        for g, lr in zip(opt.param_groups, sched.get_lr()):
            g["lr"] = lr
        sched._get_lr_called_within_step = False
        sched._last_lr = [g["lr"] for g in opt.param_groups]
    del state
    if ep >= EPOCHS:
        print(f"all {EPOCHS} epochs already done", flush=True); raise SystemExit(0)
    if n_gpu <= 1 and BATCH > 4:
        print(f"one GPU attached: batch {BATCH} -> 4 to fit a 16 GB T4", flush=True)
        BATCH = 4
    train = split_meta("train")
    items = [(m["img"], m["mask"]) for m in train]
    clean = [it for it in items if it[1] is None]
    hot = json.loads((work / "hotspots.json").read_text())["hot"] if (work / "hotspots.json").exists() else []
    print(f"train {len(items)} pages ({len(items) - len(clean)} tampered) | {len(hot)} mined false spots | "
          f"crops: {P_TAMPER_CROP:.0%} on edits, {P_HARDNEG:.0%} hard negatives, rest random", flush=True)
    print("start:", mem_line(), flush=True)

    model.train(); tot, n = 0.0, 0
    loader = torch.utils.data.DataLoader(CropSetV2(items, STEPS_PER_EPOCH * BATCH, hot, clean), batch_size=BATCH,
                                         num_workers=WORKERS, shuffle=False, drop_last=True,
                                         persistent_workers=False, pin_memory=False)
    bar = tqdm(loader, desc=f"epoch {ep + 1}/{EPOCHS}", unit="step")
    for im, dc, qt, mk in bar:
        im, dc, qt, mk = im.to(dev, non_blocking=True), dc.to(dev), qt.to(dev), mk.to(dev)
        try:
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                loss, _ = net(im, dc, qt, mk)
                loss = loss.mean()
        except torch.OutOfMemoryError:
            raise SystemExit(f"GPU out of memory at batch {BATCH}. Set BATCH = {max(2, BATCH // 2)} in the settings block and run again.")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item(); n += 1
        bar.set_postfix(loss=f"{tot / n:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
        if n % MEM_LOG_EVERY == 0:
            print(f"   step {n}/{STEPS_PER_EPOCH} | loss {tot / n:.4f} | {mem_line()}", flush=True)
        if psutil is not None and psutil.virtual_memory().available / 1e9 < MIN_FREE_RAM_GB:
            raise SystemExit(f"LOW RAM at epoch {ep + 1} step {n} ({mem_line()}). Stopped before running out of memory; "
                             "earlier epochs are saved. Run the cell again to redo this epoch.")
    del loader, bar
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    rows = score_pages(model, val_set(), f"val ep{ep + 1}")
    write_rows(rows, work / f"val_ep{ep + 1}.csv")
    s = summarise(rows)
    hist.append({"epoch": ep + 1, "train_loss": round(tot / max(n, 1), 4), **s})
    torch.save(model.state_dict(), work / f"ep{ep + 1}.pth")
    print(f"[epoch {ep + 1}] loss {tot / max(n, 1):.4f} | val caught {s['caught']:.1%} at blob >= {s['cutoff_px']} px "
          f"(clean flagged {s['clean_flagged']:.1%}) | insertions caught {s['insertion_caught']} | "
          f"pixel F1 {s['pixel_f1']} | AUC {s['auc_blob']}", flush=True)
    if s["select_score"] > best:
        torch.save(model.state_dict(), work / "best.pth")
        print(f"   new best (beats the starting model and earlier epochs) -> best.pth", flush=True)
    else:
        print(f"   not better than the best so far ({best:.4f}); best.pth unchanged", flush=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": ep}, work / "last.pt")
    (work / "history.json").write_text(json.dumps(hist, indent=1))   # written last: the runner counts epochs from it
    print("   after epoch:", mem_line(), flush=True)
    raise SystemExit(0)
'''

Path(SCRIPT).write_text("# settings written by the one-cell runner\n"
                        + "".join(f"{n} = {globals()[n]!r}\n" for n in SETTING_NAMES) + WORKER_BODY)
work = Path(WORK_DIR); work.mkdir(parents=True, exist_ok=True)
t0 = time.time()

if RESUME_DIR and not (work / "history.json").exists():
    prev = [p.parent for p in Path(RESUME_DIR).rglob("history.json") if p.parent.name == work.name]
    assert prev, f"no earlier {work.name}/history.json under RESUME_DIR = {RESUME_DIR}"
    prev = max(prev, key=lambda p: (p / "history.json").stat().st_mtime)
    for f in prev.iterdir():
        if f.is_file():
            shutil.copy(f, work / f.name)
    print(f"continuing the v2 run saved in {prev}", flush=True)

print("setting up (packages and FFDN code, once) ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jpegio", "efficientnet_pytorch", "gdown", "timm"], check=True)
if not (work / "ForensicHub").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/scu-zjz/ForensicHub.git", str(work / "ForensicHub")], check=True)
try:
    import torch
    n_gpu = torch.cuda.device_count()
except Exception:
    n_gpu = 0
NPARTS = max(1, min(2, n_gpu))
print(f"{n_gpu} GPU(s)", flush=True)


def pump(proc, keep_bars):
    for line in proc.stdout:
        if "Warning" in line or "warnings.warn" in line or "_VF.meshgrid" in line or (not keep_bars and "%|" in line):
            continue
        print(line, end="", flush=True)


def run(phase, label, parts=1):
    print(f"\n========== {label} ==========", flush=True)
    procs = []
    for part in range(parts):
        env = dict(os.environ, PYTHONUNBUFFERED="1", V2_PHASE=phase, V2_PART=str(part), V2_NPARTS=str(parts))
        if parts > 1:
            env["CUDA_VISIBLE_DEVICES"] = str(part)
        p = subprocess.Popen([sys.executable, "-u", SCRIPT], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        th = threading.Thread(target=pump, args=(p, parts == 1)); th.start()
        procs.append((p, th))
    codes = []
    for p, th in procs:
        th.join(); codes.append(p.wait())
    if any(codes):
        raise SystemExit(f"{label} failed (exit codes {codes}); see the lines above. "
                         "Finished steps are saved; run this cell again to continue.")


def history():
    h = work / "history.json"
    return json.loads(h.read_text()) if h.exists() else []


# 1. mine false spots on clean training pages (both GPUs)
if not (work / "hotspots.json").exists():
    run("mine", f"finding spots where the model wrongly fires ({MINE_PAGES} clean training pages)", NPARTS)
    pages, hot = 0, []
    for part in range(NPARTS):
        d = json.loads((work / f"hotspots_part{part}.json").read_text()); pages += d["pages"]; hot += d["hot"]
    (work / "hotspots.json").write_text(json.dumps({"pages": pages, "hot": hot}))
    print(f"{len(hot)} false spots on {len({h['image'] for h in hot})} of {pages} clean pages", flush=True)

# 2. score the starting model on the fixed validation pages
if not history():
    run("ep0", "scoring the starting model (epoch 0) on the fixed validation pages")

# 3. training, one fresh process per epoch
epoch_times = []
while len(history()) - 1 < EPOCHS:
    before, start = len(history()) - 1, time.time()
    est = max(epoch_times) if epoch_times else 0.75 * 3600
    if (time.time() - t0) + est + TEST_RESERVE_H * 3600 > BUDGET_H * 3600:
        print("time budget: no room for another epoch, moving to the final test", flush=True)
        break
    run("epoch", f"epoch {before + 1} of {EPOCHS}")
    epoch_times.append(time.time() - start)
    if len(history()) - 1 == before:
        raise SystemExit("the epoch did not finish; see the lines above. Run this cell again to continue.")
    print(f"epoch {before + 1} finished in {epoch_times[-1] / 60:.1f} min; total {(time.time() - t0) / 3600:.2f} h", flush=True)

# 4. final test with the chosen epoch (both GPUs), at the cut-off chosen on validation
run("final", "final test with the chosen epoch", NPARTS)
rows = []
for part in range(NPARTS):
    rows += list(csv.DictReader(open(work / f"test_part{part}.csv")))
with open(work / "test_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
hist = history()
chosen = max(hist, key=lambda h: h["select_score"])
cut = chosen["cutoff_px"]
tam = [r for r in rows if r["tampered"] == "1"]; cln = [r for r in rows if r["tampered"] == "0"]
ins = [r for r in tam if r["edit"] == "insertion"]
rate = lambda rs, k, t: sum(float(r[k]) >= t for r in rs) / max(1, len(rs))


def auc(pos, neg):
    allv = sorted((v, i < len(pos)) for i, v in enumerate(pos + neg))
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    rp = sum(ranks[k] for k, (v, is_pos) in enumerate(allv) if is_pos)
    return (rp - len(pos) * (len(pos) + 1) / 2) / max(1, len(pos) * len(neg))


res = {"chosen_epoch": chosen["epoch"], "cutoff_px": cut, "test_pages": len(rows),
       "caught": round(rate(tam, "blob3", cut), 4), "clean_flagged": round(rate(cln, "blob3", cut), 4),
       "insertion_caught": round(rate(ins, "blob3", cut), 4) if ins else None,
       "pixel_f1": round(sum(float(r["pixel_f1"]) for r in tam) / max(1, len(tam)), 4),
       "auc_blob": round(auc([float(r["blob3"]) for r in tam], [float(r["blob3"]) for r in cln]), 4),
       "old_rule_caught": round(rate(tam, "p999", 0.5), 4), "old_rule_clean_flagged": round(rate(cln, "p999", 0.5), 4)}
(work / "v2_summary.json").write_text(json.dumps({"test": res, "history": hist}, indent=1))
print("\nV2 DONE in %.2f h" % ((time.time() - t0) / 3600))
print("chosen epoch:", chosen["epoch"], "(0 = the starting model was not beaten)")
print(f"{'':34} | caught | clean flagged | insertions | pixel F1 | AUC")
print(f"{'v1 (first fine-tune, blob >= 246)':34} |  87.7% |      1.4%     |   ~10%     |  0.848   | 0.962")
print(f"{'v2 (blob >= ' + str(cut) + ' px)':34} | {res['caught']:6.1%} | {res['clean_flagged']:9.1%}     | "
      f"{(res['insertion_caught'] or 0):8.0%}   |  {res['pixel_f1']:.3f}   | {res['auc_blob']:.3f}")
print("download from Output -> finetune_v2: best.pth, v2_summary.json, history.json, test_v2.csv")
