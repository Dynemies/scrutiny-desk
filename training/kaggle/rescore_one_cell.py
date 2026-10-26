# ======================= PATHS AND SETTINGS (edit if needed) =======================
DATA_ROOT = "/kaggle/input/datasets/vamsikrishnasai05/data-genx"   # generator output (tamper/validation, tamper/test)
MODEL_SEARCH = "/kaggle/input"   # searched for best.pth: add the fine-tuning notebook's output as an input
MODEL_NAME = "best.pth"
WORK_DIR = "/kaggle/working/rescore"   # results: rescore_validation.csv, rescore_test.csv, rescore_examples.zip
SPLITS = ["validation", "test"]  # run side by side, one per GPU
LIMIT = 0                  # 0 = every page; e.g. 40 for a quick trial
SAVE_EXAMPLES = 40         # side-by-side crops saved for missed edits and for clean pages that light up (test split)
PROGRESS_EVERY = 50        # print a progress line every N pages
EVAL_BATCH = 8
WINDOW, STRIDE, THRESHOLD = 512, 384, 0.5
SEED = 0
START_CHECKPOINT, DOC_BENCH_FOLDER = "", ""   # unused here (kept so the shared helpers load)
SCRIPT = "/kaggle/working/rescore_ffdn.py"   # this cell writes the scoring code here
# ====================================================================================
"""
ONE CELL: re-scores the fine-tuned FFDN (best.pth) on the validation and test pages with several
page-level rules, because the old rule (99.9th-percentile pixel) cannot flag edits smaller than 0.1% of a page.
No training. Validation runs on GPU 0 and test on GPU 1 at the same time (about 1 hour on 2x T4).
"""
import os, shutil, subprocess, sys, time, threading
from pathlib import Path

SETTING_NAMES = ['DATA_ROOT', 'MODEL_SEARCH', 'MODEL_NAME', 'WORK_DIR', 'LIMIT', 'SAVE_EXAMPLES', 'PROGRESS_EVERY', 'EVAL_BATCH', 'WINDOW', 'STRIDE', 'THRESHOLD', 'SEED', 'START_CHECKPOINT', 'DOC_BENCH_FOLDER']
WORKER_BODY = r'''"""
(Shared helpers copied from finetune_ffdn.py.) Fine-tunes FFDN (ForensicHub document checkpoint, DocTamper-trained) on the synthetic
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



# ------------------------------------------------------------------ 5. re-score one split with several page rules
def page_scores(prob):
    """Candidate page scores. p999 is the old rule; the others do not depend on how big the edit is."""
    out = {"p999": float(np.percentile(prob, 99.9)),
           "peak3": float(cv2.blur(prob, (3, 3)).max()),
           "peak9": float(cv2.blur(prob, (9, 9)).max())}
    for t in (0.5, 0.3):
        n, _, stats, _ = cv2.connectedComponentsWithStats((prob >= t).astype(np.uint8), connectivity=8)
        out[f"blob{int(t * 10)}"] = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    return out


def save_pair(img_p, prob, gt, centre, path, half=160):
    rgb = cv2.imread(str(img_p))
    h, w = prob.shape
    cy, cx = centre
    y0, y1 = max(0, cy - half), min(h, cy + half); x0, x1 = max(0, cx - 2 * half), min(w, cx + 2 * half)
    page = rgb[y0:y1, x0:x1].copy()
    if gt is not None and gt.any():
        cs, _ = cv2.findContours(gt[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(page, cs, -1, (0, 200, 0), 1)
    heat = cv2.applyColorMap((prob[y0:y1, x0:x1] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(path), np.hstack([page, np.full((page.shape[0], 6, 3), 255, np.uint8), heat]), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__" or True:
    SPLIT = os.environ["RESCORE_SPLIT"]
    found = sorted(Path(MODEL_SEARCH).rglob(MODEL_NAME))
    assert found, f"no {MODEL_NAME} anywhere under {MODEL_SEARCH}: add the fine-tuning notebook's output as an input"
    best_p = max(found, key=lambda f: f.stat().st_mtime)
    model = ffdn_mod.FFDN()
    model.load_state_dict(torch.load(best_p, map_location="cpu"))
    model = model.to(dev).eval()
    root = next(p.parent for p in Path(DATA_ROOT).rglob("index.csv") if p.parent.name == SPLIT and p.parent.parent.name == "tamper")
    meta = list(csv.DictReader(open(root / "index.csv")))
    if LIMIT:
        meta = random.Random(SEED).sample(meta, min(LIMIT, len(meta)))
    print(f"[{SPLIT}] model {best_p} | {len(meta)} pages", flush=True)
    ex_dir = work / "examples"; ex_dir.mkdir(exist_ok=True)
    cols = ["image", "tampered", "edit", "doc_type", "field", "gt_area", "pixel_f1", "p999", "peak3", "peak9", "blob5", "blob3"]
    rows, t0, n_miss, n_fa = [], time.time(), 0, 0
    for i, r in enumerate(meta, 1):
        img_p = root / r["image"]
        prob = predict(model, img_p)
        gt = read_mask(root / r["mask"]) if r["tampered"] == "1" else None
        s = page_scores(prob)
        rows.append([r["image"], int(gt is not None), r["edit"], r["doc_type"], r["field"], int(gt.sum()) if gt is not None else 0,
                     round(f1(prob, gt), 4) if gt is not None else "", *[round(s[k], 4) if isinstance(s[k], float) else s[k] for k in cols[7:]]])
        if SAVE_EXAMPLES and SPLIT == "test":
            stem = Path(r["image"]).stem
            if gt is not None and s["p999"] < THRESHOLD and n_miss < SAVE_EXAMPLES:   # missed by the old rule
                ys, xs = np.nonzero(gt); n_miss += 1
                save_pair(img_p, prob, gt, (int(ys.mean()), int(xs.mean())), ex_dir / f"missed_{r['edit']}_{stem}.jpg")
            if gt is None and s["peak9"] >= THRESHOLD and n_fa < SAVE_EXAMPLES:      # clean page that lights up
                n_fa += 1
                save_pair(img_p, prob, None, np.unravel_index(cv2.blur(prob, (9, 9)).argmax(), prob.shape), ex_dir / f"clean_{r['doc_type']}_{stem}.jpg")
        if i % PROGRESS_EVERY == 0 or i == len(meta):
            el = time.time() - t0
            print(f"[{SPLIT}] {i}/{len(meta)} pages | {el / 60:.1f} min | about {el / i * (len(meta) - i) / 60:.0f} min left", flush=True)
    with open(work / f"rescore_{SPLIT}.csv", "w", newline="") as f:
        csv.writer(f).writerows([cols] + rows)
    print(f"[{SPLIT}] wrote {work / f'rescore_{SPLIT}.csv'}", flush=True)
'''

Path(SCRIPT).write_text("# settings written by the one-cell runner\n"
                        + "".join(f"{n} = {globals()[n]!r}\n" for n in SETTING_NAMES) + WORKER_BODY)
work = Path(WORK_DIR); work.mkdir(parents=True, exist_ok=True)
t0 = time.time()
print("setting up (packages and FFDN code, once) ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jpegio", "efficientnet_pytorch", "gdown", "timm"], check=True)
if not (work / "ForensicHub").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/scu-zjz/ForensicHub.git", str(work / "ForensicHub")], check=True)

try:
    import torch
    n_gpu = torch.cuda.device_count()
except Exception:
    n_gpu = 0
print(f"{n_gpu} GPU(s); running {SPLITS}", flush=True)


def pump(proc):
    for line in proc.stdout:
        if "Warning" in line or "warnings.warn" in line or "_VF.meshgrid" in line or "%|" in line:
            continue
        print(line, end="", flush=True)


procs = []
for i, split in enumerate(SPLITS):
    env = dict(os.environ, PYTHONUNBUFFERED="1", RESCORE_SPLIT=split)
    if n_gpu > 1:
        env["CUDA_VISIBLE_DEVICES"] = str(i % n_gpu)
    p = subprocess.Popen([sys.executable, "-u", SCRIPT], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    th = threading.Thread(target=pump, args=(p,)); th.start()
    procs.append((split, p, th))
    if n_gpu <= 1:          # one GPU (or CPU): run the splits one after another
        th.join(); p.wait()
codes = []
for split, p, th in procs:
    th.join(); codes.append((split, p.wait()))
bad = [s for s, c in codes if c != 0]
if bad:
    raise SystemExit(f"split(s) {bad} failed; see the lines above")
shutil.make_archive(str(work / "rescore_examples"), "zip", work, "examples")
print(f"\nRESCORE DONE in {(time.time() - t0) / 3600:.2f} h")
print("download from Output -> rescore: rescore_validation.csv, rescore_test.csv, rescore_examples.zip")
