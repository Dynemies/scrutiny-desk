# ======================= PATHS AND SETTINGS (edit if needed) =======================
DATA_ROOT = "/kaggle/input/datasets/vamsikrishnasai05/data-genx"     # generator output (tamper/train, validation, test)
WORK_DIR = "/kaggle/working/finetune"                                  # code, checkpoints, results
DOC_BENCH_FOLDER = "https://drive.google.com/drive/folders/1RZZxwYIX5e-lHKDw1CD45FwFC0QqJ7im"  # ForensicHub document checkpoints
START_CHECKPOINT = ""       # local FFDN .pth to start from; empty = download ffdn_ckpt-8.pth from the folder above
RESUME_DIR = "/kaggle/input"   # searched for the newest last.pt from an earlier run you added as input; "" = start fresh
EPOCHS = 8                  # one epoch = STEPS_PER_EPOCH batches of crops
STEPS_PER_EPOCH = 600
BATCH = 8                   # crops per step: 4 per T4 (ForensicHub trained FFDN at 6 per GPU); capped at 4 on one GPU
CROP = 512                  # training crop; multiple of 8 so JPEG blocks stay aligned
LR = 1e-4
P_TAMPER_CROP = 0.5         # share of crops centred on an edited region (the rest are random, mostly clean)
P_RECOMPRESS = 0.7          # share of crops re-saved at a random JPEG quality (DocQT-style robustness)
VAL_IMAGES = 200            # validation pages scored after every epoch (best epoch is kept)
TEST_LIMIT = 0              # 0 = score all test pages at the end; e.g. 50 for a quick trial run
TIME_BUDGET_H = 11.0        # stop training cleanly before Kaggle's 12-hour limit; the final test still runs
MIN_FREE_RAM_GB = 3.0       # if free system RAM falls below this, save and stop cleanly instead of being killed
MEM_LOG_EVERY = 100         # print a memory line every N training steps
EVAL_BATCH = 8              # windows scored at once when predicting full pages
WORKERS = 4
DEMO_PAGES_DIR = ""         # optional: the app's demo_data folder; heatmaps go to WORK_DIR/tamper_maps
WINDOW, STRIDE, THRESHOLD = 512, 384, 0.5
SEED = 0
# ====================================================================================
"""
Fine-tunes FFDN (ForensicHub document checkpoint, DocTamper-trained) on the synthetic
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


# ------------------------------------------------------------------ 5. training
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


if __name__ == "__main__" or True:
    model = ffdn_mod.FFDN()
    ONE_EPOCH = os.environ.get("FFDN_ONE_EPOCH") == "1"      # set by the runner: train one epoch, save, exit
    FINAL_ONLY = os.environ.get("FFDN_FINAL_ONLY") == "1"    # set by the runner: skip training, run the final test
    if (work / "last.pt").exists():                          # this run's own progress comes first
        last = work / "last.pt"
    elif RESUME_DIR:   # Kaggle nests a notebook's output inside the input folder, so search for it
        found = sorted(Path(RESUME_DIR).rglob("last.pt"))
        assert found, f"no last.pt anywhere under RESUME_DIR = {RESUME_DIR}"
        last = max(found, key=lambda f: f.stat().st_mtime)   # newest if several runs are attached
    else:
        last = work / "last.pt"
    state = None
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        print(f"resumed from {last} at epoch {state['epoch']}", flush=True)
    else:
        load_weights(model, fetch_checkpoint())
    model = model.to(dev)
    net = Wrapped(model)
    if n_gpu > 1:
        net = nn.DataParallel(net)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * STEPS_PER_EPOCH, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    start_ep, best, history = 0, -1.0, []
    if state:
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"]); scaler.load_state_dict(state["scaler"])
        start_ep, best, history = state["epoch"] + 1, state["best"], state["history"]
        # line the LR schedule up with the epochs actually done (safe even if EPOCHS / STEPS_PER_EPOCH changed)
        sched.total_steps = EPOCHS * STEPS_PER_EPOCH
        sched.last_epoch = min(start_ep * STEPS_PER_EPOCH, sched.total_steps - 1)
        sched._get_lr_called_within_step = True
        for g, lr in zip(opt.param_groups, sched.get_lr()):
            g["lr"] = lr
        sched._get_lr_called_within_step = False
        sched._last_lr = [g["lr"] for g in opt.param_groups]
        prev_best = last.parent / "best.pth"
        if prev_best.exists() and prev_best.resolve() != (work / "best.pth").resolve():
            import shutil
            shutil.copy(prev_best, work / "best.pth")
            print(f"copied best.pth from {last.parent} (select score {best:.4f})", flush=True)
        for h in history:
            print(f"   earlier epoch {h['epoch']}: val pixel F1 {h['pixel_f1_mean']} | val AUC {h['image_auc']}", flush=True)
        (work / "history.json").write_text(json.dumps(history, indent=1))   # the runner counts finished epochs from this file
    del state

    import gc
    try:
        import psutil
    except ImportError:
        psutil = None

    def mem_line():
        if psutil is None:
            return ""
        me = psutil.Process()
        kids = 0
        for c in me.children(recursive=True):   # unique memory only; plain RSS double-counts pages shared with the main process
            try:
                kids += c.memory_full_info().uss
            except Exception:
                pass
        vm = psutil.virtual_memory()
        return (f"RAM main {me.memory_info().rss / 1e9:.1f} GB | workers {kids / 1e9:.1f} GB | "
                f"free {vm.available / 1e9:.1f} of {vm.total / 1e9:.1f} GB")

    def low_ram():
        return psutil is not None and psutil.virtual_memory().available / 1e9 < MIN_FREE_RAM_GB

    if n_gpu <= 1 and BATCH > 4:
        print(f"one GPU attached: batch {BATCH} -> 4 to fit a 16 GB T4", flush=True)
        BATCH = 4
    train_items, val_items, test_items = split_items("train"), split_items("validation"), split_items("test")
    rng = random.Random(SEED)
    if TEST_LIMIT:
        test_items = rng.sample(test_items, min(TEST_LIMIT, len(test_items)))
    val_sub = rng.sample(val_items, min(VAL_IMAGES, len(val_items)))
    print(f"train {len(train_items)} pages ({sum(1 for i in train_items if i[1])} tampered), "
          f"validation {len(val_sub)} of {len(val_items)}, test {len(test_items)}", flush=True)
    print("start:", mem_line(), flush=True)

    for ep in range(start_ep, 0 if FINAL_ONLY else EPOCHS):
        if (time.time() - T0) / 3600 > TIME_BUDGET_H:
            print("time budget reached; stopping training", flush=True); break
        model.train(); tot, n = 0.0, 0
        # fresh workers every epoch and no pinned host memory: the first run's system RAM grew until the kernel died
        loader = torch.utils.data.DataLoader(CropSet(train_items, STEPS_PER_EPOCH * BATCH), batch_size=BATCH,
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
                raise SystemExit(f"GPU out of memory at batch {BATCH}. Set BATCH = {max(2, BATCH // 2)} in the settings block "
                                 "and run again; training resumes from last.pt if an epoch already finished.")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
            bar.set_postfix(loss=f"{tot / n:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
            if n % MEM_LOG_EVERY == 0:
                print(f"   step {n}/{STEPS_PER_EPOCH} | loss {tot / n:.4f} | {mem_line()}", flush=True)
            if low_ram():
                print(f"LOW RAM at epoch {ep + 1} step {n}: {mem_line()}", flush=True)
                done = f"Epochs 1-{ep} are saved in last.pt and best.pth. " if ep else "No epoch had finished yet. "
                raise SystemExit("Stopped before running out of memory. " + done +
                                 "Add this notebook's output as an input, set RESUME_DIR to it and run again; "
                                 f"epoch {ep + 1} restarts from its beginning.")
        del loader, bar
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
        v = evaluate(model, val_sub, f"val_ep{ep + 1}", write_csv=False)
        score = (v["pixel_f1_mean"] or 0) + (v["image_auc"] if v["image_auc"] == v["image_auc"] else 0)
        history.append({"epoch": ep + 1, "train_loss": round(tot / max(n, 1), 4), **v, "select_score": round(score, 4)})
        print(f"[epoch {ep + 1}] loss {tot / max(n, 1):.4f} | val pixel F1 {v['pixel_f1_mean']} | val AUC {v['image_auc']} | "
              f"tampered flagged {v['tampered_flagged']} | clean flagged {v['authentic_flagged']} | "
              f"elapsed {(time.time() - T0) / 3600:.2f} h", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), work / "best.pth")
            print(f"   new best (select score {score:.4f}) saved to best.pth", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "history": history}, work / "last.pt")
        (work / "history.json").write_text(json.dumps(history, indent=1))
        gc.collect()
        print("   after epoch:", mem_line(), flush=True)
        if ONE_EPOCH:
            raise SystemExit(0)      # the runner starts a fresh process for the next epoch, which releases all memory
    if ONE_EPOCH:
        print(f"all {EPOCHS} epochs already done", flush=True)
        raise SystemExit(0)

    # ------------------------------------------------------------------ 6. final test with the best epoch
    model.load_state_dict(torch.load(work / "best.pth", map_location=dev))
    final = evaluate(model, test_items, "ours_test_finetuned")
    (work / "finetune_summary.json").write_text(json.dumps({"test": final, "history": history}, indent=1))
    if DEMO_PAGES_DIR:
        pages = sorted(Path(DEMO_PAGES_DIR).glob("applications/*/*/pages/*.jpg"))
        for p in tqdm(pages, desc="demo heatmaps", unit="page"):
            prob = predict(model, p)
            out = work / "tamper_maps" / p.parent.parent.name / p.stem
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out.with_suffix(".png")), (prob * 255).astype(np.uint8))
            out.with_suffix(".json").write_text(json.dumps({"score": float(np.percentile(prob, 99.9)), "threshold": THRESHOLD,
                                                            "model": "FFDN fine-tuned on synthetic AICTE documents"}))
        import shutil
        shutil.make_archive(str(work / "tamper_maps"), "zip", work, "tamper_maps")
        print("heatmaps for the app: download tamper_maps.zip and unzip it inside the app's demo_data folder")
    print("\nFINE-TUNING DONE")
    print("baseline   (FFDN as published): pixel F1 0.1059 | image AUC 0.5403 | tampered flagged 0.8998 | clean flagged 0.8427")
    print(f"fine-tuned (best epoch)       : pixel F1 {final['pixel_f1_mean']} | image AUC {final['image_auc']} | "
          f"tampered flagged {final['tampered_flagged']} | clean flagged {final['authentic_flagged']}")
