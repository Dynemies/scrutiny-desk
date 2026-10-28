"""OCR: reads the text lines on a page (RapidOCR, PP-OCRv3 models bundled in the package, runs offline on CPU).

Lines the model is unsure about are read a second time from a padded crop. When both reads agree the line is
marked "checked"; when they differ, both readings are kept and the field goes to a person."""
import re
from functools import lru_cache

import cv2
import numpy as np

RECHECK_BELOW = 0.75      # lines under this confidence get a second read


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def _sq(s):
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def recheck(lines, img):
    """Second read of low-confidence lines from a padded crop of the page (in place)."""
    h, w = img.shape[:2]
    for ln in lines:
        if ln["conf"] >= RECHECK_BELOW or "second_read" in ln:
            continue
        x0, y0, x1, y1 = ln["box"]
        crop = img[max(0, y0 - 6):min(h, y1 + 6), max(0, x0 - 16):min(w, x1 + 16)]
        res, _ = _engine()(crop, use_det=False, use_cls=False, use_rec=True)
        txt = res[0][1] if res else ""
        ln["second_read"] = txt
        ln["checked"] = bool(txt) and _sq(txt) == _sq(ln["text"])
    return lines


def ocr_lines(image_bytes):
    """List of {text, conf, box:[x0,y0,x1,y1]} for every text line on the page, top to bottom."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    res, _ = _engine()(img)
    lines = []
    for quad, txt, conf in res or []:
        xs = [float(q[0]) for q in quad]; ys = [float(q[1]) for q in quad]
        lines.append({"text": txt, "conf": round(float(conf), 3),
                      "box": [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]})
    return recheck(sorted(lines, key=lambda l: (l["box"][1], l["box"][0])), img)
