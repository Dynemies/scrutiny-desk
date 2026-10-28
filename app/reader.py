"""Document reader: sorts a page into its document type and pulls out the printed fields from the OCR lines.

Works on the three page layouts the desk sees (a "Label: value" list, and two table styles). Field labels come from
rules/aicte_eoa.json, so a new document type or label is added there, not in code."""
import json
import re
from pathlib import Path

RULES_FILE = Path(__file__).resolve().parent / "rules" / "aicte_eoa.json"


def squash(s):
    """Lower-case letters and digits only: OCR often drops spaces, so comparisons ignore them."""
    return re.sub(r"[^0-9a-z]", "", str(s or "").lower())


def edits(a, b):
    """Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fold(s):
    """squash() plus the letter/digit look-alikes OCR confuses (O/0, I/l/1): for comparing codes and IDs."""
    return squash(s).translate(str.maketrans({"o": "0", "i": "1", "l": "1"}))


MULTILINE = {"institution_name", "address", "programmes", "lessor_name", "lessee_name", "authority"}
STOP = ("conditions", "programme", "approvedintake", "signature")


def load_rules(path=RULES_FILE):
    return json.loads(Path(path).read_text())


def classify(lines, rules):
    """Document type from the title line, e.g. 'CERTIFICATE OF ACCREDITATION' -> nba_certificate."""
    best = None
    for ln in lines:
        q = squash(ln["text"])
        for d, spec in rules["documents"].items():
            for t in spec["titles"]:
                if q.startswith(squash(t)) and (best is None or ln["box"][1] < best[1]):
                    best = (d, ln["box"][1])
    if best:
        return best[0]
    # no readable title: pick the document type whose field labels appear most often
    sq = [squash(ln["text"]) for ln in lines]
    score = {d: sum(any(q.startswith(squash(lab)) for q in sq) for lab in spec["labels"].values())
             for d, spec in rules["documents"].items()}
    d, n = max(score.items(), key=lambda kv: kv[1])
    return d if n >= 3 else None


def _starts(q, label):
    """Line begins with the label, allowing one misread letter in labels of 8+ characters (OCR: 'ADDRESSOE' for 'ADDRESSOF')."""
    if q.startswith(label):
        return True
    if len(label) < 5 or len(q) < len(label):
        return False
    head = q[:len(label)]
    return sum(x != y for x, y in zip(head, label)) <= 1


def _row(a, b):
    """True when two lines sit on the same text row."""
    ca, cb = (a["box"][1] + a["box"][3]) / 2, (b["box"][1] + b["box"][3]) / 2
    return abs(ca - cb) < 0.6 * max(a["box"][3] - a["box"][1], 8)


def _eff(ln):
    """Confidence of a line; a low-confidence line whose second read agreed counts as sure."""
    return 1.0 if ln.get("checked") else ln.get("conf", 1.0)


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _quad(b):
    x0, y0, x1, y1 = b
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def read_fields(lines, rules, doc_type=None):
    """Returns {doc_type, fields, boxes}: field values as printed (OCR text) and their positions on the page."""
    doc_type = doc_type or classify(lines, rules)
    if doc_type is None:
        return {"doc_type": None, "fields": {}, "boxes": {}}
    labels = rules["documents"][doc_type]["labels"]                      # field -> label text(s)
    lab_sq = {k: [squash(x) for x in (v if isinstance(v, list) else [v])] for k, v in labels.items()}
    all_labels = {s for v in lab_sq.values() for s in v}
    title_sq = [squash(t) for t in rules["documents"][doc_type]["titles"]]

    def label_of(ln):
        q = squash(ln["text"])
        if any(q.startswith(t) for t in title_sq):          # the document's title is never a field label
            return None, ""
        hits = [(k, s) for k, ss in lab_sq.items() for s in ss if q.startswith(s)] \
            or [(k, s) for k, ss in lab_sq.items() for s in ss if _starts(q, s)]
        if not hits:
            return None, ""
        k, s = max(hits, key=lambda h: len(h[1]))                         # longest label wins ("Certificate No." vs "Certificate Issued Date")
        # the rest of the line after the label, from the original text
        txt = ln["text"]
        rest = txt.split(":", 1)[1] if ":" in txt else ""
        if not rest:                                                       # no colon: cut the label's characters off the front
            n, i = 0, 0
            while i < len(txt) and n < len(s):
                if re.match(r"[0-9a-zA-Z]", txt[i]):
                    n += 1
                i += 1
            rest = txt[i:]
        return k, rest.strip(" :.-)(")

    fields, boxes, confs, alts, used = {}, {}, {}, {}, set()
    for i, ln in enumerate(lines):
        k, rest = label_of(ln)
        if k is None or k in fields:
            continue
        parts, pbox, pconf = [], [], []
        if rest:
            parts.append(rest); pbox.append(ln["box"]); pconf.append(_eff(ln))
        # value on the same row, to the right of the label (table layouts, or OCR split "Label:" from the value)
        right = [m for j, m in enumerate(lines) if j != i and j not in used and _row(ln, m) and m["box"][0] > ln["box"][2] - 4]
        right.sort(key=lambda m: m["box"][0])
        for m in right:
            if label_of(m)[0] is None:
                parts.append(m["text"].strip(" :")); pbox.append(m["box"]); pconf.append(_eff(m)); used.add(lines.index(m))
        # continuation lines just below the value, in the same column, that are not labels themselves
        if pbox and k in MULTILINE:
            x0, last = pbox[-1][0], pbox[-1]
            for m in lines[i + 1:]:
                if m["box"][1] <= last[1] + 2:
                    continue
                if m["box"][1] - last[3] > 0.9 * (last[3] - last[1]) or abs(m["box"][0] - x0) > 25:
                    if m["box"][1] - last[3] > 0.9 * (last[3] - last[1]):
                        break
                    continue
                if label_of(m)[0] is not None or squash(m["text"]) in all_labels or squash(m["text"]).startswith(STOP):
                    break
                if any(_row(m, n) and n["box"][0] < x0 - 40 for n in lines):     # a label sits on that row: a new field
                    break
                parts.append(m["text"].strip()); pbox.append(m["box"]); pconf.append(_eff(m)); used.add(lines.index(m)); last = m["box"]
        if parts:
            fields[k] = " ".join(parts).strip()
            boxes[k] = _quad(_union(pbox)); confs[k] = round(min(pconf), 3)
            src = [m for m in lines if m["box"] in pbox and m.get("second_read") and not m.get("checked")]
            if src:
                alts[k] = " ".join(m["second_read"] for m in src)
    # fields printed without a label, e.g. the issuing authority line above a fire NOC's title
    for k, how in rules["documents"][doc_type].get("unlabelled", {}).items():
        if how == "line_above_title":
            titles = [squash(t) for t in rules["documents"][doc_type]["titles"]]
            for i, ln in enumerate(lines):
                if any(squash(ln["text"]).startswith(t) for t in titles) and i > 0:
                    prev = [m for m in lines[:i] if m["box"][3] <= ln["box"][1] + 2]
                    if prev:
                        fields[k] = prev[-1]["text"].strip(); boxes[k] = _quad(prev[-1]["box"]); confs[k] = _eff(prev[-1])
                    break
    return {"doc_type": doc_type, "fields": fields, "boxes": boxes, "conf": confs, "second_read": alts}
