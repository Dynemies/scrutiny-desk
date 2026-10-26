# SIH26261 synthetic AICTE application generator

Makes fake but realistic Extension-of-Approval application files where every field, every planted problem and every edited pixel is known. Both models (tagging and tamper) train on its output, and its answer keys score the whole pipeline.

## Run it

```
pip install pymupdf pyhanko faker tqdm opencv-python-headless pillow numpy cryptography
python generate_aicte_synth.py
```

Edit the settings block at the top of the script first: `OUT_DIR`, `FONT_DIR` (the `fonts` folder in this zip), `N_APPLICATIONS` and `START_INDEX`. About 2.2 s and 5.2 MB per application; 2,000 applications take about 75 minutes and 10.5 GB. A progress bar and a plain `[progress]` line every 20 applications show while it runs, and a self-check runs at the end. For more data, run another batch in a new notebook with `START_INDEX = 2000` (then 4000, ...): numbering continues and the seed changes, so batches never repeat.

On Kaggle: upload this folder as a dataset, copy it to `/kaggle/working`, set `FONT_DIR = "/kaggle/working/aicte_gen/fonts"` and `OUT_DIR = "/kaggle/working/aicte_synth_out"`.

## What comes out

| Path | What it holds |
| --- | --- |
| `records/institutions.json` | Mock "AICTE records": ID, name, address, approved intake, previous approval |
| `applications/<split>/<APP>/application.json` | The submitted form: requested intake, documents uploaded |
| `applications/<split>/<APP>/docs/*.pdf` | Uploaded PDFs, DSC-signed with the test certificate |
| `applications/<split>/<APP>/pages/*.jpg` | The scanned page inside each PDF (byte-identical to the embedded JPEG) |
| `applications/<split>/<APP>/answer_key.json` | Every planted problem, the expected status per document and per application |
| `donut/<split>/metadata.jsonl` | Clean pages with their document type and printed fields, in Donut's format |
| `tamper/<split>/index.csv` | Authentic pages and tampered twins, with pixel masks |
| `pki/test_ca.pem` | Trust root for validating the test signatures |

Six document types: affidavit on e-stamp paper, lease deed extract, fire NOC, NBA accreditation certificate, previous approval letter, fee (TER) receipt. Three layouts; layout 2 appears only in the test split, and test institutions never appear in training.

Planted problems: missing document, expired document, lease shorter than the longest course, institution name mismatch, AICTE ID mismatch, intake above approval, unsigned document, edit after signing, duplicate upload, and pixel tampering (replacement, insertion, removal, copy-move, splicing).

## Known limits (v1)

- Every document is a scanned page inside a PDF; no born-digital PDFs yet.
- Copy-move is the rarest edit, because it needs two similar values on one page.
- Donut labels hold the text as printed on the page, not the true record; rules compare the two.
- Test signatures come from a self-made CA. They prove the checking logic, not real DSC trust chains.

Fonts are Noto (SIL Open Font License).
