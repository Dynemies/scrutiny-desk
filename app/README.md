# Scrutiny desk (SIH26261 prototype)

A decision-support desk for AICTE Extension of Approval applications. It checks every document before an officer opens the file and shows each finding with its evidence. Findings are advisory; the officer decides.

All applications, institutions, names and documents in `demo_data` are synthetic. Every page carries a "Synthetic sample, not a real document" footer, and no real logos or seals are used.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501 with 14 sample applications. Results are precomputed, so demo mode opens instantly. Re-run `python precompute.py` after changing the engine or the data.

## What it checks today

| Check | How |
| --- | --- |
| Missing documents | Against the Extension of Approval checklist |
| Name and AICTE ID | Each document against the AICTE record, with the differing characters marked |
| Expired certificates | Validity date against the filing date |
| Lease rule | Lease left against the longest course (Approval Process Handbook cl. 2.1(e)) |
| Intake | Requested seats against approved intake |
| Digital signature | Verified with pyHanko against the trust root |
| Changes after signing | Signed revision rendered next to the uploaded file; the changed area is boxed |
| Re-used pages | Perceptual hash, confirmed pixel by pixel, against earlier applications |

Score on 150 synthetic applications (`python score_engine.py` on a larger generated set): every planted problem of the types above was caught, with no false alarms on 748 clean documents. These numbers use a stand-in reader that knows the printed fields exactly; they will be re-measured once the tagging model reads the pages.

## Plug-in slots

- `engine.READER`: the tagging model goes here (same output shape as `stand_in_reader`).
- Tamper heatmaps: `demo_data/tamper_maps/<APP>/<doc>.png` (probability × 255) and a `.json` with `{"score": <largest marked area, px>, "threshold": 143, "unit": "px", "model": "..."}`. Made by `training/kaggle/demo_heatmaps_one_cell.py`.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `engine.py` | All checks |
| `precompute.py` | Caches results for demo mode |
| `score_engine.py` | Scores the engine against generator answer keys |
| `static/` | Martel and Mukta fonts (SIL Open Font License), served locally so the app works offline |
| `demo_data/` | 14 synthetic applications, AICTE mock records and the test trust root (no private keys) |
