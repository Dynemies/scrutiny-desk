# Training and scoring

The tamper model is [FFDN](https://github.com/scu-zjz/ForensicHub) (ECCV 2024), which reads both the page image and its JPEG
compression coefficients, fine-tuned on our synthetic AICTE pages. Start from ForensicHub's document checkpoint (trained on DocTamper).

| File | What it does |
| --- | --- |
| `baseline_ffdn.py` | Scores FFDN as published on our test pages (no training) |
| `finetune_ffdn.py` | v1 fine-tune: 512-px crops aligned to the 8×8 JPEG grid, half centred on an edit, 70% re-saved at a random JPEG quality; Dice + cross-entropy; best epoch kept |
| `kaggle/finetune_one_cell.py` | v1 as a single Kaggle cell; each epoch runs in a fresh process (no memory build-up), resumes automatically |
| `kaggle/finetune_v2_one_cell.py` | v2: mines spots where v1 wrongly fires on genuine pages, trains with 25% hard-negative crops, picks the epoch with the app's page rule, saves every epoch |
| `kaggle/rescore_one_cell.py`, `kaggle/rescore_v2_one_cell.py` | Re-score models on validation and test with five page rules (no training) |
| `kaggle/miss_analysis_cell.py` | Which edits are missed, by edit type, document and field |
| `kaggle/demo_heatmaps_one_cell.py` | Heatmaps and verdicts for the app's demo pages |
| `analysis/analyze_rescore.py`, `analysis/analyze_v2.py` | Choose the rule, cut-off and epoch on validation; report test once |

Every script has its paths and settings in a block at the top and prints progress while it runs.
Model weights are not in the repository (about 357 MB); see the main README.
