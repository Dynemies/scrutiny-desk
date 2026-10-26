# Results

Per-page scores behind every number in the main README. All pages are synthetic (see `generator/`).
Validation: 1,555 pages (571 edited, 984 genuine). Test: 1,607 pages (609 edited, 998 genuine); test institutions never appear in training, and one of the three page layouts appears only in test.

| File | Model |
| --- | --- |
| `v1_validation_scores.csv`, `v1_test_scores.csv` | Fine-tune v1 (8 epochs, best epoch 3) |
| `v2_ep3_validation_scores.csv`, `v2_ep3_test_scores.csv` | Fine-tune v2, epoch 3: **the model used in the app** |
| `v2_ep4_validation_scores.csv`, `v2_ep4_test_scores.csv` | Fine-tune v2, epoch 4 (compared, not chosen) |
| `history_v1.json` | v1 per-epoch training loss and validation scores (200 validation pages per epoch) |

Columns: `tampered` (1 = edited page), `edit` (replacement, insertion, removal, copy_move, splicing), `doc_type`, `field`,
`gt_area` (edited pixels), `pixel_f1` (overlap of the marked and the truly edited pixels, at probability 0.5),
and five page scores: `p999` (99.9th-percentile pixel), `peak3` / `peak9` (brightest spot after 3×3 / 9×9 smoothing),
`blob5` / `blob3` (largest connected area marked at probability ≥ 0.5 / ≥ 0.3, in pixels).

## How the page rule and the model were chosen

1. The cut-off for each candidate is chosen on **validation only**: the most edited pages caught while flagging at most 1% of genuine validation pages.
2. Among v2 epochs, the one catching the most validation edits wins (ties: fewer genuine pages flagged).
3. Test is scored once with that choice.

| Model, rule | Cut-off | Test: edited caught | Test: genuine flagged | Test: insertions caught | Pixel F1 | Page AUC |
| --- | --- | --- | --- | --- | --- | --- |
| v1, `p999` (old rule) | 0.5 | 82.8% | 0.4% | 0% | 0.848 | 0.932 |
| v1, `blob3` | 246 px | 87.7% | 1.4% | 9.8% | 0.848 | 0.962 |
| v2 epoch 3, `blob3` | 143 px | **96.2%** | **1.1%** | **82.9%** | **0.920** | **0.980** |
| v2 epoch 4, `blob3` | 206 px | 94.4% | 0.5% | 56.1% | 0.917 | 0.980 |

Reproduce the `blob3` rows: `python training/analysis/analyze_v2.py` from the repository root.

AUC counts ties as half. Many pages score exactly 0, and an earlier version of our scripts broke ties arbitrarily, which inflated AUC (v1 was first reported as 0.977).
