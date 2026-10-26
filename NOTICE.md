# Third-party material and credits

| What | Where it is used | Licence / terms |
| --- | --- | --- |
| **FFDN** model code, from [ForensicHub](https://github.com/scu-zjz/ForensicHub) (Sichuan University) | Fetched at run time by the training and heatmap cells; not copied into this repository | CC BY 4.0 |
| FFDN method: Chen et al., *Enhancing Tampered Text Detection through Frequency Feature Fusion and Decomposition*, ECCV 2024 | Tamper model architecture | Paper (cite when using) |
| ForensicHub FFDN document checkpoint (trained on DocTamper) | Starting weights for our fine-tune | Research use. Our fine-tuned weights are derived from it, so they are released **for research and evaluation only** |
| DocTamper: Qu et al., *Towards Robust Tampered Text Detection in Document Image*, CVPR 2023 | Data behind the starting checkpoint (not used or redistributed here) | Research use under the dataset's own agreement |
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | PDF rendering in the generator and the app | AGPL-3.0, which is why this repository is AGPL-3.0 |
| [pyHanko](https://github.com/MatthiasValvekens/pyHanko) | Signing and validating PDFs | MIT |
| [Streamlit](https://streamlit.io) | App interface | Apache 2.0 |
| Noto Serif / Sans (and Devanagari) fonts | Generated documents (`generator/fonts`) | SIL Open Font License 1.1 (`generator/fonts/OFL.txt`) |
| Martel and Mukta fonts | App interface (`app/static`) | SIL Open Font License 1.1 (`app/static/OFL-*.txt`) |

All documents, institutions, people and records in this repository are synthetic. No real AICTE, NBA, fire-service or
government document, logo or seal is reproduced.
