# How to run Scrutiny Desk

Step by step, on your own computer. No GPU needed.

## 1. What you need

- **Python 3.10 to 3.13**, from [python.org](https://www.python.org/downloads/).
  On Windows, tick **"Add python.exe to PATH"** on the first screen of the installer.
- **Git**, from [git-scm.com](https://git-scm.com/downloads). Or skip it and use *Code → Download ZIP* on the GitHub page, then unzip.
- About **3 GB of free disk** and an internet connection for the first install.

Check that Python works: open a terminal (Windows: *Command Prompt*) and type `python --version`.

## 2. Get the code

```
git clone https://github.com/Dynemies/scrutiny-desk.git
cd scrutiny-desk/app
```

(If you downloaded the ZIP instead, open a terminal inside the unzipped `scrutiny-desk-main/app` folder.)

## 3. Install, once

**Windows (Command Prompt)**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first install downloads about 1 GB (mostly PyTorch) and takes a few minutes. `(.venv)` at the start of the prompt means the environment is active.

## 4. Start the app

```
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. If it does not, open that address yourself. To stop the app, press `Ctrl + C` in the terminal.

## 5. What to look at

| Tab | What you see |
| --- | --- |
| **Application** | 14 sample applications in the left sidebar (*Needs action*, *Verified*, *All*). Pick one to see its verdict, each document, and the proof for every finding |
| **Queue overview** | How many applications need an officer, what the checks found across all of them, and the waiting list |
| **Tamper lab** | The tamper AI, page by page: heatmap, mask, true edit, close-up. Upload your own pages here |
| **Check a file** | Upload one PDF: it checks the digital signature, spots changes made after signing, and looks for the same page in other applications. Try the files in [`samples/`](samples), e.g. `03_changed_after_signing` |

### The Tamper lab

1. Open the **Tamper lab** tab. 20 test pages are already loaded with the model's results: 10 edited, 10 genuine, made after training.
2. Click **View** under a page. Above the page, switch between **On page, Heatmap, Mask, Truth, Compare** and **Original**. Scroll down for the enlarged close-up.
3. Good ones to start with: **Tampered 04** (one digit added to the AICTE ID) and **Genuine 03** (the one false alarm, on part of a name).
4. To check pages live, drag JPEG, PNG or PDF files into the upload box at the top and press **Check**. The 20 test pages are in [`samples/test_pages/`](samples/test_pages) if you want to try them yourself.
   - The **first** check downloads the model (357 MB, once) from the [Bestpath release](../../releases/tag/Bestpath).
   - After that, a page takes about a minute on a laptop CPU (about 2 seconds on a GPU).
   - Your results stay under **My pages** until you press *Clear my pages*.

## 6. Next time

```
cd scrutiny-desk/app
.venv\Scripts\activate          (macOS / Linux: source .venv/bin/activate)
streamlit run app.py
```

## If something goes wrong

| Problem | Fix |
| --- | --- |
| `python` is not recognised | Reinstall Python with *Add python.exe to PATH* ticked, or use `py` instead of `python` |
| `.venv\Scripts\activate` fails in **PowerShell** | Use Command Prompt instead, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then `.venv\Scripts\Activate.ps1` |
| `Port 8501 is not available` | Another copy is running. Close it, or start with `streamlit run app.py --server.port 8502` |
| Tamper lab says *Model packages not installed* | The environment is not active, or the install stopped part-way. Activate `.venv` and run `pip install -r requirements.txt` again |
| The model download fails or is blocked | Download `ffdn_v2_best.1.pth` from the [Bestpath release](../../releases/tag/Bestpath) yourself, rename it to `ffdn_v2_best.pth`, and put it in `app/models/` |
| You have an NVIDIA GPU and want it used | After the normal install, run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`. The app uses the GPU by itself when one is available |

Everything in the app is synthetic: no real institution, person or document is used.
