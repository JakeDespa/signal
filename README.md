Image Denoiser
===============

A small Signals & Systems project that demonstrates low-pass filtering for noise removal in images.

Features
--------
- CLI mode for quick experiments and batch noise sweeps.
- Flask web frontend for uploading images, tuning filters, and previewing results.
- Saved outputs for the original image, noisy image, and each filtered result.

Quick start
-----------
Create and activate the virtual environment (optional if you already have one):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the demo on a single image (saves results to `results/`):

```powershell
python image_denoiser.py --input tsghost2.jpg --outdir results --noise-std 0.1 --filters gaussian fft --no-show
```

Batch noise example:

```powershell
python image_denoiser.py --input tsghost2.jpg --batch-noise 0.01 0.05 0.1 0.2 --outdir experiments --no-show
```

If `--input` is omitted, the script uses the scikit-image `camera` test image.

Web frontend
------------
Start the browser UI from the project venv:

```powershell
& .venv\Scripts\python.exe web_app.py
```

Then open http://127.0.0.1:8501/ in your browser.

In the web UI you can:
- Upload an image or pick one already in the project folder.
- Set noise strength, kernel size, sigma, and FFT cutoff.
- Choose which filters to run.
- View SNR scores and download the generated PNG outputs from `web_results/`.

Troubleshooting
--------------
- If `python web_app.py` fails with `No module named 'flask'`, run the app with the venv interpreter: `& .venv\Scripts\python.exe web_app.py`.
- If dependencies are missing, install them with `& .venv\Scripts\python.exe -m pip install -r requirements.txt`.
