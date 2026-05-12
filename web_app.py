import sys
try:
    from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
except Exception:
    print("Flask is not installed in the active Python environment.\n"
          "Run the app with the project venv python:\n"
          "  & .venv\\Scripts\\python.exe web_app.py\n"
          "Or activate the venv in PowerShell and run: \n"
          "  & .venv\\Scripts\\Activate.ps1\n"
          "  python web_app.py\n")
    sys.exit(1)
from pathlib import Path
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from image_denoiser import (
    load_image,
    add_gaussian_noise,
    apply_average_filter,
    apply_gaussian_filter,
    apply_fft_lowpass_filter,
    compute_snr,
)

APP_ROOT = Path(__file__).parent
UPLOADS = APP_ROOT / "uploads"
RESULTS = APP_ROOT / "web_results"
UPLOADS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

app = Flask(__name__)


def list_local_images():
    return sorted(
        p.name
        for p in APP_ROOT.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )


def save_array_as_png(arr, path: Path):
    # arr: float array in [0,1]
    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8)
    img.save(path)


def save_frequency_spectrum(image, path: Path, title: str):
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(spectrum, cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Image Denoiser</title>
    <style>
        :root {
            --bg: #0b1020;
            --panel: rgba(17, 24, 39, 0.92);
            --panel-2: rgba(30, 41, 59, 0.9);
            --text: #e5eefc;
            --muted: #98a6c7;
            --accent: #67e8f9;
            --accent-2: #a78bfa;
            --line: rgba(148, 163, 184, 0.2);
            --shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(103, 232, 249, 0.18), transparent 26%),
                radial-gradient(circle at top right, rgba(167, 139, 250, 0.16), transparent 24%),
                linear-gradient(160deg, #050816, #0b1020 45%, #111827);
            padding: 32px;
        }
        .shell {
            max-width: 1200px;
            margin: 0 auto;
        }
        .hero {
            display: grid;
            grid-template-columns: 1.35fr 0.85fr;
            gap: 20px;
            align-items: stretch;
            margin-bottom: 20px;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }
        .hero-main {
            padding: 30px;
            position: relative;
            overflow: hidden;
        }
        .hero-main::after {
            content: "";
            position: absolute;
            inset: auto -120px -140px auto;
            width: 280px;
            height: 280px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(103, 232, 249, 0.24), transparent 70%);
            pointer-events: none;
        }
        .eyebrow {
            display: inline-flex;
            gap: 10px;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid rgba(103, 232, 249, 0.25);
            background: rgba(15, 23, 42, 0.75);
            color: var(--accent);
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        h1 {
            margin: 18px 0 10px;
            font-size: clamp(2rem, 4vw, 3.7rem);
            line-height: 0.98;
            letter-spacing: -0.04em;
        }
        .lede {
            max-width: 60ch;
            color: var(--muted);
            line-height: 1.6;
            font-size: 1.02rem;
            margin-bottom: 0;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 24px;
        }
        .stat {
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid var(--line);
        }
        .stat .k {
            display: block;
            font-size: 0.78rem;
            color: var(--muted);
            margin-bottom: 6px;
        }
        .stat .v {
            font-size: 1.02rem;
            font-weight: 700;
            color: white;
        }
        .hero-side {
            padding: 22px;
            background: linear-gradient(180deg, rgba(167, 139, 250, 0.14), rgba(103, 232, 249, 0.06));
        }
        .mini-title {
            margin: 0 0 12px;
            font-size: 1rem;
            letter-spacing: -0.02em;
        }
        .hint-list {
            list-style: none;
            padding: 0;
            margin: 0;
            display: grid;
            gap: 10px;
            color: var(--muted);
            line-height: 1.5;
        }
        .hint-list li {
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid var(--line);
        }
        .content {
            display: grid;
            grid-template-columns: 0.95fr 1.05fr;
            gap: 20px;
        }
        .card {
            padding: 24px;
        }
        .section-title {
            margin: 0 0 18px;
            font-size: 1.1rem;
            letter-spacing: -0.02em;
        }
        form {
            display: grid;
            gap: 14px;
        }
        label {
            display: grid;
            gap: 8px;
            color: #dbeafe;
            font-size: 0.95rem;
        }
        input, select {
            width: 100%;
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(2, 6, 23, 0.75);
            color: var(--text);
            outline: none;
        }
        input:focus, select:focus {
            border-color: rgba(103, 232, 249, 0.75);
            box-shadow: 0 0 0 4px rgba(103, 232, 249, 0.12);
        }
        .grid2 {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        .grid3 {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        .filters {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 12px 0 0;
        }
        .chip {
            display: inline-flex;
            gap: 8px;
            align-items: center;
            padding: 10px 12px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid var(--line);
            color: var(--text);
            font-size: 0.92rem;
        }
        .chip input { width: auto; margin: 0; }
        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 6px;
        }
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 12px 16px;
            border-radius: 14px;
            border: 1px solid transparent;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #08111f;
            font-weight: 800;
            cursor: pointer;
            text-decoration: none;
        }
        .btn.secondary {
            background: rgba(15, 23, 42, 0.85);
            border-color: var(--line);
            color: var(--text);
        }
        .note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
            margin: 0;
        }
        .guide {
            margin-top: 14px;
            display: grid;
            gap: 10px;
        }
        .guide-item {
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid var(--line);
        }
        .guide-item h3 {
            margin: 0 0 6px;
            font-size: 0.98rem;
            color: #e2f3ff;
            letter-spacing: -0.01em;
        }
        .guide-item p {
            margin: 0;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-top: 12px;
        }
        .result-card {
            padding: 16px;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--line);
        }
        .result-card img {
            width: 100%;
            display: block;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            margin-top: 10px;
        }
        .result-meta {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            align-items: center;
            margin-bottom: 8px;
            color: var(--muted);
            font-size: 0.92rem;
        }
        .footer {
            margin-top: 16px;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }
        @media (max-width: 960px) {
            body { padding: 16px; }
            .hero, .content, .grid2, .grid3, .stats { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <div class="panel hero-main">
                <div class="eyebrow">Signals & Systems Project</div>
                <h1>Image denoising with a cleaner, modern workflow.</h1>
                <p class="lede">Upload an image or reuse one from the workspace, choose the noise level and filters, then inspect the noisy input and filtered outputs side by side with SNR scores.</p>
                <div class="stats">
                    <div class="stat"><span class="k">Input options</span><span class="v">Upload or local sample</span></div>
                    <div class="stat"><span class="k">Filters</span><span class="v">Average, Gaussian, FFT</span></div>
                    <div class="stat"><span class="k">Output</span><span class="v">Saved PNG comparison set</span></div>
                </div>
            </div>
            <div class="panel hero-side">
                <h2 class="mini-title">How it works</h2>
                <ul class="hint-list">
                    <li>Pick an image from the project folder or upload your own.</li>
                    <li>Adjust the noise std and filter settings before running.</li>
                    <li>View SNR numbers and download the generated images from <code>web_results/</code>.</li>
                </ul>
            </div>
        </section>

        <section class="content">
            <div class="panel card">
                <h2 class="section-title">Run a denoising experiment</h2>
                <form method="post" enctype="multipart/form-data" action="/process">
                    <label>
                        Upload image
                        <input type="file" name="file" accept="image/*">
                    </label>
                    <label>
                        Or choose an existing file
                        <select name="existing">
                            <option value="">-- use uploaded image or default sample --</option>
                            {% for f in existing %}
                                <option value="{{f}}">{{f}}</option>
                            {% endfor %}
                        </select>
                    </label>

                    <div class="grid2">
                        <label>
                            Noise std
                            <input type="number" name="noise" value="0.1" step="0.01" min="0" max="1">
                        </label>
                        <label>
                            Seed
                            <input type="number" name="seed" value="42" step="1">
                        </label>
                    </div>

                    <div class="filters">
                        <label class="chip"><input type="checkbox" name="filters" value="average" checked> Average</label>
                        <label class="chip"><input type="checkbox" name="filters" value="gaussian" checked> Gaussian</label>
                        <label class="chip"><input type="checkbox" name="filters" value="fft" checked> FFT</label>
                    </div>

                    <div class="grid3">
                        <label>
                            Kernel size
                            <input type="number" name="kernel" value="5" min="1" step="2">
                        </label>
                        <label>
                            Sigma
                            <input type="number" name="sigma" value="2.0" min="0.1" step="0.1">
                        </label>
                        <label>
                            Cutoff
                            <input type="number" name="cutoff" value="0.12" min="0.01" max="0.5" step="0.01">
                        </label>
                    </div>

                    <div class="actions">
                        <button class="btn" type="submit">Run denoiser</button>
                        <a class="btn secondary" href="/">Reset</a>
                    </div>
                </form>
            </div>

            <div class="panel card">
                <h2 class="section-title">Project notes</h2>
                <p class="note">This frontend is designed for quick experiments: it saves the original, noisy, and filtered images in <code>web_results/</code> and prints a score for each output. For best results, start with low noise and a Gaussian filter, then compare against the FFT filter.</p>
                <div class="guide">
                    <div class="guide-item">
                        <h3>Average filter (box blur)</h3>
                        <p>Replaces each pixel with the local neighborhood mean. It removes random noise quickly but also softens edges. Spectrum view: high-frequency components (fine detail and sharp edges) are reduced broadly.</p>
                    </div>
                    <div class="guide-item">
                        <h3>Gaussian filter</h3>
                        <p>Applies a smooth, center-weighted blur controlled by sigma. It usually preserves structure better than a box filter at similar noise levels. Spectrum view: frequencies fade gradually from center to outer regions instead of a hard cutoff.</p>
                    </div>
                    <div class="guide-item">
                        <h3>FFT low-pass filter</h3>
                        <p>Moves the image to frequency domain and keeps only frequencies inside a circular radius set by cutoff. This can strongly suppress noise but may remove texture if cutoff is too small. Spectrum view: bright central disk remains, outer high-frequency ring is suppressed.</p>
                    </div>
                    <div class="guide-item">
                        <h3>How to read the spectrum</h3>
                        <p>The center contains low frequencies (overall shapes and lighting), and farther regions contain high frequencies (fine detail and noise). Better denoising usually means less scattered high-frequency energy while keeping meaningful mid-frequency structure.</p>
                    </div>
                </div>
                <p class="footer">Tip: if the UI feels too slow to render plots, use the web page for the image comparison only and keep the CLI for batch experiments.</p>
            </div>
        </section>
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    existing = list_local_images()
    return render_template_string(INDEX_HTML, existing=existing)


@app.route("/process", methods=["POST"])
def process():
    # Determine input image
    file = request.files.get("file")
    existing = request.form.get("existing")
    if file and file.filename:
        fname = uuid.uuid4().hex + Path(file.filename).suffix
        fpath = UPLOADS / fname
        file.save(fpath)
        image_path = str(fpath)
    elif existing:
        image_path = str(APP_ROOT / existing)
    else:
        image_path = None

    noise = float(request.form.get("noise", "0.1"))
    kernel = int(request.form.get("kernel", 5))
    sigma = float(request.form.get("sigma", 2.0))
    cutoff = float(request.form.get("cutoff", 0.12))
    filters = request.form.getlist("filters")

    original = load_image(image_path)
    noisy = add_gaussian_noise(original, std=noise)

    results = {}
    if "average" in filters:
        results[f"average_{kernel}x{kernel}"] = apply_average_filter(noisy, kernel_size=kernel)
    if "gaussian" in filters:
        results[f"gaussian_sigma{sigma}"] = apply_gaussian_filter(noisy, sigma=sigma)
    if "fft" in filters:
        results[f"fft_cutoff{int(cutoff*100)}pct"] = apply_fft_lowpass_filter(noisy, cutoff_ratio=cutoff)

    # Save images
    stem = (Path(image_path).stem if image_path else "camera")
    out_files = {}
    spectrum_files = {}
    orig_name = f"{stem}_orig.png"
    noisy_name = f"{stem}_noisy.png"
    save_array_as_png(original, RESULTS / orig_name)
    save_array_as_png(noisy, RESULTS / noisy_name)
    save_frequency_spectrum(original, RESULTS / f"{stem}_orig_spectrum.png", "Original spectrum")
    save_frequency_spectrum(noisy, RESULTS / f"{stem}_noisy_spectrum.png", "Noisy spectrum")
    out_files["noisy"] = noisy_name
    spectrum_files["noisy"] = f"{stem}_noisy_spectrum.png"

    snrs = {"noisy": compute_snr(original, noisy)}
    for label, arr in results.items():
        name = f"{stem}_{label}.png"
        save_array_as_png(arr, RESULTS / name)
        out_files[label] = name
        snrs[label] = compute_snr(original, arr)
        spectrum_name = f"{stem}_{label}_spectrum.png"
        save_frequency_spectrum(arr, RESULTS / spectrum_name, f"{label.replace('_', ' ')} spectrum")
        spectrum_files[label] = spectrum_name

        # Render results page
    spectrum_notes = {
        "original": "Baseline spectrum: detail and structure are intact across low and high frequencies.",
        "noisy": "Noise spreads energy into outer high-frequency regions, making the spectrum look more scattered.",
    }
    rows = []
    for key, fname in out_files.items():
        spectrum_name = spectrum_files.get(key)
        if key.startswith("average_"):
            note = "Average filtering suppresses high frequencies broadly, so edges and texture are softened."
        elif key.startswith("gaussian_"):
            note = "Gaussian filtering reduces high frequencies smoothly, usually preserving overall shapes better."
        elif key.startswith("fft_"):
            note = "FFT low-pass keeps center frequencies and cuts outer regions based on the cutoff radius."
        else:
            note = spectrum_notes.get(key, "Low frequencies near the center represent coarse structure; outer regions represent fine detail.")
        rows.append((key, f"/results/{fname}", snrs.get(key), fname, f"/results/{spectrum_name}" if spectrum_name else None, note))

    RESULT_HTML = """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Denoiser Results</title>
            <style>
                :root {
                    --bg: #0b1020;
                    --panel: rgba(17, 24, 39, 0.92);
                    --line: rgba(148, 163, 184, 0.2);
                    --text: #e5eefc;
                    --muted: #98a6c7;
                    --accent: #67e8f9;
                }
                * { box-sizing: border-box; }
                body {
                    margin: 0;
                    padding: 32px;
                    font-family: "Segoe UI", system-ui, sans-serif;
                    color: var(--text);
                    background: linear-gradient(160deg, #050816, #0b1020 45%, #111827);
                }
                .wrap { max-width: 1200px; margin: 0 auto; }
                .top {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 16px;
                    margin-bottom: 18px;
                    flex-wrap: wrap;
                }
                .card {
                    background: var(--panel);
                    border: 1px solid var(--line);
                    border-radius: 22px;
                    padding: 18px;
                }
                .btn {
                    display: inline-flex;
                    align-items: center;
                    padding: 11px 15px;
                    border-radius: 14px;
                    background: rgba(15, 23, 42, 0.85);
                    border: 1px solid var(--line);
                    color: var(--text);
                    text-decoration: none;
                }
                h1 { margin: 0; letter-spacing: -0.03em; }
                .subtitle { color: var(--muted); margin-top: 6px; }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                    gap: 16px;
                }
                .result-card img {
                    width: 100%;
                    border-radius: 14px;
                    display: block;
                    margin-top: 10px;
                    border: 1px solid var(--line);
                }
                .spectrum {
                    margin-top: 12px;
                }
                .spectrum img {
                    aspect-ratio: 1 / 1;
                    object-fit: cover;
                }
                .spectrum-note {
                    margin: 8px 0 0;
                    font-size: 0.86rem;
                    line-height: 1.45;
                    color: var(--muted);
                }
                .meta {
                    display: flex;
                    justify-content: space-between;
                    gap: 10px;
                    align-items: center;
                    font-size: 0.95rem;
                    color: var(--muted);
                }
                .snr {
                    color: var(--accent);
                    font-weight: 700;
                }
            </style>
        </head>
        <body>
            <div class="wrap">
                <div class="top">
                    <div>
                        <h1>Results</h1>
                        <div class="subtitle">Noisy, filtered outputs, and their spectrum graphs saved to <code>web_results/</code>.</div>
                    </div>
                    <a class="btn" href="/">Run another image</a>
                </div>

                <div class="grid">
                {% for label, src, snr, fname, spectrum_src, spectrum_note in rows %}
                    <div class="card result-card">
                        <div class="meta">
                            <span>{{label}}</span>
                            {% if snr is not none %}<span class="snr">{{"%.2f"|format(snr)}} dB</span>{% endif %}
                        </div>
                        <img src="{{src}}" alt="{{label}}">
                        {% if spectrum_src %}
                        <div class="spectrum">
                            <img src="{{spectrum_src}}" alt="{{label}} spectrum">
                            <p class="spectrum-note">{{spectrum_note}}</p>
                        </div>
                        {% endif %}
                        <div class="subtitle"><a class="btn" href="{{src}}" download="{{fname}}">Download PNG</a></div>
                    </div>
                {% endfor %}
                </div>
            </div>
        </body>
        </html>
    """

    return render_template_string(RESULT_HTML, rows=rows)


@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(RESULTS, filename)


if __name__ == "__main__":
    app.run(port=8501, debug=True)
