"""
Image Denoiser - Signals & Systems Project
==========================================
Demonstrates low-pass filtering for noise removal in images.

S&S Concepts:
  - Additive Gaussian noise simulation
  - 2D convolution with spatial filters (Average, Gaussian)
  - Frequency-domain filtering via FFT
  - SNR (Signal-to-Noise Ratio) as a quantitative metric

Requirements:
  pip install numpy scipy matplotlib Pillow scikit-image
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from skimage import data
from skimage.util import img_as_float
from PIL import Image
import sys
import os


# ─────────────────────────────────────────────
# 1. IMAGE LOADING
# ─────────────────────────────────────────────

def load_image(path=None):
    """
    Load a grayscale image.
    If no path is given, uses scikit-image's built-in 'camera' test image.
    """
    if path and os.path.exists(path):
        img = Image.open(path).convert("L")        # convert to grayscale
        img_array = img_as_float(np.array(img))    # normalize to [0.0, 1.0]
        print(f"Loaded image: {path}")
    else:
        img_array = img_as_float(data.camera())    # fallback: built-in test image
        print("No image path provided — using built-in 'camera' test image.")
    return img_array


# ─────────────────────────────────────────────
# 2. NOISE ADDITION
# ─────────────────────────────────────────────

def add_gaussian_noise(image, std=0.1, seed=42):
    """
    Add zero-mean Gaussian (white) noise to the image.

    Parameters:
        image : 2D float array in [0, 1]
        std   : standard deviation of noise (controls intensity)
        seed  : random seed for reproducibility

    Returns:
        noisy image clipped to [0, 1]
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=image.shape)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)


# ─────────────────────────────────────────────
# 3. FILTERS (Low-Pass)
# ─────────────────────────────────────────────

def apply_average_filter(image, kernel_size=5):
    """
    Spatial low-pass filter: simple box/average kernel.
    Every output pixel = mean of its kernel_size x kernel_size neighborhood.
    This is equivalent to convolving with a uniform kernel.
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return convolve(image, kernel)


def apply_gaussian_filter(image, sigma=2.0):
    """
    Spatial low-pass filter: Gaussian-weighted kernel.
    Smoother than the average filter — preserves edges better.
    sigma controls the spread (larger = more blurring).
    """
    return gaussian_filter(image, sigma=sigma)


def apply_fft_lowpass_filter(image, cutoff_ratio=0.1):
    """
    Frequency-domain low-pass filter using the 2D FFT.

    Steps:
      1. Compute 2D FFT and shift DC to center
      2. Create a circular mask that zeros out high frequencies
      3. Inverse FFT to recover the filtered image

    cutoff_ratio: fraction of the frequency spectrum to keep (0.1 = keep inner 10%)
    """
    # Step 1: FFT
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)       # shift zero-freq to center

    # Step 2: Build circular low-pass mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2              # center of spectrum
    radius = int(min(rows, cols) * cutoff_ratio)

    mask = np.zeros((rows, cols), dtype=np.uint8)
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    mask[dist_from_center <= radius] = 1          # keep frequencies inside circle

    # Step 3: Apply mask & inverse FFT
    f_filtered = f_shifted * mask
    f_back = np.fft.ifftshift(f_filtered)
    img_back = np.fft.ifft2(f_back)
    return np.clip(np.abs(img_back), 0.0, 1.0)


# ─────────────────────────────────────────────
# 4. SNR METRIC
# ─────────────────────────────────────────────

def compute_snr(clean, processed):
    """
    Signal-to-Noise Ratio (in dB) using PSNR as a proxy.
    Higher SNR = cleaner image = better denoising.
    PSNR is a standard metric used in image processing literature.
    """
    mse = np.mean((clean - processed) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────

def plot_results(original, noisy, results: dict, snr_baseline, outpath=None, show=True):
    """
    Plot original, noisy, and all filtered images side by side.
    Also prints SNR values for quantitative comparison.

    results: dict of {label: filtered_image}
    """
    n_filters = len(results)
    fig, axes = plt.subplots(1, 2 + n_filters, figsize=(5 * (2 + n_filters), 5))
    fig.suptitle("Image Denoiser — Signals & Systems Project", fontsize=14, fontweight="bold")

    cmap = "gray"

    # Original
    axes[0].imshow(original, cmap=cmap)
    axes[0].set_title(f"Original\n(SNR baseline)")
    axes[0].axis("off")

    # Noisy
    snr_noisy = compute_snr(original, noisy)
    axes[1].imshow(noisy, cmap=cmap)
    axes[1].set_title(f"Noisy Image\nSNR: {snr_noisy:.2f} dB")
    axes[1].axis("off")

    # Filtered results
    for ax, (label, filtered) in zip(axes[2:], results.items()):
        snr_val = compute_snr(original, filtered)
        ax.imshow(filtered, cmap=cmap)
        ax.set_title(f"{label}\nSNR: {snr_val:.2f} dB")
        ax.axis("off")

    plt.tight_layout()
    if outpath is None:
        outpath = "denoiser_results.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot → {outpath}")
    if show:
        plt.show()


def plot_frequency_spectrum(image, title="Frequency Spectrum", outpath=None, show=True):
    """
    Visualize the 2D FFT magnitude spectrum (log-scaled) of an image.
    Useful for understanding what frequencies are present.
    """
    f = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log1p(np.abs(f))               # log scale for visibility

    plt.figure(figsize=(5, 5))
    plt.imshow(magnitude, cmap="inferno")
    plt.title(title)
    plt.colorbar(label="Log magnitude")
    plt.axis("off")
    plt.tight_layout()
    if outpath is None:
        outpath = "frequency_spectrum.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved spectrum plot → {outpath}")
    if show:
        plt.show()


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Image Denoiser — S&S Project")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Image denoiser demo with configurable CLI options.")
    parser.add_argument("--input", "-i", help="Input image path or directory. If omitted, uses built-in test image.", default=None)
    parser.add_argument("--outdir", "-o", help="Output directory", default="results")
    parser.add_argument("--noise-std", type=float, default=0.2, help="Gaussian noise standard deviation")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--filters", nargs="+", choices=["average", "gaussian", "fft", "all"], default=["all"], help="Filters to run")
    parser.add_argument("--kernel-size", type=int, default=5, help="Kernel size for average filter")
    parser.add_argument("--sigma", type=float, default=2.0, help="Sigma for Gaussian filter")
    parser.add_argument("--cutoff", type=float, default=0.12, help="Cutoff ratio for FFT low-pass (fraction of min dimension)")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots (just save)")
    parser.add_argument("--batch-noise", nargs="*", type=float, help="Run multiple noise std values (overrides --noise-std when provided)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve input paths
    input_path = args.input
    if input_path is None:
        image_paths = [None]
    else:
        p = Path(input_path)
        if p.is_dir():
            # include common image extensions
            image_paths = sorted([str(x) for x in p.glob("**/*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
        else:
            image_paths = [str(p)]

    # Determine filters to run
    selected = args.filters
    if "all" in selected:
        selected = ["average", "gaussian", "fft"]

    noise_list = args.batch_noise if args.batch_noise else [args.noise_std]

    for img_path in image_paths:
        original = load_image(img_path)
        stem = "camera" if img_path is None else Path(img_path).stem

        for noise_std in noise_list:
            noisy = add_gaussian_noise(original, std=noise_std, seed=args.seed)
            print(f"\nNoise added (std={noise_std})")
            print(f"Noisy image SNR:  {compute_snr(original, noisy):.2f} dB")

            results = {}
            if "average" in selected:
                results[f"Average_{args.kernel_size}x{args.kernel_size}"] = apply_average_filter(noisy, kernel_size=args.kernel_size)
            if "gaussian" in selected:
                results[f"Gaussian_sigma{args.sigma}"] = apply_gaussian_filter(noisy, sigma=args.sigma)
            if "fft" in selected:
                results[f"FFT_cutoff{int(args.cutoff*100)}pct"] = apply_fft_lowpass_filter(noisy, cutoff_ratio=args.cutoff)

            # Print SNRs
            print("\n── SNR Comparison (higher = better) ──")
            print(f"  Noisy image:       {compute_snr(original, noisy):.2f} dB")
            for label, img in results.items():
                print(f"  {label:<30} {compute_snr(original, img):.2f} dB")

            # Save combined comparison
            comparison_name = outdir / f"{stem}_comparison_noise{noise_std:.3f}.png"
            plot_results(original, noisy, results, compute_snr(original, noisy), outpath=str(comparison_name), show=not args.no_show)

            # Save frequency spectra if requested
            spec_original = outdir / f"{stem}_spectrum_original_noise{noise_std:.3f}.png"
            spec_noisy = outdir / f"{stem}_spectrum_noisy_noise{noise_std:.3f}.png"
            plot_frequency_spectrum(original, "Spectrum — Original", outpath=str(spec_original), show=not args.no_show)
            plot_frequency_spectrum(noisy, "Spectrum — Noisy", outpath=str(spec_noisy), show=not args.no_show)

    print("\nDone! ✓")

    print("\nDone! ✓")


if __name__ == "__main__":
    main()
