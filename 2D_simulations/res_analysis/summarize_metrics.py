from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, peak_prominences

from fft_analysis_2D import load_field, radial_power_spectrum
from radial_autocor_npz import analyze_image_autocorrelation


script_path = Path(__file__).resolve()
run_dir = script_path.parents[1] / "2D_batch" / "runs" / "collect_metrics"


def base_summarize(npz, field="A_final"):
    data, field_name = load_field(npz, field)

    # Radially averaged FFT
    wavelengths, spectrum = radial_power_spectrum(data)
    valid = np.isfinite(wavelengths) & np.isfinite(spectrum) & (spectrum > 0)

    if valid.sum() > 3:
        wl = wavelengths[valid]
        ps = spectrum[valid]

        peak_idx = int(np.argmax(ps))
        peak_fft_wavelength = float(wl[peak_idx])
        peak_fft_sharpness = float(np.nanmax(ps) / np.nanmedian(ps))

        peaks, _ = find_peaks(ps)
        if len(peaks) > 0:
            main_peak = peaks[np.argmax(ps[peaks])]
            peak_fft_prominence = float(peak_prominences(ps, [main_peak])[0][0])
        else:
            peak_fft_prominence = np.nan
    else:
        peak_fft_wavelength = np.nan
        peak_fft_sharpness = np.nan
        peak_fft_prominence = np.nan

    # Radial autocorrelation
    _, autocorr = analyze_image_autocorrelation(data)
    corr = autocorr["radial_autocorrelation"].to_numpy()
    distance = autocorr["bin_centers"].to_numpy()

    zero_idx = np.where(corr <= 0)[0]
    autocorr_zero_crossing = (
        float(distance[zero_idx[0]])
        if len(zero_idx) > 0
        else np.nan
    )

    decay_idx = np.where(corr <= 1 / np.e)[0]
    autocorr_length_1e = (
        float(distance[decay_idx[0]])
        if len(decay_idx) > 0
        else np.nan
    )

    return {
        "file": npz.name,
        "field": field_name,
        "peak_fft_wavelength": peak_fft_wavelength,
        "peak_fft_sharpness": peak_fft_sharpness,
        "peak_fft_prominence": peak_fft_prominence,
        "autocorr_zero_crossing": autocorr_zero_crossing,
        "autocorr_length_1e": autocorr_length_1e,
    }


def plot_metrics(summary, run_dir):
    if summary.empty:
        print("Skipping plot: no .npz files found.")
        return None

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=False,
        squeeze=False,
    )

    for npz in sorted(run_dir.glob("*.npz")):
        data, field_name = load_field(npz, "A_final")

        # Radially averaged FFT
        wavelengths, spectrum = radial_power_spectrum(data)
        valid = np.isfinite(wavelengths) & np.isfinite(spectrum) & (spectrum > 0)

        if valid.sum() > 3:
            wl = wavelengths[valid]
            ps = spectrum[valid]
            ps = ps / np.nanmax(ps)

            axes[0, 0].plot(
                wl,
                ps,
                linewidth=1.4,
                label=npz.stem,
            )
        else:
            axes[0, 0].plot(
                [],
                [],
                linewidth=1.4,
                label=f"{npz.stem} (no spatial power)",
            )

        # Autocorrelation
        _, autocorr = analyze_image_autocorrelation(data)
        corr = autocorr["radial_autocorrelation"].to_numpy()
        distance = autocorr["bin_centers"].to_numpy()

        axes[1, 0].plot(
            distance,
            corr,
            linewidth=1.4,
            label=npz.stem,
        )

    axes[0, 0].set_title("Fourier power spectrum")
    axes[0, 0].set_xlabel("Wavelength (cells)")
    axes[0, 0].set_ylabel("Normalized power")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6)

    axes[1, 0].set_title("Autocorrelation")
    axes[1, 0].set_xlabel("Distance (cells)")
    axes[1, 0].set_ylabel("Radially Avg. Autocorr.")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=6)

    fig.tight_layout()

    save_path = run_dir / "base_metrics_plots.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return save_path

def one_png(run_dir):
    png_files = sorted(run_dir.glob("*.png"))
    png_files = [
        p for p in png_files
        if p.name not in {"base_metrics_plots.png", "all_runs.png"}
    ]

    if not png_files:
        print("No .png files found.")
        return None

    images = [plt.imread(p) for p in png_files]

    fig, axes = plt.subplots(
        len(images),
        1,
        figsize=(8, 4 * len(images)),
        squeeze=False,
    )

    for ax, img, path in zip(axes[:, 0], images, png_files):
        ax.imshow(img)
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")

    fig.tight_layout()

    save_path = run_dir / "all_runs.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path

def main():
    results = [base_summarize(p) for p in sorted(run_dir.glob("*.npz"))]
    summary = pd.DataFrame(results)

    save_path = run_dir / "base_metrics_summary.csv"
    summary.to_csv(save_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved: {save_path}")

    plot_path = plot_metrics(summary, run_dir)
    if plot_path is not None:
        print(f"Saved: {plot_path}")

    panel_path = one_png(run_dir)
    if panel_path is not None:
        print(f"Saved: {panel_path}")


if __name__ == "__main__":
    main()