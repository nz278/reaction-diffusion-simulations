from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, peak_prominences
from matplotlib.colors import Normalize

script_path = Path(__file__).resolve()

# Read parent folder
sys.path.insert(0, str(script_path.parents[1]))

run_dir = (
    script_path.parents[1]
    / "2D_batch"
    / "runs"
    / "collect_metrics"
)

from fft_analysis_2D import load_field, radial_power_spectrum
from radial_autocor_npz import analyze_image_autocorrelation
from npz_feature_distribution import otsu_threshold, component_mask
from visualize_2D import _add_hex_field


#
def threshold_data(
    data,
    threshold="auto",
    layout="even-r",
    min_size=2,
):
    if threshold == "auto":
        threshold_used = otsu_threshold(data)  # Existing function (change?)
    else:
        threshold_used = float(threshold)  # Fixed threshold

    finite = np.isfinite(data)
    raw_mask = finite & (data > threshold_used)

    components = component_mask(raw_mask, layout)
    components = [
        component
        for component in components
        if len(component) >= min_size
    ]

    # Removes small components
    filtered_mask = np.zeros_like(raw_mask, dtype=bool)

    for component in components:
        rows, cols = zip(*component)
        filtered_mask[rows, cols] = True

    return threshold_used, raw_mask, filtered_mask, components


def base_metrics(
    npz,
    field="A_final",
):
    data, field_name = load_field(npz, field)

    finite = np.isfinite(data)
    finite_data = data[finite]

    is_uniform = (
        finite_data.size == 0
        or np.allclose(
            finite_data,
            finite_data[0],
            rtol=1e-10,
            atol=1e-12,
        )
    )

    if is_uniform:
        peak_fft_wavelength = np.nan
        peak_fft_sharpness = np.nan
        peak_fft_prominence = np.nan
        autocorr_zero_crossing = np.nan
        autocorr_length_1e = np.nan

    else:
        # Radially averaged FFT
        wavelengths, spectrum = radial_power_spectrum(data)
        valid = np.isfinite(wavelengths) & np.isfinite(spectrum) & (spectrum > 0)

        if valid.sum() > 3:
            wl = wavelengths[valid]
            ps = spectrum[valid]

            peak_idx = int(np.argmax(ps))
            peak_fft_wavelength = float(wl[peak_idx])

            median_power = np.nanmedian(ps)
            peak_fft_sharpness = (
                float(np.nanmax(ps) / median_power)
                if median_power > 0
                else np.nan
            )

            peaks, _ = find_peaks(ps)
            if len(peaks) > 0:
                main_peak = peaks[np.argmax(ps[peaks])]
                peak_fft_prominence = float(
                    peak_prominences(ps, [main_peak])[0][0]
                )
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

        valid_corr = np.isfinite(corr) & np.isfinite(distance)
        corr = corr[valid_corr]
        distance = distance[valid_corr]

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

# Separated base/threshold metrics
# Metrics based on activator/inhibitor threshold
def threshold_metrics(
    npz,
    field="A_final",
    threshold="auto",
    layout="even-r",
    min_size=2,
):
    data, field_name = load_field(npz, field)

    threshold_used, raw_mask, filtered_mask, components = threshold_data(
        data,
        threshold=threshold,
        layout=layout,
        min_size=min_size,
    )

    finite = np.isfinite(data)

    activated_fraction = (
        float(np.sum(raw_mask) / np.sum(finite))
        if np.any(finite)
        else np.nan
    )

    filtered_activated_fraction = (
        float(np.sum(filtered_mask) / np.sum(finite))
        if np.any(finite)
        else np.nan
    )

    component_sizes = np.asarray(
        [len(component) for component in components],
        dtype=float,
    )

    n_components = len(components)
    mean_component_size = (
        float(np.mean(component_sizes))
        if component_sizes.size > 0
        else 0.0
    )

    return {
        "file": npz.name,
        "field": field_name,
        "threshold": threshold_used,
        "activated_fraction": activated_fraction,
        "filtered_activated_fraction": filtered_activated_fraction,
        "n_components": n_components,
        "mean_component_size": mean_component_size,
    }


def plot_binary_mask(
    npz,
    run_dir,
    field="A_final",
    threshold="auto",
    layout="even-r",
    min_size=2,
    threshold_values=None,
):
    data, field_name = load_field(npz, field)

    data_max = (
        max(1e-12, float(np.nanmax(data)))
        if np.any(np.isfinite(data))
        else 1.0
    )

    data_norm = Normalize(vmin=0.0, vmax=data_max)
    mask_norm = Normalize(vmin=0.0, vmax=1.0)

    if threshold == "fixed":
        thresholds = [
            float(threshold_value)
            for threshold_value in threshold_values
        ]

        fig = plt.figure(
            figsize=(4 * len(thresholds), 12),
            constrained_layout=True,
        )

        grid = fig.add_gridspec(
            3,
            len(thresholds),
            height_ratios=(1.25, 1.0, 1.0),
        )

        original_ax = fig.add_subplot(grid[0, :])

        data_plot = _add_hex_field(
            original_ax,
            data,
            cmap="Greens",
            norm=data_norm,
        )
        original_ax.set_title(field_name)
        original_ax.set_xlabel("x")
        original_ax.set_ylabel("y")

        fig.colorbar(
            data_plot,
            ax=original_ax,
            fraction=0.025,
            pad=0.02,
            label="Activator",
        )

        mask_axes = []

        for col, threshold_value in enumerate(thresholds):
            threshold_used, raw_mask, filtered_mask, components = threshold_data(
                data,
                threshold=threshold_value,
                layout=layout,
                min_size=min_size,
            )

            raw_ax = fig.add_subplot(grid[1, col])
            filtered_ax = fig.add_subplot(grid[2, col])

            mask_axes.extend([raw_ax, filtered_ax])

            _add_hex_field(
                raw_ax,
                raw_mask.astype(float),
                cmap="gray",
                norm=mask_norm,
            )

            _add_hex_field(
                filtered_ax,
                filtered_mask.astype(float),
                cmap="gray",
                norm=mask_norm,
            )

            raw_ax.set_title(
                f"Threshold mask\nthreshold = {threshold_used:.4g}"
            )
            raw_ax.set_xlabel("x")
            raw_ax.set_ylabel("y")

            filtered_ax.set_title(
                f"Filtered mask\n"
                f"min size = {min_size}, components = {len(components)}"
            )
            filtered_ax.set_xlabel("x")
            filtered_ax.set_ylabel("y")

        mask_plot = plt.cm.ScalarMappable(
            norm=mask_norm,
            cmap="gray",
        )
        mask_plot.set_array([])

        fig.colorbar(
            mask_plot,
            ax=mask_axes,
            fraction=0.015,
            pad=0.02,
            label="Active",
            ticks=[0, 1],
        )

    else:
        threshold_used, raw_mask, filtered_mask, components = threshold_data(
            data,
            threshold=threshold,
            layout=layout,
            min_size=min_size,
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 5),
            constrained_layout=True,
            squeeze=False,
        )

        data_plot = _add_hex_field(
            axes[0, 0],
            data,
            cmap="Greens",
            norm=data_norm,
        )
        axes[0, 0].set_title(field_name)
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")

        raw_plot = _add_hex_field(
            axes[0, 1],
            raw_mask.astype(float),
            cmap="gray",
            norm=mask_norm,
        )
        axes[0, 1].set_title(
            f"Threshold mask\nthreshold = {threshold_used:.4g}"
        )
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("y")

        filtered_plot = _add_hex_field(
            axes[0, 2],
            filtered_mask.astype(float),
            cmap="gray",
            norm=mask_norm,
        )
        axes[0, 2].set_title(
            f"Filtered mask\n"
            f"min size = {min_size}, components = {len(components)}"
        )
        axes[0, 2].set_xlabel("x")
        axes[0, 2].set_ylabel("y")

        fig.colorbar(
            data_plot,
            ax=axes[0, 0],
            fraction=0.046,
            pad=0.04,
            label="Activator",
        )
        fig.colorbar(
            raw_plot,
            ax=axes[0, 1],
            fraction=0.046,
            pad=0.04,
            label="Active",
            ticks=[0, 1],
        )
        fig.colorbar(
            filtered_plot,
            ax=axes[0, 2],
            fraction=0.046,
            pad=0.04,
            label="Active",
            ticks=[0, 1],
        )

    save_path = (
        run_dir
        / f"binary_{npz.stem}_mask.png"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


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

        finite = np.isfinite(data)
        finite_data = data[finite]

        is_uniform = (
            finite_data.size == 0
            or np.allclose(
                finite_data,
                finite_data[0],
                rtol=1e-10,
                atol=1e-12,
            )
        )

        if is_uniform:
            axes[0, 0].plot(
                [],
                [],
                linewidth=1.4,
                label=f"{npz.stem} (uniform field)",
            )
            axes[1, 0].plot(
                [],
                [],
                linewidth=1.4,
                label=f"{npz.stem} (uniform field)",
            )
            continue

        # Radially averaged FFT
        wavelengths, spectrum = radial_power_spectrum(data)
        valid = np.isfinite(wavelengths) & np.isfinite(spectrum) & (spectrum > 0)

        if valid.sum() > 3:
            wl = wavelengths[valid]
            ps = spectrum[valid]

            max_power = np.nanmax(ps)

            if max_power > 0:
                ps = ps / max_power

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

        valid_corr = np.isfinite(corr) & np.isfinite(distance)

        if np.any(valid_corr):
            axes[1, 0].plot(
                distance[valid_corr],
                corr[valid_corr],
                linewidth=1.4,
                label=npz.stem,
            )
        else:
            axes[1, 0].plot(
                [],
                [],
                linewidth=1.4,
                label=f"{npz.stem} (undefined autocorr.)",
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
    threshold = "fixed"
    layout = "even-r"
    min_size = 2

    threshold_values = (0.2, 0.6, 1.0, 1.4, 1.8)

    if threshold == "auto":
        metric_thresholds = ("auto",)
    elif threshold == "fixed":
        metric_thresholds = threshold_values
    else:
        raise ValueError(
            'threshold must be either "auto" or "fixed"'
        )

    npz_files = sorted(run_dir.glob("*.npz"))

    base_results = [
        base_metrics(p)
        for p in npz_files
    ]
    base_summary = pd.DataFrame(base_results)

    base_save_path = run_dir / "metrics_base_summary.csv"  # Could honestly consolidate into one .csv?
    base_summary.to_csv(base_save_path, index=False)

    print("Base metrics:")
    print(base_summary.to_string(index=False))
    print(f"\nSaved: {base_save_path}")

    threshold_results = [
        threshold_metrics(
            p,
            threshold=threshold_value,
            layout=layout,
            min_size=min_size,
        )
        for p in npz_files
        for threshold_value in metric_thresholds
    ]
    threshold_summary = pd.DataFrame(threshold_results)

    threshold_save_path = run_dir / "metrics_threshold_summary.csv"
    threshold_summary.to_csv(threshold_save_path, index=False)

    print("\nThreshold metrics:")
    print(threshold_summary.to_string(index=False))
    print(f"\nSaved: {threshold_save_path}")

    for npz in npz_files:
        mask_path = plot_binary_mask(
            npz,
            run_dir,
            threshold=threshold,
            layout=layout,
            min_size=min_size,
            threshold_values=threshold_values,
        )
        print(f"Saved: {mask_path}")

    plot_path = plot_metrics(base_summary, run_dir)
    if plot_path is not None:
        print(f"Saved: {plot_path}")

    panel_path = one_png(run_dir)
    if panel_path is not None:
        print(f"Saved: {panel_path}")


if __name__ == "__main__":
    main()