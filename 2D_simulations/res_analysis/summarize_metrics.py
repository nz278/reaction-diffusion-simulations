from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import existing analysis functions from Ben's repo
from fft_analysis_2D import load_field, radial_power_spectrum
from radial_autocor_npz import analyze_image_autocorrelation
# Otsu-thresholded feature metrics (activated fraction/components) excluded for now —
# every field gets measured against a different per-field threshold, which reads a
# fully-on field as "50% activated" (the artifact Dr. Morsut flagged). Re-enable once
# the group settles on a shared cutoff approach. Import kept commented, not deleted,
# so re-enabling is a one-line change.
# from npz_feature_distribution import otsu_threshold, component_mask, approx_diameter_hexagons
from scipy.signal import find_peaks, peak_widths

script_path = Path(__file__).resolve()
run_dir = script_path.parents[1] / "2D_batch" / "runs" / "collect_metrics"

# A field with ~zero variance (uniform ON, or uniform OFF) has no real spatial
# structure for the FFT/autocorrelation metrics to describe. Computing them
# anyway picks up pure floating-point noise (~1e-13 and smaller) and reports
# it as a real peak/decay length, which looks like data but isn't. The
# weakest *real* pattern we've validated against has std ~0.41, so this
# threshold sits with enormous margin on both sides of the noise floor and
# the smallest real signal seen so far.
UNIFORM_FIELD_STD_EPS = 1e-6
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


# Just base summarize function for now
def base_summarize(npz, field="A_final", threshold="auto", layout="even-r", min_size=2):
    data, field_name = load_field(npz, field)

    # Otsu-thresholded feature metrics excluded for now — see import note above.
    # threshold_val = otsu_threshold(data) if threshold == "auto" else float(threshold)
    # active_mask = np.isfinite(data) & (data > threshold_val)
    # components = [c for c in component_mask(active_mask, layout) if len(c) >= min_size]
    # component_sizes = np.array([len(c) for c in components], dtype=float)
    # diameters = np.array(
    #     [approx_diameter_hexagons(int(s)) for s in component_sizes], dtype=float
    # )

    # FFT power spectrum + autocorrelation are both undefined for a field with
    # no real spatial structure (uniform ON or uniform OFF) — without this guard
    # both return numbers that look like data but are actually just floating-
    # point noise dressed up as a peak/decay length. See UNIFORM_FIELD_STD_EPS.
    is_uniform = float(np.std(data)) < UNIFORM_FIELD_STD_EPS

    peak_wavelength = np.nan
    peak_width = np.nan
    peak_sharpness = np.nan
    zero_crossing = np.nan
    corr_length = np.nan

    if not is_uniform:
        # FFT power spectrum
        wavelengths, spectrum = radial_power_spectrum(data)
        if len(spectrum) > 0 and np.any(np.isfinite(spectrum)):
            positive = spectrum[spectrum > 0]
            peak_sharpness = (
                float(np.nanmax(spectrum) / np.nanmedian(positive))
                if len(positive) > 0
                else np.nan
            )
            peak_idx_arr, props = find_peaks(spectrum, prominence=0.0)
            if len(peak_idx_arr) > 0:
                best = np.argmax(props["prominences"])
                peak_idx = peak_idx_arr[best]
                peak_wavelength = float(wavelengths[peak_idx])
                widths_samples, _, _, _ = peak_widths(spectrum, [peak_idx], rel_height=0.5)
                local_spacing = abs(wavelengths[peak_idx] - wavelengths[peak_idx - 1]) if peak_idx > 0 else abs(wavelengths[1] - wavelengths[0])
                peak_width = float(widths_samples[0] * local_spacing)
            else:
                peak_wavelength = float(wavelengths[np.argmax(spectrum)])

        # Autocorrelation
        _, autocorr = analyze_image_autocorrelation(data)
        corr = autocorr["radial_autocorrelation"].to_numpy()
        distance = autocorr["bin_centers"].to_numpy()

        zero_idx = np.where(corr <= 0)[0]
        zero_crossing = (
            float(distance[zero_idx[0]])
            if len(zero_idx) > 0
            else np.nan
        )

        decay_idx = np.where(corr <= 1 / np.e)[0]
        corr_length = (
            float(distance[decay_idx[0]])
            if len(decay_idx) > 0
            else np.nan
        )

    # Otsu-dependent component sizing excluded for now — see note above.
    # if len(component_sizes) > 0:
    #     mean_component_size = float(component_sizes.mean())
    #     max_component_size = int(component_sizes.max())
    #     size_cv = (
    #         float(component_sizes.std() / component_sizes.mean())
    #         if component_sizes.mean() > 0
    #         else np.nan
    #     )
    #     mean_diameter = float(diameters.mean())
    # else:
    #     mean_component_size = 0.0
    #     max_component_size = 0
    #     size_cv = np.nan
    #     mean_diameter = 0.0
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
        # "threshold": threshold_val,
        # "activated_hexes": int(active_mask.sum()),
        # "activated_fraction": float(active_mask.mean()),
        # "n_components": int(len(components)),
        # "mean_component_size": mean_component_size,
        # "max_component_size": max_component_size,
        # "component_size_cv": size_cv,
        # "mean_diameter_hexes": mean_diameter,
        "peak_fft_wavelength": peak_wavelength,
        "peak_fft_width": peak_width,
        "peak_fft_sharpness": peak_sharpness,
        "autocorr_zero_crossing": zero_crossing,
        "autocorr_length_1e": corr_length,
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
):
    data, field_name = load_field(npz, field)

    threshold_used, raw_mask, filtered_mask, components = threshold_data(
        data,
        threshold=threshold,
        layout=layout,
        min_size=min_size,
    )

    data_max = (
        max(1e-12, float(np.nanmax(data)))
        if np.any(np.isfinite(data))
        else 1.0
    )

    data_norm = Normalize(vmin=0.0, vmax=data_max)
    mask_norm = Normalize(vmin=0.0, vmax=1.0)

    if "sim_number" not in metadata.columns:
            if "run" in metadata.columns:
                metadata["sim_number"] = metadata["run"]
            elif "file" in metadata.columns:
                metadata["sim_number"] = _get_sim(metadata["file"])
            else:
                metadata["sim_number"] = np.arange(len(metadata))

    metadata["sim_number"] = pd.to_numeric(metadata["sim_number"], errors="coerce").astype("Int64")
    return summary.merge(metadata, on="sim_number", how="left")

    return summary


# Plot selected metrics across parameters (matplotlib)
def plot_metrics(merged, run_dir, x_var="inh_prod_rate", col_var="inh_diffusion", hue_var="act_hill_coeff"):
    # Add more as needed to visrualize
    metrics = [
        "peak_fft_sharpness",
        "peak_fft_width",
        "autocorr_zero_crossing",
        "autocorr_length_1e",
    ]

    metrics = [m for m in metrics if m in merged.columns]
    required = [x_var, col_var, hue_var]
    missing = [c for c in required if c not in merged.columns]
    if not metrics or missing:
        print(f"Skipping plot. Missing columns: {missing}")
        return None

    # Work from copy for formatting
    merged = merged.copy()
    for col in required:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    col_values = sorted(merged[col_var].dropna().unique())
    hue_values = sorted(merged[hue_var].dropna().unique())
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

    threshold_label = (
        "auto"
        if threshold == "auto"
        else f"{float(threshold):g}".replace(".", "p")
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
    threshold = "auto"
    layout = "even-r"
    min_size = 2

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
            threshold=threshold,
            layout=layout,
            min_size=min_size,
        )
        for p in npz_files
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


# Plot Dynamic Threshold in Bar Graph
def plot_threshold_sensitivity():
    import matplotlib.pyplot as plt
    import pandas as pd

        # 1. Plug in the  data from runs
    data = {
        "Variant": ["Otsu (auto)", "Max Fraction 0.5", "Max Fraction 0.7"],
        "field_0001 (Sparse/Spots)": [16, 16, 17],
        "field_0002 (High-Density Spots)": [83, 83, 85],
        "field_0003 (Mixed Regime)": [13, 13, 13],
        "field_0004 (Labyrinth)": [2, 2, 2],
        }

        # 2. Convert to a DataFrame
    df = pd.DataFrame(data).set_index("Variant")

        # 3. Create the clustered bar plot to visually check for the plateau
    ax = df.plot(kind="bar", width=0.8, figsize=(10, 6))

        # 4. Label plot
    plt.title("Component Count Sensitivity Across Dynamic Threshold Variants", fontsize=14, pad=15)
    plt.ylabel("Connected Component Count", fontsize=12)
    plt.xlabel("Dynamic Threshold Choice", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Simulation Fields", loc="upper right")
    plt.tight_layout()

    # 5. Save the image 
    plt.savefig("threshold_plateau_plot.png", dpi=300)
    print("Diagram successfully generated and saved as 'threshold_plateau_plot.png'!")
    main()
