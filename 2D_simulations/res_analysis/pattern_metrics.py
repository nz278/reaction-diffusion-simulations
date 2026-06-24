"""
Thread B — metrics on a single saved pattern.

Per Dr. Morsut's scope: takes one saved .npz field, returns the metrics
that don't need a threshold (so they're already comparable across
patterns). No simulation inside this file — pure analysis on a field
that already exists.

Deliberately does NOT include activated_fraction / component-size /
spot-count metrics — those need an on/off cutoff (currently Otsu, which
gives every pattern a different ruler and reads a fully-on field as
"50% on"). Holding off on those until the group decides on a threshold,
per Dr. Morsut's note.

Built directly on top of Ben's existing fft_analysis_2D.py and
radial_autocor_npz.py rather than reimplementing anything — this file
should sit next to summarize_metrics.py in 2D_simulations/res_analysis/.
"""
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths

from fft_analysis_2D import load_field, radial_power_spectrum
from radial_autocor_npz import analyze_image_autocorrelation


def pattern_metrics(npz_path, field="A_final"):
    """
    Compute the threshold-free metrics for one saved pattern.

    Returns a dict with both the scalar metrics and the raw curves
    (wavelength/power, distance/autocorrelation), so the curves can be
    plotted right next to the numbers — the curve is what shows what
    the number actually means, the scalar alone doesn't.
    """
    npz_path = Path(npz_path)
    data, field_name = load_field(npz_path, field)

    # ── Structure factor (radially-averaged 2D FFT) ────────────────────
    wavelength, power = radial_power_spectrum(data)

    peak_wavelength = np.nan
    peak_width = np.nan
    peak_prominence = np.nan

    if len(power) > 0 and np.any(np.isfinite(power)):
        peak_idx_arr, props = find_peaks(power, prominence=0.0)
        if len(peak_idx_arr) > 0:
            # Take the most prominent peak, not just the first one found
            best = np.argmax(props["prominences"])
            peak_idx = peak_idx_arr[best]
            peak_prominence = float(props["prominences"][best])

            widths_samples, _, _, _ = peak_widths(power, [peak_idx], rel_height=0.5)
            # Convert width from sample-index units to wavelength units
            # (wavelength spacing is uneven, so use local spacing at the peak)
            if peak_idx > 0:
                local_spacing = abs(wavelength[peak_idx] - wavelength[peak_idx - 1])
            else:
                local_spacing = abs(wavelength[1] - wavelength[0])
            peak_width = float(widths_samples[0] * local_spacing)

            peak_wavelength = float(wavelength[peak_idx])
        else:
            # Fallback: no clean peak found by find_peaks (e.g. ON/OFF fields
            # with essentially flat spectra) — just report the argmax position
            peak_wavelength = float(wavelength[np.argmax(power)])

    # ── Autocorrelation ─────────────────────────────────────────────────
    _, autocorr_df = analyze_image_autocorrelation(data)
    distance = autocorr_df["bin_centers"].to_numpy()
    corr = autocorr_df["radial_autocorrelation"].to_numpy()

    zero_idx = np.where(corr <= 0)[0]
    characteristic_length = float(distance[zero_idx[0]]) if len(zero_idx) > 0 else np.nan

    decay_idx = np.where(corr <= 1 / np.e)[0]
    coherence_length = float(distance[decay_idx[0]]) if len(decay_idx) > 0 else np.nan

    return {
        "file": npz_path.name,
        "field": field_name,
        # structure factor
        "peak_wavelength": peak_wavelength,
        "peak_width": peak_width,
        "peak_prominence": peak_prominence,
        # autocorrelation
        "characteristic_length": characteristic_length,
        "coherence_length": coherence_length,
        # raw curves, for plotting next to the numbers
        "_fft_wavelength": wavelength,
        "_fft_power": power,
        "_autocorr_distance": distance,
        "_autocorr_values": corr,
        "_field_data": data,
    }


def plot_thread_b_summary(patterns, outfile="thread_b_summary.png"):
    """
    Build the slide-ready comparison figure: one row per pattern,
    columns = [field image, FFT structure-factor curve, autocorrelation
    curve], with the metric values annotated on each curve.

    patterns: list of (label, npz_path) tuples, e.g.
        [("ON", "pattern_bi1_ON.npz"), ("Turing", "pattern_bi5_regspots_turing.npz"), ...]
    """
    import matplotlib.pyplot as plt

    n = len(patterns)
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.6 * n), squeeze=False)

    for row, (label, npz_path) in enumerate(patterns):
        m = pattern_metrics(npz_path)

        # Column 1: the field itself
        ax = axes[row, 0]
        ax.imshow(m["_field_data"], cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        if row == 0:
            ax.set_title("Pattern", fontsize=9)

        # Column 2: structure factor curve
        ax = axes[row, 1]
        ax.plot(m["_fft_wavelength"], m["_fft_power"], color="steelblue")
        if np.isfinite(m["peak_wavelength"]):
            ax.axvline(m["peak_wavelength"], color="firebrick", linestyle="--", linewidth=1)
        ax.set_xlabel("wavelength (cells)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.text(
            0.97, 0.92,
            f"peak λ={m['peak_wavelength']:.2f}\nwidth={m['peak_width']:.2f}\nprom={m['peak_prominence']:.2g}",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
        )
        if row == 0:
            ax.set_title("Structure factor", fontsize=9)

        # Column 3: autocorrelation curve
        ax = axes[row, 2]
        ax.plot(m["_autocorr_distance"], m["_autocorr_values"], color="darkorange")
        ax.axhline(0, color="0.7", linewidth=0.8)
        if np.isfinite(m["characteristic_length"]):
            ax.axvline(m["characteristic_length"], color="firebrick", linestyle="--", linewidth=1)
        ax.set_xlabel("distance (cells)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.text(
            0.97, 0.92,
            f"char. len={m['characteristic_length']:.2f}\ncoh. len={m['coherence_length']:.2f}",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
        )
        if row == 0:
            ax.set_title("Autocorrelation", fontsize=9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    # Quick smoke test against whatever sample patterns are sitting nearby —
    # update these paths to point at your actual saved patterns.
    patterns = [
        ("ON", "pattern_bi1_ON.npz"),
        ("Turing (stripes)", "pattern_bi3_stripes_turing.npz"),
        ("Turing (spots)", "pattern_bi5_regspots_turing.npz"),
        ("Irregular", "pattern_bi12_irregular.npz"),
        ("OFF", "pattern_bi14_OFF.npz"),
    ]
    for label, p in patterns:
        m = pattern_metrics(p)
        print(f"{label:20s} peak_λ={m['peak_wavelength']:.2f}  width={m['peak_width']:.2f}  "
              f"prom={m['peak_prominence']:.3g}  char_len={m['characteristic_length']:.2f}  "
              f"coh_len={m['coherence_length']:.2f}")

    plot_thread_b_summary(patterns)
