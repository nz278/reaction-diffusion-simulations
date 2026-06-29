"""
Compare saved fields from multiple .npz simulation outputs side by side.

Built for Thread B validation: drop in any set of saved fields (e.g. the
five Fig 1C reference cases) and render them on the same hex grid, same
color scale, side by side, so visual regime identity can be checked
against the threshold-free metrics in summarize_metrics.py.

Uses the same hex-grid rendering as visualize_2D.py (_add_hex_field,
Greens colormap) rather than a plain imshow grid, so the output is
directly comparable to the rest of the team's figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from fft_analysis_2D import load_field
from visualize_2D import _add_hex_field


def compare_final_fields(
    npz_paths: Sequence[Path],
    labels: Sequence[str] | None = None,
    field: str = "A_final",
    cmap: str = "Greens",
    hex_radius: float = 1.0,
    outfile_png: Path = Path("field_comparison.png"),
) -> Path:
    """
    Render the chosen field from several .npz files side by side on a
    shared color scale.

    Parameters
    ----------
    npz_paths
        Paths to the saved .npz files to compare.
    labels
        Optional per-panel titles. Defaults to each file's name.
    field
        Which saved array to load from each file (passed to load_field).
    outfile_png
        Where to save the comparison figure.
    """
    npz_paths = [Path(p) for p in npz_paths]
    if labels is None:
        labels = [p.name for p in npz_paths]

    loaded = [load_field(p, field) for p in npz_paths]
    arrays = [data for data, _ in loaded]

    vmax = max(1e-12, max(float(a.max()) for a in arrays))
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        1, len(arrays), figsize=(4 * len(arrays), 4.2), constrained_layout=True
    )
    if len(arrays) == 1:
        axes = [axes]

    pc = None
    for ax, data, label in zip(axes, arrays, labels):
        pc = _add_hex_field(ax, data, cmap=cmap, norm=norm, hex_radius=hex_radius)
        ax.set_title(label, fontsize=10)

    fig.colorbar(pc, ax=list(axes), fraction=0.025, pad=0.02, label=field)

    outfile_png = Path(outfile_png)
    fig.savefig(outfile_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile_png.resolve()}")
    return outfile_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare A_final fields from multiple .npz files side by side."
    )
    parser.add_argument(
        "npz_files",
        type=Path,
        nargs="+",
        help="Paths to the .npz files to compare",
    )
    parser.add_argument(
        "--field",
        default="A_final",
        help="Which saved array to load from each file (default: A_final)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("field_comparison.png"),
        help="Output image file",
    )
    args = parser.parse_args()

    compare_final_fields(args.npz_files, field=args.field, outfile_png=args.output)
