from __future__ import annotations

from datetime import datetime
from pathlib import Path
import yaml

import numpy as np
import matplotlib.pyplot as plt
import string


# -----------------------------
# VISUALIZATION
# -----------------------------
def visualize(
    charts: list[np.ndarray],
    titles: list[str] | None = None,
    suptitle: str | None = None,
    cbar_label: str | None = None,
    disp_range: tuple[float, float] | None = None,
    max_cols: int = 3,
    save_dir: str | Path | None = None,
    cmap: str = "viridis",
) -> None:
    """
    Plot multiple 2D arrays with shared color scale and optional saving.
    """

    if not charts:
        raise ValueError("`charts` must contain at least one array.")

    n_charts = len(charts)
    n_cols = min(max_cols, n_charts)
    n_rows = int(np.ceil(n_charts / n_cols))

    vmin, vmax = disp_range if disp_range else (
        min(np.nanmin(c) for c in charts),
        max(np.nanmax(c) for c in charts),
    )

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(right=0.88)

    annotations = string.ascii_lowercase
    last_im = None

    for idx, chart in enumerate(charts):
        row, col = divmod(idx, n_cols)
        ax = axs[row, col]

        last_im = ax.imshow(chart, vmin=vmin, vmax=vmax, cmap=cmap)

        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=14)

        ax.text(
            0.02,
            0.98,
            f"({annotations[idx]})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color="white",
            bbox=dict(facecolor="black", alpha=0.3, edgecolor="none"),
        )

        ax.set_xticks([])
        ax.set_yticks([])

    # turn off unused axes
    for idx in range(n_charts, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axs[row, col].axis("off")

    # colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=12)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    # save or show
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        safe_name = suptitle.replace(" ", "_") if suptitle else "figure"
        fig.savefig(save_dir / f"{safe_name}.png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# -----------------------------
# NUMERICAL UTILITIES
# -----------------------------
def remove_padding(img: np.ndarray, pad: float | int) -> np.ndarray:
    """Remove symmetric padding from a 2D field using a single pad value."""

    # ---- normalize pad ----
    pad = int(round(pad))

    # ---- validation ----
    if pad < 0:
        raise ValueError(f"Padding must be >= 0, got {pad}")

    # ---- fast path ----
    if pad == 0:
        return img

    # ---- symmetric crop ----
    return img[
        pad : img.shape[0] - pad,
        pad : img.shape[1] - pad
    ]


def generate_saving_dir() -> Path:
    """Create timestamped results directory."""
    path = Path.cwd() / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=False)
    print(f"Results will be saved to: {path}")
    return path


def save_result(path: str | Path, filename: str, **matrices: np.ndarray) -> Path:
    """Save multiple arrays into .npz file."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    full_path = path / f"{filename}.npz"
    np.savez(full_path, **matrices)

    print(f"Saved {len(matrices)} matrices to {full_path}")
    return full_path


# -----------------------------
# METADATA
# -----------------------------
def save_metadata(
    opt: dict,
    general: dict,
    output_dir: str | Path,
    filename: str = "metadata.yaml",
) -> Path:
    """
    Save experiment metadata to YAML.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    data = {
        "opt": make_serializable(opt),
        "general": make_serializable(general),
    }

    path = output_dir / filename

    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return path