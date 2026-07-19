"""Shared matplotlib helpers for inspection views."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray


def phase_cmap():
    return plt.get_cmap("twilight")


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 120) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def imshow_phase(ax, field: Array, *, title: str = "") -> None:
    field = np.asarray(field, dtype=np.float64)
    im = ax.imshow(
        np.mod(field, 2.0 * np.pi),
        cmap=phase_cmap(),
        vmin=0.0,
        vmax=2.0 * np.pi,
        origin="upper",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def imshow_signed(ax, field: Array, *, title: str = "", cmap: str = "coolwarm"):
    field = np.asarray(field, dtype=np.float64)
    limit = float(np.max(np.abs(field))) if field.size else 1.0
    limit = max(limit, 1e-8)
    im = ax.imshow(
        field,
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        origin="upper",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def maybe_colorbar(fig, im, ax, *, label: Optional[str] = None) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cbar.set_label(label)
