"""Natural frequency / drive histogram and spatial map."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..geometry import as_spatial_field, reduce_channels
from ..schema import TraceBundle
from ._plotting import maybe_colorbar, save_figure
from .base import ViewContext, ViewResult


class OmegaView:
    name = "omega"

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        if bundle.omega is None:
            return ViewResult(name=self.name, skipped=True, reason="no omega")
        omega = np.asarray(bundle.omega)
        if omega.size == 0:
            return ViewResult(name=self.name, skipped=True, reason="empty omega")

        # Squeeze batch if present: (B, sites, ...) → mean over batch
        if omega.ndim >= 2 and omega.shape[0] <= 64 and omega.shape[0] != omega.shape[-1]:
            # Heuristic: leading axis looks like batch for Winfree (B, P, C)
            values = np.mean(omega, axis=0)
        else:
            values = omega
        if values.ndim >= 2:
            values = reduce_channels(values, channel_axis=-1)

        fig_cols = 2 if values.ndim == 1 else 1
        fig, axes = plt.subplots(1, fig_cols, figsize=(5.5 * fig_cols, 3.8), squeeze=False)
        ax_hist = axes[0, 0]
        flat = np.asarray(values).reshape(-1)
        ax_hist.hist(flat, bins=min(24, max(8, int(np.sqrt(flat.size)))), color="0.35")
        ax_hist.set_xlabel("omega")
        ax_hist.set_ylabel("count")
        ax_hist.set_title("Frequency distribution")
        ax_hist.grid(True, alpha=0.3)

        paths = []
        if values.ndim == 1:
            spatial, shape = as_spatial_field(values, grid_shape=bundle.grid_shape, site_axis=0)
            ax_map = axes[0, 1]
            if shape is not None:
                im = ax_map.imshow(spatial, cmap="viridis", origin="upper", aspect="equal")
                ax_map.set_title(f"omega map {shape}")
                ax_map.set_xticks([])
                ax_map.set_yticks([])
                maybe_colorbar(fig, im, ax_map, label="omega")
            else:
                ax_map.plot(values, linewidth=1.5)
                ax_map.set_xlabel("site")
                ax_map.set_ylabel("omega")
                ax_map.set_title("omega by site")
                ax_map.grid(True, alpha=0.3)
        fig.suptitle(f"Omega · {bundle.family}")
        path = save_figure(fig, context.output_dir / "omega.png", dpi=context.dpi)
        paths.append(path)
        return ViewResult(name=self.name, paths=paths)
