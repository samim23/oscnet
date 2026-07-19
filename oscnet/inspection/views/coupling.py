"""Coupling matrix / profile heatmaps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..schema import TraceBundle
from ._plotting import maybe_colorbar, save_figure
from .base import ViewContext, ViewResult


class CouplingView:
    name = "coupling"

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        matrices = []
        if bundle.coupling is not None and np.asarray(bundle.coupling).ndim == 2:
            matrices.append(("coupling", np.asarray(bundle.coupling)))
        if (
            bundle.coupling_profile is not None
            and np.asarray(bundle.coupling_profile).ndim == 2
        ):
            matrices.append(("coupling_profile", np.asarray(bundle.coupling_profile)))
        if not matrices:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="no 2D coupling / coupling_profile arrays",
            )

        n = len(matrices)
        fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.5), squeeze=False)
        paths = []
        for ax, (title, matrix) in zip(axes[0], matrices):
            limit = float(np.max(np.abs(matrix))) if matrix.size else 1.0
            limit = max(limit, 1e-8)
            im = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, origin="upper")
            ax.set_title(title)
            ax.set_xlabel("source")
            ax.set_ylabel("target")
            maybe_colorbar(fig, im, ax, label="weight")
        fig.suptitle(f"Coupling · {bundle.family}")
        path = save_figure(fig, context.output_dir / "coupling.png", dpi=context.dpi)
        paths.append(path)
        return ViewResult(name=self.name, paths=paths)
