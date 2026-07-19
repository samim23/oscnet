"""Global synchrony order R(t) from phase trajectories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from oscnet.analysis.phase_synchrony import phase_order_parameter

from ..schema import TraceBundle
from ._plotting import save_figure
from .base import ViewContext, ViewResult


class SynchronyView:
    name = "synchrony"

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        if bundle.phase_trajectory is None:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="no phase_trajectory",
            )
        order = _order_series(bundle.phase_trajectory)
        if order is None or order.size == 0:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="could not compute order parameter series",
            )

        # Mean over batch, plus selected batch.
        mean_r = np.mean(order, axis=1) if order.ndim == 2 else order
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(mean_r, linewidth=2.0, label="batch mean R(t)")
        if order.ndim == 2 and order.shape[1] > 0:
            index = int(context.batch_index) % order.shape[1]
            ax.plot(order[:, index], linewidth=1.5, alpha=0.85, label=f"batch[{index}]")
        ax.set_xlabel("settle step")
        ax.set_ylabel("order parameter R")
        ax.set_ylim(0.0, 1.05)
        ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"Synchrony · {bundle.family}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        path = save_figure(fig, context.output_dir / "synchrony.png", dpi=context.dpi)

        if bundle.energies is not None:
            energies = np.asarray(bundle.energies)
            fig_e, ax_e = plt.subplots(figsize=(6.5, 3.5))
            if energies.ndim == 2:
                ax_e.plot(np.mean(energies, axis=1), linewidth=2.0, label="batch mean")
                index = int(context.batch_index) % energies.shape[1]
                ax_e.plot(energies[:, index], alpha=0.85, label=f"batch[{index}]")
            else:
                ax_e.plot(energies.reshape(-1), linewidth=2.0)
            ax_e.set_xlabel("settle step")
            ax_e.set_ylabel("energy")
            ax_e.set_title(f"Energy · {bundle.family}")
            ax_e.grid(True, alpha=0.3)
            ax_e.legend(frameon=False)
            energy_path = save_figure(
                fig_e, context.output_dir / "energy.png", dpi=context.dpi
            )
            return ViewResult(name=self.name, paths=[path, energy_path])

        return ViewResult(name=self.name, paths=[path])


def _order_series(phase_trajectory) -> np.ndarray | None:
    theta = np.asarray(phase_trajectory)
    if theta.ndim == 3:
        # (T, B, sites)
        return phase_order_parameter(theta, axis=-1)
    if theta.ndim == 4:
        # (T, B, sites, channels)
        return phase_order_parameter(theta, axis=(-2, -1))
    if theta.ndim == 5:
        # (T, B, H, W, C)
        return phase_order_parameter(theta, axis=(-3, -2, -1))
    return None
