"""Decoded sample / reconstruction contact sheets."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from ..geometry import infer_grid_shape
from ..schema import TraceBundle
from ._plotting import save_figure
from .base import ViewContext, ViewResult


class ReadoutView:
    name = "readout"

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        if bundle.readout is None:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="no readout / generated / reconstruction array",
            )
        images = _to_image_batch(np.asarray(bundle.readout))
        if images is None:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason=f"unsupported readout shape {tuple(bundle.readout.shape)}",
            )

        n = min(images.shape[0], 16)
        cols = min(4, n)
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
        for index, ax in enumerate(axes.ravel()):
            ax.axis("off")
            if index >= n:
                continue
            ax.imshow(images[index], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"b{index}", fontsize=8)
        fig.suptitle(f"Readout · {bundle.family}")
        path = save_figure(fig, context.output_dir / "readout.png", dpi=context.dpi)
        return ViewResult(name=self.name, paths=[path])


def _to_image_batch(readout: np.ndarray):
    readout = np.asarray(readout)
    if readout.ndim == 4 and readout.shape[-1] in (1, 3):
        # (B, H, W, C)
        if readout.shape[-1] == 1:
            return readout[..., 0]
        # simple luma for RGB
        return (
            0.299 * readout[..., 0]
            + 0.587 * readout[..., 1]
            + 0.114 * readout[..., 2]
        )
    if readout.ndim == 3:
        # (B, H, W) or (T, B, F) reconstruction sequences — take last / first batch view
        if readout.shape[0] < readout.shape[1] and readout.shape[-1] > 32:
            # likely (T, B, F)
            return _flat_to_images(readout[-1])
        return readout
    if readout.ndim == 2:
        return _flat_to_images(readout)
    return None


def _flat_to_images(batch: np.ndarray):
    batch = np.asarray(batch)
    n_pix = batch.shape[-1]
    shape = infer_grid_shape(n_pix)
    if shape is None:
        # common MNIST
        if n_pix == 784:
            shape = (28, 28)
        else:
            return None
    return batch.reshape(batch.shape[0], shape[0], shape[1])
