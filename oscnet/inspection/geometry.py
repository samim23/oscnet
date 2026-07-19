"""Spatial reshape helpers for oscillator site axes."""

from __future__ import annotations

from math import ceil, floor, isqrt, sqrt
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .schema import GridShape

Array = np.ndarray


def infer_grid_shape(
    n_sites: int,
    *,
    hint: Optional[GridShape] = None,
) -> Optional[GridShape]:
    """Infer an H×W layout for a flat site count.

    Prefer an explicit hint when its product matches. Otherwise accept exact
    squares. Non-square factorizations are left unresolved so views can fall
    back to 1D encodings rather than inventing a misleading aspect ratio.
    """

    if n_sites <= 0:
        return None
    if hint is not None:
        height, width = hint
        if height * width == n_sites:
            return (int(height), int(width))
        raise ValueError(
            f"grid_shape {hint} does not match n_sites={n_sites}"
        )
    root = isqrt(n_sites)
    if root * root == n_sites:
        return (root, root)
    return None


def oscillator_pack_shape(n_sites: int) -> GridShape:
    """Near-square pack used by ``oscnet.core.coupling.oscillator_grid_coordinates``.

    May contain unused trailing slots when ``rows * cols > n_sites``.
    """

    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    rows = max(1, int(floor(sqrt(n_sites))))
    cols = max(1, int(ceil(n_sites / rows)))
    return (rows, cols)


def oscillator_site_positions(n_sites: int) -> List[Tuple[float, float]]:
    """Return ``(x, y)`` site positions matching the model coupling grid.

    Coordinates lie in ``[-1, 1]²``, same packing as
    :func:`oscnet.core.coupling.oscillator_grid_coordinates`.
    """

    if n_sites <= 0:
        return []
    rows, cols = oscillator_pack_shape(n_sites)
    ys = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    xs = np.linspace(-1.0, 1.0, cols, dtype=np.float64)
    # Row-major over the pack, truncated to n_sites (matches JAX meshgrid ij).
    positions: List[Tuple[float, float]] = []
    for r in range(rows):
        for c in range(cols):
            if len(positions) >= n_sites:
                return positions
            positions.append((float(xs[c]), float(ys[r])))
    return positions


def as_spatial_field(
    values: Array,
    *,
    grid_shape: Optional[GridShape] = None,
    site_axis: int = -1,
) -> Tuple[Array, Optional[GridShape]]:
    """Reshape a flat site axis into ``(..., H, W)`` when possible.

    Returns the (possibly unchanged) array and the grid used, or ``None`` when
    the site axis cannot be reshaped.
    """

    values = np.asarray(values)
    n_sites = values.shape[site_axis]
    shape = infer_grid_shape(n_sites, hint=grid_shape)
    if shape is None:
        return values, None
    height, width = shape
    if site_axis < 0:
        site_axis = values.ndim + site_axis
    new_shape = (
        values.shape[:site_axis]
        + (height, width)
        + values.shape[site_axis + 1 :]
    )
    return values.reshape(new_shape), shape


def select_batch(values: Array, batch_index: int, *, batch_axis: int = 1) -> Array:
    """Pick one batch element from a time-major trajectory ``(T, B, ...)``."""

    values = np.asarray(values)
    if values.ndim <= batch_axis:
        return values
    n_batch = values.shape[batch_axis]
    if n_batch == 0:
        raise ValueError("cannot select batch from empty batch axis")
    index = int(batch_index) % n_batch
    return np.take(values, index, axis=batch_axis)


def trajectory_keyframes(
    n_steps: int,
    *,
    max_frames: int = 8,
) -> Sequence[int]:
    """Evenly spaced frame indices including first and last when possible."""

    if n_steps <= 0:
        return ()
    if n_steps <= max_frames:
        return tuple(range(n_steps))
    return tuple(
        int(round(i * (n_steps - 1) / (max_frames - 1))) for i in range(max_frames)
    )


def reduce_channels(values: Array, *, channel_axis: int = -1) -> Array:
    """Collapse a trailing channel axis by circular mean for phase-like data."""

    values = np.asarray(values)
    if values.ndim == 0 or values.shape[channel_axis] == 1:
        return np.squeeze(values, axis=channel_axis) if values.ndim else values
    # For multi-channel phase fields, show the first channel by default.
    # Callers that need circular means can opt in later without guessing semantics.
    return np.take(values, 0, axis=channel_axis)
