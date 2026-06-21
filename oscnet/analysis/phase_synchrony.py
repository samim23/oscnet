"""Diagnostics for phase synchrony in oscillatory model traces."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

Array = np.ndarray


def circular_difference(a: Array, b: Array) -> Array:
    """Return wrapped phase difference ``a - b`` in ``[-pi, pi]``."""

    delta = np.asarray(a) - np.asarray(b)
    return np.arctan2(np.sin(delta), np.cos(delta))


def phase_order_parameter(theta: Array, axis=None) -> Array:
    """Compute Kuramoto-style order ``|mean(exp(i theta))|``."""

    return np.abs(np.mean(np.exp(1j * np.asarray(theta)), axis=axis))


def _grid(theta: Array, grid_shape: Tuple[int, int]) -> Array:
    height, width = grid_shape
    theta = np.asarray(theta)
    if theta.shape[-2] != height * width:
        raise ValueError("grid_shape product must match the position axis")
    return theta.reshape(*theta.shape[:-2], height, width, theta.shape[-1])


def local_group_order(
    theta: Array,
    *,
    grid_shape: Tuple[int, int],
    group_size: int,
) -> Array:
    """Compute order inside non-overlapping spatial groups.

    Non-divisible grid edges are padded and excluded from the group mean. This
    mirrors grouped Winfree layers that pad spatial grids before pooling group
    influence.
    """

    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    theta_grid = _grid(theta, grid_shape)
    height, width = grid_shape
    group_h = (height + group_size - 1) // group_size
    group_w = (width + group_size - 1) // group_size
    pad_h = group_h * group_size - height
    pad_w = group_w * group_size - width
    pad_width = [(0, 0)] * theta_grid.ndim
    pad_width[-3] = (0, pad_h)
    pad_width[-2] = (0, pad_w)
    padded = np.pad(
        theta_grid,
        pad_width,
        mode="constant",
        constant_values=np.nan,
    )
    mask = ~np.isnan(padded)

    grouped_phase = padded.reshape(
        *padded.shape[:-3],
        group_h,
        group_size,
        group_w,
        group_size,
        padded.shape[-1],
    )
    grouped_mask = mask.reshape(
        *mask.shape[:-3],
        group_h,
        group_size,
        group_w,
        group_size,
        mask.shape[-1],
    )
    grouped_phase = grouped_phase.swapaxes(-4, -3)
    grouped_mask = grouped_mask.swapaxes(-4, -3)

    phasors = np.where(grouped_mask, np.exp(1j * grouped_phase), 0.0)
    counts = np.sum(grouped_mask, axis=(-3, -2, -1))
    counts = np.maximum(counts, 1)
    mean_phasor = np.sum(phasors, axis=(-3, -2, -1)) / counts
    return np.abs(mean_phasor)


def mean_neighbor_phase_difference(
    theta: Array,
    *,
    grid_shape: Tuple[int, int],
) -> float:
    """Mean absolute wrapped phase difference between grid neighbors."""

    theta_grid = _grid(theta, grid_shape)
    diffs = []
    if grid_shape[0] > 1:
        diffs.append(np.abs(circular_difference(theta_grid[..., 1:, :, :], theta_grid[..., :-1, :, :])))
    if grid_shape[1] > 1:
        diffs.append(np.abs(circular_difference(theta_grid[..., :, 1:, :], theta_grid[..., :, :-1, :])))
    if not diffs:
        return 0.0
    return float(np.mean([np.mean(diff) for diff in diffs]))


def trace_phase_summary(
    trace_path: Path | str,
    *,
    grid_shape: Tuple[int, int],
    group_size: int = 2,
    prefix: str = "decoder",
) -> Dict[str, float]:
    """Summarize synchrony and volatility from a saved MNIST Winfree trace."""

    trace_path = Path(trace_path)
    with np.load(trace_path) as trace:
        thetas = np.asarray(trace[f"{prefix}_thetas"])
        final_theta = np.asarray(trace[f"{prefix}_final_theta"])
        energies = np.asarray(trace[f"{prefix}_energies"])

    if thetas.ndim != 4:
        raise ValueError("expected phase trace with shape (steps, batch, positions, channels)")

    global_order = phase_order_parameter(thetas, axis=(-2, -1))
    local_order = local_group_order(
        thetas,
        grid_shape=grid_shape,
        group_size=group_size,
    )
    final_global_order = phase_order_parameter(final_theta, axis=(-2, -1))
    final_local_order = local_group_order(
        final_theta,
        grid_shape=grid_shape,
        group_size=group_size,
    )
    step_delta = circular_difference(thetas[1:], thetas[:-1])
    energy_delta = np.diff(energies, axis=0)

    return {
        "global_order_mean": float(np.mean(global_order)),
        "global_order_std": float(np.std(global_order)),
        "local_order_mean": float(np.mean(local_order)),
        "local_order_std": float(np.std(local_order)),
        "local_minus_global_order": float(np.mean(local_order) - np.mean(global_order)),
        "final_global_order_mean": float(np.mean(final_global_order)),
        "final_local_order_mean": float(np.mean(final_local_order)),
        "final_local_minus_global_order": float(
            np.mean(final_local_order) - np.mean(final_global_order)
        ),
        "neighbor_phase_diff_mean": mean_neighbor_phase_difference(
            final_theta,
            grid_shape=grid_shape,
        ),
        "phase_step_abs_mean": float(np.mean(np.abs(step_delta))),
        "phase_step_abs_std": float(np.std(np.abs(step_delta))),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "energy_delta_abs_mean": float(np.mean(np.abs(energy_delta))),
        "energy_delta_std": float(np.std(energy_delta)),
    }


__all__ = [
    "circular_difference",
    "local_group_order",
    "mean_neighbor_phase_difference",
    "phase_order_parameter",
    "trace_phase_summary",
]
