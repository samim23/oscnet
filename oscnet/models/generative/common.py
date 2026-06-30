"""Implicit image generators driven by oscillator dynamics."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.core.coupling import (
    distance_decay_coupling_profile,
    local_radius_coupling_profile,
    oscillator_grid_coordinates,
)
from oscnet.models.winfree import phase_features, wrap_phase

Array = jnp.ndarray


def _activation(name: str):
    if name == "identity":
        return lambda x: x
    if name == "sigmoid":
        return jax.nn.sigmoid
    if name == "tanh":
        return jnp.tanh
    raise ValueError("output_activation must be 'identity', 'sigmoid', or 'tanh'")


def _image_hw_channels(image_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    """Normalize image shape metadata to height, width, channels."""

    if len(image_shape) == 2:
        height, width = image_shape
        channels = 1
    elif len(image_shape) == 3:
        height, width, channels = image_shape
    else:
        raise ValueError("image_shape must be (height, width) or (height, width, channels)")
    height, width, channels = int(height), int(width), int(channels)
    if height < 1 or width < 1 or channels < 1:
        raise ValueError("image_shape dimensions must be positive")
    return height, width, channels


def _spatial_basis_matrix(
    *,
    num_oscillators: int,
    image_shape: Tuple[int, ...],
    sigma: float,
) -> Array:
    """Build fixed Gaussian pixel bases for spatial phase-field readout."""

    height, width, channels = _image_hw_channels(image_shape)
    if channels != 1:
        raise ValueError("spatial_basis decoder currently supports grayscale images")
    grid_rows = max(1, int(math.floor(math.sqrt(num_oscillators))))
    grid_cols = max(1, int(math.ceil(num_oscillators / grid_rows)))
    pixel_y, pixel_x = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, height),
        jnp.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    centers_y, centers_x = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, grid_rows),
        jnp.linspace(-1.0, 1.0, grid_cols),
        indexing="ij",
    )
    centers = jnp.stack([centers_y.reshape(-1), centers_x.reshape(-1)], axis=-1)
    centers = centers[:num_oscillators]
    pixels = jnp.stack([pixel_y.reshape(-1), pixel_x.reshape(-1)], axis=-1)
    squared_distance = jnp.sum(
        (centers[:, None, :] - pixels[None, :, :]) ** 2,
        axis=-1,
    )
    sigma = float(sigma)
    if sigma <= 0.0:
        sigma = 1.25 / float(max(grid_rows, grid_cols))
    basis = jnp.exp(-squared_distance / (2.0 * sigma**2))
    return basis / jnp.maximum(jnp.max(basis, axis=-1, keepdims=True), 1e-8)


def _local_basis_tensor(
    *,
    num_oscillators: int,
    image_shape: Tuple[int, ...],
    patch_size: int,
    sigma: float,
) -> Array:
    """Build fixed local Gaussian patch bases around oscillator centers."""

    height, width, channels = _image_hw_channels(image_shape)
    if channels != 1:
        raise ValueError("local_basis decoder currently supports grayscale images")
    grid_rows = max(1, int(math.floor(math.sqrt(num_oscillators))))
    grid_cols = max(1, int(math.ceil(num_oscillators / grid_rows)))
    pixel_y, pixel_x = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, height),
        jnp.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    centers_y, centers_x = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, grid_rows),
        jnp.linspace(-1.0, 1.0, grid_cols),
        indexing="ij",
    )
    centers = jnp.stack([centers_y.reshape(-1), centers_x.reshape(-1)], axis=-1)
    centers = centers[:num_oscillators]
    pixels = jnp.stack([pixel_y.reshape(-1), pixel_x.reshape(-1)], axis=-1)

    patch_radius = patch_size // 2
    spacing = 2.0 / float(max(grid_rows, grid_cols, 2) - 1)
    offset_scale = 0.5 * spacing
    offsets_y, offsets_x = jnp.meshgrid(
        jnp.arange(-patch_radius, patch_radius + 1, dtype=jnp.float32),
        jnp.arange(-patch_radius, patch_radius + 1, dtype=jnp.float32),
        indexing="ij",
    )
    offsets = jnp.stack(
        [offsets_y.reshape(-1), offsets_x.reshape(-1)],
        axis=-1,
    ) * offset_scale

    sigma = float(sigma)
    if sigma <= 0.0:
        sigma = 0.4 * spacing
    local_centers = centers[:, None, :] + offsets[None, :, :]
    squared_distance = jnp.sum(
        (local_centers[:, :, None, :] - pixels[None, None, :, :]) ** 2,
        axis=-1,
    )
    basis = jnp.exp(-squared_distance / (2.0 * sigma**2))
    return basis / jnp.maximum(jnp.max(basis, axis=-1, keepdims=True), 1e-8)


def _oscillator_grid_coordinates(num_oscillators: int) -> Array:
    """Place oscillators on a near-square normalized grid."""

    return oscillator_grid_coordinates(num_oscillators)


def _distance_decay_coupling_profile(
    *,
    num_oscillators: int,
    length_scale: float,
    floor: float,
) -> Array:
    """Build a fixed spatial coupling profile for oscillator interactions."""

    return distance_decay_coupling_profile(
        num_oscillators=num_oscillators,
        length_scale=length_scale,
        floor=floor,
    )


def _local_radius_coupling_profile(
    *,
    num_oscillators: int,
    radius: float,
) -> Array:
    """Build a sparse local spatial coupling profile."""

    return local_radius_coupling_profile(
        num_oscillators=num_oscillators,
        radius=radius,
    )


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))
