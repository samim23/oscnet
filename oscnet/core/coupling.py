"""Coupling topology helpers for oscillator networks.

These utilities build fixed spatial profiles that can be combined with learned
coupling weights. They are intentionally small and JAX-native so experiments can
share the same topology semantics across generators, reconstruction fields, and
future multi-scale oscillator models.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import jax.numpy as jnp

Array = jnp.ndarray
CouplingNormalization = Literal["none", "row_sum"]


def oscillator_grid_coordinates(num_oscillators: int) -> Array:
    """Place oscillators on a near-square normalized 2D grid."""

    if num_oscillators < 1:
        raise ValueError("num_oscillators must be positive")
    grid_rows = max(1, int(math.floor(math.sqrt(num_oscillators))))
    grid_cols = max(1, int(math.ceil(num_oscillators / grid_rows)))
    centers_y, centers_x = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, grid_rows),
        jnp.linspace(-1.0, 1.0, grid_cols),
        indexing="ij",
    )
    centers = jnp.stack([centers_y.reshape(-1), centers_x.reshape(-1)], axis=-1)
    return centers[:num_oscillators]


def normalize_coupling_profile(
    profile: Array,
    *,
    mode: CouplingNormalization = "row_sum",
    target_row_sum: float | None = None,
    eps: float = 1e-8,
) -> Array:
    """Normalize a coupling profile without changing its sparsity pattern.

    ``mode="row_sum"`` scales each non-empty row to ``target_row_sum``. For
    oscillator updates that later divide by the source count, use
    ``target_row_sum=source_count`` to keep normalized profiles on the same
    rough gain scale as dense profiles.
    """

    if mode == "none":
        return profile
    if mode != "row_sum":
        raise ValueError("mode must be 'none' or 'row_sum'")
    profile = jnp.asarray(profile, dtype=jnp.float32)
    if profile.ndim != 2:
        raise ValueError("profile must be a matrix")
    if target_row_sum is None:
        target_row_sum = float(profile.shape[1])
    row_sum = jnp.sum(profile, axis=-1, keepdims=True)
    normalized = profile * (float(target_row_sum) / jnp.maximum(row_sum, eps))
    return jnp.where(row_sum > eps, normalized, profile)


def distance_decay_coupling_profile(
    *,
    num_oscillators: int,
    length_scale: float,
    floor: float = 0.0,
    normalization: CouplingNormalization = "none",
    target_row_sum: float | None = None,
) -> Array:
    """Build a Gaussian distance-decay profile on a normalized oscillator grid."""

    coords = oscillator_grid_coordinates(num_oscillators)
    grid_extent = max(2, int(math.ceil(math.sqrt(num_oscillators))))
    if length_scale <= 0.0:
        length_scale = 2.5 / float(grid_extent)
    length_scale = max(float(length_scale), 1e-6)
    floor = float(floor)
    if floor < 0.0 or floor > 1.0:
        raise ValueError("floor must be in [0, 1]")
    squared_distance = jnp.sum(
        (coords[:, None, :] - coords[None, :, :]) ** 2,
        axis=-1,
    )
    profile = jnp.exp(-squared_distance / (2.0 * length_scale**2))
    profile = floor + (1.0 - floor) * profile
    profile = profile * (1.0 - jnp.eye(num_oscillators, dtype=jnp.float32))
    return normalize_coupling_profile(
        profile,
        mode=normalization,
        target_row_sum=target_row_sum,
    )


def local_radius_coupling_profile(
    *,
    num_oscillators: int,
    radius: float,
    normalization: CouplingNormalization = "none",
    target_row_sum: float | None = None,
) -> Array:
    """Build a sparse local-radius profile on a normalized oscillator grid."""

    coords = oscillator_grid_coordinates(num_oscillators)
    grid_extent = max(2, int(math.ceil(math.sqrt(num_oscillators))))
    if radius <= 0.0:
        radius = 2.5 / float(grid_extent)
    radius = max(float(radius), 1e-6)
    squared_distance = jnp.sum(
        (coords[:, None, :] - coords[None, :, :]) ** 2,
        axis=-1,
    )
    profile = (squared_distance <= radius**2).astype(jnp.float32)
    profile = profile * (1.0 - jnp.eye(num_oscillators, dtype=jnp.float32))
    return normalize_coupling_profile(
        profile,
        mode=normalization,
        target_row_sum=target_row_sum,
    )


def dense_coupling_profile(
    num_oscillators: int,
    *,
    normalization: CouplingNormalization = "none",
    target_row_sum: float | None = None,
) -> Array:
    """Build a dense off-diagonal coupling profile."""

    if num_oscillators < 1:
        raise ValueError("num_oscillators must be positive")
    profile = 1.0 - jnp.eye(num_oscillators, dtype=jnp.float32)
    return normalize_coupling_profile(
        profile,
        mode=normalization,
        target_row_sum=target_row_sum,
    )


def hierarchical_coupling_profile(
    *,
    num_oscillators: int,
    inter_block_strength: float = 0.5,
    depth: int = 0,
    normalization: CouplingNormalization = "none",
    target_row_sum: float | None = None,
) -> Array:
    """Build a self-similar (fractal) ultrametric coupling profile.

    Oscillators are placed on the same near-square grid used by the local and
    distance-decay profiles, then the grid is recursively split into quadrants
    for ``depth`` levels. Two sites sharing their finest block are coupled at
    full strength; each coarser level at which they diverge multiplies the
    coupling by ``inter_block_strength``. This yields discrete self-similar
    scales with direct long-range links between distant sites -- the non-local
    structure that local nearest-neighbour coupling lacks -- while remaining far
    sparser in effective energy than a flat dense profile.
    """

    if num_oscillators < 1:
        raise ValueError("num_oscillators must be positive")
    coords = oscillator_grid_coordinates(num_oscillators)
    grid_extent = max(2, int(math.ceil(math.sqrt(num_oscillators))))
    if depth <= 0:
        depth = max(1, int(math.floor(math.log2(grid_extent))) - 1)
    depth = int(depth)
    strength = float(inter_block_strength)
    if strength <= 0.0:
        strength = 0.5
    strength = min(strength, 1.0)
    unit = jnp.clip((coords + 1.0) / 2.0, 0.0, 1.0 - 1e-6)
    shared = jnp.zeros((num_oscillators, num_oscillators), dtype=jnp.float32)
    prefix = jnp.ones((num_oscillators, num_oscillators), dtype=jnp.float32)
    for level in range(1, depth + 1):
        scale = float(2**level)
        block = jnp.floor(unit * scale)
        match = jnp.all(
            block[:, None, :] == block[None, :, :],
            axis=-1,
        ).astype(jnp.float32)
        prefix = prefix * match
        shared = shared + prefix
    weight = strength ** (float(depth) - shared)
    profile = weight * (1.0 - jnp.eye(num_oscillators, dtype=jnp.float32))
    return normalize_coupling_profile(
        profile,
        mode=normalization,
        target_row_sum=target_row_sum,
    )


def rectangular_dense_coupling_profile(
    *,
    num_targets: int,
    num_sources: int,
    normalization: CouplingNormalization = "none",
    target_row_sum: float | None = None,
) -> Array:
    """Build a dense target-source profile."""

    if num_targets < 1 or num_sources < 1:
        raise ValueError("num_targets and num_sources must be positive")
    profile = jnp.ones((num_targets, num_sources), dtype=jnp.float32)
    return normalize_coupling_profile(
        profile,
        mode=normalization,
        target_row_sum=target_row_sum,
    )


def rectangular_distance_decay_coupling_profile(
    *,
    num_targets: int,
    num_sources: int,
    length_scale: float,
    floor: float = 0.0,
    normalization: CouplingNormalization = "none",
    target_row_sum: float | None = None,
) -> Array:
    """Build a Gaussian distance-decay profile from sources to targets."""

    if num_targets < 1 or num_sources < 1:
        raise ValueError("num_targets and num_sources must be positive")
    target_coords = oscillator_grid_coordinates(num_targets)
    source_coords = oscillator_grid_coordinates(num_sources)
    grid_extent = max(
        2,
        int(math.ceil(math.sqrt(max(num_targets, num_sources)))),
    )
    if length_scale <= 0.0:
        length_scale = 2.5 / float(grid_extent)
    length_scale = max(float(length_scale), 1e-6)
    floor = float(floor)
    if floor < 0.0 or floor > 1.0:
        raise ValueError("floor must be in [0, 1]")
    squared_distance = jnp.sum(
        (target_coords[:, None, :] - source_coords[None, :, :]) ** 2,
        axis=-1,
    )
    profile = jnp.exp(-squared_distance / (2.0 * length_scale**2))
    profile = floor + (1.0 - floor) * profile
    return normalize_coupling_profile(
        profile,
        mode=normalization,
        target_row_sum=target_row_sum,
    )


def rectangular_local_radius_coupling_profile(
    *,
    num_targets: int,
    num_sources: int,
    radius: float,
    normalization: CouplingNormalization = "none",
    target_row_sum: float | None = None,
) -> Array:
    """Build a sparse local-radius source-to-target profile."""

    if num_targets < 1 or num_sources < 1:
        raise ValueError("num_targets and num_sources must be positive")
    target_coords = oscillator_grid_coordinates(num_targets)
    source_coords = oscillator_grid_coordinates(num_sources)
    grid_extent = max(
        2,
        int(math.ceil(math.sqrt(max(num_targets, num_sources)))),
    )
    if radius <= 0.0:
        radius = 2.5 / float(grid_extent)
    radius = max(float(radius), 1e-6)
    squared_distance = jnp.sum(
        (target_coords[:, None, :] - source_coords[None, :, :]) ** 2,
        axis=-1,
    )
    profile = (squared_distance <= radius**2).astype(jnp.float32)
    return normalize_coupling_profile(
        profile,
        mode=normalization,
        target_row_sum=target_row_sum,
    )


def coupling_profile_from_name(
    *,
    name: str,
    num_oscillators: int,
    length_scale: float = 0.0,
    floor: float = 0.0,
    normalization: CouplingNormalization = "none",
    target_row_sum: float | None = None,
) -> Array:
    """Create a named coupling profile."""

    if name == "dense":
        return dense_coupling_profile(
            num_oscillators,
            normalization=normalization,
            target_row_sum=target_row_sum,
        )
    if name == "distance_decay":
        return distance_decay_coupling_profile(
            num_oscillators=num_oscillators,
            length_scale=length_scale,
            floor=floor,
            normalization=normalization,
            target_row_sum=target_row_sum,
        )
    if name == "local_radius":
        return local_radius_coupling_profile(
            num_oscillators=num_oscillators,
            radius=length_scale,
            normalization=normalization,
            target_row_sum=target_row_sum,
        )
    if name == "fractal":
        return hierarchical_coupling_profile(
            num_oscillators=num_oscillators,
            inter_block_strength=length_scale if length_scale > 0.0 else 0.5,
            normalization=normalization,
            target_row_sum=target_row_sum,
        )
    raise ValueError(
        "coupling profile name must be 'dense', 'distance_decay', "
        "'local_radius', or 'fractal'"
    )


def rectangular_coupling_profile_from_name(
    *,
    name: str,
    num_targets: int,
    num_sources: int,
    length_scale: float = 0.0,
    floor: float = 0.0,
    normalization: CouplingNormalization = "none",
    target_row_sum: float | None = None,
) -> Array:
    """Create a named rectangular source-to-target coupling profile."""

    if name == "dense":
        return rectangular_dense_coupling_profile(
            num_targets=num_targets,
            num_sources=num_sources,
            normalization=normalization,
            target_row_sum=target_row_sum,
        )
    if name == "distance_decay":
        return rectangular_distance_decay_coupling_profile(
            num_targets=num_targets,
            num_sources=num_sources,
            length_scale=length_scale,
            floor=floor,
            normalization=normalization,
            target_row_sum=target_row_sum,
        )
    if name == "local_radius":
        return rectangular_local_radius_coupling_profile(
            num_targets=num_targets,
            num_sources=num_sources,
            radius=length_scale,
            normalization=normalization,
            target_row_sum=target_row_sum,
        )
    raise ValueError(
        "coupling profile name must be 'dense', 'distance_decay', or "
        "'local_radius'"
    )


def row_laplacian(profile: Array) -> Tuple[Array, Array]:
    """Return graph Laplacian and row degree for a coupling profile."""

    profile = jnp.asarray(profile, dtype=jnp.float32)
    if profile.ndim != 2 or profile.shape[0] != profile.shape[1]:
        raise ValueError("profile must be a square matrix")
    degree = jnp.sum(profile, axis=-1)
    laplacian = jnp.diag(degree) - profile
    return laplacian, degree


__all__ = [
    "CouplingNormalization",
    "coupling_profile_from_name",
    "dense_coupling_profile",
    "distance_decay_coupling_profile",
    "hierarchical_coupling_profile",
    "local_radius_coupling_profile",
    "normalize_coupling_profile",
    "oscillator_grid_coordinates",
    "rectangular_coupling_profile_from_name",
    "rectangular_dense_coupling_profile",
    "rectangular_distance_decay_coupling_profile",
    "rectangular_local_radius_coupling_profile",
    "row_laplacian",
]
