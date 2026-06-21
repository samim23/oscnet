"""Winfree phase-field model components."""

import math
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .oscillatory import FeedForwardPatchAutoencoder

Array = jnp.ndarray


def wrap_phase(theta: Array) -> Array:
    """Wrap phases to the interval [-pi, pi]."""

    return jnp.atan2(jnp.sin(theta), jnp.cos(theta))


def phase_features(theta: Array) -> Array:
    """Return periodic features [cos(theta), sin(theta)]."""

    theta = wrap_phase(theta)
    return jnp.concatenate([jnp.cos(theta), jnp.sin(theta)], axis=-1)


def _apply_positionwise(linear: eqx.nn.Linear, x: Array) -> Array:
    """Apply a Linear layer to an array with shape (batch, positions, features)."""

    return jax.vmap(jax.vmap(linear))(x)


def _apply_mlp(first: eqx.nn.Linear, second: eqx.nn.Linear, x: Array) -> Array:
    hidden = jax.nn.relu(jax.vmap(jax.vmap(first))(x))
    return jax.vmap(jax.vmap(second))(hidden)


def _apply_local_conv2d(kernel: Array, bias: Array, grid: Array) -> Array:
    """Apply a same-padded local convolution to NHWC grid data."""

    field = jax.lax.conv_general_dilated(
        grid,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return field + bias[None, None, None, :]


def _apply_output_activation(outputs: Array, activation: str) -> Array:
    if activation == "identity":
        return outputs
    if activation == "sigmoid":
        return jax.nn.sigmoid(outputs)
    if activation == "tanh01":
        return 0.5 * (jnp.tanh(outputs) + 1.0)
    raise ValueError("unknown output activation")


def _infer_grid_shape(num_positions: int) -> Tuple[int, int]:
    side = math.isqrt(num_positions)
    if side * side == num_positions:
        return side, side
    return num_positions, 1


def _infer_patch_channels(
    input_dim: Optional[int],
    patch_shape: Tuple[int, int],
) -> Tuple[int, int]:
    patch_pixels = patch_shape[0] * patch_shape[1]
    if input_dim is None:
        return patch_pixels, 1
    if input_dim < patch_pixels or input_dim % patch_pixels != 0:
        raise ValueError(
            f"input_dim must be a positive multiple of patch size ({patch_pixels})"
        )
    return int(input_dim), int(input_dim // patch_pixels)


def _images_to_patch_sequence(
    images: Array,
    image_shape: Tuple[int, int],
    patch_shape: Tuple[int, int],
    channels: int,
) -> Array:
    """Convert flattened channel-first images to time-major patch sequences."""

    height, width = image_shape
    patch_height, patch_width = patch_shape
    batch_size = images.shape[0]
    num_patches = (height // patch_height) * (width // patch_width)
    patch_dim = patch_height * patch_width * channels

    images = images.reshape(batch_size, channels, height, width)
    patches = images.reshape(
        batch_size,
        channels,
        height // patch_height,
        patch_height,
        width // patch_width,
        patch_width,
    )
    patches = patches.transpose(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(batch_size, num_patches, patch_dim)
    return patches.transpose(1, 0, 2)


def _patch_sequence_to_images(
    sequence: Array,
    image_shape: Tuple[int, int],
    patch_shape: Tuple[int, int],
    channels: int,
    *,
    flatten: bool = True,
) -> Array:
    """Convert time-major patch sequences back to flattened channel-first images."""

    height, width = image_shape
    patch_height, patch_width = patch_shape
    batch_size = sequence.shape[1]

    patches = sequence.transpose(1, 0, 2)
    images = patches.reshape(
        batch_size,
        height // patch_height,
        width // patch_width,
        channels,
        patch_height,
        patch_width,
    )
    images = images.transpose(0, 3, 1, 4, 2, 5)
    images = images.reshape(batch_size, channels, height, width)
    if channels == 1:
        images = images[:, 0]
    if flatten:
        return images.reshape(batch_size, channels * height * width)
    return images


def _spatial_decay_mask(
    grid_shape: Tuple[int, int],
    decay_length: Optional[float],
) -> Array:
    """Create a row-normalized Gaussian distance-decay coupling mask."""

    num_positions = grid_shape[0] * grid_shape[1]
    if decay_length is None:
        return jnp.ones((num_positions, num_positions))
    if decay_length <= 0.0:
        raise ValueError("coupling_decay_length must be positive when provided")

    rows = jnp.arange(grid_shape[0], dtype=jnp.float32)
    cols = jnp.arange(grid_shape[1], dtype=jnp.float32)
    rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
    coords = jnp.stack([rr, cc], axis=-1).reshape(num_positions, 2)
    distances = jnp.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    sigma = float(decay_length)
    mask = jnp.exp(-(distances**2) / (2.0 * sigma**2))
    row_rms = jnp.sqrt(jnp.mean(mask**2, axis=1, keepdims=True))
    return mask / (row_rms + 1e-8)


def _grid_transfer_weights(
    source_grid_shape: Tuple[int, int],
    target_grid_shape: Tuple[int, int],
) -> Tuple[Tuple[float, ...], ...]:
    """Create row-normalized Gaussian weights between two spatial grids."""

    source_h, source_w = source_grid_shape
    target_h, target_w = target_grid_shape
    if min(source_h, source_w, target_h, target_w) < 1:
        raise ValueError("grid dimensions must be positive")

    source_rows = (np.arange(source_h, dtype=np.float32) + 0.5) / float(source_h)
    source_cols = (np.arange(source_w, dtype=np.float32) + 0.5) / float(source_w)
    target_rows = (np.arange(target_h, dtype=np.float32) + 0.5) / float(target_h)
    target_cols = (np.arange(target_w, dtype=np.float32) + 0.5) / float(target_w)
    source_rr, source_cc = np.meshgrid(source_rows, source_cols, indexing="ij")
    target_rr, target_cc = np.meshgrid(target_rows, target_cols, indexing="ij")
    source_coords = np.stack([source_rr, source_cc], axis=-1).reshape(-1, 2)
    target_coords = np.stack([target_rr, target_cc], axis=-1).reshape(-1, 2)

    distances = np.linalg.norm(
        target_coords[:, None, :] - source_coords[None, :, :],
        axis=-1,
    )
    coarse_side = min(max(source_grid_shape), max(target_grid_shape))
    sigma = 0.5 / float(max(coarse_side, 1))
    weights = np.exp(-(distances**2) / (2.0 * sigma**2))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    return tuple(tuple(float(value) for value in row) for row in weights)


def _fixed_shuffle_permutation(num_positions: int) -> Tuple[int, ...]:
    """Create a deterministic non-identity permutation for control probes."""

    if num_positions < 1:
        raise ValueError("num_positions must be positive")
    if num_positions == 1:
        return (0,)
    rng = np.random.default_rng(1729 + num_positions)
    permutation = np.arange(num_positions, dtype=np.int32)
    rng.shuffle(permutation)
    if np.all(permutation == np.arange(num_positions, dtype=np.int32)):
        permutation = np.roll(permutation, 1)
    return tuple(int(index) for index in permutation)


def _local_neighbor_mask(
    grid_shape: Tuple[int, int],
    kernel_size: int,
) -> Array:
    """Create a boolean local-neighborhood mask over flattened grid positions."""

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    grid_h, grid_w = grid_shape
    rows = jnp.arange(grid_h)
    cols = jnp.arange(grid_w)
    rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
    coords = jnp.stack([rr, cc], axis=-1).reshape(grid_h * grid_w, 2)
    deltas = jnp.abs(coords[:, None, :] - coords[None, :, :])
    radius = kernel_size // 2
    return jnp.max(deltas, axis=-1) <= radius


def _rotary_2d_phase_grid(
    grid_shape: Tuple[int, int],
    channels: int,
    scale: float = 1.0,
) -> Array:
    """Create deterministic multi-frequency 2D phase seeds for grid positions."""

    if scale <= 0.0:
        raise ValueError("phase_init_scale must be positive")

    grid_h, grid_w = grid_shape
    rows = (jnp.arange(grid_h, dtype=jnp.float32) + 0.5) / float(grid_h)
    cols = (jnp.arange(grid_w, dtype=jnp.float32) + 0.5) / float(grid_w)
    rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
    coords = jnp.stack([rr, cc], axis=-1).reshape(grid_h * grid_w, 2)

    channel_ids = jnp.arange(channels)
    max_frequency = max(1, min(max(grid_h, grid_w), 8))
    frequencies = 1.0 + (channel_ids // 2) % max_frequency
    axes = channel_ids % 2
    selected_coords = jnp.where(axes[None, :] == 0, coords[:, :1], coords[:, 1:])
    phases = scale * 2.0 * jnp.pi * selected_coords * frequencies[None, :]
    return wrap_phase(phases)


class WinfreeFieldLayer(eqx.Module):
    """
    Recurrent Winfree dynamics over a phase field.

    The state is ``theta`` with shape ``(batch, positions, channels)`` and the
    input-conditioned drive is ``omega`` with the same shape. Each step uses the
    WONN-style trig form

    ``dtheta = omega + coupling_strength * cos(theta) * coupling(sin(theta))``.
    """

    sensitivity_in: Optional[eqx.nn.Linear]
    sensitivity_out: Optional[eqx.nn.Linear]
    influence_in: Optional[eqx.nn.Linear]
    influence_out: Optional[eqx.nn.Linear]
    adaptive_query: Optional[eqx.nn.Linear]
    adaptive_key: Optional[eqx.nn.Linear]
    adaptive_value: Optional[eqx.nn.Linear]
    channel_mix: eqx.nn.Linear
    coupling: Array
    conv_kernel: Optional[Array]
    conv_bias: Optional[Array]
    coupling_decay_mask: Array = eqx.field(static=True)
    local_coupling_mask: Array = eqx.field(static=True)

    num_positions: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    grid_shape: Tuple[int, int] = eqx.field(static=True)
    group_grid_shape: Tuple[int, int] = eqx.field(static=True)
    group_size: int = eqx.field(static=True)
    coupling_positions: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)
    coupling_decay_length: Optional[float] = eqx.field(static=True)
    coupling_mode: str = eqx.field(static=True)
    coupling_kernel_size: int = eqx.field(static=True)
    adaptive_coupling_strength: float = eqx.field(static=True)
    field_activation: str = eqx.field(static=True)
    si_func: str = eqx.field(static=True)
    si_hidden_ratio: int = eqx.field(static=True)

    def __init__(
        self,
        num_positions: int,
        channels: int,
        grid_shape: Optional[Tuple[int, int]] = None,
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "matrix",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        field_activation: str = "relu",
        si_func: str = "trig",
        si_hidden_ratio: int = 2,
        initial_coupling: Optional[Array] = None,
        *,
        key: jax.random.PRNGKey,
    ):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        if si_hidden_ratio < 1:
            raise ValueError("si_hidden_ratio must be >= 1")
        if field_activation not in {"identity", "relu", "tanh"}:
            raise ValueError("field_activation must be 'identity', 'relu', or 'tanh'")
        if si_func not in {"trig", "mlp"}:
            raise ValueError("si_func must be 'trig' or 'mlp'")
        if coupling_mode not in {
            "matrix",
            "conv",
            "adaptive",
            "conv_adaptive",
            "conv_matrix",
        }:
            raise ValueError(
                "coupling_mode must be 'matrix', 'conv', 'adaptive', "
                "'conv_adaptive', or 'conv_matrix'"
            )
        if coupling_kernel_size < 1 or coupling_kernel_size % 2 == 0:
            raise ValueError("coupling_kernel_size must be a positive odd integer")
        if adaptive_coupling_strength < 0.0:
            raise ValueError("adaptive_coupling_strength must be non-negative")

        if grid_shape is None:
            grid_shape = _infer_grid_shape(num_positions)
        if grid_shape[0] * grid_shape[1] != num_positions:
            raise ValueError("grid_shape product must match num_positions")

        group_h = (grid_shape[0] + group_size - 1) // group_size
        group_w = (grid_shape[1] + group_size - 1) // group_size
        coupling_positions = group_h * group_w

        keys = jax.random.split(key, 6)
        self.num_positions = int(num_positions)
        self.channels = int(channels)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.group_grid_shape = (int(group_h), int(group_w))
        self.group_size = int(group_size)
        self.coupling_positions = int(coupling_positions)
        self.steps = int(steps)
        self.gamma = float(gamma)
        self.coupling_strength = float(coupling_strength)
        self.coupling_decay_length = (
            None if coupling_decay_length is None else float(coupling_decay_length)
        )
        self.coupling_mode = coupling_mode
        self.coupling_kernel_size = int(coupling_kernel_size)
        self.adaptive_coupling_strength = float(adaptive_coupling_strength)
        self.field_activation = field_activation
        self.si_func = si_func
        self.si_hidden_ratio = int(si_hidden_ratio)

        if si_func == "mlp":
            sensitivity_hidden = max(channels, channels * self.si_hidden_ratio)
            influence_input_dim = group_size * group_size * channels * 2
            influence_hidden = max(channels, influence_input_dim * self.si_hidden_ratio)
            self.sensitivity_in = eqx.nn.Linear(
                2 * channels,
                sensitivity_hidden,
                key=keys[0],
            )
            self.sensitivity_out = eqx.nn.Linear(
                sensitivity_hidden,
                channels,
                key=keys[1],
            )
            self.influence_in = eqx.nn.Linear(
                influence_input_dim,
                influence_hidden,
                key=keys[2],
            )
            self.influence_out = eqx.nn.Linear(
                influence_hidden,
                channels,
                key=keys[3],
            )
        else:
            self.sensitivity_in = None
            self.sensitivity_out = None
            self.influence_in = None
            self.influence_out = None

        self.channel_mix = eqx.nn.Linear(channels, channels, key=keys[4])

        if initial_coupling is None:
            scale = 1.0 / jnp.sqrt(float(coupling_positions))
            self.coupling = jax.random.normal(
                keys[5],
                (coupling_positions, coupling_positions),
            ) * scale
        else:
            coupling = jnp.asarray(initial_coupling)
            expected = (coupling_positions, coupling_positions)
            if coupling.shape != expected:
                raise ValueError(f"initial_coupling must have shape {expected}")
            self.coupling = coupling
        mask = np.asarray(
            _spatial_decay_mask(
                self.group_grid_shape,
                self.coupling_decay_length,
            )
        )
        self.coupling_decay_mask = tuple(
            tuple(float(value) for value in row) for row in mask
        )
        local_mask = np.asarray(
            _local_neighbor_mask(
                self.group_grid_shape,
                self.coupling_kernel_size,
            )
        )
        self.local_coupling_mask = tuple(
            tuple(bool(value) for value in row) for row in local_mask
        )
        if coupling_mode in {"conv", "conv_adaptive", "conv_matrix"}:
            conv_scale = 1.0 / jnp.sqrt(
                float(coupling_kernel_size * coupling_kernel_size * channels)
            )
            conv_key = jax.random.fold_in(key, 31)
            self.conv_kernel = (
                jax.random.normal(
                    conv_key,
                    (
                        coupling_kernel_size,
                        coupling_kernel_size,
                        channels,
                        channels,
                    ),
                )
                * conv_scale
            )
            self.conv_bias = jnp.zeros((channels,))
        else:
            self.conv_kernel = None
            self.conv_bias = None

        if coupling_mode in {"adaptive", "conv_adaptive"}:
            self.adaptive_query = eqx.nn.Linear(
                channels,
                channels,
                key=jax.random.fold_in(key, 41),
            )
            self.adaptive_key = eqx.nn.Linear(
                channels,
                channels,
                key=jax.random.fold_in(key, 42),
            )
            self.adaptive_value = eqx.nn.Linear(
                channels,
                channels,
                key=jax.random.fold_in(key, 43),
            )
        else:
            self.adaptive_query = None
            self.adaptive_key = None
            self.adaptive_value = None

    def _activate_field(self, field: Array) -> Array:
        if self.field_activation == "identity":
            return field
        if self.field_activation == "tanh":
            return jnp.tanh(field)
        return jax.nn.relu(field)

    def _adaptive_coupled_field(self, influence: Array) -> Array:
        if (
            self.adaptive_query is None
            or self.adaptive_key is None
            or self.adaptive_value is None
        ):
            raise RuntimeError("adaptive coupling parameters are missing")

        query = _apply_positionwise(self.adaptive_query, influence)
        key = _apply_positionwise(self.adaptive_key, influence)
        value = _apply_positionwise(self.adaptive_value, influence)
        scale = 1.0 / jnp.sqrt(float(self.channels))
        scores = jnp.einsum("bic,bjc->bij", query, key) * scale
        local_mask = jnp.asarray(self.local_coupling_mask, dtype=bool)
        scores = jnp.where(local_mask[None, :, :], scores, -1.0e9)
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bij,bjc->bic", weights, value)

    def coupled_field(self, influence: Array) -> Array:
        """Map ``I(theta)`` to a coupling field."""

        if self.coupling_mode in {"conv", "conv_adaptive", "conv_matrix"}:
            if self.conv_kernel is None or self.conv_bias is None:
                raise RuntimeError("local convolution coupling parameters are missing")
            group_h, group_w = self.group_grid_shape
            field_grid = influence.reshape(
                influence.shape[0],
                group_h,
                group_w,
                self.channels,
            )
            field_grid = _apply_local_conv2d(
                self.conv_kernel,
                self.conv_bias,
                field_grid,
            )
            field = field_grid.reshape(
                influence.shape[0],
                self.coupling_positions,
                self.channels,
            )
            if self.coupling_mode == "conv_adaptive":
                field = (
                    field
                    + self.adaptive_coupling_strength
                    * self._adaptive_coupled_field(influence)
                )
            if self.coupling_mode == "conv_matrix":
                effective_coupling = self.coupling * jnp.asarray(
                    self.coupling_decay_mask
                )
                matrix_field = jnp.einsum(
                    "ij,bjc->bic",
                    effective_coupling,
                    influence,
                )
                field = field + self.adaptive_coupling_strength * matrix_field
        elif self.coupling_mode == "adaptive":
            field = self._adaptive_coupled_field(influence)
        else:
            effective_coupling = self.coupling * jnp.asarray(self.coupling_decay_mask)
            field = jnp.einsum("ij,bjc->bic", effective_coupling, influence)
        field = _apply_positionwise(self.channel_mix, field)
        return self._activate_field(field)

    def _to_grid(self, x: Array) -> Array:
        height, width = self.grid_shape
        return x.reshape(x.shape[0], height, width, self.channels)

    def _pad_grid(self, x: Array) -> Array:
        height, width = self.grid_shape
        pad_h = self.group_grid_shape[0] * self.group_size - height
        pad_w = self.group_grid_shape[1] * self.group_size - width
        return jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))

    def _groups_from_theta(self, theta: Array, include_cos: bool) -> Array:
        theta_grid = self._pad_grid(self._to_grid(theta))
        sin_theta = jnp.sin(theta_grid)
        if include_cos:
            features = jnp.concatenate([sin_theta, jnp.cos(theta_grid)], axis=-1)
        else:
            features = sin_theta

        batch_size = theta.shape[0]
        group_h, group_w = self.group_grid_shape
        grouped = features.reshape(
            batch_size,
            group_h,
            self.group_size,
            group_w,
            self.group_size,
            features.shape[-1],
        )
        grouped = grouped.transpose(0, 1, 3, 2, 4, 5)
        return grouped.reshape(batch_size, group_h * group_w, -1)

    def _broadcast_groups(self, group_values: Array) -> Array:
        batch_size = group_values.shape[0]
        group_h, group_w = self.group_grid_shape
        height, width = self.grid_shape

        grid = group_values.reshape(batch_size, group_h, group_w, self.channels)
        grid = jnp.repeat(grid, self.group_size, axis=1)
        grid = jnp.repeat(grid, self.group_size, axis=2)
        return grid[:, :height, :width, :].reshape(
            batch_size,
            self.num_positions,
            self.channels,
        )

    def sensitivity(self, theta: Array) -> Array:
        if self.si_func == "trig":
            return jnp.cos(theta)

        if self.sensitivity_in is None or self.sensitivity_out is None:
            raise RuntimeError("learned sensitivity parameters are missing")

        features = phase_features(theta)
        return _apply_mlp(self.sensitivity_in, self.sensitivity_out, features)

    def influence(self, theta: Array) -> Array:
        if self.group_size == 1 and self.si_func == "trig":
            return jnp.sin(theta)

        if self.si_func == "trig":
            grouped = self._groups_from_theta(theta, include_cos=False)
            grouped = grouped.reshape(theta.shape[0], self.coupling_positions, -1, self.channels)
            return jnp.mean(grouped, axis=2)

        if self.influence_in is None or self.influence_out is None:
            raise RuntimeError("learned influence parameters are missing")

        grouped = self._groups_from_theta(theta, include_cos=True)
        return _apply_mlp(self.influence_in, self.influence_out, grouped)

    def winfree_step(self, theta: Array, omega: Array) -> Tuple[Array, Array]:
        theta = wrap_phase(theta)
        sensitivity = self.sensitivity(theta)
        influence = self.influence(theta)
        field_group = self.coupled_field(influence)
        if self.coupling_positions == self.num_positions:
            field = field_group
        else:
            field = self._broadcast_groups(field_group)

        dtheta = omega + self.coupling_strength * sensitivity * field
        theta_new = wrap_phase(theta + self.gamma * dtheta)
        energy = -jnp.sum(influence * field_group, axis=(1, 2))
        return theta_new, energy

    def __call__(
        self,
        theta: Array,
        omega: Array,
        return_trajectory: bool = False,
    ):
        if theta.shape[-2:] != (self.num_positions, self.channels):
            raise ValueError(
                "theta must have shape (batch, num_positions, channels)"
            )
        if omega.shape != theta.shape:
            raise ValueError("omega must have the same shape as theta")

        theta = wrap_phase(theta)

        if return_trajectory:

            def scan_trace(carry, _):
                theta_next, energy = self.winfree_step(carry, omega)
                return theta_next, (theta_next, energy)

            final_theta, (thetas, energies) = jax.lax.scan(
                scan_trace,
                theta,
                None,
                length=self.steps,
            )
            return {
                "final_theta": final_theta,
                "thetas": thetas,
                "energies": energies,
            }

        def scan_final(carry, _):
            theta_next, _ = self.winfree_step(carry, omega)
            return theta_next, None

        final_theta, _ = jax.lax.scan(scan_final, theta, None, length=self.steps)
        return final_theta


class WinfreeFieldEncoder(eqx.Module):
    """Encode a patch sequence through a WONN-style phase field."""

    input_to_omega: eqx.nn.Linear
    to_latent: eqx.nn.Linear
    layer: WinfreeFieldLayer
    initial_theta: Array

    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    sequence_length: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        sequence_length: int,
        grid_shape: Optional[Tuple[int, int]] = None,
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "matrix",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        omega_scale: float = 1.0,
        initial_phase_scale: float = 0.1,
        field_activation: str = "relu",
        si_func: str = "trig",
        si_hidden_ratio: int = 2,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 4)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.sequence_length = int(sequence_length)
        self.omega_scale = float(omega_scale)

        self.input_to_omega = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])
        self.to_latent = eqx.nn.Linear(
            sequence_length * 2 * hidden_dim,
            latent_dim,
            key=keys[1],
        )
        self.layer = WinfreeFieldLayer(
            num_positions=sequence_length,
            channels=hidden_dim,
            grid_shape=grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            key=keys[2],
        )
        self.initial_theta = (
            jax.random.normal(keys[3], (sequence_length, hidden_dim))
            * initial_phase_scale
        )

    def _omega_from_inputs(self, inputs: Array) -> Array:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (time, batch, features)")
        if inputs.shape[0] != self.sequence_length:
            raise ValueError("input sequence length does not match the encoder")

        inputs_btf = inputs.transpose(1, 0, 2)
        omega = _apply_positionwise(self.input_to_omega, inputs_btf)
        return self.omega_scale * jnp.tanh(omega)

    def encode_with_trace(self, inputs: Array) -> Dict[str, Array]:
        omega = self._omega_from_inputs(inputs)
        batch_size = inputs.shape[1]
        theta0 = jnp.broadcast_to(
            self.initial_theta[None, :, :],
            (batch_size, self.sequence_length, self.hidden_dim),
        )
        trajectory = self.layer(theta0, omega, return_trajectory=True)
        features = phase_features(trajectory["final_theta"])
        latent = jax.vmap(self.to_latent)(features.reshape(batch_size, -1))
        return {
            "latent": latent,
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": trajectory["final_theta"],
            "thetas": trajectory["thetas"],
            "energies": trajectory["energies"],
        }

    def __call__(self, inputs: Array) -> Array:
        omega = self._omega_from_inputs(inputs)
        batch_size = inputs.shape[1]
        theta0 = jnp.broadcast_to(
            self.initial_theta[None, :, :],
            (batch_size, self.sequence_length, self.hidden_dim),
        )
        final_theta = self.layer(theta0, omega)
        features = phase_features(final_theta)
        return jax.vmap(self.to_latent)(features.reshape(batch_size, -1))


class WinfreeFieldDecoder(eqx.Module):
    """Decode a latent vector through a learned Winfree phase field."""

    latent_to_omega: eqx.nn.Linear
    latent_to_theta: eqx.nn.Linear
    latent_to_readout: Optional[eqx.nn.Linear]
    latent_to_output_skip: Optional[eqx.nn.Linear]
    theta_to_output: eqx.nn.Linear
    layer: WinfreeFieldLayer
    positional_omega: Array
    positional_theta: Array

    latent_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    sequence_length: int = eqx.field(static=True)
    latent_conditioning_strength: float = eqx.field(static=True)
    latent_readout: str = eqx.field(static=True)
    latent_readout_strength: float = eqx.field(static=True)
    latent_output_skip: str = eqx.field(static=True)
    latent_output_skip_strength: float = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True)
    output_activation: str = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        sequence_length: int,
        grid_shape: Optional[Tuple[int, int]] = None,
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "matrix",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        latent_conditioning_strength: float = 1.0,
        latent_readout: str = "none",
        latent_readout_strength: float = 1.0,
        latent_output_skip: str = "none",
        latent_output_skip_strength: float = 1.0,
        omega_scale: float = 1.0,
        initial_phase_scale: float = 0.1,
        field_activation: str = "relu",
        si_func: str = "trig",
        si_hidden_ratio: int = 2,
        output_activation: str = "identity",
        *,
        key: jax.random.PRNGKey,
    ):
        if output_activation not in {"identity", "sigmoid", "tanh01"}:
            raise ValueError(
                "output_activation must be 'identity', 'sigmoid', or 'tanh01'"
            )
        if latent_readout not in {"none", "phase_bias", "concat"}:
            raise ValueError("latent_readout must be 'none', 'phase_bias', or 'concat'")
        if latent_output_skip not in {"none", "sequence"}:
            raise ValueError("latent_output_skip must be 'none' or 'sequence'")

        keys = jax.random.split(key, 8)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.sequence_length = int(sequence_length)
        self.latent_conditioning_strength = float(latent_conditioning_strength)
        self.latent_readout = latent_readout
        self.latent_readout_strength = float(latent_readout_strength)
        self.latent_output_skip = latent_output_skip
        self.latent_output_skip_strength = float(latent_output_skip_strength)
        self.omega_scale = float(omega_scale)
        self.output_activation = output_activation

        self.latent_to_omega = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[0])
        self.latent_to_theta = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[1])
        readout_dim = 2 * hidden_dim + (latent_dim if latent_readout == "concat" else 0)
        self.theta_to_output = eqx.nn.Linear(readout_dim, output_dim, key=keys[2])
        if latent_readout == "phase_bias":
            self.latent_to_readout = eqx.nn.Linear(
                latent_dim,
                2 * hidden_dim,
                key=keys[6],
            )
        else:
            self.latent_to_readout = None
        if latent_output_skip == "sequence":
            self.latent_to_output_skip = eqx.nn.Linear(
                latent_dim,
                sequence_length * output_dim,
                key=keys[7],
            )
        else:
            self.latent_to_output_skip = None
        self.layer = WinfreeFieldLayer(
            num_positions=sequence_length,
            channels=hidden_dim,
            grid_shape=grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            key=keys[3],
        )
        self.positional_omega = (
            jax.random.normal(keys[4], (sequence_length, hidden_dim)) * 0.02
        )
        self.positional_theta = (
            jax.random.normal(keys[5], (sequence_length, hidden_dim))
            * initial_phase_scale
        )

    def _initial_fields(self, latent: Array) -> Tuple[Array, Array]:
        latent_omega = (
            jnp.tanh(jax.vmap(self.latent_to_omega)(latent))
            * self.latent_conditioning_strength
        )
        latent_theta = (
            jnp.tanh(jax.vmap(self.latent_to_theta)(latent))
            * self.latent_conditioning_strength
        )
        omega = self.omega_scale * jnp.tanh(
            latent_omega[:, None, :] + self.positional_omega[None, :, :]
        )
        theta = wrap_phase(
            latent_theta[:, None, :] + self.positional_theta[None, :, :]
        )
        return theta, omega

    def _readout(self, theta: Array, latent: Optional[Array] = None) -> Array:
        features = phase_features(theta)
        if self.latent_readout == "phase_bias":
            if latent is None:
                raise ValueError("latent is required for phase_bias readout")
            if self.latent_to_readout is None:
                raise RuntimeError("latent readout parameters are missing")
            latent_features = (
                jnp.tanh(jax.vmap(self.latent_to_readout)(latent))
                * self.latent_readout_strength
            )
            features = features + latent_features[:, None, :]
        elif self.latent_readout == "concat":
            if latent is None:
                raise ValueError("latent is required for concat readout")
            latent_features = jnp.tanh(latent) * self.latent_readout_strength
            latent_features = jnp.broadcast_to(
                latent_features[:, None, :],
                (latent.shape[0], self.sequence_length, self.latent_dim),
            )
            features = jnp.concatenate([features, latent_features], axis=-1)

        outputs = _apply_positionwise(self.theta_to_output, features)
        if self.latent_output_skip == "sequence":
            if latent is None:
                raise ValueError("latent is required for sequence output skip")
            if self.latent_to_output_skip is None:
                raise RuntimeError("latent output skip parameters are missing")
            output_skip = jax.vmap(self.latent_to_output_skip)(latent)
            output_skip = output_skip.reshape(
                latent.shape[0],
                self.sequence_length,
                self.output_dim,
            )
            outputs = outputs + self.latent_output_skip_strength * output_skip

        if self.output_activation == "sigmoid":
            return jax.nn.sigmoid(outputs)
        if self.output_activation == "tanh01":
            return 0.5 * (jnp.tanh(outputs) + 1.0)
        return outputs

    def decode_with_trace(self, latent: Array) -> Tuple[Array, Dict[str, Array]]:
        theta0, omega = self._initial_fields(latent)
        trajectory = self.layer(theta0, omega, return_trajectory=True)
        outputs = self._readout(trajectory["final_theta"], latent).transpose(1, 0, 2)
        return outputs, {
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": trajectory["final_theta"],
            "thetas": trajectory["thetas"],
            "energies": trajectory["energies"],
        }

    def __call__(self, latent: Array, sequence_length: Optional[int] = None) -> Array:
        length = sequence_length or self.sequence_length
        if length != self.sequence_length:
            raise ValueError("WinfreeFieldDecoder requires its configured length")

        theta0, omega = self._initial_fields(latent)
        final_theta = self.layer(theta0, omega)
        return self._readout(final_theta, latent).transpose(1, 0, 2)


class WinfreeFieldAutoencoder(eqx.Module):
    """Sequence autoencoder using WONN-style Winfree phase-field dynamics."""

    encoder: WinfreeFieldEncoder
    decoder: WinfreeFieldDecoder

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: Optional[int] = None,
        sequence_length: int = 16,
        grid_shape: Optional[Tuple[int, int]] = None,
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "matrix",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        latent_conditioning_strength: float = 1.0,
        latent_readout: str = "none",
        latent_readout_strength: float = 1.0,
        latent_output_skip: str = "none",
        latent_output_skip_strength: float = 1.0,
        omega_scale: float = 1.0,
        field_activation: str = "relu",
        si_func: str = "trig",
        si_hidden_ratio: int = 2,
        output_activation: str = "identity",
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 2)
        output_dim = input_dim if output_dim is None else output_dim
        self.encoder = WinfreeFieldEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            sequence_length=sequence_length,
            grid_shape=grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            omega_scale=omega_scale,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            key=keys[0],
        )
        self.decoder = WinfreeFieldDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            grid_shape=grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            latent_conditioning_strength=latent_conditioning_strength,
            latent_readout=latent_readout,
            latent_readout_strength=latent_readout_strength,
            latent_output_skip=latent_output_skip,
            latent_output_skip_strength=latent_output_skip_strength,
            omega_scale=omega_scale,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            output_activation=output_activation,
            key=keys[1],
        )

    def encode(self, inputs: Array) -> Array:
        return self.encoder(inputs)

    def decode(self, latent: Array, sequence_length: Optional[int] = None) -> Array:
        return self.decoder(latent, sequence_length=sequence_length)

    def trace(self, inputs: Array) -> Dict[str, Array]:
        encoder_trace = self.encoder.encode_with_trace(inputs)
        reconstruction, decoder_trace = self.decoder.decode_with_trace(
            encoder_trace["latent"]
        )
        return {
            "latent": encoder_trace["latent"],
            "reconstruction_sequence": reconstruction,
            "encoder_omega": encoder_trace["omega"],
            "encoder_initial_theta": encoder_trace["initial_theta"],
            "encoder_final_theta": encoder_trace["final_theta"],
            "encoder_thetas": encoder_trace["thetas"],
            "encoder_energies": encoder_trace["energies"],
            "decoder_omega": decoder_trace["omega"],
            "decoder_initial_theta": decoder_trace["initial_theta"],
            "decoder_final_theta": decoder_trace["final_theta"],
            "decoder_thetas": decoder_trace["thetas"],
            "decoder_energies": decoder_trace["energies"],
        }

    def __call__(self, inputs: Array, return_latent: bool = False):
        latent = self.encode(inputs)
        reconstruction = self.decode(latent, sequence_length=inputs.shape[0])
        if return_latent:
            return reconstruction, latent
        return reconstruction


class WinfreePatchAutoencoder(eqx.Module):
    """Flat-image wrapper around ``WinfreeFieldAutoencoder``."""

    autoencoder: WinfreeFieldAutoencoder
    image_shape: Tuple[int, int] = eqx.field(static=True)
    patch_shape: Tuple[int, int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    patch_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (7, 7),
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "matrix",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        latent_conditioning_strength: float = 1.0,
        latent_readout: str = "none",
        latent_readout_strength: float = 1.0,
        latent_output_skip: str = "none",
        latent_output_skip_strength: float = 1.0,
        omega_scale: float = 1.0,
        field_activation: str = "relu",
        si_func: str = "trig",
        si_hidden_ratio: int = 2,
        output_activation: str = "identity",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)

        height, width = image_shape
        patch_height, patch_width = patch_shape
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError("image_shape must be divisible by patch_shape")

        patch_dim = patch_height * patch_width
        if input_dim is not None and input_dim != patch_dim:
            raise ValueError(f"input_dim must match patch size ({patch_dim})")

        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.num_patches = (height // patch_height) * (width // patch_width)
        self.patch_dim = patch_dim
        grid_shape = (height // patch_height, width // patch_width)
        self.autoencoder = WinfreeFieldAutoencoder(
            input_dim=patch_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=patch_dim,
            sequence_length=self.num_patches,
            grid_shape=grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            latent_conditioning_strength=latent_conditioning_strength,
            latent_readout=latent_readout,
            latent_readout_strength=latent_readout_strength,
            latent_output_skip=latent_output_skip,
            latent_output_skip_strength=latent_output_skip_strength,
            omega_scale=omega_scale,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            output_activation=output_activation,
            key=key,
        )

    def images_to_sequence(self, images: Array) -> Array:
        height, width = self.image_shape
        patch_height, patch_width = self.patch_shape
        batch_size = images.shape[0]

        images = images.reshape(batch_size, height, width)
        patches = images.reshape(
            batch_size,
            height // patch_height,
            patch_height,
            width // patch_width,
            patch_width,
        )
        patches = patches.transpose(0, 1, 3, 2, 4)
        patches = patches.reshape(batch_size, self.num_patches, self.patch_dim)
        return patches.transpose(1, 0, 2)

    def sequence_to_images(self, sequence: Array, flatten: bool = True) -> Array:
        height, width = self.image_shape
        patch_height, patch_width = self.patch_shape
        batch_size = sequence.shape[1]

        patches = sequence.transpose(1, 0, 2)
        images = patches.reshape(
            batch_size,
            height // patch_height,
            width // patch_width,
            patch_height,
            patch_width,
        )
        images = images.transpose(0, 1, 3, 2, 4)
        images = images.reshape(batch_size, height, width)
        if flatten:
            return images.reshape(batch_size, height * width)
        return images

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        sequence = self.images_to_sequence(images)
        return self.autoencoder.encode(sequence)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        sequence = self.images_to_sequence(images)
        return self.autoencoder.trace(sequence)

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        sequence = self.images_to_sequence(images)
        reconstruction = self.autoencoder(sequence)
        return self.sequence_to_images(reconstruction, flatten=True)

    @property
    def encoder(self) -> WinfreeFieldEncoder:
        return self.autoencoder.encoder

    @property
    def decoder(self) -> WinfreeFieldDecoder:
        return self.autoencoder.decoder


class WinfreeConditionalPatchDenoiser(eqx.Module):
    """
    Direct image-to-image Winfree phase-field model.

    Unlike ``WinfreePatchAutoencoder``, this wrapper does not route through a
    global latent decoder or latent output skip. The input patch grid conditions
    the local ``theta`` and ``omega`` fields directly, the phase field evolves,
    and the clean patch grid is read from the final phase features. This is a
    closer fit for masked reconstruction and denoising tasks where iterative
    spatial completion is the object of study.
    """

    input_to_omega: eqx.nn.Linear
    input_to_theta: eqx.nn.Linear
    theta_to_output: eqx.nn.Linear
    layer: WinfreeFieldLayer
    positional_omega: Array
    positional_theta: Array

    image_shape: Tuple[int, int] = eqx.field(static=True)
    patch_shape: Tuple[int, int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    patch_dim: int = eqx.field(static=True)
    input_patch_dim: int = eqx.field(static=True)
    input_channels: int = eqx.field(static=True)
    output_channels: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True)
    input_conditioning_strength: float = eqx.field(static=True)
    phase_init: str = eqx.field(static=True)
    phase_init_scale: float = eqx.field(static=True)
    output_activation: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "conv",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        input_conditioning_strength: float = 1.0,
        omega_scale: float = 1.0,
        initial_phase_scale: float = 0.1,
        phase_init: str = "learned",
        phase_init_scale: float = 1.0,
        field_activation: str = "relu",
        si_func: str = "mlp",
        si_hidden_ratio: int = 2,
        output_activation: str = "identity",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        if output_activation not in {"identity", "sigmoid", "tanh01"}:
            raise ValueError(
                "output_activation must be 'identity', 'sigmoid', or 'tanh01'"
            )
        if phase_init not in {"learned", "rotary_2d"}:
            raise ValueError("phase_init must be 'learned' or 'rotary_2d'")
        if phase_init_scale <= 0.0:
            raise ValueError("phase_init_scale must be positive")

        height, width = image_shape
        patch_height, patch_width = patch_shape
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError("image_shape must be divisible by patch_shape")

        patch_dim = patch_height * patch_width
        input_patch_dim, input_channels = _infer_patch_channels(
            input_dim,
            patch_shape,
        )

        keys = jax.random.split(key, 6)
        grid_shape = (height // patch_height, width // patch_width)
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.num_patches = grid_shape[0] * grid_shape[1]
        self.patch_dim = patch_dim
        self.input_patch_dim = int(input_patch_dim)
        self.input_channels = int(input_channels)
        self.output_channels = 1
        self.hidden_dim = int(hidden_dim)
        self.omega_scale = float(omega_scale)
        self.input_conditioning_strength = float(input_conditioning_strength)
        self.phase_init = phase_init
        self.phase_init_scale = float(phase_init_scale)
        self.output_activation = output_activation

        self.input_to_omega = eqx.nn.Linear(
            input_patch_dim,
            hidden_dim,
            key=keys[0],
        )
        self.input_to_theta = eqx.nn.Linear(
            input_patch_dim,
            hidden_dim,
            key=keys[1],
        )
        self.theta_to_output = eqx.nn.Linear(2 * hidden_dim, patch_dim, key=keys[2])
        self.layer = WinfreeFieldLayer(
            num_positions=self.num_patches,
            channels=hidden_dim,
            grid_shape=grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            key=keys[3],
        )
        self.positional_omega = (
            jax.random.normal(keys[4], (self.num_patches, hidden_dim)) * 0.02
        )
        if phase_init == "rotary_2d":
            self.positional_theta = _rotary_2d_phase_grid(
                grid_shape,
                hidden_dim,
                scale=self.phase_init_scale,
            )
        else:
            self.positional_theta = (
                jax.random.normal(keys[5], (self.num_patches, hidden_dim))
                * initial_phase_scale
            )

    def images_to_sequence(self, images: Array) -> Array:
        return _images_to_patch_sequence(
            images,
            self.image_shape,
            self.patch_shape,
            self.input_channels,
        )

    def sequence_to_images(self, sequence: Array, flatten: bool = True) -> Array:
        return _patch_sequence_to_images(
            sequence,
            self.image_shape,
            self.patch_shape,
            self.output_channels,
            flatten=flatten,
        )

    def _initial_fields(self, images: Array) -> Tuple[Array, Array]:
        sequence = self.images_to_sequence(images).transpose(1, 0, 2)
        omega = _apply_positionwise(self.input_to_omega, sequence)
        theta = _apply_positionwise(self.input_to_theta, sequence)
        omega = self.omega_scale * jnp.tanh(
            omega * self.input_conditioning_strength
            + self.positional_omega[None, :, :]
        )
        theta = wrap_phase(
            jnp.tanh(theta) * self.input_conditioning_strength
            + self.positional_theta[None, :, :]
        )
        return theta, omega

    def _readout(self, theta: Array) -> Array:
        outputs = _apply_positionwise(self.theta_to_output, phase_features(theta))
        return _apply_output_activation(outputs, self.output_activation)

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega = self._initial_fields(images)
        final_theta = self.layer(theta0, omega)
        return jnp.mean(phase_features(final_theta), axis=1)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        theta0, omega = self._initial_fields(images)
        trajectory = self.layer(theta0, omega, return_trajectory=True)
        reconstruction_sequence = self._readout(
            trajectory["final_theta"]
        ).transpose(1, 0, 2)
        return {
            "latent": jnp.mean(phase_features(trajectory["final_theta"]), axis=1),
            "reconstruction_sequence": reconstruction_sequence,
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": trajectory["final_theta"],
            "thetas": trajectory["thetas"],
            "energies": trajectory["energies"],
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega = self._initial_fields(images)
        final_theta = self.layer(theta0, omega)
        reconstruction = self._readout(final_theta).transpose(1, 0, 2)
        return self.sequence_to_images(reconstruction, flatten=True)


class WinfreeRatePhaseConditionalPatchDenoiser(eqx.Module):
    """
    Conditional denoiser with a Winfree phase field plus content/rate state.

    The phase field still evolves with ``WinfreeFieldLayer``. A separate content
    field carries image evidence through local recurrent convolution, while
    phase features gate the content update at each settling step. This tests the
    hypothesis that phase is most useful as a coordination signal rather than
    the only reconstruction state.
    """

    input_to_omega: eqx.nn.Linear
    input_to_theta: eqx.nn.Linear
    input_to_rate: eqx.nn.Linear
    phase_to_rate_gate: eqx.nn.Linear
    state_to_output: eqx.nn.Linear
    layer: WinfreeFieldLayer
    rate_conv_kernel: Array
    rate_conv_bias: Array
    positional_omega: Array
    positional_theta: Array
    positional_rate: Array

    image_shape: Tuple[int, int] = eqx.field(static=True)
    patch_shape: Tuple[int, int] = eqx.field(static=True)
    grid_shape: Tuple[int, int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    patch_dim: int = eqx.field(static=True)
    input_patch_dim: int = eqx.field(static=True)
    input_channels: int = eqx.field(static=True)
    output_channels: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True)
    input_conditioning_strength: float = eqx.field(static=True)
    phase_init: str = eqx.field(static=True)
    phase_init_scale: float = eqx.field(static=True)
    rate_kernel_size: int = eqx.field(static=True)
    rate_update_rate: float = eqx.field(static=True)
    rate_gate_strength: float = eqx.field(static=True)
    visibility_gate: str = eqx.field(static=True)
    visibility_drive_floor: float = eqx.field(static=True)
    missing_transport_strength: float = eqx.field(static=True)
    visibility_permutation: Tuple[int, ...] = eqx.field(static=True)
    output_activation: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "conv",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        input_conditioning_strength: float = 1.0,
        omega_scale: float = 1.0,
        initial_phase_scale: float = 0.1,
        phase_init: str = "learned",
        phase_init_scale: float = 1.0,
        field_activation: str = "relu",
        si_func: str = "mlp",
        si_hidden_ratio: int = 2,
        rate_kernel_size: int = 3,
        rate_update_rate: float = 0.5,
        rate_gate_strength: float = 1.0,
        visibility_gate: str = "none",
        visibility_drive_floor: float = 0.0,
        missing_transport_strength: float = 1.0,
        output_activation: str = "identity",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        if output_activation not in {"identity", "sigmoid", "tanh01"}:
            raise ValueError(
                "output_activation must be 'identity', 'sigmoid', or 'tanh01'"
            )
        if phase_init not in {"learned", "rotary_2d"}:
            raise ValueError("phase_init must be 'learned' or 'rotary_2d'")
        if phase_init_scale <= 0.0:
            raise ValueError("phase_init_scale must be positive")
        if rate_kernel_size < 1 or rate_kernel_size % 2 == 0:
            raise ValueError("rate_kernel_size must be a positive odd integer")
        if not 0.0 <= rate_update_rate <= 1.0:
            raise ValueError("rate_update_rate must be between 0 and 1")
        if rate_gate_strength < 0.0:
            raise ValueError("rate_gate_strength must be non-negative")
        if visibility_gate not in {"none", "visibility", "shuffle"}:
            raise ValueError(
                "visibility_gate must be 'none', 'visibility', or 'shuffle'"
            )
        if not 0.0 <= visibility_drive_floor <= 1.0:
            raise ValueError("visibility_drive_floor must be between 0 and 1")
        if missing_transport_strength < 0.0:
            raise ValueError("missing_transport_strength must be non-negative")

        height, width = image_shape
        patch_height, patch_width = patch_shape
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError("image_shape must be divisible by patch_shape")

        patch_dim = patch_height * patch_width
        input_patch_dim, input_channels = _infer_patch_channels(
            input_dim,
            patch_shape,
        )

        keys = jax.random.split(key, 10)
        grid_shape = (height // patch_height, width // patch_width)
        num_patches = grid_shape[0] * grid_shape[1]
        conv_scale = 1.0 / jnp.sqrt(
            float(rate_kernel_size * rate_kernel_size * hidden_dim)
        )

        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.grid_shape = grid_shape
        self.num_patches = int(num_patches)
        self.patch_dim = int(patch_dim)
        self.input_patch_dim = int(input_patch_dim)
        self.input_channels = int(input_channels)
        self.output_channels = 1
        self.hidden_dim = int(hidden_dim)
        self.omega_scale = float(omega_scale)
        self.input_conditioning_strength = float(input_conditioning_strength)
        self.phase_init = phase_init
        self.phase_init_scale = float(phase_init_scale)
        self.rate_kernel_size = int(rate_kernel_size)
        self.rate_update_rate = float(rate_update_rate)
        self.rate_gate_strength = float(rate_gate_strength)
        self.visibility_gate = visibility_gate
        self.visibility_drive_floor = float(visibility_drive_floor)
        self.missing_transport_strength = float(missing_transport_strength)
        self.visibility_permutation = _fixed_shuffle_permutation(self.num_patches)
        self.output_activation = output_activation

        self.input_to_omega = eqx.nn.Linear(
            input_patch_dim,
            hidden_dim,
            key=keys[0],
        )
        self.input_to_theta = eqx.nn.Linear(
            input_patch_dim,
            hidden_dim,
            key=keys[1],
        )
        self.input_to_rate = eqx.nn.Linear(
            input_patch_dim,
            hidden_dim,
            key=keys[2],
        )
        self.phase_to_rate_gate = eqx.nn.Linear(2 * hidden_dim, hidden_dim, key=keys[3])
        self.state_to_output = eqx.nn.Linear(3 * hidden_dim, patch_dim, key=keys[4])
        self.layer = WinfreeFieldLayer(
            num_positions=self.num_patches,
            channels=hidden_dim,
            grid_shape=grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            key=keys[5],
        )
        self.positional_omega = (
            jax.random.normal(keys[6], (self.num_patches, hidden_dim)) * 0.02
        )
        if phase_init == "rotary_2d":
            self.positional_theta = _rotary_2d_phase_grid(
                grid_shape,
                hidden_dim,
                scale=self.phase_init_scale,
            )
        else:
            self.positional_theta = (
                jax.random.normal(keys[7], (self.num_patches, hidden_dim))
                * initial_phase_scale
            )
        self.positional_rate = (
            jax.random.normal(keys[8], (self.num_patches, hidden_dim)) * 0.02
        )
        self.rate_conv_kernel = (
            jax.random.normal(
                keys[9],
                (rate_kernel_size, rate_kernel_size, hidden_dim, hidden_dim),
            )
            * conv_scale
        )
        self.rate_conv_bias = jnp.zeros((hidden_dim,))

    def images_to_sequence(self, images: Array) -> Array:
        return _images_to_patch_sequence(
            images,
            self.image_shape,
            self.patch_shape,
            self.input_channels,
        )

    def sequence_to_images(self, sequence: Array, flatten: bool = True) -> Array:
        return _patch_sequence_to_images(
            sequence,
            self.image_shape,
            self.patch_shape,
            self.output_channels,
            flatten=flatten,
        )

    def _initial_fields(self, images: Array) -> Tuple[Array, Array, Array, Array]:
        sequence = self.images_to_sequence(images).transpose(1, 0, 2)
        omega = _apply_positionwise(self.input_to_omega, sequence)
        theta = _apply_positionwise(self.input_to_theta, sequence)
        rate_drive = _apply_positionwise(self.input_to_rate, sequence)
        omega = self.omega_scale * jnp.tanh(
            omega * self.input_conditioning_strength
            + self.positional_omega[None, :, :]
        )
        theta = wrap_phase(
            jnp.tanh(theta) * self.input_conditioning_strength
            + self.positional_theta[None, :, :]
        )
        rate_drive = jnp.tanh(
            rate_drive * self.input_conditioning_strength
            + self.positional_rate[None, :, :]
        )
        return theta, omega, rate_drive, rate_drive

    def _visibility_from_inputs(self, images: Array) -> Optional[Array]:
        if self.visibility_gate == "none":
            return None
        if self.input_channels < 2:
            raise ValueError("visibility_gate requires image_plus_mask inputs")

        sequence = self.images_to_sequence(images).transpose(1, 0, 2)
        visibility = sequence[
            :,
            :,
            self.patch_dim : 2 * self.patch_dim,
        ].mean(axis=-1, keepdims=True)
        visibility = jnp.clip(visibility, 0.0, 1.0)
        if self.visibility_gate == "shuffle":
            permutation = jnp.asarray(self.visibility_permutation, dtype=jnp.int32)
            visibility = jnp.take(visibility, permutation, axis=1)
        return visibility

    def _combine_rate_terms(
        self,
        rate_drive: Array,
        transport_term: Array,
        visibility: Optional[Array],
    ) -> Array:
        if visibility is None or self.visibility_gate == "none":
            return rate_drive + transport_term

        visibility = jnp.clip(visibility, 0.0, 1.0)
        missing = 1.0 - visibility
        drive_scale = visibility + self.visibility_drive_floor * missing
        transport_scale = 1.0 + self.missing_transport_strength * missing
        return drive_scale * rate_drive + transport_scale * transport_term

    def _rate_step(
        self,
        rate: Array,
        theta: Array,
        rate_drive: Array,
        visibility: Optional[Array] = None,
    ) -> Array:
        batch_size = rate.shape[0]
        grid_h, grid_w = self.grid_shape
        rate_grid = rate.reshape(batch_size, grid_h, grid_w, self.hidden_dim)
        local_rate = _apply_local_conv2d(
            self.rate_conv_kernel,
            self.rate_conv_bias,
            rate_grid,
        )
        local_rate = local_rate.reshape(batch_size, self.num_patches, self.hidden_dim)
        gate = jax.nn.sigmoid(
            _apply_positionwise(self.phase_to_rate_gate, phase_features(theta))
        )
        transport_term = self.rate_gate_strength * gate * local_rate
        proposal = jnp.tanh(
            self._combine_rate_terms(rate_drive, transport_term, visibility)
        )
        return (1.0 - self.rate_update_rate) * rate + self.rate_update_rate * proposal

    def _evolve(
        self,
        theta: Array,
        omega: Array,
        rate: Array,
        rate_drive: Array,
        visibility: Optional[Array] = None,
    ):
        def scan_step(carry, _):
            theta_carry, rate_carry = carry
            theta_next, energy = self.layer.winfree_step(theta_carry, omega)
            rate_next = self._rate_step(
                rate_carry,
                theta_next,
                rate_drive,
                visibility,
            )
            return (theta_next, rate_next), (theta_next, rate_next, energy)

        (final_theta, final_rate), (thetas, rates, energies) = jax.lax.scan(
            scan_step,
            (wrap_phase(theta), rate),
            None,
            length=self.layer.steps,
        )
        return final_theta, final_rate, thetas, rates, energies

    def _readout(self, theta: Array, rate: Array) -> Array:
        features = jnp.concatenate([rate, phase_features(theta)], axis=-1)
        outputs = _apply_positionwise(self.state_to_output, features)
        return _apply_output_activation(outputs, self.output_activation)

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega, rate, rate_drive = self._initial_fields(images)
        visibility = self._visibility_from_inputs(images)
        final_theta, final_rate, _, _, _ = self._evolve(
            theta0,
            omega,
            rate,
            rate_drive,
            visibility,
        )
        return jnp.mean(
            jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1),
            axis=1,
        )

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        theta0, omega, rate0, rate_drive = self._initial_fields(images)
        visibility = self._visibility_from_inputs(images)
        final_theta, final_rate, thetas, rates, energies = self._evolve(
            theta0,
            omega,
            rate0,
            rate_drive,
            visibility,
        )
        reconstruction_sequence = self._readout(final_theta, final_rate).transpose(
            1,
            0,
            2,
        )
        return {
            "latent": jnp.mean(
                jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1),
                axis=1,
            ),
            "reconstruction_sequence": reconstruction_sequence,
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": final_theta,
            "thetas": thetas,
            "energies": energies,
            "initial_rate": rate0,
            "final_rate": final_rate,
            "rate_states": rates,
            "rate_drive": rate_drive,
            "visibility": (
                jnp.ones((images.shape[0], self.num_patches, 1))
                if visibility is None
                else visibility
            ),
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega, rate, rate_drive = self._initial_fields(images)
        visibility = self._visibility_from_inputs(images)
        final_theta, final_rate, _, _, _ = self._evolve(
            theta0,
            omega,
            rate,
            rate_drive,
            visibility,
        )
        reconstruction = self._readout(final_theta, final_rate).transpose(1, 0, 2)
        return self.sequence_to_images(reconstruction, flatten=True)


class WinfreeGlobalRatePhaseConditionalPatchDenoiser(eqx.Module):
    """
    Rate-phase denoiser with a separate slow/global phase carrier.

    This keeps the local rate-phase field intact and adds a one-position
    Winfree phase band initialized from the whole corrupted image. The global
    carrier gates local content propagation during recurrent settling. This is
    the first ONN-native "slow rhythm controls fast local work" probe.
    """

    local: WinfreeRatePhaseConditionalPatchDenoiser
    image_to_global_omega: eqx.nn.Linear
    image_to_global_theta: eqx.nn.Linear
    global_phase_to_rate_gate: eqx.nn.Linear
    global_layer: WinfreeFieldLayer
    positional_global_omega: Array
    positional_global_theta: Array

    image_shape: Tuple[int, int] = eqx.field(static=True)
    patch_shape: Tuple[int, int] = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    input_patch_dim: int = eqx.field(static=True)
    input_channels: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True)
    input_conditioning_strength: float = eqx.field(static=True)
    global_gate_strength: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        global_gamma: float = 0.05,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "conv",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        input_conditioning_strength: float = 1.0,
        omega_scale: float = 1.0,
        initial_phase_scale: float = 0.1,
        phase_init: str = "learned",
        phase_init_scale: float = 1.0,
        field_activation: str = "relu",
        si_func: str = "mlp",
        si_hidden_ratio: int = 2,
        rate_kernel_size: int = 3,
        rate_update_rate: float = 0.5,
        rate_gate_strength: float = 1.0,
        global_gate_strength: float = 0.5,
        visibility_gate: str = "none",
        visibility_drive_floor: float = 0.0,
        missing_transport_strength: float = 1.0,
        output_activation: str = "identity",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        if global_gate_strength < 0.0:
            raise ValueError("global_gate_strength must be non-negative")

        height, width = image_shape
        patch_height, patch_width = patch_shape
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError("image_shape must be divisible by patch_shape")
        input_patch_dim, input_channels = _infer_patch_channels(
            input_dim,
            patch_shape,
        )

        keys = jax.random.split(key, 6)
        image_dim = height * width * input_channels

        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.hidden_dim = int(hidden_dim)
        self.input_patch_dim = int(input_patch_dim)
        self.input_channels = int(input_channels)
        self.omega_scale = float(omega_scale)
        self.input_conditioning_strength = float(input_conditioning_strength)
        self.global_gate_strength = float(global_gate_strength)

        self.local = WinfreeRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            input_conditioning_strength=input_conditioning_strength,
            omega_scale=omega_scale,
            initial_phase_scale=initial_phase_scale,
            phase_init=phase_init,
            phase_init_scale=phase_init_scale,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            rate_kernel_size=rate_kernel_size,
            rate_update_rate=rate_update_rate,
            rate_gate_strength=rate_gate_strength,
            visibility_gate=visibility_gate,
            visibility_drive_floor=visibility_drive_floor,
            missing_transport_strength=missing_transport_strength,
            output_activation=output_activation,
            key=keys[0],
        )
        self.image_to_global_omega = eqx.nn.Linear(
            image_dim,
            hidden_dim,
            key=keys[1],
        )
        self.image_to_global_theta = eqx.nn.Linear(
            image_dim,
            hidden_dim,
            key=keys[2],
        )
        self.global_phase_to_rate_gate = eqx.nn.Linear(
            2 * hidden_dim,
            hidden_dim,
            key=keys[3],
        )
        self.global_layer = WinfreeFieldLayer(
            num_positions=1,
            channels=hidden_dim,
            grid_shape=(1, 1),
            steps=steps,
            gamma=global_gamma,
            coupling_strength=0.0,
            coupling_mode="matrix",
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            key=keys[4],
        )
        self.positional_global_omega = (
            jax.random.normal(keys[5], (1, hidden_dim)) * 0.02
        )
        self.positional_global_theta = jnp.zeros((1, hidden_dim))

    def images_to_sequence(self, images: Array) -> Array:
        return self.local.images_to_sequence(images)

    def sequence_to_images(self, sequence: Array, flatten: bool = True) -> Array:
        return self.local.sequence_to_images(sequence, flatten=flatten)

    def _global_initial_fields(self, images: Array) -> Tuple[Array, Array]:
        flat = images.reshape(images.shape[0], -1)
        omega = jax.vmap(self.image_to_global_omega)(flat)
        theta = jax.vmap(self.image_to_global_theta)(flat)
        omega = self.omega_scale * jnp.tanh(
            omega[:, None, :] * self.input_conditioning_strength
            + self.positional_global_omega[None, :, :]
        )
        theta = wrap_phase(
            jnp.tanh(theta[:, None, :]) * self.input_conditioning_strength
            + self.positional_global_theta[None, :, :]
        )
        return theta, omega

    def _rate_step(
        self,
        rate: Array,
        theta: Array,
        global_theta: Array,
        rate_drive: Array,
        visibility: Optional[Array] = None,
    ) -> Array:
        batch_size = rate.shape[0]
        grid_h, grid_w = self.local.grid_shape
        rate_grid = rate.reshape(batch_size, grid_h, grid_w, self.hidden_dim)
        local_rate = _apply_local_conv2d(
            self.local.rate_conv_kernel,
            self.local.rate_conv_bias,
            rate_grid,
        )
        local_rate = local_rate.reshape(batch_size, self.local.num_patches, self.hidden_dim)
        local_gate = jax.nn.sigmoid(
            _apply_positionwise(self.local.phase_to_rate_gate, phase_features(theta))
        )
        global_gate = jax.nn.sigmoid(
            _apply_positionwise(
                self.global_phase_to_rate_gate,
                phase_features(global_theta),
            )
        )
        global_gate = jnp.broadcast_to(global_gate, local_gate.shape)
        transport_term = (
            local_gate
            + self.global_gate_strength * global_gate
        ) * local_rate
        proposal = jnp.tanh(
            self.local._combine_rate_terms(rate_drive, transport_term, visibility)
        )
        update_rate = self.local.rate_update_rate
        return (1.0 - update_rate) * rate + update_rate * proposal

    def _evolve(
        self,
        theta: Array,
        omega: Array,
        rate: Array,
        rate_drive: Array,
        global_theta: Array,
        global_omega: Array,
        visibility: Optional[Array] = None,
    ):
        def scan_step(carry, _):
            theta_carry, rate_carry, global_theta_carry = carry
            theta_next, energy = self.local.layer.winfree_step(theta_carry, omega)
            global_next, global_energy = self.global_layer.winfree_step(
                global_theta_carry,
                global_omega,
            )
            rate_next = self._rate_step(
                rate_carry,
                theta_next,
                global_next,
                rate_drive,
                visibility,
            )
            return (
                theta_next,
                rate_next,
                global_next,
            ), (
                theta_next,
                rate_next,
                global_next,
                energy,
                global_energy,
            )

        (
            final_theta,
            final_rate,
            final_global_theta,
        ), (
            thetas,
            rates,
            global_thetas,
            energies,
            global_energies,
        ) = jax.lax.scan(
            scan_step,
            (wrap_phase(theta), rate, wrap_phase(global_theta)),
            None,
            length=self.local.layer.steps,
        )
        return (
            final_theta,
            final_rate,
            final_global_theta,
            thetas,
            rates,
            global_thetas,
            energies,
            global_energies,
        )

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega, rate, rate_drive = self.local._initial_fields(images)
        visibility = self.local._visibility_from_inputs(images)
        global_theta0, global_omega = self._global_initial_fields(images)
        final_theta, final_rate, final_global_theta, *_ = self._evolve(
            theta0,
            omega,
            rate,
            rate_drive,
            global_theta0,
            global_omega,
            visibility,
        )
        local_latent = jnp.mean(
            jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1),
            axis=1,
        )
        global_latent = phase_features(final_global_theta).reshape(images.shape[0], -1)
        return jnp.concatenate([local_latent, global_latent], axis=-1)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        theta0, omega, rate0, rate_drive = self.local._initial_fields(images)
        visibility = self.local._visibility_from_inputs(images)
        global_theta0, global_omega = self._global_initial_fields(images)
        (
            final_theta,
            final_rate,
            final_global_theta,
            thetas,
            rates,
            global_thetas,
            energies,
            global_energies,
        ) = self._evolve(
            theta0,
            omega,
            rate0,
            rate_drive,
            global_theta0,
            global_omega,
            visibility,
        )
        reconstruction_sequence = self.local._readout(final_theta, final_rate).transpose(
            1,
            0,
            2,
        )
        local_latent = jnp.mean(
            jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1),
            axis=1,
        )
        global_latent = phase_features(final_global_theta).reshape(images.shape[0], -1)
        return {
            "latent": jnp.concatenate([local_latent, global_latent], axis=-1),
            "reconstruction_sequence": reconstruction_sequence,
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": final_theta,
            "thetas": thetas,
            "energies": energies,
            "initial_rate": rate0,
            "final_rate": final_rate,
            "rate_states": rates,
            "rate_drive": rate_drive,
            "visibility": (
                jnp.ones((images.shape[0], self.local.num_patches, 1))
                if visibility is None
                else visibility
            ),
            "global_omega": global_omega,
            "initial_global_theta": global_theta0,
            "final_global_theta": final_global_theta,
            "global_thetas": global_thetas,
            "global_energies": global_energies,
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega, rate, rate_drive = self.local._initial_fields(images)
        visibility = self.local._visibility_from_inputs(images)
        global_theta0, global_omega = self._global_initial_fields(images)
        final_theta, final_rate, *_ = self._evolve(
            theta0,
            omega,
            rate,
            rate_drive,
            global_theta0,
            global_omega,
            visibility,
        )
        reconstruction = self.local._readout(final_theta, final_rate).transpose(1, 0, 2)
        return self.sequence_to_images(reconstruction, flatten=True)


class WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser(eqx.Module):
    """
    Rate-phase denoiser with a separate coarse spatial phase mesh.

    The local rate-phase field stays on the fine patch grid. A slower Winfree
    field runs on a small coarse grid initialized from spatially pooled input
    patches, then projects phase gates back onto the fine field. This tests
    whether block completion benefits from a geometric slow phase band rather
    than a one-node global carrier or a U-Net-style tensor skip.
    """

    local: WinfreeRatePhaseConditionalPatchDenoiser
    coarse_input_to_omega: eqx.nn.Linear
    coarse_input_to_theta: eqx.nn.Linear
    coarse_phase_to_rate_gate: eqx.nn.Linear
    coarse_layer: WinfreeFieldLayer
    positional_coarse_omega: Array
    positional_coarse_theta: Array

    image_shape: Tuple[int, int] = eqx.field(static=True)
    patch_shape: Tuple[int, int] = eqx.field(static=True)
    grid_shape: Tuple[int, int] = eqx.field(static=True)
    coarse_grid_shape: Tuple[int, int] = eqx.field(static=True)
    coarse_positions: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    input_patch_dim: int = eqx.field(static=True)
    input_channels: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True)
    input_conditioning_strength: float = eqx.field(static=True)
    global_gate_strength: float = eqx.field(static=True)
    global_phase_control: str = eqx.field(static=True)
    fine_to_coarse_weights: Tuple[Tuple[float, ...], ...] = eqx.field(
        static=True
    )
    coarse_to_fine_weights: Tuple[Tuple[float, ...], ...] = eqx.field(
        static=True
    )
    coarse_phase_permutation: Tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        coarse_grid_shape: Tuple[int, int] = (2, 2),
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        global_gamma: float = 0.05,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "conv",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        input_conditioning_strength: float = 1.0,
        omega_scale: float = 1.0,
        initial_phase_scale: float = 0.1,
        phase_init: str = "learned",
        phase_init_scale: float = 1.0,
        field_activation: str = "relu",
        si_func: str = "mlp",
        si_hidden_ratio: int = 2,
        rate_kernel_size: int = 3,
        rate_update_rate: float = 0.5,
        rate_gate_strength: float = 1.0,
        global_gate_strength: float = 0.5,
        global_phase_control: str = "none",
        output_activation: str = "identity",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        if global_gate_strength < 0.0:
            raise ValueError("global_gate_strength must be non-negative")
        if global_phase_control not in {"none", "shuffle"}:
            raise ValueError("global_phase_control must be 'none' or 'shuffle'")
        if min(coarse_grid_shape) < 1:
            raise ValueError("coarse_grid_shape dimensions must be positive")

        height, width = image_shape
        patch_height, patch_width = patch_shape
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError("image_shape must be divisible by patch_shape")
        input_patch_dim, input_channels = _infer_patch_channels(
            input_dim,
            patch_shape,
        )

        keys = jax.random.split(key, 7)
        grid_shape = (height // patch_height, width // patch_width)
        coarse_positions = coarse_grid_shape[0] * coarse_grid_shape[1]

        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.grid_shape = grid_shape
        self.coarse_grid_shape = coarse_grid_shape
        self.coarse_positions = int(coarse_positions)
        self.hidden_dim = int(hidden_dim)
        self.input_patch_dim = int(input_patch_dim)
        self.input_channels = int(input_channels)
        self.omega_scale = float(omega_scale)
        self.input_conditioning_strength = float(input_conditioning_strength)
        self.global_gate_strength = float(global_gate_strength)
        self.global_phase_control = global_phase_control
        self.fine_to_coarse_weights = _grid_transfer_weights(
            grid_shape,
            coarse_grid_shape,
        )
        self.coarse_to_fine_weights = _grid_transfer_weights(
            coarse_grid_shape,
            grid_shape,
        )
        self.coarse_phase_permutation = _fixed_shuffle_permutation(
            self.coarse_positions
        )

        self.local = WinfreeRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            input_conditioning_strength=input_conditioning_strength,
            omega_scale=omega_scale,
            initial_phase_scale=initial_phase_scale,
            phase_init=phase_init,
            phase_init_scale=phase_init_scale,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            rate_kernel_size=rate_kernel_size,
            rate_update_rate=rate_update_rate,
            rate_gate_strength=rate_gate_strength,
            output_activation=output_activation,
            key=keys[0],
        )
        self.coarse_input_to_omega = eqx.nn.Linear(
            input_patch_dim,
            hidden_dim,
            key=keys[1],
        )
        self.coarse_input_to_theta = eqx.nn.Linear(
            input_patch_dim,
            hidden_dim,
            key=keys[2],
        )
        self.coarse_phase_to_rate_gate = eqx.nn.Linear(
            2 * hidden_dim,
            hidden_dim,
            key=keys[3],
        )
        self.coarse_layer = WinfreeFieldLayer(
            num_positions=self.coarse_positions,
            channels=hidden_dim,
            grid_shape=coarse_grid_shape,
            steps=steps,
            gamma=global_gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=None,
            coupling_mode="matrix",
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            key=keys[4],
        )
        self.positional_coarse_omega = (
            jax.random.normal(keys[5], (self.coarse_positions, hidden_dim))
            * 0.02
        )
        self.positional_coarse_theta = (
            _rotary_2d_phase_grid(
                coarse_grid_shape,
                hidden_dim,
                scale=max(phase_init_scale, 1e-6),
            )
            + jax.random.normal(keys[6], (self.coarse_positions, hidden_dim))
            * initial_phase_scale
        )

    def images_to_sequence(self, images: Array) -> Array:
        return self.local.images_to_sequence(images)

    def sequence_to_images(self, sequence: Array, flatten: bool = True) -> Array:
        return self.local.sequence_to_images(sequence, flatten=flatten)

    def _fine_to_coarse_weights_array(self, dtype) -> Array:
        return jnp.asarray(self.fine_to_coarse_weights, dtype=dtype)

    def _coarse_to_fine_weights_array(self, dtype) -> Array:
        return jnp.asarray(self.coarse_to_fine_weights, dtype=dtype)

    def _coarse_initial_fields(self, images: Array) -> Tuple[Array, Array]:
        sequence = self.local.images_to_sequence(images).transpose(1, 0, 2)
        weights = self._fine_to_coarse_weights_array(sequence.dtype)
        coarse_patches = jnp.einsum("cf,bfp->bcp", weights, sequence)
        omega = _apply_positionwise(self.coarse_input_to_omega, coarse_patches)
        theta = _apply_positionwise(self.coarse_input_to_theta, coarse_patches)
        omega = self.omega_scale * jnp.tanh(
            omega * self.input_conditioning_strength
            + self.positional_coarse_omega[None, :, :]
        )
        theta = wrap_phase(
            jnp.tanh(theta) * self.input_conditioning_strength
            + self.positional_coarse_theta[None, :, :]
        )
        return theta, omega

    def _coarse_theta_for_gate(self, coarse_theta: Array) -> Array:
        if self.global_phase_control == "none":
            return coarse_theta
        permutation = jnp.asarray(self.coarse_phase_permutation, dtype=jnp.int32)
        return jnp.take(coarse_theta, permutation, axis=1)

    def _coarse_gate_to_fine(self, coarse_theta: Array) -> Array:
        coarse_theta = self._coarse_theta_for_gate(coarse_theta)
        coarse_gate = jax.nn.sigmoid(
            _apply_positionwise(
                self.coarse_phase_to_rate_gate,
                phase_features(coarse_theta),
            )
        )
        weights = self._coarse_to_fine_weights_array(coarse_gate.dtype)
        return jnp.einsum("fc,bch->bfh", weights, coarse_gate)

    def _rate_step(
        self,
        rate: Array,
        theta: Array,
        coarse_theta: Array,
        rate_drive: Array,
    ) -> Array:
        batch_size = rate.shape[0]
        grid_h, grid_w = self.local.grid_shape
        rate_grid = rate.reshape(batch_size, grid_h, grid_w, self.hidden_dim)
        local_rate = _apply_local_conv2d(
            self.local.rate_conv_kernel,
            self.local.rate_conv_bias,
            rate_grid,
        )
        local_rate = local_rate.reshape(
            batch_size,
            self.local.num_patches,
            self.hidden_dim,
        )
        local_gate = jax.nn.sigmoid(
            _apply_positionwise(self.local.phase_to_rate_gate, phase_features(theta))
        )
        coarse_gate = self._coarse_gate_to_fine(coarse_theta)
        proposal = jnp.tanh(
            rate_drive
            + (
                self.local.rate_gate_strength * local_gate
                + self.global_gate_strength * coarse_gate
            )
            * local_rate
        )
        update_rate = self.local.rate_update_rate
        return (1.0 - update_rate) * rate + update_rate * proposal

    def _evolve(
        self,
        theta: Array,
        omega: Array,
        rate: Array,
        rate_drive: Array,
        coarse_theta: Array,
        coarse_omega: Array,
    ):
        def scan_step(carry, _):
            theta_carry, rate_carry, coarse_theta_carry = carry
            theta_next, energy = self.local.layer.winfree_step(theta_carry, omega)
            coarse_next, coarse_energy = self.coarse_layer.winfree_step(
                coarse_theta_carry,
                coarse_omega,
            )
            rate_next = self._rate_step(
                rate_carry,
                theta_next,
                coarse_next,
                rate_drive,
            )
            return (
                theta_next,
                rate_next,
                coarse_next,
            ), (
                theta_next,
                rate_next,
                coarse_next,
                energy,
                coarse_energy,
            )

        (
            final_theta,
            final_rate,
            final_coarse_theta,
        ), (
            thetas,
            rates,
            coarse_thetas,
            energies,
            coarse_energies,
        ) = jax.lax.scan(
            scan_step,
            (wrap_phase(theta), rate, wrap_phase(coarse_theta)),
            None,
            length=self.local.layer.steps,
        )
        return (
            final_theta,
            final_rate,
            final_coarse_theta,
            thetas,
            rates,
            coarse_thetas,
            energies,
            coarse_energies,
        )

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega, rate, rate_drive = self.local._initial_fields(images)
        coarse_theta0, coarse_omega = self._coarse_initial_fields(images)
        final_theta, final_rate, final_coarse_theta, *_ = self._evolve(
            theta0,
            omega,
            rate,
            rate_drive,
            coarse_theta0,
            coarse_omega,
        )
        local_latent = jnp.mean(
            jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1),
            axis=1,
        )
        coarse_latent = phase_features(final_coarse_theta).reshape(
            images.shape[0],
            -1,
        )
        return jnp.concatenate([local_latent, coarse_latent], axis=-1)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        theta0, omega, rate0, rate_drive = self.local._initial_fields(images)
        coarse_theta0, coarse_omega = self._coarse_initial_fields(images)
        (
            final_theta,
            final_rate,
            final_coarse_theta,
            thetas,
            rates,
            coarse_thetas,
            energies,
            coarse_energies,
        ) = self._evolve(
            theta0,
            omega,
            rate0,
            rate_drive,
            coarse_theta0,
            coarse_omega,
        )
        reconstruction_sequence = self.local._readout(final_theta, final_rate).transpose(
            1,
            0,
            2,
        )
        local_latent = jnp.mean(
            jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1),
            axis=1,
        )
        coarse_latent = phase_features(final_coarse_theta).reshape(
            images.shape[0],
            -1,
        )
        return {
            "latent": jnp.concatenate([local_latent, coarse_latent], axis=-1),
            "reconstruction_sequence": reconstruction_sequence,
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": final_theta,
            "thetas": thetas,
            "energies": energies,
            "initial_rate": rate0,
            "final_rate": final_rate,
            "rate_states": rates,
            "rate_drive": rate_drive,
            "coarse_omega": coarse_omega,
            "initial_coarse_theta": coarse_theta0,
            "final_coarse_theta": final_coarse_theta,
            "coarse_thetas": coarse_thetas,
            "coarse_energies": coarse_energies,
            "fine_to_coarse_weights": self._fine_to_coarse_weights_array(
                images.dtype
            ),
            "coarse_to_fine_weights": self._coarse_to_fine_weights_array(
                images.dtype
            ),
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega, rate, rate_drive = self.local._initial_fields(images)
        coarse_theta0, coarse_omega = self._coarse_initial_fields(images)
        final_theta, final_rate, *_ = self._evolve(
            theta0,
            omega,
            rate,
            rate_drive,
            coarse_theta0,
            coarse_omega,
        )
        reconstruction = self.local._readout(final_theta, final_rate).transpose(1, 0, 2)
        return self.sequence_to_images(reconstruction, flatten=True)


class WinfreeCoarseRatePhaseConditionalPatchDenoiser(
    WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser
):
    """
    Coarse-to-fine rate-phase denoiser with explicit content transport.

    This extends the coarse phase-mesh probe by giving the coarse band its own
    content/rate state. The coarse rate field evolves under coarse phase gates
    and is projected down into the fine rate field as an additive drive. This
    tests whether block completion needs global shape/content transport, not
    merely a coarse phase gate on local propagation.
    """

    coarse_input_to_rate: eqx.nn.Linear
    coarse_rate_to_fine: eqx.nn.Linear
    coarse_rate_conv_kernel: Array
    coarse_rate_conv_bias: Array
    positional_coarse_rate: Array

    global_content_strength: float = eqx.field(static=True)
    global_content_control: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        coarse_grid_shape: Tuple[int, int] = (2, 2),
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        global_gamma: float = 0.05,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "conv",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        input_conditioning_strength: float = 1.0,
        omega_scale: float = 1.0,
        initial_phase_scale: float = 0.1,
        phase_init: str = "learned",
        phase_init_scale: float = 1.0,
        field_activation: str = "relu",
        si_func: str = "mlp",
        si_hidden_ratio: int = 2,
        rate_kernel_size: int = 3,
        rate_update_rate: float = 0.5,
        rate_gate_strength: float = 1.0,
        global_gate_strength: float = 0.5,
        global_phase_control: str = "none",
        global_content_strength: float = 0.5,
        global_content_control: str = "none",
        output_activation: str = "identity",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        if global_content_strength < 0.0:
            raise ValueError("global_content_strength must be non-negative")
        if global_content_control not in {"none", "shuffle"}:
            raise ValueError("global_content_control must be 'none' or 'shuffle'")

        keys = jax.random.split(key, 5)
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            coarse_grid_shape=coarse_grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            global_gamma=global_gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            input_conditioning_strength=input_conditioning_strength,
            omega_scale=omega_scale,
            initial_phase_scale=initial_phase_scale,
            phase_init=phase_init,
            phase_init_scale=phase_init_scale,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            rate_kernel_size=rate_kernel_size,
            rate_update_rate=rate_update_rate,
            rate_gate_strength=rate_gate_strength,
            global_gate_strength=global_gate_strength,
            global_phase_control=global_phase_control,
            output_activation=output_activation,
            key=keys[0],
        )

        input_patch_dim, _ = _infer_patch_channels(input_dim, patch_shape)
        conv_scale = 1.0 / jnp.sqrt(
            float(rate_kernel_size * rate_kernel_size * hidden_dim)
        )

        self.global_content_strength = float(global_content_strength)
        self.global_content_control = global_content_control
        self.coarse_input_to_rate = eqx.nn.Linear(
            input_patch_dim,
            hidden_dim,
            key=keys[1],
        )
        self.coarse_rate_to_fine = eqx.nn.Linear(
            hidden_dim,
            hidden_dim,
            key=keys[2],
        )
        self.positional_coarse_rate = (
            jax.random.normal(keys[3], (self.coarse_positions, hidden_dim)) * 0.02
        )
        self.coarse_rate_conv_kernel = (
            jax.random.normal(
                keys[4],
                (rate_kernel_size, rate_kernel_size, hidden_dim, hidden_dim),
            )
            * conv_scale
        )
        self.coarse_rate_conv_bias = jnp.zeros((hidden_dim,))

    def _coarse_initial_fields(self, images: Array) -> Tuple[Array, Array, Array, Array]:
        sequence = self.local.images_to_sequence(images).transpose(1, 0, 2)
        weights = self._fine_to_coarse_weights_array(sequence.dtype)
        coarse_patches = jnp.einsum("cf,bfp->bcp", weights, sequence)
        omega = _apply_positionwise(self.coarse_input_to_omega, coarse_patches)
        theta = _apply_positionwise(self.coarse_input_to_theta, coarse_patches)
        rate_drive = _apply_positionwise(self.coarse_input_to_rate, coarse_patches)
        omega = self.omega_scale * jnp.tanh(
            omega * self.input_conditioning_strength
            + self.positional_coarse_omega[None, :, :]
        )
        theta = wrap_phase(
            jnp.tanh(theta) * self.input_conditioning_strength
            + self.positional_coarse_theta[None, :, :]
        )
        rate_drive = jnp.tanh(
            rate_drive * self.input_conditioning_strength
            + self.positional_coarse_rate[None, :, :]
        )
        return theta, omega, rate_drive, rate_drive

    def _coarse_rate_for_content(self, coarse_rate: Array) -> Array:
        if self.global_content_control == "none":
            return coarse_rate
        permutation = jnp.asarray(self.coarse_phase_permutation, dtype=jnp.int32)
        return jnp.take(coarse_rate, permutation, axis=1)

    def _coarse_rate_to_fine_content(self, coarse_rate: Array) -> Array:
        coarse_rate = self._coarse_rate_for_content(coarse_rate)
        weights = self._coarse_to_fine_weights_array(coarse_rate.dtype)
        fine_rate = jnp.einsum("fc,bch->bfh", weights, coarse_rate)
        return jnp.tanh(_apply_positionwise(self.coarse_rate_to_fine, fine_rate))

    def _coarse_rate_step(
        self,
        coarse_rate: Array,
        coarse_theta: Array,
        coarse_rate_drive: Array,
    ) -> Array:
        batch_size = coarse_rate.shape[0]
        grid_h, grid_w = self.coarse_grid_shape
        rate_grid = coarse_rate.reshape(batch_size, grid_h, grid_w, self.hidden_dim)
        local_rate = _apply_local_conv2d(
            self.coarse_rate_conv_kernel,
            self.coarse_rate_conv_bias,
            rate_grid,
        )
        local_rate = local_rate.reshape(
            batch_size,
            self.coarse_positions,
            self.hidden_dim,
        )
        gate = jax.nn.sigmoid(
            _apply_positionwise(
                self.coarse_phase_to_rate_gate,
                phase_features(coarse_theta),
            )
        )
        proposal = jnp.tanh(
            coarse_rate_drive + self.local.rate_gate_strength * gate * local_rate
        )
        update_rate = self.local.rate_update_rate
        return (1.0 - update_rate) * coarse_rate + update_rate * proposal

    def _rate_step(
        self,
        rate: Array,
        theta: Array,
        coarse_theta: Array,
        coarse_rate: Array,
        rate_drive: Array,
    ) -> Array:
        batch_size = rate.shape[0]
        grid_h, grid_w = self.local.grid_shape
        rate_grid = rate.reshape(batch_size, grid_h, grid_w, self.hidden_dim)
        local_rate = _apply_local_conv2d(
            self.local.rate_conv_kernel,
            self.local.rate_conv_bias,
            rate_grid,
        )
        local_rate = local_rate.reshape(
            batch_size,
            self.local.num_patches,
            self.hidden_dim,
        )
        local_gate = jax.nn.sigmoid(
            _apply_positionwise(self.local.phase_to_rate_gate, phase_features(theta))
        )
        coarse_gate = self._coarse_gate_to_fine(coarse_theta)
        coarse_content = self._coarse_rate_to_fine_content(coarse_rate)
        proposal = jnp.tanh(
            rate_drive
            + self.global_content_strength * coarse_content
            + (
                self.local.rate_gate_strength * local_gate
                + self.global_gate_strength * coarse_gate
            )
            * local_rate
        )
        update_rate = self.local.rate_update_rate
        return (1.0 - update_rate) * rate + update_rate * proposal

    def _evolve(
        self,
        theta: Array,
        omega: Array,
        rate: Array,
        rate_drive: Array,
        coarse_theta: Array,
        coarse_omega: Array,
        coarse_rate: Array,
        coarse_rate_drive: Array,
    ):
        def scan_step(carry, _):
            theta_carry, rate_carry, coarse_theta_carry, coarse_rate_carry = carry
            theta_next, energy = self.local.layer.winfree_step(theta_carry, omega)
            coarse_next, coarse_energy = self.coarse_layer.winfree_step(
                coarse_theta_carry,
                coarse_omega,
            )
            coarse_rate_next = self._coarse_rate_step(
                coarse_rate_carry,
                coarse_next,
                coarse_rate_drive,
            )
            rate_next = self._rate_step(
                rate_carry,
                theta_next,
                coarse_next,
                coarse_rate_next,
                rate_drive,
            )
            return (
                theta_next,
                rate_next,
                coarse_next,
                coarse_rate_next,
            ), (
                theta_next,
                rate_next,
                coarse_next,
                coarse_rate_next,
                energy,
                coarse_energy,
            )

        (
            final_theta,
            final_rate,
            final_coarse_theta,
            final_coarse_rate,
        ), (
            thetas,
            rates,
            coarse_thetas,
            coarse_rates,
            energies,
            coarse_energies,
        ) = jax.lax.scan(
            scan_step,
            (
                wrap_phase(theta),
                rate,
                wrap_phase(coarse_theta),
                coarse_rate,
            ),
            None,
            length=self.local.layer.steps,
        )
        return (
            final_theta,
            final_rate,
            final_coarse_theta,
            final_coarse_rate,
            thetas,
            rates,
            coarse_thetas,
            coarse_rates,
            energies,
            coarse_energies,
        )

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega, rate, rate_drive = self.local._initial_fields(images)
        (
            coarse_theta0,
            coarse_omega,
            coarse_rate0,
            coarse_rate_drive,
        ) = self._coarse_initial_fields(images)
        final_theta, final_rate, final_coarse_theta, final_coarse_rate, *_ = (
            self._evolve(
                theta0,
                omega,
                rate,
                rate_drive,
                coarse_theta0,
                coarse_omega,
                coarse_rate0,
                coarse_rate_drive,
            )
        )
        local_latent = jnp.mean(
            jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1),
            axis=1,
        )
        coarse_latent = jnp.concatenate(
            [
                final_coarse_rate.reshape(images.shape[0], -1),
                phase_features(final_coarse_theta).reshape(images.shape[0], -1),
            ],
            axis=-1,
        )
        return jnp.concatenate([local_latent, coarse_latent], axis=-1)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        theta0, omega, rate0, rate_drive = self.local._initial_fields(images)
        (
            coarse_theta0,
            coarse_omega,
            coarse_rate0,
            coarse_rate_drive,
        ) = self._coarse_initial_fields(images)
        (
            final_theta,
            final_rate,
            final_coarse_theta,
            final_coarse_rate,
            thetas,
            rates,
            coarse_thetas,
            coarse_rates,
            energies,
            coarse_energies,
        ) = self._evolve(
            theta0,
            omega,
            rate0,
            rate_drive,
            coarse_theta0,
            coarse_omega,
            coarse_rate0,
            coarse_rate_drive,
        )
        reconstruction_sequence = self.local._readout(final_theta, final_rate).transpose(
            1,
            0,
            2,
        )
        local_latent = jnp.mean(
            jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1),
            axis=1,
        )
        coarse_latent = jnp.concatenate(
            [
                final_coarse_rate.reshape(images.shape[0], -1),
                phase_features(final_coarse_theta).reshape(images.shape[0], -1),
            ],
            axis=-1,
        )
        return {
            "latent": jnp.concatenate([local_latent, coarse_latent], axis=-1),
            "reconstruction_sequence": reconstruction_sequence,
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": final_theta,
            "thetas": thetas,
            "energies": energies,
            "initial_rate": rate0,
            "final_rate": final_rate,
            "rate_states": rates,
            "rate_drive": rate_drive,
            "coarse_omega": coarse_omega,
            "initial_coarse_theta": coarse_theta0,
            "final_coarse_theta": final_coarse_theta,
            "coarse_thetas": coarse_thetas,
            "coarse_energies": coarse_energies,
            "initial_coarse_rate": coarse_rate0,
            "final_coarse_rate": final_coarse_rate,
            "coarse_rate_states": coarse_rates,
            "coarse_rate_drive": coarse_rate_drive,
            "coarse_rate_to_fine": self._coarse_rate_to_fine_content(
                final_coarse_rate
            ),
            "fine_to_coarse_weights": self._fine_to_coarse_weights_array(
                images.dtype
            ),
            "coarse_to_fine_weights": self._coarse_to_fine_weights_array(
                images.dtype
            ),
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega, rate, rate_drive = self.local._initial_fields(images)
        (
            coarse_theta0,
            coarse_omega,
            coarse_rate0,
            coarse_rate_drive,
        ) = self._coarse_initial_fields(images)
        final_theta, final_rate, *_ = self._evolve(
            theta0,
            omega,
            rate,
            rate_drive,
            coarse_theta0,
            coarse_omega,
            coarse_rate0,
            coarse_rate_drive,
        )
        reconstruction = self.local._readout(final_theta, final_rate).transpose(1, 0, 2)
        return self.sequence_to_images(reconstruction, flatten=True)


class WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser(
    WinfreeCoarseRatePhaseConditionalPatchDenoiser
):
    """
    Coarse-to-fine rate-phase denoiser with readout-only coarse prediction.

    The coarse band keeps its own rate/phase state, but it does not inject coarse
    content into the fine recurrent dynamics. Instead, the final coarse state
    predicts a broad patch-level correction that is blended into the final fine
    readout. This tests whether a slow spatial field helps block completion as a
    shape prior without creating the recurrent instability seen in additive
    coarse content transport.
    """

    coarse_state_to_patch: eqx.nn.Linear
    coarse_readout_strength: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        coarse_grid_shape: Tuple[int, int] = (2, 2),
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        global_gamma: float = 0.05,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "conv",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        input_conditioning_strength: float = 1.0,
        omega_scale: float = 1.0,
        initial_phase_scale: float = 0.1,
        phase_init: str = "learned",
        phase_init_scale: float = 1.0,
        field_activation: str = "relu",
        si_func: str = "mlp",
        si_hidden_ratio: int = 2,
        rate_kernel_size: int = 3,
        rate_update_rate: float = 0.5,
        rate_gate_strength: float = 1.0,
        global_gate_strength: float = 0.5,
        global_phase_control: str = "none",
        global_content_control: str = "none",
        coarse_readout_strength: float = 0.5,
        output_activation: str = "identity",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        if coarse_readout_strength < 0.0:
            raise ValueError("coarse_readout_strength must be non-negative")

        keys = jax.random.split(key, 2)
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            coarse_grid_shape=coarse_grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            global_gamma=global_gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            input_conditioning_strength=input_conditioning_strength,
            omega_scale=omega_scale,
            initial_phase_scale=initial_phase_scale,
            phase_init=phase_init,
            phase_init_scale=phase_init_scale,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            rate_kernel_size=rate_kernel_size,
            rate_update_rate=rate_update_rate,
            rate_gate_strength=rate_gate_strength,
            global_gate_strength=global_gate_strength,
            global_phase_control=global_phase_control,
            global_content_strength=0.0,
            global_content_control=global_content_control,
            output_activation=output_activation,
            key=keys[0],
        )
        self.coarse_readout_strength = float(coarse_readout_strength)
        self.coarse_state_to_patch = eqx.nn.Linear(
            3 * hidden_dim,
            self.local.patch_dim,
            key=keys[1],
        )

    def _rate_step(
        self,
        rate: Array,
        theta: Array,
        coarse_theta: Array,
        coarse_rate: Array,
        rate_drive: Array,
    ) -> Array:
        del coarse_rate
        batch_size = rate.shape[0]
        grid_h, grid_w = self.local.grid_shape
        rate_grid = rate.reshape(batch_size, grid_h, grid_w, self.hidden_dim)
        local_rate = _apply_local_conv2d(
            self.local.rate_conv_kernel,
            self.local.rate_conv_bias,
            rate_grid,
        )
        local_rate = local_rate.reshape(
            batch_size,
            self.local.num_patches,
            self.hidden_dim,
        )
        local_gate = jax.nn.sigmoid(
            _apply_positionwise(self.local.phase_to_rate_gate, phase_features(theta))
        )
        coarse_gate = self._coarse_gate_to_fine(coarse_theta)
        proposal = jnp.tanh(
            rate_drive
            + (
                self.local.rate_gate_strength * local_gate
                + self.global_gate_strength * coarse_gate
            )
            * local_rate
        )
        update_rate = self.local.rate_update_rate
        return (1.0 - update_rate) * rate + update_rate * proposal

    def _raw_local_readout(self, theta: Array, rate: Array) -> Array:
        features = jnp.concatenate([rate, phase_features(theta)], axis=-1)
        return _apply_positionwise(self.local.state_to_output, features)

    def _coarse_prediction_to_fine(
        self,
        coarse_theta: Array,
        coarse_rate: Array,
    ) -> Array:
        coarse_theta = self._coarse_theta_for_gate(coarse_theta)
        coarse_rate = self._coarse_rate_for_content(coarse_rate)
        features = jnp.concatenate([coarse_rate, phase_features(coarse_theta)], axis=-1)
        coarse_prediction = jnp.tanh(
            _apply_positionwise(self.coarse_state_to_patch, features)
        )
        weights = self._coarse_to_fine_weights_array(coarse_prediction.dtype)
        return jnp.einsum("fc,bcp->bfp", weights, coarse_prediction)

    def _readout(
        self,
        theta: Array,
        rate: Array,
        coarse_theta: Array,
        coarse_rate: Array,
    ) -> Array:
        local_logits = self._raw_local_readout(theta, rate)
        coarse_logits = self._coarse_prediction_to_fine(coarse_theta, coarse_rate)
        outputs = local_logits + self.coarse_readout_strength * coarse_logits
        return _apply_output_activation(outputs, self.local.output_activation)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        theta0, omega, rate0, rate_drive = self.local._initial_fields(images)
        (
            coarse_theta0,
            coarse_omega,
            coarse_rate0,
            coarse_rate_drive,
        ) = self._coarse_initial_fields(images)
        (
            final_theta,
            final_rate,
            final_coarse_theta,
            final_coarse_rate,
            thetas,
            rates,
            coarse_thetas,
            coarse_rates,
            energies,
            coarse_energies,
        ) = self._evolve(
            theta0,
            omega,
            rate0,
            rate_drive,
            coarse_theta0,
            coarse_omega,
            coarse_rate0,
            coarse_rate_drive,
        )
        local_readout = self._raw_local_readout(final_theta, final_rate)
        coarse_readout = self._coarse_prediction_to_fine(
            final_coarse_theta,
            final_coarse_rate,
        )
        reconstruction_sequence = self._readout(
            final_theta,
            final_rate,
            final_coarse_theta,
            final_coarse_rate,
        ).transpose(1, 0, 2)
        local_latent = jnp.mean(
            jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1),
            axis=1,
        )
        coarse_latent = jnp.concatenate(
            [
                final_coarse_rate.reshape(images.shape[0], -1),
                phase_features(final_coarse_theta).reshape(images.shape[0], -1),
            ],
            axis=-1,
        )
        return {
            "latent": jnp.concatenate([local_latent, coarse_latent], axis=-1),
            "reconstruction_sequence": reconstruction_sequence,
            "local_readout_sequence": local_readout.transpose(1, 0, 2),
            "coarse_readout_sequence": coarse_readout.transpose(1, 0, 2),
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": final_theta,
            "thetas": thetas,
            "energies": energies,
            "initial_rate": rate0,
            "final_rate": final_rate,
            "rate_states": rates,
            "rate_drive": rate_drive,
            "coarse_omega": coarse_omega,
            "initial_coarse_theta": coarse_theta0,
            "final_coarse_theta": final_coarse_theta,
            "coarse_thetas": coarse_thetas,
            "coarse_energies": coarse_energies,
            "initial_coarse_rate": coarse_rate0,
            "final_coarse_rate": final_coarse_rate,
            "coarse_rate_states": coarse_rates,
            "coarse_rate_drive": coarse_rate_drive,
            "coarse_readout_to_fine": coarse_readout,
            "fine_to_coarse_weights": self._fine_to_coarse_weights_array(
                images.dtype
            ),
            "coarse_to_fine_weights": self._coarse_to_fine_weights_array(
                images.dtype
            ),
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        theta0, omega, rate, rate_drive = self.local._initial_fields(images)
        (
            coarse_theta0,
            coarse_omega,
            coarse_rate0,
            coarse_rate_drive,
        ) = self._coarse_initial_fields(images)
        final_theta, final_rate, final_coarse_theta, final_coarse_rate, *_ = (
            self._evolve(
                theta0,
                omega,
                rate,
                rate_drive,
                coarse_theta0,
                coarse_omega,
                coarse_rate0,
                coarse_rate_drive,
            )
        )
        reconstruction = self._readout(
            final_theta,
            final_rate,
            final_coarse_theta,
            final_coarse_rate,
        ).transpose(1, 0, 2)
        return self.sequence_to_images(reconstruction, flatten=True)


class WinfreePriorRefinementPatchDenoiser(eqx.Module):
    """
    Feedforward semantic prior plus Winfree residual refinement.

    The feedforward branch predicts a coarse image prior. The Winfree rate-phase
    branch predicts a bounded residual correction. This tests whether
    oscillatory dynamics can improve hidden-region synthesis once a conventional
    prior already provides the semantic guess.
    """

    prior: FeedForwardPatchAutoencoder
    refiner: WinfreeGlobalRatePhaseConditionalPatchDenoiser

    refinement_strength: float = eqx.field(static=True)
    output_activation: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        feedforward_latent_output_skip: str = "sequence",
        feedforward_latent_output_skip_strength: float = 1.0,
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        global_gamma: float = 0.05,
        coupling_strength: float = 1.0,
        coupling_decay_length: Optional[float] = None,
        coupling_mode: str = "conv",
        coupling_kernel_size: int = 3,
        adaptive_coupling_strength: float = 0.1,
        input_conditioning_strength: float = 1.0,
        omega_scale: float = 1.0,
        initial_phase_scale: float = 0.1,
        phase_init: str = "learned",
        phase_init_scale: float = 1.0,
        field_activation: str = "relu",
        si_func: str = "mlp",
        si_hidden_ratio: int = 2,
        rate_kernel_size: int = 3,
        rate_update_rate: float = 0.5,
        rate_gate_strength: float = 1.0,
        global_gate_strength: float = 0.5,
        visibility_gate: str = "none",
        visibility_drive_floor: float = 0.0,
        missing_transport_strength: float = 1.0,
        refinement_strength: float = 0.25,
        output_activation: str = "identity",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        if refinement_strength < 0.0:
            raise ValueError("refinement_strength must be non-negative")
        if output_activation not in {"identity", "sigmoid", "tanh01"}:
            raise ValueError(
                "output_activation must be 'identity', 'sigmoid', or 'tanh01'"
            )

        keys = jax.random.split(key, 2)
        self.refinement_strength = float(refinement_strength)
        self.output_activation = output_activation
        self.prior = FeedForwardPatchAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            latent_output_skip=feedforward_latent_output_skip,
            latent_output_skip_strength=feedforward_latent_output_skip_strength,
            output_activation="identity",
            key=keys[0],
        )
        self.refiner = WinfreeGlobalRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            global_gamma=global_gamma,
            coupling_strength=coupling_strength,
            coupling_decay_length=coupling_decay_length,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            adaptive_coupling_strength=adaptive_coupling_strength,
            input_conditioning_strength=input_conditioning_strength,
            omega_scale=omega_scale,
            initial_phase_scale=initial_phase_scale,
            phase_init=phase_init,
            phase_init_scale=phase_init_scale,
            field_activation=field_activation,
            si_func=si_func,
            si_hidden_ratio=si_hidden_ratio,
            rate_kernel_size=rate_kernel_size,
            rate_update_rate=rate_update_rate,
            rate_gate_strength=rate_gate_strength,
            global_gate_strength=global_gate_strength,
            visibility_gate=visibility_gate,
            visibility_drive_floor=visibility_drive_floor,
            missing_transport_strength=missing_transport_strength,
            output_activation="identity",
            key=keys[1],
        )

    def images_to_sequence(self, images: Array) -> Array:
        return self.prior.images_to_sequence(images)

    def sequence_to_images(self, sequence: Array, flatten: bool = True) -> Array:
        return self.prior.sequence_to_images(sequence, flatten=flatten)

    def _raw_components(self, images: Array) -> Tuple[Array, Array, Array]:
        prior = self.prior(images)
        residual = jnp.tanh(self.refiner(images))
        combined = prior + self.refinement_strength * residual
        return prior, residual, combined

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        return jnp.concatenate(
            [
                self.prior.encode(images),
                self.refiner.encode(images),
            ],
            axis=-1,
        )

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        prior, residual, combined = self._raw_components(images)
        output = _apply_output_activation(combined, self.output_activation)
        trace = self.refiner.collect_trace(images)
        return {
            **{f"refiner_{key}": value for key, value in trace.items()},
            "latent": self.encode(images),
            "prior_reconstruction": prior,
            "residual_reconstruction": residual,
            "combined_reconstruction": combined,
            "reconstruction_sequence": _images_to_patch_sequence(
                output,
                self.prior.image_shape,
                self.prior.patch_shape,
                self.prior.output_channels,
            ),
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        _, _, combined = self._raw_components(images)
        return _apply_output_activation(combined, self.output_activation)


WONNPatchAutoencoder = WinfreePatchAutoencoder


__all__ = [
    "wrap_phase",
    "phase_features",
    "WinfreeFieldLayer",
    "WinfreeFieldEncoder",
    "WinfreeFieldDecoder",
    "WinfreeFieldAutoencoder",
    "WinfreePatchAutoencoder",
    "WinfreeConditionalPatchDenoiser",
    "WinfreeRatePhaseConditionalPatchDenoiser",
    "WinfreeGlobalRatePhaseConditionalPatchDenoiser",
    "WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser",
    "WinfreeCoarseRatePhaseConditionalPatchDenoiser",
    "WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser",
    "WinfreePriorRefinementPatchDenoiser",
    "WONNPatchAutoencoder",
]
