"""WONN-inspired Winfree phase-field model components."""

import math
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

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


def _infer_grid_shape(num_positions: int) -> Tuple[int, int]:
    side = math.isqrt(num_positions)
    if side * side == num_positions:
        return side, side
    return num_positions, 1


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
    channel_mix: eqx.nn.Linear
    coupling: Array

    num_positions: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    grid_shape: Tuple[int, int] = eqx.field(static=True)
    group_grid_shape: Tuple[int, int] = eqx.field(static=True)
    group_size: int = eqx.field(static=True)
    coupling_positions: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)
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

    def _activate_field(self, field: Array) -> Array:
        if self.field_activation == "identity":
            return field
        if self.field_activation == "tanh":
            return jnp.tanh(field)
        return jax.nn.relu(field)

    def coupled_field(self, influence: Array) -> Array:
        """Map ``I(theta)`` to a coupling field."""

        field = jnp.einsum("ij,bjc->bic", self.coupling, influence)
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
    theta_to_output: eqx.nn.Linear
    layer: WinfreeFieldLayer
    positional_omega: Array
    positional_theta: Array

    latent_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    sequence_length: int = eqx.field(static=True)
    latent_conditioning_strength: float = eqx.field(static=True)
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
        latent_conditioning_strength: float = 1.0,
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

        keys = jax.random.split(key, 6)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.sequence_length = int(sequence_length)
        self.latent_conditioning_strength = float(latent_conditioning_strength)
        self.omega_scale = float(omega_scale)
        self.output_activation = output_activation

        self.latent_to_omega = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[0])
        self.latent_to_theta = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[1])
        self.theta_to_output = eqx.nn.Linear(2 * hidden_dim, output_dim, key=keys[2])
        self.layer = WinfreeFieldLayer(
            num_positions=sequence_length,
            channels=hidden_dim,
            grid_shape=grid_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
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

    def _readout(self, theta: Array) -> Array:
        outputs = _apply_positionwise(self.theta_to_output, phase_features(theta))
        if self.output_activation == "sigmoid":
            return jax.nn.sigmoid(outputs)
        if self.output_activation == "tanh01":
            return 0.5 * (jnp.tanh(outputs) + 1.0)
        return outputs

    def decode_with_trace(self, latent: Array) -> Tuple[Array, Dict[str, Array]]:
        theta0, omega = self._initial_fields(latent)
        trajectory = self.layer(theta0, omega, return_trajectory=True)
        outputs = self._readout(trajectory["final_theta"]).transpose(1, 0, 2)
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
        return self._readout(final_theta).transpose(1, 0, 2)


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
        latent_conditioning_strength: float = 1.0,
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
            latent_conditioning_strength=latent_conditioning_strength,
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
        latent_conditioning_strength: float = 1.0,
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
            latent_conditioning_strength=latent_conditioning_strength,
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


WONNPatchAutoencoder = WinfreePatchAutoencoder


__all__ = [
    "wrap_phase",
    "phase_features",
    "WinfreeFieldLayer",
    "WinfreeFieldEncoder",
    "WinfreeFieldDecoder",
    "WinfreeFieldAutoencoder",
    "WinfreePatchAutoencoder",
    "WONNPatchAutoencoder",
]
