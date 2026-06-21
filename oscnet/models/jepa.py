"""Patch-grid representation predictors for JEPA-style probes."""

from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.models.oscillatory import (
    ConvLSTMPatchDenoiser,
    FeedForwardPatchAutoencoder,
    RecurrentConvPatchDenoiser,
)
from oscnet.models.winfree import (
    WinfreeGlobalRatePhaseConditionalPatchDenoiser,
    WinfreeRatePhaseConditionalPatchDenoiser,
    _apply_positionwise,
    phase_features,
)

Array = jnp.ndarray


def _flatten_prediction_sequence(sequence_bne: Array) -> Array:
    """Flatten a batch-major patch representation sequence."""

    return sequence_bne.reshape(sequence_bne.shape[0], -1)


class FeedForwardPatchJEPAPredictor(eqx.Module):
    """Feedforward latent baseline that predicts per-patch target embeddings."""

    base: FeedForwardPatchAutoencoder
    hidden_to_embedding: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        embedding_dim: int = 8,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        self.base = FeedForwardPatchAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            output_activation="identity",
            key=keys[0],
        )
        self.hidden_to_embedding = eqx.nn.Linear(
            hidden_dim,
            embedding_dim,
            key=keys[1],
        )
        self.embedding_dim = int(embedding_dim)

    def _hidden_sequence(self, images: Array) -> Tuple[Array, Array]:
        sequence = self.base.images_to_sequence(images)
        latent, _ = self.base._encode_sequence(sequence)
        hidden = jax.vmap(self.base.latent_to_hidden_sequence)(latent)
        hidden = hidden.reshape(
            latent.shape[0],
            self.base.num_patches,
            self.base.hidden_dim,
        )
        hidden = jax.nn.relu(
            hidden + self.base.decoder_positional_hidden[None, :, :]
        )
        return latent, hidden

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        latent, _ = self._hidden_sequence(images)
        return latent

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        latent, hidden = self._hidden_sequence(images)
        prediction = _apply_positionwise(self.hidden_to_embedding, hidden)
        return {
            "latent": latent,
            "prediction_sequence": prediction.transpose(1, 0, 2),
            "decoder_hidden": hidden,
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        _, hidden = self._hidden_sequence(images)
        prediction = _apply_positionwise(self.hidden_to_embedding, hidden)
        return _flatten_prediction_sequence(prediction)


class RecurrentConvPatchJEPAPredictor(eqx.Module):
    """Tied recurrent-conv baseline that predicts patch embeddings."""

    base: RecurrentConvPatchDenoiser
    hidden_to_embedding: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        embedding_dim: int = 8,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        steps: int = 8,
        kernel_size: int = 3,
        residual_strength: float = 0.5,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        self.base = RecurrentConvPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            steps=steps,
            kernel_size=kernel_size,
            residual_strength=residual_strength,
            output_activation="identity",
            key=keys[0],
        )
        self.hidden_to_embedding = eqx.nn.Linear(
            hidden_dim,
            embedding_dim,
            key=keys[1],
        )
        self.embedding_dim = int(embedding_dim)

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        hidden = self.base._initial_hidden(images)
        final_hidden = self.base._evolve(hidden)
        return jnp.mean(final_hidden, axis=1)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        hidden = self.base._initial_hidden(images)
        final_hidden, hidden_states = self.base._evolve(
            hidden,
            return_trajectory=True,
        )
        prediction = _apply_positionwise(self.hidden_to_embedding, final_hidden)
        return {
            "latent": jnp.mean(final_hidden, axis=1),
            "prediction_sequence": prediction.transpose(1, 0, 2),
            "initial_hidden": hidden,
            "final_hidden": final_hidden,
            "hidden_states": hidden_states,
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        hidden = self.base._initial_hidden(images)
        final_hidden = self.base._evolve(hidden)
        prediction = _apply_positionwise(self.hidden_to_embedding, final_hidden)
        return _flatten_prediction_sequence(prediction)


class ConvLSTMPatchJEPAPredictor(eqx.Module):
    """ConvLSTM baseline that predicts patch embeddings."""

    base: ConvLSTMPatchDenoiser
    hidden_to_embedding: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        embedding_dim: int = 8,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        steps: int = 8,
        kernel_size: int = 3,
        forget_bias: float = 1.0,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        self.base = ConvLSTMPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            steps=steps,
            kernel_size=kernel_size,
            forget_bias=forget_bias,
            output_activation="identity",
            key=keys[0],
        )
        self.hidden_to_embedding = eqx.nn.Linear(
            hidden_dim,
            embedding_dim,
            key=keys[1],
        )
        self.embedding_dim = int(embedding_dim)

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        drive = self.base._drive(images)
        final_hidden, _ = self.base._evolve(drive)
        return jnp.mean(final_hidden, axis=1)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        drive = self.base._drive(images)
        (final_hidden, final_cell), hidden_states, cell_states = self.base._evolve(
            drive,
            return_trajectory=True,
        )
        prediction = _apply_positionwise(self.hidden_to_embedding, final_hidden)
        return {
            "latent": jnp.mean(final_hidden, axis=1),
            "prediction_sequence": prediction.transpose(1, 0, 2),
            "drive": drive,
            "final_hidden": final_hidden,
            "final_cell": final_cell,
            "hidden_states": hidden_states,
            "cell_states": cell_states,
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        drive = self.base._drive(images)
        final_hidden, _ = self.base._evolve(drive)
        prediction = _apply_positionwise(self.hidden_to_embedding, final_hidden)
        return _flatten_prediction_sequence(prediction)


class WinfreeRatePhasePatchJEPAPredictor(eqx.Module):
    """Local Winfree rate-phase field that predicts patch embeddings."""

    base: WinfreeRatePhaseConditionalPatchDenoiser
    state_to_embedding: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        embedding_dim: int = 8,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        coupling_strength: float = 1.0,
        coupling_mode: str = "conv",
        coupling_kernel_size: int = 3,
        si_func: str = "mlp",
        rate_kernel_size: int = 3,
        rate_update_rate: float = 0.5,
        rate_gate_strength: float = 1.0,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        self.base = WinfreeRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            coupling_strength=coupling_strength,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            si_func=si_func,
            rate_kernel_size=rate_kernel_size,
            rate_update_rate=rate_update_rate,
            rate_gate_strength=rate_gate_strength,
            output_activation="identity",
            key=keys[0],
        )
        self.state_to_embedding = eqx.nn.Linear(
            3 * hidden_dim,
            embedding_dim,
            key=keys[1],
        )
        self.embedding_dim = int(embedding_dim)

    def _final_fields(self, images: Array):
        theta0, omega, rate, rate_drive = self.base._initial_fields(images)
        visibility = self.base._visibility_from_inputs(images)
        return self.base._evolve(theta0, omega, rate, rate_drive, visibility)

    def _predict_from_fields(self, theta: Array, rate: Array) -> Array:
        features = jnp.concatenate([rate, phase_features(theta)], axis=-1)
        return _apply_positionwise(self.state_to_embedding, features)

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        final_theta, final_rate, *_ = self._final_fields(images)
        features = jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1)
        return jnp.mean(features, axis=1)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        theta0, omega, rate0, rate_drive = self.base._initial_fields(images)
        visibility = self.base._visibility_from_inputs(images)
        final_theta, final_rate, thetas, rates, energies = self.base._evolve(
            theta0,
            omega,
            rate0,
            rate_drive,
            visibility,
        )
        prediction = self._predict_from_fields(final_theta, final_rate)
        features = jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1)
        return {
            "latent": jnp.mean(features, axis=1),
            "prediction_sequence": prediction.transpose(1, 0, 2),
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": final_theta,
            "thetas": thetas,
            "energies": energies,
            "initial_rate": rate0,
            "final_rate": final_rate,
            "rate_states": rates,
            "rate_drive": rate_drive,
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        final_theta, final_rate, *_ = self._final_fields(images)
        prediction = self._predict_from_fields(final_theta, final_rate)
        return _flatten_prediction_sequence(prediction)


class WinfreeGlobalRatePhasePatchJEPAPredictor(eqx.Module):
    """Slow/global Winfree rate-phase field that predicts patch embeddings."""

    base: WinfreeGlobalRatePhaseConditionalPatchDenoiser
    state_to_embedding: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        embedding_dim: int = 8,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        group_size: int = 1,
        steps: int = 8,
        gamma: float = 0.1,
        global_gamma: float = 0.05,
        coupling_strength: float = 1.0,
        coupling_mode: str = "conv",
        coupling_kernel_size: int = 3,
        si_func: str = "mlp",
        rate_kernel_size: int = 3,
        rate_update_rate: float = 0.5,
        rate_gate_strength: float = 1.0,
        global_gate_strength: float = 0.5,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        self.base = WinfreeGlobalRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            patch_shape=patch_shape,
            group_size=group_size,
            steps=steps,
            gamma=gamma,
            global_gamma=global_gamma,
            coupling_strength=coupling_strength,
            coupling_mode=coupling_mode,
            coupling_kernel_size=coupling_kernel_size,
            si_func=si_func,
            rate_kernel_size=rate_kernel_size,
            rate_update_rate=rate_update_rate,
            rate_gate_strength=rate_gate_strength,
            global_gate_strength=global_gate_strength,
            output_activation="identity",
            key=keys[0],
        )
        self.state_to_embedding = eqx.nn.Linear(
            3 * hidden_dim,
            embedding_dim,
            key=keys[1],
        )
        self.embedding_dim = int(embedding_dim)

    def _final_fields(self, images: Array):
        theta0, omega, rate, rate_drive = self.base.local._initial_fields(images)
        visibility = self.base.local._visibility_from_inputs(images)
        global_theta0, global_omega = self.base._global_initial_fields(images)
        return self.base._evolve(
            theta0,
            omega,
            rate,
            rate_drive,
            global_theta0,
            global_omega,
            visibility,
        )

    def _predict_from_fields(self, theta: Array, rate: Array) -> Array:
        features = jnp.concatenate([rate, phase_features(theta)], axis=-1)
        return _apply_positionwise(self.state_to_embedding, features)

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        final_theta, final_rate, *_ = self._final_fields(images)
        features = jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1)
        return jnp.mean(features, axis=1)

    def collect_trace(self, images: Array) -> Dict[str, Array]:
        theta0, omega, rate0, rate_drive = self.base.local._initial_fields(images)
        visibility = self.base.local._visibility_from_inputs(images)
        global_theta0, global_omega = self.base._global_initial_fields(images)
        (
            final_theta,
            final_rate,
            final_global_theta,
            thetas,
            rates,
            global_thetas,
            energies,
            global_energies,
        ) = self.base._evolve(
            theta0,
            omega,
            rate0,
            rate_drive,
            global_theta0,
            global_omega,
            visibility,
        )
        prediction = self._predict_from_fields(final_theta, final_rate)
        features = jnp.concatenate([final_rate, phase_features(final_theta)], axis=-1)
        return {
            "latent": jnp.mean(features, axis=1),
            "prediction_sequence": prediction.transpose(1, 0, 2),
            "omega": omega,
            "initial_theta": theta0,
            "final_theta": final_theta,
            "thetas": thetas,
            "energies": energies,
            "initial_rate": rate0,
            "final_rate": final_rate,
            "rate_states": rates,
            "rate_drive": rate_drive,
            "global_omega": global_omega,
            "initial_global_theta": global_theta0,
            "final_global_theta": final_global_theta,
            "global_thetas": global_thetas,
            "global_energies": global_energies,
        }

    def __call__(self, images: Array, use_phase_init: bool = False) -> Array:
        del use_phase_init
        final_theta, final_rate, *_ = self._final_fields(images)
        prediction = self._predict_from_fields(final_theta, final_rate)
        return _flatten_prediction_sequence(prediction)


__all__ = [
    "FeedForwardPatchJEPAPredictor",
    "RecurrentConvPatchJEPAPredictor",
    "ConvLSTMPatchJEPAPredictor",
    "WinfreeRatePhasePatchJEPAPredictor",
    "WinfreeGlobalRatePhasePatchJEPAPredictor",
]
