"""Wavelet-oriented oscillatory autoencoder models."""

from typing import Tuple

import equinox as eqx
import jax

from oscnet.core.oscillators import LearnableNonlinearHarmonicOscillator
from oscnet.models.oscillatory import (
    AmplitudeVelocityOscillatorCell,
    Array,
    OscillatoryAutoencoder,
)


def _wavelet_oscillator_params(
    omega_bounds: Tuple[float, float],
    gamma_bounds: Tuple[float, float],
):
    omega_min, omega_max = omega_bounds
    gamma_min, gamma_max = gamma_bounds
    return {
        "alpha": 0.08,
        "omega_init": (omega_min + omega_max) / 2,
        "gamma_init": (gamma_min + gamma_max) / 2,
        "omega_bounds": omega_bounds,
        "gamma_bounds": gamma_bounds,
        "dt": 0.01,
    }


class WaveletOscillatorCell(AmplitudeVelocityOscillatorCell):
    """Configured oscillator cell for wavelet feature sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        key: jax.random.PRNGKey,
        omega_bounds: Tuple[float, float] = (0.2, 6.0),
        gamma_bounds: Tuple[float, float] = (0.01, 0.15),
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            oscillator_class=LearnableNonlinearHarmonicOscillator,
            oscillator_params=_wavelet_oscillator_params(omega_bounds, gamma_bounds),
            gain_rec=0.3,
            initial_gain_multiplier=1.0,
            use_recurrent_velocity=False,
            readout_mode="position",
            key=key,
        )


class WaveletOscillatoryAutoencoder(eqx.Module):
    """
    Autoregressive oscillatory autoencoder configured for wavelet features.

    This is the reusable model behind the audio wavelet example. It keeps the
    old example-facing access paths, such as ``encoder_cell`` and
    ``decoder_cell``, while using the shared ``OscillatoryAutoencoder`` spine.
    """

    autoencoder: OscillatoryAutoencoder

    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    omega_bounds: Tuple[float, float] = eqx.field(static=True)
    gamma_bounds: Tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        *,
        key: jax.random.PRNGKey,
        omega_bounds: Tuple[float, float] = (0.2, 6.0),
        gamma_bounds: Tuple[float, float] = (0.01, 0.15),
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.omega_bounds = omega_bounds
        self.gamma_bounds = gamma_bounds

        self.autoencoder = OscillatoryAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            decoder_mode="autoregressive",
            oscillator_class=LearnableNonlinearHarmonicOscillator,
            oscillator_params=_wavelet_oscillator_params(omega_bounds, gamma_bounds),
            gain_rec=0.3,
            initial_gain_multiplier=1.0,
            use_recurrent_velocity=False,
            readout_mode="position",
            key=key,
        )

    def encode(self, x: Array) -> Array:
        return self.autoencoder.encode(x)

    def decode(self, latent: Array, target_length: int) -> Array:
        return self.autoencoder.decode(latent, sequence_length=target_length)

    def __call__(self, x: Array) -> Array:
        return self.autoencoder(x)

    @property
    def encoder_cell(self) -> AmplitudeVelocityOscillatorCell:
        return self.autoencoder.encoder.sequence.cell

    @property
    def decoder_cell(self) -> AmplitudeVelocityOscillatorCell:
        return self.autoencoder.decoder.cell

    @property
    def encode_projection(self) -> eqx.nn.Linear:
        return self.autoencoder.encoder.to_latent

    @property
    def decode_projection(self) -> eqx.nn.Linear:
        return self.autoencoder.decoder.latent_to_state


ProductionWaveletAutoencoder = WaveletOscillatoryAutoencoder


__all__ = [
    "WaveletOscillatorCell",
    "WaveletOscillatoryAutoencoder",
    "ProductionWaveletAutoencoder",
]
