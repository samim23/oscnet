"""Configuration objects for constructing OscNet models."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type

import jax

from oscnet.core.oscillators import NonlinearHarmonicOscillator, Oscillator
from oscnet.models.oscillatory import OscillatoryAutoencoder, PatchOscillatoryAutoencoder
from oscnet.models.phase import WinfreePhaseAutoencoder
from oscnet.models.wavelet import WaveletOscillatoryAutoencoder


@dataclass(frozen=True)
class OscillatoryAutoencoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    output_dim: Optional[int] = None
    sequence_length: Optional[int] = None
    decoder_mode: str = "repeat"
    oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator
    oscillator_params: Dict[str, Any] = field(default_factory=dict)
    gain_rec: Optional[float] = None
    initial_gain_multiplier: Optional[float] = None
    use_recurrent_velocity: bool = True
    readout_mode: str = "amplitude_velocity"
    latent_conditioning_strength: float = 1.0
    initial_amplitude: float = 0.1

    def build(self, key: jax.random.PRNGKey) -> OscillatoryAutoencoder:
        return OscillatoryAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            decoder_mode=self.decoder_mode,
            oscillator_class=self.oscillator_class,
            oscillator_params=dict(self.oscillator_params),
            gain_rec=self.gain_rec,
            initial_gain_multiplier=self.initial_gain_multiplier,
            use_recurrent_velocity=self.use_recurrent_velocity,
            readout_mode=self.readout_mode,
            latent_conditioning_strength=self.latent_conditioning_strength,
            initial_amplitude=self.initial_amplitude,
            key=key,
        )


@dataclass(frozen=True)
class PatchOscillatoryAutoencoderConfig:
    hidden_dim: int = 64
    latent_dim: int = 32
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (4, 4)
    decoder_mode: str = "repeat"
    oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator
    oscillator_params: Dict[str, Any] = field(default_factory=dict)
    gain_rec: Optional[float] = None
    initial_gain_multiplier: Optional[float] = None
    use_recurrent_velocity: bool = True
    readout_mode: str = "amplitude_velocity"
    latent_conditioning_strength: float = 1.0
    initial_amplitude: float = 0.1

    def build(self, key: jax.random.PRNGKey) -> PatchOscillatoryAutoencoder:
        return PatchOscillatoryAutoencoder(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            decoder_mode=self.decoder_mode,
            oscillator_class=self.oscillator_class,
            oscillator_params=dict(self.oscillator_params),
            gain_rec=self.gain_rec,
            initial_gain_multiplier=self.initial_gain_multiplier,
            use_recurrent_velocity=self.use_recurrent_velocity,
            readout_mode=self.readout_mode,
            latent_conditioning_strength=self.latent_conditioning_strength,
            initial_amplitude=self.initial_amplitude,
            key=key,
        )


@dataclass(frozen=True)
class WaveletAutoencoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    omega_bounds: Tuple[float, float] = (0.2, 6.0)
    gamma_bounds: Tuple[float, float] = (0.01, 0.15)

    def build(self, key: jax.random.PRNGKey) -> WaveletOscillatoryAutoencoder:
        return WaveletOscillatoryAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            omega_bounds=self.omega_bounds,
            gamma_bounds=self.gamma_bounds,
            key=key,
        )


@dataclass(frozen=True)
class WinfreePhaseAutoencoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    sequence_length: Optional[int] = None
    omega: float = 1.0
    dt: float = 0.05
    input_gain: float = 0.2
    coupling_strength: float = 0.1
    pulse_exponent: float = 1.0
    phase_response_bias: float = 1.0

    def build(self, key: jax.random.PRNGKey) -> WinfreePhaseAutoencoder:
        return WinfreePhaseAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            sequence_length=self.sequence_length,
            omega=self.omega,
            dt=self.dt,
            input_gain=self.input_gain,
            coupling_strength=self.coupling_strength,
            pulse_exponent=self.pulse_exponent,
            phase_response_bias=self.phase_response_bias,
            key=key,
        )


__all__ = [
    "OscillatoryAutoencoderConfig",
    "PatchOscillatoryAutoencoderConfig",
    "WaveletAutoencoderConfig",
    "WinfreePhaseAutoencoderConfig",
]
