"""Configuration objects for constructing OscNet models."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type

import jax

from oscnet.core.oscillators import NonlinearHarmonicOscillator, Oscillator
from oscnet.models.generative import KuramotoImageGenerator
from oscnet.models.oscillatory import (
    ConvLSTMPatchDenoiser,
    FeedForwardPatchAutoencoder,
    OscillatoryAutoencoder,
    PatchOscillatoryAutoencoder,
    RecurrentConvPatchDenoiser,
    RecurrentConvPriorRefinementPatchDenoiser,
)
from oscnet.models.phase import WinfreePhaseAutoencoder
from oscnet.models.wavelet import WaveletOscillatoryAutoencoder
from oscnet.models.winfree import (
    WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser,
    WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser,
    WinfreeCoarseRatePhaseConditionalPatchDenoiser,
    WinfreeConditionalPatchDenoiser,
    WinfreeFieldAutoencoder,
    WinfreeGlobalRatePhaseConditionalPatchDenoiser,
    WinfreePatchAutoencoder,
    WinfreePriorRefinementPatchDenoiser,
    WinfreeRatePhaseConditionalPatchDenoiser,
)


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
    output_activation: str = "identity"

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
            output_activation=self.output_activation,
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
    output_activation: str = "identity"

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
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class FeedForwardPatchAutoencoderConfig:
    hidden_dim: int = 64
    latent_dim: int = 32
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (4, 4)
    latent_output_skip: str = "sequence"
    latent_output_skip_strength: float = 1.0
    output_activation: str = "identity"

    def build(self, key: jax.random.PRNGKey) -> FeedForwardPatchAutoencoder:
        return FeedForwardPatchAutoencoder(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            latent_output_skip=self.latent_output_skip,
            latent_output_skip_strength=self.latent_output_skip_strength,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class RecurrentConvPatchDenoiserConfig:
    hidden_dim: int = 64
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (4, 4)
    steps: int = 8
    kernel_size: int = 3
    residual_strength: float = 0.5
    output_activation: str = "identity"

    def build(self, key: jax.random.PRNGKey) -> RecurrentConvPatchDenoiser:
        return RecurrentConvPatchDenoiser(
            hidden_dim=self.hidden_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            steps=self.steps,
            kernel_size=self.kernel_size,
            residual_strength=self.residual_strength,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class RecurrentConvPriorRefinementPatchDenoiserConfig:
    input_dim: Optional[int] = None
    hidden_dim: int = 64
    latent_dim: int = 64
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (4, 4)
    feedforward_latent_output_skip: str = "sequence"
    feedforward_latent_output_skip_strength: float = 1.0
    steps: int = 8
    kernel_size: int = 3
    recurrent_residual_strength: float = 0.5
    refinement_strength: float = 0.5
    output_activation: str = "identity"

    def build(
        self,
        key: jax.random.PRNGKey,
    ) -> RecurrentConvPriorRefinementPatchDenoiser:
        return RecurrentConvPriorRefinementPatchDenoiser(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            feedforward_latent_output_skip=self.feedforward_latent_output_skip,
            feedforward_latent_output_skip_strength=(
                self.feedforward_latent_output_skip_strength
            ),
            steps=self.steps,
            kernel_size=self.kernel_size,
            recurrent_residual_strength=self.recurrent_residual_strength,
            refinement_strength=self.refinement_strength,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class KuramotoImageGeneratorConfig:
    num_oscillators: int = 64
    image_shape: Tuple[int, int] = (28, 28)
    decoder_hidden_dim: int = 128
    decoder_depth: int = 2
    steps: int = 8
    dt: float = 0.1
    coupling_strength: float = 1.0
    omega_scale: float = 0.2
    coupling_init_scale: float = 0.05
    coupling_profile: str = "dense"
    coupling_length_scale: float = 0.0
    coupling_floor: float = 0.0
    coupling_bias_strength: float = 0.0
    train_dynamics: bool = True
    train_recurrent_dynamics: Optional[bool] = None
    train_conditioning_dynamics: Optional[bool] = None
    num_classes: int = 0
    label_phase_scale: float = 0.5
    num_condition_oscillators: int = 0
    conditioning_mode: str = "phase_shift"
    readout_mode: str = "absolute"
    decoder_mode: str = "mlp"
    spatial_basis_sigma: float = 0.0
    local_patch_size: int = 5
    output_activation: str = "sigmoid"
    output_bias_init: Optional[float] = None

    def build(self, key: jax.random.PRNGKey) -> KuramotoImageGenerator:
        return KuramotoImageGenerator(
            num_oscillators=self.num_oscillators,
            image_shape=self.image_shape,
            decoder_hidden_dim=self.decoder_hidden_dim,
            decoder_depth=self.decoder_depth,
            steps=self.steps,
            dt=self.dt,
            coupling_strength=self.coupling_strength,
            omega_scale=self.omega_scale,
            coupling_init_scale=self.coupling_init_scale,
            coupling_profile=self.coupling_profile,
            coupling_length_scale=self.coupling_length_scale,
            coupling_floor=self.coupling_floor,
            coupling_bias_strength=self.coupling_bias_strength,
            train_dynamics=self.train_dynamics,
            train_recurrent_dynamics=self.train_recurrent_dynamics,
            train_conditioning_dynamics=self.train_conditioning_dynamics,
            num_classes=self.num_classes,
            label_phase_scale=self.label_phase_scale,
            num_condition_oscillators=self.num_condition_oscillators,
            conditioning_mode=self.conditioning_mode,
            readout_mode=self.readout_mode,
            decoder_mode=self.decoder_mode,
            spatial_basis_sigma=self.spatial_basis_sigma,
            local_patch_size=self.local_patch_size,
            output_activation=self.output_activation,
            output_bias_init=self.output_bias_init,
            key=key,
        )


@dataclass(frozen=True)
class ConvLSTMPatchDenoiserConfig:
    input_dim: Optional[int] = None
    hidden_dim: int = 64
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (4, 4)
    steps: int = 8
    kernel_size: int = 3
    forget_bias: float = 1.0
    output_activation: str = "identity"

    def build(self, key: jax.random.PRNGKey) -> ConvLSTMPatchDenoiser:
        return ConvLSTMPatchDenoiser(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            steps=self.steps,
            kernel_size=self.kernel_size,
            forget_bias=self.forget_bias,
            output_activation=self.output_activation,
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


@dataclass(frozen=True)
class WinfreeFieldAutoencoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    output_dim: Optional[int] = None
    sequence_length: int = 16
    grid_shape: Optional[Tuple[int, int]] = None
    group_size: int = 1
    steps: int = 8
    gamma: float = 0.1
    coupling_strength: float = 1.0
    coupling_decay_length: Optional[float] = None
    coupling_mode: str = "matrix"
    coupling_kernel_size: int = 3
    adaptive_coupling_strength: float = 0.1
    latent_conditioning_strength: float = 1.0
    latent_readout: str = "none"
    latent_readout_strength: float = 1.0
    latent_output_skip: str = "none"
    latent_output_skip_strength: float = 1.0
    omega_scale: float = 1.0
    field_activation: str = "relu"
    si_func: str = "trig"
    si_hidden_ratio: int = 2
    output_activation: str = "identity"

    def build(self, key: jax.random.PRNGKey) -> WinfreeFieldAutoencoder:
        return WinfreeFieldAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            grid_shape=self.grid_shape,
            group_size=self.group_size,
            steps=self.steps,
            gamma=self.gamma,
            coupling_strength=self.coupling_strength,
            coupling_decay_length=self.coupling_decay_length,
            coupling_mode=self.coupling_mode,
            coupling_kernel_size=self.coupling_kernel_size,
            adaptive_coupling_strength=self.adaptive_coupling_strength,
            latent_conditioning_strength=self.latent_conditioning_strength,
            latent_readout=self.latent_readout,
            latent_readout_strength=self.latent_readout_strength,
            latent_output_skip=self.latent_output_skip,
            latent_output_skip_strength=self.latent_output_skip_strength,
            omega_scale=self.omega_scale,
            field_activation=self.field_activation,
            si_func=self.si_func,
            si_hidden_ratio=self.si_hidden_ratio,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class WinfreePatchAutoencoderConfig:
    hidden_dim: int = 64
    latent_dim: int = 32
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (7, 7)
    group_size: int = 1
    steps: int = 8
    gamma: float = 0.1
    coupling_strength: float = 1.0
    coupling_decay_length: Optional[float] = None
    coupling_mode: str = "matrix"
    coupling_kernel_size: int = 3
    adaptive_coupling_strength: float = 0.1
    latent_conditioning_strength: float = 1.0
    latent_readout: str = "none"
    latent_readout_strength: float = 1.0
    latent_output_skip: str = "none"
    latent_output_skip_strength: float = 1.0
    omega_scale: float = 1.0
    field_activation: str = "relu"
    si_func: str = "trig"
    si_hidden_ratio: int = 2
    output_activation: str = "identity"

    def build(self, key: jax.random.PRNGKey) -> WinfreePatchAutoencoder:
        return WinfreePatchAutoencoder(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            group_size=self.group_size,
            steps=self.steps,
            gamma=self.gamma,
            coupling_strength=self.coupling_strength,
            coupling_decay_length=self.coupling_decay_length,
            coupling_mode=self.coupling_mode,
            coupling_kernel_size=self.coupling_kernel_size,
            adaptive_coupling_strength=self.adaptive_coupling_strength,
            latent_conditioning_strength=self.latent_conditioning_strength,
            latent_readout=self.latent_readout,
            latent_readout_strength=self.latent_readout_strength,
            latent_output_skip=self.latent_output_skip,
            latent_output_skip_strength=self.latent_output_skip_strength,
            omega_scale=self.omega_scale,
            field_activation=self.field_activation,
            si_func=self.si_func,
            si_hidden_ratio=self.si_hidden_ratio,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class WinfreeConditionalPatchDenoiserConfig:
    hidden_dim: int = 64
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (4, 4)
    group_size: int = 1
    steps: int = 8
    gamma: float = 0.1
    coupling_strength: float = 1.0
    coupling_decay_length: Optional[float] = None
    coupling_mode: str = "conv"
    coupling_kernel_size: int = 3
    adaptive_coupling_strength: float = 0.1
    input_conditioning_strength: float = 1.0
    omega_scale: float = 1.0
    phase_init: str = "learned"
    phase_init_scale: float = 1.0
    field_activation: str = "relu"
    si_func: str = "mlp"
    si_hidden_ratio: int = 2
    output_activation: str = "identity"

    def build(self, key: jax.random.PRNGKey) -> WinfreeConditionalPatchDenoiser:
        return WinfreeConditionalPatchDenoiser(
            hidden_dim=self.hidden_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            group_size=self.group_size,
            steps=self.steps,
            gamma=self.gamma,
            coupling_strength=self.coupling_strength,
            coupling_decay_length=self.coupling_decay_length,
            coupling_mode=self.coupling_mode,
            coupling_kernel_size=self.coupling_kernel_size,
            adaptive_coupling_strength=self.adaptive_coupling_strength,
            input_conditioning_strength=self.input_conditioning_strength,
            omega_scale=self.omega_scale,
            phase_init=self.phase_init,
            phase_init_scale=self.phase_init_scale,
            field_activation=self.field_activation,
            si_func=self.si_func,
            si_hidden_ratio=self.si_hidden_ratio,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class WinfreeRatePhaseConditionalPatchDenoiserConfig:
    input_dim: Optional[int] = None
    hidden_dim: int = 64
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (4, 4)
    group_size: int = 1
    steps: int = 8
    gamma: float = 0.1
    coupling_strength: float = 1.0
    coupling_decay_length: Optional[float] = None
    coupling_mode: str = "conv"
    coupling_kernel_size: int = 3
    adaptive_coupling_strength: float = 0.1
    input_conditioning_strength: float = 1.0
    omega_scale: float = 1.0
    phase_init: str = "learned"
    phase_init_scale: float = 1.0
    field_activation: str = "relu"
    si_func: str = "mlp"
    si_hidden_ratio: int = 2
    rate_kernel_size: int = 3
    rate_update_rate: float = 0.5
    rate_gate_strength: float = 1.0
    visibility_gate: str = "none"
    visibility_drive_floor: float = 0.0
    missing_transport_strength: float = 1.0
    output_activation: str = "identity"

    def build(self, key: jax.random.PRNGKey) -> WinfreeRatePhaseConditionalPatchDenoiser:
        return WinfreeRatePhaseConditionalPatchDenoiser(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            group_size=self.group_size,
            steps=self.steps,
            gamma=self.gamma,
            coupling_strength=self.coupling_strength,
            coupling_decay_length=self.coupling_decay_length,
            coupling_mode=self.coupling_mode,
            coupling_kernel_size=self.coupling_kernel_size,
            adaptive_coupling_strength=self.adaptive_coupling_strength,
            input_conditioning_strength=self.input_conditioning_strength,
            omega_scale=self.omega_scale,
            phase_init=self.phase_init,
            phase_init_scale=self.phase_init_scale,
            field_activation=self.field_activation,
            si_func=self.si_func,
            si_hidden_ratio=self.si_hidden_ratio,
            rate_kernel_size=self.rate_kernel_size,
            rate_update_rate=self.rate_update_rate,
            rate_gate_strength=self.rate_gate_strength,
            visibility_gate=self.visibility_gate,
            visibility_drive_floor=self.visibility_drive_floor,
            missing_transport_strength=self.missing_transport_strength,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class WinfreeGlobalRatePhaseConditionalPatchDenoiserConfig:
    input_dim: Optional[int] = None
    hidden_dim: int = 64
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (4, 4)
    group_size: int = 1
    steps: int = 8
    gamma: float = 0.1
    global_gamma: float = 0.05
    coupling_strength: float = 1.0
    coupling_decay_length: Optional[float] = None
    coupling_mode: str = "conv"
    coupling_kernel_size: int = 3
    adaptive_coupling_strength: float = 0.1
    input_conditioning_strength: float = 1.0
    omega_scale: float = 1.0
    phase_init: str = "learned"
    phase_init_scale: float = 1.0
    field_activation: str = "relu"
    si_func: str = "mlp"
    si_hidden_ratio: int = 2
    rate_kernel_size: int = 3
    rate_update_rate: float = 0.5
    rate_gate_strength: float = 1.0
    visibility_gate: str = "none"
    visibility_drive_floor: float = 0.0
    missing_transport_strength: float = 1.0
    global_gate_strength: float = 0.5
    output_activation: str = "identity"

    def build(
        self,
        key: jax.random.PRNGKey,
    ) -> WinfreeGlobalRatePhaseConditionalPatchDenoiser:
        return WinfreeGlobalRatePhaseConditionalPatchDenoiser(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            group_size=self.group_size,
            steps=self.steps,
            gamma=self.gamma,
            global_gamma=self.global_gamma,
            coupling_strength=self.coupling_strength,
            coupling_decay_length=self.coupling_decay_length,
            coupling_mode=self.coupling_mode,
            coupling_kernel_size=self.coupling_kernel_size,
            adaptive_coupling_strength=self.adaptive_coupling_strength,
            input_conditioning_strength=self.input_conditioning_strength,
            omega_scale=self.omega_scale,
            phase_init=self.phase_init,
            phase_init_scale=self.phase_init_scale,
            field_activation=self.field_activation,
            si_func=self.si_func,
            si_hidden_ratio=self.si_hidden_ratio,
            rate_kernel_size=self.rate_kernel_size,
            rate_update_rate=self.rate_update_rate,
            rate_gate_strength=self.rate_gate_strength,
            visibility_gate=self.visibility_gate,
            visibility_drive_floor=self.visibility_drive_floor,
            missing_transport_strength=self.missing_transport_strength,
            global_gate_strength=self.global_gate_strength,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiserConfig:
    input_dim: Optional[int] = None
    hidden_dim: int = 64
    image_shape: Tuple[int, int] = (28, 28)
    patch_shape: Tuple[int, int] = (4, 4)
    coarse_grid_shape: Tuple[int, int] = (2, 2)
    group_size: int = 1
    steps: int = 8
    gamma: float = 0.1
    global_gamma: float = 0.05
    coupling_strength: float = 1.0
    coupling_decay_length: Optional[float] = None
    coupling_mode: str = "conv"
    coupling_kernel_size: int = 3
    adaptive_coupling_strength: float = 0.1
    input_conditioning_strength: float = 1.0
    omega_scale: float = 1.0
    phase_init: str = "learned"
    phase_init_scale: float = 1.0
    field_activation: str = "relu"
    si_func: str = "mlp"
    si_hidden_ratio: int = 2
    rate_kernel_size: int = 3
    rate_update_rate: float = 0.5
    rate_gate_strength: float = 1.0
    global_gate_strength: float = 0.5
    global_phase_control: str = "none"
    output_activation: str = "identity"

    def build(
        self,
        key: jax.random.PRNGKey,
    ) -> WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser:
        return WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            coarse_grid_shape=self.coarse_grid_shape,
            group_size=self.group_size,
            steps=self.steps,
            gamma=self.gamma,
            global_gamma=self.global_gamma,
            coupling_strength=self.coupling_strength,
            coupling_decay_length=self.coupling_decay_length,
            coupling_mode=self.coupling_mode,
            coupling_kernel_size=self.coupling_kernel_size,
            adaptive_coupling_strength=self.adaptive_coupling_strength,
            input_conditioning_strength=self.input_conditioning_strength,
            omega_scale=self.omega_scale,
            phase_init=self.phase_init,
            phase_init_scale=self.phase_init_scale,
            field_activation=self.field_activation,
            si_func=self.si_func,
            si_hidden_ratio=self.si_hidden_ratio,
            rate_kernel_size=self.rate_kernel_size,
            rate_update_rate=self.rate_update_rate,
            rate_gate_strength=self.rate_gate_strength,
            global_gate_strength=self.global_gate_strength,
            global_phase_control=self.global_phase_control,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class WinfreeCoarseRatePhaseConditionalPatchDenoiserConfig(
    WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiserConfig
):
    global_content_strength: float = 0.5
    global_content_control: str = "none"

    def build(
        self,
        key: jax.random.PRNGKey,
    ) -> WinfreeCoarseRatePhaseConditionalPatchDenoiser:
        return WinfreeCoarseRatePhaseConditionalPatchDenoiser(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            coarse_grid_shape=self.coarse_grid_shape,
            group_size=self.group_size,
            steps=self.steps,
            gamma=self.gamma,
            global_gamma=self.global_gamma,
            coupling_strength=self.coupling_strength,
            coupling_decay_length=self.coupling_decay_length,
            coupling_mode=self.coupling_mode,
            coupling_kernel_size=self.coupling_kernel_size,
            adaptive_coupling_strength=self.adaptive_coupling_strength,
            input_conditioning_strength=self.input_conditioning_strength,
            omega_scale=self.omega_scale,
            phase_init=self.phase_init,
            phase_init_scale=self.phase_init_scale,
            field_activation=self.field_activation,
            si_func=self.si_func,
            si_hidden_ratio=self.si_hidden_ratio,
            rate_kernel_size=self.rate_kernel_size,
            rate_update_rate=self.rate_update_rate,
            rate_gate_strength=self.rate_gate_strength,
            global_gate_strength=self.global_gate_strength,
            global_phase_control=self.global_phase_control,
            global_content_strength=self.global_content_strength,
            global_content_control=self.global_content_control,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiserConfig(
    WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiserConfig
):
    global_content_control: str = "none"
    coarse_readout_strength: float = 0.5

    def build(
        self,
        key: jax.random.PRNGKey,
    ) -> WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser:
        return WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            coarse_grid_shape=self.coarse_grid_shape,
            group_size=self.group_size,
            steps=self.steps,
            gamma=self.gamma,
            global_gamma=self.global_gamma,
            coupling_strength=self.coupling_strength,
            coupling_decay_length=self.coupling_decay_length,
            coupling_mode=self.coupling_mode,
            coupling_kernel_size=self.coupling_kernel_size,
            adaptive_coupling_strength=self.adaptive_coupling_strength,
            input_conditioning_strength=self.input_conditioning_strength,
            omega_scale=self.omega_scale,
            phase_init=self.phase_init,
            phase_init_scale=self.phase_init_scale,
            field_activation=self.field_activation,
            si_func=self.si_func,
            si_hidden_ratio=self.si_hidden_ratio,
            rate_kernel_size=self.rate_kernel_size,
            rate_update_rate=self.rate_update_rate,
            rate_gate_strength=self.rate_gate_strength,
            global_gate_strength=self.global_gate_strength,
            global_phase_control=self.global_phase_control,
            global_content_control=self.global_content_control,
            coarse_readout_strength=self.coarse_readout_strength,
            output_activation=self.output_activation,
            key=key,
        )


@dataclass(frozen=True)
class WinfreePriorRefinementPatchDenoiserConfig(
    WinfreeGlobalRatePhaseConditionalPatchDenoiserConfig
):
    latent_dim: int = 64
    feedforward_latent_output_skip: str = "sequence"
    feedforward_latent_output_skip_strength: float = 1.0
    refinement_strength: float = 0.25

    def build(
        self,
        key: jax.random.PRNGKey,
    ) -> WinfreePriorRefinementPatchDenoiser:
        return WinfreePriorRefinementPatchDenoiser(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            image_shape=self.image_shape,
            patch_shape=self.patch_shape,
            feedforward_latent_output_skip=self.feedforward_latent_output_skip,
            feedforward_latent_output_skip_strength=(
                self.feedforward_latent_output_skip_strength
            ),
            group_size=self.group_size,
            steps=self.steps,
            gamma=self.gamma,
            global_gamma=self.global_gamma,
            coupling_strength=self.coupling_strength,
            coupling_decay_length=self.coupling_decay_length,
            coupling_mode=self.coupling_mode,
            coupling_kernel_size=self.coupling_kernel_size,
            adaptive_coupling_strength=self.adaptive_coupling_strength,
            input_conditioning_strength=self.input_conditioning_strength,
            omega_scale=self.omega_scale,
            phase_init=self.phase_init,
            phase_init_scale=self.phase_init_scale,
            field_activation=self.field_activation,
            si_func=self.si_func,
            si_hidden_ratio=self.si_hidden_ratio,
            rate_kernel_size=self.rate_kernel_size,
            rate_update_rate=self.rate_update_rate,
            rate_gate_strength=self.rate_gate_strength,
            global_gate_strength=self.global_gate_strength,
            visibility_gate=self.visibility_gate,
            visibility_drive_floor=self.visibility_drive_floor,
            missing_transport_strength=self.missing_transport_strength,
            refinement_strength=self.refinement_strength,
            output_activation=self.output_activation,
            key=key,
        )


__all__ = [
    "OscillatoryAutoencoderConfig",
    "PatchOscillatoryAutoencoderConfig",
    "FeedForwardPatchAutoencoderConfig",
    "RecurrentConvPatchDenoiserConfig",
    "RecurrentConvPriorRefinementPatchDenoiserConfig",
    "KuramotoImageGeneratorConfig",
    "ConvLSTMPatchDenoiserConfig",
    "WaveletAutoencoderConfig",
    "WinfreePhaseAutoencoderConfig",
    "WinfreeFieldAutoencoderConfig",
    "WinfreePatchAutoencoderConfig",
    "WinfreeConditionalPatchDenoiserConfig",
    "WinfreeRatePhaseConditionalPatchDenoiserConfig",
    "WinfreeGlobalRatePhaseConditionalPatchDenoiserConfig",
    "WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiserConfig",
    "WinfreeCoarseRatePhaseConditionalPatchDenoiserConfig",
    "WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiserConfig",
    "WinfreePriorRefinementPatchDenoiserConfig",
]
