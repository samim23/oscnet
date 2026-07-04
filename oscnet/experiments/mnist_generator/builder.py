"""Model construction for MNIST oscillator generator experiments."""

from __future__ import annotations

import equinox as eqx
import jax

from oscnet.models import (
    CoarseToFineHORNImageGenerator,
    HORNImageGenerator,
    KuramotoImageGenerator,
    MultiscaleHORNImageGenerator,
    MultiModeHORNImageGenerator,
    StateMLPImageGenerator,
)

from .config import MNISTGeneratorExperimentConfig

def build_mnist_generator_model(
    config: MNISTGeneratorExperimentConfig,
    key: jax.random.PRNGKey,
) -> eqx.Module:
    """Build the requested oscillator generator or control."""

    if config.model_family in (
        "kuramoto",
        "horn",
        "coarse_horn",
        "multiscale_horn",
        "multimode_horn",
        "state_mlp",
    ):
        steps = config.steps
        train_dynamics = True
    elif config.model_family in (
        "decoder_only",
        "horn_decoder_only",
        "coarse_horn_decoder_only",
        "multiscale_horn_decoder_only",
        "multimode_horn_decoder_only",
        "state_mlp_decoder_only",
    ):
        steps = 0
        train_dynamics = False
    elif config.model_family in (
        "frozen_kuramoto",
        "frozen_horn",
        "frozen_coarse_horn",
        "frozen_multiscale_horn",
        "frozen_multimode_horn",
        "frozen_state_mlp",
    ):
        steps = config.steps
        train_dynamics = False
    else:
        raise ValueError(
            "model_family must be 'kuramoto', 'decoder_only', "
            "'frozen_kuramoto', 'horn', 'horn_decoder_only', 'frozen_horn', "
            "'coarse_horn', 'coarse_horn_decoder_only', 'frozen_coarse_horn', "
            "'multiscale_horn', 'multiscale_horn_decoder_only', "
            "'frozen_multiscale_horn', 'multimode_horn', "
            "'multimode_horn_decoder_only', 'frozen_multimode_horn', "
            "'state_mlp', 'state_mlp_decoder_only', or 'frozen_state_mlp'"
        )
    train_recurrent_dynamics = (
        train_dynamics
        if config.train_recurrent_dynamics is None
        else bool(config.train_recurrent_dynamics)
    )
    train_conditioning_dynamics = (
        train_dynamics
        if config.train_conditioning_dynamics is None
        else bool(config.train_conditioning_dynamics)
    )

    if config.model_family in (
        "coarse_horn",
        "coarse_horn_decoder_only",
        "frozen_coarse_horn",
    ):
        model_class = CoarseToFineHORNImageGenerator
    elif config.model_family in (
        "multiscale_horn",
        "multiscale_horn_decoder_only",
        "frozen_multiscale_horn",
    ):
        model_class = MultiscaleHORNImageGenerator
    elif config.model_family in (
        "multimode_horn",
        "multimode_horn_decoder_only",
        "frozen_multimode_horn",
    ):
        model_class = MultiModeHORNImageGenerator
    elif config.model_family in ("horn", "horn_decoder_only", "frozen_horn"):
        model_class = HORNImageGenerator
    elif config.model_family in (
        "state_mlp",
        "state_mlp_decoder_only",
        "frozen_state_mlp",
    ):
        model_class = StateMLPImageGenerator
    else:
        model_class = KuramotoImageGenerator
    model_kwargs = {
        "num_oscillators": config.num_oscillators,
        "decoder_hidden_dim": config.decoder_hidden_dim,
        "decoder_depth": config.decoder_depth,
        "steps": steps,
        "dt": config.dt,
        "coupling_strength": config.coupling_strength,
        "main_coupling_strength": config.main_coupling_strength,
        "omega_scale": config.omega_scale,
        "coupling_init_scale": config.coupling_init_scale,
        "coupling_profile": config.coupling_profile,
        "coupling_normalization": config.coupling_normalization,
        "coupling_length_scale": config.coupling_length_scale,
        "coupling_floor": config.coupling_floor,
        "coupling_bias_strength": config.coupling_bias_strength,
        "conditioning_strength": config.conditioning_strength,
        "conditioning_target_fraction": config.conditioning_target_fraction,
        "conditioning_target_pattern": config.conditioning_target_pattern,
        "train_dynamics": train_dynamics,
        "train_recurrent_dynamics": train_recurrent_dynamics,
        "train_conditioning_dynamics": train_conditioning_dynamics,
        "num_classes": config.num_classes if config.conditional else 0,
        "label_phase_scale": config.label_phase_scale,
        "num_condition_oscillators": (
            config.num_condition_oscillators if config.conditional else 0
        ),
        "conditioning_mode": config.conditioning_mode if config.conditional else "none",
        "readout_mode": config.readout_mode,
        "decoder_mode": config.decoder_mode,
        "image_shape": config.image_shape,
        "spatial_basis_sigma": config.spatial_basis_sigma,
        "local_patch_size": config.local_patch_size,
        "resize_conv_seed_shape": (
            config.resize_conv_seed_size,
            config.resize_conv_seed_size,
        ),
        "resize_conv_upsamples": config.resize_conv_upsamples,
        "resize_conv_min_channels": config.resize_conv_min_channels,
        "resize_conv_seed_layout": config.resize_conv_seed_layout,
        "resize_conv_seed_min_channels": config.resize_conv_seed_min_channels,
        "output_activation": config.output_activation,
        "output_bias_init": config.output_bias_init,
        "key": key,
    }
    if model_class in (
        HORNImageGenerator,
        CoarseToFineHORNImageGenerator,
        MultiscaleHORNImageGenerator,
        MultiModeHORNImageGenerator,
    ):
        model_kwargs.update(
            {
                "horn_frequency": config.horn_frequency,
                "horn_damping": config.horn_damping,
                "horn_nonlinearity": config.horn_nonlinearity,
                "horn_state_bound": config.horn_state_bound,
                "output_feedback_mode": config.output_feedback_mode,
                "output_feedback_strength": config.output_feedback_strength,
                "output_feedback_init_scale": config.output_feedback_init_scale,
                "output_feedback_basis_sigma": config.output_feedback_basis_sigma,
                "state_residual_readout_strength": (
                    config.state_residual_readout_strength
                ),
                "state_residual_readout_init_scale": (
                    config.state_residual_readout_init_scale
                ),
                "state_residual_readout_patch_size": (
                    config.state_residual_readout_patch_size
                ),
                "state_residual_readout_sigma": config.state_residual_readout_sigma,
                "resonant_readout_strength": config.resonant_readout_strength,
                "resonant_readout_init_scale": config.resonant_readout_init_scale,
                "resonant_readout_patch_size": config.resonant_readout_patch_size,
                "resonant_readout_sigma": config.resonant_readout_sigma,
                "state_anchor_encoder_enabled": (
                    config.state_anchor_weight > 0.0
                    and config.state_anchor_mode != "none"
                ),
                "state_anchor_encoder_kernel_size": (
                    config.state_anchor_encoder_kernel_size
                ),
            }
        )
    if model_class is MultiModeHORNImageGenerator:
        model_kwargs.update(
            {
                "num_modes": config.multimode_num_modes,
                "mode_frequency_scales": config.multimode_frequency_scales,
                "mode_coupling_strength": (
                    config.multimode_mode_coupling_strength
                ),
                "mode_coupling_profile": config.multimode_mode_coupling_profile,
            }
        )
    if model_class is CoarseToFineHORNImageGenerator:
        model_kwargs.update(
            {
                "num_coarse_oscillators": config.num_coarse_oscillators,
                "coarse_coupling_profile": config.coarse_coupling_profile,
                "coarse_coupling_normalization": (
                    config.coarse_coupling_normalization
                ),
                "coarse_coupling_length_scale": (
                    config.coarse_coupling_length_scale
                ),
                "coarse_to_fine_strength": config.coarse_to_fine_strength,
                "coarse_to_fine_profile": config.coarse_to_fine_profile,
                "coarse_to_fine_normalization": config.coarse_to_fine_normalization,
                "coarse_to_fine_length_scale": config.coarse_to_fine_length_scale,
                "coarse_to_fine_floor": config.coarse_to_fine_floor,
                "coarse_conditioning_strength": (
                    config.coarse_conditioning_strength
                ),
            }
        )
    if model_class is MultiscaleHORNImageGenerator:
        model_kwargs.update(
            {
                "multiscale_layer_sizes": config.multiscale_layer_sizes,
                "multiscale_frequency_scales": config.multiscale_frequency_scales,
                "multiscale_coupling_profile": config.multiscale_coupling_profile,
                "multiscale_coupling_normalization": (
                    config.multiscale_coupling_normalization
                ),
                "multiscale_coupling_length_scale": (
                    config.multiscale_coupling_length_scale
                ),
                "multiscale_coupling_floor": config.multiscale_coupling_floor,
                "multiscale_vertical_strength": config.multiscale_vertical_strength,
                "multiscale_feedback_strength": config.multiscale_feedback_strength,
                "multiscale_vertical_profile": config.multiscale_vertical_profile,
                "multiscale_vertical_normalization": (
                    config.multiscale_vertical_normalization
                ),
                "multiscale_vertical_length_scale": (
                    config.multiscale_vertical_length_scale
                ),
                "multiscale_vertical_floor": config.multiscale_vertical_floor,
                "multiscale_vertical_phase_lag": config.multiscale_vertical_phase_lag,
                "multiscale_feedback_phase_lag": (
                    config.multiscale_feedback_phase_lag
                ),
                "multiscale_vertical_signal_scale": (
                    config.multiscale_vertical_signal_scale
                ),
                "multiscale_feedback_signal_mode": (
                    config.multiscale_feedback_signal_mode
                ),
                "multiscale_feedback_source_gate": (
                    config.multiscale_feedback_source_gate
                ),
                "multiscale_feedback_source_mix": (
                    config.multiscale_feedback_source_mix
                ),
                "multiscale_vertical_target_gate": (
                    config.multiscale_vertical_target_gate
                ),
                "multiscale_vertical_soft_gate_floor": (
                    config.multiscale_vertical_soft_gate_floor
                ),
                "multiscale_vertical_mode": config.multiscale_vertical_mode,
                "multiscale_vertical_gain_target": (
                    config.multiscale_vertical_gain_target
                ),
                "multiscale_vertical_gain_normalization": (
                    config.multiscale_vertical_gain_normalization
                ),
                "multiscale_vertical_gain_target_std": (
                    config.multiscale_vertical_gain_target_std
                ),
                "multiscale_vertical_broad_gain_scale": (
                    config.multiscale_vertical_broad_gain_scale
                ),
                "multiscale_vertical_selective_gain_scale": (
                    config.multiscale_vertical_selective_gain_scale
                ),
                "multiscale_vertical_schedule": config.multiscale_vertical_schedule,
                "multiscale_vertical_onset_step": (
                    config.multiscale_vertical_onset_step
                ),
                "multiscale_vertical_ramp_steps": (
                    config.multiscale_vertical_ramp_steps
                ),
                "multiscale_conditioning_strength": (
                    config.multiscale_conditioning_strength
                ),
                "multiscale_auxiliary_readout_size": (
                    config.coarse_auxiliary_target_size
                ),
                "multiscale_auxiliary_readout_layer": (
                    config.multiscale_auxiliary_readout_layer
                ),
                "multiscale_readout_fusion_strength": (
                    config.multiscale_readout_fusion_strength
                ),
                "multiscale_readout_gate_mode": (
                    config.multiscale_readout_gate_mode
                ),
                "multiscale_readout_gate_strength": (
                    config.multiscale_readout_gate_strength
                ),
                "multiscale_readout_gate_init_scale": (
                    config.multiscale_readout_gate_init_scale
                ),
            }
        )
    if model_class is StateMLPImageGenerator:
        state_mlp_kwargs = {
            "state_mlp_hidden_dim": config.state_mlp_hidden_dim,
            "state_mlp_depth": config.state_mlp_depth,
            "state_mlp_residual_scale": config.state_mlp_residual_scale,
            "horn_state_bound": config.horn_state_bound,
            "output_feedback_mode": config.output_feedback_mode,
            "output_feedback_strength": config.output_feedback_strength,
            "output_feedback_init_scale": config.output_feedback_init_scale,
            "output_feedback_basis_sigma": config.output_feedback_basis_sigma,
            "state_residual_readout_strength": (
                config.state_residual_readout_strength
            ),
            "state_residual_readout_init_scale": (
                config.state_residual_readout_init_scale
            ),
            "state_residual_readout_patch_size": (
                config.state_residual_readout_patch_size
            ),
            "state_residual_readout_sigma": config.state_residual_readout_sigma,
            "resonant_readout_strength": config.resonant_readout_strength,
            "resonant_readout_init_scale": config.resonant_readout_init_scale,
            "resonant_readout_patch_size": config.resonant_readout_patch_size,
            "resonant_readout_sigma": config.resonant_readout_sigma,
            "state_anchor_encoder_enabled": (
                config.state_anchor_weight > 0.0
                and config.state_anchor_mode != "none"
            ),
            "state_anchor_encoder_kernel_size": (
                config.state_anchor_encoder_kernel_size
            ),
        }
        if (
            config.decoder_mode == "resize_conv"
            and config.resize_conv_seed_layout == "retinotopic"
        ):
            seed_sites = int(config.resize_conv_seed_size) ** 2
            if config.num_oscillators % seed_sites != 0:
                raise ValueError(
                    "retinotopic StateMLP requires num_oscillators to be "
                    "a multiple of resize_conv_seed_size ** 2"
                )
            state_mlp_kwargs.update(
                {
                    "num_spatial_sites": seed_sites,
                    "num_modes": config.num_oscillators // seed_sites,
                }
            )
        model_kwargs.update(
            state_mlp_kwargs
        )
    return model_class(**model_kwargs)
