"""Model construction for MNIST oscillator generator experiments."""

from __future__ import annotations

import equinox as eqx
import jax

from oscnet.models import (
    CoarseToFineHORNImageGenerator,
    HORNImageGenerator,
    KuramotoImageGenerator,
    StateMLPImageGenerator,
)

from .config import MNISTGeneratorExperimentConfig

def build_mnist_generator_model(
    config: MNISTGeneratorExperimentConfig,
    key: jax.random.PRNGKey,
) -> eqx.Module:
    """Build the requested oscillator generator or control."""

    if config.model_family in ("kuramoto", "horn", "coarse_horn", "state_mlp"):
        steps = config.steps
        train_dynamics = True
    elif config.model_family in (
        "decoder_only",
        "horn_decoder_only",
        "coarse_horn_decoder_only",
        "state_mlp_decoder_only",
    ):
        steps = 0
        train_dynamics = False
    elif config.model_family in (
        "frozen_kuramoto",
        "frozen_horn",
        "frozen_coarse_horn",
        "frozen_state_mlp",
    ):
        steps = config.steps
        train_dynamics = False
    else:
        raise ValueError(
            "model_family must be 'kuramoto', 'decoder_only', "
            "'frozen_kuramoto', 'horn', 'horn_decoder_only', 'frozen_horn', "
            "'coarse_horn', 'coarse_horn_decoder_only', 'frozen_coarse_horn', "
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
        "output_activation": config.output_activation,
        "output_bias_init": config.output_bias_init,
        "key": key,
    }
    if model_class in (HORNImageGenerator, CoarseToFineHORNImageGenerator):
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
    if model_class is StateMLPImageGenerator:
        model_kwargs.update(
            {
                "state_mlp_hidden_dim": config.state_mlp_hidden_dim,
                "state_mlp_depth": config.state_mlp_depth,
                "state_mlp_residual_scale": config.state_mlp_residual_scale,
                "horn_state_bound": config.horn_state_bound,
            }
        )
    return model_class(**model_kwargs)
