"""Artifact and checkpoint metadata helpers for MNIST generators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from oscnet.experiments.harness import ExperimentPaths, save_loss_curve, save_metrics_csv, write_json
from oscnet.models.generative.common import _image_hw_channels

from .common import Array
from .config import MNISTGeneratorExperimentConfig
from .metrics import InitialStateSampler, sample_generator_images

def _save_image_grid(
    images: Array,
    path: Path,
    *,
    image_shape: Tuple[int, ...],
    columns: int = 8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = _image_hw_channels(tuple(int(size) for size in image_shape))
    if channels not in (1, 3):
        raise ValueError("sample grids support grayscale or RGB images")
    images_np = np.asarray(images, dtype=np.float32).reshape(
        -1,
        channels,
        height,
        width,
    )
    if channels == 3:
        images_np = np.transpose(images_np, (0, 2, 3, 1))
    else:
        images_np = images_np[:, 0]
    rows = int(np.ceil(images_np.shape[0] / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows))
    axes_np = np.asarray(axes).reshape(rows, columns)
    for index, axis in enumerate(axes_np.flat):
        axis.axis("off")
        if index < images_np.shape[0]:
            image = np.clip(images_np[index], 0.0, 1.0)
            if channels == 1:
                axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(image, vmin=0, vmax=1)
    fig.tight_layout(pad=0.05)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_mnist_generator_artifacts(
    model: eqx.Module,
    real_images: Array,
    paths: ExperimentPaths,
    epoch: int,
    *,
    key: jax.random.PRNGKey,
    sample_count: int,
    batch_size: int,
    labels: Optional[Array] = None,
    initial_state_sampler: Optional[InitialStateSampler] = None,
) -> None:
    """Save generated samples and oscillator traces."""

    count = min(int(sample_count), int(real_images.shape[0]))
    generated = sample_generator_images(
        model,
        key=key,
        sample_count=count,
        batch_size=batch_size,
        labels=None if labels is None else labels[:count],
        initial_state_sampler=initial_state_sampler,
    )
    trace_labels = None if labels is None else labels[: min(count, batch_size)]
    trace = model.collect_trace(key, min(count, batch_size), trace_labels)
    np.savez(
        paths.artifacts / f"mnist_generator_samples_epoch_{epoch:03d}.npz",
        real=np.asarray(real_images[:count]),
        generated=np.asarray(generated),
        sample_initialization=np.asarray(
            "state_prior" if initial_state_sampler is not None else "white_noise"
        ),
        labels=(
            np.zeros((0,), dtype=np.int32)
            if labels is None
            else np.asarray(labels[:count])
        ),
    )
    np.savez(
        paths.traces / f"mnist_generator_trace_epoch_{epoch:03d}.npz",
        **{name: np.asarray(value) for name, value in trace.items()},
    )
    _save_image_grid(
        generated[: min(count, 64)],
        paths.plots / f"mnist_generator_samples_epoch_{epoch:03d}.png",
        image_shape=tuple(int(size) for size in model.image_shape),
    )


def _save_metrics_bundle(metrics: Dict[str, Any], paths: ExperimentPaths) -> None:
    write_json(paths.metrics / "history.json", metrics)
    save_metrics_csv(metrics, paths.metrics / "history.csv")
    save_loss_curve(metrics, paths.plots / "loss_curve.png")


def _checkpoint_hyperparams(config: MNISTGeneratorExperimentConfig) -> Dict[str, Any]:
    return {
        "experiment_family": "mnist_generator",
        "dataset_name": config.dataset_name,
        "data_source": config.data_source,
        "image_shape": config.image_shape,
        "model_family": config.model_family,
        "num_oscillators": config.num_oscillators,
        "decoder_hidden_dim": config.decoder_hidden_dim,
        "decoder_depth": config.decoder_depth,
        "steps": config.steps,
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
        "horn_frequency": config.horn_frequency,
        "horn_damping": config.horn_damping,
        "horn_nonlinearity": config.horn_nonlinearity,
        "horn_state_bound": config.horn_state_bound,
        "output_feedback_mode": config.output_feedback_mode,
        "output_feedback_strength": config.output_feedback_strength,
        "output_feedback_init_scale": config.output_feedback_init_scale,
        "output_feedback_basis_sigma": config.output_feedback_basis_sigma,
        "state_residual_readout_strength": config.state_residual_readout_strength,
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
        "multimode_num_modes": config.multimode_num_modes,
        "multimode_frequency_scales": config.multimode_frequency_scales,
        "multimode_mode_coupling_strength": (
            config.multimode_mode_coupling_strength
        ),
        "multimode_mode_coupling_profile": config.multimode_mode_coupling_profile,
        "num_coarse_oscillators": config.num_coarse_oscillators,
        "coarse_coupling_profile": config.coarse_coupling_profile,
        "coarse_coupling_normalization": config.coarse_coupling_normalization,
        "coarse_coupling_length_scale": config.coarse_coupling_length_scale,
        "coarse_to_fine_strength": config.coarse_to_fine_strength,
        "coarse_to_fine_profile": config.coarse_to_fine_profile,
        "coarse_to_fine_normalization": config.coarse_to_fine_normalization,
        "coarse_to_fine_length_scale": config.coarse_to_fine_length_scale,
        "coarse_to_fine_floor": config.coarse_to_fine_floor,
        "coarse_conditioning_strength": config.coarse_conditioning_strength,
        "multiscale_layer_sizes": config.multiscale_layer_sizes,
        "multiscale_frequency_scales": config.multiscale_frequency_scales,
        "multiscale_coupling_profile": config.multiscale_coupling_profile,
        "multiscale_coupling_normalization": (
            config.multiscale_coupling_normalization
        ),
        "multiscale_coupling_length_scale": config.multiscale_coupling_length_scale,
        "multiscale_coupling_floor": config.multiscale_coupling_floor,
        "multiscale_vertical_strength": config.multiscale_vertical_strength,
        "multiscale_feedback_strength": config.multiscale_feedback_strength,
        "multiscale_vertical_profile": config.multiscale_vertical_profile,
        "multiscale_vertical_normalization": config.multiscale_vertical_normalization,
        "multiscale_vertical_length_scale": config.multiscale_vertical_length_scale,
        "multiscale_vertical_floor": config.multiscale_vertical_floor,
        "multiscale_vertical_phase_lag": config.multiscale_vertical_phase_lag,
        "multiscale_feedback_phase_lag": config.multiscale_feedback_phase_lag,
        "multiscale_vertical_signal_scale": (
            config.multiscale_vertical_signal_scale
        ),
        "multiscale_feedback_signal_mode": config.multiscale_feedback_signal_mode,
        "multiscale_feedback_source_gate": config.multiscale_feedback_source_gate,
        "multiscale_feedback_source_mix": config.multiscale_feedback_source_mix,
        "multiscale_vertical_target_gate": config.multiscale_vertical_target_gate,
        "multiscale_vertical_soft_gate_floor": (
            config.multiscale_vertical_soft_gate_floor
        ),
        "multiscale_vertical_mode": config.multiscale_vertical_mode,
        "multiscale_vertical_gain_target": config.multiscale_vertical_gain_target,
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
        "multiscale_vertical_onset_step": config.multiscale_vertical_onset_step,
        "multiscale_vertical_ramp_steps": config.multiscale_vertical_ramp_steps,
        "multiscale_conditioning_strength": config.multiscale_conditioning_strength,
        "multiscale_auxiliary_readout_layer": (
            config.multiscale_auxiliary_readout_layer
        ),
        "multiscale_readout_fusion_strength": (
            config.multiscale_readout_fusion_strength
        ),
        "multiscale_readout_gate_mode": config.multiscale_readout_gate_mode,
        "multiscale_readout_gate_strength": (
            config.multiscale_readout_gate_strength
        ),
        "multiscale_readout_gate_init_scale": (
            config.multiscale_readout_gate_init_scale
        ),
        "coarse_auxiliary_weight": config.coarse_auxiliary_weight,
        "coarse_auxiliary_target_size": config.coarse_auxiliary_target_size,
        "coarse_auxiliary_loss_mode": config.coarse_auxiliary_loss_mode,
        "coarse_readout_consistency_weight": (
            config.coarse_readout_consistency_weight
        ),
        "coarse_readout_consistency_onset_epoch": (
            config.coarse_readout_consistency_onset_epoch
        ),
        "frequency_objective_weight": config.frequency_objective_weight,
        "frequency_objective_edge_weight": config.frequency_objective_edge_weight,
        "patch_objective_weight": config.patch_objective_weight,
        "patch_objective_patch_size": config.patch_objective_patch_size,
        "patch_objective_patch_sizes": config.patch_objective_patch_sizes,
        "patch_objective_stride": config.patch_objective_stride,
        "patch_objective_offsets": config.patch_objective_offsets,
        "patch_objective_projections": config.patch_objective_projections,
        "patch_objective_edge_weight": config.patch_objective_edge_weight,
        "train_recurrent_dynamics": config.train_recurrent_dynamics,
        "train_conditioning_dynamics": config.train_conditioning_dynamics,
        "conditional": config.conditional,
        "num_classes": config.num_classes,
        "label_phase_scale": config.label_phase_scale,
        "num_condition_oscillators": config.num_condition_oscillators,
        "conditioning_mode": config.conditioning_mode,
        "readout_mode": config.readout_mode,
        "decoder_mode": config.decoder_mode,
        "spatial_basis_sigma": config.spatial_basis_sigma,
        "local_patch_size": config.local_patch_size,
        "resize_conv_seed_size": config.resize_conv_seed_size,
        "resize_conv_upsamples": config.resize_conv_upsamples,
        "resize_conv_min_channels": config.resize_conv_min_channels,
        "resize_conv_seed_layout": config.resize_conv_seed_layout,
        "resize_conv_seed_min_channels": config.resize_conv_seed_min_channels,
        "output_activation": config.output_activation,
        "output_bias_init": config.output_bias_init,
        "num_projections": config.num_projections,
        "moment_weight": config.moment_weight,
        "pixel_marginal_weight": config.pixel_marginal_weight,
        "class_moment_weight": config.class_moment_weight,
        "prototype_weight": config.prototype_weight,
        "loss_mode": config.loss_mode,
        "pixel_drift_weight": config.pixel_drift_weight,
        "feature_drift_weight": config.feature_drift_weight,
        "feature_drift_mode": config.feature_drift_mode,
        "learned_feature_epochs": config.learned_feature_epochs,
        "learned_feature_kind": config.learned_feature_kind,
        "learned_feature_dim": config.learned_feature_dim,
        "learned_feature_depth": config.learned_feature_depth,
        "learned_feature_learning_rate": config.learned_feature_learning_rate,
        "learned_feature_weight_decay": config.learned_feature_weight_decay,
        "quality_classifier_epochs": config.quality_classifier_epochs,
        "quality_classifier_kind": config.quality_classifier_kind,
        "quality_classifier_dim": config.quality_classifier_dim,
        "quality_classifier_depth": config.quality_classifier_depth,
        "quality_classifier_learning_rate": config.quality_classifier_learning_rate,
        "quality_classifier_weight_decay": config.quality_classifier_weight_decay,
        "quality_classifier_train_limit": config.quality_classifier_train_limit,
        "quality_classifier_eval_limit": config.quality_classifier_eval_limit,
        "drift_queue_size": config.drift_queue_size,
        "drift_queue_num_pos": config.drift_queue_num_pos,
        "distributional_weight": config.distributional_weight,
        "drift_gamma": config.drift_gamma,
        "drift_temperatures": config.drift_temperatures,
        "state_fit_sample_count": config.state_fit_sample_count,
        "state_fit_steps": config.state_fit_steps,
        "state_fit_learning_rate": config.state_fit_learning_rate,
        "state_fit_init_scale": config.state_fit_init_scale,
        "state_fit_ridge": config.state_fit_ridge,
        "state_fit_settle_steps": config.state_fit_settle_steps,
        "state_anchor_weight": config.state_anchor_weight,
        "state_anchor_steps": config.state_anchor_steps,
        "state_anchor_noise_scale": config.state_anchor_noise_scale,
        "state_anchor_mode": config.state_anchor_mode,
        "state_anchor_encoder_kernel_size": config.state_anchor_encoder_kernel_size,
        "state_prior_sampling_mode": config.state_prior_sampling_mode,
        "state_prior_rank": config.state_prior_rank,
        "state_prior_noise_scale": config.state_prior_noise_scale,
        "state_prior_refresh_epochs": config.state_prior_refresh_epochs,
        "state_prior_start_epoch": config.state_prior_start_epoch,
        "train_settling_steps": config.train_settling_steps,
        "settling_steps": config.settling_steps,
        "attractor_variants_per_class": config.attractor_variants_per_class,
        "vertical_audit_modes": config.vertical_audit_modes,
        "vertical_audit_sample_count": config.vertical_audit_sample_count,
    }
