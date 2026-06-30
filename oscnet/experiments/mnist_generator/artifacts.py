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
from .metrics import sample_generator_images

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
) -> None:
    """Save generated samples and oscillator traces."""

    count = min(int(sample_count), int(real_images.shape[0]))
    generated = sample_generator_images(
        model,
        key=key,
        sample_count=count,
        batch_size=batch_size,
        labels=None if labels is None else labels[:count],
    )
    trace_labels = None if labels is None else labels[: min(count, batch_size)]
    trace = model.collect_trace(key, min(count, batch_size), trace_labels)
    np.savez(
        paths.artifacts / f"mnist_generator_samples_epoch_{epoch:03d}.npz",
        real=np.asarray(real_images[:count]),
        generated=np.asarray(generated),
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
        "multiscale_conditioning_strength": config.multiscale_conditioning_strength,
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
        "train_settling_steps": config.train_settling_steps,
        "settling_steps": config.settling_steps,
        "attractor_variants_per_class": config.attractor_variants_per_class,
    }
