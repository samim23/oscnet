"""Sample quality, settling, and success diagnostics for MNIST generators."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .common import Array
from .features import FeatureClassifier

def sample_generator_images(
    model: eqx.Module,
    *,
    key: jax.random.PRNGKey,
    sample_count: int,
    batch_size: int,
    labels: Optional[Array] = None,
) -> Array:
    """Generate a requested number of images in batches."""

    generated = []
    remaining = int(sample_count)
    batch_index = 0
    start = 0
    while remaining > 0:
        current = min(batch_size, remaining)
        label_batch = None if labels is None else labels[start : start + current]
        generated.append(
            model(jax.random.fold_in(key, batch_index), current, label_batch)
        )
        remaining -= current
        start += current
        batch_index += 1
    return jnp.concatenate(generated, axis=0)


def compute_generator_quality_metrics(
    real_images: Array,
    generated: Array,
    *,
    labels: Optional[Array] = None,
    prototypes: Optional[Array] = None,
    classifier: Optional[FeatureClassifier] = None,
) -> Dict[str, float]:
    """Compute lightweight distribution and diversity diagnostics."""

    real = np.asarray(real_images, dtype=np.float32).reshape(real_images.shape[0], -1)
    gen = np.asarray(generated, dtype=np.float32).reshape(generated.shape[0], -1)
    real = real[: gen.shape[0]]

    clipped = np.clip(gen, 0.0, 1.0)
    real_mean = real.mean(axis=0)
    gen_mean = clipped.mean(axis=0)
    real_std = real.std(axis=0)
    gen_std = clipped.std(axis=0)

    pairwise = np.mean((clipped[:, None, :] - real[None, :, :]) ** 2, axis=-1)
    nearest_real_mse = np.min(pairwise, axis=1)
    real_pairwise = np.mean((real[:, None, :] - real[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(real_pairwise, np.inf)

    metrics = {
        "generated_mean": float(np.mean(clipped)),
        "generated_std": float(np.std(clipped)),
        "generated_min": float(np.min(gen)),
        "generated_max": float(np.max(gen)),
        "pixel_mean_mse": float(np.mean((gen_mean - real_mean) ** 2)),
        "pixel_std_mse": float(np.mean((gen_std - real_std) ** 2)),
        "diversity_ratio": float(np.mean(gen_std) / (np.mean(real_std) + 1e-8)),
        "nearest_real_mse": float(np.mean(nearest_real_mse)),
        "real_nearest_real_mse": float(np.mean(np.min(real_pairwise, axis=1))),
    }
    if labels is not None and prototypes is not None:
        labels_np = np.asarray(labels[: gen.shape[0]], dtype=np.int32)
        prototypes_np = np.asarray(prototypes, dtype=np.float32)
        proto_mse = np.mean(
            (clipped[:, None, :] - prototypes_np[None, :, :]) ** 2,
            axis=-1,
        )
        predicted_label = np.argmin(proto_mse, axis=1)
        own_proto_mse = proto_mse[np.arange(labels_np.shape[0]), labels_np]
        metrics.update(
            {
                "prototype_mse": float(np.mean(own_proto_mse)),
                "prototype_nearest_accuracy": float(
                    np.mean(predicted_label == labels_np)
                ),
            }
        )
    if labels is not None and classifier is not None:
        labels_jnp = labels[: gen.shape[0]].astype(jnp.int32)
        logits = classifier(jnp.asarray(clipped, dtype=jnp.float32))
        probabilities = jax.nn.softmax(logits, axis=-1)
        predicted = jnp.argmax(probabilities, axis=-1)
        intended_probability = probabilities[
            jnp.arange(labels_jnp.shape[0]),
            labels_jnp,
        ]
        entropy = -jnp.sum(
            probabilities * jnp.log(jnp.maximum(probabilities, 1e-8)),
            axis=-1,
        )
        metrics.update(
            {
                "classifier_label_accuracy": float(
                    jnp.mean((predicted == labels_jnp).astype(jnp.float32))
                ),
                "classifier_label_confidence": float(jnp.mean(intended_probability)),
                "classifier_max_confidence": float(
                    jnp.mean(jnp.max(probabilities, axis=-1))
                ),
                "classifier_entropy": float(jnp.mean(entropy)),
            }
        )
    return metrics


def _model_with_steps(model: eqx.Module, steps: int) -> eqx.Module:
    """Return a view of ``model`` with a different static settling depth."""

    stepped_model = copy.copy(model)
    object.__setattr__(stepped_model, "steps", int(steps))
    return stepped_model


def compute_generator_settling_metrics(
    model: eqx.Module,
    *,
    key: jax.random.PRNGKey,
    real_images: Array,
    sample_count: int,
    batch_size: int,
    settling_steps: Sequence[int],
    labels: Optional[Array] = None,
    prototypes: Optional[Array] = None,
    classifier: Optional[FeatureClassifier] = None,
) -> Dict[str, Any]:
    """Score one trained generator at multiple test-time settling depths."""

    steps = tuple(dict.fromkeys(int(step) for step in settling_steps))
    if not steps:
        return {}
    if any(step < 0 for step in steps):
        raise ValueError("settling_steps must be non-negative")

    count = min(int(sample_count), int(real_images.shape[0]))
    label_slice = None if labels is None else labels[:count]
    by_step: Dict[str, Dict[str, float]] = {}
    for step in steps:
        step_model = _model_with_steps(model, step)
        generated = sample_generator_images(
            step_model,
            key=key,
            sample_count=count,
            batch_size=batch_size,
            labels=label_slice,
        )
        by_step[f"step_{step:03d}"] = compute_generator_quality_metrics(
            real_images[:count],
            generated,
            labels=label_slice,
            prototypes=prototypes,
            classifier=classifier,
        )

    first_key = f"step_{steps[0]:03d}"
    last_key = f"step_{steps[-1]:03d}"
    metrics: Dict[str, Any] = {
        "steps": [int(step) for step in steps],
        "by_step": by_step,
    }
    for metric_name in (
        "classifier_label_accuracy",
        "classifier_label_confidence",
        "prototype_nearest_accuracy",
        "diversity_ratio",
        "nearest_real_mse",
        "pixel_mean_mse",
        "pixel_std_mse",
    ):
        values = [
            (step, by_step[f"step_{step:03d}"].get(metric_name))
            for step in steps
            if metric_name in by_step[f"step_{step:03d}"]
        ]
        if not values:
            continue
        best_step, best_value = max(values, key=lambda item: float(item[1]))
        if metric_name in ("nearest_real_mse", "pixel_mean_mse", "pixel_std_mse"):
            best_step, best_value = min(values, key=lambda item: float(item[1]))
        first_value = by_step[first_key].get(metric_name)
        last_value = by_step[last_key].get(metric_name)
        metrics[f"{metric_name}_best_step"] = int(best_step)
        metrics[f"{metric_name}_best"] = float(best_value)
        if first_value is not None and last_value is not None:
            metrics[f"{metric_name}_last_minus_first"] = float(
                last_value - first_value
            )
    return metrics


def _array_size(value: Optional[Array]) -> int:
    if value is None:
        return 0
    return int(np.prod(tuple(value.shape)))


def compute_generator_success_diagnostics(
    model: eqx.Module,
    *,
    trace: Optional[Dict[str, Array]] = None,
    sample_count: int = 0,
    total_train_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Compute attribution and efficiency proxies for generator comparisons.

    These are digital-simulation diagnostics, not hardware energy claims. They
    exist to make oscillator results harder to over-credit when a conventional
    decoder or a frozen reservoir explains most of the behavior.
    """

    decoder_params = sum(
        _array_size(layer.weight) + _array_size(layer.bias)
        for layer in model.decoder_layers
    )
    decoder_params += _array_size(model.spatial_phase_weights) + _array_size(
        model.spatial_output_bias
    )
    decoder_params += _array_size(model.local_patch_weights)
    decoder_params += sum(
        _array_size(layer.weight) + _array_size(layer.bias)
        for layer in model.resize_conv_layers
    )
    if model.resize_conv_output is not None:
        decoder_params += _array_size(model.resize_conv_output.weight)
        decoder_params += _array_size(model.resize_conv_output.bias)
    transition_layers = tuple(getattr(model, "transition_layers", ()))
    transition_params = sum(
        _array_size(layer.weight) + _array_size(layer.bias)
        for layer in transition_layers
    )
    dynamics_family = str(getattr(model, "dynamics_family", "kuramoto"))
    recurrent_params = (
        int(transition_params)
        if dynamics_family == "state_mlp"
        else _array_size(model.omega) + _array_size(model.coupling)
    )
    conditioning_params = (
        _array_size(model.label_phase_shift)
        + _array_size(model.label_condition_phase)
        + _array_size(model.condition_omega)
        + _array_size(model.condition_coupling)
        + _array_size(model.label_condition_coupling)
    )
    total_params = int(decoder_params + recurrent_params + conditioning_params)
    trainable_main_recurrent_params = (
        int(recurrent_params) if model.train_recurrent_dynamics else 0
    )
    trainable_conditioning_params = (
        int(conditioning_params) if model.train_conditioning_dynamics else 0
    )
    trainable_recurrent_params = (
        trainable_main_recurrent_params + trainable_conditioning_params
    )
    trainable_total_params = int(decoder_params + trainable_recurrent_params)
    n = int(model.num_oscillators)
    coupling_profile = np.asarray(model.coupling_profile_matrix(), dtype=np.float32)
    effective_coupling = np.asarray(model.coupling, dtype=np.float32) * coupling_profile
    coupling_nonzero = int(np.count_nonzero(effective_coupling))
    coupling_possible = max(n * (n - 1), 1)
    off_diagonal_profile = coupling_profile[~np.eye(n, dtype=bool)]
    if off_diagonal_profile.size == 0:
        off_diagonal_profile = np.asarray([0.0], dtype=np.float32)

    condition_n = int(model.num_condition_oscillators)
    if dynamics_family == "state_mlp":
        transition_ops = sum(
            int(layer.in_features * layer.out_features)
            for layer in transition_layers
        )
        estimated_recurrent_ops_per_sample = int(model.steps * transition_ops)
    else:
        estimated_recurrent_ops_per_sample = int(
            model.steps
            * (
                n * n
                + (
                    condition_n * condition_n + n * condition_n
                    if model.conditioning_mode == "class_oscillator"
                    else 0
                )
            )
        )
    estimated_decoder_ops_per_sample = int(
        sum(layer.in_features * layer.out_features for layer in model.decoder_layers)
    )
    if model.decoder_mode == "spatial_basis":
        estimated_decoder_ops_per_sample = int(
            2 * model.num_oscillators
            + model.num_oscillators * model.image_dim
        )
    elif model.decoder_mode == "local_basis":
        patch_area = int(model.local_patch_size * model.local_patch_size)
        estimated_decoder_ops_per_sample = int(
            2 * model.num_oscillators * patch_area
            + model.num_oscillators * patch_area * model.image_dim
        )
    elif model.decoder_mode == "resize_conv":
        height = int(model.resize_conv_seed_shape[1])
        width = int(model.resize_conv_seed_shape[2])
        estimated_decoder_ops_per_sample = 0
        for layer_index in range(0, len(model.resize_conv_layers), 2):
            height *= 2
            width *= 2
            for conv in (
                model.resize_conv_layers[layer_index],
                model.resize_conv_layers[layer_index + 1],
            ):
                out_channels, in_channels, kernel_h, kernel_w = conv.weight.shape
                estimated_decoder_ops_per_sample += int(
                    height
                    * width
                    * out_channels
                    * in_channels
                    * kernel_h
                    * kernel_w
                )
        if model.resize_conv_output is not None:
            out_channels, in_channels, kernel_h, kernel_w = (
                model.resize_conv_output.weight.shape
            )
            estimated_decoder_ops_per_sample += int(
                height
                * width
                * out_channels
                * in_channels
                * kernel_h
                * kernel_w
            )
    estimated_ops_per_sample = (
        estimated_recurrent_ops_per_sample + estimated_decoder_ops_per_sample
    )

    diagnostics: Dict[str, Any] = {
        "total_params": total_params,
        "trainable_total_params": trainable_total_params,
        "decoder_params": int(decoder_params),
        "recurrent_params": int(recurrent_params),
        "transition_params": int(transition_params),
        "conditioning_params": int(conditioning_params),
        "train_recurrent_dynamics": bool(model.train_recurrent_dynamics),
        "train_conditioning_dynamics": bool(model.train_conditioning_dynamics),
        "trainable_main_recurrent_params": trainable_main_recurrent_params,
        "trainable_conditioning_params": trainable_conditioning_params,
        "trainable_recurrent_params": trainable_recurrent_params,
        "dynamics_family": dynamics_family,
        "decoder_param_fraction": (
            float(decoder_params / total_params) if total_params > 0 else 0.0
        ),
        "trainable_recurrent_param_fraction": (
            float(trainable_recurrent_params / trainable_total_params)
            if trainable_total_params > 0
            else 0.0
        ),
        "coupling_density": float(coupling_nonzero / coupling_possible),
        "coupling_profile": model.coupling_profile,
        "coupling_length_scale": float(model.coupling_length_scale),
        "coupling_floor": float(model.coupling_floor),
        "coupling_bias_strength": float(model.coupling_bias_strength),
        "conditioning_strength": float(model.conditioning_strength),
        "conditioning_mode": model.conditioning_mode,
        "readout_mode": model.readout_mode,
        "num_condition_oscillators": int(model.num_condition_oscillators),
        "coupling_profile_mean": float(np.mean(off_diagonal_profile)),
        "coupling_profile_std": float(np.std(off_diagonal_profile)),
        "coupling_profile_min": float(np.min(off_diagonal_profile)),
        "coupling_profile_max": float(np.max(off_diagonal_profile)),
        "decoder_mode": model.decoder_mode,
        "steps": int(model.steps),
        "estimated_recurrent_ops_per_sample": estimated_recurrent_ops_per_sample,
        "estimated_decoder_ops_per_sample": estimated_decoder_ops_per_sample,
        "estimated_ops_per_sample": estimated_ops_per_sample,
        "estimated_recurrent_op_fraction": (
            float(estimated_recurrent_ops_per_sample / estimated_ops_per_sample)
            if estimated_ops_per_sample > 0
            else 0.0
        ),
        "train_seconds": float(total_train_seconds),
        "samples_per_train_second": (
            float(sample_count / total_train_seconds)
            if total_train_seconds > 0.0 and sample_count > 0
            else None
        ),
    }

    if trace is not None:
        initial = np.asarray(trace["initial_theta"], dtype=np.float32)
        final = np.asarray(trace["final_theta"], dtype=np.float32)
        trajectory = np.asarray(trace["theta_trajectory"], dtype=np.float32)
        delta = np.angle(np.exp(1j * (final - initial)))
        diagnostics.update(
            {
                "phase_mean_abs_displacement": float(np.mean(np.abs(delta))),
                "phase_final_order": float(np.abs(np.mean(np.exp(1j * final)))),
                "phase_initial_order": float(np.abs(np.mean(np.exp(1j * initial)))),
            }
        )
        if trajectory.shape[0] > 1:
            step_delta = np.angle(
                np.exp(1j * np.diff(trajectory, axis=0))
            )
            diagnostics["phase_step_velocity_mean"] = float(
                np.mean(np.abs(step_delta))
            )
            order_by_step = np.abs(np.mean(np.exp(1j * trajectory), axis=-1))
            diagnostics["phase_order_delta"] = float(
                np.mean(order_by_step[-1] - order_by_step[0])
            )
        else:
            diagnostics["phase_step_velocity_mean"] = 0.0
            diagnostics["phase_order_delta"] = 0.0
        if "initial_velocity" in trace and "final_velocity" in trace:
            initial_velocity = np.asarray(trace["initial_velocity"], dtype=np.float32)
            final_velocity = np.asarray(trace["final_velocity"], dtype=np.float32)
            diagnostics["state_mean_abs_velocity_displacement"] = float(
                np.mean(np.abs(final_velocity - initial_velocity))
            )
            diagnostics["state_final_energy"] = float(
                np.mean(final**2 + final_velocity**2)
            )

    return diagnostics
