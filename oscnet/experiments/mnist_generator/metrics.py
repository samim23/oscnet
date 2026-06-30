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


def _pairwise_squared_l2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances without materializing a 3D tensor."""

    x_sq = np.sum(x * x, axis=-1, keepdims=True)
    y_sq = np.sum(y * y, axis=-1, keepdims=True).T
    return np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)


def _finite_mean(values: np.ndarray) -> float:
    """Mean over finite values, or NaN if none exist."""

    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


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
        real_jnp = jnp.asarray(real, dtype=jnp.float32)
        clipped_jnp = jnp.asarray(clipped, dtype=jnp.float32)
        logits = classifier(clipped_jnp)
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
        real_features = np.asarray(classifier.features(real_jnp), dtype=np.float32)
        gen_features = np.asarray(classifier.features(clipped_jnp), dtype=np.float32)
        feature_real_mean = real_features.mean(axis=0)
        feature_gen_mean = gen_features.mean(axis=0)
        feature_real_std = real_features.std(axis=0)
        feature_gen_std = gen_features.std(axis=0)
        feature_pairwise = _pairwise_squared_l2(gen_features, real_features)
        feature_nearest_real = np.min(feature_pairwise, axis=1)
        feature_real_pairwise = _pairwise_squared_l2(real_features, real_features)
        np.fill_diagonal(feature_real_pairwise, np.inf)
        feature_gen_pairwise = _pairwise_squared_l2(gen_features, gen_features)
        np.fill_diagonal(feature_gen_pairwise, np.inf)
        real_pairwise_mean = _finite_mean(np.min(feature_real_pairwise, axis=1))
        feature_real_pairwise_mean = _finite_mean(feature_real_pairwise)
        metrics.update(
            {
                "classifier_feature_mean_mse": float(
                    np.mean((feature_gen_mean - feature_real_mean) ** 2)
                ),
                "classifier_feature_std_mse": float(
                    np.mean((feature_gen_std - feature_real_std) ** 2)
                ),
                "classifier_feature_diversity_ratio": float(
                    np.mean(feature_gen_std) / (np.mean(feature_real_std) + 1e-8)
                ),
                "classifier_feature_nearest_real_mse": float(
                    np.mean(feature_nearest_real)
                ),
                "classifier_feature_real_nearest_real_mse": real_pairwise_mean,
                "classifier_feature_pairwise_distance_ratio": float(
                    _finite_mean(feature_gen_pairwise)
                    / (feature_real_pairwise_mean + 1e-8)
                ),
            }
        )
    return metrics


def _model_with_steps(model: eqx.Module, steps: int) -> eqx.Module:
    """Return a view of ``model`` with a different static settling depth."""

    stepped_model = copy.copy(model)
    object.__setattr__(stepped_model, "steps", int(steps))
    return stepped_model


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio when possible, otherwise NaN."""

    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _series_summary(prefix: str, values: np.ndarray) -> Dict[str, float]:
    """Summarize a per-step scalar series."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    summary = {
        f"{prefix}_initial": float(values[0]),
        f"{prefix}_final": float(values[-1]),
        f"{prefix}_mean": _finite_mean(values),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_delta": float(values[-1] - values[0]),
    }
    if values.size > 1:
        summary[f"{prefix}_last_minus_first"] = float(values[-1] - values[0])
        summary[f"{prefix}_settling_ratio"] = _safe_ratio(
            float(values[-1]),
            float(values[0]),
        )
    return summary


def _transition_summary(prefix: str, values: np.ndarray) -> Dict[str, float]:
    """Summarize a per-transition scalar series."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_initial": 0.0,
            f"{prefix}_final": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_delta": 0.0,
            f"{prefix}_settling_ratio": float("nan"),
        }
    summary = {
        f"{prefix}_initial": float(values[0]),
        f"{prefix}_final": float(values[-1]),
        f"{prefix}_mean": _finite_mean(values),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_delta": float(values[-1] - values[0]),
        f"{prefix}_settling_ratio": _safe_ratio(
            float(values[-1]),
            float(values[0]),
        ),
    }
    return summary


def _coupling_potential_proxy(position_series: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted squared-disagreement proxy over a trajectory.

    This is an energy-like diagnostic, not a proof that the dynamics optimize a
    Lyapunov energy. Absolute effective coupling weights are used so learned
    sign choices do not make the scalar cancel to near zero.
    """

    weight = np.abs(np.asarray(weight, dtype=np.float64))
    denom = float(np.sum(weight))
    if denom <= 1e-12:
        return np.zeros((position_series.shape[0],), dtype=np.float64)
    row_sum = np.sum(weight, axis=1)
    col_sum = np.sum(weight, axis=0)
    values = []
    for state in np.asarray(position_series, dtype=np.float64):
        squared = state * state
        left = np.sum(squared * row_sum[None, :], axis=1)
        right = np.sum(squared * col_sum[None, :], axis=1)
        cross = np.einsum("bi,ij,bj->b", state, weight, state)
        values.append(float(np.mean(left + right - 2.0 * cross) / denom))
    return np.asarray(values, dtype=np.float64)


def _rectangular_coupling_potential_proxy(
    target_series: np.ndarray,
    source_series: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Weighted squared-disagreement proxy for source-to-target coupling."""

    weight = np.abs(np.asarray(weight, dtype=np.float64))
    denom = float(np.sum(weight))
    if denom <= 1e-12:
        return np.zeros((target_series.shape[0],), dtype=np.float64)
    values = []
    for target, source in zip(
        np.asarray(target_series, dtype=np.float64),
        np.asarray(source_series, dtype=np.float64),
    ):
        displacement = source[:, None, :] - target[:, :, None]
        squared = displacement * displacement
        values.append(float(np.mean(np.sum(weight[None, :, :] * squared, axis=(1, 2))) / denom))
    return np.asarray(values, dtype=np.float64)


def _second_order_state_dynamics(
    prefix: str,
    position_series: np.ndarray,
    velocity_series: np.ndarray,
    *,
    dt: float,
) -> Dict[str, float]:
    """Summarize position/velocity trajectory behavior with a metric prefix."""

    diagnostics: Dict[str, float] = {}
    state_energy = np.mean(
        position_series * position_series + velocity_series * velocity_series,
        axis=(1, 2),
    )
    velocity_rms = np.sqrt(np.mean(velocity_series * velocity_series, axis=(1, 2)))
    diagnostics.update(_series_summary(f"{prefix}_energy", state_energy))
    diagnostics.update(_series_summary(f"{prefix}_velocity_rms", velocity_rms))

    if position_series.shape[0] <= 1:
        return diagnostics

    delta_position = np.diff(position_series, axis=0)
    delta_velocity = np.diff(velocity_series, axis=0)
    state_update_rms = np.sqrt(
        np.mean(
            delta_position * delta_position + delta_velocity * delta_velocity,
            axis=(1, 2),
        )
    )
    diagnostics.update(_transition_summary(f"{prefix}_update_rms", state_update_rms))
    acceleration = delta_velocity / max(dt, 1e-8)
    acceleration_rms = np.sqrt(np.mean(acceleration * acceleration, axis=(1, 2)))
    diagnostics.update(
        _transition_summary(f"{prefix}_acceleration_rms", acceleration_rms)
    )
    diagnostics[f"{prefix}_path_length_rms"] = float(np.sum(state_update_rms))
    net_displacement = position_series[-1] - position_series[0]
    net_velocity_displacement = velocity_series[-1] - velocity_series[0]
    diagnostics[f"{prefix}_net_displacement_rms"] = float(
        np.sqrt(
            np.mean(
                net_displacement * net_displacement
                + net_velocity_displacement * net_velocity_displacement
            )
        )
    )
    diagnostics[f"{prefix}_path_efficiency_ratio"] = _safe_ratio(
        diagnostics[f"{prefix}_net_displacement_rms"],
        diagnostics[f"{prefix}_path_length_rms"],
    )
    return diagnostics


def _decode_trace_outputs(model: eqx.Module, trace: Dict[str, Array]) -> Optional[np.ndarray]:
    """Decode every recorded state in a trace, when the model supports it."""

    initial = np.asarray(trace["initial_theta"], dtype=np.float32)
    trajectory = np.asarray(trace["theta_trajectory"], dtype=np.float32)
    position_series = np.concatenate([initial[None, ...], trajectory], axis=0)
    if position_series.shape[0] < 2:
        return None

    decoded = []
    if "velocity_trajectory" in trace and hasattr(model, "decode_state"):
        initial_velocity = np.asarray(trace["initial_velocity"], dtype=np.float32)
        velocity_trajectory = np.asarray(
            trace["velocity_trajectory"],
            dtype=np.float32,
        )
        velocity_series = np.concatenate(
            [initial_velocity[None, ...], velocity_trajectory],
            axis=0,
        )
        for position, velocity in zip(position_series, velocity_series):
            decoded.append(
                np.asarray(
                    model.decode_state(
                        jnp.asarray(position),
                        jnp.asarray(velocity),
                    ),
                    dtype=np.float32,
                )
            )
        return np.stack(decoded, axis=0)

    if hasattr(model, "decode_phase"):
        for phase in position_series:
            decoded.append(
                np.asarray(
                    model.decode_phase(jnp.asarray(phase)),
                    dtype=np.float32,
                )
            )
        return np.stack(decoded, axis=0)
    return None


def compute_generator_trace_dynamics(
    model: eqx.Module,
    trace: Dict[str, Array],
) -> Dict[str, float]:
    """Compute trajectory diagnostics for generator settling behavior.

    The returned values are digital-simulation probes. They are intended to
    answer practical questions such as "is the state still moving?" and "does
    the rendered image keep changing?", without claiming physical energy
    optimality for learned oscillator updates.
    """

    initial = np.asarray(trace["initial_theta"], dtype=np.float32)
    trajectory = np.asarray(trace["theta_trajectory"], dtype=np.float32)
    position_series = np.concatenate([initial[None, ...], trajectory], axis=0)
    diagnostics: Dict[str, float] = {}

    if "initial_velocity" in trace and "velocity_trajectory" in trace:
        initial_velocity = np.asarray(trace["initial_velocity"], dtype=np.float32)
        velocity_trajectory = np.asarray(
            trace["velocity_trajectory"],
            dtype=np.float32,
        )
        velocity_series = np.concatenate(
            [initial_velocity[None, ...], velocity_trajectory],
            axis=0,
        )
        diagnostics.update(
            _second_order_state_dynamics(
                "state",
                position_series,
                velocity_series,
                dt=float(getattr(model, "dt", 1.0)),
            )
        )
    elif position_series.shape[0] > 1:
        phase_delta = np.angle(np.exp(1j * np.diff(position_series, axis=0)))
        phase_update_rms = np.sqrt(np.mean(phase_delta * phase_delta, axis=(1, 2)))
        diagnostics.update(_transition_summary("phase_update_rms", phase_update_rms))

    if "coupling" in trace and "coupling_profile" in trace:
        effective_coupling = (
            np.asarray(trace["coupling"], dtype=np.float32)
            * np.asarray(trace["coupling_profile"], dtype=np.float32)
            * float(getattr(model, "main_coupling_strength", 1.0))
        )
        if effective_coupling.shape == (
            position_series.shape[-1],
            position_series.shape[-1],
        ):
            diagnostics.update(
                _series_summary(
                    "coupling_potential_proxy",
                    _coupling_potential_proxy(position_series, effective_coupling),
                )
            )

    if "coarse_initial_theta" in trace and "coarse_theta_trajectory" in trace:
        coarse_initial = np.asarray(trace["coarse_initial_theta"], dtype=np.float32)
        coarse_trajectory = np.asarray(
            trace["coarse_theta_trajectory"],
            dtype=np.float32,
        )
        coarse_position_series = np.concatenate(
            [coarse_initial[None, ...], coarse_trajectory],
            axis=0,
        )
        if (
            "coarse_initial_velocity" in trace
            and "coarse_velocity_trajectory" in trace
        ):
            coarse_initial_velocity = np.asarray(
                trace["coarse_initial_velocity"],
                dtype=np.float32,
            )
            coarse_velocity_trajectory = np.asarray(
                trace["coarse_velocity_trajectory"],
                dtype=np.float32,
            )
            coarse_velocity_series = np.concatenate(
                [coarse_initial_velocity[None, ...], coarse_velocity_trajectory],
                axis=0,
            )
            diagnostics.update(
                _second_order_state_dynamics(
                    "coarse_state",
                    coarse_position_series,
                    coarse_velocity_series,
                    dt=float(getattr(model, "dt", 1.0)),
                )
            )

        if "coarse_coupling" in trace and "coarse_coupling_profile" in trace:
            effective_coarse_coupling = (
                np.asarray(trace["coarse_coupling"], dtype=np.float32)
                * np.asarray(trace["coarse_coupling_profile"], dtype=np.float32)
                * float(getattr(model, "main_coupling_strength", 1.0))
            )
            if effective_coarse_coupling.shape == (
                coarse_position_series.shape[-1],
                coarse_position_series.shape[-1],
            ):
                diagnostics.update(
                    _series_summary(
                        "coarse_coupling_potential_proxy",
                        _coupling_potential_proxy(
                            coarse_position_series,
                            effective_coarse_coupling,
                        ),
                    )
                )

        if "coarse_to_fine_coupling" in trace and "coarse_to_fine_profile" in trace:
            effective_coarse_to_fine = (
                np.asarray(trace["coarse_to_fine_coupling"], dtype=np.float32)
                * np.asarray(trace["coarse_to_fine_profile"], dtype=np.float32)
                * float(getattr(model, "coarse_to_fine_strength", 1.0))
            )
            if effective_coarse_to_fine.shape == (
                position_series.shape[-1],
                coarse_position_series.shape[-1],
            ):
                diagnostics.update(
                    _series_summary(
                        "coarse_to_fine_potential_proxy",
                        _rectangular_coupling_potential_proxy(
                            position_series,
                            coarse_position_series,
                            effective_coarse_to_fine,
                        ),
                    )
                )

    aux_position_series = []
    for layer_index in range(int(getattr(model, "num_auxiliary_layers", 0))):
        initial_key = f"aux_{layer_index}_initial_theta"
        trajectory_key = f"aux_{layer_index}_theta_trajectory"
        if initial_key not in trace or trajectory_key not in trace:
            continue
        aux_initial = np.asarray(trace[initial_key], dtype=np.float32)
        aux_trajectory = np.asarray(trace[trajectory_key], dtype=np.float32)
        aux_positions = np.concatenate([aux_initial[None, ...], aux_trajectory], axis=0)
        aux_position_series.append(aux_positions)

        velocity_initial_key = f"aux_{layer_index}_initial_velocity"
        velocity_trajectory_key = f"aux_{layer_index}_velocity_trajectory"
        if (
            velocity_initial_key in trace
            and velocity_trajectory_key in trace
        ):
            aux_initial_velocity = np.asarray(
                trace[velocity_initial_key],
                dtype=np.float32,
            )
            aux_velocity_trajectory = np.asarray(
                trace[velocity_trajectory_key],
                dtype=np.float32,
            )
            aux_velocities = np.concatenate(
                [aux_initial_velocity[None, ...], aux_velocity_trajectory],
                axis=0,
            )
            diagnostics.update(
                _second_order_state_dynamics(
                    f"aux_{layer_index}_state",
                    aux_positions,
                    aux_velocities,
                    dt=float(getattr(model, "dt", 1.0)),
                )
            )

        coupling_key = f"aux_{layer_index}_coupling"
        coupling_profile_key = f"aux_{layer_index}_coupling_profile"
        if coupling_key in trace and coupling_profile_key in trace:
            effective_aux_coupling = (
                np.asarray(trace[coupling_key], dtype=np.float32)
                * np.asarray(trace[coupling_profile_key], dtype=np.float32)
                * float(getattr(model, "main_coupling_strength", 1.0))
            )
            if effective_aux_coupling.shape == (
                aux_positions.shape[-1],
                aux_positions.shape[-1],
            ):
                diagnostics.update(
                    _series_summary(
                        f"aux_{layer_index}_coupling_potential_proxy",
                        _coupling_potential_proxy(
                            aux_positions,
                            effective_aux_coupling,
                        ),
                    )
                )

    if aux_position_series:
        layer_position_series = [*aux_position_series, position_series]
        vertical_count = int(getattr(model, "num_vertical_couplings", 0))
        vertical_deltas = []
        for spec_index in range(vertical_count):
            coupling_key = f"vertical_{spec_index}_coupling"
            profile_key = f"vertical_{spec_index}_profile"
            source_key = f"vertical_{spec_index}_source_layer"
            target_key = f"vertical_{spec_index}_target_layer"
            if (
                coupling_key not in trace
                or profile_key not in trace
                or source_key not in trace
                or target_key not in trace
            ):
                continue
            source_layer = int(np.asarray(trace[source_key]))
            target_layer = int(np.asarray(trace[target_key]))
            if (
                source_layer >= len(layer_position_series)
                or target_layer >= len(layer_position_series)
            ):
                continue
            coupling = np.asarray(trace[coupling_key], dtype=np.float32)
            profile = np.asarray(trace[profile_key], dtype=np.float32)
            effective_vertical = coupling * profile
            if effective_vertical.shape != (
                layer_position_series[target_layer].shape[-1],
                layer_position_series[source_layer].shape[-1],
            ):
                continue
            summary = _series_summary(
                f"vertical_{spec_index}_potential_proxy",
                _rectangular_coupling_potential_proxy(
                    layer_position_series[target_layer],
                    layer_position_series[source_layer],
                    effective_vertical,
                ),
            )
            diagnostics.update(summary)
            if f"vertical_{spec_index}_potential_proxy_delta" in summary:
                vertical_deltas.append(
                    summary[f"vertical_{spec_index}_potential_proxy_delta"]
                )
        if vertical_deltas:
            diagnostics["vertical_potential_proxy_delta_mean"] = _finite_mean(
                np.asarray(vertical_deltas, dtype=np.float64)
            )

    decoded = _decode_trace_outputs(model, trace)
    if decoded is not None and decoded.shape[0] > 1:
        output_step_mse = np.mean(np.diff(decoded, axis=0) ** 2, axis=(1, 2))
        diagnostics.update(_transition_summary("output_step_mse", output_step_mse))
        diagnostics["output_path_mse"] = float(np.sum(output_step_mse))
        diagnostics["output_net_mse"] = float(np.mean((decoded[-1] - decoded[0]) ** 2))
        diagnostics["output_path_efficiency_ratio"] = _safe_ratio(
            diagnostics["output_net_mse"],
            diagnostics["output_path_mse"],
        )

    return diagnostics


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
        "classifier_feature_diversity_ratio",
        "classifier_feature_nearest_real_mse",
        "classifier_feature_pairwise_distance_ratio",
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
        if metric_name in (
            "nearest_real_mse",
            "pixel_mean_mse",
            "pixel_std_mse",
            "classifier_feature_nearest_real_mse",
        ):
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


def _mean_pairwise_distance(values: np.ndarray) -> float:
    """Mean off-diagonal squared L2 distance for a small group."""

    values = np.asarray(values, dtype=np.float32).reshape(values.shape[0], -1)
    if values.shape[0] < 2:
        return float("nan")
    pairwise = _pairwise_squared_l2(values, values)
    np.fill_diagonal(pairwise, np.nan)
    return _finite_mean(pairwise)


def _centroid_distance(centroids: np.ndarray) -> float:
    """Mean off-diagonal squared L2 distance between class centroids."""

    centroids = np.asarray(centroids, dtype=np.float32).reshape(
        centroids.shape[0],
        -1,
    )
    if centroids.shape[0] < 2:
        return float("nan")
    pairwise = _pairwise_squared_l2(centroids, centroids)
    np.fill_diagonal(pairwise, np.nan)
    return _finite_mean(pairwise)


def _attractor_diversity_score(label_accuracy: float, spread: float) -> float:
    """Collapse-aware basin score: class consistency times log spread."""

    if not np.isfinite(label_accuracy) or not np.isfinite(spread) or spread < 0.0:
        return float("nan")
    return float(label_accuracy * np.log1p(spread))


def compute_generator_attractor_robustness(
    model: eqx.Module,
    *,
    key: jax.random.PRNGKey,
    batch_size: int,
    variants_per_class: int = 4,
    num_classes: Optional[int] = None,
    classifier: Optional[FeatureClassifier] = None,
) -> Dict[str, float]:
    """Probe class-attractor consistency under repeated initial states.

    For each class, this samples several independent initial oscillator states
    with the same label. A useful class attractor should keep those samples
    class-consistent while preserving nonzero within-class diversity. These are
    diagnostics, not proof of a physical attractor.
    """

    variants = int(variants_per_class)
    if variants <= 0:
        return {}
    classes = int(num_classes if num_classes is not None else model.num_classes)
    if classes <= 0:
        return {}

    labels = jnp.repeat(jnp.arange(classes, dtype=jnp.int32), variants)
    generated = sample_generator_images(
        model,
        key=key,
        sample_count=int(labels.shape[0]),
        batch_size=batch_size,
        labels=labels,
    )
    clipped = np.clip(np.asarray(generated, dtype=np.float32), 0.0, 1.0)
    flat = clipped.reshape(clipped.shape[0], -1)
    labels_np = np.asarray(labels, dtype=np.int32)

    per_class_pairwise = []
    per_class_std = []
    class_centroids = []
    for label in range(classes):
        group = flat[labels_np == label]
        if group.size == 0:
            continue
        per_class_pairwise.append(_mean_pairwise_distance(group))
        per_class_std.append(float(np.mean(np.std(group, axis=0))))
        class_centroids.append(np.mean(group, axis=0))

    within_pixel_distance = _finite_mean(np.asarray(per_class_pairwise))
    between_pixel_distance = _centroid_distance(np.asarray(class_centroids))
    metrics: Dict[str, float] = {
        "num_classes": float(classes),
        "variants_per_class": float(variants),
        "sample_count": float(labels.shape[0]),
        "pixel_within_class_pairwise_mse": within_pixel_distance,
        "pixel_within_class_std": _finite_mean(np.asarray(per_class_std)),
        "pixel_between_class_centroid_mse": between_pixel_distance,
        "pixel_separation_ratio": _safe_ratio(
            between_pixel_distance,
            within_pixel_distance,
        ),
    }

    if classifier is None:
        return metrics

    labels_jnp = labels.astype(jnp.int32)
    clipped_jnp = jnp.asarray(flat, dtype=jnp.float32)
    logits = classifier(clipped_jnp)
    probabilities = jax.nn.softmax(logits, axis=-1)
    predicted = jnp.argmax(probabilities, axis=-1)
    intended_probability = probabilities[jnp.arange(labels_jnp.shape[0]), labels_jnp]
    entropy = -jnp.sum(
        probabilities * jnp.log(jnp.maximum(probabilities, 1e-8)),
        axis=-1,
    )
    correct = np.asarray(predicted == labels_jnp, dtype=np.float32)
    label_accuracy = float(
        jnp.mean((predicted == labels_jnp).astype(jnp.float32))
    )
    per_class_accuracy = [
        float(np.mean(correct[labels_np == label]))
        for label in range(classes)
        if np.any(labels_np == label)
    ]
    metrics.update(
        {
            "label_accuracy": label_accuracy,
            "label_confidence": float(jnp.mean(intended_probability)),
            "max_confidence": float(jnp.mean(jnp.max(probabilities, axis=-1))),
            "entropy": float(jnp.mean(entropy)),
            "class_success_fraction": float(
                np.mean(np.asarray(per_class_accuracy) >= 0.5)
            ),
            "class_accuracy_min": float(np.min(per_class_accuracy)),
            "class_accuracy_max": float(np.max(per_class_accuracy)),
            "pixel_attractor_diversity_score": _attractor_diversity_score(
                label_accuracy,
                within_pixel_distance,
            ),
        }
    )

    features = np.asarray(classifier.features(clipped_jnp), dtype=np.float32)
    feature_pairwise = []
    feature_std = []
    feature_centroids = []
    for label in range(classes):
        group = features[labels_np == label]
        if group.size == 0:
            continue
        feature_pairwise.append(_mean_pairwise_distance(group))
        feature_std.append(float(np.mean(np.std(group, axis=0))))
        feature_centroids.append(np.mean(group, axis=0))
    within_feature_distance = _finite_mean(np.asarray(feature_pairwise))
    between_feature_distance = _centroid_distance(np.asarray(feature_centroids))
    metrics.update(
        {
            "feature_within_class_pairwise_distance": within_feature_distance,
            "feature_within_class_std": _finite_mean(np.asarray(feature_std)),
            "feature_between_class_centroid_distance": between_feature_distance,
            "feature_separation_ratio": _safe_ratio(
                between_feature_distance,
                within_feature_distance,
            ),
            "feature_attractor_diversity_score": _attractor_diversity_score(
                label_accuracy,
                within_feature_distance,
            ),
        }
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
    coarse_recurrent_params = (
        _array_size(getattr(model, "coarse_omega", None))
        + _array_size(getattr(model, "coarse_coupling", None))
        + _array_size(getattr(model, "coarse_to_fine_coupling", None))
    )
    auxiliary_recurrent_params = sum(
        _array_size(value)
        for value in getattr(model, "auxiliary_omega", ())
    ) + sum(
        _array_size(value)
        for value in getattr(model, "auxiliary_coupling", ())
    )
    vertical_recurrent_params = sum(
        _array_size(value)
        for value in getattr(model, "vertical_coupling", ())
    )
    multiscale_recurrent_params = (
        auxiliary_recurrent_params + vertical_recurrent_params
    )
    output_feedback_params = _array_size(
        getattr(model, "output_feedback_gain", None)
    )
    recurrent_params = (
        int(transition_params)
        if dynamics_family == "state_mlp"
        else _array_size(model.omega) + _array_size(model.coupling)
        + coarse_recurrent_params
        + multiscale_recurrent_params
        + output_feedback_params
    )
    coarse_conditioning_params = _array_size(
        getattr(model, "coarse_label_condition_coupling", None)
    )
    multiscale_conditioning_params = sum(
        _array_size(value)
        for value in getattr(model, "auxiliary_label_condition_coupling", ())
    )
    conditioning_params = (
        _array_size(model.label_phase_shift)
        + _array_size(model.label_condition_phase)
        + _array_size(model.condition_omega)
        + _array_size(model.condition_coupling)
        + _array_size(model.label_condition_coupling)
        + coarse_conditioning_params
        + multiscale_conditioning_params
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
    coupling_profile_row_sums = np.sum(coupling_profile, axis=-1)

    condition_n = int(model.num_condition_oscillators)
    coarse_n = int(getattr(model, "num_coarse_oscillators", 0))
    coarse_to_fine_profile = (
        np.asarray(model.coarse_to_fine_profile_matrix(), dtype=np.float32)
        if hasattr(model, "coarse_to_fine_profile_matrix")
        else np.zeros((0, 0), dtype=np.float32)
    )
    coarse_to_fine_profile_nonzero = int(np.count_nonzero(coarse_to_fine_profile))
    coarse_to_fine_profile_possible = max(int(coarse_to_fine_profile.size), 1)
    coarse_to_fine_profile_row_sums = (
        np.sum(coarse_to_fine_profile, axis=-1)
        if coarse_to_fine_profile.size > 0
        else np.asarray([0.0], dtype=np.float32)
    )
    multiscale_layer_sizes = tuple(
        int(size) for size in getattr(model, "multiscale_layer_sizes", ())
    )
    vertical_profiles = []
    if hasattr(model, "vertical_profile_matrix"):
        for spec_index in range(int(getattr(model, "num_vertical_couplings", 0))):
            vertical_profiles.append(
                np.asarray(
                    model.vertical_profile_matrix(spec_index),
                    dtype=np.float32,
                )
    )
    if vertical_profiles:
        vertical_nonzero = sum(
            int(np.count_nonzero(profile))
            for profile in vertical_profiles
        )
        vertical_possible = max(
            sum(int(profile.size) for profile in vertical_profiles),
            1,
        )
        vertical_row_sums = np.concatenate(
            [np.sum(profile, axis=-1).reshape(-1) for profile in vertical_profiles]
        )
    else:
        vertical_nonzero = 0
        vertical_possible = 1
        vertical_row_sums = np.asarray([0.0], dtype=np.float32)
    if dynamics_family == "state_mlp":
        transition_ops = sum(
            int(layer.in_features * layer.out_features)
            for layer in transition_layers
        )
        estimated_recurrent_ops_per_sample = int(model.steps * transition_ops)
    else:
        multiscale_intra_ops = sum(size * size for size in multiscale_layer_sizes)
        multiscale_vertical_ops = sum(int(profile.size) for profile in vertical_profiles)
        estimated_recurrent_ops_per_sample = int(
            model.steps
            * (
                n * n
                + (coarse_n * coarse_n + n * coarse_n if coarse_n > 0 else 0)
                + multiscale_intra_ops
                + multiscale_vertical_ops
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
    output_feedback_mode = str(getattr(model, "output_feedback_mode", "none"))
    output_feedback_strength = float(getattr(model, "output_feedback_strength", 0.0))
    if output_feedback_strength <= 0.0:
        estimated_output_feedback_ops_per_sample = 0
    elif output_feedback_mode == "image":
        estimated_output_feedback_ops_per_sample = int(
            model.steps
            * (
                estimated_decoder_ops_per_sample
                + model.num_oscillators * max(model.image_dim, 1)
            )
        )
    else:
        estimated_output_feedback_ops_per_sample = int(
            model.steps * model.num_oscillators
        )
    estimated_ops_per_sample = (
        estimated_recurrent_ops_per_sample + estimated_decoder_ops_per_sample
        + estimated_output_feedback_ops_per_sample
    )
    conditioning_target_mask = tuple(
        float(value) for value in getattr(model, "conditioning_target_mask", ())
    )
    conditioning_target_count = (
        int(sum(conditioning_target_mask))
        if conditioning_target_mask
        else int(model.num_oscillators)
    )

    diagnostics: Dict[str, Any] = {
        "total_params": total_params,
        "trainable_total_params": trainable_total_params,
        "decoder_params": int(decoder_params),
        "recurrent_params": int(recurrent_params),
        "coarse_recurrent_params": int(coarse_recurrent_params),
        "auxiliary_recurrent_params": int(auxiliary_recurrent_params),
        "vertical_recurrent_params": int(vertical_recurrent_params),
        "multiscale_recurrent_params": int(multiscale_recurrent_params),
        "output_feedback_params": int(output_feedback_params),
        "transition_params": int(transition_params),
        "conditioning_params": int(conditioning_params),
        "coarse_conditioning_params": int(coarse_conditioning_params),
        "multiscale_conditioning_params": int(multiscale_conditioning_params),
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
        "coupling_normalization": getattr(model, "coupling_normalization", "none"),
        "coupling_strength": float(model.coupling_strength),
        "main_coupling_strength": float(
            getattr(model, "main_coupling_strength", model.coupling_strength)
        ),
        "coupling_length_scale": float(model.coupling_length_scale),
        "coupling_floor": float(model.coupling_floor),
        "coupling_bias_strength": float(model.coupling_bias_strength),
        "conditioning_strength": float(model.conditioning_strength),
        "conditioning_target_fraction": float(model.conditioning_target_fraction),
        "conditioning_target_pattern": model.conditioning_target_pattern,
        "conditioning_target_count": conditioning_target_count,
        "conditioning_target_effective_fraction": (
            float(conditioning_target_count / model.num_oscillators)
            if model.num_oscillators > 0
            else 0.0
        ),
        "conditioning_mode": model.conditioning_mode,
        "readout_mode": model.readout_mode,
        "num_condition_oscillators": int(model.num_condition_oscillators),
        "num_coarse_oscillators": int(coarse_n),
        "coarse_coupling_profile": getattr(
            model,
            "coarse_coupling_profile",
            "none",
        ),
        "coarse_coupling_normalization": getattr(
            model,
            "coarse_coupling_normalization",
            "none",
        ),
        "coarse_coupling_length_scale": float(
            getattr(model, "coarse_coupling_length_scale", 0.0)
        ),
        "coarse_to_fine_strength": float(
            getattr(model, "coarse_to_fine_strength", 0.0)
        ),
        "coarse_to_fine_profile": getattr(
            model,
            "coarse_to_fine_profile",
            "none",
        ),
        "coarse_to_fine_normalization": getattr(
            model,
            "coarse_to_fine_normalization",
            "none",
        ),
        "coarse_to_fine_length_scale": float(
            getattr(model, "coarse_to_fine_length_scale", 0.0)
        ),
        "coarse_to_fine_floor": float(
            getattr(model, "coarse_to_fine_floor", 0.0)
        ),
        "coarse_to_fine_profile_density": float(
            coarse_to_fine_profile_nonzero / coarse_to_fine_profile_possible
        ),
        "coarse_to_fine_profile_row_sum_mean": float(
            np.mean(coarse_to_fine_profile_row_sums)
        ),
        "coarse_to_fine_profile_row_sum_std": float(
            np.std(coarse_to_fine_profile_row_sums)
        ),
        "coarse_to_fine_profile_row_sum_min": float(
            np.min(coarse_to_fine_profile_row_sums)
        ),
        "coarse_to_fine_profile_row_sum_max": float(
            np.max(coarse_to_fine_profile_row_sums)
        ),
        "coarse_conditioning_strength": float(
            getattr(model, "coarse_conditioning_strength", 0.0)
        ),
        "multiscale_layer_sizes": list(multiscale_layer_sizes),
        "multiscale_frequency_scales": [
            float(value)
            for value in getattr(model, "multiscale_frequency_scales", ())
        ],
        "multiscale_coupling_profile": getattr(
            model,
            "multiscale_coupling_profile",
            "none",
        ),
        "multiscale_coupling_normalization": getattr(
            model,
            "multiscale_coupling_normalization",
            "none",
        ),
        "multiscale_coupling_length_scale": float(
            getattr(model, "multiscale_coupling_length_scale", 0.0)
        ),
        "multiscale_coupling_floor": float(
            getattr(model, "multiscale_coupling_floor", 0.0)
        ),
        "multiscale_vertical_strength": float(
            getattr(model, "multiscale_vertical_strength", 0.0)
        ),
        "multiscale_feedback_strength": float(
            getattr(model, "multiscale_feedback_strength", 0.0)
        ),
        "multiscale_vertical_profile": getattr(
            model,
            "multiscale_vertical_profile",
            "none",
        ),
        "multiscale_vertical_normalization": getattr(
            model,
            "multiscale_vertical_normalization",
            "none",
        ),
        "multiscale_vertical_length_scale": float(
            getattr(model, "multiscale_vertical_length_scale", 0.0)
        ),
        "multiscale_vertical_floor": float(
            getattr(model, "multiscale_vertical_floor", 0.0)
        ),
        "multiscale_vertical_phase_lag": float(
            getattr(model, "multiscale_vertical_phase_lag", 0.0)
        ),
        "multiscale_feedback_phase_lag": float(
            getattr(model, "multiscale_feedback_phase_lag", 0.0)
        ),
        "multiscale_conditioning_strength": float(
            getattr(model, "multiscale_conditioning_strength", 0.0)
        ),
        "num_auxiliary_layers": int(getattr(model, "num_auxiliary_layers", 0)),
        "num_vertical_couplings": int(getattr(model, "num_vertical_couplings", 0)),
        "vertical_profile_density": float(vertical_nonzero / vertical_possible),
        "vertical_profile_row_sum_mean": float(np.mean(vertical_row_sums)),
        "vertical_profile_row_sum_std": float(np.std(vertical_row_sums)),
        "vertical_profile_row_sum_min": float(np.min(vertical_row_sums)),
        "vertical_profile_row_sum_max": float(np.max(vertical_row_sums)),
        "output_feedback_strength": output_feedback_strength,
        "output_feedback_mode": output_feedback_mode,
        "output_feedback_init_scale": float(
            getattr(model, "output_feedback_init_scale", 0.0)
        ),
        "output_feedback_basis_sigma": float(
            getattr(model, "output_feedback_basis_sigma", 0.0)
        ),
        "coupling_profile_mean": float(np.mean(off_diagonal_profile)),
        "coupling_profile_std": float(np.std(off_diagonal_profile)),
        "coupling_profile_min": float(np.min(off_diagonal_profile)),
        "coupling_profile_max": float(np.max(off_diagonal_profile)),
        "coupling_profile_row_sum_mean": float(np.mean(coupling_profile_row_sums)),
        "coupling_profile_row_sum_std": float(np.std(coupling_profile_row_sums)),
        "coupling_profile_row_sum_min": float(np.min(coupling_profile_row_sums)),
        "coupling_profile_row_sum_max": float(np.max(coupling_profile_row_sums)),
        "decoder_mode": model.decoder_mode,
        "steps": int(model.steps),
        "estimated_recurrent_ops_per_sample": estimated_recurrent_ops_per_sample,
        "estimated_decoder_ops_per_sample": estimated_decoder_ops_per_sample,
        "estimated_output_feedback_ops_per_sample": (
            estimated_output_feedback_ops_per_sample
        ),
        "estimated_ops_per_sample": estimated_ops_per_sample,
        "estimated_recurrent_op_fraction": (
            float(estimated_recurrent_ops_per_sample / estimated_ops_per_sample)
            if estimated_ops_per_sample > 0
            else 0.0
        ),
        "estimated_output_feedback_op_fraction": (
            float(estimated_output_feedback_ops_per_sample / estimated_ops_per_sample)
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
        diagnostics.update(compute_generator_trace_dynamics(model, trace))

    return diagnostics
