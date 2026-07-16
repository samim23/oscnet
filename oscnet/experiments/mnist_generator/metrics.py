"""Sample quality, settling, and success diagnostics for MNIST generators."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from oscnet.models.generative.common import _image_hw_channels

from .common import Array
from .features import FeatureClassifier

InitialStateSampler = Callable[
    [jax.random.PRNGKey, int, Optional[Array]],
    Tuple[Array, Array],
]


def _pairwise_squared_l2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances without materializing a 3D tensor."""

    x_sq = np.sum(x * x, axis=-1, keepdims=True)
    y_sq = np.sum(y * y, axis=-1, keepdims=True).T
    return np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)


def _finite_mean(values: np.ndarray) -> float:
    """Mean over finite values, or NaN if none exist."""

    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _sqrt_psd(matrix: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Symmetric positive-semidefinite matrix square root."""

    if matrix.size == 0:
        return matrix
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values + eps)) @ vectors.T


def _frechet_feature_distance(real: np.ndarray, gen: np.ndarray) -> float:
    """FID-style Gaussian distance in classifier feature space."""

    if real.shape[0] < 2 or gen.shape[0] < 2:
        return float("nan")
    real_mean = np.mean(real, axis=0)
    gen_mean = np.mean(gen, axis=0)
    real_cov = np.cov(real, rowvar=False)
    gen_cov = np.cov(gen, rowvar=False)
    if real_cov.ndim == 0:
        real_cov = real_cov.reshape(1, 1)
        gen_cov = gen_cov.reshape(1, 1)
    sqrt_real = _sqrt_psd(real_cov)
    covmean = _sqrt_psd(sqrt_real @ gen_cov @ sqrt_real)
    distance = (
        np.sum((real_mean - gen_mean) ** 2)
        + np.trace(real_cov + gen_cov - 2.0 * covmean)
    )
    return float(max(distance, 0.0))


def _polynomial_mmd2_unbiased(real: np.ndarray, gen: np.ndarray) -> float:
    """KID-style unbiased polynomial-kernel MMD in feature space."""

    n = int(real.shape[0])
    m = int(gen.shape[0])
    if n < 2 or m < 2:
        return float("nan")
    dim = float(max(real.shape[1], 1))
    k_xx = ((real @ real.T) / dim + 1.0) ** 3
    k_yy = ((gen @ gen.T) / dim + 1.0) ** 3
    k_xy = ((real @ gen.T) / dim + 1.0) ** 3
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    return float(
        np.sum(k_xx) / (n * (n - 1))
        + np.sum(k_yy) / (m * (m - 1))
        - 2.0 * np.mean(k_xy)
    )


def _feature_precision_recall(real: np.ndarray, gen: np.ndarray) -> Dict[str, float]:
    """Nearest-neighbor feature coverage at the real manifold median radius."""

    if real.shape[0] < 2 or gen.shape[0] < 1:
        return {
            "classifier_feature_precision_at_real_median": float("nan"),
            "classifier_feature_recall_at_real_median": float("nan"),
            "classifier_feature_real_median_radius": float("nan"),
        }
    real_pairwise = _pairwise_squared_l2(real, real)
    np.fill_diagonal(real_pairwise, np.inf)
    real_radius = float(np.median(np.min(real_pairwise, axis=1)))
    gen_to_real = np.min(_pairwise_squared_l2(gen, real), axis=1)
    real_to_gen = np.min(_pairwise_squared_l2(real, gen), axis=1)
    return {
        "classifier_feature_precision_at_real_median": float(
            np.mean(gen_to_real <= real_radius)
        ),
        "classifier_feature_recall_at_real_median": float(
            np.mean(real_to_gen <= real_radius)
        ),
        "classifier_feature_real_median_radius": real_radius,
    }


def _reshape_flat_images_nchw(
    images: np.ndarray,
    image_shape: Sequence[int],
) -> np.ndarray:
    """Reshape flat MNIST/CIFAR batches to channel-first image tensors."""

    height, width, channels = _image_hw_channels(tuple(int(size) for size in image_shape))
    expected_dim = int(height * width * channels)
    flat = np.asarray(images, dtype=np.float32).reshape(images.shape[0], -1)
    if flat.shape[-1] != expected_dim:
        raise ValueError(
            f"flat images have dim {flat.shape[-1]}, but image_shape={image_shape} "
            f"implies {expected_dim}"
        )
    if channels == 1:
        return flat.reshape(flat.shape[0], 1, height, width)
    return flat.reshape(flat.shape[0], channels, height, width)


def _downsample_flat_images_np(
    images: np.ndarray,
    *,
    image_shape: Sequence[int],
    target_size: int,
) -> np.ndarray:
    """Downsample flat channel-first images with channel layout preserved."""

    target = int(target_size)
    if target <= 0:
        raise ValueError("target_size must be positive")
    nchw = _reshape_flat_images_nchw(images, image_shape)
    _, channels, height, width = nchw.shape
    if height == target and width == target:
        return nchw.reshape(nchw.shape[0], channels * target * target)
    if height % target == 0 and width % target == 0:
        y_factor = height // target
        x_factor = width // target
        pooled = nchw.reshape(
            nchw.shape[0],
            channels,
            target,
            y_factor,
            target,
            x_factor,
        ).mean(axis=(3, 5))
        return pooled.reshape(nchw.shape[0], channels * target * target)
    y_idx = np.rint(np.linspace(0, height - 1, target)).astype(np.int64)
    x_idx = np.rint(np.linspace(0, width - 1, target)).astype(np.int64)
    sampled = nchw[:, :, y_idx, :][:, :, :, x_idx]
    return sampled.reshape(nchw.shape[0], channels * target * target)


def _upsample_lowres_flat_images_np(
    lowres_images: np.ndarray,
    *,
    image_shape: Sequence[int],
    target_size: int,
) -> np.ndarray:
    """Nearest-neighbor upsample channel-first low-res flat images."""

    target = int(target_size)
    if target <= 0:
        raise ValueError("target_size must be positive")
    height, width, channels = _image_hw_channels(tuple(int(size) for size in image_shape))
    flat = np.asarray(lowres_images, dtype=np.float32).reshape(lowres_images.shape[0], -1)
    expected_dim = int(channels * target * target)
    if flat.shape[-1] != expected_dim:
        raise ValueError(
            f"lowres images have dim {flat.shape[-1]}, but target_size={target} "
            f"and image_shape={image_shape} imply {expected_dim}"
        )
    lowres = flat.reshape(flat.shape[0], channels, target, target)
    if height % target == 0 and width % target == 0:
        upsampled = np.repeat(
            np.repeat(lowres, height // target, axis=-2),
            width // target,
            axis=-1,
        )
    else:
        y_idx = np.floor(np.arange(height) * target / height).astype(np.int64)
        x_idx = np.floor(np.arange(width) * target / width).astype(np.int64)
        y_idx = np.clip(y_idx, 0, target - 1)
        x_idx = np.clip(x_idx, 0, target - 1)
        upsampled = lowres[:, :, y_idx, :][:, :, :, x_idx]
    return upsampled.reshape(flat.shape[0], height * width * channels)


def compute_frequency_diagnostics(
    real_images: Array,
    generated: Array,
    *,
    image_shape: Sequence[int],
) -> Dict[str, float]:
    """Measure frequency and edge-energy differences for generated images."""

    real = _reshape_flat_images_nchw(np.asarray(real_images), image_shape)
    gen = _reshape_flat_images_nchw(np.asarray(generated), image_shape)
    real = real[: gen.shape[0]]
    gen = np.clip(gen, 0.0, 1.0)
    if gen.shape[0] == 0:
        return {}

    real_centered = real - np.mean(real, axis=(-2, -1), keepdims=True)
    gen_centered = gen - np.mean(gen, axis=(-2, -1), keepdims=True)
    real_power = np.mean(np.abs(np.fft.rfft2(real_centered, axes=(-2, -1))) ** 2, axis=(0, 1))
    gen_power = np.mean(np.abs(np.fft.rfft2(gen_centered, axes=(-2, -1))) ** 2, axis=(0, 1))

    height, width = real.shape[-2:]
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    max_radius = float(np.max(radius))
    if max_radius > 0.0:
        radius = radius / max_radius
    low_mask = radius <= 0.25
    mid_mask = (radius > 0.25) & (radius <= 0.50)
    high_mask = radius > 0.50

    def _band_ratio(power: np.ndarray, mask: np.ndarray) -> float:
        total = float(np.sum(power))
        if total <= 0.0:
            return float("nan")
        return float(np.sum(power[mask]) / total)

    def _spectral_centroid(power: np.ndarray) -> float:
        total = float(np.sum(power))
        if total <= 0.0:
            return float("nan")
        return float(np.sum(power * radius) / total)

    def _laplacian_variance(images: np.ndarray) -> float:
        lap = (
            -4.0 * images
            + np.roll(images, 1, axis=-2)
            + np.roll(images, -1, axis=-2)
            + np.roll(images, 1, axis=-1)
            + np.roll(images, -1, axis=-1)
        )
        return float(np.mean(np.var(lap, axis=(-2, -1))))

    real_high = _band_ratio(real_power, high_mask)
    gen_high = _band_ratio(gen_power, high_mask)
    real_mid = _band_ratio(real_power, mid_mask)
    gen_mid = _band_ratio(gen_power, mid_mask)
    real_low = _band_ratio(real_power, low_mask)
    gen_low = _band_ratio(gen_power, low_mask)
    real_centroid = _spectral_centroid(real_power)
    gen_centroid = _spectral_centroid(gen_power)
    real_lap = _laplacian_variance(real)
    gen_lap = _laplacian_variance(gen)

    return {
        "frequency_real_low_power_ratio": real_low,
        "frequency_generated_low_power_ratio": gen_low,
        "frequency_real_mid_power_ratio": real_mid,
        "frequency_generated_mid_power_ratio": gen_mid,
        "frequency_real_high_power_ratio": real_high,
        "frequency_generated_high_power_ratio": gen_high,
        "frequency_high_power_ratio_delta": float(gen_high - real_high),
        "frequency_high_power_ratio": float(gen_high / (real_high + 1e-8)),
        "frequency_spectral_centroid_real": real_centroid,
        "frequency_spectral_centroid_generated": gen_centroid,
        "frequency_spectral_centroid_ratio": float(
            gen_centroid / (real_centroid + 1e-8)
        ),
        "edge_laplacian_variance_real": real_lap,
        "edge_laplacian_variance_generated": gen_lap,
        "edge_laplacian_variance_ratio": float(gen_lap / (real_lap + 1e-8)),
    }


def _state_spatial_spectrum_diagnostics(
    values: np.ndarray,
    *,
    prefix: str,
) -> Dict[str, float]:
    """Summarize spatial spectrum when oscillator count forms a square grid."""

    if values.ndim < 2:
        return {}
    flat = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    n = int(flat.shape[-1])
    side = int(round(np.sqrt(n)))
    if side * side != n or flat.shape[0] == 0:
        return {}

    grid = flat.reshape(flat.shape[0], side, side)
    centered = grid - np.mean(grid, axis=(-2, -1), keepdims=True)
    power = np.mean(np.abs(np.fft.rfft2(centered, axes=(-2, -1))) ** 2, axis=0)
    total = float(np.sum(power))
    if total <= 0.0:
        return {
            f"{prefix}_low_power_ratio": float("nan"),
            f"{prefix}_mid_power_ratio": float("nan"),
            f"{prefix}_high_power_ratio": float("nan"),
            f"{prefix}_spectral_centroid": float("nan"),
            f"{prefix}_laplacian_variance": float("nan"),
        }

    fy = np.fft.fftfreq(side)[:, None]
    fx = np.fft.rfftfreq(side)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    max_radius = float(np.max(radius))
    if max_radius > 0.0:
        radius = radius / max_radius

    low_mask = radius <= 0.25
    mid_mask = (radius > 0.25) & (radius <= 0.50)
    high_mask = radius > 0.50

    def _ratio(mask: np.ndarray) -> float:
        return float(np.sum(power[mask]) / total)

    laplacian = (
        -4.0 * grid
        + np.roll(grid, 1, axis=-2)
        + np.roll(grid, -1, axis=-2)
        + np.roll(grid, 1, axis=-1)
        + np.roll(grid, -1, axis=-1)
    )
    return {
        f"{prefix}_low_power_ratio": _ratio(low_mask),
        f"{prefix}_mid_power_ratio": _ratio(mid_mask),
        f"{prefix}_high_power_ratio": _ratio(high_mask),
        f"{prefix}_spectral_centroid": float(np.sum(power * radius) / total),
        f"{prefix}_laplacian_variance": float(
            np.mean(np.var(laplacian, axis=(-2, -1)))
        ),
    }


def sample_generator_images(
    model: eqx.Module,
    *,
    key: jax.random.PRNGKey,
    sample_count: int,
    batch_size: int,
    labels: Optional[Array] = None,
    initial_state_sampler: Optional[InitialStateSampler] = None,
) -> Array:
    """Generate a requested number of images in batches."""

    generated = []
    remaining = int(sample_count)
    batch_index = 0
    start = 0
    while remaining > 0:
        current = min(batch_size, remaining)
        label_batch = None if labels is None else labels[start : start + current]
        batch_key = jax.random.fold_in(key, batch_index)
        if initial_state_sampler is None:
            generated.append(model(batch_key, current, label_batch))
        else:
            if not hasattr(model, "evolve_state") or not hasattr(model, "decode_state"):
                raise ValueError(
                    "initial_state_sampler requires evolve_state/decode_state"
                )
            position, velocity = initial_state_sampler(
                batch_key,
                current,
                label_batch,
            )
            final_position, final_velocity = model.evolve_state(
                (position, velocity),
                label_batch,
            )
            generated.append(model.decode_state(final_position, final_velocity))
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
    image_shape: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Compute lightweight distribution and diversity diagnostics."""

    real = np.asarray(real_images, dtype=np.float32).reshape(real_images.shape[0], -1)
    gen = np.asarray(generated, dtype=np.float32).reshape(generated.shape[0], -1)

    clipped = np.clip(gen, 0.0, 1.0)
    real_mean = real.mean(axis=0)
    gen_mean = clipped.mean(axis=0)
    real_std = real.std(axis=0)
    gen_std = clipped.std(axis=0)

    image_dim = float(max(clipped.shape[-1], 1))
    pairwise = _pairwise_squared_l2(clipped, real) / image_dim
    nearest_real_mse = np.min(pairwise, axis=1)
    real_pairwise = _pairwise_squared_l2(real, real) / image_dim
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
        "nearest_real_mse_min": float(np.min(nearest_real_mse)),
        "nearest_real_mse_p01": float(np.quantile(nearest_real_mse, 0.01)),
        "nearest_real_mse_p05": float(np.quantile(nearest_real_mse, 0.05)),
        "duplicate_rate_mse_001": float(np.mean(nearest_real_mse < 0.001)),
        "duplicate_rate_mse_0025": float(np.mean(nearest_real_mse < 0.0025)),
        "duplicate_rate_mse_005": float(np.mean(nearest_real_mse < 0.005)),
        "duplicate_rate_mse_010": float(np.mean(nearest_real_mse < 0.010)),
        "real_nearest_real_mse": float(np.mean(np.min(real_pairwise, axis=1))),
    }
    if image_shape is not None:
        metrics.update(
            compute_frequency_diagnostics(
                real_images,
                generated,
                image_shape=image_shape,
            )
        )
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
                "classifier_feature_frechet_distance": _frechet_feature_distance(
                    real_features,
                    gen_features,
                ),
                "classifier_feature_kid_mmd2": _polynomial_mmd2_unbiased(
                    real_features,
                    gen_features,
                ),
            }
        )
        metrics.update(_feature_precision_recall(real_features, gen_features))
    return metrics


def _model_with_steps(model: eqx.Module, steps: int) -> eqx.Module:
    """Return a view of ``model`` with a different static settling depth."""

    stepped_model = copy.copy(model)
    object.__setattr__(stepped_model, "steps", int(steps))
    return stepped_model


def _model_with_vertical_intervention(
    model: eqx.Module,
    mode: str,
) -> eqx.Module:
    """Return a view of ``model`` with sample-time vertical intervention."""

    intervention_specs = {
        "normal": ("normal", 1.0),
        "zero": ("zero", 1.0),
        "shuffle": ("shuffle_batch", 1.0),
        "shuffle_batch": ("shuffle_batch", 1.0),
        "flip": ("flip", 1.0),
        "scale025": ("normal", 0.25),
        "scale050": ("normal", 0.5),
        "scale100": ("normal", 1.0),
    }
    if mode not in intervention_specs:
        raise ValueError(f"unknown vertical intervention mode {mode!r}")
    intervention, scale = intervention_specs[mode]
    intervened_model = copy.copy(model)
    if hasattr(intervened_model, "multiscale_vertical_intervention"):
        object.__setattr__(
            intervened_model,
            "multiscale_vertical_intervention",
            intervention,
        )
        object.__setattr__(
            intervened_model,
            "multiscale_vertical_intervention_scale",
            float(scale),
        )
    return intervened_model


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


def _array_summary(prefix: str, values: np.ndarray) -> Dict[str, float]:
    """Summarize an arbitrary array with finite-value safeguards."""

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {}
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def _fraction(values: np.ndarray, mask: np.ndarray) -> float:
    """Return the finite fraction of entries satisfying ``mask``."""

    if values.size == 0:
        return float("nan")
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return float("nan")
    return float(np.mean(mask[finite_mask]))


def _vertical_gain_diagnostics(trace: Dict[str, Array]) -> Dict[str, float]:
    """Summarize vertical gain/modulation behavior from a generator trace."""

    diagnostics: Dict[str, float] = {}
    if "vertical_gain_final" not in trace:
        return diagnostics

    gain = np.asarray(trace["vertical_gain_final"], dtype=np.float32)
    diagnostics.update(_array_summary("vertical_gain", gain))
    diagnostics["vertical_gain_negative_fraction"] = _fraction(gain, gain < 0.0)
    diagnostics["vertical_gain_near_zero_fraction"] = _fraction(
        gain,
        np.abs(gain) <= 0.05,
    )
    diagnostics["vertical_gain_below_half_fraction"] = _fraction(gain, gain < 0.5)
    diagnostics["vertical_gain_below_one_fraction"] = _fraction(gain, gain < 1.0)
    diagnostics["vertical_gain_above_one_fraction"] = _fraction(gain, gain > 1.0)
    diagnostics["vertical_gain_clip_low_fraction"] = _fraction(gain, gain <= -0.999)
    diagnostics["vertical_gain_clip_high_fraction"] = _fraction(gain, gain >= 1.999)

    if "vertical_modulation_final" in trace:
        modulation = np.asarray(trace["vertical_modulation_final"], dtype=np.float32)
        diagnostics.update(_array_summary("vertical_modulation", modulation))
        diagnostics["vertical_modulation_positive_fraction"] = _fraction(
            modulation,
            modulation > 0.0,
        )
        diagnostics["vertical_modulation_negative_fraction"] = _fraction(
            modulation,
            modulation < 0.0,
        )

    if "vertical_gain_trajectory" in trace:
        trajectory = np.asarray(trace["vertical_gain_trajectory"], dtype=np.float32)
        if trajectory.size and trajectory.shape[0] > 0:
            initial = trajectory[0]
            final = trajectory[-1]
            diagnostics["vertical_gain_initial_mean"] = float(np.mean(initial))
            diagnostics["vertical_gain_final_mean"] = float(np.mean(final))
            diagnostics["vertical_gain_mean_delta"] = float(
                np.mean(final) - np.mean(initial)
            )
            step_delta = np.diff(trajectory, axis=0)
            if step_delta.size:
                diagnostics["vertical_gain_step_delta_rms_mean"] = float(
                    np.mean(np.sqrt(np.mean(step_delta * step_delta, axis=(1, 2))))
                )

    if "conditioning_target_mask" in trace:
        target_mask = np.asarray(trace["conditioning_target_mask"], dtype=np.float32)
        if target_mask.ndim == 1 and target_mask.shape[0] == gain.shape[-1]:
            target = target_mask > 0.5
            non_target = ~target
            if np.any(target):
                diagnostics["vertical_gain_target_mean"] = float(
                    np.mean(gain[:, target])
                )
            if np.any(non_target):
                diagnostics["vertical_gain_non_target_mean"] = float(
                    np.mean(gain[:, non_target])
                )
            if np.any(target) and np.any(non_target):
                diagnostics["vertical_gain_target_minus_non_target_mean"] = (
                    diagnostics["vertical_gain_target_mean"]
                    - diagnostics["vertical_gain_non_target_mean"]
                )

    return diagnostics


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
    diagnostics.update(_vertical_gain_diagnostics(trace))

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


def _state_probe_features(
    trace: Dict[str, Array],
    theta_key: str,
    velocity_key: Optional[str],
) -> Optional[np.ndarray]:
    """Build phase/amplitude features for a traced oscillator state."""

    if theta_key not in trace:
        return None
    theta = np.asarray(trace[theta_key], dtype=np.float32).reshape(
        np.asarray(trace[theta_key]).shape[0],
        -1,
    )
    parts = [np.tanh(theta), np.sin(theta), np.cos(theta)]
    if velocity_key is not None and velocity_key in trace:
        velocity = np.asarray(trace[velocity_key], dtype=np.float32).reshape(
            theta.shape[0],
            -1,
        )
        parts.append(np.tanh(velocity))
    return np.concatenate(parts, axis=-1)


def _ridge_probe_predictions(
    features: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fit a deterministic train/test ridge probe and return test predictions."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(x.shape[0], -1)
    if x.shape[0] != y.shape[0] or x.shape[0] < 4:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    rng = np.random.default_rng(0)
    order = rng.permutation(x.shape[0])
    train_count = int(np.clip(round(0.7 * x.shape[0]), 2, x.shape[0] - 1))
    train_idx = order[:train_count]
    test_idx = order[train_count:]
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    x_train = (x_train - mean) / np.maximum(std, 1e-6)
    x_test = (x_test - mean) / np.maximum(std, 1e-6)
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=-1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=-1)
    gram = x_train.T @ x_train
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    rhs = x_train.T @ y_train
    try:
        weights = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram + penalty) @ rhs
    return x_test @ weights, y_test, test_idx


def _regression_probe_metrics(
    features: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
    prefix: str,
) -> Dict[str, float]:
    predictions = _ridge_probe_predictions(features, target, ridge=ridge)
    if predictions is None:
        return {}
    predicted, y_test, _ = predictions
    mse = float(np.mean((predicted - y_test) ** 2))
    variance = float(np.mean((y_test - np.mean(y_test, axis=0, keepdims=True)) ** 2))
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_r2": float(1.0 - mse / (variance + 1e-8)),
        f"{prefix}_target_variance": variance,
    }


def _label_probe_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    ridge: float,
) -> Dict[str, float]:
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_array.shape[0] != features.shape[0] or label_array.shape[0] < 4:
        return {}
    one_hot = np.eye(int(num_classes), dtype=np.float32)[
        np.clip(label_array, 0, int(num_classes) - 1)
    ]
    predictions = _ridge_probe_predictions(features, one_hot, ridge=ridge)
    if predictions is None:
        return {}
    predicted, y_test, test_idx = predictions
    predicted_labels = np.argmax(predicted, axis=-1)
    true_labels = label_array[test_idx]
    mse = float(np.mean((predicted - y_test) ** 2))
    variance = float(np.mean((y_test - np.mean(y_test, axis=0, keepdims=True)) ** 2))
    return {
        "label_accuracy": float(np.mean(predicted_labels == true_labels)),
        "label_mse": mse,
        "label_r2": float(1.0 - mse / (variance + 1e-8)),
    }


def compute_generator_state_information_probe(
    model: eqx.Module,
    trace: Dict[str, Array],
    *,
    labels: Optional[Array] = None,
    classifier: Optional[FeatureClassifier] = None,
    image_shape: Optional[Sequence[int]] = None,
    target_size: int = 8,
    ridge: float = 1e-3,
) -> Dict[str, Any]:
    """Probe what information is linearly decodable from traced states.

    This diagnostic is meant to separate "state formation" from "readout"
    problems. It probes the model's own traced samples, not a paired
    reconstruction target, so it should be read as an attribution compass
    rather than a standalone generative-quality score.
    """

    state_sets: Dict[str, np.ndarray] = {}
    initial = _state_probe_features(trace, "initial_theta", "initial_velocity")
    if initial is not None:
        state_sets["fine_initial"] = initial
    final = _state_probe_features(trace, "final_theta", "final_velocity")
    if final is not None:
        state_sets["fine_final"] = final

    auxiliary_features = []
    layer_index = 0
    while f"aux_{layer_index}_final_theta" in trace:
        features = _state_probe_features(
            trace,
            f"aux_{layer_index}_final_theta",
            f"aux_{layer_index}_final_velocity",
        )
        if features is not None:
            state_sets[f"aux_{layer_index}_final"] = features
            auxiliary_features.append(features)
        layer_index += 1
    if auxiliary_features and final is not None:
        state_sets["combined_final"] = np.concatenate(
            [*auxiliary_features, final],
            axis=-1,
        )

    if not state_sets:
        return {}
    sample_count = min(features.shape[0] for features in state_sets.values())
    if sample_count < 4:
        return {
            "sample_count": int(sample_count),
            "insufficient_samples": True,
        }
    state_sets = {
        name: features[:sample_count]
        for name, features in state_sets.items()
    }

    targets: Dict[str, np.ndarray] = {}
    generated = None
    if "generated" in trace and image_shape is not None:
        generated = np.asarray(trace["generated"], dtype=np.float32)[:sample_count]
        lowres = _downsample_flat_images_np(
            generated,
            image_shape=image_shape,
            target_size=target_size,
        )
        upsampled = _upsample_lowres_flat_images_np(
            lowres,
            image_shape=image_shape,
            target_size=target_size,
        )
        targets["generated_lowres"] = lowres
        targets["generated_highpass"] = generated.reshape(sample_count, -1) - upsampled
    if "auxiliary_lowres_generated" in trace:
        targets["auxiliary_lowres_generated"] = np.asarray(
            trace["auxiliary_lowres_generated"],
            dtype=np.float32,
        )[:sample_count].reshape(sample_count, -1)
    if classifier is not None and generated is not None:
        targets["classifier_features"] = np.asarray(
            classifier.features(jnp.asarray(generated)),
            dtype=np.float32,
        )

    probe: Dict[str, Any] = {
        "sample_count": int(sample_count),
        "target_size": int(target_size),
        "ridge": float(ridge),
        "state_sets": {},
    }
    label_array = None if labels is None else np.asarray(labels)[:sample_count]
    num_classes = int(getattr(model, "num_classes", 10))
    for state_name, state_features in state_sets.items():
        metrics: Dict[str, float] = {
            "feature_dim": int(state_features.shape[-1]),
        }
        if label_array is not None:
            metrics.update(
                _label_probe_metrics(
                    state_features,
                    label_array,
                    num_classes=num_classes,
                    ridge=ridge,
                )
            )
        for target_name, target in targets.items():
            metrics.update(
                _regression_probe_metrics(
                    state_features,
                    np.asarray(target, dtype=np.float32)[:sample_count],
                    ridge=ridge,
                    prefix=target_name,
                )
            )
        probe["state_sets"][state_name] = metrics
    if "fine_initial" in probe["state_sets"] and "fine_final" in probe["state_sets"]:
        for key in (
            "label_accuracy",
            "label_r2",
            "generated_lowres_r2",
            "generated_highpass_r2",
            "classifier_features_r2",
        ):
            initial_value = probe["state_sets"]["fine_initial"].get(key)
            final_value = probe["state_sets"]["fine_final"].get(key)
            if initial_value is not None and final_value is not None:
                probe[f"fine_final_minus_initial_{key}"] = float(
                    final_value - initial_value
                )
    return probe


def compute_generator_state_fitting_probe(
    model: eqx.Module,
    real_images: Array,
    *,
    key: jax.random.PRNGKey,
    labels: Optional[Array] = None,
    image_shape: Optional[Sequence[int]] = None,
    sample_count: int = 32,
    fit_steps: int = 100,
    learning_rate: float = 5e-2,
    init_scale: float = 0.05,
    settle_steps: Sequence[int] = (0, 8, 16, 32),
    ridge: float = 1e-3,
    recovery_noise_scales: Sequence[float] = (),
    recovery_settle_steps: Sequence[int] = (1, 2, 4, 8, 16),
    occlusion_fractions: Sequence[float] = (),
    return_images: bool = False,
) -> Dict[str, Any]:
    """Fit per-image final oscillator states through a frozen decoder.

    This diagnostic asks a different question than the ordinary state probe:
    can the frozen state representation and decoder reconstruct real images at
    all if we are allowed to optimize one final HORN state per image? It then
    tests whether those fitted details survive additional oscillator settling.

    When ``recovery_noise_scales`` is non-empty, a noise-then-settle recovery
    condition is added: each fitted state is perturbed with Gaussian noise
    whose per-sample norm is ``scale`` times the clean state norm, then the
    dynamics settle for each depth in ``recovery_settle_steps``. This asks
    whether settling *repairs* a corrupted detail-bearing state (drives the
    decode back toward the clean target) or degrades it further.

    When ``occlusion_fractions`` is non-empty, a structured occlude-then-settle
    condition is added for retinotopic models: a square patch of the spatial
    oscillator grid covering roughly that fraction of sites is zeroed
    (position and velocity, all modes) at a random location per sample, then
    the dynamics settle. Decode error is additionally reported separately for
    the image region under the occluded sites and the intact remainder, which
    is the associative-memory / pattern-completion version of recovery.
    """

    if not hasattr(model, "decode_state"):
        return {"unsupported": True, "reason": "model has no decode_state"}
    if sample_count <= 0:
        return {}
    count = min(int(sample_count), int(real_images.shape[0]))
    if count < 2:
        return {"sample_count": int(count), "insufficient_samples": True}
    target = jnp.asarray(real_images[:count], dtype=jnp.float32).reshape(count, -1)
    fit_labels = None
    if labels is not None:
        fit_labels = jnp.asarray(labels[:count], dtype=jnp.int32)

    position_key, velocity_key = jax.random.split(key)
    position = (
        jax.random.normal(position_key, (count, int(model.num_oscillators)))
        * float(init_scale)
    )
    velocity = (
        jax.random.normal(velocity_key, (count, int(model.num_oscillators)))
        * float(init_scale)
    )

    optimizer = optax.adam(float(learning_rate))
    opt_state = optimizer.init((position, velocity))

    def loss_fn(fit_position: Array, fit_velocity: Array) -> Array:
        reconstruction = model.decode_state(fit_position, fit_velocity)
        return jnp.mean((reconstruction - target) ** 2)

    @jax.jit
    def fit_step(
        fit_position: Array,
        fit_velocity: Array,
        fit_opt_state: optax.OptState,
    ) -> Tuple[Array, Array, optax.OptState, Array]:
        loss_value, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(
            fit_position,
            fit_velocity,
        )
        updates, next_opt_state = optimizer.update(
            grads,
            fit_opt_state,
            (fit_position, fit_velocity),
        )
        next_position, next_velocity = optax.apply_updates(
            (fit_position, fit_velocity),
            updates,
        )
        if hasattr(model, "_bound_state"):
            next_position = model._bound_state(next_position)
            next_velocity = model._bound_state(next_velocity)
        return next_position, next_velocity, next_opt_state, loss_value

    initial_loss = float(loss_fn(position, velocity))
    losses = []
    for _ in range(int(fit_steps)):
        position, velocity, opt_state, loss_value = fit_step(
            position,
            velocity,
            opt_state,
        )
        losses.append(float(loss_value))
    fitted = model.decode_state(position, velocity)
    final_loss = float(jnp.mean((fitted - target) ** 2))

    def prefixed_metrics(prefix: str, generated: Array) -> Dict[str, float]:
        generated_np = np.asarray(generated, dtype=np.float32)
        target_np = np.asarray(target, dtype=np.float32)
        clipped = np.clip(generated_np, 0.0, 1.0)
        metrics: Dict[str, float] = {
            f"{prefix}_paired_mse": float(np.mean((clipped - target_np) ** 2)),
            f"{prefix}_paired_l1": float(np.mean(np.abs(clipped - target_np))),
            f"{prefix}_generated_std": float(np.std(clipped)),
        }
        if image_shape is not None:
            for name, value in compute_frequency_diagnostics(
                target_np,
                generated_np,
                image_shape=image_shape,
            ).items():
                metrics[f"{prefix}_{name}"] = float(value)
        return metrics

    probe: Dict[str, Any] = {
        "sample_count": int(count),
        "fit_steps": int(fit_steps),
        "learning_rate": float(learning_rate),
        "init_scale": float(init_scale),
        "initial_mse": initial_loss,
        "final_mse": final_loss,
        "mse_delta": float(final_loss - initial_loss),
        "loss_tail": losses[-5:],
        "settle_steps": [int(step) for step in settle_steps],
    }
    probe.update(prefixed_metrics("fit", fitted))
    probe_images: Dict[str, np.ndarray] = {}
    if return_images:
        probe_images["target"] = np.asarray(target, dtype=np.float32)
        probe_images["fit"] = np.clip(
            np.asarray(fitted, dtype=np.float32), 0.0, 1.0
        )

    fitted_features = _state_probe_features(
        {"final_theta": position, "final_velocity": velocity},
        "final_theta",
        "final_velocity",
    )
    if fitted_features is not None and image_shape is not None:
        real_np = np.asarray(target, dtype=np.float32)
        target_size = 8
        lowres = _downsample_flat_images_np(
            real_np,
            image_shape=image_shape,
            target_size=target_size,
        )
        upsampled = _upsample_lowres_flat_images_np(
            lowres,
            image_shape=image_shape,
            target_size=target_size,
        )
        probe["fresh_readout"] = {
            "feature_dim": int(fitted_features.shape[-1]),
            "target_size": int(target_size),
            **_regression_probe_metrics(
                fitted_features,
                real_np,
                ridge=ridge,
                prefix="real_full",
            ),
            **_regression_probe_metrics(
                fitted_features,
                lowres,
                ridge=ridge,
                prefix="real_lowres",
            ),
            **_regression_probe_metrics(
                fitted_features,
                real_np - upsampled,
                ridge=ridge,
                prefix="real_highpass",
            ),
        }

    for step in settle_steps:
        step = int(step)
        if step <= 0 or not hasattr(model, "evolve_state"):
            settled_position, settled_velocity = position, velocity
        else:
            settled_model = _model_with_steps(model, step)
            settled_position, settled_velocity = settled_model.evolve_state(
                (position, velocity),
                fit_labels,
            )
        settled = model.decode_state(settled_position, settled_velocity)
        probe.update(prefixed_metrics(f"settle_{step:03d}", settled))
        if return_images:
            probe_images[f"settle_{step:03d}"] = np.clip(
                np.asarray(settled, dtype=np.float32), 0.0, 1.0
            )
        if step > 0:
            perturb_key = jax.random.fold_in(key, step)
            noise_position_key, noise_velocity_key = jax.random.split(perturb_key)
            noise_position = jax.random.normal(noise_position_key, position.shape)
            noise_velocity = jax.random.normal(noise_velocity_key, velocity.shape)
            displacement_norm = jnp.sqrt(
                jnp.sum((settled_position - position) ** 2, axis=-1)
                + jnp.sum((settled_velocity - velocity) ** 2, axis=-1)
                + 1e-12
            )
            noise_norm = jnp.sqrt(
                jnp.sum(noise_position**2, axis=-1)
                + jnp.sum(noise_velocity**2, axis=-1)
                + 1e-12
            )
            noise_scale = displacement_norm / noise_norm
            perturbed_position = position + noise_position * noise_scale[:, None]
            perturbed_velocity = velocity + noise_velocity * noise_scale[:, None]
            if hasattr(model, "_bound_state"):
                perturbed_position = model._bound_state(perturbed_position)
                perturbed_velocity = model._bound_state(perturbed_velocity)
            perturbed = model.decode_state(perturbed_position, perturbed_velocity)
            probe[f"noise_{step:03d}_matched_state_displacement_rms"] = float(
                jnp.sqrt(jnp.mean(displacement_norm**2))
            )
            probe.update(prefixed_metrics(f"noise_{step:03d}", perturbed))

    clean_state_norm = jnp.sqrt(
        jnp.sum(position**2, axis=-1) + jnp.sum(velocity**2, axis=-1) + 1e-12
    )
    clean_fit_mse = probe["fit_paired_mse"]
    fitted_np = np.clip(np.asarray(fitted, dtype=np.float32), 0.0, 1.0)
    for scale_index, noise_scale in enumerate(recovery_noise_scales):
        noise_scale = float(noise_scale)
        prefix = f"recover_n{scale_index}"
        noise_key = jax.random.fold_in(key, 90_000 + scale_index)
        noise_position_key, noise_velocity_key = jax.random.split(noise_key)
        noise_position = jax.random.normal(noise_position_key, position.shape)
        noise_velocity = jax.random.normal(noise_velocity_key, velocity.shape)
        raw_noise_norm = jnp.sqrt(
            jnp.sum(noise_position**2, axis=-1)
            + jnp.sum(noise_velocity**2, axis=-1)
            + 1e-12
        )
        per_sample_scale = noise_scale * clean_state_norm / raw_noise_norm
        perturbed_position = position + noise_position * per_sample_scale[:, None]
        perturbed_velocity = velocity + noise_velocity * per_sample_scale[:, None]
        if hasattr(model, "_bound_state"):
            perturbed_position = model._bound_state(perturbed_position)
            perturbed_velocity = model._bound_state(perturbed_velocity)
        perturbed_distance = jnp.sqrt(
            jnp.sum((perturbed_position - position) ** 2, axis=-1)
            + jnp.sum((perturbed_velocity - velocity) ** 2, axis=-1)
            + 1e-12
        )
        noisy_decode = model.decode_state(perturbed_position, perturbed_velocity)
        probe[f"{prefix}_noise_scale"] = noise_scale
        probe[f"{prefix}_state_displacement_rms"] = float(
            jnp.sqrt(jnp.mean(perturbed_distance**2))
        )
        probe.update(prefixed_metrics(f"{prefix}_settle_000", noisy_decode))
        if return_images:
            probe_images[f"{prefix}_settle_000"] = np.clip(
                np.asarray(noisy_decode, dtype=np.float32), 0.0, 1.0
            )
        noisy_mse = probe[f"{prefix}_settle_000_paired_mse"]
        for recovery_step in recovery_settle_steps:
            recovery_step = int(recovery_step)
            if recovery_step <= 0 or not hasattr(model, "evolve_state"):
                continue
            settled_model = _model_with_steps(model, recovery_step)
            settled_position, settled_velocity = settled_model.evolve_state(
                (perturbed_position, perturbed_velocity),
                fit_labels,
            )
            settled_decode = model.decode_state(settled_position, settled_velocity)
            step_prefix = f"{prefix}_settle_{recovery_step:03d}"
            probe.update(prefixed_metrics(step_prefix, settled_decode))
            if return_images:
                probe_images[step_prefix] = np.clip(
                    np.asarray(settled_decode, dtype=np.float32), 0.0, 1.0
                )
            settled_mse = probe[f"{step_prefix}_paired_mse"]
            # Negative repair delta means settling moved the decode back
            # toward the clean target relative to the un-settled noisy state.
            probe[f"{step_prefix}_repair_delta_vs_noisy"] = float(
                settled_mse - noisy_mse
            )
            probe[f"{step_prefix}_excess_mse_vs_clean_fit"] = float(
                settled_mse - clean_fit_mse
            )
            settled_np = np.clip(
                np.asarray(settled_decode, dtype=np.float32), 0.0, 1.0
            )
            probe[f"{step_prefix}_decode_drift_from_fit"] = float(
                np.mean((settled_np - fitted_np) ** 2)
            )
            settled_distance = jnp.sqrt(
                jnp.sum((settled_position - position) ** 2, axis=-1)
                + jnp.sum((settled_velocity - velocity) ** 2, axis=-1)
                + 1e-12
            )
            # Below 1.0 the settled state is closer to the clean fitted state
            # than the perturbed state was; above 1.0 settling moved it away.
            probe[f"{step_prefix}_state_return_ratio"] = float(
                jnp.mean(settled_distance / perturbed_distance)
            )

    if occlusion_fractions:
        occlusion_metrics = _state_occlusion_recovery_metrics(
            model,
            position=position,
            velocity=velocity,
            fit_labels=fit_labels,
            target=target,
            fitted_np=fitted_np,
            image_shape=image_shape,
            key=key,
            occlusion_fractions=occlusion_fractions,
            recovery_settle_steps=recovery_settle_steps,
            prefixed_metrics=prefixed_metrics,
            probe_images=probe_images if return_images else None,
        )
        probe.update(occlusion_metrics)
    if return_images:
        probe["images"] = probe_images
    return probe


def _state_occlusion_recovery_metrics(
    model: eqx.Module,
    *,
    position: Array,
    velocity: Array,
    fit_labels: Optional[Array],
    target: Array,
    fitted_np: np.ndarray,
    image_shape: Optional[Sequence[int]],
    key: jax.random.PRNGKey,
    occlusion_fractions: Sequence[float],
    recovery_settle_steps: Sequence[int],
    prefixed_metrics: Callable[[str, Array], Dict[str, float]],
    probe_images: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Occlude square patches of the retinotopic state grid, then settle.

    Requires a retinotopic state layout (oscillator index = spatial site index
    times modes plus mode, sites row-major on the resize_conv seed grid) so
    the occluded sites correspond to a contiguous image region.
    """

    metrics: Dict[str, Any] = {}
    count = int(position.shape[0])
    num_oscillators = int(model.num_oscillators)
    num_spatial_sites = int(
        getattr(model, "num_spatial_sites", num_oscillators)
    )
    num_modes = int(getattr(model, "num_modes", 1))
    seed_shape = getattr(model, "resize_conv_seed_shape", None)
    layout = getattr(model, "resize_conv_seed_layout", None)
    if (
        seed_shape is None
        or layout != "retinotopic"
        or num_spatial_sites * num_modes != num_oscillators
    ):
        metrics["occlusion_unsupported"] = True
        return metrics
    _, seed_h, seed_w = (int(size) for size in seed_shape)
    if seed_h * seed_w != num_spatial_sites:
        metrics["occlusion_unsupported"] = True
        return metrics

    pixel_geometry = None
    if image_shape is not None:
        img_h, img_w, img_c = _image_hw_channels(
            tuple(int(size) for size in image_shape)
        )
        if img_h % seed_h == 0 and img_w % seed_w == 0:
            pixel_geometry = (img_h, img_w, img_c)
    target_np = np.asarray(target, dtype=np.float32)

    for fraction_index, fraction in enumerate(occlusion_fractions):
        fraction = float(fraction)
        prefix = f"occl_f{fraction_index}"
        side_h = int(np.clip(round(seed_h * np.sqrt(fraction)), 1, seed_h))
        side_w = int(np.clip(round(seed_w * np.sqrt(fraction)), 1, seed_w))
        patch_key = jax.random.fold_in(key, 91_000 + fraction_index)
        row_key, col_key = jax.random.split(patch_key)
        row0 = jax.random.randint(row_key, (count,), 0, seed_h - side_h + 1)
        col0 = jax.random.randint(col_key, (count,), 0, seed_w - side_w + 1)
        rows = jnp.arange(seed_h)[None, :]
        cols = jnp.arange(seed_w)[None, :]
        row_mask = (rows >= row0[:, None]) & (rows < (row0 + side_h)[:, None])
        col_mask = (cols >= col0[:, None]) & (cols < (col0 + side_w)[:, None])
        site_mask = (
            row_mask[:, :, None] & col_mask[:, None, :]
        ).reshape(count, num_spatial_sites)
        state_mask = jnp.repeat(site_mask, num_modes, axis=-1)

        occluded_position = jnp.where(state_mask, 0.0, position)
        occluded_velocity = jnp.where(state_mask, 0.0, velocity)
        occluded_decode = model.decode_state(occluded_position, occluded_velocity)

        pixel_mask_flat = None
        if pixel_geometry is not None:
            img_h, img_w, img_c = pixel_geometry
            scale_h, scale_w = img_h // seed_h, img_w // seed_w
            pixel_mask = jnp.kron(
                site_mask.reshape(count, seed_h, seed_w).astype(jnp.float32),
                jnp.ones((scale_h, scale_w), dtype=jnp.float32),
            )
            pixel_mask_flat = np.asarray(
                jnp.tile(pixel_mask[:, None, :, :], (1, img_c, 1, 1)).reshape(
                    count, -1
                )
                > 0.5
            )

        def region_mse(decoded: Array, mask: np.ndarray) -> float:
            decoded_np = np.clip(np.asarray(decoded, dtype=np.float32), 0.0, 1.0)
            squared = (decoded_np - target_np) ** 2
            selected = squared[mask]
            return float(np.mean(selected)) if selected.size else float("nan")

        metrics[f"{prefix}_fraction"] = fraction
        metrics[f"{prefix}_patch_sites"] = int(side_h * side_w)
        metrics.update(prefixed_metrics(f"{prefix}_settle_000", occluded_decode))
        if probe_images is not None:
            probe_images[f"{prefix}_settle_000"] = np.clip(
                np.asarray(occluded_decode, dtype=np.float32), 0.0, 1.0
            )
        if pixel_mask_flat is not None:
            metrics[f"{prefix}_settle_000_occluded_region_mse"] = region_mse(
                occluded_decode, pixel_mask_flat
            )
            metrics[f"{prefix}_settle_000_intact_region_mse"] = region_mse(
                occluded_decode, ~pixel_mask_flat
            )
        occluded_mse = metrics[f"{prefix}_settle_000_paired_mse"]

        for recovery_step in recovery_settle_steps:
            recovery_step = int(recovery_step)
            if recovery_step <= 0 or not hasattr(model, "evolve_state"):
                continue
            settled_model = _model_with_steps(model, recovery_step)
            settled_position, settled_velocity = settled_model.evolve_state(
                (occluded_position, occluded_velocity),
                fit_labels,
            )
            settled_decode = model.decode_state(settled_position, settled_velocity)
            step_prefix = f"{prefix}_settle_{recovery_step:03d}"
            metrics.update(prefixed_metrics(step_prefix, settled_decode))
            if probe_images is not None:
                probe_images[step_prefix] = np.clip(
                    np.asarray(settled_decode, dtype=np.float32), 0.0, 1.0
                )
            settled_mse = metrics[f"{step_prefix}_paired_mse"]
            metrics[f"{step_prefix}_repair_delta_vs_occluded"] = float(
                settled_mse - occluded_mse
            )
            settled_np = np.clip(
                np.asarray(settled_decode, dtype=np.float32), 0.0, 1.0
            )
            metrics[f"{step_prefix}_decode_drift_from_fit"] = float(
                np.mean((settled_np - fitted_np) ** 2)
            )
            if pixel_mask_flat is not None:
                metrics[f"{step_prefix}_occluded_region_mse"] = region_mse(
                    settled_decode, pixel_mask_flat
                )
                metrics[f"{step_prefix}_intact_region_mse"] = region_mse(
                    settled_decode, ~pixel_mask_flat
                )
    return metrics


def _psnr_np(generated: np.ndarray, target: np.ndarray) -> float:
    """Mean per-image PSNR in dB for [0, 1] images."""

    mse = np.mean((generated - target) ** 2, axis=-1)
    mse = np.maximum(mse, 1e-10)
    return float(np.mean(10.0 * np.log10(1.0 / mse)))


def _ssim_np(
    generated: np.ndarray,
    target: np.ndarray,
    *,
    image_shape: Sequence[int],
    window: int = 7,
) -> float:
    """Mean SSIM over images/channels with a uniform window."""

    from scipy.ndimage import uniform_filter

    height, width, channels = _image_hw_channels(
        tuple(int(size) for size in image_shape)
    )
    count = generated.shape[0]
    generated = generated.reshape(count, channels, height, width)
    target = target.reshape(count, channels, height, width)
    c1, c2 = 0.01**2, 0.03**2
    size = (1, 1, window, window)
    mu_g = uniform_filter(generated, size=size)
    mu_t = uniform_filter(target, size=size)
    sigma_g = uniform_filter(generated**2, size=size) - mu_g**2
    sigma_t = uniform_filter(target**2, size=size) - mu_t**2
    sigma_gt = uniform_filter(generated * target, size=size) - mu_g * mu_t
    ssim_map = ((2 * mu_g * mu_t + c1) * (2 * sigma_gt + c2)) / (
        (mu_g**2 + mu_t**2 + c1) * (sigma_g + sigma_t + c2)
    )
    return float(np.mean(ssim_map))


def compute_generator_recovery_metrics(
    model: eqx.Module,
    real_images: Array,
    *,
    key: jax.random.PRNGKey,
    image_shape: Sequence[int],
    sample_count: int = 64,
    noise_scales: Sequence[float] = (0.25, 0.5),
    occlusion_fractions: Sequence[float] = (0.25,),
    occlusion_patch_counts: Sequence[int] = (1, 4),
    settle_steps: Sequence[int] = (0, 4, 8, 16),
) -> Dict[str, Any]:
    """Score the encode-corrupt-settle-decode recovery task on real images.

    This is the task-level evaluation for recovery-trained generators: encode
    real images through the anchor encoder, corrupt (state-space Gaussian
    noise, or image-space occlusion before encoding), settle the dynamics,
    decode, and score against the clean image with paired MSE, PSNR, and SSIM.
    Occlusion conditions additionally report the decode error inside the
    occluded image region (fill-in) and the intact remainder separately.
    """

    from .common import occlude_image_batch

    if not hasattr(model, "encode_image_state") or not hasattr(
        model, "decode_state"
    ):
        return {"unsupported": True, "reason": "model lacks encode/decode"}
    if getattr(model, "state_anchor_encoder", None) is None:
        return {"unsupported": True, "reason": "state anchor encoder disabled"}
    count = min(int(sample_count), int(real_images.shape[0]))
    if count < 2:
        return {"sample_count": int(count), "insufficient_samples": True}
    target = jnp.asarray(real_images[:count], dtype=jnp.float32).reshape(count, -1)
    target_np = np.asarray(target, dtype=np.float32)

    metrics: Dict[str, Any] = {"sample_count": int(count)}

    def score(
        prefix: str,
        decoded: Array,
        pixel_mask: Optional[np.ndarray] = None,
    ) -> None:
        decoded_np = np.clip(np.asarray(decoded, dtype=np.float32), 0.0, 1.0)
        metrics[f"{prefix}_paired_mse"] = float(
            np.mean((decoded_np - target_np) ** 2)
        )
        metrics[f"{prefix}_psnr"] = _psnr_np(decoded_np, target_np)
        metrics[f"{prefix}_ssim"] = _ssim_np(
            decoded_np, target_np, image_shape=image_shape
        )
        if pixel_mask is not None:
            squared = (decoded_np - target_np) ** 2
            occluded = squared[pixel_mask]
            intact = squared[~pixel_mask]
            metrics[f"{prefix}_occluded_region_mse"] = (
                float(np.mean(occluded)) if occluded.size else float("nan")
            )
            metrics[f"{prefix}_intact_region_mse"] = (
                float(np.mean(intact)) if intact.size else float("nan")
            )

    def settle_and_score(
        prefix: str,
        position: Array,
        velocity: Array,
        pixel_mask: Optional[np.ndarray] = None,
    ) -> None:
        for step in settle_steps:
            step = int(step)
            if step <= 0:
                settled_position, settled_velocity = position, velocity
            else:
                step_model = _model_with_steps(model, step)
                settled_position, settled_velocity = step_model.evolve_state(
                    (position, velocity),
                    None,
                )
            decoded = model.decode_state(settled_position, settled_velocity)
            score(f"{prefix}_k{step:03d}", decoded, pixel_mask)

    clean_position, clean_velocity = model.encode_image_state(target)
    settle_and_score("clean", clean_position, clean_velocity)

    for scale_index, noise_scale in enumerate(noise_scales):
        noise_key = jax.random.fold_in(key, 40_000 + scale_index)
        position_key, velocity_key = jax.random.split(noise_key)
        position = clean_position + float(noise_scale) * jax.random.normal(
            position_key, clean_position.shape
        )
        velocity = clean_velocity + float(noise_scale) * jax.random.normal(
            velocity_key, clean_velocity.shape
        )
        if hasattr(model, "_bound_state"):
            position = model._bound_state(position)
            velocity = model._bound_state(velocity)
        metrics[f"noise_s{scale_index}_scale"] = float(noise_scale)
        settle_and_score(f"noise_s{scale_index}", position, velocity)

    for fraction_index, fraction in enumerate(occlusion_fractions):
        for patch_index, patches in enumerate(occlusion_patch_counts):
            occlusion_key = jax.random.fold_in(
                key, 41_000 + fraction_index * 97 + patch_index
            )
            occluded, mask = occlude_image_batch(
                target,
                key=occlusion_key,
                image_shape=image_shape,
                fraction=float(fraction),
                patches=int(patches),
                probability=1.0,
            )
            height, width, channels = _image_hw_channels(
                tuple(int(size) for size in image_shape)
            )
            pixel_mask = np.asarray(
                jnp.tile(mask[:, None, :, :], (1, channels, 1, 1)).reshape(
                    count, -1
                )
            )
            prefix = f"occl_f{fraction_index}_p{int(patches)}"
            metrics[f"{prefix}_fraction"] = float(fraction)
            position, velocity = model.encode_image_state(occluded)
            settle_and_score(prefix, position, velocity, pixel_mask)
    return metrics


def _perturb_model_weights(
    model: eqx.Module,
    *,
    key: jax.random.PRNGKey,
    noise_scale: float,
) -> eqx.Module:
    """Add Gaussian noise scaled per-leaf by that leaf's own std.

    Emulates analog component imprecision / weight jitter across the whole
    model (recurrent core, decoder, anchor encoder). Decoder and encoder
    architectures are matched across the compared arms, so differences in
    degradation isolate the recurrent update.
    """

    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, max(len(leaves), 1))
    new_leaves = []
    for leaf, leaf_key in zip(leaves, keys):
        if leaf.size == 0:
            new_leaves.append(leaf)
            continue
        std = jnp.std(leaf)
        noise = jax.random.normal(leaf_key, leaf.shape, dtype=leaf.dtype)
        new_leaves.append(leaf + float(noise_scale) * std * noise)
    perturbed = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return eqx.combine(perturbed, static)


def _quantize_model_weights(
    model: eqx.Module,
    *,
    bits: int,
) -> eqx.Module:
    """Uniformly quantize each float leaf to ``bits`` levels (per-leaf min-max).

    Emulates a low-precision / fixed-point analog readout of the same trained
    weights.
    """

    levels = float(2**int(bits) - 1)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def quantize(leaf: Array) -> Array:
        if leaf.size == 0:
            return leaf
        lo = jnp.min(leaf)
        hi = jnp.max(leaf)
        span = hi - lo
        scaled = jnp.where(span > 1e-12, (leaf - lo) / jnp.maximum(span, 1e-12), 0.0)
        rounded = jnp.round(scaled * levels) / levels
        return jnp.where(span > 1e-12, rounded * span + lo, leaf)

    quantized = jax.tree_util.tree_map(quantize, params)
    return eqx.combine(quantized, static)


def compute_generator_robustness_metrics(
    model: eqx.Module,
    real_images: Array,
    *,
    key: jax.random.PRNGKey,
    image_shape: Sequence[int],
    sample_count: int = 128,
    settle_step: int = 8,
    weight_noise_scales: Sequence[float] = (0.02, 0.05, 0.1, 0.2),
    quant_bits: Sequence[int] = (8, 6, 4, 3),
    ood_occlusion_fractions: Sequence[float] = (0.1, 0.25, 0.4, 0.6),
    weight_noise_draws: int = 3,
) -> Dict[str, Any]:
    """Score recovery under the oscillator's "home" fitness function: stress.

    Rather than exact reconstruction at infinite-precision parity, this probes
    graceful degradation -- the property physical/analog dynamical systems are
    supposed to buy. All conditions score the contiguous single-patch
    occlusion fill-in (occluded-region MSE) and clean PSNR at a fixed settling
    depth, so the compared arms differ only in how fast quality collapses under:

    - ``weight_noise_scales``: Gaussian weight jitter (analog component noise);
    - ``quant_bits``: low-precision weight readout;
    - ``ood_occlusion_fractions``: corruption stronger than trained on.

    Returns a flat dict of per-condition ``occluded_region_mse`` and ``psnr``
    plus the clean/unperturbed baseline for reference.
    """

    if not hasattr(model, "encode_image_state") or not hasattr(
        model, "decode_state"
    ):
        return {"unsupported": True, "reason": "model lacks encode/decode"}
    if getattr(model, "state_anchor_encoder", None) is None:
        return {"unsupported": True, "reason": "state anchor encoder disabled"}
    count = min(int(sample_count), int(real_images.shape[0]))
    if count < 2:
        return {"sample_count": int(count), "insufficient_samples": True}

    step = int(settle_step)
    metrics: Dict[str, Any] = {
        "sample_count": int(count),
        "settle_step": step,
    }

    def recover(
        scored_model: eqx.Module,
        *,
        score_key: jax.random.PRNGKey,
        fraction: float,
    ) -> Tuple[float, float]:
        """Return (contiguous occluded-region MSE, clean PSNR) at fixed depth."""

        result = compute_generator_recovery_metrics(
            scored_model,
            real_images,
            key=score_key,
            image_shape=image_shape,
            sample_count=count,
            noise_scales=(),
            occlusion_fractions=(float(fraction),),
            occlusion_patch_counts=(1,),
            settle_steps=(step,),
        )
        occ = result.get(f"occl_f0_p1_k{step:03d}_occluded_region_mse", float("nan"))
        psnr = result.get(f"clean_k{step:03d}_psnr", float("nan"))
        return float(occ), float(psnr)

    base_key = jax.random.fold_in(key, 61_000)
    base_occ, base_psnr = recover(model, score_key=base_key, fraction=0.25)
    metrics["baseline_occluded_region_mse"] = base_occ
    metrics["baseline_clean_psnr"] = base_psnr

    for scale_index, scale in enumerate(weight_noise_scales):
        occ_draws = []
        psnr_draws = []
        for draw in range(max(int(weight_noise_draws), 1)):
            perturb_key = jax.random.fold_in(key, 62_000 + scale_index * 31 + draw)
            score_key = jax.random.fold_in(key, 63_000 + scale_index * 31 + draw)
            perturbed = _perturb_model_weights(
                model, key=perturb_key, noise_scale=float(scale)
            )
            occ, psnr = recover(perturbed, score_key=score_key, fraction=0.25)
            occ_draws.append(occ)
            psnr_draws.append(psnr)
        metrics[f"wnoise_s{scale_index}_scale"] = float(scale)
        metrics[f"wnoise_s{scale_index}_occluded_region_mse"] = float(
            np.mean(occ_draws)
        )
        metrics[f"wnoise_s{scale_index}_clean_psnr"] = float(np.mean(psnr_draws))

    for bit_index, bits in enumerate(quant_bits):
        score_key = jax.random.fold_in(key, 64_000 + bit_index)
        quantized = _quantize_model_weights(model, bits=int(bits))
        occ, psnr = recover(quantized, score_key=score_key, fraction=0.25)
        metrics[f"quant_b{int(bits)}_occluded_region_mse"] = occ
        metrics[f"quant_b{int(bits)}_clean_psnr"] = psnr

    for frac_index, fraction in enumerate(ood_occlusion_fractions):
        score_key = jax.random.fold_in(key, 65_000 + frac_index)
        occ, _ = recover(model, score_key=score_key, fraction=float(fraction))
        metrics[f"ood_occl_f{frac_index}_fraction"] = float(fraction)
        metrics[f"ood_occl_f{frac_index}_occluded_region_mse"] = occ

    return metrics


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
    image_shape: Optional[Sequence[int]] = None,
    initial_state_sampler: Optional[InitialStateSampler] = None,
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
            initial_state_sampler=initial_state_sampler,
        )
        by_step[f"step_{step:03d}"] = compute_generator_quality_metrics(
            real_images[:count],
            generated,
            labels=label_slice,
            prototypes=prototypes,
            classifier=classifier,
            image_shape=image_shape,
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
    initial_state_sampler: Optional[InitialStateSampler] = None,
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
        initial_state_sampler=initial_state_sampler,
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


def compute_generator_vertical_intervention_audit(
    model: eqx.Module,
    *,
    key: jax.random.PRNGKey,
    real_images: Array,
    sample_count: int,
    batch_size: int,
    labels: Optional[Array] = None,
    prototypes: Optional[Array] = None,
    classifier: Optional[FeatureClassifier] = None,
    modes: Sequence[str] = ("normal", "zero", "shuffle", "flip"),
    image_shape: Optional[Sequence[int]] = None,
    attractor_variants_per_class: int = 0,
    num_classes: Optional[int] = None,
    trace_batch_size: Optional[int] = None,
    initial_state_sampler: Optional[InitialStateSampler] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate sample-time vertical interventions on the same initial states."""

    if not modes:
        return {}
    count = min(int(sample_count), int(real_images.shape[0]))
    if count <= 0:
        return {}
    label_slice = None if labels is None else labels[:count]
    sample_key = jax.random.fold_in(key, 10_001)
    attractor_key = jax.random.fold_in(key, 10_002)
    trace_key = jax.random.fold_in(key, 10_003)
    normal_generated: Optional[Array] = None
    normal_metrics: Optional[Dict[str, float]] = None
    audit: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        mode_name = str(mode)
        audit_model = _model_with_vertical_intervention(model, mode_name)
        generated = sample_generator_images(
            audit_model,
            key=sample_key,
            sample_count=count,
            batch_size=batch_size,
            labels=label_slice,
            initial_state_sampler=initial_state_sampler,
        )
        metrics = compute_generator_quality_metrics(
            real_images[:count],
            generated,
            labels=label_slice,
            prototypes=prototypes,
            classifier=classifier,
            image_shape=image_shape,
        )
        if normal_generated is None:
            normal_generated = generated
        metrics["output_mse_vs_normal"] = float(
            jnp.mean((generated - normal_generated) ** 2)
        )

        if attractor_variants_per_class > 0:
            attractor = compute_generator_attractor_robustness(
                audit_model,
                key=attractor_key,
                batch_size=batch_size,
                variants_per_class=attractor_variants_per_class,
                num_classes=num_classes,
                classifier=classifier,
                initial_state_sampler=initial_state_sampler,
            )
            metrics.update(
                {
                    f"attractor_{name}": value
                    for name, value in attractor.items()
                }
            )

        trace_count = min(
            count,
            int(trace_batch_size or batch_size),
        )
        trace_labels = None if label_slice is None else label_slice[:trace_count]
        if hasattr(audit_model, "collect_trace") and trace_count > 0:
            trace = audit_model.collect_trace(trace_key, trace_count, trace_labels)
            trace_metrics = compute_generator_trace_dynamics(audit_model, trace)
            for name, value in trace_metrics.items():
                if name.startswith("vertical_gain") or name.startswith(
                    "vertical_modulation"
                ):
                    metrics[f"trace_{name}"] = value

        if mode_name == "normal" or normal_metrics is None:
            normal_metrics = dict(metrics)
        else:
            for key_name in (
                "classifier_label_accuracy",
                "diversity_ratio",
                "nearest_real_mse",
                "classifier_feature_diversity_ratio",
                "classifier_feature_nearest_real_mse",
                "attractor_label_accuracy",
                "attractor_pixel_attractor_diversity_score",
            ):
                if key_name in metrics and key_name in normal_metrics:
                    metrics[f"delta_{key_name}"] = (
                        float(metrics[key_name]) - float(normal_metrics[key_name])
                    )
        audit[mode_name] = metrics

    return audit


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
    state_anchor_encoder = getattr(model, "state_anchor_encoder", None)
    state_anchor_encoder_params = 0
    if state_anchor_encoder is not None:
        state_anchor_encoder_params = _array_size(
            state_anchor_encoder.weight
        ) + _array_size(state_anchor_encoder.bias)
    state_residual_readout_params = _array_size(
        getattr(model, "state_residual_readout_weight", None)
    )
    decoder_params += state_residual_readout_params
    resonant_readout_params = _array_size(
        getattr(model, "resonant_readout_weight", None)
    )
    decoder_params += resonant_readout_params
    auxiliary_readout_params = _array_size(
        getattr(model, "auxiliary_readout_weight", None)
    ) + _array_size(getattr(model, "auxiliary_readout_bias", None))
    decoder_params += auxiliary_readout_params
    if getattr(model, "multiscale_readout_gate_mode", "none") == "none":
        multiscale_readout_gate_params = 0
    else:
        multiscale_readout_gate_params = _array_size(
            getattr(model, "multiscale_readout_gate_weight", None)
        ) + _array_size(getattr(model, "multiscale_readout_gate_bias", None))
    decoder_params += multiscale_readout_gate_params
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
    total_params = int(
        decoder_params
        + recurrent_params
        + conditioning_params
        + state_anchor_encoder_params
    )
    trainable_main_recurrent_params = (
        int(recurrent_params) if model.train_recurrent_dynamics else 0
    )
    trainable_conditioning_params = (
        int(conditioning_params) if model.train_conditioning_dynamics else 0
    )
    trainable_recurrent_params = (
        trainable_main_recurrent_params + trainable_conditioning_params
    )
    trainable_total_params = int(
        decoder_params + trainable_recurrent_params + state_anchor_encoder_params
    )
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
    if auxiliary_readout_params > 0:
        estimated_decoder_ops_per_sample += _array_size(
            getattr(model, "auxiliary_readout_weight", None)
        )
    if multiscale_readout_gate_params > 0:
        estimated_decoder_ops_per_sample += _array_size(
            getattr(model, "multiscale_readout_gate_weight", None)
        )
    if state_residual_readout_params > 0:
        patch_size = int(getattr(model, "state_residual_readout_patch_size", 1))
        _, _, residual_channels = _image_hw_channels(tuple(model.image_shape))
        estimated_decoder_ops_per_sample += int(
            model.num_oscillators
            * 2
            * max(1, int(residual_channels))
            * patch_size
            * patch_size
            + model.num_oscillators * patch_size * patch_size * model.image_dim
        )
    if resonant_readout_params > 0:
        patch_size = int(getattr(model, "resonant_readout_patch_size", 1))
        _, _, resonant_channels = _image_hw_channels(tuple(model.image_shape))
        feature_count = int(
            getattr(model, "_resonant_observable_count", lambda: 0)()
        )
        estimated_decoder_ops_per_sample += int(
            model.num_oscillators
            * feature_count
            * max(1, int(resonant_channels))
            * patch_size
            * patch_size
            + model.num_oscillators * patch_size * patch_size * model.image_dim
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
        "auxiliary_readout_params": int(auxiliary_readout_params),
        "state_residual_readout_params": int(state_residual_readout_params),
        "resonant_readout_params": int(resonant_readout_params),
        "state_anchor_encoder_params": int(state_anchor_encoder_params),
        "state_anchor_encoder_enabled": bool(state_anchor_encoder is not None),
        "multiscale_readout_gate_params": int(multiscale_readout_gate_params),
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
        "num_oscillators": int(model.num_oscillators),
        "num_spatial_sites": int(
            getattr(model, "num_spatial_sites", model.num_oscillators)
        ),
        "num_modes": int(getattr(model, "num_modes", 1)),
        "mode_frequency_scales": [
            float(value) for value in getattr(model, "mode_frequency_scales", ())
        ],
        "mode_coupling_strength": float(
            getattr(model, "mode_coupling_strength", 0.0)
        ),
        "mode_coupling_profile": getattr(
            model,
            "mode_coupling_profile",
            "none",
        ),
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
        "multiscale_vertical_signal_scale": float(
            getattr(model, "multiscale_vertical_signal_scale", 1.0)
        ),
        "multiscale_feedback_signal_mode": getattr(
            model,
            "multiscale_feedback_signal_mode",
            "position",
        ),
        "multiscale_feedback_source_gate": getattr(
            model,
            "multiscale_feedback_source_gate",
            "all",
        ),
        "multiscale_feedback_source_mix": [
            float(weight)
            for weight in getattr(
                model,
                "multiscale_feedback_source_mix",
                (),
            )
        ],
        "multiscale_vertical_target_gate": getattr(
            model,
            "multiscale_vertical_target_gate",
            "all",
        ),
        "multiscale_vertical_soft_gate_floor": float(
            getattr(model, "multiscale_vertical_soft_gate_floor", 0.0)
        ),
        "multiscale_vertical_mode": getattr(
            model,
            "multiscale_vertical_mode",
            "additive",
        ),
        "multiscale_vertical_gain_target": getattr(
            model,
            "multiscale_vertical_gain_target",
            "drive",
        ),
        "multiscale_vertical_gain_normalization": getattr(
            model,
            "multiscale_vertical_gain_normalization",
            "none",
        ),
        "multiscale_vertical_gain_target_std": float(
            getattr(model, "multiscale_vertical_gain_target_std", 0.0)
        ),
        "multiscale_vertical_broad_gain_scale": float(
            getattr(model, "multiscale_vertical_broad_gain_scale", 1.0)
        ),
        "multiscale_vertical_selective_gain_scale": float(
            getattr(model, "multiscale_vertical_selective_gain_scale", 1.0)
        ),
        "multiscale_vertical_schedule": getattr(
            model,
            "multiscale_vertical_schedule",
            "constant",
        ),
        "multiscale_vertical_onset_step": int(
            getattr(model, "multiscale_vertical_onset_step", 0)
        ),
        "multiscale_vertical_ramp_steps": int(
            getattr(model, "multiscale_vertical_ramp_steps", 0)
        ),
        "multiscale_vertical_intervention": getattr(
            model,
            "multiscale_vertical_intervention",
            "normal",
        ),
        "multiscale_vertical_intervention_scale": float(
            getattr(model, "multiscale_vertical_intervention_scale", 1.0)
        ),
        "multiscale_conditioning_strength": float(
            getattr(model, "multiscale_conditioning_strength", 0.0)
        ),
        "multiscale_auxiliary_readout_layer": int(
            getattr(model, "multiscale_auxiliary_readout_layer", 0)
        ),
        "multiscale_auxiliary_readout_size": int(
            getattr(model, "multiscale_auxiliary_readout_size", 0)
        ),
        "multiscale_readout_fusion_strength": float(
            getattr(model, "multiscale_readout_fusion_strength", 0.0)
        ),
        "multiscale_readout_gate_mode": getattr(
            model,
            "multiscale_readout_gate_mode",
            "none",
        ),
        "multiscale_readout_gate_strength": float(
            getattr(model, "multiscale_readout_gate_strength", 0.0)
        ),
        "multiscale_readout_gate_init_scale": float(
            getattr(model, "multiscale_readout_gate_init_scale", 0.0)
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
        "state_residual_readout_strength": float(
            getattr(model, "state_residual_readout_strength", 0.0)
        ),
        "state_residual_readout_init_scale": float(
            getattr(model, "state_residual_readout_init_scale", 0.0)
        ),
        "state_residual_readout_patch_size": int(
            getattr(model, "state_residual_readout_patch_size", 0)
        ),
        "state_residual_readout_sigma": float(
            getattr(model, "state_residual_readout_sigma", 0.0)
        ),
        "resonant_readout_strength": float(
            getattr(model, "resonant_readout_strength", 0.0)
        ),
        "resonant_readout_init_scale": float(
            getattr(model, "resonant_readout_init_scale", 0.0)
        ),
        "resonant_readout_patch_size": int(
            getattr(model, "resonant_readout_patch_size", 0)
        ),
        "resonant_readout_sigma": float(
            getattr(model, "resonant_readout_sigma", 0.0)
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
        "resize_conv_seed_layout": getattr(model, "resize_conv_seed_layout", "flat"),
        "resize_conv_seed_min_channels": int(
            getattr(model, "resize_conv_seed_min_channels", 0)
        ),
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
        initial_spectrum = _state_spatial_spectrum_diagnostics(
            initial,
            prefix="state_initial_spatial",
        )
        final_spectrum = _state_spatial_spectrum_diagnostics(
            final,
            prefix="state_final_spatial",
        )
        diagnostics.update(initial_spectrum)
        diagnostics.update(final_spectrum)
        if (
            "state_initial_spatial_high_power_ratio" in diagnostics
            and "state_final_spatial_high_power_ratio" in diagnostics
        ):
            diagnostics["state_spatial_high_power_ratio_delta"] = float(
                diagnostics["state_final_spatial_high_power_ratio"]
                - diagnostics["state_initial_spatial_high_power_ratio"]
            )
            diagnostics["state_spatial_spectral_centroid_delta"] = float(
                diagnostics["state_final_spatial_spectral_centroid"]
                - diagnostics["state_initial_spatial_spectral_centroid"]
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
            diagnostics.update(
                _state_spatial_spectrum_diagnostics(
                    final_velocity,
                    prefix="velocity_final_spatial",
                )
            )
        diagnostics.update(compute_generator_trace_dynamics(model, trace))

    return diagnostics
