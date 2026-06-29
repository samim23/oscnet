"""Distributional and Un-0-style drift losses for MNIST generators."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from .common import Array
from .features import FeatureClassifier, mnist_feature_map

def sliced_wasserstein_loss(real: Array, generated: Array, projections: Array) -> Array:
    """Compare two batches by sorted random one-dimensional projections."""

    real_proj = real @ projections.T
    generated_proj = generated @ projections.T
    real_sorted = jnp.sort(real_proj, axis=0)
    generated_sorted = jnp.sort(generated_proj, axis=0)
    return jnp.mean((real_sorted - generated_sorted) ** 2)


def distribution_moment_loss(real: Array, generated: Array) -> Array:
    """Match first and second per-pixel moments of two image batches."""

    real_mean = jnp.mean(real, axis=0)
    generated_mean = jnp.mean(generated, axis=0)
    real_std = jnp.sqrt(jnp.var(real, axis=0) + 1e-8)
    generated_std = jnp.sqrt(jnp.var(generated, axis=0) + 1e-8)
    return jnp.mean((real_mean - generated_mean) ** 2) + jnp.mean(
        (real_std - generated_std) ** 2
    )


def pixel_marginal_loss(real: Array, generated: Array) -> Array:
    """Match each pixel's unpaired batch-value distribution."""

    real_sorted = jnp.sort(real, axis=0)
    generated_sorted = jnp.sort(generated, axis=0)
    return jnp.mean((real_sorted - generated_sorted) ** 2)


def class_moment_loss(
    real: Array,
    generated: Array,
    labels: Array,
    *,
    num_classes: int,
) -> Array:
    """Match first and second per-pixel moments within label groups."""

    labels = labels.astype(jnp.int32)
    one_hot = jax.nn.one_hot(labels, int(num_classes), dtype=real.dtype)
    counts = jnp.sum(one_hot, axis=0)
    counts_safe = jnp.maximum(counts[:, None], 1.0)
    valid = (counts > 0).astype(real.dtype)

    real_mean = (one_hot.T @ real) / counts_safe
    generated_mean = (one_hot.T @ generated) / counts_safe
    real_second = (one_hot.T @ (real**2)) / counts_safe
    generated_second = (one_hot.T @ (generated**2)) / counts_safe
    real_var = jnp.maximum(real_second - real_mean**2, 0.0)
    generated_var = jnp.maximum(generated_second - generated_mean**2, 0.0)

    per_class = jnp.mean((real_mean - generated_mean) ** 2, axis=1)
    per_class = per_class + jnp.mean((real_var - generated_var) ** 2, axis=1)
    return jnp.sum(per_class * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def class_prototype_loss(
    generated: Array,
    labels: Array,
    prototypes: Array,
) -> Array:
    """Pull generated samples toward their class-average prototype."""

    target = prototypes[labels.astype(jnp.int32)]
    return jnp.mean((generated - target) ** 2)


def _pairwise_l2(x: Array, y: Array) -> Array:
    """Pairwise Euclidean distances over the final dimension."""

    xx = jnp.sum(x * x, axis=-1, keepdims=True)
    yy = jnp.sum(y * y, axis=-1, keepdims=True).T
    distances_sq = jnp.maximum(xx + yy - 2.0 * (x @ y.T), 0.0)
    return jnp.sqrt(distances_sq + 1e-12)


def _masked_softmax(logits: Array, mask: Array, axis: int) -> Array:
    masked = jnp.where(mask, logits, -1.0e6)
    return jax.nn.softmax(masked, axis=axis) * mask.astype(logits.dtype)


def conditional_pixel_drift_loss(
    real: Array,
    generated: Array,
    labels: Array,
    *,
    num_classes: int,
    positive_labels: Optional[Array] = None,
    gamma_real: Optional[Array] = None,
    gamma_labels: Optional[Array] = None,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    gamma: float = 0.2,
) -> Array:
    """Un-0-inspired class-conditional drift loss in pixel space.

    `labels` are the generated-sample labels. By default `real` and `labels`
    also define the same-batch positive and other-class real pools. When
    `positive_labels`/`gamma_real`/`gamma_labels` are provided, positives can
    come from a per-class memory queue while other-class negatives still come
    from the current real batch. This mirrors the core shape of Un-0's
    queue-backed drift target while keeping a fixed-shape JAX implementation.
    """

    labels = labels.astype(jnp.int32)
    positive_labels = labels if positive_labels is None else positive_labels
    positive_labels = positive_labels.astype(jnp.int32)
    gamma_real = real if gamma_real is None else gamma_real
    gamma_labels = positive_labels if gamma_labels is None else gamma_labels
    gamma_labels = gamma_labels.astype(jnp.int32)
    num_classes = int(num_classes)
    gamma = float(gamma)
    if num_classes < 1:
        raise ValueError("num_classes must be positive for pixel drift loss")
    if gamma < 0.0 or gamma >= 1.0:
        raise ValueError("drift_gamma must be in [0, 1)")

    temperatures_t = jnp.asarray(tuple(temperatures), dtype=generated.dtype)
    if temperatures_t.size < 1:
        raise ValueError("drift_temperatures must contain at least one value")

    batch_size = int(generated.shape[0])
    positive_size = int(real.shape[0])
    identity = jnp.eye(batch_size, dtype=bool)
    positive_detached = jax.lax.stop_gradient(real)
    gamma_detached = jax.lax.stop_gradient(gamma_real)
    generated_detached = jax.lax.stop_gradient(generated)

    dist_pos_all = _pairwise_l2(generated_detached, positive_detached)
    dist_gen_all = _pairwise_l2(generated_detached, generated_detached)
    dist_other_all = _pairwise_l2(generated_detached, gamma_detached)
    neg_features = jnp.concatenate([generated_detached, gamma_detached], axis=0)

    total = jnp.asarray(0.0, dtype=generated.dtype)
    valid_classes = jnp.asarray(0.0, dtype=generated.dtype)

    for class_id in range(num_classes):
        gen_mask = labels == class_id
        pos_mask = positive_labels == class_id
        other_mask = gamma_labels != class_id
        n_gen = jnp.sum(gen_mask.astype(generated.dtype))
        n_pos = jnp.sum(pos_mask.astype(generated.dtype))
        n_same_neg = jnp.maximum(n_gen - 1.0, 0.0)
        n_other = jnp.sum(other_mask.astype(generated.dtype))
        valid = (n_gen > 0.0) & (n_pos > 0.0) & (
            (n_same_neg > 0.0) | ((gamma > 0.0) & (n_other > 0.0))
        )

        row_mask = gen_mask[:, None]
        pos_col_mask = pos_mask[None, :]
        same_neg_col_mask = gen_mask[None, :] & (~identity)
        other_neg_col_mask = other_mask[None, :] & (gamma > 0.0)

        pos_valid = row_mask & pos_col_mask
        same_neg_valid = row_mask & same_neg_col_mask
        other_neg_valid = row_mask & other_neg_col_mask
        neg_valid = jnp.concatenate([same_neg_valid, other_neg_valid], axis=1)

        dist_pos = dist_pos_all
        dist_neg = jnp.concatenate([dist_gen_all, dist_other_all], axis=1)
        total_distance = (
            jnp.sum(jnp.where(pos_valid, dist_pos, 0.0))
            + jnp.sum(jnp.where(same_neg_valid, dist_gen_all, 0.0))
            + jnp.sum(jnp.where(other_neg_valid, dist_other_all, 0.0))
        )
        total_count = (
            jnp.sum(pos_valid.astype(generated.dtype))
            + jnp.sum(same_neg_valid.astype(generated.dtype))
            + jnp.sum(other_neg_valid.astype(generated.dtype))
        )
        scale = jax.lax.stop_gradient(
            jnp.maximum(total_distance / jnp.maximum(total_count, 1.0), 1e-12)
        )

        logits = jnp.concatenate([-dist_pos / scale, -dist_neg / scale], axis=1)
        choice_mask = jnp.concatenate([pos_valid, neg_valid], axis=1)
        logits_t = logits[None, :, :] / temperatures_t[:, None, None]
        choice_mask_t = choice_mask[None, :, :]
        row_assignment = _masked_softmax(logits_t, choice_mask_t, axis=-1)
        col_assignment = _masked_softmax(logits_t, choice_mask_t, axis=-2)
        assignment = jnp.sqrt(
            jnp.maximum(row_assignment * col_assignment, 1e-12)
        ) * choice_mask_t.astype(generated.dtype)

        a_pos = assignment[:, :, :positive_size]
        a_neg = assignment[:, :, positive_size:]
        w_pos = a_pos * jnp.sum(a_neg, axis=-1, keepdims=True)
        w_neg = a_neg * jnp.sum(a_pos, axis=-1, keepdims=True)

        drift = (
            jnp.einsum("tbn,nd->tbd", w_pos, positive_detached)
            - jnp.einsum("tbn,nd->tbd", w_neg, neg_features)
        )
        drift_norm = jnp.linalg.norm(drift, axis=-1)
        drift_scale = jax.lax.stop_gradient(
            jnp.maximum(
                jnp.sum(drift_norm * gen_mask[None, :]) / jnp.maximum(n_gen, 1.0),
                1e-12,
            )
        )
        drift = drift / drift_scale
        target = generated_detached[None, :, :] + drift
        squared_error = (generated[None, :, :] - target) ** 2
        per_temp_row = jnp.mean(squared_error, axis=-1)
        class_loss = jnp.sum(
            jnp.sum(per_temp_row * gen_mask[None, :], axis=-1)
            / jnp.maximum(n_gen, 1.0)
        )

        total = total + jnp.where(valid, class_loss, 0.0)
        valid_classes = valid_classes + valid.astype(generated.dtype)

    return total / jnp.maximum(valid_classes, 1.0)


def conditional_feature_drift_loss(
    real: Array,
    generated: Array,
    labels: Array,
    *,
    num_classes: int,
    positive_labels: Optional[Array] = None,
    gamma_real: Optional[Array] = None,
    gamma_labels: Optional[Array] = None,
    feature_mode: str = "structural",
    feature_model: Optional[FeatureClassifier] = None,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    gamma: float = 0.2,
) -> Array:
    """Un-0-inspired conditional drift in a fixed MNIST feature space."""

    real_features = mnist_feature_map(
        real,
        mode=feature_mode,
        feature_model=feature_model,
    )
    generated_features = mnist_feature_map(
        generated,
        mode=feature_mode,
        feature_model=feature_model,
    )
    gamma_features = None
    if gamma_real is not None:
        gamma_features = mnist_feature_map(
            gamma_real,
            mode=feature_mode,
            feature_model=feature_model,
        )
    return conditional_pixel_drift_loss(
        real_features,
        generated_features,
        labels,
        num_classes=num_classes,
        positive_labels=positive_labels,
        gamma_real=gamma_features,
        gamma_labels=gamma_labels,
        temperatures=temperatures,
        gamma=gamma,
    )


def generator_distribution_loss(
    real: Array,
    generated: Array,
    projections: Array,
    *,
    labels: Optional[Array] = None,
    prototypes: Optional[Array] = None,
    feature_model: Optional[FeatureClassifier] = None,
    num_classes: int = 0,
    moment_weight: float = 0.1,
    pixel_marginal_weight: float = 1.0,
    class_moment_weight: float = 0.0,
    prototype_weight: float = 0.0,
) -> Tuple[Array, Dict[str, Array]]:
    """Unpaired distributional loss for generated MNIST samples."""

    swd = sliced_wasserstein_loss(real, generated, projections)
    moment = distribution_moment_loss(real, generated)
    marginal = pixel_marginal_loss(real, generated)
    total = swd + float(moment_weight) * moment + (
        float(pixel_marginal_weight) * marginal
    )
    class_moment = jnp.asarray(0.0, dtype=real.dtype)
    prototype = jnp.asarray(0.0, dtype=real.dtype)
    if labels is not None and num_classes > 0 and class_moment_weight > 0.0:
        class_moment = class_moment_loss(
            real,
            generated,
            labels,
            num_classes=num_classes,
        )
        total = total + float(class_moment_weight) * class_moment
    if labels is not None and prototypes is not None and prototype_weight > 0.0:
        prototype = class_prototype_loss(generated, labels, prototypes)
        total = total + float(prototype_weight) * prototype
    return total, {
        "sliced_wasserstein": swd,
        "moment_loss": moment,
        "pixel_marginal_loss": marginal,
        "class_moment_loss": class_moment,
        "prototype_loss": prototype,
        "pixel_drift_loss": jnp.asarray(0.0, dtype=real.dtype),
        "feature_drift_loss": jnp.asarray(0.0, dtype=real.dtype),
        "total_loss": total,
    }


def generator_loss(
    real: Array,
    generated: Array,
    projections: Array,
    *,
    labels: Optional[Array] = None,
    positive_batch: Optional[Array] = None,
    positive_labels: Optional[Array] = None,
    gamma_batch: Optional[Array] = None,
    gamma_labels: Optional[Array] = None,
    prototypes: Optional[Array] = None,
    feature_model: Optional[FeatureClassifier] = None,
    num_classes: int = 0,
    moment_weight: float = 0.1,
    pixel_marginal_weight: float = 1.0,
    class_moment_weight: float = 0.0,
    prototype_weight: float = 0.0,
    loss_mode: str = "distributional",
    pixel_drift_weight: float = 1.0,
    feature_drift_weight: float = 1.0,
    feature_drift_mode: str = "structural",
    distributional_weight: float = 0.0,
    drift_gamma: float = 0.2,
    drift_temperatures: Sequence[float] = (0.02, 0.05, 0.2),
) -> Tuple[Array, Dict[str, Array]]:
    """Compute the configured MNIST generator objective and diagnostics."""

    distributional_total, parts = generator_distribution_loss(
        real,
        generated,
        projections,
        labels=labels,
        prototypes=prototypes,
        num_classes=num_classes,
        moment_weight=moment_weight,
        pixel_marginal_weight=pixel_marginal_weight,
        class_moment_weight=class_moment_weight,
        prototype_weight=prototype_weight,
    )
    if loss_mode == "distributional":
        return distributional_total, parts
    if loss_mode not in ("pixel_drift", "feature_drift", "pixel_feature_drift"):
        raise ValueError(
            "loss_mode must be 'distributional', 'pixel_drift', "
            "'feature_drift', or 'pixel_feature_drift'"
        )
    if labels is None or num_classes <= 0:
        raise ValueError("drift losses require conditional labels")

    pixel_drift = jnp.asarray(0.0, dtype=real.dtype)
    feature_drift = jnp.asarray(0.0, dtype=real.dtype)
    drift_positive_batch = real if positive_batch is None else positive_batch
    drift_positive_labels = labels if positive_labels is None else positive_labels
    drift_gamma_batch = real if gamma_batch is None else gamma_batch
    drift_gamma_labels = labels if gamma_labels is None else gamma_labels
    if loss_mode in ("pixel_drift", "pixel_feature_drift"):
        pixel_drift = conditional_pixel_drift_loss(
            drift_positive_batch,
            generated,
            labels,
            num_classes=num_classes,
            positive_labels=drift_positive_labels,
            gamma_real=drift_gamma_batch,
            gamma_labels=drift_gamma_labels,
            temperatures=drift_temperatures,
            gamma=drift_gamma,
        )
    if loss_mode in ("feature_drift", "pixel_feature_drift"):
        feature_drift = conditional_feature_drift_loss(
            drift_positive_batch,
            generated,
            labels,
            num_classes=num_classes,
            positive_labels=drift_positive_labels,
            gamma_real=drift_gamma_batch,
            gamma_labels=drift_gamma_labels,
            feature_mode=feature_drift_mode,
            feature_model=feature_model,
            temperatures=drift_temperatures,
            gamma=drift_gamma,
        )
    total = (
        float(pixel_drift_weight) * pixel_drift
        + float(feature_drift_weight) * feature_drift
        + float(distributional_weight) * distributional_total
    )
    parts = {
        **parts,
        "pixel_drift_loss": pixel_drift,
        "feature_drift_loss": feature_drift,
        "total_loss": total,
    }
    return total, parts

