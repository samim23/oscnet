"""Distributional and Un-0-style drift losses for MNIST generators."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from oscnet.models.generative.common import _image_hw_channels

from .common import Array
from .features import FeatureClassifier, mnist_feature_map


def downsample_image_batch(
    images: Array,
    *,
    image_shape: Tuple[int, ...],
    target_size: int,
) -> Array:
    """Resize flat image batches to a square low-resolution target."""

    height, width, channels = _image_hw_channels(tuple(int(size) for size in image_shape))
    target_size = int(target_size)
    if target_size < 1:
        raise ValueError("target_size must be positive")
    images_chw = images.reshape(images.shape[0], channels, height, width)
    if height % target_size == 0 and width % target_size == 0:
        block_h = height // target_size
        block_w = width // target_size
        lowres = images_chw.reshape(
            images.shape[0],
            channels,
            target_size,
            block_h,
            target_size,
            block_w,
        )
        lowres = jnp.mean(lowres, axis=(3, 5))
    else:
        images_hw = jnp.transpose(images_chw, (0, 2, 3, 1))
        lowres = jax.image.resize(
            images_hw,
            (images.shape[0], target_size, target_size, channels),
            method="linear",
        )
        lowres = jnp.transpose(lowres, (0, 3, 1, 2))
    return lowres.reshape(images.shape[0], channels * target_size * target_size)


def coarse_auxiliary_image_loss(
    model,
    real: Array,
    *,
    key: jax.random.PRNGKey,
    labels: Optional[Array],
    image_shape: Tuple[int, ...],
    target_size: int,
    loss_mode: str = "mse",
    num_classes: int = 0,
) -> Array:
    """Match a multiscale auxiliary layer to a low-resolution image target.

    ``mse`` is the historical paired target. ``distributional`` asks the
    auxiliary layer to match low-resolution batch and class statistics instead,
    which is a better fit for unpaired class-conditional generation.
    """

    generated = model.sample_auxiliary_image(key, real.shape[0], labels)
    return coarse_auxiliary_image_loss_from_generated(
        generated,
        real,
        labels=labels,
        image_shape=image_shape,
        target_size=target_size,
        loss_mode=loss_mode,
        num_classes=num_classes,
    )


def coarse_auxiliary_image_loss_from_generated(
    generated: Array,
    real: Array,
    *,
    labels: Optional[Array],
    image_shape: Tuple[int, ...],
    target_size: int,
    loss_mode: str = "mse",
    num_classes: int = 0,
) -> Array:
    """Score a low-resolution auxiliary image against real-image targets."""

    target = downsample_image_batch(
        real,
        image_shape=image_shape,
        target_size=target_size,
    )
    target = jax.lax.stop_gradient(target)
    if loss_mode == "mse":
        return jnp.mean((generated - target) ** 2)
    if loss_mode == "distributional":
        loss = distribution_moment_loss(target, generated) + pixel_marginal_loss(
            target,
            generated,
        )
        if labels is not None and num_classes > 0:
            loss = loss + class_moment_loss(
                target,
                generated,
                labels,
                num_classes=int(num_classes),
        )
        return loss
    raise ValueError("coarse auxiliary loss mode must be 'mse' or 'distributional'")


def coarse_readout_consistency_loss(
    generated: Array,
    auxiliary_generated: Array,
    *,
    image_shape: Tuple[int, ...],
    target_size: int,
) -> Array:
    """Make the final readout agree with the same-sample coarse scaffold."""

    generated_lowres = downsample_image_batch(
        generated,
        image_shape=image_shape,
        target_size=target_size,
    )
    auxiliary_target = jax.lax.stop_gradient(auxiliary_generated)
    return jnp.mean((generated_lowres - auxiliary_target) ** 2)


def _reshape_flat_images_nchw(images: Array, image_shape: Tuple[int, ...]) -> Array:
    """Reshape flat image batches to channel-first tensors."""

    height, width, channels = _image_hw_channels(tuple(int(size) for size in image_shape))
    return images.reshape(images.shape[0], channels, height, width)


def _frequency_band_ratios(images: Array, image_shape: Tuple[int, ...]) -> Array:
    """Return low/mid/high spectral power ratios for flat images."""

    images_chw = _reshape_flat_images_nchw(images, image_shape)
    centered = images_chw - jnp.mean(images_chw, axis=(-2, -1), keepdims=True)
    power = jnp.mean(
        jnp.abs(jnp.fft.rfft2(centered, axes=(-2, -1))) ** 2,
        axis=(0, 1),
    )
    height, width = images_chw.shape[-2:]
    fy = jnp.fft.fftfreq(height)[:, None]
    fx = jnp.fft.rfftfreq(width)[None, :]
    radius = jnp.sqrt(fx * fx + fy * fy)
    radius = radius / jnp.maximum(jnp.max(radius), 1e-8)
    masks = (
        radius <= 0.25,
        (radius > 0.25) & (radius <= 0.50),
        radius > 0.50,
    )
    total = jnp.maximum(jnp.sum(power), 1e-8)
    return jnp.stack(
        [
            jnp.sum(jnp.where(mask, power, 0.0)) / total
            for mask in masks
        ]
    )


def _laplacian_variance(images: Array, image_shape: Tuple[int, ...]) -> Array:
    """Mean per-image/channel Laplacian variance."""

    images_chw = _reshape_flat_images_nchw(images, image_shape)
    laplacian = (
        -4.0 * images_chw
        + jnp.roll(images_chw, 1, axis=-2)
        + jnp.roll(images_chw, -1, axis=-2)
        + jnp.roll(images_chw, 1, axis=-1)
        + jnp.roll(images_chw, -1, axis=-1)
    )
    return jnp.mean(jnp.var(laplacian, axis=(-2, -1)))


def frequency_statistics_loss(
    real: Array,
    generated: Array,
    *,
    image_shape: Tuple[int, ...],
    edge_weight: float = 1.0,
) -> Array:
    """Match unpaired image spectrum and edge-energy statistics.

    This objective is deliberately distributional: it compares batch-level
    low/mid/high spectral ratios and Laplacian energy instead of pairing a
    generated sample with an arbitrary real sample.
    """

    real_bands = jax.lax.stop_gradient(_frequency_band_ratios(real, image_shape))
    generated_bands = _frequency_band_ratios(generated, image_shape)
    band_loss = jnp.mean(
        (
            jnp.log(jnp.maximum(generated_bands, 1e-8))
            - jnp.log(jnp.maximum(real_bands, 1e-8))
        )
        ** 2
    )
    real_edge = jax.lax.stop_gradient(_laplacian_variance(real, image_shape))
    generated_edge = _laplacian_variance(generated, image_shape)
    edge_loss = (
        jnp.log(jnp.maximum(generated_edge, 1e-8))
        - jnp.log(jnp.maximum(real_edge, 1e-8))
    ) ** 2
    return band_loss + float(edge_weight) * edge_loss


def extract_grid_patches(
    images: Array,
    *,
    image_shape: Tuple[int, ...],
    patch_size: int,
    stride: int,
    offset_y: int = 0,
    offset_x: int = 0,
) -> Array:
    """Extract flat channel-first patches on a fixed spatial grid."""

    patch_size = int(patch_size)
    stride = int(stride)
    offset_y = int(offset_y)
    offset_x = int(offset_x)
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    images_chw = _reshape_flat_images_nchw(images, image_shape)
    height, width = images_chw.shape[-2:]
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size must fit inside image_shape")

    patches = []
    offset_y = max(0, min(offset_y, int(height) - patch_size))
    offset_x = max(0, min(offset_x, int(width) - patch_size))
    for top in range(offset_y, int(height) - patch_size + 1, stride):
        for left in range(offset_x, int(width) - patch_size + 1, stride):
            patch = images_chw[
                :,
                :,
                top : top + patch_size,
                left : left + patch_size,
            ]
            patches.append(patch.reshape(images.shape[0], -1))
    if not patches:
        raise ValueError("patch grid is empty")
    return jnp.concatenate(patches, axis=0)


def _flat_laplacian_images(images: Array, image_shape: Tuple[int, ...]) -> Array:
    images_chw = _reshape_flat_images_nchw(images, image_shape)
    laplacian = (
        -4.0 * images_chw
        + jnp.roll(images_chw, 1, axis=-2)
        + jnp.roll(images_chw, -1, axis=-2)
        + jnp.roll(images_chw, 1, axis=-1)
        + jnp.roll(images_chw, -1, axis=-1)
    )
    return laplacian.reshape(images.shape[0], -1)


def patch_sliced_wasserstein_loss(
    real: Array,
    generated: Array,
    projections: Array,
    *,
    image_shape: Tuple[int, ...],
    patch_size: int = 5,
    patch_sizes: Sequence[int] = (),
    stride: int = 4,
    offsets: Sequence[int] = (0,),
    edge_weight: float = 0.0,
) -> Array:
    """Compare unpaired local patch distributions with random projections."""

    sizes = tuple(int(size) for size in patch_sizes) or (int(patch_size),)
    shifts = tuple(int(offset) for offset in offsets) or (0,)
    real_edges = jax.lax.stop_gradient(_flat_laplacian_images(real, image_shape))
    generated_edges = _flat_laplacian_images(generated, image_shape)
    losses = []
    for current_size in sizes:
        patch_dim = int(projections.shape[-1])
        current_dim = int(
            _image_hw_channels(tuple(int(size) for size in image_shape))[2]
            * current_size
            * current_size
        )
        if current_dim > patch_dim:
            raise ValueError("patch projections are smaller than patch dimension")
        projection_slice = projections[:, :current_dim]
        projection_slice = projection_slice / jnp.maximum(
            jnp.linalg.norm(projection_slice, axis=-1, keepdims=True),
            1e-8,
        )
        for offset_y in shifts:
            for offset_x in shifts:
                real_patches = jax.lax.stop_gradient(
                    extract_grid_patches(
                        real,
                        image_shape=image_shape,
                        patch_size=current_size,
                        stride=stride,
                        offset_y=offset_y,
                        offset_x=offset_x,
                    )
                )
                generated_patches = extract_grid_patches(
                    generated,
                    image_shape=image_shape,
                    patch_size=current_size,
                    stride=stride,
                    offset_y=offset_y,
                    offset_x=offset_x,
                )
                loss = sliced_wasserstein_loss(
                    real_patches,
                    generated_patches,
                    projection_slice,
                )
                if edge_weight > 0.0:
                    real_edge_patches = extract_grid_patches(
                        real_edges,
                        image_shape=image_shape,
                        patch_size=current_size,
                        stride=stride,
                        offset_y=offset_y,
                        offset_x=offset_x,
                    )
                    generated_edge_patches = extract_grid_patches(
                        generated_edges,
                        image_shape=image_shape,
                        patch_size=current_size,
                        stride=stride,
                        offset_y=offset_y,
                        offset_x=offset_x,
                    )
                    edge_loss = sliced_wasserstein_loss(
                        real_edge_patches,
                        generated_edge_patches,
                        projection_slice,
                    )
                    loss = loss + float(edge_weight) * edge_loss
                losses.append(loss)
    return jnp.mean(jnp.stack(losses))


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
