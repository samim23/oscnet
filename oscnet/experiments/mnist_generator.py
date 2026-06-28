"""MNIST implicit generation experiments with oscillator latent dynamics."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from oscnet.experiments.harness import (
    AutoencoderExperimentConfig,
    AutoencoderExperimentResult,
    ExperimentPaths,
    iter_input_target_batches,
    iter_sample_batches,
    prepare_experiment_paths,
    save_loss_curve,
    save_metrics_csv,
    write_json,
)
from oscnet.experiments.mnist_autoencoder import load_mnist_data
from oscnet.models import (
    HORNImageGenerator,
    KuramotoImageGenerator,
    StateMLPImageGenerator,
)
from oscnet.utils import save_equinox_checkpoint

Array = jnp.ndarray


class MNISTFeatureClassifier(eqx.Module):
    """Small MLP classifier used as a frozen MNIST feature target."""

    hidden_layers: Tuple[eqx.nn.Linear, ...]
    output_layer: eqx.nn.Linear
    feature_dim: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        image_dim: int = 28 * 28,
        feature_dim: int = 128,
        depth: int = 2,
        num_classes: int = 10,
        key: jax.random.PRNGKey,
    ):
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if depth < 1:
            raise ValueError("learned feature classifier depth must be positive")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        keys = jax.random.split(key, depth + 1)
        layers = []
        in_dim = int(image_dim)
        for layer_index in range(depth):
            layers.append(
                eqx.nn.Linear(
                    in_dim,
                    int(feature_dim),
                    key=keys[layer_index],
                )
            )
            in_dim = int(feature_dim)
        self.hidden_layers = tuple(layers)
        self.output_layer = eqx.nn.Linear(
            int(feature_dim),
            int(num_classes),
            key=keys[-1],
        )
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)

    def _features_single(self, image: Array) -> Array:
        hidden = image
        for layer in self.hidden_layers:
            hidden = jax.nn.gelu(layer(hidden))
        norm = jnp.linalg.norm(hidden)
        return hidden / jnp.maximum(norm, 1e-6)

    def features(self, images: Array) -> Array:
        """Return normalized penultimate features for a batch of flat images."""

        return jax.vmap(self._features_single)(images)

    def _logits_single(self, image: Array) -> Array:
        return self.output_layer(self._features_single(image))

    def __call__(self, images: Array) -> Array:
        """Return class logits for a batch of flat images."""

        return jax.vmap(self._logits_single)(images)


@dataclass(frozen=True)
class MNISTGeneratorExperimentConfig:
    """Task-specific controls for Un-0-style MNIST generation."""

    run: AutoencoderExperimentConfig
    model_family: str = "kuramoto"
    num_oscillators: int = 64
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
    horn_frequency: float = 1.0
    horn_damping: float = 0.15
    horn_nonlinearity: float = 0.05
    horn_state_bound: float = 3.0
    state_mlp_hidden_dim: int = 48
    state_mlp_depth: int = 1
    state_mlp_residual_scale: float = 0.1
    train_recurrent_dynamics: Optional[bool] = None
    train_conditioning_dynamics: Optional[bool] = None
    conditional: bool = False
    num_classes: int = 10
    label_phase_scale: float = 0.5
    num_condition_oscillators: int = 0
    conditioning_mode: str = "phase_shift"
    readout_mode: str = "absolute"
    decoder_mode: str = "mlp"
    spatial_basis_sigma: float = 0.0
    local_patch_size: int = 5
    resize_conv_seed_size: int = 7
    resize_conv_upsamples: int = 2
    resize_conv_min_channels: int = 8
    output_activation: str = "sigmoid"
    output_bias_init: Optional[float] = -2.0
    num_projections: int = 64
    moment_weight: float = 0.1
    pixel_marginal_weight: float = 1.0
    class_moment_weight: float = 0.0
    prototype_weight: float = 0.0
    loss_mode: str = "distributional"
    pixel_drift_weight: float = 1.0
    feature_drift_weight: float = 1.0
    feature_drift_mode: str = "structural"
    learned_feature_epochs: int = 0
    learned_feature_dim: int = 128
    learned_feature_depth: int = 2
    learned_feature_learning_rate: float = 1e-3
    learned_feature_weight_decay: float = 1e-4
    quality_classifier_epochs: int = 0
    quality_classifier_dim: int = 128
    quality_classifier_depth: int = 2
    quality_classifier_learning_rate: float = 1e-3
    quality_classifier_weight_decay: float = 1e-4
    drift_queue_size: int = 0
    drift_queue_num_pos: int = 0
    distributional_weight: float = 0.0
    drift_gamma: float = 0.2
    drift_temperatures: Tuple[float, ...] = (0.02, 0.05, 0.2)
    eval_sample_count: int = 128
    data_source: str = "idx"
    train_limit: Optional[int] = 10_000
    eval_limit: Optional[int] = 1_000


@dataclass
class MNISTDriftQueue:
    """Host-side per-class FIFO memory for conditional drift positives."""

    images: np.ndarray
    counts: np.ndarray
    write_indices: np.ndarray
    num_classes: int
    queue_size: int
    image_dim: int
    rng: np.random.Generator

    @classmethod
    def create(
        cls,
        *,
        num_classes: int,
        queue_size: int,
        image_dim: int,
        seed: int,
    ) -> "MNISTDriftQueue":
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if queue_size < 1:
            raise ValueError("drift_queue_size must be positive")
        return cls(
            images=np.zeros(
                (int(num_classes), int(queue_size), int(image_dim)),
                dtype=np.float32,
            ),
            counts=np.zeros((int(num_classes),), dtype=np.int32),
            write_indices=np.zeros((int(num_classes),), dtype=np.int32),
            num_classes=int(num_classes),
            queue_size=int(queue_size),
            image_dim=int(image_dim),
            rng=np.random.default_rng(int(seed)),
        )

    def push(self, images: Array, labels: Array) -> None:
        images_np = np.asarray(images, dtype=np.float32).reshape(-1, self.image_dim)
        labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
        for image, label in zip(images_np, labels_np):
            class_id = int(label)
            if class_id < 0 or class_id >= self.num_classes:
                continue
            slot = int(self.write_indices[class_id])
            self.images[class_id, slot] = image
            self.write_indices[class_id] = (slot + 1) % self.queue_size
            self.counts[class_id] = min(
                int(self.counts[class_id]) + 1,
                self.queue_size,
            )

    def ready(self, num_pos: int) -> bool:
        if num_pos < 1:
            raise ValueError("drift_queue_num_pos must be positive")
        return bool(np.all(self.counts >= int(num_pos)))

    def draw(self, num_pos: int) -> Tuple[Array, Array]:
        if not self.ready(num_pos):
            raise ValueError("drift queue is not ready for the requested positives")
        images = []
        labels = []
        for class_id in range(self.num_classes):
            available = int(self.counts[class_id])
            indices = self.rng.choice(available, size=int(num_pos), replace=False)
            images.append(self.images[class_id, indices])
            labels.append(np.full((int(num_pos),), class_id, dtype=np.int32))
        return jnp.asarray(np.concatenate(images, axis=0)), jnp.asarray(
            np.concatenate(labels, axis=0)
        )


def _logger() -> logging.Logger:
    logger = logging.getLogger("oscnet.experiments.mnist_generator")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def make_projection_matrix(
    key: jax.random.PRNGKey,
    *,
    image_dim: int,
    num_projections: int,
) -> Array:
    """Create normalized random projections for sliced-Wasserstein training."""

    if num_projections < 1:
        raise ValueError("num_projections must be positive")
    projections = jax.random.normal(key, (num_projections, image_dim))
    norms = jnp.linalg.norm(projections, axis=1, keepdims=True)
    return projections / jnp.maximum(norms, 1e-8)


@eqx.filter_jit
def _feature_classifier_train_step(
    classifier: MNISTFeatureClassifier,
    opt_state: Any,
    images: Array,
    labels: Array,
    optimizer: optax.GradientTransformation,
    max_grad_norm: float,
):
    def loss_fn(current_classifier):
        logits = current_classifier(images)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits,
            labels.astype(jnp.int32),
        ).mean()
        accuracy = jnp.mean(
            (jnp.argmax(logits, axis=-1) == labels.astype(jnp.int32)).astype(
                jnp.float32
            )
        )
        return loss, accuracy

    (loss_value, accuracy), grads = eqx.filter_value_and_grad(
        loss_fn,
        has_aux=True,
    )(classifier)
    grad_norm = _tree_norm(grads)
    clip = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
    grads = jax.tree.map(lambda grad: grad * clip, grads)
    updates, opt_state = optimizer.update(grads, opt_state, classifier)
    classifier = eqx.apply_updates(classifier, updates)
    return classifier, opt_state, loss_value, accuracy, grad_norm


@eqx.filter_jit
def _feature_classifier_eval_step(
    classifier: MNISTFeatureClassifier,
    images: Array,
    labels: Array,
) -> Tuple[Array, Array]:
    logits = classifier(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits,
        labels.astype(jnp.int32),
    ).mean()
    accuracy = jnp.mean(
        (jnp.argmax(logits, axis=-1) == labels.astype(jnp.int32)).astype(jnp.float32)
    )
    return loss, accuracy


def evaluate_feature_classifier(
    classifier: MNISTFeatureClassifier,
    images: Array,
    labels: Array,
    *,
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate the frozen feature classifier on labeled images."""

    losses = []
    accuracies = []
    iterator = iter_input_target_batches(
        images,
        labels,
        batch_size,
        jax.random.PRNGKey(0),
        shuffle=False,
    )
    for batch_images, batch_labels in iterator:
        loss, accuracy = _feature_classifier_eval_step(
            classifier,
            batch_images,
            batch_labels,
        )
        losses.append(float(loss))
        accuracies.append(float(accuracy))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
    }


def train_mnist_feature_classifier(
    train_images: Array,
    train_labels: Array,
    eval_images: Array,
    eval_labels: Array,
    *,
    key: jax.random.PRNGKey,
    num_classes: int,
    feature_dim: int,
    depth: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
) -> Tuple[MNISTFeatureClassifier, Dict[str, Any]]:
    """Train a small feature classifier that will be frozen for drift loss."""

    if epochs < 1:
        raise ValueError("learned feature drift requires at least one feature epoch")
    classifier = MNISTFeatureClassifier(
        image_dim=int(train_images.shape[-1]),
        feature_dim=int(feature_dim),
        depth=int(depth),
        num_classes=int(num_classes),
        key=key,
    )
    optimizer = optax.adamw(
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(classifier, eqx.is_array))
    history: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "eval_loss": [],
        "eval_accuracy": [],
        "grad_norm": [],
    }
    train_start = time.time()
    for epoch in range(1, int(epochs) + 1):
        epoch_key = jax.random.fold_in(key, epoch)
        losses = []
        accuracies = []
        grad_norms = []
        iterator = iter_input_target_batches(
            train_images,
            train_labels,
            batch_size,
            epoch_key,
            shuffle=True,
        )
        for batch_images, batch_labels in iterator:
            classifier, opt_state, loss, accuracy, grad_norm = (
                _feature_classifier_train_step(
                    classifier,
                    opt_state,
                    batch_images,
                    batch_labels,
                    optimizer,
                    max_grad_norm,
                )
            )
            losses.append(float(loss))
            accuracies.append(float(accuracy))
            grad_norms.append(float(grad_norm))
        eval_metrics = evaluate_feature_classifier(
            classifier,
            eval_images,
            eval_labels,
            batch_size=batch_size,
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["train_accuracy"].append(
            float(np.mean(accuracies)) if accuracies else float("nan")
        )
        history["eval_loss"].append(eval_metrics["loss"])
        history["eval_accuracy"].append(eval_metrics["accuracy"])
        history["grad_norm"].append(
            float(np.mean(grad_norms)) if grad_norms else float("nan")
        )
    history["train_seconds"] = float(time.time() - train_start)
    history["feature_dim"] = int(feature_dim)
    history["depth"] = int(depth)
    history["epochs"] = int(epochs)
    history["final_train_accuracy"] = history["train_accuracy"][-1]
    history["final_eval_accuracy"] = history["eval_accuracy"][-1]
    history["final_eval_loss"] = history["eval_loss"][-1]
    return classifier, history


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


def _average_pool_square(images: Array, factor: int) -> Array:
    """Average-pool square images by an integer factor."""

    if factor < 1:
        raise ValueError("pool factor must be positive")
    batch, height, width = images.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError("image shape must be divisible by pool factor")
    return images.reshape(
        batch,
        height // factor,
        factor,
        width // factor,
        factor,
    ).mean(axis=(2, 4))


def mnist_structural_features(
    images: Array,
    *,
    image_shape: Tuple[int, int] = (28, 28),
) -> Array:
    """Differentiable fixed features for MNIST-scale structural drift.

    These features are intentionally lightweight: pooled ink layout, pooled
    signed edge fields, row/column profiles, and low-order image moments. They
    are not a learned semantic encoder, but they make the drift objective care
    about shape-level organization rather than only raw pixel proximity.
    """

    height, width = (int(size) for size in image_shape)
    if images.shape[-1] != height * width:
        raise ValueError("images must be flat vectors matching image_shape")
    grid = images.reshape(images.shape[0], height, width)
    pool_factor = 4 if height % 4 == 0 and width % 4 == 0 else 2
    pooled = _average_pool_square(grid, pool_factor).reshape(images.shape[0], -1)

    dx = jnp.pad(
        grid[:, :, 1:] - grid[:, :, :-1],
        ((0, 0), (0, 0), (0, 1)),
    )
    dy = jnp.pad(
        grid[:, 1:, :] - grid[:, :-1, :],
        ((0, 0), (0, 1), (0, 0)),
    )
    edge_x = _average_pool_square(dx, pool_factor).reshape(images.shape[0], -1)
    edge_y = _average_pool_square(dy, pool_factor).reshape(images.shape[0], -1)
    row_profile = jnp.mean(grid, axis=2)
    col_profile = jnp.mean(grid, axis=1)

    y_coords, x_coords = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, height, dtype=images.dtype),
        jnp.linspace(-1.0, 1.0, width, dtype=images.dtype),
        indexing="ij",
    )
    weights = jnp.maximum(grid, 0.0)
    total = jnp.sum(weights, axis=(1, 2), keepdims=True) + 1e-6
    center_y = jnp.sum(weights * y_coords[None, :, :], axis=(1, 2), keepdims=True)
    center_x = jnp.sum(weights * x_coords[None, :, :], axis=(1, 2), keepdims=True)
    center_y = center_y / total
    center_x = center_x / total
    dy_centered = y_coords[None, :, :] - center_y
    dx_centered = x_coords[None, :, :] - center_x
    var_y = jnp.sum(weights * dy_centered**2, axis=(1, 2), keepdims=True) / total
    var_x = jnp.sum(weights * dx_centered**2, axis=(1, 2), keepdims=True) / total
    covariance = (
        jnp.sum(weights * dy_centered * dx_centered, axis=(1, 2), keepdims=True)
        / total
    )
    mass = jnp.mean(weights, axis=(1, 2), keepdims=True)
    moments = jnp.concatenate(
        [
            mass.reshape(images.shape[0], 1),
            center_y.reshape(images.shape[0], 1),
            center_x.reshape(images.shape[0], 1),
            var_y.reshape(images.shape[0], 1),
            var_x.reshape(images.shape[0], 1),
            covariance.reshape(images.shape[0], 1),
        ],
        axis=-1,
    )

    return jnp.concatenate(
        [
            pooled,
            edge_x,
            edge_y,
            row_profile,
            col_profile,
            moments,
        ],
        axis=-1,
    )


def mnist_feature_map(
    images: Array,
    *,
    mode: str = "structural",
    feature_model: Optional[MNISTFeatureClassifier] = None,
) -> Array:
    """Map flat MNIST images into the configured feature space."""

    if mode == "none":
        return images
    if mode == "structural":
        return mnist_structural_features(images)
    if mode == "learned":
        if feature_model is None:
            raise ValueError("learned feature drift requires a feature model")
        return feature_model.features(images)
    raise ValueError("feature_drift_mode must be 'none', 'structural', or 'learned'")


def compute_class_prototypes(
    images: Array,
    labels: Array,
    *,
    num_classes: int = 10,
) -> Array:
    """Compute class-average image prototypes for conditional generation."""

    labels = labels.astype(jnp.int32)
    one_hot = jax.nn.one_hot(labels, int(num_classes), dtype=images.dtype)
    counts = jnp.sum(one_hot, axis=0)[:, None]
    sums = one_hot.T @ images
    return sums / jnp.maximum(counts, 1.0)


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
    feature_model: Optional[MNISTFeatureClassifier] = None,
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
    feature_model: Optional[MNISTFeatureClassifier] = None,
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
    feature_model: Optional[MNISTFeatureClassifier] = None,
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


def _tree_norm(tree: Any) -> Array:
    if hasattr(optax, "tree") and hasattr(optax.tree, "norm"):
        return optax.tree.norm(tree)
    return optax.global_norm(tree)


def build_mnist_generator_model(
    config: MNISTGeneratorExperimentConfig,
    key: jax.random.PRNGKey,
) -> eqx.Module:
    """Build the requested oscillator generator or control."""

    if config.model_family in ("kuramoto", "horn", "state_mlp"):
        steps = config.steps
        train_dynamics = True
    elif config.model_family in (
        "decoder_only",
        "horn_decoder_only",
        "state_mlp_decoder_only",
    ):
        steps = 0
        train_dynamics = False
    elif config.model_family in (
        "frozen_kuramoto",
        "frozen_horn",
        "frozen_state_mlp",
    ):
        steps = config.steps
        train_dynamics = False
    else:
        raise ValueError(
            "model_family must be 'kuramoto', 'decoder_only', "
            "'frozen_kuramoto', 'horn', 'horn_decoder_only', 'frozen_horn', "
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

    if config.model_family in ("horn", "horn_decoder_only", "frozen_horn"):
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
        "omega_scale": config.omega_scale,
        "coupling_init_scale": config.coupling_init_scale,
        "coupling_profile": config.coupling_profile,
        "coupling_length_scale": config.coupling_length_scale,
        "coupling_floor": config.coupling_floor,
        "coupling_bias_strength": config.coupling_bias_strength,
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
    if model_class is HORNImageGenerator:
        model_kwargs.update(
            {
                "horn_frequency": config.horn_frequency,
                "horn_damping": config.horn_damping,
                "horn_nonlinearity": config.horn_nonlinearity,
                "horn_state_bound": config.horn_state_bound,
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


@eqx.filter_jit
def _train_step(
    model: eqx.Module,
    opt_state: Any,
    real_batch: Array,
    label_batch: Optional[Array],
    positive_batch: Optional[Array],
    positive_label_batch: Optional[Array],
    sample_key: jax.random.PRNGKey,
    projections: Array,
    prototypes: Optional[Array],
    feature_model: Optional[MNISTFeatureClassifier],
    optimizer: optax.GradientTransformation,
    max_grad_norm: float,
    moment_weight: float,
    pixel_marginal_weight: float,
    class_moment_weight: float,
    prototype_weight: float,
    loss_mode: str,
    pixel_drift_weight: float,
    feature_drift_weight: float,
    feature_drift_mode: str,
    distributional_weight: float,
    drift_gamma: float,
    drift_temperatures: Tuple[float, ...],
    num_classes: int,
):
    def loss_fn(current_model):
        generated = current_model(sample_key, real_batch.shape[0], label_batch)
        return generator_loss(
            real_batch,
            generated,
            projections,
            labels=label_batch,
            positive_batch=positive_batch,
            positive_labels=positive_label_batch,
            gamma_batch=real_batch,
            gamma_labels=label_batch,
            prototypes=prototypes,
            feature_model=feature_model,
            num_classes=num_classes,
            moment_weight=moment_weight,
            pixel_marginal_weight=pixel_marginal_weight,
            class_moment_weight=class_moment_weight,
            prototype_weight=prototype_weight,
            loss_mode=loss_mode,
            pixel_drift_weight=pixel_drift_weight,
            feature_drift_weight=feature_drift_weight,
            feature_drift_mode=feature_drift_mode,
            distributional_weight=distributional_weight,
            drift_gamma=drift_gamma,
            drift_temperatures=drift_temperatures,
        )

    (loss_value, loss_parts), grads = eqx.filter_value_and_grad(
        loss_fn,
        has_aux=True,
    )(model)
    grad_norm = _tree_norm(grads)
    clip = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
    grads = jax.tree.map(lambda grad: grad * clip, grads)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, grad_norm, loss_parts


@eqx.filter_jit
def _eval_step(
    model: eqx.Module,
    real_batch: Array,
    label_batch: Optional[Array],
    sample_key: jax.random.PRNGKey,
    projections: Array,
    prototypes: Optional[Array],
    feature_model: Optional[MNISTFeatureClassifier],
    moment_weight: float,
    pixel_marginal_weight: float,
    class_moment_weight: float,
    prototype_weight: float,
    loss_mode: str,
    pixel_drift_weight: float,
    feature_drift_weight: float,
    feature_drift_mode: str,
    distributional_weight: float,
    drift_gamma: float,
    drift_temperatures: Tuple[float, ...],
    num_classes: int,
):
    generated = model(sample_key, real_batch.shape[0], label_batch)
    loss, parts = generator_loss(
        real_batch,
        generated,
        projections,
        labels=label_batch,
        prototypes=prototypes,
        feature_model=feature_model,
        num_classes=num_classes,
        moment_weight=moment_weight,
        pixel_marginal_weight=pixel_marginal_weight,
        class_moment_weight=class_moment_weight,
        prototype_weight=prototype_weight,
        loss_mode=loss_mode,
        pixel_drift_weight=pixel_drift_weight,
        feature_drift_weight=feature_drift_weight,
        feature_drift_mode=feature_drift_mode,
        distributional_weight=distributional_weight,
        drift_gamma=drift_gamma,
        drift_temperatures=drift_temperatures,
    )
    return loss, parts


def evaluate_generator_loss(
    model: eqx.Module,
    real_images: Array,
    *,
    batch_size: int,
    projections: Array,
    key: jax.random.PRNGKey,
    moment_weight: float,
    pixel_marginal_weight: float,
    labels: Optional[Array] = None,
    prototypes: Optional[Array] = None,
    feature_model: Optional[MNISTFeatureClassifier] = None,
    class_moment_weight: float = 0.0,
    prototype_weight: float = 0.0,
    loss_mode: str = "distributional",
    pixel_drift_weight: float = 1.0,
    feature_drift_weight: float = 1.0,
    feature_drift_mode: str = "structural",
    distributional_weight: float = 0.0,
    drift_gamma: float = 0.2,
    drift_temperatures: Tuple[float, ...] = (0.02, 0.05, 0.2),
    num_classes: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate mean distributional loss over real-image batches."""

    losses = []
    swd_losses = []
    moment_losses = []
    marginal_losses = []
    class_moment_losses = []
    prototype_losses = []
    pixel_drift_losses = []
    feature_drift_losses = []
    if labels is None:
        iterator = (
            (batch, None)
            for batch in iter_sample_batches(
                real_images,
                batch_size,
                jax.random.PRNGKey(0),
                shuffle=False,
            )
        )
    else:
        iterator = iter_input_target_batches(
            real_images,
            labels,
            batch_size,
            jax.random.PRNGKey(0),
            shuffle=False,
        )
    for batch_index, (real_batch, label_batch) in enumerate(iterator):
        sample_key = jax.random.fold_in(key, batch_index)
        loss, parts = _eval_step(
            model,
            real_batch,
            label_batch,
            sample_key,
            projections,
            prototypes,
            feature_model,
            moment_weight,
            pixel_marginal_weight,
            class_moment_weight,
            prototype_weight,
            loss_mode,
            pixel_drift_weight,
            feature_drift_weight,
            feature_drift_mode,
            distributional_weight,
            drift_gamma,
            drift_temperatures,
            num_classes,
        )
        losses.append(float(loss))
        swd_losses.append(float(parts["sliced_wasserstein"]))
        moment_losses.append(float(parts["moment_loss"]))
        marginal_losses.append(float(parts["pixel_marginal_loss"]))
        class_moment_losses.append(float(parts["class_moment_loss"]))
        prototype_losses.append(float(parts["prototype_loss"]))
        pixel_drift_losses.append(float(parts["pixel_drift_loss"]))
        feature_drift_losses.append(float(parts["feature_drift_loss"]))
    if not losses:
        return float("nan"), {
            "eval_sliced_wasserstein": float("nan"),
            "eval_moment_loss": float("nan"),
            "eval_pixel_marginal_loss": float("nan"),
            "eval_class_moment_loss": float("nan"),
            "eval_prototype_loss": float("nan"),
            "eval_pixel_drift_loss": float("nan"),
            "eval_feature_drift_loss": float("nan"),
        }
    return float(np.mean(losses)), {
        "eval_sliced_wasserstein": float(np.mean(swd_losses)),
        "eval_moment_loss": float(np.mean(moment_losses)),
        "eval_pixel_marginal_loss": float(np.mean(marginal_losses)),
        "eval_class_moment_loss": float(np.mean(class_moment_losses)),
        "eval_prototype_loss": float(np.mean(prototype_losses)),
        "eval_pixel_drift_loss": float(np.mean(pixel_drift_losses)),
        "eval_feature_drift_loss": float(np.mean(feature_drift_losses)),
    }


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
    classifier: Optional[MNISTFeatureClassifier] = None,
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


def _save_image_grid(images: Array, path: Path, *, columns: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images_np = np.asarray(images, dtype=np.float32).reshape(-1, 28, 28)
    rows = int(np.ceil(images_np.shape[0] / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows))
    axes_np = np.asarray(axes).reshape(rows, columns)
    for index, axis in enumerate(axes_np.flat):
        axis.axis("off")
        if index < images_np.shape[0]:
            axis.imshow(np.clip(images_np[index], 0.0, 1.0), cmap="gray", vmin=0, vmax=1)
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
    )


def _save_metrics_bundle(metrics: Dict[str, Any], paths: ExperimentPaths) -> None:
    write_json(paths.metrics / "history.json", metrics)
    save_metrics_csv(metrics, paths.metrics / "history.csv")
    save_loss_curve(metrics, paths.plots / "loss_curve.png")


def _checkpoint_hyperparams(config: MNISTGeneratorExperimentConfig) -> Dict[str, Any]:
    return {
        "experiment_family": "mnist_generator",
        "model_family": config.model_family,
        "num_oscillators": config.num_oscillators,
        "decoder_hidden_dim": config.decoder_hidden_dim,
        "decoder_depth": config.decoder_depth,
        "steps": config.steps,
        "dt": config.dt,
        "coupling_strength": config.coupling_strength,
        "omega_scale": config.omega_scale,
        "coupling_init_scale": config.coupling_init_scale,
        "coupling_profile": config.coupling_profile,
        "coupling_length_scale": config.coupling_length_scale,
        "coupling_floor": config.coupling_floor,
        "coupling_bias_strength": config.coupling_bias_strength,
        "horn_frequency": config.horn_frequency,
        "horn_damping": config.horn_damping,
        "horn_nonlinearity": config.horn_nonlinearity,
        "horn_state_bound": config.horn_state_bound,
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
        "learned_feature_dim": config.learned_feature_dim,
        "learned_feature_depth": config.learned_feature_depth,
        "learned_feature_learning_rate": config.learned_feature_learning_rate,
        "learned_feature_weight_decay": config.learned_feature_weight_decay,
        "quality_classifier_epochs": config.quality_classifier_epochs,
        "quality_classifier_dim": config.quality_classifier_dim,
        "quality_classifier_depth": config.quality_classifier_depth,
        "quality_classifier_learning_rate": config.quality_classifier_learning_rate,
        "quality_classifier_weight_decay": config.quality_classifier_weight_decay,
        "drift_queue_size": config.drift_queue_size,
        "drift_queue_num_pos": config.drift_queue_num_pos,
        "distributional_weight": config.distributional_weight,
        "drift_gamma": config.drift_gamma,
        "drift_temperatures": config.drift_temperatures,
    }


def run_mnist_generator_experiment(
    config: MNISTGeneratorExperimentConfig,
) -> AutoencoderExperimentResult:
    """Train an Un-0-style oscillator generator on MNIST."""

    if config.run.mode != "train":
        raise ValueError("mnist_generator currently supports train mode only")
    uses_feature_drift = config.loss_mode in ("feature_drift", "pixel_feature_drift")
    uses_drift_loss = config.loss_mode in (
        "pixel_drift",
        "feature_drift",
        "pixel_feature_drift",
    )
    if config.feature_drift_mode == "learned":
        if not uses_feature_drift:
            raise ValueError("feature_drift_mode='learned' requires feature drift loss")
        if not config.conditional:
            raise ValueError("learned feature drift requires conditional labels")
        if config.learned_feature_epochs < 1:
            raise ValueError("learned feature drift requires learned_feature_epochs >= 1")
    if config.drift_queue_size > 0:
        if not config.conditional or not uses_drift_loss:
            raise ValueError("drift queue requires conditional drift training")
        if config.drift_queue_num_pos < 1:
            raise ValueError("drift_queue_num_pos must be positive when using a queue")
        if config.drift_queue_num_pos > config.drift_queue_size:
            raise ValueError("drift_queue_num_pos cannot exceed drift_queue_size")

    logger = _logger()
    train_images, train_labels, eval_images, eval_labels = load_mnist_data(
        source=config.data_source,
        train_limit=config.train_limit,
        eval_limit=config.eval_limit,
        seed=config.run.seed,
    )
    train_labels = train_labels.astype(jnp.int32)
    eval_labels = eval_labels.astype(jnp.int32)
    prototypes = None
    if config.conditional:
        prototypes = compute_class_prototypes(
            train_images,
            train_labels,
            num_classes=config.num_classes,
        )
    paths = prepare_experiment_paths(config.run, asdict(config))
    key = jax.random.PRNGKey(config.run.seed)
    key, model_key, projection_key, feature_key = jax.random.split(key, 4)
    feature_model: Optional[MNISTFeatureClassifier] = None
    feature_classifier_metrics: Dict[str, Any] = {}
    quality_classifier_model: Optional[MNISTFeatureClassifier] = None
    quality_classifier_metrics: Dict[str, Any] = {}
    if config.feature_drift_mode == "learned":
        logger.info(
            "training learned feature classifier epochs=%s feature_dim=%s",
            config.learned_feature_epochs,
            config.learned_feature_dim,
        )
        feature_model, feature_classifier_metrics = train_mnist_feature_classifier(
            train_images,
            train_labels,
            eval_images,
            eval_labels,
            key=feature_key,
            num_classes=config.num_classes,
            feature_dim=config.learned_feature_dim,
            depth=config.learned_feature_depth,
            epochs=config.learned_feature_epochs,
            batch_size=config.run.batch_size,
            learning_rate=config.learned_feature_learning_rate,
            weight_decay=config.learned_feature_weight_decay,
            max_grad_norm=config.run.max_grad_norm,
        )
        write_json(
            paths.metrics / "feature_classifier.json",
            feature_classifier_metrics,
        )
        logger.info(
            "feature_classifier eval_acc=%.4f eval_loss=%.4f",
            feature_classifier_metrics["final_eval_accuracy"],
            feature_classifier_metrics["final_eval_loss"],
        )
        quality_classifier_model = feature_model
        quality_classifier_metrics = feature_classifier_metrics
    if config.quality_classifier_epochs > 0 and quality_classifier_model is None:
        logger.info(
            "training quality classifier epochs=%s feature_dim=%s",
            config.quality_classifier_epochs,
            config.quality_classifier_dim,
        )
        quality_classifier_model, quality_classifier_metrics = (
            train_mnist_feature_classifier(
                train_images,
                train_labels,
                eval_images,
                eval_labels,
                key=feature_key,
                num_classes=config.num_classes,
                feature_dim=config.quality_classifier_dim,
                depth=config.quality_classifier_depth,
                epochs=config.quality_classifier_epochs,
                batch_size=config.run.batch_size,
                learning_rate=config.quality_classifier_learning_rate,
                weight_decay=config.quality_classifier_weight_decay,
                max_grad_norm=config.run.max_grad_norm,
            )
        )
        write_json(
            paths.metrics / "quality_classifier.json",
            quality_classifier_metrics,
        )
        logger.info(
            "quality_classifier eval_acc=%.4f eval_loss=%.4f",
            quality_classifier_metrics["final_eval_accuracy"],
            quality_classifier_metrics["final_eval_loss"],
        )
    model = build_mnist_generator_model(config, model_key)
    drift_queue: Optional[MNISTDriftQueue] = None
    if config.drift_queue_size > 0:
        drift_queue = MNISTDriftQueue.create(
            num_classes=config.num_classes,
            queue_size=config.drift_queue_size,
            image_dim=model.image_dim,
            seed=config.run.seed + 99_001,
        )
    projections = make_projection_matrix(
        projection_key,
        image_dim=model.image_dim,
        num_projections=config.num_projections,
    )
    optimizer = optax.adamw(
        learning_rate=config.run.learning_rate,
        weight_decay=config.run.weight_decay,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    metrics: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "eval_loss": [],
        "grad_norm": [],
        "learning_rate": [],
        "epoch_seconds": [],
        "train_sliced_wasserstein": [],
        "train_moment_loss": [],
        "train_pixel_marginal_loss": [],
        "train_class_moment_loss": [],
        "train_prototype_loss": [],
        "train_pixel_drift_loss": [],
        "train_feature_drift_loss": [],
        "train_drift_queue_ready": [],
        "eval_sliced_wasserstein": [],
        "eval_moment_loss": [],
        "eval_pixel_marginal_loss": [],
        "eval_class_moment_loss": [],
        "eval_prototype_loss": [],
        "eval_pixel_drift_loss": [],
        "eval_feature_drift_loss": [],
        "best_eval_loss": None,
        "best_epoch": None,
    }
    checkpoint_paths = []
    best_loss = float("inf")
    total_train_seconds = 0.0

    def record_checkpoint(path: str) -> None:
        if path not in checkpoint_paths:
            checkpoint_paths.append(path)

    for epoch in range(1, config.run.epochs + 1):
        epoch_start = time.time()
        key, epoch_key = jax.random.split(key)
        losses = []
        grad_norms = []
        swd_losses = []
        moment_losses = []
        marginal_losses = []
        class_moment_losses = []
        prototype_losses = []
        pixel_drift_losses = []
        feature_drift_losses = []
        drift_queue_ready_values = []

        if config.conditional:
            train_iterator = iter_input_target_batches(
                train_images,
                train_labels,
                config.run.batch_size,
                epoch_key,
                shuffle=config.run.shuffle,
            )
        else:
            train_iterator = (
                (batch, None)
                for batch in iter_sample_batches(
                    train_images,
                    config.run.batch_size,
                    epoch_key,
                    shuffle=config.run.shuffle,
                )
            )
        for batch_index, (real_batch, label_batch) in enumerate(train_iterator):
            sample_key = jax.random.fold_in(epoch_key, batch_index + 10_000)
            positive_batch = real_batch
            positive_label_batch = label_batch
            drift_queue_ready = False
            if drift_queue is not None and label_batch is not None:
                drift_queue.push(real_batch, label_batch)
                if drift_queue.ready(config.drift_queue_num_pos):
                    positive_batch, positive_label_batch = drift_queue.draw(
                        config.drift_queue_num_pos
                    )
                    drift_queue_ready = True
            model, opt_state, loss, grad_norm, parts = _train_step(
                model,
                opt_state,
                real_batch,
                label_batch,
                positive_batch,
                positive_label_batch,
                sample_key,
                projections,
                prototypes,
                feature_model,
                optimizer,
                config.run.max_grad_norm,
                config.moment_weight,
                config.pixel_marginal_weight,
                config.class_moment_weight,
                config.prototype_weight,
                config.loss_mode,
                config.pixel_drift_weight,
                config.feature_drift_weight,
                config.feature_drift_mode,
                config.distributional_weight,
                config.drift_gamma,
                config.drift_temperatures,
                config.num_classes if config.conditional else 0,
            )
            losses.append(float(loss))
            grad_norms.append(float(grad_norm))
            swd_losses.append(float(parts["sliced_wasserstein"]))
            moment_losses.append(float(parts["moment_loss"]))
            marginal_losses.append(float(parts["pixel_marginal_loss"]))
            class_moment_losses.append(float(parts["class_moment_loss"]))
            prototype_losses.append(float(parts["prototype_loss"]))
            pixel_drift_losses.append(float(parts["pixel_drift_loss"]))
            feature_drift_losses.append(float(parts["feature_drift_loss"]))
            drift_queue_ready_values.append(float(drift_queue_ready))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        grad_norm = float(np.mean(grad_norms)) if grad_norms else float("nan")
        train_swd = float(np.mean(swd_losses)) if swd_losses else float("nan")
        train_moment = float(np.mean(moment_losses)) if moment_losses else float("nan")
        train_marginal = (
            float(np.mean(marginal_losses)) if marginal_losses else float("nan")
        )
        train_class_moment = (
            float(np.mean(class_moment_losses))
            if class_moment_losses
            else float("nan")
        )
        train_prototype = (
            float(np.mean(prototype_losses)) if prototype_losses else float("nan")
        )
        train_pixel_drift = (
            float(np.mean(pixel_drift_losses))
            if pixel_drift_losses
            else float("nan")
        )
        train_feature_drift = (
            float(np.mean(feature_drift_losses))
            if feature_drift_losses
            else float("nan")
        )
        train_drift_queue_ready = (
            float(np.mean(drift_queue_ready_values))
            if drift_queue_ready_values
            else 0.0
        )
        eval_loss = None
        eval_parts = {
            "eval_sliced_wasserstein": None,
            "eval_moment_loss": None,
            "eval_pixel_marginal_loss": None,
            "eval_class_moment_loss": None,
            "eval_prototype_loss": None,
            "eval_pixel_drift_loss": None,
            "eval_feature_drift_loss": None,
        }
        if epoch % config.run.eval_every == 0:
            eval_key = jax.random.fold_in(key, epoch + 20_000)
            eval_loss, eval_parts = evaluate_generator_loss(
                model,
                eval_images,
                batch_size=config.run.batch_size,
                projections=projections,
                key=eval_key,
                labels=eval_labels if config.conditional else None,
                prototypes=prototypes,
                feature_model=feature_model,
                moment_weight=config.moment_weight,
                pixel_marginal_weight=config.pixel_marginal_weight,
                class_moment_weight=config.class_moment_weight,
                prototype_weight=config.prototype_weight,
                loss_mode=config.loss_mode,
                pixel_drift_weight=config.pixel_drift_weight,
                feature_drift_weight=config.feature_drift_weight,
                feature_drift_mode=config.feature_drift_mode,
                distributional_weight=config.distributional_weight,
                drift_gamma=config.drift_gamma,
                drift_temperatures=config.drift_temperatures,
                num_classes=config.num_classes if config.conditional else 0,
            )

        candidate_loss = eval_loss if eval_loss is not None else train_loss
        is_best = bool(candidate_loss < best_loss)
        if is_best:
            best_loss = float(candidate_loss)
            metrics["best_eval_loss"] = best_loss
            metrics["best_epoch"] = epoch

        epoch_seconds = float(time.time() - epoch_start)
        total_train_seconds += epoch_seconds

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(train_loss)
        metrics["eval_loss"].append(eval_loss)
        metrics["grad_norm"].append(grad_norm)
        metrics["learning_rate"].append(config.run.learning_rate)
        metrics["epoch_seconds"].append(epoch_seconds)
        metrics["train_sliced_wasserstein"].append(train_swd)
        metrics["train_moment_loss"].append(train_moment)
        metrics["train_pixel_marginal_loss"].append(train_marginal)
        metrics["train_class_moment_loss"].append(train_class_moment)
        metrics["train_prototype_loss"].append(train_prototype)
        metrics["train_pixel_drift_loss"].append(train_pixel_drift)
        metrics["train_feature_drift_loss"].append(train_feature_drift)
        metrics["train_drift_queue_ready"].append(train_drift_queue_ready)
        metrics["eval_sliced_wasserstein"].append(
            eval_parts["eval_sliced_wasserstein"]
        )
        metrics["eval_moment_loss"].append(eval_parts["eval_moment_loss"])
        metrics["eval_pixel_marginal_loss"].append(
            eval_parts["eval_pixel_marginal_loss"]
        )
        metrics["eval_class_moment_loss"].append(
            eval_parts["eval_class_moment_loss"]
        )
        metrics["eval_prototype_loss"].append(eval_parts["eval_prototype_loss"])
        metrics["eval_pixel_drift_loss"].append(
            eval_parts["eval_pixel_drift_loss"]
        )
        metrics["eval_feature_drift_loss"].append(
            eval_parts["eval_feature_drift_loss"]
        )

        logger.info(
            "epoch=%s train_loss=%.6f eval_loss=%s swd=%.6f grad_norm=%.6f",
            epoch,
            train_loss,
            "none" if eval_loss is None else f"{eval_loss:.6f}",
            train_swd,
            grad_norm,
        )

        if config.run.save_best and is_best:
            checkpoint_path = save_equinox_checkpoint(
                model=model,
                opt_state=opt_state,
                epoch=epoch,
                metrics={
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "grad_norm": grad_norm,
                    "is_best": is_best,
                },
                output_dir=paths.checkpoints,
                hyperparams=_checkpoint_hyperparams(config),
                is_best=True,
            )
            record_checkpoint(checkpoint_path)

        if epoch == config.run.epochs or epoch % config.run.checkpoint_every == 0:
            checkpoint_path = save_equinox_checkpoint(
                model=model,
                opt_state=opt_state,
                epoch=epoch,
                metrics={
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "grad_norm": grad_norm,
                    "is_best": is_best,
                },
                output_dir=paths.checkpoints,
                hyperparams=_checkpoint_hyperparams(config),
                is_best=False,
            )
            record_checkpoint(checkpoint_path)

        _save_metrics_bundle(metrics, paths)

        if epoch == config.run.epochs or epoch % config.run.artifact_every == 0:
            save_mnist_generator_artifacts(
                model,
                eval_images,
                paths,
                epoch,
                key=jax.random.fold_in(key, epoch + 30_000),
                sample_count=config.eval_sample_count,
                batch_size=config.run.batch_size,
                labels=eval_labels if config.conditional else None,
            )

    eval_count = min(int(config.eval_sample_count), int(eval_images.shape[0]))
    final_generated = sample_generator_images(
        model,
        key=jax.random.fold_in(key, 40_000),
        sample_count=eval_count,
        batch_size=config.run.batch_size,
        labels=eval_labels[:eval_count] if config.conditional else None,
    )
    quality = compute_generator_quality_metrics(
        eval_images[:eval_count],
        final_generated,
        labels=eval_labels[:eval_count] if config.conditional else None,
        prototypes=prototypes,
        classifier=quality_classifier_model,
    )
    diagnostic_count = min(
        eval_count,
        int(config.run.batch_size),
        int(eval_images.shape[0]),
    )
    diagnostic_labels = (
        eval_labels[:diagnostic_count] if config.conditional else None
    )
    diagnostic_trace = model.collect_trace(
        jax.random.fold_in(key, 50_000),
        diagnostic_count,
        diagnostic_labels,
    )
    success_diagnostics = compute_generator_success_diagnostics(
        model,
        trace=diagnostic_trace,
        sample_count=int(config.run.epochs)
        * int(config.train_limit or train_images.shape[0]),
        total_train_seconds=total_train_seconds,
    )
    summary = {
        "final_train_loss": metrics["train_loss"][-1],
        "final_eval_loss": metrics["eval_loss"][-1],
        "final_train_pixel_drift_loss": metrics["train_pixel_drift_loss"][-1],
        "final_eval_pixel_drift_loss": metrics["eval_pixel_drift_loss"][-1],
        "final_train_feature_drift_loss": metrics["train_feature_drift_loss"][-1],
        "final_eval_feature_drift_loss": metrics["eval_feature_drift_loss"][-1],
        "final_train_drift_queue_ready": metrics["train_drift_queue_ready"][-1],
        "best_loss": best_loss,
        "best_epoch": metrics["best_epoch"],
        "epochs": config.run.epochs,
        "final_epoch": metrics["epoch"][-1],
        "checkpoints": checkpoint_paths,
        "generator": {
            "loss": config.loss_mode,
            "distributional_loss": "sliced_wasserstein_plus_moments_and_pixel_marginals",
            "pixel_drift_weight": config.pixel_drift_weight,
            "feature_drift_weight": config.feature_drift_weight,
            "feature_drift_mode": config.feature_drift_mode,
            "feature_classifier": feature_classifier_metrics,
            "quality_classifier": quality_classifier_metrics,
            "quality_classifier_epochs": config.quality_classifier_epochs,
            "quality_classifier_dim": config.quality_classifier_dim,
            "quality_classifier_depth": config.quality_classifier_depth,
            "drift_queue_size": config.drift_queue_size,
            "drift_queue_num_pos": config.drift_queue_num_pos,
            "drift_queue_final_counts": (
                []
                if drift_queue is None
                else [int(value) for value in drift_queue.counts.tolist()]
            ),
            "distributional_weight": config.distributional_weight,
            "drift_gamma": config.drift_gamma,
            "drift_temperatures": list(config.drift_temperatures),
            "distributional_not_paired_reconstruction": True,
            "conditional": config.conditional,
            "label_phase_scale": config.label_phase_scale,
            "coupling_profile": config.coupling_profile,
            "coupling_length_scale": config.coupling_length_scale,
            "coupling_floor": config.coupling_floor,
            "coupling_bias_strength": config.coupling_bias_strength,
            "dynamics_family": str(getattr(model, "dynamics_family", "kuramoto")),
            "horn_frequency": config.horn_frequency,
            "horn_damping": config.horn_damping,
            "horn_nonlinearity": config.horn_nonlinearity,
            "horn_state_bound": config.horn_state_bound,
            "state_mlp_hidden_dim": config.state_mlp_hidden_dim,
            "state_mlp_depth": config.state_mlp_depth,
            "state_mlp_residual_scale": config.state_mlp_residual_scale,
            "train_recurrent_dynamics": model.train_recurrent_dynamics,
            "train_conditioning_dynamics": model.train_conditioning_dynamics,
            "conditioning_mode": config.conditioning_mode,
            "readout_mode": config.readout_mode,
            "decoder_mode": config.decoder_mode,
            "resize_conv_seed_size": config.resize_conv_seed_size,
            "resize_conv_upsamples": config.resize_conv_upsamples,
            "resize_conv_min_channels": config.resize_conv_min_channels,
            **quality,
            "success_diagnostics": success_diagnostics,
        },
    }
    write_json(paths.metrics / "summary.json", summary)
    metrics.update(summary)

    return AutoencoderExperimentResult(
        model=model,
        metrics=metrics,
        paths=paths,
        checkpoint_paths=checkpoint_paths,
    )


def _parse_float_tuple(value: str | Sequence[float]) -> Tuple[float, ...]:
    if isinstance(value, str):
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(float(part) for part in value)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Un-0-style MNIST generation with oscillator dynamics."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reference/mnist_generator"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--artifact-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--model-family",
        choices=[
            "kuramoto",
            "decoder_only",
            "frozen_kuramoto",
            "horn",
            "horn_decoder_only",
            "frozen_horn",
            "state_mlp",
            "state_mlp_decoder_only",
            "frozen_state_mlp",
        ],
        default="kuramoto",
    )
    parser.add_argument("--num-oscillators", type=int, default=64)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--coupling-strength", type=float, default=1.0)
    parser.add_argument("--omega-scale", type=float, default=0.2)
    parser.add_argument("--coupling-init-scale", type=float, default=0.05)
    parser.add_argument(
        "--coupling-profile",
        choices=["dense", "distance_decay"],
        default="dense",
    )
    parser.add_argument("--coupling-length-scale", type=float, default=0.0)
    parser.add_argument("--coupling-floor", type=float, default=0.0)
    parser.add_argument("--coupling-bias-strength", type=float, default=0.0)
    parser.add_argument("--horn-frequency", type=float, default=1.0)
    parser.add_argument("--horn-damping", type=float, default=0.15)
    parser.add_argument("--horn-nonlinearity", type=float, default=0.05)
    parser.add_argument("--horn-state-bound", type=float, default=3.0)
    parser.add_argument("--state-mlp-hidden-dim", type=int, default=48)
    parser.add_argument("--state-mlp-depth", type=int, default=1)
    parser.add_argument("--state-mlp-residual-scale", type=float, default=0.1)
    parser.add_argument(
        "--train-recurrent-dynamics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--train-conditioning-dynamics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--label-phase-scale", type=float, default=0.5)
    parser.add_argument("--num-condition-oscillators", type=int, default=0)
    parser.add_argument(
        "--conditioning-mode",
        choices=["none", "phase_shift", "class_coupling", "class_oscillator"],
        default="phase_shift",
    )
    parser.add_argument(
        "--readout-mode",
        choices=["absolute", "relative", "ref_oscillator", "mean_relative"],
        default="absolute",
    )
    parser.add_argument(
        "--decoder-mode",
        choices=["mlp", "spatial_basis", "local_basis", "resize_conv"],
        default="mlp",
    )
    parser.add_argument("--spatial-basis-sigma", type=float, default=0.0)
    parser.add_argument("--local-patch-size", type=int, default=5)
    parser.add_argument("--resize-conv-seed-size", type=int, default=7)
    parser.add_argument("--resize-conv-upsamples", type=int, default=2)
    parser.add_argument("--resize-conv-min-channels", type=int, default=8)
    parser.add_argument(
        "--output-activation",
        choices=["identity", "sigmoid", "tanh"],
        default="sigmoid",
    )
    parser.add_argument("--num-projections", type=int, default=64)
    parser.add_argument("--moment-weight", type=float, default=0.1)
    parser.add_argument("--pixel-marginal-weight", type=float, default=1.0)
    parser.add_argument("--class-moment-weight", type=float, default=0.0)
    parser.add_argument("--prototype-weight", type=float, default=0.0)
    parser.add_argument(
        "--loss-mode",
        choices=[
            "distributional",
            "pixel_drift",
            "feature_drift",
            "pixel_feature_drift",
        ],
        default="distributional",
    )
    parser.add_argument("--pixel-drift-weight", type=float, default=1.0)
    parser.add_argument("--feature-drift-weight", type=float, default=1.0)
    parser.add_argument(
        "--feature-drift-mode",
        choices=["none", "structural", "learned"],
        default="structural",
    )
    parser.add_argument("--learned-feature-epochs", type=int, default=0)
    parser.add_argument("--learned-feature-dim", type=int, default=128)
    parser.add_argument("--learned-feature-depth", type=int, default=2)
    parser.add_argument("--learned-feature-learning-rate", type=float, default=1e-3)
    parser.add_argument("--learned-feature-weight-decay", type=float, default=1e-4)
    parser.add_argument("--quality-classifier-epochs", type=int, default=0)
    parser.add_argument("--quality-classifier-dim", type=int, default=128)
    parser.add_argument("--quality-classifier-depth", type=int, default=2)
    parser.add_argument("--quality-classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--quality-classifier-weight-decay", type=float, default=1e-4)
    parser.add_argument("--drift-queue-size", type=int, default=0)
    parser.add_argument("--drift-queue-num-pos", type=int, default=0)
    parser.add_argument("--distributional-weight", type=float, default=0.0)
    parser.add_argument("--drift-gamma", type=float, default=0.2)
    parser.add_argument(
        "--drift-temperatures",
        type=_parse_float_tuple,
        default=(0.02, 0.05, 0.2),
        help="Comma-separated drift temperatures, e.g. '0.02,0.05,0.2'.",
    )
    parser.add_argument("--output-bias-init", type=float, default=-2.0)
    parser.add_argument("--eval-sample-count", type=int, default=128)
    parser.add_argument(
        "--data-source",
        choices=["tfds", "idx", "synthetic"],
        default="idx",
    )
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--eval-limit", type=int, default=1_000)
    return parser


def config_from_args(args: argparse.Namespace) -> MNISTGeneratorExperimentConfig:
    run = AutoencoderExperimentConfig(
        name="mnist_generator",
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        artifact_every=args.artifact_every,
    )
    return MNISTGeneratorExperimentConfig(
        run=run,
        model_family=args.model_family,
        num_oscillators=args.num_oscillators,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_depth=args.decoder_depth,
        steps=args.steps,
        dt=args.dt,
        coupling_strength=args.coupling_strength,
        omega_scale=args.omega_scale,
        coupling_init_scale=args.coupling_init_scale,
        coupling_profile=args.coupling_profile,
        coupling_length_scale=args.coupling_length_scale,
        coupling_floor=args.coupling_floor,
        coupling_bias_strength=args.coupling_bias_strength,
        horn_frequency=args.horn_frequency,
        horn_damping=args.horn_damping,
        horn_nonlinearity=args.horn_nonlinearity,
        horn_state_bound=args.horn_state_bound,
        state_mlp_hidden_dim=args.state_mlp_hidden_dim,
        state_mlp_depth=args.state_mlp_depth,
        state_mlp_residual_scale=args.state_mlp_residual_scale,
        train_recurrent_dynamics=args.train_recurrent_dynamics,
        train_conditioning_dynamics=args.train_conditioning_dynamics,
        conditional=args.conditional,
        num_classes=args.num_classes,
        label_phase_scale=args.label_phase_scale,
        num_condition_oscillators=args.num_condition_oscillators,
        conditioning_mode=args.conditioning_mode,
        readout_mode=args.readout_mode,
        decoder_mode=args.decoder_mode,
        spatial_basis_sigma=args.spatial_basis_sigma,
        local_patch_size=args.local_patch_size,
        resize_conv_seed_size=args.resize_conv_seed_size,
        resize_conv_upsamples=args.resize_conv_upsamples,
        resize_conv_min_channels=args.resize_conv_min_channels,
        output_activation=args.output_activation,
        output_bias_init=args.output_bias_init,
        num_projections=args.num_projections,
        moment_weight=args.moment_weight,
        pixel_marginal_weight=args.pixel_marginal_weight,
        class_moment_weight=args.class_moment_weight,
        prototype_weight=args.prototype_weight,
        loss_mode=args.loss_mode,
        pixel_drift_weight=args.pixel_drift_weight,
        feature_drift_weight=args.feature_drift_weight,
        feature_drift_mode=args.feature_drift_mode,
        learned_feature_epochs=args.learned_feature_epochs,
        learned_feature_dim=args.learned_feature_dim,
        learned_feature_depth=args.learned_feature_depth,
        learned_feature_learning_rate=args.learned_feature_learning_rate,
        learned_feature_weight_decay=args.learned_feature_weight_decay,
        quality_classifier_epochs=args.quality_classifier_epochs,
        quality_classifier_dim=args.quality_classifier_dim,
        quality_classifier_depth=args.quality_classifier_depth,
        quality_classifier_learning_rate=args.quality_classifier_learning_rate,
        quality_classifier_weight_decay=args.quality_classifier_weight_decay,
        drift_queue_size=args.drift_queue_size,
        drift_queue_num_pos=args.drift_queue_num_pos,
        distributional_weight=args.distributional_weight,
        drift_gamma=args.drift_gamma,
        drift_temperatures=args.drift_temperatures,
        eval_sample_count=args.eval_sample_count,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )


def main(argv: Optional[list[str]] = None) -> AutoencoderExperimentResult:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_mnist_generator_experiment(config_from_args(args))


if __name__ == "__main__":
    main()
