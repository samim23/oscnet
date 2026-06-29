"""Feature extractors and feature-space helpers for MNIST generators."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from oscnet.experiments.harness import iter_input_target_batches
from oscnet.models.generative.common import _image_hw_channels

from .common import Array, _tree_norm


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


class ConvImageFeatureClassifier(eqx.Module):
    """Small convolutional classifier for image-quality diagnostics."""

    conv_layers: Tuple[eqx.nn.Conv2d, ...]
    hidden_layer: eqx.nn.Linear
    output_layer: eqx.nn.Linear
    image_shape: Tuple[int, ...] = eqx.field(static=True)
    image_channels: int = eqx.field(static=True)
    feature_dim: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        image_dim: int = 28 * 28,
        image_shape: Tuple[int, ...] = (28, 28),
        feature_dim: int = 128,
        depth: int = 3,
        num_classes: int = 10,
        key: jax.random.PRNGKey,
    ):
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if depth < 1:
            raise ValueError("conv feature classifier depth must be positive")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        height, width, channels = _image_hw_channels(tuple(int(size) for size in image_shape))
        if height * width * channels != int(image_dim):
            raise ValueError("image_shape must match image_dim")
        keys = jax.random.split(key, depth + 2)
        conv_layers = []
        in_channels = channels
        current_height = height
        current_width = width
        for layer_index in range(depth):
            out_channels = min(int(feature_dim), 16 * (2**layer_index))
            conv_layers.append(
                eqx.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=keys[layer_index],
                )
            )
            in_channels = out_channels
            current_height = (current_height + 1) // 2
            current_width = (current_width + 1) // 2
        flattened_dim = int(in_channels * current_height * current_width)
        self.conv_layers = tuple(conv_layers)
        self.hidden_layer = eqx.nn.Linear(
            flattened_dim,
            int(feature_dim),
            key=keys[-2],
        )
        self.output_layer = eqx.nn.Linear(
            int(feature_dim),
            int(num_classes),
            key=keys[-1],
        )
        self.image_shape = (height, width, channels)
        self.image_channels = channels
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)

    def _features_single(self, image: Array) -> Array:
        height, width, channels = _image_hw_channels(self.image_shape)
        hidden = image.reshape(channels, height, width)
        for layer in self.conv_layers:
            hidden = jax.nn.gelu(layer(hidden))
        hidden = jax.nn.gelu(self.hidden_layer(hidden.reshape(-1)))
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


class ResidualConvBlock(eqx.Module):
    """Small residual image block used by the stronger diagnostic classifier."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    skip: eqx.nn.Conv2d | None

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        key: jax.random.PRNGKey,
    ):
        conv1_key, conv2_key, skip_key = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            stride=int(stride),
            padding=1,
            key=conv1_key,
        )
        self.conv2 = eqx.nn.Conv2d(
            int(out_channels),
            int(out_channels),
            kernel_size=3,
            stride=1,
            padding=1,
            key=conv2_key,
        )
        self.skip = (
            eqx.nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=1,
                stride=int(stride),
                padding=0,
                key=skip_key,
            )
            if int(in_channels) != int(out_channels) or int(stride) != 1
            else None
        )

    def __call__(self, image: Array) -> Array:
        residual = image if self.skip is None else self.skip(image)
        hidden = jax.nn.gelu(self.conv1(image))
        hidden = self.conv2(hidden)
        return jax.nn.gelu(hidden + residual)


class ResidualConvImageFeatureClassifier(eqx.Module):
    """Residual convolutional classifier for stronger image-quality diagnostics."""

    stem: eqx.nn.Conv2d
    blocks: Tuple[ResidualConvBlock, ...]
    hidden_layer: eqx.nn.Linear
    output_layer: eqx.nn.Linear
    image_shape: Tuple[int, ...] = eqx.field(static=True)
    image_channels: int = eqx.field(static=True)
    feature_dim: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        image_dim: int = 28 * 28,
        image_shape: Tuple[int, ...] = (28, 28),
        feature_dim: int = 128,
        depth: int = 3,
        num_classes: int = 10,
        key: jax.random.PRNGKey,
    ):
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if depth < 1:
            raise ValueError("residual conv feature classifier depth must be positive")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        height, width, channels = _image_hw_channels(
            tuple(int(size) for size in image_shape)
        )
        if height * width * channels != int(image_dim):
            raise ValueError("image_shape must match image_dim")
        keys = jax.random.split(key, depth + 3)
        base_channels = min(int(feature_dim), max(16, int(feature_dim) // 4))
        self.stem = eqx.nn.Conv2d(
            channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            key=keys[0],
        )
        blocks = []
        in_channels = base_channels
        for layer_index in range(int(depth)):
            out_channels = min(int(feature_dim), base_channels * (2**layer_index))
            stride = 1 if layer_index == 0 else 2
            blocks.append(
                ResidualConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    key=keys[layer_index + 1],
                )
            )
            in_channels = out_channels
        self.blocks = tuple(blocks)
        self.hidden_layer = eqx.nn.Linear(
            int(in_channels),
            int(feature_dim),
            key=keys[-2],
        )
        self.output_layer = eqx.nn.Linear(
            int(feature_dim),
            int(num_classes),
            key=keys[-1],
        )
        self.image_shape = (height, width, channels)
        self.image_channels = channels
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)

    def _features_single(self, image: Array) -> Array:
        height, width, channels = _image_hw_channels(self.image_shape)
        hidden = image.reshape(channels, height, width)
        hidden = jax.nn.gelu(self.stem(hidden))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = jnp.mean(hidden, axis=(1, 2))
        hidden = jax.nn.gelu(self.hidden_layer(hidden))
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


FeatureClassifier = Union[
    MNISTFeatureClassifier,
    ConvImageFeatureClassifier,
    ResidualConvImageFeatureClassifier,
]


def _build_feature_classifier(
    *,
    classifier_kind: str,
    image_dim: int,
    image_shape: Tuple[int, ...],
    feature_dim: int,
    depth: int,
    num_classes: int,
    key: jax.random.PRNGKey,
) -> FeatureClassifier:
    """Build a feature classifier for learned drift or sample-quality scoring."""

    if classifier_kind == "mlp":
        return MNISTFeatureClassifier(
            image_dim=int(image_dim),
            feature_dim=int(feature_dim),
            depth=int(depth),
            num_classes=int(num_classes),
            key=key,
        )
    if classifier_kind == "conv":
        return ConvImageFeatureClassifier(
            image_dim=int(image_dim),
            image_shape=tuple(int(size) for size in image_shape),
            feature_dim=int(feature_dim),
            depth=int(depth),
            num_classes=int(num_classes),
            key=key,
        )
    if classifier_kind == "residual_conv":
        return ResidualConvImageFeatureClassifier(
            image_dim=int(image_dim),
            image_shape=tuple(int(size) for size in image_shape),
            feature_dim=int(feature_dim),
            depth=int(depth),
            num_classes=int(num_classes),
            key=key,
        )
    raise ValueError("classifier_kind must be 'mlp', 'conv', or 'residual_conv'")


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
    classifier: FeatureClassifier,
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
    classifier: FeatureClassifier,
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
    classifier: FeatureClassifier,
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
    classifier_kind: str = "mlp",
    image_shape: Tuple[int, ...] = (28, 28),
) -> Tuple[FeatureClassifier, Dict[str, Any]]:
    """Train a small feature classifier that will be frozen for drift loss."""

    if epochs < 1:
        raise ValueError("learned feature drift requires at least one feature epoch")
    classifier = _build_feature_classifier(
        classifier_kind=classifier_kind,
        image_dim=int(train_images.shape[-1]),
        image_shape=tuple(int(size) for size in image_shape),
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
    history["classifier_kind"] = classifier_kind
    history["final_train_accuracy"] = history["train_accuracy"][-1]
    history["final_eval_accuracy"] = history["eval_accuracy"][-1]
    history["final_eval_loss"] = history["eval_loss"][-1]
    return classifier, history


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
    image_shape: Tuple[int, ...] = (28, 28),
) -> Array:
    """Differentiable fixed features for MNIST-scale structural drift.

    These features are intentionally lightweight: pooled ink layout, pooled
    signed edge fields, row/column profiles, and low-order image moments. They
    are not a learned semantic encoder, but they make the drift objective care
    about shape-level organization rather than only raw pixel proximity.
    """

    height, width, channels = _image_hw_channels(tuple(int(size) for size in image_shape))
    if images.shape[-1] != height * width * channels:
        raise ValueError("images must be flat vectors matching image_shape")
    grid = images.reshape(images.shape[0], channels, height, width)
    if channels == 3:
        grid = (
            0.2989 * grid[:, 0]
            + 0.5870 * grid[:, 1]
            + 0.1140 * grid[:, 2]
        )
    else:
        grid = jnp.mean(grid, axis=1)
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
    feature_model: Optional[FeatureClassifier] = None,
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
