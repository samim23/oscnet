"""MNIST rectified-flow experiments with oscillatory phase-rate fields."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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
    prepare_experiment_paths,
    save_loss_curve,
    save_metrics_csv,
    write_json,
)
from oscnet.experiments.mnist_autoencoder import load_mnist_data
from oscnet.models import (
    CoarseGlobalPhaseRateFlowField,
    PhaseRateFlowField,
    RecurrentConvFlowField,
)
from oscnet.utils import save_equinox_checkpoint

Array = jnp.ndarray
PhaseFlowModel = (
    PhaseRateFlowField | CoarseGlobalPhaseRateFlowField | RecurrentConvFlowField
)


@dataclass(frozen=True)
class MNISTPhaseFlowExperimentConfig:
    """Task-specific controls for MNIST oscillator rectified-flow training."""

    run: AutoencoderExperimentConfig
    model_family: str = "phase_flow"
    field_channels: int = 8
    steps: int = 8
    kernel_size: int = 3
    dt: float = 0.15
    coupling_strength: float = 1.0
    rate_update: float = 0.5
    input_drive_strength: float = 0.5
    global_coupling_strength: float = 0.5
    coarse_grid_size: int = 4
    omega_scale: float = 0.2
    kernel_init_scale: float = 0.05
    position_features: bool = False
    conditional: bool = True
    num_classes: int = 10
    clean_loss_weight: float = 0.25
    closure_loss_weight: float = 0.0
    t_min: float = 1e-3
    t_max: float = 0.999
    eval_sample_count: int = 64
    sample_steps: int = 16
    sample_method: str = "euler"
    target_representation: str = "pixels"
    data_source: str = "idx"
    train_limit: Optional[int] = 10_000
    eval_limit: Optional[int] = 1_000


def _logger() -> logging.Logger:
    logger = logging.getLogger("oscnet.experiments.mnist_phase_flow")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def build_mnist_phase_flow_model(
    config: MNISTPhaseFlowExperimentConfig,
    key: jax.random.PRNGKey,
) -> PhaseFlowModel:
    """Build the requested phase-flow model or matched control."""

    if config.model_family == "phase_flow":
        steps = config.steps
        train_dynamics = True
    elif config.model_family == "frozen_phase_flow":
        steps = config.steps
        train_dynamics = False
    elif config.model_family == "phase_flow_no_dynamics":
        steps = 0
        train_dynamics = False
    elif config.model_family == "coarse_phase_flow":
        return CoarseGlobalPhaseRateFlowField(
            field_channels=config.field_channels,
            steps=config.steps,
            kernel_size=config.kernel_size,
            coarse_grid_size=config.coarse_grid_size,
            dt=config.dt,
            coupling_strength=config.coupling_strength,
            rate_update=config.rate_update,
            input_drive_strength=config.input_drive_strength,
            global_coupling_strength=config.global_coupling_strength,
            omega_scale=config.omega_scale,
            kernel_init_scale=config.kernel_init_scale,
            train_dynamics=True,
            num_classes=config.num_classes if config.conditional else 0,
            position_features=config.position_features,
            key=key,
        )
    elif config.model_family == "recurrent_conv_flow":
        return RecurrentConvFlowField(
            field_channels=config.field_channels,
            steps=config.steps,
            kernel_size=config.kernel_size,
            dt=config.dt,
            coupling_strength=config.coupling_strength,
            rate_update=config.rate_update,
            input_drive_strength=config.input_drive_strength,
            kernel_init_scale=config.kernel_init_scale,
            train_dynamics=True,
            num_classes=config.num_classes if config.conditional else 0,
            position_features=config.position_features,
            key=key,
        )
    else:
        raise ValueError(
            "model_family must be 'phase_flow', 'frozen_phase_flow', "
            "'phase_flow_no_dynamics', 'coarse_phase_flow', "
            "or 'recurrent_conv_flow'"
        )
    return PhaseRateFlowField(
        field_channels=config.field_channels,
        steps=steps,
        kernel_size=config.kernel_size,
        dt=config.dt,
        coupling_strength=config.coupling_strength,
        rate_update=config.rate_update,
        input_drive_strength=config.input_drive_strength,
        omega_scale=config.omega_scale,
        kernel_init_scale=config.kernel_init_scale,
        train_dynamics=train_dynamics,
        num_classes=config.num_classes if config.conditional else 0,
        position_features=config.position_features,
        key=key,
    )


def _checkpoint_hyperparams(config: MNISTPhaseFlowExperimentConfig) -> Dict[str, Any]:
    return {
        "experiment": "mnist_phase_flow",
        "model_family": config.model_family,
        "field_channels": config.field_channels,
        "steps": config.steps,
        "kernel_size": config.kernel_size,
        "dt": config.dt,
        "coupling_strength": config.coupling_strength,
        "rate_update": config.rate_update,
        "input_drive_strength": config.input_drive_strength,
        "global_coupling_strength": config.global_coupling_strength,
        "coarse_grid_size": config.coarse_grid_size,
        "omega_scale": config.omega_scale,
        "kernel_init_scale": config.kernel_init_scale,
        "position_features": config.position_features,
        "conditional": config.conditional,
        "clean_loss_weight": config.clean_loss_weight,
        "closure_loss_weight": config.closure_loss_weight,
        "sample_steps": config.sample_steps,
        "sample_method": config.sample_method,
        "target_representation": config.target_representation,
    }


def sobel_edge_targets(images: Array, *, eps: float = 1e-6) -> Array:
    """Convert flat MNIST images to normalized Sobel edge-magnitude maps."""

    images = jnp.asarray(images)
    batch_size = images.shape[0]
    image_grid = images.reshape(batch_size, 28, 28, 1)
    sobel_x = jnp.asarray(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=images.dtype,
    ).reshape(3, 3, 1, 1)
    sobel_y = jnp.asarray(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=images.dtype,
    ).reshape(3, 3, 1, 1)
    grad_x = jax.lax.conv_general_dilated(
        image_grid,
        sobel_x,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    grad_y = jax.lax.conv_general_dilated(
        image_grid,
        sobel_y,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    magnitude = jnp.sqrt(grad_x**2 + grad_y**2 + float(eps))
    scale = jnp.max(magnitude, axis=(1, 2, 3), keepdims=True)
    magnitude = magnitude / (scale + float(eps))
    return jnp.clip(magnitude.reshape(batch_size, 28 * 28), 0.0, 1.0)


def prepare_phase_flow_targets(images: Array, representation: str) -> Array:
    """Prepare the image-domain target used by phase-flow training."""

    if representation == "pixels":
        return images
    if representation == "sobel_edges":
        return sobel_edge_targets(images)
    raise ValueError("target_representation must be 'pixels' or 'sobel_edges'")


def _mean_pool_flat_images(images: Array, grid_size: int) -> Array:
    """Average-pool flat 28x28 images to a square grid."""

    batch_size = images.shape[0]
    height = width = 28
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError("grid_size must divide 28")
    block = height // grid_size
    grid = images.reshape(batch_size, grid_size, block, grid_size, block)
    return grid.mean(axis=(2, 4))


def closure_loss(predicted: Array, target: Array) -> Array:
    """Compare coarse digit envelopes to encourage whole-shape binding.

    Pixel and velocity losses already supervise local strokes. This auxiliary
    loss adds a low-frequency pressure on the predicted clean endpoint so
    disconnected stroke fragments are less rewarding than a coherent digit
    envelope.
    """

    losses = []
    for grid_size in (14, 7):
        predicted_pool = _mean_pool_flat_images(predicted, grid_size)
        target_pool = _mean_pool_flat_images(target, grid_size)
        losses.append(jnp.mean((predicted_pool - target_pool) ** 2))
    return jnp.mean(jnp.asarray(losses))


def iter_labeled_batches(
    images: Array,
    labels: Array,
    batch_size: int,
    key: jax.random.PRNGKey,
    *,
    shuffle: bool = True,
) -> Iterable[Tuple[Array, Array]]:
    """Yield image/label batches with shared shuffling."""

    n_samples = int(images.shape[0])
    indices = jnp.arange(n_samples)
    if shuffle:
        indices = jax.random.permutation(key, n_samples)
    for start in range(0, int(indices.shape[0]), batch_size):
        batch_indices = indices[start : start + batch_size]
        if batch_indices.size == 0:
            continue
        yield images[batch_indices], labels[batch_indices]


def make_rectified_flow_batch(
    images: Array,
    key: jax.random.PRNGKey,
    *,
    t_min: float,
    t_max: float,
) -> Tuple[Array, Array, Array, Array]:
    """Corrupt clean images along a rectified-flow noise-data chord."""

    noise_key, time_key = jax.random.split(key)
    batch_size = images.shape[0]
    noise = jax.random.normal(noise_key, images.shape)
    t = jax.random.uniform(
        time_key,
        (batch_size,),
        minval=float(t_min),
        maxval=float(t_max),
    )
    noisy = (1.0 - t[:, None]) * noise + t[:, None] * images
    target_velocity = images - noise
    return noisy, t, target_velocity, noise


def phase_flow_loss(
    model: PhaseFlowModel,
    images: Array,
    labels: Array,
    key: jax.random.PRNGKey,
    *,
    clean_loss_weight: float,
    closure_loss_weight: float = 0.0,
    t_min: float,
    t_max: float,
) -> Tuple[Array, Dict[str, Array]]:
    """Return rectified-flow loss and diagnostics."""

    noisy, t, target_velocity, _ = make_rectified_flow_batch(
        images,
        key,
        t_min=t_min,
        t_max=t_max,
    )
    velocity = model(noisy, t, labels)
    velocity_loss = jnp.mean((velocity - target_velocity) ** 2)
    clean_prediction = noisy + (1.0 - t[:, None]) * velocity
    clean_loss = jnp.mean((clean_prediction - images) ** 2)
    shape_loss = closure_loss(clean_prediction, images)
    total = (
        velocity_loss
        + float(clean_loss_weight) * clean_loss
        + float(closure_loss_weight) * shape_loss
    )
    return total, {
        "velocity_loss": velocity_loss,
        "clean_loss": clean_loss,
        "closure_loss": shape_loss,
        "total_loss": total,
    }


def _tree_norm(tree: Any) -> Array:
    if hasattr(optax, "tree") and hasattr(optax.tree, "norm"):
        return optax.tree.norm(tree)
    return optax.global_norm(tree)


@eqx.filter_jit
def _train_step(
    model: PhaseFlowModel,
    opt_state: Any,
    images: Array,
    labels: Array,
    sample_key: jax.random.PRNGKey,
    optimizer: optax.GradientTransformation,
    max_grad_norm: float,
    clean_loss_weight: float,
    closure_loss_weight: float,
    t_min: float,
    t_max: float,
):
    def loss_fn(current_model):
        return phase_flow_loss(
            current_model,
            images,
            labels,
            sample_key,
            clean_loss_weight=clean_loss_weight,
            closure_loss_weight=closure_loss_weight,
            t_min=t_min,
            t_max=t_max,
        )

    (loss_value, parts), grads = eqx.filter_value_and_grad(
        loss_fn,
        has_aux=True,
    )(model)
    grad_norm = _tree_norm(grads)
    clip = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
    grads = jax.tree.map(lambda grad: grad * clip, grads)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, grad_norm, parts


@eqx.filter_jit
def _eval_step(
    model: PhaseFlowModel,
    images: Array,
    labels: Array,
    sample_key: jax.random.PRNGKey,
    clean_loss_weight: float,
    closure_loss_weight: float,
    t_min: float,
    t_max: float,
):
    return phase_flow_loss(
        model,
        images,
        labels,
        sample_key,
        clean_loss_weight=clean_loss_weight,
        closure_loss_weight=closure_loss_weight,
        t_min=t_min,
        t_max=t_max,
    )


def evaluate_phase_flow(
    model: PhaseFlowModel,
    images: Array,
    labels: Array,
    *,
    batch_size: int,
    key: jax.random.PRNGKey,
    clean_loss_weight: float,
    closure_loss_weight: float,
    t_min: float,
    t_max: float,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate rectified-flow loss over a dataset."""

    losses = []
    velocity_losses = []
    clean_losses = []
    closure_losses = []
    for batch_index, (batch_images, batch_labels) in enumerate(
        iter_labeled_batches(
            images,
            labels,
            batch_size,
            jax.random.PRNGKey(0),
            shuffle=False,
        )
    ):
        loss, parts = _eval_step(
            model,
            batch_images,
            batch_labels,
            jax.random.fold_in(key, batch_index),
            clean_loss_weight,
            closure_loss_weight,
            t_min,
            t_max,
        )
        losses.append(float(loss))
        velocity_losses.append(float(parts["velocity_loss"]))
        clean_losses.append(float(parts["clean_loss"]))
        closure_losses.append(float(parts["closure_loss"]))
    if not losses:
        return float("nan"), {
            "eval_velocity_loss": float("nan"),
            "eval_clean_loss": float("nan"),
            "eval_closure_loss": float("nan"),
        }
    return float(np.mean(losses)), {
        "eval_velocity_loss": float(np.mean(velocity_losses)),
        "eval_clean_loss": float(np.mean(clean_losses)),
        "eval_closure_loss": float(np.mean(closure_losses)),
    }


def sample_phase_flow_images(
    model: PhaseFlowModel,
    *,
    key: jax.random.PRNGKey,
    sample_count: int,
    sample_steps: int,
    sample_method: str,
    labels: Optional[Array],
    batch_size: int,
) -> Array:
    """Generate samples in batches."""

    if sample_method not in {"euler", "heun"}:
        raise ValueError("sample_method must be 'euler' or 'heun'")
    samples = []
    remaining = int(sample_count)
    start = 0
    batch_index = 0
    while remaining > 0:
        current = min(batch_size, remaining)
        current_labels = None
        if labels is not None:
            current_labels = labels[start : start + current]
        batch_key = jax.random.fold_in(key, batch_index)
        if sample_method == "euler":
            batch_samples = model.sample(
                batch_key,
                current,
                labels=current_labels,
                outer_steps=sample_steps,
            )
        else:
            batch_samples = _sample_phase_flow_heun(
                model,
                batch_key,
                current,
                labels=current_labels,
                sample_steps=sample_steps,
            )
        samples.append(batch_samples)
        start += current
        remaining -= current
        batch_index += 1
    return jnp.concatenate(samples, axis=0)


def _sample_phase_flow_heun(
    model: PhaseFlowModel,
    key: jax.random.PRNGKey,
    sample_count: int,
    *,
    labels: Optional[Array],
    sample_steps: int,
) -> Array:
    """Generate images with a second-order predictor-corrector sampler."""

    sample_count = int(sample_count)
    if sample_steps < 1:
        raise ValueError("sample_steps must be positive")
    x = jax.random.normal(key, (sample_count, model.image_dim))
    if labels is not None:
        labels = labels.astype(jnp.int32)
    step_size = 1.0 / float(sample_steps)

    def scan_fn(current_x, step_index):
        t0_value = jnp.clip(step_index.astype(current_x.dtype) * step_size, 1e-3, 0.999)
        t1_value = jnp.clip(
            (step_index.astype(current_x.dtype) + 1.0) * step_size,
            1e-3,
            0.999,
        )
        t0 = jnp.full((sample_count,), t0_value, dtype=current_x.dtype)
        t1 = jnp.full((sample_count,), t1_value, dtype=current_x.dtype)
        velocity0 = model(current_x, t0, labels)
        predictor = current_x + step_size * velocity0
        velocity1 = model(predictor, t1, labels)
        return current_x + 0.5 * step_size * (velocity0 + velocity1), None

    steps = jnp.arange(int(sample_steps), dtype=jnp.float32)
    x, _ = jax.lax.scan(scan_fn, x, steps)
    return jnp.clip(x, 0.0, 1.0)


def compute_phase_flow_quality_metrics(
    real_images: Array,
    samples: Array,
) -> Dict[str, float]:
    """Compute lightweight sample diagnostics."""

    real = np.asarray(real_images, dtype=np.float32).reshape(real_images.shape[0], -1)
    gen = np.asarray(samples, dtype=np.float32).reshape(samples.shape[0], -1)
    real_for_gen = real[: gen.shape[0]]
    pairwise = np.mean((gen[:, None, :] - real_for_gen[None, :, :]) ** 2, axis=-1)
    nearest_real_mse = np.min(pairwise, axis=1)
    real_pairwise = np.mean(
        (real_for_gen[:, None, :] - real_for_gen[None, :, :]) ** 2,
        axis=-1,
    )
    np.fill_diagonal(real_pairwise, np.inf)
    gen_std = gen.std(axis=0)
    real_std = real_for_gen.std(axis=0)
    gen_topology = _image_topology_metrics(gen)
    real_topology = _image_topology_metrics(real_for_gen)
    return {
        "sample_mean": float(np.mean(gen)),
        "sample_std": float(np.std(gen)),
        "sample_pixel_mean_mse": float(
            np.mean((gen.mean(axis=0) - real_for_gen.mean(axis=0)) ** 2)
        ),
        "sample_pixel_std_mse": float(np.mean((gen_std - real_std) ** 2)),
        "sample_diversity_ratio": float(
            np.mean(gen_std) / (np.mean(real_std) + 1e-8)
        ),
        "sample_nearest_real_mse": float(np.mean(nearest_real_mse)),
        "real_nearest_real_mse": float(np.mean(np.min(real_pairwise, axis=1))),
        **{f"sample_{key}": value for key, value in gen_topology.items()},
        **{f"real_{key}": value for key, value in real_topology.items()},
    }


def _image_topology_metrics(flat_images: np.ndarray) -> Dict[str, float]:
    """Measure simple foreground connectivity in 28x28 image samples."""

    images = np.asarray(flat_images, dtype=np.float32).reshape(-1, 28, 28)
    active_fractions = []
    component_counts = []
    largest_fractions = []
    for image in images:
        mask = image > 0.2
        active = int(mask.sum())
        active_fractions.append(active / float(mask.size))
        if active == 0:
            component_counts.append(0.0)
            largest_fractions.append(0.0)
            continue
        visited = np.zeros_like(mask, dtype=bool)
        components = 0
        largest = 0
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if not mask[y, x] or visited[y, x]:
                    continue
                components += 1
                stack = [(y, x)]
                visited[y, x] = True
                size = 0
                while stack:
                    cy, cx = stack.pop()
                    size += 1
                    for ny, nx in (
                        (cy - 1, cx),
                        (cy + 1, cx),
                        (cy, cx - 1),
                        (cy, cx + 1),
                    ):
                        if (
                            0 <= ny < mask.shape[0]
                            and 0 <= nx < mask.shape[1]
                            and mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                largest = max(largest, size)
        component_counts.append(float(components))
        largest_fractions.append(largest / float(active))
    return {
        "active_fraction": float(np.mean(active_fractions)),
        "component_count": float(np.mean(component_counts)),
        "largest_component_fraction": float(np.mean(largest_fractions)),
    }


def _save_image_grid(images: Array, path: Path, *, columns: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images_np = np.asarray(images, dtype=np.float32).reshape(-1, 28, 28)
    rows = int(np.ceil(images_np.shape[0] / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows))
    axes_np = np.asarray(axes).reshape(rows, columns)
    for index, axis in enumerate(axes_np.flat):
        axis.axis("off")
        if index < images_np.shape[0]:
            axis.imshow(
                np.clip(images_np[index], 0.0, 1.0),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
    fig.tight_layout(pad=0.05)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sample_labels(
    labels: Array,
    *,
    sample_count: int,
    conditional: bool,
    num_classes: int,
) -> Optional[Array]:
    if not conditional:
        return None
    if num_classes <= 0:
        return None
    repeated = jnp.arange(num_classes, dtype=jnp.int32)
    tiled = jnp.tile(repeated, int(np.ceil(sample_count / num_classes)))
    return tiled[:sample_count]


def save_phase_flow_artifacts(
    model: PhaseFlowModel,
    eval_images: Array,
    eval_labels: Array,
    paths: ExperimentPaths,
    epoch: int,
    *,
    key: jax.random.PRNGKey,
    sample_count: int,
    sample_steps: int,
    sample_method: str,
    batch_size: int,
    conditional: bool,
    num_classes: int,
) -> None:
    """Save denoising examples, samples, and oscillator traces."""

    count = min(int(sample_count), int(eval_images.shape[0]))
    real = eval_images[:count]
    labels = eval_labels[:count]
    noise_key, time_key, sample_key = jax.random.split(key, 3)
    noise = jax.random.normal(noise_key, real.shape)
    t = jnp.full((count,), 0.5)
    noisy = 0.5 * noise + 0.5 * real
    velocity, trace = model(noisy, t, labels, return_trace=True)
    denoised = jnp.clip(noisy + 0.5 * velocity, 0.0, 1.0)
    sample_labels = _sample_labels(
        labels,
        sample_count=count,
        conditional=conditional,
        num_classes=num_classes,
    )
    samples = sample_phase_flow_images(
        model,
        key=sample_key,
        sample_count=count,
        sample_steps=sample_steps,
        sample_method=sample_method,
        labels=sample_labels,
        batch_size=batch_size,
    )

    np.savez(
        paths.artifacts / f"mnist_phase_flow_epoch_{epoch:03d}.npz",
        real=np.asarray(real),
        noisy=np.asarray(noisy),
        denoised=np.asarray(denoised),
        samples=np.asarray(samples),
        labels=np.asarray(labels),
        sample_labels=None if sample_labels is None else np.asarray(sample_labels),
    )
    np.savez(
        paths.traces / f"mnist_phase_flow_trace_epoch_{epoch:03d}.npz",
        **{name: np.asarray(value) for name, value in trace.items()},
    )
    _save_image_grid(real[: min(count, 64)], paths.artifacts / f"real_epoch_{epoch:03d}.png")
    _save_image_grid(
        jnp.clip(noisy[: min(count, 64)], 0.0, 1.0),
        paths.artifacts / f"noisy_epoch_{epoch:03d}.png",
    )
    _save_image_grid(
        denoised[: min(count, 64)],
        paths.artifacts / f"denoised_epoch_{epoch:03d}.png",
    )
    _save_image_grid(
        samples[: min(count, 64)],
        paths.artifacts / f"samples_epoch_{epoch:03d}.png",
    )


def run_mnist_phase_flow_experiment(
    config: MNISTPhaseFlowExperimentConfig,
) -> AutoencoderExperimentResult:
    """Train/evaluate a MNIST phase-flow model."""

    logger = _logger()
    paths = prepare_experiment_paths(config.run, asdict(config))
    train_images, train_labels, eval_images, eval_labels = load_mnist_data(
        source=config.data_source,
        train_limit=config.train_limit,
        eval_limit=config.eval_limit,
        seed=config.run.seed,
    )
    train_images = prepare_phase_flow_targets(
        train_images,
        config.target_representation,
    )
    eval_images = prepare_phase_flow_targets(
        eval_images,
        config.target_representation,
    )

    key = jax.random.PRNGKey(config.run.seed)
    model_key = jax.random.fold_in(key, 1)
    model = build_mnist_phase_flow_model(config, model_key)
    optimizer = optax.adamw(
        learning_rate=config.run.learning_rate,
        weight_decay=config.run.weight_decay,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    history: Dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_velocity_loss": [],
        "train_clean_loss": [],
        "train_closure_loss": [],
        "eval_loss": [],
        "eval_velocity_loss": [],
        "eval_clean_loss": [],
        "eval_closure_loss": [],
        "grad_norm": [],
    }
    checkpoint_paths: list[str] = []
    best_loss = float("inf")
    best_epoch = 0
    train_start = time.time()

    def append_checkpoint(path: str) -> None:
        if path not in checkpoint_paths:
            checkpoint_paths.append(path)

    for epoch in range(1, config.run.epochs + 1):
        epoch_start = time.time()
        epoch_key = jax.random.fold_in(key, epoch)
        losses = []
        velocity_losses = []
        clean_losses = []
        closure_losses = []
        grad_norms = []
        for batch_index, (batch_images, batch_labels) in enumerate(
            iter_labeled_batches(
                train_images,
                train_labels,
                config.run.batch_size,
                epoch_key,
                shuffle=config.run.shuffle,
            )
        ):
            model, opt_state, loss, grad_norm, parts = _train_step(
                model,
                opt_state,
                batch_images,
                batch_labels,
                jax.random.fold_in(epoch_key, batch_index),
                optimizer,
                config.run.max_grad_norm,
                config.clean_loss_weight,
                config.closure_loss_weight,
                config.t_min,
                config.t_max,
            )
            losses.append(float(loss))
            velocity_losses.append(float(parts["velocity_loss"]))
            clean_losses.append(float(parts["clean_loss"]))
            closure_losses.append(float(parts["closure_loss"]))
            grad_norms.append(float(grad_norm))

        eval_loss, eval_parts = evaluate_phase_flow(
            model,
            eval_images,
            eval_labels,
            batch_size=config.run.batch_size,
            key=jax.random.fold_in(key, 10_000 + epoch),
            clean_loss_weight=config.clean_loss_weight,
            closure_loss_weight=config.closure_loss_weight,
            t_min=config.t_min,
            t_max=config.t_max,
        )
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            if config.run.save_best:
                checkpoint_path = save_equinox_checkpoint(
                    model=model,
                    opt_state=opt_state,
                    epoch=epoch,
                    metrics={
                        "train_loss": float(np.mean(losses)),
                        "eval_loss": eval_loss,
                        "is_best": True,
                    },
                    output_dir=paths.checkpoints,
                    hyperparams=_checkpoint_hyperparams(config),
                    is_best=True,
                )
                append_checkpoint(checkpoint_path)
        if epoch % config.run.checkpoint_every == 0 or epoch == config.run.epochs:
            checkpoint_path = save_equinox_checkpoint(
                model=model,
                opt_state=opt_state,
                epoch=epoch,
                metrics={
                    "train_loss": float(np.mean(losses)),
                    "eval_loss": eval_loss,
                    "is_best": False,
                },
                output_dir=paths.checkpoints,
                hyperparams=_checkpoint_hyperparams(config),
                is_best=False,
            )
            append_checkpoint(checkpoint_path)
        if epoch % config.run.artifact_every == 0 or epoch == config.run.epochs:
            save_phase_flow_artifacts(
                model,
                eval_images,
                eval_labels,
                paths,
                epoch,
                key=jax.random.fold_in(key, 20_000 + epoch),
                sample_count=config.eval_sample_count,
                sample_steps=config.sample_steps,
                sample_method=config.sample_method,
                batch_size=config.run.batch_size,
                conditional=config.conditional,
                num_classes=config.num_classes,
            )

        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(np.mean(losses)))
        history["train_velocity_loss"].append(float(np.mean(velocity_losses)))
        history["train_clean_loss"].append(float(np.mean(clean_losses)))
        history["train_closure_loss"].append(float(np.mean(closure_losses)))
        history["eval_loss"].append(eval_loss)
        history["eval_velocity_loss"].append(eval_parts["eval_velocity_loss"])
        history["eval_clean_loss"].append(eval_parts["eval_clean_loss"])
        history["eval_closure_loss"].append(eval_parts["eval_closure_loss"])
        history["grad_norm"].append(float(np.mean(grad_norms)))
        logger.info(
            (
                "epoch=%d train_loss=%.6f eval_loss=%.6f "
                "eval_vel=%.6f eval_clean=%.6f eval_closure=%.6f time=%.2fs"
            ),
            epoch,
            history["train_loss"][-1],
            eval_loss,
            eval_parts["eval_velocity_loss"],
            eval_parts["eval_clean_loss"],
            eval_parts["eval_closure_loss"],
            time.time() - epoch_start,
        )

    save_metrics_csv(history, paths.metrics / "history.csv")
    write_json(paths.metrics / "history.json", history)
    save_loss_curve(history, paths.plots / "loss_curve.png")

    count = min(config.eval_sample_count, int(eval_images.shape[0]))
    sample_labels = _sample_labels(
        eval_labels[:count],
        sample_count=count,
        conditional=config.conditional,
        num_classes=config.num_classes,
    )
    samples = sample_phase_flow_images(
        model,
        key=jax.random.fold_in(key, 30_001),
        sample_count=count,
        sample_steps=config.sample_steps,
        sample_method=config.sample_method,
        labels=sample_labels,
        batch_size=config.run.batch_size,
    )
    quality = compute_phase_flow_quality_metrics(eval_images[:count], samples)
    trace_noise = jax.random.normal(
        jax.random.fold_in(key, 30_002),
        eval_images[: min(count, config.run.batch_size)].shape,
    )
    trace_t = jnp.full((trace_noise.shape[0],), 0.5)
    trace_input = 0.5 * trace_noise + 0.5 * eval_images[: trace_noise.shape[0]]
    trace = model.collect_trace(
        trace_input,
        trace_t,
        eval_labels[: trace_noise.shape[0]],
    )
    phase_mean_abs_displacement = None
    state_mean_abs_displacement = None
    if "final_theta" in trace and "initial_theta" in trace:
        phase_delta = np.angle(
            np.exp(
                1j
                * (
                    np.asarray(trace["final_theta"], dtype=np.float32)
                    - np.asarray(trace["initial_theta"], dtype=np.float32)
                )
            )
        )
        phase_mean_abs_displacement = float(np.mean(np.abs(phase_delta)))
        state_mean_abs_displacement = phase_mean_abs_displacement
    elif "final_hidden" in trace and "initial_hidden" in trace:
        state_delta = np.asarray(trace["final_hidden"], dtype=np.float32) - np.asarray(
            trace["initial_hidden"],
            dtype=np.float32,
        )
        state_mean_abs_displacement = float(np.mean(np.abs(state_delta)))
    summary: Dict[str, Any] = {
        "config": asdict(config),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "final_epoch": int(config.run.epochs),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_eval_loss": float(history["eval_loss"][-1]),
        "final_eval_velocity_loss": float(history["eval_velocity_loss"][-1]),
        "final_eval_clean_loss": float(history["eval_clean_loss"][-1]),
        "final_eval_closure_loss": float(history["eval_closure_loss"][-1]),
        "phase_flow": {
            "model_family": config.model_family,
            "field_channels": int(model.field_channels),
            "steps": int(model.steps),
            "train_dynamics": bool(model.train_dynamics),
            "conditional": bool(config.conditional),
            "position_features": bool(config.position_features),
            "clean_loss_weight": float(config.clean_loss_weight),
            "closure_loss_weight": float(config.closure_loss_weight),
            "target_representation": config.target_representation,
            "sample_steps": int(config.sample_steps),
            "sample_method": config.sample_method,
            "coarse_grid_size": (
                int(model.coarse_grid_size)
                if hasattr(model, "coarse_grid_size")
                else None
            ),
            "global_coupling_strength": (
                float(model.global_coupling_strength)
                if hasattr(model, "global_coupling_strength")
                else None
            ),
            "phase_mean_abs_displacement": phase_mean_abs_displacement,
            "state_mean_abs_displacement": state_mean_abs_displacement,
            **quality,
        },
        "checkpoints": checkpoint_paths,
        "train_seconds": float(time.time() - train_start),
    }
    write_json(paths.metrics / "summary.json", summary)
    return AutoencoderExperimentResult(
        model=model,
        metrics=summary,
        paths=paths,
        checkpoint_paths=checkpoint_paths,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reference/mnist_phase_flow"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--artifact-every", type=int, default=5)
    parser.add_argument(
        "--model-family",
        choices=[
            "phase_flow",
            "frozen_phase_flow",
            "phase_flow_no_dynamics",
            "coarse_phase_flow",
            "recurrent_conv_flow",
        ],
        default="phase_flow",
    )
    parser.add_argument("--field-channels", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.15)
    parser.add_argument("--coupling-strength", type=float, default=1.0)
    parser.add_argument("--rate-update", type=float, default=0.5)
    parser.add_argument("--input-drive-strength", type=float, default=0.5)
    parser.add_argument("--global-coupling-strength", type=float, default=0.5)
    parser.add_argument("--coarse-grid-size", type=int, default=4)
    parser.add_argument("--omega-scale", type=float, default=0.2)
    parser.add_argument("--kernel-init-scale", type=float, default=0.05)
    parser.add_argument(
        "--position-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--conditional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--clean-loss-weight", type=float, default=0.25)
    parser.add_argument("--closure-loss-weight", type=float, default=0.0)
    parser.add_argument("--t-min", type=float, default=1e-3)
    parser.add_argument("--t-max", type=float, default=0.999)
    parser.add_argument("--eval-sample-count", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--sample-method", choices=["euler", "heun"], default="euler")
    parser.add_argument(
        "--target-representation",
        choices=["pixels", "sobel_edges"],
        default="pixels",
    )
    parser.add_argument("--data-source", choices=["idx", "synthetic", "tfds"], default="idx")
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--eval-limit", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_config = AutoencoderExperimentConfig(
        name="mnist_phase_flow",
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        checkpoint_every=args.checkpoint_every,
        artifact_every=args.artifact_every,
    )
    config = MNISTPhaseFlowExperimentConfig(
        run=run_config,
        model_family=args.model_family,
        field_channels=args.field_channels,
        steps=args.steps,
        kernel_size=args.kernel_size,
        dt=args.dt,
        coupling_strength=args.coupling_strength,
        rate_update=args.rate_update,
        input_drive_strength=args.input_drive_strength,
        global_coupling_strength=args.global_coupling_strength,
        coarse_grid_size=args.coarse_grid_size,
        omega_scale=args.omega_scale,
        kernel_init_scale=args.kernel_init_scale,
        position_features=args.position_features,
        conditional=args.conditional,
        num_classes=args.num_classes,
        clean_loss_weight=args.clean_loss_weight,
        closure_loss_weight=args.closure_loss_weight,
        t_min=args.t_min,
        t_max=args.t_max,
        eval_sample_count=args.eval_sample_count,
        sample_steps=args.sample_steps,
        sample_method=args.sample_method,
        target_representation=args.target_representation,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )
    result = run_mnist_phase_flow_experiment(config)
    print(json.dumps(result.metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
