"""MNIST two-stage shape-to-pixel experiments.

This harness tests the representation suggested by the phase-flow basin probe:
first settle a smooth signed-distance shape scaffold, then render pixels from
that scaffold. The renderer is still a phase-flow style recurrent field. Its
visible state has two channels:

```
channel 0: noisy/generated pixel image
channel 1: fixed signed-distance shape condition
```

During sampling, the shape channel is clamped after every integration step so
the model is judged as a shape-conditioned pixel renderer, not as another
joint pixel/shape generator.
"""

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
from oscnet.experiments.mnist_phase_flow import (
    basin_t_key,
    closure_loss,
    compute_phase_flow_quality_metrics,
    parse_basin_t_values,
    parse_noise_modes,
    phase_flow_noise_endpoint,
    signed_distance_targets,
)
from oscnet.models import (
    CoarseGlobalPhaseRateFlowField,
    PhaseRateFlowField,
    RecurrentConvFlowField,
)
from oscnet.utils import save_equinox_checkpoint

Array = jnp.ndarray
ShapePixelModel = (
    PhaseRateFlowField | CoarseGlobalPhaseRateFlowField | RecurrentConvFlowField
)


@dataclass(frozen=True)
class MNISTShapePixelExperimentConfig:
    """Controls for signed-distance conditioned MNIST pixel rendering."""

    run: AutoencoderExperimentConfig
    model_family: str = "coarse_phase_flow"
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
    shape_velocity_weight: float = 0.1
    closure_loss_weight: float = 0.0
    t_min: float = 1e-3
    t_max: float = 0.999
    eval_sample_count: int = 64
    sample_steps: int = 16
    sample_method: str = "euler"
    sample_readout_mode: str = "primary"
    basin_t_values: Tuple[float, ...] = ()
    shape_condition_t_values: Tuple[float, ...] = ()
    shape_condition_noise_modes: Tuple[str, ...] = ()
    clamp_shape: bool = True
    data_source: str = "idx"
    train_limit: Optional[int] = 10_000
    eval_limit: Optional[int] = 1_000


def _logger() -> logging.Logger:
    logger = logging.getLogger("oscnet.experiments.mnist_shape_pixel")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def build_mnist_shape_pixel_model(
    config: MNISTShapePixelExperimentConfig,
    key: jax.random.PRNGKey,
) -> ShapePixelModel:
    """Build a two-channel shape-conditioned pixel flow model."""

    common = dict(
        value_channels=2,
        field_channels=config.field_channels,
        steps=config.steps,
        kernel_size=config.kernel_size,
        dt=config.dt,
        coupling_strength=config.coupling_strength,
        rate_update=config.rate_update,
        input_drive_strength=config.input_drive_strength,
        kernel_init_scale=config.kernel_init_scale,
        num_classes=config.num_classes if config.conditional else 0,
        position_features=config.position_features,
        key=key,
    )
    if config.model_family == "phase_flow":
        return PhaseRateFlowField(
            **common,
            omega_scale=config.omega_scale,
            train_dynamics=True,
        )
    if config.model_family == "frozen_phase_flow":
        return PhaseRateFlowField(
            **common,
            omega_scale=config.omega_scale,
            train_dynamics=False,
        )
    if config.model_family == "phase_flow_no_dynamics":
        return PhaseRateFlowField(
            **{**common, "steps": 0},
            omega_scale=config.omega_scale,
            train_dynamics=False,
        )
    if config.model_family == "coarse_phase_flow":
        return CoarseGlobalPhaseRateFlowField(
            **common,
            coarse_grid_size=config.coarse_grid_size,
            global_coupling_strength=config.global_coupling_strength,
            omega_scale=config.omega_scale,
            train_dynamics=True,
        )
    if config.model_family == "recurrent_conv_flow":
        return RecurrentConvFlowField(
            **common,
            train_dynamics=True,
        )
    raise ValueError(
        "model_family must be 'phase_flow', 'frozen_phase_flow', "
        "'phase_flow_no_dynamics', 'coarse_phase_flow', or "
        "'recurrent_conv_flow'"
    )


def _checkpoint_hyperparams(config: MNISTShapePixelExperimentConfig) -> Dict[str, Any]:
    return {
        "experiment": "mnist_shape_pixel",
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
        "shape_velocity_weight": config.shape_velocity_weight,
        "closure_loss_weight": config.closure_loss_weight,
        "sample_steps": config.sample_steps,
        "sample_method": config.sample_method,
        "sample_readout_mode": config.sample_readout_mode,
        "basin_t_values": [float(value) for value in config.basin_t_values],
        "shape_condition_t_values": [
            float(value) for value in config.shape_condition_t_values
        ],
        "shape_condition_noise_modes": list(config.shape_condition_noise_modes),
        "clamp_shape": config.clamp_shape,
        "value_channels": 2,
    }


def stack_pixel_shape_channels(pixels: Array, shapes: Array) -> Array:
    """Stack flat pixel and shape images into a two-channel visible field."""

    batch_size = pixels.shape[0]
    pixel_grid = pixels.reshape(batch_size, 28, 28)
    shape_grid = shapes.reshape(batch_size, 28, 28)
    return jnp.stack([pixel_grid, shape_grid], axis=-1).reshape(batch_size, -1)


def split_pixel_shape_channels(state: Array) -> Tuple[Array, Array]:
    """Extract flat pixel and shape channels from a two-channel field."""

    batch_size = state.shape[0]
    grid = state.reshape(batch_size, 28, 28, 2)
    return grid[..., 0].reshape(batch_size, 28 * 28), grid[..., 1].reshape(
        batch_size,
        28 * 28,
    )


def replace_shape_channel(state: Array, shapes: Array) -> Array:
    """Return ``state`` with its shape channel reset to ``shapes``."""

    pixels, _ = split_pixel_shape_channels(state)
    return stack_pixel_shape_channels(pixels, shapes)


def apply_shape_pixel_readout(
    pixels: Array,
    shapes: Array,
    *,
    sample_readout_mode: str,
) -> Array:
    """Return the pixel image used for renderer sample metrics/artifacts."""

    pixels = jnp.clip(pixels, 0.0, 1.0)
    if sample_readout_mode == "primary":
        return pixels
    if sample_readout_mode == "shape_gated":
        gate = jax.nn.sigmoid(8.0 * (jnp.clip(shapes, 0.0, 1.0) - 0.35))
        return jnp.clip(pixels * gate, 0.0, 1.0)
    raise ValueError("sample_readout_mode must be 'primary' or 'shape_gated'")


def iter_shape_pixel_batches(
    pixels: Array,
    shapes: Array,
    labels: Array,
    batch_size: int,
    key: jax.random.PRNGKey,
    *,
    shuffle: bool = True,
) -> Iterable[Tuple[Array, Array, Array]]:
    """Yield aligned pixel, shape, label batches."""

    n_samples = int(pixels.shape[0])
    indices = jnp.arange(n_samples)
    if shuffle:
        indices = jax.random.permutation(key, n_samples)
    for start in range(0, int(indices.shape[0]), batch_size):
        batch_indices = indices[start : start + batch_size]
        if batch_indices.size == 0:
            continue
        yield pixels[batch_indices], shapes[batch_indices], labels[batch_indices]


def make_shape_pixel_flow_batch(
    pixels: Array,
    shapes: Array,
    key: jax.random.PRNGKey,
    *,
    t_min: float,
    t_max: float,
) -> Tuple[Array, Array, Array, Array]:
    """Create a shape-conditioned pixel rectified-flow batch."""

    noise_key, time_key = jax.random.split(key)
    batch_size = pixels.shape[0]
    noise = jax.random.normal(noise_key, pixels.shape)
    t = jax.random.uniform(
        time_key,
        (batch_size,),
        minval=float(t_min),
        maxval=float(t_max),
    )
    noisy_pixels = (1.0 - t[:, None]) * noise + t[:, None] * pixels
    state = stack_pixel_shape_channels(noisy_pixels, shapes)
    target_velocity = stack_pixel_shape_channels(
        pixels - noise,
        jnp.zeros_like(shapes),
    )
    return state, t, target_velocity, noise


def shape_pixel_loss(
    model: ShapePixelModel,
    pixels: Array,
    shapes: Array,
    labels: Array,
    key: jax.random.PRNGKey,
    *,
    clean_loss_weight: float,
    shape_velocity_weight: float,
    closure_loss_weight: float,
    t_min: float,
    t_max: float,
) -> Tuple[Array, Dict[str, Array]]:
    """Return shape-conditioned pixel-flow loss and diagnostics."""

    state, t, target_velocity, _ = make_shape_pixel_flow_batch(
        pixels,
        shapes,
        key,
        t_min=t_min,
        t_max=t_max,
    )
    velocity = model(state, t, labels)
    pixel_velocity, shape_velocity = split_pixel_shape_channels(velocity)
    target_pixel_velocity, _ = split_pixel_shape_channels(target_velocity)
    noisy_pixels, _ = split_pixel_shape_channels(state)
    clean_prediction = noisy_pixels + (1.0 - t[:, None]) * pixel_velocity

    pixel_velocity_loss = jnp.mean((pixel_velocity - target_pixel_velocity) ** 2)
    shape_velocity_loss = jnp.mean(shape_velocity**2)
    clean_loss = jnp.mean((clean_prediction - pixels) ** 2)
    shape_loss = closure_loss(clean_prediction, pixels)
    total = (
        pixel_velocity_loss
        + float(shape_velocity_weight) * shape_velocity_loss
        + float(clean_loss_weight) * clean_loss
        + float(closure_loss_weight) * shape_loss
    )
    return total, {
        "pixel_velocity_loss": pixel_velocity_loss,
        "shape_velocity_loss": shape_velocity_loss,
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
    model: ShapePixelModel,
    opt_state: Any,
    pixels: Array,
    shapes: Array,
    labels: Array,
    sample_key: jax.random.PRNGKey,
    optimizer: optax.GradientTransformation,
    max_grad_norm: float,
    clean_loss_weight: float,
    shape_velocity_weight: float,
    closure_loss_weight: float,
    t_min: float,
    t_max: float,
):
    def loss_fn(current_model):
        return shape_pixel_loss(
            current_model,
            pixels,
            shapes,
            labels,
            sample_key,
            clean_loss_weight=clean_loss_weight,
            shape_velocity_weight=shape_velocity_weight,
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
    model: ShapePixelModel,
    pixels: Array,
    shapes: Array,
    labels: Array,
    sample_key: jax.random.PRNGKey,
    clean_loss_weight: float,
    shape_velocity_weight: float,
    closure_loss_weight: float,
    t_min: float,
    t_max: float,
):
    return shape_pixel_loss(
        model,
        pixels,
        shapes,
        labels,
        sample_key,
        clean_loss_weight=clean_loss_weight,
        shape_velocity_weight=shape_velocity_weight,
        closure_loss_weight=closure_loss_weight,
        t_min=t_min,
        t_max=t_max,
    )


def evaluate_shape_pixel_model(
    model: ShapePixelModel,
    pixels: Array,
    shapes: Array,
    labels: Array,
    *,
    batch_size: int,
    key: jax.random.PRNGKey,
    clean_loss_weight: float,
    shape_velocity_weight: float,
    closure_loss_weight: float,
    t_min: float,
    t_max: float,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate shape-conditioned pixel-flow loss over a dataset."""

    losses = []
    pixel_velocity_losses = []
    shape_velocity_losses = []
    clean_losses = []
    closure_losses = []
    for batch_index, (batch_pixels, batch_shapes, batch_labels) in enumerate(
        iter_shape_pixel_batches(
            pixels,
            shapes,
            labels,
            batch_size,
            jax.random.PRNGKey(0),
            shuffle=False,
        )
    ):
        loss, parts = _eval_step(
            model,
            batch_pixels,
            batch_shapes,
            batch_labels,
            jax.random.fold_in(key, batch_index),
            clean_loss_weight,
            shape_velocity_weight,
            closure_loss_weight,
            t_min,
            t_max,
        )
        losses.append(float(loss))
        pixel_velocity_losses.append(float(parts["pixel_velocity_loss"]))
        shape_velocity_losses.append(float(parts["shape_velocity_loss"]))
        clean_losses.append(float(parts["clean_loss"]))
        closure_losses.append(float(parts["closure_loss"]))
    if not losses:
        return float("nan"), {
            "eval_pixel_velocity_loss": float("nan"),
            "eval_shape_velocity_loss": float("nan"),
            "eval_clean_loss": float("nan"),
            "eval_closure_loss": float("nan"),
        }
    return float(np.mean(losses)), {
        "eval_pixel_velocity_loss": float(np.mean(pixel_velocity_losses)),
        "eval_shape_velocity_loss": float(np.mean(shape_velocity_losses)),
        "eval_clean_loss": float(np.mean(clean_losses)),
        "eval_closure_loss": float(np.mean(closure_losses)),
    }


def sample_shape_pixel_images(
    model: ShapePixelModel,
    shapes: Array,
    *,
    key: jax.random.PRNGKey,
    sample_steps: int,
    sample_method: str,
    labels: Optional[Array],
    batch_size: int,
    sample_readout_mode: str = "primary",
    clamp_shape: bool = True,
    clip_pixels: bool = True,
) -> Array:
    """Generate pixels from fixed signed-distance shape conditions."""

    if sample_method not in {"euler", "heun"}:
        raise ValueError("sample_method must be 'euler' or 'heun'")
    if sample_steps < 1:
        raise ValueError("sample_steps must be positive")

    samples = []
    remaining = int(shapes.shape[0])
    start = 0
    batch_index = 0
    while remaining > 0:
        current = min(int(batch_size), remaining)
        batch_shapes = shapes[start : start + current]
        current_labels = None
        if labels is not None:
            current_labels = labels[start : start + current].astype(jnp.int32)
        batch_key = jax.random.fold_in(key, batch_index)
        batch_samples = _sample_shape_pixel_batch(
            model,
            batch_shapes,
            key=batch_key,
            sample_steps=sample_steps,
            sample_method=sample_method,
            sample_readout_mode=sample_readout_mode,
            labels=current_labels,
            clamp_shape=clamp_shape,
            clip_pixels=clip_pixels,
        )
        samples.append(batch_samples)
        start += current
        remaining -= current
        batch_index += 1
    return jnp.concatenate(samples, axis=0)


def _sample_shape_pixel_batch(
    model: ShapePixelModel,
    shapes: Array,
    *,
    key: jax.random.PRNGKey,
    sample_steps: int,
    sample_method: str,
    labels: Optional[Array],
    clamp_shape: bool,
    clip_pixels: bool,
    sample_readout_mode: str,
) -> Array:
    sample_count = int(shapes.shape[0])
    pixels = jax.random.normal(key, shapes.shape)
    state = stack_pixel_shape_channels(pixels, shapes)
    step_size = 1.0 / float(sample_steps)

    def clamp(current_state):
        if clamp_shape:
            return replace_shape_channel(current_state, shapes)
        return current_state

    if sample_method == "euler":

        def scan_fn(current_state, step_index):
            t_value = (step_index.astype(current_state.dtype) + 0.5) * step_size
            t = jnp.full((sample_count,), t_value, dtype=current_state.dtype)
            velocity = model(current_state, t, labels)
            return clamp(current_state + step_size * velocity), None

    else:

        def scan_fn(current_state, step_index):
            t0_value = jnp.clip(
                step_index.astype(current_state.dtype) * step_size,
                1e-3,
                0.999,
            )
            t1_value = jnp.clip(
                (step_index.astype(current_state.dtype) + 1.0) * step_size,
                1e-3,
                0.999,
            )
            t0 = jnp.full((sample_count,), t0_value, dtype=current_state.dtype)
            t1 = jnp.full((sample_count,), t1_value, dtype=current_state.dtype)
            velocity0 = model(current_state, t0, labels)
            predictor = clamp(current_state + step_size * velocity0)
            velocity1 = model(predictor, t1, labels)
            return clamp(
                current_state + 0.5 * step_size * (velocity0 + velocity1)
            ), None

    steps = jnp.arange(int(sample_steps), dtype=jnp.float32)
    state, _ = jax.lax.scan(scan_fn, state, steps)
    pixels, _ = split_pixel_shape_channels(state)
    if clip_pixels:
        return apply_shape_pixel_readout(
            pixels,
            shapes,
            sample_readout_mode=sample_readout_mode,
        )
    return pixels


def sample_shape_pixel_from_chord(
    model: ShapePixelModel,
    pixels: Array,
    shapes: Array,
    *,
    key: jax.random.PRNGKey,
    start_t: float,
    sample_steps: int,
    sample_method: str,
    labels: Optional[Array],
    batch_size: int,
    sample_readout_mode: str = "primary",
    clamp_shape: bool = True,
    clip_pixels: bool = True,
) -> Array:
    """Complete pixels from ``(1 - t) noise + t pixel`` chord states."""

    samples, _ = _sample_shape_pixel_from_chord_with_initial(
        model,
        pixels,
        shapes,
        key=key,
        start_t=start_t,
        sample_steps=sample_steps,
        sample_method=sample_method,
        sample_readout_mode=sample_readout_mode,
        labels=labels,
        batch_size=batch_size,
        clamp_shape=clamp_shape,
        clip_pixels=clip_pixels,
    )
    return samples


def _sample_shape_pixel_from_chord_with_initial(
    model: ShapePixelModel,
    pixels: Array,
    shapes: Array,
    *,
    key: jax.random.PRNGKey,
    start_t: float,
    sample_steps: int,
    sample_method: str,
    labels: Optional[Array],
    batch_size: int,
    sample_readout_mode: str = "primary",
    clamp_shape: bool = True,
    clip_pixels: bool = True,
) -> Tuple[Array, Array]:
    """Complete chord states and return the exact initial pixel states."""

    if sample_method not in {"euler", "heun"}:
        raise ValueError("sample_method must be 'euler' or 'heun'")
    if sample_steps < 1:
        raise ValueError("sample_steps must be positive")
    start_t = float(start_t)
    if start_t < 0.0 or start_t >= 1.0:
        raise ValueError("start_t must satisfy 0 <= start_t < 1")

    samples = []
    initials = []
    remaining = int(pixels.shape[0])
    start = 0
    batch_index = 0
    while remaining > 0:
        current = min(int(batch_size), remaining)
        batch_pixels = pixels[start : start + current]
        batch_shapes = shapes[start : start + current]
        current_labels = None
        if labels is not None:
            current_labels = labels[start : start + current].astype(jnp.int32)
        batch_key = jax.random.fold_in(key, batch_index)
        noise = jax.random.normal(batch_key, batch_pixels.shape)
        initial_pixels = (1.0 - start_t) * noise + start_t * batch_pixels
        initial_state = stack_pixel_shape_channels(initial_pixels, batch_shapes)
        batch_samples = _sample_shape_pixel_from_state(
            model,
            initial_state,
            batch_shapes,
            labels=current_labels,
            start_t=start_t,
            sample_steps=sample_steps,
            sample_method=sample_method,
            sample_readout_mode=sample_readout_mode,
            clamp_shape=clamp_shape,
            clip_pixels=clip_pixels,
        )
        samples.append(batch_samples)
        initials.append(initial_pixels)
        start += current
        remaining -= current
        batch_index += 1
    return jnp.concatenate(samples, axis=0), jnp.concatenate(initials, axis=0)


def _sample_shape_pixel_from_state(
    model: ShapePixelModel,
    initial_state: Array,
    shapes: Array,
    *,
    labels: Optional[Array],
    start_t: float,
    sample_steps: int,
    sample_method: str,
    sample_readout_mode: str,
    clamp_shape: bool,
    clip_pixels: bool,
) -> Array:
    """Integrate a shape-conditioned renderer from an arbitrary state/time."""

    state = initial_state
    sample_count = int(initial_state.shape[0])
    remaining_time = 1.0 - float(start_t)
    step_size = remaining_time / float(sample_steps)

    def clamp(current_state):
        if clamp_shape:
            return replace_shape_channel(current_state, shapes)
        return current_state

    if sample_method == "euler":

        def scan_fn(current_state, step_index):
            t_value = (
                float(start_t)
                + (step_index.astype(current_state.dtype) + 0.5) * step_size
            )
            t = jnp.full((sample_count,), t_value, dtype=current_state.dtype)
            velocity = model(current_state, t, labels)
            return clamp(current_state + step_size * velocity), None

    elif sample_method == "heun":

        def scan_fn(current_state, step_index):
            t0_value = jnp.clip(
                float(start_t) + step_index.astype(current_state.dtype) * step_size,
                1e-3,
                0.999,
            )
            t1_value = jnp.clip(
                float(start_t) + (step_index.astype(current_state.dtype) + 1.0) * step_size,
                1e-3,
                0.999,
            )
            t0 = jnp.full((sample_count,), t0_value, dtype=current_state.dtype)
            t1 = jnp.full((sample_count,), t1_value, dtype=current_state.dtype)
            velocity0 = model(current_state, t0, labels)
            predictor = clamp(current_state + step_size * velocity0)
            velocity1 = model(predictor, t1, labels)
            return clamp(
                current_state + 0.5 * step_size * (velocity0 + velocity1)
            ), None

    else:
        raise ValueError("sample_method must be 'euler' or 'heun'")

    steps = jnp.arange(int(sample_steps), dtype=jnp.float32)
    state, _ = jax.lax.scan(scan_fn, state, steps)
    pixels, _ = split_pixel_shape_channels(state)
    if clip_pixels:
        return apply_shape_pixel_readout(
            pixels,
            shapes,
            sample_readout_mode=sample_readout_mode,
        )
    return pixels


def compute_shape_pixel_basin_metrics(
    model: ShapePixelModel,
    pixels: Array,
    shapes: Array,
    labels: Optional[Array],
    *,
    key: jax.random.PRNGKey,
    t_values: Tuple[float, ...],
    sample_steps: int,
    sample_method: str,
    batch_size: int,
    clamp_shape: bool,
    sample_readout_mode: str = "primary",
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Measure how far from real pixels a fixed-shape renderer can recover."""

    results: Dict[str, Dict[str, float]] = {}
    for index, start_t in enumerate(t_values):
        start_t = float(start_t)
        samples, initial = _sample_shape_pixel_from_chord_with_initial(
            model,
            pixels,
            shapes,
            key=jax.random.fold_in(key, index),
            start_t=start_t,
            sample_steps=sample_steps,
            sample_method=sample_method,
            sample_readout_mode=sample_readout_mode,
            labels=labels,
            batch_size=batch_size,
            clamp_shape=clamp_shape,
            clip_pixels=True,
        )
        metrics = compute_phase_flow_quality_metrics(pixels, samples)
        initial_mse = float(jnp.mean((initial - pixels) ** 2))
        paired_mse = float(jnp.mean((samples - pixels) ** 2))
        metrics["initial_paired_mse"] = initial_mse
        metrics["paired_mse"] = paired_mse
        metrics["paired_mse_delta"] = initial_mse - paired_mse
        metrics["paired_mse_improvement_fraction"] = (
            (initial_mse - paired_mse) / (initial_mse + 1e-8)
        )
        metrics["start_t"] = start_t
        key_name = basin_t_key(start_t)
        results[key_name] = metrics
        if artifact_dir is not None:
            _save_image_grid(
                jnp.clip(initial[: min(initial.shape[0], 64)], 0.0, 1.0),
                artifact_dir / f"basin_{key_name}_initial.png",
            )
            _save_image_grid(
                samples[: min(samples.shape[0], 64)],
                artifact_dir / f"basin_{key_name}_samples.png",
            )
    return results


def corrupt_shape_conditions(
    shapes: Array,
    *,
    key: jax.random.PRNGKey,
    start_t: float,
    noise_mode: str,
    clip: bool = True,
) -> Array:
    """Return imperfect shape scaffolds on a noise-to-shape chord."""

    start_t = float(start_t)
    if start_t < 0.0 or start_t >= 1.0:
        raise ValueError("start_t must satisfy 0 <= start_t < 1")
    noise = phase_flow_noise_endpoint(key, shapes, noise_mode)
    corrupted = (1.0 - start_t) * noise + start_t * shapes
    if clip:
        return jnp.clip(corrupted, 0.0, 1.0)
    return corrupted


def compute_shape_condition_probe_metrics(
    model: ShapePixelModel,
    pixels: Array,
    shapes: Array,
    labels: Optional[Array],
    *,
    key: jax.random.PRNGKey,
    t_values: Tuple[float, ...],
    noise_modes: Tuple[str, ...],
    sample_steps: int,
    sample_method: str,
    batch_size: int,
    clamp_shape: bool,
    sample_readout_mode: str = "primary",
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Measure renderer quality when the shape scaffold is imperfect."""

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode_index, noise_mode in enumerate(noise_modes):
        mode_results: Dict[str, Dict[str, float]] = {}
        for t_index, start_t in enumerate(t_values):
            probe_key = jax.random.fold_in(jax.random.fold_in(key, mode_index), t_index)
            corrupted_shapes = corrupt_shape_conditions(
                shapes,
                key=probe_key,
                start_t=float(start_t),
                noise_mode=noise_mode,
                clip=True,
            )
            samples = sample_shape_pixel_images(
                model,
                corrupted_shapes,
                key=jax.random.fold_in(probe_key, 10_000),
                sample_steps=sample_steps,
                sample_method=sample_method,
                sample_readout_mode=sample_readout_mode,
                labels=labels,
                batch_size=batch_size,
                clamp_shape=clamp_shape,
                clip_pixels=True,
            )
            metrics = compute_phase_flow_quality_metrics(pixels, samples)
            metrics["condition_paired_mse"] = float(
                jnp.mean((corrupted_shapes - shapes) ** 2)
            )
            metrics["paired_sample_mse"] = float(jnp.mean((samples - pixels) ** 2))
            metrics["start_t"] = float(start_t)
            key_name = basin_t_key(float(start_t))
            mode_results[key_name] = metrics
            if artifact_dir is not None:
                prefix = f"shape_condition_{noise_mode}_{key_name}"
                _save_image_grid(
                    corrupted_shapes[: min(corrupted_shapes.shape[0], 64)],
                    artifact_dir / f"{prefix}_condition.png",
                )
                _save_image_grid(
                    samples[: min(samples.shape[0], 64)],
                    artifact_dir / f"{prefix}_samples.png",
                )
        results[str(noise_mode)] = mode_results
    return results


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


def _condition_labels(
    labels: Array,
    *,
    conditional: bool,
    num_classes: int,
) -> Optional[Array]:
    """Return labels paired with the fixed shape conditions, when enabled."""

    if not conditional or num_classes <= 0:
        return None
    return labels.astype(jnp.int32)


def save_shape_pixel_artifacts(
    model: ShapePixelModel,
    eval_pixels: Array,
    eval_shapes: Array,
    eval_labels: Array,
    paths: ExperimentPaths,
    epoch: int,
    *,
    key: jax.random.PRNGKey,
    sample_count: int,
    sample_steps: int,
    sample_method: str,
    sample_readout_mode: str,
    batch_size: int,
    conditional: bool,
    num_classes: int,
    clamp_shape: bool,
) -> None:
    """Save real/shape/denoised/sample artifacts."""

    count = min(int(sample_count), int(eval_pixels.shape[0]))
    pixels = eval_pixels[:count]
    shapes = eval_shapes[:count]
    labels = eval_labels[:count]
    noise_key, sample_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, pixels.shape)
    t = jnp.full((count,), 0.5)
    noisy_pixels = 0.5 * noise + 0.5 * pixels
    state = stack_pixel_shape_channels(noisy_pixels, shapes)
    velocity, trace = model(state, t, labels, return_trace=True)
    pixel_velocity, _ = split_pixel_shape_channels(velocity)
    denoised = jnp.clip(noisy_pixels + 0.5 * pixel_velocity, 0.0, 1.0)
    sample_labels = _condition_labels(
        labels,
        conditional=conditional,
        num_classes=num_classes,
    )
    sample_shapes = shapes
    samples = sample_shape_pixel_images(
        model,
        sample_shapes,
        key=sample_key,
        sample_steps=sample_steps,
        sample_method=sample_method,
        sample_readout_mode=sample_readout_mode,
        labels=sample_labels,
        batch_size=batch_size,
        clamp_shape=clamp_shape,
        clip_pixels=True,
    )
    np.savez(
        paths.artifacts / f"mnist_shape_pixel_epoch_{epoch:03d}.npz",
        real=np.asarray(pixels),
        shape=np.asarray(shapes),
        noisy=np.asarray(noisy_pixels),
        denoised=np.asarray(denoised),
        samples=np.asarray(samples),
        labels=np.asarray(labels),
        sample_labels=None if sample_labels is None else np.asarray(sample_labels),
        **{f"trace_{name}": np.asarray(value) for name, value in trace.items()},
    )
    _save_image_grid(pixels[: min(count, 64)], paths.artifacts / f"real_epoch_{epoch:03d}.png")
    _save_image_grid(shapes[: min(count, 64)], paths.artifacts / f"shape_epoch_{epoch:03d}.png")
    _save_image_grid(
        jnp.clip(noisy_pixels[: min(count, 64)], 0.0, 1.0),
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


def _prepare_dataset(
    *,
    data_source: str,
    train_limit: Optional[int],
    eval_limit: Optional[int],
    seed: int,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    train_pixels, train_labels, eval_pixels, eval_labels = load_mnist_data(
        source=data_source,
        train_limit=train_limit,
        eval_limit=eval_limit,
        seed=seed,
    )
    train_shapes = signed_distance_targets(train_pixels)
    eval_shapes = signed_distance_targets(eval_pixels)
    return train_pixels, train_shapes, train_labels, eval_pixels, eval_shapes, eval_labels


def run_mnist_shape_pixel_experiment(
    config: MNISTShapePixelExperimentConfig,
) -> AutoencoderExperimentResult:
    """Train/evaluate a signed-distance conditioned pixel renderer."""

    logger = _logger()
    paths = prepare_experiment_paths(config.run, asdict(config))
    (
        train_pixels,
        train_shapes,
        train_labels,
        eval_pixels,
        eval_shapes,
        eval_labels,
    ) = _prepare_dataset(
        data_source=config.data_source,
        train_limit=config.train_limit,
        eval_limit=config.eval_limit,
        seed=config.run.seed,
    )

    key = jax.random.PRNGKey(config.run.seed)
    model_key = jax.random.fold_in(key, 1)
    model = build_mnist_shape_pixel_model(config, model_key)
    optimizer = optax.adamw(
        learning_rate=config.run.learning_rate,
        weight_decay=config.run.weight_decay,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    history: Dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_pixel_velocity_loss": [],
        "train_shape_velocity_loss": [],
        "train_clean_loss": [],
        "train_closure_loss": [],
        "eval_loss": [],
        "eval_pixel_velocity_loss": [],
        "eval_shape_velocity_loss": [],
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
        pixel_velocity_losses = []
        shape_velocity_losses = []
        clean_losses = []
        closure_losses = []
        grad_norms = []
        for batch_index, (batch_pixels, batch_shapes, batch_labels) in enumerate(
            iter_shape_pixel_batches(
                train_pixels,
                train_shapes,
                train_labels,
                config.run.batch_size,
                epoch_key,
                shuffle=config.run.shuffle,
            )
        ):
            model, opt_state, loss, grad_norm, parts = _train_step(
                model,
                opt_state,
                batch_pixels,
                batch_shapes,
                batch_labels,
                jax.random.fold_in(epoch_key, batch_index),
                optimizer,
                config.run.max_grad_norm,
                config.clean_loss_weight,
                config.shape_velocity_weight,
                config.closure_loss_weight,
                config.t_min,
                config.t_max,
            )
            losses.append(float(loss))
            pixel_velocity_losses.append(float(parts["pixel_velocity_loss"]))
            shape_velocity_losses.append(float(parts["shape_velocity_loss"]))
            clean_losses.append(float(parts["clean_loss"]))
            closure_losses.append(float(parts["closure_loss"]))
            grad_norms.append(float(grad_norm))

        eval_loss, eval_parts = evaluate_shape_pixel_model(
            model,
            eval_pixels,
            eval_shapes,
            eval_labels,
            batch_size=config.run.batch_size,
            key=jax.random.fold_in(key, 10_000 + epoch),
            clean_loss_weight=config.clean_loss_weight,
            shape_velocity_weight=config.shape_velocity_weight,
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
            save_shape_pixel_artifacts(
                model,
                eval_pixels,
                eval_shapes,
                eval_labels,
                paths,
                epoch,
                key=jax.random.fold_in(key, 20_000 + epoch),
                sample_count=config.eval_sample_count,
                sample_steps=config.sample_steps,
                sample_method=config.sample_method,
                sample_readout_mode=config.sample_readout_mode,
                batch_size=config.run.batch_size,
                conditional=config.conditional,
                num_classes=config.num_classes,
                clamp_shape=config.clamp_shape,
            )

        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(np.mean(losses)))
        history["train_pixel_velocity_loss"].append(float(np.mean(pixel_velocity_losses)))
        history["train_shape_velocity_loss"].append(float(np.mean(shape_velocity_losses)))
        history["train_clean_loss"].append(float(np.mean(clean_losses)))
        history["train_closure_loss"].append(float(np.mean(closure_losses)))
        history["eval_loss"].append(eval_loss)
        history["eval_pixel_velocity_loss"].append(
            eval_parts["eval_pixel_velocity_loss"]
        )
        history["eval_shape_velocity_loss"].append(
            eval_parts["eval_shape_velocity_loss"]
        )
        history["eval_clean_loss"].append(eval_parts["eval_clean_loss"])
        history["eval_closure_loss"].append(eval_parts["eval_closure_loss"])
        history["grad_norm"].append(float(np.mean(grad_norms)))
        logger.info(
            (
                "epoch=%d train_loss=%.6f eval_loss=%.6f "
                "eval_pixel_vel=%.6f eval_clean=%.6f "
                "eval_shape_vel=%.6f time=%.2fs"
            ),
            epoch,
            history["train_loss"][-1],
            eval_loss,
            eval_parts["eval_pixel_velocity_loss"],
            eval_parts["eval_clean_loss"],
            eval_parts["eval_shape_velocity_loss"],
            time.time() - epoch_start,
        )

    save_metrics_csv(history, paths.metrics / "history.csv")
    write_json(paths.metrics / "history.json", history)
    save_loss_curve(history, paths.plots / "loss_curve.png")

    count = min(config.eval_sample_count, int(eval_pixels.shape[0]))
    sample_labels = _condition_labels(
        eval_labels[:count],
        conditional=config.conditional,
        num_classes=config.num_classes,
    )
    samples = sample_shape_pixel_images(
        model,
        eval_shapes[:count],
        key=jax.random.fold_in(key, 30_001),
        sample_steps=config.sample_steps,
        sample_method=config.sample_method,
        sample_readout_mode=config.sample_readout_mode,
        labels=sample_labels,
        batch_size=config.run.batch_size,
        clamp_shape=config.clamp_shape,
        clip_pixels=True,
    )
    quality = compute_phase_flow_quality_metrics(eval_pixels[:count], samples)
    paired_sample_mse = float(jnp.mean((samples - eval_pixels[:count]) ** 2))
    basin_metrics: Dict[str, Dict[str, float]] = {}
    if config.basin_t_values:
        basin_labels = eval_labels[:count] if config.conditional else None
        basin_metrics = compute_shape_pixel_basin_metrics(
            model,
            eval_pixels[:count],
            eval_shapes[:count],
            basin_labels,
            key=jax.random.fold_in(key, 30_003),
            t_values=config.basin_t_values,
            sample_steps=config.sample_steps,
            sample_method=config.sample_method,
            sample_readout_mode=config.sample_readout_mode,
            batch_size=config.run.batch_size,
            clamp_shape=config.clamp_shape,
            artifact_dir=paths.artifacts,
        )
    shape_condition_probe: Dict[str, Dict[str, Dict[str, float]]] = {}
    if config.shape_condition_t_values and config.shape_condition_noise_modes:
        probe_labels = eval_labels[:count] if config.conditional else None
        shape_condition_probe = compute_shape_condition_probe_metrics(
            model,
            eval_pixels[:count],
            eval_shapes[:count],
            probe_labels,
            key=jax.random.fold_in(key, 30_004),
            t_values=config.shape_condition_t_values,
            noise_modes=config.shape_condition_noise_modes,
            sample_steps=config.sample_steps,
            sample_method=config.sample_method,
            sample_readout_mode=config.sample_readout_mode,
            batch_size=config.run.batch_size,
            clamp_shape=config.clamp_shape,
            artifact_dir=paths.artifacts,
        )
    trace_noise = jax.random.normal(jax.random.fold_in(key, 30_002), samples.shape)
    trace_t = jnp.full((count,), 0.5)
    trace_state = stack_pixel_shape_channels(
        0.5 * trace_noise + 0.5 * eval_pixels[:count],
        eval_shapes[:count],
    )
    trace = model.collect_trace(trace_state, trace_t, eval_labels[:count])
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
        state_mean_abs_displacement = float(np.mean(np.abs(phase_delta)))
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
        "final_eval_pixel_velocity_loss": float(
            history["eval_pixel_velocity_loss"][-1]
        ),
        "final_eval_shape_velocity_loss": float(
            history["eval_shape_velocity_loss"][-1]
        ),
        "final_eval_clean_loss": float(history["eval_clean_loss"][-1]),
        "final_eval_closure_loss": float(history["eval_closure_loss"][-1]),
        "shape_pixel": {
            "model_family": config.model_family,
            "field_channels": int(model.field_channels),
            "value_channels": int(model.value_channels),
            "steps": int(model.steps),
            "train_dynamics": bool(model.train_dynamics),
            "conditional": bool(config.conditional),
            "position_features": bool(config.position_features),
            "clean_loss_weight": float(config.clean_loss_weight),
            "shape_velocity_weight": float(config.shape_velocity_weight),
            "closure_loss_weight": float(config.closure_loss_weight),
            "sample_steps": int(config.sample_steps),
            "sample_method": config.sample_method,
            "sample_readout_mode": config.sample_readout_mode,
            "basin_t_values": [float(value) for value in config.basin_t_values],
            "basin": basin_metrics,
            "shape_condition_t_values": [
                float(value) for value in config.shape_condition_t_values
            ],
            "shape_condition_noise_modes": list(config.shape_condition_noise_modes),
            "shape_condition_probe": shape_condition_probe,
            "clamp_shape": bool(config.clamp_shape),
            "paired_sample_mse": paired_sample_mse,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reference/mnist_shape_pixel"),
    )
    parser.add_argument("--name", default="mnist_shape_pixel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--artifact-every", type=int, default=20)
    parser.add_argument(
        "--model-family",
        choices=[
            "phase_flow",
            "frozen_phase_flow",
            "phase_flow_no_dynamics",
            "coarse_phase_flow",
            "recurrent_conv_flow",
        ],
        default="coarse_phase_flow",
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
    parser.add_argument("--position-features", action="store_true")
    parser.add_argument("--no-position-features", dest="position_features", action="store_false")
    parser.set_defaults(position_features=False)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--unconditional", dest="conditional", action="store_false")
    parser.set_defaults(conditional=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--clean-loss-weight", type=float, default=0.25)
    parser.add_argument("--shape-velocity-weight", type=float, default=0.1)
    parser.add_argument("--closure-loss-weight", type=float, default=0.0)
    parser.add_argument("--t-min", type=float, default=1e-3)
    parser.add_argument("--t-max", type=float, default=0.999)
    parser.add_argument("--eval-sample-count", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--sample-method", choices=["euler", "heun"], default="euler")
    parser.add_argument(
        "--sample-readout-mode",
        choices=["primary", "shape_gated"],
        default="primary",
        help=(
            "Readout used for sample metrics/artifacts. 'primary' uses the raw "
            "pixel channel; 'shape_gated' multiplies it by a smooth gate from "
            "the clamped signed-distance scaffold."
        ),
    )
    parser.add_argument(
        "--basin-t-values",
        type=parse_basin_t_values,
        default=(),
        help=(
            "Comma-separated chord start times for renderer basin diagnostics, "
            "for example '0.1,0.25,0.5,0.75,0.9'."
        ),
    )
    parser.add_argument(
        "--shape-condition-t-values",
        type=parse_basin_t_values,
        default=(),
        help=(
            "Comma-separated scaffold corruption levels for renderer robustness "
            "diagnostics. A value near 1 is close to the oracle shape; a value "
            "near 0 is close to the selected noise endpoint."
        ),
    )
    parser.add_argument(
        "--shape-condition-noise-modes",
        type=parse_noise_modes,
        default=(),
        help=(
            "Comma-separated noise endpoint modes for scaffold robustness "
            "diagnostics, for example 'uniform,salt_pepper,zeros'."
        ),
    )
    parser.add_argument("--clamp-shape", dest="clamp_shape", action="store_true")
    parser.add_argument("--no-clamp-shape", dest="clamp_shape", action="store_false")
    parser.set_defaults(clamp_shape=True)
    parser.add_argument("--data-source", choices=["idx", "synthetic", "tfds"], default="idx")
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--eval-limit", type=int, default=1_000)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_config = AutoencoderExperimentConfig(
        name=args.name,
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
    config = MNISTShapePixelExperimentConfig(
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
        shape_velocity_weight=args.shape_velocity_weight,
        closure_loss_weight=args.closure_loss_weight,
        t_min=args.t_min,
        t_max=args.t_max,
        eval_sample_count=args.eval_sample_count,
        sample_steps=args.sample_steps,
        sample_method=args.sample_method,
        sample_readout_mode=args.sample_readout_mode,
        basin_t_values=args.basin_t_values,
        shape_condition_t_values=args.shape_condition_t_values,
        shape_condition_noise_modes=args.shape_condition_noise_modes,
        clamp_shape=args.clamp_shape,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )
    result = run_mnist_shape_pixel_experiment(config)
    print(json.dumps(result.metrics, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()


__all__ = [
    "MNISTShapePixelExperimentConfig",
    "apply_shape_pixel_readout",
    "build_mnist_shape_pixel_model",
    "compute_shape_condition_probe_metrics",
    "make_shape_pixel_flow_batch",
    "run_mnist_shape_pixel_experiment",
    "compute_shape_pixel_basin_metrics",
    "corrupt_shape_conditions",
    "sample_shape_pixel_from_chord",
    "sample_shape_pixel_images",
    "shape_pixel_loss",
    "split_pixel_shape_channels",
    "stack_pixel_shape_channels",
]
