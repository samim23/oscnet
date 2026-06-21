"""Reference MNIST benchmarks for oscillatory autoencoders."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import struct
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from oscnet.core.oscillators import (
    AdaptiveNonlinearHarmonicOscillator,
    LearnableNonlinearHarmonicOscillator,
    NonlinearHarmonicOscillator,
)
from oscnet.experiments.harness import (
    AutoencoderExperimentConfig,
    AutoencoderExperimentResult,
    ExperimentPaths,
    PredictionTransform,
    collect_sequence_state_trace,
    run_eval_only,
    train_autoencoder,
    write_json,
)
from oscnet.models import (
    AmplitudeVelocityAutoencoder,
    ConvLSTMPatchDenoiser,
    FeedForwardPatchAutoencoder,
    RecurrentConvPatchDenoiser,
    RecurrentConvPriorRefinementPatchDenoiser,
    WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser,
    WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser,
    WinfreeCoarseRatePhaseConditionalPatchDenoiser,
    WinfreeConditionalPatchDenoiser,
    WinfreeGlobalRatePhaseConditionalPatchDenoiser,
    WinfreePatchAutoencoder,
    WinfreePriorRefinementPatchDenoiser,
    WinfreeRatePhaseConditionalPatchDenoiser,
)

Array = jnp.ndarray

MNIST_IDX_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "eval_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "eval_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


@dataclass(frozen=True)
class MNISTAutoencoderExperimentConfig:
    """Task-specific controls for the MNIST reference experiment."""

    run: AutoencoderExperimentConfig
    hidden_dim: int = 64
    latent_dim: int = 64
    patch_shape: Tuple[int, int] = (4, 4)
    model_family: str = "amplitude_velocity"
    decoder_mode: str = "repeat"
    latent_conditioning_strength: float = 1.0
    feedforward_latent_output_skip: str = "sequence"
    feedforward_latent_output_skip_strength: float = 1.0
    feedforward_output_activation: str = "identity"
    recurrent_conv_steps: int = 8
    recurrent_conv_kernel_size: int = 3
    recurrent_conv_residual_strength: float = 0.5
    recurrent_conv_refinement_strength: float = 0.5
    recurrent_conv_output_activation: str = "identity"
    conv_lstm_steps: int = 8
    conv_lstm_kernel_size: int = 3
    conv_lstm_forget_bias: float = 1.0
    conv_lstm_output_activation: str = "identity"
    output_activation: str = "identity"
    oscillator: str = "learnable"
    winfree_steps: int = 8
    winfree_gamma: float = 0.1
    winfree_global_gamma: float = 0.05
    winfree_coarse_grid_size: int = 2
    winfree_coupling_strength: float = 1.0
    winfree_coupling_decay_length: Optional[float] = None
    winfree_coupling_mode: str = "matrix"
    winfree_coupling_kernel_size: int = 3
    winfree_adaptive_coupling_strength: float = 0.1
    winfree_omega_scale: float = 1.0
    winfree_latent_readout: str = "none"
    winfree_latent_readout_strength: float = 1.0
    winfree_latent_output_skip: str = "none"
    winfree_latent_output_skip_strength: float = 1.0
    winfree_field_activation: str = "relu"
    winfree_si_func: str = "trig"
    winfree_si_hidden_ratio: int = 2
    winfree_group_size: int = 1
    winfree_phase_init: str = "learned"
    winfree_phase_init_scale: float = 1.0
    winfree_rate_kernel_size: int = 3
    winfree_rate_update_rate: float = 0.5
    winfree_rate_gate_strength: float = 1.0
    winfree_visibility_gate: str = "none"
    winfree_visibility_drive_floor: float = 0.0
    winfree_missing_transport_strength: float = 1.0
    winfree_global_gate_strength: float = 0.5
    winfree_global_phase_control: str = "none"
    winfree_global_content_strength: float = 0.5
    winfree_global_content_control: str = "none"
    winfree_coarse_readout_strength: float = 0.5
    winfree_refinement_strength: float = 0.25
    winfree_output_activation: str = "identity"
    corruption_mode: str = "none"
    corruption_fraction: float = 0.5
    corruption_seed: Optional[int] = None
    corruption_noise_std: float = 0.35
    corruption_mask_value: float = 0.0
    corruption_input_mode: str = "image"
    corruption_protocol: str = "full_reconstruction"
    corruption_visible_loss_weight: float = 1.0
    corruption_changed_loss_weight: float = 1.0
    corruption_change_atol: float = 1e-6
    data_source: str = "idx"
    train_limit: Optional[int] = 10_000
    eval_limit: Optional[int] = 1_000
    checkpoint: Optional[Path] = None
    eval_winfree_steps: Optional[int] = None


def _logger() -> logging.Logger:
    logger = logging.getLogger("oscnet.experiments.mnist")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _synthetic_mnist_like(n_samples: int, seed: int) -> Tuple[Array, Array]:
    """Create deterministic MNIST-shaped images for fast tests and smoke runs."""

    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, 28, 28), dtype=np.float32)
    labels = rng.integers(0, 10, size=n_samples, dtype=np.int32)

    for i, label in enumerate(labels):
        thickness = 2 + int(label % 3)
        offset = 3 + int((label * 2) % 17)
        images[i, 4:24, offset : offset + thickness] = 0.8
        if label % 2 == 0:
            images[i, offset : offset + thickness, 4:24] = 0.6
        else:
            diag = np.arange(6, 22)
            images[i, diag, np.clip(diag + label - 5, 0, 27)] = 0.9

    images += rng.normal(0.0, 0.03, size=images.shape).astype(np.float32)
    images = np.clip(images, 0.0, 1.0).reshape(n_samples, 28 * 28)
    return jnp.asarray(images), jnp.asarray(labels)


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, path)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_images, rows * cols).astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_labels).astype(np.int32)


def _load_mnist_idx(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = {}
    for name, url in MNIST_IDX_URLS.items():
        path = cache_dir / Path(url).name
        _download_file(url, path)
        paths[name] = path

    return (
        _read_idx_images(paths["train_images"]),
        _read_idx_labels(paths["train_labels"]),
        _read_idx_images(paths["eval_images"]),
        _read_idx_labels(paths["eval_labels"]),
    )


def load_mnist_data(
    subset_size: Optional[int] = None,
    *,
    source: str = "tfds",
    train_limit: Optional[int] = None,
    eval_limit: Optional[int] = None,
    seed: int = 42,
):
    """Load MNIST as flattened float32 images.

    The ``subset_size`` argument is kept for older example compatibility.
    Prefer ``train_limit`` and ``eval_limit`` in new experiment code.
    """

    if subset_size is not None:
        train_limit = subset_size
        eval_limit = subset_size

    if source == "synthetic":
        train_n = train_limit or 256
        eval_n = eval_limit or max(32, train_n // 4)
        train_images, train_labels = _synthetic_mnist_like(train_n, seed)
        eval_images, eval_labels = _synthetic_mnist_like(eval_n, seed + 1)
        return train_images, train_labels, eval_images, eval_labels

    if source not in {"tfds", "idx"}:
        raise ValueError("source must be 'tfds', 'idx', or 'synthetic'")

    if source == "tfds":
        try:
            import tensorflow_datasets as tfds

            train_ds = tfds.as_numpy(tfds.load("mnist", split="train", batch_size=-1))
            test_ds = tfds.as_numpy(tfds.load("mnist", split="test", batch_size=-1))

            train_images = train_ds["image"].astype(np.float32) / 255.0
            eval_images = test_ds["image"].astype(np.float32) / 255.0
            train_labels = train_ds["label"].astype(np.int32)
            eval_labels = test_ds["label"].astype(np.int32)

            train_images = train_images.reshape(-1, 28 * 28)
            eval_images = eval_images.reshape(-1, 28 * 28)
        except Exception as exc:
            _logger().warning(
                "TFDS MNIST loading failed (%s); falling back to direct IDX download.",
                exc,
            )
            train_images, train_labels, eval_images, eval_labels = _load_mnist_idx(
                Path.home() / ".cache" / "oscnet" / "mnist"
            )
    else:
        train_images, train_labels, eval_images, eval_labels = _load_mnist_idx(
            Path.home() / ".cache" / "oscnet" / "mnist"
        )

    if train_limit is not None:
        train_images = train_images[:train_limit]
        train_labels = train_labels[:train_limit]
    if eval_limit is not None:
        eval_images = eval_images[:eval_limit]
        eval_labels = eval_labels[:eval_limit]

    return (
        jnp.asarray(train_images),
        jnp.asarray(train_labels),
        jnp.asarray(eval_images),
        jnp.asarray(eval_labels),
    )


def corrupt_mnist_images(
    images: Array,
    *,
    mode: str,
    patch_shape: Tuple[int, int] = (4, 4),
    fraction: float = 0.5,
    seed: int = 0,
    noise_std: float = 0.35,
    mask_value: float = 0.0,
) -> Array:
    """Create deterministic corrupted MNIST inputs for reconstruction tasks."""

    corrupted, _ = corrupt_mnist_images_with_visibility(
        images,
        mode=mode,
        patch_shape=patch_shape,
        fraction=fraction,
        seed=seed,
        noise_std=noise_std,
        mask_value=mask_value,
    )
    return corrupted


def corrupt_mnist_images_with_visibility(
    images: Array,
    *,
    mode: str,
    patch_shape: Tuple[int, int] = (4, 4),
    fraction: float = 0.5,
    seed: int = 0,
    noise_std: float = 0.35,
    mask_value: float = 0.0,
) -> Tuple[Array, Array]:
    """Create corrupted inputs plus a 1=visible, 0=missing visibility mask."""

    if mode == "none":
        visibility = jnp.ones_like(images, dtype=jnp.float32)
        return images, visibility
    if mode not in {"patch_mask", "block_occlusion", "gaussian_noise"}:
        raise ValueError(
            "corruption_mode must be 'none', 'patch_mask', "
            "'block_occlusion', or 'gaussian_noise'"
        )
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("corruption_fraction must be between 0 and 1")
    if noise_std < 0.0:
        raise ValueError("corruption_noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    clean = np.asarray(images, dtype=np.float32).reshape(-1, 28, 28)
    corrupted = clean.copy()
    visibility = np.ones_like(corrupted, dtype=np.float32)

    if mode == "gaussian_noise":
        noise = rng.normal(0.0, float(noise_std), size=corrupted.shape).astype(
            np.float32
        )
        corrupted = np.clip(corrupted + noise, 0.0, 1.0)
        return (
            jnp.asarray(corrupted.reshape(images.shape)),
            jnp.asarray(visibility.reshape(images.shape)),
        )

    if mode == "block_occlusion":
        side = max(1, int(round(28.0 * np.sqrt(float(fraction)))))
        side = min(side, 28)
        for idx in range(corrupted.shape[0]):
            top = int(rng.integers(0, 28 - side + 1))
            left = int(rng.integers(0, 28 - side + 1))
            corrupted[idx, top : top + side, left : left + side] = mask_value
            visibility[idx, top : top + side, left : left + side] = 0.0
        return (
            jnp.asarray(corrupted.reshape(images.shape)),
            jnp.asarray(visibility.reshape(images.shape)),
        )

    patch_height, patch_width = patch_shape
    if 28 % patch_height != 0 or 28 % patch_width != 0:
        raise ValueError("patch_mask corruption requires patch_shape to divide 28x28")
    grid_h = 28 // patch_height
    grid_w = 28 // patch_width
    num_patches = grid_h * grid_w
    num_mask = int(round(num_patches * float(fraction)))

    for idx in range(corrupted.shape[0]):
        if num_mask == 0:
            continue
        masked = rng.choice(num_patches, size=num_mask, replace=False)
        for patch_idx in masked:
            row = int(patch_idx // grid_w)
            col = int(patch_idx % grid_w)
            top = row * patch_height
            left = col * patch_width
            corrupted[
                idx,
                top : top + patch_height,
                left : left + patch_width,
            ] = mask_value
            visibility[
                idx,
                top : top + patch_height,
                left : left + patch_width,
            ] = 0.0

    return (
        jnp.asarray(corrupted.reshape(images.shape)),
        jnp.asarray(visibility.reshape(images.shape)),
    )


def append_visibility_mask_channel(images: Array, visibility: Array) -> Array:
    """Concatenate image content and visibility as channel-first flat features."""

    if images.shape != visibility.shape:
        raise ValueError("images and visibility must have matching shapes")
    return jnp.concatenate([images, visibility], axis=-1)


def primary_image_channel(inputs: Array, *, image_pixels: int = 28 * 28) -> np.ndarray:
    """Extract the first image channel from one- or multi-channel flat inputs."""

    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 2:
        values = values.reshape(values.shape[0], -1)
    if values.shape[1] == image_pixels:
        return values
    if values.shape[1] % image_pixels != 0:
        raise ValueError("input feature dimension must be a multiple of image pixels")
    return values.reshape(values.shape[0], -1, image_pixels)[:, 0, :]


def visibility_channel_from_inputs(
    inputs: Array,
    *,
    image_pixels: int = 28 * 28,
) -> Optional[np.ndarray]:
    """Return the second flat channel as visibility when present."""

    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 2:
        values = values.reshape(values.shape[0], -1)
    if values.shape[1] == image_pixels:
        return None
    if values.shape[1] % image_pixels != 0:
        raise ValueError("input feature dimension must be a multiple of image pixels")
    channels = values.reshape(values.shape[0], -1, image_pixels)
    if channels.shape[1] < 2:
        return None
    return channels[:, 1, :]


def clamp_predictions_to_visible_inputs(
    inputs: Array,
    predictions: Array,
    *,
    image_pixels: int = 28 * 28,
) -> Array:
    """Clamp predicted visible pixels to observed input pixels.

    This is for boundary-value inpainting protocols where observed pixels are
    fixed evidence and loss is computed on missing pixels.
    """

    if inputs.ndim != 2:
        inputs = inputs.reshape(inputs.shape[0], -1)
    if predictions.ndim != 2:
        predictions = predictions.reshape(predictions.shape[0], -1)
    if inputs.shape[1] % image_pixels != 0 or inputs.shape[1] < 2 * image_pixels:
        raise ValueError("boundary clamping requires image_plus_mask inputs")
    if predictions.shape[1] != image_pixels:
        raise ValueError("predictions must contain one flat image channel")

    channels = inputs.reshape(inputs.shape[0], -1, image_pixels)
    observed = channels[:, 0, :]
    visibility = jnp.clip(channels[:, 1, :], 0.0, 1.0)
    return visibility * observed + (1.0 - visibility) * predictions


class BoundaryClampedPredictionView:
    """Artifact/metric view that applies boundary clamping without wrapping weights."""

    def __init__(self, model: eqx.Module):
        self.model = model

    def __call__(self, inputs: Array) -> Array:
        return clamp_predictions_to_visible_inputs(inputs, self.model(inputs))

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    def collect_trace(self, inputs: Array) -> Dict[str, Array]:
        trace = (
            self.model.collect_trace(inputs)
            if hasattr(self.model, "collect_trace")
            else {}
        )
        raw_reconstruction = self.model(inputs)
        clamped_reconstruction = clamp_predictions_to_visible_inputs(
            inputs,
            raw_reconstruction,
        )
        return {
            **trace,
            "raw_reconstruction": raw_reconstruction,
            "clamped_reconstruction": clamped_reconstruction,
        }


def infer_corruption_changed_mask(
    inputs: Array,
    targets: Array,
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    """Infer pixels changed by corruption from input/target image pairs."""

    inputs_np = np.asarray(inputs, dtype=np.float32)
    targets_np = np.asarray(targets, dtype=np.float32)
    if inputs_np.shape != targets_np.shape:
        raise ValueError("inputs and targets must have matching shapes")
    return np.abs(inputs_np - targets_np) > float(atol)


def compute_corruption_loss_weights(
    inputs: Array,
    targets: Array,
    *,
    visible_weight: float = 1.0,
    changed_weight: float = 1.0,
    atol: float = 1e-6,
) -> Array:
    """Create per-pixel loss weights from corruption-changed pixels."""

    if visible_weight < 0.0 or changed_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if visible_weight == 0.0 and changed_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")

    changed = infer_corruption_changed_mask(inputs, targets, atol=atol)
    weights = np.where(changed, float(changed_weight), float(visible_weight))
    return jnp.asarray(weights, dtype=jnp.float32)


def compute_visibility_loss_weights(
    visibility: Array,
    *,
    visible_weight: float = 1.0,
    changed_weight: float = 1.0,
) -> Array:
    """Create per-pixel loss weights from a 1=visible, 0=missing mask."""

    if visible_weight < 0.0 or changed_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if visible_weight == 0.0 and changed_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")

    visible = np.asarray(visibility, dtype=np.float32) >= 0.5
    weights = np.where(visible, float(visible_weight), float(changed_weight))
    return jnp.asarray(weights, dtype=jnp.float32)


def _mnist_oscillator_spec(kind: str, hidden_dim: int):
    base_omega = 2.0 * jnp.pi / 28.0
    if kind == "learnable":
        return (
            LearnableNonlinearHarmonicOscillator,
            {
                "alpha": 0.04,
                "omega_init": base_omega,
                "gamma_init": 0.01,
                "omega_bounds": (float(jnp.pi / 56.0), float(4.0 * jnp.pi / 7.0)),
                "gamma_bounds": (0.001, 0.2),
                "dt": 1.0,
            },
        )
    if kind == "adaptive":
        return (
            AdaptiveNonlinearHarmonicOscillator,
            {
                "alpha": 0.04,
                "base_omega": float(base_omega),
                "base_gamma": 0.01,
                "omega_multiplier_bounds": (0.25, 4.0),
                "gamma_multiplier_bounds": (0.1, 20.0),
                "dt": 1.0,
            },
        )
    if kind == "nonlinear":
        return (
            NonlinearHarmonicOscillator,
            {
                "alpha": 0.04,
                "omega": tuple(np.full(hidden_dim, float(base_omega))),
                "gamma": tuple(np.full(hidden_dim, 0.01)),
                "dt": 1.0,
            },
        )
    raise ValueError("oscillator must be 'learnable', 'adaptive', or 'nonlinear'")


def _mnist_input_channels(config: MNISTAutoencoderExperimentConfig) -> int:
    if config.corruption_input_mode == "image":
        return 1
    if config.corruption_input_mode == "image_plus_mask":
        return 2
    raise ValueError("corruption_input_mode must be 'image' or 'image_plus_mask'")


def _mnist_input_patch_dim(config: MNISTAutoencoderExperimentConfig) -> int:
    patch_pixels = config.patch_shape[0] * config.patch_shape[1]
    return patch_pixels * _mnist_input_channels(config)


def _select_mnist_model_inputs(
    corrupted_images: Array,
    visibility: Array,
    config: MNISTAutoencoderExperimentConfig,
) -> Array:
    if config.corruption_input_mode == "image":
        return corrupted_images
    if config.corruption_input_mode == "image_plus_mask":
        return append_visibility_mask_channel(corrupted_images, visibility)
    raise ValueError("corruption_input_mode must be 'image' or 'image_plus_mask'")


def build_mnist_model(
    config: MNISTAutoencoderExperimentConfig,
    key: jax.random.PRNGKey,
) -> eqx.Module:
    input_dim = _mnist_input_patch_dim(config)
    if config.model_family == "feedforward_patch":
        return FeedForwardPatchAutoencoder(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            patch_shape=config.patch_shape,
            latent_output_skip=config.feedforward_latent_output_skip,
            latent_output_skip_strength=(
                config.feedforward_latent_output_skip_strength
            ),
            output_activation=config.feedforward_output_activation,
            key=key,
        )

    if config.model_family == "recurrent_conv":
        return RecurrentConvPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            patch_shape=config.patch_shape,
            steps=config.recurrent_conv_steps,
            kernel_size=config.recurrent_conv_kernel_size,
            residual_strength=config.recurrent_conv_residual_strength,
            output_activation=config.recurrent_conv_output_activation,
            key=key,
        )

    if config.model_family == "recurrent_conv_prior_refinement":
        return RecurrentConvPriorRefinementPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            patch_shape=config.patch_shape,
            feedforward_latent_output_skip=config.feedforward_latent_output_skip,
            feedforward_latent_output_skip_strength=(
                config.feedforward_latent_output_skip_strength
            ),
            steps=config.recurrent_conv_steps,
            kernel_size=config.recurrent_conv_kernel_size,
            recurrent_residual_strength=config.recurrent_conv_residual_strength,
            refinement_strength=config.recurrent_conv_refinement_strength,
            output_activation=config.recurrent_conv_output_activation,
            key=key,
        )

    if config.model_family == "conv_lstm":
        return ConvLSTMPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            patch_shape=config.patch_shape,
            steps=config.conv_lstm_steps,
            kernel_size=config.conv_lstm_kernel_size,
            forget_bias=config.conv_lstm_forget_bias,
            output_activation=config.conv_lstm_output_activation,
            key=key,
        )

    if config.model_family == "winfree_conditional":
        return WinfreeConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            patch_shape=config.patch_shape,
            group_size=config.winfree_group_size,
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_decay_length=config.winfree_coupling_decay_length,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            adaptive_coupling_strength=config.winfree_adaptive_coupling_strength,
            input_conditioning_strength=config.latent_conditioning_strength,
            omega_scale=config.winfree_omega_scale,
            phase_init=config.winfree_phase_init,
            phase_init_scale=config.winfree_phase_init_scale,
            field_activation=config.winfree_field_activation,
            si_func=config.winfree_si_func,
            si_hidden_ratio=config.winfree_si_hidden_ratio,
            output_activation=config.winfree_output_activation,
            key=key,
        )

    if config.model_family == "winfree_rate_phase":
        return WinfreeRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            patch_shape=config.patch_shape,
            group_size=config.winfree_group_size,
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_decay_length=config.winfree_coupling_decay_length,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            adaptive_coupling_strength=config.winfree_adaptive_coupling_strength,
            input_conditioning_strength=config.latent_conditioning_strength,
            omega_scale=config.winfree_omega_scale,
            phase_init=config.winfree_phase_init,
            phase_init_scale=config.winfree_phase_init_scale,
            field_activation=config.winfree_field_activation,
            si_func=config.winfree_si_func,
            si_hidden_ratio=config.winfree_si_hidden_ratio,
            rate_kernel_size=config.winfree_rate_kernel_size,
            rate_update_rate=config.winfree_rate_update_rate,
            rate_gate_strength=config.winfree_rate_gate_strength,
            visibility_gate=config.winfree_visibility_gate,
            visibility_drive_floor=config.winfree_visibility_drive_floor,
            missing_transport_strength=config.winfree_missing_transport_strength,
            output_activation=config.winfree_output_activation,
            key=key,
        )

    if config.model_family == "winfree_global_rate_phase":
        return WinfreeGlobalRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            patch_shape=config.patch_shape,
            group_size=config.winfree_group_size,
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            global_gamma=config.winfree_global_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_decay_length=config.winfree_coupling_decay_length,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            adaptive_coupling_strength=config.winfree_adaptive_coupling_strength,
            input_conditioning_strength=config.latent_conditioning_strength,
            omega_scale=config.winfree_omega_scale,
            phase_init=config.winfree_phase_init,
            phase_init_scale=config.winfree_phase_init_scale,
            field_activation=config.winfree_field_activation,
            si_func=config.winfree_si_func,
            si_hidden_ratio=config.winfree_si_hidden_ratio,
            rate_kernel_size=config.winfree_rate_kernel_size,
            rate_update_rate=config.winfree_rate_update_rate,
            rate_gate_strength=config.winfree_rate_gate_strength,
            visibility_gate=config.winfree_visibility_gate,
            visibility_drive_floor=config.winfree_visibility_drive_floor,
            missing_transport_strength=config.winfree_missing_transport_strength,
            global_gate_strength=config.winfree_global_gate_strength,
            output_activation=config.winfree_output_activation,
            key=key,
        )

    if config.model_family == "winfree_prior_refinement":
        return WinfreePriorRefinementPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            patch_shape=config.patch_shape,
            feedforward_latent_output_skip=config.feedforward_latent_output_skip,
            feedforward_latent_output_skip_strength=(
                config.feedforward_latent_output_skip_strength
            ),
            group_size=config.winfree_group_size,
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            global_gamma=config.winfree_global_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_decay_length=config.winfree_coupling_decay_length,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            adaptive_coupling_strength=config.winfree_adaptive_coupling_strength,
            input_conditioning_strength=config.latent_conditioning_strength,
            omega_scale=config.winfree_omega_scale,
            phase_init=config.winfree_phase_init,
            phase_init_scale=config.winfree_phase_init_scale,
            field_activation=config.winfree_field_activation,
            si_func=config.winfree_si_func,
            si_hidden_ratio=config.winfree_si_hidden_ratio,
            rate_kernel_size=config.winfree_rate_kernel_size,
            rate_update_rate=config.winfree_rate_update_rate,
            rate_gate_strength=config.winfree_rate_gate_strength,
            visibility_gate=config.winfree_visibility_gate,
            visibility_drive_floor=config.winfree_visibility_drive_floor,
            missing_transport_strength=config.winfree_missing_transport_strength,
            global_gate_strength=config.winfree_global_gate_strength,
            refinement_strength=config.winfree_refinement_strength,
            output_activation=config.winfree_output_activation,
            key=key,
        )

    if config.model_family == "winfree_coarse_global_rate_phase":
        return WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            patch_shape=config.patch_shape,
            coarse_grid_shape=(
                config.winfree_coarse_grid_size,
                config.winfree_coarse_grid_size,
            ),
            group_size=config.winfree_group_size,
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            global_gamma=config.winfree_global_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_decay_length=config.winfree_coupling_decay_length,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            adaptive_coupling_strength=config.winfree_adaptive_coupling_strength,
            input_conditioning_strength=config.latent_conditioning_strength,
            omega_scale=config.winfree_omega_scale,
            phase_init=config.winfree_phase_init,
            phase_init_scale=config.winfree_phase_init_scale,
            field_activation=config.winfree_field_activation,
            si_func=config.winfree_si_func,
            si_hidden_ratio=config.winfree_si_hidden_ratio,
            rate_kernel_size=config.winfree_rate_kernel_size,
            rate_update_rate=config.winfree_rate_update_rate,
            rate_gate_strength=config.winfree_rate_gate_strength,
            global_gate_strength=config.winfree_global_gate_strength,
            global_phase_control=config.winfree_global_phase_control,
            output_activation=config.winfree_output_activation,
            key=key,
        )

    if config.model_family == "winfree_coarse_rate_phase":
        return WinfreeCoarseRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            patch_shape=config.patch_shape,
            coarse_grid_shape=(
                config.winfree_coarse_grid_size,
                config.winfree_coarse_grid_size,
            ),
            group_size=config.winfree_group_size,
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            global_gamma=config.winfree_global_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_decay_length=config.winfree_coupling_decay_length,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            adaptive_coupling_strength=config.winfree_adaptive_coupling_strength,
            input_conditioning_strength=config.latent_conditioning_strength,
            omega_scale=config.winfree_omega_scale,
            phase_init=config.winfree_phase_init,
            phase_init_scale=config.winfree_phase_init_scale,
            field_activation=config.winfree_field_activation,
            si_func=config.winfree_si_func,
            si_hidden_ratio=config.winfree_si_hidden_ratio,
            rate_kernel_size=config.winfree_rate_kernel_size,
            rate_update_rate=config.winfree_rate_update_rate,
            rate_gate_strength=config.winfree_rate_gate_strength,
            global_gate_strength=config.winfree_global_gate_strength,
            global_phase_control=config.winfree_global_phase_control,
            global_content_strength=config.winfree_global_content_strength,
            global_content_control=config.winfree_global_content_control,
            output_activation=config.winfree_output_activation,
            key=key,
        )

    if config.model_family == "winfree_coarse_predictive_rate_phase":
        return WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            patch_shape=config.patch_shape,
            coarse_grid_shape=(
                config.winfree_coarse_grid_size,
                config.winfree_coarse_grid_size,
            ),
            group_size=config.winfree_group_size,
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            global_gamma=config.winfree_global_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_decay_length=config.winfree_coupling_decay_length,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            adaptive_coupling_strength=config.winfree_adaptive_coupling_strength,
            input_conditioning_strength=config.latent_conditioning_strength,
            omega_scale=config.winfree_omega_scale,
            phase_init=config.winfree_phase_init,
            phase_init_scale=config.winfree_phase_init_scale,
            field_activation=config.winfree_field_activation,
            si_func=config.winfree_si_func,
            si_hidden_ratio=config.winfree_si_hidden_ratio,
            rate_kernel_size=config.winfree_rate_kernel_size,
            rate_update_rate=config.winfree_rate_update_rate,
            rate_gate_strength=config.winfree_rate_gate_strength,
            global_gate_strength=config.winfree_global_gate_strength,
            global_phase_control=config.winfree_global_phase_control,
            global_content_control=config.winfree_global_content_control,
            coarse_readout_strength=config.winfree_coarse_readout_strength,
            output_activation=config.winfree_output_activation,
            key=key,
        )

    if config.model_family == "winfree_field":
        if _mnist_input_channels(config) != 1:
            raise ValueError("winfree_field does not support image_plus_mask inputs")
        return WinfreePatchAutoencoder(
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            patch_shape=config.patch_shape,
            group_size=config.winfree_group_size,
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_decay_length=config.winfree_coupling_decay_length,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            adaptive_coupling_strength=config.winfree_adaptive_coupling_strength,
            latent_conditioning_strength=config.latent_conditioning_strength,
            latent_readout=config.winfree_latent_readout,
            latent_readout_strength=config.winfree_latent_readout_strength,
            latent_output_skip=config.winfree_latent_output_skip,
            latent_output_skip_strength=config.winfree_latent_output_skip_strength,
            omega_scale=config.winfree_omega_scale,
            field_activation=config.winfree_field_activation,
            si_func=config.winfree_si_func,
            si_hidden_ratio=config.winfree_si_hidden_ratio,
            output_activation=config.winfree_output_activation,
            key=key,
        )

    if config.model_family != "amplitude_velocity":
        raise ValueError(
            "model_family must be 'amplitude_velocity', "
            "'feedforward_patch', 'recurrent_conv', "
            "'recurrent_conv_prior_refinement', 'conv_lstm', "
            "'winfree_conditional', 'winfree_rate_phase', "
            "'winfree_global_rate_phase', "
            "'winfree_prior_refinement', "
            "'winfree_coarse_global_rate_phase', "
            "'winfree_coarse_rate_phase', "
            "'winfree_coarse_predictive_rate_phase', or 'winfree_field'"
        )

    oscillator_class, oscillator_params = _mnist_oscillator_spec(
        config.oscillator,
        config.hidden_dim,
    )
    if _mnist_input_channels(config) != 1:
        raise ValueError(
            "amplitude_velocity does not support image_plus_mask inputs"
        )
    return AmplitudeVelocityAutoencoder(
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        patch_shape=config.patch_shape,
        decoder_mode=config.decoder_mode,
        latent_conditioning_strength=config.latent_conditioning_strength,
        output_activation=config.output_activation,
        oscillator_class=oscillator_class,
        oscillator_params=oscillator_params,
        initial_amplitude=0.1,
        key=key,
    )


def _checkpoint_hyperparams(config: MNISTAutoencoderExperimentConfig) -> Dict[str, object]:
    hyperparams = {
        "model_family": config.model_family,
        "hidden_dim": config.hidden_dim,
        "latent_dim": config.latent_dim,
        "patch_shape": list(config.patch_shape),
        "input_dim": _mnist_input_patch_dim(config),
        "corruption_input_mode": config.corruption_input_mode,
        "latent_conditioning_strength": config.latent_conditioning_strength,
    }
    if config.model_family == "winfree_field":
        hyperparams.update(
            {
                "winfree_steps": config.winfree_steps,
                "winfree_gamma": config.winfree_gamma,
                "winfree_coupling_strength": config.winfree_coupling_strength,
                "winfree_coupling_decay_length": config.winfree_coupling_decay_length,
                "winfree_coupling_mode": config.winfree_coupling_mode,
                "winfree_coupling_kernel_size": config.winfree_coupling_kernel_size,
                "winfree_adaptive_coupling_strength": (
                    config.winfree_adaptive_coupling_strength
                ),
                "winfree_omega_scale": config.winfree_omega_scale,
                "winfree_latent_readout": config.winfree_latent_readout,
                "winfree_latent_readout_strength": (
                    config.winfree_latent_readout_strength
                ),
                "winfree_latent_output_skip": config.winfree_latent_output_skip,
                "winfree_latent_output_skip_strength": (
                    config.winfree_latent_output_skip_strength
                ),
                "winfree_field_activation": config.winfree_field_activation,
                "winfree_si_func": config.winfree_si_func,
                "winfree_si_hidden_ratio": config.winfree_si_hidden_ratio,
                "winfree_group_size": config.winfree_group_size,
                "winfree_output_activation": config.winfree_output_activation,
            }
        )
        return hyperparams

    if config.model_family in {
        "winfree_rate_phase",
        "winfree_global_rate_phase",
        "winfree_prior_refinement",
        "winfree_coarse_global_rate_phase",
        "winfree_coarse_rate_phase",
        "winfree_coarse_predictive_rate_phase",
    }:
        hyperparams.update(
            {
                "winfree_steps": config.winfree_steps,
                "winfree_gamma": config.winfree_gamma,
                "winfree_coupling_strength": config.winfree_coupling_strength,
                "winfree_coupling_decay_length": config.winfree_coupling_decay_length,
                "winfree_coupling_mode": config.winfree_coupling_mode,
                "winfree_coupling_kernel_size": config.winfree_coupling_kernel_size,
                "winfree_adaptive_coupling_strength": (
                    config.winfree_adaptive_coupling_strength
                ),
                "winfree_omega_scale": config.winfree_omega_scale,
                "winfree_field_activation": config.winfree_field_activation,
                "winfree_si_func": config.winfree_si_func,
                "winfree_si_hidden_ratio": config.winfree_si_hidden_ratio,
                "winfree_group_size": config.winfree_group_size,
                "winfree_phase_init": config.winfree_phase_init,
                "winfree_phase_init_scale": config.winfree_phase_init_scale,
                "winfree_rate_kernel_size": config.winfree_rate_kernel_size,
                "winfree_rate_update_rate": config.winfree_rate_update_rate,
                "winfree_rate_gate_strength": config.winfree_rate_gate_strength,
                "winfree_visibility_gate": config.winfree_visibility_gate,
                "winfree_visibility_drive_floor": (
                    config.winfree_visibility_drive_floor
                ),
                "winfree_missing_transport_strength": (
                    config.winfree_missing_transport_strength
                ),
                "winfree_output_activation": config.winfree_output_activation,
            }
        )
        if config.model_family in {
            "winfree_global_rate_phase",
            "winfree_prior_refinement",
            "winfree_coarse_global_rate_phase",
            "winfree_coarse_rate_phase",
            "winfree_coarse_predictive_rate_phase",
        }:
            hyperparams.update(
                {
                    "winfree_global_gamma": config.winfree_global_gamma,
                    "winfree_global_gate_strength": (
                        config.winfree_global_gate_strength
                    ),
                }
            )
        if config.model_family == "winfree_prior_refinement":
            hyperparams.update(
                {
                    "feedforward_latent_output_skip": (
                        config.feedforward_latent_output_skip
                    ),
                    "feedforward_latent_output_skip_strength": (
                        config.feedforward_latent_output_skip_strength
                    ),
                    "winfree_refinement_strength": (
                        config.winfree_refinement_strength
                    ),
                }
            )
        if config.model_family in {
            "winfree_coarse_global_rate_phase",
            "winfree_coarse_predictive_rate_phase",
        }:
            hyperparams.update(
                {
                    "winfree_coarse_grid_size": config.winfree_coarse_grid_size,
                    "winfree_global_phase_control": (
                        config.winfree_global_phase_control
                    ),
                }
            )
        if config.model_family == "winfree_coarse_predictive_rate_phase":
            hyperparams.update(
                {
                    "winfree_global_content_control": (
                        config.winfree_global_content_control
                    ),
                    "winfree_coarse_readout_strength": (
                        config.winfree_coarse_readout_strength
                    ),
                }
            )
        if config.model_family == "winfree_coarse_rate_phase":
            hyperparams.update(
                {
                    "winfree_coarse_grid_size": config.winfree_coarse_grid_size,
                    "winfree_global_phase_control": (
                        config.winfree_global_phase_control
                    ),
                    "winfree_global_content_strength": (
                        config.winfree_global_content_strength
                    ),
                    "winfree_global_content_control": (
                        config.winfree_global_content_control
                    ),
                }
            )
        return hyperparams

    if config.model_family == "winfree_conditional":
        hyperparams.update(
            {
                "winfree_steps": config.winfree_steps,
                "winfree_gamma": config.winfree_gamma,
                "winfree_coupling_strength": config.winfree_coupling_strength,
                "winfree_coupling_decay_length": config.winfree_coupling_decay_length,
                "winfree_coupling_mode": config.winfree_coupling_mode,
                "winfree_coupling_kernel_size": config.winfree_coupling_kernel_size,
                "winfree_adaptive_coupling_strength": (
                    config.winfree_adaptive_coupling_strength
                ),
                "winfree_omega_scale": config.winfree_omega_scale,
                "winfree_field_activation": config.winfree_field_activation,
                "winfree_si_func": config.winfree_si_func,
                "winfree_si_hidden_ratio": config.winfree_si_hidden_ratio,
                "winfree_group_size": config.winfree_group_size,
                "winfree_phase_init": config.winfree_phase_init,
                "winfree_phase_init_scale": config.winfree_phase_init_scale,
                "winfree_output_activation": config.winfree_output_activation,
            }
        )
        return hyperparams

    if config.model_family == "feedforward_patch":
        hyperparams.update(
            {
                "feedforward_latent_output_skip": (
                    config.feedforward_latent_output_skip
                ),
                "feedforward_latent_output_skip_strength": (
                    config.feedforward_latent_output_skip_strength
                ),
                "feedforward_output_activation": (
                    config.feedforward_output_activation
                ),
            }
        )
        return hyperparams

    if config.model_family == "recurrent_conv":
        hyperparams.update(
            {
                "recurrent_conv_steps": config.recurrent_conv_steps,
                "recurrent_conv_kernel_size": config.recurrent_conv_kernel_size,
                "recurrent_conv_residual_strength": (
                    config.recurrent_conv_residual_strength
                ),
                "recurrent_conv_output_activation": (
                    config.recurrent_conv_output_activation
                ),
            }
        )
        return hyperparams

    if config.model_family == "recurrent_conv_prior_refinement":
        hyperparams.update(
            {
                "feedforward_latent_output_skip": (
                    config.feedforward_latent_output_skip
                ),
                "feedforward_latent_output_skip_strength": (
                    config.feedforward_latent_output_skip_strength
                ),
                "recurrent_conv_steps": config.recurrent_conv_steps,
                "recurrent_conv_kernel_size": config.recurrent_conv_kernel_size,
                "recurrent_conv_residual_strength": (
                    config.recurrent_conv_residual_strength
                ),
                "recurrent_conv_refinement_strength": (
                    config.recurrent_conv_refinement_strength
                ),
                "recurrent_conv_output_activation": (
                    config.recurrent_conv_output_activation
                ),
            }
        )
        return hyperparams

    if config.model_family == "conv_lstm":
        hyperparams.update(
            {
                "conv_lstm_steps": config.conv_lstm_steps,
                "conv_lstm_kernel_size": config.conv_lstm_kernel_size,
                "conv_lstm_forget_bias": config.conv_lstm_forget_bias,
                "conv_lstm_output_activation": config.conv_lstm_output_activation,
            }
        )
        return hyperparams

    _, oscillator_params = _mnist_oscillator_spec(config.oscillator, config.hidden_dim)
    hyperparams.update(
        {
            "decoder_mode": config.decoder_mode,
            "oscillator": config.oscillator,
            "oscillator_params": oscillator_params,
            "output_activation": config.output_activation,
        }
    )
    return hyperparams


def _build_mnist_model_from_hyperparams(**hyperparams) -> eqx.Module:
    model_family = hyperparams.get("model_family", "amplitude_velocity")
    hidden_dim = int(hyperparams["hidden_dim"])
    patch_shape = tuple(hyperparams.get("patch_shape", (4, 4)))
    input_dim = int(hyperparams.get("input_dim", patch_shape[0] * patch_shape[1]))

    if model_family == "winfree_field":
        return WinfreePatchAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=int(hyperparams["latent_dim"]),
            patch_shape=patch_shape,
            group_size=int(hyperparams.get("winfree_group_size", 1)),
            steps=int(hyperparams.get("winfree_steps", 8)),
            gamma=float(hyperparams.get("winfree_gamma", 0.1)),
            coupling_strength=float(
                hyperparams.get("winfree_coupling_strength", 1.0)
            ),
            coupling_decay_length=(
                None
                if hyperparams.get("winfree_coupling_decay_length") is None
                else float(hyperparams["winfree_coupling_decay_length"])
            ),
            coupling_mode=hyperparams.get("winfree_coupling_mode", "matrix"),
            coupling_kernel_size=int(
                hyperparams.get("winfree_coupling_kernel_size", 3)
            ),
            adaptive_coupling_strength=float(
                hyperparams.get("winfree_adaptive_coupling_strength", 0.1)
            ),
            latent_conditioning_strength=float(
                hyperparams.get("latent_conditioning_strength", 1.0)
            ),
            latent_readout=hyperparams.get("winfree_latent_readout", "none"),
            latent_readout_strength=float(
                hyperparams.get("winfree_latent_readout_strength", 1.0)
            ),
            latent_output_skip=hyperparams.get("winfree_latent_output_skip", "none"),
            latent_output_skip_strength=float(
                hyperparams.get("winfree_latent_output_skip_strength", 1.0)
            ),
            omega_scale=float(hyperparams.get("winfree_omega_scale", 1.0)),
            field_activation=hyperparams.get("winfree_field_activation", "relu"),
            si_func=hyperparams.get("winfree_si_func", "trig"),
            si_hidden_ratio=int(hyperparams.get("winfree_si_hidden_ratio", 2)),
            output_activation=hyperparams.get("winfree_output_activation", "identity"),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "winfree_conditional":
        return WinfreeConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_shape=patch_shape,
            group_size=int(hyperparams.get("winfree_group_size", 1)),
            steps=int(hyperparams.get("winfree_steps", 8)),
            gamma=float(hyperparams.get("winfree_gamma", 0.1)),
            coupling_strength=float(
                hyperparams.get("winfree_coupling_strength", 1.0)
            ),
            coupling_decay_length=(
                None
                if hyperparams.get("winfree_coupling_decay_length") is None
                else float(hyperparams["winfree_coupling_decay_length"])
            ),
            coupling_mode=hyperparams.get("winfree_coupling_mode", "conv"),
            coupling_kernel_size=int(
                hyperparams.get("winfree_coupling_kernel_size", 3)
            ),
            adaptive_coupling_strength=float(
                hyperparams.get("winfree_adaptive_coupling_strength", 0.1)
            ),
            input_conditioning_strength=float(
                hyperparams.get("latent_conditioning_strength", 1.0)
            ),
            omega_scale=float(hyperparams.get("winfree_omega_scale", 1.0)),
            phase_init=hyperparams.get("winfree_phase_init", "learned"),
            phase_init_scale=float(
                hyperparams.get("winfree_phase_init_scale", 1.0)
            ),
            field_activation=hyperparams.get("winfree_field_activation", "relu"),
            si_func=hyperparams.get("winfree_si_func", "mlp"),
            si_hidden_ratio=int(hyperparams.get("winfree_si_hidden_ratio", 2)),
            output_activation=hyperparams.get("winfree_output_activation", "identity"),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "winfree_rate_phase":
        return WinfreeRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_shape=patch_shape,
            group_size=int(hyperparams.get("winfree_group_size", 1)),
            steps=int(hyperparams.get("winfree_steps", 8)),
            gamma=float(hyperparams.get("winfree_gamma", 0.1)),
            coupling_strength=float(
                hyperparams.get("winfree_coupling_strength", 1.0)
            ),
            coupling_decay_length=(
                None
                if hyperparams.get("winfree_coupling_decay_length") is None
                else float(hyperparams["winfree_coupling_decay_length"])
            ),
            coupling_mode=hyperparams.get("winfree_coupling_mode", "conv"),
            coupling_kernel_size=int(
                hyperparams.get("winfree_coupling_kernel_size", 3)
            ),
            adaptive_coupling_strength=float(
                hyperparams.get("winfree_adaptive_coupling_strength", 0.1)
            ),
            input_conditioning_strength=float(
                hyperparams.get("latent_conditioning_strength", 1.0)
            ),
            omega_scale=float(hyperparams.get("winfree_omega_scale", 1.0)),
            phase_init=hyperparams.get("winfree_phase_init", "learned"),
            phase_init_scale=float(
                hyperparams.get("winfree_phase_init_scale", 1.0)
            ),
            field_activation=hyperparams.get("winfree_field_activation", "relu"),
            si_func=hyperparams.get("winfree_si_func", "mlp"),
            si_hidden_ratio=int(hyperparams.get("winfree_si_hidden_ratio", 2)),
            rate_kernel_size=int(hyperparams.get("winfree_rate_kernel_size", 3)),
            rate_update_rate=float(
                hyperparams.get("winfree_rate_update_rate", 0.5)
            ),
            rate_gate_strength=float(
                hyperparams.get("winfree_rate_gate_strength", 1.0)
            ),
            visibility_gate=hyperparams.get("winfree_visibility_gate", "none"),
            visibility_drive_floor=float(
                hyperparams.get("winfree_visibility_drive_floor", 0.0)
            ),
            missing_transport_strength=float(
                hyperparams.get("winfree_missing_transport_strength", 1.0)
            ),
            output_activation=hyperparams.get("winfree_output_activation", "identity"),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "winfree_global_rate_phase":
        return WinfreeGlobalRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_shape=patch_shape,
            group_size=int(hyperparams.get("winfree_group_size", 1)),
            steps=int(hyperparams.get("winfree_steps", 8)),
            gamma=float(hyperparams.get("winfree_gamma", 0.1)),
            global_gamma=float(hyperparams.get("winfree_global_gamma", 0.05)),
            coupling_strength=float(
                hyperparams.get("winfree_coupling_strength", 1.0)
            ),
            coupling_decay_length=(
                None
                if hyperparams.get("winfree_coupling_decay_length") is None
                else float(hyperparams["winfree_coupling_decay_length"])
            ),
            coupling_mode=hyperparams.get("winfree_coupling_mode", "conv"),
            coupling_kernel_size=int(
                hyperparams.get("winfree_coupling_kernel_size", 3)
            ),
            adaptive_coupling_strength=float(
                hyperparams.get("winfree_adaptive_coupling_strength", 0.1)
            ),
            input_conditioning_strength=float(
                hyperparams.get("latent_conditioning_strength", 1.0)
            ),
            omega_scale=float(hyperparams.get("winfree_omega_scale", 1.0)),
            phase_init=hyperparams.get("winfree_phase_init", "learned"),
            phase_init_scale=float(
                hyperparams.get("winfree_phase_init_scale", 1.0)
            ),
            field_activation=hyperparams.get("winfree_field_activation", "relu"),
            si_func=hyperparams.get("winfree_si_func", "mlp"),
            si_hidden_ratio=int(hyperparams.get("winfree_si_hidden_ratio", 2)),
            rate_kernel_size=int(hyperparams.get("winfree_rate_kernel_size", 3)),
            rate_update_rate=float(
                hyperparams.get("winfree_rate_update_rate", 0.5)
            ),
            rate_gate_strength=float(
                hyperparams.get("winfree_rate_gate_strength", 1.0)
            ),
            visibility_gate=hyperparams.get("winfree_visibility_gate", "none"),
            visibility_drive_floor=float(
                hyperparams.get("winfree_visibility_drive_floor", 0.0)
            ),
            missing_transport_strength=float(
                hyperparams.get("winfree_missing_transport_strength", 1.0)
            ),
            global_gate_strength=float(
                hyperparams.get("winfree_global_gate_strength", 0.5)
            ),
            output_activation=hyperparams.get("winfree_output_activation", "identity"),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "winfree_prior_refinement":
        return WinfreePriorRefinementPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=int(hyperparams["latent_dim"]),
            patch_shape=patch_shape,
            feedforward_latent_output_skip=hyperparams.get(
                "feedforward_latent_output_skip",
                "sequence",
            ),
            feedforward_latent_output_skip_strength=float(
                hyperparams.get("feedforward_latent_output_skip_strength", 1.0)
            ),
            group_size=int(hyperparams.get("winfree_group_size", 1)),
            steps=int(hyperparams.get("winfree_steps", 8)),
            gamma=float(hyperparams.get("winfree_gamma", 0.1)),
            global_gamma=float(hyperparams.get("winfree_global_gamma", 0.05)),
            coupling_strength=float(
                hyperparams.get("winfree_coupling_strength", 1.0)
            ),
            coupling_decay_length=(
                None
                if hyperparams.get("winfree_coupling_decay_length") is None
                else float(hyperparams["winfree_coupling_decay_length"])
            ),
            coupling_mode=hyperparams.get("winfree_coupling_mode", "conv"),
            coupling_kernel_size=int(
                hyperparams.get("winfree_coupling_kernel_size", 3)
            ),
            adaptive_coupling_strength=float(
                hyperparams.get("winfree_adaptive_coupling_strength", 0.1)
            ),
            input_conditioning_strength=float(
                hyperparams.get("latent_conditioning_strength", 1.0)
            ),
            omega_scale=float(hyperparams.get("winfree_omega_scale", 1.0)),
            phase_init=hyperparams.get("winfree_phase_init", "learned"),
            phase_init_scale=float(
                hyperparams.get("winfree_phase_init_scale", 1.0)
            ),
            field_activation=hyperparams.get("winfree_field_activation", "relu"),
            si_func=hyperparams.get("winfree_si_func", "mlp"),
            si_hidden_ratio=int(hyperparams.get("winfree_si_hidden_ratio", 2)),
            rate_kernel_size=int(hyperparams.get("winfree_rate_kernel_size", 3)),
            rate_update_rate=float(
                hyperparams.get("winfree_rate_update_rate", 0.5)
            ),
            rate_gate_strength=float(
                hyperparams.get("winfree_rate_gate_strength", 1.0)
            ),
            visibility_gate=hyperparams.get("winfree_visibility_gate", "none"),
            visibility_drive_floor=float(
                hyperparams.get("winfree_visibility_drive_floor", 0.0)
            ),
            missing_transport_strength=float(
                hyperparams.get("winfree_missing_transport_strength", 1.0)
            ),
            global_gate_strength=float(
                hyperparams.get("winfree_global_gate_strength", 0.5)
            ),
            refinement_strength=float(
                hyperparams.get("winfree_refinement_strength", 0.25)
            ),
            output_activation=hyperparams.get("winfree_output_activation", "identity"),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "winfree_coarse_global_rate_phase":
        coarse_size = int(hyperparams.get("winfree_coarse_grid_size", 2))
        return WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_shape=patch_shape,
            coarse_grid_shape=(coarse_size, coarse_size),
            group_size=int(hyperparams.get("winfree_group_size", 1)),
            steps=int(hyperparams.get("winfree_steps", 8)),
            gamma=float(hyperparams.get("winfree_gamma", 0.1)),
            global_gamma=float(hyperparams.get("winfree_global_gamma", 0.05)),
            coupling_strength=float(
                hyperparams.get("winfree_coupling_strength", 1.0)
            ),
            coupling_decay_length=(
                None
                if hyperparams.get("winfree_coupling_decay_length") is None
                else float(hyperparams["winfree_coupling_decay_length"])
            ),
            coupling_mode=hyperparams.get("winfree_coupling_mode", "conv"),
            coupling_kernel_size=int(
                hyperparams.get("winfree_coupling_kernel_size", 3)
            ),
            adaptive_coupling_strength=float(
                hyperparams.get("winfree_adaptive_coupling_strength", 0.1)
            ),
            input_conditioning_strength=float(
                hyperparams.get("latent_conditioning_strength", 1.0)
            ),
            omega_scale=float(hyperparams.get("winfree_omega_scale", 1.0)),
            phase_init=hyperparams.get("winfree_phase_init", "learned"),
            phase_init_scale=float(
                hyperparams.get("winfree_phase_init_scale", 1.0)
            ),
            field_activation=hyperparams.get("winfree_field_activation", "relu"),
            si_func=hyperparams.get("winfree_si_func", "mlp"),
            si_hidden_ratio=int(hyperparams.get("winfree_si_hidden_ratio", 2)),
            rate_kernel_size=int(hyperparams.get("winfree_rate_kernel_size", 3)),
            rate_update_rate=float(
                hyperparams.get("winfree_rate_update_rate", 0.5)
            ),
            rate_gate_strength=float(
                hyperparams.get("winfree_rate_gate_strength", 1.0)
            ),
            global_gate_strength=float(
                hyperparams.get("winfree_global_gate_strength", 0.5)
            ),
            global_phase_control=hyperparams.get(
                "winfree_global_phase_control",
                "none",
            ),
            output_activation=hyperparams.get("winfree_output_activation", "identity"),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "winfree_coarse_rate_phase":
        coarse_size = int(hyperparams.get("winfree_coarse_grid_size", 2))
        return WinfreeCoarseRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_shape=patch_shape,
            coarse_grid_shape=(coarse_size, coarse_size),
            group_size=int(hyperparams.get("winfree_group_size", 1)),
            steps=int(hyperparams.get("winfree_steps", 8)),
            gamma=float(hyperparams.get("winfree_gamma", 0.1)),
            global_gamma=float(hyperparams.get("winfree_global_gamma", 0.05)),
            coupling_strength=float(
                hyperparams.get("winfree_coupling_strength", 1.0)
            ),
            coupling_decay_length=(
                None
                if hyperparams.get("winfree_coupling_decay_length") is None
                else float(hyperparams["winfree_coupling_decay_length"])
            ),
            coupling_mode=hyperparams.get("winfree_coupling_mode", "conv"),
            coupling_kernel_size=int(
                hyperparams.get("winfree_coupling_kernel_size", 3)
            ),
            adaptive_coupling_strength=float(
                hyperparams.get("winfree_adaptive_coupling_strength", 0.1)
            ),
            input_conditioning_strength=float(
                hyperparams.get("latent_conditioning_strength", 1.0)
            ),
            omega_scale=float(hyperparams.get("winfree_omega_scale", 1.0)),
            phase_init=hyperparams.get("winfree_phase_init", "learned"),
            phase_init_scale=float(
                hyperparams.get("winfree_phase_init_scale", 1.0)
            ),
            field_activation=hyperparams.get("winfree_field_activation", "relu"),
            si_func=hyperparams.get("winfree_si_func", "mlp"),
            si_hidden_ratio=int(hyperparams.get("winfree_si_hidden_ratio", 2)),
            rate_kernel_size=int(hyperparams.get("winfree_rate_kernel_size", 3)),
            rate_update_rate=float(
                hyperparams.get("winfree_rate_update_rate", 0.5)
            ),
            rate_gate_strength=float(
                hyperparams.get("winfree_rate_gate_strength", 1.0)
            ),
            global_gate_strength=float(
                hyperparams.get("winfree_global_gate_strength", 0.5)
            ),
            global_phase_control=hyperparams.get(
                "winfree_global_phase_control",
                "none",
            ),
            global_content_strength=float(
                hyperparams.get("winfree_global_content_strength", 0.5)
            ),
            global_content_control=hyperparams.get(
                "winfree_global_content_control",
                "none",
            ),
            output_activation=hyperparams.get("winfree_output_activation", "identity"),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "winfree_coarse_predictive_rate_phase":
        coarse_size = int(hyperparams.get("winfree_coarse_grid_size", 2))
        return WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_shape=patch_shape,
            coarse_grid_shape=(coarse_size, coarse_size),
            group_size=int(hyperparams.get("winfree_group_size", 1)),
            steps=int(hyperparams.get("winfree_steps", 8)),
            gamma=float(hyperparams.get("winfree_gamma", 0.1)),
            global_gamma=float(hyperparams.get("winfree_global_gamma", 0.05)),
            coupling_strength=float(
                hyperparams.get("winfree_coupling_strength", 1.0)
            ),
            coupling_decay_length=(
                None
                if hyperparams.get("winfree_coupling_decay_length") is None
                else float(hyperparams["winfree_coupling_decay_length"])
            ),
            coupling_mode=hyperparams.get("winfree_coupling_mode", "conv"),
            coupling_kernel_size=int(
                hyperparams.get("winfree_coupling_kernel_size", 3)
            ),
            adaptive_coupling_strength=float(
                hyperparams.get("winfree_adaptive_coupling_strength", 0.1)
            ),
            input_conditioning_strength=float(
                hyperparams.get("latent_conditioning_strength", 1.0)
            ),
            omega_scale=float(hyperparams.get("winfree_omega_scale", 1.0)),
            phase_init=hyperparams.get("winfree_phase_init", "learned"),
            phase_init_scale=float(
                hyperparams.get("winfree_phase_init_scale", 1.0)
            ),
            field_activation=hyperparams.get("winfree_field_activation", "relu"),
            si_func=hyperparams.get("winfree_si_func", "mlp"),
            si_hidden_ratio=int(hyperparams.get("winfree_si_hidden_ratio", 2)),
            rate_kernel_size=int(hyperparams.get("winfree_rate_kernel_size", 3)),
            rate_update_rate=float(
                hyperparams.get("winfree_rate_update_rate", 0.5)
            ),
            rate_gate_strength=float(
                hyperparams.get("winfree_rate_gate_strength", 1.0)
            ),
            global_gate_strength=float(
                hyperparams.get("winfree_global_gate_strength", 0.5)
            ),
            global_phase_control=hyperparams.get(
                "winfree_global_phase_control",
                "none",
            ),
            global_content_control=hyperparams.get(
                "winfree_global_content_control",
                "none",
            ),
            coarse_readout_strength=float(
                hyperparams.get("winfree_coarse_readout_strength", 0.5)
            ),
            output_activation=hyperparams.get("winfree_output_activation", "identity"),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "feedforward_patch":
        return FeedForwardPatchAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=int(hyperparams["latent_dim"]),
            patch_shape=patch_shape,
            latent_output_skip=hyperparams.get(
                "feedforward_latent_output_skip",
                "sequence",
            ),
            latent_output_skip_strength=float(
                hyperparams.get("feedforward_latent_output_skip_strength", 1.0)
            ),
            output_activation=hyperparams.get(
                "feedforward_output_activation",
                "identity",
            ),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "recurrent_conv":
        return RecurrentConvPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_shape=patch_shape,
            steps=int(hyperparams.get("recurrent_conv_steps", 8)),
            kernel_size=int(hyperparams.get("recurrent_conv_kernel_size", 3)),
            residual_strength=float(
                hyperparams.get("recurrent_conv_residual_strength", 0.5)
            ),
            output_activation=hyperparams.get(
                "recurrent_conv_output_activation",
                "identity",
            ),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "recurrent_conv_prior_refinement":
        return RecurrentConvPriorRefinementPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=int(hyperparams["latent_dim"]),
            patch_shape=patch_shape,
            feedforward_latent_output_skip=hyperparams.get(
                "feedforward_latent_output_skip",
                "sequence",
            ),
            feedforward_latent_output_skip_strength=float(
                hyperparams.get("feedforward_latent_output_skip_strength", 1.0)
            ),
            steps=int(hyperparams.get("recurrent_conv_steps", 8)),
            kernel_size=int(hyperparams.get("recurrent_conv_kernel_size", 3)),
            recurrent_residual_strength=float(
                hyperparams.get("recurrent_conv_residual_strength", 0.5)
            ),
            refinement_strength=float(
                hyperparams.get("recurrent_conv_refinement_strength", 0.5)
            ),
            output_activation=hyperparams.get(
                "recurrent_conv_output_activation",
                "identity",
            ),
            key=jax.random.PRNGKey(0),
        )

    if model_family == "conv_lstm":
        return ConvLSTMPatchDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_shape=patch_shape,
            steps=int(hyperparams.get("conv_lstm_steps", 8)),
            kernel_size=int(hyperparams.get("conv_lstm_kernel_size", 3)),
            forget_bias=float(hyperparams.get("conv_lstm_forget_bias", 1.0)),
            output_activation=hyperparams.get(
                "conv_lstm_output_activation",
                "identity",
            ),
            key=jax.random.PRNGKey(0),
        )

    if model_family != "amplitude_velocity":
        raise ValueError(
            "model_family must be 'amplitude_velocity', "
            "'feedforward_patch', 'recurrent_conv', "
            "'recurrent_conv_prior_refinement', 'conv_lstm', "
            "'winfree_conditional', 'winfree_rate_phase', "
            "'winfree_global_rate_phase', "
            "'winfree_prior_refinement', "
            "'winfree_coarse_global_rate_phase', "
            "'winfree_coarse_rate_phase', "
            "'winfree_coarse_predictive_rate_phase', or 'winfree_field'"
        )

    oscillator = hyperparams.get("oscillator", "learnable")
    oscillator_class, _ = _mnist_oscillator_spec(oscillator, hidden_dim)
    oscillator_params = dict(hyperparams.get("oscillator_params", {}))
    for key in (
        "omega_bounds",
        "gamma_bounds",
        "omega_multiplier_bounds",
        "gamma_multiplier_bounds",
        "omega",
        "gamma",
    ):
        if key in oscillator_params and isinstance(oscillator_params[key], list):
            oscillator_params[key] = tuple(oscillator_params[key])

    return AmplitudeVelocityAutoencoder(
        hidden_dim=hidden_dim,
        latent_dim=int(hyperparams["latent_dim"]),
        patch_shape=patch_shape,
        decoder_mode=hyperparams.get("decoder_mode", "repeat"),
        latent_conditioning_strength=float(
            hyperparams.get("latent_conditioning_strength", 1.0)
        ),
        output_activation=hyperparams.get("output_activation", "identity"),
        oscillator_class=oscillator_class,
        oscillator_params=oscillator_params,
        initial_amplitude=0.1,
        key=jax.random.PRNGKey(0),
    )


def _checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_metadata.json")


def _load_checkpoint_bundle(
    checkpoint_path: Path,
    hyperparam_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[eqx.Module, Dict[str, Any], Dict[str, Any]]:
    with open(checkpoint_path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        if hyperparam_overrides:
            hyperparams.update(hyperparam_overrides)
        model = _build_mnist_model_from_hyperparams(**hyperparams)
        model = eqx.tree_deserialise_leaves(f, model)

    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    return model, hyperparams, metadata


def _load_checkpoint(checkpoint_path: Path) -> eqx.Module:
    model, _, _ = _load_checkpoint_bundle(checkpoint_path)
    return model


def _checkpoint_metric_loss(metadata: Dict[str, Any]) -> Optional[float]:
    metrics = metadata.get("metrics", {})
    loss = metrics.get("eval_loss")
    if loss is None:
        loss = metrics.get("train_loss")
    return None if loss is None else float(loss)


def prepare_reconstructions(model, samples, targets: Optional[Array] = None):
    """Prepare 28x28 original/reconstruction arrays for visualization."""

    originals_source = samples if targets is None else targets
    originals = originals_source.reshape(originals_source.shape[0], 28, 28)
    reconstructions = model(samples).reshape(samples.shape[0], 28, 28)
    return originals, reconstructions


def compute_mnist_baselines(train_images: Array, eval_images: Array) -> Dict[str, float]:
    """Compute trivial reconstruction baselines for MNIST comparisons."""

    train = np.asarray(train_images)
    eval_set = np.asarray(eval_images)
    scalar_mean = np.full_like(eval_set, train.mean())
    pixel_mean = np.broadcast_to(train.mean(axis=0), eval_set.shape)
    return {
        "zero_mse": float(np.mean(eval_set**2)),
        "scalar_mean_mse": float(np.mean((scalar_mean - eval_set) ** 2)),
        "pixel_mean_mse": float(np.mean((pixel_mean - eval_set) ** 2)),
    }


def _predict_images_in_batches(
    model: eqx.Module,
    images: Array,
    batch_size: int,
    prediction_transform: Optional[PredictionTransform] = None,
) -> np.ndarray:
    predictions = []
    n_images = int(images.shape[0])
    for start in range(0, n_images, batch_size):
        batch = images[start : start + batch_size]
        prediction = model(batch)
        if prediction_transform is not None:
            prediction = prediction_transform(batch, prediction)
        predictions.append(np.asarray(prediction))
    return np.concatenate(predictions, axis=0)


def compute_mnist_quality_metrics(
    model: eqx.Module,
    eval_images: Array,
    *,
    target_images: Optional[Array] = None,
    batch_size: int,
    threshold: float = 0.25,
    change_atol: float = 1e-6,
    prediction_transform: Optional[PredictionTransform] = None,
) -> Dict[str, float]:
    """Compute image-quality diagnostics for MNIST reconstructions."""

    target_source = eval_images if target_images is None else target_images
    originals = np.asarray(target_source).reshape(target_source.shape[0], -1)
    inputs = primary_image_channel(eval_images).reshape(originals.shape)
    visibility = visibility_channel_from_inputs(eval_images)
    reconstructions = _predict_images_in_batches(
        model,
        eval_images,
        batch_size,
        prediction_transform=prediction_transform,
    )
    reconstructions = reconstructions.reshape(originals.shape)
    clipped = np.clip(reconstructions, 0.0, 1.0)

    original_centered = originals - originals.mean(axis=1, keepdims=True)
    recon_centered = clipped - clipped.mean(axis=1, keepdims=True)
    numerator = np.sum(original_centered * recon_centered, axis=1)
    denominator = (
        np.linalg.norm(original_centered, axis=1)
        * np.linalg.norm(recon_centered, axis=1)
        + 1e-8
    )
    sample_correlations = numerator / denominator

    original_mask = originals > threshold
    recon_mask = clipped > threshold
    intersection = np.logical_and(original_mask, recon_mask).sum(axis=1)
    union = np.logical_or(original_mask, recon_mask).sum(axis=1)
    predicted = recon_mask.sum(axis=1)
    actual = original_mask.sum(axis=1)
    iou = intersection / (union + 1e-8)
    precision = intersection / (predicted + 1e-8)
    recall = intersection / (actual + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    original_diversity = originals.std(axis=0).mean()
    recon_diversity = clipped.std(axis=0).mean()

    metrics = {
        "mae": float(np.mean(np.abs(reconstructions - originals))),
        "mse": float(np.mean((reconstructions - originals) ** 2)),
        "clipped_mse": float(np.mean((clipped - originals) ** 2)),
        "pixel_correlation": float(np.mean(sample_correlations)),
        "foreground_iou": float(np.mean(iou)),
        "foreground_f1": float(np.mean(f1)),
        "output_min": float(np.min(reconstructions)),
        "output_max": float(np.max(reconstructions)),
        "output_mean": float(np.mean(reconstructions)),
        "output_std": float(np.std(reconstructions)),
        "diversity_ratio": float(recon_diversity / (original_diversity + 1e-8)),
    }

    if target_images is not None:
        if visibility is None:
            changed = infer_corruption_changed_mask(
                inputs,
                originals,
                atol=change_atol,
            ).reshape(originals.shape)
        else:
            changed = np.asarray(visibility).reshape(originals.shape) < 0.5
        unchanged = ~changed
        error = reconstructions - originals
        clipped_error = clipped - originals
        input_error = inputs - originals

        def safe_mean(values: np.ndarray) -> float:
            if values.size == 0:
                return float("nan")
            return float(np.mean(values))

        changed_input_mse = safe_mean(input_error[changed] ** 2)
        changed_mse = safe_mean(error[changed] ** 2)
        metrics.update(
            {
                "changed_fraction": float(np.mean(changed)),
                "input_mse": float(np.mean(input_error**2)),
                "changed_input_mse": changed_input_mse,
                "changed_mse": changed_mse,
                "changed_clipped_mse": safe_mean(clipped_error[changed] ** 2),
                "changed_mae": safe_mean(np.abs(error[changed])),
                "changed_improvement": changed_input_mse - changed_mse,
                "unchanged_mse": safe_mean(error[unchanged] ** 2),
                "unchanged_clipped_mse": safe_mean(clipped_error[unchanged] ** 2),
                "unchanged_mae": safe_mean(np.abs(error[unchanged])),
                "changed_target_mean": safe_mean(originals[changed]),
                "changed_output_mean": safe_mean(reconstructions[changed]),
            }
        )

    return metrics


def annotate_mnist_summary(
    result: AutoencoderExperimentResult,
    baselines: Dict[str, float],
    quality_metrics: Optional[Dict[str, float]] = None,
    *,
    loss_protocol: str = "full_reconstruction",
) -> None:
    """Attach MNIST baseline comparisons to the saved experiment summary."""

    summary_path = result.paths.metrics / "summary.json"
    if not summary_path.exists():
        return

    with open(summary_path) as f:
        summary = json.load(f)

    comparison_loss = summary.get("final_eval_loss", summary.get("eval_loss"))
    summary["baselines"] = baselines
    summary["loss_protocol"] = loss_protocol
    if quality_metrics is not None:
        summary["quality"] = quality_metrics
    if loss_protocol == "boundary_clamped":
        summary["primary_metric"] = "hidden_region_mse"
        summary["full_image_metrics_are_secondary"] = True
    elif comparison_loss is not None:
        pixel_mean_mse = baselines["pixel_mean_mse"]
        summary["beats_pixel_mean"] = bool(comparison_loss < pixel_mean_mse)
        summary["margin_vs_pixel_mean"] = float(pixel_mean_mse - comparison_loss)

    write_json(summary_path, summary)


def save_mnist_artifacts(
    model: eqx.Module,
    batch: Optional[Array | Tuple[Array, Array]],
    paths: ExperimentPaths,
    epoch: int,
    metrics: Dict[str, object],
) -> None:
    """Save reconstructions plus latent and oscillator-state traces."""

    if batch is None:
        return

    if isinstance(batch, tuple):
        batch_inputs, batch_targets = batch
    else:
        batch_inputs = batch
        batch_targets = None

    n_examples = min(8, int(batch_inputs.shape[0]))
    samples = batch_inputs[:n_examples]
    targets = None if batch_targets is None else batch_targets[:n_examples]
    originals, reconstructions = prepare_reconstructions(model, samples, targets)
    inputs = primary_image_channel(samples).reshape(samples.shape[0], 28, 28)

    n_rows = 2 if targets is None else 3
    fig, axes = plt.subplots(n_rows, n_examples, figsize=(1.6 * n_examples, 1.6 * n_rows))
    if n_examples == 1:
        axes = np.asarray(axes).reshape(n_rows, 1)
    for idx in range(n_examples):
        row = 0
        if targets is not None:
            axes[row, idx].imshow(np.asarray(inputs[idx]), cmap="gray", vmin=0, vmax=1)
            axes[row, idx].axis("off")
            row += 1
        axes[row, idx].imshow(np.asarray(originals[idx]), cmap="gray", vmin=0, vmax=1)
        axes[row, idx].axis("off")
        row += 1
        axes[row, idx].imshow(
            np.asarray(reconstructions[idx]), cmap="gray", vmin=0, vmax=1
        )
        axes[row, idx].axis("off")
    if targets is not None:
        axes[0, 0].set_ylabel("input")
        axes[1, 0].set_ylabel("target")
        axes[2, 0].set_ylabel("recon")
    else:
        axes[0, 0].set_ylabel("input")
        axes[1, 0].set_ylabel("recon")
    fig.suptitle(f"MNIST Reconstructions - Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(paths.plots / f"mnist_reconstructions_epoch_{epoch:03d}.png", dpi=150)
    plt.close(fig)

    if hasattr(model, "collect_trace"):
        trace = model.collect_trace(samples)
        np.savez(
            paths.traces / f"mnist_latent_state_epoch_{epoch:03d}.npz",
            **{key: np.asarray(value) for key, value in trace.items()},
        )
    else:
        sequence = model.images_to_sequence(samples)
        latent = model.encode(samples, use_phase_init=True)
        trace = collect_sequence_state_trace(
            model.encoder.rnn,
            sequence,
            use_phase_init=True,
        )

        np.savez(
            paths.traces / f"mnist_latent_state_epoch_{epoch:03d}.npz",
            latent=np.asarray(latent),
            encoder_outputs=np.asarray(trace["outputs"]),
            encoder_positions=np.asarray(trace["positions"]),
            encoder_velocities=np.asarray(trace["velocities"]),
            final_position=np.asarray(trace["final_position"]),
            final_velocity=np.asarray(trace["final_velocity"]),
        )

    np.savez(
        paths.artifacts / f"mnist_reconstructions_epoch_{epoch:03d}.npz",
        inputs=np.asarray(inputs),
        originals=np.asarray(originals),
        reconstructions=np.asarray(reconstructions),
    )


def export_encoder_complex_states(
    model: AmplitudeVelocityAutoencoder,
    images: Array,
    save_path: str,
    use_phase_init: bool = True,
    eps: float = 1e-8,
) -> None:
    """Export final encoder states as amplitude and phase arrays."""

    patch_sequence = model.images_to_sequence(images)
    result = model.encoder.rnn(
        patch_sequence,
        return_trajectories=True,
        use_phase_init=use_phase_init,
    )
    x_state, v_state = result["final_state"]

    oscillator = model.encoder.rnn.cell.oscillator
    omega = getattr(oscillator, "omega", 1.0)
    if isinstance(omega, (int, float)):
        omega = jnp.ones((x_state.shape[-1],), dtype=jnp.float32) * float(omega)
    else:
        omega = jnp.asarray(omega)
        if omega.ndim == 0:
            omega = jnp.ones((x_state.shape[-1],), dtype=omega.dtype) * omega
        if omega.shape[-1] != x_state.shape[-1]:
            omega = jnp.broadcast_to(omega, (x_state.shape[-1],))

    z = x_state + 1j * (v_state / (omega + eps))
    np.savez(save_path, amplitude=np.asarray(jnp.abs(z)), phase=np.asarray(jnp.angle(z)))


def run_mnist_experiment(
    config: MNISTAutoencoderExperimentConfig,
) -> AutoencoderExperimentResult:
    """Run the canonical MNIST oscillator autoencoder benchmark."""

    logger = _logger()
    train_images, _, eval_images, _ = load_mnist_data(
        source=config.data_source,
        train_limit=config.train_limit,
        eval_limit=config.eval_limit,
        seed=config.run.seed,
    )
    corruption_seed = (
        config.run.seed + 10_000
        if config.corruption_seed is None
        else int(config.corruption_seed)
    )
    train_inputs, train_visibility = corrupt_mnist_images_with_visibility(
        train_images,
        mode=config.corruption_mode,
        patch_shape=config.patch_shape,
        fraction=config.corruption_fraction,
        seed=corruption_seed,
        noise_std=config.corruption_noise_std,
        mask_value=config.corruption_mask_value,
    )
    eval_inputs, eval_visibility = corrupt_mnist_images_with_visibility(
        eval_images,
        mode=config.corruption_mode,
        patch_shape=config.patch_shape,
        fraction=config.corruption_fraction,
        seed=corruption_seed + 1,
        noise_std=config.corruption_noise_std,
        mask_value=config.corruption_mask_value,
    )
    uses_corruption = config.corruption_mode != "none"
    if not uses_corruption and config.corruption_input_mode != "image":
        raise ValueError("image_plus_mask input mode requires a corruption_mode")
    if config.corruption_protocol not in {
        "full_reconstruction",
        "boundary_clamped",
    }:
        raise ValueError(
            "corruption_protocol must be 'full_reconstruction' or "
            "'boundary_clamped'"
        )
    if config.corruption_protocol == "boundary_clamped":
        if not uses_corruption:
            raise ValueError("boundary_clamped protocol requires a corruption_mode")
        if config.corruption_input_mode != "image_plus_mask":
            raise ValueError(
                "boundary_clamped protocol requires image_plus_mask inputs"
            )
    if config.eval_winfree_steps is not None:
        if config.eval_winfree_steps < 1:
            raise ValueError("eval_winfree_steps must be >= 1")
        if config.run.mode != "eval":
            raise ValueError("eval_winfree_steps is only supported in eval mode")
        if config.checkpoint is None:
            raise ValueError("eval_winfree_steps requires a checkpoint")
    train_model_inputs = _select_mnist_model_inputs(
        train_inputs,
        train_visibility,
        config,
    )
    eval_model_inputs = _select_mnist_model_inputs(
        eval_inputs,
        eval_visibility,
        config,
    )
    if (
        config.corruption_visible_loss_weight < 0.0
        or config.corruption_changed_loss_weight < 0.0
    ):
        raise ValueError("corruption loss weights must be non-negative")
    if (
        config.corruption_visible_loss_weight == 0.0
        and config.corruption_changed_loss_weight == 0.0
    ):
        raise ValueError("at least one corruption loss weight must be positive")
    uses_weighted_corruption_loss = uses_corruption and (
        config.corruption_visible_loss_weight != 1.0
        or config.corruption_changed_loss_weight != 1.0
    )
    if config.corruption_protocol == "boundary_clamped":
        prediction_transform = clamp_predictions_to_visible_inputs
        train_loss_weights = compute_visibility_loss_weights(
            train_visibility,
            visible_weight=0.0,
            changed_weight=1.0,
        )
        eval_loss_weights = compute_visibility_loss_weights(
            eval_visibility,
            visible_weight=0.0,
            changed_weight=1.0,
        )
    else:
        prediction_transform = None
        train_loss_weights = (
            compute_visibility_loss_weights(
                train_visibility,
                visible_weight=config.corruption_visible_loss_weight,
                changed_weight=config.corruption_changed_loss_weight,
            )
            if uses_weighted_corruption_loss
            else None
        )
        eval_loss_weights = (
            compute_visibility_loss_weights(
                eval_visibility,
                visible_weight=config.corruption_visible_loss_weight,
                changed_weight=config.corruption_changed_loss_weight,
            )
            if uses_weighted_corruption_loss
            else None
        )
    baselines = compute_mnist_baselines(train_images, eval_images)

    def artifact_callback(
        model: eqx.Module,
        batch: Optional[Array | Tuple[Array, Array]],
        paths: ExperimentPaths,
        epoch: int,
        metrics: Dict[str, object],
    ) -> None:
        artifact_model = (
            BoundaryClampedPredictionView(model)
            if prediction_transform is not None
            else model
        )
        save_mnist_artifacts(artifact_model, batch, paths, epoch, metrics)

    if config.run.mode == "eval":
        if config.checkpoint is not None:
            checkpoint_overrides = (
                {"winfree_steps": int(config.eval_winfree_steps)}
                if config.eval_winfree_steps is not None
                else None
            )
            model, _, _ = _load_checkpoint_bundle(
                config.checkpoint,
                hyperparam_overrides=checkpoint_overrides,
            )
        else:
            model = build_mnist_model(config, jax.random.PRNGKey(config.run.seed))
        result = run_eval_only(
            model,
            eval_model_inputs,
            config.run,
            sample_axis=0,
            eval_targets=eval_images if uses_corruption else None,
            eval_loss_weights=eval_loss_weights,
            prediction_transform=prediction_transform,
            task_config=asdict(config),
            artifact_callback=artifact_callback,
        )
        quality_metrics = compute_mnist_quality_metrics(
            result.model,
            eval_model_inputs,
            target_images=eval_images if uses_corruption else None,
            batch_size=config.run.batch_size,
            change_atol=config.corruption_change_atol,
            prediction_transform=prediction_transform,
        )
        annotate_mnist_summary(
            result,
            baselines,
            quality_metrics,
            loss_protocol=config.corruption_protocol,
        )
        return result

    start_epoch = 0
    initial_best_loss = None
    initial_best_epoch = None
    checkpoint_hyperparams = _checkpoint_hyperparams(config)
    if config.checkpoint is not None:
        model, checkpoint_hyperparams, checkpoint_metadata = _load_checkpoint_bundle(
            config.checkpoint
        )
        start_epoch = int(checkpoint_metadata.get("epoch", 0))
        initial_best_loss = _checkpoint_metric_loss(checkpoint_metadata)
        if initial_best_loss is not None and start_epoch > 0:
            initial_best_epoch = start_epoch
        logger.info(
            "resuming model weights from %s at epoch=%s",
            config.checkpoint,
            start_epoch,
        )
    else:
        model = build_mnist_model(config, jax.random.PRNGKey(config.run.seed))

    result = train_autoencoder(
        model,
        train_model_inputs,
        eval_model_inputs,
        config.run,
        sample_axis=0,
        train_targets=train_images if uses_corruption else None,
        eval_targets=eval_images if uses_corruption else None,
        train_loss_weights=train_loss_weights,
        eval_loss_weights=eval_loss_weights,
        prediction_transform=prediction_transform,
        task_config=asdict(config),
        checkpoint_hyperparams=checkpoint_hyperparams,
        artifact_callback=artifact_callback,
        logger=logger,
        start_epoch=start_epoch,
        resume_from_checkpoint=config.checkpoint,
        initial_best_loss=initial_best_loss,
        initial_best_epoch=initial_best_epoch,
    )
    quality_metrics = compute_mnist_quality_metrics(
        result.model,
        eval_model_inputs,
        target_images=eval_images if uses_corruption else None,
        batch_size=config.run.batch_size,
        change_atol=config.corruption_change_atol,
        prediction_transform=prediction_transform,
    )
    annotate_mnist_summary(
        result,
        baselines,
        quality_metrics,
        loss_protocol=config.corruption_protocol,
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OscNet MNIST oscillator autoencoder reference benchmark."
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reference/mnist"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--output-bounds-penalty", type=float, default=0.0)
    parser.add_argument("--latent-variance-weight", type=float, default=0.0)
    parser.add_argument("--latent-std-floor", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--artifact-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument(
        "--model-family",
        choices=[
            "amplitude_velocity",
            "feedforward_patch",
            "recurrent_conv",
            "recurrent_conv_prior_refinement",
            "conv_lstm",
            "winfree_conditional",
            "winfree_rate_phase",
            "winfree_global_rate_phase",
            "winfree_prior_refinement",
            "winfree_coarse_global_rate_phase",
            "winfree_coarse_rate_phase",
            "winfree_coarse_predictive_rate_phase",
            "winfree_field",
        ],
        default="amplitude_velocity",
    )
    parser.add_argument(
        "--decoder-mode",
        choices=["repeat", "autoregressive", "positional"],
        default="repeat",
    )
    parser.add_argument("--latent-conditioning-strength", type=float, default=1.0)
    parser.add_argument(
        "--feedforward-latent-output-skip",
        choices=["none", "sequence"],
        default="sequence",
    )
    parser.add_argument(
        "--feedforward-latent-output-skip-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--feedforward-output-activation",
        choices=["identity", "sigmoid", "tanh01"],
        default="identity",
    )
    parser.add_argument("--recurrent-conv-steps", type=int, default=8)
    parser.add_argument("--recurrent-conv-kernel-size", type=int, default=3)
    parser.add_argument(
        "--recurrent-conv-residual-strength",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--recurrent-conv-refinement-strength",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--recurrent-conv-output-activation",
        choices=["identity", "sigmoid", "tanh01"],
        default="identity",
    )
    parser.add_argument("--conv-lstm-steps", type=int, default=8)
    parser.add_argument("--conv-lstm-kernel-size", type=int, default=3)
    parser.add_argument("--conv-lstm-forget-bias", type=float, default=1.0)
    parser.add_argument(
        "--conv-lstm-output-activation",
        choices=["identity", "sigmoid", "tanh01"],
        default="identity",
    )
    parser.add_argument(
        "--output-activation",
        choices=["identity", "sigmoid", "tanh01"],
        default="identity",
    )
    parser.add_argument(
        "--oscillator",
        choices=["learnable", "adaptive", "nonlinear"],
        default="learnable",
    )
    parser.add_argument("--winfree-steps", type=int, default=8)
    parser.add_argument("--winfree-gamma", type=float, default=0.1)
    parser.add_argument("--winfree-global-gamma", type=float, default=0.05)
    parser.add_argument("--winfree-coarse-grid-size", type=int, default=2)
    parser.add_argument("--winfree-coupling-strength", type=float, default=1.0)
    parser.add_argument("--winfree-coupling-decay-length", type=float, default=None)
    parser.add_argument(
        "--winfree-coupling-mode",
        choices=["matrix", "conv", "adaptive", "conv_adaptive", "conv_matrix"],
        default="matrix",
    )
    parser.add_argument("--winfree-coupling-kernel-size", type=int, default=3)
    parser.add_argument("--winfree-adaptive-coupling-strength", type=float, default=0.1)
    parser.add_argument("--winfree-omega-scale", type=float, default=1.0)
    parser.add_argument(
        "--winfree-latent-readout",
        choices=["none", "phase_bias", "concat"],
        default="none",
    )
    parser.add_argument("--winfree-latent-readout-strength", type=float, default=1.0)
    parser.add_argument(
        "--winfree-latent-output-skip",
        choices=["none", "sequence"],
        default="none",
    )
    parser.add_argument(
        "--winfree-latent-output-skip-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--winfree-field-activation",
        choices=["identity", "relu", "tanh"],
        default="relu",
    )
    parser.add_argument(
        "--winfree-si-func",
        choices=["trig", "mlp"],
        default="trig",
    )
    parser.add_argument("--winfree-si-hidden-ratio", type=int, default=2)
    parser.add_argument("--winfree-group-size", type=int, default=1)
    parser.add_argument(
        "--winfree-phase-init",
        choices=["learned", "rotary_2d"],
        default="learned",
    )
    parser.add_argument("--winfree-phase-init-scale", type=float, default=1.0)
    parser.add_argument("--winfree-rate-kernel-size", type=int, default=3)
    parser.add_argument("--winfree-rate-update-rate", type=float, default=0.5)
    parser.add_argument("--winfree-rate-gate-strength", type=float, default=1.0)
    parser.add_argument(
        "--winfree-visibility-gate",
        choices=["none", "visibility", "shuffle"],
        default="none",
    )
    parser.add_argument("--winfree-visibility-drive-floor", type=float, default=0.0)
    parser.add_argument(
        "--winfree-missing-transport-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument("--winfree-global-gate-strength", type=float, default=0.5)
    parser.add_argument(
        "--winfree-global-phase-control",
        choices=["none", "shuffle"],
        default="none",
    )
    parser.add_argument("--winfree-global-content-strength", type=float, default=0.5)
    parser.add_argument(
        "--winfree-global-content-control",
        choices=["none", "shuffle"],
        default="none",
    )
    parser.add_argument("--winfree-coarse-readout-strength", type=float, default=0.5)
    parser.add_argument("--winfree-refinement-strength", type=float, default=0.25)
    parser.add_argument(
        "--winfree-output-activation",
        choices=["identity", "sigmoid", "tanh01"],
        default="identity",
    )
    parser.add_argument(
        "--corruption-mode",
        choices=["none", "patch_mask", "block_occlusion", "gaussian_noise"],
        default="none",
    )
    parser.add_argument("--corruption-fraction", type=float, default=0.5)
    parser.add_argument("--corruption-seed", type=int, default=None)
    parser.add_argument("--corruption-noise-std", type=float, default=0.35)
    parser.add_argument("--corruption-mask-value", type=float, default=0.0)
    parser.add_argument(
        "--corruption-input-mode",
        choices=["image", "image_plus_mask"],
        default="image",
    )
    parser.add_argument(
        "--corruption-protocol",
        choices=["full_reconstruction", "boundary_clamped"],
        default="full_reconstruction",
    )
    parser.add_argument("--corruption-visible-loss-weight", type=float, default=1.0)
    parser.add_argument("--corruption-changed-loss-weight", type=float, default=1.0)
    parser.add_argument("--corruption-change-atol", type=float, default=1e-6)
    parser.add_argument(
        "--data-source",
        choices=["tfds", "idx", "synthetic"],
        default="idx",
    )
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--eval-limit", type=int, default=1_000)
    parser.add_argument("--eval-winfree-steps", type=int, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> MNISTAutoencoderExperimentConfig:
    run = AutoencoderExperimentConfig(
        name="mnist_oscillator_autoencoder",
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_bounds_penalty=args.output_bounds_penalty,
        latent_variance_weight=args.latent_variance_weight,
        latent_std_floor=args.latent_std_floor,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        artifact_every=args.artifact_every,
    )
    return MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        patch_shape=(args.patch_size, args.patch_size),
        model_family=args.model_family,
        decoder_mode=args.decoder_mode,
        latent_conditioning_strength=args.latent_conditioning_strength,
        feedforward_latent_output_skip=args.feedforward_latent_output_skip,
        feedforward_latent_output_skip_strength=(
            args.feedforward_latent_output_skip_strength
        ),
        feedforward_output_activation=args.feedforward_output_activation,
        recurrent_conv_steps=args.recurrent_conv_steps,
        recurrent_conv_kernel_size=args.recurrent_conv_kernel_size,
        recurrent_conv_residual_strength=args.recurrent_conv_residual_strength,
        recurrent_conv_refinement_strength=args.recurrent_conv_refinement_strength,
        recurrent_conv_output_activation=args.recurrent_conv_output_activation,
        conv_lstm_steps=args.conv_lstm_steps,
        conv_lstm_kernel_size=args.conv_lstm_kernel_size,
        conv_lstm_forget_bias=args.conv_lstm_forget_bias,
        conv_lstm_output_activation=args.conv_lstm_output_activation,
        output_activation=args.output_activation,
        oscillator=args.oscillator,
        winfree_steps=args.winfree_steps,
        winfree_gamma=args.winfree_gamma,
        winfree_global_gamma=args.winfree_global_gamma,
        winfree_coarse_grid_size=args.winfree_coarse_grid_size,
        winfree_coupling_strength=args.winfree_coupling_strength,
        winfree_coupling_decay_length=args.winfree_coupling_decay_length,
        winfree_coupling_mode=args.winfree_coupling_mode,
        winfree_coupling_kernel_size=args.winfree_coupling_kernel_size,
        winfree_adaptive_coupling_strength=args.winfree_adaptive_coupling_strength,
        winfree_omega_scale=args.winfree_omega_scale,
        winfree_latent_readout=args.winfree_latent_readout,
        winfree_latent_readout_strength=args.winfree_latent_readout_strength,
        winfree_latent_output_skip=args.winfree_latent_output_skip,
        winfree_latent_output_skip_strength=args.winfree_latent_output_skip_strength,
        winfree_field_activation=args.winfree_field_activation,
        winfree_si_func=args.winfree_si_func,
        winfree_si_hidden_ratio=args.winfree_si_hidden_ratio,
        winfree_group_size=args.winfree_group_size,
        winfree_phase_init=args.winfree_phase_init,
        winfree_phase_init_scale=args.winfree_phase_init_scale,
        winfree_rate_kernel_size=args.winfree_rate_kernel_size,
        winfree_rate_update_rate=args.winfree_rate_update_rate,
        winfree_rate_gate_strength=args.winfree_rate_gate_strength,
        winfree_visibility_gate=args.winfree_visibility_gate,
        winfree_visibility_drive_floor=args.winfree_visibility_drive_floor,
        winfree_missing_transport_strength=args.winfree_missing_transport_strength,
        winfree_global_gate_strength=args.winfree_global_gate_strength,
        winfree_global_phase_control=args.winfree_global_phase_control,
        winfree_global_content_strength=args.winfree_global_content_strength,
        winfree_global_content_control=args.winfree_global_content_control,
        winfree_coarse_readout_strength=args.winfree_coarse_readout_strength,
        winfree_refinement_strength=args.winfree_refinement_strength,
        winfree_output_activation=args.winfree_output_activation,
        corruption_mode=args.corruption_mode,
        corruption_fraction=args.corruption_fraction,
        corruption_seed=args.corruption_seed,
        corruption_noise_std=args.corruption_noise_std,
        corruption_mask_value=args.corruption_mask_value,
        corruption_input_mode=args.corruption_input_mode,
        corruption_protocol=args.corruption_protocol,
        corruption_visible_loss_weight=args.corruption_visible_loss_weight,
        corruption_changed_loss_weight=args.corruption_changed_loss_weight,
        corruption_change_atol=args.corruption_change_atol,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        checkpoint=args.checkpoint,
        eval_winfree_steps=args.eval_winfree_steps,
    )


def main(argv: Optional[list[str]] = None) -> AutoencoderExperimentResult:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_mnist_experiment(config_from_args(args))


if __name__ == "__main__":
    main()
