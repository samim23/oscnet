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
from typing import Dict, Optional, Tuple

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
    collect_sequence_state_trace,
    run_eval_only,
    train_autoencoder,
    write_json,
)
from oscnet.models import AmplitudeVelocityAutoencoder, WinfreePatchAutoencoder

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
    oscillator: str = "learnable"
    winfree_steps: int = 8
    winfree_gamma: float = 0.1
    winfree_coupling_strength: float = 1.0
    winfree_omega_scale: float = 1.0
    winfree_field_activation: str = "relu"
    winfree_si_func: str = "trig"
    winfree_si_hidden_ratio: int = 2
    winfree_group_size: int = 1
    winfree_output_activation: str = "identity"
    data_source: str = "idx"
    train_limit: Optional[int] = 10_000
    eval_limit: Optional[int] = 1_000
    checkpoint: Optional[Path] = None


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


def build_mnist_model(
    config: MNISTAutoencoderExperimentConfig,
    key: jax.random.PRNGKey,
) -> eqx.Module:
    if config.model_family == "winfree_field":
        return WinfreePatchAutoencoder(
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            patch_shape=config.patch_shape,
            group_size=config.winfree_group_size,
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            coupling_strength=config.winfree_coupling_strength,
            latent_conditioning_strength=config.latent_conditioning_strength,
            omega_scale=config.winfree_omega_scale,
            field_activation=config.winfree_field_activation,
            si_func=config.winfree_si_func,
            si_hidden_ratio=config.winfree_si_hidden_ratio,
            output_activation=config.winfree_output_activation,
            key=key,
        )

    if config.model_family != "amplitude_velocity":
        raise ValueError("model_family must be 'amplitude_velocity' or 'winfree_field'")

    oscillator_class, oscillator_params = _mnist_oscillator_spec(
        config.oscillator,
        config.hidden_dim,
    )
    return AmplitudeVelocityAutoencoder(
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        patch_shape=config.patch_shape,
        decoder_mode=config.decoder_mode,
        latent_conditioning_strength=config.latent_conditioning_strength,
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
        "latent_conditioning_strength": config.latent_conditioning_strength,
    }
    if config.model_family == "winfree_field":
        hyperparams.update(
            {
                "winfree_steps": config.winfree_steps,
                "winfree_gamma": config.winfree_gamma,
                "winfree_coupling_strength": config.winfree_coupling_strength,
                "winfree_omega_scale": config.winfree_omega_scale,
                "winfree_field_activation": config.winfree_field_activation,
                "winfree_si_func": config.winfree_si_func,
                "winfree_si_hidden_ratio": config.winfree_si_hidden_ratio,
                "winfree_group_size": config.winfree_group_size,
                "winfree_output_activation": config.winfree_output_activation,
            }
        )
        return hyperparams

    _, oscillator_params = _mnist_oscillator_spec(config.oscillator, config.hidden_dim)
    hyperparams.update(
        {
            "decoder_mode": config.decoder_mode,
            "oscillator": config.oscillator,
            "oscillator_params": oscillator_params,
        }
    )
    return hyperparams


def _build_mnist_model_from_hyperparams(**hyperparams) -> eqx.Module:
    model_family = hyperparams.get("model_family", "amplitude_velocity")
    hidden_dim = int(hyperparams["hidden_dim"])
    patch_shape = tuple(hyperparams.get("patch_shape", (4, 4)))

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
            latent_conditioning_strength=float(
                hyperparams.get("latent_conditioning_strength", 1.0)
            ),
            omega_scale=float(hyperparams.get("winfree_omega_scale", 1.0)),
            field_activation=hyperparams.get("winfree_field_activation", "relu"),
            si_func=hyperparams.get("winfree_si_func", "trig"),
            si_hidden_ratio=int(hyperparams.get("winfree_si_hidden_ratio", 2)),
            output_activation=hyperparams.get("winfree_output_activation", "identity"),
            key=jax.random.PRNGKey(0),
        )

    if model_family != "amplitude_velocity":
        raise ValueError("model_family must be 'amplitude_velocity' or 'winfree_field'")

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
        oscillator_class=oscillator_class,
        oscillator_params=oscillator_params,
        initial_amplitude=0.1,
        key=jax.random.PRNGKey(0),
    )


def _load_checkpoint(checkpoint_path: Path) -> eqx.Module:
    with open(checkpoint_path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = _build_mnist_model_from_hyperparams(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)


def prepare_reconstructions(model, samples):
    """Prepare 28x28 original/reconstruction arrays for visualization."""

    originals = samples.reshape(samples.shape[0], 28, 28)
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
) -> np.ndarray:
    predictions = []
    n_images = int(images.shape[0])
    for start in range(0, n_images, batch_size):
        batch = images[start : start + batch_size]
        predictions.append(np.asarray(model(batch)))
    return np.concatenate(predictions, axis=0)


def compute_mnist_quality_metrics(
    model: eqx.Module,
    eval_images: Array,
    *,
    batch_size: int,
    threshold: float = 0.25,
) -> Dict[str, float]:
    """Compute image-quality diagnostics for MNIST reconstructions."""

    originals = np.asarray(eval_images).reshape(eval_images.shape[0], -1)
    reconstructions = _predict_images_in_batches(model, eval_images, batch_size)
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

    return {
        "mae": float(np.mean(np.abs(reconstructions - originals))),
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


def annotate_mnist_summary(
    result: AutoencoderExperimentResult,
    baselines: Dict[str, float],
    quality_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Attach MNIST baseline comparisons to the saved experiment summary."""

    summary_path = result.paths.metrics / "summary.json"
    if not summary_path.exists():
        return

    with open(summary_path) as f:
        summary = json.load(f)

    comparison_loss = summary.get("final_eval_loss", summary.get("eval_loss"))
    summary["baselines"] = baselines
    if quality_metrics is not None:
        summary["quality"] = quality_metrics
    if comparison_loss is not None:
        pixel_mean_mse = baselines["pixel_mean_mse"]
        summary["beats_pixel_mean"] = bool(comparison_loss < pixel_mean_mse)
        summary["margin_vs_pixel_mean"] = float(pixel_mean_mse - comparison_loss)

    write_json(summary_path, summary)


def save_mnist_artifacts(
    model: eqx.Module,
    batch: Optional[Array],
    paths: ExperimentPaths,
    epoch: int,
    metrics: Dict[str, object],
) -> None:
    """Save reconstructions plus latent and oscillator-state traces."""

    if batch is None:
        return

    n_examples = min(8, int(batch.shape[0]))
    samples = batch[:n_examples]
    originals, reconstructions = prepare_reconstructions(model, samples)

    fig, axes = plt.subplots(2, n_examples, figsize=(1.6 * n_examples, 3.2))
    if n_examples == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for idx in range(n_examples):
        axes[0, idx].imshow(np.asarray(originals[idx]), cmap="gray", vmin=0, vmax=1)
        axes[0, idx].axis("off")
        axes[1, idx].imshow(
            np.asarray(reconstructions[idx]), cmap="gray", vmin=0, vmax=1
        )
        axes[1, idx].axis("off")
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
            latent=np.asarray(trace["latent"]),
            encoder_omega=np.asarray(trace["encoder_omega"]),
            encoder_initial_theta=np.asarray(trace["encoder_initial_theta"]),
            encoder_final_theta=np.asarray(trace["encoder_final_theta"]),
            encoder_thetas=np.asarray(trace["encoder_thetas"]),
            encoder_energies=np.asarray(trace["encoder_energies"]),
            decoder_omega=np.asarray(trace["decoder_omega"]),
            decoder_initial_theta=np.asarray(trace["decoder_initial_theta"]),
            decoder_final_theta=np.asarray(trace["decoder_final_theta"]),
            decoder_thetas=np.asarray(trace["decoder_thetas"]),
            decoder_energies=np.asarray(trace["decoder_energies"]),
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
    baselines = compute_mnist_baselines(train_images, eval_images)

    if config.run.mode == "eval":
        if config.checkpoint is not None:
            model = _load_checkpoint(config.checkpoint)
        else:
            model = build_mnist_model(config, jax.random.PRNGKey(config.run.seed))
        result = run_eval_only(
            model,
            eval_images,
            config.run,
            sample_axis=0,
            task_config=asdict(config),
            artifact_callback=save_mnist_artifacts,
        )
        quality_metrics = compute_mnist_quality_metrics(
            result.model,
            eval_images,
            batch_size=config.run.batch_size,
        )
        annotate_mnist_summary(result, baselines, quality_metrics)
        return result

    model = build_mnist_model(config, jax.random.PRNGKey(config.run.seed))
    result = train_autoencoder(
        model,
        train_images,
        eval_images,
        config.run,
        sample_axis=0,
        task_config=asdict(config),
        checkpoint_hyperparams=_checkpoint_hyperparams(config),
        artifact_callback=save_mnist_artifacts,
        logger=logger,
    )
    quality_metrics = compute_mnist_quality_metrics(
        result.model,
        eval_images,
        batch_size=config.run.batch_size,
    )
    annotate_mnist_summary(result, baselines, quality_metrics)
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
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--artifact-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument(
        "--model-family",
        choices=["amplitude_velocity", "winfree_field"],
        default="amplitude_velocity",
    )
    parser.add_argument(
        "--decoder-mode",
        choices=["repeat", "autoregressive", "positional"],
        default="repeat",
    )
    parser.add_argument("--latent-conditioning-strength", type=float, default=1.0)
    parser.add_argument(
        "--oscillator",
        choices=["learnable", "adaptive", "nonlinear"],
        default="learnable",
    )
    parser.add_argument("--winfree-steps", type=int, default=8)
    parser.add_argument("--winfree-gamma", type=float, default=0.1)
    parser.add_argument("--winfree-coupling-strength", type=float, default=1.0)
    parser.add_argument("--winfree-omega-scale", type=float, default=1.0)
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
        "--winfree-output-activation",
        choices=["identity", "sigmoid", "tanh01"],
        default="identity",
    )
    parser.add_argument(
        "--data-source",
        choices=["tfds", "idx", "synthetic"],
        default="idx",
    )
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--eval-limit", type=int, default=1_000)
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
        oscillator=args.oscillator,
        winfree_steps=args.winfree_steps,
        winfree_gamma=args.winfree_gamma,
        winfree_coupling_strength=args.winfree_coupling_strength,
        winfree_omega_scale=args.winfree_omega_scale,
        winfree_field_activation=args.winfree_field_activation,
        winfree_si_func=args.winfree_si_func,
        winfree_si_hidden_ratio=args.winfree_si_hidden_ratio,
        winfree_group_size=args.winfree_group_size,
        winfree_output_activation=args.winfree_output_activation,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        checkpoint=args.checkpoint,
    )


def main(argv: Optional[list[str]] = None) -> AutoencoderExperimentResult:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_mnist_experiment(config_from_args(args))


if __name__ == "__main__":
    main()
