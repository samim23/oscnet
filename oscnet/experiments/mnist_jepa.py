"""JEPA-style MNIST patch representation prediction experiments."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from oscnet.experiments.harness import (
    ArtifactBatch,
    AutoencoderExperimentConfig,
    AutoencoderExperimentResult,
    ExperimentPaths,
    train_autoencoder,
    write_json,
)
from oscnet.experiments.mnist_autoencoder import (
    append_visibility_mask_channel,
    corrupt_mnist_images_with_visibility,
    load_mnist_data,
)
from oscnet.models import (
    ConvLSTMPatchJEPAPredictor,
    FeedForwardPatchJEPAPredictor,
    RecurrentConvPatchJEPAPredictor,
    WinfreeGlobalRatePhasePatchJEPAPredictor,
    WinfreeRatePhasePatchJEPAPredictor,
)

Array = jnp.ndarray


@dataclass(frozen=True)
class MNISTJEPAExperimentConfig:
    """Task-specific controls for MNIST patch representation prediction."""

    run: AutoencoderExperimentConfig
    model_family: str = "winfree_global_rate_phase"
    hidden_dim: int = 64
    latent_dim: int = 96
    embedding_dim: int = 8
    patch_shape: Tuple[int, int] = (4, 4)
    target_encoder: str = "dct_lowfreq"
    corruption_mode: str = "block_occlusion"
    corruption_fraction: float = 0.5
    corruption_seed: Optional[int] = None
    corruption_mask_value: float = 0.0
    corruption_input_mode: str = "image_plus_mask"
    recurrent_conv_steps: int = 8
    recurrent_conv_kernel_size: int = 3
    recurrent_conv_residual_strength: float = 0.5
    conv_lstm_steps: int = 8
    conv_lstm_kernel_size: int = 3
    conv_lstm_forget_bias: float = 1.0
    winfree_steps: int = 8
    winfree_gamma: float = 0.1
    winfree_global_gamma: float = 0.05
    winfree_coupling_strength: float = 1.0
    winfree_coupling_mode: str = "conv"
    winfree_coupling_kernel_size: int = 3
    winfree_si_func: str = "mlp"
    winfree_rate_kernel_size: int = 3
    winfree_rate_update_rate: float = 0.5
    winfree_rate_gate_strength: float = 1.0
    winfree_global_gate_strength: float = 0.5
    data_source: str = "idx"
    train_limit: Optional[int] = 10_000
    eval_limit: Optional[int] = 1_000


def _logger() -> logging.Logger:
    logger = logging.getLogger("oscnet.experiments.mnist_jepa")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _mnist_input_channels(config: MNISTJEPAExperimentConfig) -> int:
    if config.corruption_input_mode == "image":
        return 1
    if config.corruption_input_mode == "image_plus_mask":
        return 2
    raise ValueError("corruption_input_mode must be 'image' or 'image_plus_mask'")


def _mnist_input_patch_dim(config: MNISTJEPAExperimentConfig) -> int:
    return config.patch_shape[0] * config.patch_shape[1] * _mnist_input_channels(
        config
    )


def _select_model_inputs(
    corrupted_images: Array,
    visibility: Array,
    config: MNISTJEPAExperimentConfig,
) -> Array:
    if config.corruption_input_mode == "image":
        return corrupted_images
    if config.corruption_input_mode == "image_plus_mask":
        return append_visibility_mask_channel(corrupted_images, visibility)
    raise ValueError("corruption_input_mode must be 'image' or 'image_plus_mask'")


def _patchify_images(images: Array, patch_shape: Tuple[int, int]) -> Array:
    height = width = 28
    patch_height, patch_width = patch_shape
    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError("patch_shape must divide 28x28")
    batch_size = images.shape[0]
    patches = images.reshape(batch_size, height, width)
    patches = patches.reshape(
        batch_size,
        height // patch_height,
        patch_height,
        width // patch_width,
        patch_width,
    )
    patches = patches.transpose(0, 1, 3, 2, 4)
    return patches.reshape(
        batch_size,
        (height // patch_height) * (width // patch_width),
        patch_height * patch_width,
    )


def dct_lowfreq_basis(
    patch_shape: Tuple[int, int],
    embedding_dim: int,
) -> Array:
    """Create an orthonormal low-frequency 2D DCT basis for patch targets."""

    patch_height, patch_width = patch_shape
    patch_dim = patch_height * patch_width
    if embedding_dim < 1 or embedding_dim > patch_dim:
        raise ValueError("embedding_dim must be between 1 and patch size")

    y = np.arange(patch_height, dtype=np.float32)
    x = np.arange(patch_width, dtype=np.float32)
    basis = []
    for u in range(patch_height):
        for v in range(patch_width):
            alpha_u = np.sqrt(1.0 / patch_height) if u == 0 else np.sqrt(2.0 / patch_height)
            alpha_v = np.sqrt(1.0 / patch_width) if v == 0 else np.sqrt(2.0 / patch_width)
            yy, xx = np.meshgrid(y, x, indexing="ij")
            values = (
                alpha_u
                * alpha_v
                * np.cos(np.pi * (2.0 * yy + 1.0) * u / (2.0 * patch_height))
                * np.cos(np.pi * (2.0 * xx + 1.0) * v / (2.0 * patch_width))
            )
            basis.append((u + v, u, v, values.reshape(-1)))
    basis.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = np.stack([item[3] for item in basis[:embedding_dim]], axis=1)
    return jnp.asarray(selected, dtype=jnp.float32)


def compute_patch_embeddings(
    images: Array,
    *,
    patch_shape: Tuple[int, int],
    embedding_dim: int,
    target_encoder: str = "dct_lowfreq",
) -> Array:
    """Encode clean images into flattened per-patch target embeddings."""

    if target_encoder != "dct_lowfreq":
        raise ValueError("target_encoder must be 'dct_lowfreq'")
    patches = _patchify_images(images, patch_shape)
    basis = dct_lowfreq_basis(patch_shape, embedding_dim)
    embeddings = jnp.einsum("bnp,pe->bne", patches, basis)
    return embeddings.reshape(images.shape[0], -1)


def compute_hidden_patch_weights(
    visibility: Array,
    *,
    patch_shape: Tuple[int, int],
    embedding_dim: int,
) -> Array:
    """Create embedding loss weights for hidden target patches only."""

    patch_visibility = _patchify_images(visibility, patch_shape).mean(axis=-1)
    hidden = patch_visibility < 0.999
    weights = jnp.broadcast_to(
        hidden[:, :, None],
        (hidden.shape[0], hidden.shape[1], embedding_dim),
    )
    return weights.astype(jnp.float32).reshape(visibility.shape[0], -1)


def _weighted_mse(prediction: Array, target: Array, weights: Array) -> float:
    squared = (prediction - target) ** 2
    value = jnp.sum(squared * weights) / jnp.maximum(jnp.sum(weights), 1e-8)
    return float(value)


def compute_jepa_baselines(
    train_targets: Array,
    eval_targets: Array,
    eval_weights: Array,
) -> Dict[str, float]:
    zero = jnp.zeros_like(eval_targets)
    train_mean = jnp.mean(train_targets, axis=0, keepdims=True)
    mean_prediction = jnp.broadcast_to(train_mean, eval_targets.shape)
    return {
        "zero_embedding_mse": _weighted_mse(zero, eval_targets, eval_weights),
        "train_mean_embedding_mse": _weighted_mse(
            mean_prediction,
            eval_targets,
            eval_weights,
        ),
    }


def build_mnist_jepa_model(
    config: MNISTJEPAExperimentConfig,
    key: jax.random.PRNGKey,
) -> eqx.Module:
    input_dim = _mnist_input_patch_dim(config)
    common = {
        "input_dim": input_dim,
        "hidden_dim": config.hidden_dim,
        "embedding_dim": config.embedding_dim,
        "patch_shape": config.patch_shape,
        "key": key,
    }
    if config.model_family == "feedforward_patch":
        return FeedForwardPatchJEPAPredictor(
            latent_dim=config.latent_dim,
            **common,
        )
    if config.model_family == "recurrent_conv":
        return RecurrentConvPatchJEPAPredictor(
            steps=config.recurrent_conv_steps,
            kernel_size=config.recurrent_conv_kernel_size,
            residual_strength=config.recurrent_conv_residual_strength,
            **common,
        )
    if config.model_family == "conv_lstm":
        return ConvLSTMPatchJEPAPredictor(
            steps=config.conv_lstm_steps,
            kernel_size=config.conv_lstm_kernel_size,
            forget_bias=config.conv_lstm_forget_bias,
            **common,
        )
    if config.model_family == "winfree_rate_phase":
        return WinfreeRatePhasePatchJEPAPredictor(
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            si_func=config.winfree_si_func,
            rate_kernel_size=config.winfree_rate_kernel_size,
            rate_update_rate=config.winfree_rate_update_rate,
            rate_gate_strength=config.winfree_rate_gate_strength,
            **common,
        )
    if config.model_family == "winfree_global_rate_phase":
        return WinfreeGlobalRatePhasePatchJEPAPredictor(
            steps=config.winfree_steps,
            gamma=config.winfree_gamma,
            global_gamma=config.winfree_global_gamma,
            coupling_strength=config.winfree_coupling_strength,
            coupling_mode=config.winfree_coupling_mode,
            coupling_kernel_size=config.winfree_coupling_kernel_size,
            si_func=config.winfree_si_func,
            rate_kernel_size=config.winfree_rate_kernel_size,
            rate_update_rate=config.winfree_rate_update_rate,
            rate_gate_strength=config.winfree_rate_gate_strength,
            global_gate_strength=config.winfree_global_gate_strength,
            **common,
        )
    raise ValueError(
        "model_family must be 'feedforward_patch', 'recurrent_conv', "
        "'conv_lstm', 'winfree_rate_phase', or 'winfree_global_rate_phase'"
    )


def save_mnist_jepa_artifacts(
    model: eqx.Module,
    batch: Optional[ArtifactBatch],
    paths: ExperimentPaths,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    del metrics
    if batch is None:
        return
    if isinstance(batch, tuple):
        samples, targets = batch
    else:
        samples = batch
        targets = batch
    predictions = model(samples)
    trace = model.collect_trace(samples)
    np.savez(
        paths.artifacts / f"mnist_jepa_predictions_epoch_{epoch:03d}.npz",
        inputs=np.asarray(samples),
        targets=np.asarray(targets),
        predictions=np.asarray(predictions),
    )
    np.savez(
        paths.traces / f"mnist_jepa_trace_epoch_{epoch:03d}.npz",
        **{key: np.asarray(value) for key, value in trace.items()},
    )


def annotate_mnist_jepa_summary(
    result: AutoencoderExperimentResult,
    baselines: Dict[str, float],
) -> None:
    summary_path = result.paths.metrics / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    final_loss = float(summary.get("final_eval_loss", summary.get("best_loss")))
    zero_loss = float(baselines["zero_embedding_mse"])
    mean_loss = float(baselines["train_mean_embedding_mse"])
    summary["jepa"] = {
        **baselines,
        "beats_zero_embedding": final_loss < zero_loss,
        "beats_train_mean_embedding": final_loss < mean_loss,
        "margin_vs_zero_embedding": zero_loss - final_loss,
        "margin_vs_train_mean_embedding": mean_loss - final_loss,
    }
    write_json(summary_path, summary)
    result.metrics.update(summary)


def _checkpoint_hyperparams(config: MNISTJEPAExperimentConfig) -> Dict[str, object]:
    return {
        "experiment_family": "mnist_jepa",
        "model_family": config.model_family,
        "hidden_dim": config.hidden_dim,
        "latent_dim": config.latent_dim,
        "embedding_dim": config.embedding_dim,
        "patch_shape": list(config.patch_shape),
        "target_encoder": config.target_encoder,
        "input_dim": _mnist_input_patch_dim(config),
    }


def run_mnist_jepa_experiment(
    config: MNISTJEPAExperimentConfig,
) -> AutoencoderExperimentResult:
    """Run MNIST JEPA-lite hidden patch representation prediction."""

    if config.run.mode != "train":
        raise ValueError("mnist_jepa currently supports train mode only")
    if config.corruption_mode == "none":
        raise ValueError("mnist_jepa requires a corruption mode")

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
    train_corrupted, train_visibility = corrupt_mnist_images_with_visibility(
        train_images,
        mode=config.corruption_mode,
        patch_shape=config.patch_shape,
        fraction=config.corruption_fraction,
        seed=corruption_seed,
        mask_value=config.corruption_mask_value,
    )
    eval_corrupted, eval_visibility = corrupt_mnist_images_with_visibility(
        eval_images,
        mode=config.corruption_mode,
        patch_shape=config.patch_shape,
        fraction=config.corruption_fraction,
        seed=corruption_seed + 1,
        mask_value=config.corruption_mask_value,
    )

    train_inputs = _select_model_inputs(train_corrupted, train_visibility, config)
    eval_inputs = _select_model_inputs(eval_corrupted, eval_visibility, config)
    train_targets = compute_patch_embeddings(
        train_images,
        patch_shape=config.patch_shape,
        embedding_dim=config.embedding_dim,
        target_encoder=config.target_encoder,
    )
    eval_targets = compute_patch_embeddings(
        eval_images,
        patch_shape=config.patch_shape,
        embedding_dim=config.embedding_dim,
        target_encoder=config.target_encoder,
    )
    train_weights = compute_hidden_patch_weights(
        train_visibility,
        patch_shape=config.patch_shape,
        embedding_dim=config.embedding_dim,
    )
    eval_weights = compute_hidden_patch_weights(
        eval_visibility,
        patch_shape=config.patch_shape,
        embedding_dim=config.embedding_dim,
    )
    baselines = compute_jepa_baselines(train_targets, eval_targets, eval_weights)

    model = build_mnist_jepa_model(config, jax.random.PRNGKey(config.run.seed))
    result = train_autoencoder(
        model,
        train_inputs,
        eval_inputs,
        config.run,
        sample_axis=0,
        train_targets=train_targets,
        eval_targets=eval_targets,
        train_loss_weights=train_weights,
        eval_loss_weights=eval_weights,
        task_config=asdict(config),
        checkpoint_hyperparams=_checkpoint_hyperparams(config),
        artifact_callback=save_mnist_jepa_artifacts,
        logger=logger,
    )
    annotate_mnist_jepa_summary(result, baselines)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OscNet MNIST JEPA-lite representation benchmark."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reference/mnist_jepa"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--artifact-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument(
        "--model-family",
        choices=[
            "feedforward_patch",
            "recurrent_conv",
            "conv_lstm",
            "winfree_rate_phase",
            "winfree_global_rate_phase",
        ],
        default="winfree_global_rate_phase",
    )
    parser.add_argument("--target-encoder", choices=["dct_lowfreq"], default="dct_lowfreq")
    parser.add_argument(
        "--corruption-mode",
        choices=["patch_mask", "block_occlusion"],
        default="block_occlusion",
    )
    parser.add_argument("--corruption-fraction", type=float, default=0.5)
    parser.add_argument("--corruption-seed", type=int, default=None)
    parser.add_argument("--corruption-mask-value", type=float, default=0.0)
    parser.add_argument(
        "--corruption-input-mode",
        choices=["image", "image_plus_mask"],
        default="image_plus_mask",
    )
    parser.add_argument("--recurrent-conv-steps", type=int, default=8)
    parser.add_argument("--recurrent-conv-kernel-size", type=int, default=3)
    parser.add_argument("--recurrent-conv-residual-strength", type=float, default=0.5)
    parser.add_argument("--conv-lstm-steps", type=int, default=8)
    parser.add_argument("--conv-lstm-kernel-size", type=int, default=3)
    parser.add_argument("--conv-lstm-forget-bias", type=float, default=1.0)
    parser.add_argument("--winfree-steps", type=int, default=8)
    parser.add_argument("--winfree-gamma", type=float, default=0.1)
    parser.add_argument("--winfree-global-gamma", type=float, default=0.05)
    parser.add_argument("--winfree-coupling-strength", type=float, default=1.0)
    parser.add_argument(
        "--winfree-coupling-mode",
        choices=["matrix", "conv", "adaptive", "conv_adaptive", "conv_matrix"],
        default="conv",
    )
    parser.add_argument("--winfree-coupling-kernel-size", type=int, default=3)
    parser.add_argument("--winfree-si-func", choices=["trig", "mlp"], default="mlp")
    parser.add_argument("--winfree-rate-kernel-size", type=int, default=3)
    parser.add_argument("--winfree-rate-update-rate", type=float, default=0.5)
    parser.add_argument("--winfree-rate-gate-strength", type=float, default=1.0)
    parser.add_argument("--winfree-global-gate-strength", type=float, default=0.5)
    parser.add_argument(
        "--data-source",
        choices=["tfds", "idx", "synthetic"],
        default="idx",
    )
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--eval-limit", type=int, default=1_000)
    return parser


def config_from_args(args: argparse.Namespace) -> MNISTJEPAExperimentConfig:
    run = AutoencoderExperimentConfig(
        name="mnist_jepa",
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        artifact_every=args.artifact_every,
    )
    return MNISTJEPAExperimentConfig(
        run=run,
        model_family=args.model_family,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        patch_shape=(args.patch_size, args.patch_size),
        target_encoder=args.target_encoder,
        corruption_mode=args.corruption_mode,
        corruption_fraction=args.corruption_fraction,
        corruption_seed=args.corruption_seed,
        corruption_mask_value=args.corruption_mask_value,
        corruption_input_mode=args.corruption_input_mode,
        recurrent_conv_steps=args.recurrent_conv_steps,
        recurrent_conv_kernel_size=args.recurrent_conv_kernel_size,
        recurrent_conv_residual_strength=args.recurrent_conv_residual_strength,
        conv_lstm_steps=args.conv_lstm_steps,
        conv_lstm_kernel_size=args.conv_lstm_kernel_size,
        conv_lstm_forget_bias=args.conv_lstm_forget_bias,
        winfree_steps=args.winfree_steps,
        winfree_gamma=args.winfree_gamma,
        winfree_global_gamma=args.winfree_global_gamma,
        winfree_coupling_strength=args.winfree_coupling_strength,
        winfree_coupling_mode=args.winfree_coupling_mode,
        winfree_coupling_kernel_size=args.winfree_coupling_kernel_size,
        winfree_si_func=args.winfree_si_func,
        winfree_rate_kernel_size=args.winfree_rate_kernel_size,
        winfree_rate_update_rate=args.winfree_rate_update_rate,
        winfree_rate_gate_strength=args.winfree_rate_gate_strength,
        winfree_global_gate_strength=args.winfree_global_gate_strength,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )


def main(argv: Optional[list[str]] = None) -> AutoencoderExperimentResult:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_mnist_jepa_experiment(config_from_args(args))


if __name__ == "__main__":
    main()
