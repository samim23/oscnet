"""MNIST-native phase VAE experiments.

This experiment is deliberately easier than the unpaired oscillator generator:
it gives the model the same kind of paired reconstruction signal that ordinary
MNIST VAEs use, while still routing the latent code through oscillator phase
dynamics before decoding.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    iter_sample_batches,
    prepare_experiment_paths,
    save_loss_curve,
    save_metrics_csv,
    write_json,
)
from oscnet.experiments.mnist_autoencoder import load_mnist_data
from oscnet.models import KuramotoPhaseVAE
from oscnet.utils import save_equinox_checkpoint

Array = jnp.ndarray


@dataclass(frozen=True)
class MNISTPhaseVAEExperimentConfig:
    """Task-specific controls for MNIST phase-VAE training."""

    run: AutoencoderExperimentConfig
    model_family: str = "phase_vae"
    latent_dim: int = 32
    hidden_dim: int = 256
    encoder_depth: int = 2
    decoder_depth: int = 2
    steps: int = 4
    dt: float = 0.1
    coupling_strength: float = 1.0
    omega_scale: float = 0.2
    coupling_init_scale: float = 0.05
    phase_readout_mode: str = "absolute"
    kl_weight: float = 1e-3
    reconstruction_loss: str = "bce"
    eval_sample_count: int = 64
    data_source: str = "idx"
    train_limit: Optional[int] = 10_000
    eval_limit: Optional[int] = 1_000


def _logger() -> logging.Logger:
    logger = logging.getLogger("oscnet.experiments.mnist_phase_vae")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def build_mnist_phase_vae_model(
    config: MNISTPhaseVAEExperimentConfig,
    key: jax.random.PRNGKey,
) -> KuramotoPhaseVAE:
    """Build the requested phase-VAE model or control."""

    if config.model_family == "phase_vae":
        steps = config.steps
        train_dynamics = True
    elif config.model_family == "frozen_phase_vae":
        steps = config.steps
        train_dynamics = False
    elif config.model_family == "phase_vae_no_dynamics":
        steps = 0
        train_dynamics = False
    else:
        raise ValueError(
            "model_family must be 'phase_vae', 'frozen_phase_vae', "
            "or 'phase_vae_no_dynamics'"
        )
    return KuramotoPhaseVAE(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        encoder_depth=config.encoder_depth,
        decoder_depth=config.decoder_depth,
        steps=steps,
        dt=config.dt,
        coupling_strength=config.coupling_strength,
        omega_scale=config.omega_scale,
        coupling_init_scale=config.coupling_init_scale,
        train_dynamics=train_dynamics,
        phase_readout_mode=config.phase_readout_mode,
        output_activation="sigmoid",
        key=key,
    )


def _checkpoint_hyperparams(config: MNISTPhaseVAEExperimentConfig) -> Dict[str, Any]:
    return {
        "experiment": "mnist_phase_vae",
        "model_family": config.model_family,
        "latent_dim": config.latent_dim,
        "hidden_dim": config.hidden_dim,
        "encoder_depth": config.encoder_depth,
        "decoder_depth": config.decoder_depth,
        "steps": config.steps,
        "dt": config.dt,
        "coupling_strength": config.coupling_strength,
        "omega_scale": config.omega_scale,
        "coupling_init_scale": config.coupling_init_scale,
        "phase_readout_mode": config.phase_readout_mode,
        "kl_weight": config.kl_weight,
        "reconstruction_loss": config.reconstruction_loss,
    }


def bernoulli_reconstruction_loss(target: Array, reconstruction: Array) -> Array:
    """Mean binary cross-entropy over pixels."""

    eps = jnp.asarray(1e-6, dtype=reconstruction.dtype)
    clipped = jnp.clip(reconstruction, eps, 1.0 - eps)
    return -jnp.mean(
        target * jnp.log(clipped) + (1.0 - target) * jnp.log(1.0 - clipped)
    )


def reconstruction_loss(
    target: Array,
    reconstruction: Array,
    *,
    mode: str,
) -> Array:
    """Compute the configured reconstruction loss."""

    if mode == "bce":
        return bernoulli_reconstruction_loss(target, reconstruction)
    if mode == "mse":
        return jnp.mean((target - reconstruction) ** 2)
    raise ValueError("reconstruction_loss must be 'bce' or 'mse'")


def gaussian_kl_loss(mu: Array, logvar: Array) -> Array:
    """Average KL divergence to a standard Gaussian prior."""

    per_sample = -0.5 * jnp.sum(
        1.0 + logvar - mu**2 - jnp.exp(logvar),
        axis=-1,
    )
    return jnp.mean(per_sample)


def phase_vae_loss(
    model: KuramotoPhaseVAE,
    images: Array,
    key: jax.random.PRNGKey,
    *,
    kl_weight: float,
    reconstruction_mode: str,
    deterministic: bool = False,
) -> Tuple[Array, Dict[str, Array]]:
    """Return total phase-VAE loss and diagnostics."""

    reconstruction, latent = model(
        images,
        key,
        deterministic=deterministic,
        return_latent=True,
    )
    recon = reconstruction_loss(
        images,
        reconstruction,
        mode=reconstruction_mode,
    )
    mse = jnp.mean((images - reconstruction) ** 2)
    kl = gaussian_kl_loss(latent["mu"], latent["logvar"])
    total = recon + float(kl_weight) * kl
    return total, {
        "reconstruction_loss": recon,
        "mse": mse,
        "kl_loss": kl,
        "total_loss": total,
    }


def _tree_norm(tree: Any) -> Array:
    if hasattr(optax, "tree") and hasattr(optax.tree, "norm"):
        return optax.tree.norm(tree)
    return optax.global_norm(tree)


@eqx.filter_jit
def _train_step(
    model: KuramotoPhaseVAE,
    opt_state: Any,
    images: Array,
    sample_key: jax.random.PRNGKey,
    optimizer: optax.GradientTransformation,
    max_grad_norm: float,
    kl_weight: float,
    reconstruction_mode: str,
):
    def loss_fn(current_model):
        return phase_vae_loss(
            current_model,
            images,
            sample_key,
            kl_weight=kl_weight,
            reconstruction_mode=reconstruction_mode,
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
    model: KuramotoPhaseVAE,
    images: Array,
    sample_key: jax.random.PRNGKey,
    kl_weight: float,
    reconstruction_mode: str,
):
    return phase_vae_loss(
        model,
        images,
        sample_key,
        kl_weight=kl_weight,
        reconstruction_mode=reconstruction_mode,
        deterministic=True,
    )


def evaluate_phase_vae(
    model: KuramotoPhaseVAE,
    images: Array,
    *,
    batch_size: int,
    key: jax.random.PRNGKey,
    kl_weight: float,
    reconstruction_mode: str,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate deterministic reconstruction loss over a dataset."""

    losses = []
    recon_losses = []
    mse_losses = []
    kl_losses = []
    for batch_index, batch in enumerate(
        iter_sample_batches(
            images,
            batch_size,
            jax.random.PRNGKey(0),
            shuffle=False,
        )
    ):
        loss, parts = _eval_step(
            model,
            batch,
            jax.random.fold_in(key, batch_index),
            kl_weight,
            reconstruction_mode,
        )
        losses.append(float(loss))
        recon_losses.append(float(parts["reconstruction_loss"]))
        mse_losses.append(float(parts["mse"]))
        kl_losses.append(float(parts["kl_loss"]))
    if not losses:
        return float("nan"), {
            "eval_reconstruction_loss": float("nan"),
            "eval_mse": float("nan"),
            "eval_kl_loss": float("nan"),
        }
    return float(np.mean(losses)), {
        "eval_reconstruction_loss": float(np.mean(recon_losses)),
        "eval_mse": float(np.mean(mse_losses)),
        "eval_kl_loss": float(np.mean(kl_losses)),
    }


def sample_phase_vae_images(
    model: KuramotoPhaseVAE,
    *,
    key: jax.random.PRNGKey,
    sample_count: int,
    batch_size: int,
) -> Array:
    """Generate prior samples in batches."""

    samples = []
    remaining = int(sample_count)
    batch_index = 0
    while remaining > 0:
        current = min(batch_size, remaining)
        samples.append(model.sample(jax.random.fold_in(key, batch_index), current))
        remaining -= current
        batch_index += 1
    return jnp.concatenate(samples, axis=0)


def compute_phase_vae_quality_metrics(
    real_images: Array,
    reconstructions: Array,
    samples: Array,
) -> Dict[str, float]:
    """Compute lightweight reconstruction and sample diagnostics."""

    real = np.asarray(real_images, dtype=np.float32).reshape(real_images.shape[0], -1)
    recon = np.asarray(reconstructions, dtype=np.float32).reshape(
        reconstructions.shape[0],
        -1,
    )
    gen = np.asarray(samples, dtype=np.float32).reshape(samples.shape[0], -1)
    real_for_recon = real[: recon.shape[0]]
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
    return {
        "reconstruction_mse": float(np.mean((real_for_recon - recon) ** 2)),
        "sample_mean": float(np.mean(gen)),
        "sample_std": float(np.std(gen)),
        "sample_pixel_mean_mse": float(
            np.mean((gen.mean(axis=0) - real_for_gen.mean(axis=0)) ** 2)
        ),
        "sample_pixel_std_mse": float(
            np.mean((gen_std - real_std) ** 2)
        ),
        "sample_diversity_ratio": float(
            np.mean(gen_std) / (np.mean(real_std) + 1e-8)
        ),
        "sample_nearest_real_mse": float(np.mean(nearest_real_mse)),
        "real_nearest_real_mse": float(np.mean(np.min(real_pairwise, axis=1))),
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


def save_phase_vae_artifacts(
    model: KuramotoPhaseVAE,
    eval_images: Array,
    paths: ExperimentPaths,
    epoch: int,
    *,
    key: jax.random.PRNGKey,
    sample_count: int,
    batch_size: int,
) -> None:
    """Save reconstructions, prior samples, and latent traces."""

    count = min(int(sample_count), int(eval_images.shape[0]))
    real = eval_images[:count]
    recon, _ = model(
        real,
        key,
        deterministic=True,
        return_latent=True,
    )
    samples = sample_phase_vae_images(
        model,
        key=jax.random.fold_in(key, 1),
        sample_count=count,
        batch_size=batch_size,
    )
    trace = model.collect_trace(real[: min(count, batch_size)], key)

    np.savez(
        paths.artifacts / f"mnist_phase_vae_epoch_{epoch:03d}.npz",
        real=np.asarray(real),
        reconstruction=np.asarray(recon),
        samples=np.asarray(samples),
    )
    np.savez(
        paths.traces / f"mnist_phase_vae_trace_epoch_{epoch:03d}.npz",
        **{name: np.asarray(value) for name, value in trace.items()},
    )
    _save_image_grid(real[: min(count, 64)], paths.artifacts / f"real_epoch_{epoch:03d}.png")
    _save_image_grid(
        recon[: min(count, 64)],
        paths.artifacts / f"reconstruction_epoch_{epoch:03d}.png",
    )
    _save_image_grid(
        samples[: min(count, 64)],
        paths.artifacts / f"samples_epoch_{epoch:03d}.png",
    )


def run_mnist_phase_vae_experiment(
    config: MNISTPhaseVAEExperimentConfig,
) -> AutoencoderExperimentResult:
    """Train/evaluate a MNIST phase VAE."""

    logger = _logger()
    paths = prepare_experiment_paths(config.run, asdict(config))
    train_images, _, eval_images, _ = load_mnist_data(
        source=config.data_source,
        train_limit=config.train_limit,
        eval_limit=config.eval_limit,
        seed=config.run.seed,
    )

    key = jax.random.PRNGKey(config.run.seed)
    model_key, projection_key = jax.random.split(key)
    del projection_key
    model = build_mnist_phase_vae_model(config, model_key)
    optimizer = optax.adamw(
        learning_rate=config.run.learning_rate,
        weight_decay=config.run.weight_decay,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    history: Dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_reconstruction_loss": [],
        "train_mse": [],
        "train_kl_loss": [],
        "eval_loss": [],
        "eval_reconstruction_loss": [],
        "eval_mse": [],
        "eval_kl_loss": [],
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
        recon_losses = []
        mse_losses = []
        kl_losses = []
        grad_norms = []
        for batch_index, batch in enumerate(
            iter_sample_batches(
                train_images,
                config.run.batch_size,
                epoch_key,
                shuffle=config.run.shuffle,
            )
        ):
            model, opt_state, loss, grad_norm, parts = _train_step(
                model,
                opt_state,
                batch,
                jax.random.fold_in(epoch_key, batch_index),
                optimizer,
                config.run.max_grad_norm,
                config.kl_weight,
                config.reconstruction_loss,
            )
            losses.append(float(loss))
            recon_losses.append(float(parts["reconstruction_loss"]))
            mse_losses.append(float(parts["mse"]))
            kl_losses.append(float(parts["kl_loss"]))
            grad_norms.append(float(grad_norm))

        eval_loss, eval_parts = evaluate_phase_vae(
            model,
            eval_images,
            batch_size=config.run.batch_size,
            key=jax.random.fold_in(key, 10_000 + epoch),
            kl_weight=config.kl_weight,
            reconstruction_mode=config.reconstruction_loss,
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
            save_phase_vae_artifacts(
                model,
                eval_images,
                paths,
                epoch,
                key=jax.random.fold_in(key, 20_000 + epoch),
                sample_count=config.eval_sample_count,
                batch_size=config.run.batch_size,
            )

        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(np.mean(losses)))
        history["train_reconstruction_loss"].append(float(np.mean(recon_losses)))
        history["train_mse"].append(float(np.mean(mse_losses)))
        history["train_kl_loss"].append(float(np.mean(kl_losses)))
        history["eval_loss"].append(eval_loss)
        history["eval_reconstruction_loss"].append(eval_parts["eval_reconstruction_loss"])
        history["eval_mse"].append(eval_parts["eval_mse"])
        history["eval_kl_loss"].append(eval_parts["eval_kl_loss"])
        history["grad_norm"].append(float(np.mean(grad_norms)))
        logger.info(
            "epoch=%d train_loss=%.6f eval_loss=%.6f eval_mse=%.6f kl=%.6f time=%.2fs",
            epoch,
            history["train_loss"][-1],
            eval_loss,
            eval_parts["eval_mse"],
            eval_parts["eval_kl_loss"],
            time.time() - epoch_start,
        )

    save_metrics_csv(history, paths.metrics / "history.csv")
    write_json(paths.metrics / "history.json", history)
    save_loss_curve(history, paths.plots / "loss_curve.png")

    count = min(config.eval_sample_count, int(eval_images.shape[0]))
    recon, _ = model(
        eval_images[:count],
        jax.random.fold_in(key, 30_001),
        deterministic=True,
        return_latent=True,
    )
    samples = sample_phase_vae_images(
        model,
        key=jax.random.fold_in(key, 30_002),
        sample_count=count,
        batch_size=config.run.batch_size,
    )
    quality = compute_phase_vae_quality_metrics(eval_images[:count], recon, samples)
    trace = model.collect_trace(
        eval_images[: min(count, config.run.batch_size)],
        jax.random.fold_in(key, 30_003),
    )
    phase_delta = np.angle(
        np.exp(
            1j
            * (
                np.asarray(trace["final_theta"], dtype=np.float32)
                - np.asarray(trace["initial_theta"], dtype=np.float32)
            )
        )
    )
    summary: Dict[str, Any] = {
        "config": asdict(config),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "final_epoch": int(config.run.epochs),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_eval_loss": float(history["eval_loss"][-1]),
        "final_eval_reconstruction_loss": float(history["eval_reconstruction_loss"][-1]),
        "final_eval_mse": float(history["eval_mse"][-1]),
        "final_eval_kl_loss": float(history["eval_kl_loss"][-1]),
        "phase_vae": {
            "model_family": config.model_family,
            "latent_dim": int(model.latent_dim),
            "steps": int(model.steps),
            "train_dynamics": bool(model.train_dynamics),
            "phase_readout_mode": model.phase_readout_mode,
            "kl_weight": float(config.kl_weight),
            "reconstruction_loss": config.reconstruction_loss,
            "phase_mean_abs_displacement": float(np.mean(np.abs(phase_delta))),
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reference/mnist_phase_vae"))
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
        choices=["phase_vae", "frozen_phase_vae", "phase_vae_no_dynamics"],
        default="phase_vae",
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--coupling-strength", type=float, default=1.0)
    parser.add_argument("--omega-scale", type=float, default=0.2)
    parser.add_argument("--coupling-init-scale", type=float, default=0.05)
    parser.add_argument(
        "--phase-readout-mode",
        choices=["absolute", "mean_relative", "ref_oscillator"],
        default="absolute",
    )
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--reconstruction-loss", choices=["bce", "mse"], default="bce")
    parser.add_argument("--eval-sample-count", type=int, default=64)
    parser.add_argument("--data-source", choices=["idx", "synthetic", "tfds"], default="idx")
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--eval-limit", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_config = AutoencoderExperimentConfig(
        name="mnist_phase_vae",
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
    config = MNISTPhaseVAEExperimentConfig(
        run=run_config,
        model_family=args.model_family,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        steps=args.steps,
        dt=args.dt,
        coupling_strength=args.coupling_strength,
        omega_scale=args.omega_scale,
        coupling_init_scale=args.coupling_init_scale,
        phase_readout_mode=args.phase_readout_mode,
        kl_weight=args.kl_weight,
        reconstruction_loss=args.reconstruction_loss,
        eval_sample_count=args.eval_sample_count,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )
    result = run_mnist_phase_vae_experiment(config)
    print(json.dumps(result.metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
