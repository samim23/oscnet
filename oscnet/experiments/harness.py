"""Shared training and artifact harness for reference autoencoder experiments."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from oscnet.utils import save_equinox_checkpoint

Array = jnp.ndarray
ArtifactBatch = Array | Tuple[Array, Array]
ArtifactCallback = Callable[
    [eqx.Module, Optional[ArtifactBatch], "ExperimentPaths", int, Dict[str, Any]],
    None,
]
PredictionTransform = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class AutoencoderExperimentConfig:
    """Generic run controls shared by reference autoencoder experiments."""

    name: str
    output_dir: Path = Path("outputs/reference")
    mode: str = "train"
    seed: int = 42
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    output_bounds_penalty: float = 0.0
    latent_variance_weight: float = 0.0
    latent_std_floor: float = 1.0
    eval_every: int = 1
    checkpoint_every: int = 5
    artifact_every: int = 5
    shuffle: bool = True
    save_best: bool = True


@dataclass(frozen=True)
class ExperimentPaths:
    """Directory layout for one experiment run."""

    root: Path
    checkpoints: Path
    metrics: Path
    plots: Path
    traces: Path
    artifacts: Path


@dataclass
class AutoencoderExperimentResult:
    """Result bundle returned by train/eval helpers."""

    model: eqx.Module
    metrics: Dict[str, Any]
    paths: ExperimentPaths
    checkpoint_paths: List[str]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def prepare_experiment_paths(
    config: AutoencoderExperimentConfig,
    task_config: Optional[Dict[str, Any]] = None,
) -> ExperimentPaths:
    """Create the canonical directory layout and persist the run config."""

    root = Path(config.output_dir)
    paths = ExperimentPaths(
        root=root,
        checkpoints=root / "checkpoints",
        metrics=root / "metrics",
        plots=root / "plots",
        traces=root / "traces",
        artifacts=root / "artifacts",
    )
    for path in asdict(paths).values():
        Path(path).mkdir(parents=True, exist_ok=True)

    write_json(
        root / "config.json",
        {
            "experiment": config,
            "task": task_config or {},
            "created_at_unix": time.time(),
        },
    )
    return paths


def _target_like_prediction(batch: Array, prediction: Array) -> Array:
    if batch.shape == prediction.shape:
        return batch
    if batch.size == prediction.size:
        return jnp.reshape(batch, prediction.shape)
    raise ValueError(
        f"Cannot compare prediction shape {prediction.shape} to batch shape {batch.shape}"
    )


def _reconstruction_loss(
    prediction: Array,
    targets: Array,
    loss_weights: Optional[Array] = None,
) -> Array:
    target = _target_like_prediction(targets, prediction)
    squared_error = (prediction - target) ** 2
    if loss_weights is None:
        return jnp.mean(squared_error)

    weights = _target_like_prediction(loss_weights, prediction)
    weighted_error = squared_error * weights
    return jnp.sum(weighted_error) / jnp.maximum(jnp.sum(weights), 1e-8)


def _tree_norm(tree: Any) -> Array:
    if hasattr(optax, "tree") and hasattr(optax.tree, "norm"):
        return optax.tree.norm(tree)
    return optax.global_norm(tree)


def _latent_variance_loss(model: eqx.Module, batch: Array, std_floor: float) -> Array:
    latent = model.encode(batch)
    latent = jnp.reshape(latent, (latent.shape[0], -1))
    std = jnp.sqrt(jnp.var(latent, axis=0) + 1e-4)
    return jnp.mean(jnp.maximum(float(std_floor) - std, 0.0) ** 2)


@eqx.filter_jit
def _train_step(
    model: eqx.Module,
    opt_state: Any,
    inputs: Array,
    targets: Array,
    loss_weights: Optional[Array],
    prediction_transform: Optional[PredictionTransform],
    optimizer: optax.GradientTransformation,
    max_grad_norm: float,
    output_bounds_penalty: float,
    latent_variance_weight: float,
    latent_std_floor: float,
):
    def loss_fn(current_model):
        prediction = current_model(inputs)
        if prediction_transform is not None:
            prediction = prediction_transform(inputs, prediction)
        reconstruction_loss = _reconstruction_loss(
            prediction,
            targets,
            loss_weights,
        )
        lower_overshoot = jnp.maximum(-prediction, 0.0)
        upper_overshoot = jnp.maximum(prediction - 1.0, 0.0)
        bounds_loss = jnp.mean(lower_overshoot**2 + upper_overshoot**2)
        latent_loss = 0.0
        if latent_variance_weight > 0.0:
            latent_loss = _latent_variance_loss(
                current_model,
                inputs,
                latent_std_floor,
            )
        return (
            reconstruction_loss
            + output_bounds_penalty * bounds_loss
            + latent_variance_weight * latent_loss
        )

    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model)
    grad_norm = _tree_norm(grads)
    clip = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
    grads = jax.tree.map(lambda grad: grad * clip, grads)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, grad_norm


@eqx.filter_jit
def _eval_loss(
    model: eqx.Module,
    inputs: Array,
    targets: Array,
    loss_weights: Optional[Array] = None,
    prediction_transform: Optional[PredictionTransform] = None,
):
    prediction = model(inputs)
    if prediction_transform is not None:
        prediction = prediction_transform(inputs, prediction)
    return _reconstruction_loss(prediction, targets, loss_weights)


def iter_sample_batches(
    data: Array,
    batch_size: int,
    key: jax.random.PRNGKey,
    *,
    sample_axis: int = 0,
    shuffle: bool = True,
    drop_remainder: bool = False,
) -> Iterable[Array]:
    """Yield batches while preserving non-sample axes, such as time-major audio."""

    n_samples = int(data.shape[sample_axis])
    indices = jnp.arange(n_samples)
    if shuffle:
        indices = jax.random.permutation(key, n_samples)

    if drop_remainder:
        usable = (n_samples // batch_size) * batch_size
        indices = indices[:usable]

    for start in range(0, int(indices.shape[0]), batch_size):
        batch_indices = indices[start : start + batch_size]
        if batch_indices.size == 0:
            continue
        yield jnp.take(data, batch_indices, axis=sample_axis)


def iter_input_target_batches(
    inputs: Array,
    targets: Optional[Array],
    batch_size: int,
    key: jax.random.PRNGKey,
    *,
    sample_axis: int = 0,
    shuffle: bool = True,
    drop_remainder: bool = False,
) -> Iterable[Tuple[Array, Array]]:
    """Yield aligned input/target batches, defaulting targets to inputs."""

    n_samples = int(inputs.shape[sample_axis])
    if targets is not None and int(targets.shape[sample_axis]) != n_samples:
        raise ValueError("targets must have the same sample count as inputs")

    indices = jnp.arange(n_samples)
    if shuffle:
        indices = jax.random.permutation(key, n_samples)

    if drop_remainder:
        usable = (n_samples // batch_size) * batch_size
        indices = indices[:usable]

    for start in range(0, int(indices.shape[0]), batch_size):
        batch_indices = indices[start : start + batch_size]
        if batch_indices.size == 0:
            continue
        input_batch = jnp.take(inputs, batch_indices, axis=sample_axis)
        target_source = inputs if targets is None else targets
        target_batch = jnp.take(target_source, batch_indices, axis=sample_axis)
        yield input_batch, target_batch


def iter_weighted_input_target_batches(
    inputs: Array,
    targets: Optional[Array],
    loss_weights: Optional[Array],
    batch_size: int,
    key: jax.random.PRNGKey,
    *,
    sample_axis: int = 0,
    shuffle: bool = True,
    drop_remainder: bool = False,
) -> Iterable[Tuple[Array, Array, Optional[Array]]]:
    """Yield aligned input/target/weight batches, defaulting targets to inputs."""

    n_samples = int(inputs.shape[sample_axis])
    if targets is not None and int(targets.shape[sample_axis]) != n_samples:
        raise ValueError("targets must have the same sample count as inputs")
    if loss_weights is not None and int(loss_weights.shape[sample_axis]) != n_samples:
        raise ValueError("loss_weights must have the same sample count as inputs")

    indices = jnp.arange(n_samples)
    if shuffle:
        indices = jax.random.permutation(key, n_samples)

    if drop_remainder:
        usable = (n_samples // batch_size) * batch_size
        indices = indices[:usable]

    for start in range(0, int(indices.shape[0]), batch_size):
        batch_indices = indices[start : start + batch_size]
        if batch_indices.size == 0:
            continue
        input_batch = jnp.take(inputs, batch_indices, axis=sample_axis)
        target_source = inputs if targets is None else targets
        target_batch = jnp.take(target_source, batch_indices, axis=sample_axis)
        weight_batch = (
            None
            if loss_weights is None
            else jnp.take(loss_weights, batch_indices, axis=sample_axis)
        )
        yield input_batch, target_batch, weight_batch


def evaluate_autoencoder(
    model: eqx.Module,
    data: Optional[Array],
    *,
    batch_size: int,
    sample_axis: int = 0,
    targets: Optional[Array] = None,
    loss_weights: Optional[Array] = None,
    prediction_transform: Optional[PredictionTransform] = None,
) -> Optional[float]:
    """Evaluate mean reconstruction loss over a dataset."""

    if data is None or int(data.shape[sample_axis]) == 0:
        return None

    losses = []
    for inputs, target_batch, weight_batch in iter_weighted_input_target_batches(
        data,
        targets,
        loss_weights,
        batch_size,
        jax.random.PRNGKey(0),
        sample_axis=sample_axis,
        shuffle=False,
        drop_remainder=False,
    ):
        losses.append(
            float(
                _eval_loss(
                    model,
                    inputs,
                    target_batch,
                    weight_batch,
                    prediction_transform,
                )
            )
        )
    if not losses:
        return None
    return float(np.mean(losses))


def collect_sequence_state_trace(
    sequence_layer: eqx.Module,
    inputs: Array,
    *,
    use_phase_init: bool = False,
) -> Dict[str, Array]:
    """Collect output, position, and velocity traces from a sequence layer."""

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (time, batch, features)")

    state = sequence_layer.cell.initial_state(inputs.shape[1], use_phase_init)

    def scan_fn(carry, x_t):
        output, new_state = sequence_layer.cell(x_t, carry)
        x_state, v_state = new_state
        return new_state, (output, x_state, v_state)

    final_state, (outputs, positions, velocities) = jax.lax.scan(
        scan_fn, state, inputs
    )
    final_position, final_velocity = final_state
    return {
        "outputs": outputs,
        "positions": positions,
        "velocities": velocities,
        "final_position": final_position,
        "final_velocity": final_velocity,
    }


def save_loss_curve(metrics: Dict[str, Any], path: Path) -> None:
    """Save a comparable train/eval loss curve."""

    epochs = metrics.get("epoch", [])
    train_loss = metrics.get("train_loss", [])
    eval_loss = metrics.get("eval_loss", [])

    if not epochs:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if any(value is not None for value in train_loss):
        valid_epochs = [
            epoch for epoch, value in zip(epochs, train_loss) if value is not None
        ]
        valid_losses = [value for value in train_loss if value is not None]
        ax.plot(valid_epochs, valid_losses, label="train")
    if any(value is not None for value in eval_loss):
        valid_epochs = [
            epoch for epoch, value in zip(epochs, eval_loss) if value is not None
        ]
        valid_losses = [value for value in eval_loss if value is not None]
        ax.plot(valid_epochs, valid_losses, label="eval")
    if not ax.lines:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics_csv(metrics: Dict[str, Any], path: Path) -> None:
    """Save epoch metrics in a spreadsheet-friendly format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    epochs = metrics.get("epoch", [])
    for i, epoch in enumerate(epochs):
        rows.append(
            {
                "epoch": epoch,
                "train_loss": metrics.get("train_loss", [None] * len(epochs))[i],
                "eval_loss": metrics.get("eval_loss", [None] * len(epochs))[i],
                "grad_norm": metrics.get("grad_norm", [None] * len(epochs))[i],
                "learning_rate": metrics.get("learning_rate", [None] * len(epochs))[i],
                "epoch_seconds": metrics.get("epoch_seconds", [None] * len(epochs))[i],
            }
        )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "eval_loss",
                "grad_norm",
                "learning_rate",
                "epoch_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_metrics_bundle(metrics: Dict[str, Any], paths: ExperimentPaths) -> None:
    write_json(paths.metrics / "history.json", metrics)
    save_metrics_csv(metrics, paths.metrics / "history.csv")
    save_loss_curve(metrics, paths.plots / "loss_curve.png")


def _first_batch(
    data: Optional[Array],
    batch_size: int,
    sample_axis: int,
    targets: Optional[Array] = None,
) -> Optional[ArtifactBatch]:
    if data is None or int(data.shape[sample_axis]) == 0:
        return None
    input_batch, target_batch = next(
        iter_input_target_batches(
            data,
            targets,
            min(batch_size, int(data.shape[sample_axis])),
            jax.random.PRNGKey(0),
            sample_axis=sample_axis,
            shuffle=False,
        )
    )
    if targets is None:
        return input_batch
    return input_batch, target_batch


def train_autoencoder(
    model: eqx.Module,
    train_data: Array,
    eval_data: Optional[Array],
    config: AutoencoderExperimentConfig,
    *,
    sample_axis: int = 0,
    train_targets: Optional[Array] = None,
    eval_targets: Optional[Array] = None,
    train_loss_weights: Optional[Array] = None,
    eval_loss_weights: Optional[Array] = None,
    prediction_transform: Optional[PredictionTransform] = None,
    task_config: Optional[Dict[str, Any]] = None,
    checkpoint_hyperparams: Optional[Dict[str, Any]] = None,
    artifact_callback: Optional[ArtifactCallback] = None,
    logger: Optional[logging.Logger] = None,
    start_epoch: int = 0,
    resume_from_checkpoint: Optional[Path] = None,
    initial_best_loss: Optional[float] = None,
    initial_best_epoch: Optional[int] = None,
) -> AutoencoderExperimentResult:
    """Train an autoencoder and save comparable reference artifacts."""

    if config.mode != "train":
        raise ValueError("train_autoencoder requires config.mode == 'train'")
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if start_epoch < 0:
        raise ValueError("start_epoch must be >= 0")

    logger = logger or logging.getLogger(__name__)
    paths = prepare_experiment_paths(config, task_config)
    optimizer = optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    key = jax.random.fold_in(jax.random.PRNGKey(config.seed), int(start_epoch))

    metrics: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "eval_loss": [],
        "grad_norm": [],
        "learning_rate": [],
        "epoch_seconds": [],
        "best_eval_loss": initial_best_loss,
        "best_epoch": initial_best_epoch,
    }
    checkpoint_paths: List[str] = []
    best_loss = float(initial_best_loss) if initial_best_loss is not None else float("inf")

    def record_checkpoint(path: str) -> None:
        if path not in checkpoint_paths:
            checkpoint_paths.append(path)

    for local_epoch in range(1, config.epochs + 1):
        epoch = start_epoch + local_epoch
        epoch_start = time.time()
        key, epoch_key = jax.random.split(key)
        losses = []
        grad_norms = []

        for inputs, targets, loss_weights in iter_weighted_input_target_batches(
            train_data,
            train_targets,
            train_loss_weights,
            config.batch_size,
            epoch_key,
            sample_axis=sample_axis,
            shuffle=config.shuffle,
            drop_remainder=False,
        ):
            model, opt_state, loss_value, grad_norm = _train_step(
                model,
                opt_state,
                inputs,
                targets,
                loss_weights,
                prediction_transform,
                optimizer,
                config.max_grad_norm,
                config.output_bounds_penalty,
                config.latent_variance_weight,
                config.latent_std_floor,
            )
            losses.append(float(loss_value))
            grad_norms.append(float(grad_norm))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        grad_norm = float(np.mean(grad_norms)) if grad_norms else float("nan")
        eval_loss = None
        if eval_data is not None and epoch % config.eval_every == 0:
            eval_loss = evaluate_autoencoder(
                model,
                eval_data,
                batch_size=config.batch_size,
                sample_axis=sample_axis,
                targets=eval_targets,
                loss_weights=eval_loss_weights,
                prediction_transform=prediction_transform,
            )

        candidate_loss = eval_loss if eval_loss is not None else train_loss
        is_best = bool(candidate_loss < best_loss)
        if is_best:
            best_loss = candidate_loss
            metrics["best_eval_loss"] = candidate_loss
            metrics["best_epoch"] = epoch

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(train_loss)
        metrics["eval_loss"].append(eval_loss)
        metrics["grad_norm"].append(grad_norm)
        metrics["learning_rate"].append(config.learning_rate)
        metrics["epoch_seconds"].append(float(time.time() - epoch_start))

        logger.info(
            "epoch=%s train_loss=%.6f eval_loss=%s grad_norm=%.6f",
            epoch,
            train_loss,
            "none" if eval_loss is None else f"{eval_loss:.6f}",
            grad_norm,
        )

        if config.save_best and is_best:
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
                hyperparams=checkpoint_hyperparams or {},
                is_best=True,
            )
            record_checkpoint(checkpoint_path)

        should_checkpoint = (
            local_epoch == config.epochs or epoch % config.checkpoint_every == 0
        )
        if should_checkpoint:
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
                hyperparams=checkpoint_hyperparams or {},
                is_best=False,
            )
            record_checkpoint(checkpoint_path)

        _save_metrics_bundle(metrics, paths)

        should_save_artifacts = (
            local_epoch == config.epochs or epoch % config.artifact_every == 0
        )
        if artifact_callback is not None and should_save_artifacts:
            artifact_source = eval_data if eval_data is not None else train_data
            artifact_targets = eval_targets if eval_data is not None else train_targets
            artifact_batch = _first_batch(
                artifact_source,
                config.batch_size,
                sample_axis,
                artifact_targets,
            )
            artifact_callback(model, artifact_batch, paths, epoch, metrics)

    write_json(
        paths.metrics / "summary.json",
        {
            "final_train_loss": metrics["train_loss"][-1],
            "final_eval_loss": metrics["eval_loss"][-1],
            "best_loss": best_loss,
            "best_epoch": metrics["best_epoch"],
            "epochs": config.epochs,
            "start_epoch": start_epoch,
            "final_epoch": metrics["epoch"][-1],
            "resume_from_checkpoint": (
                str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
            ),
            "checkpoints": checkpoint_paths,
        },
    )

    return AutoencoderExperimentResult(
        model=model,
        metrics=metrics,
        paths=paths,
        checkpoint_paths=checkpoint_paths,
    )


def run_eval_only(
    model: eqx.Module,
    eval_data: Array,
    config: AutoencoderExperimentConfig,
    *,
    sample_axis: int = 0,
    eval_targets: Optional[Array] = None,
    eval_loss_weights: Optional[Array] = None,
    prediction_transform: Optional[PredictionTransform] = None,
    task_config: Optional[Dict[str, Any]] = None,
    artifact_callback: Optional[ArtifactCallback] = None,
) -> AutoencoderExperimentResult:
    """Evaluate a model, saving the same metrics/artifact layout as training."""

    paths = prepare_experiment_paths(config, task_config)
    eval_loss = evaluate_autoencoder(
        model,
        eval_data,
        batch_size=config.batch_size,
        sample_axis=sample_axis,
        targets=eval_targets,
        loss_weights=eval_loss_weights,
        prediction_transform=prediction_transform,
    )
    metrics = {
        "epoch": [0],
        "train_loss": [None],
        "eval_loss": [eval_loss],
        "grad_norm": [None],
        "learning_rate": [None],
        "epoch_seconds": [0.0],
    }
    _save_metrics_bundle(metrics, paths)
    write_json(paths.metrics / "summary.json", {"eval_loss": eval_loss})

    if artifact_callback is not None:
        artifact_batch = _first_batch(
            eval_data,
            config.batch_size,
            sample_axis,
            eval_targets,
        )
        artifact_callback(model, artifact_batch, paths, 0, metrics)

    return AutoencoderExperimentResult(
        model=model,
        metrics=metrics,
        paths=paths,
        checkpoint_paths=[],
    )


__all__ = [
    "Array",
    "ArtifactCallback",
    "AutoencoderExperimentConfig",
    "AutoencoderExperimentResult",
    "ExperimentPaths",
    "PredictionTransform",
    "collect_sequence_state_trace",
    "evaluate_autoencoder",
    "iter_input_target_batches",
    "iter_sample_batches",
    "prepare_experiment_paths",
    "run_eval_only",
    "save_loss_curve",
    "train_autoencoder",
    "write_json",
]
