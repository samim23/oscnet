"""Training loop for MNIST oscillator generator experiments."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from oscnet.experiments.harness import (
    AutoencoderExperimentResult,
    iter_input_target_batches,
    iter_sample_batches,
    prepare_experiment_paths,
    write_json,
)
from oscnet.experiments.mnist_autoencoder import load_mnist_data
from oscnet.utils import save_equinox_checkpoint

from .artifacts import (
    _checkpoint_hyperparams,
    _save_metrics_bundle,
    save_mnist_generator_artifacts,
)
from .builder import build_mnist_generator_model
from .common import Array, _logger, _tree_norm
from .config import MNISTGeneratorExperimentConfig
from .features import (
    FeatureClassifier,
    compute_class_prototypes,
    make_projection_matrix,
    train_mnist_feature_classifier,
)
from .losses import generator_loss
from .metrics import (
    _model_with_steps,
    compute_generator_attractor_robustness,
    compute_generator_quality_metrics,
    compute_generator_settling_metrics,
    compute_generator_success_diagnostics,
    sample_generator_images,
)
from .queue import MNISTDriftQueue

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
    feature_model: Optional[FeatureClassifier],
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
    train_settling_steps: Tuple[int, ...],
    num_classes: int,
):
    def loss_fn(current_model):
        step_depths = (
            train_settling_steps
            if train_settling_steps
            else (int(current_model.steps),)
        )
        losses = []
        parts_by_name: Dict[str, list[Array]] = {}
        for step_index, step_depth in enumerate(step_depths):
            step_model = _model_with_steps(current_model, int(step_depth))
            generated = step_model(
                jax.random.fold_in(sample_key, step_index),
                real_batch.shape[0],
                label_batch,
            )
            loss, parts = generator_loss(
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
            losses.append(loss)
            for name, value in parts.items():
                parts_by_name.setdefault(name, []).append(value)
        mean_parts = {
            name: jnp.mean(jnp.stack(values))
            for name, values in parts_by_name.items()
        }
        return jnp.mean(jnp.stack(losses)), mean_parts

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
    feature_model: Optional[FeatureClassifier],
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
    feature_model: Optional[FeatureClassifier] = None,
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


def run_mnist_generator_experiment(
    config: MNISTGeneratorExperimentConfig,
) -> AutoencoderExperimentResult:
    """Train an Un-0-style oscillator generator on MNIST."""

    if config.run.mode != "train":
        raise ValueError("mnist_generator currently supports train mode only")
    if any(step < 0 for step in config.train_settling_steps):
        raise ValueError("train_settling_steps must be non-negative")
    if any(step < 0 for step in config.settling_steps):
        raise ValueError("settling_steps must be non-negative")
    if config.attractor_variants_per_class < 0:
        raise ValueError("attractor_variants_per_class must be non-negative")
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
        dataset_name=config.dataset_name,
        train_limit=config.train_limit,
        eval_limit=config.eval_limit,
        seed=config.run.seed,
    )
    train_labels = train_labels.astype(jnp.int32)
    eval_labels = eval_labels.astype(jnp.int32)
    expected_image_dim = int(np.prod(tuple(int(size) for size in config.image_shape)))
    if train_images.shape[-1] != expected_image_dim:
        raise ValueError(
            f"dataset {config.dataset_name!r} produced image_dim={train_images.shape[-1]}, "
            f"but config.image_shape={config.image_shape} implies {expected_image_dim}"
        )
    prototypes = None
    if config.conditional:
        prototypes = compute_class_prototypes(
            train_images,
            train_labels,
            num_classes=config.num_classes,
        )
    paths = prepare_experiment_paths(config.run, asdict(config))
    key = jax.random.PRNGKey(config.run.seed)
    key, model_key, projection_key, feature_key, quality_key = jax.random.split(key, 5)
    feature_model: Optional[FeatureClassifier] = None
    feature_classifier_metrics: Dict[str, Any] = {}
    quality_classifier_model: Optional[FeatureClassifier] = None
    quality_classifier_metrics: Dict[str, Any] = {}
    if config.feature_drift_mode == "learned":
        logger.info(
            "training learned feature classifier kind=%s epochs=%s feature_dim=%s",
            config.learned_feature_kind,
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
            classifier_kind=config.learned_feature_kind,
            image_shape=config.image_shape,
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
        if config.quality_classifier_epochs <= 0:
            quality_classifier_model = feature_model
            quality_classifier_metrics = feature_classifier_metrics
    if config.quality_classifier_epochs > 0:
        quality_train_images = train_images
        quality_train_labels = train_labels
        quality_eval_images = eval_images
        quality_eval_labels = eval_labels
        if (
            config.quality_classifier_train_limit is not None
            or config.quality_classifier_eval_limit is not None
        ):
            quality_train_limit = (
                config.quality_classifier_train_limit
                if config.quality_classifier_train_limit is not None
                else config.train_limit
            )
            quality_eval_limit = (
                config.quality_classifier_eval_limit
                if config.quality_classifier_eval_limit is not None
                else config.eval_limit
            )
            (
                quality_train_images,
                quality_train_labels,
                quality_eval_images,
                quality_eval_labels,
            ) = load_mnist_data(
                source=config.data_source,
                dataset_name=config.dataset_name,
                train_limit=quality_train_limit,
                eval_limit=quality_eval_limit,
                seed=config.run.seed,
            )
            quality_train_labels = quality_train_labels.astype(jnp.int32)
            quality_eval_labels = quality_eval_labels.astype(jnp.int32)
        logger.info(
            "training quality classifier kind=%s epochs=%s feature_dim=%s train_n=%s eval_n=%s",
            config.quality_classifier_kind,
            config.quality_classifier_epochs,
            config.quality_classifier_dim,
            int(quality_train_images.shape[0]),
            int(quality_eval_images.shape[0]),
        )
        quality_classifier_model, quality_classifier_metrics = (
            train_mnist_feature_classifier(
                quality_train_images,
                quality_train_labels,
                quality_eval_images,
                quality_eval_labels,
                key=quality_key,
                num_classes=config.num_classes,
                feature_dim=config.quality_classifier_dim,
                depth=config.quality_classifier_depth,
                epochs=config.quality_classifier_epochs,
                batch_size=config.run.batch_size,
                learning_rate=config.quality_classifier_learning_rate,
                weight_decay=config.quality_classifier_weight_decay,
                max_grad_norm=config.run.max_grad_norm,
                classifier_kind=config.quality_classifier_kind,
                image_shape=config.image_shape,
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
                config.train_settling_steps,
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
    settling = compute_generator_settling_metrics(
        model,
        key=jax.random.fold_in(key, 45_000),
        real_images=eval_images[:eval_count],
        sample_count=eval_count,
        batch_size=config.run.batch_size,
        settling_steps=config.settling_steps,
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
    attractor_robustness = compute_generator_attractor_robustness(
        model,
        key=jax.random.fold_in(key, 55_000),
        batch_size=config.run.batch_size,
        variants_per_class=(
            config.attractor_variants_per_class if config.conditional else 0
        ),
        num_classes=config.num_classes,
        classifier=quality_classifier_model,
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
            "dataset_name": config.dataset_name,
            "data_source": config.data_source,
            "image_shape": list(config.image_shape),
            "loss": config.loss_mode,
            "distributional_loss": "sliced_wasserstein_plus_moments_and_pixel_marginals",
            "pixel_drift_weight": config.pixel_drift_weight,
            "feature_drift_weight": config.feature_drift_weight,
            "feature_drift_mode": config.feature_drift_mode,
            "learned_feature_kind": config.learned_feature_kind,
            "learned_feature_epochs": config.learned_feature_epochs,
            "learned_feature_dim": config.learned_feature_dim,
            "learned_feature_depth": config.learned_feature_depth,
            "feature_classifier": feature_classifier_metrics,
            "quality_classifier": quality_classifier_metrics,
            "quality_classifier_epochs": config.quality_classifier_epochs,
            "quality_classifier_kind": config.quality_classifier_kind,
            "quality_classifier_dim": config.quality_classifier_dim,
            "quality_classifier_depth": config.quality_classifier_depth,
            "quality_classifier_train_limit": config.quality_classifier_train_limit,
            "quality_classifier_eval_limit": config.quality_classifier_eval_limit,
            "drift_queue_size": config.drift_queue_size,
            "drift_queue_num_pos": config.drift_queue_num_pos,
            "drift_queue_final_counts": (
                []
                if drift_queue is None
                else [int(value) for value in drift_queue.counts.tolist()]
            ),
            "distributional_weight": config.distributional_weight,
            "class_moment_weight": config.class_moment_weight,
            "prototype_weight": config.prototype_weight,
            "moment_weight": config.moment_weight,
            "pixel_marginal_weight": config.pixel_marginal_weight,
            "num_projections": config.num_projections,
            "drift_gamma": config.drift_gamma,
            "drift_temperatures": list(config.drift_temperatures),
            "train_settling_steps": list(config.train_settling_steps),
            "attractor_variants_per_class": config.attractor_variants_per_class,
            "distributional_not_paired_reconstruction": True,
            "conditional": config.conditional,
            "label_phase_scale": config.label_phase_scale,
            "coupling_profile": model.coupling_profile,
            "coupling_normalization": getattr(
                model,
                "coupling_normalization",
                "none",
            ),
            "coupling_strength": float(model.coupling_strength),
            "main_coupling_strength": float(
                getattr(model, "main_coupling_strength", model.coupling_strength)
            ),
            "coupling_length_scale": float(model.coupling_length_scale),
            "coupling_floor": float(model.coupling_floor),
            "coupling_bias_strength": float(model.coupling_bias_strength),
            "conditioning_strength": float(model.conditioning_strength),
            "conditioning_target_fraction": float(
                model.conditioning_target_fraction
            ),
            "conditioning_target_pattern": model.conditioning_target_pattern,
            "conditioning_target_count": int(
                sum(getattr(model, "conditioning_target_mask", ()))
                if getattr(model, "conditioning_target_mask", ())
                else model.num_oscillators
            ),
            "dynamics_family": str(getattr(model, "dynamics_family", "kuramoto")),
            "horn_frequency": config.horn_frequency,
            "horn_damping": config.horn_damping,
            "horn_nonlinearity": config.horn_nonlinearity,
            "horn_state_bound": config.horn_state_bound,
            "output_feedback_mode": getattr(
                model,
                "output_feedback_mode",
                "none",
            ),
            "output_feedback_strength": float(
                getattr(model, "output_feedback_strength", 0.0)
            ),
            "output_feedback_init_scale": float(
                getattr(model, "output_feedback_init_scale", 0.0)
            ),
            "output_feedback_basis_sigma": float(
                getattr(model, "output_feedback_basis_sigma", 0.0)
            ),
            "num_coarse_oscillators": getattr(model, "num_coarse_oscillators", 0),
            "coarse_coupling_profile": getattr(
                model,
                "coarse_coupling_profile",
                "none",
            ),
            "coarse_coupling_normalization": getattr(
                model,
                "coarse_coupling_normalization",
                "none",
            ),
            "coarse_coupling_length_scale": float(
                getattr(model, "coarse_coupling_length_scale", 0.0)
            ),
            "coarse_to_fine_strength": float(
                getattr(model, "coarse_to_fine_strength", 0.0)
            ),
            "coarse_to_fine_profile": getattr(
                model,
                "coarse_to_fine_profile",
                "none",
            ),
            "coarse_to_fine_normalization": getattr(
                model,
                "coarse_to_fine_normalization",
                "none",
            ),
            "coarse_to_fine_length_scale": float(
                getattr(model, "coarse_to_fine_length_scale", 0.0)
            ),
            "coarse_to_fine_floor": float(
                getattr(model, "coarse_to_fine_floor", 0.0)
            ),
            "coarse_conditioning_strength": float(
                getattr(model, "coarse_conditioning_strength", 0.0)
            ),
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
            "settling": settling,
            "attractor_robustness": attractor_robustness,
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
