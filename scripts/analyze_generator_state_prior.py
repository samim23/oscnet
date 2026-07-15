"""Probe state-space priors for trained HORN image generators."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from oscnet.analysis.generator_checkpoints import load_generator_checkpoint_model
from oscnet.analysis.generator_state_prior import (
    evaluate_state_prior_sampling_probe,
    save_state_prior_contact_sheet,
    state_prior_to_json,
    write_state_prior_probe_csv,
)
from oscnet.experiments.harness import write_json
from oscnet.experiments.mnist_autoencoder import load_mnist_data
from oscnet.experiments.mnist_generator import (
    compute_class_prototypes,
    train_mnist_feature_classifier,
)


DEFAULT_PRESET = "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030"


def _maybe_train_classifier(
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    eval_images: jnp.ndarray,
    eval_labels: jnp.ndarray,
    *,
    config: Any,
    key: jax.random.PRNGKey,
    epochs: int,
    batch_size: int,
):
    if int(epochs) <= 0:
        return None, {"epochs": 0}
    classifier, history = train_mnist_feature_classifier(
        train_images,
        train_labels,
        eval_images,
        eval_labels,
        key=key,
        num_classes=int(config.num_classes),
        feature_dim=int(config.quality_classifier_dim),
        depth=int(config.quality_classifier_depth),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(config.quality_classifier_learning_rate),
        weight_decay=float(config.quality_classifier_weight_decay),
        max_grad_norm=float(config.run.max_grad_norm),
        classifier_kind=str(config.quality_classifier_kind),
        image_shape=tuple(int(size) for size in config.image_shape),
    )
    return classifier, history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether a trained anchor HORN checkpoint samples better "
            "from a transparent per-class state prior than from white noise."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--prior-rank", type=int, default=32)
    parser.add_argument("--prior-noise-scale", type=float, default=1.0)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--classifier-epochs", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument(
        "--output-prefix",
        default="generator_state_prior_probe",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model, config, checkpoint_hparams = load_generator_checkpoint_model(
        args.checkpoint,
        preset=args.preset,
        seed=args.seed,
    )
    train_images, train_labels, eval_images, eval_labels = load_mnist_data(
        source=config.data_source,
        dataset_name=config.dataset_name,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        seed=args.seed,
    )
    prototypes = compute_class_prototypes(
        train_images,
        train_labels,
        num_classes=int(config.num_classes),
    )
    classifier, classifier_history = _maybe_train_classifier(
        train_images,
        train_labels,
        eval_images,
        eval_labels,
        config=config,
        key=jax.random.PRNGKey(args.seed + 3107),
        epochs=args.classifier_epochs,
        batch_size=args.batch_size,
    )
    metrics, variants, prior = evaluate_state_prior_sampling_probe(
        model,
        train_images=train_images,
        train_labels=train_labels,
        eval_images=eval_images,
        eval_labels=eval_labels,
        prototypes=prototypes,
        classifier=classifier,
        image_shape=tuple(int(size) for size in config.image_shape),
        sample_count=args.sample_count,
        prior_rank=args.prior_rank,
        settle_steps=args.settle_steps,
        batch_size=args.batch_size,
        key=jax.random.PRNGKey(args.seed + 9109),
        prior_noise_scale=args.prior_noise_scale,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}.csv"
    json_path = output_dir / f"{args.output_prefix}.json"
    samples_path = output_dir / f"{args.output_prefix}_samples.npz"
    contact_path = output_dir / f"{args.output_prefix}_contact_sheet.png"

    write_state_prior_probe_csv(metrics, csv_path)
    write_json(
        json_path,
        {
            "checkpoint": str(args.checkpoint),
            "preset": args.preset,
            "seed": args.seed,
            "config": {
                "dataset_name": config.dataset_name,
                "image_shape": config.image_shape,
                "model_family": config.model_family,
                "num_classes": config.num_classes,
                "steps": config.steps,
            },
            "checkpoint_hyperparams": checkpoint_hparams,
            "classifier_history": classifier_history,
            "prior": state_prior_to_json(prior),
            "metrics": metrics,
        },
    )
    np.savez(
        samples_path,
        labels=np.asarray(eval_labels[: int(args.sample_count)]),
        **{name: np.asarray(images) for name, images in variants.items()},
    )
    save_state_prior_contact_sheet(
        variants,
        contact_path,
        image_shape=tuple(int(size) for size in config.image_shape),
    )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {samples_path}")
    print(f"wrote {contact_path}")


if __name__ == "__main__":
    main()

