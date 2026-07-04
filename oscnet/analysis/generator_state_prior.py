"""State-space prior probes for oscillator image generators."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from oscnet.experiments.mnist_generator.metrics import (
    _model_with_steps,
    compute_generator_quality_metrics,
    sample_generator_images,
)
from oscnet.models.generative.common import _image_hw_channels


@dataclass(frozen=True)
class ClassStatePrior:
    """Per-class low-rank Gaussian prior over concatenated HORN states."""

    means: np.ndarray
    components: np.ndarray
    scales: np.ndarray
    counts: np.ndarray
    rank: int
    state_dim: int

    @property
    def num_classes(self) -> int:
        return int(self.means.shape[0])

    @property
    def oscillator_dim(self) -> int:
        return int(self.state_dim // 2)


def encode_anchor_state_distribution(
    model: eqx.Module,
    images: jnp.ndarray,
    *,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode images into anchor position/velocity states."""

    if not hasattr(model, "encode_image_state"):
        raise ValueError("model does not expose encode_image_state")
    positions = []
    velocities = []
    total = int(images.shape[0])
    for start in range(0, total, int(batch_size)):
        batch = images[start : start + int(batch_size)]
        position, velocity = model.encode_image_state(jnp.asarray(batch))
        positions.append(np.asarray(position, dtype=np.float32))
        velocities.append(np.asarray(velocity, dtype=np.float32))
    position_np = np.concatenate(positions, axis=0)
    velocity_np = np.concatenate(velocities, axis=0)
    states = np.concatenate([position_np, velocity_np], axis=-1)
    return position_np, velocity_np, states


def fit_class_state_prior(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    rank: int = 32,
) -> ClassStatePrior:
    """Fit per-class mean plus low-rank PCA Gaussian state priors."""

    states = np.asarray(states, dtype=np.float32).reshape(states.shape[0], -1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if states.shape[0] != labels.shape[0]:
        raise ValueError("states and labels must have the same first dimension")
    if states.shape[1] % 2 != 0:
        raise ValueError("state dimension must concatenate position and velocity")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    rank = int(max(rank, 0))
    state_dim = int(states.shape[1])
    means = np.zeros((int(num_classes), state_dim), dtype=np.float32)
    components = np.zeros((int(num_classes), rank, state_dim), dtype=np.float32)
    scales = np.zeros((int(num_classes), rank), dtype=np.float32)
    counts = np.zeros((int(num_classes),), dtype=np.int32)

    global_mean = np.mean(states, axis=0)
    global_components = np.zeros((rank, state_dim), dtype=np.float32)
    global_scales = np.zeros((rank,), dtype=np.float32)
    if states.shape[0] >= 2 and rank > 0:
        centered = states - global_mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        take = min(rank, vt.shape[0])
        global_components[:take] = vt[:take].astype(np.float32)
        global_scales[:take] = (
            singular_values[:take] / np.sqrt(float(max(states.shape[0] - 1, 1)))
        ).astype(np.float32)

    for class_index in range(int(num_classes)):
        class_states = states[labels == class_index]
        counts[class_index] = int(class_states.shape[0])
        if class_states.shape[0] == 0:
            means[class_index] = global_mean
            components[class_index] = global_components
            scales[class_index] = global_scales
            continue
        mean = np.mean(class_states, axis=0)
        means[class_index] = mean.astype(np.float32)
        if class_states.shape[0] < 2 or rank == 0:
            continue
        centered = class_states - mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        take = min(rank, vt.shape[0])
        components[class_index, :take] = vt[:take].astype(np.float32)
        scales[class_index, :take] = (
            singular_values[:take]
            / np.sqrt(float(max(class_states.shape[0] - 1, 1)))
        ).astype(np.float32)

    return ClassStatePrior(
        means=means,
        components=components,
        scales=scales,
        counts=counts,
        rank=rank,
        state_dim=state_dim,
    )


def sample_class_state_prior(
    prior: ClassStatePrior,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
    mean_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample position/velocity states from a fitted class prior."""

    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if np.any(labels < 0) or np.any(labels >= prior.num_classes):
        raise ValueError("labels are outside the fitted prior class range")
    states = np.array(prior.means[labels], copy=True)
    if not mean_only and prior.rank > 0 and noise_scale > 0.0:
        class_components = prior.components[labels]
        class_scales = prior.scales[labels] * float(noise_scale)
        coefficients = rng.normal(size=class_scales.shape).astype(np.float32)
        noise = np.einsum(
            "br,brd->bd",
            coefficients * class_scales,
            class_components,
        )
        states = states + noise.astype(np.float32)
    split = prior.oscillator_dim
    return states[:, :split].astype(np.float32), states[:, split:].astype(np.float32)


def generate_from_initial_states(
    model: eqx.Module,
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    labels: Optional[np.ndarray],
    settle_steps: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Settle a batch of explicit HORN states and decode images."""

    if not hasattr(model, "evolve_state") or not hasattr(model, "decode_state"):
        raise ValueError("model must expose evolve_state and decode_state")
    step_model = _model_with_steps(model, int(settle_steps))
    outputs = []
    total = int(position.shape[0])
    for start in range(0, total, int(batch_size)):
        stop = min(start + int(batch_size), total)
        label_batch = (
            None
            if labels is None
            else jnp.asarray(labels[start:stop], dtype=jnp.int32)
        )
        final_position, final_velocity = step_model.evolve_state(
            (
                jnp.asarray(position[start:stop], dtype=jnp.float32),
                jnp.asarray(velocity[start:stop], dtype=jnp.float32),
            ),
            label_batch,
        )
        generated = step_model.decode_state(final_position, final_velocity)
        outputs.append(np.asarray(generated, dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def nearest_reference_mse_summary(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.001, 0.0025, 0.005, 0.010),
) -> Dict[str, float]:
    """Summarize nearest-reference pixel MSE and duplicate rates."""

    gen = np.clip(
        np.asarray(generated, dtype=np.float32).reshape(generated.shape[0], -1),
        0.0,
        1.0,
    )
    ref = np.asarray(reference, dtype=np.float32).reshape(reference.shape[0], -1)
    dim = float(max(gen.shape[-1], 1))
    gen_sq = np.sum(gen * gen, axis=-1, keepdims=True)
    ref_sq = np.sum(ref * ref, axis=-1, keepdims=True).T
    pairwise = np.maximum(gen_sq + ref_sq - 2.0 * (gen @ ref.T), 0.0) / dim
    nearest = np.min(pairwise, axis=1)
    summary = {
        "nearest_reference_mse": float(np.mean(nearest)),
        "nearest_reference_mse_min": float(np.min(nearest)),
        "nearest_reference_mse_p01": float(np.quantile(nearest, 0.01)),
        "nearest_reference_mse_p05": float(np.quantile(nearest, 0.05)),
    }
    for threshold in thresholds:
        threshold_milli = float(threshold) * 1000.0
        if abs(threshold_milli - round(threshold_milli)) < 1e-9:
            tag = f"{int(round(threshold_milli)):03d}"
        else:
            tag = f"{int(round(float(threshold) * 10000.0)):04d}"
        summary[f"duplicate_rate_mse_{tag}"] = float(np.mean(nearest < threshold))
    return summary


def evaluate_state_prior_sampling_probe(
    model: eqx.Module,
    *,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    eval_images: jnp.ndarray,
    eval_labels: jnp.ndarray,
    prototypes: Optional[jnp.ndarray],
    classifier: Optional[object] = None,
    image_shape: Sequence[int],
    sample_count: int,
    prior_rank: int,
    settle_steps: int,
    batch_size: int,
    key: jax.random.PRNGKey,
    prior_noise_scale: float = 1.0,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], ClassStatePrior]:
    """Run white-noise and class-prior init controls on one checkpoint."""

    _, _, encoded_states = encode_anchor_state_distribution(
        model,
        train_images,
        batch_size=batch_size,
    )
    prior = fit_class_state_prior(
        encoded_states,
        np.asarray(train_labels),
        num_classes=int(getattr(model, "num_classes", 10)),
        rank=int(prior_rank),
    )

    count = min(int(sample_count), int(eval_images.shape[0]))
    labels = np.asarray(eval_labels[:count], dtype=np.int32)
    eval_reference = np.asarray(eval_images, dtype=np.float32)
    train_reference = np.asarray(train_images, dtype=np.float32)
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    _, white_key = jax.random.split(key)

    sample_labels = jnp.asarray(labels, dtype=jnp.int32)
    step_model = _model_with_steps(model, int(settle_steps))
    variants: Dict[str, np.ndarray] = {
        "white_noise": np.asarray(
            sample_generator_images(
                step_model,
                key=white_key,
                sample_count=count,
                batch_size=batch_size,
                labels=sample_labels,
            ),
            dtype=np.float32,
        )
    }

    mean_position, mean_velocity = sample_class_state_prior(
        prior,
        labels,
        rng=rng,
        mean_only=True,
    )
    variants["prior_mean"] = generate_from_initial_states(
        model,
        mean_position,
        mean_velocity,
        labels=labels,
        settle_steps=settle_steps,
        batch_size=batch_size,
    )

    sample_position, sample_velocity = sample_class_state_prior(
        prior,
        labels,
        rng=rng,
        noise_scale=prior_noise_scale,
    )
    variants["prior_sample"] = generate_from_initial_states(
        model,
        sample_position,
        sample_velocity,
        labels=labels,
        settle_steps=settle_steps,
        batch_size=batch_size,
    )

    offsets = rng.integers(1, prior.num_classes, size=labels.shape[0], dtype=np.int32)
    shuffled_prior_labels = (labels + offsets) % prior.num_classes
    shuffled_position, shuffled_velocity = sample_class_state_prior(
        prior,
        shuffled_prior_labels,
        rng=rng,
        noise_scale=prior_noise_scale,
    )
    variants["shuffled_prior"] = generate_from_initial_states(
        model,
        shuffled_position,
        shuffled_velocity,
        labels=labels,
        settle_steps=settle_steps,
        batch_size=batch_size,
    )

    metrics: Dict[str, Dict[str, float]] = {}
    for name, generated in variants.items():
        quality = compute_generator_quality_metrics(
            eval_reference,
            generated,
            labels=sample_labels,
            prototypes=prototypes,
            classifier=classifier,
            image_shape=image_shape,
        )
        train_duplicate = nearest_reference_mse_summary(generated, train_reference)
        quality.update({f"train_{key}": value for key, value in train_duplicate.items()})
        quality["sample_count"] = float(count)
        quality["settle_steps"] = float(settle_steps)
        quality["prior_rank"] = float(prior_rank)
        quality["prior_noise_scale"] = float(prior_noise_scale)
        metrics[name] = quality
    return metrics, variants, prior


def write_state_prior_probe_csv(
    metrics: Mapping[str, Mapping[str, float]],
    path: Path,
) -> None:
    """Write one row per state-prior variant."""

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in metrics.values() for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", *keys])
        writer.writeheader()
        for variant, row in metrics.items():
            writer.writerow({"variant": variant, **row})


def save_state_prior_contact_sheet(
    variants: Mapping[str, np.ndarray],
    path: Path,
    *,
    image_shape: Sequence[int],
    columns: int = 16,
    max_per_variant: int = 16,
) -> None:
    """Save a row-per-variant generated-sample contact sheet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = _image_hw_channels(tuple(int(size) for size in image_shape))
    rows = len(variants)
    columns = int(max(1, columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns, max(1, rows) * 1.2),
        squeeze=False,
    )
    for row_index, (name, images) in enumerate(variants.items()):
        images_np = np.asarray(images, dtype=np.float32).reshape(
            images.shape[0],
            channels,
            height,
            width,
        )
        if channels == 3:
            images_np = np.transpose(images_np, (0, 2, 3, 1))
        else:
            images_np = images_np[:, 0]
        for column_index, axis in enumerate(axes[row_index]):
            axis.axis("off")
            if column_index == 0:
                axis.set_title(name, fontsize=8)
            if column_index < min(max_per_variant, images_np.shape[0]):
                image = np.clip(images_np[column_index], 0.0, 1.0)
                if channels == 1:
                    axis.imshow(image, cmap="gray", vmin=0, vmax=1)
                else:
                    axis.imshow(image, vmin=0, vmax=1)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def state_prior_to_json(prior: ClassStatePrior) -> Dict[str, object]:
    """Small JSON-safe summary of a fitted state prior."""

    payload = asdict(prior)
    payload["means"] = {
        "shape": list(prior.means.shape),
        "mean_abs": float(np.mean(np.abs(prior.means))),
        "std": float(np.std(prior.means)),
    }
    payload["components"] = {
        "shape": list(prior.components.shape),
        "mean_abs": float(np.mean(np.abs(prior.components))),
    }
    payload["scales"] = {
        "shape": list(prior.scales.shape),
        "mean": float(np.mean(prior.scales)),
        "max": float(np.max(prior.scales)),
    }
    payload["counts"] = prior.counts.tolist()
    return payload
