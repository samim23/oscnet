"""Training loop for MNIST oscillator generator experiments."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
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
from .common import Array, _logger, _tree_norm, occlude_image_batch
from .config import MNISTGeneratorExperimentConfig
from .features import (
    FeatureClassifier,
    compute_class_prototypes,
    make_projection_matrix,
    train_mnist_feature_classifier,
)
from .losses import (
    coarse_auxiliary_image_loss,
    coarse_auxiliary_image_loss_from_generated,
    coarse_readout_consistency_loss,
    frequency_statistics_loss,
    generator_loss,
    patch_sliced_wasserstein_loss,
)
from .metrics import (
    InitialStateSampler,
    _model_with_steps,
    compute_generator_attractor_robustness,
    compute_generator_quality_metrics,
    compute_generator_settling_metrics,
    compute_generator_recovery_metrics,
    compute_generator_robustness_metrics,
    compute_generator_state_fitting_probe,
    compute_generator_state_information_probe,
    compute_generator_success_diagnostics,
    compute_generator_vertical_intervention_audit,
    sample_generator_images,
)
from .queue import MNISTDriftQueue


@dataclass
class _StatePriorParams:
    """Host-side low-rank Gaussian prior over HORN position/velocity states."""

    means: np.ndarray
    components: np.ndarray
    scales: np.ndarray
    counts: np.ndarray
    rank: int
    state_dim: int
    mode: str

    @property
    def num_prior_classes(self) -> int:
        return int(self.means.shape[0])

    @property
    def oscillator_dim(self) -> int:
        return int(self.state_dim // 2)


def _encode_anchor_states_np(
    model: eqx.Module,
    images: Array,
    *,
    batch_size: int,
) -> np.ndarray:
    """Encode images into concatenated HORN position/velocity states on host."""

    if not hasattr(model, "encode_image_state"):
        raise ValueError("state-prior sampling requires encode_image_state")
    encoded = []
    total = int(images.shape[0])
    for start in range(0, total, int(batch_size)):
        batch = images[start : start + int(batch_size)]
        position, velocity = model.encode_image_state(batch)
        encoded.append(
            np.concatenate(
                [
                    np.asarray(position, dtype=np.float32),
                    np.asarray(velocity, dtype=np.float32),
                ],
                axis=-1,
            )
        )
    return np.concatenate(encoded, axis=0)


def _fit_low_rank_state_prior(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str,
    num_classes: int,
    rank: int,
) -> _StatePriorParams:
    """Fit a global or per-class PCA Gaussian prior over anchor states."""

    states = np.asarray(states, dtype=np.float32).reshape(states.shape[0], -1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if states.shape[0] != labels.shape[0]:
        raise ValueError("state-prior states and labels must align")
    if states.shape[-1] % 2 != 0:
        raise ValueError("state-prior state dimension must be even")
    if mode not in ("global", "class"):
        raise ValueError("state-prior mode must be 'global' or 'class'")
    prior_classes = 1 if mode == "global" else int(num_classes)
    rank = int(max(rank, 0))
    state_dim = int(states.shape[-1])
    means = np.zeros((prior_classes, state_dim), dtype=np.float32)
    components = np.zeros((prior_classes, rank, state_dim), dtype=np.float32)
    scales = np.zeros((prior_classes, rank), dtype=np.float32)
    counts = np.zeros((prior_classes,), dtype=np.int32)
    prior_labels = np.zeros_like(labels) if mode == "global" else labels
    for class_index in range(prior_classes):
        class_states = states[prior_labels == class_index]
        counts[class_index] = int(class_states.shape[0])
        if class_states.shape[0] == 0:
            class_states = states
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
    return _StatePriorParams(
        means=means,
        components=components,
        scales=scales,
        counts=counts,
        rank=rank,
        state_dim=state_dim,
        mode=mode,
    )


def _fit_anchor_state_prior(
    model: eqx.Module,
    train_images: Array,
    train_labels: Array,
    *,
    mode: str,
    num_classes: int,
    rank: int,
    batch_size: int,
) -> _StatePriorParams:
    """Fit a low-rank state prior from current anchor-encoder outputs."""

    states = _encode_anchor_states_np(model, train_images, batch_size=batch_size)
    return _fit_low_rank_state_prior(
        states,
        np.asarray(train_labels, dtype=np.int32),
        mode=mode,
        num_classes=num_classes,
        rank=rank,
    )


def _sample_state_prior_batch(
    prior: _StatePriorParams,
    labels: Optional[Array],
    *,
    rng: np.random.Generator,
    noise_scale: float,
    batch_size: int,
    prior_labels: Optional[np.ndarray] = None,
) -> Tuple[Array, Array]:
    """Draw one batch of HORN position/velocity states from a host prior."""

    if prior.mode == "global":
        prior_labels = np.zeros((int(batch_size),), dtype=np.int32)
    elif prior_labels is not None:
        prior_labels = np.asarray(prior_labels, dtype=np.int32).reshape(-1)
    else:
        if labels is None:
            raise ValueError("class state-prior sampling requires labels")
        prior_labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if prior_labels.shape[0] != int(batch_size):
        raise ValueError("state-prior label count must match batch_size")
    states = np.array(prior.means[prior_labels], copy=True)
    if prior.rank > 0 and noise_scale > 0.0:
        class_components = prior.components[prior_labels]
        class_scales = prior.scales[prior_labels] * float(noise_scale)
        coefficients = rng.normal(size=class_scales.shape).astype(np.float32)
        states = states + np.einsum(
            "br,brd->bd",
            coefficients * class_scales,
            class_components,
        ).astype(np.float32)
    split = prior.oscillator_dim
    return (
        jnp.asarray(states[:, :split], dtype=jnp.float32),
        jnp.asarray(states[:, split:], dtype=jnp.float32),
    )


def _make_state_prior_sampler(
    prior: _StatePriorParams,
    *,
    noise_scale: float,
    seed_offset: int = 0,
    shuffle_prior_labels: bool = False,
) -> InitialStateSampler:
    """Return a deterministic eval sampler for explicit HORN initial states."""

    def sampler(
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array],
    ) -> Tuple[Array, Array]:
        rng_seed = int(jax.random.randint(key, (), 0, 2**31 - 1)) + int(seed_offset)
        rng = np.random.default_rng(rng_seed)
        prior_labels = None
        if shuffle_prior_labels and prior.mode != "global":
            if labels is None:
                raise ValueError("shuffled state-prior eval requires labels")
            labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
            offsets = rng.integers(
                1,
                prior.num_prior_classes,
                size=labels_np.shape[0],
                dtype=np.int32,
            )
            prior_labels = (labels_np + offsets) % prior.num_prior_classes
        return _sample_state_prior_batch(
            prior,
            labels,
            rng=rng,
            noise_scale=noise_scale,
            batch_size=batch_size,
            prior_labels=prior_labels,
        )

    return sampler


def _write_state_prior_artifacts(
    prior: _StatePriorParams,
    path_prefix: Path,
) -> Dict[str, Any]:
    """Persist fitted state prior arrays and a small JSON summary."""

    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = path_prefix.with_suffix(".npz")
    json_path = path_prefix.with_suffix(".json")
    np.savez(
        npz_path,
        means=prior.means,
        components=prior.components,
        scales=prior.scales,
        counts=prior.counts,
        rank=np.asarray(prior.rank, dtype=np.int32),
        state_dim=np.asarray(prior.state_dim, dtype=np.int32),
        mode=np.asarray(prior.mode),
    )
    payload = {
        "mode": prior.mode,
        "rank": int(prior.rank),
        "state_dim": int(prior.state_dim),
        "num_prior_classes": int(prior.num_prior_classes),
        "oscillator_dim": int(prior.oscillator_dim),
        "means_shape": list(prior.means.shape),
        "components_shape": list(prior.components.shape),
        "scales_shape": list(prior.scales.shape),
        "counts": prior.counts.tolist(),
        "means_mean_abs": float(np.mean(np.abs(prior.means))),
        "means_std": float(np.std(prior.means)),
        "components_mean_abs": float(np.mean(np.abs(prior.components))),
        "scales_mean": float(np.mean(prior.scales)),
        "scales_max": float(np.max(prior.scales)) if prior.scales.size else 0.0,
        "npz_path": str(npz_path),
    }
    write_json(json_path, payload)
    return {"json_path": str(json_path), "npz_path": str(npz_path), **payload}


def _stop_gradient_if_array(value: Any) -> Any:
    if value is None:
        return None
    if eqx.is_array(value):
        return jax.lax.stop_gradient(value)
    return value


def _model_with_frozen_anchor_dynamics(model: eqx.Module) -> eqx.Module:
    """Stop gradients through recurrent/conditioning dynamics for anchor loss."""

    names = tuple(
        name
        for name in (
            "omega",
            "coupling",
            "condition_omega",
            "condition_coupling",
            "label_phase_shift",
            "label_condition_phase",
            "label_condition_coupling",
            "output_feedback_gain",
        )
        if hasattr(model, name)
    )
    replacements = tuple(
        _stop_gradient_if_array(getattr(model, name))
        for name in names
    )
    return eqx.tree_at(
        lambda module: tuple(getattr(module, name) for name in names),
        model,
        replacements,
        is_leaf=lambda value: value is None,
    )


def _state_anchor_image_loss(
    model: eqx.Module,
    real_batch: Array,
    *,
    key: jax.random.PRNGKey,
    state_anchor_weight: float,
    state_anchor_steps: Tuple[int, ...],
    state_anchor_noise_scale: float,
    state_anchor_mode: str,
    state_anchor_occlusion_fraction: float = 0.0,
    state_anchor_occlusion_patches: int = 4,
    state_anchor_occlusion_probability: float = 0.5,
    state_anchor_clean_weight: float = 0.0,
) -> Array:
    """Train a local image-to-state anchor to survive HORN settling.

    The encoder is a training-time probe: free sampling still starts from random
    HORN state. ``reconstruct`` is the k=0 autoencoder control; ``settle`` lets
    dynamics learn through the anchor; ``frozen_dynamics`` stops gradients on
    recurrent/conditioning parameters only inside this anchor path.

    Corruption for the recovery objective happens in two places: occlusion is
    applied to the image *before* encoding (zeroed square patches, so the task
    is "corrupted image in, clean image out after settling"), while Gaussian
    noise is added to the encoded state. The optional clean fixed-point term
    settles the uncorrupted encoded state and scores it against the clean
    image, training clean states to survive the dynamics.
    """

    if state_anchor_weight <= 0.0 or state_anchor_mode == "none":
        return jnp.asarray(0.0, dtype=real_batch.dtype)
    if not hasattr(model, "encode_image_state") or not hasattr(model, "decode_state"):
        return jnp.asarray(0.0, dtype=real_batch.dtype)
    if getattr(model, "state_anchor_encoder", None) is None:
        return jnp.asarray(0.0, dtype=real_batch.dtype)
    if state_anchor_mode not in ("reconstruct", "settle", "frozen_dynamics"):
        raise ValueError(
            "state_anchor_mode must be 'none', 'reconstruct', 'settle', "
            "or 'frozen_dynamics'"
        )

    if state_anchor_mode == "reconstruct":
        position, velocity = model.encode_image_state(real_batch)
        generated = model.decode_state(position, velocity)
        return jnp.mean((generated - real_batch) ** 2)

    if not state_anchor_steps:
        state_anchor_steps = (int(model.steps),)

    position_key, velocity_key, step_key, occlusion_key = jax.random.split(key, 4)
    encoder_input = real_batch
    if state_anchor_occlusion_fraction > 0.0:
        encoder_input, _ = occlude_image_batch(
            real_batch,
            key=occlusion_key,
            image_shape=model.image_shape,
            fraction=float(state_anchor_occlusion_fraction),
            patches=int(state_anchor_occlusion_patches),
            probability=float(state_anchor_occlusion_probability),
        )
    position, velocity = model.encode_image_state(encoder_input)
    if state_anchor_noise_scale > 0.0:
        position = position + float(state_anchor_noise_scale) * jax.random.normal(
            position_key,
            position.shape,
            dtype=position.dtype,
        )
        velocity = velocity + float(state_anchor_noise_scale) * jax.random.normal(
            velocity_key,
            velocity.shape,
            dtype=velocity.dtype,
        )
        if hasattr(model, "_bound_state"):
            position = model._bound_state(position)
            velocity = model._bound_state(velocity)

    dynamics_model = (
        _model_with_frozen_anchor_dynamics(model)
        if state_anchor_mode == "frozen_dynamics"
        else model
    )
    branch_index = jax.random.randint(
        step_key,
        shape=(),
        minval=0,
        maxval=len(state_anchor_steps),
    )

    clean_state = None
    if state_anchor_clean_weight > 0.0:
        if state_anchor_occlusion_fraction > 0.0 or state_anchor_noise_scale > 0.0:
            clean_state = model.encode_image_state(real_batch)
        else:
            # Without corruption the settled path already scores clean states.
            clean_state = (position, velocity)

    def branch(step_depth: int):
        def _run() -> Array:
            step_model = _model_with_steps(dynamics_model, int(step_depth))
            settled_position, settled_velocity = step_model.evolve_state(
                (position, velocity),
                None,
            )
            generated = model.decode_state(settled_position, settled_velocity)
            loss = jnp.mean((generated - real_batch) ** 2)
            if clean_state is not None:
                clean_position, clean_velocity = step_model.evolve_state(
                    clean_state,
                    None,
                )
                clean_decode = model.decode_state(clean_position, clean_velocity)
                loss = loss + float(state_anchor_clean_weight) * jnp.mean(
                    (clean_decode - real_batch) ** 2
                )
            return loss

        return _run

    return jax.lax.switch(
        branch_index,
        tuple(branch(int(step_depth)) for step_depth in state_anchor_steps),
    )


@eqx.filter_jit
def _train_step(
    model: eqx.Module,
    opt_state: Any,
    real_batch: Array,
    label_batch: Optional[Array],
    positive_batch: Optional[Array],
    positive_label_batch: Optional[Array],
    prior_position_batch: Optional[Array],
    prior_velocity_batch: Optional[Array],
    sample_key: jax.random.PRNGKey,
    projections: Array,
    patch_projections: Array,
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
    coarse_auxiliary_weight: float,
    coarse_auxiliary_target_size: int,
    coarse_auxiliary_loss_mode: str,
    coarse_readout_consistency_weight: float,
    frequency_objective_weight: float,
    frequency_objective_edge_weight: float,
    patch_objective_weight: float,
    patch_objective_patch_size: int,
    patch_objective_patch_sizes: Tuple[int, ...],
    patch_objective_stride: int,
    patch_objective_offsets: Tuple[int, ...],
    patch_objective_edge_weight: float,
    state_anchor_weight: float,
    state_anchor_steps: Tuple[int, ...],
    state_anchor_noise_scale: float,
    state_anchor_mode: str,
    state_anchor_occlusion_fraction: float,
    state_anchor_occlusion_patches: int,
    state_anchor_occlusion_probability: float,
    state_anchor_clean_weight: float,
    state_prior_sampling_mode: str,
    image_shape: Tuple[int, ...],
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
            step_key = jax.random.fold_in(sample_key, step_index)
            auxiliary_generated = None
            if (
                state_prior_sampling_mode != "none"
                and prior_position_batch is not None
                and prior_velocity_batch is not None
            ):
                if not hasattr(step_model, "evolve_state") or not hasattr(
                    step_model,
                    "decode_state",
                ):
                    raise ValueError(
                        "state-prior sampling requires evolve_state/decode_state"
                    )
                final_position, final_velocity = step_model.evolve_state(
                    (prior_position_batch, prior_velocity_batch),
                    label_batch,
                )
                generated = step_model.decode_state(final_position, final_velocity)
            elif coarse_readout_consistency_weight > 0.0 and hasattr(
                step_model,
                "sample_with_auxiliary_image",
            ):
                generated, auxiliary_generated = step_model.sample_with_auxiliary_image(
                    step_key,
                    real_batch.shape[0],
                    label_batch,
                )
            else:
                generated = step_model(
                    step_key,
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
            if coarse_auxiliary_weight > 0.0 and hasattr(
                step_model,
                "sample_auxiliary_image",
            ):
                if auxiliary_generated is None:
                    auxiliary_loss = coarse_auxiliary_image_loss(
                        step_model,
                        real_batch,
                        key=jax.random.fold_in(sample_key, step_index + 100_003),
                        labels=label_batch,
                        image_shape=image_shape,
                        target_size=coarse_auxiliary_target_size,
                        loss_mode=coarse_auxiliary_loss_mode,
                        num_classes=num_classes,
                    )
                else:
                    auxiliary_loss = coarse_auxiliary_image_loss_from_generated(
                        auxiliary_generated,
                        real_batch,
                        labels=label_batch,
                        image_shape=image_shape,
                        target_size=coarse_auxiliary_target_size,
                        loss_mode=coarse_auxiliary_loss_mode,
                        num_classes=num_classes,
                    )
                loss = loss + float(coarse_auxiliary_weight) * auxiliary_loss
                parts["coarse_auxiliary_loss"] = auxiliary_loss
            else:
                parts["coarse_auxiliary_loss"] = jnp.asarray(
                    0.0,
                    dtype=real_batch.dtype,
                )
            if (
                coarse_readout_consistency_weight > 0.0
                and auxiliary_generated is not None
            ):
                consistency_loss = coarse_readout_consistency_loss(
                    generated,
                    auxiliary_generated,
                    image_shape=image_shape,
                    target_size=coarse_auxiliary_target_size,
                )
                loss = (
                    loss
                    + float(coarse_readout_consistency_weight) * consistency_loss
                )
                parts["coarse_readout_consistency_loss"] = consistency_loss
            else:
                parts["coarse_readout_consistency_loss"] = jnp.asarray(
                    0.0,
                    dtype=real_batch.dtype,
                )
            if frequency_objective_weight > 0.0:
                frequency_loss = frequency_statistics_loss(
                    real_batch,
                    generated,
                    image_shape=image_shape,
                    edge_weight=frequency_objective_edge_weight,
                )
                loss = loss + float(frequency_objective_weight) * frequency_loss
                parts["frequency_objective_loss"] = frequency_loss
            else:
                parts["frequency_objective_loss"] = jnp.asarray(
                    0.0,
                    dtype=real_batch.dtype,
                )
            if patch_objective_weight > 0.0:
                patch_loss = patch_sliced_wasserstein_loss(
                    real_batch,
                    generated,
                    patch_projections,
                    image_shape=image_shape,
                    patch_size=patch_objective_patch_size,
                    patch_sizes=patch_objective_patch_sizes,
                    stride=patch_objective_stride,
                    offsets=patch_objective_offsets,
                    edge_weight=patch_objective_edge_weight,
                )
                loss = loss + float(patch_objective_weight) * patch_loss
                parts["patch_objective_loss"] = patch_loss
            else:
                parts["patch_objective_loss"] = jnp.asarray(
                    0.0,
                    dtype=real_batch.dtype,
                )
            losses.append(loss)
            for name, value in parts.items():
                parts_by_name.setdefault(name, []).append(value)
        mean_parts = {
            name: jnp.mean(jnp.stack(values))
            for name, values in parts_by_name.items()
        }
        total_loss = jnp.mean(jnp.stack(losses))
        state_anchor_loss = _state_anchor_image_loss(
            current_model,
            real_batch,
            key=jax.random.fold_in(sample_key, 701_001),
            state_anchor_weight=state_anchor_weight,
            state_anchor_steps=state_anchor_steps,
            state_anchor_noise_scale=state_anchor_noise_scale,
            state_anchor_mode=state_anchor_mode,
            state_anchor_occlusion_fraction=state_anchor_occlusion_fraction,
            state_anchor_occlusion_patches=state_anchor_occlusion_patches,
            state_anchor_occlusion_probability=state_anchor_occlusion_probability,
            state_anchor_clean_weight=state_anchor_clean_weight,
        )
        total_loss = total_loss + float(state_anchor_weight) * state_anchor_loss
        mean_parts["state_anchor_loss"] = state_anchor_loss
        return total_loss, mean_parts

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
    prior_position_batch: Optional[Array],
    prior_velocity_batch: Optional[Array],
    sample_key: jax.random.PRNGKey,
    projections: Array,
    patch_projections: Array,
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
    coarse_auxiliary_weight: float,
    coarse_auxiliary_target_size: int,
    coarse_auxiliary_loss_mode: str,
    coarse_readout_consistency_weight: float,
    frequency_objective_weight: float,
    frequency_objective_edge_weight: float,
    patch_objective_weight: float,
    patch_objective_patch_size: int,
    patch_objective_patch_sizes: Tuple[int, ...],
    patch_objective_stride: int,
    patch_objective_offsets: Tuple[int, ...],
    patch_objective_edge_weight: float,
    state_anchor_weight: float,
    state_anchor_steps: Tuple[int, ...],
    state_anchor_noise_scale: float,
    state_anchor_mode: str,
    state_anchor_occlusion_fraction: float,
    state_anchor_occlusion_patches: int,
    state_anchor_occlusion_probability: float,
    state_anchor_clean_weight: float,
    state_prior_sampling_mode: str,
    image_shape: Tuple[int, ...],
    num_classes: int,
):
    auxiliary_generated = None
    if (
        state_prior_sampling_mode != "none"
        and prior_position_batch is not None
        and prior_velocity_batch is not None
    ):
        if not hasattr(model, "evolve_state") or not hasattr(model, "decode_state"):
            raise ValueError("state-prior eval requires evolve_state/decode_state")
        final_position, final_velocity = model.evolve_state(
            (prior_position_batch, prior_velocity_batch),
            label_batch,
        )
        generated = model.decode_state(final_position, final_velocity)
    elif coarse_readout_consistency_weight > 0.0 and hasattr(
        model,
        "sample_with_auxiliary_image",
    ):
        generated, auxiliary_generated = model.sample_with_auxiliary_image(
            sample_key,
            real_batch.shape[0],
            label_batch,
        )
    else:
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
    if coarse_auxiliary_weight > 0.0 and hasattr(model, "sample_auxiliary_image"):
        if auxiliary_generated is None:
            auxiliary_loss = coarse_auxiliary_image_loss(
                model,
                real_batch,
                key=jax.random.fold_in(sample_key, 200_003),
                labels=label_batch,
                image_shape=image_shape,
                target_size=coarse_auxiliary_target_size,
                loss_mode=coarse_auxiliary_loss_mode,
                num_classes=num_classes,
            )
        else:
            auxiliary_loss = coarse_auxiliary_image_loss_from_generated(
                auxiliary_generated,
                real_batch,
                labels=label_batch,
                image_shape=image_shape,
                target_size=coarse_auxiliary_target_size,
                loss_mode=coarse_auxiliary_loss_mode,
                num_classes=num_classes,
            )
        loss = loss + float(coarse_auxiliary_weight) * auxiliary_loss
        parts["coarse_auxiliary_loss"] = auxiliary_loss
    else:
        parts["coarse_auxiliary_loss"] = jnp.asarray(0.0, dtype=real_batch.dtype)
    if coarse_readout_consistency_weight > 0.0 and auxiliary_generated is not None:
        consistency_loss = coarse_readout_consistency_loss(
            generated,
            auxiliary_generated,
            image_shape=image_shape,
            target_size=coarse_auxiliary_target_size,
        )
        loss = loss + float(coarse_readout_consistency_weight) * consistency_loss
        parts["coarse_readout_consistency_loss"] = consistency_loss
    else:
        parts["coarse_readout_consistency_loss"] = jnp.asarray(
            0.0,
            dtype=real_batch.dtype,
        )
    if frequency_objective_weight > 0.0:
        frequency_loss = frequency_statistics_loss(
            real_batch,
            generated,
            image_shape=image_shape,
            edge_weight=frequency_objective_edge_weight,
        )
        loss = loss + float(frequency_objective_weight) * frequency_loss
        parts["frequency_objective_loss"] = frequency_loss
    else:
        parts["frequency_objective_loss"] = jnp.asarray(0.0, dtype=real_batch.dtype)
    if patch_objective_weight > 0.0:
        patch_loss = patch_sliced_wasserstein_loss(
            real_batch,
            generated,
            patch_projections,
            image_shape=image_shape,
            patch_size=patch_objective_patch_size,
            patch_sizes=patch_objective_patch_sizes,
            stride=patch_objective_stride,
            offsets=patch_objective_offsets,
            edge_weight=patch_objective_edge_weight,
        )
        loss = loss + float(patch_objective_weight) * patch_loss
        parts["patch_objective_loss"] = patch_loss
    else:
        parts["patch_objective_loss"] = jnp.asarray(0.0, dtype=real_batch.dtype)
    state_anchor_loss = _state_anchor_image_loss(
        model,
        real_batch,
        key=jax.random.fold_in(sample_key, 702_001),
        state_anchor_weight=state_anchor_weight,
        state_anchor_steps=state_anchor_steps,
        state_anchor_noise_scale=state_anchor_noise_scale,
        state_anchor_mode=state_anchor_mode,
        state_anchor_occlusion_fraction=state_anchor_occlusion_fraction,
        state_anchor_occlusion_patches=state_anchor_occlusion_patches,
        state_anchor_occlusion_probability=state_anchor_occlusion_probability,
        state_anchor_clean_weight=state_anchor_clean_weight,
    )
    loss = loss + float(state_anchor_weight) * state_anchor_loss
    parts["state_anchor_loss"] = state_anchor_loss
    return loss, parts


def evaluate_generator_loss(
    model: eqx.Module,
    real_images: Array,
    *,
    batch_size: int,
    projections: Array,
    patch_projections: Array,
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
    coarse_auxiliary_weight: float = 0.0,
    coarse_auxiliary_target_size: int = 8,
    coarse_auxiliary_loss_mode: str = "mse",
    coarse_readout_consistency_weight: float = 0.0,
    frequency_objective_weight: float = 0.0,
    frequency_objective_edge_weight: float = 1.0,
    patch_objective_weight: float = 0.0,
    patch_objective_patch_size: int = 5,
    patch_objective_patch_sizes: Tuple[int, ...] = (),
    patch_objective_stride: int = 4,
    patch_objective_offsets: Tuple[int, ...] = (0,),
    patch_objective_edge_weight: float = 0.25,
    state_anchor_weight: float = 0.0,
    state_anchor_steps: Tuple[int, ...] = (4, 8, 16),
    state_anchor_noise_scale: float = 0.05,
    state_anchor_mode: str = "none",
    state_anchor_occlusion_fraction: float = 0.0,
    state_anchor_occlusion_patches: int = 4,
    state_anchor_occlusion_probability: float = 0.5,
    state_anchor_clean_weight: float = 0.0,
    initial_state_sampler: Optional[InitialStateSampler] = None,
    state_prior_sampling_mode: str = "none",
    image_shape: Tuple[int, ...] = (28, 28),
    num_classes: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate mean distributional loss over real-image batches."""

    effective_state_prior_sampling_mode = (
        state_prior_sampling_mode
        if initial_state_sampler is None or state_prior_sampling_mode != "none"
        else "explicit"
    )
    losses = []
    swd_losses = []
    moment_losses = []
    marginal_losses = []
    class_moment_losses = []
    prototype_losses = []
    pixel_drift_losses = []
    feature_drift_losses = []
    coarse_auxiliary_losses = []
    coarse_readout_consistency_losses = []
    frequency_objective_losses = []
    patch_objective_losses = []
    state_anchor_losses = []
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
        prior_position_batch = None
        prior_velocity_batch = None
        if initial_state_sampler is not None:
            prior_position_batch, prior_velocity_batch = initial_state_sampler(
                sample_key,
                int(real_batch.shape[0]),
                label_batch,
            )
        loss, parts = _eval_step(
            model,
            real_batch,
            label_batch,
            prior_position_batch,
            prior_velocity_batch,
            sample_key,
            projections,
            patch_projections,
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
            coarse_auxiliary_weight,
            coarse_auxiliary_target_size,
            coarse_auxiliary_loss_mode,
            coarse_readout_consistency_weight,
            frequency_objective_weight,
            frequency_objective_edge_weight,
            patch_objective_weight,
            patch_objective_patch_size,
            patch_objective_patch_sizes,
            patch_objective_stride,
            patch_objective_offsets,
            patch_objective_edge_weight,
            state_anchor_weight,
            state_anchor_steps,
            state_anchor_noise_scale,
            state_anchor_mode,
            state_anchor_occlusion_fraction,
            state_anchor_occlusion_patches,
            state_anchor_occlusion_probability,
            state_anchor_clean_weight,
            effective_state_prior_sampling_mode,
            image_shape,
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
        coarse_auxiliary_losses.append(float(parts["coarse_auxiliary_loss"]))
        coarse_readout_consistency_losses.append(
            float(parts["coarse_readout_consistency_loss"])
        )
        frequency_objective_losses.append(float(parts["frequency_objective_loss"]))
        patch_objective_losses.append(float(parts["patch_objective_loss"]))
        state_anchor_losses.append(float(parts["state_anchor_loss"]))
    if not losses:
        return float("nan"), {
            "eval_sliced_wasserstein": float("nan"),
            "eval_moment_loss": float("nan"),
            "eval_pixel_marginal_loss": float("nan"),
            "eval_class_moment_loss": float("nan"),
            "eval_prototype_loss": float("nan"),
            "eval_pixel_drift_loss": float("nan"),
            "eval_feature_drift_loss": float("nan"),
            "eval_coarse_auxiliary_loss": float("nan"),
            "eval_coarse_readout_consistency_loss": float("nan"),
            "eval_frequency_objective_loss": float("nan"),
            "eval_patch_objective_loss": float("nan"),
            "eval_state_anchor_loss": float("nan"),
        }
    return float(np.mean(losses)), {
        "eval_sliced_wasserstein": float(np.mean(swd_losses)),
        "eval_moment_loss": float(np.mean(moment_losses)),
        "eval_pixel_marginal_loss": float(np.mean(marginal_losses)),
        "eval_class_moment_loss": float(np.mean(class_moment_losses)),
        "eval_prototype_loss": float(np.mean(prototype_losses)),
        "eval_pixel_drift_loss": float(np.mean(pixel_drift_losses)),
        "eval_feature_drift_loss": float(np.mean(feature_drift_losses)),
        "eval_coarse_auxiliary_loss": float(np.mean(coarse_auxiliary_losses)),
        "eval_coarse_readout_consistency_loss": float(
            np.mean(coarse_readout_consistency_losses)
        ),
        "eval_frequency_objective_loss": float(np.mean(frequency_objective_losses)),
        "eval_patch_objective_loss": float(np.mean(patch_objective_losses)),
        "eval_state_anchor_loss": float(np.mean(state_anchor_losses)),
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
    if any(step < 0 for step in config.state_anchor_steps):
        raise ValueError("state_anchor_steps must be non-negative")
    if config.state_anchor_weight < 0.0:
        raise ValueError("state_anchor_weight must be non-negative")
    if config.state_anchor_noise_scale < 0.0:
        raise ValueError("state_anchor_noise_scale must be non-negative")
    if config.state_anchor_encoder_kernel_size < 1 or (
        config.state_anchor_encoder_kernel_size % 2 != 1
    ):
        raise ValueError("state_anchor_encoder_kernel_size must be a positive odd integer")
    if config.state_anchor_mode not in (
        "none",
        "reconstruct",
        "settle",
        "frozen_dynamics",
    ):
        raise ValueError(
            "state_anchor_mode must be 'none', 'reconstruct', 'settle', "
            "or 'frozen_dynamics'"
        )
    if config.state_anchor_weight > 0.0 and config.state_anchor_mode == "none":
        raise ValueError("state_anchor_weight > 0 requires state_anchor_mode != 'none'")
    if config.state_prior_sampling_mode not in ("none", "global", "class"):
        raise ValueError(
            "state_prior_sampling_mode must be 'none', 'global', or 'class'"
        )
    if config.state_prior_rank < 0:
        raise ValueError("state_prior_rank must be non-negative")
    if config.state_prior_noise_scale < 0.0:
        raise ValueError("state_prior_noise_scale must be non-negative")
    if config.state_prior_refresh_epochs < 1:
        raise ValueError("state_prior_refresh_epochs must be positive")
    if config.state_prior_start_epoch < 1:
        raise ValueError("state_prior_start_epoch must be positive")
    if config.state_prior_sampling_mode == "class" and not config.conditional:
        raise ValueError("class state-prior sampling requires conditional labels")
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
    if config.state_prior_sampling_mode != "none":
        if not hasattr(model, "encode_image_state") or not hasattr(
            model,
            "evolve_state",
        ):
            raise ValueError(
                "state-prior sampling requires an anchor-enabled HORN-style model"
            )
        if getattr(model, "state_anchor_encoder", None) is None:
            raise ValueError(
                "state-prior sampling requires state_anchor_weight > 0 and "
                "state_anchor_mode != 'none' so an anchor encoder is built"
            )
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
    patch_channels = int(config.image_shape[2]) if len(config.image_shape) == 3 else 1
    patch_sizes = tuple(int(size) for size in config.patch_objective_patch_sizes)
    max_patch_size = max(patch_sizes or (int(config.patch_objective_patch_size),))
    patch_dim = int(
        patch_channels
        * max_patch_size
        * max_patch_size
    )
    patch_projections = make_projection_matrix(
        jax.random.fold_in(projection_key, 17_031),
        image_dim=patch_dim,
        num_projections=config.patch_objective_projections,
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
        "train_coarse_auxiliary_loss": [],
        "train_coarse_readout_consistency_loss": [],
        "train_frequency_objective_loss": [],
        "train_patch_objective_loss": [],
        "train_state_anchor_loss": [],
        "train_state_prior_sampling_active": [],
        "state_prior_refit_seconds": [],
        "train_drift_queue_ready": [],
        "eval_sliced_wasserstein": [],
        "eval_moment_loss": [],
        "eval_pixel_marginal_loss": [],
        "eval_class_moment_loss": [],
        "eval_prototype_loss": [],
        "eval_pixel_drift_loss": [],
        "eval_feature_drift_loss": [],
        "eval_coarse_auxiliary_loss": [],
        "eval_coarse_readout_consistency_loss": [],
        "eval_frequency_objective_loss": [],
        "eval_patch_objective_loss": [],
        "eval_state_anchor_loss": [],
        "best_eval_loss": None,
        "best_epoch": None,
    }
    checkpoint_paths = []
    best_loss = float("inf")
    total_train_seconds = 0.0
    state_prior: Optional[_StatePriorParams] = None
    state_prior_rng = np.random.default_rng(config.run.seed + 515_151)

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
        coarse_auxiliary_losses = []
        coarse_readout_consistency_losses = []
        frequency_objective_losses = []
        patch_objective_losses = []
        state_anchor_losses = []
        drift_queue_ready_values = []
        consistency_active = (
            config.coarse_readout_consistency_weight > 0.0
            and epoch >= config.coarse_readout_consistency_onset_epoch
        )
        effective_consistency_weight = (
            config.coarse_readout_consistency_weight if consistency_active else 0.0
        )
        state_prior_refit_seconds = 0.0
        state_prior_active = (
            config.state_prior_sampling_mode != "none"
            and epoch >= config.state_prior_start_epoch
        )
        should_refit_state_prior = (
            state_prior_active
            and (
                state_prior is None
                or (
                    (epoch - config.state_prior_start_epoch)
                    % config.state_prior_refresh_epochs
                    == 0
                )
            )
        )
        if should_refit_state_prior:
            prior_start = time.time()
            state_prior = _fit_anchor_state_prior(
                model,
                train_images,
                train_labels,
                mode=config.state_prior_sampling_mode,
                num_classes=config.num_classes,
                rank=config.state_prior_rank,
                batch_size=config.run.batch_size,
            )
            state_prior_refit_seconds = float(time.time() - prior_start)
            logger.info(
                "state_prior mode=%s rank=%s refit_seconds=%.2f",
                config.state_prior_sampling_mode,
                config.state_prior_rank,
                state_prior_refit_seconds,
            )

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
            prior_position_batch = None
            prior_velocity_batch = None
            if state_prior_active and state_prior is not None:
                prior_position_batch, prior_velocity_batch = _sample_state_prior_batch(
                    state_prior,
                    label_batch,
                    rng=state_prior_rng,
                    noise_scale=config.state_prior_noise_scale,
                    batch_size=int(real_batch.shape[0]),
                )
            model, opt_state, loss, grad_norm, parts = _train_step(
                model,
                opt_state,
                real_batch,
                label_batch,
                positive_batch,
                positive_label_batch,
                prior_position_batch,
                prior_velocity_batch,
                sample_key,
                projections,
                patch_projections,
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
                config.coarse_auxiliary_weight,
                config.coarse_auxiliary_target_size,
                config.coarse_auxiliary_loss_mode,
                effective_consistency_weight,
                config.frequency_objective_weight,
                config.frequency_objective_edge_weight,
                config.patch_objective_weight,
                config.patch_objective_patch_size,
                config.patch_objective_patch_sizes,
                config.patch_objective_stride,
                config.patch_objective_offsets,
                config.patch_objective_edge_weight,
                config.state_anchor_weight,
                config.state_anchor_steps,
                config.state_anchor_noise_scale,
                config.state_anchor_mode,
                config.state_anchor_occlusion_fraction,
                config.state_anchor_occlusion_patches,
                config.state_anchor_occlusion_probability,
                config.state_anchor_clean_weight,
                config.state_prior_sampling_mode if state_prior_active else "none",
                config.image_shape,
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
            coarse_auxiliary_losses.append(float(parts["coarse_auxiliary_loss"]))
            coarse_readout_consistency_losses.append(
                float(parts["coarse_readout_consistency_loss"])
            )
            frequency_objective_losses.append(float(parts["frequency_objective_loss"]))
            patch_objective_losses.append(float(parts["patch_objective_loss"]))
            state_anchor_losses.append(float(parts["state_anchor_loss"]))
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
        train_coarse_auxiliary = (
            float(np.mean(coarse_auxiliary_losses))
            if coarse_auxiliary_losses
            else float("nan")
        )
        train_coarse_readout_consistency = (
            float(np.mean(coarse_readout_consistency_losses))
            if coarse_readout_consistency_losses
            else float("nan")
        )
        train_frequency_objective = (
            float(np.mean(frequency_objective_losses))
            if frequency_objective_losses
            else float("nan")
        )
        train_patch_objective = (
            float(np.mean(patch_objective_losses))
            if patch_objective_losses
            else float("nan")
        )
        train_state_anchor = (
            float(np.mean(state_anchor_losses))
            if state_anchor_losses
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
            "eval_coarse_auxiliary_loss": None,
            "eval_coarse_readout_consistency_loss": None,
            "eval_frequency_objective_loss": None,
            "eval_patch_objective_loss": None,
            "eval_state_anchor_loss": None,
        }
        if epoch % config.run.eval_every == 0:
            eval_key = jax.random.fold_in(key, epoch + 20_000)
            eval_initial_state_sampler = (
                _make_state_prior_sampler(
                    state_prior,
                    noise_scale=config.state_prior_noise_scale,
                    seed_offset=epoch * 1009,
                )
                if state_prior_active and state_prior is not None
                else None
            )
            eval_loss, eval_parts = evaluate_generator_loss(
                model,
                eval_images,
                batch_size=config.run.batch_size,
                projections=projections,
                patch_projections=patch_projections,
                key=eval_key,
                initial_state_sampler=eval_initial_state_sampler,
                state_prior_sampling_mode=(
                    config.state_prior_sampling_mode
                    if eval_initial_state_sampler is not None
                    else "none"
                ),
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
                coarse_auxiliary_weight=config.coarse_auxiliary_weight,
                coarse_auxiliary_target_size=config.coarse_auxiliary_target_size,
                coarse_auxiliary_loss_mode=config.coarse_auxiliary_loss_mode,
                coarse_readout_consistency_weight=effective_consistency_weight,
                frequency_objective_weight=config.frequency_objective_weight,
                frequency_objective_edge_weight=config.frequency_objective_edge_weight,
                patch_objective_weight=config.patch_objective_weight,
                patch_objective_patch_size=config.patch_objective_patch_size,
                patch_objective_patch_sizes=config.patch_objective_patch_sizes,
                patch_objective_stride=config.patch_objective_stride,
                patch_objective_offsets=config.patch_objective_offsets,
                patch_objective_edge_weight=config.patch_objective_edge_weight,
                state_anchor_weight=config.state_anchor_weight,
                state_anchor_steps=config.state_anchor_steps,
                state_anchor_noise_scale=config.state_anchor_noise_scale,
                state_anchor_mode=config.state_anchor_mode,
                state_anchor_occlusion_fraction=config.state_anchor_occlusion_fraction,
                state_anchor_occlusion_patches=config.state_anchor_occlusion_patches,
                state_anchor_occlusion_probability=(
                    config.state_anchor_occlusion_probability
                ),
                state_anchor_clean_weight=config.state_anchor_clean_weight,
                image_shape=config.image_shape,
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
        metrics["train_coarse_auxiliary_loss"].append(train_coarse_auxiliary)
        metrics["train_coarse_readout_consistency_loss"].append(
            train_coarse_readout_consistency
        )
        metrics["train_frequency_objective_loss"].append(train_frequency_objective)
        metrics["train_patch_objective_loss"].append(train_patch_objective)
        metrics["train_state_anchor_loss"].append(train_state_anchor)
        metrics["train_state_prior_sampling_active"].append(float(state_prior_active))
        metrics["state_prior_refit_seconds"].append(state_prior_refit_seconds)
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
        metrics["eval_coarse_auxiliary_loss"].append(
            eval_parts["eval_coarse_auxiliary_loss"]
        )
        metrics["eval_coarse_readout_consistency_loss"].append(
            eval_parts["eval_coarse_readout_consistency_loss"]
        )
        metrics["eval_frequency_objective_loss"].append(
            eval_parts["eval_frequency_objective_loss"]
        )
        metrics["eval_patch_objective_loss"].append(
            eval_parts["eval_patch_objective_loss"]
        )
        metrics["eval_state_anchor_loss"].append(
            eval_parts["eval_state_anchor_loss"]
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
                    "train_frequency_objective_loss": train_frequency_objective,
                    "eval_frequency_objective_loss": (
                        eval_parts["eval_frequency_objective_loss"]
                    ),
                    "train_patch_objective_loss": train_patch_objective,
                    "eval_patch_objective_loss": (
                        eval_parts["eval_patch_objective_loss"]
                    ),
                    "train_state_anchor_loss": train_state_anchor,
                    "eval_state_anchor_loss": (
                        eval_parts["eval_state_anchor_loss"]
                    ),
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
                    "train_frequency_objective_loss": train_frequency_objective,
                    "eval_frequency_objective_loss": (
                        eval_parts["eval_frequency_objective_loss"]
                    ),
                    "train_patch_objective_loss": train_patch_objective,
                    "eval_patch_objective_loss": (
                        eval_parts["eval_patch_objective_loss"]
                    ),
                    "train_state_anchor_loss": train_state_anchor,
                    "eval_state_anchor_loss": (
                        eval_parts["eval_state_anchor_loss"]
                    ),
                    "is_best": is_best,
                },
                output_dir=paths.checkpoints,
                hyperparams=_checkpoint_hyperparams(config),
                is_best=False,
            )
            record_checkpoint(checkpoint_path)

        _save_metrics_bundle(metrics, paths)

        if epoch == config.run.epochs or epoch % config.run.artifact_every == 0:
            artifact_sampler = (
                _make_state_prior_sampler(
                    state_prior,
                    noise_scale=config.state_prior_noise_scale,
                    seed_offset=epoch * 2011,
                )
                if state_prior_active and state_prior is not None
                else None
            )
            save_mnist_generator_artifacts(
                model,
                eval_images,
                paths,
                epoch,
                key=jax.random.fold_in(key, epoch + 30_000),
                sample_count=config.eval_sample_count,
                batch_size=config.run.batch_size,
                labels=eval_labels if config.conditional else None,
                initial_state_sampler=artifact_sampler,
            )

    eval_count = min(int(config.eval_sample_count), int(eval_images.shape[0]))
    final_state_prior_artifacts: Dict[str, Any] = {}
    final_prior_sampler: Optional[InitialStateSampler] = None
    shuffled_prior_sampler: Optional[InitialStateSampler] = None
    if config.state_prior_sampling_mode != "none":
        prior_start = time.time()
        state_prior = _fit_anchor_state_prior(
            model,
            train_images,
            train_labels,
            mode=config.state_prior_sampling_mode,
            num_classes=config.num_classes,
            rank=config.state_prior_rank,
            batch_size=config.run.batch_size,
        )
        final_state_prior_refit_seconds = float(time.time() - prior_start)
        final_state_prior_artifacts = _write_state_prior_artifacts(
            state_prior,
            paths.checkpoints / "state_prior_final",
        )
        final_state_prior_artifacts["final_refit_seconds"] = (
            final_state_prior_refit_seconds
        )
        final_prior_sampler = _make_state_prior_sampler(
            state_prior,
            noise_scale=config.state_prior_noise_scale,
            seed_offset=90_001,
        )
        if config.state_prior_sampling_mode == "class":
            shuffled_prior_sampler = _make_state_prior_sampler(
                state_prior,
                noise_scale=config.state_prior_noise_scale,
                seed_offset=90_777,
                shuffle_prior_labels=True,
            )
    final_generated = sample_generator_images(
        model,
        key=jax.random.fold_in(key, 40_000),
        sample_count=eval_count,
        batch_size=config.run.batch_size,
        labels=eval_labels[:eval_count] if config.conditional else None,
        initial_state_sampler=final_prior_sampler,
    )
    quality = compute_generator_quality_metrics(
        eval_images[:eval_count],
        final_generated,
        labels=eval_labels[:eval_count] if config.conditional else None,
        prototypes=prototypes,
        classifier=quality_classifier_model,
        image_shape=config.image_shape,
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
        image_shape=config.image_shape,
        initial_state_sampler=final_prior_sampler,
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
    state_information_probe: Dict[str, Any] = {}
    if config.state_probe_sample_count > 0:
        state_probe_count = min(
            eval_count,
            int(config.run.batch_size),
            int(config.state_probe_sample_count),
            int(eval_images.shape[0]),
        )
        if state_probe_count >= 4:
            state_probe_labels = (
                eval_labels[:state_probe_count] if config.conditional else None
            )
            if state_probe_count == diagnostic_count:
                state_probe_trace = diagnostic_trace
            else:
                state_probe_trace = model.collect_trace(
                    jax.random.fold_in(key, 50_333),
                    state_probe_count,
                    state_probe_labels,
                )
            state_information_probe = compute_generator_state_information_probe(
                model,
                state_probe_trace,
                labels=state_probe_labels,
                classifier=quality_classifier_model,
                image_shape=config.image_shape,
                target_size=config.state_probe_target_size,
                ridge=config.state_probe_ridge,
            )
        else:
            state_information_probe = {
                "sample_count": int(state_probe_count),
                "insufficient_samples": True,
            }
    state_fitting_probe: Dict[str, Any] = {}
    if config.state_fit_sample_count > 0:
        state_fit_count = min(
            eval_count,
            int(config.state_fit_sample_count),
            int(eval_images.shape[0]),
        )
        if state_fit_count >= 2:
            state_fitting_probe = compute_generator_state_fitting_probe(
                model,
                eval_images[:state_fit_count],
                key=jax.random.fold_in(key, 50_777),
                labels=(
                    eval_labels[:state_fit_count] if config.conditional else None
                ),
                image_shape=config.image_shape,
                sample_count=state_fit_count,
                fit_steps=config.state_fit_steps,
                learning_rate=config.state_fit_learning_rate,
                init_scale=config.state_fit_init_scale,
                settle_steps=config.state_fit_settle_steps,
                ridge=config.state_fit_ridge,
            )
        else:
            state_fitting_probe = {
                "sample_count": int(state_fit_count),
                "insufficient_samples": True,
            }
    recovery_metrics: Dict[str, Any] = {}
    if config.recovery_eval_sample_count > 0:
        recovery_metrics = compute_generator_recovery_metrics(
            model,
            eval_images,
            key=jax.random.fold_in(key, 51_333),
            image_shape=config.image_shape,
            sample_count=config.recovery_eval_sample_count,
            noise_scales=config.recovery_eval_noise_scales,
            occlusion_fractions=config.recovery_eval_occlusion_fractions,
            occlusion_patch_counts=config.recovery_eval_occlusion_patches,
            settle_steps=config.recovery_eval_settle_steps,
        )
    robustness_metrics: Dict[str, Any] = {}
    if config.robustness_eval_sample_count > 0:
        robustness_metrics = compute_generator_robustness_metrics(
            model,
            eval_images,
            key=jax.random.fold_in(key, 52_777),
            image_shape=config.image_shape,
            sample_count=config.robustness_eval_sample_count,
            settle_step=config.robustness_eval_settle_step,
            weight_noise_scales=config.robustness_eval_weight_noise_scales,
            quant_bits=config.robustness_eval_quant_bits,
            ood_occlusion_fractions=config.robustness_eval_occlusion_fractions,
            weight_noise_draws=config.robustness_eval_weight_noise_draws,
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
        initial_state_sampler=final_prior_sampler,
    )
    vertical_intervention_audit = compute_generator_vertical_intervention_audit(
        model,
        key=jax.random.fold_in(key, 56_000),
        real_images=eval_images[:eval_count],
        sample_count=(
            config.vertical_audit_sample_count
            if config.vertical_audit_sample_count > 0
            else eval_count
        ),
        batch_size=config.run.batch_size,
        labels=eval_labels[:eval_count] if config.conditional else None,
        prototypes=prototypes,
        classifier=quality_classifier_model,
        modes=config.vertical_audit_modes,
        image_shape=config.image_shape,
        attractor_variants_per_class=(
            config.attractor_variants_per_class
            if config.conditional and config.vertical_audit_modes
            else 0
        ),
        num_classes=config.num_classes,
        trace_batch_size=diagnostic_count,
        initial_state_sampler=final_prior_sampler,
    )
    white_noise_quality: Dict[str, float] = {}
    white_noise_settling: Dict[str, Any] = {}
    white_noise_attractor_robustness: Dict[str, float] = {}
    if final_prior_sampler is not None:
        white_noise_generated = sample_generator_images(
            model,
            key=jax.random.fold_in(key, 57_000),
            sample_count=eval_count,
            batch_size=config.run.batch_size,
            labels=eval_labels[:eval_count] if config.conditional else None,
        )
        white_noise_quality = compute_generator_quality_metrics(
            eval_images[:eval_count],
            white_noise_generated,
            labels=eval_labels[:eval_count] if config.conditional else None,
            prototypes=prototypes,
            classifier=quality_classifier_model,
            image_shape=config.image_shape,
        )
        white_noise_settling = compute_generator_settling_metrics(
            model,
            key=jax.random.fold_in(key, 57_500),
            real_images=eval_images[:eval_count],
            sample_count=eval_count,
            batch_size=config.run.batch_size,
            settling_steps=config.settling_steps,
            labels=eval_labels[:eval_count] if config.conditional else None,
            prototypes=prototypes,
            classifier=quality_classifier_model,
            image_shape=config.image_shape,
        )
        white_noise_attractor_robustness = compute_generator_attractor_robustness(
            model,
            key=jax.random.fold_in(key, 58_000),
            batch_size=config.run.batch_size,
            variants_per_class=(
                config.attractor_variants_per_class if config.conditional else 0
            ),
            num_classes=config.num_classes,
            classifier=quality_classifier_model,
        )
    shuffled_prior_quality: Dict[str, float] = {}
    shuffled_prior_settling: Dict[str, Any] = {}
    shuffled_prior_attractor_robustness: Dict[str, float] = {}
    if shuffled_prior_sampler is not None:
        shuffled_prior_generated = sample_generator_images(
            model,
            key=jax.random.fold_in(key, 58_500),
            sample_count=eval_count,
            batch_size=config.run.batch_size,
            labels=eval_labels[:eval_count] if config.conditional else None,
            initial_state_sampler=shuffled_prior_sampler,
        )
        shuffled_prior_quality = compute_generator_quality_metrics(
            eval_images[:eval_count],
            shuffled_prior_generated,
            labels=eval_labels[:eval_count] if config.conditional else None,
            prototypes=prototypes,
            classifier=quality_classifier_model,
            image_shape=config.image_shape,
        )
        shuffled_prior_settling = compute_generator_settling_metrics(
            model,
            key=jax.random.fold_in(key, 59_000),
            real_images=eval_images[:eval_count],
            sample_count=eval_count,
            batch_size=config.run.batch_size,
            settling_steps=config.settling_steps,
            labels=eval_labels[:eval_count] if config.conditional else None,
            prototypes=prototypes,
            classifier=quality_classifier_model,
            image_shape=config.image_shape,
            initial_state_sampler=shuffled_prior_sampler,
        )
        shuffled_prior_attractor_robustness = (
            compute_generator_attractor_robustness(
                model,
                key=jax.random.fold_in(key, 59_500),
                batch_size=config.run.batch_size,
                variants_per_class=(
                    config.attractor_variants_per_class
                    if config.conditional
                    else 0
                ),
                num_classes=config.num_classes,
                classifier=quality_classifier_model,
                initial_state_sampler=shuffled_prior_sampler,
            )
        )
    summary = {
        "final_train_loss": metrics["train_loss"][-1],
        "final_eval_loss": metrics["eval_loss"][-1],
        "final_train_pixel_drift_loss": metrics["train_pixel_drift_loss"][-1],
        "final_eval_pixel_drift_loss": metrics["eval_pixel_drift_loss"][-1],
        "final_train_feature_drift_loss": metrics["train_feature_drift_loss"][-1],
        "final_eval_feature_drift_loss": metrics["eval_feature_drift_loss"][-1],
        "final_train_coarse_auxiliary_loss": metrics[
            "train_coarse_auxiliary_loss"
        ][-1],
        "final_eval_coarse_auxiliary_loss": metrics[
            "eval_coarse_auxiliary_loss"
        ][-1],
        "final_train_coarse_readout_consistency_loss": metrics[
            "train_coarse_readout_consistency_loss"
        ][-1],
        "final_eval_coarse_readout_consistency_loss": metrics[
            "eval_coarse_readout_consistency_loss"
        ][-1],
        "final_train_frequency_objective_loss": metrics[
            "train_frequency_objective_loss"
        ][-1],
        "final_eval_frequency_objective_loss": metrics[
            "eval_frequency_objective_loss"
        ][-1],
        "final_train_patch_objective_loss": metrics[
            "train_patch_objective_loss"
        ][-1],
        "final_eval_patch_objective_loss": metrics[
            "eval_patch_objective_loss"
        ][-1],
        "final_train_state_anchor_loss": metrics[
            "train_state_anchor_loss"
        ][-1],
        "final_train_state_prior_sampling_active": metrics[
            "train_state_prior_sampling_active"
        ][-1],
        "final_state_prior_refit_seconds": metrics[
            "state_prior_refit_seconds"
        ][-1],
        "final_eval_state_anchor_loss": metrics[
            "eval_state_anchor_loss"
        ][-1],
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
            "state_probe_sample_count": config.state_probe_sample_count,
            "state_probe_target_size": config.state_probe_target_size,
            "state_probe_ridge": config.state_probe_ridge,
            "state_fit_sample_count": config.state_fit_sample_count,
            "state_fit_steps": config.state_fit_steps,
            "state_fit_learning_rate": config.state_fit_learning_rate,
            "state_fit_init_scale": config.state_fit_init_scale,
            "state_fit_ridge": config.state_fit_ridge,
            "state_fit_settle_steps": list(config.state_fit_settle_steps),
            "state_anchor_weight": config.state_anchor_weight,
            "state_anchor_steps": list(config.state_anchor_steps),
            "state_anchor_noise_scale": config.state_anchor_noise_scale,
            "state_anchor_mode": config.state_anchor_mode,
            "state_anchor_encoder_kernel_size": (
                config.state_anchor_encoder_kernel_size
            ),
            "state_anchor_occlusion_fraction": (
                config.state_anchor_occlusion_fraction
            ),
            "state_anchor_occlusion_patches": config.state_anchor_occlusion_patches,
            "state_anchor_occlusion_probability": (
                config.state_anchor_occlusion_probability
            ),
            "state_anchor_clean_weight": config.state_anchor_clean_weight,
            "recovery_eval_sample_count": config.recovery_eval_sample_count,
            "recovery_eval_noise_scales": list(config.recovery_eval_noise_scales),
            "recovery_eval_occlusion_fractions": list(
                config.recovery_eval_occlusion_fractions
            ),
            "recovery_eval_occlusion_patches": list(
                config.recovery_eval_occlusion_patches
            ),
            "recovery_eval_settle_steps": list(config.recovery_eval_settle_steps),
            "robustness_eval_sample_count": config.robustness_eval_sample_count,
            "robustness_eval_settle_step": config.robustness_eval_settle_step,
            "robustness_eval_weight_noise_scales": list(
                config.robustness_eval_weight_noise_scales
            ),
            "robustness_eval_quant_bits": list(config.robustness_eval_quant_bits),
            "robustness_eval_occlusion_fractions": list(
                config.robustness_eval_occlusion_fractions
            ),
            "robustness_eval_weight_noise_draws": (
                config.robustness_eval_weight_noise_draws
            ),
            "state_prior_sampling_mode": config.state_prior_sampling_mode,
            "state_prior_rank": config.state_prior_rank,
            "state_prior_noise_scale": config.state_prior_noise_scale,
            "state_prior_refresh_epochs": config.state_prior_refresh_epochs,
            "state_prior_start_epoch": config.state_prior_start_epoch,
            "sample_initialization": (
                "state_prior" if final_prior_sampler is not None else "white_noise"
            ),
            "state_prior_artifacts": final_state_prior_artifacts,
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
            "num_oscillators": int(model.num_oscillators),
            "num_spatial_sites": int(
                getattr(model, "num_spatial_sites", model.num_oscillators)
            ),
            "num_modes": int(getattr(model, "num_modes", 1)),
            "mode_frequency_scales": [
                float(scale)
                for scale in getattr(model, "mode_frequency_scales", ())
            ],
            "mode_coupling_strength": float(
                getattr(model, "mode_coupling_strength", 0.0)
            ),
            "mode_coupling_profile": getattr(
                model,
                "mode_coupling_profile",
                "none",
            ),
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
            "state_residual_readout_strength": float(
                getattr(model, "state_residual_readout_strength", 0.0)
            ),
            "state_residual_readout_init_scale": float(
                getattr(model, "state_residual_readout_init_scale", 0.0)
            ),
            "state_residual_readout_patch_size": int(
                getattr(model, "state_residual_readout_patch_size", 0)
            ),
            "state_residual_readout_sigma": float(
                getattr(model, "state_residual_readout_sigma", 0.0)
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
            "coarse_frequency_scale": float(
                getattr(model, "coarse_frequency_scale", 1.0)
            ),
            "multiscale_layer_sizes": [
                int(size) for size in getattr(model, "multiscale_layer_sizes", ())
            ],
            "multiscale_frequency_scales": [
                float(scale)
                for scale in getattr(model, "multiscale_frequency_scales", ())
            ],
            "multiscale_coupling_profile": getattr(
                model,
                "multiscale_coupling_profile",
                "none",
            ),
            "multiscale_coupling_normalization": getattr(
                model,
                "multiscale_coupling_normalization",
                "none",
            ),
            "multiscale_coupling_length_scale": float(
                getattr(model, "multiscale_coupling_length_scale", 0.0)
            ),
            "multiscale_coupling_floor": float(
                getattr(model, "multiscale_coupling_floor", 0.0)
            ),
            "multiscale_vertical_strength": float(
                getattr(model, "multiscale_vertical_strength", 0.0)
            ),
            "multiscale_feedback_strength": float(
                getattr(model, "multiscale_feedback_strength", 0.0)
            ),
            "multiscale_vertical_profile": getattr(
                model,
                "multiscale_vertical_profile",
                "none",
            ),
            "multiscale_vertical_normalization": getattr(
                model,
                "multiscale_vertical_normalization",
                "none",
            ),
            "multiscale_vertical_length_scale": float(
                getattr(model, "multiscale_vertical_length_scale", 0.0)
            ),
            "multiscale_vertical_floor": float(
                getattr(model, "multiscale_vertical_floor", 0.0)
            ),
            "multiscale_vertical_phase_lag": float(
                getattr(model, "multiscale_vertical_phase_lag", 0.0)
            ),
            "multiscale_feedback_phase_lag": float(
                getattr(model, "multiscale_feedback_phase_lag", 0.0)
            ),
            "multiscale_vertical_signal_scale": float(
                getattr(model, "multiscale_vertical_signal_scale", 1.0)
            ),
            "multiscale_feedback_signal_mode": getattr(
                model,
                "multiscale_feedback_signal_mode",
                "position",
            ),
            "multiscale_feedback_source_gate": getattr(
                model,
                "multiscale_feedback_source_gate",
                "all",
            ),
            "multiscale_feedback_source_mix": [
                float(weight)
                for weight in getattr(
                    model,
                    "multiscale_feedback_source_mix",
                    (),
                )
            ],
            "multiscale_vertical_target_gate": getattr(
                model,
                "multiscale_vertical_target_gate",
                "all",
            ),
            "multiscale_vertical_soft_gate_floor": float(
                getattr(model, "multiscale_vertical_soft_gate_floor", 0.0)
            ),
            "multiscale_vertical_mode": getattr(
                model,
                "multiscale_vertical_mode",
                "additive",
            ),
            "multiscale_vertical_gain_target": getattr(
                model,
                "multiscale_vertical_gain_target",
                "drive",
            ),
            "multiscale_vertical_gain_normalization": getattr(
                model,
                "multiscale_vertical_gain_normalization",
                "none",
            ),
            "multiscale_vertical_gain_target_std": float(
                getattr(model, "multiscale_vertical_gain_target_std", 0.0)
            ),
            "multiscale_vertical_broad_gain_scale": float(
                getattr(model, "multiscale_vertical_broad_gain_scale", 1.0)
            ),
            "multiscale_vertical_selective_gain_scale": float(
                getattr(model, "multiscale_vertical_selective_gain_scale", 1.0)
            ),
            "multiscale_vertical_schedule": getattr(
                model,
                "multiscale_vertical_schedule",
                "constant",
            ),
            "multiscale_vertical_onset_step": int(
                getattr(model, "multiscale_vertical_onset_step", 0)
            ),
            "multiscale_vertical_ramp_steps": int(
                getattr(model, "multiscale_vertical_ramp_steps", 0)
            ),
            "multiscale_conditioning_strength": float(
                getattr(model, "multiscale_conditioning_strength", 0.0)
            ),
            "multiscale_auxiliary_readout_layer": int(
                getattr(model, "multiscale_auxiliary_readout_layer", 0)
            ),
            "multiscale_auxiliary_readout_size": int(
                getattr(model, "multiscale_auxiliary_readout_size", 0)
            ),
            "multiscale_readout_fusion_strength": float(
                getattr(model, "multiscale_readout_fusion_strength", 0.0)
            ),
            "multiscale_readout_gate_mode": getattr(
                model,
                "multiscale_readout_gate_mode",
                "none",
            ),
            "multiscale_readout_gate_strength": float(
                getattr(model, "multiscale_readout_gate_strength", 0.0)
            ),
            "multiscale_readout_gate_init_scale": float(
                getattr(model, "multiscale_readout_gate_init_scale", 0.0)
            ),
            "coarse_auxiliary_weight": config.coarse_auxiliary_weight,
            "coarse_auxiliary_target_size": config.coarse_auxiliary_target_size,
            "coarse_auxiliary_loss_mode": config.coarse_auxiliary_loss_mode,
            "coarse_readout_consistency_weight": (
                config.coarse_readout_consistency_weight
            ),
            "coarse_readout_consistency_onset_epoch": (
                config.coarse_readout_consistency_onset_epoch
            ),
            "frequency_objective_weight": config.frequency_objective_weight,
            "frequency_objective_edge_weight": (
                config.frequency_objective_edge_weight
            ),
            "patch_objective_weight": config.patch_objective_weight,
            "patch_objective_patch_size": config.patch_objective_patch_size,
            "patch_objective_patch_sizes": list(config.patch_objective_patch_sizes),
            "patch_objective_stride": config.patch_objective_stride,
            "patch_objective_offsets": list(config.patch_objective_offsets),
            "patch_objective_projections": config.patch_objective_projections,
            "patch_objective_edge_weight": config.patch_objective_edge_weight,
            "resonant_readout_strength": config.resonant_readout_strength,
            "resonant_readout_init_scale": config.resonant_readout_init_scale,
            "resonant_readout_patch_size": config.resonant_readout_patch_size,
            "resonant_readout_sigma": config.resonant_readout_sigma,
            "num_auxiliary_layers": int(getattr(model, "num_auxiliary_layers", 0)),
            "num_vertical_couplings": int(
                getattr(model, "num_vertical_couplings", 0)
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
            "resize_conv_seed_layout": config.resize_conv_seed_layout,
            "resize_conv_seed_min_channels": config.resize_conv_seed_min_channels,
            "settling": settling,
            "attractor_robustness": attractor_robustness,
            "vertical_intervention_audit": vertical_intervention_audit,
            "white_noise_quality": white_noise_quality,
            "white_noise_settling": white_noise_settling,
            "white_noise_attractor_robustness": white_noise_attractor_robustness,
            "shuffled_prior_quality": shuffled_prior_quality,
            "shuffled_prior_settling": shuffled_prior_settling,
            "shuffled_prior_attractor_robustness": (
                shuffled_prior_attractor_robustness
            ),
            "state_information_probe": state_information_probe,
            "state_fitting_probe": state_fitting_probe,
            "recovery": recovery_metrics,
            "robustness": robustness_metrics,
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
