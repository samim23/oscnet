"""Named local presets for MNIST generator experiments."""

from __future__ import annotations

from typing import Any, Dict

SPARSE_HORN_MNIST_BASE: Dict[str, Any] = {
    "conditional": True,
    "model_family": "horn",
    "label_phase_scale": 0.0,
    "conditioning_mode": "phase_shift",
    "readout_mode": "mean_relative",
    "decoder_mode": "resize_conv",
    "resize_conv_seed_size": 7,
    "resize_conv_upsamples": 2,
    "resize_conv_min_channels": 8,
    "num_oscillators": 196,
    "decoder_hidden_dim": 256,
    "decoder_depth": 0,
    "steps": 16,
    "train_settling_steps": (8, 16, 32),
    "settling_steps": (0, 1, 2, 4, 8, 16, 32),
    "coupling_profile": "local_radius",
    "coupling_length_scale": 0.24,
    "loss_mode": "pixel_drift",
    "pixel_drift_weight": 1.0,
    "feature_drift_weight": 0.0,
    "distributional_weight": 0.0,
    "drift_gamma": 0.2,
    "drift_temperatures": (0.02, 0.05, 0.2),
    "drift_queue_size": 512,
    "drift_queue_num_pos": 32,
    "class_moment_weight": 0.0,
    "prototype_weight": 0.0,
    "epochs": 20,
    "train_limit": 500,
    "eval_limit": 1000,
    "eval_sample_count": 512,
    "batch_size": 128,
    "dt": 0.1,
    "coupling_strength": 1.0,
    "conditioning_strength": 1.0,
    "omega_scale": 0.1,
    "coupling_init_scale": 0.05,
    "horn_frequency": 1.0,
    "horn_damping": 0.15,
    "horn_nonlinearity": 0.05,
    "horn_state_bound": 3.0,
    "num_projections": 256,
    "moment_weight": 0.1,
    "pixel_marginal_weight": 1.0,
    "quality_classifier_epochs": 5,
    "quality_classifier_dim": 128,
    "quality_classifier_depth": 2,
    "output_bias_init": -2.0,
    "artifact_every": 20,
    "checkpoint_every": 20,
    "output_dir": "outputs/reference/mnist_generator_sparse_horn_mnist",
}


def _preset(**overrides: Any) -> Dict[str, Any]:
    values = dict(SPARSE_HORN_MNIST_BASE)
    values.update(overrides)
    return values


GENERATOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "sparse_horn_mnist": _preset(),
    "sparse_horn_mnist_frozen": _preset(
        model_family="frozen_horn",
        output_dir="outputs/reference/mnist_generator_sparse_horn_mnist_frozen",
    ),
    "sparse_horn_mnist_decoder_only": _preset(
        model_family="horn_decoder_only",
        train_settling_steps=(),
        output_dir="outputs/reference/mnist_generator_sparse_horn_mnist_decoder_only",
    ),
    "sparse_horn_mnist_state_mlp": _preset(
        model_family="state_mlp",
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        output_dir="outputs/reference/mnist_generator_sparse_horn_mnist_state_mlp",
    ),
    "sparse_horn_mnist_state_mlp_frozen": _preset(
        model_family="frozen_state_mlp",
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_frozen"
        ),
    ),
    "sparse_horn_mnist_state_mlp_decoder_only": _preset(
        model_family="state_mlp_decoder_only",
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        train_settling_steps=(),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_decoder_only"
        ),
    ),
    "sparse_horn_mnist_step1": _preset(
        steps=1,
        train_settling_steps=(1,),
        settling_steps=(0, 1),
        output_dir="outputs/reference/mnist_generator_sparse_horn_mnist_step1",
    ),
    "sparse_horn_mnist_class_oscillator": _preset(
        conditioning_mode="class_oscillator",
        num_condition_oscillators=32,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_oscillator"
        ),
    ),
    "sparse_horn_mnist_class_oscillator_step1": _preset(
        conditioning_mode="class_oscillator",
        num_condition_oscillators=32,
        steps=1,
        train_settling_steps=(1,),
        settling_steps=(0, 1),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_oscillator_step1"
        ),
    ),
    "sparse_horn_mnist_class_oscillator_frozen": _preset(
        model_family="frozen_horn",
        conditioning_mode="class_oscillator",
        num_condition_oscillators=32,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_oscillator_frozen"
        ),
    ),
    "sparse_horn_mnist_class_coupling": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling"
        ),
    ),
    "sparse_horn_mnist_class_coupling_step1": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        steps=1,
        train_settling_steps=(1,),
        settling_steps=(0, 1),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_step1"
        ),
    ),
    "sparse_horn_mnist_class_coupling_long": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_long"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strong": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=2.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strong"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strength4": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=4.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strength4"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strength8": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strength8"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strong_frozen": _preset(
        model_family="frozen_horn",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=2.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strong_frozen"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strong_decoder_only": _preset(
        model_family="horn_decoder_only",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=2.0,
        steps=32,
        train_settling_steps=(),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strong_decoder_only"
        ),
    ),
    "sparse_horn_mnist_state_mlp_class_coupling_strong": _preset(
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=2.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_class_coupling_strong"
        ),
    ),
    "sparse_horn_mnist_state_mlp_class_coupling_strong_frozen": _preset(
        model_family="frozen_state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=2.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_class_coupling_strong_frozen"
        ),
    ),
    "sparse_horn_mnist_class_coupling_anchor": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        label_phase_scale=0.5,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_anchor"
        ),
    ),
}


def preset_defaults(name: str | None) -> Dict[str, Any]:
    if name in (None, "none"):
        return {}
    try:
        return dict(GENERATOR_PRESETS[name])
    except KeyError as exc:
        options = ", ".join(sorted(GENERATOR_PRESETS))
        raise ValueError(
            f"unknown MNIST generator preset {name!r}; choose one of: {options}"
        ) from exc
