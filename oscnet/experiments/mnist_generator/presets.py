"""Named local presets for MNIST generator experiments."""

from __future__ import annotations

from typing import Any, Dict

RECOMMENDED_GENERATOR_PRESET = "sparse_horn_mnist_recommended"
CURRENT_MNIST_GENERATOR_PRESET = RECOMMENDED_GENERATOR_PRESET
CURRENT_CIFAR10_RGB_GENERATOR_PRESET = "sparse_horn_cifar10_rgb_current"
CURRENT_CIFAR10_RGB_HIERARCHY_PRESET = "sparse_horn_cifar10_rgb_hierarchy_lead"

_CIFAR10_RGB_STABLE_WINNER_PRESET = "sparse_horn_cifar10_rgb_recommended_normlocal"
_CIFAR10_RGB_HIERARCHY_LEAD_PRESET = (
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
    "signed_gain_conditioning_vscale30_center_feedback_state_mix50_50"
)

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
    "sparse_horn_mnist_class_coupling_strength8_dist001": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.01,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strength8_dist001"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strength8_dist0025": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.025,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strength8_dist0025"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strength8_dist005": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.05,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strength8_dist005"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strength8_freq13": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_frequency=1.3,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strength8_freq13"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strength8_damp030": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_strength8_damp030"
        ),
    ),
    "sparse_horn_mnist_class_coupling_strength8_freq13_dist0025": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_frequency=1.3,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.025,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_class_coupling_"
            "strength8_freq13_dist0025"
        ),
    ),
    "sparse_horn_mnist_recommended": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir="outputs/reference/mnist_generator_sparse_horn_mnist_recommended",
    ),
    "sparse_horn_mnist_strict": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir="outputs/reference/mnist_generator_sparse_horn_mnist_strict",
    ),
    "sparse_horn_mnist_quality": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.025,
        output_dir="outputs/reference/mnist_generator_sparse_horn_mnist_quality",
    ),
    "sparse_horn_mnist_dynamics_quality": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_dynamics_quality"
        ),
    ),
    "sparse_horn_mnist_dynamics_quality_dist001": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.01,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_dynamics_quality_dist001"
        ),
    ),
    "sparse_horn_mnist_dynamics_quality_dist0025": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.025,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_dynamics_quality_dist0025"
        ),
    ),
    "sparse_horn_mnist_dynamics_quality_dist005": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.05,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_dynamics_quality_dist005"
        ),
    ),
    "sparse_horn_mnist_recommended_no_main_coupling": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_strength=0.0,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_recommended_no_main_coupling"
        ),
    ),
    "sparse_horn_mnist_recommended_frozen_recurrent": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        train_recurrent_dynamics=False,
        train_conditioning_dynamics=True,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_recommended_frozen_recurrent"
        ),
    ),
    "sparse_horn_mnist_recommended_frozen_conditioning": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        train_recurrent_dynamics=True,
        train_conditioning_dynamics=False,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_recommended_frozen_conditioning"
        ),
    ),
    "sparse_horn_mnist_recommended_frozen": _preset(
        model_family="frozen_horn",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_recommended_frozen"
        ),
    ),
    "sparse_horn_mnist_recommended_decoder_only": _preset(
        model_family="horn_decoder_only",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        train_settling_steps=(),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_recommended_decoder_only"
        ),
    ),
    "sparse_horn_mnist_recommended_step1": _preset(
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=1,
        train_settling_steps=(1,),
        settling_steps=(0, 1),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_recommended_step1"
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
    "sparse_horn_mnist_state_mlp_class_coupling_strength8": _preset(
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_class_coupling_strength8"
        ),
    ),
    "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist005": _preset(
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.05,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_class_coupling_"
            "strength8_dist005"
        ),
    ),
    "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01": _preset(
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.1,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_class_coupling_"
            "strength8_dist01"
        ),
    ),
    "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01_class": _preset(
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.1,
        class_moment_weight=1.0,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_class_coupling_"
            "strength8_dist01_class"
        ),
    ),
    "sparse_horn_fashion_mnist_recommended": _preset(
        dataset_name="fashion_mnist",
        data_source="idx",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir="outputs/reference/mnist_generator_sparse_horn_fashion_mnist",
    ),
    "sparse_horn_fashion_mnist_state_mlp_strength8": _preset(
        dataset_name="fashion_mnist",
        data_source="idx",
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_fashion_mnist_state_mlp_strength8"
        ),
    ),
    "sparse_horn_fashion_mnist_state_mlp_strength8_dist005": _preset(
        dataset_name="fashion_mnist",
        data_source="idx",
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.05,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_fashion_mnist_state_mlp_strength8_dist005"
        ),
    ),
    "sparse_horn_fashion_mnist_recommended_ch16": _preset(
        dataset_name="fashion_mnist",
        data_source="idx",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        resize_conv_min_channels=16,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_fashion_mnist_ch16"
        ),
    ),
    "sparse_horn_fashion_mnist_state_mlp_strength8_ch16": _preset(
        dataset_name="fashion_mnist",
        data_source="idx",
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        resize_conv_min_channels=16,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_fashion_mnist_state_mlp_strength8_ch16"
        ),
    ),
    "sparse_horn_fashion_mnist_recommended_dist0025": _preset(
        dataset_name="fashion_mnist",
        data_source="idx",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.025,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_fashion_mnist_dist0025"
        ),
    ),
    "sparse_horn_fashion_mnist_recommended_dist005": _preset(
        dataset_name="fashion_mnist",
        data_source="idx",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        distributional_weight=0.05,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_fashion_mnist_dist005"
        ),
    ),
    "sparse_horn_cifar10_gray_recommended": _preset(
        dataset_name="cifar10_gray",
        data_source="idx",
        image_shape=(32, 32),
        resize_conv_seed_size=8,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=1000,
        eval_limit=1000,
        output_dir="outputs/reference/mnist_generator_sparse_horn_cifar10_gray",
    ),
    "sparse_horn_cifar10_gray_recommended_dist005": _preset(
        dataset_name="cifar10_gray",
        data_source="idx",
        image_shape=(32, 32),
        resize_conv_seed_size=8,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=1000,
        eval_limit=1000,
        distributional_weight=0.05,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_gray_dist005"
        ),
    ),
    "sparse_horn_cifar10_gray_state_mlp_strength8": _preset(
        dataset_name="cifar10_gray",
        data_source="idx",
        image_shape=(32, 32),
        resize_conv_seed_size=8,
        num_oscillators=256,
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_gray_state_mlp_strength8"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir="outputs/reference/mnist_generator_sparse_horn_cifar10_rgb",
    ),
    "sparse_horn_cifar10_rgb_recommended_drive025": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_drive025"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_normlocal": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_normlocal"
        ),
    ),
    "sparse_horn_cifar10_rgb_coarse16_normlocal": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="coarse_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        num_coarse_oscillators=16,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_strength=1.0,
        coarse_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_coarse16_normlocal"
        ),
    ),
    "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="coarse_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        num_coarse_oscillators=16,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_strength=0.25,
        coarse_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_coarse16_normlocal_gentle"
        ),
    ),
    "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_dist050": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="coarse_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        num_coarse_oscillators=16,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_strength=0.25,
        coarse_to_fine_profile="distance_decay",
        coarse_to_fine_normalization="row_sum",
        coarse_to_fine_length_scale=0.5,
        coarse_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_dist050"
        ),
    ),
    "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="coarse_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        num_coarse_oscillators=16,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_strength=0.25,
        coarse_to_fine_profile="local_radius",
        coarse_to_fine_normalization="row_sum",
        coarse_to_fine_length_scale=0.5,
        coarse_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050"
        ),
    ),
    "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_ch32": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="coarse_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=32,
        num_oscillators=256,
        num_coarse_oscillators=16,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_strength=0.25,
        coarse_to_fine_profile="local_radius",
        coarse_to_fine_normalization="row_sum",
        coarse_to_fine_length_scale=0.5,
        coarse_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_coarse16_normlocal_"
            "gentle_local050_ch32"
        ),
    ),
    "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_dist0025": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="coarse_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        num_coarse_oscillators=16,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_strength=0.25,
        coarse_to_fine_profile="local_radius",
        coarse_to_fine_normalization="row_sum",
        coarse_to_fine_length_scale=0.5,
        coarse_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        distributional_weight=0.025,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_coarse16_normlocal_"
            "gentle_local050_dist0025"
        ),
    ),
    "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_ch32_dist0025": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="coarse_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=32,
        num_oscillators=256,
        num_coarse_oscillators=16,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_strength=0.25,
        coarse_to_fine_profile="local_radius",
        coarse_to_fine_normalization="row_sum",
        coarse_to_fine_length_scale=0.5,
        coarse_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        distributional_weight=0.025,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_coarse16_normlocal_"
            "gentle_local050_ch32_dist0025"
        ),
    ),
    "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_feedback050": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="coarse_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        num_coarse_oscillators=16,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_strength=0.25,
        coarse_to_fine_profile="local_radius",
        coarse_to_fine_normalization="row_sum",
        coarse_to_fine_length_scale=0.5,
        coarse_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        output_feedback_mode="state_proxy",
        output_feedback_strength=0.5,
        output_feedback_init_scale=0.05,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_coarse16_normlocal_"
            "gentle_local050_feedback050"
        ),
    ),
    "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_ch32_dist0025_feedback050": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="coarse_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=32,
        num_oscillators=256,
        num_coarse_oscillators=16,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_strength=0.25,
        coarse_to_fine_profile="local_radius",
        coarse_to_fine_normalization="row_sum",
        coarse_to_fine_length_scale=0.5,
        coarse_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        output_feedback_mode="state_proxy",
        output_feedback_strength=0.5,
        output_feedback_init_scale=0.05,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        distributional_weight=0.025,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_coarse16_normlocal_"
            "gentle_local050_ch32_dist0025_feedback050"
        ),
    ),
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="multiscale_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        multiscale_layer_sizes=(16, 64),
        multiscale_frequency_scales=(0.45, 0.75),
        multiscale_coupling_profile="local_radius",
        multiscale_coupling_normalization="row_sum",
        multiscale_coupling_length_scale=0.5,
        multiscale_vertical_strength=0.25,
        multiscale_feedback_strength=0.05,
        multiscale_vertical_profile="local_radius",
        multiscale_vertical_normalization="row_sum",
        multiscale_vertical_length_scale=0.5,
        multiscale_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005"
        ),
    ),
    "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        model_family="multiscale_horn",
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        multiscale_layer_sizes=(16, 64),
        multiscale_frequency_scales=(0.45, 0.75),
        multiscale_coupling_profile="local_radius",
        multiscale_coupling_normalization="row_sum",
        multiscale_coupling_length_scale=0.5,
        multiscale_vertical_strength=0.0,
        multiscale_feedback_strength=0.0,
        multiscale_vertical_profile="local_radius",
        multiscale_vertical_normalization="row_sum",
        multiscale_vertical_length_scale=0.5,
        multiscale_conditioning_strength=1.0,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_normalization="row_sum",
        main_coupling_strength=1.0,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=0.25,
        feature_drift_mode="learned",
        learned_feature_kind="residual_conv",
        learned_feature_epochs=10,
        learned_feature_dim=256,
        learned_feature_depth=3,
        quality_classifier_kind="residual_conv",
        quality_classifier_train_limit=10000,
        quality_classifier_eval_limit=5000,
        quality_classifier_epochs=15,
        quality_classifier_dim=256,
        quality_classifier_depth=3,
        attractor_variants_per_class=8,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        train_limit=2000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
            "no_vertical"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_drive025_spatial_grid": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        conditioning_target_pattern="spatial_grid",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_drive025_spatial_grid"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_drive025_center_block": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        conditioning_target_pattern="center_block",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_drive025_center_block"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_drive010": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.10,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_drive010"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_dist005": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        distributional_weight=0.05,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_dist005"
        ),
    ),
    "sparse_horn_cifar10_rgb_state_mlp_strength8": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        model_family="state_mlp",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        state_mlp_hidden_dim=48,
        state_mlp_depth=1,
        state_mlp_residual_scale=0.1,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_state_mlp_strength8"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_step1": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=1,
        train_settling_steps=(1,),
        settling_steps=(0, 1),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_step1"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_frozen_recurrent": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        train_recurrent_dynamics=False,
        train_conditioning_dynamics=True,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_frozen_recurrent"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_frozen_conditioning": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        train_recurrent_dynamics=True,
        train_conditioning_dynamics=False,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_frozen_conditioning"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_no_main_interaction": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_init_scale=0.0,
        train_recurrent_dynamics=False,
        train_conditioning_dynamics=True,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_no_main_interaction"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_init_scale=0.0,
        train_recurrent_dynamics=False,
        train_conditioning_dynamics=True,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_no_main_interaction_drive025"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025_spatial_grid": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        conditioning_target_pattern="spatial_grid",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_init_scale=0.0,
        train_recurrent_dynamics=False,
        train_conditioning_dynamics=True,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_no_main_interaction_"
            "drive025_spatial_grid"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025_center_block": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        conditioning_target_pattern="center_block",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_init_scale=0.0,
        train_recurrent_dynamics=False,
        train_conditioning_dynamics=True,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_no_main_interaction_"
            "drive025_center_block"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive010": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.10,
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        coupling_init_scale=0.0,
        train_recurrent_dynamics=False,
        train_conditioning_dynamics=True,
        steps=32,
        train_settling_steps=(16, 32, 48),
        settling_steps=(0, 1, 8, 16, 32, 48, 64),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_no_main_interaction_drive010"
        ),
    ),
    "sparse_horn_cifar10_rgb_recommended_decoder_only": _preset(
        dataset_name="cifar10_rgb",
        data_source="idx",
        image_shape=(32, 32, 3),
        resize_conv_seed_size=8,
        resize_conv_min_channels=16,
        num_oscillators=256,
        model_family="horn_decoder_only",
        conditioning_mode="class_coupling",
        num_condition_oscillators=32,
        conditioning_strength=8.0,
        horn_damping=0.30,
        steps=32,
        train_settling_steps=(),
        settling_steps=(0,),
        quality_classifier_kind="conv",
        train_limit=1000,
        eval_limit=1000,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_cifar10_rgb_decoder_only"
        ),
    ),
    "sparse_horn_mnist_state_mlp_class_coupling_strong_dist005": _preset(
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
        distributional_weight=0.05,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_class_coupling_"
            "strong_dist005"
        ),
    ),
    "sparse_horn_mnist_state_mlp_class_coupling_strong_dist01": _preset(
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
        distributional_weight=0.1,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_class_coupling_"
            "strong_dist01"
        ),
    ),
    "sparse_horn_mnist_state_mlp_class_coupling_strong_dist01_class": _preset(
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
        distributional_weight=0.1,
        class_moment_weight=1.0,
        output_dir=(
            "outputs/reference/"
            "mnist_generator_sparse_horn_mnist_state_mlp_class_coupling_"
            "strong_dist01_class"
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


GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
] = {
    **GENERATOR_PRESETS["sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005"],
    "coarse_auxiliary_weight": 0.05,
    "coarse_auxiliary_target_size": 8,
    "multiscale_auxiliary_readout_layer": 0,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8"
] = {
    **GENERATOR_PRESETS["sparse_horn_cifar10_rgb_multiscale16_64_no_vertical"],
    "coarse_auxiliary_weight": 0.05,
    "coarse_auxiliary_target_size": 8,
    "multiscale_auxiliary_readout_layer": 0,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "no_vertical_auxlow8"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
    ],
    "coarse_auxiliary_loss_mode": "distributional",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxdist8"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8"
    ],
    "coarse_auxiliary_loss_mode": "distributional",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "no_vertical_auxdist8"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_vgate_conditioning"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
    ],
    "multiscale_vertical_target_gate": "conditioning",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_vgate_conditioning"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_vgate_non_conditioning"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
    ],
    "multiscale_vertical_target_gate": "non_conditioning",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_vgate_non_conditioning"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_all"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
    ],
    "multiscale_vertical_mode": "gain_modulation",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_all"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_conditioning"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
    ],
    "multiscale_vertical_mode": "gain_modulation",
    "multiscale_vertical_target_gate": "conditioning",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_conditioning"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_all"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
    ],
    "multiscale_vertical_mode": "signed_gain",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_all"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_conditioning"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
    ],
    "multiscale_vertical_mode": "signed_gain",
    "multiscale_vertical_target_gate": "conditioning",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8_drive2"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8"
    ],
    "conditioning_strength": 2.0,
    "multiscale_conditioning_strength": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "no_vertical_auxlow8_drive2"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_vgate_conditioning_drive2"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_vgate_conditioning"
    ],
    "conditioning_strength": 2.0,
    "multiscale_conditioning_strength": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_vgate_conditioning_drive2"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_all_drive2"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_all"
    ],
    "conditioning_strength": 2.0,
    "multiscale_conditioning_strength": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_all_drive2"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_conditioning_drive2"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_conditioning"
    ],
    "conditioning_strength": 2.0,
    "multiscale_conditioning_strength": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_conditioning_drive2"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_all_drive2"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_all"
    ],
    "conditioning_strength": 2.0,
    "multiscale_conditioning_strength": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_all_drive2"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_conditioning_drive2"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning"
    ],
    "conditioning_strength": 2.0,
    "multiscale_conditioning_strength": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_drive2"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_conditioning_soft025"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_conditioning"
    ],
    "multiscale_vertical_soft_gate_floor": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_conditioning_soft025"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_conditioning_soft025"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning"
    ],
    "multiscale_vertical_soft_gate_floor": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_soft025"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_conditioning_soft025_drive2"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_conditioning_soft025"
    ],
    "conditioning_strength": 2.0,
    "multiscale_conditioning_strength": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_conditioning_soft025_drive2"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_conditioning_soft025_drive2"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_soft025"
    ],
    "conditioning_strength": 2.0,
    "multiscale_conditioning_strength": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_soft025_drive2"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_all_vscale10"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_all"
    ],
    "multiscale_vertical_signal_scale": 10.0,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_all_vscale10"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_all_vscale30"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_all"
    ],
    "multiscale_vertical_signal_scale": 30.0,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_all_vscale30"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_conditioning_vscale10"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning"
    ],
    "multiscale_vertical_signal_scale": 10.0,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale10"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_conditioning_vscale30"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning"
    ],
    "multiscale_vertical_signal_scale": 30.0,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale30"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_conditioning_vscale30_center"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale30"
    ],
    "multiscale_vertical_gain_normalization": "center",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale30_center"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_conditioning_vscale30_center_feedback_state"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_"
        "signed_gain_conditioning_vscale30_center"
    ],
    "multiscale_feedback_signal_mode": "state",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale30_"
        "center_feedback_state"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_"
        "signed_gain_conditioning_vscale30_center"
    ],
    "coarse_auxiliary_loss_mode": "distributional",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center"
    ],
    "multiscale_feedback_signal_mode": "state",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_source_conditioning"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state"
    ],
    "multiscale_feedback_source_gate": "conditioning",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_source_conditioning"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_source_non_conditioning"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state"
    ],
    "multiscale_feedback_source_gate": "non_conditioning",
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_source_non_conditioning"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_mix75_25"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state"
    ],
    "multiscale_feedback_source_gate": "weighted",
    "multiscale_feedback_source_mix": (0.75, 0.25),
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_mix75_25"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_mix75_25_fusion010"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state_mix75_25"
    ],
    "multiscale_readout_fusion_strength": 0.10,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_mix75_25_fusion010"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_mix75_25_consistency005"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state_mix75_25"
    ],
    "coarse_readout_consistency_weight": 0.05,
    "coarse_readout_consistency_onset_epoch": 5,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_mix75_25_consistency005"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_mix50_50"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state"
    ],
    "multiscale_feedback_source_gate": "weighted",
    "multiscale_feedback_source_mix": (0.5, 0.5),
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_mix50_50"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_mix50_50_consistency005"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state_mix50_50"
    ],
    "coarse_readout_consistency_weight": 0.05,
    "coarse_readout_consistency_onset_epoch": 5,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_mix50_50_consistency005"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_mix50_50_consistency010"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state_mix50_50"
    ],
    "coarse_readout_consistency_weight": 0.10,
    "coarse_readout_consistency_onset_epoch": 5,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_mix50_50_consistency010"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_mix50_50_fusion010"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state_mix50_50"
    ],
    "multiscale_readout_fusion_strength": 0.10,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_mix50_50_fusion010"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_mix50_50_fusion025"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state_mix50_50"
    ],
    "multiscale_readout_fusion_strength": 0.25,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_mix50_50_fusion025"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center_feedback_state_mix25_75"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8_"
        "signed_gain_conditioning_vscale30_center_feedback_state"
    ],
    "multiscale_feedback_source_gate": "weighted",
    "multiscale_feedback_source_mix": (0.25, 0.75),
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
        "center_feedback_state_mix25_75"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_dual_gain_conditioning_vscale10"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
    ],
    "multiscale_vertical_mode": "dual_gain",
    "multiscale_vertical_target_gate": "conditioning",
    "multiscale_vertical_signal_scale": 10.0,
    "multiscale_vertical_broad_gain_scale": 1.0,
    "multiscale_vertical_selective_gain_scale": 1.0,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_dual_gain_conditioning_vscale10"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_dual_gain_conditioning_vscale30"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8"
    ],
    "multiscale_vertical_mode": "dual_gain",
    "multiscale_vertical_target_gate": "conditioning",
    "multiscale_vertical_signal_scale": 30.0,
    "multiscale_vertical_broad_gain_scale": 1.0,
    "multiscale_vertical_selective_gain_scale": 1.0,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_dual_gain_conditioning_vscale30"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_gain_all_vscale30_normstd015"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_all_vscale30"
    ],
    "multiscale_vertical_gain_normalization": "center_rms",
    "multiscale_vertical_gain_target_std": 0.015,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_gain_all_vscale30_normstd015"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_signed_gain_conditioning_vscale30_normstd015"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale30"
    ],
    "multiscale_vertical_gain_normalization": "center_rms",
    "multiscale_vertical_gain_target_std": 0.015,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale30_normstd015"
    ),
}

GENERATOR_PRESETS[
    "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8_dual_gain_conditioning_vscale30_normstd015"
] = {
    **GENERATOR_PRESETS[
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_dual_gain_conditioning_vscale30"
    ],
    "multiscale_vertical_gain_normalization": "center_rms",
    "multiscale_vertical_gain_target_std": 0.015,
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_dual_gain_conditioning_vscale30_normstd015"
    ),
}

GENERATOR_PRESETS[CURRENT_CIFAR10_RGB_GENERATOR_PRESET] = {
    **GENERATOR_PRESETS[_CIFAR10_RGB_STABLE_WINNER_PRESET],
    "output_dir": "outputs/reference/mnist_generator_sparse_horn_cifar10_rgb_current",
}

GENERATOR_PRESETS[CURRENT_CIFAR10_RGB_HIERARCHY_PRESET] = {
    **GENERATOR_PRESETS[_CIFAR10_RGB_HIERARCHY_LEAD_PRESET],
    "output_dir": (
        "outputs/reference/"
        "mnist_generator_sparse_horn_cifar10_rgb_hierarchy_lead"
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
