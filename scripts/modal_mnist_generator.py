"""Run OscNet MNIST oscillator-generator experiments on Modal GPUs."""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "oscnet-mnist-generator"
VOLUME_NAME = os.environ.get("OSCNET_MODAL_VOLUME", "oscnet-runs")
VOLUME_MOUNT = Path("/mnt/oscnet-runs")
GPU = os.environ.get("OSCNET_MODAL_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("OSCNET_MODAL_TIMEOUT_SECONDS", "10800"))
JAX_PACKAGE = os.environ.get("OSCNET_MODAL_JAX", "jax[cuda13]")
MAX_CONTAINERS = int(os.environ.get("OSCNET_MODAL_MAX_CONTAINERS", "1"))

DEFAULT_SMOKE_ARGS = (
    "--data-source synthetic --model-family kuramoto --seed 0 --epochs 1 "
    "--train-limit 8 --eval-limit 4 --eval-sample-count 4 --batch-size 4 "
    "--num-oscillators 8 --decoder-hidden-dim 12 --decoder-depth 1 "
    "--steps 1 --num-projections 8 --artifact-every 1 --checkpoint-every 1"
)

SWEEP_CSVS = {
    "mnist_generator_core": Path(
        "outputs/analysis/modal_mnist_generator_core.csv"
    ),
    "mnist_generator_conditional_core": Path(
        "outputs/analysis/modal_mnist_generator_conditional_core.csv"
    ),
    "mnist_generator_un0_coupled_core": Path(
        "outputs/analysis/modal_mnist_generator_un0_coupled_core.csv"
    ),
    "mnist_generator_spatial_readout_core": Path(
        "outputs/analysis/modal_mnist_generator_spatial_readout_core.csv"
    ),
    "mnist_generator_local_readout_core": Path(
        "outputs/analysis/modal_mnist_generator_local_readout_core.csv"
    ),
    "mnist_generator_spatial_coupling_core": Path(
        "outputs/analysis/modal_mnist_generator_spatial_coupling_core.csv"
    ),
    "mnist_generator_trainability_attribution_core": Path(
        "outputs/analysis/modal_mnist_generator_trainability_attribution_core.csv"
    ),
    "mnist_generator_unconditional_local_readout_core": Path(
        "outputs/analysis/modal_mnist_generator_unconditional_local_readout_core.csv"
    ),
    "mnist_generator_resize_conv_core": Path(
        "outputs/analysis/modal_mnist_generator_resize_conv_core.csv"
    ),
    "mnist_generator_resize_conv_pixel_drift_core": Path(
        "outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_core.csv"
    ),
    "mnist_generator_resize_conv_pixel_drift_queue_core": Path(
        "outputs/analysis/"
        "modal_mnist_generator_resize_conv_pixel_drift_queue_core.csv"
    ),
    "mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core": Path(
        "outputs/analysis/"
        "modal_mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core.csv"
    ),
    "mnist_generator_resize_conv_pixel_drift_queue_distributional_core": Path(
        "outputs/analysis/"
        "modal_mnist_generator_resize_conv_pixel_drift_queue_distributional_core.csv"
    ),
    "mnist_generator_resize_conv_feature_drift_core": Path(
        "outputs/analysis/modal_mnist_generator_resize_conv_feature_drift_core.csv"
    ),
    "mnist_generator_resize_conv_learned_feature_drift_core": Path(
        "outputs/analysis/"
        "modal_mnist_generator_resize_conv_learned_feature_drift_core.csv"
    ),
    "mnist_generator_resize_conv_learned_feature_drift_queue_core": Path(
        "outputs/analysis/"
        "modal_mnist_generator_resize_conv_learned_feature_drift_queue_core.csv"
    ),
    "mnist_generator_horn_resize_conv_core": Path(
        "outputs/analysis/modal_mnist_generator_horn_resize_conv_core.csv"
    ),
    "mnist_generator_horn_conditioning_attribution_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_horn_conditioning_attribution_probe.csv"
    ),
    "mnist_generator_horn_label0_replication_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_horn_label0_replication_probe.csv"
    ),
    "mnist_generator_state_mlp_label0_control_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_state_mlp_label0_control_probe.csv"
    ),
    "mnist_generator_horn_state_mlp_low_data_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_horn_state_mlp_low_data_probe.csv"
    ),
    "mnist_generator_horn_state_mlp_settling_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_horn_state_mlp_settling_probe.csv"
    ),
    "mnist_generator_horn_settling_train_probe": Path(
        "outputs/analysis/modal_mnist_generator_horn_settling_train_probe.csv"
    ),
    "mnist_generator_horn_structured_coupling_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_horn_structured_coupling_probe.csv"
    ),
    "mnist_generator_horn_sparse_coupling_probe": Path(
        "outputs/analysis/modal_mnist_generator_horn_sparse_coupling_probe.csv"
    ),
    "mnist_generator_sparse_horn_state_mlp_replication_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_state_mlp_replication_probe.csv"
    ),
    "mnist_generator_sparse_horn_attribution_probe": Path(
        "outputs/analysis/modal_mnist_generator_sparse_horn_attribution_probe.csv"
    ),
    "mnist_generator_sparse_horn_conditioning_route_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_conditioning_route_probe.csv"
    ),
    "mnist_generator_sparse_horn_class_coupling_sharpen_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_class_coupling_sharpen_probe.csv"
    ),
    "mnist_generator_sparse_horn_class_coupling_strong_control_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_class_coupling_strong_control_probe.csv"
    ),
    "mnist_generator_sparse_horn_class_coupling_strength_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_class_coupling_strength_probe.csv"
    ),
    "mnist_generator_sparse_horn_state_mlp_diversity_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_state_mlp_diversity_probe.csv"
    ),
    "mnist_generator_sparse_horn_distributional_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_distributional_probe.csv"
    ),
    "mnist_generator_sparse_horn_quality_classifier_audit": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_quality_classifier_audit.csv"
    ),
    "mnist_generator_sparse_horn_dynamics_quality_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_dynamics_quality_probe.csv"
    ),
    "mnist_generator_sparse_horn_damping_distributional_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_damping_distributional_probe.csv"
    ),
    "mnist_generator_sparse_horn_recommended_ablation_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_recommended_ablation_probe.csv"
    ),
    "mnist_generator_sparse_horn_state_mlp_strength8_diversity_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_sparse_horn_state_mlp_strength8_diversity_probe.csv"
    ),
    "mnist_generator_fashion_mnist_frontier_probe": Path(
        "outputs/analysis/modal_mnist_generator_fashion_mnist_frontier_probe.csv"
    ),
    "mnist_generator_fashion_mnist_readout_capacity_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_fashion_mnist_readout_capacity_probe.csv"
    ),
    "mnist_generator_fashion_mnist_horn_calibration_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_fashion_mnist_horn_calibration_probe.csv"
    ),
    "mnist_generator_cifar10_gray_frontier_probe": Path(
        "outputs/analysis/modal_mnist_generator_cifar10_gray_frontier_probe.csv"
    ),
    "mnist_generator_cifar10_gray_convjudge_frontier_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_gray_convjudge_frontier_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_frontier_probe": Path(
        "outputs/analysis/modal_mnist_generator_cifar10_rgb_frontier_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_feature_metric_audit": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_feature_metric_audit.csv"
    ),
    "mnist_generator_cifar10_rgb_judge_audit": Path(
        "outputs/analysis/modal_mnist_generator_cifar10_rgb_judge_audit.csv"
    ),
    "mnist_generator_cifar10_rgb_semantic_feature_drift_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_semantic_feature_drift_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_semantic_feature_drift_attribution": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_semantic_feature_drift_attribution.csv"
    ),
    "mnist_generator_cifar10_rgb_attractor_robustness_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_attractor_robustness_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat.csv"
    ),
    "mnist_generator_cifar10_rgb_main_coupling_strength_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_main_coupling_strength_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat.csv"
    ),
    "mnist_generator_cifar10_rgb_main_coupling_fine_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_main_coupling_fine_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_main_coupling_current_replication": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_main_coupling_current_replication.csv"
    ),
    "mnist_generator_cifar10_rgb_normalized_distance_decay_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_normalized_distance_decay_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_normalized_local_radius_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_normalized_local_radius_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_normalized_local_radius_sweep": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_normalized_local_radius_sweep.csv"
    ),
    "mnist_generator_cifar10_rgb_coarse_to_fine_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coarse_to_fine_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_coarse_to_fine_gain_sweep": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coarse_to_fine_gain_sweep.csv"
    ),
    "mnist_generator_cifar10_rgb_coarse_to_fine_profile_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coarse_to_fine_profile_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_audit": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_audit.csv"
    ),
    "mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat.csv"
    ),
    "mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_coarse_to_fine_feedback_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coarse_to_fine_feedback_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_multiscale_layered_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_multiscale_layered_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_multiscale_auxiliary_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_multiscale_auxiliary_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_multiscale_gated_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_multiscale_gated_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_multiscale_gain_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_multiscale_gain_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_multiscale_weak_drive_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_multiscale_weak_drive_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_multiscale_signed_gain_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_multiscale_signed_gain_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_multiscale_soft_gate_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_multiscale_soft_gate_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_vertical_causality_audit": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_vertical_causality_audit.csv"
    ),
    "mnist_generator_cifar10_rgb_vertical_calibration_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_vertical_calibration_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_dual_gain_probe": Path(
        "outputs/analysis/modal_mnist_generator_cifar10_rgb_dual_gain_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_vertical_homeostasis_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_vertical_homeostasis_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration.csv"
    ),
    "mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_centered_signed_gain_target_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_centered_signed_gain_target_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_coarse_objective_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coarse_objective_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_feedback_signal_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_feedback_signal_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_feedback_source_gate_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_feedback_source_gate_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_feedback_source_mix_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_feedback_source_mix_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_feedback_source_mix_auxfix_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_feedback_source_mix_auxfix_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_readout_fusion_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_readout_fusion_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_coarse_readout_consistency_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coarse_readout_consistency_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_readout_gate_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_readout_gate_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_frequency_objective_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_frequency_objective_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_patch_objective_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_patch_objective_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_patch_v2_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_patch_v2_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_state_information_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_state_information_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_state_residual_readout_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_state_residual_readout_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_state_residual_longer_pilot": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_state_residual_longer_pilot.csv"
    ),
    "mnist_generator_cifar10_rgb_resonant_readout_pilot": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_resonant_readout_pilot.csv"
    ),
    "mnist_generator_cifar10_rgb_capacity_probe": Path(
        "outputs/analysis/modal_mnist_generator_cifar10_rgb_capacity_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_multimode_probe": Path(
        "outputs/analysis/modal_mnist_generator_cifar10_rgb_multimode_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_retinotopic_readout_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_retinotopic_readout_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_retinotopic_control_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_retinotopic_control_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_state_anchor_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_state_anchor_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_state_prior_training_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_state_prior_training_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1.csv"
    ),
    "mnist_generator_cifar10_rgb_state_prior_control_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_state_prior_control_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_attribution_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_attribution_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_sparse_drive_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_sparse_drive_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_sparse_drive_seed_repeat": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_sparse_drive_seed_repeat.csv"
    ),
    "mnist_generator_cifar10_rgb_structured_drive_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_structured_drive_probe.csv"
    ),
    "mnist_generator_cifar10_rgb_coherent_drive_probe": Path(
        "outputs/analysis/"
        "modal_mnist_generator_cifar10_rgb_coherent_drive_probe.csv"
    ),
}

REMOTE_PACKAGES = [
    JAX_PACKAGE,
    "equinox>=0.10.0",
    "diffrax>=0.4.0",
    "optax>=0.1.7",
    "numpy>=1.20.0",
    "matplotlib>=3.5.0",
    "networkx>=2.8.0",
]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*REMOTE_PACKAGES)
    .env(
        {
            "MPLBACKEND": "Agg",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_COMPILATION_CACHE_DIR": str(VOLUME_MOUNT / "jax_cache"),
        }
    )
    .add_local_python_source("oscnet")
)


def _safe_run_name(name: str) -> str:
    name = name.strip() or time.strftime("mnist-generator-%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip(".-") or "mnist-generator-run"


def _has_arg(args: list[str], flag: str) -> bool:
    return flag in args or any(arg.startswith(f"{flag}=") for arg in args)


def _with_default_arg(args: list[str], flag: str, value: str | Path) -> list[str]:
    if _has_arg(args, flag):
        return args
    return [*args, flag, str(value)]


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _summary_metric(summary: dict[str, Any], dotted_key: str) -> Any:
    value: Any = summary
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _mnist_generator_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        ("kuramoto", ["--model-family kuramoto"]),
        ("frozen_kuramoto", ["--model-family frozen_kuramoto"]),
        ("decoder_only", ["--model-family decoder_only"]),
    ]
    common = [
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 256",
        "--decoder-hidden-dim 256",
        "--decoder-depth 2",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n256_h256_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_conditional_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        ("kuramoto", ["--model-family kuramoto"]),
        ("frozen_kuramoto", ["--model-family frozen_kuramoto"]),
        ("decoder_only", ["--model-family decoder_only"]),
    ]
    common = [
        "--conditional",
        "--num-classes 10",
        "--label-phase-scale 0.75",
        "--class-moment-weight 0.5",
        "--prototype-weight 0.2",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 256",
        "--decoder-hidden-dim 256",
        "--decoder-depth 2",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_conditional_"
                f"{suffix}_n256_h256_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_un0_coupled_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "class_coupled_kuramoto",
            [
                "--model-family kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
        (
            "class_coupled_frozen",
            [
                "--model-family frozen_kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
        (
            "phase_shift_decoder",
            [
                "--model-family decoder_only",
                "--conditioning-mode phase_shift",
                "--num-condition-oscillators 0",
            ],
        ),
    ]
    common = [
        "--conditional",
        "--num-classes 10",
        "--label-phase-scale 1.0",
        "--readout-mode relative",
        "--class-moment-weight 0.5",
        "--prototype-weight 0.2",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 256",
        "--decoder-hidden-dim 256",
        "--decoder-depth 2",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_un0_"
                f"{suffix}_n256_h256_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_spatial_readout_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "spatial_class_coupled_kuramoto",
            [
                "--model-family kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
        (
            "spatial_class_coupled_frozen",
            [
                "--model-family frozen_kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
        (
            "spatial_phase_shift_decoder",
            [
                "--model-family decoder_only",
                "--conditioning-mode phase_shift",
                "--num-condition-oscillators 0",
            ],
        ),
    ]
    common = [
        "--conditional",
        "--num-classes 10",
        "--label-phase-scale 1.0",
        "--readout-mode relative",
        "--decoder-mode spatial_basis",
        "--spatial-basis-sigma 0.0",
        "--class-moment-weight 0.5",
        "--prototype-weight 0.2",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 256",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n256_basis_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_local_readout_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "local_class_coupled_kuramoto",
            [
                "--model-family kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
        (
            "local_class_coupled_frozen",
            [
                "--model-family frozen_kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
        (
            "local_phase_shift_decoder",
            [
                "--model-family decoder_only",
                "--conditioning-mode phase_shift",
                "--num-condition-oscillators 0",
            ],
        ),
    ]
    common = [
        "--conditional",
        "--num-classes 10",
        "--label-phase-scale 1.0",
        "--readout-mode relative",
        "--decoder-mode local_basis",
        "--local-patch-size 5",
        "--spatial-basis-sigma 0.0",
        "--class-moment-weight 0.5",
        "--prototype-weight 0.2",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 256",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n256_local5_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_spatial_coupling_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "spatial_coupled_kuramoto",
            [
                "--model-family kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
        (
            "spatial_coupled_frozen",
            [
                "--model-family frozen_kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
        (
            "spatial_phase_shift_decoder",
            [
                "--model-family decoder_only",
                "--conditioning-mode phase_shift",
                "--num-condition-oscillators 0",
            ],
        ),
    ]
    common = [
        "--conditional",
        "--num-classes 10",
        "--label-phase-scale 1.0",
        "--readout-mode relative",
        "--decoder-mode local_basis",
        "--local-patch-size 5",
        "--spatial-basis-sigma 0.0",
        "--coupling-profile distance_decay",
        "--coupling-length-scale 0.35",
        "--coupling-floor 0.05",
        "--coupling-bias-strength 0.05",
        "--class-moment-weight 0.5",
        "--prototype-weight 0.2",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 256",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n256_local5_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_trainability_attribution_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "attrib_all_trained",
            [
                "--model-family kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
        (
            "attrib_conditioning_only",
            [
                "--model-family kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
                "--no-train-recurrent-dynamics",
                "--train-conditioning-dynamics",
            ],
        ),
        (
            "attrib_recurrent_only",
            [
                "--model-family kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
                "--train-recurrent-dynamics",
                "--no-train-conditioning-dynamics",
            ],
        ),
        (
            "attrib_frozen",
            [
                "--model-family frozen_kuramoto",
                "--conditioning-mode class_coupling",
                "--num-condition-oscillators 32",
            ],
        ),
    ]
    common = [
        "--conditional",
        "--num-classes 10",
        "--label-phase-scale 1.0",
        "--readout-mode relative",
        "--decoder-mode local_basis",
        "--local-patch-size 5",
        "--spatial-basis-sigma 0.0",
        "--class-moment-weight 0.5",
        "--prototype-weight 0.2",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 256",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n256_local5_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_unconditional_local_readout_core_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("uncond_local_kuramoto", ["--model-family kuramoto"]),
        ("uncond_local_frozen", ["--model-family frozen_kuramoto"]),
        ("uncond_local_decoder", ["--model-family decoder_only"]),
    ]
    common = [
        "--conditioning-mode none",
        "--readout-mode relative",
        "--decoder-mode local_basis",
        "--local-patch-size 5",
        "--spatial-basis-sigma 0.0",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 256",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n256_local5_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_resize_conv_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        ("resize_conv_kuramoto", ["--model-family kuramoto"]),
        ("resize_conv_frozen", ["--model-family frozen_kuramoto"]),
        ("resize_conv_decoder", ["--model-family decoder_only"]),
    ]
    common = [
        "--conditional",
        "--conditioning-mode class_coupling",
        "--num-condition-oscillators 8",
        "--readout-mode relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--class-moment-weight 0.5",
        "--prototype-weight 0.2",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n196_resizeconv_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_resize_conv_pixel_drift_core_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("resize_conv_drift_kuramoto", ["--model-family kuramoto"]),
        ("resize_conv_drift_frozen", ["--model-family frozen_kuramoto"]),
        ("resize_conv_drift_decoder", ["--model-family decoder_only"]),
    ]
    common = [
        "--conditional",
        "--conditioning-mode class_coupling",
        "--num-condition-oscillators 8",
        "--readout-mode relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n196_resizeconv_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_resize_conv_feature_drift_core_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("resize_conv_feature_kuramoto", ["--model-family kuramoto"]),
        ("resize_conv_feature_frozen", ["--model-family frozen_kuramoto"]),
        ("resize_conv_feature_decoder", ["--model-family decoder_only"]),
    ]
    common = [
        "--conditional",
        "--conditioning-mode class_coupling",
        "--num-condition-oscillators 8",
        "--readout-mode relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_feature_drift",
        "--pixel-drift-weight 0.5",
        "--feature-drift-weight 1.0",
        "--feature-drift-mode structural",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n196_resizeconv_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_resize_conv_pixel_drift_queue_core_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("resize_conv_drift_queue_kuramoto", ["--model-family kuramoto"]),
        ("resize_conv_drift_queue_frozen", ["--model-family frozen_kuramoto"]),
        ("resize_conv_drift_queue_decoder", ["--model-family decoder_only"]),
    ]
    common = [
        "--conditional",
        "--conditioning-mode class_coupling",
        "--num-condition-oscillators 8",
        "--readout-mode relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n196_resizeconv_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("resize_conv_drift_queue_un0_condition_kuramoto", ["--model-family kuramoto"]),
        (
            "resize_conv_drift_queue_un0_condition_frozen",
            ["--model-family frozen_kuramoto"],
        ),
        (
            "resize_conv_drift_queue_un0_condition_decoder",
            ["--model-family decoder_only"],
        ),
    ]
    common = [
        "--conditional",
        "--conditioning-mode class_oscillator",
        "--num-condition-oscillators 8",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n196_resizeconv_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_resize_conv_learned_feature_drift_core_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("resize_conv_learned_feature_kuramoto", ["--model-family kuramoto"]),
        ("resize_conv_learned_feature_frozen", ["--model-family frozen_kuramoto"]),
        ("resize_conv_learned_feature_decoder", ["--model-family decoder_only"]),
    ]
    common = [
        "--conditional",
        "--conditioning-mode class_coupling",
        "--num-condition-oscillators 8",
        "--readout-mode relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_feature_drift",
        "--pixel-drift-weight 0.5",
        "--feature-drift-weight 1.0",
        "--feature-drift-mode learned",
        "--learned-feature-epochs 5",
        "--learned-feature-dim 128",
        "--learned-feature-depth 2",
        "--learned-feature-learning-rate 0.001",
        "--learned-feature-weight-decay 0.0001",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n196_resizeconv_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_resize_conv_pixel_drift_queue_distributional_core_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("kuramoto", ["--model-family kuramoto"]),
        ("frozen", ["--model-family frozen_kuramoto"]),
        ("decoder", ["--model-family decoder_only"]),
    ]
    regularizers = [
        ("dist005", "0.005"),
        ("dist01", "0.01"),
    ]
    common = [
        "--conditional",
        "--conditioning-mode class_coupling",
        "--num-condition-oscillators 8",
        "--readout-mode relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for reg_suffix, reg_weight in regularizers:
        for seed in (11, 12):
            for variant, model_args in variants:
                suffix = f"resize_conv_drift_queue_{reg_suffix}_{variant}"
                run_name = (
                    "mnist_generator_"
                    f"{suffix}_n196_resizeconv_steps16_train5000_seed{seed}_20e"
                )
                args = shlex.split(
                    " ".join(
                        [
                            f"--seed {seed}",
                            *common,
                            f"--distributional-weight {reg_weight}",
                            *model_args,
                        ]
                    )
                )
                output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
                args = _with_default_arg(args, "--output-dir", output_dir)
                entries.append((args, run_name))
    return entries


def _mnist_generator_resize_conv_learned_feature_drift_queue_core_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        (
            "resize_conv_learned_feature_queue_kuramoto",
            ["--model-family kuramoto"],
        ),
        (
            "resize_conv_learned_feature_queue_frozen",
            ["--model-family frozen_kuramoto"],
        ),
        (
            "resize_conv_learned_feature_queue_decoder",
            ["--model-family decoder_only"],
        ),
    ]
    common = [
        "--conditional",
        "--conditioning-mode class_coupling",
        "--num-condition-oscillators 8",
        "--readout-mode relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_feature_drift",
        "--pixel-drift-weight 0.5",
        "--feature-drift-weight 1.0",
        "--feature-drift-mode learned",
        "--learned-feature-epochs 5",
        "--learned-feature-dim 128",
        "--learned-feature-depth 2",
        "--learned-feature-learning-rate 0.001",
        "--learned-feature-weight-decay 0.0001",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n196_resizeconv_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_horn_resize_conv_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        ("horn", ["--model-family horn"]),
        ("frozen_horn", ["--model-family frozen_horn"]),
        ("horn_decoder", ["--model-family horn_decoder_only"]),
        ("kuramoto", ["--model-family kuramoto"]),
        ("frozen_kuramoto", ["--model-family frozen_kuramoto"]),
        ("kuramoto_decoder", ["--model-family decoder_only"]),
    ]
    common = [
        "--conditional",
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-frequency 1.0",
        "--horn-damping 0.15",
        "--horn-nonlinearity 0.05",
        "--horn-state-bound 3.0",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_generator_"
                f"{suffix}_n196_resizeconv_steps16_train5000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_horn_conditioning_attribution_probe_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("horn", ["--model-family horn"]),
        ("horn_decoder", ["--model-family horn_decoder_only"]),
        ("frozen_horn", ["--model-family frozen_horn"]),
    ]
    conditioning_variants = [
        ("label0_train", ["--conditional", "--label-phase-scale 0.0"]),
        ("label01_train", ["--conditional", "--label-phase-scale 0.1"]),
        ("label05_train", ["--conditional", "--label-phase-scale 0.5"]),
    ]
    common = [
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-frequency 1.0",
        "--horn-damping 0.15",
        "--horn-nonlinearity 0.05",
        "--horn-state-bound 3.0",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--quality-classifier-epochs 5",
        "--quality-classifier-dim 128",
        "--quality-classifier-depth 2",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11,):
        for condition_suffix, condition_args in conditioning_variants:
            for model_suffix, model_args in variants:
                run_name = (
                    "mnist_generator_horn_conditioning_"
                    f"{condition_suffix}_{model_suffix}_"
                    f"n196_resizeconv_steps16_train5000_seed{seed}_20e"
                )
                args = shlex.split(
                    " ".join(
                        [
                            f"--seed {seed}",
                            *common,
                            *condition_args,
                            *model_args,
                        ]
                    )
                )
                output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
                args = _with_default_arg(args, "--output-dir", output_dir)
                entries.append((args, run_name))
    return entries


def _mnist_generator_horn_label0_replication_probe_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("horn", ["--model-family horn"]),
        ("horn_decoder", ["--model-family horn_decoder_only"]),
        ("frozen_horn", ["--model-family frozen_horn"]),
    ]
    common = [
        "--conditional",
        "--label-phase-scale 0.0",
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-frequency 1.0",
        "--horn-damping 0.15",
        "--horn-nonlinearity 0.05",
        "--horn-state-bound 3.0",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--quality-classifier-epochs 5",
        "--quality-classifier-dim 128",
        "--quality-classifier-depth 2",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (12, 13):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_generator_horn_label0_replication_"
                f"{model_suffix}_n196_resizeconv_steps16_train5000_"
                f"seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_state_mlp_label0_control_probe_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("state_mlp", ["--model-family state_mlp"]),
        ("state_mlp_decoder", ["--model-family state_mlp_decoder_only"]),
        ("frozen_state_mlp", ["--model-family frozen_state_mlp"]),
    ]
    common = [
        "--conditional",
        "--label-phase-scale 0.0",
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 5000",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-state-bound 3.0",
        "--state-mlp-hidden-dim 48",
        "--state-mlp-depth 1",
        "--state-mlp-residual-scale 0.1",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--quality-classifier-epochs 5",
        "--quality-classifier-dim 128",
        "--quality-classifier-depth 2",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12, 13):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_generator_state_mlp_label0_control_"
                f"{model_suffix}_n196_resizeconv_steps16_train5000_"
                f"seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_horn_state_mlp_low_data_probe_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("horn", ["--model-family horn"]),
        ("state_mlp", ["--model-family state_mlp"]),
    ]
    common = [
        "--conditional",
        "--label-phase-scale 0.0",
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 500",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-frequency 1.0",
        "--horn-damping 0.15",
        "--horn-nonlinearity 0.05",
        "--horn-state-bound 3.0",
        "--state-mlp-hidden-dim 48",
        "--state-mlp-depth 1",
        "--state-mlp-residual-scale 0.1",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--quality-classifier-epochs 5",
        "--quality-classifier-dim 128",
        "--quality-classifier-depth 2",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12, 13):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_generator_horn_state_mlp_low_data_"
                f"{model_suffix}_n196_resizeconv_steps16_train500_"
                f"seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_horn_state_mlp_settling_probe_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("horn", ["--model-family horn"]),
        ("state_mlp", ["--model-family state_mlp"]),
    ]
    common = [
        "--conditional",
        "--label-phase-scale 0.0",
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 500",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--settling-steps 0,1,2,4,8,16,32",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-frequency 1.0",
        "--horn-damping 0.15",
        "--horn-nonlinearity 0.05",
        "--horn-state-bound 3.0",
        "--state-mlp-hidden-dim 48",
        "--state-mlp-depth 1",
        "--state-mlp-residual-scale 0.1",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--quality-classifier-epochs 5",
        "--quality-classifier-dim 128",
        "--quality-classifier-depth 2",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for model_suffix, model_args in variants:
        run_name = (
            "mnist_generator_horn_state_mlp_settling_"
            f"{model_suffix}_n196_resizeconv_steps16_train500_seed11_20e"
        )
        args = shlex.split(" ".join(["--seed 11", *common, *model_args]))
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_horn_settling_train_probe_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        ("baseline", []),
        ("train_steps_8_16_32", ["--train-settling-steps 8,16,32"]),
    ]
    common = [
        "--model-family horn",
        "--conditional",
        "--label-phase-scale 0.0",
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 500",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--settling-steps 0,1,2,4,8,16,32",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-frequency 1.0",
        "--horn-damping 0.15",
        "--horn-nonlinearity 0.05",
        "--horn-state-bound 3.0",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--quality-classifier-epochs 5",
        "--quality-classifier-dim 128",
        "--quality-classifier-depth 2",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for variant_suffix, variant_args in variants:
        run_name = (
            "mnist_generator_horn_settling_train_"
            f"{variant_suffix}_n196_resizeconv_steps16_train500_seed11_20e"
        )
        args = shlex.split(" ".join(["--seed 11", *common, *variant_args]))
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_horn_structured_coupling_probe_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        (
            "decay_l035_f002",
            [
                "--coupling-profile distance_decay",
                "--coupling-length-scale 0.35",
                "--coupling-floor 0.02",
            ],
        ),
        (
            "decay_l060_f002",
            [
                "--coupling-profile distance_decay",
                "--coupling-length-scale 0.60",
                "--coupling-floor 0.02",
            ],
        ),
    ]
    common = [
        "--model-family horn",
        "--conditional",
        "--label-phase-scale 0.0",
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 500",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--train-settling-steps 8,16,32",
        "--settling-steps 0,1,2,4,8,16,32",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-frequency 1.0",
        "--horn-damping 0.15",
        "--horn-nonlinearity 0.05",
        "--horn-state-bound 3.0",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--quality-classifier-epochs 5",
        "--quality-classifier-dim 128",
        "--quality-classifier-depth 2",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for variant_suffix, variant_args in variants:
        run_name = (
            "mnist_generator_horn_structured_coupling_"
            f"{variant_suffix}_n196_resizeconv_steps16_train500_seed11_20e"
        )
        args = shlex.split(" ".join(["--seed 11", *common, *variant_args]))
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_horn_sparse_coupling_probe_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "local_r024",
            [
                "--coupling-profile local_radius",
                "--coupling-length-scale 0.24",
            ],
        ),
        (
            "local_r035",
            [
                "--coupling-profile local_radius",
                "--coupling-length-scale 0.35",
            ],
        ),
    ]
    common = [
        "--model-family horn",
        "--conditional",
        "--label-phase-scale 0.0",
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 500",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--train-settling-steps 8,16,32",
        "--settling-steps 0,1,2,4,8,16,32",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-frequency 1.0",
        "--horn-damping 0.15",
        "--horn-nonlinearity 0.05",
        "--horn-state-bound 3.0",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--quality-classifier-epochs 5",
        "--quality-classifier-dim 128",
        "--quality-classifier-depth 2",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for variant_suffix, variant_args in variants:
        run_name = (
            "mnist_generator_horn_sparse_coupling_"
            f"{variant_suffix}_n196_resizeconv_steps16_train500_seed11_20e"
        )
        args = shlex.split(" ".join(["--seed 11", *common, *variant_args]))
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_state_mlp_replication_probe_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        (
            "sparse_horn_r024",
            [
                "--model-family horn",
                "--coupling-profile local_radius",
                "--coupling-length-scale 0.24",
            ],
        ),
        (
            "state_mlp",
            [
                "--model-family state_mlp",
                "--state-mlp-hidden-dim 48",
                "--state-mlp-depth 1",
                "--state-mlp-residual-scale 0.1",
            ],
        ),
    ]
    common = [
        "--conditional",
        "--label-phase-scale 0.0",
        "--conditioning-mode phase_shift",
        "--readout-mode mean_relative",
        "--decoder-mode resize_conv",
        "--resize-conv-seed-size 7",
        "--resize-conv-upsamples 2",
        "--resize-conv-min-channels 8",
        "--loss-mode pixel_drift",
        "--pixel-drift-weight 1.0",
        "--feature-drift-weight 0.0",
        "--distributional-weight 0.0",
        "--drift-gamma 0.2",
        "--drift-temperatures 0.02,0.05,0.2",
        "--drift-queue-size 512",
        "--drift-queue-num-pos 32",
        "--class-moment-weight 0.0",
        "--prototype-weight 0.0",
        "--epochs 20",
        "--train-limit 500",
        "--eval-limit 1000",
        "--eval-sample-count 512",
        "--batch-size 128",
        "--num-oscillators 196",
        "--decoder-hidden-dim 256",
        "--decoder-depth 0",
        "--steps 16",
        "--train-settling-steps 8,16,32",
        "--settling-steps 0,1,2,4,8,16,32",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.1",
        "--coupling-init-scale 0.05",
        "--horn-frequency 1.0",
        "--horn-damping 0.15",
        "--horn-nonlinearity 0.05",
        "--horn-state-bound 3.0",
        "--num-projections 256",
        "--moment-weight 0.1",
        "--pixel-marginal-weight 1.0",
        "--quality-classifier-epochs 5",
        "--quality-classifier-dim 128",
        "--quality-classifier-depth 2",
        "--output-bias-init -2.0",
        "--artifact-every 20",
        "--checkpoint-every 20",
    ]
    for seed in (11, 12, 13):
        for variant_suffix, variant_args in variants:
            run_name = (
                "mnist_generator_sparse_horn_state_mlp_replication_"
                f"{variant_suffix}_n196_resizeconv_steps16_train500_"
                f"seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *variant_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_attribution_probe_sweep() -> list[
    tuple[list[str], str]
]:
    entries = []
    variants = [
        ("horn", "sparse_horn_mnist"),
        ("frozen_horn", "sparse_horn_mnist_frozen"),
        ("horn_decoder", "sparse_horn_mnist_decoder_only"),
        ("horn_step1", "sparse_horn_mnist_step1"),
        ("state_mlp", "sparse_horn_mnist_state_mlp"),
        ("frozen_state_mlp", "sparse_horn_mnist_state_mlp_frozen"),
        ("state_mlp_decoder", "sparse_horn_mnist_state_mlp_decoder_only"),
    ]
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_attribution_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_conditioning_route_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Probe whether HORN still works when labels enter only through dynamics."""

    entries = []
    variants = [
        ("phase_shift_full", "sparse_horn_mnist"),
        ("phase_shift_step1", "sparse_horn_mnist_step1"),
        ("class_osc_full", "sparse_horn_mnist_class_oscillator"),
        ("class_osc_step1", "sparse_horn_mnist_class_oscillator_step1"),
        ("class_osc_frozen", "sparse_horn_mnist_class_oscillator_frozen"),
        ("class_coupling_full", "sparse_horn_mnist_class_coupling"),
        ("class_coupling_step1", "sparse_horn_mnist_class_coupling_step1"),
    ]
    for seed in (11,):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_conditioning_route_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_class_coupling_sharpen_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Try to sharpen class coupling without reintroducing phase-shift labels."""

    entries = []
    variants = [
        ("baseline", "sparse_horn_mnist_class_coupling"),
        ("long", "sparse_horn_mnist_class_coupling_long"),
        ("strong", "sparse_horn_mnist_class_coupling_strong"),
        ("anchor", "sparse_horn_mnist_class_coupling_anchor"),
    ]
    for seed in (11,):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_class_coupling_sharpen_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_class_coupling_strong_control_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Replicate strong class coupling against matched dynamics controls."""

    entries = []
    variants = [
        ("horn_strong", "sparse_horn_mnist_class_coupling_strong"),
        ("horn_frozen", "sparse_horn_mnist_class_coupling_strong_frozen"),
        ("horn_decoder", "sparse_horn_mnist_class_coupling_strong_decoder_only"),
        ("state_mlp", "sparse_horn_mnist_state_mlp_class_coupling_strong"),
        (
            "state_mlp_frozen",
            "sparse_horn_mnist_state_mlp_class_coupling_strong_frozen",
        ),
    ]
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_class_coupling_strong_control_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_class_coupling_strength_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Probe whether stronger HORN class drive fixes seed instability."""

    entries = []
    variants = [
        ("strength2", "sparse_horn_mnist_class_coupling_strong"),
        ("strength4", "sparse_horn_mnist_class_coupling_strength4"),
        ("strength8", "sparse_horn_mnist_class_coupling_strength8"),
    ]
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_class_coupling_strength_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_state_mlp_diversity_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Test whether distributional pressure gives StateMLP HORN-like diversity."""

    entries = []
    variants = [
        ("horn_strength8", "sparse_horn_mnist_class_coupling_strength8"),
        ("state_mlp", "sparse_horn_mnist_state_mlp_class_coupling_strong"),
        (
            "state_mlp_dist005",
            "sparse_horn_mnist_state_mlp_class_coupling_strong_dist005",
        ),
        (
            "state_mlp_dist01",
            "sparse_horn_mnist_state_mlp_class_coupling_strong_dist01",
        ),
        (
            "state_mlp_dist01_class",
            "sparse_horn_mnist_state_mlp_class_coupling_strong_dist01_class",
        ),
    ]
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_state_mlp_diversity_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_state_mlp_strength8_diversity_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Test whether strength-8 StateMLP can recover HORN-like diversity."""

    entries = []
    variants = [
        ("horn_recommended", "sparse_horn_mnist_recommended"),
        ("state_mlp_strength8", "sparse_horn_mnist_state_mlp_class_coupling_strength8"),
        (
            "state_mlp_strength8_dist005",
            "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist005",
        ),
        (
            "state_mlp_strength8_dist01",
            "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01",
        ),
        (
            "state_mlp_strength8_dist01_class",
            "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01_class",
        ),
    ]
    classifier_args = (
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_state_mlp_strength8_diversity_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_distributional_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Test whether small distributional pressure improves strict HORN quality."""

    entries = []
    variants = [
        ("strength8", "sparse_horn_mnist_class_coupling_strength8"),
        ("strength8_dist001", "sparse_horn_mnist_class_coupling_strength8_dist001"),
        ("strength8_dist0025", "sparse_horn_mnist_class_coupling_strength8_dist0025"),
        ("strength8_dist005", "sparse_horn_mnist_class_coupling_strength8_dist005"),
    ]
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_distributional_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_quality_classifier_audit_sweep() -> list[
    tuple[list[str], str]
]:
    """Re-score leading strict generators with a stronger MNIST quality classifier."""

    entries = []
    variants = [
        ("horn_strength8", "sparse_horn_mnist_class_coupling_strength8"),
        ("horn_strength8_dist0025", "sparse_horn_mnist_class_coupling_strength8_dist0025"),
        ("state_mlp", "sparse_horn_mnist_state_mlp_class_coupling_strong"),
    ]
    classifier_args = (
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_quality_classifier_audit_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_dynamics_quality_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Probe HORN frequency/damping quality knobs with the stronger evaluator."""

    entries = []
    variants = [
        ("strength8", "sparse_horn_mnist_class_coupling_strength8"),
        ("strength8_dist0025", "sparse_horn_mnist_class_coupling_strength8_dist0025"),
        ("strength8_freq13", "sparse_horn_mnist_class_coupling_strength8_freq13"),
        ("strength8_damp030", "sparse_horn_mnist_class_coupling_strength8_damp030"),
        (
            "strength8_freq13_dist0025",
            "sparse_horn_mnist_class_coupling_strength8_freq13_dist0025",
        ),
    ]
    classifier_args = (
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_dynamics_quality_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_damping_distributional_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Test whether damping and small distributional pressure compound."""

    entries = []
    variants = [
        ("strict", "sparse_horn_mnist_strict"),
        ("quality_dist0025", "sparse_horn_mnist_quality"),
        ("damp030", "sparse_horn_mnist_dynamics_quality"),
        ("damp030_dist001", "sparse_horn_mnist_dynamics_quality_dist001"),
        ("damp030_dist0025", "sparse_horn_mnist_dynamics_quality_dist0025"),
        ("damp030_dist005", "sparse_horn_mnist_dynamics_quality_dist005"),
    ]
    classifier_args = (
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_damping_distributional_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_sparse_horn_recommended_ablation_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Ablate the recommended HORN route without changing the readout/loss."""

    entries = []
    variants = [
        ("recommended", "sparse_horn_mnist_recommended"),
        ("no_main_coupling", "sparse_horn_mnist_recommended_no_main_coupling"),
        ("frozen_recurrent", "sparse_horn_mnist_recommended_frozen_recurrent"),
        (
            "frozen_conditioning",
            "sparse_horn_mnist_recommended_frozen_conditioning",
        ),
        ("frozen_all", "sparse_horn_mnist_recommended_frozen"),
        ("decoder_only", "sparse_horn_mnist_recommended_decoder_only"),
        ("step1", "sparse_horn_mnist_recommended_step1"),
        (
            "state_mlp_strength8",
            "sparse_horn_mnist_state_mlp_class_coupling_strength8",
        ),
    ]
    classifier_args = (
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_sparse_horn_recommended_ablation_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_fashion_mnist_frontier_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """First non-MNIST HORN-vs-StateMLP frontier probe."""

    entries = []
    variants = [
        ("horn_recommended", "sparse_horn_fashion_mnist_recommended"),
        ("state_mlp_strength8", "sparse_horn_fashion_mnist_state_mlp_strength8"),
        (
            "state_mlp_strength8_dist005",
            "sparse_horn_fashion_mnist_state_mlp_strength8_dist005",
        ),
    ]
    classifier_args = (
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_fashion_mnist_frontier_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_fashion_mnist_readout_capacity_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Check whether Fashion-MNIST HORN quality is readout-width limited."""

    entries = []
    variants = [
        ("horn_ch16", "sparse_horn_fashion_mnist_recommended_ch16"),
        ("state_mlp_strength8_ch16", "sparse_horn_fashion_mnist_state_mlp_strength8_ch16"),
    ]
    classifier_args = (
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_fashion_mnist_readout_capacity_"
                f"{variant_suffix}_n196_resizeconv16_train500_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_fashion_mnist_horn_calibration_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Check whether small distributional pressure improves Fashion HORN quality."""

    entries = []
    variants = [
        ("horn_dist0025", "sparse_horn_fashion_mnist_recommended_dist0025"),
        ("horn_dist005", "sparse_horn_fashion_mnist_recommended_dist005"),
    ]
    classifier_args = (
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_fashion_mnist_horn_calibration_"
                f"{variant_suffix}_n196_resizeconv_train500_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_gray_frontier_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """First 32x32 natural-image grayscale HORN-vs-control frontier probe."""

    entries = []
    variants = [
        ("horn_recommended", "sparse_horn_cifar10_gray_recommended"),
        ("horn_dist005", "sparse_horn_cifar10_gray_recommended_dist005"),
        ("state_mlp_strength8", "sparse_horn_cifar10_gray_state_mlp_strength8"),
    ]
    classifier_args = (
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_gray_frontier_"
                f"{variant_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_gray_convjudge_frontier_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Rerun CIFAR-gray frontier with a convolutional quality classifier."""

    entries = []
    variants = [
        ("horn_recommended", "sparse_horn_cifar10_gray_recommended"),
        ("horn_dist005", "sparse_horn_cifar10_gray_recommended_dist005"),
        ("state_mlp_strength8", "sparse_horn_cifar10_gray_state_mlp_strength8"),
    ]
    classifier_args = (
        "--quality-classifier-kind conv "
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_gray_convjudge_frontier_"
                f"{variant_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_frontier_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """First full-color CIFAR-10 HORN-vs-control frontier probe."""

    entries = []
    variants = [
        ("horn_recommended", "sparse_horn_cifar10_rgb_recommended"),
        ("horn_dist005", "sparse_horn_cifar10_rgb_recommended_dist005"),
        ("state_mlp_strength8", "sparse_horn_cifar10_rgb_state_mlp_strength8"),
    ]
    classifier_args = (
        "--quality-classifier-kind conv "
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 12, 13):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_frontier_"
                f"{variant_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_feature_metric_audit_sweep() -> list[
    tuple[list[str], str]
]:
    """One-seed RGB frontier audit with classifier feature-space metrics."""

    entries = []
    variants = [
        ("horn_recommended", "sparse_horn_cifar10_rgb_recommended"),
        ("horn_dist005", "sparse_horn_cifar10_rgb_recommended_dist005"),
        ("state_mlp_strength8", "sparse_horn_cifar10_rgb_state_mlp_strength8"),
    ]
    classifier_args = (
        "--quality-classifier-kind conv "
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    seed = 11
    for variant_suffix, local_preset in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_feature_metrics_"
            f"{variant_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
        )
        args = shlex.split(
            f"--seed {seed} --preset {local_preset} {classifier_args}"
        )
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_judge_audit_sweep() -> list[
    tuple[list[str], str]
]:
    """Compare old conv and stronger residual-conv CIFAR RGB quality judges."""

    entries = []
    variants = [
        ("horn_prefix025", "sparse_horn_cifar10_rgb_recommended_drive025"),
        (
            "horn_no_main_prefix025",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025",
        ),
    ]
    classifier_variants = [
        (
            "convjudge",
            "--quality-classifier-kind conv "
            "--quality-classifier-train-limit 10000 "
            "--quality-classifier-eval-limit 5000 "
            "--quality-classifier-epochs 15 "
            "--quality-classifier-dim 256 "
            "--quality-classifier-depth 3",
        ),
        (
            "resconvjudge",
            "--quality-classifier-kind residual_conv "
            "--quality-classifier-train-limit 10000 "
            "--quality-classifier-eval-limit 5000 "
            "--quality-classifier-epochs 15 "
            "--quality-classifier-dim 256 "
            "--quality-classifier-depth 3",
        ),
    ]
    seed = 11
    for variant_suffix, local_preset in variants:
        for judge_suffix, classifier_args in classifier_variants:
            run_name = (
                "mnist_generator_cifar10_rgb_judge_audit_"
                f"{variant_suffix}_{judge_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_semantic_feature_drift_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Ask whether residual-conv feature drift improves strict CIFAR semantics."""

    entries = []
    variants = [
        ("horn_pixel", []),
        (
            "horn_resfeat025",
            [
                "--loss-mode pixel_feature_drift",
                "--pixel-drift-weight 0.5",
                "--feature-drift-weight 0.25",
                "--feature-drift-mode learned",
                "--learned-feature-kind residual_conv",
                "--learned-feature-epochs 10",
                "--learned-feature-dim 256",
                "--learned-feature-depth 3",
            ],
        ),
        (
            "horn_resfeat100",
            [
                "--loss-mode pixel_feature_drift",
                "--pixel-drift-weight 0.5",
                "--feature-drift-weight 1.0",
                "--feature-drift-mode learned",
                "--learned-feature-kind residual_conv",
                "--learned-feature-epochs 10",
                "--learned-feature-dim 256",
                "--learned-feature-depth 3",
            ],
        ),
    ]
    common = [
        "--preset sparse_horn_cifar10_rgb_recommended_drive025",
        "--train-limit 2000",
        "--eval-limit 1000",
        "--quality-classifier-kind residual_conv",
        "--quality-classifier-train-limit 10000",
        "--quality-classifier-eval-limit 5000",
        "--quality-classifier-epochs 15",
        "--quality-classifier-dim 256",
        "--quality-classifier-depth 3",
    ]
    seed = 11
    for variant_suffix, variant_args in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_semantic_feature_drift_"
            f"{variant_suffix}_n256_resizeconv_train2000_seed{seed}_20e"
        )
        args = shlex.split(
            " ".join([f"--seed {seed}", *common, *variant_args])
        )
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_semantic_feature_drift_attribution_sweep() -> list[
    tuple[list[str], str]
]:
    """Repeat residual feature drift against no-main controls across seeds."""

    entries = []
    feature_args = [
        "--loss-mode pixel_feature_drift",
        "--pixel-drift-weight 0.5",
        "--feature-drift-weight 0.25",
        "--feature-drift-mode learned",
        "--learned-feature-kind residual_conv",
        "--learned-feature-epochs 10",
        "--learned-feature-dim 256",
        "--learned-feature-depth 3",
    ]
    variants = [
        ("horn_pixel", "sparse_horn_cifar10_rgb_recommended_drive025", []),
        (
            "horn_resfeat025",
            "sparse_horn_cifar10_rgb_recommended_drive025",
            feature_args,
        ),
        (
            "horn_no_main_pixel",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025",
            [],
        ),
        (
            "horn_no_main_resfeat025",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025",
            feature_args,
        ),
    ]
    common = [
        "--train-limit 2000",
        "--eval-limit 1000",
        "--quality-classifier-kind residual_conv",
        "--quality-classifier-train-limit 10000",
        "--quality-classifier-eval-limit 5000",
        "--quality-classifier-epochs 15",
        "--quality-classifier-dim 256",
        "--quality-classifier-depth 3",
    ]
    for seed in (11, 23):
        for variant_suffix, local_preset, variant_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_semantic_feature_drift_attribution_"
                f"{variant_suffix}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--preset {local_preset}",
                        *common,
                        *variant_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_attractor_robustness_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Compact strict-judge attractor-basin probe for current RGB candidates."""

    return _mnist_generator_cifar10_rgb_attractor_robustness_entries(
        seeds=(11,),
        name_prefix="mnist_generator_cifar10_rgb_attractor_robustness",
    )


def _mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat_sweep() -> list[
    tuple[list[str], str]
]:
    """Two-seed repeat of the compact attractor-basin RGB probe."""

    return _mnist_generator_cifar10_rgb_attractor_robustness_entries(
        seeds=(11, 23),
        name_prefix="mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat",
    )


def _mnist_generator_cifar10_rgb_attractor_robustness_entries(
    *,
    seeds: tuple[int, ...],
    name_prefix: str,
) -> list[tuple[list[str], str]]:
    """Build strict-judge attractor-basin probe entries."""

    entries = []
    feature_args = [
        "--loss-mode pixel_feature_drift",
        "--pixel-drift-weight 0.5",
        "--feature-drift-weight 0.25",
        "--feature-drift-mode learned",
        "--learned-feature-kind residual_conv",
        "--learned-feature-epochs 10",
        "--learned-feature-dim 256",
        "--learned-feature-depth 3",
    ]
    variants = [
        (
            "horn_resfeat025",
            "sparse_horn_cifar10_rgb_recommended_drive025",
        ),
        (
            "horn_no_main_resfeat025",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025",
        ),
        (
            "state_mlp_resfeat025",
            "sparse_horn_cifar10_rgb_state_mlp_strength8",
        ),
    ]
    common = [
        "--train-limit 2000",
        "--eval-limit 1000",
        "--quality-classifier-kind residual_conv",
        "--quality-classifier-train-limit 10000",
        "--quality-classifier-eval-limit 5000",
        "--quality-classifier-epochs 15",
        "--quality-classifier-dim 256",
        "--quality-classifier-depth 3",
        "--attractor-variants-per-class 8",
    ]
    for seed in seeds:
        for variant_suffix, local_preset in variants:
            run_name = (
                f"{name_prefix}_{variant_suffix}_n256_resizeconv_train2000_"
                f"seed{seed}_20e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--preset {local_preset}",
                        *common,
                        *feature_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_main_coupling_strength_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Sweep recurrent HORN coupling while keeping class drive fixed."""

    return _mnist_generator_cifar10_rgb_main_coupling_strength_entries(
        seeds=(11,),
        name_prefix="mnist_generator_cifar10_rgb_main_coupling_strength",
        include_state_mlp=False,
    )


def _mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat_sweep() -> list[
    tuple[list[str], str]
]:
    """Repeat the main-coupling probe against StateMLP across two seeds."""

    return _mnist_generator_cifar10_rgb_main_coupling_strength_entries(
        seeds=(11, 23),
        name_prefix="mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat",
        include_state_mlp=True,
    )


def _mnist_generator_cifar10_rgb_main_coupling_fine_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Fine sweep around the moderate recurrent-coupling Goldilocks region."""

    return _mnist_generator_cifar10_rgb_main_coupling_strength_entries(
        seeds=(11, 23),
        name_prefix="mnist_generator_cifar10_rgb_main_coupling_fine",
        include_state_mlp=False,
        strengths=(("main025", 0.25), ("main050", 0.5), ("main075", 0.75)),
    )


def _mnist_generator_cifar10_rgb_main_coupling_current_replication_sweep() -> list[
    tuple[list[str], str]
]:
    """Current-code replication of all recurrent-coupling strengths."""

    return _mnist_generator_cifar10_rgb_main_coupling_strength_entries(
        seeds=(11, 23),
        name_prefix="mnist_generator_cifar10_rgb_main_coupling_current",
        include_state_mlp=False,
        strengths=(
            ("main000", 0.0),
            ("main025", 0.25),
            ("main050", 0.5),
            ("main075", 0.75),
            ("main100", 1.0),
        ),
    )


def _mnist_generator_cifar10_rgb_normalized_distance_decay_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Probe normalized distance-decay recurrent coupling on CIFAR-10 RGB."""

    return _mnist_generator_cifar10_rgb_main_coupling_strength_entries(
        seeds=(11, 23),
        name_prefix="mnist_generator_cifar10_rgb_normdist",
        include_state_mlp=False,
        strengths=(("main025", 0.25), ("main050", 0.5), ("main100", 1.0)),
        extra_args=(
            "--coupling-profile distance_decay",
            "--coupling-normalization row_sum",
            "--coupling-length-scale 0.24",
            "--coupling-floor 0.0",
        ),
    )


def _mnist_generator_cifar10_rgb_normalized_local_radius_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Probe sparse row-normalized local recurrent coupling on CIFAR-10 RGB."""

    return _mnist_generator_cifar10_rgb_main_coupling_strength_entries(
        seeds=(11, 23),
        name_prefix="mnist_generator_cifar10_rgb_normlocal",
        include_state_mlp=False,
        strengths=(("main025", 0.25), ("main050", 0.5), ("main100", 1.0)),
        extra_args=(
            "--coupling-profile local_radius",
            "--coupling-normalization row_sum",
            "--coupling-length-scale 0.24",
            "--coupling-floor 0.0",
        ),
    )


def _mnist_generator_cifar10_rgb_normalized_local_radius_sweep() -> list[
    tuple[list[str], str]
]:
    """Sweep sparse row-normalized local coupling radius on CIFAR-10 RGB."""

    entries: list[tuple[list[str], str]] = []
    for radius_label, radius in (("r016", 0.16), ("r024", 0.24), ("r032", 0.32)):
        entries.extend(
            _mnist_generator_cifar10_rgb_main_coupling_strength_entries(
                seeds=(11, 23),
                name_prefix="mnist_generator_cifar10_rgb_normlocal_radius",
                include_state_mlp=False,
                strengths=((radius_label, 1.0),),
                extra_args=(
                    "--coupling-profile local_radius",
                    "--coupling-normalization row_sum",
                    f"--coupling-length-scale {radius}",
                    "--coupling-floor 0.0",
                ),
            )
        )
    return entries


def _mnist_generator_cifar10_rgb_coarse_to_fine_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Probe whether a coarse HORN field improves normalized-local HORN."""

    entries = []
    variants = (
        (
            "horn_normlocal",
            "sparse_horn_cifar10_rgb_recommended_normlocal",
            (),
        ),
        (
            "coarse16_c2f000",
            "sparse_horn_cifar10_rgb_coarse16_normlocal",
            ("--coarse-to-fine-strength 0.0",),
        ),
        (
            "coarse16_c2f100",
            "sparse_horn_cifar10_rgb_coarse16_normlocal",
            ("--coarse-to-fine-strength 1.0",),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name, extra_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_coarse_to_fine_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            variant_args = [
                f"--seed {seed}",
                f"--preset {preset_name}",
                *extra_args,
            ]
            args = shlex.split(" ".join(variant_args))
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_coarse_to_fine_gain_sweep() -> list[
    tuple[list[str], str]
]:
    """Sweep gentler coarse-to-fine drive strengths on CIFAR-10 RGB."""

    entries = []
    variants = (
        ("coarse16_c2f025", 0.25),
        ("coarse16_c2f050", 0.50),
        ("coarse16_c2f075", 0.75),
    )
    for seed in (11, 23):
        for variant_name, strength in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_coarse_to_fine_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            variant_args = [
                f"--seed {seed}",
                "--preset sparse_horn_cifar10_rgb_coarse16_normlocal",
                f"--coarse-to-fine-strength {strength}",
            ]
            args = shlex.split(" ".join(variant_args))
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_coarse_to_fine_profile_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Probe spatially regularized coarse-to-fine projection profiles."""

    entries = []
    variants = (
        (
            "coarse16_c2f025_dense",
            (),
        ),
        (
            "coarse16_c2f025_local050",
            (
                "--coarse-to-fine-profile local_radius",
                "--coarse-to-fine-length-scale 0.5",
                "--coarse-to-fine-normalization row_sum",
            ),
        ),
        (
            "coarse16_c2f025_dist050",
            (
                "--coarse-to-fine-profile distance_decay",
                "--coarse-to-fine-length-scale 0.5",
                "--coarse-to-fine-normalization row_sum",
            ),
        ),
    )
    for seed in (11, 23):
        for variant_name, extra_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_coarse_to_fine_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            variant_args = [
                f"--seed {seed}",
                "--preset sparse_horn_cifar10_rgb_coarse16_normlocal_gentle",
                *extra_args,
            ]
            args = shlex.split(" ".join(variant_args))
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_audit() -> list[
    tuple[list[str], str]
]:
    """One-seed audit for coarse-to-fine settling/clamping diagnostics."""

    entries = []
    variants = (
        (
            "horn_normlocal",
            "sparse_horn_cifar10_rgb_recommended_normlocal",
            (),
        ),
        (
            "coarse16_c2f000",
            "sparse_horn_cifar10_rgb_coarse16_normlocal",
            ("--coarse-to-fine-strength 0.0",),
        ),
        (
            "coarse16_c2f025_dense",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle",
            (),
        ),
        (
            "coarse16_c2f025_dist050",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_dist050",
            (),
        ),
        (
            "coarse16_c2f025_local050",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050",
            (),
        ),
    )
    seed = 11
    for variant_name, preset_name, extra_args in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_"
            f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
        )
        variant_args = [
            f"--seed {seed}",
            f"--preset {preset_name}",
            *extra_args,
        ]
        args = shlex.split(" ".join(variant_args))
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat() -> list[
    tuple[list[str], str]
]:
    """Seed-repeat the local-radius coarse-to-fine HORN lead."""

    entries = []
    variants = (
        (
            "horn_normlocal",
            "sparse_horn_cifar10_rgb_recommended_normlocal",
            (),
        ),
        (
            "coarse16_c2f000",
            "sparse_horn_cifar10_rgb_coarse16_normlocal",
            ("--coarse-to-fine-strength 0.0",),
        ),
        (
            "coarse16_c2f025_local050",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle",
            (
                "--coarse-to-fine-profile local_radius",
                "--coarse-to-fine-length-scale 0.5",
                "--coarse-to-fine-normalization row_sum",
            ),
        ),
    )
    for seed in (11, 23, 37, 41):
        for variant_name, preset_name, extra_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            variant_args = [
                f"--seed {seed}",
                f"--preset {preset_name}",
                *extra_args,
            ]
            args = shlex.split(" ".join(variant_args))
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe() -> list[
    tuple[list[str], str]
]:
    """Test whether readout/objective upgrades convert local C2F basin gains."""

    entries = []
    variants = (
        (
            "coarse16_c2f000_base",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050",
            ("--coarse-to-fine-strength 0.0",),
        ),
        (
            "coarse16_c2f025_local050_base",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050",
            (),
        ),
        (
            "coarse16_c2f000_ch32",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_ch32",
            ("--coarse-to-fine-strength 0.0",),
        ),
        (
            "coarse16_c2f025_local050_ch32",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_ch32",
            (),
        ),
        (
            "coarse16_c2f000_dist0025",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_dist0025",
            ("--coarse-to-fine-strength 0.0",),
        ),
        (
            "coarse16_c2f025_local050_dist0025",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_dist0025",
            (),
        ),
        (
            "coarse16_c2f000_ch32_dist0025",
            (
                "sparse_horn_cifar10_rgb_coarse16_normlocal_"
                "gentle_local050_ch32_dist0025"
            ),
            ("--coarse-to-fine-strength 0.0",),
        ),
        (
            "coarse16_c2f025_local050_ch32_dist0025",
            (
                "sparse_horn_cifar10_rgb_coarse16_normlocal_"
                "gentle_local050_ch32_dist0025"
            ),
            (),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name, extra_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_coarse_to_fine_conversion_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            variant_args = [
                f"--seed {seed}",
                f"--preset {preset_name}",
                *extra_args,
            ]
            args = shlex.split(" ".join(variant_args))
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_coarse_to_fine_feedback_probe() -> list[
    tuple[list[str], str]
]:
    """Test whether output feedback converts C2F basin gains into quality."""

    entries = []
    variants = (
        (
            "coarse16_c2f000_feedback050_base",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_feedback050",
            ("--coarse-to-fine-strength 0.0",),
        ),
        (
            "coarse16_c2f025_local050_feedback050_base",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_feedback050",
            (),
        ),
        (
            "coarse16_c2f000_ch32_dist0025_feedback050",
            (
                "sparse_horn_cifar10_rgb_coarse16_normlocal_"
                "gentle_local050_ch32_dist0025_feedback050"
            ),
            ("--coarse-to-fine-strength 0.0",),
        ),
        (
            "coarse16_c2f025_local050_ch32_dist0025_feedback050",
            (
                "sparse_horn_cifar10_rgb_coarse16_normlocal_"
                "gentle_local050_ch32_dist0025_feedback050"
            ),
            (),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name, extra_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_coarse_to_fine_feedback_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            variant_args = [
                f"--seed {seed}",
                f"--preset {preset_name}",
                *extra_args,
            ]
            args = shlex.split(" ".join(variant_args))
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_multiscale_layered_probe() -> list[
    tuple[list[str], str]
]:
    """Test layered HORN vertical coupling against matched controls."""

    entries = []
    variants = (
        (
            "normlocal_plain",
            "sparse_horn_cifar10_rgb_recommended_normlocal",
            (),
        ),
        (
            "multiscale_no_vertical",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical",
            (),
        ),
        (
            "multiscale_local050_fb005",
            "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005",
            (),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name, extra_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_multiscale_layered_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            variant_args = [
                f"--seed {seed}",
                f"--preset {preset_name}",
                *extra_args,
            ]
            args = shlex.split(" ".join(variant_args))
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_multiscale_auxiliary_probe() -> list[
    tuple[list[str], str]
]:
    """Test whether a coarse low-res target makes vertical hierarchy useful."""

    entries = []
    variants = (
        (
            "no_vertical",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical",
        ),
        (
            "no_vertical_auxlow8",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8",
        ),
        (
            "local050_fb005",
            "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005",
        ),
        (
            "local050_fb005_auxlow8",
            "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8",
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_multiscale_auxiliary_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            variant_args = [
                f"--seed {seed}",
                f"--preset {preset_name}",
            ]
            args = shlex.split(" ".join(variant_args))
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_multiscale_gated_probe() -> list[
    tuple[list[str], str]
]:
    """Test selective vertical routing for auxiliary-supervised multiscale HORN."""

    entries = []
    variants = (
        (
            "no_vertical_auxlow8",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8",
        ),
        (
            "local050_fb005_auxlow8",
            "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8",
        ),
        (
            "vgate_conditioning_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_vgate_conditioning"
            ),
        ),
        (
            "vgate_non_conditioning_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_vgate_non_conditioning"
            ),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_multiscale_gated_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {preset_name}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_multiscale_gain_probe() -> list[
    tuple[list[str], str]
]:
    """Compare additive vertical drive with coarse-to-fine gain modulation."""

    entries = []
    variants = (
        (
            "no_vertical_auxlow8",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8",
        ),
        (
            "vgate_conditioning_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_vgate_conditioning"
            ),
        ),
        (
            "gain_all_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_all"
            ),
        ),
        (
            "gain_conditioning_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_conditioning"
            ),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_multiscale_gain_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {preset_name}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_multiscale_weak_drive_probe() -> list[
    tuple[list[str], str]
]:
    """Test whether gain hierarchy helps when class drive is weaker."""

    entries = []
    variants = (
        (
            "no_vertical_auxlow8_drive2",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "no_vertical_auxlow8_drive2"
            ),
        ),
        (
            "vgate_conditioning_auxlow8_drive2",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_vgate_conditioning_drive2"
            ),
        ),
        (
            "gain_all_auxlow8_drive2",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_all_drive2"
            ),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_multiscale_weak_drive_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {preset_name}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_multiscale_signed_gain_probe() -> list[
    tuple[list[str], str]
]:
    """Compare selective and signed gain hierarchy variants."""

    entries = []
    variants = (
        (
            "gain_conditioning_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_conditioning"
            ),
        ),
        (
            "signed_gain_all_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_all"
            ),
        ),
        (
            "signed_gain_conditioning_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning"
            ),
        ),
        (
            "gain_conditioning_auxlow8_drive2",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_conditioning_drive2"
            ),
        ),
        (
            "signed_gain_all_auxlow8_drive2",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_all_drive2"
            ),
        ),
        (
            "signed_gain_conditioning_auxlow8_drive2",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning_drive2"
            ),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_multiscale_signed_gain_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {preset_name}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_multiscale_soft_gate_probe() -> list[
    tuple[list[str], str]
]:
    """Test soft selective vertical gain as a middle ground."""

    entries = []
    variants = (
        (
            "gain_conditioning_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_conditioning"
            ),
        ),
        (
            "gain_conditioning_soft025_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_conditioning_soft025"
            ),
        ),
        (
            "signed_gain_conditioning_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning"
            ),
        ),
        (
            "signed_gain_conditioning_soft025_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning_soft025"
            ),
        ),
        (
            "gain_conditioning_soft025_auxlow8_drive2",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_conditioning_soft025_drive2"
            ),
        ),
        (
            "signed_gain_conditioning_soft025_auxlow8_drive2",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning_soft025_drive2"
            ),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_multiscale_soft_gate_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {preset_name}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_vertical_causality_audit() -> list[
    tuple[list[str], str]
]:
    """Audit sample-time vertical hierarchy causality on current best variants."""

    entries = []
    audit_args = (
        "--batch-size 32 "
        "--vertical-audit-modes normal,zero "
        "--vertical-audit-sample-count 64 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "no_vertical_auxlow8",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8",
        ),
        (
            "gain_all_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_all"
            ),
        ),
        (
            "signed_gain_conditioning_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning"
            ),
        ),
        (
            "gain_conditioning_soft025_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_conditioning_soft025"
            ),
        ),
    )
    for seed in (11, 23, 37):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_vertical_causality_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {preset_name} {audit_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_vertical_calibration_probe() -> list[
    tuple[list[str], str]
]:
    """Test whether calibrated vertical signal scale creates causal hierarchy."""

    entries = []
    audit_args = (
        "--batch-size 32 "
        "--vertical-audit-modes normal,zero "
        "--vertical-audit-sample-count 64 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "no_vertical_auxlow8",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8",
        ),
        (
            "gain_all_auxlow8",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_all"
            ),
        ),
        (
            "gain_all_vscale10",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_all_vscale10"
            ),
        ),
        (
            "gain_all_vscale30",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_all_vscale30"
            ),
        ),
        (
            "signed_gain_conditioning_vscale10",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning_vscale10"
            ),
        ),
        (
            "signed_gain_conditioning_vscale30",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning_vscale30"
            ),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_vertical_calibration_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {preset_name} {audit_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_dual_gain_probe() -> list[tuple[list[str], str]]:
    """Compare calibrated single-route gain against dual-route vertical gain."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "no_vertical_auxlow8",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8",
        ),
        (
            "gain_all_vscale30",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_all_vscale30"
            ),
        ),
        (
            "signed_gain_conditioning_vscale30",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning_vscale30"
            ),
        ),
        (
            "dual_gain_conditioning_vscale10",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_dual_gain_conditioning_vscale10"
            ),
        ),
        (
            "dual_gain_conditioning_vscale30",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_dual_gain_conditioning_vscale30"
            ),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_dual_gain_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {preset_name} {audit_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_main_coupling_strength_entries(
    *,
    seeds: tuple[int, ...],
    name_prefix: str,
    include_state_mlp: bool,
    extra_args: tuple[str, ...] = (),
    strengths: tuple[tuple[str, float], ...] = (
        ("main000", 0.0),
        ("main025", 0.25),
        ("main050", 0.5),
        ("main100", 1.0),
    ),
) -> list[tuple[list[str], str]]:
    """Build strict-judge recurrent-coupling strength probe entries."""

    entries = []
    common = [
        "--train-limit 2000",
        "--eval-limit 1000",
        "--quality-classifier-kind residual_conv",
        "--quality-classifier-train-limit 10000",
        "--quality-classifier-eval-limit 5000",
        "--quality-classifier-epochs 15",
        "--quality-classifier-dim 256",
        "--quality-classifier-depth 3",
        "--attractor-variants-per-class 8",
        "--loss-mode pixel_feature_drift",
        "--pixel-drift-weight 0.5",
        "--feature-drift-weight 0.25",
        "--feature-drift-mode learned",
        "--learned-feature-kind residual_conv",
        "--learned-feature-epochs 10",
        "--learned-feature-dim 256",
        "--learned-feature-depth 3",
    ]
    variants = list(strengths)
    if include_state_mlp:
        variants = [
            ("main000", 0.0),
            ("main050", 0.5),
            ("main100", 1.0),
            ("state_mlp", None),
        ]
    for seed in seeds:
        for strength_label, main_strength in variants:
            is_state_mlp = main_strength is None
            local_preset = (
                "sparse_horn_cifar10_rgb_state_mlp_strength8"
                if is_state_mlp
                else "sparse_horn_cifar10_rgb_recommended_drive025"
            )
            variant_name = (
                "state_mlp_resfeat025"
                if is_state_mlp
                else f"horn_resfeat025_{strength_label}"
            )
            run_name = (
                f"{name_prefix}_{variant_name}_n256_resizeconv_train2000_"
                f"seed{seed}_20e"
            )
            variant_args = [
                f"--seed {seed}",
                f"--preset {local_preset}",
                *common,
                *extra_args,
            ]
            if not is_state_mlp:
                variant_args.extend(
                    [
                        "--coupling-strength 1.0",
                        f"--main-coupling-strength {main_strength}",
                    ]
                )
            args = shlex.split(" ".join(variant_args))
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_vertical_homeostasis_probe() -> list[
    tuple[list[str], str]
]:
    """Compare raw calibrated gain against homeostatic normalized gain."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "no_vertical_auxlow8",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8",
        ),
        (
            "gain_all_vscale30",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_all_vscale30"
            ),
        ),
        (
            "signed_gain_conditioning_vscale30",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning_vscale30"
            ),
        ),
        (
            "dual_gain_conditioning_vscale30",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_dual_gain_conditioning_vscale30"
            ),
        ),
        (
            "gain_all_vscale30_normstd015",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_gain_all_vscale30_normstd015"
            ),
        ),
        (
            "signed_gain_conditioning_vscale30_normstd015",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_signed_gain_conditioning_"
                "vscale30_normstd015"
            ),
        ),
        (
            "dual_gain_conditioning_vscale30_normstd015",
            (
                "sparse_horn_cifar10_rgb_multiscale16_64_"
                "local050_fb005_auxlow8_dual_gain_conditioning_"
                "vscale30_normstd015"
            ),
        ),
    )
    for seed in (11, 23):
        for variant_name, preset_name in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_vertical_homeostasis_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {preset_name} {audit_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration() -> list[
    tuple[list[str], str]
]:
    """Calibrate normalized selective signed vertical gain around the winner."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    base_preset = (
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale30"
    )
    variants = (
        (
            "signed_gain_conditioning_vscale30_center",
            "--multiscale-vertical-gain-normalization center "
            "--multiscale-vertical-gain-target-std 0.0",
        ),
        (
            "signed_gain_conditioning_vscale30_normstd010",
            "--multiscale-vertical-gain-normalization center_rms "
            "--multiscale-vertical-gain-target-std 0.010",
        ),
        (
            "signed_gain_conditioning_vscale30_normstd015",
            "--multiscale-vertical-gain-normalization center_rms "
            "--multiscale-vertical-gain-target-std 0.015",
        ),
        (
            "signed_gain_conditioning_vscale30_normstd020",
            "--multiscale-vertical-gain-normalization center_rms "
            "--multiscale-vertical-gain-target-std 0.020",
        ),
    )
    for seed in (11, 23):
        for variant_name, variant_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_vertical_homeostasis_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {base_preset} "
                f"{variant_args} {audit_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe() -> list[
    tuple[list[str], str]
]:
    """Test whether centered top-down signed gain should arrive late or ramp in."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    base_preset = (
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale30_center"
    )
    variants = (
        (
            "center_constant",
            "--multiscale-vertical-schedule constant "
            "--multiscale-vertical-onset-step 0 "
            "--multiscale-vertical-ramp-steps 0",
        ),
        (
            "center_delayed8",
            "--multiscale-vertical-schedule delayed "
            "--multiscale-vertical-onset-step 8 "
            "--multiscale-vertical-ramp-steps 0",
        ),
        (
            "center_delayed16",
            "--multiscale-vertical-schedule delayed "
            "--multiscale-vertical-onset-step 16 "
            "--multiscale-vertical-ramp-steps 0",
        ),
        (
            "center_ramp8_16",
            "--multiscale-vertical-schedule linear_ramp "
            "--multiscale-vertical-onset-step 8 "
            "--multiscale-vertical-ramp-steps 16",
        ),
    )
    for seed in (11, 23):
        for variant_name, variant_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_vertical_timing_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {base_preset} "
                f"{variant_args} {audit_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_centered_signed_gain_target_probe() -> list[
    tuple[list[str], str]
]:
    """Route centered top-down gain into specific fine HORN terms."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    base_preset = (
        "sparse_horn_cifar10_rgb_multiscale16_64_"
        "local050_fb005_auxlow8_signed_gain_conditioning_vscale30_center"
    )
    variants = (
        ("center_drive", "--multiscale-vertical-gain-target drive"),
        ("center_coupling", "--multiscale-vertical-gain-target coupling"),
        ("center_conditioning", "--multiscale-vertical-gain-target conditioning"),
        ("center_damping", "--multiscale-vertical-gain-target damping"),
    )
    for seed in (11, 23):
        for variant_name, variant_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_vertical_gain_target_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {base_preset} "
                f"{variant_args} {audit_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_coarse_objective_probe() -> list[
    tuple[list[str], str]
]:
    """Compare paired and distributional coarse supervision for hierarchy."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "center_mse",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_vscale30_center",
        ),
        (
            "center_dist",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center",
        ),
        (
            "no_vertical_mse",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8",
        ),
        (
            "no_vertical_dist",
            "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxdist8",
        ),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_coarse_objective_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_attribution_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """One-seed RGB HORN attribution controls with feature/dynamics metrics."""

    entries = []
    variants = [
        ("horn_recommended", "sparse_horn_cifar10_rgb_recommended"),
        ("horn_step1", "sparse_horn_cifar10_rgb_recommended_step1"),
        (
            "horn_frozen_recurrent",
            "sparse_horn_cifar10_rgb_recommended_frozen_recurrent",
        ),
        (
            "horn_frozen_conditioning",
            "sparse_horn_cifar10_rgb_recommended_frozen_conditioning",
        ),
        (
            "horn_no_main_interaction",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction",
        ),
        ("horn_decoder_only", "sparse_horn_cifar10_rgb_recommended_decoder_only"),
    ]
    classifier_args = (
        "--quality-classifier-kind conv "
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    seed = 11
    for variant_suffix, local_preset in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_attribution_"
            f"{variant_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
        )
        args = shlex.split(
            f"--seed {seed} --preset {local_preset} {classifier_args}"
        )
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_sparse_drive_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """CIFAR RGB probe that asks whether main coupling propagates sparse drive."""

    entries = []
    variants = [
        ("horn_drive100", "sparse_horn_cifar10_rgb_recommended"),
        ("horn_drive025", "sparse_horn_cifar10_rgb_recommended_drive025"),
        ("horn_drive010", "sparse_horn_cifar10_rgb_recommended_drive010"),
        (
            "horn_no_main_drive100",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction",
        ),
        (
            "horn_no_main_drive025",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025",
        ),
        (
            "horn_no_main_drive010",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive010",
        ),
    ]
    classifier_args = (
        "--quality-classifier-kind conv "
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    seed = 11
    for variant_suffix, local_preset in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_sparse_drive_"
            f"{variant_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
        )
        args = shlex.split(
            f"--seed {seed} --preset {local_preset} {classifier_args}"
        )
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_sparse_drive_seed_repeat_sweep() -> list[
    tuple[list[str], str]
]:
    """Repeat the 25% sparse-drive coupling attribution probe across seeds."""

    entries = []
    variants = [
        ("horn_drive025", "sparse_horn_cifar10_rgb_recommended_drive025"),
        (
            "horn_no_main_drive025",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025",
        ),
    ]
    classifier_args = (
        "--quality-classifier-kind conv "
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (7, 11, 23, 37):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_sparse_drive_seed_repeat_"
                f"{variant_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_structured_drive_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Compare contiguous and spatial-grid sparse class-drive topologies."""

    entries = []
    variants = [
        ("horn_prefix025", "sparse_horn_cifar10_rgb_recommended_drive025"),
        (
            "horn_grid025",
            "sparse_horn_cifar10_rgb_recommended_drive025_spatial_grid",
        ),
        (
            "horn_no_main_prefix025",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025",
        ),
        (
            "horn_no_main_grid025",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025_spatial_grid",
        ),
    ]
    classifier_args = (
        "--quality-classifier-kind conv "
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 23):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_structured_drive_"
                f"{variant_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_coherent_drive_probe_sweep() -> list[
    tuple[list[str], str]
]:
    """Compare top-band and centered-block sparse class-drive regions."""

    entries = []
    variants = [
        ("horn_prefix025", "sparse_horn_cifar10_rgb_recommended_drive025"),
        (
            "horn_center025",
            "sparse_horn_cifar10_rgb_recommended_drive025_center_block",
        ),
        (
            "horn_no_main_prefix025",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025",
        ),
        (
            "horn_no_main_center025",
            "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025_center_block",
        ),
    ]
    classifier_args = (
        "--quality-classifier-kind conv "
        "--quality-classifier-train-limit 5000 "
        "--quality-classifier-eval-limit 2000 "
        "--quality-classifier-epochs 10 "
        "--quality-classifier-dim 256 "
        "--quality-classifier-depth 3"
    )
    for seed in (11, 23):
        for variant_suffix, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_coherent_drive_"
                f"{variant_suffix}_n256_resizeconv_train1000_seed{seed}_20e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {classifier_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_feedback_signal_probe() -> list[
    tuple[list[str], str]
]:
    """Compare position-only and state feedback in active hierarchy."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "mse_position",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_vscale30_center",
        ),
        (
            "mse_state",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_vscale30_"
            "center_feedback_state",
        ),
        (
            "dist_position",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_center",
        ),
        (
            "dist_state",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state",
        ),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_feedback_signal_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_feedback_source_gate_probe() -> list[
    tuple[list[str], str]
]:
    """Compare which fine columns bottom-up state feedback should listen to."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "source_all",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state",
        ),
        (
            "source_conditioning",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_source_conditioning",
        ),
        (
            "source_non_conditioning",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_source_non_conditioning",
        ),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_feedback_source_gate_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_feedback_source_mix_entries(
    *,
    run_prefix: str,
) -> list[tuple[list[str], str]]:
    """Return source-mix entries for a named run prefix."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "source_all",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state",
        ),
        (
            "mix75_25",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix75_25",
        ),
        (
            "mix50_50",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50",
        ),
        (
            "mix25_75",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix25_75",
        ),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                f"mnist_generator_cifar10_rgb_{run_prefix}_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_feedback_source_mix_probe() -> list[
    tuple[list[str], str]
]:
    """Compare mean-normalized soft source mixes for state feedback."""

    return _mnist_generator_cifar10_rgb_feedback_source_mix_entries(
        run_prefix="feedback_source_mix",
    )


def _mnist_generator_cifar10_rgb_feedback_source_mix_auxfix_probe() -> list[
    tuple[list[str], str]
]:
    """Rerun source-mix after fixing RGB coarse-auxiliary target layout."""

    return _mnist_generator_cifar10_rgb_feedback_source_mix_entries(
        run_prefix="feedback_source_mix_auxfix",
    )


def _mnist_generator_cifar10_rgb_readout_fusion_probe() -> list[
    tuple[list[str], str]
]:
    """Test whether a low-capacity coarse readout scaffold improves rendering."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "mix50_50",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50",
        ),
        (
            "mix50_50_fusion010",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50_fusion010",
        ),
        (
            "mix50_50_fusion025",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50_fusion025",
        ),
        (
            "mix75_25_fusion010",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix75_25_fusion010",
        ),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_readout_fusion_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_coarse_readout_consistency_probe() -> list[
    tuple[list[str], str]
]:
    """Train the fine readout to agree with the same-trajectory coarse scaffold."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        (
            "mix50_50",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50",
        ),
        (
            "mix50_50_consistency005",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50_consistency005",
        ),
        (
            "mix50_50_consistency010",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50_consistency010",
        ),
        (
            "mix75_25_consistency005",
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix75_25_consistency005",
        ),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_coarse_readout_consistency_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_readout_gate_probe() -> list[
    tuple[list[str], str]
]:
    """Test learned coarse-to-fine readout modulation without RGB blending."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("hierarchy_lead", "sparse_horn_cifar10_rgb_hierarchy_lead"),
        ("gate010", "sparse_horn_cifar10_rgb_hierarchy_gate010"),
        ("gate025", "sparse_horn_cifar10_rgb_hierarchy_gate025"),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_readout_gate_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_frequency_objective_probe() -> list[
    tuple[list[str], str]
]:
    """Test whether image-spectrum statistics recover missing CIFAR detail."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("hierarchy_lead", "sparse_horn_cifar10_rgb_hierarchy_lead"),
        ("freq001", "sparse_horn_cifar10_rgb_hierarchy_freq001"),
        ("freq003", "sparse_horn_cifar10_rgb_hierarchy_freq003"),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_frequency_objective_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_patch_objective_probe() -> list[
    tuple[list[str], str]
]:
    """Test local patch-detail statistics against global spectrum matching."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero,shuffle,flip,scale025,scale050 "
        "--vertical-audit-sample-count 128 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("hierarchy_lead", "sparse_horn_cifar10_rgb_hierarchy_lead"),
        ("patch005", "sparse_horn_cifar10_rgb_hierarchy_patch005"),
        ("patch010", "sparse_horn_cifar10_rgb_hierarchy_patch010"),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_patch_objective_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_patch_v2_probe() -> list[
    tuple[list[str], str]
]:
    """Test overlap/multiscale patch objectives for grid-artifact reduction."""

    entries = []
    audit_args = (
        "--batch-size 64 "
        "--vertical-audit-modes normal,zero "
        "--vertical-audit-sample-count 64 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("hierarchy_lead", "sparse_horn_cifar10_rgb_hierarchy_lead"),
        ("patch010", "sparse_horn_cifar10_rgb_hierarchy_patch010"),
        ("patch010_overlap", "sparse_horn_cifar10_rgb_hierarchy_patch010_overlap"),
        (
            "patch010_multiscale",
            "sparse_horn_cifar10_rgb_hierarchy_patch010_multiscale",
        ),
        (
            "patch010_multiscale_overlap",
            "sparse_horn_cifar10_rgb_hierarchy_patch010_multiscale_overlap",
        ),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_patch_v2_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {audit_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_state_information_probe() -> list[
    tuple[list[str], str]
]:
    """Ask whether traced oscillator states expose scaffold/detail information."""

    entries = []
    probe_args = (
        "--batch-size 64 "
        "--state-probe-sample-count 64 "
        "--state-probe-target-size 8 "
        "--vertical-audit-modes normal,zero "
        "--vertical-audit-sample-count 64 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("current", "sparse_horn_cifar10_rgb_current"),
        ("hierarchy_lead", "sparse_horn_cifar10_rgb_hierarchy_lead"),
        (
            "patch010_multiscale",
            "sparse_horn_cifar10_rgb_hierarchy_patch010_multiscale",
        ),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_state_information_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {probe_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_state_residual_readout_probe() -> list[
    tuple[list[str], str]
]:
    """Test direct local final-state residual readout on hierarchy lead."""

    entries = []
    probe_args = (
        "--batch-size 64 "
        "--state-probe-sample-count 64 "
        "--state-probe-target-size 8 "
        "--vertical-audit-modes normal,zero "
        "--vertical-audit-sample-count 64 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("hierarchy_lead", "sparse_horn_cifar10_rgb_hierarchy_lead"),
        (
            "state_residual005",
            "sparse_horn_cifar10_rgb_hierarchy_state_residual005",
        ),
        (
            "state_residual010",
            "sparse_horn_cifar10_rgb_hierarchy_state_residual010",
        ),
    )
    for seed in (11, 23):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_state_residual_readout_"
                f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {probe_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_state_residual_longer_pilot() -> list[
    tuple[list[str], str]
]:
    """Run a longer visual-maturation pilot for the residual readout candidate."""

    entries = []
    pilot_args = (
        "--epochs 40 "
        "--checkpoint-every 20 "
        "--artifact-every 20 "
        "--batch-size 64 "
        "--eval-sample-count 256 "
        "--vertical-audit-modes normal,zero "
        "--vertical-audit-sample-count 64 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("current", "sparse_horn_cifar10_rgb_current"),
        ("hierarchy_lead", "sparse_horn_cifar10_rgb_hierarchy_lead"),
        (
            "state_residual005",
            "sparse_horn_cifar10_rgb_hierarchy_state_residual005",
        ),
    )
    seed = 23
    for variant_name, local_preset in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_state_residual_longer_"
            f"{variant_name}_n256_resizeconv_train2000_seed{seed}_40e"
        )
        args = shlex.split(f"--seed {seed} --preset {local_preset} {pilot_args}")
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_resonant_readout_pilot() -> list[
    tuple[list[str], str]
]:
    """Test shared HORN resonant filter-bank readout on current CIFAR recipe."""

    entries = []
    pilot_args = (
        "--epochs 20 "
        "--checkpoint-every 20 "
        "--artifact-every 20 "
        "--batch-size 64 "
        "--eval-sample-count 256 "
        "--vertical-audit-modes normal,zero "
        "--vertical-audit-sample-count 64 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("current", "sparse_horn_cifar10_rgb_current"),
        ("resonant005", "sparse_horn_cifar10_rgb_current_resonant005"),
        ("resonant010", "sparse_horn_cifar10_rgb_current_resonant010"),
    )
    seed = 23
    for variant_name, local_preset in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_resonant_readout_"
            f"{variant_name}_n256_resizeconv_train2000_seed{seed}_20e"
        )
        args = shlex.split(f"--seed {seed} --preset {local_preset} {pilot_args}")
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_capacity_probe() -> list[tuple[list[str], str]]:
    """Test whether CIFAR RGB HORN is bottlenecked by oscillator-site count."""

    entries = []
    probe_args = (
        "--epochs 20 "
        "--checkpoint-every 20 "
        "--artifact-every 20 "
        "--batch-size 32 "
        "--eval-sample-count 256 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("n256_current", "sparse_horn_cifar10_rgb_current"),
        ("n512_current", "sparse_horn_cifar10_rgb_current_n512"),
        ("n512_resonant005", "sparse_horn_cifar10_rgb_current_n512_resonant005"),
    )
    seed = 23
    for variant_name, local_preset in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_capacity_"
            f"{variant_name}_resizeconv_train2000_seed{seed}_20e"
        )
        args = shlex.split(f"--seed {seed} --preset {local_preset} {probe_args}")
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_multimode_probe() -> list[tuple[list[str], str]]:
    """Compare flat site-count scaling with structured frequency modes per site."""

    entries = []
    probe_args = (
        "--epochs 20 "
        "--checkpoint-every 20 "
        "--artifact-every 20 "
        "--batch-size 32 "
        "--eval-sample-count 256 "
        "--attractor-variants-per-class 8"
    )
    variants = (
        ("n256_current", "sparse_horn_cifar10_rgb_current"),
        ("n512_flat", "sparse_horn_cifar10_rgb_current_n512"),
        ("multimode2", "sparse_horn_cifar10_rgb_current_multimode2"),
    )
    seed = 23
    for variant_name, local_preset in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_multimode_"
            f"{variant_name}_resizeconv_train2000_seed{seed}_20e"
        )
        args = shlex.split(f"--seed {seed} --preset {local_preset} {probe_args}")
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_retinotopic_readout_probe() -> list[
    tuple[list[str], str]
]:
    """Compare scrambled resize-conv seeds with retinotopic HORN state seeds."""

    entries = []
    probe_args = (
        "--epochs 20 "
        "--checkpoint-every 20 "
        "--artifact-every 20 "
        "--batch-size 32 "
        "--eval-sample-count 256 "
        "--attractor-variants-per-class 8 "
        "--state-fit-sample-count 32 "
        "--state-fit-steps 80 "
        "--state-fit-settle-steps 0,8,16,32"
    )
    variants = (
        ("n256_current", "sparse_horn_cifar10_rgb_current"),
        ("n256_retino", "sparse_horn_cifar10_rgb_current_retinotopic"),
        ("multimode2", "sparse_horn_cifar10_rgb_current_multimode2"),
        (
            "multimode2_retino",
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic",
        ),
    )
    seed = 23
    for variant_name, local_preset in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_retinotopic_"
            f"{variant_name}_train2000_seed{seed}_20e"
        )
        args = shlex.split(f"--seed {seed} --preset {local_preset} {probe_args}")
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_retinotopic_control_probe() -> list[
    tuple[list[str], str]
]:
    """Control retinotopic decoder size, seed width, and fitted-state brittleness."""

    entries = []
    probe_args = (
        "--epochs 20 "
        "--checkpoint-every 20 "
        "--artifact-every 20 "
        "--batch-size 32 "
        "--eval-sample-count 256 "
        "--attractor-variants-per-class 8 "
        "--state-fit-sample-count 32 "
        "--state-fit-steps 80 "
        "--state-fit-settle-steps 0,1,2,4,8,16,32"
    )
    variants = (
        ("n256_flat", "sparse_horn_cifar10_rgb_current"),
        ("n256_retino_ch30", "sparse_horn_cifar10_rgb_current_retinotopic_ch30"),
        (
            "n256_retino_seed4_ch30",
            "sparse_horn_cifar10_rgb_current_retinotopic_seed4_ch30",
        ),
        ("multimode2_flat", "sparse_horn_cifar10_rgb_current_multimode2"),
        (
            "multimode2_retino_ch30",
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_ch30",
        ),
    )
    seed = 23
    for variant_name, local_preset in variants:
        run_name = (
            "mnist_generator_cifar10_rgb_retino_control_"
            f"{variant_name}_train2000_seed{seed}_20e"
        )
        args = shlex.split(f"--seed {seed} --preset {local_preset} {probe_args}")
        output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_state_anchor_probe() -> list[
    tuple[list[str], str]
]:
    """Probe whether local image-to-state anchors improve texture survival."""

    entries = []
    probe_args = (
        "--epochs 20 "
        "--checkpoint-every 20 "
        "--artifact-every 20 "
        "--batch-size 32 "
        "--eval-sample-count 256 "
        "--attractor-variants-per-class 8 "
        "--state-fit-sample-count 32 "
        "--state-fit-steps 80 "
        "--state-fit-settle-steps 0,1,2,4,8,16,32"
    )
    variants = (
        (
            "no_anchor",
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_ch30",
        ),
        (
            "anchor_reconstruct010",
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor_reconstruct010",
        ),
        (
            "anchor_frozen010",
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor_frozen010",
        ),
        (
            "anchor010",
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor010",
        ),
        (
            "anchor030",
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030",
        ),
    )
    for seed in (23, 24):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_state_anchor_"
                f"{variant_name}_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {probe_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_state_prior_training_probe() -> list[
    tuple[list[str], str]
]:
    """Train anchor HORN with generated samples drawn from state priors."""

    entries = []
    probe_args = (
        "--epochs 20 "
        "--checkpoint-every 20 "
        "--artifact-every 20 "
        "--batch-size 32 "
        "--eval-sample-count 256 "
        "--attractor-variants-per-class 8 "
        "--state-fit-sample-count 32 "
        "--state-fit-steps 80 "
        "--state-fit-settle-steps 0,1,2,4,8,16,32"
    )
    variants = (
        (
            "anchor030",
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030",
        ),
        (
            "prior_global",
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor030_prior_global",
        ),
        (
            "prior_class",
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor030_prior_class",
        ),
    )
    for seed in (23, 24):
        for variant_name, local_preset in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_state_prior_training_"
                f"{variant_name}_train2000_seed{seed}_20e"
            )
            args = shlex.split(f"--seed {seed} --preset {local_preset} {probe_args}")
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1() -> list[
    tuple[list[str], str]
]:
    """Diagnostic scale gate for prior-aware CIFAR RGB HORN generators."""

    entries = []
    probe_args = (
        "--epochs 40 "
        "--train-limit 10000 "
        "--eval-limit 5000 "
        "--checkpoint-every 40 "
        "--artifact-every 40 "
        "--batch-size 32 "
        "--eval-sample-count 512 "
        "--quality-classifier-train-limit 20000 "
        "--quality-classifier-eval-limit 5000 "
        "--attractor-variants-per-class 8 "
        "--state-fit-sample-count 32 "
        "--state-fit-steps 80 "
        "--state-fit-settle-steps 0,1,2,4,8,16,32"
    )
    variants = (
        (
            "prior_global_b32",
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor030_prior_global",
            "",
        ),
        (
            "prior_class_b32",
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor030_prior_class",
            "",
        ),
        (
            "prior_class_patch005_offset_b32",
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor030_prior_class_patch005",
            "",
        ),
    )
    for seed in (23, 24):
        for variant_name, local_preset, extra_args in variants:
            run_name = (
                "mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1_"
                f"{variant_name}_train10000_seed{seed}_40e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {probe_args} {extra_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))

    return entries


def _mnist_generator_cifar10_rgb_state_prior_control_probe() -> list[
    tuple[list[str], str]
]:
    """Rung-1 control matrix for state-prior HORN vs StateMLP stacks."""

    entries = []
    probe_args = (
        "--epochs 40 "
        "--train-limit 10000 "
        "--eval-limit 5000 "
        "--checkpoint-every 40 "
        "--artifact-every 40 "
        "--batch-size 32 "
        "--eval-sample-count 512 "
        "--quality-classifier-train-limit 20000 "
        "--quality-classifier-eval-limit 5000 "
        "--attractor-variants-per-class 8 "
        "--state-fit-sample-count 32 "
        "--state-fit-steps 80 "
        "--state-fit-settle-steps 0,1,2,4,8,16,32"
    )
    variants = (
        (
            "prior_global_patch005_b32",
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor030_prior_global_patch005",
            (23, 24),
            "",
        ),
        (
            "state_mlp_prior_class_patch005_b32",
            "sparse_horn_cifar10_rgb_current_state_mlp_"
            "retinotopic_anchor030_prior_class_patch005",
            (23, 24),
            "",
        ),
        (
            "prior_class_patch005_queue64_b32",
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor030_prior_class_patch005",
            (23,),
            "--drift-queue-size 64 --drift-queue-num-pos 64",
        ),
    )
    for variant_name, local_preset, seeds, extra_args in variants:
        for seed in seeds:
            run_name = (
                "mnist_generator_cifar10_rgb_state_prior_control_probe_"
                f"{variant_name}_train10000_seed{seed}_40e"
            )
            args = shlex.split(
                f"--seed {seed} --preset {local_preset} {probe_args} {extra_args}"
            )
            output_dir = VOLUME_MOUNT / "mnist_generator" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))

    return entries


def _sweep_entries(preset: str) -> list[tuple[list[str], str]]:
    if preset == "mnist_generator_core":
        return _mnist_generator_core_sweep()
    if preset == "mnist_generator_conditional_core":
        return _mnist_generator_conditional_core_sweep()
    if preset == "mnist_generator_un0_coupled_core":
        return _mnist_generator_un0_coupled_core_sweep()
    if preset == "mnist_generator_spatial_readout_core":
        return _mnist_generator_spatial_readout_core_sweep()
    if preset == "mnist_generator_local_readout_core":
        return _mnist_generator_local_readout_core_sweep()
    if preset == "mnist_generator_spatial_coupling_core":
        return _mnist_generator_spatial_coupling_core_sweep()
    if preset == "mnist_generator_trainability_attribution_core":
        return _mnist_generator_trainability_attribution_core_sweep()
    if preset == "mnist_generator_unconditional_local_readout_core":
        return _mnist_generator_unconditional_local_readout_core_sweep()
    if preset == "mnist_generator_resize_conv_core":
        return _mnist_generator_resize_conv_core_sweep()
    if preset == "mnist_generator_resize_conv_pixel_drift_core":
        return _mnist_generator_resize_conv_pixel_drift_core_sweep()
    if preset == "mnist_generator_resize_conv_pixel_drift_queue_core":
        return _mnist_generator_resize_conv_pixel_drift_queue_core_sweep()
    if preset == "mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core":
        return _mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core_sweep()
    if preset == "mnist_generator_resize_conv_pixel_drift_queue_distributional_core":
        return _mnist_generator_resize_conv_pixel_drift_queue_distributional_core_sweep()
    if preset == "mnist_generator_resize_conv_feature_drift_core":
        return _mnist_generator_resize_conv_feature_drift_core_sweep()
    if preset == "mnist_generator_resize_conv_learned_feature_drift_core":
        return _mnist_generator_resize_conv_learned_feature_drift_core_sweep()
    if preset == "mnist_generator_resize_conv_learned_feature_drift_queue_core":
        return _mnist_generator_resize_conv_learned_feature_drift_queue_core_sweep()
    if preset == "mnist_generator_horn_resize_conv_core":
        return _mnist_generator_horn_resize_conv_core_sweep()
    if preset == "mnist_generator_horn_conditioning_attribution_probe":
        return _mnist_generator_horn_conditioning_attribution_probe_sweep()
    if preset == "mnist_generator_horn_label0_replication_probe":
        return _mnist_generator_horn_label0_replication_probe_sweep()
    if preset == "mnist_generator_state_mlp_label0_control_probe":
        return _mnist_generator_state_mlp_label0_control_probe_sweep()
    if preset == "mnist_generator_horn_state_mlp_low_data_probe":
        return _mnist_generator_horn_state_mlp_low_data_probe_sweep()
    if preset == "mnist_generator_horn_state_mlp_settling_probe":
        return _mnist_generator_horn_state_mlp_settling_probe_sweep()
    if preset == "mnist_generator_horn_settling_train_probe":
        return _mnist_generator_horn_settling_train_probe_sweep()
    if preset == "mnist_generator_horn_structured_coupling_probe":
        return _mnist_generator_horn_structured_coupling_probe_sweep()
    if preset == "mnist_generator_horn_sparse_coupling_probe":
        return _mnist_generator_horn_sparse_coupling_probe_sweep()
    if preset == "mnist_generator_sparse_horn_state_mlp_replication_probe":
        return _mnist_generator_sparse_horn_state_mlp_replication_probe_sweep()
    if preset == "mnist_generator_sparse_horn_attribution_probe":
        return _mnist_generator_sparse_horn_attribution_probe_sweep()
    if preset == "mnist_generator_sparse_horn_conditioning_route_probe":
        return _mnist_generator_sparse_horn_conditioning_route_probe_sweep()
    if preset == "mnist_generator_sparse_horn_class_coupling_sharpen_probe":
        return _mnist_generator_sparse_horn_class_coupling_sharpen_probe_sweep()
    if preset == "mnist_generator_sparse_horn_class_coupling_strong_control_probe":
        return (
            _mnist_generator_sparse_horn_class_coupling_strong_control_probe_sweep()
        )
    if preset == "mnist_generator_sparse_horn_class_coupling_strength_probe":
        return _mnist_generator_sparse_horn_class_coupling_strength_probe_sweep()
    if preset == "mnist_generator_sparse_horn_state_mlp_diversity_probe":
        return _mnist_generator_sparse_horn_state_mlp_diversity_probe_sweep()
    if preset == "mnist_generator_sparse_horn_distributional_probe":
        return _mnist_generator_sparse_horn_distributional_probe_sweep()
    if preset == "mnist_generator_sparse_horn_quality_classifier_audit":
        return _mnist_generator_sparse_horn_quality_classifier_audit_sweep()
    if preset == "mnist_generator_sparse_horn_dynamics_quality_probe":
        return _mnist_generator_sparse_horn_dynamics_quality_probe_sweep()
    if preset == "mnist_generator_sparse_horn_damping_distributional_probe":
        return _mnist_generator_sparse_horn_damping_distributional_probe_sweep()
    if preset == "mnist_generator_sparse_horn_recommended_ablation_probe":
        return _mnist_generator_sparse_horn_recommended_ablation_probe_sweep()
    if preset == "mnist_generator_sparse_horn_state_mlp_strength8_diversity_probe":
        return _mnist_generator_sparse_horn_state_mlp_strength8_diversity_probe_sweep()
    if preset == "mnist_generator_fashion_mnist_frontier_probe":
        return _mnist_generator_fashion_mnist_frontier_probe_sweep()
    if preset == "mnist_generator_fashion_mnist_readout_capacity_probe":
        return _mnist_generator_fashion_mnist_readout_capacity_probe_sweep()
    if preset == "mnist_generator_fashion_mnist_horn_calibration_probe":
        return _mnist_generator_fashion_mnist_horn_calibration_probe_sweep()
    if preset == "mnist_generator_cifar10_gray_frontier_probe":
        return _mnist_generator_cifar10_gray_frontier_probe_sweep()
    if preset == "mnist_generator_cifar10_gray_convjudge_frontier_probe":
        return _mnist_generator_cifar10_gray_convjudge_frontier_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_frontier_probe":
        return _mnist_generator_cifar10_rgb_frontier_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_feature_metric_audit":
        return _mnist_generator_cifar10_rgb_feature_metric_audit_sweep()
    if preset == "mnist_generator_cifar10_rgb_judge_audit":
        return _mnist_generator_cifar10_rgb_judge_audit_sweep()
    if preset == "mnist_generator_cifar10_rgb_semantic_feature_drift_probe":
        return _mnist_generator_cifar10_rgb_semantic_feature_drift_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_semantic_feature_drift_attribution":
        return (
            _mnist_generator_cifar10_rgb_semantic_feature_drift_attribution_sweep()
        )
    if preset == "mnist_generator_cifar10_rgb_attractor_robustness_probe":
        return _mnist_generator_cifar10_rgb_attractor_robustness_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat":
        return _mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat_sweep()
    if preset == "mnist_generator_cifar10_rgb_main_coupling_strength_probe":
        return _mnist_generator_cifar10_rgb_main_coupling_strength_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat":
        return _mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat_sweep()
    if preset == "mnist_generator_cifar10_rgb_main_coupling_fine_probe":
        return _mnist_generator_cifar10_rgb_main_coupling_fine_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_main_coupling_current_replication":
        return _mnist_generator_cifar10_rgb_main_coupling_current_replication_sweep()
    if preset == "mnist_generator_cifar10_rgb_normalized_distance_decay_probe":
        return _mnist_generator_cifar10_rgb_normalized_distance_decay_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_normalized_local_radius_probe":
        return _mnist_generator_cifar10_rgb_normalized_local_radius_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_normalized_local_radius_sweep":
        return _mnist_generator_cifar10_rgb_normalized_local_radius_sweep()
    if preset == "mnist_generator_cifar10_rgb_coarse_to_fine_probe":
        return _mnist_generator_cifar10_rgb_coarse_to_fine_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_coarse_to_fine_gain_sweep":
        return _mnist_generator_cifar10_rgb_coarse_to_fine_gain_sweep()
    if preset == "mnist_generator_cifar10_rgb_coarse_to_fine_profile_probe":
        return _mnist_generator_cifar10_rgb_coarse_to_fine_profile_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_audit":
        return _mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_audit()
    if preset == "mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat":
        return _mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat()
    if preset == "mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe":
        return _mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe()
    if preset == "mnist_generator_cifar10_rgb_coarse_to_fine_feedback_probe":
        return _mnist_generator_cifar10_rgb_coarse_to_fine_feedback_probe()
    if preset == "mnist_generator_cifar10_rgb_multiscale_layered_probe":
        return _mnist_generator_cifar10_rgb_multiscale_layered_probe()
    if preset == "mnist_generator_cifar10_rgb_multiscale_auxiliary_probe":
        return _mnist_generator_cifar10_rgb_multiscale_auxiliary_probe()
    if preset == "mnist_generator_cifar10_rgb_multiscale_gated_probe":
        return _mnist_generator_cifar10_rgb_multiscale_gated_probe()
    if preset == "mnist_generator_cifar10_rgb_multiscale_gain_probe":
        return _mnist_generator_cifar10_rgb_multiscale_gain_probe()
    if preset == "mnist_generator_cifar10_rgb_multiscale_weak_drive_probe":
        return _mnist_generator_cifar10_rgb_multiscale_weak_drive_probe()
    if preset == "mnist_generator_cifar10_rgb_multiscale_signed_gain_probe":
        return _mnist_generator_cifar10_rgb_multiscale_signed_gain_probe()
    if preset == "mnist_generator_cifar10_rgb_multiscale_soft_gate_probe":
        return _mnist_generator_cifar10_rgb_multiscale_soft_gate_probe()
    if preset == "mnist_generator_cifar10_rgb_vertical_causality_audit":
        return _mnist_generator_cifar10_rgb_vertical_causality_audit()
    if preset == "mnist_generator_cifar10_rgb_vertical_calibration_probe":
        return _mnist_generator_cifar10_rgb_vertical_calibration_probe()
    if preset == "mnist_generator_cifar10_rgb_dual_gain_probe":
        return _mnist_generator_cifar10_rgb_dual_gain_probe()
    if preset == "mnist_generator_cifar10_rgb_vertical_homeostasis_probe":
        return _mnist_generator_cifar10_rgb_vertical_homeostasis_probe()
    if preset == "mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration":
        return _mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration()
    if preset == "mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe":
        return _mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe()
    if preset == "mnist_generator_cifar10_rgb_centered_signed_gain_target_probe":
        return _mnist_generator_cifar10_rgb_centered_signed_gain_target_probe()
    if preset == "mnist_generator_cifar10_rgb_coarse_objective_probe":
        return _mnist_generator_cifar10_rgb_coarse_objective_probe()
    if preset == "mnist_generator_cifar10_rgb_feedback_signal_probe":
        return _mnist_generator_cifar10_rgb_feedback_signal_probe()
    if preset == "mnist_generator_cifar10_rgb_feedback_source_gate_probe":
        return _mnist_generator_cifar10_rgb_feedback_source_gate_probe()
    if preset == "mnist_generator_cifar10_rgb_feedback_source_mix_probe":
        return _mnist_generator_cifar10_rgb_feedback_source_mix_probe()
    if preset == "mnist_generator_cifar10_rgb_feedback_source_mix_auxfix_probe":
        return _mnist_generator_cifar10_rgb_feedback_source_mix_auxfix_probe()
    if preset == "mnist_generator_cifar10_rgb_readout_fusion_probe":
        return _mnist_generator_cifar10_rgb_readout_fusion_probe()
    if preset == "mnist_generator_cifar10_rgb_coarse_readout_consistency_probe":
        return _mnist_generator_cifar10_rgb_coarse_readout_consistency_probe()
    if preset == "mnist_generator_cifar10_rgb_readout_gate_probe":
        return _mnist_generator_cifar10_rgb_readout_gate_probe()
    if preset == "mnist_generator_cifar10_rgb_frequency_objective_probe":
        return _mnist_generator_cifar10_rgb_frequency_objective_probe()
    if preset == "mnist_generator_cifar10_rgb_patch_objective_probe":
        return _mnist_generator_cifar10_rgb_patch_objective_probe()
    if preset == "mnist_generator_cifar10_rgb_patch_v2_probe":
        return _mnist_generator_cifar10_rgb_patch_v2_probe()
    if preset == "mnist_generator_cifar10_rgb_state_information_probe":
        return _mnist_generator_cifar10_rgb_state_information_probe()
    if preset == "mnist_generator_cifar10_rgb_state_residual_readout_probe":
        return _mnist_generator_cifar10_rgb_state_residual_readout_probe()
    if preset == "mnist_generator_cifar10_rgb_state_residual_longer_pilot":
        return _mnist_generator_cifar10_rgb_state_residual_longer_pilot()
    if preset == "mnist_generator_cifar10_rgb_resonant_readout_pilot":
        return _mnist_generator_cifar10_rgb_resonant_readout_pilot()
    if preset == "mnist_generator_cifar10_rgb_capacity_probe":
        return _mnist_generator_cifar10_rgb_capacity_probe()
    if preset == "mnist_generator_cifar10_rgb_multimode_probe":
        return _mnist_generator_cifar10_rgb_multimode_probe()
    if preset == "mnist_generator_cifar10_rgb_retinotopic_readout_probe":
        return _mnist_generator_cifar10_rgb_retinotopic_readout_probe()
    if preset == "mnist_generator_cifar10_rgb_retinotopic_control_probe":
        return _mnist_generator_cifar10_rgb_retinotopic_control_probe()
    if preset == "mnist_generator_cifar10_rgb_state_anchor_probe":
        return _mnist_generator_cifar10_rgb_state_anchor_probe()
    if preset == "mnist_generator_cifar10_rgb_state_prior_training_probe":
        return _mnist_generator_cifar10_rgb_state_prior_training_probe()
    if preset == "mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1":
        return _mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1()
    if preset == "mnist_generator_cifar10_rgb_state_prior_control_probe":
        return _mnist_generator_cifar10_rgb_state_prior_control_probe()
    if preset == "mnist_generator_cifar10_rgb_attribution_probe":
        return _mnist_generator_cifar10_rgb_attribution_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_sparse_drive_probe":
        return _mnist_generator_cifar10_rgb_sparse_drive_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_sparse_drive_seed_repeat":
        return _mnist_generator_cifar10_rgb_sparse_drive_seed_repeat_sweep()
    if preset == "mnist_generator_cifar10_rgb_structured_drive_probe":
        return _mnist_generator_cifar10_rgb_structured_drive_probe_sweep()
    if preset == "mnist_generator_cifar10_rgb_coherent_drive_probe":
        return _mnist_generator_cifar10_rgb_coherent_drive_probe_sweep()
    raise ValueError("unknown sweep preset")


def _vertical_audit_metric_names() -> list[str]:
    """CSV columns for sample-time vertical intervention audit metrics."""

    modes = ("normal", "zero", "shuffle", "flip", "scale025", "scale050")
    base_metrics = (
        "classifier_label_accuracy",
        "diversity_ratio",
        "nearest_real_mse",
        "classifier_feature_diversity_ratio",
        "classifier_feature_nearest_real_mse",
        "attractor_label_accuracy",
        "attractor_pixel_attractor_diversity_score",
        "output_mse_vs_normal",
        "trace_vertical_gain_mean",
        "trace_vertical_gain_std",
        "trace_vertical_gain_negative_fraction",
        "trace_vertical_gain_below_one_fraction",
        "trace_vertical_gain_target_minus_non_target_mean",
        "trace_vertical_modulation_mean",
        "trace_vertical_modulation_negative_fraction",
    )
    delta_metrics = (
        "delta_classifier_label_accuracy",
        "delta_diversity_ratio",
        "delta_nearest_real_mse",
        "delta_classifier_feature_diversity_ratio",
        "delta_classifier_feature_nearest_real_mse",
        "delta_attractor_label_accuracy",
        "delta_attractor_pixel_attractor_diversity_score",
    )
    names = []
    for mode in modes:
        prefix = f"generator.vertical_intervention_audit.{mode}"
        names.extend(f"{prefix}.{metric}" for metric in base_metrics)
        if mode != "normal":
            names.extend(f"{prefix}.{metric}" for metric in delta_metrics)
    return names


def _write_sweep_csv(results: list[dict[str, Any]], path: Path) -> None:
    metric_names = [
        "final_eval_loss",
        "final_train_coarse_auxiliary_loss",
        "final_eval_coarse_auxiliary_loss",
        "final_train_coarse_readout_consistency_loss",
        "final_eval_coarse_readout_consistency_loss",
        "final_train_frequency_objective_loss",
        "final_eval_frequency_objective_loss",
        "final_train_patch_objective_loss",
        "final_eval_patch_objective_loss",
        "final_train_state_anchor_loss",
        "final_eval_state_anchor_loss",
        "final_train_state_prior_sampling_active",
        "final_state_prior_refit_seconds",
        "best_loss",
        "best_epoch",
        "final_epoch",
        "generator.diversity_ratio",
        "generator.pixel_mean_mse",
        "generator.pixel_std_mse",
        "generator.nearest_real_mse",
        "generator.nearest_real_mse_min",
        "generator.nearest_real_mse_p01",
        "generator.nearest_real_mse_p05",
        "generator.duplicate_rate_mse_001",
        "generator.duplicate_rate_mse_0025",
        "generator.duplicate_rate_mse_005",
        "generator.duplicate_rate_mse_010",
        "generator.real_nearest_real_mse",
        "generator.prototype_mse",
        "generator.prototype_nearest_accuracy",
        "generator.dynamics_family",
        "generator.num_oscillators",
        "generator.num_spatial_sites",
        "generator.num_modes",
        "generator.mode_frequency_scales",
        "generator.mode_coupling_strength",
        "generator.mode_coupling_profile",
        "generator.horn_frequency",
        "generator.horn_damping",
        "generator.horn_nonlinearity",
        "generator.horn_state_bound",
        "generator.output_feedback_mode",
        "generator.output_feedback_strength",
        "generator.output_feedback_init_scale",
        "generator.output_feedback_basis_sigma",
        "generator.state_residual_readout_strength",
        "generator.state_residual_readout_init_scale",
        "generator.state_residual_readout_patch_size",
        "generator.state_residual_readout_sigma",
        "generator.resonant_readout_strength",
        "generator.resonant_readout_init_scale",
        "generator.resonant_readout_patch_size",
        "generator.resonant_readout_sigma",
        "generator.state_mlp_hidden_dim",
        "generator.state_mlp_depth",
        "generator.state_mlp_residual_scale",
        "generator.num_coarse_oscillators",
        "generator.coarse_coupling_profile",
        "generator.coarse_coupling_normalization",
        "generator.coarse_coupling_length_scale",
        "generator.coarse_to_fine_strength",
        "generator.coarse_to_fine_profile",
        "generator.coarse_to_fine_normalization",
        "generator.coarse_to_fine_length_scale",
        "generator.coarse_to_fine_floor",
        "generator.coarse_conditioning_strength",
        "generator.coarse_auxiliary_weight",
        "generator.coarse_auxiliary_target_size",
        "generator.coarse_auxiliary_loss_mode",
        "generator.coarse_readout_consistency_weight",
        "generator.coarse_readout_consistency_onset_epoch",
        "generator.frequency_objective_weight",
        "generator.frequency_objective_edge_weight",
        "generator.patch_objective_weight",
        "generator.patch_objective_patch_size",
        "generator.patch_objective_patch_sizes",
        "generator.patch_objective_stride",
        "generator.patch_objective_offsets",
        "generator.patch_objective_projections",
        "generator.patch_objective_edge_weight",
        "generator.state_anchor_weight",
        "generator.state_anchor_steps",
        "generator.state_anchor_noise_scale",
        "generator.state_anchor_mode",
        "generator.state_anchor_encoder_kernel_size",
        "generator.state_prior_sampling_mode",
        "generator.state_prior_rank",
        "generator.state_prior_noise_scale",
        "generator.state_prior_refresh_epochs",
        "generator.state_prior_start_epoch",
        "generator.sample_initialization",
        "generator.state_prior_artifacts.json_path",
        "generator.state_prior_artifacts.npz_path",
        "generator.state_prior_artifacts.final_refit_seconds",
        "generator.white_noise_quality.classifier_label_accuracy",
        "generator.white_noise_quality.classifier_feature_diversity_ratio",
        "generator.white_noise_quality.classifier_feature_nearest_real_mse",
        "generator.white_noise_quality.nearest_real_mse",
        "generator.white_noise_settling.classifier_label_accuracy_best",
        "generator.white_noise_attractor_robustness.label_accuracy",
        "generator.shuffled_prior_quality.classifier_label_accuracy",
        "generator.shuffled_prior_quality.classifier_feature_diversity_ratio",
        "generator.shuffled_prior_quality.classifier_feature_nearest_real_mse",
        "generator.shuffled_prior_quality.nearest_real_mse",
        "generator.shuffled_prior_settling.classifier_label_accuracy_best",
        "generator.shuffled_prior_attractor_robustness.label_accuracy",
        "generator.multiscale_feedback_signal_mode",
        "generator.multiscale_feedback_source_gate",
        "generator.multiscale_feedback_source_mix",
        "generator.multiscale_readout_fusion_strength",
        "generator.multiscale_readout_gate_mode",
        "generator.multiscale_readout_gate_strength",
        "generator.multiscale_readout_gate_init_scale",
        "generator.conditional",
        "generator.label_phase_scale",
        "generator.conditioning_mode",
        "generator.coupling_profile",
        "generator.coupling_normalization",
        "generator.coupling_strength",
        "generator.main_coupling_strength",
        "generator.coupling_length_scale",
        "generator.coupling_floor",
        "generator.coupling_bias_strength",
        "generator.conditioning_strength",
        "generator.conditioning_target_fraction",
        "generator.conditioning_target_pattern",
        "generator.conditioning_target_count",
        "generator.train_recurrent_dynamics",
        "generator.train_conditioning_dynamics",
        "generator.readout_mode",
        "generator.decoder_mode",
        "generator.loss",
        "generator.pixel_drift_weight",
        "generator.feature_drift_weight",
        "generator.feature_drift_mode",
        "generator.learned_feature_kind",
        "generator.learned_feature_epochs",
        "generator.learned_feature_dim",
        "generator.learned_feature_depth",
        "generator.feature_classifier.classifier_kind",
        "generator.feature_classifier.final_eval_accuracy",
        "generator.feature_classifier.final_eval_loss",
        "generator.feature_classifier.final_train_accuracy",
        "generator.feature_classifier.epochs",
        "generator.feature_classifier.feature_dim",
        "generator.quality_classifier.final_eval_accuracy",
        "generator.quality_classifier.final_eval_loss",
        "generator.quality_classifier.final_train_accuracy",
        "generator.quality_classifier.epochs",
        "generator.quality_classifier.feature_dim",
        "generator.quality_classifier.classifier_kind",
        "generator.quality_classifier_epochs",
        "generator.quality_classifier_kind",
        "generator.quality_classifier_dim",
        "generator.quality_classifier_depth",
        "generator.quality_classifier_train_limit",
        "generator.quality_classifier_eval_limit",
        "generator.dataset_name",
        "generator.data_source",
        "generator.image_shape",
        "generator.classifier_label_accuracy",
        "generator.classifier_label_confidence",
        "generator.classifier_max_confidence",
        "generator.classifier_entropy",
        "generator.classifier_feature_mean_mse",
        "generator.classifier_feature_std_mse",
        "generator.classifier_feature_diversity_ratio",
        "generator.classifier_feature_nearest_real_mse",
        "generator.classifier_feature_real_nearest_real_mse",
        "generator.classifier_feature_pairwise_distance_ratio",
        "generator.classifier_feature_frechet_distance",
        "generator.classifier_feature_kid_mmd2",
        "generator.classifier_feature_precision_at_real_median",
        "generator.classifier_feature_recall_at_real_median",
        "generator.classifier_feature_real_median_radius",
        "generator.frequency_real_low_power_ratio",
        "generator.frequency_generated_low_power_ratio",
        "generator.frequency_real_mid_power_ratio",
        "generator.frequency_generated_mid_power_ratio",
        "generator.frequency_real_high_power_ratio",
        "generator.frequency_generated_high_power_ratio",
        "generator.frequency_high_power_ratio_delta",
        "generator.frequency_high_power_ratio",
        "generator.frequency_spectral_centroid_real",
        "generator.frequency_spectral_centroid_generated",
        "generator.frequency_spectral_centroid_ratio",
        "generator.edge_laplacian_variance_real",
        "generator.edge_laplacian_variance_generated",
        "generator.edge_laplacian_variance_ratio",
        "generator.drift_queue_size",
        "generator.drift_queue_num_pos",
        "generator.distributional_weight",
        "generator.class_moment_weight",
        "generator.prototype_weight",
        "generator.moment_weight",
        "generator.pixel_marginal_weight",
        "generator.num_projections",
        "generator.drift_gamma",
        "generator.train_settling_steps",
        "generator.attractor_variants_per_class",
        "generator.state_probe_sample_count",
        "generator.state_probe_target_size",
        "generator.state_fit_sample_count",
        "generator.state_fit_steps",
        "generator.state_fit_learning_rate",
        "generator.state_fit_init_scale",
        "generator.state_fit_ridge",
        "generator.state_fit_settle_steps",
        "generator.state_information_probe.sample_count",
        "generator.state_information_probe.fine_final_minus_initial_label_accuracy",
        "generator.state_information_probe.fine_final_minus_initial_label_r2",
        "generator.state_information_probe.fine_final_minus_initial_generated_lowres_r2",
        "generator.state_information_probe.fine_final_minus_initial_generated_highpass_r2",
        "generator.state_information_probe.fine_final_minus_initial_classifier_features_r2",
        "generator.state_information_probe.state_sets.fine_initial.label_accuracy",
        "generator.state_information_probe.state_sets.fine_initial.generated_lowres_r2",
        "generator.state_information_probe.state_sets.fine_initial.generated_highpass_r2",
        "generator.state_information_probe.state_sets.fine_final.label_accuracy",
        "generator.state_information_probe.state_sets.fine_final.generated_lowres_r2",
        "generator.state_information_probe.state_sets.fine_final.generated_highpass_r2",
        "generator.state_information_probe.state_sets.fine_final.classifier_features_r2",
        "generator.state_information_probe.state_sets.combined_final.label_accuracy",
        "generator.state_information_probe.state_sets.combined_final.generated_lowres_r2",
        "generator.state_information_probe.state_sets.combined_final.generated_highpass_r2",
        "generator.state_information_probe.state_sets.combined_final.classifier_features_r2",
        "generator.state_fitting_probe.sample_count",
        "generator.state_fitting_probe.fit_steps",
        "generator.state_fitting_probe.learning_rate",
        "generator.state_fitting_probe.initial_mse",
        "generator.state_fitting_probe.final_mse",
        "generator.state_fitting_probe.mse_delta",
        "generator.state_fitting_probe.fit_paired_mse",
        "generator.state_fitting_probe.fit_paired_l1",
        "generator.state_fitting_probe.fit_generated_std",
        "generator.state_fitting_probe.fit_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.fit_frequency_spectral_centroid_ratio",
        "generator.state_fitting_probe.fit_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.settle_000_paired_mse",
        "generator.state_fitting_probe.settle_001_paired_mse",
        "generator.state_fitting_probe.settle_002_paired_mse",
        "generator.state_fitting_probe.settle_004_paired_mse",
        "generator.state_fitting_probe.settle_008_paired_mse",
        "generator.state_fitting_probe.settle_016_paired_mse",
        "generator.state_fitting_probe.settle_032_paired_mse",
        "generator.state_fitting_probe.settle_001_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.settle_002_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.settle_004_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.settle_008_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.settle_016_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.settle_032_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.settle_001_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.settle_002_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.settle_004_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.settle_008_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.settle_016_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.settle_032_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.noise_001_paired_mse",
        "generator.state_fitting_probe.noise_002_paired_mse",
        "generator.state_fitting_probe.noise_004_paired_mse",
        "generator.state_fitting_probe.noise_008_paired_mse",
        "generator.state_fitting_probe.noise_001_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.noise_002_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.noise_004_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.noise_008_frequency_generated_high_power_ratio",
        "generator.state_fitting_probe.noise_001_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.noise_002_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.noise_004_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.noise_008_edge_laplacian_variance_ratio",
        "generator.state_fitting_probe.fresh_readout.real_full_r2",
        "generator.state_fitting_probe.fresh_readout.real_lowres_r2",
        "generator.state_fitting_probe.fresh_readout.real_highpass_r2",
        "generator.resize_conv_seed_size",
        "generator.resize_conv_upsamples",
        "generator.resize_conv_min_channels",
        "generator.resize_conv_seed_layout",
        "generator.resize_conv_seed_min_channels",
        "generator.success_diagnostics.total_params",
        "generator.success_diagnostics.recurrent_params",
        "generator.success_diagnostics.decoder_params",
        "generator.success_diagnostics.estimated_ops_per_sample",
        "generator.success_diagnostics.estimated_recurrent_ops_per_sample",
        "generator.success_diagnostics.estimated_decoder_ops_per_sample",
        "generator.success_diagnostics.dynamics_family",
        "generator.success_diagnostics.decoder_mode",
        "generator.success_diagnostics.decoder_param_fraction",
        "generator.success_diagnostics.resize_conv_seed_layout",
        "generator.success_diagnostics.resize_conv_seed_min_channels",
        "generator.success_diagnostics.trainable_recurrent_param_fraction",
        "generator.success_diagnostics.train_recurrent_dynamics",
        "generator.success_diagnostics.train_conditioning_dynamics",
        "generator.success_diagnostics.trainable_main_recurrent_params",
        "generator.success_diagnostics.trainable_conditioning_params",
        "generator.success_diagnostics.coarse_recurrent_params",
        "generator.success_diagnostics.coarse_conditioning_params",
        "generator.success_diagnostics.auxiliary_recurrent_params",
        "generator.success_diagnostics.vertical_recurrent_params",
        "generator.success_diagnostics.multiscale_recurrent_params",
        "generator.success_diagnostics.multiscale_conditioning_params",
        "generator.success_diagnostics.transition_params",
        "generator.success_diagnostics.estimated_recurrent_op_fraction",
        "generator.success_diagnostics.coupling_profile",
        "generator.success_diagnostics.coupling_normalization",
        "generator.success_diagnostics.coupling_strength",
        "generator.success_diagnostics.main_coupling_strength",
        "generator.success_diagnostics.coupling_density",
        "generator.success_diagnostics.coupling_profile_mean",
        "generator.success_diagnostics.coupling_profile_std",
        "generator.success_diagnostics.coupling_profile_min",
        "generator.success_diagnostics.coupling_profile_max",
        "generator.success_diagnostics.coupling_profile_row_sum_mean",
        "generator.success_diagnostics.coupling_profile_row_sum_std",
        "generator.success_diagnostics.coupling_profile_row_sum_min",
        "generator.success_diagnostics.coupling_profile_row_sum_max",
        "generator.success_diagnostics.conditioning_strength",
        "generator.success_diagnostics.conditioning_target_fraction",
        "generator.success_diagnostics.conditioning_target_pattern",
        "generator.success_diagnostics.conditioning_target_count",
        "generator.success_diagnostics.conditioning_target_effective_fraction",
        "generator.success_diagnostics.num_coarse_oscillators",
        "generator.success_diagnostics.coarse_coupling_profile",
        "generator.success_diagnostics.coarse_coupling_normalization",
        "generator.success_diagnostics.coarse_coupling_length_scale",
        "generator.success_diagnostics.coarse_to_fine_strength",
        "generator.success_diagnostics.coarse_to_fine_profile",
        "generator.success_diagnostics.coarse_to_fine_normalization",
        "generator.success_diagnostics.coarse_to_fine_length_scale",
        "generator.success_diagnostics.coarse_to_fine_floor",
        "generator.success_diagnostics.coarse_to_fine_profile_density",
        "generator.success_diagnostics.coarse_to_fine_profile_row_sum_mean",
        "generator.success_diagnostics.coarse_to_fine_profile_row_sum_std",
        "generator.success_diagnostics.coarse_to_fine_profile_row_sum_min",
        "generator.success_diagnostics.coarse_to_fine_profile_row_sum_max",
        "generator.success_diagnostics.coarse_conditioning_strength",
        "generator.success_diagnostics.auxiliary_readout_params",
        "generator.success_diagnostics.multiscale_readout_gate_params",
        "generator.success_diagnostics.multiscale_layer_sizes",
        "generator.success_diagnostics.multiscale_frequency_scales",
        "generator.success_diagnostics.multiscale_coupling_profile",
        "generator.success_diagnostics.multiscale_coupling_normalization",
        "generator.success_diagnostics.multiscale_coupling_length_scale",
        "generator.success_diagnostics.multiscale_vertical_strength",
        "generator.success_diagnostics.multiscale_feedback_strength",
        "generator.success_diagnostics.multiscale_vertical_profile",
        "generator.success_diagnostics.multiscale_vertical_normalization",
        "generator.success_diagnostics.multiscale_vertical_length_scale",
        "generator.success_diagnostics.multiscale_vertical_phase_lag",
        "generator.success_diagnostics.multiscale_feedback_phase_lag",
        "generator.success_diagnostics.multiscale_vertical_signal_scale",
        "generator.success_diagnostics.multiscale_feedback_signal_mode",
        "generator.success_diagnostics.multiscale_feedback_source_gate",
        "generator.success_diagnostics.multiscale_feedback_source_mix",
        "generator.success_diagnostics.multiscale_vertical_target_gate",
        "generator.success_diagnostics.multiscale_vertical_soft_gate_floor",
        "generator.success_diagnostics.multiscale_vertical_mode",
        "generator.success_diagnostics.multiscale_vertical_gain_target",
        "generator.success_diagnostics.multiscale_vertical_gain_normalization",
        "generator.success_diagnostics.multiscale_vertical_gain_target_std",
        "generator.success_diagnostics.multiscale_vertical_broad_gain_scale",
        "generator.success_diagnostics.multiscale_vertical_selective_gain_scale",
        "generator.success_diagnostics.multiscale_vertical_schedule",
        "generator.success_diagnostics.multiscale_vertical_onset_step",
        "generator.success_diagnostics.multiscale_vertical_ramp_steps",
        "generator.success_diagnostics.multiscale_conditioning_strength",
        "generator.success_diagnostics.multiscale_auxiliary_readout_layer",
        "generator.success_diagnostics.multiscale_auxiliary_readout_size",
        "generator.success_diagnostics.multiscale_readout_fusion_strength",
        "generator.success_diagnostics.multiscale_readout_gate_mode",
        "generator.success_diagnostics.multiscale_readout_gate_strength",
        "generator.success_diagnostics.multiscale_readout_gate_init_scale",
        "generator.success_diagnostics.num_auxiliary_layers",
        "generator.success_diagnostics.num_vertical_couplings",
        "generator.success_diagnostics.vertical_profile_density",
        "generator.success_diagnostics.vertical_profile_row_sum_mean",
        "generator.success_diagnostics.vertical_profile_row_sum_std",
        "generator.success_diagnostics.vertical_gain_mean",
        "generator.success_diagnostics.vertical_gain_std",
        "generator.success_diagnostics.vertical_gain_min",
        "generator.success_diagnostics.vertical_gain_max",
        "generator.success_diagnostics.vertical_gain_negative_fraction",
        "generator.success_diagnostics.vertical_gain_below_one_fraction",
        "generator.success_diagnostics.vertical_gain_above_one_fraction",
        "generator.success_diagnostics.vertical_gain_target_mean",
        "generator.success_diagnostics.vertical_gain_non_target_mean",
        "generator.success_diagnostics.vertical_gain_target_minus_non_target_mean",
        "generator.success_diagnostics.vertical_modulation_mean",
        "generator.success_diagnostics.vertical_modulation_negative_fraction",
        "generator.success_diagnostics.state_initial_spatial_high_power_ratio",
        "generator.success_diagnostics.state_final_spatial_high_power_ratio",
        "generator.success_diagnostics.state_spatial_high_power_ratio_delta",
        "generator.success_diagnostics.state_initial_spatial_spectral_centroid",
        "generator.success_diagnostics.state_final_spatial_spectral_centroid",
        "generator.success_diagnostics.state_spatial_spectral_centroid_delta",
        "generator.success_diagnostics.velocity_final_spatial_high_power_ratio",
        "generator.success_diagnostics.output_feedback_mode",
        "generator.success_diagnostics.output_feedback_strength",
        "generator.success_diagnostics.output_feedback_init_scale",
        "generator.success_diagnostics.output_feedback_basis_sigma",
        "generator.success_diagnostics.state_residual_readout_params",
        "generator.success_diagnostics.state_residual_readout_strength",
        "generator.success_diagnostics.state_residual_readout_init_scale",
        "generator.success_diagnostics.state_residual_readout_patch_size",
        "generator.success_diagnostics.state_residual_readout_sigma",
        "generator.success_diagnostics.resonant_readout_params",
        "generator.success_diagnostics.state_anchor_encoder_params",
        "generator.success_diagnostics.state_anchor_encoder_enabled",
        "generator.success_diagnostics.resonant_readout_strength",
        "generator.success_diagnostics.resonant_readout_init_scale",
        "generator.success_diagnostics.resonant_readout_patch_size",
        "generator.success_diagnostics.resonant_readout_sigma",
        "generator.success_diagnostics.num_oscillators",
        "generator.success_diagnostics.num_spatial_sites",
        "generator.success_diagnostics.num_modes",
        "generator.success_diagnostics.mode_frequency_scales",
        "generator.success_diagnostics.mode_coupling_strength",
        "generator.success_diagnostics.mode_coupling_profile",
        "generator.success_diagnostics.output_feedback_params",
        "generator.success_diagnostics.estimated_output_feedback_ops_per_sample",
        "generator.success_diagnostics.estimated_output_feedback_op_fraction",
        "generator.success_diagnostics.samples_per_train_second",
        "generator.success_diagnostics.phase_mean_abs_displacement",
        "generator.success_diagnostics.phase_final_order",
        "generator.success_diagnostics.phase_order_delta",
        "generator.success_diagnostics.state_final_energy",
        "generator.success_diagnostics.state_mean_abs_velocity_displacement",
        "generator.success_diagnostics.state_energy_initial",
        "generator.success_diagnostics.state_energy_final",
        "generator.success_diagnostics.state_energy_delta",
        "generator.success_diagnostics.state_velocity_rms_initial",
        "generator.success_diagnostics.state_velocity_rms_final",
        "generator.success_diagnostics.state_velocity_rms_delta",
        "generator.success_diagnostics.state_update_rms_initial",
        "generator.success_diagnostics.state_update_rms_final",
        "generator.success_diagnostics.state_update_rms_mean",
        "generator.success_diagnostics.state_update_rms_settling_ratio",
        "generator.success_diagnostics.state_acceleration_rms_initial",
        "generator.success_diagnostics.state_acceleration_rms_final",
        "generator.success_diagnostics.state_acceleration_rms_mean",
        "generator.success_diagnostics.state_acceleration_rms_settling_ratio",
        "generator.success_diagnostics.state_path_length_rms",
        "generator.success_diagnostics.state_net_displacement_rms",
        "generator.success_diagnostics.state_path_efficiency_ratio",
        "generator.success_diagnostics.coupling_potential_proxy_initial",
        "generator.success_diagnostics.coupling_potential_proxy_final",
        "generator.success_diagnostics.coupling_potential_proxy_delta",
        "generator.success_diagnostics.coarse_state_energy_initial",
        "generator.success_diagnostics.coarse_state_energy_final",
        "generator.success_diagnostics.coarse_state_energy_delta",
        "generator.success_diagnostics.coarse_state_update_rms_initial",
        "generator.success_diagnostics.coarse_state_update_rms_final",
        "generator.success_diagnostics.coarse_state_update_rms_mean",
        "generator.success_diagnostics.coarse_state_update_rms_settling_ratio",
        "generator.success_diagnostics.coarse_state_acceleration_rms_initial",
        "generator.success_diagnostics.coarse_state_acceleration_rms_final",
        "generator.success_diagnostics.coarse_state_acceleration_rms_mean",
        "generator.success_diagnostics.coarse_state_acceleration_rms_settling_ratio",
        "generator.success_diagnostics.coarse_state_path_length_rms",
        "generator.success_diagnostics.coarse_state_net_displacement_rms",
        "generator.success_diagnostics.coarse_state_path_efficiency_ratio",
        "generator.success_diagnostics.coarse_coupling_potential_proxy_initial",
        "generator.success_diagnostics.coarse_coupling_potential_proxy_final",
        "generator.success_diagnostics.coarse_coupling_potential_proxy_delta",
        "generator.success_diagnostics.coarse_to_fine_potential_proxy_initial",
        "generator.success_diagnostics.coarse_to_fine_potential_proxy_final",
        "generator.success_diagnostics.coarse_to_fine_potential_proxy_delta",
        "generator.success_diagnostics.vertical_potential_proxy_delta_mean",
        "generator.success_diagnostics.output_step_mse_initial",
        "generator.success_diagnostics.output_step_mse_final",
        "generator.success_diagnostics.output_step_mse_mean",
        "generator.success_diagnostics.output_step_mse_settling_ratio",
        "generator.success_diagnostics.output_path_mse",
        "generator.success_diagnostics.output_net_mse",
        "generator.success_diagnostics.output_path_efficiency_ratio",
        "generator.attractor_robustness.num_classes",
        "generator.attractor_robustness.variants_per_class",
        "generator.attractor_robustness.sample_count",
        "generator.attractor_robustness.label_accuracy",
        "generator.attractor_robustness.label_confidence",
        "generator.attractor_robustness.max_confidence",
        "generator.attractor_robustness.entropy",
        "generator.attractor_robustness.class_success_fraction",
        "generator.attractor_robustness.class_accuracy_min",
        "generator.attractor_robustness.class_accuracy_max",
        "generator.attractor_robustness.pixel_within_class_pairwise_mse",
        "generator.attractor_robustness.pixel_within_class_std",
        "generator.attractor_robustness.pixel_between_class_centroid_mse",
        "generator.attractor_robustness.pixel_separation_ratio",
        "generator.attractor_robustness.pixel_attractor_diversity_score",
        "generator.attractor_robustness.feature_within_class_pairwise_distance",
        "generator.attractor_robustness.feature_within_class_std",
        "generator.attractor_robustness.feature_between_class_centroid_distance",
        "generator.attractor_robustness.feature_separation_ratio",
        "generator.attractor_robustness.feature_attractor_diversity_score",
        *_vertical_audit_metric_names(),
        "generator.settling.classifier_label_accuracy_best_step",
        "generator.settling.classifier_label_accuracy_best",
        "generator.settling.classifier_label_accuracy_last_minus_first",
        "generator.settling.classifier_label_confidence_best_step",
        "generator.settling.classifier_label_confidence_best",
        "generator.settling.classifier_label_confidence_last_minus_first",
        "generator.settling.classifier_feature_diversity_ratio_best_step",
        "generator.settling.classifier_feature_diversity_ratio_best",
        "generator.settling.classifier_feature_diversity_ratio_last_minus_first",
        "generator.settling.classifier_feature_nearest_real_mse_best_step",
        "generator.settling.classifier_feature_nearest_real_mse_best",
        "generator.settling.classifier_feature_nearest_real_mse_last_minus_first",
        "generator.settling.classifier_feature_pairwise_distance_ratio_best_step",
        "generator.settling.classifier_feature_pairwise_distance_ratio_best",
        "generator.settling.classifier_feature_pairwise_distance_ratio_last_minus_first",
        "generator.settling.diversity_ratio_best_step",
        "generator.settling.diversity_ratio_best",
        "generator.settling.diversity_ratio_last_minus_first",
        "generator.settling.nearest_real_mse_best_step",
        "generator.settling.nearest_real_mse_best",
        "generator.settling.nearest_real_mse_last_minus_first",
        "generator.settling.by_step.step_000.classifier_label_accuracy",
        "generator.settling.by_step.step_001.classifier_label_accuracy",
        "generator.settling.by_step.step_002.classifier_label_accuracy",
        "generator.settling.by_step.step_004.classifier_label_accuracy",
        "generator.settling.by_step.step_008.classifier_label_accuracy",
        "generator.settling.by_step.step_016.classifier_label_accuracy",
        "generator.settling.by_step.step_032.classifier_label_accuracy",
        "generator.settling.by_step.step_048.classifier_label_accuracy",
        "generator.settling.by_step.step_064.classifier_label_accuracy",
        "generator.settling.by_step.step_000.classifier_feature_diversity_ratio",
        "generator.settling.by_step.step_016.classifier_feature_diversity_ratio",
        "generator.settling.by_step.step_032.classifier_feature_diversity_ratio",
        "generator.settling.by_step.step_048.classifier_feature_diversity_ratio",
        "generator.settling.by_step.step_064.classifier_feature_diversity_ratio",
        "generator.settling.by_step.step_000.classifier_feature_nearest_real_mse",
        "generator.settling.by_step.step_016.classifier_feature_nearest_real_mse",
        "generator.settling.by_step.step_032.classifier_feature_nearest_real_mse",
        "generator.settling.by_step.step_048.classifier_feature_nearest_real_mse",
        "generator.settling.by_step.step_064.classifier_feature_nearest_real_mse",
        "generator.settling.by_step.step_000.diversity_ratio",
        "generator.settling.by_step.step_016.diversity_ratio",
        "generator.settling.by_step.step_032.diversity_ratio",
        "generator.settling.by_step.step_048.diversity_ratio",
        "generator.settling.by_step.step_064.diversity_ratio",
        "generator.settling.by_step.step_000.nearest_real_mse",
        "generator.settling.by_step.step_016.nearest_real_mse",
        "generator.settling.by_step.step_032.nearest_real_mse",
        "generator.settling.by_step.step_048.nearest_real_mse",
        "generator.settling.by_step.step_064.nearest_real_mse",
        "generator.generated_mean",
        "generator.generated_std",
        "final_train_pixel_drift_loss",
        "final_eval_pixel_drift_loss",
        "final_train_feature_drift_loss",
        "final_eval_feature_drift_loss",
        "final_train_drift_queue_ready",
    ]
    fieldnames = ["run", "root", "error", *metric_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(
            results,
            key=lambda row: _summary_metric(row.get("summary", {}), "final_eval_loss")
            or float("inf"),
        ):
            summary = result.get("summary", {})
            writer.writerow(
                {
                    "run": result.get("run_name"),
                    "root": result.get("output_dir"),
                    "error": result.get("error", ""),
                    **{
                        metric: _summary_metric(summary, metric)
                        for metric in metric_names
                    },
                }
            )


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    volumes={VOLUME_MOUNT: volume},
)
def run_mnist_generator_remote(
    experiment_args: list[str],
    run_name: str,
) -> dict[str, Any]:
    """Run one MNIST oscillator-generator experiment remotely."""

    import os

    safe_name = _safe_run_name(run_name)
    output_dir = VOLUME_MOUNT / "mnist_generator" / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (VOLUME_MOUNT / "jax_cache").mkdir(parents=True, exist_ok=True)

    cache_target = VOLUME_MOUNT / "cache" / "oscnet"
    cache_target.mkdir(parents=True, exist_ok=True)
    local_cache = Path.home() / ".cache" / "oscnet"
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    if not local_cache.exists():
        os.symlink(cache_target, local_cache, target_is_directory=True)

    import jax

    from oscnet.experiments.mnist_generator import (
        config_from_args,
        parse_args,
        run_mnist_generator_experiment,
    )

    args = _with_default_arg(list(experiment_args), "--output-dir", output_dir)
    parsed = parse_args(args)
    result = run_mnist_generator_experiment(config_from_args(parsed))

    summary_path = result.paths.metrics / "summary.json"
    history_path = result.paths.metrics / "history.json"
    summary = _load_json_if_exists(summary_path)
    history = _load_json_if_exists(history_path)
    history_tail = {
        key: values[-3:] if isinstance(values, list) else values
        for key, values in history.items()
    }
    payload = {
        "app": APP_NAME,
        "run_name": safe_name,
        "gpu": GPU,
        "jax_package": JAX_PACKAGE,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "output_dir": str(result.paths.root),
        "summary_path": str(summary_path),
        "history_path": str(history_path),
        "checkpoint_paths": result.checkpoint_paths,
        "summary": summary,
        "history_tail": history_tail,
    }
    with open(result.paths.root / "modal_result.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    try:
        volume.commit()
        payload["volume_commit_ok"] = True
    except Exception as exc:
        payload["volume_commit_ok"] = False
        payload["volume_commit_error"] = repr(exc)
        print(f"warning: volume.commit failed for {safe_name}: {exc!r}")
    return payload


@app.local_entrypoint()
def main(
    experiment_args: str = DEFAULT_SMOKE_ARGS,
    run_name: str = "",
    sweep_preset: str = "",
    sweep_csv: str = "",
    print_only: bool = False,
) -> None:
    """Launch a remote MNIST oscillator-generator experiment."""

    if sweep_preset:
        entries = _sweep_entries(sweep_preset)
        csv_path = Path(sweep_csv or SWEEP_CSVS[sweep_preset])
        request = {
            "sweep_preset": sweep_preset,
            "runs": [
                {"run_name": name, "experiment_args": args}
                for args, name in entries
            ],
            "gpu": GPU,
            "max_containers": MAX_CONTAINERS,
            "jax_package": JAX_PACKAGE,
            "timeout_seconds": TIMEOUT_SECONDS,
            "volume": VOLUME_NAME,
            "sweep_csv": str(csv_path),
        }
        if print_only:
            print(json.dumps(request, indent=2, sort_keys=True))
            return
        print(json.dumps(request, indent=2, sort_keys=True))
        raw_results = list(
            run_mnist_generator_remote.starmap(
                entries,
                order_outputs=True,
                return_exceptions=True,
            )
        )
        results = []
        for (_args, name), result in zip(entries, raw_results):
            if isinstance(result, BaseException):
                results.append(
                    {
                        "run_name": name,
                        "experiment_args": _args,
                        "error": repr(result),
                        "summary": {},
                    }
                )
            else:
                results.append(result)
        _write_sweep_csv(results, csv_path)
        with open(csv_path.with_suffix(".json"), "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"wrote {csv_path}")
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    safe_name = _safe_run_name(run_name)
    args = shlex.split(experiment_args)
    output_dir = VOLUME_MOUNT / "mnist_generator" / safe_name
    args = _with_default_arg(args, "--output-dir", output_dir)
    request = {
        "run_name": safe_name,
        "gpu": GPU,
        "max_containers": MAX_CONTAINERS,
        "jax_package": JAX_PACKAGE,
        "timeout_seconds": TIMEOUT_SECONDS,
        "volume": VOLUME_NAME,
        "experiment_args": args,
    }
    if print_only:
        print(json.dumps(request, indent=2, sort_keys=True))
        return
    print(json.dumps(request, indent=2, sort_keys=True))
    result = run_mnist_generator_remote.remote(args, safe_name)
    print(json.dumps(result, indent=2, sort_keys=True))
