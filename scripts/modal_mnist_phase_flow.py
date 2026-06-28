"""Run MNIST phase-flow experiments on Modal GPUs."""

from __future__ import annotations

import base64
import csv
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "oscnet-mnist-phase-flow"
VOLUME_NAME = os.environ.get("OSCNET_MODAL_VOLUME", "oscnet-runs")
VOLUME_MOUNT = Path("/mnt/oscnet-runs")
GPU = os.environ.get("OSCNET_MODAL_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("OSCNET_MODAL_TIMEOUT_SECONDS", "10800"))
JAX_PACKAGE = os.environ.get("OSCNET_MODAL_JAX", "jax[cuda13]")
MAX_CONTAINERS = int(os.environ.get("OSCNET_MODAL_MAX_CONTAINERS", "3"))

DEFAULT_SMOKE_ARGS = (
    "--data-source synthetic --model-family phase_flow --seed 0 --epochs 1 "
    "--train-limit 8 --eval-limit 4 --eval-sample-count 4 --batch-size 4 "
    "--field-channels 2 --steps 1 --sample-steps 2 "
    "--checkpoint-every 1 --artifact-every 1"
)

SWEEP_CSVS = {
    "mnist_phase_flow_core": Path(
        "outputs/analysis/modal_mnist_phase_flow_core.csv"
    ),
    "mnist_phase_flow_conv_control_core": Path(
        "outputs/analysis/modal_mnist_phase_flow_conv_control_core.csv"
    ),
    "mnist_phase_flow_recurrent_conv_control": Path(
        "outputs/analysis/modal_mnist_phase_flow_recurrent_conv_control.csv"
    ),
    "mnist_phase_flow_coarse_global_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_coarse_global_probe.csv"
    ),
    "mnist_phase_flow_coarse_heun_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_coarse_heun_probe.csv"
    ),
    "mnist_phase_flow_coarse_position_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_coarse_position_probe.csv"
    ),
    "mnist_phase_flow_coarse_closure_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_coarse_closure_probe.csv"
    ),
    "mnist_phase_flow_edge_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_edge_probe.csv"
    ),
    "mnist_phase_flow_signed_distance_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_signed_distance_probe.csv"
    ),
    "mnist_phase_flow_signed_distance_flow_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_signed_distance_flow_probe.csv"
    ),
    "mnist_phase_flow_pixel_shape_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_pixel_shape_probe.csv"
    ),
    "mnist_phase_flow_centered_pixel_shape_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_centered_pixel_shape_probe.csv"
    ),
    "mnist_phase_flow_centered_shape_gated_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_centered_shape_gated_probe.csv"
    ),
    "mnist_phase_flow_shape_guided_sampler_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_shape_guided_sampler_probe.csv"
    ),
    "mnist_phase_flow_shape_gated_audit": Path(
        "outputs/analysis/modal_mnist_phase_flow_shape_gated_audit.csv"
    ),
    "mnist_phase_flow_basin_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_basin_probe.csv"
    ),
    "mnist_phase_flow_signed_distance_flow_basin_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_signed_distance_flow_basin_probe.csv"
    ),
    "mnist_phase_flow_signed_distance_noise_basin_probe": Path(
        "outputs/analysis/modal_mnist_phase_flow_signed_distance_noise_basin_probe.csv"
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
    name = name.strip() or time.strftime("mnist-phase-flow-%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip(".-") or "mnist-phase-flow-run"


def _has_arg(args: list[str], flag: str) -> bool:
    return flag in args or any(arg.startswith(f"{flag}=") for arg in args)


def _with_default_arg(args: list[str], flag: str, value: str | Path) -> list[str]:
    if _has_arg(args, flag):
        return args
    return [*args, flag, str(value)]


def _summary_metric(summary: dict[str, Any], dotted_key: str) -> Any:
    value: Any = summary
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _mnist_phase_flow_core_sweep(
    *,
    include_conv_control: bool = False,
) -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        ("phase_flow", ["--model-family phase_flow"]),
        ("frozen_phase_flow", ["--model-family frozen_phase_flow"]),
        ("no_dynamics", ["--model-family phase_flow_no_dynamics"]),
    ]
    if include_conv_control:
        variants.append(("recurrent_conv_flow", ["--model-family recurrent_conv_flow"]))
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--no-position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--sample-steps 16",
        "--sample-method euler",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
    ]
    for seed in (31,):
        for suffix, model_args in variants:
            run_name = (
                "mnist_phase_flow_"
                f"{suffix}_c8_steps8_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_phase_flow_recurrent_conv_control_sweep() -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--no-position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--sample-steps 16",
        "--sample-method euler",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
        "--model-family recurrent_conv_flow",
    ]
    entries = []
    for seed in (31,):
        run_name = f"mnist_phase_flow_recurrent_conv_flow_c8_steps8_seed{seed}_20e"
        args = shlex.split(" ".join([f"--seed {seed}", *common]))
        output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_phase_flow_coarse_global_probe_sweep() -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--no-position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--sample-steps 16",
        "--sample-method euler",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
        "--model-family coarse_phase_flow",
    ]
    entries = []
    for seed in (31,):
        run_name = f"mnist_phase_flow_coarse4_phase_flow_c8_steps8_seed{seed}_20e"
        args = shlex.split(" ".join([f"--seed {seed}", *common]))
        output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_phase_flow_coarse_heun_probe_sweep() -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--no-position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--sample-steps 32",
        "--sample-method heun",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
        "--model-family coarse_phase_flow",
    ]
    entries = []
    for seed in (31,):
        run_name = (
            "mnist_phase_flow_coarse4_phase_flow_"
            f"heun32_c8_steps8_seed{seed}_20e"
        )
        args = shlex.split(" ".join([f"--seed {seed}", *common]))
        output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_phase_flow_coarse_position_probe_sweep() -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--sample-steps 16",
        "--sample-method euler",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
        "--model-family coarse_phase_flow",
    ]
    entries = []
    for seed in (31,):
        run_name = (
            "mnist_phase_flow_coarse4_position_phase_flow_"
            f"c8_steps8_seed{seed}_20e"
        )
        args = shlex.split(" ".join([f"--seed {seed}", *common]))
        output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _mnist_phase_flow_coarse_closure_probe_sweep() -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--closure-loss-weight 1.0",
        "--sample-steps 16",
        "--sample-method euler",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
        "--model-family coarse_phase_flow",
    ]
    variants = [
        ("closure", ["--no-position-features"]),
        ("position_closure", ["--position-features"]),
    ]
    entries = []
    for seed in (31,):
        for suffix, variant_args in variants:
            run_name = (
                "mnist_phase_flow_coarse4_"
                f"{suffix}_c8_steps8_seed{seed}_20e"
            )
            args = shlex.split(" ".join([f"--seed {seed}", *common, *variant_args]))
            output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_phase_flow_target_probe_sweep(
    target_representation: str,
    *,
    sample_readout_mode: str = "primary",
    sample_schedule: str = "standard",
) -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--no-position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--closure-loss-weight 0.0",
        f"--target-representation {target_representation}",
        f"--sample-schedule {sample_schedule}",
        f"--sample-readout-mode {sample_readout_mode}",
        "--sample-steps 16",
        "--sample-method euler",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
    ]
    variants = [
        ("coarse_phase_flow", ["--model-family coarse_phase_flow"]),
        ("recurrent_conv_flow", ["--model-family recurrent_conv_flow"]),
    ]
    entries = []
    for seed in (31,):
        for suffix, variant_args in variants:
            run_name = (
                f"mnist_phase_flow_{target_representation}_{sample_schedule}_"
                f"{sample_readout_mode}_"
                f"{suffix}_c8_steps8_seed{seed}_20e"
            )
            args = shlex.split(" ".join([f"--seed {seed}", *common, *variant_args]))
            output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_phase_flow_edge_probe_sweep() -> list[tuple[list[str], str]]:
    return _mnist_phase_flow_target_probe_sweep("sobel_edges")


def _mnist_phase_flow_signed_distance_probe_sweep() -> list[tuple[list[str], str]]:
    return _mnist_phase_flow_target_probe_sweep("signed_distance")


def _mnist_phase_flow_signed_distance_flow_probe_sweep() -> list[tuple[list[str], str]]:
    return _mnist_phase_flow_target_probe_sweep("signed_distance_flow")


def _mnist_phase_flow_pixel_shape_probe_sweep() -> list[tuple[list[str], str]]:
    return _mnist_phase_flow_target_probe_sweep("pixels_signed_distance")


def _mnist_phase_flow_centered_pixel_shape_probe_sweep() -> list[tuple[list[str], str]]:
    return _mnist_phase_flow_target_probe_sweep("centered_pixels_signed_distance")


def _mnist_phase_flow_centered_shape_gated_probe_sweep() -> list[tuple[list[str], str]]:
    return _mnist_phase_flow_target_probe_sweep(
        "centered_pixels_signed_distance",
        sample_readout_mode="shape_gated",
    )


def _mnist_phase_flow_shape_guided_sampler_probe_sweep() -> list[tuple[list[str], str]]:
    return _mnist_phase_flow_target_probe_sweep(
        "centered_pixels_signed_distance",
        sample_readout_mode="shape_gated",
        sample_schedule="shape_guided",
    )


def _mnist_phase_flow_shape_gated_audit_sweep() -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--no-position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--closure-loss-weight 0.0",
        "--target-representation centered_pixels_signed_distance",
        "--sample-schedule standard",
        "--sample-readout-mode shape_gated",
        "--sample-steps 16",
        "--sample-method euler",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
    ]
    variants = [
        ("coarse_phase_flow", ["--model-family coarse_phase_flow"]),
        ("phase_flow", ["--model-family phase_flow"]),
        ("frozen_phase_flow", ["--model-family frozen_phase_flow"]),
        ("no_dynamics", ["--model-family phase_flow_no_dynamics"]),
        ("recurrent_conv_flow", ["--model-family recurrent_conv_flow"]),
    ]
    entries = []
    for seed in (31, 32, 33, 34, 35):
        for suffix, variant_args in variants:
            run_name = (
                "mnist_phase_flow_shape_gated_audit_"
                f"{suffix}_c8_steps8_seed{seed}_20e"
            )
            args = shlex.split(" ".join([f"--seed {seed}", *common, *variant_args]))
            output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_phase_flow_basin_probe_sweep() -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--no-position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--closure-loss-weight 0.0",
        "--sample-schedule standard",
        "--sample-steps 16",
        "--sample-method euler",
        "--basin-t-values 0.1,0.25,0.5,0.75,0.9",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
    ]
    variants = [
        (
            "centered_pixel_shape_coarse_phase_flow",
            [
                "--target-representation centered_pixels_signed_distance",
                "--sample-readout-mode shape_gated",
                "--model-family coarse_phase_flow",
            ],
        ),
        (
            "centered_pixel_shape_recurrent_conv_flow",
            [
                "--target-representation centered_pixels_signed_distance",
                "--sample-readout-mode shape_gated",
                "--model-family recurrent_conv_flow",
            ],
        ),
        (
            "signed_distance_coarse_phase_flow",
            [
                "--target-representation signed_distance",
                "--sample-readout-mode primary",
                "--model-family coarse_phase_flow",
            ],
        ),
        (
            "signed_distance_recurrent_conv_flow",
            [
                "--target-representation signed_distance",
                "--sample-readout-mode primary",
                "--model-family recurrent_conv_flow",
            ],
        ),
    ]
    entries = []
    for seed in (31, 32):
        for suffix, variant_args in variants:
            run_name = f"mnist_phase_flow_basin_{suffix}_seed{seed}_20e"
            args = shlex.split(" ".join([f"--seed {seed}", *common, *variant_args]))
            output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_phase_flow_signed_distance_flow_basin_probe_sweep() -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--no-position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--closure-loss-weight 0.0",
        "--target-representation signed_distance_flow",
        "--sample-readout-mode primary",
        "--sample-schedule standard",
        "--sample-steps 16",
        "--sample-method euler",
        "--basin-t-values 0.1,0.25,0.5,0.75,0.9",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
    ]
    variants = [
        ("coarse_phase_flow", ["--model-family coarse_phase_flow"]),
        ("recurrent_conv_flow", ["--model-family recurrent_conv_flow"]),
    ]
    entries = []
    for seed in (31, 32):
        for suffix, variant_args in variants:
            run_name = (
                "mnist_phase_flow_basin_signed_distance_flow_"
                f"{suffix}_seed{seed}_20e"
            )
            args = shlex.split(" ".join([f"--seed {seed}", *common, *variant_args]))
            output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_phase_flow_signed_distance_noise_basin_probe_sweep() -> list[tuple[list[str], str]]:
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--field-channels 8",
        "--steps 8",
        "--kernel-size 3",
        "--dt 0.15",
        "--coupling-strength 1.0",
        "--rate-update 0.5",
        "--input-drive-strength 0.5",
        "--global-coupling-strength 0.5",
        "--coarse-grid-size 4",
        "--omega-scale 0.2",
        "--kernel-init-scale 0.05",
        "--no-position-features",
        "--conditional",
        "--clean-loss-weight 0.25",
        "--closure-loss-weight 0.0",
        "--target-representation signed_distance",
        "--sample-readout-mode primary",
        "--sample-schedule standard",
        "--sample-steps 16",
        "--sample-method euler",
        "--basin-t-values 0.1,0.25,0.5,0.75,0.9",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
    ]
    variants = [
        ("coarse_phase_flow", ["--model-family coarse_phase_flow"]),
        ("recurrent_conv_flow", ["--model-family recurrent_conv_flow"]),
    ]
    entries = []
    for seed in (31, 32):
        for noise_mode in ("uniform", "salt_pepper", "zeros"):
            for suffix, variant_args in variants:
                run_name = (
                    "mnist_phase_flow_basin_signed_distance_"
                    f"{noise_mode}_{suffix}_seed{seed}_20e"
                )
                args = shlex.split(
                    " ".join(
                        [
                            f"--seed {seed}",
                            *common,
                            f"--basin-noise-mode {noise_mode}",
                            *variant_args,
                        ]
                    )
                )
                output_dir = VOLUME_MOUNT / "mnist_phase_flow" / run_name
                args = _with_default_arg(args, "--output-dir", output_dir)
                entries.append((args, run_name))
    return entries


def _sweep_entries(preset: str) -> list[tuple[list[str], str]]:
    if preset == "mnist_phase_flow_core":
        return _mnist_phase_flow_core_sweep()
    if preset == "mnist_phase_flow_conv_control_core":
        return _mnist_phase_flow_core_sweep(include_conv_control=True)
    if preset == "mnist_phase_flow_recurrent_conv_control":
        return _mnist_phase_flow_recurrent_conv_control_sweep()
    if preset == "mnist_phase_flow_coarse_global_probe":
        return _mnist_phase_flow_coarse_global_probe_sweep()
    if preset == "mnist_phase_flow_coarse_heun_probe":
        return _mnist_phase_flow_coarse_heun_probe_sweep()
    if preset == "mnist_phase_flow_coarse_position_probe":
        return _mnist_phase_flow_coarse_position_probe_sweep()
    if preset == "mnist_phase_flow_coarse_closure_probe":
        return _mnist_phase_flow_coarse_closure_probe_sweep()
    if preset == "mnist_phase_flow_edge_probe":
        return _mnist_phase_flow_edge_probe_sweep()
    if preset == "mnist_phase_flow_signed_distance_probe":
        return _mnist_phase_flow_signed_distance_probe_sweep()
    if preset == "mnist_phase_flow_signed_distance_flow_probe":
        return _mnist_phase_flow_signed_distance_flow_probe_sweep()
    if preset == "mnist_phase_flow_pixel_shape_probe":
        return _mnist_phase_flow_pixel_shape_probe_sweep()
    if preset == "mnist_phase_flow_centered_pixel_shape_probe":
        return _mnist_phase_flow_centered_pixel_shape_probe_sweep()
    if preset == "mnist_phase_flow_centered_shape_gated_probe":
        return _mnist_phase_flow_centered_shape_gated_probe_sweep()
    if preset == "mnist_phase_flow_shape_guided_sampler_probe":
        return _mnist_phase_flow_shape_guided_sampler_probe_sweep()
    if preset == "mnist_phase_flow_shape_gated_audit":
        return _mnist_phase_flow_shape_gated_audit_sweep()
    if preset == "mnist_phase_flow_basin_probe":
        return _mnist_phase_flow_basin_probe_sweep()
    if preset == "mnist_phase_flow_signed_distance_flow_basin_probe":
        return _mnist_phase_flow_signed_distance_flow_basin_probe_sweep()
    if preset == "mnist_phase_flow_signed_distance_noise_basin_probe":
        return _mnist_phase_flow_signed_distance_noise_basin_probe_sweep()
    raise ValueError("unknown sweep preset")


def _write_sweep_csv(results: list[dict[str, Any]], path: Path) -> None:
    metric_names = [
        "final_eval_loss",
        "best_loss",
        "best_epoch",
        "final_epoch",
        "final_eval_velocity_loss",
        "final_eval_clean_loss",
        "final_eval_closure_loss",
        "phase_flow.model_family",
        "phase_flow.field_channels",
        "phase_flow.steps",
        "phase_flow.train_dynamics",
        "phase_flow.conditional",
        "phase_flow.position_features",
        "phase_flow.clean_loss_weight",
        "phase_flow.closure_loss_weight",
        "phase_flow.target_representation",
        "phase_flow.target_channels",
        "phase_flow.sample_steps",
        "phase_flow.sample_method",
        "phase_flow.sample_schedule",
        "phase_flow.sample_readout_mode",
        "phase_flow.basin_t_values",
        "phase_flow.basin_noise_mode",
        "phase_flow.sample_mean",
        "phase_flow.sample_std",
        "phase_flow.sample_pixel_mean_mse",
        "phase_flow.sample_pixel_std_mse",
        "phase_flow.sample_diversity_ratio",
        "phase_flow.sample_nearest_real_mse",
        "phase_flow.real_nearest_real_mse",
        "phase_flow.sample_active_fraction",
        "phase_flow.sample_component_count",
        "phase_flow.sample_largest_component_fraction",
        "phase_flow.real_active_fraction",
        "phase_flow.real_component_count",
        "phase_flow.real_largest_component_fraction",
        "phase_flow.coarse_grid_size",
        "phase_flow.global_coupling_strength",
        "phase_flow.phase_mean_abs_displacement",
        "phase_flow.state_mean_abs_displacement",
        "train_seconds",
    ]
    for basin_key in ("t0_100", "t0_250", "t0_500", "t0_750", "t0_900"):
        for metric in (
            "initial_paired_mse",
            "paired_mse",
            "paired_mse_delta",
            "paired_mse_improvement_fraction",
            "sample_nearest_real_mse",
            "sample_active_fraction",
            "sample_component_count",
            "sample_largest_component_fraction",
        ):
            metric_names.append(f"phase_flow.basin.{basin_key}.{metric}")
    fieldnames = ["run", "root", *metric_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
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
def run_mnist_phase_flow_remote(
    experiment_args: list[str],
    run_name: str,
) -> dict[str, Any]:
    """Run one MNIST phase-flow experiment remotely."""

    import base64
    import os

    safe_name = _safe_run_name(run_name)
    output_dir = VOLUME_MOUNT / "mnist_phase_flow" / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (VOLUME_MOUNT / "jax_cache").mkdir(parents=True, exist_ok=True)

    cache_target = VOLUME_MOUNT / "cache" / "oscnet"
    cache_target.mkdir(parents=True, exist_ok=True)
    local_cache = Path.home() / ".cache" / "oscnet"
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    if not local_cache.exists():
        os.symlink(cache_target, local_cache, target_is_directory=True)

    import jax

    from oscnet.experiments.harness import AutoencoderExperimentConfig
    from oscnet.experiments.mnist_phase_flow import (
        MNISTPhaseFlowExperimentConfig,
        build_arg_parser,
        run_mnist_phase_flow_experiment,
    )

    parser = build_arg_parser()
    args = parser.parse_args(experiment_args)
    run_config = AutoencoderExperimentConfig(
        name=safe_name,
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
    config = MNISTPhaseFlowExperimentConfig(
        run=run_config,
        model_family=args.model_family,
        field_channels=args.field_channels,
        steps=args.steps,
        kernel_size=args.kernel_size,
        dt=args.dt,
        coupling_strength=args.coupling_strength,
        rate_update=args.rate_update,
        input_drive_strength=args.input_drive_strength,
        global_coupling_strength=args.global_coupling_strength,
        coarse_grid_size=args.coarse_grid_size,
        omega_scale=args.omega_scale,
        kernel_init_scale=args.kernel_init_scale,
        position_features=args.position_features,
        conditional=args.conditional,
        num_classes=args.num_classes,
        clean_loss_weight=args.clean_loss_weight,
        closure_loss_weight=args.closure_loss_weight,
        t_min=args.t_min,
        t_max=args.t_max,
        eval_sample_count=args.eval_sample_count,
        sample_steps=args.sample_steps,
        sample_method=args.sample_method,
        sample_schedule=args.sample_schedule,
        sample_readout_mode=args.sample_readout_mode,
        basin_t_values=args.basin_t_values,
        basin_noise_mode=args.basin_noise_mode,
        target_representation=args.target_representation,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )
    result = run_mnist_phase_flow_experiment(config)
    summary_path = result.paths.metrics / "summary.json"
    history_path = result.paths.metrics / "history.json"
    with summary_path.open() as f:
        summary = json.load(f)
    with history_path.open() as f:
        history = json.load(f)
    epochs = history.get("epoch", [])
    tail_indices = range(max(0, len(epochs) - 3), len(epochs))
    history_tail = {
        key: [values[index] for index in tail_indices]
        for key, values in history.items()
        if isinstance(values, list)
    }

    artifact_epoch = int(summary["final_epoch"])
    sample_png = result.paths.artifacts / f"samples_epoch_{artifact_epoch:03d}.png"
    denoised_png = result.paths.artifacts / f"denoised_epoch_{artifact_epoch:03d}.png"
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
        "samples_png_b64": (
            base64.b64encode(sample_png.read_bytes()).decode("ascii")
            if sample_png.exists()
            else None
        ),
        "denoised_png_b64": (
            base64.b64encode(denoised_png.read_bytes()).decode("ascii")
            if denoised_png.exists()
            else None
        ),
    }
    with (result.paths.root / "modal_result.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    volume.commit()
    return payload


def _write_pngs(results: list[dict[str, Any]], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for result in results:
        run = result["run_name"]
        if result.get("samples_png_b64"):
            (root / f"{run}_samples.png").write_bytes(
                base64.b64decode(result["samples_png_b64"])
            )
        if result.get("denoised_png_b64"):
            (root / f"{run}_denoised.png").write_bytes(
                base64.b64decode(result["denoised_png_b64"])
            )


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"samples_png_b64", "denoised_png_b64"}
    }


def _compact_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_result(result) for result in results]


@app.local_entrypoint()
def main(
    experiment_args: str = DEFAULT_SMOKE_ARGS,
    run_name: str = "",
    sweep_preset: str = "",
    sweep_csv: str = "",
    print_only: bool = False,
) -> None:
    """Launch a remote MNIST phase-flow experiment."""

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
        results = list(
            run_mnist_phase_flow_remote.starmap(
                entries,
                order_outputs=False,
                return_exceptions=False,
            )
        )
        _write_sweep_csv(results, csv_path)
        compact_results = _compact_results(results)
        with csv_path.with_suffix(".json").open("w") as f:
            json.dump(compact_results, f, indent=2, sort_keys=True)
        _write_pngs(results, Path("outputs/analysis/modal_mnist_phase_flow_samples"))
        print(f"wrote {csv_path}")
        print(json.dumps(compact_results, indent=2, sort_keys=True))
        return

    safe_name = _safe_run_name(run_name)
    args = shlex.split(experiment_args)
    output_dir = VOLUME_MOUNT / "mnist_phase_flow" / safe_name
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
    result = run_mnist_phase_flow_remote.remote(args, safe_name)
    _write_pngs([result], Path("outputs/analysis/modal_mnist_phase_flow_samples"))
    print(json.dumps(_compact_result(result), indent=2, sort_keys=True))
