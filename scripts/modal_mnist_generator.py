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
    raise ValueError("unknown sweep preset")


def _write_sweep_csv(results: list[dict[str, Any]], path: Path) -> None:
    metric_names = [
        "final_eval_loss",
        "best_loss",
        "best_epoch",
        "final_epoch",
        "generator.diversity_ratio",
        "generator.pixel_mean_mse",
        "generator.pixel_std_mse",
        "generator.nearest_real_mse",
        "generator.real_nearest_real_mse",
        "generator.prototype_mse",
        "generator.prototype_nearest_accuracy",
        "generator.dynamics_family",
        "generator.horn_frequency",
        "generator.horn_damping",
        "generator.horn_nonlinearity",
        "generator.horn_state_bound",
        "generator.conditional",
        "generator.label_phase_scale",
        "generator.conditioning_mode",
        "generator.coupling_profile",
        "generator.coupling_length_scale",
        "generator.coupling_floor",
        "generator.coupling_bias_strength",
        "generator.train_recurrent_dynamics",
        "generator.train_conditioning_dynamics",
        "generator.readout_mode",
        "generator.decoder_mode",
        "generator.loss",
        "generator.pixel_drift_weight",
        "generator.feature_drift_weight",
        "generator.feature_drift_mode",
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
        "generator.quality_classifier_epochs",
        "generator.quality_classifier_dim",
        "generator.quality_classifier_depth",
        "generator.classifier_label_accuracy",
        "generator.classifier_label_confidence",
        "generator.classifier_max_confidence",
        "generator.classifier_entropy",
        "generator.drift_queue_size",
        "generator.drift_queue_num_pos",
        "generator.distributional_weight",
        "generator.drift_gamma",
        "generator.resize_conv_seed_size",
        "generator.resize_conv_upsamples",
        "generator.resize_conv_min_channels",
        "generator.success_diagnostics.total_params",
        "generator.success_diagnostics.dynamics_family",
        "generator.success_diagnostics.decoder_mode",
        "generator.success_diagnostics.decoder_param_fraction",
        "generator.success_diagnostics.trainable_recurrent_param_fraction",
        "generator.success_diagnostics.train_recurrent_dynamics",
        "generator.success_diagnostics.train_conditioning_dynamics",
        "generator.success_diagnostics.trainable_main_recurrent_params",
        "generator.success_diagnostics.trainable_conditioning_params",
        "generator.success_diagnostics.estimated_recurrent_op_fraction",
        "generator.success_diagnostics.coupling_profile",
        "generator.success_diagnostics.coupling_profile_mean",
        "generator.success_diagnostics.coupling_profile_std",
        "generator.success_diagnostics.coupling_profile_min",
        "generator.success_diagnostics.coupling_profile_max",
        "generator.success_diagnostics.samples_per_train_second",
        "generator.success_diagnostics.phase_mean_abs_displacement",
        "generator.success_diagnostics.phase_final_order",
        "generator.success_diagnostics.phase_order_delta",
        "generator.success_diagnostics.state_final_energy",
        "generator.success_diagnostics.state_mean_abs_velocity_displacement",
        "generator.generated_mean",
        "generator.generated_std",
        "final_train_pixel_drift_loss",
        "final_eval_pixel_drift_loss",
        "final_train_feature_drift_loss",
        "final_eval_feature_drift_loss",
        "final_train_drift_queue_ready",
    ]
    fieldnames = ["run", "root", *metric_names]
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
        build_arg_parser,
        config_from_args,
        run_mnist_generator_experiment,
    )

    args = _with_default_arg(list(experiment_args), "--output-dir", output_dir)
    parser = build_arg_parser()
    parsed = parser.parse_args(args)
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
    volume.commit()
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
        results = list(
            run_mnist_generator_remote.starmap(
                entries,
                order_outputs=False,
                return_exceptions=False,
            )
        )
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
