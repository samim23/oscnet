"""Run OscNet MNIST experiments on Modal GPU workers.

This is intentionally an outer launch adapter. The OscNet model and experiment
code stay Modal-free; this script only builds a remote environment, mounts a
persistent output volume, and calls the existing MNIST experiment API.

Examples:

    modal run scripts/modal_mnist.py

    modal run scripts/modal_mnist.py --run-name block50_global_seed11 \
      --experiment-args "--model-family winfree_global_rate_phase --seed 11 \
      --corruption-mode block_occlusion --corruption-fraction 0.5 \
      --corruption-seed 11 --epochs 10 --train-limit 2000 --eval-limit 500"

Environment overrides:

    OSCNET_MODAL_GPU=A100
    OSCNET_MODAL_JAX=jax[cuda12]
    OSCNET_MODAL_VOLUME=oscnet-runs
    OSCNET_MODAL_TIMEOUT_SECONDS=10800
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
import csv
from pathlib import Path
from typing import Any

import modal

APP_NAME = "oscnet-mnist"
VOLUME_NAME = os.environ.get("OSCNET_MODAL_VOLUME", "oscnet-runs")
VOLUME_MOUNT = Path("/mnt/oscnet-runs")
GPU = os.environ.get("OSCNET_MODAL_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("OSCNET_MODAL_TIMEOUT_SECONDS", "10800"))
JAX_PACKAGE = os.environ.get("OSCNET_MODAL_JAX", "jax[cuda13]")
MAX_CONTAINERS = int(os.environ.get("OSCNET_MODAL_MAX_CONTAINERS", "3"))

DEFAULT_SMOKE_ARGS = (
    "--data-source synthetic --model-family winfree_rate_phase "
    "--corruption-mode block_occlusion --corruption-fraction 0.5 "
    "--seed 0 --corruption-seed 0 --epochs 1 --train-limit 8 --eval-limit 4 "
    "--batch-size 4 --hidden-dim 4 --latent-dim 4 --patch-size 7 "
    "--winfree-steps 1 --artifact-every 1 --checkpoint-every 1"
)

DEFAULT_SWEEP_CSV = Path("outputs/analysis/modal_block50_coarse_phase_mesh.csv")
DEFAULT_ROBUSTNESS_CSV = Path("outputs/analysis/modal_checkpoint_robustness.csv")
DEFAULT_SETTLING_CSV = Path("outputs/analysis/modal_anytime_settling.csv")
SWEEP_CSVS = {
    "block50_coarse_phase_mesh": DEFAULT_SWEEP_CSV,
    "block50_coarse_rate_phase": Path(
        "outputs/analysis/modal_block50_coarse_rate_phase.csv"
    ),
    "block50_mask_aware_core": Path(
        "outputs/analysis/modal_block50_mask_aware_core.csv"
    ),
    "block50_mask_weight_sweep": Path(
        "outputs/analysis/modal_block50_mask_weight_sweep.csv"
    ),
    "block50_missing_marker_core": Path(
        "outputs/analysis/modal_block50_missing_marker_core.csv"
    ),
    "block50_image_plus_mask_core": Path(
        "outputs/analysis/modal_block50_image_plus_mask_core.csv"
    ),
    "block50_visibility_gated_winfree": Path(
        "outputs/analysis/modal_block50_visibility_gated_winfree.csv"
    ),
    "block50_conv_lstm_control": Path(
        "outputs/analysis/modal_block50_conv_lstm_control.csv"
    ),
    "block50_coarse_predictive_readout": Path(
        "outputs/analysis/modal_block50_coarse_predictive_readout.csv"
    ),
    "block50_boundary_clamped_core": Path(
        "outputs/analysis/modal_block50_boundary_clamped_core.csv"
    ),
    "block50_prior_refinement": Path(
        "outputs/analysis/modal_block50_prior_refinement.csv"
    ),
    "block50_recurrent_prior_refinement": Path(
        "outputs/analysis/modal_block50_recurrent_prior_refinement.csv"
    ),
}
ROBUSTNESS_METRICS = (
    "eval_loss",
    "final_eval_loss",
    "best_loss",
    "quality.changed_mse",
    "quality.changed_mae",
    "quality.unchanged_mse",
    "quality.pixel_correlation",
    "quality.foreground_f1",
    "quality.diversity_ratio",
    "quality.changed_improvement",
)
SETTLING_METRICS = ROBUSTNESS_METRICS

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
    name = name.strip() or time.strftime("mnist-%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip(".-") or "mnist-run"


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


def _get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _parse_seed_from_run_name(run_name: str) -> int:
    match = re.search(r"seed(\d+)", run_name)
    if match is None:
        return 0
    return int(match.group(1))


def _variant_from_run_name(run_name: str) -> str:
    if "recurrent_prior_refine_s050" in run_name:
        return "recurrent_prior_refine_s050"
    if "prior_refine_s050" in run_name:
        return "prior_refine_s050"
    if "prior_refine_s025" in run_name:
        return "prior_refine_s025"
    if "feedforward" in run_name:
        return "feedforward"
    if "winfree" in run_name:
        return "winfree"
    return "unknown"


def _parse_int_list(values: str) -> list[int]:
    parsed = [int(value.strip()) for value in values.split(",") if value.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    if any(value < 1 for value in parsed):
        raise ValueError("all integer values must be >= 1")
    return parsed


def _parse_name_list(values: str) -> list[str]:
    parsed = [value.strip() for value in values.split(",") if value.strip()]
    if not parsed:
        raise ValueError("expected at least one name")
    return parsed


def _block50_coarse_phase_mesh_sweep() -> list[tuple[list[str], str]]:
    entries = []
    for seed in (11, 12):
        for coarse_size in (2, 4):
            for phase_control in ("none", "shuffle"):
                control_suffix = "" if phase_control == "none" else "_shuffle"
                run_name = (
                    "mnist_block50_winfree_coarse_global_rate_phase_"
                    f"coarse{coarse_size}{control_suffix}_"
                    f"h64_steps8_train2000_seed{seed}_10e"
                )
                args = shlex.split(
                    " ".join(
                        [
                            "--model-family winfree_coarse_global_rate_phase",
                            f"--seed {seed}",
                            f"--corruption-seed {seed}",
                            "--corruption-mode block_occlusion",
                            "--corruption-fraction 0.5",
                            "--epochs 10",
                            "--train-limit 2000",
                            "--eval-limit 500",
                            "--batch-size 64",
                            "--hidden-dim 64",
                            "--latent-dim 64",
                            "--patch-size 4",
                            "--winfree-steps 8",
                            "--winfree-coupling-mode conv",
                            "--winfree-coupling-kernel-size 3",
                            "--winfree-si-func mlp",
                            "--winfree-global-gamma 0.05",
                            "--winfree-global-gate-strength 0.5",
                            f"--winfree-coarse-grid-size {coarse_size}",
                            f"--winfree-global-phase-control {phase_control}",
                            "--winfree-output-activation sigmoid",
                            "--artifact-every 10",
                            "--checkpoint-every 10",
                        ]
                    )
                )
                output_dir = VOLUME_MOUNT / "mnist" / run_name
                args = _with_default_arg(args, "--output-dir", output_dir)
                entries.append((args, run_name))
    return entries


def _block50_coarse_rate_phase_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "content05_gate05",
            "--winfree-global-content-strength 0.5 "
            "--winfree-global-content-control none "
            "--winfree-global-gate-strength 0.5 "
            "--winfree-global-phase-control none",
        ),
        (
            "content05_gate05_contentshuffle",
            "--winfree-global-content-strength 0.5 "
            "--winfree-global-content-control shuffle "
            "--winfree-global-gate-strength 0.5 "
            "--winfree-global-phase-control none",
        ),
        (
            "content05_gate00",
            "--winfree-global-content-strength 0.5 "
            "--winfree-global-content-control none "
            "--winfree-global-gate-strength 0.0 "
            "--winfree-global-phase-control none",
        ),
        (
            "content00_gate05",
            "--winfree-global-content-strength 0.0 "
            "--winfree-global-content-control none "
            "--winfree-global-gate-strength 0.5 "
            "--winfree-global-phase-control none",
        ),
    ]
    for seed in (11, 12):
        for suffix, controls in variants:
            run_name = (
                "mnist_block50_winfree_coarse_rate_phase_"
                f"coarse4_{suffix}_h64_steps8_train2000_seed{seed}_10e"
            )
            args = shlex.split(
                " ".join(
                    [
                        "--model-family winfree_coarse_rate_phase",
                        f"--seed {seed}",
                        f"--corruption-seed {seed}",
                        "--corruption-mode block_occlusion",
                        "--corruption-fraction 0.5",
                        "--epochs 10",
                        "--train-limit 2000",
                        "--eval-limit 500",
                        "--batch-size 64",
                        "--hidden-dim 64",
                        "--latent-dim 64",
                        "--patch-size 4",
                        "--winfree-steps 8",
                        "--winfree-coupling-mode conv",
                        "--winfree-coupling-kernel-size 3",
                        "--winfree-si-func mlp",
                        "--winfree-global-gamma 0.05",
                        "--winfree-coarse-grid-size 4",
                        controls,
                        "--winfree-output-activation sigmoid",
                        "--artifact-every 10",
                        "--checkpoint-every 10",
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _block50_mask_aware_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            [
                "--model-family feedforward_patch",
                "--feedforward-output-activation sigmoid",
            ],
        ),
        (
            "recurrent_conv",
            [
                "--model-family recurrent_conv",
                "--recurrent-conv-steps 8",
                "--recurrent-conv-kernel-size 3",
                "--recurrent-conv-residual-strength 0.5",
                "--recurrent-conv-output-activation sigmoid",
            ],
        ),
        (
            "winfree_rate_phase",
            [
                "--model-family winfree_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-visible-loss-weight 0.25",
        "--corruption-changed-loss-weight 2.0",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
    ]
    for seed in (11, 12):
        for suffix, model_args in variants:
            run_name = (
                "mnist_block50_maskaware_"
                f"{suffix}_h64_l96_train2000_seed{seed}_10e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--corruption-seed {seed}",
                        *common,
                        *model_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _block50_mask_weight_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            [
                "--model-family feedforward_patch",
                "--feedforward-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
    ]
    weight_settings = [
        ("changed15_visible05", 1.5, 0.5),
        ("changed125_visible075", 1.25, 0.75),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
    ]
    for seed in (11, 12):
        for weight_suffix, changed_weight, visible_weight in weight_settings:
            for model_suffix, model_args in variants:
                run_name = (
                    "mnist_block50_maskweight_"
                    f"{weight_suffix}_{model_suffix}_"
                    f"h64_l96_train2000_seed{seed}_10e"
                )
                args = shlex.split(
                    " ".join(
                        [
                            f"--seed {seed}",
                            f"--corruption-seed {seed}",
                            *common,
                            f"--corruption-visible-loss-weight {visible_weight}",
                            f"--corruption-changed-loss-weight {changed_weight}",
                            *model_args,
                        ]
                    )
                )
                output_dir = VOLUME_MOUNT / "mnist" / run_name
                args = _with_default_arg(args, "--output-dir", output_dir)
                entries.append((args, run_name))
    return entries


def _block50_missing_marker_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            [
                "--model-family feedforward_patch",
                "--feedforward-output-activation sigmoid",
            ],
        ),
        (
            "recurrent_conv",
            [
                "--model-family recurrent_conv",
                "--recurrent-conv-steps 8",
                "--recurrent-conv-kernel-size 3",
                "--recurrent-conv-residual-strength 0.5",
                "--recurrent-conv-output-activation sigmoid",
            ],
        ),
        (
            "winfree_rate_phase",
            [
                "--model-family winfree_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-mask-value -1.0",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
    ]
    for seed in (11, 12):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_block50_missing_marker_"
                f"{model_suffix}_h64_l96_train2000_seed{seed}_10e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--corruption-seed {seed}",
                        *common,
                        *model_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _block50_image_plus_mask_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            [
                "--model-family feedforward_patch",
                "--feedforward-output-activation sigmoid",
            ],
        ),
        (
            "recurrent_conv",
            [
                "--model-family recurrent_conv",
                "--recurrent-conv-steps 8",
                "--recurrent-conv-kernel-size 3",
                "--recurrent-conv-residual-strength 0.5",
                "--recurrent-conv-output-activation sigmoid",
            ],
        ),
        (
            "winfree_rate_phase",
            [
                "--model-family winfree_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-mask-value 0.0",
        "--corruption-input-mode image_plus_mask",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
    ]
    for seed in (11, 12):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_block50_image_plus_mask_"
                f"{model_suffix}_h64_l96_train2000_seed{seed}_10e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--corruption-seed {seed}",
                        *common,
                        *model_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _block50_visibility_gated_winfree_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            [
                "--model-family feedforward_patch",
                "--feedforward-output-activation sigmoid",
            ],
        ),
        (
            "winfree_rate_phase_ungated",
            [
                "--model-family winfree_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_rate_phase_gated",
            [
                "--model-family winfree_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-visibility-gate visibility",
                "--winfree-visibility-drive-floor 0.0",
                "--winfree-missing-transport-strength 1.0",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_rate_phase_gated_shuffle",
            [
                "--model-family winfree_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-visibility-gate shuffle",
                "--winfree-visibility-drive-floor 0.0",
                "--winfree-missing-transport-strength 1.0",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase_ungated",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase_gated",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-visibility-gate visibility",
                "--winfree-visibility-drive-floor 0.0",
                "--winfree-missing-transport-strength 1.0",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase_gated_shuffle",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-visibility-gate shuffle",
                "--winfree-visibility-drive-floor 0.0",
                "--winfree-missing-transport-strength 1.0",
                "--winfree-output-activation sigmoid",
            ],
        ),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-mask-value 0.0",
        "--corruption-input-mode image_plus_mask",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
    ]
    for seed in (11, 12):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_block50_visibility_"
                f"{model_suffix}_h64_l96_train2000_seed{seed}_10e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--corruption-seed {seed}",
                        *common,
                        *model_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _block50_conv_lstm_control_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            [
                "--model-family feedforward_patch",
                "--feedforward-output-activation sigmoid",
            ],
        ),
        (
            "recurrent_conv",
            [
                "--model-family recurrent_conv",
                "--recurrent-conv-steps 8",
                "--recurrent-conv-kernel-size 3",
                "--recurrent-conv-residual-strength 0.5",
                "--recurrent-conv-output-activation sigmoid",
            ],
        ),
        (
            "conv_lstm",
            [
                "--model-family conv_lstm",
                "--conv-lstm-steps 8",
                "--conv-lstm-kernel-size 3",
                "--conv-lstm-forget-bias 1.0",
                "--conv-lstm-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-mask-value 0.0",
        "--corruption-input-mode image_plus_mask",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
    ]
    for seed in (11, 12):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_block50_convlstm_control_"
                f"{model_suffix}_h64_l96_train2000_seed{seed}_10e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--corruption-seed {seed}",
                        *common,
                        *model_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _block50_coarse_predictive_readout_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            [
                "--model-family feedforward_patch",
                "--feedforward-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_coarse_predictive_c2",
            [
                "--model-family winfree_coarse_predictive_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-coarse-grid-size 2",
                "--winfree-coarse-readout-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_coarse_predictive_c4",
            [
                "--model-family winfree_coarse_predictive_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-coarse-grid-size 4",
                "--winfree-coarse-readout-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-mask-value 0.0",
        "--corruption-input-mode image_plus_mask",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
    ]
    for seed in (11, 12):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_block50_coarse_predictive_"
                f"{model_suffix}_h64_l96_train2000_seed{seed}_10e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--corruption-seed {seed}",
                        *common,
                        *model_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _block50_boundary_clamped_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            [
                "--model-family feedforward_patch",
                "--feedforward-output-activation sigmoid",
            ],
        ),
        (
            "winfree_global_rate_phase",
            [
                "--model-family winfree_global_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "winfree_coarse_predictive_c4",
            [
                "--model-family winfree_coarse_predictive_rate_phase",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-coarse-grid-size 4",
                "--winfree-coarse-readout-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-mask-value 0.0",
        "--corruption-input-mode image_plus_mask",
        "--corruption-protocol boundary_clamped",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
    ]
    for seed in (11, 12):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_block50_boundary_clamped_"
                f"{model_suffix}_h64_l96_train2000_seed{seed}_10e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--corruption-seed {seed}",
                        *common,
                        *model_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _block50_prior_refinement_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            [
                "--model-family feedforward_patch",
                "--feedforward-output-activation sigmoid",
            ],
        ),
        (
            "prior_refine_s025",
            [
                "--model-family winfree_prior_refinement",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-refinement-strength 0.25",
                "--winfree-output-activation sigmoid",
            ],
        ),
        (
            "prior_refine_s050",
            [
                "--model-family winfree_prior_refinement",
                "--winfree-steps 8",
                "--winfree-coupling-mode conv",
                "--winfree-coupling-kernel-size 3",
                "--winfree-si-func mlp",
                "--winfree-global-gamma 0.05",
                "--winfree-global-gate-strength 0.5",
                "--winfree-refinement-strength 0.5",
                "--winfree-output-activation sigmoid",
            ],
        ),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-mask-value 0.0",
        "--corruption-input-mode image_plus_mask",
        "--corruption-protocol boundary_clamped",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
    ]
    for seed in (11, 12):
        for model_suffix, model_args in variants:
            run_name = (
                "mnist_block50_prior_refinement_"
                f"{model_suffix}_h64_l96_train2000_seed{seed}_10e"
            )
            args = shlex.split(
                " ".join(
                    [
                        f"--seed {seed}",
                        f"--corruption-seed {seed}",
                        *common,
                        *model_args,
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _block50_recurrent_prior_refinement_sweep() -> list[tuple[list[str], str]]:
    entries = []
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-mask-value 0.0",
        "--corruption-input-mode image_plus_mask",
        "--corruption-protocol boundary_clamped",
        "--epochs 10",
        "--train-limit 2000",
        "--eval-limit 500",
        "--batch-size 64",
        "--hidden-dim 64",
        "--latent-dim 96",
        "--patch-size 4",
        "--artifact-every 10",
        "--checkpoint-every 10",
        "--model-family recurrent_conv_prior_refinement",
        "--recurrent-conv-steps 8",
        "--recurrent-conv-kernel-size 3",
        "--recurrent-conv-residual-strength 0.5",
        "--recurrent-conv-refinement-strength 0.5",
        "--recurrent-conv-output-activation sigmoid",
    ]
    for seed in (11, 12):
        run_name = (
            "mnist_block50_recurrent_prior_refinement_"
            f"recurrent_prior_refine_s050_h64_l96_train2000_seed{seed}_10e"
        )
        args = shlex.split(
            " ".join(
                [
                    f"--seed {seed}",
                    f"--corruption-seed {seed}",
                    *common,
                ]
            )
        )
        output_dir = VOLUME_MOUNT / "mnist" / run_name
        args = _with_default_arg(args, "--output-dir", output_dir)
        entries.append((args, run_name))
    return entries


def _sweep_entries(preset: str) -> list[tuple[list[str], str]]:
    if preset == "block50_coarse_phase_mesh":
        return _block50_coarse_phase_mesh_sweep()
    if preset == "block50_coarse_rate_phase":
        return _block50_coarse_rate_phase_sweep()
    if preset == "block50_mask_aware_core":
        return _block50_mask_aware_core_sweep()
    if preset == "block50_mask_weight_sweep":
        return _block50_mask_weight_sweep()
    if preset == "block50_missing_marker_core":
        return _block50_missing_marker_core_sweep()
    if preset == "block50_image_plus_mask_core":
        return _block50_image_plus_mask_core_sweep()
    if preset == "block50_visibility_gated_winfree":
        return _block50_visibility_gated_winfree_sweep()
    if preset == "block50_conv_lstm_control":
        return _block50_conv_lstm_control_sweep()
    if preset == "block50_coarse_predictive_readout":
        return _block50_coarse_predictive_readout_sweep()
    if preset == "block50_boundary_clamped_core":
        return _block50_boundary_clamped_core_sweep()
    if preset == "block50_prior_refinement":
        return _block50_prior_refinement_sweep()
    if preset == "block50_recurrent_prior_refinement":
        return _block50_recurrent_prior_refinement_sweep()
    raise ValueError("unknown sweep preset")


def _robustness_scenarios(preset: str) -> list[dict[str, Any]]:
    if preset != "mask_stress":
        raise ValueError("unknown robustness preset")
    return [
        {
            "name": "block25",
            "mode": "block_occlusion",
            "fraction": 0.25,
            "seed_offset": 0,
        },
        {
            "name": "block50",
            "mode": "block_occlusion",
            "fraction": 0.50,
            "seed_offset": 0,
        },
        {
            "name": "block75",
            "mode": "block_occlusion",
            "fraction": 0.75,
            "seed_offset": 0,
        },
        {
            "name": "patch50",
            "mode": "patch_mask",
            "fraction": 0.50,
            "seed_offset": 100,
        },
        {
            "name": "patch75",
            "mode": "patch_mask",
            "fraction": 0.75,
            "seed_offset": 100,
        },
    ]


def _best_checkpoint_path(result: dict[str, Any]) -> str | None:
    checkpoint_paths = result.get("checkpoint_paths") or []
    for checkpoint_path in checkpoint_paths:
        if str(checkpoint_path).endswith("best_model.eqx"):
            return str(checkpoint_path)
    if checkpoint_paths:
        return str(checkpoint_paths[0])
    return None


def _robustness_entries_from_sweep_json(
    source_json: Path,
    *,
    preset: str,
    include_regex: str = "",
) -> tuple[list[tuple[list[str], str]], dict[str, dict[str, Any]]]:
    with open(source_json) as f:
        source_results = json.load(f)
    include = re.compile(include_regex) if include_regex else None
    scenarios = _robustness_scenarios(preset)
    entries: list[tuple[list[str], str]] = []
    metadata_by_run: dict[str, dict[str, Any]] = {}
    for source in source_results:
        source_run = str(source.get("run_name", ""))
        if include is not None and include.search(source_run) is None:
            continue
        checkpoint_path = _best_checkpoint_path(source)
        if checkpoint_path is None:
            continue
        seed = _parse_seed_from_run_name(source_run)
        variant = _variant_from_run_name(source_run)
        for scenario in scenarios:
            scenario_name = str(scenario["name"])
            corruption_mode = str(scenario["mode"])
            corruption_fraction = float(scenario["fraction"])
            corruption_seed = seed + int(scenario["seed_offset"])
            run_name = _safe_run_name(f"{source_run}_eval_{scenario_name}")
            args = shlex.split(
                " ".join(
                    [
                        "--mode eval",
                        f"--checkpoint {checkpoint_path}",
                        f"--seed {seed}",
                        f"--corruption-seed {corruption_seed}",
                        f"--corruption-mode {corruption_mode}",
                        f"--corruption-fraction {corruption_fraction}",
                        "--corruption-mask-value 0.0",
                        "--corruption-input-mode image_plus_mask",
                        "--corruption-protocol boundary_clamped",
                        "--epochs 0",
                        "--train-limit 2000",
                        "--eval-limit 500",
                        "--batch-size 64",
                        "--patch-size 4",
                        "--artifact-every 1000000",
                        "--checkpoint-every 1000000",
                    ]
                )
            )
            output_dir = VOLUME_MOUNT / "mnist" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
            metadata_by_run[run_name] = {
                "source_run": source_run,
                "variant": variant,
                "seed": seed,
                "scenario": scenario_name,
                "corruption_mode": corruption_mode,
                "corruption_fraction": corruption_fraction,
                "corruption_seed": corruption_seed,
                "checkpoint": checkpoint_path,
            }
    return entries, metadata_by_run


def _settling_entries_from_sweep_json(
    source_json: Path,
    *,
    step_values: list[int],
    scenario_names: list[str],
    include_regex: str = "",
) -> tuple[list[tuple[list[str], str]], dict[str, dict[str, Any]]]:
    with open(source_json) as f:
        source_results = json.load(f)
    include = re.compile(include_regex) if include_regex else None
    scenario_catalog = {
        str(scenario["name"]): scenario
        for scenario in _robustness_scenarios("mask_stress")
    }
    scenarios = []
    for name in scenario_names:
        if name not in scenario_catalog:
            raise ValueError(f"unknown settling scenario: {name}")
        scenarios.append(scenario_catalog[name])

    entries: list[tuple[list[str], str]] = []
    metadata_by_run: dict[str, dict[str, Any]] = {}
    for source in source_results:
        source_run = str(source.get("run_name", ""))
        if include is not None and include.search(source_run) is None:
            continue
        checkpoint_path = _best_checkpoint_path(source)
        if checkpoint_path is None:
            continue
        seed = _parse_seed_from_run_name(source_run)
        variant = _variant_from_run_name(source_run)
        for scenario in scenarios:
            scenario_name = str(scenario["name"])
            corruption_mode = str(scenario["mode"])
            corruption_fraction = float(scenario["fraction"])
            corruption_seed = seed + int(scenario["seed_offset"])
            for step_count in step_values:
                run_name = _safe_run_name(
                    f"{source_run}_settle_s{step_count}_{scenario_name}"
                )
                args = shlex.split(
                    " ".join(
                        [
                            "--mode eval",
                            f"--checkpoint {checkpoint_path}",
                            f"--eval-winfree-steps {step_count}",
                            f"--seed {seed}",
                            f"--corruption-seed {corruption_seed}",
                            f"--corruption-mode {corruption_mode}",
                            f"--corruption-fraction {corruption_fraction}",
                            "--corruption-mask-value 0.0",
                            "--corruption-input-mode image_plus_mask",
                            "--corruption-protocol boundary_clamped",
                            "--epochs 0",
                            "--train-limit 2000",
                            "--eval-limit 500",
                            "--batch-size 64",
                            "--patch-size 4",
                            "--artifact-every 1000000",
                            "--checkpoint-every 1000000",
                        ]
                    )
                )
                output_dir = VOLUME_MOUNT / "mnist" / run_name
                args = _with_default_arg(args, "--output-dir", output_dir)
                entries.append((args, run_name))
                metadata_by_run[run_name] = {
                    "source_run": source_run,
                    "variant": variant,
                    "seed": seed,
                    "settling_steps": step_count,
                    "scenario": scenario_name,
                    "corruption_mode": corruption_mode,
                    "corruption_fraction": corruption_fraction,
                    "corruption_seed": corruption_seed,
                    "checkpoint": checkpoint_path,
                }
    return entries, metadata_by_run


def _summary_metric(summary: dict[str, Any], dotted_key: str) -> Any:
    value: Any = summary
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _write_sweep_csv(results: list[dict[str, Any]], path: Path) -> None:
    metric_names = [
        "final_eval_loss",
        "best_loss",
        "best_epoch",
        "final_epoch",
        "quality.pixel_correlation",
        "quality.foreground_f1",
        "quality.diversity_ratio",
        "quality.mae",
        "quality.mse",
        "quality.changed_mse",
        "quality.changed_mae",
        "quality.unchanged_mse",
        "quality.unchanged_mae",
        "quality.changed_improvement",
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


def _write_robustness_csv(
    results: list[dict[str, Any]],
    metadata_by_run: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    fieldnames = [
        "run",
        "root",
        "source_run",
        "variant",
        "seed",
        "scenario",
        "corruption_mode",
        "corruption_fraction",
        "corruption_seed",
        "checkpoint",
        *ROBUSTNESS_METRICS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(
            results,
            key=lambda row: (
                metadata_by_run.get(str(row.get("run_name", "")), {}).get(
                    "scenario", ""
                ),
                metadata_by_run.get(str(row.get("run_name", "")), {}).get(
                    "variant", ""
                ),
                metadata_by_run.get(str(row.get("run_name", "")), {}).get(
                    "seed", 0
                ),
            ),
        ):
            run_name = str(result.get("run_name", ""))
            summary = result.get("summary", {})
            metadata = metadata_by_run.get(run_name, {})
            writer.writerow(
                {
                    "run": run_name,
                    "root": result.get("output_dir"),
                    **metadata,
                    **{
                        metric: _get_nested(summary, metric)
                        for metric in ROBUSTNESS_METRICS
                    },
                }
            )


def _write_settling_csv(
    results: list[dict[str, Any]],
    metadata_by_run: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    fieldnames = [
        "run",
        "root",
        "source_run",
        "variant",
        "seed",
        "scenario",
        "settling_steps",
        "corruption_mode",
        "corruption_fraction",
        "corruption_seed",
        "checkpoint",
        *SETTLING_METRICS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(
            results,
            key=lambda row: (
                metadata_by_run.get(str(row.get("run_name", "")), {}).get(
                    "scenario", ""
                ),
                metadata_by_run.get(str(row.get("run_name", "")), {}).get(
                    "settling_steps", 0
                ),
                metadata_by_run.get(str(row.get("run_name", "")), {}).get(
                    "seed", 0
                ),
            ),
        ):
            run_name = str(result.get("run_name", ""))
            summary = result.get("summary", {})
            metadata = metadata_by_run.get(run_name, {})
            writer.writerow(
                {
                    "run": run_name,
                    "root": result.get("output_dir"),
                    **metadata,
                    **{
                        metric: _get_nested(summary, metric)
                        for metric in SETTLING_METRICS
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
def run_mnist_remote(experiment_args: list[str], run_name: str) -> dict[str, Any]:
    """Run one MNIST experiment remotely and return a compact summary."""

    import os

    safe_name = _safe_run_name(run_name)
    output_dir = VOLUME_MOUNT / "mnist" / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (VOLUME_MOUNT / "jax_cache").mkdir(parents=True, exist_ok=True)

    cache_target = VOLUME_MOUNT / "cache" / "oscnet"
    cache_target.mkdir(parents=True, exist_ok=True)
    local_cache = Path.home() / ".cache" / "oscnet"
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    if not local_cache.exists():
        os.symlink(cache_target, local_cache, target_is_directory=True)

    import jax

    from oscnet.experiments.mnist_autoencoder import (
        build_arg_parser,
        config_from_args,
        run_mnist_experiment,
    )

    args = list(experiment_args)
    args = _with_default_arg(args, "--output-dir", output_dir)

    parser = build_arg_parser()
    parsed = parser.parse_args(args)
    config = config_from_args(parsed)
    result = run_mnist_experiment(config)

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
    sweep_csv: str = str(DEFAULT_SWEEP_CSV),
    robustness_from_json: str = "",
    robustness_preset: str = "mask_stress",
    robustness_csv: str = str(DEFAULT_ROBUSTNESS_CSV),
    robustness_include_regex: str = "",
    settling_from_json: str = "",
    settling_csv: str = str(DEFAULT_SETTLING_CSV),
    settling_include_regex: str = "s050",
    settling_steps: str = "1,2,4,8,16,32",
    settling_scenarios: str = "block50,patch75",
    print_only: bool = False,
) -> None:
    """Launch a remote MNIST experiment.

    ``experiment_args`` is parsed exactly like the normal MNIST CLI arguments.
    Use ``print_only=True`` to inspect the remote command without starting a GPU
    worker.
    """

    if settling_from_json:
        step_values = _parse_int_list(settling_steps)
        scenario_names = _parse_name_list(settling_scenarios)
        entries, metadata_by_run = _settling_entries_from_sweep_json(
            Path(settling_from_json),
            step_values=step_values,
            scenario_names=scenario_names,
            include_regex=settling_include_regex,
        )
        if not entries:
            raise ValueError("settling source produced no eval entries")
        request = {
            "settling_from_json": settling_from_json,
            "settling_include_regex": settling_include_regex,
            "settling_steps": step_values,
            "settling_scenarios": scenario_names,
            "runs": [
                {
                    "run_name": name,
                    "experiment_args": args,
                    **metadata_by_run.get(name, {}),
                }
                for args, name in entries
            ],
            "gpu": GPU,
            "max_containers": MAX_CONTAINERS,
            "jax_package": JAX_PACKAGE,
            "timeout_seconds": TIMEOUT_SECONDS,
            "volume": VOLUME_NAME,
            "settling_csv": settling_csv,
        }
        if print_only:
            print(json.dumps(request, indent=2, sort_keys=True))
            return

        print(json.dumps(request, indent=2, sort_keys=True))
        results = list(
            run_mnist_remote.starmap(
                entries,
                order_outputs=False,
                return_exceptions=False,
            )
        )
        csv_path = Path(settling_csv)
        _write_settling_csv(results, metadata_by_run, csv_path)
        enriched_results = []
        for result in results:
            run = str(result.get("run_name", ""))
            enriched = dict(result)
            enriched["settling"] = metadata_by_run.get(run, {})
            enriched_results.append(enriched)
        with open(csv_path.with_suffix(".json"), "w") as f:
            json.dump(enriched_results, f, indent=2, sort_keys=True)
        print(f"wrote {csv_path}")
        print(json.dumps(enriched_results, indent=2, sort_keys=True))
        return

    if robustness_from_json:
        entries, metadata_by_run = _robustness_entries_from_sweep_json(
            Path(robustness_from_json),
            preset=robustness_preset,
            include_regex=robustness_include_regex,
        )
        if not entries:
            raise ValueError("robustness source produced no eval entries")
        request = {
            "robustness_from_json": robustness_from_json,
            "robustness_preset": robustness_preset,
            "robustness_include_regex": robustness_include_regex,
            "runs": [
                {
                    "run_name": name,
                    "experiment_args": args,
                    **metadata_by_run.get(name, {}),
                }
                for args, name in entries
            ],
            "gpu": GPU,
            "max_containers": MAX_CONTAINERS,
            "jax_package": JAX_PACKAGE,
            "timeout_seconds": TIMEOUT_SECONDS,
            "volume": VOLUME_NAME,
            "robustness_csv": robustness_csv,
        }
        if print_only:
            print(json.dumps(request, indent=2, sort_keys=True))
            return

        print(json.dumps(request, indent=2, sort_keys=True))
        results = list(
            run_mnist_remote.starmap(
                entries,
                order_outputs=False,
                return_exceptions=False,
            )
        )
        csv_path = Path(robustness_csv)
        _write_robustness_csv(results, metadata_by_run, csv_path)
        enriched_results = []
        for result in results:
            run = str(result.get("run_name", ""))
            enriched = dict(result)
            enriched["robustness"] = metadata_by_run.get(run, {})
            enriched_results.append(enriched)
        with open(csv_path.with_suffix(".json"), "w") as f:
            json.dump(enriched_results, f, indent=2, sort_keys=True)
        print(f"wrote {csv_path}")
        print(json.dumps(enriched_results, indent=2, sort_keys=True))
        return

    if sweep_preset:
        entries = _sweep_entries(sweep_preset)
        if sweep_csv == str(DEFAULT_SWEEP_CSV):
            sweep_csv = str(SWEEP_CSVS.get(sweep_preset, DEFAULT_SWEEP_CSV))
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
            "sweep_csv": sweep_csv,
        }
        if print_only:
            print(json.dumps(request, indent=2, sort_keys=True))
            return

        print(json.dumps(request, indent=2, sort_keys=True))
        results = list(
            run_mnist_remote.starmap(
                entries,
                order_outputs=False,
                return_exceptions=False,
            )
        )
        csv_path = Path(sweep_csv)
        _write_sweep_csv(results, csv_path)
        with open(csv_path.with_suffix(".json"), "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"wrote {csv_path}")
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    safe_name = _safe_run_name(run_name)
    args = shlex.split(experiment_args)
    output_dir = VOLUME_MOUNT / "mnist" / safe_name
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
    result = run_mnist_remote.remote(args, safe_name)
    print(json.dumps(result, indent=2, sort_keys=True))
