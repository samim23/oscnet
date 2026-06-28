"""Run MNIST shape-to-pixel renderer experiments on Modal GPUs."""

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

APP_NAME = "oscnet-mnist-shape-pixel"
VOLUME_NAME = os.environ.get("OSCNET_MODAL_VOLUME", "oscnet-runs")
VOLUME_MOUNT = Path("/mnt/oscnet-runs")
GPU = os.environ.get("OSCNET_MODAL_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("OSCNET_MODAL_TIMEOUT_SECONDS", "10800"))
JAX_PACKAGE = os.environ.get("OSCNET_MODAL_JAX", "jax[cuda13]")
MAX_CONTAINERS = int(os.environ.get("OSCNET_MODAL_MAX_CONTAINERS", "3"))

DEFAULT_SMOKE_ARGS = (
    "--data-source synthetic --model-family coarse_phase_flow --seed 0 --epochs 1 "
    "--train-limit 8 --eval-limit 4 --eval-sample-count 4 --batch-size 4 "
    "--field-channels 2 --steps 1 --sample-steps 2 "
    "--checkpoint-every 1 --artifact-every 1"
)

SWEEP_CSVS = {
    "mnist_shape_pixel_core": Path(
        "outputs/analysis/modal_mnist_shape_pixel_core.csv"
    ),
    "mnist_shape_pixel_basin_probe": Path(
        "outputs/analysis/modal_mnist_shape_pixel_basin_probe.csv"
    ),
    "mnist_shape_pixel_shape_condition_probe": Path(
        "outputs/analysis/modal_mnist_shape_pixel_shape_condition_probe.csv"
    ),
    "mnist_shape_pixel_shape_gated_probe": Path(
        "outputs/analysis/modal_mnist_shape_pixel_shape_gated_probe.csv"
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
    name = name.strip() or time.strftime("mnist-shape-pixel-%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip(".-") or "mnist-shape-pixel-run"


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


def _mnist_shape_pixel_core_sweep(
    *,
    include_basin: bool = False,
    include_shape_condition_probe: bool = False,
    sample_readout_mode: str = "primary",
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
        "--shape-velocity-weight 0.1",
        "--closure-loss-weight 0.0",
        "--sample-steps 16",
        "--sample-method euler",
        f"--sample-readout-mode {sample_readout_mode}",
        "--clamp-shape",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
    ]
    if include_basin:
        common.append("--basin-t-values 0.1,0.25,0.5,0.75,0.9")
    if include_shape_condition_probe:
        common.append("--shape-condition-t-values 0.1,0.5,0.9")
        common.append("--shape-condition-noise-modes uniform,salt_pepper,zeros")
    variants = [
        ("coarse_phase_flow", ["--model-family coarse_phase_flow"]),
        ("phase_flow", ["--model-family phase_flow"]),
        ("no_dynamics", ["--model-family phase_flow_no_dynamics"]),
        ("recurrent_conv_flow", ["--model-family recurrent_conv_flow"]),
    ]
    readout_suffix = "" if sample_readout_mode == "primary" else f"_{sample_readout_mode}"
    entries = []
    for seed in (31, 32):
        for suffix, model_args in variants:
            run_name = f"mnist_shape_pixel_{suffix}{readout_suffix}_seed{seed}_20e"
            args = shlex.split(" ".join([f"--seed {seed}", *common, *model_args]))
            output_dir = VOLUME_MOUNT / "mnist_shape_pixel" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _sweep_entries(preset: str) -> list[tuple[list[str], str]]:
    if preset == "mnist_shape_pixel_core":
        return _mnist_shape_pixel_core_sweep()
    if preset == "mnist_shape_pixel_basin_probe":
        return _mnist_shape_pixel_core_sweep(include_basin=True)
    if preset == "mnist_shape_pixel_shape_condition_probe":
        return _mnist_shape_pixel_core_sweep(include_shape_condition_probe=True)
    if preset == "mnist_shape_pixel_shape_gated_probe":
        return _mnist_shape_pixel_core_sweep(
            include_shape_condition_probe=True,
            sample_readout_mode="shape_gated",
        )
    raise ValueError("unknown sweep preset")


def _write_sweep_csv(results: list[dict[str, Any]], path: Path) -> None:
    metric_names = [
        "final_eval_loss",
        "best_loss",
        "best_epoch",
        "final_epoch",
        "final_eval_pixel_velocity_loss",
        "final_eval_shape_velocity_loss",
        "final_eval_clean_loss",
        "final_eval_closure_loss",
        "config.run.seed",
        "shape_pixel.model_family",
        "shape_pixel.field_channels",
        "shape_pixel.steps",
        "shape_pixel.train_dynamics",
        "shape_pixel.conditional",
        "shape_pixel.position_features",
        "shape_pixel.clean_loss_weight",
        "shape_pixel.shape_velocity_weight",
        "shape_pixel.closure_loss_weight",
        "shape_pixel.sample_steps",
        "shape_pixel.sample_method",
        "shape_pixel.sample_readout_mode",
        "shape_pixel.basin_t_values",
        "shape_pixel.shape_condition_t_values",
        "shape_pixel.shape_condition_noise_modes",
        "shape_pixel.clamp_shape",
        "shape_pixel.paired_sample_mse",
        "shape_pixel.sample_mean",
        "shape_pixel.sample_std",
        "shape_pixel.sample_pixel_mean_mse",
        "shape_pixel.sample_pixel_std_mse",
        "shape_pixel.sample_diversity_ratio",
        "shape_pixel.sample_nearest_real_mse",
        "shape_pixel.real_nearest_real_mse",
        "shape_pixel.sample_active_fraction",
        "shape_pixel.sample_component_count",
        "shape_pixel.sample_largest_component_fraction",
        "shape_pixel.real_active_fraction",
        "shape_pixel.real_component_count",
        "shape_pixel.real_largest_component_fraction",
        "shape_pixel.coarse_grid_size",
        "shape_pixel.global_coupling_strength",
        "shape_pixel.state_mean_abs_displacement",
        "shape_pixel.basin.t0_100.initial_paired_mse",
        "shape_pixel.basin.t0_100.paired_mse",
        "shape_pixel.basin.t0_100.paired_mse_improvement_fraction",
        "shape_pixel.basin.t0_100.sample_active_fraction",
        "shape_pixel.basin.t0_100.sample_mean",
        "shape_pixel.basin.t0_250.initial_paired_mse",
        "shape_pixel.basin.t0_250.paired_mse",
        "shape_pixel.basin.t0_250.paired_mse_improvement_fraction",
        "shape_pixel.basin.t0_250.sample_active_fraction",
        "shape_pixel.basin.t0_250.sample_mean",
        "shape_pixel.basin.t0_500.initial_paired_mse",
        "shape_pixel.basin.t0_500.paired_mse",
        "shape_pixel.basin.t0_500.paired_mse_improvement_fraction",
        "shape_pixel.basin.t0_500.sample_active_fraction",
        "shape_pixel.basin.t0_500.sample_mean",
        "shape_pixel.basin.t0_750.initial_paired_mse",
        "shape_pixel.basin.t0_750.paired_mse",
        "shape_pixel.basin.t0_750.paired_mse_improvement_fraction",
        "shape_pixel.basin.t0_750.sample_active_fraction",
        "shape_pixel.basin.t0_750.sample_mean",
        "shape_pixel.basin.t0_900.initial_paired_mse",
        "shape_pixel.basin.t0_900.paired_mse",
        "shape_pixel.basin.t0_900.paired_mse_improvement_fraction",
        "shape_pixel.basin.t0_900.sample_active_fraction",
        "shape_pixel.basin.t0_900.sample_mean",
        "shape_pixel.shape_condition_probe.uniform.t0_100.condition_paired_mse",
        "shape_pixel.shape_condition_probe.uniform.t0_100.paired_sample_mse",
        "shape_pixel.shape_condition_probe.uniform.t0_100.sample_nearest_real_mse",
        "shape_pixel.shape_condition_probe.uniform.t0_100.sample_active_fraction",
        "shape_pixel.shape_condition_probe.uniform.t0_500.condition_paired_mse",
        "shape_pixel.shape_condition_probe.uniform.t0_500.paired_sample_mse",
        "shape_pixel.shape_condition_probe.uniform.t0_500.sample_nearest_real_mse",
        "shape_pixel.shape_condition_probe.uniform.t0_500.sample_active_fraction",
        "shape_pixel.shape_condition_probe.uniform.t0_900.condition_paired_mse",
        "shape_pixel.shape_condition_probe.uniform.t0_900.paired_sample_mse",
        "shape_pixel.shape_condition_probe.uniform.t0_900.sample_nearest_real_mse",
        "shape_pixel.shape_condition_probe.uniform.t0_900.sample_active_fraction",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_100.condition_paired_mse",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_100.paired_sample_mse",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_100.sample_nearest_real_mse",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_100.sample_active_fraction",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_500.condition_paired_mse",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_500.paired_sample_mse",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_500.sample_nearest_real_mse",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_500.sample_active_fraction",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_900.condition_paired_mse",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_900.paired_sample_mse",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_900.sample_nearest_real_mse",
        "shape_pixel.shape_condition_probe.salt_pepper.t0_900.sample_active_fraction",
        "shape_pixel.shape_condition_probe.zeros.t0_100.condition_paired_mse",
        "shape_pixel.shape_condition_probe.zeros.t0_100.paired_sample_mse",
        "shape_pixel.shape_condition_probe.zeros.t0_100.sample_nearest_real_mse",
        "shape_pixel.shape_condition_probe.zeros.t0_100.sample_active_fraction",
        "shape_pixel.shape_condition_probe.zeros.t0_500.condition_paired_mse",
        "shape_pixel.shape_condition_probe.zeros.t0_500.paired_sample_mse",
        "shape_pixel.shape_condition_probe.zeros.t0_500.sample_nearest_real_mse",
        "shape_pixel.shape_condition_probe.zeros.t0_500.sample_active_fraction",
        "shape_pixel.shape_condition_probe.zeros.t0_900.condition_paired_mse",
        "shape_pixel.shape_condition_probe.zeros.t0_900.paired_sample_mse",
        "shape_pixel.shape_condition_probe.zeros.t0_900.sample_nearest_real_mse",
        "shape_pixel.shape_condition_probe.zeros.t0_900.sample_active_fraction",
        "train_seconds",
    ]
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
def run_mnist_shape_pixel_remote(
    experiment_args: list[str],
    run_name: str,
) -> dict[str, Any]:
    """Run one MNIST shape-to-pixel experiment remotely."""

    import os

    safe_name = _safe_run_name(run_name)
    output_dir = VOLUME_MOUNT / "mnist_shape_pixel" / safe_name
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
    from oscnet.experiments.mnist_shape_pixel import (
        MNISTShapePixelExperimentConfig,
        build_arg_parser,
        run_mnist_shape_pixel_experiment,
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
    config = MNISTShapePixelExperimentConfig(
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
        shape_velocity_weight=args.shape_velocity_weight,
        closure_loss_weight=args.closure_loss_weight,
        t_min=args.t_min,
        t_max=args.t_max,
        eval_sample_count=args.eval_sample_count,
        sample_steps=args.sample_steps,
        sample_method=args.sample_method,
        sample_readout_mode=args.sample_readout_mode,
        basin_t_values=args.basin_t_values,
        shape_condition_t_values=args.shape_condition_t_values,
        shape_condition_noise_modes=args.shape_condition_noise_modes,
        clamp_shape=args.clamp_shape,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )
    result = run_mnist_shape_pixel_experiment(config)
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
    artifacts = result.paths.artifacts
    png_payload: dict[str, str | None] = {}
    for name in ("shape", "samples", "denoised"):
        path = artifacts / f"{name}_epoch_{artifact_epoch:03d}.png"
        png_payload[f"{name}_png_b64"] = (
            base64.b64encode(path.read_bytes()).decode("ascii")
            if path.exists()
            else None
        )
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
        **png_payload,
    }
    with (result.paths.root / "modal_result.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    volume.commit()
    return payload


def _write_pngs(results: list[dict[str, Any]], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for result in results:
        run = result["run_name"]
        for name in ("shape", "samples", "denoised"):
            payload_key = f"{name}_png_b64"
            if result.get(payload_key):
                (root / f"{run}_{name}.png").write_bytes(
                    base64.b64decode(result[payload_key])
                )


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"shape_png_b64", "samples_png_b64", "denoised_png_b64"}
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
    """Launch a remote MNIST shape-to-pixel experiment."""

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
            run_mnist_shape_pixel_remote.starmap(
                entries,
                order_outputs=False,
                return_exceptions=False,
            )
        )
        _write_sweep_csv(results, csv_path)
        compact_results = _compact_results(results)
        with csv_path.with_suffix(".json").open("w") as f:
            json.dump(compact_results, f, indent=2, sort_keys=True)
        _write_pngs(results, Path("outputs/analysis/modal_mnist_shape_pixel_samples"))
        print(f"wrote {csv_path}")
        print(json.dumps(compact_results, indent=2, sort_keys=True))
        return

    safe_name = _safe_run_name(run_name)
    args = shlex.split(experiment_args)
    output_dir = VOLUME_MOUNT / "mnist_shape_pixel" / safe_name
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
    result = run_mnist_shape_pixel_remote.remote(args, safe_name)
    _write_pngs([result], Path("outputs/analysis/modal_mnist_shape_pixel_samples"))
    print(json.dumps(_compact_result(result), indent=2, sort_keys=True))
