"""Run MNIST phase-VAE experiments on Modal GPUs."""

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

APP_NAME = "oscnet-mnist-phase-vae"
VOLUME_NAME = os.environ.get("OSCNET_MODAL_VOLUME", "oscnet-runs")
VOLUME_MOUNT = Path("/mnt/oscnet-runs")
GPU = os.environ.get("OSCNET_MODAL_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("OSCNET_MODAL_TIMEOUT_SECONDS", "10800"))
JAX_PACKAGE = os.environ.get("OSCNET_MODAL_JAX", "jax[cuda13]")
MAX_CONTAINERS = int(os.environ.get("OSCNET_MODAL_MAX_CONTAINERS", "3"))

DEFAULT_SMOKE_ARGS = (
    "--data-source synthetic --model-family phase_vae --seed 0 --epochs 1 "
    "--train-limit 8 --eval-limit 4 --eval-sample-count 4 --batch-size 4 "
    "--latent-dim 8 --hidden-dim 12 --encoder-depth 1 --decoder-depth 1 "
    "--steps 1 --checkpoint-every 1 --artifact-every 1"
)

SWEEP_CSVS = {
    "mnist_phase_vae_core": Path(
        "outputs/analysis/modal_mnist_phase_vae_core.csv"
    ),
    "mnist_phase_vae_forced_dynamics_core": Path(
        "outputs/analysis/modal_mnist_phase_vae_forced_dynamics_core.csv"
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
    name = name.strip() or time.strftime("mnist-phase-vae-%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip(".-") or "mnist-phase-vae-run"


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


def _mnist_phase_vae_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        ("phase_vae", ["--model-family phase_vae"]),
        ("frozen_phase_vae", ["--model-family frozen_phase_vae"]),
        ("no_dynamics", ["--model-family phase_vae_no_dynamics"]),
    ]
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--latent-dim 32",
        "--hidden-dim 256",
        "--encoder-depth 2",
        "--decoder-depth 2",
        "--steps 4",
        "--dt 0.1",
        "--coupling-strength 1.0",
        "--omega-scale 0.2",
        "--coupling-init-scale 0.05",
        "--kl-weight 0.001",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
    ]
    for seed in (21,):
        for suffix, model_args in variants:
            run_name = f"mnist_phase_vae_{suffix}_latent32_h256_steps4_seed{seed}_20e"
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_phase_vae" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _mnist_phase_vae_forced_dynamics_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        ("phase_vae", ["--model-family phase_vae"]),
        ("frozen_phase_vae", ["--model-family frozen_phase_vae"]),
        ("no_dynamics", ["--model-family phase_vae_no_dynamics"]),
    ]
    common = [
        "--data-source idx",
        "--epochs 20",
        "--train-limit 10000",
        "--eval-limit 1000",
        "--eval-sample-count 64",
        "--batch-size 128",
        "--latent-dim 128",
        "--hidden-dim 256",
        "--encoder-depth 2",
        "--decoder-depth 2",
        "--steps 8",
        "--dt 0.2",
        "--coupling-strength 2.0",
        "--omega-scale 0.5",
        "--coupling-init-scale 0.2",
        "--phase-readout-mode mean_relative",
        "--kl-weight 0.001",
        "--learning-rate 0.001",
        "--weight-decay 0.0001",
        "--checkpoint-every 20",
        "--artifact-every 20",
    ]
    for seed in (21,):
        for suffix, model_args in variants:
            run_name = (
                "mnist_phase_vae_forced_"
                f"{suffix}_latent128_h256_steps8_seed{seed}_20e"
            )
            args = shlex.split(
                " ".join([f"--seed {seed}", *common, *model_args])
            )
            output_dir = VOLUME_MOUNT / "mnist_phase_vae" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _sweep_entries(preset: str) -> list[tuple[list[str], str]]:
    if preset == "mnist_phase_vae_core":
        return _mnist_phase_vae_core_sweep()
    if preset == "mnist_phase_vae_forced_dynamics_core":
        return _mnist_phase_vae_forced_dynamics_sweep()
    raise ValueError("unknown sweep preset")


def _write_sweep_csv(results: list[dict[str, Any]], path: Path) -> None:
    metric_names = [
        "final_eval_loss",
        "best_loss",
        "best_epoch",
        "final_epoch",
        "final_eval_reconstruction_loss",
        "final_eval_mse",
        "final_eval_kl_loss",
        "phase_vae.model_family",
        "phase_vae.latent_dim",
        "phase_vae.steps",
        "phase_vae.train_dynamics",
        "phase_vae.phase_readout_mode",
        "phase_vae.kl_weight",
        "phase_vae.reconstruction_mse",
        "phase_vae.sample_mean",
        "phase_vae.sample_std",
        "phase_vae.sample_pixel_mean_mse",
        "phase_vae.sample_pixel_std_mse",
        "phase_vae.sample_diversity_ratio",
        "phase_vae.sample_nearest_real_mse",
        "phase_vae.real_nearest_real_mse",
        "phase_vae.phase_mean_abs_displacement",
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
def run_mnist_phase_vae_remote(
    experiment_args: list[str],
    run_name: str,
) -> dict[str, Any]:
    """Run one MNIST phase-VAE experiment remotely."""

    import base64
    import os

    safe_name = _safe_run_name(run_name)
    output_dir = VOLUME_MOUNT / "mnist_phase_vae" / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (VOLUME_MOUNT / "jax_cache").mkdir(parents=True, exist_ok=True)

    cache_target = VOLUME_MOUNT / "cache" / "oscnet"
    cache_target.mkdir(parents=True, exist_ok=True)
    local_cache = Path.home() / ".cache" / "oscnet"
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    if not local_cache.exists():
        os.symlink(cache_target, local_cache, target_is_directory=True)

    import jax

    from oscnet.experiments.mnist_phase_vae import (
        build_arg_parser,
        run_mnist_phase_vae_experiment,
        MNISTPhaseVAEExperimentConfig,
    )
    from oscnet.experiments.harness import AutoencoderExperimentConfig

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
    config = MNISTPhaseVAEExperimentConfig(
        run=run_config,
        model_family=args.model_family,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        steps=args.steps,
        dt=args.dt,
        coupling_strength=args.coupling_strength,
        omega_scale=args.omega_scale,
        coupling_init_scale=args.coupling_init_scale,
        phase_readout_mode=args.phase_readout_mode,
        kl_weight=args.kl_weight,
        reconstruction_loss=args.reconstruction_loss,
        eval_sample_count=args.eval_sample_count,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )
    result = run_mnist_phase_vae_experiment(config)
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
    recon_png = result.paths.artifacts / f"reconstruction_epoch_{artifact_epoch:03d}.png"
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
        "reconstruction_png_b64": (
            base64.b64encode(recon_png.read_bytes()).decode("ascii")
            if recon_png.exists()
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
        if result.get("reconstruction_png_b64"):
            (root / f"{run}_reconstructions.png").write_bytes(
                base64.b64decode(result["reconstruction_png_b64"])
            )


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop inline image payloads before writing logs or JSON summaries."""

    return {
        key: value
        for key, value in result.items()
        if key not in {"samples_png_b64", "reconstruction_png_b64"}
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
    """Launch a remote MNIST phase-VAE experiment."""

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
            run_mnist_phase_vae_remote.starmap(
                entries,
                order_outputs=False,
                return_exceptions=False,
            )
        )
        _write_sweep_csv(results, csv_path)
        compact_results = _compact_results(results)
        with csv_path.with_suffix(".json").open("w") as f:
            json.dump(compact_results, f, indent=2, sort_keys=True)
        _write_pngs(results, Path("outputs/analysis/modal_mnist_phase_vae_samples"))
        print(f"wrote {csv_path}")
        print(json.dumps(compact_results, indent=2, sort_keys=True))
        return

    safe_name = _safe_run_name(run_name)
    args = shlex.split(experiment_args)
    output_dir = VOLUME_MOUNT / "mnist_phase_vae" / safe_name
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
    result = run_mnist_phase_vae_remote.remote(args, safe_name)
    _write_pngs([result], Path("outputs/analysis/modal_mnist_phase_vae_samples"))
    print(json.dumps(_compact_result(result), indent=2, sort_keys=True))
