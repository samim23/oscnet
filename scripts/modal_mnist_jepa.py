"""Run OscNet MNIST JEPA-lite experiments on Modal GPU workers."""

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

APP_NAME = "oscnet-mnist-jepa"
VOLUME_NAME = os.environ.get("OSCNET_MODAL_VOLUME", "oscnet-runs")
VOLUME_MOUNT = Path("/mnt/oscnet-runs")
GPU = os.environ.get("OSCNET_MODAL_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("OSCNET_MODAL_TIMEOUT_SECONDS", "10800"))
JAX_PACKAGE = os.environ.get("OSCNET_MODAL_JAX", "jax[cuda13]")
MAX_CONTAINERS = int(os.environ.get("OSCNET_MODAL_MAX_CONTAINERS", "3"))

DEFAULT_SMOKE_ARGS = (
    "--data-source synthetic --model-family winfree_global_rate_phase "
    "--corruption-mode block_occlusion --corruption-fraction 0.5 "
    "--seed 0 --corruption-seed 0 --epochs 1 --train-limit 8 --eval-limit 4 "
    "--batch-size 4 --hidden-dim 4 --latent-dim 4 --embedding-dim 4 "
    "--patch-size 7 --winfree-steps 1 --artifact-every 1 --checkpoint-every 1"
)

SWEEP_CSVS = {
    "block50_jepa_core": Path("outputs/analysis/modal_block50_jepa_core.csv"),
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
    name = name.strip() or time.strftime("mnist-jepa-%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip(".-") or "mnist-jepa-run"


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


def _block50_jepa_core_sweep() -> list[tuple[list[str], str]]:
    entries = []
    variants = [
        (
            "feedforward",
            ["--model-family feedforward_patch"],
        ),
        (
            "recurrent_conv",
            [
                "--model-family recurrent_conv",
                "--recurrent-conv-steps 8",
                "--recurrent-conv-kernel-size 3",
                "--recurrent-conv-residual-strength 0.5",
            ],
        ),
        (
            "conv_lstm",
            [
                "--model-family conv_lstm",
                "--conv-lstm-steps 8",
                "--conv-lstm-kernel-size 3",
                "--conv-lstm-forget-bias 1.0",
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
            ],
        ),
    ]
    common = [
        "--corruption-mode block_occlusion",
        "--corruption-fraction 0.5",
        "--corruption-mask-value 0.0",
        "--corruption-input-mode image_plus_mask",
        "--target-encoder dct_lowfreq",
        "--embedding-dim 8",
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
                "mnist_block50_jepa_"
                f"{suffix}_dct8_h64_l96_train2000_seed{seed}_10e"
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
            output_dir = VOLUME_MOUNT / "mnist_jepa" / run_name
            args = _with_default_arg(args, "--output-dir", output_dir)
            entries.append((args, run_name))
    return entries


def _sweep_entries(preset: str) -> list[tuple[list[str], str]]:
    if preset == "block50_jepa_core":
        return _block50_jepa_core_sweep()
    raise ValueError("unknown sweep preset")


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
        "jepa.zero_embedding_mse",
        "jepa.train_mean_embedding_mse",
        "jepa.margin_vs_zero_embedding",
        "jepa.margin_vs_train_mean_embedding",
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
def run_mnist_jepa_remote(
    experiment_args: list[str],
    run_name: str,
) -> dict[str, Any]:
    """Run one MNIST JEPA-lite experiment remotely."""

    import os

    safe_name = _safe_run_name(run_name)
    output_dir = VOLUME_MOUNT / "mnist_jepa" / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (VOLUME_MOUNT / "jax_cache").mkdir(parents=True, exist_ok=True)

    cache_target = VOLUME_MOUNT / "cache" / "oscnet"
    cache_target.mkdir(parents=True, exist_ok=True)
    local_cache = Path.home() / ".cache" / "oscnet"
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    if not local_cache.exists():
        os.symlink(cache_target, local_cache, target_is_directory=True)

    import jax

    from oscnet.experiments.mnist_jepa import (
        build_arg_parser,
        config_from_args,
        run_mnist_jepa_experiment,
    )

    args = _with_default_arg(list(experiment_args), "--output-dir", output_dir)
    parser = build_arg_parser()
    parsed = parser.parse_args(args)
    result = run_mnist_jepa_experiment(config_from_args(parsed))

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
    """Launch a remote MNIST JEPA-lite experiment."""

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
            run_mnist_jepa_remote.starmap(
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
    output_dir = VOLUME_MOUNT / "mnist_jepa" / safe_name
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
    result = run_mnist_jepa_remote.remote(args, safe_name)
    print(json.dumps(result, indent=2, sort_keys=True))
