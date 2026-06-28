"""Run the upstream Un-0 reference generator on Modal.

This is intentionally separate from the OscNet JAX experiment runners. It is a
calibration helper: load an official Un-0 checkpoint, generate a small sample
grid, and record lightweight reference metrics without adding PyTorch as an
OscNet dependency.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "oscnet-un0-reference"
VOLUME_NAME = os.environ.get("OSCNET_MODAL_VOLUME", "oscnet-runs")
VOLUME_MOUNT = Path("/mnt/oscnet-runs")
GPU = os.environ.get("OSCNET_MODAL_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("OSCNET_MODAL_TIMEOUT_SECONDS", "10800"))
MAX_CONTAINERS = int(os.environ.get("OSCNET_MODAL_MAX_CONTAINERS", "1"))
UN0_COMMIT = os.environ.get("OSCNET_UN0_COMMIT", "43f2587")

REMOTE_PACKAGES = [
    *os.environ.get("OSCNET_MODAL_TORCH_PACKAGES", "torch torchvision").split(),
    "torchdiffeq>=0.2.5",
    "huggingface_hub>=0.36.0",
    "numpy>=1.24.0",
    "Pillow>=9.0.0",
    "tqdm>=4.65.0",
]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(*REMOTE_PACKAGES)
    .run_commands(
        "python -m pip install --no-deps "
        f"git+https://github.com/unconv-ai/Un-0.git@{UN0_COMMIT}"
    )
    .env(
        {
            "HF_HOME": str(VOLUME_MOUNT / "hf_cache"),
            "TORCH_HOME": str(VOLUME_MOUNT / "torch_cache"),
        }
    )
)


def _flatten_metrics(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metrics(name, value))
        else:
            flat[name] = value
    return flat


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    volumes={VOLUME_MOUNT: volume},
)
def run_un0_reference_remote(
    *,
    pretrained: str,
    classes: list[int],
    samples_per_class: int,
    seed: int,
    num_steps: int | None,
) -> dict[str, Any]:
    """Generate a reference sample grid with upstream Un-0 code."""

    import base64
    import json

    import torch

    from un0.common import resolve_device, save_sample_grid, seed_everything
    from un0.model import ConditionalImplicitKuramotoGenerator

    seed_everything(int(seed))
    device = resolve_device("auto")
    model = ConditionalImplicitKuramotoGenerator.from_pretrained(
        pretrained,
        device=device,
    )
    if num_steps is not None:
        model.num_steps = int(num_steps)
    model.eval()

    class_ids = torch.tensor(classes, device=device, dtype=torch.long)
    class_ids = class_ids.repeat_interleave(int(samples_per_class))

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        samples = model.sample(class_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    generation_seconds = time.time() - start

    image_size = round((samples.shape[1] // 3) ** 0.5)
    safe_name = pretrained.replace("/", "_")
    run_root = VOLUME_MOUNT / "un0_reference" / f"{safe_name}_seed{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    sample_path = run_root / "samples.png"
    save_sample_grid(
        samples,
        sample_path,
        image_size=image_size,
        nrow=int(samples_per_class),
    )

    samples_cpu = samples.detach().float().cpu()
    samples_01 = ((samples_cpu + 1.0) * 0.5).clamp(0.0, 1.0)
    param_count = sum(param.numel() for param in model.parameters())
    trainable_param_count = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    metrics = {
        "pretrained": pretrained,
        "un0_commit": UN0_COMMIT,
        "device": str(device),
        "torch_version": torch.__version__,
        "num_samples": int(samples.shape[0]),
        "image_size": int(image_size),
        "num_steps": int(model.num_steps),
        "n_oscillators": int(model.dynamics.n),
        "n_conditional_oscillators": int(model.dynamics.n_cond),
        "readout_relativization": model.readout.relativization,
        "readout_encoding": model.readout.encoding,
        "solver": model.solver,
        "class_dropout_prob": float(model.class_dropout_prob),
        "param_count": int(param_count),
        "trainable_param_count": int(trainable_param_count),
        "generation_seconds": float(generation_seconds),
        "samples_per_second": float(samples.shape[0] / generation_seconds)
        if generation_seconds > 0
        else None,
        "generated_mean": float(samples_01.mean()),
        "generated_std": float(samples_01.std()),
        "generated_min": float(samples_01.min()),
        "generated_max": float(samples_01.max()),
        "remote_sample_path": str(sample_path),
    }

    return {
        "metrics_json": json.dumps(metrics),
        "sample_png_b64": base64.b64encode(sample_path.read_bytes()).decode("ascii"),
    }


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    volumes={VOLUME_MOUNT: volume},
)
def run_un0_step_sweep_remote(
    *,
    pretrained: str,
    classes: list[int],
    samples_per_class: int,
    seed: int,
    step_values: list[int],
) -> dict[str, Any]:
    """Generate comparable reference grids across inference step counts."""

    import base64
    import json

    import torch

    from un0.common import resolve_device, save_sample_grid, seed_everything
    from un0.model import ConditionalImplicitKuramotoGenerator

    device = resolve_device("auto")
    model = ConditionalImplicitKuramotoGenerator.from_pretrained(
        pretrained,
        device=device,
    )
    model.eval()
    class_ids = torch.tensor(classes, device=device, dtype=torch.long)
    class_ids = class_ids.repeat_interleave(int(samples_per_class))
    image_size = 32
    safe_name = pretrained.replace("/", "_")
    run_root = VOLUME_MOUNT / "un0_reference" / f"{safe_name}_seed{seed}_step_sweep"
    run_root.mkdir(parents=True, exist_ok=True)

    outputs = []
    for num_steps in step_values:
        seed_everything(int(seed))
        model.num_steps = int(num_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            samples = model.sample(class_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        generation_seconds = time.time() - start

        image_size = round((samples.shape[1] // 3) ** 0.5)
        sample_path = run_root / f"steps{num_steps:02d}.png"
        save_sample_grid(
            samples,
            sample_path,
            image_size=image_size,
            nrow=int(samples_per_class),
        )

        samples_cpu = samples.detach().float().cpu()
        samples_01 = ((samples_cpu + 1.0) * 0.5).clamp(0.0, 1.0)
        metrics = {
            "pretrained": pretrained,
            "un0_commit": UN0_COMMIT,
            "device": str(device),
            "torch_version": torch.__version__,
            "num_samples": int(samples.shape[0]),
            "image_size": int(image_size),
            "num_steps": int(model.num_steps),
            "n_oscillators": int(model.dynamics.n),
            "n_conditional_oscillators": int(model.dynamics.n_cond),
            "readout_relativization": model.readout.relativization,
            "readout_encoding": model.readout.encoding,
            "solver": model.solver,
            "class_dropout_prob": float(model.class_dropout_prob),
            "param_count": int(sum(param.numel() for param in model.parameters())),
            "trainable_param_count": int(
                sum(param.numel() for param in model.parameters() if param.requires_grad)
            ),
            "generation_seconds": float(generation_seconds),
            "samples_per_second": float(samples.shape[0] / generation_seconds)
            if generation_seconds > 0
            else None,
            "generated_mean": float(samples_01.mean()),
            "generated_std": float(samples_01.std()),
            "generated_min": float(samples_01.min()),
            "generated_max": float(samples_01.max()),
            "remote_sample_path": str(sample_path),
        }
        outputs.append(
            {
                "metrics_json": json.dumps(metrics),
                "sample_png_b64": base64.b64encode(
                    sample_path.read_bytes()
                ).decode("ascii"),
            }
        )

    return {"outputs_json": json.dumps(outputs)}


@app.local_entrypoint()
def main(
    pretrained: str = "cifar10/n1024",
    classes: str = "0,1,2,3,4,5,6,7,8,9",
    samples_per_class: int = 4,
    seed: int = 42,
    num_steps: int | None = None,
    step_sweep: str = "",
    output_dir: str = "outputs/analysis/un0_reference",
) -> None:
    class_ids = [int(value) for value in classes.split(",") if value.strip()]
    if not class_ids:
        raise ValueError("classes must contain at least one class id")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_name = pretrained.replace("/", "_")
    csv_path = root / "un0_reference_runs.csv"

    if step_sweep:
        step_values = [int(value) for value in step_sweep.split(",") if value.strip()]
        if not step_values:
            raise ValueError("step_sweep must contain at least one integer")
        result = run_un0_step_sweep_remote.remote(
            pretrained=pretrained,
            classes=class_ids,
            samples_per_class=int(samples_per_class),
            seed=int(seed),
            step_values=step_values,
        )
        outputs = json.loads(result["outputs_json"])
        metrics_rows = []
        for output in outputs:
            metrics = json.loads(output["metrics_json"])
            step_count = int(metrics["num_steps"])
            stem = f"{safe_name}_seed{seed}_steps{step_count}"
            png_path = root / f"{stem}.png"
            json_path = root / f"{stem}.json"
            png_path.write_bytes(base64.b64decode(output["sample_png_b64"]))
            metrics = {**metrics, "local_sample_path": str(png_path)}
            json_path.write_text(json.dumps(metrics, indent=2))
            metrics_rows.append(metrics)

        sweep_path = root / f"{safe_name}_seed{seed}_step_sweep.csv"
        fieldnames = sorted(
            {
                key
                for row in metrics_rows
                for key in _flatten_metrics("", row).keys()
            }
        )
        with sweep_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_rows:
                writer.writerow(_flatten_metrics("", row))

        exists = csv_path.exists()
        aggregate_fieldnames = fieldnames
        if exists:
            with csv_path.open(newline="") as f:
                reader = csv.reader(f)
                aggregate_fieldnames = next(reader)
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_fieldnames)
            if not exists:
                writer.writeheader()
            for row in metrics_rows:
                writer.writerow(_flatten_metrics("", row))

        print(json.dumps(metrics_rows, indent=2))
        return

    result = run_un0_reference_remote.remote(
        pretrained=pretrained,
        classes=class_ids,
        samples_per_class=int(samples_per_class),
        seed=int(seed),
        num_steps=num_steps,
    )

    stem = f"{safe_name}_seed{seed}_steps{num_steps or 'ckpt'}"
    png_path = root / f"{stem}.png"
    json_path = root / f"{stem}.json"

    png_path.write_bytes(base64.b64decode(result["sample_png_b64"]))
    metrics = {
        **json.loads(result["metrics_json"]),
        "local_sample_path": str(png_path),
    }
    json_path.write_text(json.dumps(metrics, indent=2))

    fieldnames = sorted(_flatten_metrics("", metrics).keys())
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(_flatten_metrics("", metrics))

    print(json.dumps(metrics, indent=2))
