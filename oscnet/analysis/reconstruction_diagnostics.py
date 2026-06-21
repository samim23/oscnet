"""Diagnostics for saved reconstruction artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class ReconstructionArtifactSummary:
    """Compact metrics from a saved reconstruction artifact."""

    run: str
    artifact_path: str
    n_examples: int
    changed_fraction: float
    mse: float
    clipped_mse: float
    mae: float
    input_mse: float
    changed_mse: float
    changed_input_mse: float
    changed_mae: float
    unchanged_mse: float
    changed_improvement: float
    output_mean: float
    output_std: float
    output_min: float
    output_max: float
    target_mean: float
    changed_target_mean: float
    changed_output_mean: float


@dataclass(frozen=True)
class RunDiagnosticSummary:
    """Artifact diagnostics plus optional training summary fields."""

    run: str
    artifact_path: str
    final_eval_loss: Optional[float]
    best_loss: Optional[float]
    max_grad_norm: Optional[float]
    final_grad_norm: Optional[float]
    changed_fraction: float
    changed_mse: float
    changed_input_mse: float
    changed_improvement: float
    unchanged_mse: float
    mse: float
    clipped_mse: float
    output_std: float
    changed_target_mean: float
    changed_output_mean: float


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_max(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.max(np.asarray(values, dtype=np.float64)))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def infer_changed_mask(
    inputs: np.ndarray,
    originals: np.ndarray,
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    """Infer pixels modified by corruption from saved input/target arrays."""

    if inputs.shape != originals.shape:
        raise ValueError("inputs and originals must have the same shape")
    return np.abs(np.asarray(inputs) - np.asarray(originals)) > atol


def summarize_reconstruction_artifact(
    artifact_path: Path | str,
    *,
    run: Optional[str] = None,
    change_atol: float = 1e-6,
) -> ReconstructionArtifactSummary:
    """Summarize full and corruption-changed-region reconstruction errors."""

    artifact_path = Path(artifact_path)
    with np.load(artifact_path) as artifact:
        inputs = np.asarray(artifact["inputs"], dtype=np.float32)
        originals = np.asarray(artifact["originals"], dtype=np.float32)
        reconstructions = np.asarray(artifact["reconstructions"], dtype=np.float32)

    if inputs.shape != originals.shape or originals.shape != reconstructions.shape:
        raise ValueError("inputs, originals, and reconstructions must have matching shapes")

    changed = infer_changed_mask(inputs, originals, atol=change_atol)
    unchanged = ~changed
    error = reconstructions - originals
    input_error = inputs - originals
    clipped = np.clip(reconstructions, 0.0, 1.0)

    changed_input_mse = _safe_mean(np.square(input_error[changed]))
    changed_mse = _safe_mean(np.square(error[changed]))
    changed_improvement = changed_input_mse - changed_mse

    return ReconstructionArtifactSummary(
        run=run or artifact_path.parent.parent.name,
        artifact_path=str(artifact_path),
        n_examples=int(originals.shape[0]),
        changed_fraction=float(np.mean(changed)),
        mse=float(np.mean(np.square(error))),
        clipped_mse=float(np.mean(np.square(clipped - originals))),
        mae=float(np.mean(np.abs(error))),
        input_mse=float(np.mean(np.square(input_error))),
        changed_mse=changed_mse,
        changed_input_mse=changed_input_mse,
        changed_mae=_safe_mean(np.abs(error[changed])),
        unchanged_mse=_safe_mean(np.square(error[unchanged])),
        changed_improvement=changed_improvement,
        output_mean=float(np.mean(reconstructions)),
        output_std=float(np.std(reconstructions)),
        output_min=float(np.min(reconstructions)),
        output_max=float(np.max(reconstructions)),
        target_mean=float(np.mean(originals)),
        changed_target_mean=_safe_mean(originals[changed]),
        changed_output_mean=_safe_mean(reconstructions[changed]),
    )


def latest_reconstruction_artifact(run_root: Path | str) -> Path:
    """Return the latest saved MNIST reconstruction artifact under a run root."""

    run_root = Path(run_root)
    artifact_dir = run_root / "artifacts"
    matches = sorted(artifact_dir.glob("mnist_reconstructions_epoch_*.npz"))
    if not matches:
        raise FileNotFoundError(f"no MNIST reconstruction artifacts under {artifact_dir}")
    return matches[-1]


def summarize_run_diagnostics(
    run_root: Path | str,
    *,
    artifact_path: Optional[Path | str] = None,
    change_atol: float = 1e-6,
) -> RunDiagnosticSummary:
    """Summarize artifact and optional summary/history metrics for one run."""

    run_root = Path(run_root)
    artifact = (
        Path(artifact_path)
        if artifact_path is not None
        else latest_reconstruction_artifact(run_root)
    )
    artifact_summary = summarize_reconstruction_artifact(
        artifact,
        run=run_root.name,
        change_atol=change_atol,
    )
    summary = _load_json(run_root / "metrics" / "summary.json")
    history = _load_json(run_root / "metrics" / "history.json")
    grad_norms = [
        float(value)
        for value in history.get("grad_norm", [])
        if value is not None and np.isfinite(float(value))
    ]

    return RunDiagnosticSummary(
        run=artifact_summary.run,
        artifact_path=artifact_summary.artifact_path,
        final_eval_loss=summary.get("final_eval_loss"),
        best_loss=summary.get("best_loss"),
        max_grad_norm=_safe_max(grad_norms),
        final_grad_norm=grad_norms[-1] if grad_norms else None,
        changed_fraction=artifact_summary.changed_fraction,
        changed_mse=artifact_summary.changed_mse,
        changed_input_mse=artifact_summary.changed_input_mse,
        changed_improvement=artifact_summary.changed_improvement,
        unchanged_mse=artifact_summary.unchanged_mse,
        mse=artifact_summary.mse,
        clipped_mse=artifact_summary.clipped_mse,
        output_std=artifact_summary.output_std,
        changed_target_mean=artifact_summary.changed_target_mean,
        changed_output_mean=artifact_summary.changed_output_mean,
    )


def write_run_diagnostics_csv(
    rows: Iterable[RunDiagnosticSummary],
    path: Path | str,
) -> None:
    """Write run diagnostic summaries to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(RunDiagnosticSummary.__dataclass_fields__)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


__all__ = [
    "ReconstructionArtifactSummary",
    "RunDiagnosticSummary",
    "infer_changed_mask",
    "latest_reconstruction_artifact",
    "summarize_reconstruction_artifact",
    "summarize_run_diagnostics",
    "write_run_diagnostics_csv",
]
