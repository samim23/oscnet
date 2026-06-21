"""Utilities for comparing saved experiment result summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_RESULT_METRICS = (
    "final_eval_loss",
    "best_loss",
    "best_epoch",
    "final_epoch",
    "quality.pixel_correlation",
    "quality.foreground_f1",
    "quality.diversity_ratio",
    "quality.mae",
)


@dataclass(frozen=True)
class ExperimentSummaryRow:
    """Flattened metric row for one saved experiment run."""

    run: str
    root: Path
    metrics: Dict[str, Any]

    def as_dict(self, metric_names: Sequence[str]) -> Dict[str, Any]:
        return {
            "run": self.run,
            "root": str(self.root),
            **{name: self.metrics.get(name) for name in metric_names},
        }


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _summary_path(path: Path) -> Path:
    if path.name == "summary.json":
        return path
    if path.name == "metrics":
        return path / "summary.json"
    return path / "metrics" / "summary.json"


def _run_root(path: Path) -> Path:
    if path.name == "summary.json":
        return path.parent.parent
    if path.name == "metrics":
        return path.parent
    return path


def _get_nested(mapping: Dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def find_experiment_runs(root: Path, pattern: str = "*") -> List[Path]:
    """Find run directories below ``root`` that contain a metrics summary."""

    root = Path(root)
    runs = []
    for summary_path in sorted(root.glob(f"{pattern}/metrics/summary.json")):
        runs.append(summary_path.parent.parent)
    return runs


def load_experiment_summary(
    path: Path,
    metric_names: Sequence[str] = DEFAULT_RESULT_METRICS,
) -> ExperimentSummaryRow:
    """Load selected metrics from one experiment run or summary file."""

    path = Path(path)
    summary_path = _summary_path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f"missing experiment summary: {summary_path}")

    root = _run_root(path)
    summary = _read_json(summary_path)
    metrics = {name: _get_nested(summary, name) for name in metric_names}
    return ExperimentSummaryRow(run=root.name, root=root, metrics=metrics)


def collect_experiment_summaries(
    paths: Iterable[Path],
    metric_names: Sequence[str] = DEFAULT_RESULT_METRICS,
    sort_by: Optional[str] = "final_eval_loss",
    descending: bool = False,
) -> List[ExperimentSummaryRow]:
    """Load and optionally sort summary rows from multiple experiment paths."""

    rows = [load_experiment_summary(Path(path), metric_names) for path in paths]
    if sort_by is None:
        return rows
    if sort_by not in metric_names:
        raise ValueError(f"sort_by must be one of {list(metric_names)}")
    present = [row for row in rows if row.metrics.get(sort_by) is not None]
    missing = [row for row in rows if row.metrics.get(sort_by) is None]
    present.sort(key=lambda row: row.metrics.get(sort_by), reverse=descending)
    return present + missing


def write_comparison_csv(
    rows: Sequence[ExperimentSummaryRow],
    path: Path,
    metric_names: Sequence[str] = DEFAULT_RESULT_METRICS,
) -> None:
    """Write comparison rows to a CSV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "root", *metric_names]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict(metric_names))


def format_comparison_table(
    rows: Sequence[ExperimentSummaryRow],
    metric_names: Sequence[str] = DEFAULT_RESULT_METRICS,
) -> str:
    """Format comparison rows as a compact Markdown table."""

    headers = ["run", *metric_names]
    table_rows = [
        [row.run, *[row.metrics.get(name) for name in metric_names]]
        for row in rows
    ]

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    rendered = ["| " + " | ".join(headers) + " |"]
    rendered.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in table_rows:
        rendered.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(rendered)


__all__ = [
    "DEFAULT_RESULT_METRICS",
    "ExperimentSummaryRow",
    "collect_experiment_summaries",
    "find_experiment_runs",
    "format_comparison_table",
    "load_experiment_summary",
    "write_comparison_csv",
]
