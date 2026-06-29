"""Summarize generator quality/diversity frontier sweeps."""

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt


DEFAULT_VARIANT_PATTERNS = (
    re.compile(r"fashion_mnist_frontier_(.+?)_n196"),
    re.compile(r"fashion_mnist_readout_capacity_(.+?)_n196"),
    re.compile(r"fashion_mnist_horn_calibration_(.+?)_n196"),
    re.compile(r"cifar10_gray_frontier_(.+?)_n256"),
    re.compile(r"state_mlp_strength8_diversity_(.+?)_n196"),
    re.compile(r"recommended_ablation_(.+?)_n196"),
    re.compile(r"damping_distributional_(.+?)_n196"),
    re.compile(r"state_mlp_diversity_(.+?)_n196"),
)


@dataclass
class GeneratorFrontierSummary:
    """Aggregate metrics for one generator sweep variant."""

    variant: str
    runs: int
    generated_accuracy_mean: float
    generated_accuracy_std: float
    diversity_ratio_mean: float
    diversity_ratio_std: float
    nearest_real_mse_mean: float
    nearest_real_mse_std: float
    pixel_mean_mse_mean: float
    pixel_std_mse_mean: float
    state_energy_mean: float
    quality_judge_accuracy_mean: float
    coupling_density_mean: float
    total_params_mean: float
    decoder_param_fraction_mean: float
    recurrent_op_fraction_mean: float
    trainable_recurrent_fraction_mean: float
    samples_per_second_mean: float
    distributional_weight_mean: float
    class_moment_weight_mean: float
    pareto_frontier: bool = False


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text == "True":
        return 1.0
    if text == "False":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def _mean(values: Iterable[object]) -> float:
    nums = [value for value in (_to_float(item) for item in values) if value is not None]
    return statistics.fmean(nums) if nums else float("nan")


def _pstdev(values: Iterable[object]) -> float:
    nums = [value for value in (_to_float(item) for item in values) if value is not None]
    if len(nums) < 2:
        return 0.0
    return statistics.pstdev(nums)


def infer_generator_variant(run_name: str, variant_regex: str | None = None) -> str:
    """Infer a compact variant label from a Modal generator run name."""

    if variant_regex:
        match = re.search(variant_regex, run_name)
        if match:
            return match.group(1) if match.groups() else match.group(0)

    for pattern in DEFAULT_VARIANT_PATTERNS:
        match = pattern.search(run_name)
        if match:
            return match.group(1)

    compact = re.sub(r"_seed\d+.*$", "", run_name)
    compact = re.sub(r"^mnist_generator_sparse_horn_", "", compact)
    return compact


def read_generator_sweep_csv(
    csv_path: str | Path,
    *,
    variant_regex: str | None = None,
) -> Dict[str, List[dict[str, str]]]:
    """Load a generator sweep CSV and group rows by inferred variant."""

    grouped: Dict[str, List[dict[str, str]]] = {}
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_name = row.get("run", "")
            variant = infer_generator_variant(run_name, variant_regex)
            grouped.setdefault(variant, []).append(row)
    return grouped


def _summarize_variant(variant: str, rows: Sequence[dict[str, str]]) -> GeneratorFrontierSummary:
    def column(name: str) -> List[object]:
        return [row.get(name, "") for row in rows]

    return GeneratorFrontierSummary(
        variant=variant,
        runs=len(rows),
        generated_accuracy_mean=_mean(column("generator.classifier_label_accuracy")),
        generated_accuracy_std=_pstdev(column("generator.classifier_label_accuracy")),
        diversity_ratio_mean=_mean(column("generator.diversity_ratio")),
        diversity_ratio_std=_pstdev(column("generator.diversity_ratio")),
        nearest_real_mse_mean=_mean(column("generator.nearest_real_mse")),
        nearest_real_mse_std=_pstdev(column("generator.nearest_real_mse")),
        pixel_mean_mse_mean=_mean(column("generator.pixel_mean_mse")),
        pixel_std_mse_mean=_mean(column("generator.pixel_std_mse")),
        state_energy_mean=_mean(
            column("generator.success_diagnostics.state_final_energy")
        ),
        quality_judge_accuracy_mean=_mean(
            column("generator.quality_classifier.final_eval_accuracy")
        ),
        coupling_density_mean=_mean(
            column("generator.success_diagnostics.coupling_density")
        ),
        total_params_mean=_mean(column("generator.success_diagnostics.total_params")),
        decoder_param_fraction_mean=_mean(
            column("generator.success_diagnostics.decoder_param_fraction")
        ),
        recurrent_op_fraction_mean=_mean(
            column("generator.success_diagnostics.estimated_recurrent_op_fraction")
        ),
        trainable_recurrent_fraction_mean=_mean(
            column(
                "generator.success_diagnostics.trainable_recurrent_param_fraction"
            )
        ),
        samples_per_second_mean=_mean(
            column("generator.success_diagnostics.samples_per_train_second")
        ),
        distributional_weight_mean=_mean(column("generator.distributional_weight")),
        class_moment_weight_mean=_mean(column("generator.class_moment_weight")),
    )


def summarize_generator_frontier(
    grouped_rows: Dict[str, Sequence[dict[str, str]]],
    *,
    accuracy_floor: float = 0.0,
) -> List[GeneratorFrontierSummary]:
    """Aggregate rows and tag quality/diversity Pareto-frontier variants."""

    summaries = [
        _summarize_variant(variant, rows)
        for variant, rows in sorted(grouped_rows.items())
        if rows
    ]
    for summary in summaries:
        summary.pareto_frontier = _is_frontier_member(
            summary, summaries, accuracy_floor=accuracy_floor
        )
    return summaries


def _is_frontier_member(
    candidate: GeneratorFrontierSummary,
    summaries: Sequence[GeneratorFrontierSummary],
    *,
    accuracy_floor: float = 0.0,
) -> bool:
    if candidate.generated_accuracy_mean < accuracy_floor:
        return False
    for other in summaries:
        if other is candidate:
            continue
        if other.generated_accuracy_mean < accuracy_floor:
            continue
        no_worse = (
            other.generated_accuracy_mean >= candidate.generated_accuracy_mean
            and other.diversity_ratio_mean >= candidate.diversity_ratio_mean
            and other.nearest_real_mse_mean <= candidate.nearest_real_mse_mean
        )
        strictly_better = (
            other.generated_accuracy_mean > candidate.generated_accuracy_mean
            or other.diversity_ratio_mean > candidate.diversity_ratio_mean
            or other.nearest_real_mse_mean < candidate.nearest_real_mse_mean
        )
        if no_worse and strictly_better:
            return False
    return True


def write_frontier_csv(
    summaries: Sequence[GeneratorFrontierSummary], path: str | Path
) -> None:
    """Write frontier summaries to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(GeneratorFrontierSummary)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({name: getattr(summary, name) for name in names})


def write_frontier_markdown(
    summaries: Sequence[GeneratorFrontierSummary],
    path: str | Path,
    *,
    title: str = "MNIST Generator Frontier",
    accuracy_floor: float = 0.0,
) -> None:
    """Write a compact markdown table for reports and README links."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "| Variant | Runs | Frontier | Acc | Diversity | Nearest-real MSE | "
        "State energy | Coupling density | Params | Samples/sec |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.variant,
                    str(summary.runs),
                    "yes" if summary.pareto_frontier else "",
                    _fmt(summary.generated_accuracy_mean),
                    _fmt(summary.diversity_ratio_mean),
                    _fmt(summary.nearest_real_mse_mean),
                    _fmt(summary.state_energy_mean),
                    _fmt(summary.coupling_density_mean),
                    _fmt(summary.total_params_mean, digits=0),
                    _fmt(summary.samples_per_second_mean),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Frontier variants are non-dominated over generated-label accuracy "
            "(higher is better), diversity ratio (higher is better), and "
            "nearest-real MSE (lower is better).",
        ]
    )
    if accuracy_floor > 0.0:
        lines.append(
            f"Variants below generated-label accuracy `{accuracy_floor:.3f}` "
            "are excluded from frontier marking."
        )
    path.write_text("\n".join(lines) + "\n")


def plot_frontier(
    summaries: Sequence[GeneratorFrontierSummary],
    path: str | Path,
    *,
    title: str = "MNIST generator quality/diversity frontier",
) -> None:
    """Plot diversity versus nearest-real MSE with labels."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for summary in summaries:
        marker = "*" if summary.pareto_frontier else "o"
        size = 220 if summary.pareto_frontier else 120
        color = summary.generated_accuracy_mean
        alpha = 1.0 if summary.pareto_frontier else 0.55
        ax.scatter(
            summary.nearest_real_mse_mean,
            summary.diversity_ratio_mean,
            c=[color],
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            s=size,
            marker=marker,
            edgecolors="black",
            linewidths=0.8,
            alpha=alpha,
        )
        if summary.pareto_frontier:
            offset = (-118, 8) if summary.variant.startswith("horn") else (8, 8)
            ax.annotate(
                summary.variant,
                (summary.nearest_real_mse_mean, summary.diversity_ratio_mean),
                textcoords="offset points",
                xytext=offset,
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75},
            )
    ax.set_title(title)
    ax.set_xlabel("nearest-real MSE (lower is closer to training examples)")
    ax.set_ylabel("diversity ratio (higher is more varied)")
    ax.grid(alpha=0.25)
    if summaries:
        xs = [summary.nearest_real_mse_mean for summary in summaries]
        ys = [summary.diversity_ratio_mean for summary in summaries]
        x_margin = max((max(xs) - min(xs)) * 0.18, 0.001)
        y_margin = max((max(ys) - min(ys)) * 0.18, 0.02)
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_clim(0.0, 1.0)
    fig.colorbar(sm, ax=ax, label="generated-label accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _fmt(value: float, *, digits: int = 4) -> str:
    if value != value:
        return ""
    if digits == 0:
        return str(int(round(value)))
    return f"{value:.{digits}f}"
