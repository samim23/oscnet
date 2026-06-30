"""Summarize generator quality/diversity frontier sweeps."""

from __future__ import annotations

import csv
import math
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
    re.compile(r"cifar10_rgb_coherent_drive_(.+?)_n256"),
    re.compile(r"cifar10_rgb_structured_drive_(.+?)_n256"),
    re.compile(r"cifar10_rgb_sparse_drive_seed_repeat_(.+?)_n256"),
    re.compile(r"cifar10_rgb_sparse_drive_(.+?)_n256"),
    re.compile(r"cifar10_rgb_judge_audit_(.+?)_n256"),
    re.compile(r"cifar10_rgb_semantic_feature_drift_attribution_(.+?)_n256"),
    re.compile(r"cifar10_rgb_semantic_feature_drift_(.+?)_n256"),
    re.compile(r"cifar10_rgb_coarse_to_fine_feedback_(.+?)_n256"),
    re.compile(r"cifar10_rgb_coarse_to_fine_conversion_(.+?)_n256"),
    re.compile(r"cifar10_rgb_coarse_to_fine_local_repeat_(.+?)_n256"),
    re.compile(r"cifar10_rgb_coarse_to_fine_dynamics_(.+?)_n256"),
    re.compile(r"cifar10_rgb_coarse_to_fine_(.+?)_n256"),
    re.compile(r"cifar10_rgb_multiscale_layered_(.+?)_n256"),
    re.compile(r"cifar10_rgb_normlocal_radius_(.+?)_n256"),
    re.compile(r"cifar10_rgb_normlocal_(.+?)_n256"),
    re.compile(r"cifar10_rgb_normdist_(.+?)_n256"),
    re.compile(r"cifar10_rgb_main_coupling_current_(.+?)_n256"),
    re.compile(r"cifar10_rgb_main_coupling_fine_(.+?)_n256"),
    re.compile(r"cifar10_rgb_main_coupling_strength_seed_repeat_(.+?)_n256"),
    re.compile(r"cifar10_rgb_main_coupling_strength_(.+?)_n256"),
    re.compile(r"cifar10_rgb_attractor_robustness_seed_repeat_(.+?)_n256"),
    re.compile(r"cifar10_rgb_attractor_robustness_(.+?)_n256"),
    re.compile(r"cifar10_rgb_attribution_(.+?)_n256"),
    re.compile(r"cifar10_rgb_feature_metrics_(.+?)_n256"),
    re.compile(r"cifar10_rgb_frontier_(.+?)_n256"),
    re.compile(r"cifar10_gray_convjudge_frontier_(.+?)_n256"),
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
    feature_diversity_ratio_mean: float
    feature_nearest_real_mse_mean: float
    feature_pairwise_distance_ratio_mean: float
    pixel_mean_mse_mean: float
    pixel_std_mse_mean: float
    state_energy_mean: float
    state_update_settling_ratio_mean: float
    state_acceleration_settling_ratio_mean: float
    output_step_mse_settling_ratio_mean: float
    coupling_potential_delta_mean: float
    coarse_state_energy_mean: float
    coarse_state_update_settling_ratio_mean: float
    coarse_state_acceleration_settling_ratio_mean: float
    coarse_coupling_potential_delta_mean: float
    coarse_to_fine_potential_delta_mean: float
    coarse_to_fine_profile_density_mean: float
    attractor_label_accuracy_mean: float
    attractor_class_success_fraction_mean: float
    attractor_pixel_within_class_pairwise_mse_mean: float
    attractor_pixel_separation_ratio_mean: float
    attractor_pixel_diversity_score_mean: float
    attractor_feature_within_class_distance_mean: float
    attractor_feature_separation_ratio_mean: float
    attractor_feature_diversity_score_mean: float
    quality_judge_accuracy_mean: float
    coupling_density_mean: float
    total_params_mean: float
    decoder_param_fraction_mean: float
    recurrent_op_fraction_mean: float
    trainable_recurrent_fraction_mean: float
    conditioning_target_fraction_mean: float
    conditioning_target_count_mean: float
    samples_per_second_mean: float
    distributional_weight_mean: float
    class_moment_weight_mean: float
    pareto_frontier: bool = False


@dataclass(frozen=True)
class PairedMetricSpec:
    """Metric used for same-seed paired generator attribution."""

    key: str
    label: str
    higher_is_better: bool = True


@dataclass(frozen=True)
class GeneratorPairedDeltaSummary:
    """Same-seed target-minus-baseline delta for one metric."""

    baseline_variant: str
    target_variant: str
    metric: str
    label: str
    direction: str
    paired_runs: int
    seeds: str
    baseline_mean: float
    target_mean: float
    delta_mean: float
    target_wins: int


DEFAULT_PAIRED_METRICS = (
    PairedMetricSpec("generator.classifier_label_accuracy", "generated acc", True),
    PairedMetricSpec("generator.diversity_ratio", "diversity", True),
    PairedMetricSpec("generator.nearest_real_mse", "nearest-real MSE", False),
    PairedMetricSpec(
        "generator.classifier_feature_diversity_ratio",
        "feature diversity",
        True,
    ),
    PairedMetricSpec(
        "generator.classifier_feature_nearest_real_mse",
        "feature nearest-real",
        False,
    ),
    PairedMetricSpec(
        "generator.attractor_robustness.label_accuracy",
        "attractor acc",
        True,
    ),
    PairedMetricSpec(
        "generator.attractor_robustness.pixel_attractor_diversity_score",
        "basin score",
        True,
    ),
    PairedMetricSpec(
        "generator.success_diagnostics.output_step_mse_settling_ratio",
        "output settle",
        False,
    ),
)


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


def _collapse_aware_score(accuracy: object, spread: object) -> Optional[float]:
    accuracy_value = _to_float(accuracy)
    spread_value = _to_float(spread)
    if accuracy_value is None or spread_value is None or spread_value < 0.0:
        return None
    return float(accuracy_value * math.log1p(spread_value))


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


def infer_generator_seed(run_name: str) -> Optional[int]:
    """Extract the seed from a generator run name, when present."""

    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", run_name)
    return int(match.group(1)) if match else None


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


def paired_generator_metric_deltas(
    grouped_rows: Dict[str, Sequence[dict[str, str]]],
    *,
    baseline_variant: str,
    target_variants: Sequence[str],
    metric_specs: Sequence[PairedMetricSpec] = DEFAULT_PAIRED_METRICS,
) -> List[GeneratorPairedDeltaSummary]:
    """Summarize same-seed target-minus-baseline deltas for variants.

    This is an attribution helper, not a frontier metric: it asks whether one
    mechanism improves metrics when seed/run scaffolding is paired.
    """

    baseline_rows = grouped_rows.get(baseline_variant, ())
    baseline_by_seed = {
        seed: row
        for row in baseline_rows
        if (seed := infer_generator_seed(row.get("run", ""))) is not None
    }
    summaries: List[GeneratorPairedDeltaSummary] = []
    for target_variant in target_variants:
        target_rows = grouped_rows.get(target_variant, ())
        target_by_seed = {
            seed: row
            for row in target_rows
            if (seed := infer_generator_seed(row.get("run", ""))) is not None
        }
        paired_seeds = sorted(set(baseline_by_seed).intersection(target_by_seed))
        for spec in metric_specs:
            baseline_values: List[float] = []
            target_values: List[float] = []
            seeds_with_metric: List[int] = []
            target_wins = 0
            for seed in paired_seeds:
                baseline_value = _to_float(baseline_by_seed[seed].get(spec.key, ""))
                target_value = _to_float(target_by_seed[seed].get(spec.key, ""))
                if baseline_value is None or target_value is None:
                    continue
                baseline_values.append(baseline_value)
                target_values.append(target_value)
                seeds_with_metric.append(seed)
                if spec.higher_is_better:
                    target_wins += int(target_value > baseline_value)
                else:
                    target_wins += int(target_value < baseline_value)
            if not baseline_values:
                continue
            baseline_mean = statistics.fmean(baseline_values)
            target_mean = statistics.fmean(target_values)
            summaries.append(
                GeneratorPairedDeltaSummary(
                    baseline_variant=baseline_variant,
                    target_variant=target_variant,
                    metric=spec.key,
                    label=spec.label,
                    direction="higher" if spec.higher_is_better else "lower",
                    paired_runs=len(baseline_values),
                    seeds=",".join(str(seed) for seed in seeds_with_metric),
                    baseline_mean=baseline_mean,
                    target_mean=target_mean,
                    delta_mean=target_mean - baseline_mean,
                    target_wins=target_wins,
                )
            )
    return summaries


def _summarize_variant(variant: str, rows: Sequence[dict[str, str]]) -> GeneratorFrontierSummary:
    def column(name: str) -> List[object]:
        return [row.get(name, "") for row in rows]

    def score_column(
        direct_name: str,
        accuracy_name: str,
        spread_name: str,
    ) -> List[object]:
        values: List[object] = []
        for row in rows:
            direct_value = _to_float(row.get(direct_name, ""))
            if direct_value is not None:
                values.append(direct_value)
                continue
            score = _collapse_aware_score(
                row.get(accuracy_name, ""),
                row.get(spread_name, ""),
            )
            if score is not None:
                values.append(score)
        return values

    return GeneratorFrontierSummary(
        variant=variant,
        runs=len(rows),
        generated_accuracy_mean=_mean(column("generator.classifier_label_accuracy")),
        generated_accuracy_std=_pstdev(column("generator.classifier_label_accuracy")),
        diversity_ratio_mean=_mean(column("generator.diversity_ratio")),
        diversity_ratio_std=_pstdev(column("generator.diversity_ratio")),
        nearest_real_mse_mean=_mean(column("generator.nearest_real_mse")),
        nearest_real_mse_std=_pstdev(column("generator.nearest_real_mse")),
        feature_diversity_ratio_mean=_mean(
            column("generator.classifier_feature_diversity_ratio")
        ),
        feature_nearest_real_mse_mean=_mean(
            column("generator.classifier_feature_nearest_real_mse")
        ),
        feature_pairwise_distance_ratio_mean=_mean(
            column("generator.classifier_feature_pairwise_distance_ratio")
        ),
        pixel_mean_mse_mean=_mean(column("generator.pixel_mean_mse")),
        pixel_std_mse_mean=_mean(column("generator.pixel_std_mse")),
        state_energy_mean=_mean(
            column("generator.success_diagnostics.state_final_energy")
        ),
        state_update_settling_ratio_mean=_mean(
            column("generator.success_diagnostics.state_update_rms_settling_ratio")
        ),
        state_acceleration_settling_ratio_mean=_mean(
            column(
                "generator.success_diagnostics.state_acceleration_rms_settling_ratio"
            )
        ),
        output_step_mse_settling_ratio_mean=_mean(
            column("generator.success_diagnostics.output_step_mse_settling_ratio")
        ),
        coupling_potential_delta_mean=_mean(
            column("generator.success_diagnostics.coupling_potential_proxy_delta")
        ),
        coarse_state_energy_mean=_mean(
            column("generator.success_diagnostics.coarse_state_energy_final")
        ),
        coarse_state_update_settling_ratio_mean=_mean(
            column(
                "generator.success_diagnostics."
                "coarse_state_update_rms_settling_ratio"
            )
        ),
        coarse_state_acceleration_settling_ratio_mean=_mean(
            column(
                "generator.success_diagnostics."
                "coarse_state_acceleration_rms_settling_ratio"
            )
        ),
        coarse_coupling_potential_delta_mean=_mean(
            column(
                "generator.success_diagnostics."
                "coarse_coupling_potential_proxy_delta"
            )
        ),
        coarse_to_fine_potential_delta_mean=_mean(
            column(
                "generator.success_diagnostics."
                "coarse_to_fine_potential_proxy_delta"
            )
        ),
        coarse_to_fine_profile_density_mean=_mean(
            column("generator.success_diagnostics.coarse_to_fine_profile_density")
        ),
        attractor_label_accuracy_mean=_mean(
            column("generator.attractor_robustness.label_accuracy")
        ),
        attractor_class_success_fraction_mean=_mean(
            column("generator.attractor_robustness.class_success_fraction")
        ),
        attractor_pixel_within_class_pairwise_mse_mean=_mean(
            column("generator.attractor_robustness.pixel_within_class_pairwise_mse")
        ),
        attractor_pixel_separation_ratio_mean=_mean(
            column("generator.attractor_robustness.pixel_separation_ratio")
        ),
        attractor_pixel_diversity_score_mean=_mean(
            score_column(
                "generator.attractor_robustness."
                "pixel_attractor_diversity_score",
                "generator.attractor_robustness.label_accuracy",
                "generator.attractor_robustness.pixel_within_class_pairwise_mse",
            )
        ),
        attractor_feature_within_class_distance_mean=_mean(
            column(
                "generator.attractor_robustness."
                "feature_within_class_pairwise_distance"
            )
        ),
        attractor_feature_separation_ratio_mean=_mean(
            column("generator.attractor_robustness.feature_separation_ratio")
        ),
        attractor_feature_diversity_score_mean=_mean(
            score_column(
                "generator.attractor_robustness."
                "feature_attractor_diversity_score",
                "generator.attractor_robustness.label_accuracy",
                "generator.attractor_robustness."
                "feature_within_class_pairwise_distance",
            )
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
        conditioning_target_fraction_mean=_mean(
            column("generator.success_diagnostics.conditioning_target_effective_fraction")
        ),
        conditioning_target_count_mean=_mean(
            column("generator.success_diagnostics.conditioning_target_count")
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
    include_feature_metrics = any(
        summary.feature_diversity_ratio_mean == summary.feature_diversity_ratio_mean
        or summary.feature_nearest_real_mse_mean
        == summary.feature_nearest_real_mse_mean
        for summary in summaries
    )
    include_attractor_metrics = any(
        summary.attractor_label_accuracy_mean == summary.attractor_label_accuracy_mean
        or summary.attractor_pixel_within_class_pairwise_mse_mean
        == summary.attractor_pixel_within_class_pairwise_mse_mean
        for summary in summaries
    )
    include_coarse_metrics = any(
        summary.coarse_state_energy_mean == summary.coarse_state_energy_mean
        or summary.coarse_to_fine_potential_delta_mean
        == summary.coarse_to_fine_potential_delta_mean
        for summary in summaries
    )
    header = "| Variant | Runs | Frontier | Acc | Diversity | Nearest-real MSE | "
    separator = "| --- | ---: | :---: | ---: | ---: | ---: | "
    if include_feature_metrics:
        header += "Feature diversity | Feature nearest-real | "
        separator += "---: | ---: | "
    if include_attractor_metrics:
        header += "Attractor acc | Basin score | Attractor within | Attractor sep | "
        separator += "---: | ---: | ---: | ---: | "
    header += (
        "State energy | Update settle | Output settle | Coupling delta | "
        "Coupling density | "
    )
    separator += "---: | ---: | ---: | ---: | ---: | "
    if include_coarse_metrics:
        header += (
            "Coarse energy | Coarse update settle | Coarse coupling delta | "
            "C2F delta | C2F density | "
        )
        separator += "---: | ---: | ---: | ---: | ---: | "
    header += "Drive frac | Drive count | Params | Samples/sec |"
    separator += "---: | ---: | ---: | ---: |"
    lines = [
        f"# {title}",
        "",
        header,
        separator,
    ]
    for summary in summaries:
        row = [
            summary.variant,
            str(summary.runs),
            "yes" if summary.pareto_frontier else "",
            _fmt(summary.generated_accuracy_mean),
            _fmt(summary.diversity_ratio_mean),
            _fmt(summary.nearest_real_mse_mean),
        ]
        if include_feature_metrics:
            row.extend(
                [
                    _fmt(summary.feature_diversity_ratio_mean),
                    _fmt(summary.feature_nearest_real_mse_mean),
                ]
            )
        if include_attractor_metrics:
            row.extend(
                [
                    _fmt(summary.attractor_label_accuracy_mean),
                    _fmt(summary.attractor_pixel_diversity_score_mean),
                    _fmt(summary.attractor_pixel_within_class_pairwise_mse_mean),
                    _fmt(summary.attractor_pixel_separation_ratio_mean),
                ]
            )
        row.extend(
            [
                _fmt(summary.state_energy_mean),
                _fmt(summary.state_update_settling_ratio_mean),
                _fmt(summary.output_step_mse_settling_ratio_mean),
                _fmt(summary.coupling_potential_delta_mean),
                _fmt(summary.coupling_density_mean),
            ]
        )
        if include_coarse_metrics:
            row.extend(
                [
                    _fmt(summary.coarse_state_energy_mean),
                    _fmt(summary.coarse_state_update_settling_ratio_mean),
                    _fmt(summary.coarse_coupling_potential_delta_mean),
                    _fmt(summary.coarse_to_fine_potential_delta_mean),
                    _fmt(summary.coarse_to_fine_profile_density_mean),
                ]
            )
        row.extend(
            [
                _fmt(summary.conditioning_target_fraction_mean),
                _fmt(summary.conditioning_target_count_mean, digits=0),
                _fmt(summary.total_params_mean, digits=0),
                _fmt(summary.samples_per_second_mean),
            ]
        )
        lines.append(
            "| "
            + " | ".join(row)
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


def write_paired_delta_csv(
    summaries: Sequence[GeneratorPairedDeltaSummary],
    path: str | Path,
) -> None:
    """Write same-seed paired metric deltas to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(GeneratorPairedDeltaSummary)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({name: getattr(summary, name) for name in names})


def write_paired_delta_markdown(
    summaries: Sequence[GeneratorPairedDeltaSummary],
    path: str | Path,
    *,
    title: str = "Generator Paired Deltas",
) -> None:
    """Write same-seed paired metric deltas to Markdown."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "Deltas are `target - baseline` on matched seeds. A positive delta is "
        "good only when the direction column says `higher`; for `lower` metrics, "
        "a negative delta is good.",
        "",
        "| Target | Baseline | Metric | Direction | Pairs | Seeds | Baseline | Target | Delta | Target wins |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.target_variant,
                    summary.baseline_variant,
                    summary.label,
                    summary.direction,
                    str(summary.paired_runs),
                    summary.seeds,
                    _fmt(summary.baseline_mean),
                    _fmt(summary.target_mean),
                    _fmt(summary.delta_mean),
                    str(summary.target_wins),
                ]
            )
            + " |"
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
