"""Create same-seed paired attribution deltas for generator sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from oscnet.analysis.generator_frontier import (
    paired_generator_metric_deltas,
    read_generator_sweep_csv,
    write_paired_delta_csv,
    write_paired_delta_markdown,
)


DEFAULT_INPUT = (
    "outputs/analysis/"
    "modal_mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat.csv"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=DEFAULT_INPUT,
        help="Input Modal sweep CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/cifar10_rgb_coarse_to_fine_local_repeat",
        help="Directory for paired-delta CSV/Markdown artifacts.",
    )
    parser.add_argument(
        "--variant-regex",
        default=None,
        help=(
            "Optional regex used to extract a variant label from the run name. "
            "If it has a capture group, group 1 is used."
        ),
    )
    parser.add_argument(
        "--baseline-variant",
        default="coarse16_c2f000",
        help="Variant used as the same-seed baseline.",
    )
    parser.add_argument(
        "--target-variant",
        action="append",
        default=None,
        help=(
            "Variant to compare against the baseline. Repeat this flag for "
            "multiple targets. Defaults to the local C2F and plain HORN rows "
            "from the CIFAR RGB local-repeat sweep."
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help=(
            "Matched comparison as 'baseline:target'. Repeat for multiple "
            "pairs. When provided, this overrides --baseline-variant and "
            "--target-variant."
        ),
    )
    parser.add_argument(
        "--title",
        default="CIFAR-10 RGB coarse-to-fine paired attribution deltas",
        help="Title for the Markdown report.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    grouped = read_generator_sweep_csv(args.csv, variant_regex=args.variant_regex)
    if args.pair:
        summaries = []
        for pair in args.pair:
            if ":" not in pair:
                raise SystemExit("--pair must be formatted as 'baseline:target'")
            baseline_variant, target_variant = pair.split(":", 1)
            summaries.extend(
                paired_generator_metric_deltas(
                    grouped,
                    baseline_variant=baseline_variant,
                    target_variants=[target_variant],
                )
            )
    else:
        target_variants = args.target_variant or [
            "coarse16_c2f025_local050",
            "horn_normlocal",
        ]
        summaries = paired_generator_metric_deltas(
            grouped,
            baseline_variant=args.baseline_variant,
            target_variants=target_variants,
        )

    csv_path = output_dir / "paired_deltas.csv"
    md_path = output_dir / "paired_deltas.md"
    write_paired_delta_csv(summaries, csv_path)
    write_paired_delta_markdown(summaries, md_path, title=args.title)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
