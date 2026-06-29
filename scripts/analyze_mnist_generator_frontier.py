"""Create quality/diversity frontier artifacts for MNIST generator sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from oscnet.analysis.generator_frontier import (
    plot_frontier,
    read_generator_sweep_csv,
    summarize_generator_frontier,
    write_frontier_csv,
    write_frontier_markdown,
)


DEFAULT_INPUT = (
    "outputs/analysis/"
    "modal_mnist_generator_sparse_horn_state_mlp_strength8_diversity_probe.csv"
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
        default="outputs/analysis/mnist_generator_frontier",
        help="Directory for frontier CSV/Markdown/PNG artifacts.",
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
        "--title",
        default="MNIST generator quality/diversity frontier",
        help="Title for the Markdown report and plot.",
    )
    parser.add_argument(
        "--accuracy-floor",
        type=float,
        default=0.99,
        help=(
            "Generated-label accuracy required for frontier marking. "
            "Set to 0 to include every variant."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    grouped = read_generator_sweep_csv(args.csv, variant_regex=args.variant_regex)
    summaries = summarize_generator_frontier(grouped, accuracy_floor=args.accuracy_floor)

    csv_path = output_dir / "frontier_summary.csv"
    md_path = output_dir / "frontier_summary.md"
    png_path = output_dir / "frontier_plot.png"

    write_frontier_csv(summaries, csv_path)
    write_frontier_markdown(
        summaries,
        md_path,
        title=args.title,
        accuracy_floor=args.accuracy_floor,
    )
    if not args.no_plot:
        plot_frontier(summaries, png_path, title=args.title)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if not args.no_plot:
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
