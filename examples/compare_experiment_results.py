"""Compare saved OscNet experiment summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from oscnet.experiments.results import (
    DEFAULT_RESULT_METRICS,
    collect_experiment_summaries,
    find_experiment_runs,
    format_comparison_table,
    write_comparison_csv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare saved experiment metrics from summary.json files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Run directories, metrics directories, or summary.json files.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/reference"),
        help="Root used with --pattern when explicit paths are omitted.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Run-name glob under --root when explicit paths are omitted.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric key to include. Supports dotted keys such as quality.mae.",
    )
    parser.add_argument("--sort-by", default="final_eval_loss")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    metric_names = tuple(args.metrics) if args.metrics else DEFAULT_RESULT_METRICS
    paths = args.paths or find_experiment_runs(args.root, args.pattern)
    rows = collect_experiment_summaries(
        paths,
        metric_names=metric_names,
        sort_by=args.sort_by,
        descending=args.descending,
    )
    print(format_comparison_table(rows, metric_names=metric_names))

    if args.csv is not None:
        write_comparison_csv(rows, args.csv, metric_names=metric_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
