"""Analyze saved MNIST reconstruction artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from oscnet.analysis import (
    summarize_run_diagnostics,
    write_run_diagnostics_csv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize MNIST reconstruction artifacts by changed-region error."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Experiment run roots containing artifacts/ and metrics/ directories.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--change-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for inferring changed pixels.",
    )
    return parser


def _format(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rows = [
        summarize_run_diagnostics(run, change_atol=args.change_atol)
        for run in args.runs
    ]
    rows.sort(key=lambda row: row.changed_mse)

    columns = [
        "run",
        "final_eval_loss",
        "changed_mse",
        "changed_input_mse",
        "changed_improvement",
        "unchanged_mse",
        "output_std",
        "max_grad_norm",
    ]
    widths = {
        column: max(
            len(column),
            *(len(_format(asdict(row)[column])) for row in rows),
        )
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        data = asdict(row)
        print(
            "  ".join(
                _format(data[column]).ljust(widths[column])
                for column in columns
            )
        )

    if args.csv is not None:
        write_run_diagnostics_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
