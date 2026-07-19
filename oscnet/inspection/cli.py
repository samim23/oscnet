"""CLI for oscillatory trace inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .report import inspect_run_traces, inspect_trace
from .views import DEFAULT_VIEW_NAMES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an OscNet NPZ trace: coupling, phase fields, synchrony, "
            "omega, and readout panels."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Trace NPZ file, or an experiment run directory containing traces/.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for inspection artifacts (default: <path>_inspect or run/inspect).",
    )
    parser.add_argument(
        "--family",
        choices=("generator", "winfree", "phase_flow", "generic"),
        help="Force adapter family instead of auto-detecting.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=sorted(DEFAULT_VIEW_NAMES),
        help="Subset of views to render (default: all).",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Batch item for field movies (default: 0).",
    )
    parser.add_argument(
        "--grid-shape",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Optional spatial layout for flat oscillator sites.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Max keyframes in trajectory strips (default: 8).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="PNG DPI (default: 120).",
    )
    parser.add_argument(
        "--winfree-prefix",
        default="decoder",
        help="Winfree trajectory prefix (default: decoder).",
    )
    parser.add_argument(
        "--all-traces",
        action="store_true",
        help="When path is a run dir, inspect every traces/*.npz (default: latest only).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip writing the interactive index.html (PNG/JSON only).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open index.html in the default browser when written.",
    )
    return parser


def _default_output_dir(path: Path) -> Path:
    if path.is_dir():
        return path / "inspect"
    return path.with_name(path.stem + "_inspect")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    path: Path = args.path
    output_dir = args.output_dir or _default_output_dir(path)
    grid_shape: Optional[Tuple[int, int]] = (
        (int(args.grid_shape[0]), int(args.grid_shape[1]))
        if args.grid_shape is not None
        else None
    )
    common = dict(
        views=args.views,
        family=args.family,
        batch_index=args.batch_index,
        grid_shape=grid_shape,
        max_frames=args.max_frames,
        dpi=args.dpi,
        winfree_prefix=args.winfree_prefix,
        html=not args.no_html,
    )

    if path.is_dir():
        reports = inspect_run_traces(
            path,
            output_dir,
            latest_only=not args.all_traces,
            **common,
        )
        print(f"inspected {len(reports)} trace(s) → {output_dir}")
        for report in reports:
            print(
                f"  [{report.family}] {report.source.name}: "
                f"{len(report.artifacts)} artifacts, {len(report.skipped)} skipped"
            )
            if report.html_path is not None:
                print(f"    html: {report.html_path}")
        if args.open:
            for report in reports:
                _open_html(report.html_path)
        return 0

    if not path.is_file():
        parser.error(f"path not found: {path}")

    report = inspect_trace(path, output_dir, **common)
    print(f"family: {report.family}")
    print(f"output: {report.output_dir}")
    if report.html_path is not None:
        print(f"html: {report.html_path}")
    print(f"artifacts ({len(report.artifacts)}):")
    for artifact in report.artifacts:
        print(f"  {artifact}")
    if report.skipped:
        print(f"skipped ({len(report.skipped)}):")
        for item in report.skipped:
            print(f"  {item['view']}: {item['reason']}")
    if args.open:
        _open_html(report.html_path)
    return 0


def _open_html(path: Optional[Path]) -> None:
    if path is None or not path.is_file():
        return
    import webbrowser

    webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    raise SystemExit(main())
