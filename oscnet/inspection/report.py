"""Compose adapters + views into an inspection report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .adapters import load_trace
from .html_report import write_html_report
from .schema import GridShape
from .views import ViewContext, ViewResult, get_views


@dataclass
class InspectReport:
    """Result of inspecting one trace NPZ."""

    source: Path
    output_dir: Path
    family: str
    overview: dict
    artifacts: List[str] = field(default_factory=list)
    skipped: List[dict] = field(default_factory=list)
    html_path: Optional[Path] = None

    def write_manifest(self) -> Path:
        path = self.output_dir / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "source": str(self.source),
                    "output_dir": str(self.output_dir),
                    "family": self.family,
                    "overview": self.overview,
                    "artifacts": self.artifacts,
                    "skipped": self.skipped,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return path


def inspect_trace(
    path: Path | str,
    output_dir: Path | str,
    *,
    views: Optional[Sequence[str]] = None,
    family: Optional[str] = None,
    batch_index: int = 0,
    grid_shape: Optional[GridShape] = None,
    max_frames: int = 8,
    dpi: int = 120,
    winfree_prefix: str = "decoder",
    html: bool = True,
) -> InspectReport:
    """Load a trace NPZ and render selected inspection views.

    Parameters
    ----------
    path:
        Path to a saved experiment ``*.npz`` trace.
    output_dir:
        Directory for PNGs / JSON artifacts (created if needed).
    views:
        Optional subset of view names. Default renders all registered views;
        each view no-ops cleanly when its inputs are absent.
    family:
        Force adapter family (``generator``, ``winfree``, ``phase_flow``,
        ``generic``). Default auto-detects from keys.
    batch_index:
        Which batch item to show in field movies.
    grid_shape:
        Optional ``(H, W)`` for flat oscillator sites.
    max_frames:
        Max keyframes in trajectory strips.
    dpi:
        Figure DPI for saved PNGs.
    winfree_prefix:
        Preferred Winfree trajectory prefix (``decoder`` / ``encoder``).
    html:
        If True (default), also write a self-contained ``index.html`` with
        tabs and a phase scrubber.
    """

    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_trace(
        path,
        family=family,
        grid_shape=grid_shape,
        prefix=winfree_prefix,
    )
    context = ViewContext(
        output_dir=output_dir,
        batch_index=batch_index,
        max_frames=max_frames,
        dpi=dpi,
    )
    results: List[ViewResult] = []
    for view in get_views(views):
        results.append(view.render(bundle, context))

    artifacts: List[str] = []
    skipped: List[dict] = []
    for result in results:
        if result.skipped:
            skipped.append({"view": result.name, "reason": result.reason})
            continue
        for artifact in result.paths:
            artifacts.append(str(artifact.relative_to(output_dir)))

    html_path = None
    if html:
        html_path = write_html_report(
            bundle,
            output_dir,
            artifacts=sorted(artifacts),
            skipped=skipped,
            batch_index=batch_index,
            max_frames=max(max_frames, 32),
            dpi=dpi,
        )
        artifacts.append(str(html_path.relative_to(output_dir)))

    report = InspectReport(
        source=path,
        output_dir=output_dir,
        family=bundle.family,
        overview=bundle.overview(),
        artifacts=sorted(artifacts),
        skipped=skipped,
        html_path=html_path,
    )
    report.write_manifest()
    return report


def inspect_run_traces(
    run_dir: Path | str,
    output_dir: Path | str,
    *,
    pattern: str = "traces/*.npz",
    latest_only: bool = True,
    **kwargs,
) -> List[InspectReport]:
    """Inspect one or more traces under an experiment run directory."""

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    traces = sorted(run_dir.glob(pattern))
    if not traces:
        raise FileNotFoundError(f"no traces matching {pattern!r} under {run_dir}")
    if latest_only:
        traces = traces[-1:]
    reports = []
    for trace_path in traces:
        stem = trace_path.stem
        reports.append(
            inspect_trace(
                trace_path,
                output_dir / stem,
                **kwargs,
            )
        )
    return reports
