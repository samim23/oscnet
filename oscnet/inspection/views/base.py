"""View protocol for rendering TraceBundle panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Protocol, runtime_checkable

from ..schema import TraceBundle


@dataclass
class ViewResult:
    """Artifacts produced by one view."""

    name: str
    paths: List[Path] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""


@dataclass
class ViewContext:
    """Shared rendering options passed to every view."""

    output_dir: Path
    batch_index: int = 0
    max_frames: int = 8
    dpi: int = 120


@runtime_checkable
class TraceView(Protocol):
    name: str

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        """Write one or more artifacts under ``context.output_dir``."""
