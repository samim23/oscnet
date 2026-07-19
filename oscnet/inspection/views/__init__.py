"""Pluggable inspection views over TraceBundle."""

from __future__ import annotations

from .architecture import ArchitectureView
from .base import TraceView, ViewContext, ViewResult
from .coupling import CouplingView
from .fields import PhaseFieldView, RateFieldView, VerticalGainView
from .omega import OmegaView
from .overview import OverviewView
from .readout import ReadoutView
from .synchrony import SynchronyView

DEFAULT_VIEW_NAMES = (
    "architecture",
    "overview",
    "coupling",
    "omega",
    "phase_fields",
    "rate_fields",
    "vertical_gain",
    "synchrony",
    "readout",
)


def default_views() -> list:
    return [
        ArchitectureView(),
        OverviewView(),
        CouplingView(),
        OmegaView(),
        PhaseFieldView(),
        RateFieldView(),
        VerticalGainView(),
        SynchronyView(),
        ReadoutView(),
    ]


def get_views(names: list[str] | tuple[str, ...] | None = None) -> list:
    pool = {view.name: view for view in default_views()}
    if names is None:
        return default_views()
    missing = [name for name in names if name not in pool]
    if missing:
        raise ValueError(
            f"unknown view(s) {missing}; available: {sorted(pool)}"
        )
    return [pool[name] for name in names]


__all__ = [
    "ArchitectureView",
    "CouplingView",
    "DEFAULT_VIEW_NAMES",
    "OmegaView",
    "OverviewView",
    "PhaseFieldView",
    "RateFieldView",
    "ReadoutView",
    "SynchronyView",
    "TraceView",
    "VerticalGainView",
    "ViewContext",
    "ViewResult",
    "default_views",
    "get_views",
]
