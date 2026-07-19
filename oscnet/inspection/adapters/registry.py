"""Registry that selects the best TraceAdapter for a raw NPZ."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from ..schema import GridShape, TraceBundle
from .base import Array, TraceAdapter, load_npz_arrays
from .generator import GeneratorTraceAdapter
from .generic import GenericTraceAdapter
from .phase_flow import PhaseFlowTraceAdapter
from .winfree import WinfreeTraceAdapter


def default_adapters() -> List[TraceAdapter]:
    """Specialized adapters first; generic always last as fallback."""

    return [
        PhaseFlowTraceAdapter(),
        GeneratorTraceAdapter(),
        WinfreeTraceAdapter(),
        GenericTraceAdapter(),
    ]


def detect_family(
    arrays: Mapping[str, Array],
    *,
    adapters: Optional[Sequence[TraceAdapter]] = None,
) -> str:
    for adapter in adapters or default_adapters():
        if adapter.matches(arrays):
            return adapter.name
    return "unknown"


def adapt_arrays(
    arrays: Mapping[str, Array],
    *,
    source: Path | str,
    family: Optional[str] = None,
    grid_shape: Optional[GridShape] = None,
    adapters: Optional[Sequence[TraceAdapter]] = None,
    **adapter_kwargs,
) -> TraceBundle:
    """Normalize raw arrays, optionally forcing a family name."""

    pool: Iterable[TraceAdapter] = adapters or default_adapters()
    if family is not None:
        for adapter in pool:
            if adapter.name == family:
                return adapter.adapt(
                    arrays,
                    source=Path(source),
                    grid_shape=grid_shape,
                    **adapter_kwargs,
                )
        raise ValueError(f"unknown trace family {family!r}")

    for adapter in pool:
        if adapter.matches(arrays):
            return adapter.adapt(
                arrays,
                source=Path(source),
                grid_shape=grid_shape,
                **adapter_kwargs,
            )
    raise ValueError("no adapter matched the provided arrays")


def load_trace(
    path: Path | str,
    *,
    family: Optional[str] = None,
    grid_shape: Optional[GridShape] = None,
    adapters: Optional[Sequence[TraceAdapter]] = None,
    **adapter_kwargs,
) -> TraceBundle:
    """Load an NPZ trace and return a normalized :class:`TraceBundle`."""

    path = Path(path)
    arrays = load_npz_arrays(path)
    return adapt_arrays(
        arrays,
        source=path,
        family=family,
        grid_shape=grid_shape,
        adapters=adapters,
        **adapter_kwargs,
    )
