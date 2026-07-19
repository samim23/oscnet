"""Trace adapters: raw NPZ → TraceBundle."""

from .base import TraceAdapter, load_npz_arrays
from .generator import GeneratorTraceAdapter
from .generic import GenericTraceAdapter
from .phase_flow import PhaseFlowTraceAdapter
from .registry import adapt_arrays, default_adapters, detect_family, load_trace
from .winfree import WinfreeTraceAdapter

__all__ = [
    "GeneratorTraceAdapter",
    "GenericTraceAdapter",
    "PhaseFlowTraceAdapter",
    "TraceAdapter",
    "WinfreeTraceAdapter",
    "adapt_arrays",
    "default_adapters",
    "detect_family",
    "load_npz_arrays",
    "load_trace",
]
