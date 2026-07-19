"""
OscNet inspection — modular visualization of oscillatory traces.

This package turns saved experiment NPZ traces into a structured, extensible
report:

```text
NPZ trace
  → adapter (family-specific) → TraceBundle
  → views (coupling, phase fields, synchrony, ...) → PNGs / JSON
  → index.html (tabs + phase scrubber)
```

Public entrypoints:

- :func:`load_trace` — normalize one NPZ
- :func:`inspect_trace` — render a full report directory (+ ``index.html``)
- :func:`main` — CLI (also ``python examples/inspect_trace.py``)

Add a new model family by registering a :class:`TraceAdapter`. Add a new panel
by implementing a :class:`TraceView` and listing it in ``views.default_views``.
"""

from .adapters import (
    GeneratorTraceAdapter,
    GenericTraceAdapter,
    PhaseFlowTraceAdapter,
    TraceAdapter,
    WinfreeTraceAdapter,
    adapt_arrays,
    default_adapters,
    detect_family,
    load_trace,
)
from .cli import main
from .html_report import write_html_report
from .report import InspectReport, inspect_run_traces, inspect_trace
from .schema import ArrayRef, TraceBundle
from .views import (
    DEFAULT_VIEW_NAMES,
    TraceView,
    ViewContext,
    ViewResult,
    default_views,
    get_views,
)

__all__ = [
    "ArrayRef",
    "DEFAULT_VIEW_NAMES",
    "GeneratorTraceAdapter",
    "GenericTraceAdapter",
    "InspectReport",
    "PhaseFlowTraceAdapter",
    "TraceAdapter",
    "TraceBundle",
    "TraceView",
    "ViewContext",
    "ViewResult",
    "WinfreeTraceAdapter",
    "adapt_arrays",
    "default_adapters",
    "default_views",
    "detect_family",
    "get_views",
    "inspect_run_traces",
    "inspect_trace",
    "load_trace",
    "main",
    "write_html_report",
]
