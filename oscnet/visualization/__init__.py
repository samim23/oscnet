"""
Visualization package for oscillatory neural networks.

This package provides tools for visualizing oscillatory neural networks,
their dynamics, and their outputs.
"""

from importlib import import_module

_OSCILLATOR_ANALYSIS_EXPORTS = {
    "analyze_oscillator_families",
    "analyze_parameter_distributions",
    "visualize_oscillator_families",
    "visualize_specialization_map",
    "generate_discovery_report",
    "comprehensive_oscillator_analysis",
}

__all__ = sorted(_OSCILLATOR_ANALYSIS_EXPORTS)


def __getattr__(name):
    """Load sklearn-dependent oscillator analysis tools only when requested."""
    if name in _OSCILLATOR_ANALYSIS_EXPORTS:
        module = import_module(f"{__name__}.oscillator_analysis")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _OSCILLATOR_ANALYSIS_EXPORTS)
