# OscNet: Oscillatory Neural Networks Library
"""
OscNet: Oscillatory Neural Networks Library
==========================================

A comprehensive library for building and training neural networks based on 
oscillatory dynamics, including harmonic oscillators, Van der Pol oscillators,
and other dynamical systems.

Key Features:
- Oscillator-based neural network architectures (HORN, etc.)
- Edge-of-chaos initialization and criticality analysis
- Advanced training utilities and optimizations
- Comprehensive evaluation and visualization tools
- JAX-based implementation for high performance

Modules:
    core: Core oscillator implementations and base architectures
    models: Pre-built oscillatory neural network models
    learning: Training utilities, schedulers, and optimization tools
    evaluation: Model evaluation and analysis tools
    experiments: Reference experiment CLIs and artifact harnesses
    visualization: Plotting and visualization utilities
    analysis: Advanced analysis tools for oscillatory dynamics
    inspection: Modular NPZ trace inspection (coupling, fields, synchrony)
    utils: General utilities for JAX optimization, logging, and checkpointing
"""

from importlib import import_module

_SUBMODULES = {
    "core",
    "models",
    "learning",
    "evaluation",
    "experiments",
    "analysis",
    "visualization",
    "inspection",
    "utils",
}

# Version information
__version__ = "0.1.0"

# Core oscillator components
from .core.oscillators import (
    Oscillator,
    LinearHarmonicOscillator,
    NonlinearHarmonicOscillator,
    FitzHughNagumoOscillator,
    VanDerPolOscillator,
    StuartLandauOscillator,
    KuramotoOscillator
)


def __getattr__(name):
    """Lazily import subpackages so core imports stay lightweight."""
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)

__all__ = [
    "core",
    "models", 
    "learning",
    "evaluation",
    "experiments",
    "analysis",
    "visualization",
    "inspection",
    "utils",
    # Oscillators
    "Oscillator",
    "LinearHarmonicOscillator", 
    "NonlinearHarmonicOscillator",
    "FitzHughNagumoOscillator",
    "VanDerPolOscillator",
    "StuartLandauOscillator", 
    "KuramotoOscillator",
] 
