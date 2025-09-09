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
    visualization: Plotting and visualization utilities
    analysis: Advanced analysis tools for oscillatory dynamics
    utils: General utilities for JAX optimization, logging, and checkpointing
"""

# Core oscillator and architecture components
from . import core
from . import models

# Learning and training utilities
from . import learning

# Evaluation and analysis tools
from . import evaluation
from . import analysis

# Visualization tools
from . import visualization

# General utilities
from . import utils

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

__all__ = [
    "core",
    "models", 
    "learning",
    "evaluation",
    "analysis",
    "visualization",
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