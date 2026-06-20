"""
Core Components for OscNet

This module contains the fundamental building blocks for oscillatory neural networks,
including oscillator dynamics and core interfaces.
"""

from .oscillators import (
    # Base class
    Oscillator,
    
    # Harmonic oscillators
    LinearHarmonicOscillator,
    NonlinearHarmonicOscillator,
    
    # Nonlinear oscillators
    VanDerPolOscillator,
    StuartLandauOscillator,
    KuramotoOscillator,
    FitzHughNagumoOscillator,
    HORNOscillator,
    
    # Pure functions for dynamics
    harmonic_oscillator_update,
    nonlinear_harmonic_oscillator_update,
    van_der_pol_update,
    stuart_landau_update,
    kuramoto_update,
    fitzhugh_nagumo_update
)
from .dynamics import solve_ode
from .interfaces import ArgsType, Array, StateType, TimeType

from .fractal_coupling import (
    HierarchicalCouplingLayer,
    AdaptiveFractalCouplingLayer,
    FractalCouplingLayer,
    create_hierarchical_coupling,
    create_power_law_coupling,
    create_log_periodic_coupling,
    create_coupling_matrix
)

__all__ = [
    # Base class
    "Oscillator",
    
    # Oscillator implementations
    "LinearHarmonicOscillator",
    "NonlinearHarmonicOscillator", 
    "VanDerPolOscillator",
    "StuartLandauOscillator",
    "KuramotoOscillator",
    "FitzHughNagumoOscillator",
    "HORNOscillator",
    
    # Pure functions
    "harmonic_oscillator_update",
    "nonlinear_harmonic_oscillator_update",
    "van_der_pol_update",
    "stuart_landau_update",
    "kuramoto_update",
    "fitzhugh_nagumo_update",
    "solve_ode",
    "Array",
    "StateType",
    "TimeType",
    "ArgsType",
    
    # Fractal coupling
    "HierarchicalCouplingLayer",
    "AdaptiveFractalCouplingLayer",
    "FractalCouplingLayer",
    "create_hierarchical_coupling",
    "create_power_law_coupling",
    "create_log_periodic_coupling",
    "create_coupling_matrix"
] 
