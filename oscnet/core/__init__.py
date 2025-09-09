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
    
    # Pure functions for dynamics
    harmonic_oscillator_update,
    nonlinear_harmonic_oscillator_update,
    van_der_pol_update,
    stuart_landau_update,
    kuramoto_update,
    fitzhugh_nagumo_update
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
    
    # Pure functions
    "harmonic_oscillator_update",
    "nonlinear_harmonic_oscillator_update",
    "van_der_pol_update",
    "stuart_landau_update",
    "kuramoto_update",
    "fitzhugh_nagumo_update"
] 