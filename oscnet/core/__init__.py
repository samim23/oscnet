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
from .coupling import (
    coupling_profile_from_name,
    dense_coupling_profile,
    distance_decay_coupling_profile,
    local_radius_coupling_profile,
    normalize_coupling_profile,
    oscillator_grid_coordinates,
    rectangular_coupling_profile_from_name,
    rectangular_dense_coupling_profile,
    rectangular_distance_decay_coupling_profile,
    rectangular_local_radius_coupling_profile,
    row_laplacian,
)
from .layered import (
    InterLayerCouplingSpec,
    OscillatorLayerSpec,
    adjacent_inter_layer_specs,
    inter_layer_profile,
    intra_layer_profile,
    validate_layer_specs,
)

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

    # Coupling topology helpers
    "coupling_profile_from_name",
    "dense_coupling_profile",
    "distance_decay_coupling_profile",
    "local_radius_coupling_profile",
    "normalize_coupling_profile",
    "oscillator_grid_coordinates",
    "rectangular_coupling_profile_from_name",
    "rectangular_dense_coupling_profile",
    "rectangular_distance_decay_coupling_profile",
    "rectangular_local_radius_coupling_profile",
    "row_laplacian",
    "InterLayerCouplingSpec",
    "OscillatorLayerSpec",
    "adjacent_inter_layer_specs",
    "inter_layer_profile",
    "intra_layer_profile",
    "validate_layer_specs",
    
    # Fractal coupling
    "HierarchicalCouplingLayer",
    "AdaptiveFractalCouplingLayer",
    "FractalCouplingLayer",
    "create_hierarchical_coupling",
    "create_power_law_coupling",
    "create_log_periodic_coupling",
    "create_coupling_matrix"
] 
