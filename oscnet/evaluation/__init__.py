"""
Comprehensive evaluation suite for OSCNet models
"""

from .model_analysis import (
    count_model_parameters,
    print_model_summary,
    compare_model_sizes,
    analyze_model_efficiency
)

from .deep_evaluation import (
    comprehensive_model_evaluation,
    test_noise_robustness,
    test_latent_space_properties,
    test_interpolation_quality,
    create_evaluation_report
)

from .visualizations import (
    visualize_latent_interpolation,
    visualize_hybrid_latent_space,
    analyze_hybrid_phase_amplitude,
    visualize_reconstruction_quality_by_digit,
    create_enhanced_visualizations
)

# Import resonance analysis functions
from .resonance_analysis import (
    analyze_oscillator_frequencies,
    analyze_standing_waves, 
    analyze_phase_coherence,
    create_resonance_visualizations,
    comprehensive_resonance_analysis
)

__all__ = [
    # Model analysis
    'count_model_parameters',
    'print_model_summary', 
    'compare_model_sizes',
    'analyze_model_efficiency',
    
    # Deep evaluation
    'comprehensive_model_evaluation',
    'test_noise_robustness',
    'test_latent_space_properties',
    'test_interpolation_quality',
    'create_evaluation_report',
    
    # Visualizations
    'visualize_latent_interpolation',
    'visualize_hybrid_latent_space',
    'analyze_hybrid_phase_amplitude',
    'visualize_reconstruction_quality_by_digit',
    'create_enhanced_visualizations',

    # Resonance analysis
    'analyze_oscillator_frequencies',
    'analyze_standing_waves',
    'analyze_phase_coherence',
    'create_resonance_visualizations',
    'comprehensive_resonance_analysis'
] 