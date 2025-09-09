"""
Criticality Initialization for Oscillatory Neural Networks
==========================================================

This module provides utilities for initializing oscillatory neural networks
at the edge of chaos - the critical regime where complex systems exhibit
optimal computational capabilities.

Key Features:
- Parameter optimization for target Lyapunov exponents
- Phase relationship optimization for wave interference
- Combined parameter + phase criticality scoring
- Multiple phase initialization strategies

The edge of chaos is characterized by:
- Lyapunov exponents near zero (neither stable nor chaotic)
- Rich temporal dynamics and memory
- Optimal information processing capabilities
- Maximum computational expressivity

Classes:
    CriticalityInitializer: Main utility for edge-of-chaos initialization

Example:
    ```python
    from oscnet.learning.criticality import CriticalityInitializer
    
    # Initialize parameters for edge of chaos
    params = CriticalityInitializer.initialize_for_criticality(
        dim=64,
        target_lyapunov=0.01,
        include_phases=True,
        phase_strategy="optimized"
    )
    
    # Analyze the configuration
    analysis = CriticalityInitializer.analyze_criticality_configuration(params)
    print(analysis["overall_assessment"])
    ```
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional


class CriticalityInitializer:
    """
    Initialize parameters and phases for edge of chaos dynamics.
    
    This enhanced initializer considers both:
    1. Parameter criticality (α, ω, γ) for edge-of-chaos dynamics
    2. Phase relationships for optimal wave interference and criticality
    
    The edge of chaos is the critical regime between order and chaos where
    complex systems exhibit optimal computational capabilities, memory, and
    adaptability. This initializer helps oscillatory neural networks start
    in this optimal regime.
    """
    
    @staticmethod
    def compute_lyapunov_estimate(alpha: float, omega: float, gamma: float) -> float:
        """
        Rough estimate of largest Lyapunov exponent for HORN oscillator.
        
        For a nonlinear harmonic oscillator with dynamics:
        dx/dt = v
        dv/dt = -ω²x - 2γv + α*tanh(input)
        
        The Lyapunov exponent approximation is:
        λ ≈ α - 2γ - ω²/(4α)
        
        Args:
            alpha: Nonlinearity strength parameter
            omega: Oscillation frequency
            gamma: Damping coefficient
            
        Returns:
            Estimated Lyapunov exponent
        """
        return alpha - 2 * gamma - omega**2 / (4 * alpha + 1e-8)
    
    @staticmethod
    def compute_phase_criticality_score(phases: jnp.ndarray) -> float:
        """
        Compute a criticality score based on phase distribution.
        
        Optimal criticality often occurs with:
        - Moderate phase diversity (not all synchronized, not completely random)
        - Some phase clustering for local coherence
        - Sufficient phase separation for interference patterns
        
        The score combines:
        1. Phase diversity (entropy-like measure)
        2. Phase coherence (local clustering)
        3. Interference potential (constructive/destructive interference)
        
        Args:
            phases: Array of phases (shape: dim)
            
        Returns:
            Criticality score (higher = better for edge-of-chaos)
        """
        # Normalize phases to [0, 2π]
        phases_norm = jnp.mod(phases, 2 * jnp.pi)
        
        # 1. Phase diversity score (entropy-like measure)
        # Bin phases and compute distribution
        n_bins = 8
        bin_edges = jnp.linspace(0, 2 * jnp.pi, n_bins + 1)
        hist, _ = jnp.histogram(phases_norm, bins=bin_edges)
        hist_norm = hist / jnp.sum(hist)
        
        # Entropy of phase distribution (higher = more diverse)
        entropy = -jnp.sum(hist_norm * jnp.log(hist_norm + 1e-8))
        diversity_score = entropy / jnp.log(n_bins)  # Normalize to [0,1]
        
        # 2. Phase coherence score (measure of local clustering)
        # Compute pairwise phase differences
        phase_diffs = jnp.abs(phases_norm[:, None] - phases_norm[None, :])
        phase_diffs = jnp.minimum(phase_diffs, 2 * jnp.pi - phase_diffs)  # Circular distance
        
        # Mean phase coherence (lower = more coherent)
        mean_coherence = jnp.mean(phase_diffs)
        coherence_score = 1.0 - (mean_coherence / jnp.pi)  # Normalize to [0,1]
        
        # 3. Interference potential (measure of constructive/destructive interference)
        # Sum of complex exponentials
        complex_sum = jnp.sum(jnp.exp(1j * phases_norm))
        interference_magnitude = jnp.abs(complex_sum) / len(phases)
        
        # Optimal interference is moderate (not fully constructive or destructive)
        interference_score = 1.0 - jnp.abs(interference_magnitude - 0.5) * 2
        
        # Combined criticality score (weighted combination)
        criticality_score = (
            0.4 * diversity_score +      # Want some diversity
            0.3 * coherence_score +      # Want some local coherence  
            0.3 * interference_score     # Want moderate interference
        )
        
        return float(criticality_score)
    
    @staticmethod
    def initialize_for_criticality(
        dim: int, 
        target_lyapunov: float = 0.01,
        include_phases: bool = True,
        phase_strategy: str = "optimized",
        key: Optional[jax.random.PRNGKey] = None
    ) -> Dict:
        """
        Initialize oscillator parameters and phases near edge of chaos.
        
        This method uses mathematical optimization to precisely target the
        edge of chaos regime by solving for parameters that yield the desired
        Lyapunov exponent.
        
        Args:
            dim: Number of oscillators
            target_lyapunov: Target Lyapunov exponent for criticality
                           - 0.0: Exactly at edge of chaos
                           - 0.01: Slightly chaotic (recommended)
                           - -0.01: Slightly stable
            include_phases: Whether to include phase initialization
            phase_strategy: Strategy for phase initialization
                - "optimized": Optimize phases for criticality
                - "random": Random phases
                - "clustered": Create phase clusters
                - "progressive": Progressive phase shifts
            key: Random key for initialization
            
        Returns:
            Dictionary with parameters and optionally phases:
            - "alpha": Nonlinearity strength
            - "omega": Frequency array (per oscillator)
            - "gamma": Damping array (per oscillator)
            - "dt": Time step
            - "phases": Phase array (if include_phases=True)
            - "phase_criticality_score": Phase score (if include_phases=True)
        """
        if key is None:
            key = jax.random.PRNGKey(42)
            
        keys = jax.random.split(key, 5)
        
        # 1. Initialize base parameters for criticality using iterative approach
        base_omega = 2.0 * jnp.pi / 7.0
        
        # Generate random variations for each oscillator
        omega_noise = jax.random.normal(keys[1], (dim,)) * 0.1
        omega_values = base_omega * (1.0 + omega_noise)
        omega_values = jnp.clip(omega_values, 0.5, 5.0)
        
        # For each oscillator, solve for alpha and gamma to achieve target Lyapunov
        # Lyapunov ≈ alpha - 2*gamma - omega²/(4*alpha)
        # Rearranging: alpha - 2*gamma - omega²/(4*alpha) = target_lyapunov
        
        alpha_values = []
        gamma_values = []
        
        for omega in omega_values:
            # Use a reasonable gamma value and solve for alpha
            gamma = 0.05 + jax.random.normal(keys[2]) * 0.01  # Small variation around 0.05
            gamma = jnp.clip(gamma, 0.02, 0.08)
            
            # Solve quadratic equation for alpha:
            # alpha - 2*gamma - omega²/(4*alpha) = target_lyapunov
            # 4*alpha² - (8*gamma + 4*target_lyapunov)*alpha - omega² = 0
            
            a = 4.0
            b = -(8.0 * gamma + 4.0 * target_lyapunov)
            c = -omega**2
            
            # Quadratic formula: alpha = (-b ± sqrt(b²-4ac)) / 2a
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                alpha1 = (-b + jnp.sqrt(discriminant)) / (2*a)
                alpha2 = (-b - jnp.sqrt(discriminant)) / (2*a)
                # Choose positive alpha that's in reasonable range
                alpha = alpha1 if alpha1 > 0 else alpha2
                alpha = jnp.clip(alpha, 0.3, 1.2)
            else:
                # Fallback if no real solution
                alpha = 0.6
            
            alpha_values.append(float(alpha))
            gamma_values.append(float(gamma))
        
        alpha_values = jnp.array(alpha_values)
        gamma_values = jnp.array(gamma_values)
        
        result = {
            "alpha": float(jnp.mean(alpha_values)),
            "omega": omega_values,
            "gamma": gamma_values,
            "dt": 1.0
        }
        
        # 2. Initialize phases if requested
        if include_phases:
            if phase_strategy == "optimized":
                # Optimize phases for criticality
                best_phases = None
                best_score = -1.0
                
                # Try multiple random initializations and pick the best
                for _ in range(10):
                    key_temp, subkey = jax.random.split(keys[3])
                    keys = keys.at[3].set(key_temp)  # Update keys array properly
                    candidate_phases = jax.random.uniform(
                        subkey, (dim,), minval=0.0, maxval=2.0 * jnp.pi
                    )
                    score = CriticalityInitializer.compute_phase_criticality_score(candidate_phases)
                    
                    if score > best_score:
                        best_score = score
                        best_phases = candidate_phases
                
                result["phases"] = best_phases
                result["phase_criticality_score"] = best_score
                
            elif phase_strategy == "random":
                result["phases"] = jax.random.uniform(
                    keys[3], (dim,), minval=0.0, maxval=2.0 * jnp.pi
                )
                
            elif phase_strategy == "clustered":
                # Create phase clusters for local coherence
                n_clusters = max(2, dim // 8)
                cluster_centers = jax.random.uniform(
                    keys[3], (n_clusters,), minval=0.0, maxval=2.0 * jnp.pi
                )
                cluster_assignments = jax.random.randint(
                    keys[4], (dim,), 0, n_clusters
                )
                
                # Add noise around cluster centers
                cluster_noise = jax.random.normal(keys[4], (dim,)) * 0.3
                phases = cluster_centers[cluster_assignments] + cluster_noise
                phases = jnp.mod(phases, 2.0 * jnp.pi)
                
                result["phases"] = phases
                
            elif phase_strategy == "progressive":
                # Progressive phase shifts with some randomness
                base_progression = jnp.linspace(0.0, 2.0 * jnp.pi, dim)
                phase_noise = jax.random.normal(keys[3], (dim,)) * 0.2
                phases = jnp.mod(base_progression + phase_noise, 2.0 * jnp.pi)
                
                result["phases"] = phases
                
            else:
                raise ValueError(f"Unknown phase_strategy: {phase_strategy}")
            
            # Compute criticality score for any phase strategy
            if "phase_criticality_score" not in result:
                result["phase_criticality_score"] = CriticalityInitializer.compute_phase_criticality_score(
                    result["phases"]
                )
        
        return result
    
    @staticmethod
    def analyze_criticality_configuration(params: Dict) -> Dict:
        """
        Analyze the criticality properties of a parameter/phase configuration.
        
        This method provides comprehensive analysis of how well a configuration
        achieves edge-of-chaos dynamics, including both parameter-based and
        phase-based criticality measures.
        
        Args:
            params: Dictionary with oscillator parameters and optionally phases
                   (as returned by initialize_for_criticality)
            
        Returns:
            Analysis results dictionary containing:
            - Lyapunov estimates and statistics
            - Phase criticality scores and assessments
            - Combined criticality scores
            - Overall assessment and recommendations
        """
        analysis = {}
        
        # 1. Parameter-based criticality analysis
        if "alpha" in params and "omega" in params and "gamma" in params:
            alpha = params["alpha"]
            omega = params["omega"] if hasattr(params["omega"], "__len__") else [params["omega"]]
            gamma = params["gamma"] if hasattr(params["gamma"], "__len__") else [params["gamma"]]
            
            # Compute Lyapunov estimates for each oscillator
            lyapunov_estimates = []
            for w, g in zip(omega, gamma):
                lyap = CriticalityInitializer.compute_lyapunov_estimate(alpha, w, g)
                lyapunov_estimates.append(lyap)
            
            analysis["lyapunov_estimates"] = lyapunov_estimates
            analysis["mean_lyapunov"] = float(jnp.mean(jnp.array(lyapunov_estimates)))
            analysis["lyapunov_std"] = float(jnp.std(jnp.array(lyapunov_estimates)))
            
            # Criticality assessment based on Lyapunov exponent
            mean_lyap = analysis["mean_lyapunov"]
            if -0.01 <= mean_lyap <= 0.05:
                analysis["criticality_assessment"] = "EXCELLENT - Near edge of chaos"
            elif -0.05 <= mean_lyap <= 0.1:
                analysis["criticality_assessment"] = "GOOD - Close to critical regime"
            elif mean_lyap < -0.05:
                analysis["criticality_assessment"] = "STABLE - May be too damped"
            else:
                analysis["criticality_assessment"] = "CHAOTIC - May be too unstable"
        
        # 2. Phase-based criticality analysis
        if "phases" in params:
            phases = params["phases"]
            analysis["phase_criticality_score"] = CriticalityInitializer.compute_phase_criticality_score(phases)
            analysis["phase_diversity"] = float(jnp.std(phases))
            analysis["phase_range"] = float(jnp.max(phases) - jnp.min(phases))
            
            # Phase assessment
            phase_score = analysis["phase_criticality_score"]
            if phase_score >= 0.7:
                analysis["phase_assessment"] = "EXCELLENT - Optimal phase distribution"
            elif phase_score >= 0.5:
                analysis["phase_assessment"] = "GOOD - Reasonable phase distribution"
            else:
                analysis["phase_assessment"] = "POOR - Suboptimal phase distribution"
        
        # 3. Combined assessment
        if "lyapunov_estimates" in analysis and "phase_criticality_score" in analysis:
            param_score = 1.0 - abs(analysis["mean_lyapunov"]) / 0.1  # Normalize around 0
            param_score = max(0.0, min(1.0, param_score))
            
            combined_score = 0.6 * param_score + 0.4 * analysis["phase_criticality_score"]
            analysis["combined_criticality_score"] = combined_score
            
            # Overall assessment with realistic thresholds
            if combined_score >= 0.7:
                analysis["overall_assessment"] = "EXCELLENT - Optimal criticality configuration"
            elif combined_score >= 0.5:
                analysis["overall_assessment"] = "GOOD - Well-configured for criticality"
            elif combined_score >= 0.3:
                analysis["overall_assessment"] = "FAIR - Reasonable configuration with room for improvement"
            else:
                analysis["overall_assessment"] = "NEEDS IMPROVEMENT - Suboptimal configuration"
        
        return analysis


# Convenience functions for common use cases

def initialize_edge_of_chaos(dim: int, key: Optional[jax.random.PRNGKey] = None) -> Dict:
    """
    Quick initialization for edge of chaos with default settings.
    
    Args:
        dim: Number of oscillators
        key: Random key
        
    Returns:
        Parameters optimized for edge of chaos
    """
    return CriticalityInitializer.initialize_for_criticality(
        dim=dim,
        target_lyapunov=0.01,
        include_phases=True,
        phase_strategy="optimized",
        key=key
    )


def assess_criticality(params: Dict) -> str:
    """
    Quick assessment of criticality configuration.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Overall assessment string
    """
    analysis = CriticalityInitializer.analyze_criticality_configuration(params)
    return analysis.get("overall_assessment", "Unable to assess") 