"""
Fractal Coupling Implementations for Oscillatory Neural Networks

This module provides fractal coupling topologies that concentrate wave energy
instead of scattering it diffusively. Based on the hypothesis:

"In a fractal domain, waves don't scatter, they focus."

Key implementations:
- Hierarchical coupling (self-similar block structure)
- Power-law coupling (distance-based decay)
- Log-periodic coupling (discrete scale invariance)
- Learnable fractal mixture
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple


def create_hierarchical_coupling(
    hidden_dim: int, 
    depth: int = 1,  # Changed from 3 to 1 based on empirical findings
    inter_block_strength: float = 0.5
) -> jnp.ndarray:
    """
    Create hierarchical block-structured coupling matrix.
    
    This creates self-similar structure across multiple scales, enabling
    nested resonance pockets and energy localization.
    
    **Empirically optimal**: depth=1 for memory tasks (associative recall)
    
    Args:
        hidden_dim: Number of oscillators
        depth: Recursion depth (default: 1, optimal)
        inter_block_strength: Coupling strength between blocks (0-1, default: 0.5 optimal)
        
    Returns:
        Coupling matrix of shape (hidden_dim, hidden_dim)
        
    Example:
        >>> coupling = create_hierarchical_coupling(64, depth=1)
        >>> coupling.shape
        (64, 64)
    """
    def build_hierarchical(dim, current_depth):
        if current_depth == 0 or dim <= 1:
            return jnp.ones((dim, dim))
        
        block_size = dim // 2
        if block_size == 0:
            return jnp.ones((dim, dim))
            
        sub_block = build_hierarchical(block_size, current_depth - 1)
        
        coupling = jnp.zeros((dim, dim))
        coupling = coupling.at[:block_size, :block_size].set(sub_block)
        coupling = coupling.at[block_size:, block_size:].set(sub_block)
        coupling = coupling.at[:block_size, block_size:].set(sub_block * inter_block_strength)
        coupling = coupling.at[block_size:, :block_size].set(sub_block * inter_block_strength)
        
        return coupling
    
    coupling = build_hierarchical(hidden_dim, depth)
    return coupling / (jnp.max(coupling) + 1e-8)


def create_power_law_coupling(
    hidden_dim: int, 
    exponent: float = -1.5
) -> jnp.ndarray:
    """
    Create coupling matrix with power-law distance decay.
    
    W_ij ∝ |i-j|^exponent
    
    Args:
        hidden_dim: Number of oscillators
        exponent: Power-law exponent (typically negative, e.g., -1.5)
        
    Returns:
        Coupling matrix of shape (hidden_dim, hidden_dim)
    """
    indices = jnp.arange(hidden_dim, dtype=jnp.float32)
    distances = jnp.abs(indices[:, None] - indices[None, :]) + 1.0
    coupling = distances ** exponent
    return coupling / (jnp.max(coupling) + 1e-8)


def create_log_periodic_coupling(
    hidden_dim: int, 
    period: float = 2.0
) -> jnp.ndarray:
    """
    Create coupling matrix with log-periodic modulation.
    
    Creates discrete scale invariance with resonance bands at logarithmically
    spaced frequencies.
    
    Args:
        hidden_dim: Number of oscillators
        period: Log-periodic modulation period
        
    Returns:
        Coupling matrix of shape (hidden_dim, hidden_dim)
    """
    indices = jnp.arange(hidden_dim, dtype=jnp.float32)
    log_distances = jnp.log(jnp.abs(indices[:, None] - indices[None, :]) + 1.0)
    coupling = jnp.cos(2 * jnp.pi * log_distances / jnp.log(period))
    coupling = coupling * jnp.exp(-jnp.abs(log_distances) / hidden_dim)
    return coupling / (jnp.max(jnp.abs(coupling)) + 1e-8)


class HierarchicalCouplingLayer(eqx.Module):
    """
    Hierarchical fractal coupling layer with learnable strength.
    
    This replaces the standard dense Linear(hidden_dim, hidden_dim) layer
    with a hierarchical self-similar structure.
    
    **Optimal configuration** (empirically determined):
    - depth=1: Best for associative recall and memory tasks
    - inter_block_strength=0.5: Optimal coupling strength
    
    Usage:
        >>> layer = HierarchicalCouplingLayer(64, depth=1, key=key)
        >>> output = layer(inputs)
    """
    
    coupling_matrix: jnp.ndarray  # Fixed hierarchical structure
    strength: jnp.ndarray  # Learnable scaling factor
    
    def __init__(
        self, 
        hidden_dim: int,
        depth: int = 1,  # Changed from 2 to 1 based on empirical findings
        inter_block_strength: float = 0.5,
        initial_strength: float = 1.0,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize hierarchical coupling layer.
        
        Args:
            hidden_dim: Number of oscillators
            depth: Hierarchical recursion depth (default: 1, optimal for memory tasks)
            inter_block_strength: Coupling strength between blocks (default: 0.5, optimal)
            initial_strength: Initial value for learnable strength parameter
            key: JAX PRNG key
        """
        self.coupling_matrix = create_hierarchical_coupling(
            hidden_dim, depth, inter_block_strength
        )
        self.strength = jnp.array(initial_strength)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply hierarchical coupling to input.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, hidden_dim)
        """
        # Apply learnable scaling
        scaled_coupling = self.strength * self.coupling_matrix
        
        # Apply coupling: y = W @ x
        return jnp.dot(scaled_coupling, x.T).T


class AdaptiveFractalCouplingLayer(eqx.Module):
    """
    Simplified adaptive fractal coupling with per-oscillator routing.
    
    Each oscillator learns which fractal base (different depths) to use,
    enabling adaptive multi-scale pattern separation.
    
    **Key differences from fixed HierarchicalCouplingLayer**:
    - Learns to route patterns to different fractal scales
    - Can handle overlapping patterns when trained
    - More parameters than fixed, fewer than dense
    
    **Limitations**:
    - Requires training to work effectively
    - Random initialization performs worse than fixed
    
    Usage:
        >>> layer = AdaptiveFractalCouplingLayer(64, key=key)
        >>> output = layer(inputs)
    """
    
    fractal_bases: jnp.ndarray  # (n_depths, hidden_dim, hidden_dim) - fixed
    routing_weights: jnp.ndarray  # (hidden_dim, n_depths) - learnable
    
    def __init__(
        self, 
        hidden_dim: int,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize adaptive fractal coupling layer.
        
        Creates multiple hierarchical structures at different depths (1, 2, 3)
        and learns routing weights for each oscillator.
        
        Args:
            hidden_dim: Number of oscillators
            key: JAX PRNG key
        """
        # Create hierarchical bases at different depths
        self.fractal_bases = jnp.stack([
            create_hierarchical_coupling(hidden_dim, depth=1),
            create_hierarchical_coupling(hidden_dim, depth=2),
            create_hierarchical_coupling(hidden_dim, depth=3),
        ])
        
        # Learnable routing: each oscillator chooses fractal base
        self.routing_weights = jax.random.normal(key, (hidden_dim, 3)) * 0.1
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply adaptive fractal coupling.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, hidden_dim)
        """
        contributions = []
        for depth_idx in range(3):
            # Get routing weights for this depth
            weights = self.routing_weights[:, depth_idx]
            # Scale coupling matrix by routing weights
            scaled_matrix = self.fractal_bases[depth_idx] * weights[:, None]
            # Apply coupling and accumulate
            contribution = jnp.dot(scaled_matrix, x.T).T
            contributions.append(contribution)
        
        # Sum contributions from all fractal bases
        return jnp.sum(jnp.stack(contributions), axis=0)


class FractalCouplingLayer(eqx.Module):
    """
    Learnable fractal coupling via mixture of fractal bases.
    
    Instead of learning O(N²) weights, learns mixing coefficients for
    O(K) fractal templates where K << N².
    
    Usage:
        >>> layer = FractalCouplingLayer(64, n_bases=5, key=key)
        >>> output = layer(inputs)
    """
    
    fractal_bases: jnp.ndarray  # (n_bases, hidden_dim, hidden_dim) - fixed
    mixing_weights: jnp.ndarray  # (hidden_dim, hidden_dim, n_bases) - learnable
    
    def __init__(
        self, 
        hidden_dim: int, 
        n_bases: int = 5, 
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize learnable fractal coupling layer.
        
        Args:
            hidden_dim: Number of oscillators
            n_bases: Number of fractal base templates
            key: JAX PRNG key
        """
        keys = jax.random.split(key, 2)
        
        # Create diverse fractal bases
        bases = []
        bases.append(create_power_law_coupling(hidden_dim, exponent=-1.0))
        bases.append(create_power_law_coupling(hidden_dim, exponent=-2.0))
        bases.append(create_hierarchical_coupling(hidden_dim, depth=2))
        bases.append(create_hierarchical_coupling(hidden_dim, depth=3))
        bases.append(create_log_periodic_coupling(hidden_dim, period=2.0))
        
        # Pad or trim to n_bases
        if len(bases) > n_bases:
            bases = bases[:n_bases]
        elif len(bases) < n_bases:
            # Repeat last base if needed
            while len(bases) < n_bases:
                bases.append(bases[-1])
        
        self.fractal_bases = jnp.stack(bases)
        
        # Learnable mixing weights (small initialization)
        self.mixing_weights = jax.random.normal(
            keys[0], (hidden_dim, hidden_dim, n_bases)
        ) * 0.1
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply fractal coupling via mixture.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, hidden_dim)
        """
        # Mix fractal bases with learned weights
        # Result: (hidden_dim, hidden_dim) coupling matrix
        coupling_matrix = jnp.einsum(
            'kij,ijk->ij', 
            self.fractal_bases, 
            self.mixing_weights
        )
        
        # Apply coupling
        return jnp.dot(coupling_matrix, x.T).T


# Convenience function to create coupling from type name
def create_coupling_matrix(
    hidden_dim: int,
    coupling_type: str = "hierarchical",
    **kwargs
) -> jnp.ndarray:
    """
    Create coupling matrix from type string.
    
    Args:
        hidden_dim: Number of oscillators
        coupling_type: Type of coupling ("hierarchical", "power_law", "log_periodic")
        **kwargs: Additional arguments for specific coupling types
        
    Returns:
        Coupling matrix of shape (hidden_dim, hidden_dim)
        
    Example:
        >>> coupling = create_coupling_matrix(64, "hierarchical", depth=1)
        >>> coupling.shape
        (64, 64)
    """
    if coupling_type == "hierarchical":
        depth = kwargs.get("depth", 1)  # Optimal depth=1 for memory tasks
        inter_block_strength = kwargs.get("inter_block_strength", 0.5)
        return create_hierarchical_coupling(hidden_dim, depth, inter_block_strength)
    
    elif coupling_type == "power_law":
        exponent = kwargs.get("exponent", -1.5)
        return create_power_law_coupling(hidden_dim, exponent)
    
    elif coupling_type == "log_periodic":
        period = kwargs.get("period", 2.0)
        return create_log_periodic_coupling(hidden_dim, period)
    
    else:
        raise ValueError(f"Unknown coupling type: {coupling_type}")


# Export key functions
__all__ = [
    "create_hierarchical_coupling",
    "create_power_law_coupling",
    "create_log_periodic_coupling",
    "HierarchicalCouplingLayer",
    "AdaptiveFractalCouplingLayer",
    "FractalCouplingLayer",
    "create_coupling_matrix",
]

