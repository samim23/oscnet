"""
Core Oscillator Dynamics and Implementations for Oscillatory Neural Networks

This module contains:
1. Pure functions for oscillator dynamics
2. Oscillator module implementations using JAX and Equinox

Separated from the main architecture to provide clean, reusable oscillator components.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Union, Optional, Dict
from abc import abstractmethod


# ======== 1. CORE OSCILLATOR DYNAMICS (PURE FUNCTIONS) ========

def harmonic_oscillator_update(
    x: jnp.ndarray,
    v: jnp.ndarray, 
    inputs: jnp.ndarray,
    omega_squared: jnp.ndarray,
    gamma_factor: jnp.ndarray,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pure function implementing linear harmonic oscillator dynamics.
    
    dx/dt = v
    dv/dt = inputs - omega^2 * x - 2*gamma * v
    
    Args:
        x: Position state
        v: Velocity state
        inputs: External forcing input
        omega_squared: Square of natural frequency
        gamma_factor: Damping factor (2*gamma)
        dt: Integration timestep
        
    Returns:
        Tuple of updated (x, v) states
    """
    # Update equations (symplectic Euler integration)
    x_new = x + dt * v
    v_new = v + dt * (inputs - omega_squared * x_new - gamma_factor * v)
    return x_new, v_new


def nonlinear_harmonic_oscillator_update(
    x: jnp.ndarray, 
    v: jnp.ndarray, 
    inputs: jnp.ndarray, 
    alpha: float, 
    omega_squared: jnp.ndarray, 
    gamma_factor: jnp.ndarray, 
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pure function implementing nonlinear harmonic oscillator dynamics.
    
    dx/dt = v
    dv/dt = alpha * tanh(inputs) - omega^2 * x - 2*gamma * v
    
    Args:
        x: Position state
        v: Velocity state
        inputs: External forcing input
        alpha: Nonlinearity strength parameter
        omega_squared: Square of natural frequency
        gamma_factor: Damping factor (2*gamma)
        dt: Integration timestep
        
    Returns:
        Tuple of updated (x, v) states
    """
    # Nonlinear forcing term
    forcing = alpha * jnp.tanh(inputs)
    
    # Update equations (symplectic Euler integration)
    x_new = x + dt * v
    v_new = v + dt * (forcing - omega_squared * x_new - gamma_factor * v)
    return x_new, v_new


def van_der_pol_update(
    x: jnp.ndarray,
    v: jnp.ndarray,
    inputs: jnp.ndarray,
    mu: float,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pure function implementing Van der Pol oscillator dynamics.
    
    dx/dt = v
    dv/dt = mu * (1 - x^2) * v - x + inputs
    
    Args:
        x: Position state
        v: Velocity state
        inputs: External forcing input
        mu: Nonlinearity and damping strength parameter
        dt: Integration timestep
        
    Returns:
        Tuple of updated (x, v) states
    """
    # Update equations (forward Euler integration)
    x_new = x + dt * v
    v_new = v + dt * (mu * (1 - x**2) * v - x + inputs)
    return x_new, v_new


def stuart_landau_update(
    x: jnp.ndarray,
    y: jnp.ndarray,
    inputs: jnp.ndarray,
    alpha: float,
    omega: float,
    beta: float,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pure function implementing Stuart-Landau oscillator dynamics.
    
    dx/dt = αx - ωy - βx(x² + y²) + inputs
    dy/dt = ωx + αy - βy(x² + y²)
    
    Args:
        x: First state component
        y: Second state component
        inputs: External forcing input (applied to x equation)
        alpha: Linear growth rate
        omega: Natural frequency
        beta: Nonlinear saturation
        dt: Integration timestep
        
    Returns:
        Tuple of updated (x, y) states
    """
    r_squared = x**2 + y**2
    
    # Update equations (forward Euler integration)
    x_new = x + dt * (alpha * x - omega * y - beta * x * r_squared + inputs)
    y_new = y + dt * (omega * x + alpha * y - beta * y * r_squared)
    return x_new, y_new


def kuramoto_update(
    theta: jnp.ndarray,
    inputs: jnp.ndarray,
    omega: float,
    dt: float
) -> jnp.ndarray:
    """
    Pure function implementing Kuramoto phase oscillator dynamics.
    
    dθ/dt = ω + inputs
    
    Args:
        theta: Phase state
        inputs: External forcing input
        omega: Natural frequency
        dt: Integration timestep
        
    Returns:
        Updated phase state
    """
    # Update equation (forward Euler integration)
    theta_new = theta + dt * (omega + inputs)
    return theta_new


def fitzhugh_nagumo_update(
    v: jnp.ndarray,  # Membrane potential
    w: jnp.ndarray,  # Recovery variable
    inputs: jnp.ndarray,  # External current
    a: float = 0.7,  # Recovery rate
    b: float = 0.8,  # Self-feedback
    tau: float = 12.5,  # Time constant
    dt: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pure function implementing FitzHugh-Nagumo neuron model.
    
    dv/dt = v - v^3/3 - w + inputs
    dw/dt = (v + a - b*w)/tau
    
    Args:
        v: Membrane potential
        w: Recovery variable
        inputs: External current
        a, b, tau: Model parameters
        dt: Integration timestep
        
    Returns:
        Tuple of updated (v, w) states
    """
    # Update equations (forward Euler integration)
    v_new = v + dt * (v - v**3/3 - w + inputs)
    w_new = w + dt * (v + a - b*w) / tau
    return v_new, w_new


# ======== 2. OSCILLATOR MODULE IMPLEMENTATIONS ========

class Oscillator(eqx.Module):
    """Base oscillator class defining the interface"""
    
    @abstractmethod
    def step(
        self, 
        x: jnp.ndarray, 
        v: jnp.ndarray, 
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform a single integration step"""
        raise NotImplementedError


class LinearHarmonicOscillator(Oscillator):
    """Linear harmonic oscillator implementation"""
    
    # Parameters
    omega: jnp.ndarray  # Trainable
    gamma: jnp.ndarray  # Trainable
    dt: float = eqx.field(static=True)
    
    def __init__(
        self, 
        dim: int,
        omega: Union[float, jnp.ndarray] = 2.0 * jnp.pi / 28.0,
        gamma: Union[float, jnp.ndarray] = 0.01,
        dt: float = 1.0,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize a linear harmonic oscillator.
        
        Args:
            dim: Dimension of the oscillator
            omega: Natural frequency (or array of frequencies)
            gamma: Damping coefficient (or array of coefficients)
            dt: Integration timestep
            key: JAX PRNG key
        """
        self.dt = dt
        
        # Handle scalar or array parameters
        if isinstance(omega, (int, float)):
            self.omega = jnp.ones(dim) * omega
        else:
            self.omega = omega
            
        if isinstance(gamma, (int, float)):
            self.gamma = jnp.ones(dim) * gamma
        else:
            self.gamma = gamma
    
    def step(
        self, 
        x: jnp.ndarray, 
        v: jnp.ndarray, 
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform a single integration step"""
        omega_squared = self.omega**2
        gamma_factor = 2.0 * self.gamma
        return harmonic_oscillator_update(
            x, v, inputs, omega_squared, gamma_factor, self.dt
        )


class NonlinearHarmonicOscillator(Oscillator):
    """Nonlinear harmonic oscillator implementation"""
    
    # Parameters - static
    alpha: float = eqx.field(static=True)
    omega: Union[float, jnp.ndarray] = eqx.field(static=True)  # not trainable
    gamma: Union[float, jnp.ndarray] = eqx.field(static=True)  # not trainable
    dt: float = eqx.field(static=True)
    
    def __init__(
        self, 
        dim: int,
        alpha: float = 0.04, 
        omega: Union[float, jnp.ndarray] = 2.0 * jnp.pi / 28.0,
        gamma: Union[float, jnp.ndarray] = 0.01,
        dt: float = 1.0,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize a nonlinear harmonic oscillator.
        
        Args:
            dim: Dimension of the oscillator
            alpha: Nonlinearity strength parameter
            omega: Natural frequency (or array of frequencies)
            gamma: Damping coefficient (or array of coefficients)
            dt: Integration timestep
            key: JAX PRNG key
        """
        self.alpha = alpha
        self.dt = dt
        
        # Handle scalar or array parameters
        # Store as tuples to avoid JAX array warnings in static fields
        if isinstance(omega, (int, float)):
            self.omega = float(omega)  # Store scalar as float
        else:
            self.omega = tuple(jnp.asarray(omega).tolist())  # Store as tuple
            
        if isinstance(gamma, (int, float)):
            self.gamma = float(gamma)  # Store scalar as float
        else:
            self.gamma = tuple(jnp.asarray(gamma).tolist())  # Store as tuple
    
    def step(
        self, 
        x: jnp.ndarray, 
        v: jnp.ndarray, 
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform a single integration step"""
        # Convert stored parameters to proper JAX arrays for computation
        if isinstance(self.omega, (int, float)):
            omega_array = jnp.ones(x.shape[-1]) * self.omega
        else:
            omega_array = jnp.asarray(self.omega)
            
        if isinstance(self.gamma, (int, float)):
            gamma_array = jnp.ones(x.shape[-1]) * self.gamma  
        else:
            gamma_array = jnp.asarray(self.gamma)
            
        omega_squared = omega_array**2
        gamma_factor = 2.0 * gamma_array
        
        return nonlinear_harmonic_oscillator_update(
            x, v, inputs, self.alpha, omega_squared, gamma_factor, self.dt
        )


class VanDerPolOscillator(Oscillator):
    """Van der Pol oscillator implementation"""
    
    # Parameters
    mu: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    
    def __init__(
        self,
        dim: int,
        mu: float = 1.0,
        dt: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize a Van der Pol oscillator.
        
        Args:
            dim: Dimension of the oscillator (not used, kept for interface compatibility)
            mu: Nonlinearity and damping strength parameter
            dt: Integration timestep
            key: JAX PRNG key
        """
        self.mu = mu
        self.dt = dt
    
    def step(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform a single integration step"""
        return van_der_pol_update(x, v, inputs, self.mu, self.dt)


class StuartLandauOscillator(Oscillator):
    """Stuart-Landau oscillator implementation"""
    
    # Parameters
    alpha: float = eqx.field(static=True)
    omega: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    
    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        omega: float = 1.0,
        beta: float = 1.0,
        dt: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize a Stuart-Landau oscillator.
        
        Args:
            dim: Dimension of the oscillator (not used, kept for interface compatibility)
            alpha: Linear growth rate
            omega: Natural frequency
            beta: Nonlinear saturation
            dt: Integration timestep
            key: JAX PRNG key
        """
        self.alpha = alpha
        self.omega = omega
        self.beta = beta
        self.dt = dt
    
    def step(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform a single integration step"""
        return stuart_landau_update(x, y, inputs, self.alpha, self.omega, self.beta, self.dt)


class KuramotoOscillator(Oscillator):
    """Kuramoto phase oscillator implementation"""
    
    # Parameters
    omega: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    
    def __init__(
        self,
        dim: int,
        omega: float = 1.0,
        dt: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize a Kuramoto oscillator.
        
        Args:
            dim: Dimension of the oscillator (not used, kept for interface compatibility)
            omega: Natural frequency
            dt: Integration timestep
            key: JAX PRNG key
        """
        self.omega = omega
        self.dt = dt
    
    def step(
        self,
        theta: jnp.ndarray,
        dummy: jnp.ndarray,  # Not used, but kept for interface compatibility
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform a single integration step"""
        theta_new = kuramoto_update(theta, inputs, self.omega, self.dt)
        return theta_new, dummy  # Return dummy as second component for interface compatibility


class FitzHughNagumoOscillator(Oscillator):
    """FitzHugh-Nagumo neural oscillator implementation"""
    
    # Parameters
    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    
    def __init__(
        self,
        dim: int,
        a: float = 0.7,
        b: float = 0.8,
        tau: float = 12.5, 
        dt: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize a FitzHugh-Nagumo neural oscillator.
        
        Args:
            dim: Dimension of the oscillator
            a: Recovery rate
            b: Self-feedback
            tau: Time constant
            dt: Integration timestep
            key: JAX PRNG key
        """
        self.a = a
        self.b = b
        self.tau = tau
        self.dt = dt
    
    def step(
        self, 
        v: jnp.ndarray, 
        w: jnp.ndarray, 
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform a single integration step"""
        return fitzhugh_nagumo_update(
            v, w, inputs, self.a, self.b, self.tau, self.dt
        )


class LearnableNonlinearHarmonicOscillator(Oscillator):
    """Learnable nonlinear harmonic oscillator with trainable ω and γ parameters"""
    
    # Fixed parameters (for stability)
    alpha: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    omega_bounds: Tuple[float, float] = eqx.field(static=True)
    gamma_bounds: Tuple[float, float] = eqx.field(static=True)
    
    # Learnable parameters (for task adaptation)
    omega: jnp.ndarray  # Trainable frequencies - each oscillator learns its optimal frequency
    gamma: jnp.ndarray  # Trainable damping - each oscillator learns its stability/expressiveness trade-off
    
    def __init__(
        self, 
        dim: int,
        alpha: float = 0.04, 
        omega_init: Union[float, jnp.ndarray] = 2.0 * jnp.pi / 28.0,
        gamma_init: Union[float, jnp.ndarray] = 0.01,
        omega_bounds: Tuple[float, float] = (jnp.pi/56, 4*jnp.pi/7),  # Reasonable frequency range
        gamma_bounds: Tuple[float, float] = (0.001, 0.2),  # Stable damping range
        dt: float = 1.0,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize a learnable nonlinear harmonic oscillator.
        
        Args:
            dim: Dimension of the oscillator
            alpha: Nonlinearity strength parameter (kept static for stability)
            omega_init: Initial frequency values (scalar or array)
            gamma_init: Initial damping values (scalar or array) 
            omega_bounds: (min, max) bounds for frequency learning
            gamma_bounds: (min, max) bounds for damping learning
            dt: Integration timestep
            key: JAX PRNG key
        """
        self.alpha = alpha
        self.dt = dt
        self.omega_bounds = omega_bounds
        self.gamma_bounds = gamma_bounds
        
        # Initialize learnable parameters
        keys = jax.random.split(key, 2)
        
        # Initialize omega with small random variations around the base frequency
        if isinstance(omega_init, (int, float)):
            # Add small random variations to break symmetry and encourage specialization
            omega_base = jnp.ones(dim) * omega_init
            omega_noise = jax.random.normal(keys[0], (dim,)) * (omega_init * 0.1)  # 10% variation
            self.omega = jnp.clip(omega_base + omega_noise, omega_bounds[0], omega_bounds[1])
        else:
            self.omega = jnp.clip(jnp.asarray(omega_init), omega_bounds[0], omega_bounds[1])
            
        # Initialize gamma with small random variations around the base damping
        if isinstance(gamma_init, (int, float)):
            # Add small random variations for specialization
            gamma_base = jnp.ones(dim) * gamma_init  
            gamma_noise = jax.random.normal(keys[1], (dim,)) * (gamma_init * 0.2)  # 20% variation
            self.gamma = jnp.clip(gamma_base + gamma_noise, gamma_bounds[0], gamma_bounds[1])
        else:
            self.gamma = jnp.clip(jnp.asarray(gamma_init), gamma_bounds[0], gamma_bounds[1])
    
    def step(
        self, 
        x: jnp.ndarray, 
        v: jnp.ndarray, 
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform a single integration step with learnable parameters"""
        # Apply bounds during forward pass to ensure stability
        omega_clipped = jnp.clip(self.omega, self.omega_bounds[0], self.omega_bounds[1])
        gamma_clipped = jnp.clip(self.gamma, self.gamma_bounds[0], self.gamma_bounds[1])
        
        omega_squared = omega_clipped**2
        gamma_factor = 2.0 * gamma_clipped
        
        return nonlinear_harmonic_oscillator_update(
            x, v, inputs, self.alpha, omega_squared, gamma_factor, self.dt
        )


class AdaptiveNonlinearHarmonicOscillator(Oscillator):
    """Adaptive oscillator with learnable coupling gain multiplier + learnable ω,γ"""
    
    # Fixed parameters
    alpha: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    base_omega: float = eqx.field(static=True)  # Base frequency for initialization
    base_gamma: float = eqx.field(static=True)  # Base damping for initialization
    omega_multiplier_bounds: Tuple[float, float] = eqx.field(static=True)
    gamma_multiplier_bounds: Tuple[float, float] = eqx.field(static=True)
    
    # Learnable parameters  
    omega_multipliers: jnp.ndarray  # Multipliers on base frequency
    gamma_multipliers: jnp.ndarray  # Multipliers on base damping
    
    def __init__(
        self, 
        dim: int,
        alpha: float = 0.04,
        base_omega: float = 2.0 * jnp.pi / 28.0,
        base_gamma: float = 0.01,
        omega_multiplier_bounds: Tuple[float, float] = (0.25, 4.0),  # 0.25x to 4x base frequency
        gamma_multiplier_bounds: Tuple[float, float] = (0.1, 20.0),  # 0.1x to 20x base damping 
        dt: float = 1.0,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize adaptive oscillator with multiplier-based learning.
        
        This approach learns multipliers on base values rather than absolute values,
        which can be more stable and interpretable.
        
        Args:
            dim: Dimension of the oscillator
            alpha: Nonlinearity strength (fixed)
            base_omega: Base natural frequency 
            base_gamma: Base damping coefficient
            omega_multiplier_bounds: Bounds on frequency multipliers
            gamma_multiplier_bounds: Bounds on damping multipliers
            dt: Integration timestep
            key: JAX PRNG key
        """
        self.alpha = alpha
        self.dt = dt
        self.base_omega = base_omega
        self.base_gamma = base_gamma
        self.omega_multiplier_bounds = omega_multiplier_bounds
        self.gamma_multiplier_bounds = gamma_multiplier_bounds
        
        # Initialize multipliers with small random variations around 1.0
        keys = jax.random.split(key, 2)
        
        # Omega multipliers: start near 1.0 with small variations
        omega_noise = jax.random.normal(keys[0], (dim,)) * 0.2  # 20% variation
        self.omega_multipliers = jnp.clip(
            jnp.ones(dim) + omega_noise, 
            omega_multiplier_bounds[0], 
            omega_multiplier_bounds[1]
        )
        
        # Gamma multipliers: start near 1.0 with small variations  
        gamma_noise = jax.random.normal(keys[1], (dim,)) * 0.3  # 30% variation
        self.gamma_multipliers = jnp.clip(
            jnp.ones(dim) + gamma_noise,
            gamma_multiplier_bounds[0],
            gamma_multiplier_bounds[1] 
        )
    
    def step(
        self, 
        x: jnp.ndarray, 
        v: jnp.ndarray, 
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Perform integration step with adaptive parameters"""
        # Apply bounds and compute effective parameters
        omega_mult_clipped = jnp.clip(
            self.omega_multipliers, 
            self.omega_multiplier_bounds[0], 
            self.omega_multiplier_bounds[1]
        )
        gamma_mult_clipped = jnp.clip(
            self.gamma_multipliers,
            self.gamma_multiplier_bounds[0],
            self.gamma_multiplier_bounds[1]
        )
        
        # Compute effective omega and gamma
        effective_omega = self.base_omega * omega_mult_clipped
        effective_gamma = self.base_gamma * gamma_mult_clipped
        
        omega_squared = effective_omega**2
        gamma_factor = 2.0 * effective_gamma
        
        return nonlinear_harmonic_oscillator_update(
            x, v, inputs, self.alpha, omega_squared, gamma_factor, self.dt
        ) 


# ===== PHASE-AWARE OSCILLATOR UTILITIES =====

def compute_phase(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Compute instantaneous phase from amplitude and velocity"""
    return jnp.arctan2(v, x)

def compute_kuramoto_order_parameter(phases: jnp.ndarray) -> jnp.ndarray:
    """Compute global phase synchronization (Kuramoto order parameter)"""
    complex_phases = jnp.exp(1j * phases)
    order_param = jnp.abs(jnp.mean(complex_phases, axis=-1))
    return order_param

def phase_difference_matrix(phases: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise phase differences between oscillators"""
    phase_diffs = phases[..., :, None] - phases[..., None, :]
    # Wrap to [-π, π]
    phase_diffs = jnp.angle(jnp.exp(1j * phase_diffs))
    return phase_diffs

def compute_phase_coherence_matrix(phases: jnp.ndarray) -> jnp.ndarray:
    """Compute phase coherence matrix between all oscillator pairs"""
    phase_diffs = phase_difference_matrix(phases)
    coherence = jnp.abs(jnp.mean(jnp.exp(1j * phase_diffs), axis=0))  # Average over time
    return coherence


class PhaseAwareLearnableOscillator(Oscillator):
    """
    🎵 PHASE-AWARE RESONANT OSCILLATOR 🎵
    
    Enhanced learnable oscillator with explicit phase tracking and coordination.
    This enables true "resonant computing" with harmonic phase relationships.
    
    Key Features:
    - Explicit phase tracking and coordination
    - Phase-based coupling between oscillators  
    - Learnable phase bias for harmonic relationships
    - Phase coherence optimization
    """
    
    # Fixed parameters (for stability)
    alpha: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    omega_bounds: Tuple[float, float] = eqx.field(static=True)
    gamma_bounds: Tuple[float, float] = eqx.field(static=True)
    
    # Learnable parameters (for task adaptation)
    omega: jnp.ndarray  # Trainable frequencies
    gamma: jnp.ndarray  # Trainable damping
    phase_bias: jnp.ndarray  # Trainable phase offsets for harmonic relationships
    phase_coupling_strength: jnp.ndarray  # How strongly each oscillator couples to global phase
    
    def __init__(
        self, 
        dim: int,
        alpha: float = 0.04, 
        omega_init: Union[float, jnp.ndarray] = 2.0 * jnp.pi / 28.0,
        gamma_init: Union[float, jnp.ndarray] = 0.01,
        omega_bounds: Tuple[float, float] = (jnp.pi/56, 4*jnp.pi/7),
        gamma_bounds: Tuple[float, float] = (0.001, 0.2),
        dt: float = 1.0,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Initialize phase-aware learnable oscillator.
        
        Args:
            dim: Number of oscillators in the bank
            alpha: Nonlinearity strength (fixed for stability)
            omega_init: Initial frequency (or frequencies)
            gamma_init: Initial damping (or damping values)
            omega_bounds: (min_freq, max_freq) for learnable frequencies
            gamma_bounds: (min_damp, max_damp) for learnable damping
            dt: Integration timestep
            key: JAX PRNG key for initialization
        """
        self.alpha = alpha
        self.dt = dt
        self.omega_bounds = omega_bounds
        self.gamma_bounds = gamma_bounds
        
        keys = jax.random.split(key, 4)
        
        # Initialize learnable frequencies
        if isinstance(omega_init, (int, float)):
            omega_init_array = jnp.ones(dim) * omega_init
        else:
            omega_init_array = jnp.asarray(omega_init)
        self.omega = omega_init_array + jax.random.normal(keys[0], (dim,)) * 0.1
        
        # Initialize learnable damping
        if isinstance(gamma_init, (int, float)):
            gamma_init_array = jnp.ones(dim) * gamma_init
        else:
            gamma_init_array = jnp.asarray(gamma_init)
        self.gamma = gamma_init_array + jax.random.normal(keys[1], (dim,)) * 0.01
        
        # NEW: Initialize phase biases for harmonic relationships
        # Start with small random phase offsets
        self.phase_bias = jax.random.uniform(keys[2], (dim,), minval=0, maxval=2*jnp.pi)
        
        # NEW: Initialize phase coupling strengths
        # Start with moderate coupling, let training optimize
        self.phase_coupling_strength = jnp.ones(dim) * 0.1 + jax.random.normal(keys[3], (dim,)) * 0.02
    
    def step(
        self, 
        x: jnp.ndarray, 
        v: jnp.ndarray, 
        inputs: jnp.ndarray,
        global_phase_context: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform phase-aware oscillator step with harmonic coordination.
        
        Args:
            x: Current positions 
            v: Current velocities
            inputs: External input drive
            global_phase_context: Mean phase of all oscillators for coordination
            
        Returns:
            (new_x, new_v): Updated oscillator states
        """
        # Clamp learnable parameters to bounds
        omega_clamped = jnp.clip(self.omega, self.omega_bounds[0], self.omega_bounds[1])
        gamma_clamped = jnp.clip(self.gamma, self.gamma_bounds[0], self.gamma_bounds[1])
        
        # Compute current local phases
        local_phases = compute_phase(x, v)
        
        # Standard nonlinear harmonic dynamics
        omega_squared = omega_clamped**2
        gamma_factor = 2.0 * gamma_clamped
        
        # Apply standard oscillator update
        x_new, v_new = nonlinear_harmonic_oscillator_update(
            x, v, inputs, self.alpha, omega_squared, gamma_factor, self.dt
        )
        
        # NEW: Phase-aware coordination
        if global_phase_context is not None:
            # Compute desired phase relationships
            target_phases = global_phase_context + self.phase_bias
            current_phases = compute_phase(x_new, v_new)
            
            # Phase error (how far we are from desired harmonic relationship)
            phase_errors = jnp.angle(jnp.exp(1j * (target_phases - current_phases)))
            
            # Apply phase-based corrections
            # Convert phase error back to (x, v) space and apply gentle correction
            amplitude = jnp.sqrt(x_new**2 + v_new**2) + 1e-8
            correction_strength = self.phase_coupling_strength * jnp.abs(phase_errors) * 0.1
            
            # Apply phase corrections (small nudges toward harmonic relationships)
            phase_correction_x = correction_strength * amplitude * jnp.cos(target_phases)
            phase_correction_v = correction_strength * amplitude * jnp.sin(target_phases)
            
            x_new = x_new + phase_correction_x * self.dt
            v_new = v_new + phase_correction_v * self.dt
        
        return x_new, v_new
    
    def get_current_phases(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Get current phases of all oscillators"""
        return compute_phase(x, v)
    
    def get_effective_frequencies(self) -> jnp.ndarray:
        """Get current effective frequencies (clamped to bounds)"""
        return jnp.clip(self.omega, self.omega_bounds[0], self.omega_bounds[1])
    
    def get_effective_damping(self) -> jnp.ndarray:
        """Get current effective damping (clamped to bounds)"""
        return jnp.clip(self.gamma, self.gamma_bounds[0], self.gamma_bounds[1])


# ===== PHASE-AWARE LOSS FUNCTIONS =====

def phase_coherence_loss(
    oscillator_states: Dict[str, jnp.ndarray], 
    target_coherence: float = 0.8,
    coherence_weight: float = 0.1
) -> jnp.ndarray:
    """
    🎵 Phase Coherence Loss 🎵
    
    Encourages oscillators to maintain strong phase relationships.
    
    Args:
        oscillator_states: Dict with 'x' and 'v' arrays of shape (seq_len, batch, dim)
        target_coherence: Target Kuramoto order parameter (0.8 = strong sync)
        coherence_weight: Weight for this loss component
        
    Returns:
        Loss value encouraging phase coherence
    """
    x, v = oscillator_states['x'], oscillator_states['v']
    
    # Compute phases for each timestep
    phases = compute_phase(x, v)  # Shape: (seq_len, batch, dim)
    
    # Compute mean coherence across time and batch
    coherences = []
    for t in range(phases.shape[0]):
        for b in range(phases.shape[1]):
            coherence = compute_kuramoto_order_parameter(phases[t, b])
            coherences.append(coherence)
    
    mean_coherence = jnp.mean(jnp.array(coherences))
    
    # Loss: encourage high coherence
    coherence_loss = -jnp.log(mean_coherence + 1e-8) 
    
    return coherence_weight * coherence_loss


def oscillation_regularity_loss(
    oscillator_states: Dict[str, jnp.ndarray],
    regularity_weight: float = 0.05
) -> jnp.ndarray:
    """
    📈 Oscillation Regularity Loss 📈
    
    Encourages clean, periodic oscillations by maximizing autocorrelation.
    
    Args:
        oscillator_states: Dict with 'x' and 'v' arrays
        regularity_weight: Weight for this loss component
        
    Returns:
        Loss value encouraging regular oscillations
    """
    x, v = oscillator_states['x'], oscillator_states['v']
    
    # Compute energy (combined amplitude measure)
    energy = x**2 + v**2  # Shape: (seq_len, batch, dim)
    
    # For each oscillator, compute autocorrelation at estimated period
    seq_len = energy.shape[0]
    estimated_period = max(4, seq_len // 8)  # Rough estimate
    
    if seq_len <= estimated_period:
        return jnp.array(0.0)  # Can't compute autocorrelation
    
    autocorrs = []
    for dim_idx in range(energy.shape[2]):
        for batch_idx in range(energy.shape[1]):
            signal = energy[:, batch_idx, dim_idx]
            
            # Compute autocorrelation at estimated period
            signal_1 = signal[:-estimated_period]
            signal_2 = signal[estimated_period:]
            
            # Normalized correlation
            corr = jnp.corrcoef(signal_1, signal_2)[0, 1]
            autocorrs.append(jnp.nan_to_num(corr, nan=0.0))
    
    mean_autocorr = jnp.mean(jnp.array(autocorrs))
    
    # Loss: encourage high autocorrelation (regular oscillations)
    regularity_loss = -mean_autocorr
    
    return regularity_weight * regularity_loss


def harmonic_relationship_loss(
    oscillator_states: Dict[str, jnp.ndarray],
    learnable_oscillator: PhaseAwareLearnableOscillator,
    harmonic_weight: float = 0.02
) -> jnp.ndarray:
    """
    🎼 Harmonic Relationship Loss 🎼
    
    Encourages oscillators to form musical harmonic relationships.
    
    Args:
        oscillator_states: Dict with 'x' and 'v' arrays  
        learnable_oscillator: The oscillator to extract frequencies from
        harmonic_weight: Weight for this loss component
        
    Returns:
        Loss encouraging harmonic frequency relationships
    """
    frequencies = learnable_oscillator.get_effective_frequencies()
    
    # Find fundamental frequency (lowest frequency)
    fundamental = jnp.min(frequencies)
    
    # Compute frequency ratios
    ratios = frequencies / (fundamental + 1e-8)
    
    # Encourage ratios to be close to harmonic series: 1, 2, 3, 4, 5, 6, ...
    harmonic_series = jnp.arange(1, len(frequencies) + 1, dtype=jnp.float32)
    
    # Find best matching harmonic for each frequency
    harmonic_losses = []
    for i, ratio in enumerate(ratios):
        # Distance to nearest harmonic
        distances = jnp.abs(ratio - harmonic_series)
        min_distance = jnp.min(distances)
        harmonic_losses.append(min_distance)
    
    mean_harmonic_loss = jnp.mean(jnp.array(harmonic_losses))
    
    return harmonic_weight * mean_harmonic_loss


# ===== MULTI-SCALE OSCILLATOR BANK =====

class MultiScalePhaseAwareOscillatorBank(eqx.Module):
    """
    🏗️ Multi-Scale Phase-Aware Oscillator Bank 🏗️
    
    Creates a hierarchy of oscillator banks operating at different frequency scales:
    - Low frequency: Global structure, slow patterns
    - Mid frequency: Local patterns, textures  
    - High frequency: Fine details, edges
    """
    
    low_freq_bank: PhaseAwareLearnableOscillator
    mid_freq_bank: PhaseAwareLearnableOscillator  
    high_freq_bank: PhaseAwareLearnableOscillator
    scale_weights: jnp.ndarray  # Learnable weights for combining scales
    
    def __init__(
        self,
        dim_per_scale: int = 21,  # 21 oscillators per scale (64 total ≈ your current setup)
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize multi-scale oscillator bank"""
        keys = jax.random.split(key, 4)
        
        # Low frequency bank: 0.5-4 Hz (global structure)
        self.low_freq_bank = PhaseAwareLearnableOscillator(
            dim=dim_per_scale,
            omega_bounds=(0.5 * 2 * jnp.pi, 4.0 * 2 * jnp.pi),
            key=keys[0]
        )
        
        # Mid frequency bank: 4-20 Hz (local patterns) 
        self.mid_freq_bank = PhaseAwareLearnableOscillator(
            dim=dim_per_scale,
            omega_bounds=(4.0 * 2 * jnp.pi, 20.0 * 2 * jnp.pi),
            key=keys[1]
        )
        
        # High frequency bank: 20-100 Hz (fine details)
        self.high_freq_bank = PhaseAwareLearnableOscillator(
            dim=dim_per_scale,
            omega_bounds=(20.0 * 2 * jnp.pi, 100.0 * 2 * jnp.pi),
            key=keys[2]
        )
        
        # Learnable scale combination weights
        self.scale_weights = jnp.ones(3) / 3.0 + jax.random.normal(keys[3], (3,)) * 0.1
    
    def step(
        self,
        x_low: jnp.ndarray, v_low: jnp.ndarray,
        x_mid: jnp.ndarray, v_mid: jnp.ndarray, 
        x_high: jnp.ndarray, v_high: jnp.ndarray,
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Step all oscillator banks with cross-scale phase coordination"""
        
        # Compute global phase contexts for coordination
        phase_low = jnp.mean(compute_phase(x_low, v_low))
        phase_mid = jnp.mean(compute_phase(x_mid, v_mid))
        phase_high = jnp.mean(compute_phase(x_high, v_high))
        
        # Step each bank with cross-scale phase awareness
        x_low_new, v_low_new = self.low_freq_bank.step(x_low, v_low, inputs, phase_mid)
        x_mid_new, v_mid_new = self.mid_freq_bank.step(x_mid, v_mid, inputs, phase_low)
        x_high_new, v_high_new = self.high_freq_bank.step(x_high, v_high, inputs, phase_mid)
        
        return x_low_new, v_low_new, x_mid_new, v_mid_new, x_high_new, v_high_new 
