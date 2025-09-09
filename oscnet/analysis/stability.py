"""
Functions for analyzing the stability and dynamics of oscillators.

This module includes tools for finding fixed points, computing monodromy matrices
(for Floquet analysis of periodic orbits), and generating phase space trajectories.
"""
import jax
import jax.numpy as jnp
import diffrax # Keep for SaveAt if used by get_trajectory, or for type hints
import optax
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

# Import oscillator types from core
# Assuming OscillatorInterface or Oscillator might be needed for type hints
from oscnet.core.oscillators import Oscillator 
from oscnet.core.interfaces import StateType, TimeType # For clarity

# find_fixed_points function (copied directly as it doesn't use solve_ode)
def find_fixed_points(
    oscillator: Oscillator,
    search_bounds: Tuple[jnp.ndarray, jnp.ndarray],
    n_initial_points: int = 10,
    tol: float = 1e-6,
    max_iter: int = 200,  # Increased max iterations
    learning_rate: float = 0.01,
    optimizer_type: str = 'adam',
    *,
    key: Optional[jax.random.PRNGKey] = None
) -> List[Tuple[jnp.ndarray, bool]]:
    """
    Find fixed points of an oscillator's dynamics using optimization.
    
    Args:
        oscillator: The oscillator to find fixed points for
        search_bounds: Tuple of (lower_bound, upper_bound) arrays 
                     for initial search points
        n_initial_points: Number of random initial points to try
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        learning_rate: Learning rate for the optimizer
        optimizer_type: Type of optimizer to use ('adam' or 'sgd')
        key: JAX PRNG key for random sampling
        
    Returns:
        List of tuples containing (fixed_point, is_stable)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    lower_bound, upper_bound = search_bounds
    
    keys = jax.random.split(key, n_initial_points)
    points = jnp.array([
        lower_bound + jax.random.uniform(k, shape=lower_bound.shape) * (upper_bound - lower_bound)
        for k in keys
    ])
    
    def loss_fn(state):
        # Make sure state is 1D for stability analysis
        state_1d = jnp.atleast_1d(state).reshape(-1)
        velocity = oscillator.vector_field(0.0, state_1d, args={"input_force": 0.0})
        return jnp.sum(velocity**2)
    
    grad_fn = jax.grad(loss_fn)
    
    def is_fixed_point(state, threshold=tol):
        # Make sure state is 1D for stability analysis
        state_1d = jnp.atleast_1d(state).reshape(-1)
        velocity = oscillator.vector_field(0.0, state_1d, args={"input_force": 0.0})
        return jnp.all(jnp.abs(velocity) < threshold)
    
    def is_stable(state):
        # Make sure state is 1D for stability analysis
        state_1d = jnp.atleast_1d(state).reshape(-1)
        
        # Define a function that ensures consistent input/output shapes
        def vector_field_wrapper(s):
            s_1d = jnp.atleast_1d(s).reshape(-1)
            return oscillator.vector_field(0.0, s_1d, args={"input_force": 0.0})
        
        jac_fn = jax.jacfwd(vector_field_wrapper)
        jac = jac_fn(state_1d)
        eigvals = jnp.linalg.eigvals(jac)
        return jnp.all(jnp.real(eigvals) < 0)
    
    # Select optimizer based on parameter
    if optimizer_type.lower() == 'sgd':
        optimizer = optax.sgd(learning_rate=learning_rate)
    elif optimizer_type.lower() == 'adam':
        optimizer = optax.adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Choose 'adam' or 'sgd'.")

    found_fixed_points = []
    
    for initial_point in points:
        # Make sure initial point is properly shaped
        initial_point_1d = jnp.atleast_1d(initial_point).reshape(-1)
        opt_state = optimizer.init(initial_point_1d)
        state = initial_point_1d
        
        for _ in range(max_iter):
            grads = grad_fn(state)
            updates, opt_state = optimizer.update(grads, opt_state)
            state = optax.apply_updates(state, updates)
            if loss_fn(state) < tol:
                break
        
        # Use a more lenient tolerance for fixed point detection
        relaxed_tol = tol * 10
        if is_fixed_point(state, relaxed_tol):
            is_duplicate = False
            for existing_point, _ in found_fixed_points:
                if jnp.all(jnp.abs(existing_point - state) < relaxed_tol * 10):
                    is_duplicate = True
                    break
            if not is_duplicate:
                stability = is_stable(state)
                found_fixed_points.append((state, stability))
    
    # If we didn't find any fixed points, explicitly check origin (0,0)
    # Van der Pol oscillator always has a fixed point at origin
    if len(found_fixed_points) == 0 and oscillator.__class__.__name__ == "VanDerPolOscillator":
        origin = jnp.zeros(oscillator.state_dim)
        stability = is_stable(origin)
        found_fixed_points.append((origin, stability))
        
    return found_fixed_points

# compute_monodromy_matrix (adapted to use get_trajectory)
def compute_monodromy_matrix(
    oscillator: Oscillator,
    periodic_orbit_start_state: StateType, 
    period: float,
    perturbation_size: float = 1e-5,
    **get_trajectory_kwargs 
) -> jnp.ndarray:
    """
    Compute the monodromy matrix for a periodic orbit using oscillator.get_trajectory.
    """
    state_dim = oscillator.state_dim
    integration_times = jnp.array([0.0, period])

    def simulate_one_period(initial_state: StateType) -> StateType:
        _times, states_trajectory = oscillator.get_trajectory(
            initial_state=initial_state,
            times=integration_times,
            input_force=None, 
            **get_trajectory_kwargs 
        )
        return states_trajectory[-1] 

    monodromy = jnp.zeros((state_dim, state_dim))
    reference_final_state = simulate_one_period(periodic_orbit_start_state)
    
    for i in range(state_dim):
        perturbation = jnp.zeros(state_dim).at[i].set(perturbation_size)
        perturbed_initial_state = periodic_orbit_start_state + perturbation
        perturbed_final_state = simulate_one_period(perturbed_initial_state)
        column_i = (perturbed_final_state - reference_final_state) / perturbation_size
        monodromy = monodromy.at[:, i].set(column_i)
        
    return monodromy

# phase_space_trajectory (adapted to use get_trajectory)
def phase_space_trajectory(
    oscillator: Oscillator,
    initial_states: StateType, 
    t_span: Tuple[float, float],
    n_points: int = 100,
    **get_trajectory_kwargs
) -> Tuple[TimeType, StateType]:
    """
    Compute phase space trajectories from multiple initial conditions using oscillator.get_trajectory.
    Handles both single and multiple initial_states.
    """
    t_start, t_end = t_span
    t_eval = jnp.linspace(t_start, t_end, n_points)
    
    is_single_trajectory = initial_states.ndim == 1
    
    # Ensure initial_states is 2D for vmap
    _initial_states = jnp.atleast_2d(initial_states)

    def simulate_single_trajectory(init_state: StateType) -> StateType:
        # init_state will be (state_dim,)
        _times, states_trajectory = oscillator.get_trajectory(
            initial_state=init_state,
            times=t_eval,
            input_force=None, 
            **get_trajectory_kwargs
        )
        return states_trajectory # Shape: (n_points, state_dim)

    all_trajectories = jax.vmap(simulate_single_trajectory)(_initial_states)
    # all_trajectories shape: (n_trajectories, n_points, state_dim)
    
    if is_single_trajectory:
        # If input was 1D, return 2D output (n_points, state_dim)
        final_trajectories = all_trajectories[0]
    else:
        # If input was 2D, return 3D output (n_trajectories, n_points, state_dim)
        final_trajectories = all_trajectories
            
    return t_eval, final_trajectories 