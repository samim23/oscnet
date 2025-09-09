"""
Floquet analysis tools for oscillatory systems.

This module provides functions to compute Floquet multipliers and 
exponents for periodic orbits, and related stability measures.
"""

import jax
import jax.numpy as jnp
import diffrax
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from jax import jacfwd # For Jacobian computation

from oscnet.core.dynamics import solve_ode
from oscnet.core.oscillators import Oscillator

# TODO: Implement functions for monodromy matrix, Floquet multipliers, etc.

def find_periodic_orbit_by_simulation(
    oscillator: Oscillator,
    y_guess: jnp.ndarray,
    t_sim_total: float,
    t_period_guess: float,
    num_points_per_period: int = 100,
    ode_solver_options: dict | None = None
) -> tuple[diffrax.Solution, jnp.ndarray, float]:
    '''Finds a periodic orbit by simulating until convergence and then extracting one period.

    Args:
        oscillator: The oscillator model.
        y_guess: Initial guess for a point on or near the limit cycle.
        t_sim_total: Total simulation time to allow for convergence to the limit cycle.
        t_period_guess: An approximate guess of the period of the limit cycle.
        num_points_per_period: Number of points to save for the final orbit trajectory.
        ode_solver_options: Optional dictionary of arguments for the ODE solver
                             (passed to `solve_ode`).

    Returns:
        A tuple containing:
            - orbit_solution: Diffrax solution object for one period of the orbit.
            - y_final_on_orbit: The state vector at the end of t_sim_total (a point on the orbit).
            - estimated_period: The t_period_guess used for extracting the orbit.
    '''
    if ode_solver_options is None:
        ode_solver_options = {}

    # Simulate for a long time to converge to the limit cycle
    # We only need the final state from this simulation
    convergence_sol = solve_ode(
        oscillator,
        y_guess,
        t_start=0.0,
        t_end=t_sim_total,
        saveat=diffrax.SaveAt(t1=True), # Save only at the end
        **ode_solver_options
    )
    y_on_orbit = convergence_sol.ys[-1] # Get the last state

    # Now simulate for one estimated period, starting from y_on_orbit, saving densely
    t_eval_orbit = jnp.linspace(0.0, t_period_guess, num_points_per_period)
    orbit_solution = solve_ode(
        oscillator,
        y_on_orbit,
        t_start=0.0,
        t_end=t_period_guess,
        saveat=diffrax.SaveAt(ts=t_eval_orbit),
        **ode_solver_options
    )

    return orbit_solution, y_on_orbit, t_period_guess


def _augmented_vector_field_for_monodromy(
    t: float,
    state_phi_flat: jnp.ndarray,
    args_tuple: tuple[Oscillator, int]
):
    '''Vector field for the augmented system (original state + flattened Phi matrix).'''
    oscillator, dim = args_tuple
    original_state = state_phi_flat[:dim]
    phi_flat = state_phi_flat[dim:]
    phi_matrix = phi_flat.reshape((dim, dim))

    # Original dynamics
    dy_dt = oscillator.vector_field(t, original_state, None) # Assuming args for vector_field is None or handled inside

    # Jacobian of the original vector field
    # The vector_field method takes (self, t, y, args), so we compute jacobian wrt y (arg_nums=2 for a method)
    # However, oscillator is an object, and vector_field is a method. We need to make it a static function for jacfwd
    # or pass `oscillator.vector_field` directly if it's compatible with jacfwd's argnums for `y`.
    # Let's assume we can get the jacobian J(t,y) correctly.
    # For an eqx.Module, methods are pure if all their attributes are JAX types or static.
    # `oscillator.vector_field` can be passed to `jacfwd` if `oscillator` itself is a pytree.
    # The `argnums=1` here assumes `vector_field_for_jacobian(t, y, args_for_vf)` where `y` is the second argument.
    def vf_for_jac(y_jac, t_jac, args_vf_jac): # Wrapper for jacobian calculation
        return oscillator.vector_field(t_jac, y_jac, args_vf_jac)
    
    jacobian_matrix = jacfwd(vf_for_jac)(original_state, t, None)

    # Variational equation: d(Phi)/dt = J * Phi
    dPhi_dt_matrix = jnp.dot(jacobian_matrix, phi_matrix)
    dPhi_dt_flat = dPhi_dt_matrix.flatten()

    return jnp.concatenate([dy_dt, dPhi_dt_flat])


def compute_monodromy_matrix(
    oscillator: Oscillator,
    y0_on_orbit: jnp.ndarray,
    period: float,
    ode_solver_options: dict | None = None
) -> jnp.ndarray:
    '''Computes the monodromy matrix for a periodic orbit.

    Args:
        oscillator: The oscillator model.
        y0_on_orbit: A point on the periodic orbit (initial condition).
        period: The period of the orbit.
        ode_solver_options: Optional dictionary of arguments for the ODE solver.

    Returns:
        The monodromy matrix (dim x dim JAX array).
    '''
    if ode_solver_options is None:
        ode_solver_options = {}

    dim = y0_on_orbit.shape[0]
    phi0_flat = jnp.eye(dim).flatten() # Initial condition for Phi is the identity matrix, flattened
    initial_augmented_state = jnp.concatenate([y0_on_orbit, phi0_flat])

    # Create the ODETerm for the augmented system
    # The `args` for `_augmented_vector_field_for_monodromy` will be `(oscillator, dim)`
    term = diffrax.ODETerm(_augmented_vector_field_for_monodromy)
    
    # Solve the augmented system ODE for one period
    solution = diffrax.diffeqsolve(
        term,
        solver=ode_solver_options.get("solver", diffrax.Tsit5()), # Default if not provided
        t0=0.0,
        t1=period,
        dt0=ode_solver_options.get("dt0", None),
        y0=initial_augmented_state,
        args=(oscillator, dim), # Pass oscillator and dim as args to the augmented vector field
        saveat=diffrax.SaveAt(t1=True), # We only need the final state of Phi
        stepsize_controller=ode_solver_options.get(
            "stepsize_controller", diffrax.PIDController(rtol=1e-7, atol=1e-7)
        ),
        max_steps=ode_solver_options.get("max_steps", 16**4)
    )

    final_augmented_state = solution.ys[-1]
    final_phi_flat = final_augmented_state[dim:]
    monodromy_matrix = final_phi_flat.reshape((dim, dim))

    return monodromy_matrix

def compute_floquet_multipliers(monodromy_matrix: jnp.ndarray) -> jnp.ndarray:
    '''Computes Floquet multipliers (eigenvalues of the monodromy matrix).'''
    return jnp.linalg.eigvals(monodromy_matrix) 