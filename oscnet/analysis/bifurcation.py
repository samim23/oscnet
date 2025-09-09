"""
Bifurcation Analysis

Tools for creating and analyzing bifurcation diagrams for oscillatory systems.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Any, Union, Callable
import time
from pathlib import Path
from tqdm import tqdm

from ..core.oscillators import Oscillator
from .model_interface import OscillatoryModelInterface, ParamDict, adapt_model


def _filter_constructor_parameters(oscillator_params: Dict[str, Any], oscillator_type: str = None) -> Dict[str, Any]:
    """
    Filter parameters to include only those compatible with the oscillator constructor.
    
    Args:
        oscillator_params: Dictionary of oscillator parameters
        oscillator_type: Optional oscillator type name
        
    Returns:
        Filtered dictionary with only constructor-compatible parameters
    """
    # List of known derived parameters that should not be passed to constructors
    derived_params = ['omega_squared', 'gamma_factor']
    
    # Create a new dictionary with filtered parameters
    filtered_params = {}
    
    # Filter out derived parameters and convert JAX arrays
    for k, v in oscillator_params.items():
        if k in derived_params:
            continue
            
        # Handle JAX arrays and convert them to Python floats
        try:
            if hasattr(v, 'item'):  # Check if it's a JAX array or similar
                try:
                    # Convert single-value arrays to Python float
                    filtered_params[k] = float(v.item())
                except (ValueError, AttributeError, TypeError):
                    # If item() fails, try converting to a numpy array first
                    try:
                        filtered_params[k] = float(np.array(v).item())
                    except (ValueError, AttributeError, TypeError):
                        # If that also fails, just convert to float directly
                        filtered_params[k] = float(v)
            else:
                # For non-array values, try to convert to float if it's numeric
                if isinstance(v, (int, float, np.number)):
                    filtered_params[k] = float(v)
                else:
                    filtered_params[k] = v
        except Exception as e:
            print(f"Warning: Failed to process parameter {k}={v}: {str(e)}")
            # Keep original value as a fallback
            filtered_params[k] = v
    
    # Determine oscillator type if not provided
    if oscillator_type is None and hasattr(oscillator_params.get('__class__', None), 'get_type_name'):
        oscillator_type = oscillator_params['__class__'].get_type_name()
    
    # Handle specific oscillator types
    if oscillator_type == 'horn':
        # HORN oscillator only accepts alpha, omega, gamma, h
        valid_params = ['alpha', 'omega', 'gamma', 'h']
        filtered_params = {k: v for k, v in filtered_params.items() if k in valid_params}
    elif oscillator_type == 'van_der_pol':
        # Van der Pol only accepts mu
        valid_params = ['mu']
        filtered_params = {k: v for k, v in filtered_params.items() if k in valid_params}
    
    # Print the filtered parameters for debugging
    print(f"Filtered parameters for {oscillator_type}: {filtered_params}")
    
    return filtered_params


def validate_parameters(oscillator_type: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate oscillator parameters before creating an oscillator instance.
    
    Args:
        oscillator_type: Type of oscillator ('horn', 'van_der_pol', etc.)
        parameters: Dictionary of parameters to validate
        
    Returns:
        Tuple of (valid, message) where valid is a boolean and message is a string
    """
    # Check for required parameters based on oscillator type
    if oscillator_type == 'horn':
        required_params = ['alpha', 'omega', 'gamma', 'h']
        for param in required_params:
            if param not in parameters:
                return False, f"Missing required parameter '{param}' for HORN oscillator"
                
        # HORN-specific validations
        if 'alpha' in parameters and parameters['alpha'] <= 0:
            return False, f"Parameter 'alpha' must be positive, got {parameters['alpha']}"
            
        if 'omega' in parameters and parameters['omega'] <= 0:
            return False, f"Parameter 'omega' must be positive, got {parameters['omega']}"
            
        if 'gamma' in parameters and parameters['gamma'] < 0:
            return False, f"Parameter 'gamma' must be non-negative, got {parameters['gamma']}"
            
    elif oscillator_type == 'van_der_pol':
        if 'mu' not in parameters:
            return False, f"Missing required parameter 'mu' for Van der Pol oscillator"
            
        # Van der Pol-specific validations
        if 'mu' in parameters and parameters['mu'] <= 0:
            return False, f"Parameter 'mu' must be positive, got {parameters['mu']}"
    
    # No validation errors
    return True, "Parameters are valid"


def create_bifurcation_diagram(
    oscillator: Oscillator,
    param_name: str,
    param_range: Tuple[float, float],
    n_points: int = 50,
    t_span: Tuple[float, float] = (0.0, 100.0),
    dt: float = 0.1,
    transient_time: float = 50.0,
    input_force: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a bifurcation diagram for an oscillator parameter.
    
    Args:
        oscillator: The oscillator to analyze.
        param_name: Name of the parameter to vary.
        param_range: Range of parameter values (min, max).
        n_points: Number of parameter values to sample.
        t_span: Time span for simulation.
        dt: Time step for integration.
        transient_time: Time to discard as transient behavior.
        input_force: Optional input force to the oscillator.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure with the bifurcation diagram.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate parameter values
    min_val, max_val = param_range
    param_values = jnp.linspace(min_val, max_val, n_points)
    
    # Generate time points
    t_start, t_end = t_span
    t = jnp.arange(t_start, t_end, dt)
    
    # Find index corresponding to transient time
    transient_idx = int(transient_time / dt) if transient_time > 0 else 0
    
    # Setup initial conditions
    initial_state = jnp.zeros(oscillator.state_dim)
    
    # Define args for vector field
    args = input_force if input_force is not None else 0.0
    
    # Determine oscillator type if possible
    oscillator_type = None
    if hasattr(oscillator, 'get_type_name'):
        oscillator_type = oscillator.get_type_name()
    else:
        # Try to infer from class name
        osc_class_name = oscillator.__class__.__name__.lower()
        if 'horn' in osc_class_name:
            oscillator_type = 'horn'
        elif 'vanderpol' in osc_class_name or 'van_der_pol' in osc_class_name:
            oscillator_type = 'van_der_pol'
            
    if oscillator_type is None:
        print("Warning: Could not determine oscillator type. Parameter validation may be incomplete.")
        oscillator_type = 'unknown'
    
    # Track successful parameter values for plotting
    plotted_params = []
    peaks_data = []
    
    # For each parameter value
    for param_idx, param_val in enumerate(tqdm(param_values, desc=f"Computing bifurcation diagram for {param_name}")):
        try:
            # Update oscillator parameter (make a deep copy to avoid modifying original)
            oscillator_params = dict(oscillator.params)
            
            # Convert param_val to a Python float, handling JAX arrays
            if hasattr(param_val, 'item'):
                param_val = float(param_val.item())
                
            # Update the parameter
            oscillator_params[param_name] = param_val
            
            # Filter parameters for constructor compatibility
            filtered_params = _filter_constructor_parameters(oscillator_params, oscillator_type)
            
            # Validate parameters before creating oscillator
            valid, message = validate_parameters(oscillator_type, filtered_params)
            if not valid:
                print(f"Skipping param value {param_name}={param_val}: {message}")
                continue
            
            # Create updated oscillator
            try:
                updated_oscillator = type(oscillator)(**filtered_params)
            except Exception as e:
                print(f"Error creating oscillator with parameters {filtered_params}: {str(e)}")
                continue
            
            # Define ODE function
            def ode_system(state, t):
                return updated_oscillator.vector_field(t, state, args)
            
            # Solve ODE using forward Euler (for simplicity)
            states = [initial_state]
            curr_state = initial_state
            
            for i in range(1, len(t)):
                try:
                    # Forward Euler step
                    next_state = curr_state + dt * ode_system(curr_state, t[i-1])
                    
                    # Check for NaN or infinity
                    if jnp.any(jnp.isnan(next_state)) or jnp.any(jnp.isinf(next_state)):
                        print(f"Warning: NaN or Inf encountered at t={t[i-1]} for {param_name}={param_val}. Stopping simulation.")
                        break
                        
                    curr_state = next_state
                    states.append(curr_state)
                except Exception as e:
                    print(f"Error in ODE step at t={t[i-1]} for {param_name}={param_val}: {str(e)}")
                    break
            
            # Convert to array
            states = jnp.array(states)
            
            # Skip if we don't have enough data
            if len(states) <= transient_idx + 5:  # Need at least a few points beyond transient
                print(f"Not enough data points for {param_name}={param_val}, skipping")
                continue
                
            # Skip transient part
            steady_states = states[transient_idx:]
            
            # Determine peaks (local maxima) for the first state variable
            peaks = []
            for i in range(1, len(steady_states) - 1):
                if (steady_states[i, 0] > steady_states[i-1, 0] and 
                    steady_states[i, 0] > steady_states[i+1, 0]):
                    peaks.append(steady_states[i, 0])
            
            # Store successful parameter value and peaks
            if peaks:
                plotted_params.append(param_val)
                peaks_data.append(peaks)
                
                # Plot peaks at this parameter value
                for peak in peaks:
                    ax.plot(param_val, peak, 'k.', markersize=1)
                    
        except Exception as e:
            print(f"Error processing {param_name}={param_val}: {str(e)}")
            continue
    
    # Add debug info about successes
    successful_percentage = len(plotted_params) / len(param_values) * 100
    print(f"Successfully processed {len(plotted_params)}/{len(param_values)} parameter values ({successful_percentage:.1f}%)")
    
    if not plotted_params:
        ax.text(0.5, 0.5, "No valid data points generated", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='red')
    
    # Set labels and title
    ax.set_xlabel(param_name)
    ax.set_ylabel("State Variable Peaks")
    ax.set_title(f"Bifurcation Diagram: {param_name}")
    ax.grid(True, alpha=0.3)
    
    return fig


def analyze_parameter_space(
    oscillator: Oscillator,
    param1_name: str,
    param1_range: Tuple[float, float],
    param2_name: str,
    param2_range: Tuple[float, float],
    metric: str = "lyapunov",
    n_points: int = 10,
    t_span: Tuple[float, float] = (0.0, 50.0),
    dt: float = 0.1,
    input_force: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Analyze the parameter space of an oscillator.
    
    Args:
        oscillator: The oscillator to analyze.
        param1_name: Name of the first parameter to vary.
        param1_range: Range of first parameter values (min, max).
        param2_name: Name of the second parameter to vary.
        param2_range: Range of second parameter values (min, max).
        metric: Metric to compute ('lyapunov', 'amplitude', 'frequency', 'complexity').
        n_points: Number of parameter values to sample per dimension.
        t_span: Time span for simulation.
        dt: Time step for integration.
        input_force: Optional input force to the oscillator.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure with the parameter space analysis.
    """
    from .edge_of_chaos import compute_largest_lyapunov_exponent
    
    min1, max1 = param1_range
    min2, max2 = param2_range
    
    param1_values = jnp.linspace(min1, max1, n_points)
    param2_values = jnp.linspace(min2, max2, n_points)
    
    metric_grid = jnp.zeros((n_points, n_points))
    valid_data_mask = jnp.zeros((n_points, n_points), dtype=bool)
    
    # Determine oscillator type if possible
    oscillator_type = None
    if hasattr(oscillator, 'get_type_name'):
        oscillator_type = oscillator.get_type_name()
    else:
        # Try to infer from class name
        osc_class_name = oscillator.__class__.__name__.lower()
        if 'horn' in osc_class_name:
            oscillator_type = 'horn'
        elif 'vanderpol' in osc_class_name or 'van_der_pol' in osc_class_name:
            oscillator_type = 'van_der_pol'
    
    if oscillator_type is None:
        print("Warning: Could not determine oscillator type. Parameter validation may be incomplete.")
        oscillator_type = 'unknown'
    
    # Count successful computations
    success_count = 0
    
    # For each parameter combination
    for i, p1 in enumerate(tqdm(param1_values, desc=f"Analyzing parameter space")):
        for j, p2 in enumerate(param2_values):
            try:
                # Convert parameters to Python floats, handling JAX arrays
                if hasattr(p1, 'item'):
                    p1 = float(p1.item())
                if hasattr(p2, 'item'):
                    p2 = float(p2.item())
                
                # Update oscillator parameters
                oscillator_params = dict(oscillator.params)
                oscillator_params[param1_name] = p1
                oscillator_params[param2_name] = p2
                
                # Filter parameters for constructor compatibility
                filtered_params = _filter_constructor_parameters(oscillator_params, oscillator_type)
                
                # Validate parameters before creating oscillator
                valid, message = validate_parameters(oscillator_type, filtered_params)
                if not valid:
                    print(f"Skipping params {param1_name}={p1}, {param2_name}={p2}: {message}")
                    metric_grid = metric_grid.at[i, j].set(float('nan'))
                    continue
                
                # Create updated oscillator
                try:
                    updated_oscillator = type(oscillator)(**filtered_params)
                except Exception as e:
                    print(f"Error creating oscillator with parameters {filtered_params}: {str(e)}")
                    metric_grid = metric_grid.at[i, j].set(float('nan'))
                    continue
                
                # Compute metric based on selection
                try:
                    if metric == "lyapunov":
                        # Compute Lyapunov exponent
                        metric_value = compute_largest_lyapunov_exponent(
                            updated_oscillator,
                            t_span=t_span,
                            dt=dt,
                            input_force=input_force
                        )
                    elif metric == "amplitude":
                        # Simulate and compute amplitude
                        metric_value = _compute_oscillation_amplitude(
                            updated_oscillator,
                            t_span=t_span,
                            dt=dt,
                            input_force=input_force
                        )
                    elif metric == "frequency":
                        # Simulate and compute frequency
                        metric_value = _compute_oscillation_frequency(
                            updated_oscillator,
                            t_span=t_span,
                            dt=dt,
                            input_force=input_force
                        )
                    elif metric == "complexity":
                        # Compute a complexity metric
                        metric_value = _compute_oscillation_complexity(
                            updated_oscillator,
                            t_span=t_span,
                            dt=dt,
                            input_force=input_force
                        )
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    
                    # Check for invalid values
                    if jnp.isnan(metric_value) or jnp.isinf(metric_value):
                        print(f"Invalid metric value ({metric_value}) for {param1_name}={p1}, {param2_name}={p2}")
                        metric_grid = metric_grid.at[i, j].set(float('nan'))
                    else:
                        # Valid data point
                        metric_grid = metric_grid.at[i, j].set(metric_value)
                        valid_data_mask = valid_data_mask.at[i, j].set(True)
                        success_count += 1
                        
                except Exception as e:
                    print(f"Error computing {metric} for {param1_name}={p1}, {param2_name}={p2}: {str(e)}")
                    metric_grid = metric_grid.at[i, j].set(float('nan'))
            except Exception as e:
                print(f"Unexpected error for {param1_name}={p1}, {param2_name}={p2}: {str(e)}")
                metric_grid = metric_grid.at[i, j].set(float('nan'))
    
    # Print success rate
    total_points = n_points * n_points
    success_rate = (success_count / total_points) * 100
    print(f"Successfully computed metrics for {success_count}/{total_points} parameter combinations ({success_rate:.1f}%)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create meshgrid for plotting
    X, Y = jnp.meshgrid(param1_values, param2_values)
    
    # Check if we have any valid data
    if success_count == 0:
        ax.text(0.5, 0.5, "No valid data points generated", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='red')
        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        ax.set_title(f"Parameter Space Analysis: {metric.capitalize()} (Failed)")
        return fig
    
    # Apply mask to handle NaN values better
    metric_grid_masked = jnp.where(valid_data_mask, metric_grid, jnp.nan)
    
    # For better visualization, clip extreme values
    if success_count > 0:
        # Get valid values
        valid_values = metric_grid_masked[~jnp.isnan(metric_grid_masked)]
        if len(valid_values) > 0:
            # Compute statistics on valid data
            vmin = jnp.nanpercentile(valid_values, 5)  # 5th percentile
            vmax = jnp.nanpercentile(valid_values, 95)  # 95th percentile
            
            # Clip data for visualization
            clipped_data = jnp.clip(metric_grid_masked, vmin, vmax)
            
            # Create contour plot with clipped data
            levels = 20
            contour = ax.contourf(X, Y, clipped_data.T, levels=levels, cmap='viridis', 
                                  alpha=0.8, vmin=vmin, vmax=vmax, extend='both')
            
            # Add colorbar
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label(metric.capitalize())
            
            # Add contour lines
            ax.contour(X, Y, clipped_data.T, levels=levels, colors='k', alpha=0.3, linewidths=0.5)
    else:
        # Fallback if stats computation fails
        contour = ax.contourf(X, Y, metric_grid.T, levels=20, cmap='viridis')
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(metric.capitalize())
    
    # Set labels and title
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_title(f"Parameter Space Analysis: {metric.capitalize()}")
    ax.grid(True, alpha=0.3)
    
    return fig


def _compute_oscillation_amplitude(
    oscillator: Oscillator,
    t_span: Tuple[float, float] = (0.0, 100.0),
    dt: float = 0.1,
    transient_time: float = 50.0,
    input_force: Optional[float] = None
) -> float:
    """Compute the amplitude of oscillations."""
    # Generate time points
    t_start, t_end = t_span
    t = jnp.arange(t_start, t_end, dt)
    
    # Find index corresponding to transient time
    transient_idx = int(transient_time / dt) if transient_time > 0 else 0
    
    # Setup initial conditions
    initial_state = jnp.zeros(oscillator.state_dim)
    
    # Define args for vector field
    args = input_force if input_force is not None else 0.0
    
    # Define ODE function
    def ode_system(state, t):
        return oscillator.vector_field(t, state, args)
    
    # Solve ODE using forward Euler
    states = [initial_state]
    curr_state = initial_state
    
    for i in range(1, len(t)):
        # Forward Euler step
        curr_state = curr_state + dt * ode_system(curr_state, t[i-1])
        states.append(curr_state)
    
    # Convert to array
    states = jnp.array(states)
    
    # Skip transient part
    steady_states = states[transient_idx:]
    
    # Compute amplitude (max - min)
    amplitude = jnp.max(steady_states[:, 0]) - jnp.min(steady_states[:, 0])
    
    return float(amplitude)


def _compute_oscillation_frequency(
    oscillator: Oscillator,
    t_span: Tuple[float, float] = (0.0, 100.0),
    dt: float = 0.1,
    transient_time: float = 50.0,
    input_force: Optional[float] = None
) -> float:
    """Compute the frequency of oscillations using zero crossings."""
    # Generate time points
    t_start, t_end = t_span
    t = jnp.arange(t_start, t_end, dt)
    
    # Find index corresponding to transient time
    transient_idx = int(transient_time / dt) if transient_time > 0 else 0
    
    # Setup initial conditions
    initial_state = jnp.zeros(oscillator.state_dim)
    
    # Define args for vector field
    args = input_force if input_force is not None else 0.0
    
    # Define ODE function
    def ode_system(state, t):
        return oscillator.vector_field(t, state, args)
    
    # Solve ODE using forward Euler
    states = [initial_state]
    curr_state = initial_state
    
    for i in range(1, len(t)):
        # Forward Euler step
        curr_state = curr_state + dt * ode_system(curr_state, t[i-1])
        states.append(curr_state)
    
    # Convert to array
    states = jnp.array(states)
    
    # Skip transient part
    steady_states = states[transient_idx:]
    steady_t = t[transient_idx:]
    
    # Find zero crossings (positive to negative)
    zero_crossings = 0
    for i in range(1, len(steady_states)):
        if steady_states[i-1, 0] >= 0 and steady_states[i, 0] < 0:
            zero_crossings += 1
    
    # Compute frequency (cycles per time unit)
    if zero_crossings > 0:
        total_time = steady_t[-1] - steady_t[0]
        frequency = zero_crossings / total_time
    else:
        frequency = 0.0
    
    return float(frequency)


def _compute_oscillation_complexity(
    oscillator: Oscillator,
    t_span: Tuple[float, float] = (0.0, 100.0),
    dt: float = 0.1,
    transient_time: float = 50.0,
    input_force: Optional[float] = None
) -> float:
    """
    Compute a complexity metric based on spectral analysis.
    
    Uses the number of peaks in the power spectrum as a complexity measure.
    """
    # Generate time points
    t_start, t_end = t_span
    t = jnp.arange(t_start, t_end, dt)
    
    # Find index corresponding to transient time
    transient_idx = int(transient_time / dt) if transient_time > 0 else 0
    
    # Setup initial conditions
    initial_state = jnp.zeros(oscillator.state_dim)
    
    # Define args for vector field
    args = input_force if input_force is not None else 0.0
    
    # Define ODE function
    def ode_system(state, t):
        return oscillator.vector_field(t, state, args)
    
    # Solve ODE using forward Euler
    states = [initial_state]
    curr_state = initial_state
    
    for i in range(1, len(t)):
        # Forward Euler step
        curr_state = curr_state + dt * ode_system(curr_state, t[i-1])
        states.append(curr_state)
    
    # Convert to array
    states = jnp.array(states)
    
    # Skip transient part
    steady_states = states[transient_idx:, 0]  # First state variable
    
    # Compute FFT
    spectrum = jnp.abs(jnp.fft.fft(steady_states))
    
    # Count peaks in spectrum
    peak_count = 0
    threshold = 0.1 * jnp.max(spectrum)  # 10% of maximum as threshold
    
    for i in range(1, len(spectrum) // 2 - 1):  # Only first half is meaningful
        if (spectrum[i] > spectrum[i-1] and 
            spectrum[i] > spectrum[i+1] and 
            spectrum[i] > threshold):
            peak_count += 1
    
    # Normalize to [0, 1] range (assuming max 10 peaks is maximum complexity)
    complexity = min(1.0, peak_count / 10.0)
    
    return float(complexity)


# High-level interface functions

def plot(
    model: Any,
    param: str,
    param_range: Optional[Tuple[float, float]] = None,
    n_points: int = 50,
    output_dir: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a bifurcation diagram for a model parameter.
    
    Args:
        model: The model to analyze (will be adapted to OscillatoryModelInterface).
        param: Name of the parameter to vary.
        param_range: Range of parameter values (min, max). If None, a default range will be used.
        n_points: Number of parameter values to sample.
        output_dir: Directory to save the plot (None to skip saving).
        show: Whether to display the plot.
        
    Returns:
        Matplotlib figure with the bifurcation diagram.
    """
    # Adapt model to interface
    adapted_model = adapt_model(model)
    
    # Extract current parameters
    current_params = adapted_model.extract_parameters()
    
    # List of derived parameters that shouldn't be directly varied
    derived_params = ['omega_squared', 'gamma_factor']
    
    # Check if param is a derived parameter
    if param in derived_params:
        # Map to base parameter
        if param == 'omega_squared':
            param = 'omega'
            print(f"Warning: 'omega_squared' is a derived parameter. Using 'omega' instead.")
            # If we have a range for omega_squared, convert it to omega
            if param_range is not None:
                min_val, max_val = param_range
                param_range = (jnp.sqrt(min_val), jnp.sqrt(max_val))
        elif param == 'gamma_factor':
            param = 'gamma'
            print(f"Warning: 'gamma_factor' is a derived parameter. Using 'gamma' instead.")
            # If we have a range for gamma_factor, convert it to gamma
            if param_range is not None:
                min_val, max_val = param_range
                param_range = (min_val / 2.0, max_val / 2.0)
    
    if param not in current_params and param not in derived_params:
        # Try to find alternative parameters in the model
        available_params = list(current_params.keys())
        raise ValueError(f"Parameter '{param}' not found in model. Available parameters: {available_params}")
    
    # Define default parameter range if not provided
    if param_range is None:
        # Default ranges based on parameter name
        default_ranges = {
            "alpha": (0.01, 1.0),
            "omega": (0.01, 2.0),
            "gamma": (0.01, 0.5),
            "h": (0.5, 2.0),
            "mu": (0.1, 5.0)
        }
        
        if param in default_ranges:
            param_range = default_ranges[param]
        else:
            # Use a range around current value
            if param in current_params:
                current_val = current_params[param]
                # Ensure current_val is a scalar
                if hasattr(current_val, 'shape') and current_val.shape:
                    current_val = float(current_val.flatten()[0])
                # Ensure positive range
                min_val = max(0.01, 0.5 * current_val) if current_val > 0 else 0.01
                max_val = 2.0 * current_val if current_val > 0 else 1.0
                param_range = (min_val, max_val)
            else:
                # Fallback default range
                param_range = (0.01, 1.0)
    
    # Get an oscillator for bifurcation analysis
    from oscnet.core.oscillators import HORNOscillator
    
    # Filter parameters for constructor
    filtered_params = _filter_constructor_parameters(current_params, 'horn')
    
    # Create oscillator with filtered parameters
    try:
        oscillator = HORNOscillator(**filtered_params)
    except Exception as e:
        # Fallback to default parameters if constructor fails
        print(f"Failed to create oscillator with parameters {filtered_params}: {e}")
        print("Using default parameters instead.")
        oscillator = HORNOscillator()
    
    # Create bifurcation diagram
    fig = create_bifurcation_diagram(
        oscillator=oscillator,
        param_name=param,
        param_range=param_range,
        n_points=n_points,
        t_span=(0.0, 50.0),  # Shorter time for faster execution
        dt=0.1,
        transient_time=25.0  # Half of simulation time
    )
    
    # Save figure if requested
    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"bifurcation_{param}.png"
        fig.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def parameter_grid(
    model: Any,
    param1: str,
    param2: str,
    param1_range: Optional[Tuple[float, float]] = None,
    param2_range: Optional[Tuple[float, float]] = None,
    metric: str = "lyapunov",
    n_points: int = 10,
    output_dir: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Analyze a 2D parameter space of a model.
    
    Args:
        model: The model to analyze (will be adapted to OscillatoryModelInterface).
        param1: Name of the first parameter to vary.
        param2: Name of the second parameter to vary.
        param1_range: Range of first parameter values (min, max). If None, a default range will be used.
        param2_range: Range of second parameter values (min, max). If None, a default range will be used.
        metric: Metric to compute ('lyapunov', 'amplitude', 'frequency', 'complexity').
        n_points: Number of parameter values to sample per dimension.
        output_dir: Directory to save the plot (None to skip saving).
        show: Whether to display the plot.
        
    Returns:
        Matplotlib figure with the parameter space analysis.
    """
    # Adapt model to interface
    adapted_model = adapt_model(model)
    
    # Extract current parameters
    current_params = adapted_model.extract_parameters()
    
    # List of derived parameters that shouldn't be directly varied
    derived_params = ['omega_squared', 'gamma_factor']
    
    # Check if param1 is a derived parameter
    if param1 in derived_params:
        # Map to base parameter
        if param1 == 'omega_squared':
            param1 = 'omega'
            print(f"Warning: 'omega_squared' is a derived parameter. Using 'omega' instead.")
            # If we have a range for omega_squared, convert it to omega
            if param1_range is not None:
                min_val, max_val = param1_range
                param1_range = (jnp.sqrt(min_val), jnp.sqrt(max_val))
        elif param1 == 'gamma_factor':
            param1 = 'gamma'
            print(f"Warning: 'gamma_factor' is a derived parameter. Using 'gamma' instead.")
            # If we have a range for gamma_factor, convert it to gamma
            if param1_range is not None:
                min_val, max_val = param1_range
                param1_range = (min_val / 2.0, max_val / 2.0)
    
    # Check if param2 is a derived parameter
    if param2 in derived_params:
        # Map to base parameter
        if param2 == 'omega_squared':
            param2 = 'omega'
            print(f"Warning: 'omega_squared' is a derived parameter. Using 'omega' instead.")
            # If we have a range for omega_squared, convert it to omega
            if param2_range is not None:
                min_val, max_val = param2_range
                param2_range = (jnp.sqrt(min_val), jnp.sqrt(max_val))
        elif param2 == 'gamma_factor':
            param2 = 'gamma'
            print(f"Warning: 'gamma_factor' is a derived parameter. Using 'gamma' instead.")
            # If we have a range for gamma_factor, convert it to gamma
            if param2_range is not None:
                min_val, max_val = param2_range
                param2_range = (min_val / 2.0, max_val / 2.0)
    
    if param1 not in current_params and param1 not in derived_params:
        raise ValueError(f"Parameter '{param1}' not found in model. Available parameters: {list(current_params.keys())}")
    
    if param2 not in current_params and param2 not in derived_params:
        raise ValueError(f"Parameter '{param2}' not found in model. Available parameters: {list(current_params.keys())}")
    
    # Define default parameter ranges if not provided
    default_ranges = {
        "alpha": (0.01, 1.0),
        "omega": (0.01, 2.0),
        "gamma": (0.01, 0.5),
        "h": (0.5, 2.0),
        "mu": (0.1, 5.0)
    }
    
    if param1_range is None:
        if param1 in default_ranges:
            param1_range = default_ranges[param1]
        else:
            # Use a range around current value
            current_val = current_params.get(param1)
            if current_val is not None:
                # Ensure current_val is a scalar
                if hasattr(current_val, 'shape') and current_val.shape:
                    current_val = float(current_val.flatten()[0])
                # Ensure positive range
                min_val = max(0.01, 0.5 * current_val) if current_val > 0 else 0.01
                max_val = 2.0 * current_val if current_val > 0 else 1.0
                param1_range = (min_val, max_val)
            else:
                param1_range = (0.01, 1.0)
    
    if param2_range is None:
        if param2 in default_ranges:
            param2_range = default_ranges[param2]
        else:
            # Use a range around current value
            current_val = current_params.get(param2)
            if current_val is not None:
                # Ensure current_val is a scalar
                if hasattr(current_val, 'shape') and current_val.shape:
                    current_val = float(current_val.flatten()[0])
                # Ensure positive range
                min_val = max(0.01, 0.5 * current_val) if current_val > 0 else 0.01
                max_val = 2.0 * current_val if current_val > 0 else 1.0
                param2_range = (min_val, max_val)
            else:
                param2_range = (0.01, 1.0)
    
    # Get an oscillator for analysis
    from oscnet.core.oscillators import HORNOscillator
    
    # Filter parameters for constructor
    filtered_params = _filter_constructor_parameters(current_params, 'horn')
    
    # Create oscillator with filtered parameters
    try:
        oscillator = HORNOscillator(**filtered_params)
    except Exception as e:
        # Fallback to default parameters if constructor fails
        print(f"Failed to create oscillator with parameters {filtered_params}: {e}")
        print("Using default parameters instead.")
        oscillator = HORNOscillator()
    
    # Create parameter space analysis
    fig = analyze_parameter_space(
        oscillator=oscillator,
        param1_name=param1,
        param1_range=param1_range,
        param2_name=param2,
        param2_range=param2_range,
        metric=metric,
        n_points=n_points,
        t_span=(0.0, 30.0),  # Shorter time for faster execution
        dt=0.1
    )
    
    # Save figure if requested
    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"parameter_grid_{param1}_{param2}_{metric}.png"
        fig.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig 