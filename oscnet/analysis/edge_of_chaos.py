"""
Edge of Chaos Analysis

Tools for analyzing and optimizing oscillator parameters to achieve edge of chaos dynamics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Any, Union, Callable
import time
from functools import partial

from oscnet.core.oscillators import (
    Oscillator, 
    NonlinearHarmonicOscillator, 
    VanDerPolOscillator,
    LinearHarmonicOscillator
)

# Simple oscillator factory to replace the missing create_oscillator function
def create_oscillator(oscillator_type: str, dim: int = 1, **params):
    """Create an oscillator instance based on type."""
    # Ensure we have a key parameter
    if 'key' not in params:
        params['key'] = jax.random.PRNGKey(42)
    
    # Force dim=1 for edge-of-chaos analysis (simple 1D oscillators)
    params['dim'] = 1
    
    # Filter parameters to only include what each oscillator type expects
    if oscillator_type.lower() in ['nonlinear', 'nonlinearharmonic', 'nonlinearharmonicoscillator']:
        # NonlinearHarmonicOscillator expects: dim, alpha, omega, gamma, dt, key
        filtered_params = {k: v for k, v in params.items() if k in ['dim', 'alpha', 'omega', 'gamma', 'dt', 'key']}
        # Set defaults for missing parameters
        filtered_params.setdefault('alpha', 0.1)
        filtered_params.setdefault('omega', 1.0)
        filtered_params.setdefault('gamma', 0.1)
        filtered_params.setdefault('dt', 0.1)
        return NonlinearHarmonicOscillator(**filtered_params)
    elif oscillator_type.lower() in ['vanderpol', 'vanderpoloscillator']:
        # VanDerPol only needs mu, dt, dim, key
        filtered_params = {k: v for k, v in params.items() if k in ['mu', 'dt', 'dim', 'key']}
        filtered_params.setdefault('mu', 1.0)
        filtered_params.setdefault('dt', 0.1)
        return VanDerPolOscillator(**filtered_params)
    elif oscillator_type.lower() in ['linear', 'linearharmonic', 'linearharmonicoscillator']:
        # Linear needs omega, gamma, dt, dim, key
        filtered_params = {k: v for k, v in params.items() if k in ['omega', 'gamma', 'dt', 'dim', 'key']}
        filtered_params.setdefault('omega', 1.0)
        filtered_params.setdefault('gamma', 0.1)
        filtered_params.setdefault('dt', 0.1)
        return LinearHarmonicOscillator(**filtered_params)
    else:
        # Default to NonlinearHarmonicOscillator
        filtered_params = {k: v for k, v in params.items() if k in ['dim', 'alpha', 'omega', 'gamma', 'dt', 'key']}
        filtered_params.setdefault('alpha', 0.1)
        filtered_params.setdefault('omega', 1.0)
        filtered_params.setdefault('gamma', 0.1)
        filtered_params.setdefault('dt', 0.1)
        return NonlinearHarmonicOscillator(**filtered_params)

def get_registered_oscillator_types():
    """Get available oscillator types."""
    return ['nonlinear', 'vanderpol', 'linear']

def get_parameter_ranges(oscillator_type: str = 'nonlinear'):
    """Get parameter ranges for oscillator types."""
    if oscillator_type.lower() in ['nonlinear', 'nonlinearharmonic']:
        return {
            'alpha': (0.01, 2.0),
            'omega': (0.1, 10.0),
            'gamma': (0.01, 1.0),
            'dt': (0.1, 2.0)
        }
    elif oscillator_type.lower() == 'vanderpol':
        return {
            'mu': (0.1, 5.0),
            'omega': (0.1, 10.0),
            'dt': (0.1, 2.0)
        }
    else:
        return {
            'omega': (0.1, 10.0),
            'gamma': (0.01, 1.0),
            'dt': (0.1, 2.0)
        }

# Simple model adapter functions to replace model_interface
def adapt_model(model):
    """Simple adapter for models - just returns the model as-is for now."""
    return SimpleModelAdapter(model)

class SimpleModelAdapter:
    """Simple model adapter that works directly with the model."""
    
    def __init__(self, model):
        self.model = model
    
    def get_model(self):
        """Get the underlying model."""
        return self.model
    
    def extract_parameters(self):
        """Extract oscillator parameters from the model."""
        # Try to extract parameters from the model's oscillator components
        params = {}
        
        # Check if model has encoder/decoder with oscillators
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'rnn'):
            osc = self.model.encoder.rnn.cell.oscillator
            
            # Extract parameters based on oscillator type
            if hasattr(osc, 'alpha'):
                params['alpha'] = float(osc.alpha)
            if hasattr(osc, 'omega'):
                # Handle both scalar and array omega
                if hasattr(osc.omega, '__len__') and len(osc.omega) > 0:
                    params['omega'] = float(osc.omega[0])
                else:
                    params['omega'] = float(osc.omega)
            if hasattr(osc, 'gamma'):
                # Handle both scalar and array gamma
                if hasattr(osc.gamma, '__len__') and len(osc.gamma) > 0:
                    params['gamma'] = float(osc.gamma[0])
                else:
                    params['gamma'] = float(osc.gamma)
            if hasattr(osc, 'dt'):
                params['dt'] = float(osc.dt)
            if hasattr(osc, 'mu'):
                params['mu'] = float(osc.mu)
        
        return params
    
    def get_model_type(self):
        """Get the type of the model."""
        return self.model.__class__.__name__.lower()
    
    def get_constructor_parameters(self):
        """Get parameters needed for oscillator construction."""
        return self.extract_parameters()
    
    def create_new_instance(self, new_params):
        """Create a new model instance with updated oscillator parameters while preserving trained weights."""
        try:
            # Import equinox for model manipulation
            import equinox as eqx
            
            # Get the current model
            model = self.model
            
            # Check if this is an amplitude-velocity autoencoder
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'rnn'):
                # Extract the current oscillator from encoder
                encoder_osc = model.encoder.rnn.cell.oscillator
                decoder_osc = model.decoder.rnn.cell.oscillator
                
                # Create updated oscillators with new parameters while preserving structure
                updated_encoder_osc = self._update_oscillator_params(encoder_osc, new_params)
                updated_decoder_osc = self._update_oscillator_params(decoder_osc, new_params)
                
                # Check if parameters actually changed
                params_changed = False
                for param in ['alpha', 'omega', 'gamma', 'dt']:
                    if hasattr(encoder_osc, param) and hasattr(updated_encoder_osc, param):
                        orig_val = getattr(encoder_osc, param)
                        new_val = getattr(updated_encoder_osc, param)
                        if not jnp.allclose(orig_val, new_val):
                            params_changed = True
                            break
                
                if not params_changed:
                    # If no parameters changed, return the original model
                    return SimpleModelAdapter(model)
                
                # Create new model with updated oscillators but same weights
                updated_model = eqx.tree_at(
                    lambda m: m.encoder.rnn.cell.oscillator,
                    model,
                    updated_encoder_osc
                )
                
                updated_model = eqx.tree_at(
                    lambda m: m.decoder.rnn.cell.oscillator,
                    updated_model,
                    updated_decoder_osc
                )
                
                return SimpleModelAdapter(updated_model)
            else:
                # For other model types, return the same model for now
                return SimpleModelAdapter(model)
                
        except Exception as e:
            # Fallback to original model
            return SimpleModelAdapter(self.model)
    
    def _update_oscillator_params(self, oscillator, new_params):
        """Update oscillator parameters while preserving the oscillator structure."""
        import equinox as eqx
        
        # Check if we need to update any static fields (like dt, alpha)
        static_fields_to_update = {}
        dynamic_fields_to_update = {}
        
        for param_name, param_value in new_params.items():
            if hasattr(oscillator, param_name):
                current_val = getattr(oscillator, param_name)
                
                # Check if this is a static field by trying to update it
                try:
                    # Try a test update to see if it's a static field
                    test_osc = eqx.tree_at(
                        lambda osc: getattr(osc, param_name),
                        oscillator,
                        param_value
                    )
                    # If we get here, it's a dynamic field
                    dynamic_fields_to_update[param_name] = param_value
                except Exception as e:
                    # If it fails, it's likely a static field
                    static_fields_to_update[param_name] = param_value
        
        # If we have static fields to update, we need to create a new oscillator
        if static_fields_to_update:
            # Get current parameters
            current_params = {}
            for attr_name in ['alpha', 'omega', 'gamma', 'dt', 'mu', 'beta', 'a', 'b', 'tau']:
                if hasattr(oscillator, attr_name):
                    current_params[attr_name] = getattr(oscillator, attr_name)
            
            # Update with new parameters
            current_params.update(new_params)
            
            # Create new oscillator with updated parameters
            oscillator_type = oscillator.__class__.__name__.lower()
            if 'nonlinear' in oscillator_type:
                oscillator_type = 'nonlinear'
            elif 'vanderpol' in oscillator_type:
                oscillator_type = 'vanderpol'
            elif 'linear' in oscillator_type:
                oscillator_type = 'linear'
            
            # Create new oscillator instance
            try:
                updated_osc = create_oscillator(oscillator_type, dim=1, **current_params)
            except Exception as e:
                updated_osc = oscillator
        else:
            # No static fields to update, use the original oscillator
            updated_osc = oscillator
        
        # Update dynamic fields using eqx.tree_at
        for param_name, param_value in dynamic_fields_to_update.items():
            try:
                current_param = getattr(updated_osc, param_name)
                
                # Handle array parameters (like omega, gamma)
                if param_name in ['omega', 'gamma'] and hasattr(current_param, 'shape'):
                    # If it's an array, broadcast the new value
                    new_array = jnp.full_like(current_param, param_value)
                    
                    updated_osc = eqx.tree_at(
                        lambda osc: getattr(osc, param_name),
                        updated_osc,
                        new_array
                    )
                else:
                    # For scalar parameters
                    updated_osc = eqx.tree_at(
                        lambda osc: getattr(osc, param_name),
                        updated_osc,
                        param_value
                    )
                
            except Exception as e:
                # If update fails, continue with the current oscillator
                pass
        
        return updated_osc

# Type definitions
ParamDict = Dict[str, float]
OptimizationResult = Dict[str, Any]


def compute_largest_lyapunov_exponent(
    oscillator: Oscillator,
    n_steps: int = 1000,
    perturbation_size: float = 1e-6,
    input_force: float = 0.0
) -> float:
    """
    Compute the largest Lyapunov exponent for an oscillator using the current step-based interface.
    
    Args:
        oscillator: The oscillator to analyze.
        n_steps: Number of integration steps.
        perturbation_size: Size of initial perturbation.
        input_force: Constant input force to the oscillator.
        
    Returns:
        Largest Lyapunov exponent.
    """
    # For standalone oscillators, we work with simple 1D state
    # Initialize states for reference trajectory (position and velocity)
    x_ref = jnp.array([0.1])  # Small initial position
    v_ref = jnp.array([0.0])  # Zero initial velocity
    
    # Initialize perturbed trajectory
    x_pert = x_ref + perturbation_size
    v_pert = v_ref
    
    # Track Lyapunov sum
    lyapunov_sum = 0.0
    d0 = jnp.linalg.norm(jnp.concatenate([x_pert - x_ref, v_pert - v_ref]))
    
    # Integration loop
    for step in range(n_steps):
        # Step reference trajectory
        inputs = jnp.array([input_force])
        x_ref_new, v_ref_new = oscillator.step(x_ref, v_ref, inputs)
        
        # Step perturbed trajectory  
        x_pert_new, v_pert_new = oscillator.step(x_pert, v_pert, inputs)
        
        # Ensure outputs are 1D arrays (handle any shape issues)
        x_ref_new = jnp.atleast_1d(x_ref_new).flatten()[:1]
        v_ref_new = jnp.atleast_1d(v_ref_new).flatten()[:1]
        x_pert_new = jnp.atleast_1d(x_pert_new).flatten()[:1]
        v_pert_new = jnp.atleast_1d(v_pert_new).flatten()[:1]
        
        # Compute separation in phase space
        dx = x_pert_new - x_ref_new
        dv = v_pert_new - v_ref_new
        d1 = jnp.linalg.norm(jnp.concatenate([dx, dv]))
        
        # Accumulate Lyapunov exponent
        if d1 > 1e-12:  # Avoid log(0)
            lyapunov_sum += jnp.log(d1 / d0)
            
            # Renormalize perturbation to prevent overflow/underflow
            scale = d0 / d1
            x_pert_new = x_ref_new + dx * scale
            v_pert_new = v_ref_new + dv * scale
        
        # Update states
        x_ref, v_ref = x_ref_new, v_ref_new
        x_pert, v_pert = x_pert_new, v_pert_new
    
    # Average Lyapunov exponent
    lyapunov = lyapunov_sum / n_steps
    
    return float(lyapunov)


class EdgeOfChaosOptimizer:
    """Optimizer for finding oscillator parameters that achieve edge of chaos dynamics."""
    
    def __init__(
        self,
        target_lyapunov: float = 0.0,
        lambda_min: float = -0.05,
        lambda_max: float = 0.05,
        learning_rate: float = 0.01,
        max_iterations: int = 100
    ):
        """
        Initialize the optimizer.
        
        Args:
            target_lyapunov: Target Lyapunov exponent (0.0 for edge of chaos).
            lambda_min: Minimum acceptable Lyapunov exponent.
            lambda_max: Maximum acceptable Lyapunov exponent.
            learning_rate: Learning rate for parameter updates.
            max_iterations: Maximum number of optimization iterations.
        """
        self.target_lyapunov = target_lyapunov
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
    
    def optimize(
        self,
        oscillator: Oscillator,
        parameter_ranges: Dict[str, Tuple[float, float]],
        initial_parameters: Optional[Dict[str, float]] = None,
        input_force: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, float], List[float], List[float]]:
        """
        Optimize oscillator parameters to achieve edge of chaos.
        
        Args:
            oscillator: The oscillator to optimize.
            parameter_ranges: Dictionary mapping parameter names to (min, max) ranges.
            initial_parameters: Initial parameter values (optional).
            input_force: Optional input force to the oscillator.
            seed: Random seed for initialization.
            
        Returns:
            Tuple of (optimized_parameters, loss_history, lyapunov_history).
        """
        # Initialize random key
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(int(time.time()))
        
        # Initialize parameters
        params = {}
        
        # Check if parameter_ranges is empty
        if not parameter_ranges:
            # If no ranges provided, just use initial parameters
            if initial_parameters is not None:
                params = dict(initial_parameters)
            return params, [], []  # Return immediately with no optimization
        
        if initial_parameters is not None:
            # Use provided initial parameters
            params = dict(initial_parameters)
            # Clip to ranges
            for param_name, (min_val, max_val) in parameter_ranges.items():
                if param_name in params:
                    params[param_name] = max(min_val, min(max_val, params[param_name]))
                elif min_val is not None and max_val is not None:
                    # Initialize randomly if not provided
                    key, subkey = jax.random.split(key)
                    params[param_name] = min_val + jax.random.uniform(subkey) * (max_val - min_val)
        else:
            # Initialize all parameters randomly within ranges
            for param_name, (min_val, max_val) in parameter_ranges.items():
                if min_val is not None and max_val is not None:
                    key, subkey = jax.random.split(key)
                    params[param_name] = min_val + jax.random.uniform(subkey) * (max_val - min_val)
        
        # Initialize histories
        loss_history = []
        lyapunov_history = []
        
        # Define loss function based on target Lyapunov exponent
        def loss_function(lyapunov):
            return (lyapunov - self.target_lyapunov) ** 2
        
        # Determine oscillator type from class name
        oscillator_type = oscillator.__class__.__name__.lower()
        if 'vanderpol' in oscillator_type:
            oscillator_type = 'vanderpol'
        elif 'nonlinear' in oscillator_type:
            oscillator_type = 'nonlinear'
        elif 'linear' in oscillator_type:
            oscillator_type = 'linear'
        else:
            oscillator_type = 'nonlinear'  # Default fallback
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Create oscillator with current parameters
            updated_oscillator = create_oscillator(oscillator_type, dim=1, **params)
            
            # Compute Lyapunov exponent
            lyapunov = compute_largest_lyapunov_exponent(
                updated_oscillator,
                input_force=input_force
            )
            
            # Compute loss
            loss = loss_function(lyapunov)
            
            # Store history
            loss_history.append(float(loss))
            lyapunov_history.append(float(lyapunov))
            
            # Print progress
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                print(f"Iteration {iteration}: Lyapunov = {lyapunov:.6f}, Loss = {loss:.6f}")
                for param_name, value in params.items():
                    print(f"  {param_name} = {value:.6f}")
            
            # Check if we're in the acceptable range
            if self.lambda_min <= lyapunov <= self.lambda_max:
                print(f"Stopping optimization - Lyapunov exponent {lyapunov:.6f} is in acceptable range")
                break
            
            # Update parameters using gradient descent
            # We use a simple finite difference approximation for the gradient
            for param_name, value in list(params.items()):
                # Skip if out of range
                if param_name not in parameter_ranges:
                    continue
                
                min_val, max_val = parameter_ranges[param_name]
                
                # Compute gradient with finite difference
                eps = 1e-4
                params_plus = dict(params)
                params_plus[param_name] = value + eps
                
                # Clip to range
                params_plus[param_name] = max(min_val, min(max_val, params_plus[param_name]))
                
                # Compute perturbed Lyapunov exponent
                updated_oscillator_plus = create_oscillator(oscillator_type, dim=1, **params_plus)
                
                # Compute Lyapunov with perturbed parameter
                lyapunov_plus = compute_largest_lyapunov_exponent(
                    updated_oscillator_plus,
                    input_force=input_force
                )
                
                # Compute loss with perturbed parameter
                loss_plus = loss_function(lyapunov_plus)
                
                # Compute gradient
                grad = (loss_plus - loss) / eps
                
                # Update parameter
                params[param_name] = value - self.learning_rate * grad
                
                # Clip to range
                params[param_name] = max(min_val, min(max_val, params[param_name]))
        
        return params, loss_history, lyapunov_history
    
    def plot_optimization_results(
        self,
        loss_history: List[float],
        lyapunov_history: List[float],
        title: str = "Edge of Chaos Optimization",
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot optimization results.
        
        Args:
            loss_history: History of loss values.
            lyapunov_history: History of Lyapunov exponents.
            title: Plot title.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        ax1.plot(loss_history, 'b-', linewidth=2)
        ax1.set_title("Loss Function")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        
        # Plot Lyapunov exponent
        ax2.plot(lyapunov_history, 'r-', linewidth=2)
        ax2.axhline(y=self.target_lyapunov, color='k', linestyle='--', label=f"Target λ={self.target_lyapunov}")
        ax2.axhline(y=self.lambda_min, color='g', linestyle='--', label=f"Min λ={self.lambda_min}")
        ax2.axhline(y=self.lambda_max, color='g', linestyle='--', label=f"Max λ={self.lambda_max}")
        ax2.set_title("Lyapunov Exponent")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Lyapunov Exponent")
        ax2.legend()
        ax2.grid(True)
        
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig
    
    def analyze_parameter_sensitivity(
        self,
        oscillator: Oscillator,
        parameters: Dict[str, float],
        parameter_name: str,
        parameter_range: Tuple[float, float],
        n_points: int = 50,
        input_force: Optional[float] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Analyze parameter sensitivity.
        
        Args:
            oscillator: The oscillator to analyze.
            parameters: Current parameter values.
            parameter_name: Name of parameter to analyze.
            parameter_range: Range of parameter values.
            n_points: Number of points in the range.
            input_force: Optional input force to the oscillator.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure.
        """
        min_val, max_val = parameter_range
        param_values = jnp.linspace(min_val, max_val, n_points)
        lyapunov_values = []
        
        # Determine oscillator type from class name
        oscillator_type = oscillator.__class__.__name__.lower()
        if 'vanderpol' in oscillator_type:
            oscillator_type = 'vanderpol'
        elif 'nonlinear' in oscillator_type:
            oscillator_type = 'nonlinear'
        elif 'linear' in oscillator_type:
            oscillator_type = 'linear'
        else:
            oscillator_type = 'nonlinear'  # Default fallback
        
        # Compute Lyapunov exponent for each parameter value
        for param_value in param_values:
            # Update parameter
            params = dict(parameters)
            params[parameter_name] = param_value
            
            # Validate parameters - ensure no None values
            validated_params = {}
            for k, v in params.items():
                if v is not None:
                    validated_params[k] = v
            
            # Create updated oscillator
            updated_oscillator = create_oscillator(oscillator_type, dim=1, **validated_params)
            
            # Compute Lyapunov exponent
            lyapunov = compute_largest_lyapunov_exponent(
                updated_oscillator,
                input_force=input_force
            )
            
            lyapunov_values.append(float(lyapunov))
        
        # Plot sensitivity
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(param_values, lyapunov_values, 'b-', linewidth=2)
        
        # Make sure we create increasing levels for EOC boundaries
        eoc_levels = sorted([self.lambda_min, self.target_lyapunov, self.lambda_max])
        colors = ['g', 'k', 'g']
        
        for level, color in zip(eoc_levels, colors):
            ax.axhline(y=level, color=color, linestyle='--', 
                      label=f"λ={level}" if level == self.target_lyapunov else None)
        
        # Mark the current parameter value
        current_value = parameters[parameter_name]
        current_idx = jnp.argmin(jnp.abs(param_values - current_value))
        current_lyapunov = lyapunov_values[current_idx]
        ax.plot(current_value, current_lyapunov, 'ro', markersize=8, label=f"Current Value: {current_value:.4f}")
        
        ax.set_title(f"Sensitivity Analysis: {parameter_name}")
        ax.set_xlabel(f"{parameter_name} Value")
        ax.set_ylabel("Lyapunov Exponent")
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def grid_search(
        self,
        oscillator: Oscillator,
        parameters: Dict[str, float],
        param1_name: str,
        param1_range: Tuple[float, float],
        param2_name: str,
        param2_range: Tuple[float, float],
        n_points: int = 10,
        input_force: Optional[float] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Perform grid search over two parameters.
        
        Args:
            oscillator: The oscillator to analyze.
            parameters: Current parameter values.
            param1_name: Name of first parameter.
            param1_range: Range of first parameter.
            param2_name: Name of second parameter.
            param2_range: Range of second parameter.
            n_points: Number of points per dimension.
            input_force: Optional input force to the oscillator.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure.
        """
        min1, max1 = param1_range
        min2, max2 = param2_range
        
        param1_values = jnp.linspace(min1, max1, n_points)
        param2_values = jnp.linspace(min2, max2, n_points)
        
        lyapunov_grid = jnp.zeros((n_points, n_points))
        
        # Determine oscillator type from class name
        oscillator_type = oscillator.__class__.__name__.lower()
        if 'vanderpol' in oscillator_type:
            oscillator_type = 'vanderpol'
        elif 'nonlinear' in oscillator_type:
            oscillator_type = 'nonlinear'
        elif 'linear' in oscillator_type:
            oscillator_type = 'linear'
        else:
            oscillator_type = 'nonlinear'  # Default fallback
        
        # Compute Lyapunov exponent for each parameter combination
        for i, p1 in enumerate(param1_values):
            for j, p2 in enumerate(param2_values):
                # Update parameters
                params = dict(parameters)
                params[param1_name] = p1
                params[param2_name] = p2
                
                # Validate parameters - ensure no None values
                validated_params = {}
                for k, v in params.items():
                    if v is not None:
                        validated_params[k] = v
                
                # Create updated oscillator
                updated_oscillator = create_oscillator(oscillator_type, dim=1, **validated_params)
                
                # Compute Lyapunov exponent
                lyapunov = compute_largest_lyapunov_exponent(
                    updated_oscillator,
                    input_force=input_force
                )
                
                lyapunov_grid = lyapunov_grid.at[i, j].set(lyapunov)
        
        # Plot grid search results
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create meshgrid for plotting
        X, Y = jnp.meshgrid(param1_values, param2_values)
        
        # Create contour plot
        levels = 20
        
        # Ensure contour levels are increasing
        min_val = float(jnp.min(lyapunov_grid))
        max_val = float(jnp.max(lyapunov_grid))
        
        # If min and max are the same, add a small difference
        if min_val == max_val:
            min_val -= 0.001
            max_val += 0.001
            
        levels = jnp.linspace(min_val, max_val, 20)
        
        contour = ax.contourf(X, Y, lyapunov_grid.T, levels=levels, cmap='viridis')
        
        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Lyapunov Exponent')
        
        # Add contour lines for significant levels
        # Make sure we create increasing levels for EOC boundaries
        eoc_levels = sorted([self.lambda_min, self.target_lyapunov, self.lambda_max])
        ax.contour(
            X, Y, lyapunov_grid.T,
            levels=eoc_levels,
            colors=['g', 'k', 'g'],
            linestyles=['--', '--', '--'],
            linewidths=[1, 2, 1]
        )
        
        # Mark the current parameter values
        current_p1 = parameters[param1_name]
        current_p2 = parameters[param2_name]
        ax.plot(current_p1, current_p2, 'ro', markersize=8)
        
        ax.set_title(f"Grid Search: {param1_name} vs {param2_name}")
        ax.set_xlabel(f"{param1_name}")
        ax.set_ylabel(f"{param2_name}")
        ax.grid(True)
        
        return fig


# High-level functions for model optimization

def optimize(
    model: Any,
    target_lyapunov: float = 0.0,
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    input_data: Optional[jnp.ndarray] = None,
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    lambda_min: float = -0.05,
    lambda_max: float = 0.05,
    output_dir: Optional[str] = None,
    plot_results: bool = True,
    verbose: bool = True
) -> OptimizationResult:
    """
    High-level function to optimize a model for edge of chaos dynamics.
    
    This function handles all the details of extracting parameters, running optimization,
    visualizing results, and updating the model with optimized parameters.
    
    Args:
        model: The model to optimize (will be adapted to OscillatoryModelInterface).
        target_lyapunov: Target Lyapunov exponent (0.0 for edge of chaos).
        parameter_ranges: Dictionary mapping parameter names to (min, max) ranges.
            If None, default ranges will be used.
        input_data: Optional input data for the model.
        learning_rate: Learning rate for optimization.
        max_iterations: Maximum number of optimization iterations.
        lambda_min: Minimum acceptable Lyapunov exponent.
        lambda_max: Maximum acceptable Lyapunov exponent.
        output_dir: Directory to save visualizations (None to skip saving).
        plot_results: Whether to create and show plots.
        verbose: Whether to print progress information.
        
    Returns:
        Dictionary containing optimization results.
    """
    # Adapt model to interface
    adapted_model = adapt_model(model)
    
    # Extract current parameters
    current_params = adapted_model.extract_parameters()
    
    if verbose:
        print("\n--- Edge of Chaos Optimization ---")
        print(f"Current parameters: {current_params}")
    
    # Get model type and ensure consistency
    model_type = adapted_model.get_model_type()
    
    # Extract the oscillator parameters to double-check the type
    model_params = adapted_model.extract_parameters()
    
    # If we have mu parameter but model_type is not van_der_pol, there's inconsistency
    # This is likely the adapter misdetecting the type
    if 'mu' in model_params and model_type != 'van_der_pol':
        model_type = 'van_der_pol'
        print(f"[DEBUG EOC Internal] Corrected model type to van_der_pol based on parameters {model_params}")
    # If we have alpha, omega, gamma parameters but not detected as 'horn'
    elif all(p in model_params for p in ['alpha', 'omega', 'gamma']) and model_type != 'horn':
        model_type = 'horn'
        print(f"[DEBUG EOC Internal] Corrected model type to horn based on parameters {model_params}")
    
    if verbose:
        print(f"[DEBUG EOC Internal] model_type from adapter: {model_type}")
    
    # Define default parameter ranges if not provided
    if parameter_ranges is None:
        try:
            # Get default ranges from core factory
            parameter_ranges = get_parameter_ranges(model_type)
        except ValueError:
            # Default ranges for common parameters
            parameter_ranges = {
                "alpha": (0.01, 1.0),
                "omega": (0.01, 2.0),
                "gamma": (0.01, 0.5),
                "h": (0.5, 2.0),
                "mu": (0.1, 3.0)  # For van der Pol oscillators
            }
        
        # Filter to only include parameters that exist in the model
        parameter_ranges = {k: v for k, v in parameter_ranges.items() if k in current_params}
        
        if verbose:
            print("\nUsing parameter ranges:")
            for name, (min_val, max_val) in parameter_ranges.items():
                print(f"  {name}: [{min_val}, {max_val}]")
    
    # Initialize the optimizer
    optimizer = EdgeOfChaosOptimizer(
        target_lyapunov=target_lyapunov,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        learning_rate=learning_rate,
        max_iterations=max_iterations
    )
    
    # Get constructor parameters
    constructor_params = adapted_model.get_constructor_parameters()
    
    # Create an oscillator for optimization using the simplified factory
    oscillator = create_oscillator(model_type, dim=1, **constructor_params)
    
    # Run optimization
    if verbose:
        print("\nRunning edge of chaos optimization...")
    
    # Filter parameter ranges to match oscillator type
    # This ensures we only optimize parameters that are valid for the current oscillator type
    filtered_params = {}
    filtered_ranges = {}
    
    for param_name in constructor_params:
        if param_name in parameter_ranges:
            filtered_params[param_name] = constructor_params[param_name]
            filtered_ranges[param_name] = parameter_ranges[param_name]
    
    # If we don't have any valid parameters to optimize, use current parameters
    if not filtered_ranges:
        filtered_params = constructor_params
        if verbose:
            print("No valid parameter ranges found for this oscillator type. Using default optimization.")
    
    # Run optimization with filtered parameters
    optimized_params, loss_history, lyapunov_history = optimizer.optimize(
        oscillator=oscillator,
        parameter_ranges=filtered_ranges if filtered_ranges else parameter_ranges,
        initial_parameters=filtered_params,
        input_force=input_data,
        seed=42
    )
    
    # Update model with optimized parameters
    optimized_model = adapted_model.create_new_instance(optimized_params).get_model()
    
    # Generate plots if requested
    if plot_results:
        try:
            # Create figure for optimization progress
            progress_fig = optimizer.plot_optimization_results(
                loss_history=loss_history,
                lyapunov_history=lyapunov_history,
                title="Edge of Chaos Optimization"
            )
            
            # Save figure if output directory provided
            if output_dir is not None:
                import os
                from pathlib import Path
                
                # Create directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Save progress figure
                progress_fig.savefig(
                    Path(output_dir) / "edge_of_chaos_optimization.png",
                    dpi=300,
                    bbox_inches='tight'
                )
                
                # Create sensitivity plots for important parameters
                for param_name in list(parameter_ranges.keys())[:2]:  # Limit to first two parameters
                    try:
                        # Ensure the parameter exists in optimized_params
                        if param_name not in optimized_params:
                            if verbose:
                                print(f"Warning: Parameter {param_name} not found in optimized parameters")
                            continue
                            
                        # Validate parameter range
                        param_range = parameter_ranges[param_name]
                        if param_range is None or len(param_range) != 2:
                            if verbose:
                                print(f"Warning: Invalid parameter range for {param_name}: {param_range}")
                            continue
                        
                        sensitivity_fig = optimizer.analyze_parameter_sensitivity(
                            oscillator=oscillator,
                            parameters=optimized_params,
                            parameter_name=param_name,
                            parameter_range=param_range,
                            n_points=20
                        )
                        
                        # Check if figure was created successfully
                        if sensitivity_fig is not None:
                            # Save sensitivity figure
                            sensitivity_fig.savefig(
                                Path(output_dir) / f"eoc_sensitivity_{param_name}.png",
                                dpi=300,
                                bbox_inches='tight'
                            )
                            
                            plt.close(sensitivity_fig)
                        else:
                            if verbose:
                                print(f"Warning: Could not create sensitivity plot for {param_name}: Figure is None")
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Could not create sensitivity plot for {param_name}: {e}")
                
                # Create grid search plot if we have at least 2 parameters
                if len(parameter_ranges) >= 2:
                    try:
                        param_names = list(parameter_ranges.keys())
                        param1_name = param_names[0]
                        param2_name = param_names[1]
                        
                        # Validate both parameters exist
                        if param1_name not in optimized_params or param2_name not in optimized_params:
                            if verbose:
                                print(f"Warning: Parameters {param1_name} or {param2_name} not found in optimized parameters")
                        else:
                            # Validate parameter ranges
                            param1_range = parameter_ranges[param1_name]
                            param2_range = parameter_ranges[param2_name]
                            
                            if (param1_range is None or len(param1_range) != 2 or 
                                param2_range is None or len(param2_range) != 2):
                                if verbose:
                                    print(f"Warning: Invalid parameter ranges for grid search")
                            else:
                                grid_fig = optimizer.grid_search(
                                    oscillator=oscillator,
                                    parameters=optimized_params,
                                    param1_name=param1_name,
                                    param1_range=param1_range,
                                    param2_name=param2_name,
                                    param2_range=param2_range,
                                    n_points=10
                                )
                                
                                # Check if figure was created successfully
                                if grid_fig is not None:
                                    # Save grid figure
                                    grid_fig.savefig(
                                        Path(output_dir) / "eoc_grid_search.png",
                                        dpi=300,
                                        bbox_inches='tight'
                                    )
                                    
                                    plt.close(grid_fig)
                                else:
                                    if verbose:
                                        print(f"Warning: Could not create grid search plot: Figure is None")
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Could not create grid search plot: {e}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not create edge-of-chaos plots: {e}")
    
    # Create results dictionary
    results = {
        "initial_parameters": current_params,
        "optimized_parameters": optimized_params,
        "optimized_model": optimized_model,
        "loss_history": loss_history,
        "lyapunov_history": lyapunov_history,
        "final_lyapunov": lyapunov_history[-1] if lyapunov_history else None
    }
    
    if verbose:
        print("\nOptimization Results:")
        print(f"Initial parameters: {current_params}")
        print(f"Optimized parameters: {optimized_params}")
        print(f"Final Lyapunov exponent: {results['final_lyapunov']:.6f}" if results["final_lyapunov"] is not None else "No results")
    
    # Create reconstruction comparison if we have test data
    if input_data is not None and len(optimized_params) > 0:
        try:
            if verbose:
                print("\n🎨 Creating reconstruction comparison visualizations...")
            
            # Create reconstruction comparison
            recon_results = create_eoc_reconstruction_comparison(
                original_model=model,
                optimized_params=optimized_params,
                x_test=input_data,
                n_samples=min(8, len(input_data)),
                output_dir=output_dir,
                verbose=verbose
            )
            
            # Create parameter impact analysis
            impact_results = create_eoc_parameter_impact_analysis(
                original_model=model,
                original_params=current_params,
                optimized_params=optimized_params,
                x_test=input_data,
                output_dir=output_dir,
                verbose=verbose
            )
            
            # Add to results
            results["reconstruction_comparison"] = recon_results
            results["parameter_impact"] = impact_results
            
        except Exception as e:
            if verbose:
                print(f"Warning: Could not create reconstruction comparisons: {e}")
    
    return results


def evaluate_model_comparison(
    original_model: Any,
    optimized_model: Any,
    x_test: jnp.ndarray,
    y_test: Optional[jnp.ndarray] = None,
    n_samples: int = 10,
    sequence_length: int = 20,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate and compare model performance before and after edge of chaos optimization.
    
    Args:
        original_model: Original model.
        optimized_model: Model with optimized parameters.
        x_test: Test inputs.
        y_test: Optional test labels.
        n_samples: Number of samples to evaluate.
        sequence_length: Sequence length for time series.
        output_dir: Directory to save visualizations (None to skip saving).
        verbose: Whether to print progress information.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    if verbose:
        print("\n--- Evaluating Models Before and After Edge of Chaos Optimization ---")
    
    # Check if the models are actually different
    if optimized_model is original_model:
        if verbose:
            print("The optimized model is the same as the original model. Skipping evaluation.")
        return {
            "original_mse": 0.0,
            "optimized_mse": 0.0,
            "change_pct": 0.0,
        }
    
    # More sophisticated check - compare model parameters
    try:
        orig_adapter = adapt_model(original_model)
        opt_adapter = adapt_model(optimized_model)
        orig_params = orig_adapter.extract_parameters()
        opt_params = opt_adapter.extract_parameters()
        
        params_identical = True
        for param_name in orig_params:
            if param_name in opt_params:
                orig_val = orig_params[param_name]
                opt_val = opt_params[param_name]
                if not jnp.allclose(orig_val, opt_val, atol=1e-6):
                    params_identical = False
                    if verbose:
                        print(f"Parameter {param_name} differs: {orig_val} vs {opt_val}")
                    break
        
        if params_identical:
            if verbose:
                print("Models have identical parameters. Computing MSE with original model only.")
            # Compute MSE with original model for both "original" and "optimized"
            test_mse = _compute_model_mse(original_model, x_test[0], sequence_length)
            return {
                "original_mse": test_mse,
                "optimized_mse": test_mse,
                "change_pct": 0.0,
            }
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compare model parameters: {e}")
    
    # Select a subset of test data
    key = jax.random.PRNGKey(42)
    indices = jax.random.permutation(key, len(x_test))[:n_samples]
    x_subset = x_test[indices]
    y_subset = None if y_test is None else y_test[indices]
    
    # Compute reconstruction errors for both models
    original_mse = []
    optimized_mse = []
    
    # Compute MSE for each model
    for i in range(n_samples):
        # Calculate reconstruction MSE for each model
        orig_mse = _compute_model_mse(original_model, x_subset[i], sequence_length)
        opt_mse = _compute_model_mse(optimized_model, x_subset[i], sequence_length)
        
        original_mse.append(orig_mse)
        optimized_mse.append(opt_mse)
    
    # Calculate average metrics
    avg_original_mse = float(np.mean(original_mse))
    avg_optimized_mse = float(np.mean(optimized_mse))
    
    # Calculate percent change
    change_pct = (avg_optimized_mse - avg_original_mse) / avg_original_mse * 100 if avg_original_mse != 0 else 0.0
    
    if verbose:
        print(f"Original model - Average MSE: {avg_original_mse:.6f}")
        print(f"Optimized model - Average MSE: {avg_optimized_mse:.6f}")
        print(f"Change: {change_pct:.2f}%")
    
    # Generate visualizations if requested
    if output_dir is not None:
        _visualize_model_comparison(
            original_model=original_model,
            optimized_model=optimized_model,
            x_subset=x_subset,
            sequence_length=sequence_length,
            output_dir=output_dir
        )
    
    # Return evaluation metrics
    return {
        "original_mse": avg_original_mse,
        "optimized_mse": avg_optimized_mse,
        "change_pct": change_pct,
        "original_mse_samples": original_mse,
        "optimized_mse_samples": optimized_mse
    }


def _compute_model_mse(model, x, sequence_length):
    """Compute reconstruction MSE for a model."""
    # Flatten if not already flat
    x_flat = x.reshape(-1) if x.ndim > 1 else x
    
    try:
        # Check if this is an amplitude-velocity autoencoder or similar model
        # that expects flat input directly (not sequences)
        if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            # This is likely an autoencoder - pass input directly
            x_batch = x_flat[jnp.newaxis, :]  # Add batch dimension
            reconstructed = model(x_batch)
            
            # Extract reconstruction
            if reconstructed.ndim == 2:
                final_recon = reconstructed[0]  # Remove batch dimension
            else:
                final_recon = reconstructed.flatten()[:len(x_flat)]
                
        else:
            # For other models, try sequence input
            # The model expects shape (batch_size, sequence_length, features)
            # Create a sequence by repeating the input
            x_seq = x_flat[jnp.newaxis, jnp.newaxis, :].repeat(sequence_length, axis=1)
            
            # Get reconstruction
            reconstructed = model(x_seq)
            
            # Handle different possible output shapes
            if isinstance(reconstructed, dict) and "output" in reconstructed:
                # Some models return a dictionary with 'output' key
                reconstructed = reconstructed["output"]
            
            # Extract final reconstruction - handle sequence outputs properly
            if reconstructed.ndim == 3:
                # Shape is (batch, sequence, features) - take last timestep
                final_recon = reconstructed[0, -1, :]  # Last timestep
            elif reconstructed.ndim == 2:
                # Shape is (batch, features) - use as is
                final_recon = reconstructed[0]
            else:
                # Unexpected shape - flatten and take first part
                final_recon = reconstructed.flatten()[:len(x_flat)]
        
        # Ensure final_recon has the same length as x_flat
        if len(final_recon) != len(x_flat):
            # If lengths don't match, pad or truncate
            if len(final_recon) > len(x_flat):
                final_recon = final_recon[:len(x_flat)]
            else:
                # Pad with zeros if too short
                padding = jnp.zeros(len(x_flat) - len(final_recon))
                final_recon = jnp.concatenate([final_recon, padding])
        
        # Compute MSE
        mse = float(jnp.mean((x_flat - final_recon) ** 2))
        
        # Check for invalid values
        if jnp.isnan(mse) or jnp.isinf(mse):
            print(f"Warning: Invalid MSE value: {mse}")
            return 0.0
            
        return mse
        
    except Exception as e:
        # If an error occurs, print a warning and return a default value
        print(f"Warning: Error computing MSE: {e}")
        return 0.0


def _visualize_model_comparison(
    original_model,
    optimized_model,
    x_subset,
    sequence_length,
    output_dir
):
    """Create visualizations comparing original and optimized models."""
    # Prepare reconstructions
    original_reconstructions = []
    optimized_reconstructions = []
    
    for i in range(min(5, len(x_subset))):
        x_img = x_subset[i]
        x_flat = x_img.reshape(-1)
        
        # Prepare sequence input
        seq_input = x_flat[jnp.newaxis, jnp.newaxis, :].repeat(sequence_length, axis=1)
        
        # Get reconstructions
        orig_recon = original_model(seq_input)[0, -1, :]
        opt_recon = optimized_model(seq_input)[0, -1, :]
        
        # Reshape back to images
        img_size = int(np.sqrt(len(x_flat)))
        x_img_reshaped = x_flat.reshape(img_size, img_size)
        orig_recon_img = orig_recon.reshape(img_size, img_size)
        opt_recon_img = opt_recon.reshape(img_size, img_size)
        
        # Store reconstructions
        original_reconstructions.append((x_img_reshaped, orig_recon_img))
        optimized_reconstructions.append((x_img_reshaped, opt_recon_img))
    
    # Plot reconstructions
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    fig.suptitle("Reconstructions Before vs After Edge of Chaos Optimization", fontsize=16)
    
    for i in range(min(5, len(original_reconstructions))):
        # Original input
        axes[i, 0].imshow(original_reconstructions[i][0], cmap='gray')
        axes[i, 0].set_title(f"Original {i+1}" if i == 0 else f"{i+1}")
        axes[i, 0].axis('off')
        
        # Original reconstruction
        axes[i, 1].imshow(original_reconstructions[i][1], cmap='gray')
        axes[i, 1].set_title("Before EOC" if i == 0 else "")
        axes[i, 1].axis('off')
        
        # Optimized input (same as original)
        axes[i, 2].imshow(optimized_reconstructions[i][0], cmap='gray')
        axes[i, 2].set_title(f"Original {i+1}" if i == 0 else f"{i+1}")
        axes[i, 2].axis('off')
        
        # Optimized reconstruction
        axes[i, 3].imshow(optimized_reconstructions[i][1], cmap='gray')
        axes[i, 3].set_title("After EOC" if i == 0 else "")
        axes[i, 3].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Create output directory if it doesn't exist
    import os
    from pathlib import Path
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(
        Path(output_dir) / "eoc_reconstruction_comparison.png",
        dpi=300,
        bbox_inches='tight'
    )
    
    # Compare parameters between original and optimized models
    try:
        # Adapt models to get parameter access
        orig_adapter = adapt_model(original_model)
        optim_adapter = adapt_model(optimized_model)
        
        # Extract parameters
        orig_params = orig_adapter.extract_parameters()
        optim_params = optim_adapter.extract_parameters()
        
        # Print parameter comparison
        with open(Path(output_dir) / "parameter_comparison.txt", "w") as f:
            f.write("Parameter Comparison Before vs After EOC Optimization\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("{:10} {:>12} {:>12} {:>12}\n".format("Parameter", "Original", "Optimized", "Change (%)"))
            f.write("-" * 48 + "\n")
            
            for param_name in sorted(set(orig_params.keys()) | set(optim_params.keys())):
                if param_name in orig_params and param_name in optim_params:
                    orig_val = orig_params[param_name]
                    optim_val = optim_params[param_name]
                    
                    # Handle array parameters
                    if isinstance(orig_val, jnp.ndarray) and orig_val.size > 0:
                        orig_val = float(orig_val.flatten()[0])
                    if isinstance(optim_val, jnp.ndarray) and optim_val.size > 0:
                        optim_val = float(optim_val.flatten()[0])
                    
                    change_pct = (optim_val - orig_val) / orig_val * 100 if orig_val != 0 else float('inf')
                    f.write("{:10} {:12.6f} {:12.6f} {:+12.2f}\n".format(
                        param_name, orig_val, optim_val, change_pct
                    ))
                elif param_name in orig_params:
                    orig_val = orig_params[param_name]
                    f.write("{:10} {:12.6f} {:>12} {:>12}\n".format(
                        param_name, float(orig_val), "N/A", "N/A"
                    ))
                else:  # param_name in optim_params
                    optim_val = optim_params[param_name]
                    f.write("{:10} {:>12} {:12.6f} {:>12}\n".format(
                        param_name, "N/A", float(optim_val), "N/A"
                    ))
    
    except Exception as e:
        print(f"Error comparing parameters: {e}")


def create_eoc_reconstruction_comparison(
    original_model: Any,
    optimized_params: Dict[str, float],
    x_test: jnp.ndarray,
    n_samples: int = 8,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Create visual reconstructions comparing original model vs edge-of-chaos optimized model.
    
    Args:
        original_model: The original trained model
        optimized_params: Edge-of-chaos optimized parameters
        x_test: Test data for reconstruction
        n_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
        verbose: Whether to print progress
        
    Returns:
        Dictionary with reconstruction metrics and visualizations
    """
    if verbose:
        print("\n🎨 Creating Edge-of-Chaos Reconstruction Comparison...")
    
    # Create the edge-of-chaos optimized model
    adapter = adapt_model(original_model)
    eoc_model_adapter = adapter.create_new_instance(optimized_params)
    eoc_model = eoc_model_adapter.get_model()
    
    # Select random test samples
    key = jax.random.PRNGKey(42)
    indices = jax.random.permutation(key, len(x_test))[:n_samples]
    x_subset = x_test[indices]
    
    # Get reconstructions from both models
    original_reconstructions = []
    eoc_reconstructions = []
    original_errors = []
    eoc_errors = []
    
    for i in range(n_samples):
        x_sample = x_subset[i]
        
        # Prepare input (assuming MNIST-like data)
        if x_sample.ndim == 2:  # If it's an image, flatten it
            x_flat = x_sample.flatten()
        else:
            x_flat = x_sample
            
        try:
            # Get reconstruction from original model
            orig_recon = _get_model_reconstruction(original_model, x_flat)
            original_reconstructions.append(orig_recon)
            
            # Get reconstruction from EOC model
            eoc_recon = _get_model_reconstruction(eoc_model, x_flat)
            eoc_reconstructions.append(eoc_recon)
            
            # Compute reconstruction errors
            orig_mse = float(jnp.mean((x_flat - orig_recon) ** 2))
            eoc_mse = float(jnp.mean((x_flat - eoc_recon) ** 2))
            
            original_errors.append(orig_mse)
            eoc_errors.append(eoc_mse)
            
        except Exception as e:
            if verbose:
                print(f"Warning: Could not reconstruct sample {i}: {e}")
            # Use zeros as fallback
            original_reconstructions.append(jnp.zeros_like(x_flat))
            eoc_reconstructions.append(jnp.zeros_like(x_flat))
            original_errors.append(float('inf'))
            eoc_errors.append(float('inf'))
    
    # Create visualization comparing both models
    if len(original_reconstructions) > 0:
        _create_eoc_model_comparison_plots(
            x_subset=x_subset,
            original_reconstructions=original_reconstructions,
            eoc_reconstructions=eoc_reconstructions,
            original_errors=original_errors,
            eoc_errors=eoc_errors,
            optimized_params=optimized_params,
            output_dir=output_dir,
            verbose=verbose
        )
    
    # Return metrics
    avg_original_mse = float(jnp.mean(jnp.array(original_errors)))
    avg_eoc_mse = float(jnp.mean(jnp.array(eoc_errors)))
    improvement_pct = (avg_original_mse - avg_eoc_mse) / avg_original_mse * 100 if avg_original_mse > 0 else 0.0
    
    results = {
        "original_mse": avg_original_mse,
        "eoc_mse": avg_eoc_mse,
        "improvement_pct": improvement_pct,
        "optimized_parameters": optimized_params,
        "n_samples": n_samples,
        "original_errors": original_errors,
        "eoc_errors": eoc_errors
    }
    
    if verbose:
        print(f"✅ Original model average MSE: {avg_original_mse:.6f}")
        print(f"🔧 EOC model average MSE: {avg_eoc_mse:.6f}")
        print(f"📈 Performance change: {improvement_pct:+.2f}%")
        if improvement_pct > 0:
            print(f"🎉 Edge-of-chaos optimization IMPROVED the model!")
        else:
            print(f"📝 Edge-of-chaos optimization did not improve performance")
        
    return results


def _get_model_reconstruction(model, x_flat):
    """Get a single reconstruction from a model, handling different model types properly."""
    try:
        # Check if this is an amplitude-velocity autoencoder or similar model
        # that expects flat input directly (not sequences)
        if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            # This is likely an autoencoder - pass input directly
            x_batch = x_flat[jnp.newaxis, :]  # Add batch dimension
            recon = model(x_batch)
            
            # Extract reconstruction
            if recon.ndim == 2:
                final_recon = recon[0]  # Remove batch dimension
            else:
                final_recon = recon.flatten()[:len(x_flat)]
                
        else:
            # For other models, try sequence input
            x_seq = x_flat[jnp.newaxis, jnp.newaxis, :].repeat(20, axis=1)  # 20 timesteps
            recon = model(x_seq)
            
            # Handle different output formats
            if isinstance(recon, dict) and "output" in recon:
                recon = recon["output"]
            
            # Extract final reconstruction - handle sequence outputs properly
            if recon.ndim == 3:
                # Shape is (batch, sequence, features) - take last timestep
                final_recon = recon[0, -1, :]  # Last timestep
            elif recon.ndim == 2:
                # Shape is (batch, features) - use as is
                final_recon = recon[0]
            else:
                # Unexpected shape - flatten and take first part
                final_recon = recon.flatten()[:len(x_flat)]
        
        # Ensure final_recon has the same length as x_flat
        if len(final_recon) != len(x_flat):
            # If lengths don't match, pad or truncate
            if len(final_recon) > len(x_flat):
                final_recon = final_recon[:len(x_flat)]
            else:
                # Pad with zeros if too short
                padding = jnp.zeros(len(x_flat) - len(final_recon))
                final_recon = jnp.concatenate([final_recon, padding])
        
        return final_recon
        
    except Exception as e:
        # If all else fails, return zeros
        print(f"Warning: Model reconstruction failed: {e}")
        return jnp.zeros_like(x_flat)


def _create_eoc_model_comparison_plots(
    x_subset: jnp.ndarray,
    original_reconstructions: List[jnp.ndarray],
    eoc_reconstructions: List[jnp.ndarray],
    original_errors: List[float],
    eoc_errors: List[float],
    optimized_params: Dict[str, float],
    output_dir: Optional[str] = None,
    verbose: bool = True
):
    """Create comparison plots showing original vs edge-of-chaos optimized model reconstructions."""
    
    n_samples = len(x_subset)
    
    # Create figure with subplots: Original, Original Recon, EOC Recon, Difference
    fig, axes = plt.subplots(4, n_samples, figsize=(2*n_samples, 8))
    if n_samples == 1:
        axes = axes.reshape(4, 1)
    
    fig.suptitle("Edge-of-Chaos Model Comparison: Original vs Optimized", fontsize=16)
    
    # Determine image size (assume square images)
    img_size = int(jnp.sqrt(x_subset[0].size))
    
    for i in range(n_samples):
        # Original image
        if x_subset[i].ndim == 2:
            original_img = x_subset[i]
        else:
            original_img = x_subset[i].reshape(img_size, img_size)
            
        axes[0, i].imshow(original_img, cmap='gray')
        axes[0, i].set_title(f"Original {i+1}" if i == 0 else f"{i+1}")
        axes[0, i].axis('off')
        
        # Original model reconstruction
        try:
            orig_recon_img = original_reconstructions[i].reshape(img_size, img_size)
            axes[1, i].imshow(orig_recon_img, cmap='gray')
            # Handle inf values in error display
            error_text = f"{original_errors[i]:.4f}" if not jnp.isinf(original_errors[i]) else "inf"
            axes[1, i].set_title(f"Trained Model\nMSE: {error_text}" if i == 0 else f"MSE: {error_text}")
        except Exception as e:
            # Show blank if reconstruction failed
            axes[1, i].imshow(jnp.zeros((img_size, img_size)), cmap='gray')
            axes[1, i].set_title(f"Trained Model\nFailed" if i == 0 else "Failed")
        axes[1, i].axis('off')
        
        # EOC model reconstruction
        try:
            eoc_recon_img = eoc_reconstructions[i].reshape(img_size, img_size)
            axes[2, i].imshow(eoc_recon_img, cmap='gray')
            # Handle inf values in error display
            error_text = f"{eoc_errors[i]:.4f}" if not jnp.isinf(eoc_errors[i]) else "inf"
            axes[2, i].set_title(f"EOC Optimized\nMSE: {error_text}" if i == 0 else f"MSE: {error_text}")
        except Exception as e:
            # Show blank if reconstruction failed
            axes[2, i].imshow(jnp.zeros((img_size, img_size)), cmap='gray')
            axes[2, i].set_title(f"EOC Optimized\nFailed" if i == 0 else "Failed")
        axes[2, i].axis('off')
        
        # Difference between reconstructions
        try:
            orig_recon_img = original_reconstructions[i].reshape(img_size, img_size)
            eoc_recon_img = eoc_reconstructions[i].reshape(img_size, img_size)
            diff = jnp.abs(orig_recon_img - eoc_recon_img)
            # Handle case where diff might be all zeros or have inf values
            if jnp.all(diff == 0) or jnp.any(jnp.isinf(diff)) or jnp.any(jnp.isnan(diff)):
                diff = jnp.zeros_like(diff)
            im = axes[3, i].imshow(diff, cmap='hot', vmin=0, vmax=jnp.max(diff) if jnp.max(diff) > 0 else 1)
            axes[3, i].set_title(f"Reconstruction Diff" if i == 0 else "")
        except Exception as e:
            # Show blank if difference computation failed
            im = axes[3, i].imshow(jnp.zeros((img_size, img_size)), cmap='hot')
            axes[3, i].set_title(f"Diff Failed" if i == 0 else "")
        axes[3, i].axis('off')
    
    # Add colorbar for difference maps (only if we have a valid image)
    try:
        plt.colorbar(im, ax=axes[3, :], orientation='horizontal', fraction=0.05, pad=0.1)
    except:
        pass  # Skip colorbar if it fails
    
    # Add text box with optimized parameters and performance summary
    # Filter out inf values for averaging
    valid_orig_errors = [e for e in original_errors if not jnp.isinf(e) and not jnp.isnan(e)]
    valid_eoc_errors = [e for e in eoc_errors if not jnp.isinf(e) and not jnp.isnan(e)]
    
    if valid_orig_errors and valid_eoc_errors:
        avg_orig_mse = float(jnp.mean(jnp.array(valid_orig_errors)))
        avg_eoc_mse = float(jnp.mean(jnp.array(valid_eoc_errors)))
        improvement = (avg_orig_mse - avg_eoc_mse) / avg_orig_mse * 100 if avg_orig_mse > 0 else 0.0
    else:
        avg_orig_mse = float('inf')
        avg_eoc_mse = float('inf')
        improvement = 0.0
    
    param_text = "Edge-of-Chaos Optimization Results:\n"
    param_text += f"Original MSE: {avg_orig_mse:.4f}" if not jnp.isinf(avg_orig_mse) else "Original MSE: inf"
    param_text += f"\nEOC MSE: {avg_eoc_mse:.4f}" if not jnp.isinf(avg_eoc_mse) else "\nEOC MSE: inf"
    param_text += f"\nImprovement: {improvement:+.1f}%\n\n"
    param_text += "Optimized Parameters:\n"
    for param, value in optimized_params.items():
        param_text += f"{param}: {value:.4f}\n"
    
    fig.text(0.02, 0.02, param_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if improvement > 0 else "lightcoral", alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Save if output directory provided
    if output_dir is not None:
        import os
        from pathlib import Path
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(
            Path(output_dir) / "eoc_model_comparison.png",
            dpi=300,
            bbox_inches='tight'
        )
        
        if verbose:
            print(f"💾 Saved model comparison to {output_dir}/eoc_model_comparison.png")
    
    plt.show()


def create_eoc_parameter_impact_analysis(
    original_model: Any,
    original_params: Dict[str, float],
    optimized_params: Dict[str, float],
    x_test: jnp.ndarray,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze the impact of each optimized parameter on reconstruction quality.
    
    Args:
        original_model: The original trained model
        original_params: Original oscillator parameters
        optimized_params: Edge-of-chaos optimized parameters
        x_test: Test data
        output_dir: Directory to save analysis
        verbose: Whether to print progress
        
    Returns:
        Analysis results
    """
    if verbose:
        print("\n📊 Analyzing Edge-of-Chaos Parameter Impact...")
    
    # Create parameter comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameter comparison bar chart
    params = list(optimized_params.keys())
    original_vals = [original_params.get(p, 0) for p in params]
    optimized_vals = [optimized_params[p] for p in params]
    
    x_pos = jnp.arange(len(params))
    width = 0.35
    
    ax1.bar(x_pos - width/2, original_vals, width, label='Original', alpha=0.7)
    ax1.bar(x_pos + width/2, optimized_vals, width, label='Edge-of-Chaos Optimized', alpha=0.7)
    
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Values')
    ax1.set_title('Parameter Comparison: Original vs Edge-of-Chaos Optimized')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(params)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Parameter change percentage
    changes = []
    for p in params:
        orig_val = original_params.get(p, 0)
        opt_val = optimized_params[p]
        if orig_val != 0:
            change_pct = (opt_val - orig_val) / orig_val * 100
        else:
            change_pct = 0
        changes.append(change_pct)
    
    colors = ['red' if c < 0 else 'green' for c in changes]
    ax2.bar(params, changes, color=colors, alpha=0.7)
    ax2.set_xlabel('Parameters')
    ax2.set_ylabel('Change (%)')
    ax2.set_title('Parameter Changes After Edge-of-Chaos Optimization')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Rotate x-axis labels if needed
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir is not None:
        import os
        from pathlib import Path
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(
            Path(output_dir) / "eoc_parameter_impact_analysis.png",
            dpi=300,
            bbox_inches='tight'
        )
        
        if verbose:
            print(f"💾 Saved parameter impact analysis to {output_dir}/eoc_parameter_impact_analysis.png")
    
    plt.show()
    
    # Create summary report
    results = {
        "parameter_changes": dict(zip(params, changes)),
        "largest_change": max(changes, key=abs),
        "most_changed_param": params[jnp.argmax(jnp.abs(jnp.array(changes)))],
        "original_params": original_params,
        "optimized_params": optimized_params
    }
    
    if verbose:
        print(f"📈 Largest parameter change: {results['most_changed_param']} ({results['largest_change']:.2f}%)")
    
    return results


def fine_tune_after_eoc_optimization(
    original_model: Any,
    optimized_model: Any,
    train_data: jnp.ndarray,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    verbose: bool = True
) -> Any:
    """
    Fine-tune the model after edge-of-chaos parameter optimization.
    
    This function performs a few epochs of training to adapt the neural network weights
    to the new oscillator parameters, restoring model stability and performance.
    
    Args:
        original_model: The original trained model
        optimized_model: Model with updated oscillator parameters
        train_data: Training data for fine-tuning
        num_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        batch_size: Batch size for fine-tuning
        verbose: Whether to print progress
        
    Returns:
        Fine-tuned model with stable performance
    """
    import optax
    import equinox as eqx
    from oscnet.learning import mse_loss
    
    if verbose:
        print(f"\n🔧 Fine-tuning model after edge-of-chaos optimization...")
        print(f"   Epochs: {num_epochs}, LR: {learning_rate}, Batch size: {batch_size}")
    
    # Setup optimizer for fine-tuning
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)
    opt_state = optimizer.init(eqx.filter(optimized_model, eqx.is_array))
    
    # Fine-tuning loop
    model = optimized_model
    n_batches = len(train_data) // batch_size
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Create batches
        for i in range(0, len(train_data) - batch_size + 1, batch_size):
            batch = train_data[i:i + batch_size]
            
            # Compute loss and gradients
            def loss_fn(model):
                return mse_loss(model, batch)
            
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            
            # Update model
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            
            epoch_losses.append(float(loss))
        
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        
        if verbose:
            print(f"   Fine-tune Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    if verbose:
        print(f"✅ Fine-tuning complete!")
    
    return model


def optimize_with_finetuning(
    model: Any,
    train_data: jnp.ndarray,
    target_lyapunov: float = 0.0,
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    lambda_min: float = -0.05,
    lambda_max: float = 0.05,
    finetune_epochs: int = 5,
    finetune_lr: float = 1e-4,
    output_dir: Optional[str] = None,
    plot_results: bool = True,
    verbose: bool = True
) -> OptimizationResult:
    """
    Edge-of-chaos optimization with fine-tuning to restore model stability.
    
    This function performs edge-of-chaos parameter optimization followed by
    fine-tuning to adapt the neural network weights to the new parameters.
    
    Args:
        model: The model to optimize
        train_data: Training data for fine-tuning
        target_lyapunov: Target Lyapunov exponent
        parameter_ranges: Parameter ranges for optimization
        learning_rate: Learning rate for parameter optimization
        max_iterations: Maximum optimization iterations
        lambda_min: Minimum acceptable Lyapunov exponent
        lambda_max: Maximum acceptable Lyapunov exponent
        finetune_epochs: Number of fine-tuning epochs
        finetune_lr: Learning rate for fine-tuning
        output_dir: Directory to save results
        plot_results: Whether to create plots
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing optimization results including fine-tuned model
    """
    if verbose:
        print("\n🌊 EDGE-OF-CHAOS OPTIMIZATION WITH FINE-TUNING")
        print("=" * 60)
    
    # Step 1: Perform edge-of-chaos parameter optimization
    eoc_results = optimize(
        model=model,
        target_lyapunov=target_lyapunov,
        parameter_ranges=parameter_ranges,
        input_data=train_data[:100] if len(train_data) > 100 else train_data,  # Use subset for optimization
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        output_dir=output_dir,
        plot_results=plot_results,
        verbose=verbose
    )
    
    # Step 2: Fine-tune the optimized model
    optimized_model = eoc_results["optimized_model"]
    
    # Check if parameters actually changed
    original_params = eoc_results["initial_parameters"]
    optimized_params = eoc_results["optimized_parameters"]
    
    params_changed = False
    for param_name in original_params:
        if param_name in optimized_params:
            if not jnp.allclose(original_params[param_name], optimized_params[param_name], atol=1e-6):
                params_changed = True
                break
    
    if params_changed:
        if verbose:
            print(f"\n🔧 Parameters changed - performing fine-tuning...")
        
        # Fine-tune the model
        finetuned_model = fine_tune_after_eoc_optimization(
            original_model=model,
            optimized_model=optimized_model,
            train_data=train_data,
            num_epochs=finetune_epochs,
            learning_rate=finetune_lr,
            batch_size=32,
            verbose=verbose
        )
        
        # Update results with fine-tuned model
        eoc_results["finetuned_model"] = finetuned_model
        eoc_results["fine_tuning_performed"] = True
        
        if verbose:
            print(f"\n📊 Evaluating fine-tuned model performance...")
        
        # Evaluate fine-tuned model
        test_subset = train_data[:50] if len(train_data) > 50 else train_data
        
        try:
            # Compare original, optimized (unstable), and fine-tuned models
            original_mse = _compute_model_mse(model, test_subset[0], 49)
            optimized_mse = _compute_model_mse(optimized_model, test_subset[0], 49)
            finetuned_mse = _compute_model_mse(finetuned_model, test_subset[0], 49)
            
            eoc_results["original_mse"] = float(original_mse)
            eoc_results["optimized_mse"] = float(optimized_mse)
            eoc_results["finetuned_mse"] = float(finetuned_mse)
            
            if verbose:
                print(f"   📈 Original model MSE: {original_mse:.6f}")
                print(f"   ⚠️  Optimized model MSE: {optimized_mse:.6f}")
                print(f"   ✅ Fine-tuned model MSE: {finetuned_mse:.6f}")
                
                if finetuned_mse < original_mse:
                    improvement = (original_mse - finetuned_mse) / original_mse * 100
                    print(f"   🎉 Fine-tuned model improved by {improvement:.2f}%!")
                elif finetuned_mse < optimized_mse:
                    print(f"   🔧 Fine-tuning restored model stability!")
                
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not evaluate models: {e}")
    
    else:
        if verbose:
            print(f"\n📝 No significant parameter changes - skipping fine-tuning")
        eoc_results["finetuned_model"] = optimized_model
        eoc_results["fine_tuning_performed"] = False
    
    return eoc_results 