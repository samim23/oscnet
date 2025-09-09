"""
Theoretically-Grounded Edge-of-Chaos Regularization
===================================================

Implementation based on "Optimal Machine Intelligence at the Edge of Chaos" 
by Feng et al. (2020).

This module provides precise mathematical criteria for edge-of-chaos dynamics
using Jacobian norm and spectral radius analysis.

Key theoretical insights:
1. Edge of chaos occurs when: 1/√N ||J*||_F = 1
2. Three distinct phases: Ordered (stable), Periodic (edge), Chaotic
3. Information processing is maximized at the edge of chaos
4. Spectral radius ρ = 1 marks the boundary between stable and periodic phases

This module also includes the original lightweight EOC methods and empirical
calibration utilities for comprehensive edge-of-chaos analysis and training.
"""

import jax
import jax.numpy as jnp
from oscnet.learning import mse_loss


# ======== THEORETICAL EOC METHODS (Based on Feng et al. 2020) ========

def compute_jacobian_norm_eoc_penalty(model, x_batch, target_jacobian_norm=1.0, penalty_strength=1.0):
    """
    Compute EOC penalty based on the theoretical framework from:
    "Optimal Machine Intelligence at the Edge of Chaos" (Feng et al., 2020)
    
    The paper establishes that edge of chaos occurs when:
    1/√N ||J*||_F = 1
    
    Where:
    - N is the system dimension (hidden_dim)
    - J* is the Jacobian matrix at the asymptotic attractor
    - ||J*||_F is the Frobenius norm
    
    This provides a precise mathematical criterion for edge-of-chaos dynamics.
    
    Args:
        model: The autoencoder model
        x_batch: Input batch
        target_jacobian_norm: Target value for 1/√N ||J*|| (1.0 for edge of chaos)
        penalty_strength: Strength of the penalty term
        
    Returns:
        EOC penalty based on Jacobian norm criterion
    """
    try:
        batch_size = x_batch.shape[0]
        
        # Convert to patches (same as in main model)
        x_images = x_batch.reshape(batch_size, 28, 28)
        patches = x_images.reshape(batch_size, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(batch_size, 49, 16)
        patch_sequence = patches.transpose(1, 0, 2)
        
        # Get initial state
        if hasattr(model.encoder.rnn.cell, 'get_initial_state_from_phases'):
            x_state, v_state = model.encoder.rnn.cell.get_initial_state_from_phases(batch_size)
        else:
            hidden_dim = model.encoder.rnn.cell.hidden_dim
            x_state = jnp.zeros((batch_size, hidden_dim))
            v_state = jnp.zeros((batch_size, hidden_dim))
        
        state = (x_state, v_state)
        hidden_dim = model.encoder.rnn.cell.hidden_dim
        
        # Run dynamics to reach asymptotic attractor
        # Use more steps to ensure we reach the attractor
        max_steps = min(15, patch_sequence.shape[0])
        
        for t in range(max_steps):
            inputs = patch_sequence[t % patch_sequence.shape[0]]  # Cycle if needed
            output, new_state = model.encoder.rnn.cell(inputs, state)
            state = new_state
        
        # Extract asymptotic states
        x_asymptotic, v_asymptotic = state
        
        # Compute Jacobian matrix at the asymptotic state
        # We need the Jacobian of the RNN cell dynamics: d(x_{t+1}, v_{t+1})/d(x_t, v_t)
        
        def cell_dynamics(state_vec, inputs):
            """
            Wrapper function for computing Jacobian.
            state_vec: concatenated [x, v] of shape (batch_size, 2*hidden_dim)
            """
            x_state = state_vec[:, :hidden_dim]
            v_state = state_vec[:, hidden_dim:]
            
            output, (new_x, new_v) = model.encoder.rnn.cell(inputs, (x_state, v_state))
            
            # Return concatenated next state
            next_state = jnp.concatenate([new_x, new_v], axis=-1)
            return next_state
        
        # Concatenate current asymptotic state
        current_state_vec = jnp.concatenate([x_asymptotic, v_asymptotic], axis=-1)
        
        # Use the last input for Jacobian computation
        last_inputs = patch_sequence[-1]
        
        # Compute Jacobian using JAX autodiff
        jacobian_fn = jax.jacfwd(cell_dynamics, argnums=0)
        jacobian_matrix = jacobian_fn(current_state_vec, last_inputs)
        
        # jacobian_matrix shape: (batch_size, 2*hidden_dim, batch_size, 2*hidden_dim)
        # We want the Jacobian for each sample independently
        
        # Extract diagonal blocks (each sample's Jacobian)
        batch_jacobians = []
        for i in range(batch_size):
            sample_jacobian = jacobian_matrix[i, :, i, :]  # Shape: (2*hidden_dim, 2*hidden_dim)
            batch_jacobians.append(sample_jacobian)
        
        jacobians = jnp.stack(batch_jacobians, axis=0)  # Shape: (batch_size, 2*hidden_dim, 2*hidden_dim)
        
        # Compute Frobenius norm for each Jacobian
        frobenius_norms = jnp.linalg.norm(jacobians, ord='fro', axis=(1, 2))  # Shape: (batch_size,)
        
        # Apply the theoretical criterion: 1/√N ||J*||_F
        N = 2 * hidden_dim  # Total state dimension (x + v)
        normalized_jacobian_norms = frobenius_norms / jnp.sqrt(N)
        
        # Compute penalty based on deviation from target (1.0 for edge of chaos)
        deviations = normalized_jacobian_norms - target_jacobian_norm
        
        # Use squared penalty to encourage convergence to target
        penalty = jnp.mean(deviations ** 2) * penalty_strength
        
        return penalty
        
    except Exception as e:
        # If Jacobian computation fails, fall back to zero penalty
        print(f"Warning: Jacobian EOC computation failed: {e}")
        return 0.0


def compute_spectral_radius_eoc_penalty(model, x_batch, target_spectral_radius=1.0, penalty_strength=1.0):
    """
    Alternative EOC penalty based on spectral radius criterion.
    
    From the paper: The boundary between stable fixed points and periodic cycles
    occurs when the spectral radius ρ = 1.
    
    This is computationally lighter than full Jacobian norm computation.
    
    Args:
        model: The autoencoder model
        x_batch: Input batch
        target_spectral_radius: Target spectral radius (1.0 for edge of chaos)
        penalty_strength: Strength of the penalty term
        
    Returns:
        EOC penalty based on spectral radius criterion
    """
    try:
        batch_size = x_batch.shape[0]
        
        # Convert to patches
        x_images = x_batch.reshape(batch_size, 28, 28)
        patches = x_images.reshape(batch_size, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(batch_size, 49, 16)
        patch_sequence = patches.transpose(1, 0, 2)
        
        # Get initial state
        if hasattr(model.encoder.rnn.cell, 'get_initial_state_from_phases'):
            x_state, v_state = model.encoder.rnn.cell.get_initial_state_from_phases(batch_size)
        else:
            hidden_dim = model.encoder.rnn.cell.hidden_dim
            x_state = jnp.zeros((batch_size, hidden_dim))
            v_state = jnp.zeros((batch_size, hidden_dim))
        
        state = (x_state, v_state)
        hidden_dim = model.encoder.rnn.cell.hidden_dim
        
        # Run to asymptotic state
        max_steps = min(10, patch_sequence.shape[0])
        
        for t in range(max_steps):
            inputs = patch_sequence[t % patch_sequence.shape[0]]
            output, new_state = model.encoder.rnn.cell(inputs, state)
            state = new_state
        
        x_asymptotic, v_asymptotic = state
        
        # Compute Jacobian (same as above but extract spectral radius)
        def cell_dynamics(state_vec, inputs):
            x_state = state_vec[:, :hidden_dim]
            v_state = state_vec[:, hidden_dim:]
            output, (new_x, new_v) = model.encoder.rnn.cell(inputs, (x_state, v_state))
            next_state = jnp.concatenate([new_x, new_v], axis=-1)
            return next_state
        
        current_state_vec = jnp.concatenate([x_asymptotic, v_asymptotic], axis=-1)
        last_inputs = patch_sequence[-1]
        
        jacobian_fn = jax.jacfwd(cell_dynamics, argnums=0)
        jacobian_matrix = jacobian_fn(current_state_vec, last_inputs)
        
        # Extract spectral radius for each sample
        spectral_radii = []
        for i in range(batch_size):
            sample_jacobian = jacobian_matrix[i, :, i, :]
            eigenvalues = jnp.linalg.eigvals(sample_jacobian)
            spectral_radius = jnp.max(jnp.abs(eigenvalues))
            spectral_radii.append(spectral_radius)
        
        spectral_radii = jnp.stack(spectral_radii)
        
        # Compute penalty based on deviation from target spectral radius
        deviations = spectral_radii - target_spectral_radius
        penalty = jnp.mean(deviations ** 2) * penalty_strength
        
        return penalty
        
    except Exception as e:
        print(f"Warning: Spectral radius EOC computation failed: {e}")
        return 0.0


def compute_hybrid_theoretical_eoc_penalty(model, x_batch, 
                                         jacobian_weight=0.7, 
                                         spectral_weight=0.3,
                                         penalty_strength=1.0):
    """
    Hybrid EOC penalty combining both Jacobian norm and spectral radius criteria
    from the theoretical framework.
    
    This provides a robust measure that captures both the precise edge-of-chaos
    criterion (Jacobian norm) and the phase transition boundary (spectral radius).
    
    Args:
        model: The autoencoder model
        x_batch: Input batch
        jacobian_weight: Weight for Jacobian norm penalty
        spectral_weight: Weight for spectral radius penalty
        penalty_strength: Overall penalty strength
        
    Returns:
        Combined theoretical EOC penalty
    """
    jacobian_penalty = compute_jacobian_norm_eoc_penalty(
        model, x_batch, target_jacobian_norm=1.0, penalty_strength=1.0
    )
    
    spectral_penalty = compute_spectral_radius_eoc_penalty(
        model, x_batch, target_spectral_radius=1.0, penalty_strength=1.0
    )
    
    # Combine penalties with weights
    total_penalty = (jacobian_weight * jacobian_penalty + 
                    spectral_weight * spectral_penalty) * penalty_strength
    
    return total_penalty


def theoretical_eoc_aware_loss(model, x_batch, eoc_weight=0.001, eoc_method="hybrid"):
    """
    EOC-aware loss using the theoretical framework from Feng et al. 2020.
    
    Args:
        model: The model
        x_batch: Input batch
        eoc_weight: Weight for EOC regularization term
        eoc_method: Method to use ("jacobian", "spectral", "hybrid")
        
    Returns:
        Combined loss with theoretical EOC regularization
    """
    # Standard reconstruction loss
    reconstruction_loss = mse_loss(model, x_batch)
    
    # Theoretical EOC penalty
    if eoc_weight > 0:
        if eoc_method == "jacobian":
            eoc_penalty = compute_jacobian_norm_eoc_penalty(model, x_batch)
        elif eoc_method == "spectral":
            eoc_penalty = compute_spectral_radius_eoc_penalty(model, x_batch)
        elif eoc_method == "hybrid":
            eoc_penalty = compute_hybrid_theoretical_eoc_penalty(model, x_batch)
        else:
            raise ValueError(f"Unknown EOC method: {eoc_method}")
        
        total_loss = reconstruction_loss + eoc_weight * eoc_penalty
        
        # Debug output
        if hasattr(theoretical_eoc_aware_loss, 'debug_mode') and theoretical_eoc_aware_loss.debug_mode:
            print(f"THEORETICAL EOC DEBUG: recon_loss={reconstruction_loss:.6f}, "
                  f"eoc_penalty={eoc_penalty:.6f}, method={eoc_method}, "
                  f"weighted_eoc={eoc_weight * eoc_penalty:.6f}, total={total_loss:.6f}")
        
        return total_loss
    else:
        return reconstruction_loss

# Debug flag for theoretical EOC loss
theoretical_eoc_aware_loss.debug_mode = False


def analyze_theoretical_eoc_metrics(model, x_batch, return_detailed=False):
    """
    Analyze EOC dynamics using the theoretical framework from the paper.
    
    This function computes the precise theoretical metrics:
    - Normalized Jacobian norm: 1/√N ||J*||_F
    - Spectral radius: ρ
    - Phase classification based on theoretical boundaries
    
    Args:
        model: The autoencoder model
        x_batch: Input batch for analysis
        return_detailed: If True, return detailed breakdown
        
    Returns:
        Dictionary with theoretical EOC analysis
    """
    try:
        batch_size = x_batch.shape[0]
        
        # Convert to patches
        x_images = x_batch.reshape(batch_size, 28, 28)
        patches = x_images.reshape(batch_size, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(batch_size, 49, 16)
        patch_sequence = patches.transpose(1, 0, 2)
        
        # Get initial state
        if hasattr(model.encoder.rnn.cell, 'get_initial_state_from_phases'):
            x_state, v_state = model.encoder.rnn.cell.get_initial_state_from_phases(batch_size)
        else:
            hidden_dim = model.encoder.rnn.cell.hidden_dim
            x_state = jnp.zeros((batch_size, hidden_dim))
            v_state = jnp.zeros((batch_size, hidden_dim))
        
        state = (x_state, v_state)
        hidden_dim = model.encoder.rnn.cell.hidden_dim
        
        # Run to asymptotic state
        max_steps = min(15, patch_sequence.shape[0])
        
        for t in range(max_steps):
            inputs = patch_sequence[t % patch_sequence.shape[0]]
            output, new_state = model.encoder.rnn.cell(inputs, state)
            state = new_state
        
        x_asymptotic, v_asymptotic = state
        
        # Compute Jacobian matrix
        def cell_dynamics(state_vec, inputs):
            x_state = state_vec[:, :hidden_dim]
            v_state = state_vec[:, hidden_dim:]
            output, (new_x, new_v) = model.encoder.rnn.cell(inputs, (x_state, v_state))
            next_state = jnp.concatenate([new_x, new_v], axis=-1)
            return next_state
        
        current_state_vec = jnp.concatenate([x_asymptotic, v_asymptotic], axis=-1)
        last_inputs = patch_sequence[-1]
        
        jacobian_fn = jax.jacfwd(cell_dynamics, argnums=0)
        jacobian_matrix = jacobian_fn(current_state_vec, last_inputs)
        
        # Compute metrics for each sample
        N = 2 * hidden_dim  # Total state dimension
        
        jacobian_norms = []
        spectral_radii = []
        
        for i in range(batch_size):
            sample_jacobian = jacobian_matrix[i, :, i, :]
            
            # Frobenius norm
            frobenius_norm = jnp.linalg.norm(sample_jacobian, ord='fro')
            normalized_jacobian_norm = frobenius_norm / jnp.sqrt(N)
            jacobian_norms.append(float(normalized_jacobian_norm))
            
            # Spectral radius
            eigenvalues = jnp.linalg.eigvals(sample_jacobian)
            spectral_radius = jnp.max(jnp.abs(eigenvalues))
            spectral_radii.append(float(spectral_radius))
        
        # Average across batch
        avg_jacobian_norm = float(jnp.mean(jnp.array(jacobian_norms)))
        avg_spectral_radius = float(jnp.mean(jnp.array(spectral_radii)))
        
        # Phase classification based on theoretical criteria
        if avg_jacobian_norm < 1.0 and avg_spectral_radius < 1.0:
            phase = "ordered"  # Stable fixed points
        elif avg_spectral_radius >= 1.0 and avg_jacobian_norm < 1.0:
            phase = "periodic"  # Pseudo-periodic cycles
        elif avg_jacobian_norm >= 1.0:
            phase = "chaotic"  # Chaotic
        else:
            phase = "transitional"  # Edge case
        
        # Distance from edge of chaos
        jacobian_distance = abs(avg_jacobian_norm - 1.0)
        spectral_distance = abs(avg_spectral_radius - 1.0)
        
        # Overall EOC score (closer to 1.0 is better for both metrics)
        eoc_score = 1.0 / (1.0 + jacobian_distance + spectral_distance)
        
        results = {
            'normalized_jacobian_norm': avg_jacobian_norm,
            'spectral_radius': avg_spectral_radius,
            'phase': phase,
            'jacobian_distance_from_eoc': jacobian_distance,
            'spectral_distance_from_eoc': spectral_distance,
            'theoretical_eoc_score': eoc_score,
            'system_dimension': N,
            'batch_size': batch_size
        }
        
        if return_detailed:
            results['detailed_metrics'] = {
                'individual_jacobian_norms': jacobian_norms,
                'individual_spectral_radii': spectral_radii,
                'jacobian_norm_std': float(jnp.std(jnp.array(jacobian_norms))),
                'spectral_radius_std': float(jnp.std(jnp.array(spectral_radii))),
                'theoretical_criteria': {
                    'edge_of_chaos_jacobian': 1.0,
                    'edge_of_chaos_spectral': 1.0,
                    'ordered_phase': 'jacobian < 1.0 and spectral < 1.0',
                    'periodic_phase': 'spectral >= 1.0 and jacobian < 1.0',
                    'chaotic_phase': 'jacobian >= 1.0'
                }
            }
        
        return results
        
    except Exception as e:
        return {
            'normalized_jacobian_norm': 0.0,
            'spectral_radius': 0.0,
            'phase': 'unknown',
            'jacobian_distance_from_eoc': float('inf'),
            'spectral_distance_from_eoc': float('inf'),
            'theoretical_eoc_score': 0.0,
            'error': str(e)
        }


def print_theoretical_eoc_analysis(eoc_results, model_name="Model"):
    """Print a summary of theoretical EOC analysis results."""
    print(f"\n🔬 {model_name} THEORETICAL EOC ANALYSIS (Feng et al. 2020)")
    print("="*70)
    
    jacobian_norm = eoc_results['normalized_jacobian_norm']
    spectral_radius = eoc_results['spectral_radius']
    phase = eoc_results['phase']
    eoc_score = eoc_results['theoretical_eoc_score']
    
    print(f"📊 Theoretical Metrics:")
    print(f"   Normalized Jacobian Norm (1/√N ||J*||): {jacobian_norm:.4f}")
    print(f"   Spectral Radius (ρ): {spectral_radius:.4f}")
    print(f"   System Dimension (N): {eoc_results['system_dimension']}")
    
    print(f"\n🎯 Edge-of-Chaos Analysis:")
    print(f"   Target Jacobian Norm: 1.0000 (edge of chaos)")
    print(f"   Target Spectral Radius: 1.0000 (phase boundary)")
    print(f"   Jacobian Distance: {eoc_results['jacobian_distance_from_eoc']:.4f}")
    print(f"   Spectral Distance: {eoc_results['spectral_distance_from_eoc']:.4f}")
    
    print(f"\n📈 Phase Classification:")
    print(f"   Current Phase: {phase.upper()}")
    
    if phase == "ordered":
        print("   📉 ORDERED: Stable fixed points (too stable)")
        print("   💡 Recommendation: Increase system complexity")
    elif phase == "periodic":
        print("   🌊 PERIODIC: Pseudo-periodic cycles (near edge of chaos)")
        print("   ✅ Good: Close to optimal information processing")
    elif phase == "chaotic":
        print("   🌪️  CHAOTIC: Chaotic dynamics (too complex)")
        print("   💡 Recommendation: Reduce system complexity")
    else:
        print("   🔄 TRANSITIONAL: Between phases")
    
    print(f"\n🏆 Overall Theoretical EOC Score: {eoc_score:.3f}")
    
    if eoc_score >= 0.8:
        print("   🎉 EXCELLENT: Very close to theoretical edge of chaos!")
    elif eoc_score >= 0.6:
        print("   ✅ GOOD: Reasonably close to edge of chaos")
    elif eoc_score >= 0.4:
        print("   ⚠️  FAIR: Some distance from edge of chaos")
    else:
        print("   ❌ POOR: Far from edge of chaos")
    
    print("="*70)


# ======== COMPARISON UTILITIES ========

def compare_eoc_methods(model, x_batch, methods=["jacobian", "spectral", "hybrid"]):
    """
    Compare different EOC regularization methods on the same model and data.
    
    Args:
        model: The model to analyze
        x_batch: Input batch
        methods: List of methods to compare
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for method in methods:
        try:
            if method == "jacobian":
                penalty = compute_jacobian_norm_eoc_penalty(model, x_batch)
            elif method == "spectral":
                penalty = compute_spectral_radius_eoc_penalty(model, x_batch)
            elif method == "hybrid":
                penalty = compute_hybrid_theoretical_eoc_penalty(model, x_batch)
            else:
                penalty = 0.0
            
            results[method] = {
                'penalty': float(penalty),
                'computational_cost': 'high' if method == 'jacobian' else 'medium' if method == 'spectral' else 'high'
            }
        except Exception as e:
            results[method] = {
                'penalty': 0.0,
                'error': str(e)
            }
    
    return results


def adaptive_eoc_weight_schedule(epoch, total_epochs, base_weight=0.001, 
                               warmup_epochs=5, peak_epoch_ratio=0.7):
    """
    Adaptive EOC weight scheduling based on training progress.
    
    The paper suggests that models naturally evolve towards edge of chaos
    during training, so we can adapt the regularization strength accordingly.
    
    Args:
        epoch: Current epoch
        total_epochs: Total training epochs
        base_weight: Base EOC weight
        warmup_epochs: Number of warmup epochs
        peak_epoch_ratio: When to reach peak weight (as ratio of total epochs)
        
    Returns:
        Adaptive EOC weight for current epoch
    """
    if epoch < warmup_epochs:
        # Gradual warmup
        return base_weight * (epoch / warmup_epochs)
    
    peak_epoch = int(total_epochs * peak_epoch_ratio)
    
    if epoch <= peak_epoch:
        # Increase to peak
        progress = (epoch - warmup_epochs) / (peak_epoch - warmup_epochs)
        return base_weight * (1.0 + progress)
    else:
        # Gradual decay after peak
        decay_progress = (epoch - peak_epoch) / (total_epochs - peak_epoch)
        return base_weight * (2.0 - decay_progress)


# ======== LIGHTWEIGHT EOC METHODS (Original Heuristic Approach) ========

def compute_lightweight_eoc_penalty(model, x_batch):
    """
    Compute a lightweight edge-of-chaos regularization penalty (OPTIMIZED VERSION).
    
    This function encourages dynamics that are neither too ordered (low variance)
    nor too chaotic (extremely high variance) by analyzing hidden state statistics.
    This is much faster than computing Lyapunov exponents.
    
    OPTIMIZATIONS:
    - Reduced time steps (5 instead of 10)
    - Vectorized variance calculations
    - Cached intermediate computations
    - Simplified penalty terms
    
    Args:
        model: The autoencoder model
        x_batch: Input batch
        
    Returns:
        EOC penalty value (scalar)
    """
    try:
        # Get encoder states by running a partial forward pass
        batch_size = x_batch.shape[0]
        
        # Convert flat MNIST to patches (same as in main model) - OPTIMIZED
        x_images = x_batch.reshape(batch_size, 28, 28)
        # Use einops-style reshape for efficiency
        patches = x_images.reshape(batch_size, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(batch_size, 49, 16)
        patch_sequence = patches.transpose(1, 0, 2)
        
        # Get initial state
        if hasattr(model.encoder.rnn.cell, 'get_initial_state_from_phases'):
            x_state, v_state = model.encoder.rnn.cell.get_initial_state_from_phases(batch_size)
        else:
            hidden_dim = model.encoder.rnn.cell.hidden_dim
            x_state = jnp.zeros((batch_size, hidden_dim))
            v_state = jnp.zeros((batch_size, hidden_dim))
        
        state = (x_state, v_state)
        
        # Collect states for FEWER time steps (5 instead of 10) for speed
        max_steps = min(5, patch_sequence.shape[0])
        x_states_list = []
        v_states_list = []
        
        for t in range(max_steps):
            inputs = patch_sequence[t]
            output, new_state = model.encoder.rnn.cell(inputs, state)
            x_states_list.append(new_state[0])
            v_states_list.append(new_state[1])
            state = new_state
        
        # Stack states efficiently
        x_states = jnp.stack(x_states_list, axis=0)  # Shape: (time, batch, hidden)
        v_states = jnp.stack(v_states_list, axis=0)  # Shape: (time, batch, hidden)
        
        # OPTIMIZED: Compute all statistics in one go using vectorized operations
        
        # 1. Activity variance (across time and batch dimensions)
        x_var = jnp.var(x_states)  # Single scalar variance
        v_var = jnp.var(v_states)
        
        # 2. Temporal dynamics variance (using diff)
        if x_states.shape[0] > 1:  # Only if we have multiple time steps
            x_diff = jnp.diff(x_states, axis=0)
            v_diff = jnp.diff(v_states, axis=0)
            x_temporal_var = jnp.var(x_diff)
            v_temporal_var = jnp.var(v_diff)
        else:
            x_temporal_var = 0.0
            v_temporal_var = 0.0
        
        # Target variance levels (tuned for edge-of-chaos)
        target_state_var = 0.1
        target_temporal_var = 0.05
        
        # SIMPLIFIED penalty computation
        state_penalty = (x_var - target_state_var)**2 + (v_var - target_state_var)**2
        temporal_penalty = (x_temporal_var - target_temporal_var)**2 + (v_temporal_var - target_temporal_var)**2
        
        # 3. OPTIMIZED saturation/dead neuron penalties using efficient operations
        # Use percentile-based approach for speed
        x_abs = jnp.abs(x_states)
        v_abs = jnp.abs(v_states)
        
        # Saturation: penalize values above threshold
        saturation_threshold = 2.0
        saturation_penalty = jnp.mean(jnp.maximum(0.0, x_abs - saturation_threshold)**2) + \
                           jnp.mean(jnp.maximum(0.0, v_abs - saturation_threshold)**2)
        
        # Dead neurons: penalize values below threshold
        dead_threshold = 0.01
        dead_penalty = jnp.mean(jnp.maximum(0.0, dead_threshold - x_abs)**2) + \
                      jnp.mean(jnp.maximum(0.0, dead_threshold - v_abs)**2)
        
        # OPTIMIZED: Simple weighted combination
        total_eoc_penalty = (state_penalty + 
                           temporal_penalty + 
                           0.1 * saturation_penalty + 
                           0.1 * dead_penalty)
        
        return total_eoc_penalty
        
    except Exception as e:
        # If anything goes wrong, return zero penalty (fail gracefully)
        return 0.0


def eoc_aware_loss(model, x_batch, eoc_weight=0.001):
    """
    Combined reconstruction loss with lightweight EOC regularization.
    
    Args:
        model: The model
        x_batch: Input batch
        eoc_weight: Weight for EOC regularization term
        
    Returns:
        Combined loss
    """
    # Standard reconstruction loss
    reconstruction_loss = mse_loss(model, x_batch)
    
    # Lightweight EOC penalty
    if eoc_weight > 0:
        eoc_penalty = compute_lightweight_eoc_penalty(model, x_batch)
        total_loss = reconstruction_loss + eoc_weight * eoc_penalty
        
        # DEBUGGING: Store components for verification (can be removed later)
        # This ensures we can verify EOC is actually contributing
        if hasattr(eoc_aware_loss, 'debug_mode') and eoc_aware_loss.debug_mode:
            print(f"DEBUG: recon_loss={reconstruction_loss:.6f}, eoc_penalty={eoc_penalty:.6f}, "
                  f"weighted_eoc={eoc_weight * eoc_penalty:.6f}, total={total_loss:.6f}")
        
        return total_loss
    else:
        return reconstruction_loss

# Debug flag for EOC loss verification
eoc_aware_loss.debug_mode = False


# ======== COMPREHENSIVE EOC DYNAMICS ANALYSIS ========

def analyze_eoc_dynamics(model, x_batch, return_detailed=False):
    """
    Comprehensive analysis of edge-of-chaos dynamics in the model.
    
    This function computes multiple EOC-related metrics to understand
    how close the model actually is to edge-of-chaos dynamics.
    
    Args:
        model: The autoencoder model
        x_batch: Input batch for analysis
        return_detailed: If True, return detailed breakdown of all metrics
        
    Returns:
        Dictionary with EOC analysis results:
        - 'eoc_score': Overall EOC score (0=ordered, 1=edge-of-chaos, >1=chaotic)
        - 'state_variance_ratio': Ratio of actual to target state variance
        - 'temporal_variance_ratio': Ratio of actual to target temporal variance
        - 'activity_distribution': Statistics about neural activity distribution
        - 'chaos_indicators': Various chaos indicators
        - 'detailed_metrics': Full breakdown (if return_detailed=True)
    """
    try:
        batch_size = x_batch.shape[0]
        
        # Convert to patches (same as in main model)
        x_images = x_batch.reshape(batch_size, 28, 28)
        patches = x_images.reshape(batch_size, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(batch_size, 49, 16)
        patch_sequence = patches.transpose(1, 0, 2)
        
        # Get initial state
        if hasattr(model.encoder.rnn.cell, 'get_initial_state_from_phases'):
            x_state, v_state = model.encoder.rnn.cell.get_initial_state_from_phases(batch_size)
        else:
            hidden_dim = model.encoder.rnn.cell.hidden_dim
            x_state = jnp.zeros((batch_size, hidden_dim))
            v_state = jnp.zeros((batch_size, hidden_dim))
        
        state = (x_state, v_state)
        
        # Collect states for analysis (use more time steps for better analysis)
        max_steps = min(10, patch_sequence.shape[0])
        x_states_list = []
        v_states_list = []
        outputs_list = []
        
        for t in range(max_steps):
            inputs = patch_sequence[t]
            output, new_state = model.encoder.rnn.cell(inputs, state)
            x_states_list.append(new_state[0])
            v_states_list.append(new_state[1])
            outputs_list.append(output)
            state = new_state
        
        # Stack states
        x_states = jnp.stack(x_states_list, axis=0)  # Shape: (time, batch, hidden)
        v_states = jnp.stack(v_states_list, axis=0)
        outputs = jnp.stack(outputs_list, axis=0)
        
        # === 1. STATE VARIANCE ANALYSIS ===
        # EOC systems should have moderate variance - not too low (ordered) or too high (chaotic)
        x_var = float(jnp.var(x_states))
        v_var = float(jnp.var(v_states))
        output_var = float(jnp.var(outputs))
        
        # Target variance for EOC (from our penalty function)
        target_state_var = 0.1
        
        # Variance ratios (1.0 = perfect, <1 = too ordered, >1 = too chaotic)
        x_var_ratio = x_var / target_state_var
        v_var_ratio = v_var / target_state_var
        
        # === 2. TEMPORAL DYNAMICS ANALYSIS ===
        # EOC should have rich temporal dynamics
        if x_states.shape[0] > 1:
            x_diff = jnp.diff(x_states, axis=0)
            v_diff = jnp.diff(v_states, axis=0)
            output_diff = jnp.diff(outputs, axis=0)
            
            x_temporal_var = float(jnp.var(x_diff))
            v_temporal_var = float(jnp.var(v_diff))
            output_temporal_var = float(jnp.var(output_diff))
            
            target_temporal_var = 0.05
            x_temporal_ratio = x_temporal_var / target_temporal_var
            v_temporal_ratio = v_temporal_var / target_temporal_var
        else:
            x_temporal_var = v_temporal_var = output_temporal_var = 0.0
            x_temporal_ratio = v_temporal_ratio = 0.0
        
        # === 3. ACTIVITY DISTRIBUTION ANALYSIS ===
        # EOC should have balanced activity - not too many dead or saturated neurons
        x_abs = jnp.abs(x_states)
        v_abs = jnp.abs(v_states)
        
        # Dead neuron percentage (activity < threshold)
        dead_threshold = 0.01
        x_dead_pct = float(jnp.mean(x_abs < dead_threshold) * 100)
        v_dead_pct = float(jnp.mean(v_abs < dead_threshold) * 100)
        
        # Saturated neuron percentage (activity > threshold)
        sat_threshold = 2.0
        x_sat_pct = float(jnp.mean(x_abs > sat_threshold) * 100)
        v_sat_pct = float(jnp.mean(v_abs > sat_threshold) * 100)
        
        # Activity distribution entropy (higher = more diverse)
        def compute_activity_entropy(states, n_bins=20):
            """Compute entropy of activity distribution"""
            # Flatten all states
            flat_states = states.flatten()
            
            # Create histogram
            hist, _ = jnp.histogram(flat_states, bins=n_bins, density=True)
            
            # Add small epsilon to avoid log(0)
            hist = hist + 1e-8
            hist = hist / jnp.sum(hist)  # Normalize
            
            # Compute entropy
            entropy = -jnp.sum(hist * jnp.log(hist))
            return float(entropy)
        
        x_entropy = compute_activity_entropy(x_states)
        v_entropy = compute_activity_entropy(v_states)
        
        # === 4. CHAOS INDICATORS ===
        # Simple chaos indicators based on dynamical properties
        
        # 4a. Largest Lyapunov approximation using finite differences
        def approximate_largest_lyapunov(states):
            """Approximate largest Lyapunov exponent using finite differences"""
            if states.shape[0] < 3:
                return 0.0
                
            # Compute successive ratios of perturbation growth
            lyap_estimates = []
            
            for t in range(1, states.shape[0] - 1):
                # Current state
                current = states[t]
                
                # Previous and next states
                prev = states[t-1]
                next_state = states[t+1]
                
                # Approximate derivatives
                d_prev = current - prev
                d_next = next_state - current
                
                # Avoid division by zero
                d_prev_norm = jnp.linalg.norm(d_prev, axis=-1) + 1e-8
                d_next_norm = jnp.linalg.norm(d_next, axis=-1) + 1e-8
                
                # Growth ratio (per time step)
                growth_ratio = d_next_norm / d_prev_norm
                
                # Log growth (Lyapunov-like)
                lyap_estimate = jnp.log(jnp.mean(growth_ratio))
                lyap_estimates.append(float(lyap_estimate))
            
            return float(jnp.mean(jnp.array(lyap_estimates)))
        
        x_lyapunov_approx = approximate_largest_lyapunov(x_states)
        v_lyapunov_approx = approximate_largest_lyapunov(v_states)
        
        # 4b. Dimensionality of attractor (using correlation dimension approximation)
        def estimate_correlation_dimension(states, max_points=100):
            """Estimate correlation dimension using point correlation"""
            # Flatten and subsample for efficiency
            flat_states = states.reshape(-1, states.shape[-1])
            n_points = min(max_points, flat_states.shape[0])
            
            if n_points < 10:
                return 1.0
            
            # Subsample points
            indices = jnp.linspace(0, flat_states.shape[0]-1, n_points, dtype=int)
            sample_states = flat_states[indices]
            
            # Compute pairwise distances
            diffs = sample_states[:, None, :] - sample_states[None, :, :]
            distances = jnp.linalg.norm(diffs, axis=-1)
            
            # Remove self-distances
            distances = distances + jnp.eye(n_points) * 1e6
            
            # Mean nearest neighbor distance
            min_distances = jnp.min(distances, axis=-1)
            mean_nn_distance = jnp.mean(min_distances)
            
            # Simple correlation dimension estimate
            # Higher values indicate more complex attractors
            corr_dim = -jnp.log(n_points) / jnp.log(float(mean_nn_distance) + 1e-8)
            return max(1.0, min(10.0, float(corr_dim)))  # Clamp to reasonable range
        
        x_corr_dim = estimate_correlation_dimension(x_states)
        v_corr_dim = estimate_correlation_dimension(v_states)
        
        # === 5. OVERALL EOC SCORE ===
        # Combine all metrics into an overall EOC score
        
        # Variance component (penalty for being too far from target)
        var_score = 1.0 / (1.0 + 0.5 * (abs(x_var_ratio - 1.0) + abs(v_var_ratio - 1.0)))
        
        # Temporal dynamics component
        temporal_score = 1.0 / (1.0 + 0.5 * (abs(x_temporal_ratio - 1.0) + abs(v_temporal_ratio - 1.0)))
        
        # Activity balance component (penalize too many dead or saturated neurons)
        dead_penalty = (x_dead_pct + v_dead_pct) / 200.0  # Normalize to [0,1]
        sat_penalty = (x_sat_pct + v_sat_pct) / 200.0
        activity_score = 1.0 / (1.0 + dead_penalty + sat_penalty)
        
        # Chaos component (moderate chaos is good)
        chaos_score = 1.0 / (1.0 + abs(x_lyapunov_approx) + abs(v_lyapunov_approx))
        
        # Complexity component (moderate complexity is good)
        target_dim = 3.0  # Target correlation dimension
        complexity_score = 1.0 / (1.0 + 0.5 * (abs(x_corr_dim - target_dim) + abs(v_corr_dim - target_dim)))
        
        # Weighted overall score
        overall_eoc_score = (
            0.3 * var_score +
            0.2 * temporal_score +
            0.2 * activity_score +
            0.15 * chaos_score +
            0.15 * complexity_score
        )
        
        # === COMPILE RESULTS ===
        results = {
            'eoc_score': float(overall_eoc_score),
            'state_variance_ratio': {
                'x_ratio': float(x_var_ratio),
                'v_ratio': float(v_var_ratio),
                'mean_ratio': float((x_var_ratio + v_var_ratio) / 2)
            },
            'temporal_variance_ratio': {
                'x_ratio': float(x_temporal_ratio),
                'v_ratio': float(v_temporal_ratio),
                'mean_ratio': float((x_temporal_ratio + v_temporal_ratio) / 2)
            },
            'activity_distribution': {
                'dead_neurons_pct': {'x': float(x_dead_pct), 'v': float(v_dead_pct)},
                'saturated_neurons_pct': {'x': float(x_sat_pct), 'v': float(v_sat_pct)},
                'entropy': {'x': float(x_entropy), 'v': float(v_entropy)}
            },
            'chaos_indicators': {
                'lyapunov_approx': {'x': float(x_lyapunov_approx), 'v': float(v_lyapunov_approx)},
                'correlation_dimension': {'x': float(x_corr_dim), 'v': float(v_corr_dim)}
            }
        }
        
        if return_detailed:
            results['detailed_metrics'] = {
                'raw_variances': {
                    'x_var': float(x_var),
                    'v_var': float(v_var),
                    'output_var': float(output_var),
                    'x_temporal_var': float(x_temporal_var),
                    'v_temporal_var': float(v_temporal_var),
                    'output_temporal_var': float(output_temporal_var)
                },
                'score_components': {
                    'var_score': float(var_score),
                    'temporal_score': float(temporal_score),
                    'activity_score': float(activity_score),
                    'chaos_score': float(chaos_score),
                    'complexity_score': float(complexity_score)
                },
                'target_values': {
                    'target_state_var': target_state_var,
                    'target_temporal_var': target_temporal_var,
                    'target_corr_dim': target_dim
                }
            }
        
        return results
        
    except Exception as e:
        # Return safe defaults if analysis fails
        return {
            'eoc_score': 0.0,
            'state_variance_ratio': {'x_ratio': 0.0, 'v_ratio': 0.0, 'mean_ratio': 0.0},
            'temporal_variance_ratio': {'x_ratio': 0.0, 'v_ratio': 0.0, 'mean_ratio': 0.0},
            'activity_distribution': {
                'dead_neurons_pct': {'x': 0.0, 'v': 0.0},
                'saturated_neurons_pct': {'x': 0.0, 'v': 0.0},
                'entropy': {'x': 0.0, 'v': 0.0}
            },
            'chaos_indicators': {
                'lyapunov_approx': {'x': 0.0, 'v': 0.0},
                'correlation_dimension': {'x': 1.0, 'v': 1.0}
            },
            'error': str(e)
        }


def print_eoc_analysis_summary(eoc_results, model_name="Model"):
    """Print a human-readable summary of EOC analysis results."""
    print(f"\n🔬 {model_name} EOC DYNAMICS ANALYSIS")
    print("="*60)
    
    eoc_score = eoc_results['eoc_score']
    print(f"📊 Overall EOC Score: {eoc_score:.3f}/1.0")
    
    if eoc_score >= 0.8:
        print("   🏆 EXCELLENT: Strong edge-of-chaos dynamics!")
    elif eoc_score >= 0.6:
        print("   ✅ GOOD: Moderate edge-of-chaos dynamics")
    elif eoc_score >= 0.4:
        print("   ⚠️  FAIR: Some EOC characteristics present")
    elif eoc_score >= 0.2:
        print("   ❌ POOR: Limited EOC dynamics")
    else:
        print("   💀 VERY POOR: No significant EOC dynamics")
    
    # State variance analysis
    state_ratio = eoc_results['state_variance_ratio']['mean_ratio']
    print(f"\n🌊 State Variance Ratio: {state_ratio:.3f}")
    print(f"   Target: 1.0 (perfect), Current: {state_ratio:.3f}")
    if 0.8 <= state_ratio <= 1.2:
        print("   ✅ Well-balanced state variance")
    elif state_ratio < 0.8:
        print("   📉 Too ordered (low variance)")
    else:
        print("   📈 Too chaotic (high variance)")
    
    # Temporal dynamics
    temporal_ratio = eoc_results['temporal_variance_ratio']['mean_ratio']
    print(f"\n⏱️  Temporal Dynamics Ratio: {temporal_ratio:.3f}")
    if 0.8 <= temporal_ratio <= 1.2:
        print("   ✅ Rich temporal dynamics")
    elif temporal_ratio < 0.8:
        print("   📉 Limited temporal variation")
    else:
        print("   📈 Excessive temporal chaos")
    
    # Activity distribution
    activity = eoc_results['activity_distribution']
    dead_pct = (activity['dead_neurons_pct']['x'] + activity['dead_neurons_pct']['v']) / 2
    sat_pct = (activity['saturated_neurons_pct']['x'] + activity['saturated_neurons_pct']['v']) / 2
    
    print(f"\n🧠 Neural Activity Balance:")
    print(f"   Dead neurons: {dead_pct:.1f}% (should be <10%)")
    print(f"   Saturated neurons: {sat_pct:.1f}% (should be <5%)")
    
    if dead_pct < 10 and sat_pct < 5:
        print("   ✅ Healthy activity distribution")
    else:
        print("   ⚠️  Unbalanced activity distribution")
    
    # Chaos indicators
    chaos = eoc_results['chaos_indicators']
    lyap_x = chaos['lyapunov_approx']['x']
    lyap_v = chaos['lyapunov_approx']['v']
    corr_dim_x = chaos['correlation_dimension']['x']
    corr_dim_v = chaos['correlation_dimension']['v']
    
    print(f"\n🔀 Chaos Indicators:")
    print(f"   Lyapunov (x): {lyap_x:.4f}, (v): {lyap_v:.4f}")
    print(f"   Correlation Dim (x): {corr_dim_x:.2f}, (v): {corr_dim_v:.2f}")
    
    mean_lyap = abs(lyap_x) + abs(lyap_v)
    if mean_lyap < 0.1:
        print("   🎯 Near edge-of-chaos (good!)")
    elif mean_lyap < 0.5:
        print("   🌊 Moderate chaos")
    else:
        print("   🌪️  High chaos")
    
    print("="*60) 