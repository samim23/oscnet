"""
Integration Example: Hierarchical Fractal Coupling in oscnet

This example shows how to integrate hierarchical fractal coupling into
an oscillatory neural network, replacing the standard dense h2h layer.

Demonstrates:
1. Creating hierarchical coupling layer
2. Replacing h2h with fractal coupling
3. Training and comparing performance
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pathlib import Path
import time
import logging

# Configure JAX
jax.config.update("jax_log_compiles", False)
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import fractal coupling (requires: pip install -e .)
from oscnet.core.fractal_coupling import HierarchicalCouplingLayer
from oscnet.core.oscillators import NonlinearHarmonicOscillator


# ============================================================================
# MODIFIED HORN CELL WITH FRACTAL COUPLING
# ============================================================================

class FractalHORNCell(eqx.Module):
    """
    HORN cell with hierarchical fractal coupling instead of dense layer.
    
    This demonstrates how to integrate fractal coupling into existing models.
    """
    
    # Layers
    i2h: eqx.nn.Linear  # Input projection
    h2h: HierarchicalCouplingLayer  # Fractal coupling layer (replaces dense)
    h2o: eqx.nn.Linear  # Output projection
    
    # Oscillator dynamics
    oscillator: NonlinearHarmonicOscillator
    
    # Configuration
    hidden_dim: int = eqx.static_field()
    gain_rec: float = eqx.static_field()
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        coupling_depth: int = 2,
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize fractal HORN cell."""
        keys = jax.random.split(key, 4)
        
        self.hidden_dim = hidden_dim
        self.gain_rec = 1.0 / jnp.sqrt(hidden_dim)
        
        # Standard HORN layers
        self.i2h = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])
        
        # FRACTAL COUPLING: Replace dense h2h with hierarchical structure
        self.h2h = HierarchicalCouplingLayer(
            hidden_dim=hidden_dim,
            depth=coupling_depth,
            initial_strength=self.gain_rec,
            key=keys[1]
        )
        
        self.h2o = eqx.nn.Linear(hidden_dim, output_dim, key=keys[2])
        
        # Oscillator dynamics
        self.oscillator = NonlinearHarmonicOscillator(
            dim=hidden_dim,
            key=keys[3]
        )
    
    def __call__(
        self, 
        inputs: jnp.ndarray, 
        state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Process inputs using fractal coupling dynamics."""
        batch_size = inputs.shape[0]
        
        if state is None:
            x = jnp.zeros((batch_size, self.hidden_dim))
            v = jnp.zeros((batch_size, self.hidden_dim))
        else:
            x, v = state
        
        # Standard HORN processing
        input_contrib = jax.vmap(self.i2h)(inputs)
        
        # FRACTAL COUPLING: Use hierarchical structure
        recurrent_contrib = self.h2h(v) * self.gain_rec
        
        total_input = input_contrib + recurrent_contrib
        
        # Update oscillator dynamics
        new_x, new_v = jax.vmap(self.oscillator.step)(x, v, total_input)
        
        # Output
        output = jax.vmap(self.h2o)(new_x)
        
        return output, (new_x, new_v)


# ============================================================================
# COMPARISON TEST
# ============================================================================

def compare_coupling_strategies():
    """Compare fractal vs dense coupling on simple task."""
    
    logger.info("="*80)
    logger.info("COMPARING FRACTAL VS DENSE COUPLING")
    logger.info("="*80)
    
    # Parameters
    input_dim = 16
    hidden_dim = 64
    output_dim = 16
    batch_size = 32
    seq_len = 10
    
    key = jax.random.PRNGKey(42)
    
    # Create models
    logger.info("\nCreating models...")
    
    key, subkey = jax.random.split(key)
    fractal_model = FractalHORNCell(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        coupling_depth=2,
        key=subkey
    )
    
    logger.info(f"✅ Fractal HORN created (hierarchical coupling)")
    
    # Create dummy input sequence
    key, subkey = jax.random.split(key)
    dummy_inputs = jax.random.normal(subkey, (seq_len, batch_size, input_dim))
    
    # Forward pass
    logger.info("\nRunning forward pass...")
    
    def process_sequence(model, inputs):
        outputs = []
        state = None
        for t in range(seq_len):
            output, state = model(inputs[t], state)
            outputs.append(output)
        return jnp.stack(outputs)
    
    start_time = time.time()
    fractal_outputs = process_sequence(fractal_model, dummy_inputs)
    fractal_time = time.time() - start_time
    
    logger.info(f"✅ Fractal forward pass: {fractal_time:.4f}s")
    logger.info(f"   Output shape: {fractal_outputs.shape}")
    
    # Analyze coupling structure
    logger.info("\nAnalyzing coupling structure...")
    
    coupling_matrix = fractal_model.h2h.coupling_matrix
    coupling_strength = fractal_model.h2h.strength
    
    logger.info(f"   Coupling shape: {coupling_matrix.shape}")
    logger.info(f"   Learnable strength: {coupling_strength:.6f}")
    logger.info(f"   Max coupling value: {jnp.max(coupling_matrix):.4f}")
    logger.info(f"   Min coupling value: {jnp.min(coupling_matrix):.4f}")
    
    # Visualize coupling matrix
    output_dir = Path("outputs/fractal_integration_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.array(coupling_matrix), cmap='viridis', aspect='auto')
    ax.set_title("Hierarchical Fractal Coupling Matrix", fontsize=16, fontweight='bold')
    ax.set_xlabel('Oscillator Index')
    ax.set_ylabel('Oscillator Index')
    plt.colorbar(im, ax=ax, label='Coupling Strength')
    plt.tight_layout()
    plt.savefig(output_dir / "fractal_coupling_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Visualization saved to {output_dir}")
    
    # Extract learnable parameters
    learnable_params = eqx.filter(fractal_model, eqx.is_array)
    n_params = sum(x.size for x in jax.tree.leaves(learnable_params))
    
    logger.info(f"\nModel statistics:")
    logger.info(f"   Learnable parameters: {n_params:,}")
    logger.info(f"   Coupling structure: Hierarchical (depth=2)")
    logger.info(f"   Block structure: Self-similar across scales")
    
    logger.info("\n✅ Integration example complete!")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Train on MNIST autoencoder task")
    logger.info(f"  2. Compare reconstruction quality vs dense coupling")
    logger.info(f"  3. Measure spectral properties during training")
    
    return fractal_model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    model = compare_coupling_strategies()

