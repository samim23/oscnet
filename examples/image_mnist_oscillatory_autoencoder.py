"""
Oscillatory Autoencoder
==============================================

This script implements an autoencoder using oscillator dynamics that capture
both amplitude (position) and velocity information from HORN (Harmonic Oscillator
Recurrent Network) cells.

This implementation uses the oscnet library utilities for:
- JAX optimization and device management
- Training infrastructure and JIT compilation
- Checkpointing and logging
- Memory optimization

Key features:
- Uses amplitude (x) and velocity (v) from oscillator dynamics
- Patch-based processing for MNIST images
- Edge-of-chaos parameter initialization
- Focused training loop using oscnet utilities
- External evaluation suite for comprehensive analysis
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import time
import logging
import json
import numpy as np

# PRODUCTION JAX CONFIGURATION - Clean and optimized
jax.config.update("jax_log_compiles", False)
jax.config.update("jax_enable_x64", False)  # Use float32 for better performance
jax.config.update("jax_default_matmul_precision", "float32")

# Import the base architecture components
from oscnet.core.oscillators import (
    NonlinearHarmonicOscillator, 
    LearnableNonlinearHarmonicOscillator,
    AdaptiveNonlinearHarmonicOscillator
)

# Import oscnet utilities
from oscnet.utils import (
    get_device_info,
    setup_memory_optimization,
    optimize_for_device,
    monitor_memory_usage,
    disable_nan_debugging,
    disable_verbose_jax_logging,
    setup_application_logger,
    save_equinox_checkpoint,
    save_training_metrics,
    create_batch_iterator
)

from oscnet.learning import (
    create_scheduler,
    wrap_optimizer_with_scheduler,
    CriticalityInitializer,
    train_step,
    memory_efficient_train_step,
    train_epoch_scan,
    warmup_model_compilation,
    mse_loss
)

# Import visualization and evaluation
from oscnet.visualization.model_insights import (
    visualize_reconstructions,
    visualize_training_metrics,
)

from oscnet.evaluation import (
    print_model_summary,
    analyze_model_efficiency,
    comprehensive_model_evaluation,
    create_enhanced_visualizations,
    comprehensive_resonance_analysis
)


# ======== AMPLITUDE-VELOCITY HORN CELL ========

class AmplitudeVelocityHORNCell(eqx.Module):
    """
    Enhanced HORN cell that outputs both amplitude (x) and velocity (v).
    
    This cell extends the standard HORN architecture by using both the position (x)
    and velocity (v) states from the harmonic oscillator, providing richer dynamics
    and better temporal modeling capabilities.
    
    Supports explicit phase initialization for enhanced oscillator dynamics.
    """
    
    # Layers
    i2h: eqx.nn.Linear  # Input projection
    h2h: eqx.nn.Linear  # Recurrent projection  
    h2o: eqx.nn.Linear  # Output projection
    
    # Oscillator dynamics
    oscillator: NonlinearHarmonicOscillator
    
    # Configuration
    hidden_dim: int = eqx.static_field()
    gain_rec: float = eqx.static_field()
    gain_multiplier: jnp.ndarray  # Learnable multiplier for coupling strength
    
    # Phase initialization parameters
    initial_phases: Optional[jnp.ndarray] = eqx.static_field()
    initial_amplitude: float = eqx.static_field()
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        oscillator_class = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict] = None,
        gain_rec: Optional[float] = None,
        initial_phases: Optional[jnp.ndarray] = None,
        initial_amplitude: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize amplitude-velocity HORN cell."""
        keys = jax.random.split(key, 5)
        
        self.hidden_dim = hidden_dim
        # HYBRID APPROACH: HORN scaling + learnable multiplier
        base_gain = 1.0 / jnp.sqrt(hidden_dim)  # Theoretical HORN scaling
        if gain_rec is not None:
            self.gain_rec = gain_rec
        else:
            self.gain_rec = base_gain
        
        # Initialize learnable multiplier to achieve strong coupling (8x boost)
        initial_multiplier = 1.0 / base_gain  # This gives us gain_rec ≈ 1.0 initially
        self.gain_multiplier = jnp.array([initial_multiplier])
        
        self.initial_amplitude = initial_amplitude
        
        # Initialize phases
        if initial_phases is None:
            self.initial_phases = jax.random.uniform(
                keys[4], (hidden_dim,), minval=0.0, maxval=2.0 * jnp.pi
            )
        else:
            assert initial_phases.shape == (hidden_dim,), f"initial_phases must have shape ({hidden_dim},)"
            self.initial_phases = initial_phases
        
        # Standard HORN layers
        self.i2h = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])
        self.h2h = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1])
        
        # AMPLITUDE-VELOCITY OUTPUT: Project from 2*hidden_dim (x + v) to output_dim
        self.h2o = eqx.nn.Linear(2 * hidden_dim, output_dim, key=keys[2])
        
        # Oscillator dynamics
        self.oscillator = oscillator_class(
            dim=hidden_dim,
            **(oscillator_params or {}),
            key=keys[3]
        )
    
    def get_initial_state_from_phases(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate initial (x, v) states from the stored phases."""
        if isinstance(self.oscillator, NonlinearHarmonicOscillator) and hasattr(self.oscillator, 'omega'):
            # Convert omega from static storage format to JAX array
            if isinstance(self.oscillator.omega, (int, float)):
                omega = jnp.ones(self.hidden_dim) * self.oscillator.omega
            else:
                omega = jnp.asarray(self.oscillator.omega)
            
            x_init = self.initial_amplitude * jnp.cos(self.initial_phases)
            v_init = -self.initial_amplitude * omega * jnp.sin(self.initial_phases)
        else:
            # Fallback for other oscillators
            x_init = self.initial_amplitude * 0.1 * jnp.ones(self.hidden_dim)
            v_init = self.initial_amplitude * 0.1 * jnp.ones(self.hidden_dim)

        # Broadcast to batch size
        x_batch = jnp.broadcast_to(x_init[None, :], (batch_size, self.hidden_dim))
        v_batch = jnp.broadcast_to(v_init[None, :], (batch_size, self.hidden_dim))
        
        return x_batch, v_batch
    
    def __call__(
        self, 
        inputs: jnp.ndarray, 
        state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        use_phase_init: bool = False
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Process inputs using amplitude-velocity dynamics."""
        batch_size = inputs.shape[0]
        
        if state is None:
            if use_phase_init:
                x, v = self.get_initial_state_from_phases(batch_size)
            else:
                x = jnp.zeros((batch_size, self.hidden_dim))
                v = jnp.zeros((batch_size, self.hidden_dim))
        else:
            x, v = state
        
        # Standard HORN processing with learnable gain multiplier
        input_contrib = jax.vmap(self.i2h)(inputs)
        effective_gain = self.gain_rec * self.gain_multiplier[0]  # Learnable coupling strength
        recurrent_contrib = jax.vmap(self.h2h)(v) * effective_gain
        total_input = input_contrib + recurrent_contrib
        
        # Update oscillator dynamics
        new_x, new_v = jax.vmap(self.oscillator.step)(x, v, total_input)
        
        # AMPLITUDE-VELOCITY OUTPUT: Concatenate both x and v
        amplitude_velocity_state = jnp.concatenate([new_x, new_v], axis=-1)
        output = jax.vmap(self.h2o)(amplitude_velocity_state)
        
        return output, (new_x, new_v)


# ======== AMPLITUDE-VELOCITY HORN RNN ========

class AmplitudeVelocityHORN(eqx.Module):
    """Enhanced HORN RNN using amplitude-velocity cells."""
    
    cell: AmplitudeVelocityHORNCell
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        oscillator_class = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict] = None,
        gain_rec: Optional[float] = None,
        initial_phases: Optional[jnp.ndarray] = None,
        initial_amplitude: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize amplitude-velocity HORN RNN."""
        self.cell = AmplitudeVelocityHORNCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            gain_rec=gain_rec,
            initial_phases=initial_phases,
            initial_amplitude=initial_amplitude,
            key=key
        )
    
    def __call__(
        self, 
        inputs: jnp.ndarray,
        initial_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        return_trajectories: bool = False,
        use_phase_init: bool = False
    ) -> jnp.ndarray:
        """Process input sequence using amplitude-velocity dynamics."""
        seq_len, batch_size, input_dim = inputs.shape
        
        if initial_state is None:
            if use_phase_init:
                x, v = self.cell.get_initial_state_from_phases(batch_size)
                initial_state = (x, v)
            else:
                x = jnp.zeros((batch_size, self.cell.hidden_dim))
                v = jnp.zeros((batch_size, self.cell.hidden_dim))
                initial_state = (x, v)
        
        def scan_fn(carry, x_input):
            output, new_state = self.cell(x_input, carry, use_phase_init=False)
            return new_state, output
        
        final_state, outputs = jax.lax.scan(scan_fn, initial_state, inputs)
        
        if return_trajectories:
            return {"outputs": outputs, "final_state": final_state}
        else:
            return outputs


# ======== AMPLITUDE-VELOCITY ENCODER/DECODER ========

class AmplitudeVelocityEncoder(eqx.Module):
    """Enhanced encoder using amplitude-velocity HORN dynamics."""
    
    rnn: AmplitudeVelocityHORN
    to_latent: eqx.nn.Linear
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        oscillator_class = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict] = None,
        initial_phases: Optional[jnp.ndarray] = None,
        initial_amplitude: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize amplitude-velocity encoder."""
        keys = jax.random.split(key, 2)
        
        self.rnn = AmplitudeVelocityHORN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            initial_phases=initial_phases,
            initial_amplitude=initial_amplitude,
            key=keys[0]
        )
        
        self.to_latent = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[1])
    
    def __call__(
        self, 
        inputs: jnp.ndarray, 
        use_phase_init: bool = False
    ) -> jnp.ndarray:
        """Encode sequence to latent representation."""
        outputs = self.rnn(inputs, use_phase_init=use_phase_init)
        final_output = outputs[-1]
        latent = jax.vmap(self.to_latent)(final_output)
        return latent


class AmplitudeVelocityDecoder(eqx.Module):
    """Enhanced decoder using amplitude-velocity HORN dynamics."""
    
    from_latent: eqx.nn.Linear
    rnn: AmplitudeVelocityHORN
    sequence_length: int = eqx.static_field()
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        sequence_length: int,
        oscillator_class = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict] = None,
        initial_phases: Optional[jnp.ndarray] = None,
        initial_amplitude: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize amplitude-velocity decoder."""
        keys = jax.random.split(key, 2)
        
        self.sequence_length = sequence_length
        self.from_latent = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[0])
        
        self.rnn = AmplitudeVelocityHORN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            initial_phases=initial_phases,
            initial_amplitude=initial_amplitude,
            key=keys[1]
        )
    
    def __call__(
        self, 
        latent: jnp.ndarray, 
        use_phase_init: bool = False
    ) -> jnp.ndarray:
        """Decode latent representation to output sequence."""
        batch_size = latent.shape[0]
        
        # Project latent to hidden space
        hidden = jax.vmap(self.from_latent)(latent)
        
        # Create input sequence by repeating latent projection
        inputs = jnp.broadcast_to(
            hidden[None, :, :], 
            (self.sequence_length, batch_size, hidden.shape[-1])
        )
        
        # Process through amplitude-velocity RNN
        outputs = self.rnn(inputs, use_phase_init=use_phase_init)
        
        return outputs


# ======== AMPLITUDE-VELOCITY AUTOENCODER ========

class AmplitudeVelocityAutoencoder(eqx.Module):
    """Complete autoencoder using amplitude-velocity oscillator dynamics."""
    
    encoder: AmplitudeVelocityEncoder
    decoder: AmplitudeVelocityDecoder
    
    def __init__(
        self,
        input_dim: int = 16,  # 4x4 patch
        hidden_dim: int = 64,
        latent_dim: int = 32,
        oscillator_class = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict] = None,
        encoder_phases: Optional[jnp.ndarray] = None,
        decoder_phases: Optional[jnp.ndarray] = None,
        initial_amplitude: float = 0.1,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """Initialize amplitude-velocity autoencoder."""
        if key is None:
            key = jax.random.PRNGKey(42)
            
        keys = jax.random.split(key, 2)
        
        self.encoder = AmplitudeVelocityEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            initial_phases=encoder_phases,
            initial_amplitude=initial_amplitude,
            key=keys[0]
        )
        
        self.decoder = AmplitudeVelocityDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            sequence_length=49,  # 7x7 patches for 28x28 images
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            initial_phases=decoder_phases,
            initial_amplitude=initial_amplitude,
            key=keys[1]
        )
    
    def __call__(
        self, 
        x: jnp.ndarray, 
        use_phase_init: bool = True
    ) -> jnp.ndarray:
        """Forward pass with patch-based processing."""
        batch_size = x.shape[0]
        
        # Convert flat MNIST to patches (4x4 patches in 7x7 grid)
        x_images = x.reshape(batch_size, 28, 28)
        patches = x_images.reshape(batch_size, 7, 4, 7, 4)
        patches = patches.transpose(0, 1, 3, 2, 4)
        patches = patches.reshape(batch_size, 49, 16)
        
        # Convert to sequence format: (49, batch, 16)
        patch_sequence = patches.transpose(1, 0, 2)
        
        # Encode patches to latent using amplitude-velocity dynamics
        latent = self.encoder(patch_sequence, use_phase_init=use_phase_init)
        
        # Decode latent to patch sequence using amplitude-velocity dynamics
        decoded_patches = self.decoder(latent, use_phase_init=use_phase_init)
        
        # Convert back to image format
        decoded_patches = decoded_patches.transpose(1, 0, 2)
        reconstructed = decoded_patches.reshape(batch_size, 7, 7, 4, 4)
        reconstructed = reconstructed.transpose(0, 1, 3, 2, 4)
        reconstructed = reconstructed.reshape(batch_size, 28, 28)
        
        # Flatten back to (batch_size, 784)
        return reconstructed.reshape(batch_size, 784)


# ======== DATA LOADING ========

def load_mnist_data(subset_size=None):
    """Load MNIST dataset from TensorFlow Datasets"""
    import tensorflow_datasets as tfds
    
    train_ds = tfds.as_numpy(tfds.load('mnist', split='train', batch_size=-1))
    test_ds = tfds.as_numpy(tfds.load('mnist', split='test', batch_size=-1))
    
    train_images = train_ds['image'].astype(jnp.float32) / 255.0
    test_images = test_ds['image'].astype(jnp.float32) / 255.0
    train_labels = train_ds['label']
    test_labels = test_ds['label']
    
    train_images = train_images.reshape(-1, 28*28)
    test_images = test_images.reshape(-1, 28*28)
    
    if subset_size is not None:
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]
        test_images = test_images[:subset_size]
        test_labels = test_labels[:subset_size]
    
    return train_images, train_labels, test_images, test_labels


def prepare_reconstructions(model, samples):
    """Prepare reconstructions for visualization"""
    originals = samples
    reconstructions = model(samples)
    
    h, w = (28, 28)
    batch_size = samples.shape[0]
    
    # Ensure originals are in the right format for visualization
    if originals.ndim == 2 and originals.shape[1] == 784:
        originals_reshaped = originals.reshape(batch_size, h, w)
    elif originals.ndim == 3:
        originals_reshaped = originals
    else:
        originals_reshaped = originals.reshape(batch_size, h, w)
    
    # Ensure reconstructions are in the right format
    if reconstructions.ndim == 2 and reconstructions.shape[1] == 784:
        reconstructions_reshaped = reconstructions.reshape(batch_size, h, w)
    elif reconstructions.ndim == 3:
        reconstructions_reshaped = reconstructions
    else:
        reconstructions_reshaped = reconstructions.reshape(batch_size, h, w)
    
    return originals_reshaped, reconstructions_reshaped


# ======== EXPORT ENCODER COMPLEX STATES ========

def export_encoder_complex_states(model: AmplitudeVelocityAutoencoder,
                                  images: jnp.ndarray,
                                  save_path: str,
                                  use_phase_init: bool = True,
                                  eps: float = 1e-8) -> None:
    """Export encoder final complex states z = x + i*(v/omega) to NPZ.

    Saves arrays with keys: 'amplitude' (N,H), 'phase' (N,H).
    """
    # Prepare patch sequence as in model forward
    batch_size = images.shape[0]
    x_images = images.reshape(batch_size, 28, 28)
    patches = x_images.reshape(batch_size, 7, 4, 7, 4)
    patches = patches.transpose(0, 1, 3, 2, 4)
    patches = patches.reshape(batch_size, 49, 16)
    patch_sequence = patches.transpose(1, 0, 2)  # (49, B, 16)

    # Run encoder RNN to get final (x, v) state
    r = model.encoder.rnn(patch_sequence, return_trajectories=True, use_phase_init=use_phase_init)
    (x_state, v_state) = r["final_state"]  # shapes: (B, H)

    # Get per-unit omega (scalar or vector)
    osc = model.encoder.rnn.cell.oscillator
    if hasattr(osc, 'omega'):
        omega = osc.omega
        if isinstance(omega, (int, float)):
            omega = jnp.ones((x_state.shape[-1],), dtype=jnp.float32) * float(omega)
        else:
            omega = jnp.asarray(omega)
            if omega.ndim == 0:
                omega = jnp.ones((x_state.shape[-1],), dtype=omega.dtype) * float(omega)
            if omega.shape[-1] != x_state.shape[-1]:
                omega = jnp.broadcast_to(omega, (x_state.shape[-1],))
    else:
        omega = jnp.ones((x_state.shape[-1],), dtype=jnp.float32)

    z = x_state + 1j * (v_state / (omega + eps))
    amplitude = jnp.abs(z)
    phase = jnp.angle(z)

    # Save to NPZ
    np.savez(save_path,
             amplitude=np.asarray(amplitude),
             phase=np.asarray(phase))

# ======== MAIN TRAINING FUNCTION ========

def main():
    """Train amplitude-velocity autoencoder using oscnet utilities"""
    
    # ======== SETUP USING OSCNET UTILITIES ========
    
    # Define output directory
    output_dir = Path("outputs/amplitude_velocity_autoencoder_testE_learnable_vs_adaptive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger using oscnet utilities
    enable_debug_logging = False
    enable_file_logging = True
    
    if enable_debug_logging:
        log_file = output_dir / "training_debug.log"
        logger = setup_application_logger(
            name="amplitude_velocity_autoencoder",
            level=logging.DEBUG, 
            enable_file_logging=enable_file_logging, 
            log_file=str(log_file)
        )
    else:
        log_file = output_dir / "training.log"
        logger = setup_application_logger(
            name="amplitude_velocity_autoencoder",
            level=logging.INFO,
            enable_file_logging=enable_file_logging,
            log_file=str(log_file)
        )
    
    logger.info("🚀 INITIALIZING AMPLITUDE-VELOCITY AUTOENCODER")
    logger.info("="*60)
    
    # Setup JAX optimizations using oscnet utilities
    if enable_debug_logging:
        pass  # Keep verbose logging for debugging
    else:
        disable_verbose_jax_logging()
    
    device_type = optimize_for_device()
    primary_device = get_device_info()
    setup_memory_optimization()
    disable_nan_debugging()  # For production performance
    monitor_memory_usage()
    
    # Set random seed
    main_key = jax.random.PRNGKey(42)
    
    # Create organized subdirectories
    checkpoints_dir = output_dir / "checkpoints"
    visualizations_dir = output_dir / "visualizations" 
    metrics_dir = output_dir / "metrics"
    
    checkpoints_dir.mkdir(exist_ok=True)
    visualizations_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("\n📊 Loading MNIST data...")
    train_images, train_labels, test_images, test_labels = load_mnist_data(subset_size=10000)
    
    # Place data on optimal device using oscnet utilities
    logger.info(f"📍 Placing data on {primary_device}")
    train_images = jax.device_put(train_images, primary_device)
    test_images = jax.device_put(test_images, primary_device)
    
    # Model parameters
    batch_size = 32
    n_batches = len(train_images) // batch_size
    hidden_dim = 64
    latent_dim = 64
    num_epochs = 50
    initial_lr = 2e-3
    min_lr = 1e-6
    use_memory_efficient = True
    use_criticality_init = True
    
    logger.info(f"\n🏗️  MODEL CONFIGURATION:")
    logger.info(f"   Architecture: {hidden_dim}→{latent_dim}")
    logger.info(f"   Training: {num_epochs} epochs, {batch_size} batch size")
    logger.info(f"   Optimizations: Memory-efficient={use_memory_efficient}, Criticality={use_criticality_init}")

    # Oscillator parameters using oscnet criticality utilities
    if use_criticality_init:
        logger.info("\n🧪 Initializing parameters for edge of chaos...")
        main_key, subkey = jax.random.split(main_key)
        osc_params = CriticalityInitializer.initialize_for_criticality(
            dim=hidden_dim, 
            target_lyapunov=0.01,
            include_phases=True,
            phase_strategy="optimized",
            key=subkey
        )
        
        encoder_phases = osc_params.get("phases", None)
        decoder_phases = osc_params.get("phases", None)
        
        # Clean oscillator parameters
        valid_osc_keys = ['alpha', 'omega', 'gamma', 'dt']
        osc_params_clean = {k: v for k, v in osc_params.items() if k in valid_osc_keys}
        
        # Ensure arrays for vector parameters
        if 'omega' in osc_params_clean and not isinstance(osc_params_clean["omega"], jnp.ndarray):
            osc_params_clean["omega"] = jnp.ones(hidden_dim) * osc_params_clean["omega"]
        if 'gamma' in osc_params_clean and not isinstance(osc_params_clean["gamma"], jnp.ndarray):
            osc_params_clean["gamma"] = jnp.ones(hidden_dim) * osc_params_clean["gamma"]
        if 'dt' not in osc_params_clean:
            osc_params_clean['dt'] = 1.0 
        if 'alpha' not in osc_params_clean:
             osc_params_clean['alpha'] = 0.04

        logger.info(f"   ✅ Critical params: α={osc_params_clean.get('alpha', 'N/A'):.3f}")
        logger.info(f"   🌊 Phase criticality score: {osc_params.get('phase_criticality_score', 0.0):.3f}")
       
    else:
        logger.info("\n📊 Using standard oscillator parameters...")
        osc_params_clean = {
            "alpha": 0.5,
            "omega": 2.0 * jnp.pi / 7.0,
            "gamma": 0.1,
            "dt": 1.0
        }
        if not isinstance(osc_params_clean["omega"], jnp.ndarray):
            osc_params_clean["omega"] = jnp.ones(hidden_dim) * osc_params_clean["omega"]
        if not isinstance(osc_params_clean["gamma"], jnp.ndarray):
            osc_params_clean["gamma"] = jnp.ones(hidden_dim) * osc_params_clean["gamma"]
            
        encoder_phases = None
        decoder_phases = None

    # ======== 🌊 ADAPTIVE OSCILLATOR CONFIGURATION ========
    # Choose your oscillator type for maximum performance:
    
    # OPTION 1: Traditional HORN (baseline, weaker performance)
    # oscillator_class = NonlinearHarmonicOscillator
    # oscillator_params = osc_params_clean
    
    # OPTION 2: 🚀 LEARNABLE OSCILLATORS (recommended!)
    # Each oscillator learns task-specific ω and γ for optimal performance
    oscillator_class = LearnableNonlinearHarmonicOscillator
    oscillator_params = {
        "alpha": osc_params_clean.get("alpha", 0.04),  # Keep α fixed for stability
        "omega_init": osc_params_clean.get("omega", 2.0 * jnp.pi / 28.0),
        "gamma_init": osc_params_clean.get("gamma", 0.01),
        "omega_bounds": (jnp.pi/56, 4*jnp.pi/7),  # Wide range for specialization
        "gamma_bounds": (0.001, 0.2),  # Stable damping range
        "dt": osc_params_clean.get("dt", 1.0)
    }
    
    # OPTION 3: 🔥 ADAPTIVE MULTIPLIER OSCILLATORS (research)
    # Uses multipliers on base frequencies - more interpretable
    # oscillator_class = AdaptiveNonlinearHarmonicOscillator
    # oscillator_params = {
    #     "alpha": osc_params_clean.get("alpha", 0.04),
    #     "base_omega": 2.0 * jnp.pi / 28.0,
    #     "base_gamma": 0.01,
    #     "omega_multiplier_bounds": (0.25, 4.0),  # 0.25x to 4x base frequency
    #     "gamma_multiplier_bounds": (0.1, 20.0),  # 0.1x to 20x base damping
    #     "dt": 1.0
    # }
    
    logger.info(f"\n🎯 OSCILLATOR CONFIGURATION:")
    logger.info(f"   Type: {oscillator_class.__name__}")
    if oscillator_class == LearnableNonlinearHarmonicOscillator:
        logger.info(f"   🎵 Frequency range: [{oscillator_params['omega_bounds'][0]:.3f}, {oscillator_params['omega_bounds'][1]:.3f}]")
        logger.info(f"   ⚖️  Damping range: [{oscillator_params['gamma_bounds'][0]:.3f}, {oscillator_params['gamma_bounds'][1]:.3f}]")
        logger.info(f"   🚀 Expected: Each oscillator will specialize for different temporal patterns!")
    elif oscillator_class == AdaptiveNonlinearHarmonicOscillator:
        logger.info(f"   🎼 MUSICAL APPROACH: Learning harmonic multipliers on base frequencies!")
        logger.info(f"   🎵 Base frequency: {oscillator_params['base_omega']:.4f} rad/s")
        logger.info(f"   🎶 Frequency multiplier range: {oscillator_params['omega_multiplier_bounds']} (like musical harmonics!)")
        logger.info(f"   🥁 Damping multiplier range: {oscillator_params['gamma_multiplier_bounds']}")
        logger.info(f"   📊 Base γ: {oscillator_params['base_gamma']:.4f}")
        logger.info(f"   🎹 Expected: Oscillators will discover harmonic relationships in the data!")
    else:
        logger.info(f"   📊 Traditional fixed parameters")
        logger.info(f"   ⚠️  Note: Using learnable oscillators can give 1.65x better performance!")

    # Create amplitude-velocity model
    logger.info("\n🏗️  Creating amplitude-velocity autoencoder...")
    main_key, subkey = jax.random.split(main_key)
    
    model = AmplitudeVelocityAutoencoder(
        input_dim=16,  # 4x4 patches
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        oscillator_class=oscillator_class,
        oscillator_params=oscillator_params,
        encoder_phases=encoder_phases,
        decoder_phases=decoder_phases,
        initial_amplitude=0.1,
        key=subkey
    )
    
    # Place model on optimal device
    model = jax.device_put(model, primary_device)
    
    # Print model summary using oscnet utilities
    logger.info("\n" + "="*80)
    logger.info("📊 MODEL ARCHITECTURE ANALYSIS")
    logger.info("="*80)
    param_summary = print_model_summary(model, "Amplitude-Velocity Autoencoder")
    efficiency_analysis = analyze_model_efficiency(model, input_dim=784, output_dim=784)
    
    # Setup optimizer using oscnet utilities
    logger.info("\n⚙️  Setting up optimizer...")

    scheduler = create_scheduler(
        scheduler_type="cosine",
        initial_lr=initial_lr,
        min_lr=min_lr,
        cycle_steps=num_epochs * n_batches
    )
    
    optimizer = wrap_optimizer_with_scheduler(
        lambda lr: optax.adamw(learning_rate=lr, weight_decay=2e-4),
        scheduler
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # JIT compilation warmup using oscnet utilities
    logger.info("\n" + "="*60)
    logger.info("🔥 JIT COMPILATION WARMUP")
    logger.info("="*60)
    sample_batch = train_images[:batch_size]
    
    # Use oscnet warmup utility
    warmup_model_compilation(model, opt_state, optimizer, sample_batch, mse_loss)
    
    # Test initial loss
    initial_loss = mse_loss(model, sample_batch)
    logger.info(f"🎯 Initial loss: {initial_loss:.6f}")
    
    # Initialize metrics tracking
    metrics = {
        "train_loss": [], 
        "learning_rate": [], 
        "gradient_norm": [],
        "loss_improvement": [],
        "epoch_times": [],
        "memory_efficient": use_memory_efficient,
        "device_type": device_type,
        "stochastic_forcing": False
    }
    global_step = 0
    
    # Create hyperparameters dictionary for checkpointing
    hyperparams = {
        "input_dim": 16,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "oscillator_class": oscillator_class.__name__,
        "oscillator_params": oscillator_params,
        "encoder_phases": encoder_phases.tolist() if encoder_phases is not None else None,
        "decoder_phases": decoder_phases.tolist() if decoder_phases is not None else None,
        "initial_amplitude": 0.1,
        "use_criticality_init": use_criticality_init,
        "stochastic_forcing_config": None
    }
    
    # ======== TRAINING LOOP USING OSCNET UTILITIES ========
    logger.info(f"\n🏋️ Beginning amplitude-velocity training...")
    logger.info(f"⚡ Using oscnet optimized training utilities")
    
    training_start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Split key for this epoch
        epoch_key, main_key = jax.random.split(main_key)
        
        # Create batches using oscnet utility
        batches = create_batch_iterator(train_images, batch_size, epoch_key, device=primary_device)
        
        # Time the epoch
        epoch_start = time.time()
        
        # Use standard reconstruction loss and training
        current_loss_fn = mse_loss
        loss_type = "RECON"
        
        # Run optimized epoch training using oscnet utilities
        model, opt_state, losses, grad_norms = train_epoch_scan(
            model, opt_state, batches, optimizer, current_loss_fn, 
            max_norm=1.0, use_memory_efficient=use_memory_efficient
        )
        
        # Update global step counter
        global_step += n_batches
        
        epoch_time = time.time() - epoch_start
        avg_loss = jnp.mean(losses)
        avg_grad_norm = jnp.mean(grad_norms)
        
        # Calculate performance metrics
        batches_per_sec = n_batches / epoch_time
        samples_per_sec = (n_batches * batch_size) / epoch_time
        
        # Check for improvement
        improvement = best_loss - avg_loss if best_loss != float('inf') else 0.0
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        # Track metrics
        metrics["train_loss"].append(float(avg_loss))
        metrics["gradient_norm"].append(float(avg_grad_norm))
        metrics["loss_improvement"].append(float(improvement))
        metrics["epoch_times"].append(float(epoch_time))
        
        # Track learning rate
        current_lr = scheduler(global_step) if hasattr(scheduler, '__call__') else 2e-3
        metrics["learning_rate"].append(float(current_lr))
        
        # Enhanced progress display
        progress_bar = "🔥" if is_best else "📈"
        
        logger.info(f"{progress_bar} Epoch {epoch}/{num_epochs} | "
              f"Loss: {avg_loss:.6f} | "
              f"Time: {epoch_time:.2f}s | "
              f"{batches_per_sec:.1f} batch/s | "
              f"{samples_per_sec:.0f} samples/s | "
              f"{loss_type} | "
              f"Improvement: {improvement:.6f}")
        
        # Save checkpoints using oscnet utilities
        save_checkpoint_this_epoch = (epoch % 20 == 0) or (epoch == num_epochs)
        
        # Save best model checkpoint using oscnet utilities
        if is_best and save_checkpoint_this_epoch:
            save_equinox_checkpoint(
                model, opt_state, epoch, 
                {"loss": float(avg_loss), "grad_norm": float(avg_grad_norm)},
                checkpoints_dir, hyperparams, is_best=True
            )
        
        # Save regular checkpoint every 20 epochs using oscnet utilities
        if epoch % 20 == 0:
            save_equinox_checkpoint(
                model, opt_state, epoch,
                {"loss": float(avg_loss), "grad_norm": float(avg_grad_norm)},
                checkpoints_dir, hyperparams, is_best=False
            )
            
            # Save metrics using oscnet utilities
            save_training_metrics(metrics, metrics_dir, epoch)
        
        # Save visualizations every 20 epochs
        if epoch % 20 == 0:
            sample_indices = jnp.arange(8)
            sample_batch = train_images[sample_indices]
            originals, reconstructions = prepare_reconstructions(
                model, sample_batch
            )
            
            try:
                # Use oscnet visualization utilities
                recon_fig = visualize_reconstructions(
                    original=originals,
                    reconstructed=reconstructions,
                    title=f"Amplitude-Velocity MNIST Reconstructions (Epoch {epoch})"
                )
                
                viz_path = visualizations_dir / f"reconstructions_epoch_{epoch:03d}.png"
                recon_fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close(recon_fig)
                
                # Generate training metrics visualization
                if len(metrics["train_loss"]) > 1:
                    metrics_fig = visualize_training_metrics(
                        metrics=metrics,
                        title=f"Amplitude-Velocity Training Progress (Epoch {epoch})",
                        save_path=visualizations_dir / f"training_progress_epoch_{epoch:03d}.png",
                        show=False
                    )
                    plt.close(metrics_fig)
                
            except Exception as e:
                logger.warning(f"⚠️  Error creating visualizations: {e}")
        
        # Live training progress visualization (updates every 5 epochs)
        if epoch % 5 == 0 and len(metrics["train_loss"]) > 1:
            try:
                live_metrics_fig = visualize_training_metrics(
                    metrics=metrics,
                    title=f"LIVE Training Progress (Epoch {epoch}/{num_epochs})",
                    save_path=visualizations_dir / "training_progress_live.png",
                    show=False
                )
                plt.close(live_metrics_fig)
                
            except Exception as e:
                logger.warning(f"⚠️  Error creating live visualization: {e}")
        
        # Early stopping check
        if avg_loss < 0.001:
            logger.info(f"🎯 Early stopping: Loss {avg_loss:.6f} below threshold")
            break
    
    # Performance summary
    total_time = time.time() - training_start_time
    logger.info(f"\n🎉 Amplitude-velocity training complete!")
    logger.info(f"⏱️  Total training time: {total_time:.2f}s")
    logger.info(f"📊 Final loss: {best_loss:.6f}")
    logger.info(f"🚀 Average speed: {num_epochs * n_batches * batch_size / total_time:.0f} samples/s")
    
    # Save final checkpoint using oscnet utilities
    logger.info("💾 Saving final checkpoint...")
    save_equinox_checkpoint(
        model, opt_state, epoch, 
        {"final_loss": float(best_loss), "total_epochs": epoch},
        checkpoints_dir, hyperparams, is_best=False
    )
    
    save_equinox_checkpoint(
        model, opt_state, epoch, 
        {"final_best_loss": float(best_loss), "total_epochs": epoch},
        checkpoints_dir, hyperparams, is_best=True
    )
    
    # Generate final visualizations
    try:
        logger.info("\n🎨 Creating final visualizations...")
        
        final_sample_batch = test_images[:16]
        final_originals, final_reconstructions = prepare_reconstructions(
            model, final_sample_batch
        )
        
        final_enhanced_fig = visualize_reconstructions(
            original=final_originals,
            reconstructed=final_reconstructions,
            title="Amplitude-Velocity MNIST Reconstructions (Final)"
        )
        
        final_viz_path = visualizations_dir / "reconstructions_final.png"
        final_enhanced_fig.savefig(final_viz_path, dpi=150, bbox_inches='tight')
        plt.close(final_enhanced_fig)
        
        # Final training metrics visualization
        if len(metrics["train_loss"]) > 1:
            final_metrics_fig = visualize_training_metrics(
                metrics=metrics,
                title="Amplitude-Velocity Training Progress (Complete)",
                save_path=visualizations_dir / "training_progress_final.png",
                show=False
            )
            plt.close(final_metrics_fig)
        
    except Exception as e:
        logger.warning(f"⚠️  Error creating final visualizations: {e}")
    
    # Optional comprehensive evaluation
    perform_comprehensive_evaluation = True

    if perform_comprehensive_evaluation:
        # ======== COMPREHENSIVE EVALUATION USING OSCNET UTILITIES ========
        logger.info(f"\n🔬 STARTING COMPREHENSIVE EVALUATION SUITE...")
        logger.info(f"═══════════════════════════════════════════════")
        
        try:
            # Comprehensive model evaluation using oscnet utilities
            eval_results = comprehensive_model_evaluation(
                model=model,
                test_images=test_images,
                test_labels=test_labels,
                output_dir=output_dir,
                training_metrics=metrics
            )
            
            # Print evaluation summary
            overall_score = eval_results.get('overall_score', 0.0)
            logger.info(f"\n🏆 EVALUATION RESULTS SUMMARY")
            logger.info(f"═══════════════════════════════════════════════")
            logger.info(f"📊 Overall Quality Score: {overall_score:.3f}/5.0")
            
            if overall_score >= 4.0:
                logger.info(f"   🏆 EXCELLENT: Outstanding performance!")
            elif overall_score >= 3.0:
                logger.info(f"   🥉 GOOD: Solid performance.")
            elif overall_score >= 2.0:
                logger.info(f"   ⚠️  FAIR: Acceptable performance.")
            else:
                logger.info(f"   ❌ POOR: Needs improvement.")
            
            logger.info(f"\n✨ COMPREHENSIVE EVALUATION COMPLETE!")
            
        except Exception as e:
            logger.warning(f"❌ Error during comprehensive evaluation: {e}")
        
        # Enhanced evaluation using oscnet utilities
        try:
            logger.info("\n🔬 COMPREHENSIVE RESONANCE ANALYSIS")
            logger.info("="*60)
            
            evaluation_results = comprehensive_resonance_analysis(
                model, test_images, output_dir,
                model_name="Amplitude-Velocity Autoencoder",
                n_samples=100
            )
            
            logger.info("🔍 Resonance analysis complete!")
            
            create_enhanced_visualizations(
                model, test_images, test_labels, output_dir
            )
            
            logger.info("🎨 Enhanced visualization suite complete!")
            
        except Exception as e:
            logger.warning(f"❌ Error during enhanced evaluation: {e}")
        
    logger.info(f"\n🎉 Training complete! Results saved to {output_dir}")
    logger.info(f"📊 Best loss achieved: {best_loss:.6f}")
    logger.info(f"⚡ Powered by oscnet utilities for maximum performance!")
    
    # Save final training metrics using oscnet utilities
    metrics["final_loss"] = float(best_loss)
    metrics["total_epochs"] = epoch
    metrics["total_training_time"] = total_time
    
    save_training_metrics(metrics, metrics_dir, epoch)


if __name__ == "__main__":
    main()
