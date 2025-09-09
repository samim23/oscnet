"""
Training utilities for oscillatory neural networks.

This module provides optimized training functions including JIT-compiled
training steps, memory-efficient training, and compilation warmup utilities.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import logging
from typing import Tuple, Any, Callable

# Get logger for this module
logger = logging.getLogger(__name__)


@eqx.filter_jit
def train_step(
    model: eqx.Module,
    opt_state: Any,
    batch: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    max_norm: float = 1.0
) -> Tuple[eqx.Module, Any, float, float]:
    """
    Enhanced training step with gradient clipping and proper JIT compilation.
    
    Args:
        model: Equinox model
        opt_state: Optimizer state
        batch: Training batch
        optimizer: Optax optimizer
        loss_fn: Loss function
        max_norm: Maximum gradient norm for clipping
        
    Returns:
        Tuple of (new_model, new_opt_state, loss_value, grad_norm)
    """
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    
    # Calculate gradient norm for monitoring
    grad_norm = optax.global_norm(grads)
    
    # Clip gradients for stability
    grads = jax.tree.map(
        lambda g: g * jnp.minimum(1.0, max_norm / (grad_norm + 1e-8)), 
        grads
    )
    
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss_value, grad_norm


@eqx.filter_jit
def memory_efficient_train_step(
    model: eqx.Module,
    opt_state: Any,
    batch: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    max_norm: float = 1.0
) -> Tuple[eqx.Module, Any, float, float]:
    """
    Memory-efficient training step with gradient checkpointing.
    
    Args:
        model: Equinox model
        opt_state: Optimizer state
        batch: Training batch
        optimizer: Optax optimizer
        loss_fn: Loss function
        max_norm: Maximum gradient norm for clipping
        
    Returns:
        Tuple of (new_model, new_opt_state, loss_value, grad_norm)
    """
    # For MSE loss, use memory-efficient checkpointing
    if loss_fn == mse_loss:
        # Create memory-efficient loss function with checkpointing
        def checkpointed_loss_fn(model, batch):
            def checkpointed_forward(model, batch):
                return model(batch)
            
            # Apply checkpointing to reduce memory usage during backprop
            reconstruction = jax.checkpoint(checkpointed_forward)(model, batch)
            batch_flat = batch.reshape(batch.shape[0], -1)
            return jnp.mean((reconstruction - batch_flat) ** 2)
        
        # Use memory-efficient loss computation
        loss_value, grads = eqx.filter_value_and_grad(checkpointed_loss_fn)(model, batch)
    else:
        # For complex loss functions (like EOC), we can't checkpoint easily
        # because they need access to model internals. Use standard computation.
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    
    # Calculate gradient norm for monitoring
    grad_norm = optax.global_norm(grads)
    
    # Gradient clipping with adaptive scaling
    clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
    grads = jax.tree.map(lambda g: g * clip_factor, grads)
    
    # Apply updates
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss_value, grad_norm


@eqx.filter_jit
def train_epoch_scan(
    model: eqx.Module,
    opt_state: Any,
    batches: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    max_norm: float = 1.0,
    use_memory_efficient: bool = True
) -> Tuple[eqx.Module, Any, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled epoch training using scan for maximum performance.
    Processes all batches in one compiled function call.
    
    Args:
        model: Equinox model
        opt_state: Optimizer state
        batches: All batches for the epoch (n_batches, batch_size, ...)
        optimizer: Optax optimizer
        loss_fn: Loss function
        max_norm: Maximum gradient norm for clipping
        use_memory_efficient: Whether to use memory-efficient training
        
    Returns:
        Tuple of (final_model, final_opt_state, losses, grad_norms)
    """
    def scan_fn(carry, batch):
        model, opt_state = carry
        if use_memory_efficient:
            new_model, new_opt_state, loss, grad_norm = memory_efficient_train_step(
                model, opt_state, batch, optimizer, loss_fn, max_norm
            )
        else:
            new_model, new_opt_state, loss, grad_norm = train_step(
                model, opt_state, batch, optimizer, loss_fn, max_norm
            )
        return (new_model, new_opt_state), (loss, grad_norm)
    
    (final_model, final_opt_state), (losses, grad_norms) = jax.lax.scan(
        scan_fn, (model, opt_state), batches
    )
    
    return final_model, final_opt_state, losses, grad_norms


def warmup_model_compilation(
    model: eqx.Module,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    sample_batch: jnp.ndarray,
    loss_fn: Callable
):
    """
    Warm up all JIT compilations to exclude compilation time from training timing.
    This is critical for accurate performance measurement.
    
    Args:
        model: Equinox model
        opt_state: Optimizer state
        optimizer: Optax optimizer
        sample_batch: Sample batch for warmup
        loss_fn: Loss function
    """
    logger.info("🔥 Warming up JIT compilations...")
    
    # Warm up forward pass
    _ = model(sample_batch)
    logger.info("   ✅ Forward pass compiled")
    
    # Warm up loss computation
    _ = loss_fn(model, sample_batch)
    logger.info("   ✅ Loss computation compiled")
    
    # Warm up standard training step
    _, _, _, _ = train_step(model, opt_state, sample_batch, optimizer, loss_fn)
    logger.info("   ✅ Standard training step compiled")
    
    # Warm up memory-efficient training step
    _, _, _, _ = memory_efficient_train_step(model, opt_state, sample_batch, optimizer, loss_fn)
    logger.info("   ✅ Memory-efficient training step compiled")
    
    # Warm up batch processing (create small batch set for warmup)
    small_batches = sample_batch[None, :8, :]  # (1, 8, ...)
    _, _, _, _ = train_epoch_scan(model, opt_state, small_batches, optimizer, loss_fn)
    logger.info("   ✅ Epoch scan compiled")
    
    logger.info("🚀 All compilations complete - ready for fast training!")


# Common loss functions
@eqx.filter_jit
def mse_loss(model: eqx.Module, batch: jnp.ndarray) -> float:
    """
    Compute MSE loss between input and reconstruction.
    
    Args:
        model: Equinox model
        batch: Input batch
        
    Returns:
        float: MSE loss value
    """
    reconstruction = model(batch)
    batch_flat = batch.reshape(batch.shape[0], -1)
    return jnp.mean((reconstruction - batch_flat) ** 2) 