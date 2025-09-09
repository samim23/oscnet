"""
JAX optimization and device management utilities.

This module provides utilities for optimizing JAX performance, managing devices,
and configuring JAX settings for different hardware configurations.
"""

import jax
import jax.numpy as jnp
import logging
import os
from typing import Any

# Get logger for this module
logger = logging.getLogger(__name__)


def get_device_info():
    """
    Get information about available devices and return the optimal device.
    
    Returns:
        jax.Device: The optimal device (GPU preferred, then CPU)
    """
    devices = jax.devices()
    logger.info(f"🖥️  Available devices: {[d.device_kind for d in devices]}")
    
    # Prefer GPU if available
    gpu_devices = [d for d in devices if d.device_kind == 'gpu']
    if gpu_devices:
        logger.info(f"   🚀 Using GPU: {gpu_devices[0]}")
        return gpu_devices[0]
    else:
        logger.info(f"   💻 Using CPU: {devices[0]}")
        return devices[0]


def setup_memory_optimization():
    """
    Configure JAX for optimal memory usage.
    
    Sets environment variables for dynamic memory allocation and
    limits memory fraction to prevent OOM errors.
    """
    # Enable dynamic memory allocation for better performance
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    
    logger.info("🧠 Memory optimization configured:")
    logger.info("   - Dynamic memory allocation enabled")
    logger.info("   - Memory fraction limited to 80%")


def optimize_for_device():
    """
    Optimize JAX settings based on available hardware.
    
    Returns:
        str: Device type ('gpu', 'tpu', or 'cpu')
    """
    try:
        devices = jax.devices()
        device_kinds = [d.device_kind for d in devices]
        
        if 'gpu' in device_kinds:
            logger.info("🚀 GPU detected - optimizing for GPU performance")
            # GPU-specific optimizations
            jax.config.update("jax_default_matmul_precision", "float32")
            return "gpu"
        elif 'tpu' in device_kinds:
            logger.info("⚡ TPU detected - optimizing for TPU performance") 
            # TPU-specific optimizations
            jax.config.update("jax_default_matmul_precision", "bfloat16")
            return "tpu"
        else:
            logger.info("💻 CPU detected - optimizing for CPU performance")
            # CPU-specific optimizations
            jax.config.update("jax_default_matmul_precision", "float32")
            return "cpu"
    except Exception as e:
        logger.warning(f"⚠️  Device optimization error: {e}")
        logger.info("💻 Defaulting to CPU optimizations")
        jax.config.update("jax_default_matmul_precision", "float32")
        return "cpu"


def monitor_memory_usage():
    """
    Monitor JAX memory usage across available devices.
    
    Attempts to provide memory information for GPU devices when available.
    """
    try:
        # Try to get GPU memory info if available
        devices = jax.devices()
        for device in devices:
            if device.device_kind == 'gpu':
                # This is device-specific and may not work on all systems
                logger.info(f"🖥️  Device: {device}")
                # Memory monitoring would be device-specific
    except Exception as e:
        logger.warning(f"⚠️  Could not monitor memory usage: {e}")


def setup_nan_debugging():
    """
    Enable NaN debugging for development.
    
    Warning: This adds significant overhead and should be disabled in production.
    """
    jax.config.update("jax_debug_nans", True)
    logger.info("🐛 NaN debugging enabled (disable in production)")


def disable_nan_debugging():
    """
    Disable NaN debugging for production performance.
    """
    jax.config.update("jax_debug_nans", False)
    logger.info("🚀 NaN debugging disabled for production performance")


def enable_verbose_jax_logging():
    """
    Enable verbose JAX logging for debugging.
    
    Useful for debugging compilation issues and understanding JAX behavior.
    """
    jax.config.update("jax_log_compiles", True)
    logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.WARNING)
    logging.getLogger("jax._src.dispatch").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logger.info("🔊 JAX verbose logging enabled for debugging")


def disable_verbose_jax_logging():
    """
    Disable verbose JAX logging for clean output (default).
    
    Recommended for production and clean training output.
    """
    jax.config.update("jax_log_compiles", False)
    logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.ERROR)
    logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.info("🔇 JAX verbose logging disabled for clean output")


def create_batch_iterator(data, batch_size, key, device=None):
    """
    Create shuffled batches with optional device placement.
    
    Args:
        data: Input data array
        batch_size: Size of each batch
        key: JAX random key for shuffling
        device: Optional device for data placement
        
    Returns:
        jnp.ndarray: Batched data (n_batches, batch_size, ...)
    """
    n_samples = len(data)
    n_batches = n_samples // batch_size
    
    # Shuffle data
    perm = jax.random.permutation(key, n_samples)
    shuffled_data = data[perm]
    
    # Create batches and optionally place on specific device
    batches = shuffled_data[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    
    if device is not None:
        # Place data on specific device for optimal performance
        batches = jax.device_put(batches, device)
    
    return batches 