"""
Utility functions for oscillatory neural networks.

This module provides general-purpose utilities for JAX optimization,
logging, checkpointing, and other common tasks.

Modules:
    jax_utils: JAX optimization and device management utilities
    logging: Logging configuration and setup
    checkpoints: Model checkpointing and serialization utilities
"""

from .jax_utils import (
    get_device_info,
    setup_memory_optimization,
    optimize_for_device,
    monitor_memory_usage,
    setup_nan_debugging,
    disable_nan_debugging,
    enable_verbose_jax_logging,
    disable_verbose_jax_logging,
    create_batch_iterator
)

from .logging import (
    setup_application_logger
)

from .checkpoints import (
    save_equinox_checkpoint,
    load_equinox_checkpoint,
    save_training_metrics
)

__all__ = [
    # JAX utilities
    "get_device_info",
    "setup_memory_optimization", 
    "optimize_for_device",
    "monitor_memory_usage",
    "setup_nan_debugging",
    "disable_nan_debugging",
    "enable_verbose_jax_logging",
    "disable_verbose_jax_logging",
    "create_batch_iterator",
    
    # Logging
    "setup_application_logger",
    
    # Checkpointing
    "save_equinox_checkpoint",
    "load_equinox_checkpoint",
    "save_training_metrics"
] 