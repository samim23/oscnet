"""
Learning utilities for oscillatory neural networks.

This module provides training optimization tools including:
- Learning rate schedulers
- Criticality initialization for edge-of-chaos dynamics
- Training utilities and optimizers
- Annealed stochastic forcing for enhanced oscillator synchronization

Modules:
    schedulers: Learning rate scheduling strategies
    criticality: Edge-of-chaos initialization utilities
    training: Optimized training functions and utilities
    stochastic_forcing: Annealed stochastic forcing techniques
"""

from .schedulers import (
    create_scheduler,
    wrap_optimizer_with_scheduler,
    LRScheduler,
    ConstantScheduler,
    ExponentialDecayScheduler,
    CosineAnnealingScheduler,
    WarmupCooldownScheduler,
    ReduceLROnPlateauScheduler
)

from .criticality import (
    CriticalityInitializer,
    initialize_edge_of_chaos,
    assess_criticality
)

from .training import (
    train_step,
    memory_efficient_train_step,
    train_epoch_scan,
    warmup_model_compilation,
    mse_loss
)

from .stochastic_forcing import (
    StochasticForcingConfig,
    NoiseType,
    NoiseSchedule,
    generate_pink_noise,
    apply_velocity_stochastic_forcing,
    train_step_with_stochastic_forcing,
    train_epoch_with_stochastic_forcing,
    create_stochastic_forcing_config,
    apply_stochastic_forcing_both_states,
    generate_resonant_driving,
    create_resonant_driving_config
)

__all__ = [
    # Schedulers
    "create_scheduler",
    "wrap_optimizer_with_scheduler", 
    "LRScheduler",
    "ConstantScheduler",
    "ExponentialDecayScheduler",
    "CosineAnnealingScheduler",
    "WarmupCooldownScheduler",
    "ReduceLROnPlateauScheduler",
    
    # Criticality
    "CriticalityInitializer",
    "initialize_edge_of_chaos",
    "assess_criticality",
    
    # Training
    "train_step",
    "memory_efficient_train_step", 
    "train_epoch_scan",
    "warmup_model_compilation",
    "mse_loss",
    
    # Stochastic Forcing
    "StochasticForcingConfig",
    "NoiseType",
    "NoiseSchedule", 
    "generate_pink_noise",
    "apply_velocity_stochastic_forcing",
    "train_step_with_stochastic_forcing",
    "train_epoch_with_stochastic_forcing",
    "create_stochastic_forcing_config",
    "apply_stochastic_forcing_both_states",
    "generate_resonant_driving",
    "create_resonant_driving_config"
] 