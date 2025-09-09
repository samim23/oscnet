"""
Learning rate scheduling utilities for oscillatory neural networks.

This module provides various learning rate scheduling techniques that can be
used with JAX-based optimizers (optax) to improve training performance.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Optional, Tuple, Union, Any
import optax


class LRScheduler:
    """Base class for all learning rate schedulers."""
    
    def __init__(self, initial_lr: float):
        """
        Initialize the learning rate scheduler.
        
        Args:
            initial_lr: Initial learning rate
        """
        self.initial_lr = initial_lr
    
    def __call__(self, step: int) -> float:
        """
        Calculate the learning rate for the current step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate for the current step
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def create_schedule(self) -> Callable[[int], float]:
        """
        Create a schedule function that can be passed to optax.
        
        Returns:
            A function mapping step counts to learning rates
        """
        return self.__call__


class ConstantScheduler(LRScheduler):
    """Scheduler that maintains a constant learning rate."""
    
    def __call__(self, step: int) -> float:
        """Return the constant learning rate."""
        return self.initial_lr
    
    def wrap_optimizer(self, optimizer_fn: Callable) -> optax.GradientTransformation:
        """
        Wrap an optimizer with this scheduler.
        
        Args:
            optimizer_fn: Function that creates an optimizer given a learning rate
            
        Returns:
            Wrapped optimizer with scheduling
        """
        return optimizer_fn(self.initial_lr)


class ExponentialDecayScheduler(LRScheduler):
    """
    Exponentially decays the learning rate.
    
    The learning rate is calculated as:
    lr = initial_lr * decay_rate ^ (step / decay_steps)
    """
    
    def __init__(
        self,
        initial_lr: float,
        decay_rate: float,
        decay_steps: int,
        staircase: bool = False
    ):
        """
        Initialize the exponential decay scheduler.
        
        Args:
            initial_lr: Initial learning rate
            decay_rate: Rate of decay (e.g., 0.96)
            decay_steps: Number of steps over which to decay the learning rate
            staircase: If True, decay in discrete intervals
        """
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase
    
    def __call__(self, step: int) -> float:
        """Calculate the exponentially decayed learning rate."""
        if self.staircase:
            power = step // self.decay_steps
        else:
            power = step / self.decay_steps
        
        return self.initial_lr * (self.decay_rate ** power)
    
    def wrap_optimizer(self, optimizer_fn: Callable) -> optax.GradientTransformation:
        """
        Wrap an optimizer with this scheduler.
        
        Args:
            optimizer_fn: Function that creates an optimizer given a learning rate
            
        Returns:
            Wrapped optimizer with scheduling
        """
        schedule_fn = optax.exponential_decay(
            init_value=self.initial_lr,
            transition_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase
        )
        return optimizer_fn(schedule_fn)


class CosineAnnealingScheduler(LRScheduler):
    """
    Cosine annealing learning rate scheduler.
    
    The learning rate follows a cosine curve from the initial learning rate
    to the minimum learning rate over the specified number of steps.
    """
    
    def __init__(
        self,
        initial_lr: float,
        min_lr: float,
        cycle_steps: int,
        warmup_steps: int = 0
    ):
        """
        Initialize the cosine annealing scheduler.
        
        Args:
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            cycle_steps: Number of steps in a complete cycle
            warmup_steps: Number of steps for linear warmup
        """
        super().__init__(initial_lr)
        self.min_lr = min_lr
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
    
    def __call__(self, step: int) -> float:
        """Calculate the cosine annealed learning rate with optional warmup."""
        # Linear warmup
        if step < self.warmup_steps:
            return self.initial_lr * (step / max(1, self.warmup_steps))
        
        # Adjust step to account for warmup
        adjusted_step = step - self.warmup_steps
        
        # Cosine annealing
        progress = min(1.0, adjusted_step / self.cycle_steps)
        cosine_factor = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
    
    def wrap_optimizer(self, optimizer_fn: Callable) -> optax.GradientTransformation:
        """
        Wrap an optimizer with this scheduler.
        
        Args:
            optimizer_fn: Function that creates an optimizer given a learning rate
            
        Returns:
            Wrapped optimizer with scheduling
        """
        if self.warmup_steps > 0:
            warmup_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=self.initial_lr,
                transition_steps=self.warmup_steps
            )
            cosine_schedule = optax.cosine_decay_schedule(
                init_value=self.initial_lr,
                decay_steps=self.cycle_steps,
                alpha=self.min_lr / self.initial_lr
            )
            schedule_fn = optax.join_schedules(
                schedules=[warmup_schedule, cosine_schedule],
                boundaries=[self.warmup_steps]
            )
        else:
            schedule_fn = optax.cosine_decay_schedule(
                init_value=self.initial_lr,
                decay_steps=self.cycle_steps,
                alpha=self.min_lr / self.initial_lr
            )
        
        return optimizer_fn(schedule_fn)


class WarmupCooldownScheduler(LRScheduler):
    """
    Learning rate scheduler with warmup and cooldown phases.
    
    The learning rate increases linearly during warmup, remains constant
    during the middle phase, and then decreases during cooldown.
    """
    
    def __init__(
        self,
        initial_lr: float,
        warmup_steps: int,
        constant_steps: int,
        cooldown_steps: int,
        final_lr: float = 0.0
    ):
        """
        Initialize the warmup-cooldown scheduler.
        
        Args:
            initial_lr: Target learning rate after warmup
            warmup_steps: Number of steps for linear warmup
            constant_steps: Number of steps with constant learning rate
            cooldown_steps: Number of steps for linear cooldown
            final_lr: Final learning rate after cooldown
        """
        super().__init__(initial_lr)
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.cooldown_steps = cooldown_steps
        self.final_lr = final_lr
        self.total_steps = warmup_steps + constant_steps + cooldown_steps
    
    def __call__(self, step: int) -> float:
        """Calculate the learning rate according to warmup and cooldown."""
        if step < self.warmup_steps:
            # Linear warmup
            return (step / max(1, self.warmup_steps)) * self.initial_lr
        
        if step < self.warmup_steps + self.constant_steps:
            # Constant phase
            return self.initial_lr
        
        if step < self.total_steps:
            # Linear cooldown
            cooldown_progress = (step - self.warmup_steps - self.constant_steps) / max(1, self.cooldown_steps)
            return self.initial_lr - (self.initial_lr - self.final_lr) * cooldown_progress
        
        # After the end of the schedule
        return self.final_lr
    
    def wrap_optimizer(self, optimizer_fn: Callable) -> optax.GradientTransformation:
        """
        Wrap an optimizer with this scheduler.
        
        Args:
            optimizer_fn: Function that creates an optimizer given a learning rate
            
        Returns:
            Wrapped optimizer with scheduling
        """
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=self.initial_lr,
            transition_steps=self.warmup_steps
        )
        
        constant_schedule = optax.constant_schedule(self.initial_lr)
        
        cooldown_schedule = optax.linear_schedule(
            init_value=self.initial_lr,
            end_value=self.final_lr,
            transition_steps=self.cooldown_steps
        )
        
        schedule_fn = optax.join_schedules(
            schedules=[warmup_schedule, constant_schedule, cooldown_schedule],
            boundaries=[self.warmup_steps, self.warmup_steps + self.constant_steps]
        )
        
        return optimizer_fn(schedule_fn)


class ReduceLROnPlateauScheduler:
    """
    Reduce learning rate when a metric stops improving.
    
    Note: This doesn't fit the standard LRScheduler interface since it depends
    on metric values, not just step counts. It requires special handling in the training loop.
    """
    
    def __init__(
        self,
        initial_lr: float,
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 0.0,
        verbose: bool = False
    ):
        """
        Initialize the reduce-on-plateau scheduler.
        
        Args:
            initial_lr: Initial learning rate
            factor: Factor by which to reduce the learning rate (e.g., 0.1)
            patience: Number of epochs with no improvement before reducing LR
            threshold: Minimum change to qualify as an improvement
            min_lr: Minimum learning rate
            verbose: Whether to print learning rate changes
        """
        self.current_lr = initial_lr
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        
        # State
        self.best_value = float('inf')
        self.wait_count = 0
    
    def step(self, metric_value: float) -> float:
        """
        Update the learning rate based on the metric value.
        
        Args:
            metric_value: The monitored metric value (e.g., validation loss)
            
        Returns:
            Updated learning rate
        """
        # Check if metric improved
        if metric_value < self.best_value - self.threshold:
            # Metric improved, reset counter
            self.best_value = metric_value
            self.wait_count = 0
        else:
            # Metric did not improve, increment counter
            self.wait_count += 1
            
            # If waited for patience epochs, reduce learning rate
            if self.wait_count >= self.patience:
                new_lr = max(self.current_lr * self.factor, self.min_lr)
                
                # Only update if LR actually changes
                if new_lr < self.current_lr:
                    if self.verbose:
                        print(f"Reducing learning rate from {self.current_lr:.6f} to {new_lr:.6f}")
                    self.current_lr = new_lr
                    self.wait_count = 0  # Reset wait count after reducing
        
        return self.current_lr


def create_scheduler(
    scheduler_type: str,
    initial_lr: float,
    **kwargs
) -> Union[LRScheduler, ReduceLROnPlateauScheduler]:
    """
    Factory function to create a learning rate scheduler.
    
    Args:
        scheduler_type: Type of scheduler ('constant', 'exponential', 'cosine', 'warmup', 'plateau')
        initial_lr: Initial learning rate
        **kwargs: Additional arguments specific to the scheduler
    
    Returns:
        Learning rate scheduler instance
    """
    if scheduler_type == 'constant':
        return ConstantScheduler(initial_lr)
    
    elif scheduler_type == 'exponential':
        return ExponentialDecayScheduler(
            initial_lr=initial_lr,
            decay_rate=kwargs.get('decay_rate', 0.9),
            decay_steps=kwargs.get('decay_steps', 1000),
            staircase=kwargs.get('staircase', False)
        )
    
    elif scheduler_type == 'cosine':
        return CosineAnnealingScheduler(
            initial_lr=initial_lr,
            min_lr=kwargs.get('min_lr', 0.0),
            cycle_steps=kwargs.get('cycle_steps', 1000),
            warmup_steps=kwargs.get('warmup_steps', 0)
        )
    
    elif scheduler_type == 'warmup':
        return WarmupCooldownScheduler(
            initial_lr=initial_lr,
            warmup_steps=kwargs.get('warmup_steps', 100),
            constant_steps=kwargs.get('constant_steps', 1000),
            cooldown_steps=kwargs.get('cooldown_steps', 100),
            final_lr=kwargs.get('final_lr', 0.0)
        )
    
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateauScheduler(
            initial_lr=initial_lr,
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 0.0),
            verbose=kwargs.get('verbose', False)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def wrap_optimizer_with_scheduler(
    optimizer_fn: Callable,
    scheduler: Union[LRScheduler, str],
    initial_lr: Optional[float] = None,
    **kwargs
) -> optax.GradientTransformation:
    """
    Create an optimizer with a learning rate schedule.
    
    Args:
        optimizer_fn: Function that creates an optimizer given a learning rate
                      (e.g., lambda lr: optax.adam(lr))
        scheduler: Either a LRScheduler instance or a string specifying the scheduler type
        initial_lr: Initial learning rate (required if scheduler is a string)
        **kwargs: Additional arguments for the scheduler (if creating from string)
    
    Returns:
        Optimizer with learning rate schedule
    """
    if isinstance(scheduler, str):
        if initial_lr is None:
            raise ValueError("initial_lr must be provided when scheduler is specified as a string")
        scheduler = create_scheduler(scheduler, initial_lr, **kwargs)
    
    if isinstance(scheduler, LRScheduler):
        return scheduler.wrap_optimizer(optimizer_fn)
    else:
        # For non-standard schedulers like ReduceLROnPlateauScheduler
        if initial_lr is None:
            initial_lr = getattr(scheduler, 'initial_lr', 1e-3)
        return optimizer_fn(initial_lr) 