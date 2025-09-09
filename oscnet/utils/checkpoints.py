"""
Model checkpointing and serialization utilities.

This module provides utilities for saving and loading Equinox models with
hyperparameters, training metrics, and optimizer states.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, Optional
import jax
import jax.numpy as jnp
import equinox as eqx

# Get logger for this module
logger = logging.getLogger(__name__)


def save_equinox_checkpoint(
    model: eqx.Module,
    opt_state: Any,
    epoch: int,
    metrics: Dict[str, Any],
    output_dir: Path,
    hyperparams: Dict[str, Any],
    is_best: bool = False
) -> str:
    """
    Save checkpoint using proper Equinox serialization with hyperparameters.
    
    Args:
        model: Equinox model to save
        opt_state: Optimizer state
        epoch: Current epoch number
        metrics: Training metrics dictionary
        output_dir: Directory to save checkpoint
        hyperparams: Model hyperparameters
        is_best: Whether this is the best model so far
        
    Returns:
        str: Path to saved checkpoint file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_best:
        checkpoint_path = output_dir / "best_model.eqx"
        metadata_path = output_dir / "best_model_metadata.json"
    else:
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.eqx"
        metadata_path = output_dir / f"checkpoint_epoch_{epoch:03d}_metadata.json"
    
    # Save model using Equinox serialization
    def save_model_with_hyperparams(filename, hyperparams, model):
        # Convert hyperparams to JSON-serializable format
        serializable_hyperparams = {}
        for key, value in hyperparams.items():
            if hasattr(value, 'tolist'):  # JAX/numpy arrays
                serializable_hyperparams[key] = value.tolist()
            elif isinstance(value, dict):
                # Handle nested dictionaries (like oscillator_params)
                serializable_dict = {}
                for k, v in value.items():
                    if hasattr(v, 'tolist'):
                        serializable_dict[k] = v.tolist()
                    elif hasattr(v, 'item'):  # scalar arrays
                        serializable_dict[k] = v.item()
                    else:
                        serializable_dict[k] = v
                serializable_hyperparams[key] = serializable_dict
            elif hasattr(value, 'item'):  # scalar arrays
                serializable_hyperparams[key] = value.item()
            else:
                serializable_hyperparams[key] = value
        
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(serializable_hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)
    
    save_model_with_hyperparams(checkpoint_path, hyperparams, model)
    
    # Save additional metadata (optimizer state, metrics, etc.)
    metadata = {
        "epoch": epoch,
        "metrics": metrics,
        "optimizer_state": {
            # Convert optimizer state to serializable format
            "step": int(opt_state[0].count) if hasattr(opt_state[0], 'count') else 0,
            # Note: Full optimizer state reconstruction would need more work
            # For now, we'll just save the step count
        }
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    
    return str(checkpoint_path)


def load_equinox_checkpoint(
    checkpoint_path: Path,
    make_model_fn: Callable,
    make_optimizer_fn: Callable
) -> Tuple[eqx.Module, Any, Dict[str, Any], Dict[str, Any]]:
    """
    Load checkpoint using proper Equinox deserialization.
    
    Args:
        checkpoint_path: Path to checkpoint file
        make_model_fn: Function to create model instance
        make_optimizer_fn: Function to create optimizer instance
        
    Returns:
        Tuple of (model, opt_state, metadata, hyperparams)
    """
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix('').with_suffix('_metadata.json')
    
    # Load model with hyperparameters
    def load_model_with_hyperparams(filename):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = make_model_fn(**hyperparams)
            return eqx.tree_deserialise_leaves(f, model), hyperparams
    
    model, hyperparams = load_model_with_hyperparams(checkpoint_path)
    
    # Load metadata if available
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # Recreate optimizer state (simplified - would need full reconstruction for complete restoration)
    opt_state = make_optimizer_fn().init(eqx.filter(model, eqx.is_array))
    
    return model, opt_state, metadata, hyperparams


def save_training_metrics(metrics: Dict[str, Any], metrics_dir: Path, epoch: int) -> str:
    """
    Save training metrics to JSON file.
    
    Args:
        metrics: Dictionary of training metrics
        metrics_dir: Directory to save metrics
        epoch: Current epoch number
        
    Returns:
        str: Path to saved metrics file
    """
    metrics_path = metrics_dir / f"training_metrics_epoch_{epoch:03d}.json"
    
    # Convert JAX arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'tolist'):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'item'):
            serializable_metrics[key] = [v.item() for v in value]
        else:
            serializable_metrics[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    return str(metrics_path) 