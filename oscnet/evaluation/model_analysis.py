"""
Model parameter analysis and summary utilities
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict


def count_model_parameters(model: eqx.Module) -> Dict[str, int]:
    """
    Count model parameters similar to Keras model.summary()
    Returns total, trainable, and non-trainable parameter counts
    """
    # Get all parameters using eqx.filter
    params, static = eqx.partition(model, eqx.is_array)
    
    # Count parameters by flattening the PyTree and summing array sizes
    param_leaves = jax.tree_util.tree_leaves(params)
    
    total_params = 0
    for leaf in param_leaves:
        if isinstance(leaf, jnp.ndarray):
            total_params += leaf.size
    
    # In JAX/Equinox, all parameters are typically trainable
    trainable_params = total_params
    non_trainable_params = 0
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params
    }


def print_model_summary(model: eqx.Module, model_name: str = "Model"):
    """
    Print a detailed model summary similar to Keras model.summary()
    """
    param_counts = count_model_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    
    # Print architecture overview
    print(f"Architecture: {type(model).__name__}")
    
    # Analyze each major component
    if hasattr(model, 'encoder'):
        encoder_params = count_model_parameters(model.encoder)
        print(f"Encoder parameters: {encoder_params['total_params']:,}")
        
        if hasattr(model.encoder, 'rnn'):
            rnn_params = count_model_parameters(model.encoder.rnn)
            print(f"  ├── RNN: {rnn_params['total_params']:,}")
        if hasattr(model.encoder, 'to_latent'):
            linear_params = count_model_parameters(model.encoder.to_latent)
            print(f"  └── Linear projection: {linear_params['total_params']:,}")
    
    if hasattr(model, 'decoder'):
        decoder_params = count_model_parameters(model.decoder)
        print(f"Decoder parameters: {decoder_params['total_params']:,}")
        
        if hasattr(model.decoder, 'from_latent'):
            linear_params = count_model_parameters(model.decoder.from_latent)
            print(f"  ├── Linear projection: {linear_params['total_params']:,}")
        if hasattr(model.decoder, 'rnn'):
            rnn_params = count_model_parameters(model.decoder.rnn)
            print(f"  └── RNN: {rnn_params['total_params']:,}")
    
    print(f"\n{'-'*60}")
    print(f"Total params: {param_counts['total_params']:,}")
    print(f"Trainable params: {param_counts['trainable_params']:,}")
    print(f"Non-trainable params: {param_counts['non_trainable_params']:,}")
    print(f"{'-'*60}")
    
    # Calculate memory usage estimate
    memory_mb = (param_counts['total_params'] * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"Estimated memory usage: {memory_mb:.2f} MB")
    print(f"{'='*60}\n")
    
    return param_counts


def compare_model_sizes(models: Dict[str, eqx.Module]):
    """
    Compare parameter counts across multiple models
    """
    print(f"\n{'='*80}")
    print(f"Model Size Comparison")
    print(f"{'='*80}")
    
    model_stats = {}
    for name, model in models.items():
        params = count_model_parameters(model)
        model_stats[name] = params
        
    # Print comparison table
    print(f"{'Model':<25} {'Total Params':<15} {'Trainable':<15} {'Memory (MB)':<12}")
    print(f"{'-'*80}")
    
    for name, stats in model_stats.items():
        memory_mb = (stats['total_params'] * 4) / (1024 * 1024)
        print(f"{name:<25} {stats['total_params']:<15,} {stats['trainable_params']:<15,} {memory_mb:<12.2f}")
    
    print(f"{'='*80}\n")
    return model_stats


def analyze_model_efficiency(model: eqx.Module, input_dim: int = 784, output_dim: int = 784):
    """
    Analyze model parameter efficiency and capacity
    """
    param_counts = count_model_parameters(model)
    total_params = param_counts['total_params']
    
    print(f"\n{'='*60}")
    print(f"📊 PARAMETER EFFICIENCY ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate parameter density
    theoretical_minimum = input_dim + output_dim  # Linear mapping
    efficiency_ratio = total_params / theoretical_minimum
    
    print(f"   📏 Input dimension: {input_dim}")
    print(f"   📐 Output dimension: {output_dim}")
    print(f"   🔢 Theoretical minimum (linear): {theoretical_minimum:,} params")
    print(f"   📈 Actual parameters: {total_params:,} params")
    print(f"   ⚡ Parameter efficiency ratio: {efficiency_ratio:.2f}x")
    
    if efficiency_ratio < 5:
        efficiency_note = "EXCELLENT - Very parameter efficient"
    elif efficiency_ratio < 20:
        efficiency_note = "GOOD - Reasonable parameter usage"
    elif efficiency_ratio < 100:
        efficiency_note = "MODERATE - Higher parameter count but acceptable"
    else:
        efficiency_note = "HIGH - Consider architecture optimization"
    
    print(f"   🎯 Efficiency assessment: {efficiency_note}")
    
    # Model size category
    if total_params < 100_000:
        size_category = "COMPACT"
        deployment_note = "Perfect for edge deployment and mobile devices"
    elif total_params < 1_000_000:
        size_category = "MEDIUM"
        deployment_note = "Good balance of capacity and efficiency"
    else:
        size_category = "LARGE"
        deployment_note = "High capacity model, may require more resources"
    
    print(f"   📏 Model size category: {size_category}")
    print(f"   🚀 Deployment suitability: {deployment_note}")
    print(f"{'='*60}\n")
    
    return {
        'efficiency_ratio': efficiency_ratio,
        'size_category': size_category,
        'total_params': total_params,
        'memory_mb': (total_params * 4) / (1024 * 1024)
    } 