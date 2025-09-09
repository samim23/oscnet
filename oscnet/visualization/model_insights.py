"""
Visualization tools for model evaluation and analysis.

This module provides functions for visualizing model results such as reconstructions,
latent spaces, training metrics, and comparative evaluations.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def visualize_reconstructions(
    original: jnp.ndarray,
    reconstructed: jnp.ndarray,
    n_samples: int = 10,
    title: str = "Original vs Reconstructed",
    figsize: Tuple[int, int] = None,
    cmap: str = 'gray'
):
    """
    Visualize original images and their reconstructions side by side.
    
    Args:
        original: Original images array with shape (n_images, height, width)
        reconstructed: Reconstructed images array with shape (n_images, height, width)
        n_samples: Number of samples to display
        title: Figure title
        figsize: Figure size (calculated automatically if None)
        cmap: Colormap for images
        
    Returns:
        Matplotlib figure
    """
    n_samples = min(n_samples, original.shape[0])
    
    # Calculate appropriate figure size if not provided
    if figsize is None:
        figsize = (n_samples * 2, 4)
    
    # Create figure
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    for i in range(n_samples):
        # Display original
        axes[0, i].imshow(original[i], cmap=cmap)
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Display reconstruction
        axes[1, i].imshow(reconstructed[i], cmap=cmap)
        axes[1, i].set_title(f"Recon {i+1}")
        axes[1, i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def visualize_latent_space(
    latent: jnp.ndarray,
    labels: Optional[jnp.ndarray] = None,
    method: str = 'auto',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'tab10',
    alpha: float = 0.8,
    marker_size: int = 30,
    title: str = "Latent Space Visualization"
):
    """
    Visualize latent space representations.
    
    Args:
        latent: Latent vectors with shape (n_samples, latent_dim)
        labels: Optional class labels for coloring points
        method: Visualization method: 'auto', '1d', '2d', or '3d'
        figsize: Figure size
        cmap: Colormap for labels
        alpha: Point transparency
        marker_size: Point size
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    latent_dim = latent.shape[1]
    
    # Determine method automatically if set to 'auto'
    if method == 'auto':
        if latent_dim == 1:
            method = '1d'
        elif latent_dim == 2:
            method = '2d'
        elif latent_dim >= 3:
            method = '3d'
    
    fig = plt.figure(figsize=figsize)
    
    # 1D visualization
    if method == '1d':
        ax = fig.add_subplot(111)
        if labels is not None:
            scatter = ax.scatter(
                latent[:, 0], jnp.zeros_like(latent[:, 0]),
                c=labels, cmap=cmap, s=marker_size, alpha=alpha
            )
            plt.colorbar(scatter, label='Class')
        else:
            ax.scatter(
                latent[:, 0], jnp.zeros_like(latent[:, 0]),
                s=marker_size, alpha=alpha
            )
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_yticks([])
        ax.set_title(title)
    
    # 2D visualization    
    elif method == '2d':
        ax = fig.add_subplot(111)
        if labels is not None:
            scatter = ax.scatter(
                latent[:, 0], latent[:, 1],
                c=labels, cmap=cmap, s=marker_size, alpha=alpha
            )
            plt.colorbar(scatter, label='Class')
        else:
            ax.scatter(
                latent[:, 0], latent[:, 1],
                s=marker_size, alpha=alpha
            )
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
    # 3D visualization
    elif method == '3d':
        ax = fig.add_subplot(111, projection='3d')
        if labels is not None:
            scatter = ax.scatter(
                latent[:, 0], latent[:, 1], latent[:, 2],
                c=labels, cmap=cmap, s=marker_size, alpha=alpha
            )
            plt.colorbar(scatter, label='Class')
        else:
            ax.scatter(
                latent[:, 0], latent[:, 1], latent[:, 2],
                s=marker_size, alpha=alpha
            )
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_zlabel('Latent Dimension 3')
        ax.set_title(title)
        
    plt.tight_layout()
    return fig

def plot_training_curves(
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Training Metrics",
    xlabel: str = "Epoch",
    grid: bool = True
):
    """
    Plot training metrics over epochs.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        figsize: Figure size
        title: Figure title
        xlabel: Label for x-axis
        grid: Whether to show grid
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, values in metrics.items():
        ax.plot(values, label=name)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.set_title(title)
    
    if grid:
        ax.grid(True, alpha=0.3)
        
    ax.legend()
    plt.tight_layout()
    
    return fig

def compare_model_metrics(
    model_results: Dict[str, Dict[str, List[float]]],
    metric_name: str = "loss",
    figsize: Tuple[int, int] = (10, 6),
    title: str = None,
    xlabel: str = "Epoch",
    ylabel: str = None,
    grid: bool = True
):
    """
    Compare metrics across multiple models.
    
    Args:
        model_results: Dictionary mapping model names to metrics dictionaries
        metric_name: Name of metric to plot
        figsize: Figure size
        title: Figure title (defaults to f"{metric_name} Comparison")
        xlabel: Label for x-axis
        ylabel: Label for y-axis (defaults to metric_name)
        grid: Whether to show grid
        
    Returns:
        Matplotlib figure
    """
    if title is None:
        title = f"{metric_name} Comparison"
        
    if ylabel is None:
        ylabel = metric_name.capitalize()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, metrics in model_results.items():
        if metric_name in metrics:
            ax.plot(metrics[metric_name], label=model_name)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if grid:
        ax.grid(True, alpha=0.3)
        
    ax.legend()
    plt.tight_layout()
    
    return fig

def visualize_oscillator_dynamics(
    oscillator_states: Dict[str, jnp.ndarray],
    title: str = "Oscillator Dynamics",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Visualize the oscillator states and trajectories.
    
    Args:
        oscillator_states: Dictionary with trajectory data
        title: Figure title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    if "trajectory_x" in oscillator_states and "trajectory_v" in oscillator_states:
        traj_x = oscillator_states["trajectory_x"]
        traj_v = oscillator_states["trajectory_v"]
    else:
        # If no trajectory data, return early
        print("No trajectory data found in oscillator_states")
        return None
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Get dimensions of trajectory data
    if len(traj_x.shape) == 3:  # shape: (time_steps, batch_size, n_oscillators)
        # Extract first batch item
        traj_x = traj_x[:, 0, :]
        traj_v = traj_v[:, 0, :]
    
    # Number of oscillators to visualize (limited to first 3)
    n_oscillators = min(3, traj_x.shape[1])
    
    # Create appropriate visualization based on dimension
    if n_oscillators == 1:
        # 2D phase space of single oscillator (x vs v)
        ax = fig.add_subplot(111)
        ax.plot(traj_x[:, 0], traj_v[:, 0], 'o-', alpha=0.7, markersize=4)
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Velocity (v)")
        ax.set_title(f"{title} - Oscillator 1")
        ax.grid(True, alpha=0.3)
    
    elif n_oscillators == 2:
        # 2D plot of two oscillators' positions
        ax = fig.add_subplot(111)
        ax.plot(traj_x[:, 0], traj_x[:, 1], 'o-', alpha=0.7, markersize=4)
        ax.set_xlabel("Oscillator 1 position (x)")
        ax.set_ylabel("Oscillator 2 position (x)")
        ax.set_title(f"{title} - Oscillator Positions")
        ax.grid(True, alpha=0.3)
    
    else:  # n_oscillators >= 3
        # 3D visualization of three oscillators
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj_x[:, 0], traj_x[:, 1], traj_x[:, 2], 'o-', alpha=0.7, markersize=4)
        ax.set_xlabel("Oscillator 1 position (x)")
        ax.set_ylabel("Oscillator 2 position (x)")
        ax.set_zlabel("Oscillator 3 position (x)")
        ax.set_title(f"{title} - Oscillator Positions")
    
    plt.tight_layout()
    return fig

def visualize_training_metrics(
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Training Metrics",
    apply_smoothing: bool = True,
    smoothing_window: int = None,
    include_lr: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create an enhanced visualization of training metrics with multiple plots.
    
    Args:
        metrics: Dictionary containing metrics:
                - Required: 'train_loss' list of training loss values
                - Optional: 'val_loss' list of validation loss values
                - Optional: 'learning_rate' list of learning rate values
        figsize: Figure size
        title: Main figure title
        apply_smoothing: Whether to apply smoothing to training loss
        smoothing_window: Size of smoothing window (auto-calculated if None)
        include_lr: Whether to include learning rate subplot
        save_path: Optional path to save the figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    # Determine number of subplots based on available metrics
    has_lr = 'learning_rate' in metrics and include_lr
    n_plots = 2 if has_lr else 1
    
    # Create figure
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, 
                            gridspec_kw={'height_ratios': [3, 1]} if has_lr else None)
    
    # If only one subplot, convert axes to list for consistent indexing
    if n_plots == 1:
        axes = [axes]
    
    # Plot training and validation loss
    ax = axes[0]
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Training loss
    ax.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    
    # Add smoothed training loss
    if apply_smoothing and len(epochs) > 2:
        if smoothing_window is None:
            # Auto-calculate a reasonable window size
            smoothing_window = max(2, len(epochs) // 5)
        
        # Apply smoothing using convolution
        if len(metrics['train_loss']) > smoothing_window:
            # Create kernel for smoothing
            kernel = np.ones(smoothing_window) / smoothing_window
            # Apply convolution for smoothing
            train_loss_smooth = np.convolve(metrics['train_loss'], kernel, mode='valid')
            # Plot smoothed curve
            epochs_smooth = epochs[smoothing_window-1:]
            ax.plot(epochs_smooth, train_loss_smooth, 'g--', linewidth=1.5, 
                   label=f'Smoothed Train Loss (window={smoothing_window})')
    
    # Validation loss if available
    if 'val_loss' in metrics:
        ax.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    
    # Add labels and styling
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add learning rate subplot if available
    if has_lr:
        ax = axes[1]
        ax.plot(epochs, metrics['learning_rate'], 'g-', label='Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
    else:
        # Add xlabel to main plot if there's no LR subplot
        axes[0].set_xlabel('Epoch')
    
    # Add main title
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Adjust for main title
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        
    # Show if requested
    if show:
        plt.show()
        
    return fig

def visualize_reconstruction_progression(
    model_checkpoints: List[Any],
    base_model: Any,
    test_data: jnp.ndarray,
    checkpoint_epochs: List[int] = None,
    n_samples: int = 5,
    sequence_length: int = 20,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = 'gray',
    save_path: Optional[str] = None,
    show: bool = True,
    image_encoder_fn: Optional[Callable] = None
) -> plt.Figure:
    """
    Visualize how reconstructions evolve across training epochs.
    
    Args:
        model_checkpoints: List of serialized model checkpoints or loaded model objects
        base_model: Base model structure for deserializing checkpoints (if needed)
        test_data: Test data to use for reconstructions
        checkpoint_epochs: Corresponding epoch numbers (if None, uses 1-based indices)
        n_samples: Number of test samples to visualize
        sequence_length: Sequence length for processing
        figsize: Figure size (auto-calculated if None)
        cmap: Colormap for visualizations
        save_path: Optional path to save the figure
        show: Whether to display the figure
        image_encoder_fn: Custom function to encode images (if None, uses simple normalization)
        
    Returns:
        Matplotlib figure
    """
    # Use checkpoint indices if epochs not provided
    if checkpoint_epochs is None:
        checkpoint_epochs = list(range(1, len(model_checkpoints) + 1))
    
    # Select samples
    n_samples = min(n_samples, test_data.shape[0])
    x_test_subset = test_data[:n_samples]
    
    # Calculate figure size if not provided
    n_cols = len(model_checkpoints) + 1  # +1 for originals
    if figsize is None:
        figsize = (n_cols * 2, n_samples * 2)
    
    # Create figure
    fig, axes = plt.subplots(n_samples, n_cols, figsize=figsize)
    
    # If only one sample, ensure axes is 2D
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Plot original images in the first column
    for i in range(n_samples):
        axes[i, 0].imshow(x_test_subset[i], cmap=cmap)
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].axis('off')
    
    # Preprocessing function
    def preprocess_inputs(imgs):
        # Default preprocessing if no custom encoder provided
        if image_encoder_fn is None:
            # Flatten and normalize
            flattened = imgs.reshape(imgs.shape[0], -1)
            if flattened.max() > 1.0:
                flattened = flattened / 255.0
            
            # Add sequence dimension
            return flattened[:, jnp.newaxis, :].repeat(sequence_length, axis=1)
        else:
            return image_encoder_fn(imgs, sequence_length)
    
    # Process each checkpoint
    for col, (epoch_idx, checkpoint) in enumerate(zip(checkpoint_epochs, model_checkpoints), 1):
        # Check if checkpoint is a path or a model object
        if isinstance(checkpoint, (str, Path)) and base_model is not None:
            # Deserialize the model
            try:
                import equinox as eqx
                epoch_model = eqx.tree_deserialise_leaves(checkpoint, base_model)
            except Exception as e:
                print(f"Error deserializing checkpoint {epoch_idx}: {e}")
                # Set empty images for this column
                for i in range(n_samples):
                    axes[i, col].text(0.5, 0.5, "Error", ha='center', va='center')
                    axes[i, col].axis('off')
                continue
        else:
            # Checkpoint is already a model object
            epoch_model = checkpoint
        
        # Preprocess inputs
        x_encoded = preprocess_inputs(x_test_subset)
        
        # Forward pass through model
        try:
            reconstructions = epoch_model(x_encoded)
        except Exception as e:
            print(f"Error running model for epoch {epoch_idx}: {e}")
            # Set error messages for this column
            for i in range(n_samples):
                axes[i, col].text(0.5, 0.5, "Error", ha='center', va='center')
                axes[i, col].axis('off')
            continue
        
        # Reshape outputs to images
        img_size = int(np.sqrt(reconstructions.shape[2]))
        recon_images = reconstructions[:, -1, :].reshape(n_samples, img_size, img_size)
        
        # Plot reconstructions
        for i in range(n_samples):
            axes[i, col].imshow(recon_images[i], cmap=cmap)
            if i == 0:
                axes[i, col].set_title(f"Epoch {epoch_idx}")
            axes[i, col].axis('off')
    
    # Layout adjustments
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        
    # Show if requested
    if show:
        plt.show()
        
    return fig 