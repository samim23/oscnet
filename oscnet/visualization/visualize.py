"""
Visualization tools for oscillator networks.

This module provides functions for visualizing oscillator network states,
synchronization measures, and network connectivity.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

def plot_reconstruction(
    originals,
    reconstructions,
    title: str = "Reconstructions",
    num_images: int = 10,
    figsize: tuple = (10, 3)
):
    """
    Plot original and reconstructed MNIST images side by side.
    Args:
        originals: Array of original images, shape (N, 784) or (N, 28, 28)
        reconstructions: Array of reconstructed images, shape (N, 784) or (N, 28, 28)
        title: Plot title
        num_images: Number of images to display
        figsize: Figure size
    """
    originals = np.array(originals)
    reconstructions = np.array(reconstructions)
    if originals.shape[-1] == 784:
        originals = originals.reshape(-1, 28, 28)
    if reconstructions.shape[-1] == 784:
        reconstructions = reconstructions.reshape(-1, 28, 28)
    num_images = min(num_images, originals.shape[0], reconstructions.shape[0])
    fig, axes = plt.subplots(2, num_images, figsize=figsize)
    for i in range(num_images):
        axes[0, i].imshow(originals[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructions[i], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=12)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_network_connectivity(
    adjacency_matrix: jnp.ndarray,
    node_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    node_colors: Optional[Union[List, jnp.ndarray]] = None,
    edge_weights: bool = True,
    layout: str = 'spring',
    title: str = "Network Connectivity",
    ax: Optional[plt.Axes] = None,
):
    """
    Plot the network connectivity using networkx.
    
    Args:
        adjacency_matrix: Adjacency matrix (n_oscillators × n_oscillators)
        node_positions: Optional dictionary mapping node indices to (x, y) positions
        node_colors: Optional list or array of node colors
        edge_weights: Whether to use edge weights for line thickness
        layout: Layout algorithm ('spring', 'circular', 'spectral', 'random', 'shell')
        title: Plot title
        ax: Optional matplotlib axis for plotting
        
    Returns:
        Created matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
        
    # Convert JAX array to numpy for compatibility with networkx
    adj_matrix_np = np.array(adjacency_matrix)
    
    # Create directed graph from adjacency matrix
    G = nx.DiGraph(adj_matrix_np)
    
    # Determine node positions
    if node_positions is None:
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        else:
            raise ValueError(f"Unknown layout: {layout}")
    else:
        pos = node_positions
    
    # Set default node colors if not provided
    if node_colors is None:
        node_colors = 'skyblue'
    
    # Calculate edge widths based on weights if requested
    if edge_weights:
        edge_weights_list = [G[u][v]['weight'] * 2.0 for u, v in G.edges()]
    else:
        edge_weights_list = [1.0 for _ in G.edges()]

    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights_list, alpha=0.5, 
                           edge_color='gray', arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    
    return ax

def plot_oscillator_states(
    times: jnp.ndarray,
    states: jnp.ndarray,
    oscillator_indices: Optional[List[int]] = None,
    dimension_indices: Optional[List[int]] = None,
    title: str = "Oscillator States",
    ax: Optional[plt.Axes] = None,
):
    """
    Plot the time evolution of oscillator states.
    
    Args:
        times: Time points (n_times,)
        states: State trajectory, shape (n_times, n_oscillators, state_dims)
                or (n_times, n_oscillators) for scalar states
        oscillator_indices: Indices of oscillators to plot. If None, plot all.
        dimension_indices: For vector states, which dimensions to plot. If None, plot all.
        title: Plot title
        ax: Optional matplotlib axis for plotting
        
    Returns:
        Created matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    is_scalar_state = states.ndim == 2
    n_times, n_oscillators = states.shape[0], states.shape[1]
    
    # Set default oscillator indices if none specified
    if oscillator_indices is None:
        if n_oscillators <= 10:
            oscillator_indices = list(range(n_oscillators))
        else:
            # Plot first 10 oscillators if more than 10
            oscillator_indices = list(range(10))
    
    # For scalar states, just plot the values
    if is_scalar_state:
        for i in oscillator_indices:
            ax.plot(times, states[:, i], label=f"Oscillator {i}")
    else:
        # For vector states, plot specified dimensions
        state_dims = states.shape[2]
        
        if dimension_indices is None:
            if state_dims <= 3:
                dimension_indices = list(range(state_dims))
            else:
                # Plot first 3 dimensions if more than 3
                dimension_indices = list(range(3))
        
        for i in oscillator_indices:
            for d in dimension_indices:
                ax.plot(times, states[:, i, d], 
                        label=f"Oscillator {i}, Dim {d}")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_title(title)
    ax.legend(loc='best')
    
    return ax

def plot_synchronization_measures(
    times: jnp.ndarray,
    sync_measures: Dict[str, jnp.ndarray],
    title: str = "Synchronization Measures",
    ax: Optional[plt.Axes] = None,
):
    """
    Plot synchronization measures over time.
    
    Args:
        times: Time points (n_times,)
        sync_measures: Dictionary of synchronization measures as returned by
                      OscillatorNetwork.compute_synchronization()
        title: Plot title
        ax: Optional matplotlib axis for plotting
        
    Returns:
        Created matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    for name, measure in sync_measures.items():
        ax.plot(times, measure, label=name)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Measure Value")
    ax.set_title(title)
    ax.legend(loc='best')
    
    return ax

def plot_phase_distribution(
    phases: jnp.ndarray,
    title: str = "Phase Distribution",
    time_index: int = -1,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot the distribution of oscillator phases on a unit circle.
    
    Args:
        phases: Phase values, shape (n_times, n_oscillators) or
               (n_times, n_oscillators, state_dims) where the first dimension
               is assumed to be the phase
        title: Plot title
        time_index: Which time index to plot (-1 for final state)
        ax: Optional matplotlib axis for plotting
        
    Returns:
        Created matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    # Extract phases for the specified time
    if phases.ndim == 2:
        # Direct phases
        phase_values = phases[time_index]
    else:
        # Extract phase from first dimension
        phase_values = phases[time_index, :, 0]
    
    # Ensure phases are within [0, 2π]
    phase_values = np.mod(np.array(phase_values), 2 * np.pi)
    
    # Plot phases on unit circle
    ax.scatter(phase_values, np.ones_like(phase_values), alpha=0.7)
    
    # Add unit circle
    ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100), 'k--', alpha=0.3)
    
    # Calculate order parameter
    complex_phases = np.exp(1j * phase_values)
    order_param = np.abs(np.mean(complex_phases))
    mean_phase = np.angle(np.mean(complex_phases))
    
    # Add order parameter vector
    ax.arrow(mean_phase, 0, 0, order_param, alpha=0.8, width=0.05,
             head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax.set_rticks([0.5, 1.0])
    ax.set_title(f"{title}\nOrder Parameter: {order_param:.3f}")
    
    return ax

def create_network_animation(
    times: jnp.ndarray,
    states: jnp.ndarray,
    adjacency_matrix: jnp.ndarray,
    node_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    output_file: Optional[str] = None,
    fps: int = 10,
    dpi: int = 100,
    title: str = "Network Dynamics",
):
    """
    Create an animation of the network dynamics.
    
    Args:
        times: Time points (n_times,)
        states: State trajectory, shape (n_times, n_oscillators, state_dims)
                or (n_times, n_oscillators) for scalar states
        adjacency_matrix: Adjacency matrix (n_oscillators × n_oscillators)
        node_positions: Optional dictionary mapping node indices to (x, y) positions
        output_file: Path to save animation (if None, animation is displayed)
        fps: Frames per second
        dpi: Resolution
        title: Animation title
        
    Returns:
        matplotlib animation
    """
    n_oscillators = adjacency_matrix.shape[0]
    
    # Setup figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create directed graph
    G = nx.DiGraph(np.array(adjacency_matrix))
    
    # Determine node positions
    if node_positions is None:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = node_positions
    
    # Determine node coloring based on states
    is_scalar_state = states.ndim == 2
    
    if is_scalar_state:
        # For scalar states, use the phase directly for coloring
        vmin, vmax = 0, 2 * np.pi
        cmap = plt.cm.hsv
    else:
        # For vector states, use the first component
        vmin = np.min(states[:, :, 0])
        vmax = np.max(states[:, :, 0])
        cmap = plt.cm.viridis
    
    # Determine edge widths
    edge_weights = [G[u][v]['weight'] * 2.0 for u, v in G.edges()]
    
    # Setup colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Phase" if is_scalar_state else "State[0]")
    
    # Animation function
    def animate(frame):
        ax.clear()
        
        # Get node colors for current frame
        if is_scalar_state:
            node_colors = np.mod(np.array(states[frame]), 2 * np.pi)
        else:
            node_colors = np.array(states[frame, :, 0])
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap,
                              vmin=vmin, vmax=vmax, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5,
                              edge_color='gray', arrows=True, ax=ax)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        # Set title with current time
        ax.set_title(f"{title} - Time: {times[frame]:.2f}")
        ax.axis('off')
        
        return []
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=len(times),
                                  interval=1000/fps, blit=True)
    
    # Save animation if requested
    if output_file is not None:
        ani.save(output_file, fps=fps, dpi=dpi)
        plt.close(fig)
    
    return ani

def plot_phase_sync(
    times: jnp.ndarray,
    states: jnp.ndarray,
    groups: Optional[Dict[str, List[int]]] = None,
    title: str = "Phase Synchronization",
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None,
):
    """
    Plot phase synchronization measures for groups of oscillators over time.
    
    Args:
        times: Time points (n_times,)
        states: State trajectory, shape (n_times, n_oscillators)
        groups: Dict mapping group names to lists of oscillator indices.
               If None, will compute synchronization for all oscillators together.
        title: Plot title
        figsize: Figure size as (width, height) in inches
        ax: Optional matplotlib axis for plotting
        
    Returns:
        Created matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    # Ensure states is a JAX array
    states = jnp.asarray(states)
    
    # If no groups provided, use all oscillators as a single group
    if groups is None:
        n_oscillators = states.shape[1]
        groups = {"All Oscillators": list(range(n_oscillators))}
    
    # Calculate order parameter R for each group at each time
    for group_name, oscillator_indices in groups.items():
        # Extract phases for this group
        group_phases = states[:, oscillator_indices]
        
        # Calculate Kuramoto order parameter R = |∑ e^(i*θ_j)|/N
        complex_phases = jnp.exp(1j * group_phases)
        sync_order = jnp.abs(jnp.sum(complex_phases, axis=1)) / len(oscillator_indices)
        
        # Plot this group's synchronization
        ax.plot(times, sync_order, label=f"{group_name} (R)")
    
    # Add reference line for perfect synchronization
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label="Perfect Sync (R=1)")
    
    # Formatting
    ax.set_xlabel("Time")
    ax.set_ylabel("Order Parameter (R)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_phase_space(
    states: jnp.ndarray,
    dimensions: Tuple[int, int] = (0, 1),
    oscillator_indices: Optional[List[int]] = None,
    trajectory: bool = True,
    final_state: bool = True,
    title: str = "Phase Space",
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[plt.Axes] = None,
):
    """
    Plot oscillator trajectories in phase space (2D projection).
    
    Args:
        states: State trajectory, shape (n_times, n_oscillators, state_dims)
        dimensions: Which two dimensions to plot (x, y)
        oscillator_indices: Which oscillators to plot. If None, plots all oscillators.
        trajectory: Whether to plot the full trajectory
        final_state: Whether to highlight the final state
        title: Plot title
        figsize: Figure size in inches
        ax: Optional matplotlib axis for plotting
        
    Returns:
        Created matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        
    # Ensure states is a JAX array
    states = jnp.asarray(states)
    
    # Check dimensionality
    if states.ndim < 3:
        raise ValueError("States must have at least 3 dimensions (time, oscillators, state_dims)")
    
    n_times, n_oscillators, state_dims = states.shape
    
    # Set default oscillator indices if none specified
    if oscillator_indices is None:
        if n_oscillators <= 10:
            oscillator_indices = list(range(n_oscillators))
        else:
            # Plot first 10 oscillators if more than 10
            oscillator_indices = list(range(10))
    
    # Check if dimensions are valid
    dim_x, dim_y = dimensions
    if dim_x >= state_dims or dim_y >= state_dims:
        raise ValueError(f"Dimensions {dimensions} out of bounds for state with {state_dims} dimensions")
    
    # Plot trajectories for each oscillator
    for i in oscillator_indices:
        # Get x and y coordinates for this oscillator
        x = states[:, i, dim_x]
        y = states[:, i, dim_y]
        
        # Plot full trajectory if requested
        if trajectory:
            ax.plot(x, y, '-', alpha=0.6, label=f"Oscillator {i}")
        
        # Plot final state if requested
        if final_state:
            ax.plot(x[-1], y[-1], 'o', markersize=8)
    
    # Formatting
    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")
    ax.set_title(title)
    if len(oscillator_indices) <= 10:  # Only show legend if not too cluttered
        ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax 
