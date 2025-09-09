"""
Phase space visualization tools for oscillatory neural networks.

This module provides specialized visualization tools for analyzing the dynamics
of oscillatory neural networks in phase space, including dimensionality
reduction techniques for high-dimensional systems.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import equinox as eqx
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from functools import partial
import diffrax

# For dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, Isomap
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PhaseSpaceVisualizer:
    """
    Visualize oscillator dynamics in phase space.
    
    This class provides various methods for visualizing the behavior
    of oscillatory systems in phase space, including 2D and 3D plots,
    animations, and dimensionality reduction for high-dimensional systems.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "viridis",
        n_dim_reduction_components: int = 2,
        dim_reduction_method: str = "pca"
    ):
        """
        Initialize the phase space visualizer.
        
        Args:
            figsize: Figure size
            cmap: Colormap for plots
            n_dim_reduction_components: Number of components for dimensionality reduction
            dim_reduction_method: Method for dimensionality reduction ('pca', 'tsne', 'isomap')
        """
        self.figsize = figsize
        self.cmap = cmap
        self.n_components = n_dim_reduction_components
        self.dim_reduction_method = dim_reduction_method
        
        # Check if sklearn is available for dimensionality reduction
        if not SKLEARN_AVAILABLE and dim_reduction_method != "pca":
            print("Warning: sklearn not available. Using simple PCA for dimensionality reduction.")
            self.dim_reduction_method = "pca"
    
    def visualize_2d_trajectory(
        self,
        trajectory: jnp.ndarray,
        axis_labels: List[str] = None,
        title: str = "Oscillator Phase Space Trajectory",
        color_by_time: bool = True,
        ax: Any = None,
        show_arrows: bool = True,
        arrow_frequency: int = 10,
        marker_size: float = 10,
        marker_alpha: float = 0.7,
        show: bool = True
    ) -> Any:
        """
        Visualize a 2D phase space trajectory.
        
        Args:
            trajectory: Array of states with shape (time_steps, 2)
            axis_labels: Labels for x and y axes
            title: Plot title
            color_by_time: Whether to color points by time
            ax: Matplotlib axis to plot on (creates new one if None)
            show_arrows: Whether to show direction arrows
            arrow_frequency: Frequency of arrows along trajectory
            marker_size: Size of markers
            marker_alpha: Alpha transparency of markers
            show: Whether to show the plot
            
        Returns:
            Matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract x and y coordinates
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        
        # Set default axis labels if not provided
        if axis_labels is None:
            axis_labels = ["x", "v"]
        
        # Plot trajectory
        if color_by_time:
            # Color by time using colormap
            points = ax.scatter(
                x, y, 
                c=np.arange(len(x)), 
                cmap=self.cmap,
                s=marker_size,
                alpha=marker_alpha
            )
            plt.colorbar(points, ax=ax, label="Time")
        else:
            # Single color with connecting lines
            ax.plot(x, y, 'o-', markersize=marker_size/2, alpha=marker_alpha)
        
        # Add direction arrows
        if show_arrows and len(x) > 1:
            arrow_indices = np.arange(0, len(x) - 1, arrow_frequency)
            for i in arrow_indices:
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                ax.arrow(
                    x[i], y[i], dx, dy,
                    head_width=0.02 * np.max([np.ptp(x), np.ptp(y)]),
                    head_length=0.03 * np.max([np.ptp(x), np.ptp(y)]),
                    fc='black', ec='black', alpha=0.7
                )
        
        # Add labels and title
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    def visualize_3d_trajectory(
        self,
        trajectory: jnp.ndarray,
        axis_labels: List[str] = None,
        title: str = "3D Oscillator Phase Space Trajectory",
        color_by_time: bool = True,
        ax: Any = None,
        show_arrows: bool = True,
        arrow_frequency: int = 20,
        marker_size: float = 8,
        marker_alpha: float = 0.7,
        view_angles: Tuple[float, float] = (30, 45),
        show: bool = True
    ) -> Any:
        """
        Visualize a 3D phase space trajectory.
        
        Args:
            trajectory: Array of states with shape (time_steps, 3)
            axis_labels: Labels for x, y, and z axes
            title: Plot title
            color_by_time: Whether to color points by time
            ax: Matplotlib 3D axis to plot on (creates new one if None)
            show_arrows: Whether to show direction arrows
            arrow_frequency: Frequency of arrows along trajectory
            marker_size: Size of markers
            marker_alpha: Alpha transparency of markers
            view_angles: (elevation, azimuth) viewing angles
            show: Whether to show the plot
            
        Returns:
            Matplotlib 3D axis object
        """
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        # Extract x, y, and z coordinates
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]
        
        # Set default axis labels if not provided
        if axis_labels is None:
            axis_labels = ["x", "y", "z"]
        
        # Plot trajectory
        if color_by_time:
            # Color by time using colormap
            points = ax.scatter(
                x, y, z,
                c=np.arange(len(x)),
                cmap=self.cmap,
                s=marker_size,
                alpha=marker_alpha
            )
            plt.colorbar(points, ax=ax, label="Time")
        else:
            # Single color with connecting lines
            ax.plot(x, y, z, 'o-', markersize=marker_size/2, alpha=marker_alpha)
        
        # Add direction arrows
        if show_arrows and len(x) > 1:
            arrow_indices = np.arange(0, len(x) - 1, arrow_frequency)
            for i in arrow_indices:
                ax.quiver(
                    x[i], y[i], z[i],
                    x[i+1] - x[i], y[i+1] - y[i], z[i+1] - z[i],
                    color='black', alpha=0.7,
                    arrow_length_ratio=0.3,
                    normalize=True
                )
        
        # Add labels and title
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        ax.set_title(title)
        
        # Set view angle
        if view_angles is not None:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    def animate_trajectory(
        self,
        trajectory: jnp.ndarray,
        axis_labels: List[str] = None,
        title: str = "Animated Phase Space Trajectory",
        interval: int = 50,
        tail_length: int = 20,
        blit: bool = True,
        save_path: Optional[str] = None,
        dimension: str = '2d'
    ) -> FuncAnimation:
        """
        Create an animation of a phase space trajectory.
        
        Args:
            trajectory: Array of states with shape (time_steps, state_dim)
            axis_labels: Labels for axes
            title: Animation title
            interval: Time between frames in milliseconds
            tail_length: Length of the trailing tail
            blit: Whether to use blitting for improved performance
            save_path: Path to save animation (if None, not saved)
            dimension: '2d' or '3d'
            
        Returns:
            Matplotlib animation object
        """
        if dimension == '3d' and trajectory.shape[1] < 3:
            raise ValueError("3D animation requires trajectory with at least 3 dimensions")
        
        # Set up the figure and axis
        if dimension == '2d':
            fig, ax = plt.subplots(figsize=self.figsize)
            x_data, y_data = trajectory[:, 0], trajectory[:, 1]
            
            # Set default axis labels if not provided
            if axis_labels is None:
                axis_labels = ["x", "v"]
            
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            
            # Set axis limits
            x_min, x_max = np.min(x_data), np.max(x_data)
            y_min, y_max = np.min(y_data), np.max(y_data)
            x_range = x_max - x_min
            y_range = y_max - y_min
            ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            # Initialize the line and point
            line, = ax.plot([], [], 'o-', alpha=0.7, markersize=4)
            point, = ax.plot([], [], 'o', color='red', markersize=8)
            
            # Animation update function
            def update(frame):
                # Determine the range of points to show (for the tail)
                start = max(0, frame - tail_length)
                x_slice = x_data[start:frame+1]
                y_slice = y_data[start:frame+1]
                
                # Update the line and point
                line.set_data(x_slice, y_slice)
                if frame < len(x_data):
                    point.set_data([x_data[frame]], [y_data[frame]])
                
                return line, point
        
        elif dimension == '3d':
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            x_data, y_data, z_data = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
            
            # Set default axis labels if not provided
            if axis_labels is None:
                axis_labels = ["x", "y", "z"]
            
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_zlabel(axis_labels[2])
            
            # Set axis limits
            x_min, x_max = np.min(x_data), np.max(x_data)
            y_min, y_max = np.min(y_data), np.max(y_data)
            z_min, z_max = np.min(z_data), np.max(z_data)
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)
            
            # Initialize the line and point
            line, = ax.plot([], [], [], 'o-', alpha=0.7, markersize=4)
            point, = ax.plot([], [], [], 'o', color='red', markersize=8)
            
            # Animation update function
            def update(frame):
                # Determine the range of points to show (for the tail)
                start = max(0, frame - tail_length)
                x_slice = x_data[start:frame+1]
                y_slice = y_data[start:frame+1]
                z_slice = z_data[start:frame+1]
                
                # Update the line and point
                line.set_data(x_slice, y_slice)
                line.set_3d_properties(z_slice)
                if frame < len(x_data):
                    point.set_data([x_data[frame]], [y_data[frame]])
                    point.set_3d_properties([z_data[frame]])
                
                return line, point
        
        # Add title
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Create the animation
        anim = FuncAnimation(
            fig, update, frames=len(trajectory),
            interval=interval, blit=blit
        )
        
        # Save if path provided
        if save_path is not None:
            anim.save(save_path, writer='pillow', fps=30)
        
        plt.tight_layout()
        plt.close()  # Close the figure to avoid displaying it
        
        return anim
    
    def reduce_dimensions(
        self,
        trajectory: jnp.ndarray,
        method: Optional[str] = None
    ) -> jnp.ndarray:
        """
        Reduce dimensionality of a high-dimensional trajectory.
        
        Args:
            trajectory: Array of states with shape (time_steps, state_dim)
            method: Dimensionality reduction method ('pca', 'tsne', 'isomap')
                    If None, uses the method specified in the constructor
            
        Returns:
            Reduced trajectory with shape (time_steps, n_components)
        """
        if method is None:
            method = self.dim_reduction_method
        
        # Convert to numpy for sklearn compatibility
        trajectory_np = np.array(trajectory)
        
        if method == "pca" or not SKLEARN_AVAILABLE:
            if not SKLEARN_AVAILABLE:
                # Simple PCA implementation if sklearn not available
                # Center the data
                centered = trajectory_np - np.mean(trajectory_np, axis=0)
                
                # Compute covariance matrix
                cov = np.cov(centered, rowvar=False)
                
                # Compute eigenvalues and eigenvectors
                evals, evecs = np.linalg.eigh(cov)
                
                # Sort eigenvectors by eigenvalues in descending order
                idx = np.argsort(evals)[::-1]
                evecs = evecs[:, idx]
                
                # Select top components
                components = evecs[:, :self.n_components]
                
                # Project data
                reduced = np.dot(centered, components)
            else:
                # Use sklearn's PCA
                pca = PCA(n_components=self.n_components)
                reduced = pca.fit_transform(trajectory_np)
                
        elif method == "tsne":
            # t-SNE for nonlinear dimensionality reduction
            tsne = TSNE(n_components=self.n_components, random_state=42)
            reduced = tsne.fit_transform(trajectory_np)
            
        elif method == "isomap":
            # Isomap for manifold learning
            isomap = Isomap(n_components=self.n_components)
            reduced = isomap.fit_transform(trajectory_np)
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Convert back to jax array
        return jnp.array(reduced)
    
    def visualize_high_dim_trajectory(
        self,
        trajectory: jnp.ndarray,
        title: str = "High-Dimensional Phase Space Trajectory",
        method: Optional[str] = None,
        plot_type: str = "scatter",
        color_by_time: bool = True,
        show_arrows: bool = True,
        arrow_frequency: int = 10,
        marker_size: float = 10,
        marker_alpha: float = 0.7,
        show: bool = True
    ) -> Tuple[Any, jnp.ndarray]:
        """
        Visualize a high-dimensional phase space trajectory using dimensionality reduction.
        
        Args:
            trajectory: High-dimensional trajectory with shape (time_steps, state_dim)
            title: Plot title
            method: Dimensionality reduction method
            plot_type: Type of plot ('scatter', '2d', or '3d')
            color_by_time: Whether to color points by time
            show_arrows: Whether to show direction arrows
            arrow_frequency: Frequency of arrows along trajectory
            marker_size: Size of markers
            marker_alpha: Alpha transparency of markers
            show: Whether to show the plot
            
        Returns:
            Tuple of (axis, reduced_trajectory)
        """
        # Reduce dimensions
        reduced = self.reduce_dimensions(trajectory, method)
        
        # Determine dimension of reduced trajectory
        dim = min(3, reduced.shape[1])
        
        # Get appropriate visualization method
        if plot_type == "scatter" or dim < 2:
            # Scatter plot of first two components
            plt.figure(figsize=self.figsize)
            plt.scatter(
                reduced[:, 0],
                reduced[:, 1] if reduced.shape[1] > 1 else np.zeros_like(reduced[:, 0]),
                c=np.arange(len(reduced)) if color_by_time else 'b',
                cmap=self.cmap,
                s=marker_size,
                alpha=marker_alpha
            )
            plt.colorbar(label="Time")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2" if reduced.shape[1] > 1 else "")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            if show:
                plt.tight_layout()
                plt.show()
                
            return plt.gca(), reduced
            
        elif dim == 2 or plot_type == "2d":
            # Use 2D trajectory visualization
            ax = self.visualize_2d_trajectory(
                reduced[:, :2],
                axis_labels=["Component 1", "Component 2"],
                title=title,
                color_by_time=color_by_time,
                show_arrows=show_arrows,
                arrow_frequency=arrow_frequency,
                marker_size=marker_size,
                marker_alpha=marker_alpha,
                show=show
            )
            return ax, reduced
            
        else:  # dim >= 3 or plot_type == "3d"
            # Use 3D trajectory visualization
            ax = self.visualize_3d_trajectory(
                reduced[:, :3],
                axis_labels=["Component 1", "Component 2", "Component 3"],
                title=title,
                color_by_time=color_by_time,
                show_arrows=show_arrows,
                arrow_frequency=arrow_frequency,
                marker_size=marker_size,
                marker_alpha=marker_alpha,
                show=show
            )
            return ax, reduced
    
    def visualize_oscillator_dynamics(
        self,
        oscillator: Any,
        initial_state: jnp.ndarray,
        t_span: Tuple[float, float],
        n_points: int = 1000,
        parameters: Optional[Dict[str, float]] = None,
        title: str = "Oscillator Dynamics",
        plot_type: str = "auto",
        **kwargs
    ) -> Tuple[Any, jnp.ndarray]:
        """
        Visualize the dynamics of an oscillator starting from an initial state.
        
        Args:
            oscillator: Oscillator instance with vector_field method
            initial_state: Initial state for simulation
            t_span: Time span for simulation (t_start, t_end)
            n_points: Number of time points
            parameters: Optional parameters to override oscillator parameters
            title: Plot title
            plot_type: Plot type ('auto', '2d', '3d', or 'high_dim')
            **kwargs: Additional arguments for visualization methods
            
        Returns:
            Tuple of (axis, trajectory)
        """
        # Apply parameters if provided
        if parameters is not None:
            oscillator = eqx.tree_at(
                lambda o: o.params,
                oscillator,
                parameters
            )
        
        # Define vector field function
        def vector_field(t, state, args=None):
            return oscillator.vector_field(t, state, args)
        
        # Create time points
        t0, t1 = t_span
        ts = jnp.linspace(t0, t1, n_points)
        
        # Solve ODE
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Dopri5()
        
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=(t1 - t0) / n_points,
            y0=initial_state,
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=10000,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
        )
        
        # Extract trajectory
        trajectory = solution.ys
        
        # Determine plot type based on dimension if 'auto'
        if plot_type == "auto":
            dim = len(initial_state)
            if dim == 1:
                # 1D: Plot against time
                plt.figure(figsize=self.figsize)
                plt.plot(ts, trajectory, **kwargs.get("plot_kwargs", {}))
                plt.xlabel("Time")
                plt.ylabel("State")
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                return plt.gca(), trajectory
            elif dim == 2:
                plot_type = "2d"
            elif dim == 3:
                plot_type = "3d"
            else:
                plot_type = "high_dim"
        
        # Visualize based on plot type
        if plot_type == "2d":
            ax = self.visualize_2d_trajectory(
                trajectory,
                title=title,
                **kwargs
            )
            return ax, trajectory
        
        elif plot_type == "3d":
            ax = self.visualize_3d_trajectory(
                trajectory,
                title=title,
                **kwargs
            )
            return ax, trajectory
        
        elif plot_type == "high_dim":
            return self.visualize_high_dim_trajectory(
                trajectory,
                title=title,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def plot_oscillator_phase_portrait(
        self,
        oscillator: Any,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        n_grid: int = 20,
        n_trajectories: int = 5,
        traj_length: float = 5.0,
        traj_points: int = 200,
        parameters: Optional[Dict[str, float]] = None,
        title: str = "Phase Portrait",
        axis_labels: Optional[List[str]] = None,
        show_nullclines: bool = True,
        show_fixed_points: bool = True,
        fixed_point_tol: float = 1e-6,
        cmap: str = "viridis",
        show: bool = True
    ) -> Any:
        """
        Plot a phase portrait of a 2D oscillator.
        
        Args:
            oscillator: Oscillator instance with vector_field method
            x_range: Range of x values (min, max)
            y_range: Range of y values (min, max)
            n_grid: Number of grid points in each dimension
            n_trajectories: Number of trajectories to plot
            traj_length: Length of each trajectory in time
            traj_points: Number of points in each trajectory
            parameters: Optional parameters to override oscillator parameters
            title: Plot title
            axis_labels: Labels for x and y axes
            show_nullclines: Whether to show nullclines
            show_fixed_points: Whether to locate and show fixed points
            fixed_point_tol: Tolerance for fixed point detection
            cmap: Colormap for vector field
            show: Whether to show the plot
            
        Returns:
            Matplotlib axis object
        """
        # Apply parameters if provided
        if parameters is not None:
            oscillator = eqx.tree_at(
                lambda o: o.params,
                oscillator,
                parameters
            )
        
        # Define vector field function
        def vector_field(t, state, args=None):
            return oscillator.vector_field(t, state, args)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], n_grid)
        y = np.linspace(y_range[0], y_range[1], n_grid)
        X, Y = np.meshgrid(x, y)
        
        # Compute vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(n_grid):
            for j in range(n_grid):
                state = jnp.array([X[i, j], Y[i, j]])
                derivatives = vector_field(0.0, state)
                U[i, j] = derivatives[0]
                V[i, j] = derivatives[1]
        
        # Normalize for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        U_norm = U / (magnitude + 1e-10)
        V_norm = V / (magnitude + 1e-10)
        
        # Plot vector field
        quiver = ax.quiver(
            X, Y, U_norm, V_norm,
            magnitude,
            cmap=cmap,
            scale=30,
            alpha=0.7
        )
        plt.colorbar(quiver, ax=ax, label="Magnitude")
        
        # Plot nullclines if requested
        if show_nullclines:
            # X-nullcline (where dx/dt = 0)
            x_nullcline = np.abs(U) < 0.05 * np.max(np.abs(U))
            ax.scatter(
                X[x_nullcline], Y[x_nullcline],
                color='red', s=5, alpha=0.5, label="x-nullcline"
            )
            
            # Y-nullcline (where dy/dt = 0)
            y_nullcline = np.abs(V) < 0.05 * np.max(np.abs(V))
            ax.scatter(
                X[y_nullcline], Y[y_nullcline],
                color='blue', s=5, alpha=0.5, label="y-nullcline"
            )
            
            # Add legend
            ax.legend()
        
        # Find fixed points if requested
        if show_fixed_points:
            fixed_points = []
            
            # Look for points where both U and V are small
            potential_fp = (np.abs(U) < 0.1 * np.max(np.abs(U))) & (np.abs(V) < 0.1 * np.max(np.abs(V)))
            
            # Check each potential fixed point
            for i, j in zip(*np.where(potential_fp)):
                state = jnp.array([X[i, j], Y[i, j]])
                
                # Refine the fixed point using optimization
                def fp_error(state):
                    derivatives = vector_field(0.0, state)
                    return jnp.sum(derivatives**2)
                
                # Simple gradient descent refinement
                refined_state = state
                for _ in range(50):
                    # Compute gradient
                    grad_fn = jax.grad(fp_error)
                    gradient = grad_fn(refined_state)
                    
                    # Update state
                    refined_state = refined_state - 0.01 * gradient
                    
                    # Check if we've converged
                    error = fp_error(refined_state)
                    if error < fixed_point_tol:
                        break
                
                # Check if the refined point is a fixed point
                derivatives = vector_field(0.0, refined_state)
                if jnp.all(jnp.abs(derivatives) < fixed_point_tol):
                    fixed_points.append(tuple(refined_state))
            
            # Remove duplicates
            unique_fps = []
            for fp in fixed_points:
                is_duplicate = False
                for existing_fp in unique_fps:
                    if jnp.sqrt(jnp.sum((jnp.array(fp) - jnp.array(existing_fp))**2)) < 0.1:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_fps.append(fp)
            
            # Plot fixed points
            for i, fp in enumerate(unique_fps):
                ax.scatter(
                    fp[0], fp[1],
                    color='black', s=100, marker='*',
                    edgecolor='white', linewidth=1.5,
                    label=f"Fixed Point {i+1}" if i == 0 else ""
                )
                
                # Print fixed point coordinates
                print(f"Fixed Point {i+1}: ({fp[0]:.4f}, {fp[1]:.4f})")
                
                # Classify fixed point by eigenvalues if possible
                try:
                    # Compute Jacobian
                    def vector_field_batch(state):
                        return vector_field(0.0, state)
                    
                    jacobian_fn = jax.jacfwd(vector_field_batch)
                    jacobian = jacobian_fn(jnp.array(fp))
                    
                    # Compute eigenvalues
                    eigenvalues = jnp.linalg.eigvals(jacobian)
                    
                    # Classify based on eigenvalues
                    real_parts = jnp.real(eigenvalues)
                    imag_parts = jnp.imag(eigenvalues)
                    
                    if jnp.all(real_parts < 0):
                        fp_type = "Stable"
                    elif jnp.all(real_parts > 0):
                        fp_type = "Unstable"
                    else:
                        fp_type = "Saddle"
                    
                    if jnp.any(jnp.abs(imag_parts) > 1e-6):
                        fp_type += " Spiral" if "Stable" in fp_type or "Unstable" in fp_type else " Focus"
                    else:
                        fp_type += " Node" if "Stable" in fp_type or "Unstable" in fp_type else ""
                    
                    print(f"  Type: {fp_type}")
                    print(f"  Eigenvalues: {eigenvalues}")
                    
                    # Annotate fixed point
                    ax.annotate(
                        fp_type,
                        xy=(fp[0], fp[1]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        ha='left',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                    )
                
                except Exception as e:
                    print(f"  Error classifying fixed point: {e}")
        
        # Plot sample trajectories
        for i in range(n_trajectories):
            # Generate random initial state
            x0 = np.random.uniform(x_range[0], x_range[1])
            y0 = np.random.uniform(y_range[0], y_range[1])
            initial_state = jnp.array([x0, y0])
            
            # Simulate trajectory
            term = diffrax.ODETerm(vector_field)
            solver = diffrax.Dopri5()
            
            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=traj_length,
                dt0=traj_length / traj_points,
                y0=initial_state,
                saveat=diffrax.SaveAt(steps=traj_points),
                max_steps=10000,
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
            )
            
            # Plot trajectory
            trajectory = solution.ys
            self.visualize_2d_trajectory(
                trajectory,
                color_by_time=True,
                show_arrows=True,
                show=False,
                ax=ax,
                title=""
            )
        
        # Set labels and title
        if axis_labels is None:
            axis_labels = ["x", "v"]
            
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax 