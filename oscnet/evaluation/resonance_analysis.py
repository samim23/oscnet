"""
Resonance and Wave Analysis for Oscillatory Neural Networks

This module provides comprehensive analysis of wave phenomena in trained oscillatory models:
- Resonant frequency detection
- Standing wave pattern analysis  
- Phase coherence measurement
- Spatial-temporal interference patterns
- Oscillatory dynamics characterization

Can be applied to any model with oscillatory components.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

def analyze_oscillator_frequencies(model, test_data: jnp.ndarray, n_samples: int = 100) -> Dict[str, Any]:
    """
    Analyze the natural frequencies of oscillators in a trained model.
    
    Args:
        model: Trained oscillatory model
        test_data: Test dataset for analysis
        n_samples: Number of samples to analyze
        
    Returns:
        Dictionary containing frequency analysis results
    """
    print("🔍 Analyzing oscillator frequencies...")
    
    # Get a batch of test data
    batch = test_data[:n_samples]
    
    # Extract oscillator states during forward pass
    def get_oscillator_states(x):
        """Extract amplitude and velocity states during encoding"""
        batch_size = x.shape[0]
        
        # Handle different input formats
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'rnn'):
            # Autoencoder with RNN encoder
            if x.ndim == 2 and x.shape[1] == 784:  # MNIST format
                # Convert to patches
                x_images = x.reshape(batch_size, 28, 28)
                patches = x_images.reshape(batch_size, 7, 4, 7, 4)
                patches = patches.transpose(0, 1, 3, 2, 4)
                patches = patches.reshape(batch_size, 49, 16)
                patch_sequence = patches.transpose(1, 0, 2)
            else:
                # Assume already in sequence format
                patch_sequence = x.transpose(1, 0, 2) if x.ndim == 3 else x
            
            # Track states through encoder RNN
            seq_len, batch_size, input_dim = patch_sequence.shape
            hidden_dim = model.encoder.rnn.cell.hidden_dim
            
            # Initialize states
            x_states = jnp.zeros((batch_size, hidden_dim))  # amplitude
            v_states = jnp.zeros((batch_size, hidden_dim))  # velocity
            
            # Store trajectory
            amplitude_trajectory = []
            velocity_trajectory = []
            
            for t in range(seq_len):
                # Process one timestep
                output, (x_states, v_states) = model.encoder.rnn.cell(
                    patch_sequence[t], (x_states, v_states)
                )
                amplitude_trajectory.append(x_states)
                velocity_trajectory.append(v_states)
            
            return jnp.array(amplitude_trajectory), jnp.array(velocity_trajectory)
        
        else:
            # Generic oscillatory model - try to extract states
            # This is a fallback for other model types
            try:
                # Assume model has a method to get oscillator states
                if hasattr(model, 'get_oscillator_states'):
                    return model.get_oscillator_states(x)
                else:
                    # Create dummy trajectories for non-oscillatory models
                    seq_len, hidden_dim = 49, 64
                    amplitude_traj = jnp.zeros((seq_len, batch_size, hidden_dim))
                    velocity_traj = jnp.zeros((seq_len, batch_size, hidden_dim))
                    return amplitude_traj, velocity_traj
            except Exception as e:
                print(f"Warning: Could not extract oscillator states: {e}")
                # Return dummy data
                seq_len, hidden_dim = 49, 64
                amplitude_traj = jnp.zeros((seq_len, batch_size, hidden_dim))
                velocity_traj = jnp.zeros((seq_len, batch_size, hidden_dim))
                return amplitude_traj, velocity_traj
    
    # Get oscillator trajectories
    amplitude_traj, velocity_traj = get_oscillator_states(batch)
    
    # Analyze frequencies using FFT
    seq_len, batch_size, hidden_dim = amplitude_traj.shape
    
    # Compute FFT for each oscillator dimension
    freqs = jnp.fft.fftfreq(seq_len, d=1.0)  # Assuming dt=1
    
    amplitude_fft = jnp.fft.fft(amplitude_traj, axis=0)
    velocity_fft = jnp.fft.fft(velocity_traj, axis=0)
    
    # Compute power spectral density
    amplitude_psd = jnp.mean(jnp.abs(amplitude_fft)**2, axis=1)  # Average over batch
    velocity_psd = jnp.mean(jnp.abs(velocity_fft)**2, axis=1)    # Average over batch
    
    # Find dominant frequencies
    positive_freqs = freqs[freqs > 0]
    amplitude_psd_pos = amplitude_psd[freqs > 0]
    velocity_psd_pos = velocity_psd[freqs > 0]
    
    return {
        'frequencies': positive_freqs,
        'amplitude_psd': amplitude_psd_pos,
        'velocity_psd': velocity_psd_pos,
        'amplitude_trajectory': amplitude_traj,
        'velocity_trajectory': velocity_traj,
        'sequence_length': seq_len,
        'hidden_dim': hidden_dim
    }

def analyze_standing_waves(model, test_data: jnp.ndarray, n_samples: int = 50) -> Dict[str, Any]:
    """
    Analyze standing wave patterns in the model's spatial processing.
    
    Args:
        model: Trained oscillatory model
        test_data: Test dataset for analysis
        n_samples: Number of samples to analyze
        
    Returns:
        Dictionary containing standing wave analysis results
    """
    print("🌊 Analyzing standing wave patterns...")
    
    batch = test_data[:n_samples]
    
    # Get spatial activations
    def get_spatial_activations(x):
        """Get activations for spatial analysis"""
        batch_size = x.shape[0]
        
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'rnn'):
            # Autoencoder with spatial structure
            if x.ndim == 2 and x.shape[1] == 784:  # MNIST format
                # Convert to patches
                x_images = x.reshape(batch_size, 28, 28)
                patches = x_images.reshape(batch_size, 7, 4, 7, 4)
                patches = patches.transpose(0, 1, 3, 2, 4)
                patches = patches.reshape(batch_size, 49, 16)
                patch_sequence = patches.transpose(1, 0, 2)
                
                # Get encoder outputs for each patch
                outputs = model.encoder.rnn(patch_sequence)  # (49, batch, hidden_dim)
                
                return outputs, patches
            else:
                # Generic sequence processing
                outputs = model(x) if callable(model) else x
                return outputs, x
        else:
            # Generic model
            try:
                outputs = model(x) if callable(model) else x
                return outputs, x
            except Exception as e:
                print(f"Warning: Could not get spatial activations: {e}")
                # Return dummy data
                seq_len, hidden_dim = 49, 64
                outputs = jnp.zeros((seq_len, batch_size, hidden_dim))
                return outputs, x
    
    spatial_outputs, input_data = get_spatial_activations(batch)
    
    # Handle different output shapes
    if spatial_outputs.ndim == 3:
        seq_len, batch_size, hidden_dim = spatial_outputs.shape
        
        # Try to reshape to spatial grid if possible
        if seq_len == 49:  # 7x7 grid
            spatial_grid = spatial_outputs.reshape(7, 7, batch_size, hidden_dim)
        else:
            # Create a square grid as close as possible
            grid_size = int(jnp.sqrt(seq_len))
            if grid_size * grid_size == seq_len:
                spatial_grid = spatial_outputs.reshape(grid_size, grid_size, batch_size, hidden_dim)
            else:
                # Pad or truncate to make square
                target_size = grid_size * grid_size
                if seq_len > target_size:
                    spatial_outputs_truncated = spatial_outputs[:target_size]
                else:
                    padding = target_size - seq_len
                    spatial_outputs_truncated = jnp.concatenate([
                        spatial_outputs, 
                        jnp.zeros((padding, batch_size, hidden_dim))
                    ], axis=0)
                spatial_grid = spatial_outputs_truncated.reshape(grid_size, grid_size, batch_size, hidden_dim)
    else:
        # Handle 2D outputs
        batch_size, feature_dim = spatial_outputs.shape
        grid_size = int(jnp.sqrt(feature_dim))
        if grid_size * grid_size == feature_dim:
            spatial_grid = spatial_outputs.reshape(batch_size, grid_size, grid_size, 1)
            spatial_grid = spatial_grid.transpose(1, 2, 0, 3)  # (grid, grid, batch, 1)
            hidden_dim = 1
        else:
            # Create dummy spatial grid
            spatial_grid = jnp.zeros((7, 7, batch_size, 64))
            hidden_dim = 64
    
    # Compute spatial correlations
    spatial_correlations = []
    grid_h, grid_w, batch_size, hidden_dim = spatial_grid.shape
    
    for dim in range(min(8, hidden_dim)):  # Analyze first 8 dimensions
        dim_outputs = spatial_grid[:, :, :, dim]  # (grid_h, grid_w, batch)
        
        # Compute 2D spatial FFT for each sample
        spatial_fft = jnp.fft.fft2(dim_outputs, axes=(0, 1))  # FFT over spatial dimensions
        spatial_psd = jnp.mean(jnp.abs(spatial_fft)**2, axis=2)  # Average over batch
        
        spatial_correlations.append(spatial_psd)
    
    return {
        'spatial_outputs': spatial_grid,
        'spatial_correlations': jnp.array(spatial_correlations),
        'input_data': input_data,
        'grid_shape': (grid_h, grid_w)
    }

def analyze_phase_coherence(frequency_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze phase coherence between amplitude and velocity oscillations.
    
    Args:
        frequency_data: Output from analyze_oscillator_frequencies
        
    Returns:
        Dictionary containing phase coherence analysis results
    """
    print("🔄 Analyzing phase coherence...")
    
    amplitude_traj = frequency_data['amplitude_trajectory']
    velocity_traj = frequency_data['velocity_trajectory']
    
    seq_len, batch_size, hidden_dim = amplitude_traj.shape
    
    # Compute instantaneous phase using Hilbert transform
    def compute_phase(signal):
        """Compute instantaneous phase of a signal"""
        analytic_signal = jnp.fft.fft(signal, axis=0)
        # Zero out negative frequencies for Hilbert transform
        analytic_signal = analytic_signal.at[seq_len//2+1:].set(0)
        analytic_signal = analytic_signal.at[1:seq_len//2+1].multiply(2)
        analytic_time = jnp.fft.ifft(analytic_signal, axis=0)
        return jnp.angle(analytic_time)
    
    # Compute phases
    amplitude_phases = jax.vmap(compute_phase, in_axes=2, out_axes=2)(amplitude_traj)
    velocity_phases = jax.vmap(compute_phase, in_axes=2, out_axes=2)(velocity_traj)
    
    # Compute phase differences
    phase_diff = amplitude_phases - velocity_phases
    
    # Compute phase coherence (how consistent the phase relationship is)
    phase_coherence = jnp.abs(jnp.mean(jnp.exp(1j * phase_diff), axis=(0, 1)))
    
    return {
        'amplitude_phases': amplitude_phases,
        'velocity_phases': velocity_phases,
        'phase_difference': phase_diff,
        'phase_coherence': phase_coherence
    }

def create_resonance_visualizations(
    frequency_data: Dict[str, Any], 
    standing_wave_data: Dict[str, Any], 
    phase_data: Dict[str, Any], 
    output_dir: Path,
    model_name: str = "Oscillatory Model"
) -> None:
    """
    Create comprehensive visualizations of resonance effects.
    
    Args:
        frequency_data: Frequency analysis results
        standing_wave_data: Standing wave analysis results
        phase_data: Phase coherence analysis results
        output_dir: Directory to save visualizations
        model_name: Name of the model for titles
    """
    print("📊 Creating resonance visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Resonance and Wave Analysis: {model_name}', fontsize=20, y=0.98)
    
    # 1. Frequency spectrum
    ax1 = plt.subplot(3, 4, 1)
    freqs = frequency_data['frequencies']
    amp_psd = jnp.mean(frequency_data['amplitude_psd'], axis=1)
    vel_psd = jnp.mean(frequency_data['velocity_psd'], axis=1)
    
    plt.semilogy(freqs, amp_psd, 'b-', label='Amplitude', linewidth=2)
    plt.semilogy(freqs, vel_psd, 'r-', label='Velocity', linewidth=2)
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.title('Oscillator Frequency Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Dominant frequencies
    ax2 = plt.subplot(3, 4, 2)
    # Find peaks in amplitude spectrum
    amp_psd_mean = jnp.mean(frequency_data['amplitude_psd'], axis=1)
    peak_indices = []
    for i in range(1, len(amp_psd_mean)-1):
        if amp_psd_mean[i] > amp_psd_mean[i-1] and amp_psd_mean[i] > amp_psd_mean[i+1]:
            if amp_psd_mean[i] > 0.1 * jnp.max(amp_psd_mean):  # Only significant peaks
                peak_indices.append(i)
    
    if peak_indices:
        # Convert list to JAX array for indexing
        peak_indices_array = jnp.array(peak_indices)
        peak_freqs = freqs[peak_indices_array]
        peak_powers = amp_psd_mean[peak_indices_array]
        plt.bar(range(len(peak_freqs)), peak_powers, alpha=0.7, color='skyblue')
        plt.xticks(range(len(peak_freqs)), [f'{f:.3f}' for f in peak_freqs], rotation=45)
        plt.ylabel('Peak Power')
        plt.title('Resonant Frequencies')
    else:
        plt.text(0.5, 0.5, 'No clear peaks found', ha='center', va='center', transform=ax2.transAxes)
        plt.title('Resonant Frequencies')
    
    # 3. Spatial standing wave patterns (first few modes)
    for i in range(4):
        ax = plt.subplot(3, 4, 5 + i)
        if i < len(standing_wave_data['spatial_correlations']):
            spatial_pattern = standing_wave_data['spatial_correlations'][i]
            im = plt.imshow(jnp.abs(spatial_pattern), cmap='viridis', aspect='equal')
            plt.title(f'Spatial Mode {i+1}')
            plt.colorbar(im, shrink=0.8)
        else:
            plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            plt.title(f'Spatial Mode {i+1}')
    
    # 4. Phase coherence
    ax3 = plt.subplot(3, 4, 3)
    coherence = phase_data['phase_coherence']
    n_dims = min(16, len(coherence))
    plt.bar(range(n_dims), coherence[:n_dims], color='orange', alpha=0.7)
    plt.xlabel('Oscillator Dimension')
    plt.ylabel('Phase Coherence')
    plt.title('Amplitude-Velocity Phase Coherence')
    plt.xticks(range(0, n_dims, 2))
    
    # 5. Phase difference distribution
    ax4 = plt.subplot(3, 4, 4)
    phase_diff_flat = phase_data['phase_difference'].flatten()
    plt.hist(phase_diff_flat, bins=50, alpha=0.7, density=True, color='purple')
    plt.xlabel('Phase Difference (radians)')
    plt.ylabel('Density')
    plt.title('Phase Difference Distribution')
    plt.axvline(jnp.pi/2, color='r', linestyle='--', label='π/2 (Quadrature)')
    plt.axvline(-jnp.pi/2, color='r', linestyle='--')
    plt.legend()
    
    # 6. Temporal evolution of oscillator states
    ax5 = plt.subplot(3, 4, 9)
    time_steps = range(frequency_data['amplitude_trajectory'].shape[0])
    n_show = min(3, frequency_data['amplitude_trajectory'].shape[2])
    colors = ['blue', 'green', 'red']
    for i in range(n_show):
        amp_trace = frequency_data['amplitude_trajectory'][:, 0, i]  # First sample
        plt.plot(time_steps, amp_trace, label=f'Osc {i+1}', linewidth=2, color=colors[i])
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    plt.title('Oscillator Amplitude Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Velocity evolution
    ax6 = plt.subplot(3, 4, 10)
    for i in range(n_show):
        vel_trace = frequency_data['velocity_trajectory'][:, 0, i]  # First sample
        plt.plot(time_steps, vel_trace, label=f'Osc {i+1}', linewidth=2, color=colors[i])
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')
    plt.title('Oscillator Velocity Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Phase space trajectory
    ax7 = plt.subplot(3, 4, 11)
    # Plot phase space (amplitude vs velocity) for first oscillator
    amp_trace = frequency_data['amplitude_trajectory'][:, 0, 0]
    vel_trace = frequency_data['velocity_trajectory'][:, 0, 0]
    plt.plot(amp_trace, vel_trace, 'b-', alpha=0.7, linewidth=2)
    plt.scatter(amp_trace[0], vel_trace[0], color='green', s=50, label='Start', zorder=5)
    plt.scatter(amp_trace[-1], vel_trace[-1], color='red', s=50, label='End', zorder=5)
    plt.xlabel('Amplitude')
    plt.ylabel('Velocity')
    plt.title('Phase Space Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Resonance strength across dimensions
    ax8 = plt.subplot(3, 4, 12)
    # Compute resonance strength as peak-to-mean ratio in frequency domain
    resonance_strength = []
    n_dims = min(16, frequency_data['amplitude_psd'].shape[1])
    for dim in range(n_dims):
        psd_dim = frequency_data['amplitude_psd'][:, dim]
        peak_power = jnp.max(psd_dim)
        mean_power = jnp.mean(psd_dim)
        resonance_strength.append(peak_power / (mean_power + 1e-8))
    
    plt.bar(range(len(resonance_strength)), resonance_strength, color='coral', alpha=0.7)
    plt.xlabel('Oscillator Dimension')
    plt.ylabel('Resonance Strength (Peak/Mean)')
    plt.title('Resonance Strength by Dimension')
    plt.xticks(range(0, len(resonance_strength), 2))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'resonance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    with open(output_dir / 'resonance_report.txt', 'w') as f:
        f.write(f"RESONANCE AND STANDING WAVE ANALYSIS REPORT\n")
        f.write(f"Model: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. FREQUENCY ANALYSIS:\n")
        f.write(f"   - Analyzed {len(freqs)} frequency bins\n")
        f.write(f"   - Peak amplitude frequency: {freqs[jnp.argmax(amp_psd_mean)]:.4f}\n")
        f.write(f"   - Peak velocity frequency: {freqs[jnp.argmax(vel_psd)]:.4f}\n")
        
        f.write("\n2. PHASE COHERENCE:\n")
        f.write(f"   - Mean phase coherence: {jnp.mean(coherence):.4f}\n")
        f.write(f"   - Max phase coherence: {jnp.max(coherence):.4f}\n")
        f.write(f"   - Highly coherent dimensions: {jnp.sum(coherence > 0.8)}\n")
        
        f.write("\n3. RESONANCE EFFECTS:\n")
        if resonance_strength:
            f.write(f"   - Mean resonance strength: {jnp.mean(jnp.array(resonance_strength)):.4f}\n")
            f.write(f"   - Max resonance strength: {jnp.max(jnp.array(resonance_strength)):.4f}\n")
            f.write(f"   - Strong resonators (>5x): {jnp.sum(jnp.array(resonance_strength) > 5)}\n")
        
        f.write("\n4. STANDING WAVE PATTERNS:\n")
        f.write(f"   - Analyzed {len(standing_wave_data['spatial_correlations'])} spatial modes\n")
        f.write(f"   - Spatial grid shape: {standing_wave_data['grid_shape']}\n")
        
        f.write("\n5. WAVE PHENOMENA SUMMARY:\n")
        # Determine if strong wave effects are present
        mean_resonance = jnp.mean(jnp.array(resonance_strength)) if resonance_strength else 0
        mean_coherence = jnp.mean(coherence)
        
        if mean_resonance > 10 and mean_coherence > 0.5:
            f.write("   🌊 STRONG WAVE PHENOMENA DETECTED!\n")
            f.write("   - Model exhibits significant resonance and coherence\n")
            f.write("   - Wave-based computation is likely occurring\n")
        elif mean_resonance > 5 or mean_coherence > 0.3:
            f.write("   🌊 MODERATE WAVE PHENOMENA DETECTED\n")
            f.write("   - Some oscillatory dynamics present\n")
        else:
            f.write("   📊 LIMITED WAVE PHENOMENA\n")
            f.write("   - Minimal oscillatory behavior detected\n")

def comprehensive_resonance_analysis(
    model, 
    test_data: jnp.ndarray, 
    output_dir: Path,
    model_name: str = "Oscillatory Model",
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Perform comprehensive resonance and wave analysis on a trained model.
    
    Args:
        model: Trained oscillatory model
        test_data: Test dataset for analysis
        output_dir: Directory to save results
        model_name: Name of the model for reports
        n_samples: Number of samples to analyze
        
    Returns:
        Dictionary containing all analysis results
    """
    print(f"\n🌊 COMPREHENSIVE RESONANCE ANALYSIS: {model_name}")
    print("=" * 60)
    
    # Create resonance analysis subdirectory
    resonance_dir = output_dir / "resonance_analysis"
    resonance_dir.mkdir(exist_ok=True)
    
    try:
        # Perform all analyses
        frequency_data = analyze_oscillator_frequencies(model, test_data, n_samples)
        standing_wave_data = analyze_standing_waves(model, test_data, n_samples//2)
        phase_data = analyze_phase_coherence(frequency_data)
        
        # Create visualizations
        create_resonance_visualizations(
            frequency_data, standing_wave_data, phase_data, 
            resonance_dir, model_name
        )
        
        # Compile results
        results = {
            'frequency_analysis': frequency_data,
            'standing_wave_analysis': standing_wave_data,
            'phase_coherence_analysis': phase_data,
            'summary': {
                'model_name': model_name,
                'samples_analyzed': n_samples,
                'resonance_detected': True,  # Will be refined based on analysis
                'output_directory': str(resonance_dir)
            }
        }
        
        print(f"\n✅ Resonance analysis complete! Results saved to {resonance_dir}")
        print("🔍 Check resonance_analysis.png for visual analysis")
        print("📄 Check resonance_report.txt for detailed summary")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during resonance analysis: {e}")
        return {
            'error': str(e),
            'summary': {
                'model_name': model_name,
                'samples_analyzed': 0,
                'resonance_detected': False,
                'output_directory': str(resonance_dir)
            }
        } 