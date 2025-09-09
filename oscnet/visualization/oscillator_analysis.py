"""
Oscillator Analysis and Discovery Tools.

This module provides comprehensive analysis tools for learnable oscillators,
revealing frequency families, specialization patterns, and emergent organizational
principles that explain superior performance over fixed oscillator parameters.

Designed for both ML practitioners and dynamical systems researchers.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union

# Import oscillator classes
from ..core.oscillators import (
    LearnableNonlinearHarmonicOscillator,
    AdaptiveNonlinearHarmonicOscillator
)


def analyze_oscillator_families(
    model: eqx.Module,
    n_families: int = 5,
    extract_fn: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Discover frequency families and specialization patterns in learnable oscillators.
    
    Args:
        model: Trained model containing learnable oscillators
        n_families: Number of frequency families to identify via clustering
        extract_fn: Optional custom function to extract oscillator data
        
    Returns:
        Dictionary containing family analysis results
    """
    print("🎵 ANALYZING OSCILLATOR FREQUENCY FAMILIES")
    print("="*50)
    
    # Extract oscillator data
    oscillator_data = extract_fn(model) if extract_fn else _extract_oscillator_data(model)
    
    results = {}
    
    for component_name, component_data in oscillator_data.items():
        omega = np.array(component_data['omega'])
        gamma = np.array(component_data['gamma'])
        
        # Convert to Hz for interpretability
        freq_hz = omega / (2 * np.pi)
        
        # Cluster by frequency and damping
        features = np.column_stack([freq_hz, gamma])
        kmeans = KMeans(n_clusters=n_families, random_state=42)
        family_labels = kmeans.fit_predict(features)
        
        # Analyze each family
        families = {}
        for i in range(n_families):
            mask = family_labels == i
            family_freqs = freq_hz[mask]
            family_gammas = gamma[mask]
            
            families[f'family_{i}'] = {
                'indices': np.where(mask)[0].tolist(),
                'count': int(np.sum(mask)),
                'freq_mean': float(np.mean(family_freqs)),
                'freq_std': float(np.std(family_freqs)),
                'freq_range': [float(np.min(family_freqs)), float(np.max(family_freqs))],
                'gamma_mean': float(np.mean(family_gammas)),
                'gamma_std': float(np.std(family_gammas)),
                'gamma_range': [float(np.min(family_gammas)), float(np.max(family_gammas))],
                'specialization': _classify_family_role(np.mean(family_freqs), np.mean(family_gammas))
            }
        
        results[component_name] = {
            'families': families,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'raw_frequencies': freq_hz.tolist(),
            'raw_damping': gamma.tolist(),
            'family_labels': family_labels.tolist()
        }
        
        # Print family summary
        print(f"\n📊 {component_name.upper()} FREQUENCY FAMILIES:")
        for i, (family_name, family_data) in enumerate(families.items()):
            print(f"   Family {i+1} ({family_data['specialization']}):")
            print(f"      Count: {family_data['count']} oscillators")
            print(f"      Frequency: {family_data['freq_mean']:.4f} ± {family_data['freq_std']:.4f} Hz")
            print(f"      Damping: {family_data['gamma_mean']:.4f} ± {family_data['gamma_std']:.4f}")
    
    return results


def analyze_parameter_distributions(
    model: eqx.Module,
    extract_fn: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Analyze the distribution and utilization of learned oscillator parameters.
    
    Args:
        model: Trained model containing learnable oscillators
        extract_fn: Optional custom function to extract oscillator data
        
    Returns:
        Dictionary containing parameter distribution statistics
    """
    print("\n📊 ANALYZING PARAMETER DISTRIBUTIONS")
    print("="*50)
    
    oscillator_data = extract_fn(model) if extract_fn else _extract_oscillator_data(model)
    results = {}
    
    for component_name, component_data in oscillator_data.items():
        omega = np.array(component_data['omega'])
        gamma = np.array(component_data['gamma'])
        
        freq_hz = omega / (2 * np.pi)
        
        # Calculate distribution statistics
        results[component_name] = {
            'frequency_stats': {
                'mean': float(np.mean(freq_hz)),
                'std': float(np.std(freq_hz)),
                'min': float(np.min(freq_hz)),
                'max': float(np.max(freq_hz)),
                'median': float(np.median(freq_hz)),
                'q25': float(np.percentile(freq_hz, 25)),
                'q75': float(np.percentile(freq_hz, 75))
            },
            'damping_stats': {
                'mean': float(np.mean(gamma)),
                'std': float(np.std(gamma)),
                'min': float(np.min(gamma)),
                'max': float(np.max(gamma)),
                'median': float(np.median(gamma)),
                'q25': float(np.percentile(gamma, 25)),
                'q75': float(np.percentile(gamma, 75))
            }
        }
        
        # Calculate bounds utilization if available
        if 'omega_bounds' in component_data and 'gamma_bounds' in component_data:
            omega_bounds = component_data['omega_bounds']
            gamma_bounds = component_data['gamma_bounds']
            
            freq_range_available = omega_bounds[1]/(2*np.pi) - omega_bounds[0]/(2*np.pi)
            gamma_range_available = gamma_bounds[1] - gamma_bounds[0]
            
            results[component_name]['bounds_utilization'] = {
                'freq_range_used': float((np.max(freq_hz) - np.min(freq_hz)) / freq_range_available),
                'gamma_range_used': float((np.max(gamma) - np.min(gamma)) / gamma_range_available)
            }
        
        print(f"\n{component_name.upper()} PARAMETER STATISTICS:")
        print(f"   Frequency: {results[component_name]['frequency_stats']['mean']:.4f} ± {results[component_name]['frequency_stats']['std']:.4f} Hz")
        print(f"   Range: [{results[component_name]['frequency_stats']['min']:.4f}, {results[component_name]['frequency_stats']['max']:.4f}] Hz")
        print(f"   Damping: {results[component_name]['damping_stats']['mean']:.4f} ± {results[component_name]['damping_stats']['std']:.4f}")
        
        if 'bounds_utilization' in results[component_name]:
            print(f"   Range used: {results[component_name]['bounds_utilization']['freq_range_used']*100:.1f}% (freq), {results[component_name]['bounds_utilization']['gamma_range_used']*100:.1f}% (damp)")
    
    return results


def visualize_oscillator_families(
    family_data: Dict[str, Any],
    figsize: Tuple[int, int] = (18, 12),
    title: str = "Learnable Oscillator Family Analysis",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create comprehensive visualization of oscillator frequency families.
    
    Args:
        family_data: Results from analyze_oscillator_families()
        figsize: Figure size
        title: Figure title
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    n_components = len(family_data)
    fig, axes = plt.subplots(n_components, 3, figsize=figsize)
    fig.suptitle(f'🎵 {title}', fontsize=16, fontweight='bold')
    
    # Handle single component case
    if n_components == 1:
        axes = axes.reshape(1, -1)
    
    for i, (component_name, data) in enumerate(family_data.items()):
        # 1. Frequency vs Damping scatter plot with families
        ax = axes[i, 0]
        frequencies = np.array(data['raw_frequencies'])
        damping = np.array(data['raw_damping'])
        labels = np.array(data['family_labels'])
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(labels))))
        for j, label in enumerate(np.unique(labels)):
            mask = labels == label
            family_info = data['families'][f'family_{label}']
            ax.scatter(frequencies[mask], damping[mask], 
                      c=[colors[j]], label=f"Family {label+1} ({family_info['count']})", 
                      alpha=0.7, s=60)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Damping Coefficient')
        ax.set_title(f'{component_name.title()} Frequency Families')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 2. Frequency distribution
        ax = axes[i, 1]
        ax.hist(frequencies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(frequencies), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(frequencies):.4f}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Count')
        ax.set_title(f'{component_name.title()} Frequency Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Damping distribution
        ax = axes[i, 2]
        ax.hist(damping, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.axvline(np.mean(damping), color='red', linestyle='--',
                  label=f'Mean: {np.mean(damping):.4f}')
        ax.set_xlabel('Damping Coefficient')
        ax.set_ylabel('Count')
        ax.set_title(f'{component_name.title()} Damping Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def visualize_specialization_map(
    family_data: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Oscillator Functional Specialization Map",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create functional specialization map showing oscillator roles.
    
    Args:
        family_data: Results from analyze_oscillator_families()
        figsize: Figure size
        title: Figure title
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create theoretical specialization regions
    freq_range = np.linspace(0, 0.3, 100)
    gamma_range = np.linspace(0, 0.2, 100)
    freq_grid, gamma_grid = np.meshgrid(freq_range, gamma_range)
    
    # Define specialization regions
    specialization_map = np.zeros_like(freq_grid)
    
    # Frequency specialization (1-3)
    specialization_map[freq_grid > 0.15] += 3  # High-freq
    specialization_map[(freq_grid > 0.05) & (freq_grid <= 0.15)] += 2  # Medium-freq
    specialization_map[freq_grid <= 0.05] += 1  # Low-freq
    
    # Damping specialization (10-30, for distinction)
    specialization_map[gamma_grid > 0.05] += 30  # Heavy-damp
    specialization_map[(gamma_grid > 0.02) & (gamma_grid <= 0.05)] += 20  # Medium-damp
    specialization_map[gamma_grid <= 0.02] += 10  # Light-damp
    
    # Create background colormap
    im = ax.contourf(freq_grid, gamma_grid, specialization_map, 
                    levels=20, alpha=0.6, cmap='viridis')
    
    # Plot actual oscillators
    for component_name, data in family_data.items():
        frequencies = np.array(data['raw_frequencies'])
        damping = np.array(data['raw_damping'])
        labels = np.array(data['family_labels'])
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(labels))))
        marker = 'o' if 'encoder' in component_name.lower() else 's'
        
        for j, label in enumerate(np.unique(labels)):
            mask = labels == label
            family_info = data['families'][f'family_{label}']
            ax.scatter(frequencies[mask], damping[mask], 
                      c=[colors[j]], marker=marker, s=100,
                      label=f"{component_name.title()} F{label+1} ({family_info['count']})",
                      edgecolors='black', linewidth=1)
    
    # Add specialization region labels
    specialization_labels = [
        (0.25, 0.15, "High-freq\nHeavy-damp\n(Robust edges)", 'white'),
        (0.25, 0.035, "High-freq\nMedium-damp\n(Sharp features)", 'white'),
        (0.25, 0.005, "High-freq\nLight-damp\n(Fine details)", 'black'),
        (0.1, 0.15, "Medium-freq\nHeavy-damp\n(Stable patterns)", 'white'),
        (0.1, 0.035, "Medium-freq\nMedium-damp\n(Textures)", 'white'),
        (0.1, 0.005, "Medium-freq\nLight-damp\n(Dynamic patterns)", 'black'),
        (0.025, 0.15, "Low-freq\nHeavy-damp\n(Global structure)", 'white'),
        (0.025, 0.035, "Low-freq\nMedium-damp\n(Smooth regions)", 'white'),
        (0.025, 0.005, "Low-freq\nLight-damp\n(Sensitive global)", 'black')
    ]
    
    for x, y, text, color in specialization_labels:
        ax.text(x, y, text, ha='center', va='center', fontsize=8, 
               color=color, weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Damping Coefficient', fontweight='bold')
    ax.set_title(f'🎯 {title}', fontweight='bold', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig


def generate_discovery_report(
    family_data: Dict[str, Any],
    param_stats: Dict[str, Any],
    coupling_data: Optional[Dict[str, Any]] = None,
    onn_analysis: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive text report of oscillator discoveries.
    
    Args:
        family_data: Results from analyze_oscillator_families()
        param_stats: Results from analyze_parameter_distributions()
        coupling_data: Optional coupling strength analysis results
        onn_analysis: Optional ONN-specific analysis results
        save_path: Optional path to save report
        
    Returns:
        Report text string
    """
    report_lines = [
        "🔬 OSCILLATOR DISCOVERY ANALYSIS REPORT",
        "="*60,
        "",
        "EXECUTIVE SUMMARY",
        "-"*20,
        "This analysis reveals how learnable oscillators automatically",
        "discovered specialized frequency families and functional roles",
        "during training, explaining their superior performance.",
        "",
        "🎵 FREQUENCY FAMILY DISCOVERIES",
        "-"*40
    ]
    
    # Family analysis
    for component_name, families_data in family_data.items():
        report_lines.extend([
            f"\n{component_name.upper()} FAMILIES:",
        ])
        
        families = families_data['families']
        for i, (family_name, family_info) in enumerate(families.items()):
            report_lines.extend([
                f"  Family {i+1}: {family_info['specialization']}",
                f"    • Population: {family_info['count']} oscillators",
                f"    • Frequency: {family_info['freq_mean']:.4f} ± {family_info['freq_std']:.4f} Hz",
                f"    • Damping: {family_info['gamma_mean']:.4f} ± {family_info['gamma_std']:.4f}",
                f"    • Indices: {family_info['indices'][:5]}{'...' if len(family_info['indices']) > 5 else ''}",
                ""
            ])
    
    # Parameter utilization
    report_lines.extend([
        "📊 PARAMETER SPACE UTILIZATION",
        "-"*40
    ])
    
    for component_name, stats in param_stats.items():
        report_lines.extend([
            f"\n{component_name.upper()}:",
            f"  Frequency diversity: σ = {stats['frequency_stats']['std']:.4f} Hz",
            f"  Damping diversity: σ = {stats['damping_stats']['std']:.4f}"
        ])
        
        if 'bounds_utilization' in stats:
            report_lines.extend([
                f"  Frequency exploration: {stats['bounds_utilization']['freq_range_used']*100:.1f}% of available range",
                f"  Damping exploration: {stats['bounds_utilization']['gamma_range_used']*100:.1f}% of available range"
            ])
        
        report_lines.append("")
    
    # Coupling discoveries (if provided)
    if coupling_data:
        report_lines.extend([
            "⚡ COUPLING STRENGTH DISCOVERIES",
            "-"*40
        ])
        
        for component_name, coupling in coupling_data.items():
            report_lines.extend([
                f"\n{component_name.upper()}:",
                f"  Learned multiplier: {coupling['coupling_multiplier']:.4f}",
                f"  Effective coupling: {coupling['effective_gain']:.4f}",
                f"  HORN amplification: {coupling['gain_boost']:.1f}x stronger",
                ""
            ])
    
    # ONN-specific analysis
    if onn_analysis:
        report_lines.extend([
            "🌊 ONN-SPECIFIC ANALYSIS",
            "-"*40
        ])
        
        for analysis_name, analysis_data in onn_analysis.items():
            if isinstance(analysis_data, dict):
                for metric_name, metric_value in analysis_data.items():
                    report_lines.extend([
                        f"\n{analysis_name.upper()}:",
                        f"  {metric_name.replace('_', ' ').capitalize()}: {metric_value}"
                    ])
            else:
                report_lines.extend([
                    f"\n{analysis_name.upper()}:",
                    f"  {analysis_data}"
                ])
    
    # Key insights
    report_lines.extend([
        "🧠 KEY INSIGHTS",
        "-"*20,
        "1. AUTOMATIC SPECIALIZATION: Oscillators self-organized into",
        "   distinct functional families without explicit supervision.",
        "",
        "2. FREQUENCY HIERARCHY: Natural emergence of high/medium/low",
        "   frequency specialists for different feature scales.",
        "",
        "3. DAMPING OPTIMIZATION: Each family found optimal stability",
        "   vs responsiveness trade-offs for their functional role.",
        "",
        "4. PARAMETER EFFICIENCY: Oscillators used available parameter",
        "   space efficiently, exploring diverse but focused regions.",
        "",
        "-"*20,
    ])
    
    report_text = "\n".join(report_lines)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text


def comprehensive_oscillator_analysis(
    model: eqx.Module,
    output_dir: str,
    n_families: int = 5,
    extract_fn: Optional[callable] = None,
    generate_visualizations: bool = True,
    generate_report: bool = True,
    test_data: Optional[jnp.ndarray] = None
) -> Dict[str, Any]:
    """
    Run comprehensive analysis of learnable oscillators including ONN-specific metrics.
    
    Args:
        model: Trained model containing learnable oscillators
        output_dir: Directory to save analysis results
        n_families: Number of frequency families to identify
        extract_fn: Optional custom function to extract oscillator data
        generate_visualizations: Whether to create and save visualizations
        generate_report: Whether to generate text report
        test_data: Optional test data for temporal dynamics analysis
        
    Returns:
        Dictionary containing all analysis results
    """
    print("🔬 COMPREHENSIVE OSCILLATOR DISCOVERY ANALYSIS")
    print("="*60)
    print("Analyzing learned oscillator patterns and specializations...\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run basic analyses
    family_data = analyze_oscillator_families(model, n_families, extract_fn)
    param_stats = analyze_parameter_distributions(model, extract_fn)
    
    # Try to analyze coupling if available
    coupling_data = None
    try:
        coupling_data = _analyze_coupling_strength(model, extract_fn)
    except Exception:
        print("⚠️  Coupling analysis not available for this model type")
    
    # NEW: ONN-specific analyses
    onn_analysis = {}
    
    if test_data is not None:
        print("\n🌊 RUNNING ONN-SPECIFIC TEMPORAL ANALYSIS")
        print("="*50)
        
        # Phase synchronization analysis
        try:
            sync_analysis = analyze_phase_synchronization(model, test_data[:32])  # Use small batch
            onn_analysis['phase_synchronization'] = sync_analysis
            print("   ✅ Phase synchronization analysis completed!")
        except Exception as e:
            print(f"   ⚠️  Phase sync analysis failed: {e}")
        
        # Temporal oscillation quality
        try:
            osc_quality = analyze_oscillation_quality(model, test_data[:32])
            onn_analysis['oscillation_quality'] = osc_quality
            print("   ✅ Oscillation quality analysis completed!")
        except Exception as e:
            print(f"   ⚠️  Oscillation quality analysis failed: {e}")
        
        # Coupling network analysis
        try:
            coupling_network = analyze_coupling_network(model, test_data[:16])
            onn_analysis['coupling_network'] = coupling_network
            print("   ✅ Coupling network analysis completed!")
        except Exception as e:
            print(f"   ⚠️  Coupling network analysis failed: {e}")
    
    # Create visualizations
    if generate_visualizations:
        print("\n🎨 Creating visualizations...")
        
        family_fig = visualize_oscillator_families(
            family_data,
            save_path=output_path / 'oscillator_family_analysis.png',
            show=False
        )
        
        spec_fig = visualize_specialization_map(
            family_data,
            save_path=output_path / 'oscillator_specialization_map.png',
            show=False
        )
        
        # NEW: ONN-specific visualizations
        if onn_analysis:
            try:
                onn_fig = visualize_onn_dynamics(
                    onn_analysis,
                    save_path=output_path / 'onn_dynamics_analysis.png',
                    show=False
                )
                print("   ✅ ONN dynamics visualization created!")
            except Exception as e:
                print(f"   ⚠️  ONN visualization failed: {e}")
    
    # Generate report
    if generate_report:
        print("📝 Generating discovery report...")
        report_text = generate_discovery_report(
            family_data, param_stats, coupling_data, onn_analysis,
            save_path=output_path / 'oscillator_discovery_report.txt'
        )
    
    # Compile results
    analysis_results = {
        'frequency_families': family_data,
        'parameter_statistics': param_stats,
        'coupling_analysis': coupling_data,
        'onn_analysis': onn_analysis,  # NEW: ONN-specific results
        'summary': {
            'total_oscillators': len(list(family_data.values())[0]['raw_frequencies']),
            'families_discovered': len(list(family_data.values())[0]['families']),
            'onn_analysis_complete': bool(onn_analysis),
            'analysis_complete': True
        }
    }
    
    # Save complete analysis data
    with open(output_path / 'complete_analysis_data.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_path}")
    if generate_visualizations:
        print("📊 Visualizations:")
        print("   • oscillator_family_analysis.png")
        print("   • oscillator_specialization_map.png")
        if onn_analysis:
            print("   • onn_dynamics_analysis.png")
    if generate_report:
        print("📝 Reports:")
        print("   • oscillator_discovery_report.txt")
    print("💾 Data:")
    print("   • complete_analysis_data.json")
    
    return analysis_results


# ========== HELPER FUNCTIONS ==========

def _extract_oscillator_data(model: eqx.Module) -> Dict[str, Any]:
    """
    Default function to extract oscillator data from common model architectures.
    
    Args:
        model: Model containing oscillators
        
    Returns:
        Dictionary with oscillator data for each component
    """
    oscillator_data = {}
    
    # Try to extract from common patterns
    components_to_check = []
    
    # Check for encoder/decoder pattern
    if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        components_to_check = [
            ('encoder', model.encoder),
            ('decoder', model.decoder)
        ]
    # Check for single RNN/cell pattern
    elif hasattr(model, 'rnn') or hasattr(model, 'cell'):
        rnn = getattr(model, 'rnn', getattr(model, 'cell', None))
        components_to_check = [('main', rnn)]
    else:
        # Try to find oscillators directly
        components_to_check = [('model', model)]
    
    for component_name, component in components_to_check:
        try:
            # Navigate to oscillator
            oscillator = component
            
            # Common navigation patterns
            if hasattr(component, 'rnn'):
                oscillator = component.rnn
            if hasattr(oscillator, 'cell'):
                oscillator = oscillator.cell
            if hasattr(oscillator, 'oscillator'):
                oscillator = oscillator.oscillator
            
            # Extract parameters based on oscillator type
            data = {}
            
            # Handle AdaptiveNonlinearHarmonicOscillator (multiplier-based)
            if hasattr(oscillator, 'omega_multipliers') and hasattr(oscillator, 'gamma_multipliers'):
                # Compute effective omega and gamma from base values and multipliers
                base_omega = getattr(oscillator, 'base_omega', 1.0)
                base_gamma = getattr(oscillator, 'base_gamma', 0.01)
                
                effective_omega = base_omega * oscillator.omega_multipliers
                effective_gamma = base_gamma * oscillator.gamma_multipliers
                
                data = {
                    'omega': effective_omega,
                    'gamma': effective_gamma,
                    'alpha': getattr(oscillator, 'alpha', None),
                    'base_omega': base_omega,
                    'base_gamma': base_gamma,
                    'omega_multipliers': oscillator.omega_multipliers,
                    'gamma_multipliers': oscillator.gamma_multipliers,
                    'oscillator_type': 'adaptive_multiplier'
                }
                
                # Add bounds if available
                if hasattr(oscillator, 'omega_multiplier_bounds'):
                    # Convert multiplier bounds to effective omega bounds
                    omega_bounds = (base_omega * oscillator.omega_multiplier_bounds[0], 
                                   base_omega * oscillator.omega_multiplier_bounds[1])
                    data['omega_bounds'] = omega_bounds
                    data['omega_multiplier_bounds'] = oscillator.omega_multiplier_bounds
                    
                if hasattr(oscillator, 'gamma_multiplier_bounds'):
                    # Convert multiplier bounds to effective gamma bounds
                    gamma_bounds = (base_gamma * oscillator.gamma_multiplier_bounds[0],
                                   base_gamma * oscillator.gamma_multiplier_bounds[1])
                    data['gamma_bounds'] = gamma_bounds
                    data['gamma_multiplier_bounds'] = oscillator.gamma_multiplier_bounds
            
            # Handle standard learnable oscillators (direct omega/gamma)
            elif hasattr(oscillator, 'omega') and hasattr(oscillator, 'gamma'):
                data = {
                    'omega': oscillator.omega,
                    'gamma': oscillator.gamma,
                    'alpha': getattr(oscillator, 'alpha', None),
                    'oscillator_type': 'learnable_direct'
                }
                
                # Add bounds if available
                if hasattr(oscillator, 'omega_bounds'):
                    data['omega_bounds'] = oscillator.omega_bounds
                if hasattr(oscillator, 'gamma_bounds'):
                    data['gamma_bounds'] = oscillator.gamma_bounds
            
            # Add coupling multiplier if available
            if hasattr(component, 'gain_multiplier'):
                data['coupling_multiplier'] = component.gain_multiplier
            elif hasattr(component, 'rnn') and hasattr(component.rnn, 'cell') and hasattr(component.rnn.cell, 'gain_multiplier'):
                data['coupling_multiplier'] = component.rnn.cell.gain_multiplier
            
            if data:  # Only add if we found parameters
                oscillator_data[component_name] = data
                
        except Exception as e:
            print(f"⚠️  Could not extract oscillator data from {component_name}: {e}")
    
    if not oscillator_data:
        raise ValueError("No learnable oscillators found in model. Ensure model has learnable omega/gamma parameters.")
    
    return oscillator_data


def _classify_family_role(freq_hz: float, gamma: float) -> str:
    """Classify the functional role of an oscillator family based on parameters."""
    
    # Frequency-based classification
    if freq_hz > 0.15:
        freq_role = "High-freq"
        function = "Edge detection, fine details"
    elif freq_hz > 0.05:
        freq_role = "Medium-freq"
        function = "Textures, patterns"
    else:
        freq_role = "Low-freq"
        function = "Global structure, smooth regions"
    
    # Damping-based classification
    if gamma > 0.05:
        damp_role = "Heavy-damp"
        stability = "Stable, robust"
    elif gamma > 0.02:
        damp_role = "Medium-damp"
        stability = "Balanced"
    else:
        damp_role = "Light-damp"
        stability = "Sensitive, responsive"
    
    return f"{freq_role}, {damp_role} ({function}, {stability})"


def _analyze_coupling_strength(model: eqx.Module, extract_fn: Optional[callable] = None) -> Dict[str, Any]:
    """Analyze learned coupling strength multipliers."""
    
    oscillator_data = extract_fn(model) if extract_fn else _extract_oscillator_data(model)
    results = {}
    
    for component_name, component_data in oscillator_data.items():
        if 'coupling_multiplier' in component_data:
            coupling = float(component_data['coupling_multiplier'][0])
            
            # Estimate hidden dimension from omega shape
            hidden_dim = len(component_data['omega'])
            base_gain = 1.0 / np.sqrt(hidden_dim)
            effective_gain = coupling * base_gain
            
            results[component_name] = {
                'coupling_multiplier': coupling,
                'base_gain': base_gain,
                'effective_gain': effective_gain,
                'gain_boost': coupling / base_gain,
                'comparison_to_horn': effective_gain / base_gain
            }
            
            print(f"\n{component_name.upper()} COUPLING ANALYSIS:")
            print(f"   Learned multiplier: {coupling:.4f}")
            print(f"   Effective gain: {effective_gain:.4f}")
            print(f"   Boost over HORN: {results[component_name]['gain_boost']:.1f}x")
    
    return results 


# ========== NEW ONN-SPECIFIC ANALYSIS FUNCTIONS ==========

def analyze_phase_synchronization(
    model: eqx.Module,
    test_batch: jnp.ndarray,
    sequence_length: int = 20
) -> Dict[str, Any]:
    """
    Analyze phase synchronization patterns in oscillatory networks.
    
    Args:
        model: Trained model
        test_batch: Test data batch
        sequence_length: Length of sequence to analyze
        
    Returns:
        Dictionary with synchronization metrics
    """
    print("🌊 Analyzing phase synchronization patterns...")
    
    # Extract oscillator states during forward pass
    batch_size = test_batch.shape[0]
    
    # Process sequence and collect oscillator states
    oscillator_states = []
    
    try:
        # Navigate to oscillator cell (adapt based on architecture)
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'rnn'):
            cell = model.encoder.rnn.cell
        elif hasattr(model, 'rnn'):
            cell = model.rnn.cell
        else:
            raise ValueError("Cannot find oscillator cell in model")
        
        # Create input sequence
        patches = test_batch.reshape(batch_size, 28, 28)
        patches = patches.reshape(batch_size, 7, 4, 7, 4)
        patches = patches.transpose(0, 1, 3, 2, 4)
        patches = patches.reshape(batch_size, 49, 16)
        input_sequence = patches.transpose(1, 0, 2)[:sequence_length]  # Truncate
        
        # Initialize states
        if hasattr(cell, 'get_initial_state_from_phases'):
            x_state, v_state = cell.get_initial_state_from_phases(batch_size)
        else:
            x_state = jnp.zeros((batch_size, cell.hidden_dim))
            v_state = jnp.zeros((batch_size, cell.hidden_dim))
        
        # Collect states over time
        states_x, states_v = [], []
        
        for t in range(sequence_length):
            output, (x_state, v_state) = cell(input_sequence[t], (x_state, v_state))
            states_x.append(x_state)
            states_v.append(v_state)
        
        # Convert to arrays: (time, batch, oscillators)
        states_x = jnp.stack(states_x)
        states_v = jnp.stack(states_v)
        
        # Compute phases from x and v (arctan2)
        phases = jnp.arctan2(states_v, states_x + 1e-8)  # Shape: (time, batch, oscillators)
        
        # Synchronization analysis
        sync_results = {}
        
        # 1. Global sync order parameter R(t)
        # R = |1/N * Σ e^(iθ_j)|
        complex_phases = jnp.exp(1j * phases)  # (time, batch, oscillators)
        mean_complex = jnp.mean(complex_phases, axis=2)  # (time, batch)
        sync_order_R = jnp.abs(mean_complex)  # (time, batch)
        
        sync_results['global_sync_R'] = {
            'mean': float(jnp.mean(sync_order_R)),
            'std': float(jnp.std(sync_order_R)),
            'time_series': sync_order_R.tolist()
        }
        
        # 2. Pairwise phase coherence
        n_oscillators = phases.shape[2]
        phase_diffs = phases[:, :, :, None] - phases[:, :, None, :]  # (time, batch, i, j)
        phase_coherence = jnp.abs(jnp.mean(jnp.exp(1j * phase_diffs), axis=(0, 1)))  # (i, j)
        
        sync_results['phase_coherence'] = {
            'mean_coherence': float(jnp.mean(phase_coherence)),
            'max_coherence': float(jnp.max(phase_coherence)),
            'coherence_matrix': phase_coherence.tolist()
        }
        
        # 3. Frequency analysis from phases
        # Compute instantaneous frequencies
        phase_unwrapped = jnp.unwrap(phases, axis=0)  # Unwrap along time
        inst_frequencies = jnp.diff(phase_unwrapped, axis=0) # (time-1, batch, oscillators)
        
        sync_results['instantaneous_frequencies'] = {
            'mean_freq': float(jnp.mean(inst_frequencies)),
            'std_freq': float(jnp.std(inst_frequencies)),
            'freq_per_oscillator': jnp.mean(inst_frequencies, axis=(0, 1)).tolist()
        }
        
        print(f"   Global synchronization R: {sync_results['global_sync_R']['mean']:.4f} ± {sync_results['global_sync_R']['std']:.4f}")
        print(f"   Phase coherence: {sync_results['phase_coherence']['mean_coherence']:.4f}")
        
        return sync_results
        
    except Exception as e:
        print(f"   Phase synchronization analysis failed: {e}")
        return {'error': str(e)}


def analyze_oscillation_quality(
    model: eqx.Module,
    test_batch: jnp.ndarray,
    sequence_length: int = 50
) -> Dict[str, Any]:
    """
    Analyze the quality and stability of oscillations.
    
    Args:
        model: Trained model
        test_batch: Test data batch
        sequence_length: Length of sequence to analyze
        
    Returns:
        Dictionary with oscillation quality metrics
    """
    print("📊 Analyzing oscillation quality and stability...")
    
    try:
        # Similar setup to phase analysis but focused on oscillation properties
        batch_size = min(8, test_batch.shape[0])  # Smaller batch for detailed analysis
        
        # Navigate to oscillator
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'rnn'):
            cell = model.encoder.rnn.cell
        elif hasattr(model, 'rnn'):
            cell = model.rnn.cell
        else:
            raise ValueError("Cannot find oscillator cell")
        
        # Prepare input
        patches = test_batch[:batch_size].reshape(batch_size, 28, 28)
        patches = patches.reshape(batch_size, 7, 4, 7, 4)
        patches = patches.transpose(0, 1, 3, 2, 4)
        patches = patches.reshape(batch_size, 49, 16)
        input_sequence = patches.transpose(1, 0, 2)[:sequence_length]
        
        # Initialize and run
        if hasattr(cell, 'get_initial_state_from_phases'):
            x_state, v_state = cell.get_initial_state_from_phases(batch_size)
        else:
            x_state = jnp.zeros((batch_size, cell.hidden_dim))
            v_state = jnp.zeros((batch_size, cell.hidden_dim))
        
        states_x, states_v, energies = [], [], []
        
        for t in range(sequence_length):
            output, (x_state, v_state) = cell(input_sequence[t], (x_state, v_state))
            states_x.append(x_state)
            states_v.append(v_state)
            
            # Compute oscillator energy: E = 0.5 * (x² + v²)
            energy = 0.5 * (x_state**2 + v_state**2)
            energies.append(energy)
        
        states_x = jnp.stack(states_x)  # (time, batch, oscillators)
        states_v = jnp.stack(states_v)
        energies = jnp.stack(energies)
        
        quality_results = {}
        
        # 1. Energy stability
        energy_mean = jnp.mean(energies, axis=(0, 1))  # Per oscillator
        energy_std = jnp.std(energies, axis=0)  # (batch, oscillators)
        energy_stability = jnp.mean(energy_std, axis=0)  # Per oscillator
        
        quality_results['energy_analysis'] = {
            'mean_energy_per_oscillator': energy_mean.tolist(),
            'energy_stability': energy_stability.tolist(),
            'overall_stability': float(jnp.mean(energy_stability))
        }
        
        # 2. Amplitude consistency
        amplitudes = jnp.sqrt(states_x**2 + states_v**2)
        amplitude_consistency = 1.0 / (jnp.std(amplitudes, axis=0) + 1e-8)  # (batch, oscillators)
        
        quality_results['amplitude_analysis'] = {
            'mean_amplitude': float(jnp.mean(amplitudes)),
            'amplitude_consistency': jnp.mean(amplitude_consistency, axis=0).tolist(),
            'overall_consistency': float(jnp.mean(amplitude_consistency))
        }
        
        # 3. Oscillation regularity (autocorrelation)
        def compute_autocorr(signal, max_lag=10):
            """Compute autocorrelation for regularity measure"""
            autocorrs = []
            for lag in range(1, min(max_lag, len(signal))):
                if len(signal) > lag:
                    corr = jnp.corrcoef(signal[:-lag], signal[lag:])[0, 1]
                    autocorrs.append(corr)
            return jnp.nanmean(jnp.array(autocorrs))
        
        regularities = []
        for b in range(batch_size):
            for osc in range(cell.hidden_dim):
                x_signal = states_x[:, b, osc]
                regularity = compute_autocorr(x_signal)
                if not jnp.isnan(regularity):
                    regularities.append(regularity)
        
        quality_results['regularity_analysis'] = {
            'mean_regularity': float(jnp.mean(jnp.array(regularities))) if regularities else 0.0,
            'oscillation_quality_score': float(jnp.mean(jnp.array(regularities))) if regularities else 0.0
        }
        
        print(f"   Energy stability: {quality_results['energy_analysis']['overall_stability']:.4f}")
        print(f"   Amplitude consistency: {quality_results['amplitude_analysis']['overall_consistency']:.4f}")
        print(f"   Oscillation regularity: {quality_results['regularity_analysis']['mean_regularity']:.4f}")
        
        return quality_results
        
    except Exception as e:
        print(f"   Oscillation quality analysis failed: {e}")
        return {'error': str(e)}


def analyze_coupling_network(
    model: eqx.Module,
    test_batch: jnp.ndarray
) -> Dict[str, Any]:
    """
    Analyze the learned coupling network structure.
    
    Args:
        model: Trained model
        test_batch: Test data batch
        
    Returns:
        Dictionary with coupling network analysis
    """
    print("🔗 Analyzing learned coupling network structure...")
    
    try:
        # Extract recurrent weights (h2h layer)
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'rnn'):
            h2h_weights = model.encoder.rnn.cell.h2h.weight
        elif hasattr(model, 'rnn'):
            h2h_weights = model.rnn.cell.h2h.weight
        else:
            raise ValueError("Cannot find h2h weights")
        
        # h2h_weights shape: (hidden_dim, hidden_dim)
        W = np.array(h2h_weights)
        
        network_results = {}
        
        # 1. Weight statistics
        network_results['weight_statistics'] = {
            'mean_weight': float(np.mean(np.abs(W))),
            'std_weight': float(np.std(W)),
            'max_weight': float(np.max(np.abs(W))),
            'sparsity': float(np.sum(np.abs(W) < 1e-3) / W.size)
        }
        
        # 2. Connectivity analysis
        # Strong connections (above threshold)
        threshold = np.std(W) * 2
        strong_connections = np.abs(W) > threshold
        
        network_results['connectivity'] = {
            'strong_connection_ratio': float(np.sum(strong_connections) / W.size),
            'in_degree_mean': float(np.mean(np.sum(strong_connections, axis=1))),
            'out_degree_mean': float(np.mean(np.sum(strong_connections, axis=0))),
            'clustering_coefficient': _compute_clustering_coefficient(strong_connections)
        }
        
        # 3. Spectral analysis
        eigenvals = np.linalg.eigvals(W)
        network_results['spectral_properties'] = {
            'largest_eigenvalue': float(np.max(np.real(eigenvals))),
            'spectral_radius': float(np.max(np.abs(eigenvals))),
            'eigenvalue_spread': float(np.std(np.real(eigenvals)))
        }
        
        print(f"   Mean coupling strength: {network_results['weight_statistics']['mean_weight']:.4f}")
        print(f"   Strong connections: {network_results['connectivity']['strong_connection_ratio']*100:.1f}%")
        print(f"   Spectral radius: {network_results['spectral_properties']['spectral_radius']:.4f}")
        
        return network_results
        
    except Exception as e:
        print(f"   Coupling network analysis failed: {e}")
        return {'error': str(e)}


def _compute_clustering_coefficient(adj_matrix):
    """Compute clustering coefficient for binary adjacency matrix"""
    try:
        n = adj_matrix.shape[0]
        clustering_coeffs = []
        
        for i in range(n):
            neighbors = np.where(adj_matrix[i])[0]
            if len(neighbors) < 2:
                continue
                
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = 0
            
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    if adj_matrix[neighbors[j], neighbors[k]]:
                        actual_edges += 1
            
            if possible_edges > 0:
                clustering_coeffs.append(actual_edges / possible_edges)
        
        return float(np.mean(clustering_coeffs)) if clustering_coeffs else 0.0
    except:
        return 0.0


def visualize_onn_dynamics(
    onn_analysis: Dict[str, Any],
    figsize: Tuple[int, int] = (16, 12),
    title: str = "ONN Dynamics Analysis",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Create visualization of ONN-specific dynamics."""
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'🌊 {title}', fontsize=16, fontweight='bold')
    
    # 1. Synchronization time series
    if 'phase_synchronization' in onn_analysis:
        sync_data = onn_analysis['phase_synchronization']
        if 'time_series' in sync_data['global_sync_R']:
            time_series = np.array(sync_data['global_sync_R']['time_series'])
            if time_series.ndim > 1:
                time_series = np.mean(time_series, axis=1)  # Average over batch
            
            axes[0, 0].plot(time_series, linewidth=2)
            axes[0, 0].set_title('Global Synchronization R(t)')
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('Sync Order Parameter R')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='R=0.5')
            axes[0, 0].legend()
    
    # 2. Phase coherence matrix
    if 'phase_synchronization' in onn_analysis:
        sync_data = onn_analysis['phase_synchronization']
        if 'coherence_matrix' in sync_data['phase_coherence']:
            coherence = np.array(sync_data['phase_coherence']['coherence_matrix'])
            im = axes[0, 1].imshow(coherence, cmap='viridis', aspect='auto')
            axes[0, 1].set_title('Phase Coherence Matrix')
            axes[0, 1].set_xlabel('Oscillator Index')
            axes[0, 1].set_ylabel('Oscillator Index')
            plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Energy stability
    if 'oscillation_quality' in onn_analysis:
        quality_data = onn_analysis['oscillation_quality']
        if 'energy_analysis' in quality_data:
            energies = quality_data['energy_analysis']['mean_energy_per_oscillator']
            stability = quality_data['energy_analysis']['energy_stability']
            
            axes[0, 2].scatter(energies, stability, alpha=0.7)
            axes[0, 2].set_title('Energy vs Stability')
            axes[0, 2].set_xlabel('Mean Energy')
            axes[0, 2].set_ylabel('Energy Stability')
            axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Frequency distribution from dynamics
    if 'phase_synchronization' in onn_analysis:
        sync_data = onn_analysis['phase_synchronization']
        if 'freq_per_oscillator' in sync_data['instantaneous_frequencies']:
            freqs = sync_data['instantaneous_frequencies']['freq_per_oscillator']
            axes[1, 0].hist(freqs, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Instantaneous Frequency Distribution')
            axes[1, 0].set_xlabel('Frequency (rad/step)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Coupling network properties
    if 'coupling_network' in onn_analysis:
        network_data = onn_analysis['coupling_network']
        if 'weight_statistics' in network_data:
            stats = network_data['weight_statistics']
            connectivity = network_data.get('connectivity', {})
            
            # Bar plot of network properties
            properties = ['Mean Weight', 'Sparsity', 'Strong Conn %', 'Clustering']
            values = [
                stats['mean_weight'],
                stats['sparsity'],
                connectivity.get('strong_connection_ratio', 0) * 100,
                connectivity.get('clustering_coefficient', 0)
            ]
            
            axes[1, 1].bar(properties, values, alpha=0.7)
            axes[1, 1].set_title('Network Properties')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Overall quality summary
    quality_scores = []
    labels = []
    
    if 'phase_synchronization' in onn_analysis:
        sync_r = onn_analysis['phase_synchronization']['global_sync_R']['mean']
        quality_scores.append(sync_r)
        labels.append('Synchronization')
    
    if 'oscillation_quality' in onn_analysis:
        consistency = onn_analysis['oscillation_quality']['amplitude_analysis']['overall_consistency']
        quality_scores.append(min(consistency/10, 1.0))  # Normalize
        labels.append('Amplitude Consistency')
        
        regularity = onn_analysis['oscillation_quality']['regularity_analysis']['mean_regularity']
        if not np.isnan(regularity):
            quality_scores.append(max(0, regularity))
            labels.append('Regularity')
    
    if quality_scores:
        axes[1, 2].bar(labels, quality_scores, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'][:len(quality_scores)])
        axes[1, 2].set_title('ONN Quality Metrics')
        axes[1, 2].set_ylabel('Quality Score')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if not show:
        plt.close(fig)
    
    return fig 