"""
Deep evaluation and robustness testing for models
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    PCA = None
    NearestNeighbors = None
    SKLEARN_AVAILABLE = False


def comprehensive_model_evaluation(
    model,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    output_dir: Path,
    training_metrics: Optional[Dict] = None
):
    """
    Run comprehensive evaluation suite and generate summary report
    """
    print("🔬 Running comprehensive model evaluation...")
    
    # Create evaluation directory
    eval_dir = output_dir / "deep_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Noise Robustness Testing
    print("\n1️⃣  NOISE ROBUSTNESS ANALYSIS")
    noise_results = test_noise_robustness(model, test_images, eval_dir)
    results['noise_robustness'] = noise_results
    
    # 2. Latent Space Analysis
    print("\n2️⃣  LATENT SPACE ANALYSIS")
    latent_results = test_latent_space_properties(model, test_images, test_labels, eval_dir)
    results['latent_space'] = latent_results
    
    # 3. Computational Efficiency
    print("\n3️⃣  COMPUTATIONAL EFFICIENCY")
    efficiency_results = evaluate_computational_efficiency(model, test_images)
    results['efficiency'] = efficiency_results
    
    # 4. Oscillator Discovery Analysis (if available)
    print("\n4️⃣  OSCILLATOR DISCOVERY ANALYSIS")
    try:
        from ..visualization.oscillator_analysis import comprehensive_oscillator_analysis
        
        oscillator_dir = output_dir / "oscillator_analysis"
        oscillator_results = comprehensive_oscillator_analysis(
            model=model,
            output_dir=str(oscillator_dir),
            n_families=5,
            generate_visualizations=True,
            generate_report=True,
            test_data=test_images
        )
        results['oscillator_analysis'] = oscillator_results
        print("   ✅ Oscillator discovery analysis completed!")
        
    except Exception as e:
        print(f"   ⚠️  Oscillator analysis skipped: {e}")
        results['oscillator_analysis'] = None
    
    # 5. Generate comprehensive report
    print("\n5️⃣  GENERATING COMPREHENSIVE REPORT")
    overall_score = create_evaluation_report(results, training_metrics, eval_dir)
    results['overall_score'] = overall_score
    
    print("✅ Comprehensive evaluation completed!")
    return results


def test_noise_robustness(
    model,
    test_images: jnp.ndarray,
    output_dir: Path
):
    """Test model robustness to various types of noise"""
    print("🔊 Testing noise robustness...")
    
    n_samples = min(100, len(test_images))
    clean_images = test_images[:n_samples]
    
    noise_types = ['gaussian', 'salt_pepper', 'uniform', 'dropout']
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = {}
    
    for noise_type in noise_types:
        results[noise_type] = []
        
        for noise_level in noise_levels:
            # Add noise
            if noise_type == 'gaussian':
                key = jax.random.PRNGKey(42)
                noise = jax.random.normal(key, clean_images.shape) * noise_level
                noisy_images = jnp.clip(clean_images + noise, 0, 1)
            
            elif noise_type == 'salt_pepper':
                key = jax.random.PRNGKey(42)
                mask = jax.random.uniform(key, clean_images.shape) < noise_level
                noisy_images = jnp.where(mask, 
                                       jax.random.uniform(key, clean_images.shape) > 0.5,
                                       clean_images)
            
            elif noise_type == 'uniform':
                key = jax.random.PRNGKey(42)
                noise = (jax.random.uniform(key, clean_images.shape) - 0.5) * noise_level
                noisy_images = jnp.clip(clean_images + noise, 0, 1)
            
            elif noise_type == 'dropout':
                key = jax.random.PRNGKey(42)
                mask = jax.random.uniform(key, clean_images.shape) > noise_level
                noisy_images = clean_images * mask
            
            # Test reconstruction quality
            clean_recons = model(clean_images)
            noisy_recons = model(noisy_images)
            
            # Metrics
            clean_mse = float(jnp.mean((clean_images - clean_recons) ** 2))
            noisy_mse = float(jnp.mean((clean_images - noisy_recons) ** 2))
            degradation = (noisy_mse - clean_mse) / clean_mse
            
            results[noise_type].append({
                'noise_level': noise_level,
                'clean_mse': clean_mse,
                'noisy_mse': noisy_mse,
                'degradation': degradation
            })
    
    # Visualize robustness
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, noise_type in enumerate(noise_types):
        noise_levels_plot = [r['noise_level'] for r in results[noise_type]]
        degradations = [r['degradation'] for r in results[noise_type]]
        
        axes[i].plot(noise_levels_plot, degradations, 'o-', linewidth=2, markersize=6)
        axes[i].set_title(f'{noise_type.title()} Noise Robustness')
        axes[i].set_xlabel('Noise Level')
        axes[i].set_ylabel('Reconstruction Degradation')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "noise_robustness_analysis.png", dpi=150)
    plt.close(fig)
    
    # Save detailed results
    with open(output_dir / "noise_robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("   ✅ Noise robustness analysis completed!")
    return results


def test_latent_space_properties(
    model,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    output_dir: Path
):
    """Deep analysis of latent space properties"""
    print("🧠 Analyzing latent space properties...")
    
    n_samples = min(1000, len(test_images))
    sample_images = test_images[:n_samples]
    sample_labels = test_labels[:n_samples]
    
    # Get latent representations
    latent_codes = []
    for i in range(0, n_samples, 32):
        batch_end = min(i + 32, n_samples)
        batch = sample_images[i:batch_end]
        
        # Encode to latent space
        batch_size = batch.shape[0]
        patches = batch.reshape(batch_size, 28, 28)
        patches = patches.reshape(batch_size, 7, 4, 7, 4)
        patches = patches.transpose(0, 1, 3, 2, 4)
        patches = patches.reshape(batch_size, 49, 16)
        patch_sequence = patches.transpose(1, 0, 2)
        latent_batch = model.encoder(patch_sequence)
        latent_codes.append(latent_batch)
    
    all_latents = jnp.concatenate(latent_codes, axis=0)
    
    properties = {}
    
    # 1. Intrinsic dimensionality (if sklearn available)
    if SKLEARN_AVAILABLE:
        pca = PCA()
        pca.fit(np.array(all_latents))
        
        # Calculate effective dimensionality (95% variance explained)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = np.argmax(cumvar >= 0.95) + 1
        properties['effective_dimensionality'] = int(effective_dim)
        properties['total_dimensionality'] = all_latents.shape[1]
        
        # 2. Latent space smoothness (local neighborhood analysis)
        nbrs = NearestNeighbors(n_neighbors=5).fit(np.array(all_latents))
        distances, indices = nbrs.kneighbors(np.array(all_latents))
        
        # Check label consistency in neighborhoods
        label_consistency = []
        for i in range(len(sample_labels)):
            neighbor_labels = sample_labels[indices[i]]
            consistency = np.mean(neighbor_labels == sample_labels[i])
            label_consistency.append(consistency)
        
        properties['latent_smoothness'] = float(np.mean(jnp.array(label_consistency)))
    else:
        # Fallback analysis without sklearn
        properties['effective_dimensionality'] = all_latents.shape[1]
        properties['total_dimensionality'] = all_latents.shape[1]
        properties['latent_smoothness'] = 0.5  # Placeholder
    
    # 3. Interpolation quality
    interpolation_quality = test_interpolation_quality(model, sample_images, sample_labels)
    properties['interpolation_quality'] = interpolation_quality
    
    # 4. Latent utilization
    latent_std = float(jnp.mean(jnp.std(all_latents, axis=0)))
    properties['latent_utilization'] = latent_std
    
    # Visualize properties
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if SKLEARN_AVAILABLE:
        # PCA variance explained
        axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                        np.cumsum(pca.explained_variance_ratio_), 'o-')
        axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        axes[0, 0].axvline(x=effective_dim, color='r', linestyle='--', label=f'Effective dim: {effective_dim}')
        axes[0, 0].set_xlabel('Principal Components')
        axes[0, 0].set_ylabel('Cumulative Variance Explained')
        axes[0, 0].set_title('Latent Space Dimensionality')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Label consistency histogram
        axes[0, 1].hist(label_consistency, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Neighborhood Label Consistency')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Latent Smoothness (Mean: {properties["latent_smoothness"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distance distribution
        axes[1, 0].hist(distances[:, 1:].flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Neighbor Distance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Latent Space Distance Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        # Simplified visualizations without sklearn
        axes[0, 0].text(0.5, 0.5, 'PCA analysis\nrequires sklearn', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, 'Neighborhood analysis\nrequires sklearn', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[1, 0].text(0.5, 0.5, 'Distance analysis\nrequires sklearn', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Properties summary
    axes[1, 1].text(0.1, 0.7, f"Effective Dimensionality: {properties['effective_dimensionality']}/{properties['total_dimensionality']}", 
                    fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f"Latent Smoothness: {properties['latent_smoothness']:.3f}", 
                    fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.3, f"Interpolation Quality: {interpolation_quality:.3f}", 
                    fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Latent Space Properties Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "latent_space_properties.png", dpi=150)
    plt.close(fig)
    
    # Save results
    with open(output_dir / "latent_space_properties.json", "w") as f:
        json.dump(properties, f, indent=2)
    
    print("   ✅ Latent space properties analysis completed!")
    return properties


def test_interpolation_quality(model, test_images, test_labels):
    """Test quality of latent space interpolations"""
    
    # Select pairs of same and different digits
    same_digit_pairs = []
    diff_digit_pairs = []
    
    for i in range(10):
        # Same digit pairs
        digit_indices = jnp.where(test_labels == i)[0]
        if len(digit_indices) >= 2:
            same_digit_pairs.append((digit_indices[0], digit_indices[1]))
        
        # Different digit pairs
        if i < 9:
            other_digit_indices = jnp.where(test_labels == i+1)[0]
            if len(other_digit_indices) > 0:
                diff_digit_pairs.append((digit_indices[0], other_digit_indices[0]))
    
    # Test interpolation smoothness
    same_smoothness = []
    diff_smoothness = []
    
    for pair in same_digit_pairs[:5]:  # Test first 5 pairs
        smoothness = compute_interpolation_smoothness(model, test_images[pair[0]], test_images[pair[1]])
        same_smoothness.append(smoothness)
    
    for pair in diff_digit_pairs[:5]:
        smoothness = compute_interpolation_smoothness(model, test_images[pair[0]], test_images[pair[1]])
        diff_smoothness.append(smoothness)
    
    # Overall quality score (higher same-digit smoothness is better)
    if len(same_smoothness) > 0 and len(diff_smoothness) > 0:
        quality_score = float(jnp.mean(jnp.array(same_smoothness)) / (jnp.mean(jnp.array(diff_smoothness)) + 1e-8))
    else:
        quality_score = 1.0
    
    return quality_score


def compute_interpolation_smoothness(model, img1, img2, n_steps=10):
    """Compute smoothness of interpolation path"""
    
    # Get latent codes
    def encode_image(img):
        patches = img.reshape(28, 28)
        patches = patches.reshape(7, 4, 7, 4)
        patches = patches.transpose(0, 2, 1, 3)
        patches = patches.reshape(49, 16)
        patch_sequence = patches[None, :, :].transpose(1, 0, 2)
        return model.encoder(patch_sequence)[0]
    
    z1 = encode_image(img1)
    z2 = encode_image(img2)
    
    # Generate interpolation path
    alphas = jnp.linspace(0, 1, n_steps)
    reconstructions = []
    
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # Decode
        decoded_patches = model.decoder(z_interp[None, :])
        decoded_patches = decoded_patches.transpose(1, 0, 2)
        reconstructed = decoded_patches.reshape(1, 7, 7, 4, 4)
        reconstructed = reconstructed.transpose(0, 1, 3, 2, 4)
        reconstructed = reconstructed.reshape(28, 28)
        
        reconstructions.append(reconstructed)
    
    # Compute smoothness (lower variation is smoother)
    variations = []
    for i in range(len(reconstructions) - 1):
        diff = jnp.mean((reconstructions[i+1] - reconstructions[i]) ** 2)
        variations.append(float(diff))
    
    # Return inverse of variation (higher = smoother)
    return 1.0 / (jnp.mean(jnp.array(variations)) + 1e-8)


def evaluate_computational_efficiency(model, test_images: jnp.ndarray):
    """Evaluate computational efficiency"""
    print("⚡ Evaluating computational efficiency...")
    
    # Warm up
    _ = model(test_images[:1])
    
    # Time forward pass
    batch_sizes = [1, 8, 32]
    timing_results = {}
    
    for batch_size in batch_sizes:
        batch = test_images[:batch_size]
        
        # Time multiple runs
        times = []
        for _ in range(10):
            start_time = time.time()
            _ = model(batch)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = float(jnp.mean(jnp.array(times)))
        timing_results[f'time_batch_{batch_size}'] = avg_time
        timing_results[f'time_per_sample_batch_{batch_size}'] = avg_time / batch_size
    
    print("   ✅ Computational efficiency analysis completed!")
    return timing_results


def create_evaluation_report(
    results: Dict, 
    training_metrics: Optional[Dict],
    output_dir: Path
):
    """Create comprehensive evaluation report"""
    print("📋 Generating evaluation report...")
    
    # Extract key metrics
    noise_results = results.get('noise_robustness', {})
    latent_results = results.get('latent_space', {})
    efficiency_results = results.get('efficiency', {})
    oscillator_results = results.get('oscillator_analysis', {})
    
    # Calculate summary metrics
    summary_report = {
        'model_architecture': 'Hybrid Patch-Based Autoencoder',
        'noise_robustness': {},
        'latent_space_quality': latent_results,
        'computational_efficiency': efficiency_results
    }
    
    # Add oscillator discovery insights if available
    if oscillator_results and oscillator_results.get('summary', {}).get('analysis_complete'):
        oscillator_summary = oscillator_results['summary']
        family_data = oscillator_results.get('frequency_families', {})
        
        summary_report['oscillator_discoveries'] = {
            'total_oscillators': oscillator_summary.get('total_oscillators', 0),
            'families_discovered': oscillator_summary.get('families_discovered', 0),
            'specialization_emerged': True,
            'family_insights': {}
        }
        
        # Extract family insights
        for component_name, component_data in family_data.items():
            families = component_data.get('families', {})
            family_roles = []
            for family_name, family_info in families.items():
                family_roles.append({
                    'specialization': family_info.get('specialization', 'Unknown'),
                    'population': family_info.get('count', 0),
                    'frequency_mean': family_info.get('freq_mean', 0.0)
                })
            
            summary_report['oscillator_discoveries']['family_insights'][component_name] = family_roles
        
        print(f"   🎵 Oscillator insights: {oscillator_summary.get('families_discovered', 0)} frequency families discovered!")
    
    # Noise robustness summary
    for noise_type in ['gaussian', 'salt_pepper', 'uniform', 'dropout']:
        if noise_type in noise_results:
            degradations = [r['degradation'] for r in noise_results[noise_type]]
            summary_report['noise_robustness'][f'{noise_type}_degradation_mean'] = float(jnp.mean(jnp.array(degradations)))
    
    # Add training metrics if available
    if training_metrics:
        summary_report['training_config'] = {
            'final_loss': float(training_metrics['train_loss'][-1]) if training_metrics.get('train_loss') else None,
            'epochs_trained': len(training_metrics['train_loss']) if training_metrics.get('train_loss') else None
        }
    
    # Calculate overall quality score
    quality_metrics = {
        'reconstruction_quality': 1.0 / (summary_report['training_config'].get('final_loss', 0.1) + 1e-8) if training_metrics else 50.0,
        'latent_utilization': latent_results.get('latent_utilization', 0.0),
        'interpolation_quality': latent_results.get('interpolation_quality', 0.0),
        'noise_robustness': 1.0 / (summary_report['noise_robustness'].get('gaussian_degradation_mean', 1.0) + 1.0),
        'dimensionality_efficiency': latent_results.get('effective_dimensionality', 32) / latent_results.get('total_dimensionality', 32)
    }
    
    # Add oscillator specialization bonus
    if oscillator_results and oscillator_results.get('summary', {}).get('analysis_complete'):
        families_discovered = oscillator_results['summary'].get('families_discovered', 0)
        quality_metrics['oscillator_specialization'] = min(families_discovered / 5.0, 1.0)  # Bonus for discovering families
    
    # Weighted overall score
    weights = {'reconstruction_quality': 0.25, 'latent_utilization': 0.15, 'interpolation_quality': 0.15, 
              'noise_robustness': 0.15, 'dimensionality_efficiency': 0.1, 'oscillator_specialization': 0.2}
    
    # Adjust weights if oscillator analysis not available
    if 'oscillator_specialization' not in quality_metrics:
        weights = {'reconstruction_quality': 0.3, 'latent_utilization': 0.2, 'interpolation_quality': 0.2, 
                  'noise_robustness': 0.2, 'dimensionality_efficiency': 0.1}
        quality_metrics['oscillator_specialization'] = 0.0
    
    overall_score = sum(quality_metrics[metric] * weights[metric] for metric in quality_metrics if metric in weights)
    summary_report['overall_quality_score'] = float(overall_score)
    summary_report['quality_breakdown'] = quality_metrics
    
    # Save comprehensive summary
    with open(output_dir / "comprehensive_evaluation_summary.json", "w") as f:
        json.dump(summary_report, f, indent=2)
    
    print("   ✅ Evaluation report generated!")
    return overall_score 
