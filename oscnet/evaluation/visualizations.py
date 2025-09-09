"""
Enhanced visualization functions for model evaluation
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional

# Import existing visualization functions
from oscnet.visualization.model_insights import visualize_latent_space


def visualize_latent_interpolation(
    model,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    output_dir: Path,
    n_interpolations: int = 10
):
    """
    Create latent space interpolations between different digit pairs.
    """
    print("🎨 Creating latent space interpolations...")
    
    # Find examples of different digits
    digit_examples = {}
    for label in range(10):
        indices = jnp.where(test_labels == label)[0]
        if len(indices) > 0:
            digit_examples[label] = test_images[indices[0]]
    
    # Select interesting pairs to interpolate between
    pairs = [(1, 7), (3, 8), (4, 9), (0, 6), (2, 5)]
    
    for pair in pairs:
        if pair[0] in digit_examples and pair[1] in digit_examples:
            source = digit_examples[pair[0]]
            target = digit_examples[pair[1]]
            
            # Get latent representations
            source_batch = source[None, :]  # Add batch dimension
            target_batch = target[None, :]
            
            # Convert to patches and encode
            source_patches = source_batch.reshape(1, 28, 28)
            source_patches = source_patches.reshape(1, 7, 4, 7, 4)
            source_patches = source_patches.transpose(0, 1, 3, 2, 4)
            source_patches = source_patches.reshape(1, 49, 16)
            source_sequence = source_patches.transpose(1, 0, 2)
            z1 = model.encoder(source_sequence)  # (1, latent_dim)
            
            target_patches = target_batch.reshape(1, 28, 28)
            target_patches = target_patches.reshape(1, 7, 4, 7, 4)
            target_patches = target_patches.transpose(0, 1, 3, 2, 4)
            target_patches = target_patches.reshape(1, 49, 16)
            target_sequence = target_patches.transpose(1, 0, 2)
            z2 = model.encoder(target_sequence)  # (1, latent_dim)
            
            # Create interpolation
            fig, axes = plt.subplots(2, n_interpolations + 2, figsize=(18, 4))
            
            for i, alpha in enumerate(jnp.linspace(0, 1, n_interpolations)):
                # Interpolate in latent space
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Decode interpolated latent
                decoded_patches = model.decoder(z_interp)  # (49, 1, 16)
                decoded_patches = decoded_patches.transpose(1, 0, 2)  # (1, 49, 16)
                reconstructed = decoded_patches.reshape(1, 7, 7, 4, 4)
                reconstructed = reconstructed.transpose(0, 1, 3, 2, 4)
                reconstructed = reconstructed.reshape(1, 28, 28)
                
                # Plot original images on top row (only at ends)
                if i == 0:
                    axes[0, i].imshow(source.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
                    axes[0, i].set_title(f"Digit {pair[0]}")
                elif i == n_interpolations - 1:
                    axes[0, i+2].imshow(target.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
                    axes[0, i+2].set_title(f"Digit {pair[1]}")
                
                # Plot interpolated image on bottom row
                col_idx = i + 1  # Offset by 1 to leave space for source
                axes[1, col_idx].imshow(reconstructed[0], cmap='gray', vmin=0, vmax=1)
                axes[1, col_idx].set_title(f"α={alpha:.1f}")
                axes[1, col_idx].axis('off')
            
            # Hide unused axes
            for i in range(n_interpolations + 2):
                axes[0, i].axis('off')
                if i == 0 or i == n_interpolations + 1:
                    axes[1, i].axis('off')
            
            plt.suptitle(f'Hybrid Latent Space Interpolation: {pair[0]} → {pair[1]}', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / f"latent_interpolation_{pair[0]}_to_{pair[1]}.png", dpi=150)
            plt.close(fig)
    
    print("   ✅ Latent interpolations saved!")


def visualize_hybrid_latent_space(
    model,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    output_dir: Path,
    n_samples: int = 1000
):
    """
    Visualize the latent space representation with PCA if needed.
    Shows how different digits cluster in the hybrid latent space.
    """
    print("🌌 Visualizing hybrid latent space...")
    
    # Limit samples for visualization
    n_samples = min(n_samples, len(test_images))
    sample_images = test_images[:n_samples]
    sample_labels = test_labels[:n_samples]
    
    # Get latent representations
    latent_codes = []
    for i in range(0, n_samples, 32):  # Process in batches
        batch_end = min(i + 32, n_samples)
        batch = sample_images[i:batch_end]
        
        # Convert to patch format
        batch_size = batch.shape[0]
        patches = batch.reshape(batch_size, 28, 28)
        patches = patches.reshape(batch_size, 7, 4, 7, 4)
        patches = patches.transpose(0, 1, 3, 2, 4)
        patches = patches.reshape(batch_size, 49, 16)
        patch_sequence = patches.transpose(1, 0, 2)
        
        # Encode to latent
        latent_batch = model.encoder(patch_sequence)  # (batch_size, latent_dim)
        latent_codes.append(latent_batch)
    
    # Concatenate all latent codes
    all_latents = jnp.concatenate(latent_codes, axis=0)  # (n_samples, latent_dim)
    
    # Create latent space visualization
    fig = visualize_latent_space(
        latent=all_latents,
        labels=sample_labels,
        title="Hybrid Autoencoder Latent Space",
        figsize=(12, 10)
    )
    
    plt.savefig(output_dir / "hybrid_latent_space.png", dpi=150)
    plt.close(fig)
    
    print("   ✅ Latent space visualization saved!")


def analyze_hybrid_phase_amplitude(
    model,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    output_dir: Path,
    n_samples: int = 200
):
    """
    Analyze the relationship between amplitude (x) and frequency (v) in hybrid model.
    """
    print("🔬 Analyzing hybrid phase-amplitude relationships...")
    
    # Get a sample of different digits
    sample_images = []
    sample_labels_list = []
    
    for digit in range(10):
        digit_indices = jnp.where(test_labels == digit)[0]
        if len(digit_indices) > 0:
            # Take multiple samples per digit
            n_per_digit = min(n_samples // 10, len(digit_indices))
            selected_indices = digit_indices[:n_per_digit]
            sample_images.extend([test_images[i] for i in selected_indices])
            sample_labels_list.extend([digit] * n_per_digit)
    
    sample_images = jnp.array(sample_images)
    sample_labels = jnp.array(sample_labels_list)
    
    # Get latent representations
    latent_codes = []
    for i in range(len(sample_images)):
        image = sample_images[i]
        
        # Convert to patch format
        patches = image.reshape(28, 28)
        patches = patches.reshape(7, 4, 7, 4)
        patches = patches.transpose(0, 2, 1, 3)
        patches = patches.reshape(49, 16)
        patch_sequence = patches[None, :, :].transpose(1, 0, 2)
        
        # Get latent representation
        latent = model.encoder(patch_sequence)  # (1, latent_dim)
        latent_codes.append(latent[0])
    
    latent_codes = jnp.array(latent_codes)  # (n_samples, latent_dim)
    
    # Analyze latent space structure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Latent magnitudes by digit
    axes[0, 0].scatter(jnp.arange(len(latent_codes)), jnp.linalg.norm(latent_codes, axis=1), 
                      c=sample_labels, cmap='tab10', alpha=0.7)
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Latent Magnitude')
    axes[0, 0].set_title('Latent Representation Magnitudes')
    
    # Plot 2: First two latent dimensions
    scatter = axes[0, 1].scatter(latent_codes[:, 0], latent_codes[:, 1], 
                                c=sample_labels, cmap='tab10', alpha=0.7)
    axes[0, 1].set_xlabel('Latent Dimension 1')
    axes[0, 1].set_ylabel('Latent Dimension 2')
    axes[0, 1].set_title('Latent Space (First 2 Dims)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Digit')
    
    # Plot 3: Latent variance by dimension
    latent_var = jnp.var(latent_codes, axis=0)
    axes[1, 0].bar(range(len(latent_var)), latent_var)
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title('Latent Dimension Variances')
    
    # Plot 4: Mean latent by digit
    for digit in range(10):
        digit_mask = sample_labels == digit
        if jnp.sum(digit_mask) > 0:
            mean_latent = jnp.mean(latent_codes[digit_mask], axis=0)
            axes[1, 1].plot(mean_latent, label=f'Digit {digit}', alpha=0.7)
    
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Mean Activation')
    axes[1, 1].set_title('Mean Latent Patterns by Digit')
    axes[1, 1].legend()
    
    plt.suptitle('Hybrid Model Phase-Amplitude Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "hybrid_phase_amplitude_analysis.png", dpi=150)
    plt.close(fig)
    
    print("   ✅ Phase-amplitude analysis saved!")


def visualize_reconstruction_quality_by_digit(
    model,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    output_dir: Path
):
    """
    Analyze reconstruction quality for each digit class.
    """
    print("📊 Analyzing reconstruction quality by digit...")
    
    digit_losses = []
    digit_examples = []
    
    for digit in range(10):
        digit_indices = jnp.where(test_labels == digit)[0]
        if len(digit_indices) > 0:
            # Take first 100 examples of this digit
            n_examples = min(100, len(digit_indices))
            digit_images = test_images[digit_indices[:n_examples]]
            
            # Compute reconstruction loss for this digit
            reconstructions = model(digit_images)
            losses = jnp.mean((digit_images - reconstructions) ** 2, axis=1)
            mean_loss = jnp.mean(losses)
            
            digit_losses.append(float(mean_loss))
            
            # Store best and worst examples
            best_idx = jnp.argmin(losses)
            worst_idx = jnp.argmax(losses)
            
            digit_examples.append({
                'digit': digit,
                'best_original': digit_images[best_idx],
                'best_reconstruction': reconstructions[best_idx],
                'best_loss': float(losses[best_idx]),
                'worst_original': digit_images[worst_idx],
                'worst_reconstruction': reconstructions[worst_idx],
                'worst_loss': float(losses[worst_idx])
            })
    
    # Plot reconstruction quality by digit
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    
    # Top row: Mean loss by digit
    axes[0, 4].bar(range(10), digit_losses)
    axes[0, 4].set_xlabel('Digit')
    axes[0, 4].set_ylabel('Mean Reconstruction Loss')
    axes[0, 4].set_title('Reconstruction Quality by Digit')
    for i in range(10):
        if i != 4:
            axes[0, i].axis('off')
    
    # Middle and bottom rows: Best and worst examples
    for i, example in enumerate(digit_examples):
        # Best example
        axes[1, i].imshow(example['best_original'].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f"Best {example['digit']}\nLoss: {example['best_loss']:.4f}")
        axes[1, i].axis('off')
        
        # Worst example
        axes[2, i].imshow(example['worst_original'].reshape(28, 28), cmap='gray')
        axes[2, i].set_title(f"Worst {example['digit']}\nLoss: {example['worst_loss']:.4f}")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_quality_by_digit.png", dpi=150)
    plt.close(fig)
    
    print("   ✅ Reconstruction quality analysis saved!")


def create_enhanced_visualizations(
    model,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    output_dir: Path,
    quick_mode: bool = False
):
    """
    Create all enhanced visualizations in one call
    """
    print("\n🎨 Creating enhanced visualizations...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Adjust parameters for quick mode
    n_interpolations = 6 if quick_mode else 8
    n_samples = 500 if quick_mode else 1000
    n_phase_samples = 100 if quick_mode else 200
    
    try:
        # 1. Latent Space Interpolation
        visualize_latent_interpolation(
            model=model,
            test_images=test_images,
            test_labels=test_labels,
            output_dir=output_dir,
            n_interpolations=n_interpolations
        )
        
        # 2. Latent Space Visualization
        visualize_hybrid_latent_space(
            model=model,
            test_images=test_images,
            test_labels=test_labels,
            output_dir=output_dir,
            n_samples=n_samples
        )
        
        # 3. Phase-Amplitude Analysis
        analyze_hybrid_phase_amplitude(
            model=model,
            test_images=test_images,
            test_labels=test_labels,
            output_dir=output_dir,
            n_samples=n_phase_samples
        )
        
        # 4. Reconstruction Quality Analysis
        visualize_reconstruction_quality_by_digit(
            model=model,
            test_images=test_images,
            test_labels=test_labels,
            output_dir=output_dir
        )
        
        print("✨ All enhanced visualizations completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating enhanced visualizations: {e}")
        return False 