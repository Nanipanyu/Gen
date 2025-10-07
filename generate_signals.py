"""
Generation Script for Signal Diffusion Transformer

This script generates broadband earthquake signals from low-frequency inputs
using the trained Signal DiT model and PGA predictor.

Author: Adapted for earthquake signal generation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

from signal_dit import SignalDiT
from signal_diff_utils import SignalDiffusion, SignalMetrics
from pga_predictor import PGAPredictor
from signal_datasets import EarthquakeSignalDataset
from signal_config import signal_config


def load_models(checkpoint_path, device='cpu'):
    """Load trained models from checkpoint"""
    print(f"Loading models from {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', signal_config)
    
    # Initialize Signal DiT
    signal_dit = SignalDiT(
        seq_len=config.get('seq_len', 6000),
        dim=config.get('dim', 300),
        patch_size=config.get('patch_size', 20),
        encoder_depth=config.get('encoder_depth', 4),
        decoder_depth=config.get('decoder_depth', 8),
        heads=config.get('heads', 10),
        mlp_dim=config.get('mlp_dim', 512),
        k=config.get('k', 64)
    ).to(device)
    
    # Load EMA weights for better quality
    signal_dit.load_state_dict(ckpt['ema'])
    signal_dit.eval()
    
    # Initialize diffusion
    diffusion = SignalDiffusion(
        P_mean=config.get('P_mean', -1.2),
        P_std=config.get('P_std', 1.2),
        sigma_data=config.get('sigma_data', 0.66)
    )
    
    # Initialize PGA predictor
    pga_predictor = PGAPredictor(
        seq_len=config.get('seq_len', 4096),
        cnn_filters=config.get('cnn_filters', [32, 64, 128, 256]),
        lstm_hidden=config.get('lstm_hidden', [512, 256]),
        dropout=config.get('pga_dropout', 0.1)
    ).to(device)
    
    # Load PGA predictor weights if available
    if 'pga_predictor' in ckpt:
        pga_predictor.load_state_dict(ckpt['pga_predictor'])
    pga_predictor.eval()
    
    print("Models loaded successfully!")
    return signal_dit, diffusion, pga_predictor, config

def generate_broadband_signals(signal_dit, diffusion, x_low, steps=100, seed=None):
    """
    Generate broadband signals conditioned on low-frequency inputs
    
    Args:
        signal_dit: Trained Signal DiT model
        diffusion: Diffusion process
        x_low: Low-frequency conditioning signals (batch, seq_len)
        steps: Number of denoising steps
        seed: Random seed for reproducible generation
    
    Returns:
        Generated broadband signals (batch, seq_len)
    """
    device = next(signal_dit.parameters()).device
    batch_size, seq_len = x_low.shape
    sz = (batch_size, seq_len)
    
    with torch.no_grad():
        generated_signals = diffusion.sample(
            signal_dit, sz, steps=steps, seed=seed, x_cond=x_low
        )
    
    return generated_signals.to(device)

def predict_pga_and_scale(pga_predictor, x_low, y_broad_normalized):
    """
    Predict PGA and scale the normalized broadband signal
    
    Args:
        pga_predictor: Trained PGA predictor
        x_low: Low-frequency signals (unnormalized)
        y_broad_normalized: Generated broadband signals (normalized)
    
    Returns:
        pga_predictions: Predicted PGA values
        y_broad_scaled: Scaled broadband signals
    """
    with torch.no_grad():
        pga_predictions = pga_predictor(x_low, y_broad_normalized)
        
        # Scale the normalized signals by predicted PGA
        y_broad_scaled = y_broad_normalized * pga_predictions.unsqueeze(-1)
    
    return pga_predictions, y_broad_scaled

def plot_generation_results(x_low, y_broad_norm, y_broad_scaled, pga_pred, 
                          save_path, sample_rate=100.0, n_samples=4):
    """Plot generation results for visualization"""
    fig, axes = plt.subplots(n_samples, 3, figsize=(20, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    time_axis = np.arange(x_low.shape[-1]) / sample_rate
    
    for i in range(min(n_samples, x_low.shape[0])):
        # Low-frequency input
        axes[i, 0].plot(time_axis, x_low[i].cpu().numpy(), 'b-', linewidth=0.8)
        axes[i, 0].set_title(f'Low-frequency Input {i+1}')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Acceleration')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Generated broadband (normalized)
        axes[i, 1].plot(time_axis, y_broad_norm[i].cpu().numpy(), 'r-', linewidth=0.8)
        axes[i, 1].set_title(f'Generated Broadband (Normalized) {i+1}')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('Acceleration (normalized)')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Scaled broadband with PGA
        axes[i, 2].plot(time_axis, y_broad_scaled[i].cpu().numpy(), 'g-', linewidth=0.8)
        axes[i, 2].set_title(f'Final Broadband (PGA={pga_pred[i]:.4f}) {i+1}')
        axes[i, 2].set_xlabel('Time (s)')
        axes[i, 2].set_ylabel('Acceleration')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results plotted and saved to {save_path}")

def plot_frequency_analysis(y_broad_scaled, sample_rate=100.0, save_path=None):
    """Plot frequency analysis of generated signals"""
    # Compute frequency content
    freqs, magnitude = SignalMetrics.compute_frequency_content(y_broad_scaled, sample_rate)
    
    # Focus on 0-30 Hz range
    freq_mask = (freqs >= 0) & (freqs <= 30)
    freqs_plot = freqs[freq_mask].cpu().numpy()
    magnitude_plot = magnitude[:, freq_mask].cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    
    # Plot individual spectra
    for i in range(min(4, magnitude_plot.shape[0])):
        plt.semilogy(freqs_plot, magnitude_plot[i], alpha=0.7, label=f'Signal {i+1}')
    
    # Plot mean spectrum
    mean_spectrum = np.mean(magnitude_plot, axis=0)
    plt.semilogy(freqs_plot, mean_spectrum, 'k-', linewidth=2, label='Mean Spectrum')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Content of Generated Broadband Signals (0-30 Hz)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 30)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Frequency analysis saved to {save_path}")
    
    plt.show()

def save_signals(signals, metadata, output_dir, prefix="generated"):
    """Save generated signals to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (signal, meta) in enumerate(zip(signals, metadata)):
        filename = os.path.join(output_dir, f"{prefix}_signal_{i:06d}.npz")
        
        np.savez(filename,
                signal=signal.cpu().numpy(),
                pga=meta['pga'],
                sample_rate=meta['sample_rate'],
                length_sec=len(signal) / meta['sample_rate'],
                generation_steps=meta.get('steps', 100),
                signal_id=i)
    
    print(f"Saved {len(signals)} signals to {output_dir}")

def generate_from_low_frequency_file(model_path, low_freq_file, output_dir, 
                                   steps=100, seed=42, device='cpu'):
    """Generate broadband signal from a single low-frequency file"""
    
    # Load models
    signal_dit, diffusion, pga_predictor, config = load_models(model_path, device)
    sample_rate = config.get('sample_rate', 100.0)
    
    # Load low-frequency signal
    print(f"Loading low-frequency signal from {low_freq_file}")
    data = np.load(low_freq_file)
    x_low_np = data['signal']
    
    # Ensure correct length
    seq_len = config.get('seq_len', 4096)
    if len(x_low_np) > seq_len:
        x_low_np = x_low_np[:seq_len]
    elif len(x_low_np) < seq_len:
        x_low_np = np.pad(x_low_np, (0, seq_len - len(x_low_np)), mode='constant')
    
    # Convert to tensor and add batch dimension
    x_low = torch.tensor(x_low_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Generate broadband signal
    print("Generating broadband signal...")
    y_broad_norm = generate_broadband_signals(
        signal_dit, diffusion, x_low, steps=steps, seed=seed
    )
    
    # Predict PGA and scale
    print("Predicting PGA and scaling...")
    pga_pred, y_broad_scaled = predict_pga_and_scale(
        pga_predictor, x_low, y_broad_norm
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot results
    plot_path = os.path.join(output_dir, 'generation_results.png')
    plot_generation_results(
        x_low, y_broad_norm, y_broad_scaled, pga_pred.squeeze(),
        plot_path, sample_rate, n_samples=1
    )
    
    # Plot frequency analysis
    freq_path = os.path.join(output_dir, 'frequency_analysis.png')
    plot_frequency_analysis(y_broad_scaled, sample_rate, freq_path)
    
    # Save signals
    metadata = [{
        'pga': pga_pred.squeeze().item(),
        'sample_rate': sample_rate,
        'steps': steps
    }]
    save_signals(y_broad_scaled, metadata, output_dir, "broadband")
    
    # Save low-frequency input for reference
    np.savez(os.path.join(output_dir, 'input_lowfreq.npz'),
            signal=x_low_np,
            sample_rate=sample_rate,
            length_sec=len(x_low_np) / sample_rate)
    
    print(f"Generation complete! Results saved to {output_dir}")
    print(f"Predicted PGA: {pga_pred.squeeze().item():.6f}")

def batch_generate_from_directory(model_path, input_dir, output_dir, 
                                batch_size=8, steps=100, seed=42, device='cpu'):
    """Generate broadband signals from a directory of low-frequency files"""
    
    # Load models
    signal_dit, diffusion, pga_predictor, config = load_models(model_path, device)
    sample_rate = config.get('sample_rate', 100.0)
    seq_len = config.get('seq_len', 4096)
    
    # Find all signal files
    signal_files = []
    for ext in ['.npz', '.npy']:
        signal_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
    
    print(f"Found {len(signal_files)} signal files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in batches
    n_batches = (len(signal_files) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(signal_files))
        batch_files = signal_files[start_idx:end_idx]
        
        # Load batch of low-frequency signals
        x_low_batch = []
        for file_path in batch_files:
            data = np.load(file_path)
            signal = data['signal']
            
            # Ensure correct length
            if len(signal) > seq_len:
                signal = signal[:seq_len]
            elif len(signal) < seq_len:
                signal = np.pad(signal, (0, seq_len - len(signal)), mode='constant')
            
            x_low_batch.append(signal)
        
        x_low_batch = torch.tensor(np.array(x_low_batch), dtype=torch.float32).to(device)
        
        # Generate broadband signals
        y_broad_norm = generate_broadband_signals(
            signal_dit, diffusion, x_low_batch, steps=steps, seed=seed
        )
        
        # Predict PGA and scale
        pga_pred, y_broad_scaled = predict_pga_and_scale(
            pga_predictor, x_low_batch, y_broad_norm
        )
        
        # Save results
        for i, (y_broad, pga) in enumerate(zip(y_broad_scaled, pga_pred)):
            file_idx = start_idx + i
            
            # Save broadband signal
            output_file = os.path.join(output_dir, f'broadband_{file_idx:06d}.npz')
            np.savez(output_file,
                    signal=y_broad.cpu().numpy(),
                    pga=pga.item(),
                    sample_rate=sample_rate,
                    length_sec=seq_len / sample_rate,
                    original_file=batch_files[i],
                    generation_steps=steps)
    
    print(f"Batch generation complete! {len(signal_files)} signals processed.")

def main():
    parser = argparse.ArgumentParser(description='Generate broadband signals using Signal DiT')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input low-frequency signal file or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for generated signals')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for generation')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing (when input is directory)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        print("Processing single file...")
        generate_from_low_frequency_file(
            args.model_path, args.input, args.output_dir,
            steps=args.steps, seed=args.seed, device=device
        )
    elif os.path.isdir(args.input):
        print("Processing directory...")
        batch_generate_from_directory(
            args.model_path, args.input, args.output_dir,
            batch_size=args.batch_size, steps=args.steps, 
            seed=args.seed, device=device
        )
    else:
        raise ValueError(f"Input path {args.input} is neither a file nor a directory")

if __name__ == '__main__':
    import glob
    main()
