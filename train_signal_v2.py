"""
Training Script for Signal Diffusion Transformer V2 - Simplified

New architecture with:
- RoPE and packed attention for variable-length sequences
- Spectral ControlNet for frequency-domain conditioning  
- Standard Gaussian diffusion (NO log-space)
- NO normalization (works with raw signals)
- NO metadata conditioning
"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

from signal_dit_v2 import SignalDiTV2
from signal_datasets_v2 import EarthquakeSignalDatasetV2, create_dataloader_v2
from signal_config_v2 import CONFIG_V2, get_config
import glob

warnings.filterwarnings("ignore")


def get_noise_schedule(num_steps=1000, beta_start=0.0001, beta_end=0.02):
    """
    Create linear noise schedule for standard DDPM diffusion.
    
    Args:
        num_steps: Number of diffusion timesteps
        beta_start: Starting noise level
        beta_end: Ending noise level
    
    Returns:
        dict with alpha, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod, SNR
    """
    betas = torch.linspace(beta_start, beta_end, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # Compute SNR (Signal-to-Noise Ratio) for loss weighting
    # SNR = alpha_cumprod / (1 - alpha_cumprod)
    snr = alphas_cumprod / (1.0 - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'snr': snr,
    }


def compute_dataset_statistics(signal_paths, max_samples=100):
    """
    Compute mean and std of signals AND envelopes for proper normalization.
    
    Args:
        signal_paths: List of paths to signal files
        max_samples: Maximum number of samples to use for statistics
    
    Returns:
        dict with signal_mean, signal_std, envelope_mean, envelope_std
    """
    print("📊 Computing dataset statistics for normalization...")
    from scipy.signal import hilbert
    
    signal_values = []
    envelope_values = []
    
    for i, path in enumerate(signal_paths[:max_samples]):
        try:
            data = np.load(path)
            # Get signal
            if 'signal_broadband' in data:
                signal = data['signal_broadband']
                pga = data.get('pga_broadband', np.max(np.abs(signal)))
                signal_denorm = signal * pga
            elif 'signal_normalized' in data:
                signal_denorm = data['signal_normalized']
            else:
                continue
            
            # Compute envelope
            try:
                analytic = hilbert(signal_denorm)
                envelope = np.abs(analytic)
            except:
                envelope = np.abs(signal_denorm)
            
            signal_values.append(signal_denorm)
            envelope_values.append(envelope)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if not signal_values:
        print("Warning: No valid signals found, using default statistics")
        return {'signal_mean': 0.0, 'signal_std': 1.0, 'envelope_mean': 0.0, 'envelope_std': 1.0}
    
    # Signal statistics
    all_signals = np.concatenate(signal_values)
    signal_mean = float(np.mean(all_signals))
    signal_std = float(np.std(all_signals))
    
    # Envelope statistics (always positive, different distribution!)
    all_envelopes = np.concatenate(envelope_values)
    envelope_mean = float(np.mean(all_envelopes))
    envelope_std = float(np.std(all_envelopes))
    
    print(f"   Signal - Mean: {signal_mean:.6f}, Std: {signal_std:.6f}")
    print(f"   Signal - Range: [{np.min(all_signals):.6f}, {np.max(all_signals):.6f}]")
    print(f"   Envelope - Mean: {envelope_mean:.6f}, Std: {envelope_std:.6f}")
    print(f"   Envelope - Range: [{np.min(all_envelopes):.6f}, {np.max(all_envelopes):.6f}]")
    
    return {
        'signal_mean': signal_mean,
        'signal_std': signal_std,
        'envelope_mean': envelope_mean,
        'envelope_std': envelope_std
    }


def compute_snr_weights(snr, loss_type='min_snr', min_snr_gamma=5.0):
    """
    Compute loss weights based on SNR to prevent gradient explosion.
    
    Args:
        snr: Signal-to-Noise Ratio tensor [num_timesteps]
        loss_type: Type of SNR weighting ('min_snr', 'truncated_snr', 'uniform')
        min_snr_gamma: Gamma parameter for min-SNR weighting (default: 5.0)
    
    Returns:
        weights: Loss weights [num_timesteps]
    """
    if loss_type == 'min_snr':
        # Min-SNR weighting: https://arxiv.org/abs/2303.09556
        # Prevents large gradients at low noise levels
        weights = torch.minimum(snr, torch.ones_like(snr) * min_snr_gamma)
    elif loss_type == 'truncated_snr':
        # Truncated SNR: clip both high and low SNR
        weights = torch.clamp(snr, min=0.1, max=min_snr_gamma)
    else:  # uniform
        weights = torch.ones_like(snr)
    
    return weights


def compute_loss_with_weighting(
    predicted_noise,
    target_noise,
    snr_weights,
    timesteps,
    use_l1=False
):
    """
    Compute weighted diffusion loss with SNR weighting.
    
    Args:
        predicted_noise: Model predicted noise [batch, seq_len]
        target_noise: Ground truth noise [batch, seq_len]
        snr_weights: SNR-based weights [num_timesteps]
        timesteps: Timestep indices [batch]
        use_l1: Use L1 loss instead of MSE
    
    Returns:
        weighted_loss: Scalar loss value
    """
    # Get weights for this batch
    batch_weights = snr_weights[timesteps].view(-1, 1)  # [batch, 1]
    
    # Compute per-sample loss
    if use_l1:
        sample_losses = torch.abs(predicted_noise - target_noise).mean(dim=1, keepdim=True)
    else:
        sample_losses = ((predicted_noise - target_noise) ** 2).mean(dim=1, keepdim=True)
    
    # Apply SNR weights
    weighted_loss = (sample_losses * batch_weights).mean()
    
    return weighted_loss


def monitor_gradients(model, max_norm=10.0):
    """
    Monitor gradient statistics for debugging.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm threshold
    
    Returns:
        dict with gradient statistics
    """
    total_norm = 0.0
    max_grad = 0.0
    grad_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_grad = max(max_grad, param_norm.item())
            grad_count += 1
    
    total_norm = total_norm ** 0.5
    
    return {
        'total_norm': total_norm,
        'max_grad': max_grad,
        'avg_grad': total_norm / max(grad_count, 1),
        'exploded': total_norm > max_norm
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train Signal DiT V2')
    parser.add_argument('--data_dir', type=str, default='data_prep_acc/processed_dynamic',
                        help='Directory containing processed signal files')
    parser.add_argument('--output_dir', type=str, default='signal_model_v3',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (number of packed batches)')
    parser.add_argument('--pack_size', type=int, default=1,
                        help='Number of sequences to pack together')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Normalize signals to target std (default: False for original scale)')
    parser.add_argument('--target_std', type=float, default=1.0,
                        help='Target standard deviation for normalized signals')
    parser.add_argument('--loss_type', type=str, default='min_snr',
                        choices=['min_snr', 'truncated_snr', 'uniform'],
                        help='Loss weighting strategy for stable gradients')
    parser.add_argument('--min_snr_gamma', type=float, default=5.0,
                        help='Gamma parameter for min-SNR loss weighting')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--use_l1_loss', action='store_true', default=False,
                        help='Use L1 loss instead of MSE (more robust)')
    parser.add_argument('--envelope_loss_weight', type=float, default=0.3,
                        help='Weight for envelope loss (0.0-1.0, default: 0.3, reaches full weight at epoch 50)')
    
    return parser.parse_args()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """Update Exponential Moving Average model"""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def plot_signals_v2(
    generated,
    lowpass,
    original=None,
    save_path='samples.png',
    n_samples=4,
    sample_rate=100.0,
    actual_length=None
):
    """
    Plot generated signals in ORIGINAL scale (no log-space conversion needed).
    
    Args:
        generated: Generated signals in ORIGINAL scale [batch, seq_len]
        lowpass: Lowpass conditioning in ORIGINAL scale [batch, seq_len]
        original: Original signals in ORIGINAL scale (optional) [batch, seq_len]
        save_path: Path to save figure
        n_samples: Number of samples to plot
        sample_rate: Sampling rate
        actual_length: Actual length (excluding padding)
    """
    # DEBUG: Print value ranges
    print(f"\nDEBUG - generated stats:")
    print(f"  Shape: {generated.shape}")
    print(f"  Range: [{generated.min().item():.6f}, {generated.max().item():.6f}]")
    print(f"  Mean: {generated.mean().item():.6f}, Std: {generated.std().item():.6f}")
    
    print(f"DEBUG - lowpass stats:")
    print(f"  Range: [{lowpass.min().item():.6f}, {lowpass.max().item():.6f}]")
    print(f"  Mean: {lowpass.mean().item():.6f}")
    
    if original is not None:
        print(f"DEBUG - original stats:")
        print(f"  Range: [{original.min().item():.6f}, {original.max().item():.6f}]")
        print(f"  Mean: {original.mean().item():.6f}")
    
    n_cols = 3 if original is not None else 2
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(15 if n_cols == 2 else 20, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(n_samples, generated.shape[0])):
        # Only plot up to actual_length to avoid padding artifacts
        if actual_length is not None:
            plot_len = actual_length
        else:
            plot_len = generated.shape[1]
        
        seq_len = plot_len
        time_axis = np.arange(seq_len) / sample_rate
        
        # Generated broadband
        axes[i, 0].plot(time_axis, generated[i, :plot_len].cpu().numpy(), 'b-', linewidth=0.5, alpha=0.8)
        axes[i, 0].set_title(f'Generated Broadband {i+1}')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Acceleration')
        axes[i, 0].grid(True, alpha=0.3)
        
        col_idx = 1
        
        # Original broadband if provided
        if original is not None:
            axes[i, 1].plot(time_axis, original[i, :plot_len].cpu().numpy(), 'g-', linewidth=0.8)
            axes[i, 1].set_title(f'Original Broadband {i+1}')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].set_ylabel('Acceleration')
            axes[i, 1].grid(True, alpha=0.3)
            col_idx = 2
        
        # Lowpass conditioning
        axes[i, col_idx].plot(time_axis, lowpass[i, :plot_len].cpu().numpy(), 'r-', linewidth=0.8)
        axes[i, col_idx].set_title(f'Lowpass Conditioning {i+1}')
        axes[i, col_idx].set_xlabel('Time (s)')
        axes[i, col_idx].set_ylabel('Acceleration')
        axes[i, col_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def compute_envelope_torch(x, window_size=200):
    """Compute envelope using moving RMS with longer window for stability"""
    x_squared = x ** 2
    # Use 1D convolution for moving average
    kernel = torch.ones(1, 1, window_size, device=x.device) / window_size
    # Pad to maintain length
    x_padded = torch.nn.functional.pad(x_squared.unsqueeze(1), (window_size//2, window_size//2), mode='reflect')
    rms = torch.nn.functional.conv1d(x_padded, kernel)
    envelope = torch.sqrt(rms.squeeze(1) + 1e-8)
    return envelope


def compute_relative_envelope_loss(pred_envelope, target_envelope, epsilon=1e-8):
    """Compute relative envelope loss (ratio-based, scale-invariant)
    
    Uses mean+std normalization for better stability than max normalization.
    Focuses on envelope shape matching rather than absolute amplitude.
    """
    # Normalize both envelopes using mean and std (more stable than max)
    pred_mean = pred_envelope.mean(dim=-1, keepdim=True)
    pred_std = pred_envelope.std(dim=-1, keepdim=True) + epsilon
    pred_norm = (pred_envelope - pred_mean) / pred_std
    
    target_mean = target_envelope.mean(dim=-1, keepdim=True) 
    target_std = target_envelope.std(dim=-1, keepdim=True) + epsilon
    target_norm = (target_envelope - target_mean) / target_std
    
    # MSE on normalized envelopes
    mse_loss = torch.nn.functional.mse_loss(pred_norm, target_norm)
    
    # Add correlation loss to encourage shape matching
    # Compute correlation coefficient
    pred_centered = pred_envelope - pred_mean
    target_centered = target_envelope - target_mean
    
    correlation = (pred_centered * target_centered).sum(dim=-1) / (
        torch.sqrt((pred_centered ** 2).sum(dim=-1) * (target_centered ** 2).sum(dim=-1)) + epsilon
    )
    correlation_loss = 1.0 - correlation.mean()  # 1 - corr, so lower is better
    
    # Combined: 70% MSE + 30% correlation
    combined_loss = 0.7 * mse_loss + 0.3 * correlation_loss
    
    return combined_loss


def train_epoch(
    model,
    ema_model,
    dataloader,
    noise_schedule,
    optimizer,
    device,
    epoch,
    writer,
    signal_mean=0.0,
    signal_std=1.0,
    envelope_mean=0.0,
    envelope_std=1.0,
    normalize=True,
    target_std=1.0,
    loss_type='min_snr',
    min_snr_gamma=5.0,
    grad_clip=1.0,
    use_l1_loss=False,
    envelope_loss_weight=0.3
):
    """Train for one epoch using standard DDPM diffusion with envelope-aware loss"""
    model.train()
    
    total_loss = 0.0
    total_waveform_loss = 0.0
    total_envelope_loss = 0.0
    n_batches = 0
    grad_stats_accum = {'total_norm': 0, 'max_grad': 0, 'exploded_count': 0}
    
    sqrt_alphas_cumprod = noise_schedule['sqrt_alphas_cumprod'].to(device)
    sqrt_one_minus_alphas_cumprod = noise_schedule['sqrt_one_minus_alphas_cumprod'].to(device)
    snr = noise_schedule['snr'].to(device)
    num_timesteps = len(sqrt_alphas_cumprod)
    
    # Compute SNR weights for loss
    snr_weights = compute_snr_weights(snr, loss_type=loss_type, min_snr_gamma=min_snr_gamma)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        broadband = batch['broadband'].to(device)  # [pack_size, max_len] - ORIGINAL scale
        lowpass = batch['lowpass'].to(device)  # ORIGINAL scale
        envelope = batch['envelope'].to(device)  # ORIGINAL scale
        metadata = batch['metadata'].to(device)  # Kept for interface compatibility (ignored by model)
        boundaries = batch['boundaries']  # List of tuples, stays on CPU
        actual_length = batch['actual_length']  # Integer, stays on CPU
        position_ids = batch['position_ids'].to(device)
        
        # Normalize signals if enabled
        if normalize:
            broadband = (broadband - signal_mean) / signal_std * target_std
            lowpass = (lowpass - signal_mean) / signal_std * target_std
            # Use separate envelope statistics!
            envelope = (envelope - envelope_mean) / envelope_std * target_std
        
        batch_size = broadband.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
        
        # CRITICAL FIX: Sample noise at the SAME scale as the signal
        # This ensures diffusion happens at the correct amplitude range
        if normalize:
            # When normalized, signals are at target_std (1.0), so noise should match
            noise = torch.randn_like(broadband)  # std ≈ 1.0
        else:
            # When NOT normalized, signals are at original scale (std ≈ 0.002)
            # So noise must be scaled down to match!
            noise = torch.randn_like(broadband) * signal_std  # std ≈ 0.002
        
        # Add noise to clean signal: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        noisy_broadband = sqrt_alpha_t * broadband + sqrt_one_minus_alpha_t * noise
        
        # Predict noise
        predicted_noise = model(
            noisy_broadband,
            t.float() / num_timesteps,  # Normalize timesteps to [0, 1]
            x_lowpass=lowpass,
            x_envelope=envelope,
            metadata=metadata,  # Passed but ignored by model
            boundaries=boundaries,
            actual_length=actual_length,
            position_ids=position_ids
        )
        
        # Compute waveform loss with SNR weighting
        waveform_loss = compute_loss_with_weighting(
            predicted_noise,
            noise,
            snr_weights,
            t,
            use_l1=use_l1_loss
        )
        
        # FIX BUG #2: Verify loss is computed in correct space
        # Add sanity checks to catch scale mismatches early
        if batch_idx == 0 and epoch % 10 == 0:  # Periodic check
            pred_max = predicted_noise.abs().max().item()
            target_max = noise.abs().max().item()
            signal_max = broadband.abs().max().item()
            print(f"  🔍 Scale check - Pred noise: {pred_max:.6f}, Target noise: {target_max:.6f}, Signal: {signal_max:.6f}")
        
        # Compute envelope loss with adaptive weighting
        envelope_loss_value = torch.tensor(0.0, device=device)
        # Faster ramp-up: reach full weight by epoch 50 (was 100)
        # This gives stronger envelope learning signal earlier
        adaptive_env_weight = envelope_loss_weight * min(1.0, epoch / 50.0)
        
        if adaptive_env_weight > 0:
            # Predict clean signal from noise prediction
            sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            pred_x0 = (noisy_broadband - sqrt_one_minus_alpha_t * predicted_noise) / (sqrt_alpha_t + 1e-8)
            
            # Compute envelopes with longer window for stability
            pred_envelope = compute_envelope_torch(pred_x0, window_size=200)
            target_envelope = compute_envelope_torch(broadband, window_size=200)
            
            # Use relative envelope loss (scale-invariant with correlation)
            envelope_loss_value = compute_relative_envelope_loss(pred_envelope, target_envelope)
            
            # Clip envelope loss to prevent explosions (more generous clipping)
            envelope_loss_value = torch.clamp(envelope_loss_value, 0.0, 5.0)
        
        # Combined loss with adaptive envelope weight
        loss = (1 - adaptive_env_weight) * waveform_loss + adaptive_env_weight * envelope_loss_value
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Monitor gradients before clipping
        grad_stats = monitor_gradients(model, max_norm=grad_clip * 2)
        grad_stats_accum['total_norm'] += grad_stats['total_norm']
        grad_stats_accum['max_grad'] = max(grad_stats_accum['max_grad'], grad_stats['max_grad'])
        if grad_stats['exploded']:
            grad_stats_accum['exploded_count'] += 1
        
        # IMPROVED: Adaptive gradient clipping to prevent explosion
        # More aggressive clipping during early training and when gradients are large
        if grad_stats['total_norm'] > grad_clip * 3:
            # Strong explosion detected - clip very aggressively
            effective_clip = grad_clip * 0.3
        elif epoch < 20:
            # Early training - moderate clipping
            effective_clip = grad_clip * 0.5
        else:
            # Normal training
            effective_clip = grad_clip
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=effective_clip)
        
        optimizer.step()
        
        # Update EMA
        update_ema(ema_model, model, decay=0.999)
        
        total_loss += loss.item()
        total_waveform_loss += waveform_loss.item()
        total_envelope_loss += envelope_loss_value.item()
        n_batches += 1
        
        # Logging
        if batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Loss/waveform', waveform_loss.item(), global_step)
            writer.add_scalar('Loss/envelope', envelope_loss_value.item(), global_step)
            writer.add_scalar('Gradients/norm', grad_stats['total_norm'], global_step)
            writer.add_scalar('Gradients/max', grad_stats['max_grad'], global_step)
            
            # Compute envelope correlation for monitoring
            if adaptive_env_weight > 0:
                with torch.no_grad():
                    pred_centered = pred_envelope - pred_envelope.mean(dim=-1, keepdim=True)
                    target_centered = target_envelope - target_envelope.mean(dim=-1, keepdim=True)
                    corr = (pred_centered * target_centered).sum() / (
                        torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-8
                    )
                    writer.add_scalar('Envelope/correlation', corr.item(), global_step)
                    
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}, "
                          f"Wave: {waveform_loss.item():.6f}, Env: {envelope_loss_value.item():.6f}, "
                          f"EnvWeight: {adaptive_env_weight:.3f}, EnvCorr: {corr.item():.3f}, GradNorm: {grad_stats['total_norm']:.3f}")
            else:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}, "
                      f"Wave: {waveform_loss.item():.6f}, Env: {envelope_loss_value.item():.6f}, "
                      f"EnvWeight: {adaptive_env_weight:.3f}, GradNorm: {grad_stats['total_norm']:.3f}")
        
        if batch_idx % 50 == 0 and batch_idx > 0:
            avg_grad = grad_stats_accum['total_norm'] / n_batches
            avg_env_loss = total_envelope_loss / n_batches
            print(f"  [Progress] Processed {batch_idx} batches, avg loss: {total_loss/n_batches:.6f}, "
                  f"avg env loss: {avg_env_loss:.6f}, avg grad: {avg_grad:.3f}")
            if grad_stats_accum['exploded_count'] > 0:
                print(f"  ⚠️  Warning: {grad_stats_accum['exploded_count']} gradient explosions detected")
    
    avg_loss = total_loss / n_batches
    avg_waveform_loss = total_waveform_loss / n_batches
    avg_envelope_loss = total_envelope_loss / n_batches
    
    # Log gradient and loss statistics
    avg_grad_norm = grad_stats_accum['total_norm'] / n_batches
    writer.add_scalar('Loss/train_waveform_epoch', avg_waveform_loss, epoch)
    writer.add_scalar('Loss/train_envelope_epoch', avg_envelope_loss, epoch)
    writer.add_scalar('Gradients/epoch_avg_norm', avg_grad_norm, epoch)
    writer.add_scalar('Gradients/epoch_max', grad_stats_accum['max_grad'], epoch)
    
    if grad_stats_accum['exploded_count'] > 0:
        print(f"  ⚠️  Epoch {epoch}: {grad_stats_accum['exploded_count']}/{n_batches} batches had gradient explosion")
    
    print(f"  📊 Epoch {epoch} - Waveform Loss: {avg_waveform_loss:.6f}, Envelope Loss: {avg_envelope_loss:.6f}")
    
    # Report envelope learning progress
    current_adaptive_weight = envelope_loss_weight * min(1.0, epoch / 50.0)
    print(f"  📈 Envelope Weight: {current_adaptive_weight:.3f} (reaches {envelope_loss_weight:.3f} at epoch 50)")
    
    return avg_loss


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    noise_schedule,
    device,
    epoch,
    output_dir,
    num_samples=4,
    signal_mean=0.0,
    signal_std=1.0,
    envelope_mean=0.0,
    envelope_std=1.0,
    normalize=True,
    target_std=1.0
):
    """Generate samples for evaluation using DDIM sampling"""
    model.eval()
    
    # Get one batch
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("Warning: Validation dataloader is empty, skipping evaluation")
        return
    
    broadband = batch['broadband'].to(device)
    lowpass = batch['lowpass'].to(device)
    envelope = batch['envelope'].to(device)
    metadata = batch['metadata'].to(device)
    boundaries = batch['boundaries']
    actual_length = batch['actual_length']
    position_ids = batch['position_ids'].to(device)
    
    # Store original for comparison
    broadband_original = broadband.clone()
    
    # Normalize conditioning signals if enabled
    if normalize:
        lowpass = (lowpass - signal_mean) / signal_std * target_std
        # Use separate envelope statistics!
        envelope = (envelope - envelope_mean) / envelope_std * target_std
    
    print(f"\n=== EVALUATION (Epoch {epoch}) ===")
    print(f"broadband shape: {broadband.shape}")
    print(f"lowpass shape: {lowpass.shape}")
    print(f"actual_length: {actual_length}")
    
    # DDIM sampling (simplified - can be improved)
    sqrt_alphas_cumprod = noise_schedule['sqrt_alphas_cumprod'].to(device)
    sqrt_one_minus_alphas_cumprod = noise_schedule['sqrt_one_minus_alphas_cumprod'].to(device)
    alphas_cumprod = noise_schedule['alphas_cumprod'].to(device)
    
    # CRITICAL FIX: Start sampling at the SAME scale as training
    shape = broadband.shape
    if normalize:
        # When normalized, start with noise at target_std (1.0)
        x = torch.randn(shape, device=device)  # std ≈ 1.0 (target_std is already 1.0)
        print(f"Starting noise - Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}, Range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    else:
        # When NOT normalized, start with noise at original signal scale
        x = torch.randn(shape, device=device) * signal_std  # std ≈ 0.002
    
    # Improved DDIM sampling with proper noise scheduling
    num_inference_steps = 50
    timesteps = torch.linspace(len(sqrt_alphas_cumprod) - 1, 0, num_inference_steps).long()
    eta = 0.0  # DDIM deterministic (eta=0), increase for stochasticity
    
    for i, t in enumerate(timesteps):
        t_batch = t.expand(shape[0]).to(device)
        t_normalized = t_batch.float() / len(sqrt_alphas_cumprod)
        
        # Predict noise
        predicted_noise = model(
            x,
            t_normalized,
            x_lowpass=lowpass,
            x_envelope=envelope,
            metadata=metadata,
            boundaries=boundaries,
            actual_length=actual_length,
            position_ids=position_ids
        )
        
        # FIXED DDIM step - simplified and stable formula
        alpha_t = alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_t_prev = alphas_cumprod[t_prev]
        else:
            alpha_t_prev = torch.ones_like(alpha_t)
        
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
        
        # Predict x0 from noise (standard DDPM formula)
        pred_x0 = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        
        # FIX BUG #1: Remove hard clamp - let model learn the correct scale
        # Only apply very loose clipping to prevent numerical instability
        # Signals have std ≈ 0.002, so ±0.05 is ~25σ (extremely conservative)
        if not normalize:
            pred_x0 = torch.clamp(pred_x0, -0.05, 0.05)  # Loose clamp at physical scale
        
        # DDIM update (deterministic when eta=0)
        # Standard formula: x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1}) * pred_noise
        dir_xt = sqrt_one_minus_alpha_t_prev * predicted_noise
        x = sqrt_alpha_t_prev * pred_x0 + dir_xt
        
        # Add noise if eta > 0 (stochastic sampling)
        if eta > 0 and i < len(timesteps) - 1:
            noise_scale = eta * sqrt_one_minus_alpha_t_prev
            x = x + noise_scale * torch.randn_like(x)
    
    generated = x
    
    # CRITICAL: Proper denormalization from normalized space
    if normalize:
        print(f"Before denorm - Mean: {generated.mean().item():.6f}, Std: {generated.std().item():.6f}, Range: [{generated.min().item():.3f}, {generated.max().item():.3f}]")
        
        # Denormalize: reverse the normalization formula
        # Original: normalized = (signal - mean) / std * target_std
        # Reverse: signal = (normalized / target_std) * std + mean
        generated = (generated / target_std) * signal_std + signal_mean
        
        print(f"After denorm - Mean: {generated.mean().item():.6f}, Std: {generated.std().item():.6f}, Range: [{generated.min().item():.3f}, {generated.max().item():.3f}]")
        print(f"Expected - Signal std: {signal_std:.6f}")
        
        # Denormalize conditioning signals
        lowpass = (lowpass / target_std) * signal_std + signal_mean
        # FIX: envelope denormalization was cancelling out!
        envelope = (envelope / target_std) * envelope_std + envelope_mean
    
    print(f"Generated range (denormalized): [{generated.min().item():.6f}, {generated.max().item():.6f}]")
    print(f"Original range: [{broadband_original.min().item():.6f}, {broadband_original.max().item():.6f}]")
    
    # Plot samples
    save_path = os.path.join(output_dir, 'generated', f'epoch_{epoch:04d}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plot_signals_v2(
        generated[:num_samples],
        lowpass[:num_samples],
        original=broadband_original[:num_samples],
        save_path=save_path,
        n_samples=num_samples,
        actual_length=actual_length
    )
    
    print(f"Generated samples saved to {save_path}")


def save_checkpoint(
    model,
    ema_model,
    optimizer,
    scheduler,
    epoch,
    loss,
    output_dir,
    filename='checkpoint.pt',
    signal_mean=0.0,
    signal_std=1.0,
    envelope_mean=0.0,
    envelope_std=1.0
):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'signal_mean': signal_mean,
        'signal_std': signal_std,
        'envelope_mean': envelope_mean,
        'envelope_std': envelope_std
    }
    
    save_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model,
    ema_model,
    optimizer,
    scheduler,
    checkpoint_path
):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    # Load normalization statistics if available
    signal_mean = checkpoint.get('signal_mean', 0.0)
    signal_std = checkpoint.get('signal_std', 1.0)
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, loss {loss:.6f}")
    print(f"Signal statistics - Mean: {signal_mean:.6f}, Std: {signal_std:.6f}")
    
    return epoch, loss, signal_mean, signal_std


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'generated'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tensorboard'), exist_ok=True)
    
    # Get signal paths
    signal_paths = glob.glob(os.path.join(args.data_dir, '*.npz'))
    print(f"Found {len(signal_paths)} signal files")
    
    if len(signal_paths) == 0:
        raise ValueError(f"No signal files found in {args.data_dir}")
    
    # Split train/val
    np.random.shuffle(signal_paths)
    split_idx = int(0.9 * len(signal_paths))
    train_paths = signal_paths[:split_idx]
    val_paths = signal_paths[split_idx:]
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create dataloaders
    train_loader = create_dataloader_v2(
        train_paths,
        batch_size=args.batch_size,
        pack_size=args.pack_size,
        max_length=150000,  # Accommodate longest signals
        num_workers=args.num_workers,
        shuffle=True,
        sample_rate=100.0,
        min_length=5000,
        validate_signals=True
    )
    
    val_loader = create_dataloader_v2(
        val_paths,
        batch_size=args.batch_size,
        pack_size=args.pack_size,
        max_length=150000,  # Accommodate longest signals
        num_workers=0,
        shuffle=False,
        sample_rate=100.0,
        min_length=5000,
        validate_signals=True
    )
    
    # Create model
    model = SignalDiTV2(
        patch_size=100,
        hidden_dim=384,
        num_encoder_layers=4,
        num_decoder_layers=5,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        use_rope=True,
        rope_base=10.0,
        stft_n_fft=512,
        stft_hop_length=256,
        metadata_dim=3,
        metadata_hidden_dims=[64, 128, 256]
    ).to(device)
    
    # Create EMA model
    ema_model = SignalDiTV2(
        patch_size=100,
        hidden_dim=384,
        num_encoder_layers=4,
        num_decoder_layers=5,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        use_rope=True,
        rope_base=10.0,
        stft_n_fft=512,
        stft_hop_length=256,
        metadata_dim=3,
        metadata_hidden_dims=[64, 128, 256]
    ).to(device)
    
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Create standard DDPM noise schedule
    noise_schedule = get_noise_schedule(
        num_steps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
    print(f"Noise schedule: {len(noise_schedule['betas'])} timesteps")
    print(f"Beta range: [{noise_schedule['betas'][0]:.6f}, {noise_schedule['betas'][-1]:.6f}]")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Compute dataset statistics for normalization
    signal_mean = 0.0
    signal_std = 1.0
    envelope_mean = 0.0
    envelope_std = 1.0
    
    if args.normalize:
        print(f"\n{'='*50}")
        print("Computing dataset statistics for normalization...")
        print(f"{'='*50}")
        stats = compute_dataset_statistics(train_paths, max_samples=100)
        signal_mean = stats['signal_mean']
        signal_std = stats['signal_std']
        envelope_mean = stats['envelope_mean']
        envelope_std = stats['envelope_std']
        print(f"Will normalize signals to target_std={args.target_std}")
        print(f"\nLoss configuration:")
        print(f"  Loss weighting: {args.loss_type}")
        print(f"  Min-SNR gamma: {args.min_snr_gamma}")
        print(f"  Gradient clipping: {args.grad_clip}")
        print(f"  Loss function: {'L1' if args.use_l1_loss else 'MSE'}")
    
    # Auto-resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    # WARNING: Old checkpoints incompatible with new simplified architecture
    # Priority: 1) --resume argument, 2) best_checkpoint.pt, 3) last_checkpoint.pt
    checkpoint_to_load = None
    if args.resume:
        checkpoint_to_load = args.resume
        print(f"Resuming from specified checkpoint: {checkpoint_to_load}")
    else:
        # Auto-resume from best or last checkpoint
        best_ckpt = os.path.join(args.output_dir, 'best_checkpoint.pt')
        last_ckpt = os.path.join(args.output_dir, 'last_checkpoint.pt')
        if os.path.exists(best_ckpt):
            checkpoint_to_load = best_ckpt
            print(f"Auto-resuming from best checkpoint: {best_ckpt}")
        elif os.path.exists(last_ckpt):
            checkpoint_to_load = last_ckpt
            print(f"Auto-resuming from last checkpoint: {last_ckpt}")
    
    if checkpoint_to_load:
        try:
            start_epoch, loaded_loss, loaded_mean, loaded_std = load_checkpoint(
                model, ema_model, optimizer, scheduler, checkpoint_to_load
            )
            best_loss = loaded_loss  # Initialize best_loss from checkpoint
            start_epoch += 1
            
            # Use loaded statistics if normalization is enabled
            if args.normalize:
                signal_mean = loaded_mean
                signal_std = loaded_std
                # envelope_mean and envelope_std already computed above from compute_dataset_statistics
                print(f"   Using loaded signal statistics - Mean: {signal_mean:.6f}, Std: {signal_std:.6f}")
                print(f"   Using computed envelope statistics - Mean: {envelope_mean:.6f}, Std: {envelope_std:.6f}")
            
            print(f"✅ Successfully resumed from epoch {start_epoch}, loss: {loaded_loss:.6f}")
            print(f"📊 Envelope loss weight: {args.envelope_loss_weight}")
        except RuntimeError as e:
            print(f"❌ ERROR loading checkpoint: {e}")
            print(f"   Architecture mismatch detected (old checkpoint has metadata_encoder)")
            print(f"   Starting training from scratch instead...")
            start_epoch = 0
            best_loss = float('inf')
    else:
        print("Starting training from scratch")
    
    # Training loop
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model,
            ema_model,
            train_loader,
            noise_schedule,
            optimizer,
            device,
            epoch,
            writer,
            signal_mean=signal_mean,
            signal_std=signal_std,
            envelope_mean=envelope_mean,
            envelope_std=envelope_std,
            normalize=args.normalize,
            target_std=args.target_std,
            loss_type=args.loss_type,
            min_snr_gamma=args.min_snr_gamma,
            grad_clip=args.grad_clip,
            use_l1_loss=args.use_l1_loss,
            envelope_loss_weight=args.envelope_loss_weight
        )
        
        # Log
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.6f}")
        
        # Scheduler step
        scheduler.step()
        
        # Evaluate and generate samples every 5 epochs
        epochs_since_start = epoch - start_epoch
        if epochs_since_start % 5 == 0 or epoch % 5 == 0 or epoch == args.epochs - 1:
            # Use regular model for evaluation, not EMA
            eval_model = model if epoch < 20 else ema_model
            evaluate(
                eval_model,
                val_loader,
                noise_schedule,
                device,
                epoch,
                args.output_dir,
                num_samples=4,
                signal_mean=signal_mean,
                signal_std=signal_std,
                envelope_mean=envelope_mean,
                envelope_std=envelope_std,
                normalize=args.normalize,
                target_std=args.target_std
            )
        
        # Save checkpoints
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model,
                ema_model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                args.output_dir,
                filename=f'checkpoint_epoch_{epoch:04d}.pt',
                signal_mean=signal_mean,
                signal_std=signal_std,
                envelope_mean=envelope_mean,
                envelope_std=envelope_std
            )
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(
                model,
                ema_model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                args.output_dir,
                filename='best_checkpoint.pt',
                signal_mean=signal_mean,
                signal_std=signal_std,
                envelope_mean=envelope_mean,
                envelope_std=envelope_std
            )
        
        # Save last checkpoint
        save_checkpoint(
            model,
            ema_model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            args.output_dir,
            filename='last_checkpoint.pt',
            signal_mean=signal_mean,
            signal_std=signal_std,
            envelope_mean=envelope_mean,
            envelope_std=envelope_std
        )
    
    writer.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
