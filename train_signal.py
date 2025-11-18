"""
Training Script for Signal Diffusion Transformer

This script trains a diffusion transformer to generate broadband earthquake 
ground motion signals conditioned on low-frequency simulations.

Author: Adapted for earthquake signal generation
"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from collections import OrderedDict
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings

from signal_dit import SignalDiT
from signal_diff_utils import SignalDiffusion, gen_signal_batches, SignalMetrics, apply_lowpass_filter
from signal_datasets import create_signal_loader, preprocess_at2_files_for_training
from pga_predictor import PGAPredictor, PGALoss, EarthquakeDataProcessor
from signal_config import signal_config, get_config_value
from utils import Config

warnings.filterwarnings("ignore")

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """Update Exponential Moving Average model"""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model"""
    for p in model.parameters():
        p.requires_grad = flag

def plot_signal_samples(signals_broad, signals_low, save_path, sample_rate=100.0, n_samples=4, original_broad=None):
    """Plot sample signals for visualization"""
    # Use 3 columns if original broadband is provided, otherwise 2
    n_cols = 3 if original_broad is not None else 2
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(15 if n_cols == 2 else 20, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    time_axis = np.arange(signals_broad.shape[-1]) / sample_rate
    
    for i in range(min(n_samples, signals_broad.shape[0])):
        # Generated broadband signal
        axes[i, 0].plot(time_axis, signals_broad[i].cpu().numpy(), 'b-', linewidth=0.8)
        axes[i, 0].set_title(f'Generated Broadband Signal {i+1}')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Acceleration (normalized)')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Original broadband signal (if provided) - PGA normalized (same as generated)
        if original_broad is not None:
            axes[i, 1].plot(time_axis, original_broad[i].cpu().numpy(), 'g-', linewidth=0.8)
            axes[i, 1].set_title(f'Original Broadband (Target) {i+1}')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].set_ylabel('Acceleration (normalized)')
            axes[i, 1].grid(True, alpha=0.3)
            col_idx = 2  # Low-freq goes in column 2
        else:
            col_idx = 1  # Low-freq goes in column 1
        
        # Low-frequency conditioning signal
        if signals_low is not None:
            axes[i, col_idx].plot(time_axis, signals_low[i].cpu().numpy(), 'r-', linewidth=0.8)
            axes[i, col_idx].set_title(f'Low-frequency Conditioning {i+1}')
            axes[i, col_idx].set_xlabel('Time (s)')
            axes[i, col_idx].set_ylabel('Acceleration (normalized)')
            axes[i, col_idx].grid(True, alpha=0.3)
        else:
            axes[i, col_idx].text(0.5, 0.5, 'No Conditioning', 
                          transform=axes[i, col_idx].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_signal_quality(generated_signals, real_signals, sample_rate=100.0):
    """Evaluate the quality of generated signals"""
    metrics = {}
    
    # Compute PGA statistics
    gen_pga = SignalMetrics.compute_pga(generated_signals)
    real_pga = SignalMetrics.compute_pga(real_signals)
    
    metrics['gen_pga_mean'] = gen_pga.mean().item()
    metrics['gen_pga_std'] = gen_pga.std().item()
    metrics['real_pga_mean'] = real_pga.mean().item()
    metrics['real_pga_std'] = real_pga.std().item()
    
    # Compute frequency content comparison
    gen_freqs, gen_magnitude = SignalMetrics.compute_frequency_content(generated_signals, sample_rate)
    real_freqs, real_magnitude = SignalMetrics.compute_frequency_content(real_signals, sample_rate)
    
    # Focus on 0-30 Hz range
    freq_mask = (gen_freqs >= 0) & (gen_freqs <= 30)
    gen_spectrum = gen_magnitude[:, freq_mask].mean(dim=0)
    real_spectrum = real_magnitude[:, freq_mask].mean(dim=0)
    
    # Compute spectral similarity (simplified)
    spectral_error = torch.nn.functional.mse_loss(gen_spectrum, real_spectrum)
    metrics['spectral_mse'] = spectral_error.item()
    
    return metrics

def train_signal_dit(model_dir, data_dir, eval_interval, log_interval, conf):
    """Main training function for Signal DiT"""
    
    # Setup directories
    os.makedirs(model_dir, exist_ok=True)
    gen_dir = os.path.join(model_dir, 'generated')
    log_img_dir = os.path.join(model_dir, 'log_img')
    log_dir = os.path.join(model_dir, 'tensorboard')
    
    for dir_path in [gen_dir, log_img_dir, log_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loader
    print("Loading training data...")
    train_loader, val_loader = create_signal_loader(
        data_dir, 
        seq_len=conf.seq_len,
        batch_size=conf.batch_size,
        sample_rate=conf.sample_rate,
        num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )
    
    # Convert to iterator for infinite sampling
    train_iter = iter(train_loader)
    
    # Create fixed evaluation dataset - use first batch for consistent visualization
    print("Creating fixed evaluation dataset...")
    eval_batch = next(iter(train_loader))
    
    fixed_eval_lowpass = eval_batch['lowfreq'][:7]  # Fixed 7 low-pass signals
    fixed_eval_broadband = eval_batch['broadband'][:7]  # Corresponding true broadband (PGA normalized)
    
    # Get the file paths from the evaluation batch metadata to load the SAME original signals
    fixed_eval_original = []
    
    batch_metadata = eval_batch['metadata']
    n_samples = min(7, len(fixed_eval_broadband))
    
    try:
        for i in range(n_samples):
            # Handle different metadata batching structures
            if isinstance(batch_metadata, list):
                # Metadata is a list of dicts
                file_path = batch_metadata[i]['file_path']
            elif isinstance(batch_metadata, dict) and 'file_path' in batch_metadata:
                # Metadata is a dict with batched values
                file_paths = batch_metadata['file_path']
                if isinstance(file_paths, (list, tuple)):
                    file_path = file_paths[i]
                else:
                    # Single file path - use it for all samples
                    file_path = file_paths
            else:
                raise ValueError(f"Unexpected metadata structure: {type(batch_metadata)}")
            
            print(f"Loading original signal from: {os.path.basename(file_path)}")
            
            npz_data = np.load(file_path)
            original_signal = npz_data['signal_broadband']  # Raw signal
            
            # Ensure correct length (same processing as dataset)
            if len(original_signal) != conf.seq_len:
                if len(original_signal) > conf.seq_len:
                    original_signal = original_signal[:conf.seq_len]
                else:
                    padded = np.zeros(conf.seq_len)
                    padded[:len(original_signal)] = original_signal
                    original_signal = padded
            
            # Apply SAME PGA normalization as the dataset does
            pga_broad = np.max(np.abs(original_signal))
            pga_threshold = 1e-6  # Same threshold as dataset
            if pga_broad > pga_threshold:
                original_signal = original_signal / pga_broad
            
            fixed_eval_original.append(torch.tensor(original_signal, dtype=torch.float32))
        
        fixed_eval_original = torch.stack(fixed_eval_original)
        
    except Exception as e:
        print(f"Error loading original signals: {e}")
        print("Using normalized broadband signals from dataset as fallback...")
        # Fallback: use the already normalized signals from the dataset
        fixed_eval_original = fixed_eval_broadband.clone()
    print(f"Fixed evaluation set created with {fixed_eval_lowpass.shape[0]} signals (same files for all comparisons)")
    
    # Initialize models
    print("Initializing models...")
    model = SignalDiT(
        seq_len=conf.seq_len,
        dim=conf.dim,
        patch_size=conf.patch_size,
        encoder_depth=conf.encoder_depth,
        decoder_depth=conf.decoder_depth,
        heads=conf.heads,
        mlp_dim=conf.mlp_dim,
        k=conf.k
    ).to(device)
    
    # Initialize diffusion with CRITICAL sigma_max parameter
    diffusion = SignalDiffusion(
        P_mean=conf.P_mean,
        P_std=conf.P_std,
        sigma_data=conf.sigma_data,
        sigma_max=conf.sigma_max  # CRITICAL: Pass sigma_max from config
    )
    
    # Initialize PGA predictor (conditional)
    train_pga = getattr(conf, 'train_pga_predictor', False)  # Default to False if not found
    if train_pga:
        pga_predictor = PGAPredictor(
            seq_len=conf.seq_len,
            cnn_filters=conf.cnn_filters,
            lstm_hidden=conf.lstm_hidden,
            dropout=conf.pga_dropout
        ).to(device)
    else:
        pga_predictor = None
    
    # Optimizers - Use CONSERVATIVE learning rate to prevent explosion
    # CRITICAL FIX: Reduce learning rate to prevent instability after iter 1000
    effective_lr = conf.lr * 0.5  # 0.001 -> 0.0005 (stable learning, prevents explosion)
    print(f"🔧 STABILITY FIX: Learning rate {effective_lr:.6f} (0.5x for stability)")
    optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=0.01, eps=1e-8)
    
    # Learning rate scheduler - reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, 
        threshold=0.01, min_lr=1e-6
    )
    
    if train_pga and pga_predictor is not None:
        pga_optimizer = optim.AdamW(pga_predictor.parameters(), lr=conf.pga_lr, weight_decay=0.01)
    else:
        pga_optimizer = None
    
    # Loss functions
    signal_loss_fn = nn.MSELoss()
    if train_pga:
        pga_loss_fn = PGALoss()
    else:
        pga_loss_fn = None
    
    # EMA model
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    # Tensorboard logging
    writer = SummaryWriter(log_dir)
    
    # Checkpoint management
    last_ckpt = os.path.join(model_dir, 'last_ckpt.pt')
    best_ckpt = os.path.join(model_dir, 'best_ckpt.pt')
    
    start_iter = 0
    best_loss = float('inf')
    best_quality = 0.0
    quality_patience = 15  # Stop if quality degrades for 15 evaluations (reasonable patience)
    quality_patience_counter = 0
    early_stop = False
    
    # Disable early stopping to see full learning progression
    disable_early_stop = True
    
    if os.path.exists(last_ckpt):
        print("Loading checkpoint...")
        ckpt = torch.load(last_ckpt, map_location=device)
        start_iter = ckpt['iter']
        best_loss = ckpt.get('best_loss', best_loss)
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # PGA predictor loading disabled
        # if 'pga_predictor' in ckpt:
        #     pga_predictor.load_state_dict(ckpt['pga_predictor'])
        #     pga_optimizer.load_state_dict(ckpt['pga_optimizer'])
        print(f'Checkpoint restored at iter {start_iter}; best loss: {best_loss:.6f} (PGA disabled)')
        
        # CRITICAL FIX: Scale final layer weights to match noise magnitude  
        # Target noise magnitude is ~5.0, but model predicts ~0.1 (50x too small)
        with torch.no_grad():
            if hasattr(model, 'final_layer') and hasattr(model.final_layer, 'linear'):
                current_std = model.final_layer.linear.weight.std().item()
                if current_std < 0.1:  # If weights are too small (< 0.1)
                    scale_factor = 10.0  # Scale up by 10x 
                    model.final_layer.linear.weight.data *= scale_factor
                    model.final_layer.linear.bias.data *= scale_factor
                    print(f"🔧 CRITICAL FIX: Scaled final layer weights by {scale_factor}x (std: {current_std:.4f} → {model.final_layer.linear.weight.std().item():.4f})")
    else:
        print("Starting new training")
    
    # Training loop
    print("Starting training...")
    model.train()
    ema.eval()
    if train_pga and pga_predictor is not None:
        pga_predictor.train()
    
    # CRITICAL DEBUG: Check if model parameters have gradients enabled
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"🔍 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    if trainable_params == 0:
        print("❌ CRITICAL ERROR: NO TRAINABLE PARAMETERS! Enabling gradients...")
        requires_grad(model, True)
    else:
        print("✅ Model has trainable parameters")
    
    running_signal_loss = 0.0
    running_envelope_loss = 0.0
    running_pga_loss = 0.0
    start_time = time.time()
    
    # Track best loss for explosion detection
    best_loss_so_far = float('inf')
    loss_explosion_threshold = 10.0  # If loss > 10x best, consider it exploded
    
    for idx in range(conf.n_iter):
        try:
            i = idx + start_iter
            
            # Get batch - recreate iterator if exhausted
            try:
                batch = next(train_iter)
            except StopIteration:
                # Dataset exhausted, create new iterator for next epoch
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            y_broad = batch['broadband'].to(device)
            x_low = batch['lowfreq'].to(device)
            pga_broad_true = batch['pga_broadband'].to(device)
            
            # === Train Signal DiT ===
            optimizer.zero_grad()

            # Diffuse broadband signal - get noisy input and target
            xt, sigma, noise_target, signal_target = diffusion.diffuse(y_broad)
            
            # Forward pass with conditioning (predicts noise) - use sigma as timestep embedding
            predicted_noise = model(xt, sigma, x_low)
            
            # 🔍 DIAGNOSTICS - Check every 100 iterations  
            if i % 100 == 0:
                print(f"🔍 DIAGNOSTICS (iter {i}):")
                print(f"   Target clean signal: [{y_broad.min():.3f}, {y_broad.max():.3f}]")
                print(f"   Conditioning signal: [{x_low.min():.3f}, {x_low.max():.3f}]")
                print(f"   Noise level σ: [{sigma.min():.3f}, {sigma.max():.3f}]")
                print(f"   Noisy input: [{xt.min():.3f}, {xt.max():.3f}]")
            
            # SIMPLE CORRECT TRAINING: Just predict the clean signal directly!
            # 
            # The model takes:
            # - Noisy input xt (signal + noise)
            # - Noise level sigma
            # - Conditioning x_low
            # 
            # And predicts the CLEAN SIGNAL directly (v-prediction / x0-prediction)
            # This is simpler and actually works!
            
            sigma_expanded = sigma.view(-1, 1)
            
            # Model predicts CLEAN SIGNAL (not noise!)
            predicted_clean = model(xt, sigma, x_low)
            
            # SINGLE OBJECTIVE: Predict clean signal
            # With strong weighting on envelope matching
            signal_loss = signal_loss_fn(predicted_clean, y_broad)
            
            # Add envelope consistency loss
            predicted_lowfreq = apply_lowpass_filter(predicted_clean, cutoff_freq=1.0, sample_rate=conf.sample_rate)
            envelope_loss = signal_loss_fn(predicted_lowfreq, x_low)
            
            # Combined loss with envelope emphasis
            signal_loss = 0.6 * signal_loss + 0.4 * envelope_loss
            
            current_denoising_loss = signal_loss.item()
            
            # CRITICAL: Check for loss explosion BEFORE backward pass
            current_loss_value = signal_loss.item()
            
            # Detect catastrophic loss explosion
            if current_loss_value > 1000.0:
                print(f"🚨 CATASTROPHIC LOSS EXPLOSION: {current_loss_value:.1f} at iter {i}")
                print(f"   Rolling back to best checkpoint and reducing learning rate...")
                
                # Load best checkpoint
                if os.path.exists(best_ckpt):
                    ckpt = torch.load(best_ckpt, map_location=device)
                    model.load_state_dict(ckpt['model'])
                    ema.load_state_dict(ckpt['ema'])
                    optimizer.load_state_dict(ckpt['optimizer'])
                    print(f"   Restored checkpoint from iter {ckpt['iter']}")
                
                # Reduce learning rate dramatically
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * 0.1
                    print(f"   Reduced LR: {old_lr:.6f} → {param_group['lr']:.6f}")
                
                optimizer.zero_grad()
                continue
            
            signal_loss.backward()
            
            # CRITICAL: Check for NaN/Inf in loss AFTER backward
            if not torch.isfinite(signal_loss):
                print(f"❌ WARNING: Loss is NaN/Inf at iter {i}! Skipping update.")
                optimizer.zero_grad()
                continue
            
            # Calculate gradient norm BEFORE clipping for monitoring
            total_grad_norm = 0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    if not np.isfinite(param_norm):
                        print(f"❌ WARNING: NaN/Inf gradient detected at iter {i}! Skipping update.")
                        optimizer.zero_grad()
                        continue
                    total_grad_norm += param_norm ** 2
                    param_count += 1
            total_grad_norm = total_grad_norm ** 0.5
            
            # CRITICAL: Aggressive gradient clipping to prevent explosion
            # If gradients are huge (>10), clip to 5.0; otherwise clip to 10.0
            max_grad_norm = 5.0 if total_grad_norm > 10.0 else 10.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # Recompute grad norm after clipping
            clipped_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    clipped_grad_norm += param.grad.data.norm(2).item() ** 2
            clipped_grad_norm = clipped_grad_norm ** 0.5
            
            # Store a parameter value before update to check if it changes
            first_param_before = None
            for param in model.parameters():
                first_param_before = param.data.clone()
                break
            
            optimizer.step()
            
            # Check if parameter actually changed
            first_param_after = None
            for param in model.parameters():
                first_param_after = param.data.clone()
                break
            
            param_change = (first_param_after - first_param_before).abs().max().item() if first_param_before is not None else 0
            
            # Check EMA model state before and after update
            # Update EMA model
            update_ema(ema, model, conf.ema_decay)
            
            running_signal_loss += signal_loss.item()
            running_envelope_loss = running_envelope_loss + envelope_loss.item() if 'running_envelope_loss' in locals() else envelope_loss.item()
            
            # Quick validation every 10 iterations to check if fixes are working
            if i % 10 == 0 and i > 0:
                with torch.no_grad():
                    # Check prediction quality - how well does predicted clean match target
                    signal_correlation = torch.corrcoef(torch.stack([predicted_clean.flatten(), y_broad.flatten()]))[0, 1]
                    # Check ENVELOPE matching quality
                    envelope_correlation = torch.corrcoef(torch.stack([predicted_lowfreq.flatten(), x_low.flatten()]))[0, 1]
                    pred_range = (predicted_clean.min().item(), predicted_clean.max().item())
                    target_range = (y_broad.min().item(), y_broad.max().item())
                    
                    # CRITICAL: Check for model weight explosion
                    max_weight = max(p.abs().max().item() for p in model.parameters())
                    if max_weight > 100.0:
                        print(f"⚠️ WARNING: Model weights exploding! Max weight: {max_weight:.1f}")
                    
                    print(f"  Quick check - Loss: {signal_loss.item():.4f}, Env: {envelope_loss.item():.4f}, "
                          f"Signal corr: {signal_correlation.item():.3f}, Envelope corr: {envelope_correlation.item():.3f}, "
                          f"Pred: [{pred_range[0]:.2f},{pred_range[1]:.2f}], Target: [{target_range[0]:.2f},{target_range[1]:.2f}]")
            
        # === Train PGA Predictor ===
        # TEMPORARILY DISABLED DUE TO MEMORY ISSUES
        # TODO: Fix PGA predictor memory consumption
        # pga_optimizer.zero_grad()
        
        # # Use EMA model to generate samples for PGA training
        # ema.eval()
        # # Generate normalized broadband signals
        # sz = (conf.pga_batch_size, conf.seq_len)
        # if y_broad.shape[0] >= conf.pga_batch_size:
        #     x_low_pga = x_low[:conf.pga_batch_size]
        #     pga_true_batch = pga_broad_true[:conf.pga_batch_size]
        #     
        #     # Generate samples (these will be normalized)
        #     with torch.no_grad():
        #         generated_broad = diffusion.sample(
        #             ema, sz, steps=20,  # Use fewer steps for training efficiency
        #             x_cond=x_low_pga
        #         )
        #     generated_broad = generated_broad.to(device)
        #     
        #     # Predict PGA (outside no_grad to allow gradients)
        #     pga_pred = pga_predictor(x_low_pga, generated_broad.detach())
        #     
        #     # Compute PGA loss
        #     pga_total_loss, pga_mse, pga_l1 = pga_loss_fn(pga_pred, pga_true_batch)
        #     pga_total_loss.backward()
        #     
        #     pga_optimizer.step()
        #     running_pga_loss += pga_total_loss.item()
        # else:
        #     # Skip PGA training if batch is too small
        #     pass            # Enhanced logging with loss components
            if i % log_interval == 0:
                elapsed = time.time() - start_time
                
                # Calculate number of actual iterations that contributed to running losses
                actual_iterations = min(i + 1, log_interval) if i == 0 else log_interval
                
                avg_signal_loss = running_signal_loss / actual_iterations
                avg_pga_loss = running_pga_loss / actual_iterations
                
                # Log loss components for analysis - Simple direct prediction
                avg_envelope_loss = running_envelope_loss / actual_iterations
                
                print(f'Iter {i:07d} | Time: {elapsed:.2f}s | '
                      f'Total: {avg_signal_loss:.6f} | '
                      f'Envelope: {avg_envelope_loss:.6f} | '
                      f'Grad: {total_grad_norm:.6f}')
                
                # Tensorboard logging - Simple metrics
                writer.add_scalar('Loss/Total_Signal', avg_signal_loss, i)
                writer.add_scalar('Loss/Envelope_Matching', avg_envelope_loss, i)
                writer.add_scalar('Loss/PGA', avg_pga_loss, i)
                writer.add_scalar('Training/Grad_Norm', total_grad_norm, i)
                writer.add_scalar('Training/Grad_Norm_Clipped', clipped_grad_norm, i)
                writer.add_scalar('Learning_Rate/Signal', optimizer.param_groups[0]['lr'], i)
                
                # CRITICAL: Update learning rate scheduler based on loss
                scheduler.step(avg_signal_loss)
                if train_pga and pga_optimizer is not None:
                    writer.add_scalar('Learning_Rate/PGA', pga_optimizer.param_groups[0]['lr'], i)
                writer.flush()
                
                running_signal_loss = 0.0
                running_envelope_loss = 0.0
                running_pga_loss = 0.0
                start_time = time.time()
            
            # Evaluation and visualization  
            if i % conf.eval_interval == 0:
                print("Generating evaluation samples...")
                ema.eval()
                if train_pga and pga_predictor is not None:
                    pga_predictor.eval()
                
                with torch.no_grad():
                    # Use fixed evaluation signals for consistent visualization
                    n_vis = min(7, len(fixed_eval_lowpass))  # Use available samples
                    sz = (n_vis, conf.seq_len)
                    
                    # Use fixed conditioning signals (same across all evaluations)
                    x_cond_vis = fixed_eval_lowpass[:n_vis].to(device)
                    
                    # If we have fewer samples than n_vis, repeat them to match
                    if len(x_cond_vis) < n_vis:
                        repeats = (n_vis + len(x_cond_vis) - 1) // len(x_cond_vis)  # Ceiling division
                        x_cond_vis = x_cond_vis.repeat(repeats, 1)[:n_vis]
                    
                    # Generate signals - DON'T use fixed seed for evaluation to see model changes
                    generated_signals = diffusion.sample(
                        ema, sz, steps=conf.steps,
                        x_cond=x_cond_vis, seed=None
                    ).to(device)
                    
                    # Create visualization with original unnormalized broadband signals for comparison
                    plot_path = os.path.join(log_img_dir, f'samples_{i:07d}.png')
                    plot_signal_samples(
                        generated_signals, x_cond_vis, plot_path, 
                        sample_rate=conf.sample_rate, n_samples=n_vis,
                        original_broad=fixed_eval_original.to(device)
                    )
                    
                    # Evaluate signal quality against fixed true signals
                    real_signals = fixed_eval_broadband.to(device)
                    quality_metrics = evaluate_signal_quality(
                        generated_signals, real_signals, conf.sample_rate
                    )
                    
                    # Calculate comparison loss between generated and original normalized signals
                    # Both signals are PGA-normalized, so direct comparison is meaningful
                    original_normalized = fixed_eval_broadband.to(device)  # Already PGA normalized from dataset
                    
                    # MSE loss between generated and original normalized broadband signals
                    broadband_comparison_loss = torch.nn.functional.mse_loss(
                        generated_signals, original_normalized
                    ).item()
                    
                    # L1 loss for additional comparison
                    broadband_l1_loss = torch.nn.functional.l1_loss(
                        generated_signals, original_normalized
                    ).item()
                    
                    # Print comparison losses to terminal
                    print(f'         Generated vs Original - MSE: {broadband_comparison_loss:.6f} | '
                          f'L1: {broadband_l1_loss:.6f}')
                    
                    # Log quality metrics
                    for metric_name, metric_value in quality_metrics.items():
                        writer.add_scalar(f'Quality/{metric_name}', metric_value, i)
                    
                    # Log comparison losses to TensorBoard
                    writer.add_scalar('Comparison/Generated_vs_Original_MSE', broadband_comparison_loss, i)
                    writer.add_scalar('Comparison/Generated_vs_Original_L1', broadband_l1_loss, i)
                    
                    # Enhanced signal quality metrics for early stopping
                    with torch.no_grad():
                        # 1. Amplitude envelope correlation
                        gen_envelope = torch.abs(generated_signals)
                        orig_envelope = torch.abs(original_normalized)
                        envelope_corr = torch.corrcoef(torch.stack([
                            gen_envelope.flatten(), orig_envelope.flatten()
                        ]))[0,1]
                        
                        # 2. Peak timing accuracy
                        gen_peaks = torch.argmax(torch.abs(generated_signals), dim=1) 
                        orig_peaks = torch.argmax(torch.abs(original_normalized), dim=1)
                        peak_timing_error = torch.mean(torch.abs(gen_peaks - orig_peaks).float())
                        
                        # 3. Conditioning consistency
                        gen_lowfreq = apply_lowpass_filter(generated_signals, cutoff_freq=1.0, sample_rate=conf.sample_rate)
                        cond_consistency = torch.nn.functional.mse_loss(gen_lowfreq, x_cond_vis)
                        
                        # 4. Signal-to-noise ratio
                        signal_power = torch.mean(generated_signals ** 2)
                        noise_power = torch.mean((generated_signals - original_normalized) ** 2) 
                        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                        
                        # 5. Composite quality score
                        quality_score = (
                            0.4 * envelope_corr +
                            0.3 * max(0, 1 - peak_timing_error/1000) +  # Normalize timing error
                            0.2 * max(0, 1 - cond_consistency) +        # Lower is better
                            0.1 * min(1, snr.item()/20)                 # Normalize SNR
                        )
                        
                        # Log enhanced metrics
                        writer.add_scalar('Quality/Envelope_Correlation', envelope_corr.item(), i)
                        writer.add_scalar('Quality/Peak_Timing_Error', peak_timing_error.item(), i)
                        writer.add_scalar('Quality/Conditioning_Consistency', cond_consistency.item(), i)
                        writer.add_scalar('Quality/SNR_dB', snr.item(), i)
                        writer.add_scalar('Quality/Composite_Score', quality_score, i)
                        
                        print(f'         Quality - Envelope: {envelope_corr.item():.3f} | '
                              f'Peak Error: {peak_timing_error.item():.1f} | '
                              f'SNR: {snr.item():.1f}dB | Score: {quality_score:.3f}')
                        
                        # Early stopping based on signal quality
                        if quality_score > best_quality:
                            best_quality = quality_score
                            quality_patience_counter = 0
                            # Save best quality model
                            best_quality_ckpt = os.path.join(model_dir, 'best_quality_ckpt.pt')
                            torch.save({
                                'iter': i,
                                'model': model.state_dict(),
                                'ema': ema.state_dict(),
                                'quality_score': quality_score,
                                'config': signal_config
                            }, best_quality_ckpt)
                            print(f'         New best quality: {quality_score:.3f} - saved checkpoint')
                        elif i > 2000 and quality_score < best_quality - 0.05:
                            quality_patience_counter += 1
                            print(f'         Quality degrading ({quality_patience_counter}/{quality_patience})')
                            if not disable_early_stop and quality_patience_counter >= quality_patience:
                                print("Signal quality consistently degrading - implementing early stopping")
                                early_stop = True
                                break
                    
                    # PGA prediction on generated signals - DISABLED
                    # pga_predictions = pga_predictor(x_cond_vis, generated_signals)
                    # true_pga = SignalMetrics.compute_pga(generated_signals)
                    # pga_error = torch.mean(torch.abs(pga_predictions.squeeze() - true_pga))
                    # writer.add_scalar('Quality/PGA_Error', pga_error.item(), i)
                    
                    writer.flush()
                
                model.train()
                # pga_predictor.train()  # DISABLED
            
            # Save checkpoint - MORE FREQUENT during critical learning phase
            # Save every 100 iters during first 2000 iterations (when loss is unstable)
            # Then save every checkpoint_interval as normal
            should_save = (i % 100 == 0 and i < 2000) or (i % conf.checkpoint_interval == 0)
            
            if should_save:
                # Calculate current average losses for checkpoint
                current_signal_loss = running_signal_loss / max(1, i % log_interval if i % log_interval != 0 else log_interval)
                current_pga_loss = 0.0  # PGA training disabled
                total_loss = current_signal_loss + current_pga_loss
                
                ckpt_data = {
                    'iter': i,
                    'model': model.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'pga_predictor': pga_predictor.state_dict(),  # DISABLED
                    # 'pga_optimizer': pga_optimizer.state_dict(),  # DISABLED
                    'signal_loss': current_signal_loss,
                    'pga_loss': current_pga_loss,
                    'total_loss': total_loss,
                    'best_loss': min(total_loss, best_loss),
                    'config': conf.__dict__
                }
                
                torch.save(ckpt_data, last_ckpt)
                
                # Save best model
                if total_loss < best_loss:
                    best_loss = total_loss
                    torch.save(ckpt_data, best_ckpt)
                    print(f'New best model saved at iter {i} with loss {best_loss:.6f}')
                
                print(f'Checkpoint saved at iter {i}')
                
            # Check for early stopping
            if early_stop:
                print(f"Early stopping triggered at iteration {i}")
                break
                
        except Exception as e:
            print(f"ERROR in training loop at iteration {i}: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    if early_stop:
        print(f"Training terminated early due to quality degradation!")
    else:
        print(f"Training completed after {conf.n_iter} iterations!")
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train Signal Diffusion Transformer')
    parser.add_argument('--model_dir', type=str, default='signal_model_v1',
                       help='Directory to save model')
    parser.add_argument('--data_dir', type=str, default='data_prep_acc/processed_dynamic',
                       help='Directory containing training signals (.npz files)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                       help='Evaluation interval')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--process_at2', action='store_true',
                       help='Process AT2 files before training')
    parser.add_argument('--at2_input_dir', type=str,
                       help='Directory containing AT2 files (if processing)')
    
    args = parser.parse_args()
    
    # Process AT2 files if requested
    if args.process_at2:
        if not args.at2_input_dir:
            raise ValueError("--at2_input_dir must be specified when using --process_at2")
        
        print("Processing AT2 files...")
        preprocess_at2_files_for_training(args.at2_input_dir, args.data_dir)
        print("AT2 processing complete!")
    
    # Load configuration with dynamic seq_len resolution
    resolved_config = {}
    for key, value in signal_config.items():
        resolved_config[key] = get_config_value(key, signal_config, args.data_dir)
    
    # CRITICAL: Validate model dimensions to prevent shape errors
    from signal_config import validate_model_dimensions
    resolved_config = validate_model_dimensions(resolved_config)
    
    print(f"📊 Configuration loaded:")
    print(f"   seq_len: {resolved_config['seq_len']:,} samples")
    print(f"   sample_rate: {resolved_config['sample_rate']} Hz")
    print(f"   patch_size: {resolved_config['patch_size']} (num_patches: {resolved_config['seq_len']//resolved_config['patch_size']})")
    print(f"   dim: {resolved_config['dim']}, heads: {resolved_config['heads']} (head_dim: {resolved_config['dim']//resolved_config['heads']})")
    if resolved_config['seq_len'] != signal_config['seq_len']:
        print(f"   ✅ Dynamic seq_len resolved from dataset")
    
    conf = Config(resolved_config, args.model_dir)
    
    # REMOVED HARMFUL OVERRIDE: batch_size=1 prevents proper learning
    # Proper batch training enables gradient averaging and stable BatchNorm
    print(f"✅ Using proper batch_size: {conf.batch_size} (enables gradient averaging)")
    
    # Start training
    train_signal_dit(
        args.model_dir,
        args.data_dir, 
        args.eval_interval,
        args.log_interval,
        conf
    )

if __name__ == '__main__':
    main()
