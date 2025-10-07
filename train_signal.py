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
from signal_config import signal_config
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
        sample_rate=conf.sample_rate
    )
    
    # Convert to iterator for infinite sampling
    train_iter = iter(train_loader)
    
    # Create fixed evaluation dataset - use first batch for consistent visualization
    print("Creating fixed evaluation dataset...")
    eval_batch = next(iter(train_loader))
    
    # Debug: Check the structure of eval_batch
    print(f"Eval batch keys: {eval_batch.keys()}")
    print(f"Metadata type: {type(eval_batch['metadata'])}")
    if isinstance(eval_batch['metadata'], dict):
        print(f"Metadata dict keys: {eval_batch['metadata'].keys()}")
        # Check if file_path exists and its structure
        if 'file_path' in eval_batch['metadata']:
            fp = eval_batch['metadata']['file_path']
            print(f"File_path type: {type(fp)}, length: {len(fp) if hasattr(fp, '__len__') else 'N/A'}")
    
    fixed_eval_lowpass = eval_batch['lowfreq'][:7]  # Fixed 7 low-pass signals
    fixed_eval_broadband = eval_batch['broadband'][:7]  # Corresponding true broadband (PGA normalized)
    
    # Get the file paths from the evaluation batch metadata to load the SAME original signals
    print("Loading original unnormalized signals for the SAME files as in evaluation batch...")
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
    
    # Initialize diffusion
    diffusion = SignalDiffusion(
        P_mean=conf.P_mean,
        P_std=conf.P_std,
        sigma_data=conf.sigma_data
    )
    
    # Initialize PGA predictor
    pga_predictor = PGAPredictor(
        seq_len=conf.seq_len,
        cnn_filters=conf.cnn_filters,
        lstm_hidden=conf.lstm_hidden,
        dropout=conf.pga_dropout
    ).to(device)
    
    # Optimizers
    optimizer = optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=0.01)
    pga_optimizer = optim.AdamW(pga_predictor.parameters(), lr=conf.pga_lr, weight_decay=0.01)
    
    # Loss functions
    signal_loss_fn = nn.MSELoss()
    pga_loss_fn = PGALoss()
    
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
    quality_patience = 5  # Stop if quality degrades for 5 evaluations
    quality_patience_counter = 0
    early_stop = False
    
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
    else:
        print("Starting new training")
    
    # Training loop
    print("Starting training...")
    model.train()
    ema.eval()
    pga_predictor.train()
    
    running_signal_loss = 0.0
    running_pga_loss = 0.0
    start_time = time.time()
    
    for idx in range(conf.n_iter):
        try:
            i = idx + start_iter
            
            # Get batch
            batch = next(train_iter)
            y_broad = batch['broadband'].to(device)
            x_low = batch['lowfreq'].to(device)
            pga_broad_true = batch['pga_broadband'].to(device)
            
            # === Train Signal DiT with Multi-Objective Loss ===
            optimizer.zero_grad()
            
            # Enhanced diffusion process with multi-objective targets
            xt, t, noise_target, signal_target = diffusion.diffuse(y_broad)
            
            # Forward pass with conditioning (predicts noise)
            predicted_noise = model(xt, t, x_low)
            
            # Multi-objective loss computation
            # Loss 1: Denoising loss (standard diffusion)
            denoising_loss = signal_loss_fn(predicted_noise, noise_target)
            
            # Loss 2: Signal reconstruction loss
            sigma_expanded = t.view(-1, 1)
            reconstructed_signal = xt - predicted_noise * sigma_expanded
            reconstruction_loss = signal_loss_fn(reconstructed_signal, signal_target)
            
            # Loss 3: Conditioning consistency loss
            reconstructed_lowfreq = apply_lowpass_filter(reconstructed_signal, cutoff_freq=1.0, sample_rate=conf.sample_rate)
            conditioning_loss = signal_loss_fn(reconstructed_lowfreq, x_low)
            
            # Progressive training weights based on iteration
            if i < 1000:
                # Phase 1: Learn basic denoising
                w_denoise, w_reconstruct, w_condition = 1.0, 0.0, 0.1
            elif i < 2500:
                # Phase 2: Add signal reconstruction
                w_denoise, w_reconstruct, w_condition = 0.8, 0.4, 0.3
            else:
                # Phase 3: Emphasize signal quality
                w_denoise, w_reconstruct, w_condition = 0.6, 0.8, 0.5
            
            # Combined loss
            signal_loss = (
                w_denoise * denoising_loss +
                w_reconstruct * reconstruction_loss +
                w_condition * conditioning_loss
            )
            
            signal_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            update_ema(ema, model, conf.ema_decay)
            
            running_signal_loss += signal_loss.item()
            
            # Quick validation every 10 iterations to check if fixes are working
            if i % 10 == 0 and i > 0:
                with torch.no_grad():
                    # Check prediction quality
                    correlation = torch.corrcoef(torch.stack([predicted_noise.flatten(), noise_target.flatten()]))[0, 1]
                    noise_range = (noise_target.min().item(), noise_target.max().item())
                    pred_range = (predicted_noise.min().item(), predicted_noise.max().item())
                    
                    print(f"  Quick check - Loss: {signal_loss.item():.6f}, "
                          f"Correlation: {correlation.item():.4f}, "
                          f"Noise range: [{noise_range[0]:.3f}, {noise_range[1]:.3f}], "
                          f"Pred range: [{pred_range[0]:.3f}, {pred_range[1]:.3f}]")
            
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
                
                # Log individual loss components for analysis
                with torch.no_grad():
                    # Get current loss components for display
                    print(f'Iter {i:07d} | Time: {elapsed:.2f}s | '
                          f'Total: {avg_signal_loss:.6f} | '
                          f'Denoise: {denoising_loss.item():.6f} | '
                          f'Reconstruct: {reconstruction_loss.item():.6f} | '
                          f'Condition: {conditioning_loss.item():.6f} | '
                          f'Weights: D{w_denoise:.1f}/R{w_reconstruct:.1f}/C{w_condition:.1f}')
                
                # Tensorboard logging with detailed metrics
                writer.add_scalar('Loss/Total_Signal', avg_signal_loss, i)
                writer.add_scalar('Loss/Denoising', denoising_loss.item(), i)
                writer.add_scalar('Loss/Reconstruction', reconstruction_loss.item(), i)
                writer.add_scalar('Loss/Conditioning', conditioning_loss.item(), i)
                writer.add_scalar('Loss/PGA', avg_pga_loss, i)
                writer.add_scalar('Training/Weight_Denoising', w_denoise, i)
                writer.add_scalar('Training/Weight_Reconstruction', w_reconstruct, i)
                writer.add_scalar('Training/Weight_Conditioning', w_condition, i)
                writer.add_scalar('Learning_Rate/Signal', optimizer.param_groups[0]['lr'], i)
                writer.add_scalar('Learning_Rate/PGA', pga_optimizer.param_groups[0]['lr'], i)
                writer.flush()
                
                running_signal_loss = 0.0
                running_pga_loss = 0.0
                start_time = time.time()
            
            # Evaluation and visualization
            if i % conf.eval_interval == 0:
                print("Generating evaluation samples...")
                ema.eval()
                pga_predictor.eval()
                
                with torch.no_grad():
                    # Use fixed evaluation signals for consistent visualization
                    n_vis = 7
                    sz = (n_vis, conf.seq_len)
                    
                    # Use fixed conditioning signals (same across all evaluations)
                    x_cond_vis = fixed_eval_lowpass.to(device)
                    
                    # Generate signals
                    generated_signals = diffusion.sample(
                        ema, sz, steps=conf.steps,
                        x_cond=x_cond_vis, seed=conf.seed
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
                            if quality_patience_counter >= quality_patience:
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
            
            # Save checkpoint
            if i % conf.checkpoint_interval == 0:
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
    parser.add_argument('--data_dir', type=str, default='data_prep_acc/processed_signals',
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
    
    # Load configuration
    conf = Config(signal_config, args.model_dir)
    
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
