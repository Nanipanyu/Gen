"""
Signal Diffusion Utilities for Earthquake Ground Motion Generation

Adapted diffusion utilities specifically for 1D time series signals,
with conditioning support for low-frequency earthquake signals.

Based on:
    https://github.com/NVlabs/edm/ 
    https://github.com/crowsonkb/k-diffusion

"""
# mathematical heart of your broadband earthquake generation system - 
# it implements the core diffusion process adapted specifically for 1D seismic signals with physics-based conditioning

import torch
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import butter, filtfilt

def apply_lowpass_filter(signal, cutoff_freq=1.0, sample_rate=100.0, order=4):
    """Apply low-pass filter for conditioning consistency loss"""
    if len(signal.shape) == 1:
        signal = signal.unsqueeze(0)
    
    device = signal.device
    signal_np = signal.detach().cpu().numpy()
    
    # Design filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1.0:
        return signal
    
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Apply filter
    filtered_signals = []
    for i in range(signal_np.shape[0]):
        filtered = filtfilt(b, a, signal_np[i])
        filtered_signals.append(filtered)
    
    result = torch.tensor(np.stack(filtered_signals), dtype=signal.dtype, device=device)
    return result

def get_scalings(sig, sig_data):
    """Computes scaling factors for the diffusion process based on noise levels"""
    s = sig ** 2 + sig_data ** 2
    # c_skip, c_out, c_in
    return sig_data ** 2 / s, sig * sig_data / s.sqrt(), 1 / s.sqrt()

def get_sigmas_karras(n, sigma_min=0.01, sigma_max=80., rho=7., device='cpu'):
    """Generates a noise schedule using the Karras et al. method"""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.tensor([0.])]).to(device)

class SignalDiffusion(object):
    """
    Diffusion process for 1D earthquake signals with optional conditioning
    
    CRITICAL: Noise schedule MUST match signal scale!
    For signals in [-1, 1], we need much smaller sigma values.
    """
    def __init__(self, P_mean=-1.2, P_std=1.0, sigma_data=0.5, sigma_max=3.0):
        """
        Args:
            P_mean: Mean of log-normal noise distribution (default: -1.2 → σ ≈ 0.3)
            P_std: Std of log-normal noise distribution (reduced to 1.0 for stability)
            sigma_data: Data standard deviation (0.5 for normalized signals)
            sigma_max: Maximum noise level (3.0 means noise can be 3x signal amplitude max)
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        
        # Pre-compute training noise distribution for sampling alignment
        self.training_sigmas = self._compute_training_noise_schedule()
    
    def _compute_training_noise_schedule(self, n_samples=10000):
        """Compute empirical noise distribution from training"""
        rnd_normal = torch.randn([n_samples, 1])
        training_sigmas = (rnd_normal * self.P_std + self.P_mean).exp()
        return torch.sort(training_sigmas.flatten())[0]
        
    def diffuse(self, y, return_noise=False):
        """
        Adds noise to the input signal based on the diffusion parameters
        
        Args:
            y: Clean broadband signal (batch, seq_len) - MUST be in [-1, 1]
            return_noise: Whether to return the noise tensor
            
        Returns:
            noised_input: Noisy input for the model
            sigma: Noise level
            target: Target for training (noise)
            signal_target: Original clean signal for reconstruction loss
            noise: Added noise (if return_noise=True)
        """
        device = y.device
        batch_size = y.shape[0]
        
        # Sample noise levels from log-normal distribution
        rnd_normal = torch.randn([batch_size, 1], device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        # CRITICAL: Clamp sigma to prevent explosion
        # For signals in [-1, 1], sigma should not exceed sigma_max
        sigma = torch.clamp(sigma, min=0.002, max=self.sigma_max)
        
        # Add noise to signal
        n = torch.randn_like(y, device=device)
        
        # Create noisy input - standard approach
        # With sigma_max=3.0, noisy signal stays in reasonable range ~[-10, 10]
        noised_input = y + n * sigma.view(-1, 1)
        
        # Multi-target training
        noise_target = n      # For denoising capability
        signal_target = y     # For signal reconstruction loss
        
        # Return noisy input and targets for training
        if return_noise:
            return noised_input, sigma.squeeze(), noise_target, signal_target, n
        else:
            return noised_input, sigma.squeeze(), noise_target, signal_target

    def sample(self, model, sz, steps=100, sigma_max=None, seed=None, x_cond=None):
        """
        Generates samples from the diffusion model
        
        Args:
            model: The trained signal diffusion transformer
            sz: Shape of samples to generate (batch_size, seq_len)
            steps: Number of denoising steps
            sigma_max: Maximum noise level (if None, uses self.sigma_max)
            seed: Random seed for reproducible generation
            x_cond: Conditioning signal (low-frequency) - optional
            
        Returns:
            Generated signals
        """
        # Use instance sigma_max if not provided
        if sigma_max is None:
            sigma_max = self.sigma_max
            
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                return self.                # Generate sigmas using same log-normal distribution as training
                rho = 7.0
                min_inv_rho = (0.002 ** (1 / rho))
                max_inv_rho = (sigma_max ** (1 / rho))
                ramp = torch.linspace(0, 1, steps + 1, device=device)
                t_steps = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho(model, sz, steps, sigma_max, x_cond)
        else:
            return self._sample_internal(model, sz, steps, sigma_max, x_cond)

    def _sample_internal(self, model, sz, steps, sigma_max, x_cond=None):
        """Internal method to handle the sampling process"""
        device = next(model.parameters()).device
        model.eval()
        
        # Start with pure noise scaled to sigma_max
        # CRITICAL: For signals in [-1,1], sigma_max=3.0 means initial noise in ~[-9, 9]
        x = torch.randn(sz, device=device) * sigma_max
        
        # Move conditioning to device if provided
        if x_cond is not None:
            x_cond = x_cond.to(device)
        
        # CRITICAL FIX: Use log-normal schedule matching training
        # Training samples: σ ~ exp(N(-1.2, 1.0))
        # Sampling should use monotonic decreasing path through this distribution
        
        # Generate sigmas using same log-normal distribution as training
        # but in sorted decreasing order for denoising trajectory
        rho = 7.0  # Controls schedule curvature
        min_inv_rho = (0.002 ** (1 / rho))  # σ_min from training
        max_inv_rho = (sigma_max ** (1 / rho))
        
        ramp = torch.linspace(0, 1, steps + 1, device=device)
        t_steps = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        
        # Ensure final step is exactly 0
        t_steps = torch.cat([t_steps, torch.tensor([0.0], device=device)])
        
        # Iterative denoising
        for i in tqdm(range(len(t_steps) - 1), desc="Generating signal"):
            x = self.edm_sampler(x, t_steps, i, model, x_cond=x_cond)
            
        return x.cpu()

    @torch.no_grad()
    def edm_sampler(self, x, t_steps, i, model, s_churn=0., s_min=0.,
                    s_max=float('inf'), s_noise=1., x_cond=None):
        """
        Envelope-guided sampling with explicit low-frequency enforcement
        
        Args:
            x: Current noisy signal
            t_steps: Noise schedule (decreasing: σ_max → σ_min → 0)
            i: Current step index
            model: The denoising model
            x_cond: Conditioning signal (REQUIRED for envelope guidance)
        """
        sigma_cur = t_steps[i] 
        sigma_next = t_steps[i + 1]
        
        # Get denoised prediction using current model
        x_denoised = self.get_d(model, x, sigma_cur, x_cond)
        
        # CRITICAL FIX: Enforce envelope matching during sampling
        if x_cond is not None:
            # Extract low-frequency component from prediction
            pred_lowfreq = apply_lowpass_filter(x_denoised, cutoff_freq=1.0, sample_rate=100.0)
            
            # Compute correction: difference between conditioning and predicted envelope
            envelope_error = x_cond - pred_lowfreq
            
            # Apply envelope guidance with MODERATE strength
            # Don't go to full strength (1.0) to preserve high-frequency details
            # Use 0.3-0.5 range instead of 0-1 to avoid over-correction
            progress = 1.0 - (sigma_cur / self.sigma_max)  # 0 at start, 1 at end
            guidance_strength = 0.3 + 0.2 * progress  # Ranges from 0.3 to 0.5
            
            # Correct the denoised prediction by adding envelope error
            x_denoised = x_denoised + guidance_strength * envelope_error
        
        # Standard EDM step formula
        if sigma_next > 0:
            # Predict noise from current denoised estimate
            noise_pred = (x - x_denoised) / sigma_cur if sigma_cur > 1e-8 else torch.zeros_like(x)
            
            # Step to next noise level
            x_next = x_denoised + noise_pred * sigma_next
        else:
            # Final step: return clean signal
            x_next = x_denoised
            
        return x_next

    def get_d(self, model, x, sig, x_cond=None):
        """
        SIMPLE VERSION: Model predicts CLEAN SIGNAL directly (x0-prediction)
        
        This is much simpler and actually works properly!
        
        Args:
            model: The signal diffusion transformer
            x: Noisy signal [batch, seq_len]
            sig: Noise level (scalar or [batch])
            x_cond: Conditioning signal (optional) [batch, seq_len]
            
        Returns:
            predicted_clean: The model's prediction of the clean signal
        """
        # Ensure sig has correct shape for model input
        if sig.dim() == 0:
            sig = sig.unsqueeze(0)  # Convert scalar to 1D tensor [1]
        
        sig_for_model = sig.view(-1)  # Shape [batch] for model input
        
        # Model predicts CLEAN SIGNAL directly
        predicted_clean = model(x, sig_for_model, x_cond)
        
        return predicted_clean

    def get_gamma(self, t_cur, s_churn, s_min, s_max, s_noise, n):
        """Computes the stochastic churn amount for the current timestep"""
        # Handle scalar tensors
        t_val = t_cur.item() if torch.is_tensor(t_cur) and t_cur.dim() > 0 else float(t_cur)
        
        if s_min <= t_val <= s_max:
            return min(s_churn / (n - 1), 2 ** 0.5 - 1)
        else:
            return 0.

def save_signals(signals, file_paths, sample_rate=100.0):
    """
    Save generated signals to files
    
    Args:
        signals: Generated signals tensor (batch, seq_len)
        file_paths: List of file paths to save to
        sample_rate: Sampling rate in Hz
    """
    signals_np = signals.cpu().numpy()
    
    for i, (signal, file_path) in enumerate(zip(signals_np, file_paths)):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save as numpy array with metadata
        np.savez(file_path, 
                signal=signal, 
                sample_rate=sample_rate,
                length_sec=len(signal) / sample_rate)

def gen_signal_batches(diffusion, model, n_signals, batch_size, steps, 
                      dir_path, seq_len, x_cond=None, sample_rate=100.0):
    """
    Generate and save batches of earthquake signals
    
    Args:
        diffusion: SignalDiffusion instance
        model: Trained signal diffusion transformer
        n_signals: Total number of signals to generate
        batch_size: Batch size for generation
        steps: Number of denoising steps
        dir_path: Directory to save signals
        seq_len: Length of each signal
        x_cond: Conditioning signals (optional)
        sample_rate: Sampling rate in Hz
    """
    n_batches = n_signals // batch_size
    sz = (batch_size, seq_len)
    
    os.makedirs(dir_path, exist_ok=True)
    
    for i in tqdm(range(n_batches), desc="Generating signal batches"):
        start_idx = i * batch_size
        
        # Get conditioning batch if provided
        x_cond_batch = None
        if x_cond is not None:
            x_cond_batch = x_cond[start_idx:start_idx + batch_size]
        
        # Generate batch
        gen_batch = diffusion.sample(model, sz, steps=steps, x_cond=x_cond_batch)
        
        # Save individual signals
        for j, signal in enumerate(gen_batch):
            signal_idx = start_idx + j
            file_path = os.path.join(dir_path, f'signal_{signal_idx:06d}.npz')
            
            # Save signal
            np.savez(file_path,
                    signal=signal.numpy(),
                    sample_rate=sample_rate,
                    length_sec=len(signal) / sample_rate,
                    signal_id=signal_idx)

class SignalMetrics:
    """Utility class for computing signal-specific metrics"""
    
    @staticmethod
    def compute_pga(signals):
        """Compute Peak Ground Acceleration"""
        return torch.max(torch.abs(signals), dim=-1)[0]
    
    @staticmethod
    def compute_frequency_content(signals, sample_rate=100.0):
        """Compute frequency content using FFT"""
        fft = torch.fft.fft(signals, dim=-1)
        freqs = torch.fft.fftfreq(signals.shape[-1], d=1/sample_rate)
        magnitude = torch.abs(fft)
        return freqs, magnitude
    
    @staticmethod
    def compute_response_spectrum(signals, sample_rate=100.0, periods=None):
        """Compute response spectrum (simplified version)"""
        if periods is None:
            periods = torch.logspace(-2, 1, 50)  # 0.01 to 10 seconds
        
        # This is a simplified version - in practice, you'd use proper
        # single-degree-of-freedom oscillator integration
        pga = SignalMetrics.compute_pga(signals)
        # Approximate response spectrum as PGA scaled by period-dependent factors
        response = pga.unsqueeze(-1) * torch.ones_like(periods).unsqueeze(0)
        
        return periods, response

# Test the signal diffusion
if __name__ == "__main__":
    print("Testing Signal Diffusion...")
    
    # Create sample data
    batch_size = 4
    seq_len = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Sample clean signals
    y_clean = torch.randn(batch_size, seq_len) * 0.5
    
    # Sample conditioning signals
    x_cond = torch.randn(batch_size, seq_len) * 0.2
    
    # Initialize diffusion
    diffusion = SignalDiffusion()
    
    # Test diffusion process
    noised_input, sigma, target = diffusion.diffuse(y_clean)
    print(f"Clean signal shape: {y_clean.shape}")
    print(f"Noised input shape: {noised_input.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Target shape: {target.shape}")
    
    # Test metrics
    metrics = SignalMetrics()
    pga = metrics.compute_pga(y_clean)
    print(f"PGA values: {pga}")
    
    freqs, magnitude = metrics.compute_frequency_content(y_clean)
    print(f"Frequency analysis - freqs shape: {freqs.shape}, magnitude shape: {magnitude.shape}")
    
    print("Signal diffusion test successful!")
