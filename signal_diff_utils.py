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
    signal_np = signal.cpu().numpy()
    
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
    """
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.66):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
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
            y: Clean broadband signal (batch, seq_len)
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
        
        # Sample noise levels
        rnd_normal = torch.randn([batch_size, 1], device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        # Add noise to signal
        n = torch.randn_like(y, device=device)
        
        # Create noisy input - standard approach
        noised_input = y + n * sigma.view(-1, 1)
        
        # Multi-target training
        noise_target = n      # For denoising capability
        signal_target = y     # For signal reconstruction loss
        
        # Return noisy input and targets for training
        if return_noise:
            return noised_input, sigma.squeeze(), noise_target, signal_target, n
        else:
            return noised_input, sigma.squeeze(), noise_target, signal_target

    def sample(self, model, sz, steps=100, sigma_max=80., seed=None, x_cond=None):
        """
        Generates samples from the diffusion model
        
        Args:
            model: The trained signal diffusion transformer
            sz: Shape of samples to generate (batch_size, seq_len)
            steps: Number of denoising steps
            sigma_max: Maximum noise level
            seed: Random seed for reproducible generation
            x_cond: Conditioning signal (low-frequency) - optional
            
        Returns:
            Generated signals
        """
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                return self._sample_internal(model, sz, steps, sigma_max, x_cond)
        else:
            return self._sample_internal(model, sz, steps, sigma_max, x_cond)

    def _sample_internal(self, model, sz, steps, sigma_max, x_cond=None):
        """Internal method to handle the sampling process"""
        device = next(model.parameters()).device
        model.eval()
        
        # Start with pure noise
        x = torch.randn(sz, device=device) * sigma_max
        
        # Move conditioning to device if provided
        if x_cond is not None:
            x_cond = x_cond.to(device)
        
        # Generate noise schedule - use training-aligned distribution
        indices = torch.linspace(0, len(self.training_sigmas)-1, steps+1).long()
        t_steps = self.training_sigmas[indices].to(device)
        t_steps = torch.cat([t_steps, torch.tensor([0.0], device=device)])
        
        # Iterative denoising
        for i in tqdm(range(len(t_steps) - 1), desc="Generating signal"):
            x = self.edm_sampler(x, t_steps, i, model, x_cond=x_cond)
            
        return x.cpu()

    @torch.no_grad()
    def edm_sampler(self, x, t_steps, i, model, s_churn=0., s_min=0.,
                    s_max=float('inf'), s_noise=1., x_cond=None):
        """
        Implements the EDM sampling algorithm (second-order solver) for denoising
        
        Args:
            x: Current noisy signal
            t_steps: Noise schedule
            i: Current step index
            model: The denoising model
            x_cond: Conditioning signal (optional)
        """
        n = len(t_steps)
        gamma = self.get_gamma(t_steps[i], s_churn, s_min, s_max, s_noise, n)
        eps = torch.randn_like(x) * s_noise
        t_hat = t_steps[i] + gamma * t_steps[i]
        
        if gamma > 0:
            x_hat = x + eps * (t_hat ** 2 - t_steps[i] ** 2) ** 0.5
        else:
            x_hat = x
            
        # Simple Euler step - get denoised signal directly
        denoised = self.get_d(model, x_hat, t_hat, x_cond)
        
        # Simple linear interpolation between current noisy and denoised
        # Step size based on noise level decrease
        t_hat_val = t_hat.item() if t_hat.dim() > 0 else float(t_hat)
        t_next_val = t_steps[i + 1].item() if t_steps[i + 1].dim() > 0 else float(t_steps[i + 1])
        
        step_size = (t_hat_val - t_next_val) / t_hat_val if t_hat_val > 0 else 0
        x_next = x_hat + step_size * (denoised - x_hat)
            
        return x_next

    def get_d(self, model, x, sig, x_cond=None):
        """
        Computes the denoising direction using the trained model
        
        Args:
            model: The signal diffusion transformer
            x: Noisy signal
            sig: Noise level
            x_cond: Conditioning signal (optional)
        """
        # Ensure sig has correct shape for model input and broadcasting
        if sig.dim() == 0:
            sig = sig.unsqueeze(0)  # Convert scalar to 1D tensor
        
        sig_for_broadcast = sig.view(-1, 1)  # Shape for broadcasting
        sig_for_model = sig.view(-1)  # Shape for model input
        
        # Forward pass through model - model predicts noise
        predicted_noise = model(x, sig_for_model, x_cond)
        
        # Correct epsilon parameterization: 
        # During training: noisy = clean + noise * sigma
        # During sampling: clean = noisy - predicted_noise * sigma
        denoised_signal = x - predicted_noise * sig_for_broadcast
        
        return denoised_signal

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
