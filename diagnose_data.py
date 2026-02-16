"""
Quick diagnostic to check log-space transformation of actual data.
"""

import torch
import numpy as np
from pathlib import Path
from log_diffusion import LogSpaceDiffusion

# Load actual data
data_path = Path("data_prep_acc/processed_dynamic")
npz_files = list(data_path.glob("*.npz"))

print("=" * 60)
print("DATA LOG-SPACE ANALYSIS")
print("=" * 60)

diffusion = LogSpaceDiffusion(epsilon=1e-6, sigma_min=0.02, sigma_max=1.5, num_steps=100, schedule='cosine')

# Analyze multiple files
all_log_mins = []
all_log_maxs = []
all_log_means = []

for npz_file in npz_files[:10]:  # Check first 10 files
    data = np.load(npz_file)
    broadband = torch.from_numpy(data['signal_broadband']).float()
    
    # Convert to log-space
    broadband_log, broadband_sign = diffusion.to_log_space(broadband.unsqueeze(0))
    broadband_log = broadband_log.squeeze(0)
    
    print(f"\n{npz_file.name}:")
    print(f"  Raw signal range: [{broadband.min():.6f}, {broadband.max():.6f}]")
    print(f"  Log-space range:  [{broadband_log.min():.6f}, {broadband_log.max():.6f}]")
    print(f"  Log-space mean:   {broadband_log.mean():.6f}")
    
    all_log_mins.append(broadband_log.min().item())
    all_log_maxs.append(broadband_log.max().item())
    all_log_means.append(broadband_log.mean().item())

print("\n" + "=" * 60)
print("SUMMARY STATISTICS (Log-Space)")
print("=" * 60)
print(f"Min log value:  {min(all_log_mins):.4f}")
print(f"Max log value:  {max(all_log_maxs):.4f}")
print(f"Mean of means:  {np.mean(all_log_means):.4f}")
print(f"Std of means:   {np.std(all_log_means):.4f}")

print("\n" + "=" * 60)
print("DIFFUSION SCHEDULE CHECK")
print("=" * 60)
sigmas = diffusion.sigmas
print(f"Sigma schedule: {diffusion.schedule}")
print(f"Sigma min: {diffusion.sigma_min}, max: {diffusion.sigma_max}")
print(f"First 5 sigmas: {sigmas[:5].tolist()}")
print(f"Last 5 sigmas: {sigmas[-5:].tolist()}")

# Check if sigma_max makes sense for this data range
data_range = max(all_log_maxs) - min(all_log_mins)
print(f"\nData range in log-space: {data_range:.4f}")
print(f"Sigma max: {diffusion.sigma_max}")
print(f"Ratio (data_range / sigma_max): {data_range / diffusion.sigma_max:.2f}")

# The sigma_max should be roughly 1-3x the data range for good coverage
if diffusion.sigma_max > data_range * 3:
    print("⚠️  WARNING: sigma_max may be too high, causing over-noising")
elif diffusion.sigma_max < data_range * 0.3:
    print("⚠️  WARNING: sigma_max may be too low, insufficient noise range")
else:
    print("✓  Sigma max looks reasonable for data range")

# Test noise addition
print("\n" + "=" * 60)
print("NOISE ADDITION TEST")
print("=" * 60)

# Use first data sample
data = np.load(npz_files[0])
x = torch.from_numpy(data['signal_broadband']).float().unsqueeze(0)
x_log, sign = diffusion.to_log_space(x)

print(f"Original log signal: mean={x_log.mean():.4f}, std={x_log.std():.4f}")

for sigma_val in [0.02, 0.1, 0.5, 1.0, 1.5]:
    sigma = torch.tensor([sigma_val])
    x_noisy, noise = diffusion.add_noise(x_log, sigma)
    print(f"Sigma={sigma_val:.2f}: noisy mean={x_noisy.mean():.4f}, std={x_noisy.std():.4f}, actual noise std={noise.std():.4f}")

# Check signal-to-noise ratio
print("\n" + "=" * 60)
print("SIGNAL-TO-NOISE RATIO ANALYSIS")
print("=" * 60)

signal_std = x_log.std().item()
for sigma_val in [0.02, 0.1, 0.5, 1.0, 1.5]:
    snr = signal_std / sigma_val
    print(f"Sigma={sigma_val:.2f}: SNR = {snr:.2f}")

print("\nFor good training:")
print("  - At sigma_max: SNR should be < 0.5 (signal buried in noise)")
print("  - At sigma_min: SNR should be > 10 (signal dominates)")
