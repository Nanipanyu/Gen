"""
Debug why x_log is exploding during sampling
"""
import torch
from log_diffusion import LogSpaceDiffusion

# Create diffusion with correct params
diffusion = LogSpaceDiffusion(epsilon=1e-6, sigma_min=0.02, sigma_max=5.0, num_steps=100, schedule='cosine')

print("=" * 60)
print("SIGMA SCHEDULE CHECK")
print("=" * 60)

sigmas = diffusion.sigmas
print(f"Schedule: {diffusion.schedule}")
print(f"Num steps: {diffusion.num_steps}")
print(f"Sigma min: {diffusion.sigma_min}, max: {diffusion.sigma_max}")

print(f"\nFirst 10 sigmas: {sigmas[:10].tolist()}")
print(f"Last 10 sigmas: {sigmas[-10:].tolist()}")

# Check for duplicates
unique_sigmas = torch.unique(sigmas)
print(f"\nTotal sigmas: {len(sigmas)}")
print(f"Unique sigmas: {len(unique_sigmas)}")
if len(unique_sigmas) < len(sigmas):
    print("⚠️  WARNING: Duplicate sigma values detected!")

# Check the deltas
deltas = sigmas[:-1] - sigmas[1:]
print(f"\nSigma deltas (should all be positive, decreasing noise):")
print(f"  Min delta: {deltas.min().item():.6f}")
print(f"  Max delta: {deltas.max().item():.6f}")
print(f"  Mean delta: {deltas.mean().item():.6f}")

negative_deltas = (deltas < 0).sum().item()
if negative_deltas > 0:
    print(f"⚠️  WARNING: {negative_deltas} negative deltas (sigma increasing!)")

print("\n" + "=" * 60)
print("DENOISING STEP SIMULATION")
print("=" * 60)

# Simulate denoising with realistic values
x_t = torch.randn(1, 1000) * 5.0 - 5.0  # Start near data mean
sigma_t = sigmas[0]  # First sigma
sigma_next = sigmas[1]  # Next sigma
predicted_noise = torch.randn_like(x_t) * 0.8  # Realistic noise prediction

print(f"\nInitial x_t: range=[{x_t.min():.2f}, {x_t.max():.2f}], mean={x_t.mean():.2f}")
print(f"sigma_t: {sigma_t:.4f}")
print(f"sigma_next: {sigma_next:.4f}")
print(f"predicted_noise: range=[{predicted_noise.min():.2f}, {predicted_noise.max():.2f}], std={predicted_noise.std():.2f}")

# Apply current denoising step
x_0_pred = x_t - sigma_t * predicted_noise
print(f"\nx_0_pred = x_t - sigma_t * noise:")
print(f"  range=[{x_0_pred.min():.2f}, {x_0_pred.max():.2f}], mean={x_0_pred.mean():.2f}")

direction = (x_t - x_0_pred) / (sigma_t + 1e-8)
x_next = x_0_pred + sigma_next * direction

print(f"\nx_next = x_0_pred + sigma_next * direction:")
print(f"  range=[{x_next.min():.2f}, {x_next.max():.2f}], mean={x_next.mean():.2f}")

# Check change
delta_x = x_next - x_t
print(f"\nChange (x_next - x_t):")
print(f"  range=[{delta_x.min():.2f}, {delta_x.max():.2f}], mean={delta_x.mean():.2f}")

print("\n" + "=" * 60)
print("FULL SAMPLING SIMULATION")
print("=" * 60)

# Simulate 10 steps
x = torch.randn(1, 1000) * 5.0 - 5.0
print(f"\nStart: range=[{x.min():.2f}, {x.max():.2f}], mean={x.mean():.2f}")

for i in range(10):
    sigma_t = sigmas[i]
    sigma_next = sigmas[i+1] if i+1 < len(sigmas) else torch.tensor(0.0)
    
    # Simulate noise prediction (random, as model would output)
    noise_pred = torch.randn_like(x) * 0.8
    
    # Denoise
    x_0_pred = x - sigma_t * noise_pred
    direction = (x - x_0_pred) / (sigma_t + 1e-8)
    x = x_0_pred + sigma_next * direction
    
    if i < 3 or i == 9:
        print(f"Step {i}: range=[{x.min():.2f}, {x.max():.2f}], mean={x.mean():.2f}, std={x.std():.2f}")

print(f"\nAfter 10 steps: range=[{x.min():.2f}, {x.max():.2f}]")
print(f"Expected range: [-14, 0] (data range)")

if x.max() > 5 or x.min() < -20:
    print("⚠️  Signal is drifting outside expected range!")
