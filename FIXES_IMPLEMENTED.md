# Critical Fixes Implemented

## Problem Analysis
After 50+ iterations, the model was generating random noise instead of matching the envelope shape of conditioning signals. Root cause: **Training enforces envelope matching via loss, but sampling has NO such enforcement.**

## Fixes Implemented

### Fix 1: Envelope-Guided Sampling ✅ CRITICAL
**Location:** `signal_diff_utils.py` line ~213

**Problem:** During training, the model learns with envelope loss constraint:
```python
loss = 0.6 * MSE(predicted, target) + 0.4 * MSE(lowpass(predicted), conditioning)
```

But during sampling, there was NO enforcement - the model could drift to any envelope.

**Solution:** Added explicit envelope guidance in the sampling loop:
```python
# Extract predicted low-frequency component
pred_lowfreq = apply_lowpass_filter(x_denoised, cutoff_freq=1.0, sample_rate=100.0)

# Compute envelope error
envelope_error = x_cond - pred_lowfreq

# Apply guidance with increasing strength (0 at start, 1 at end)
guidance_strength = 1.0 - (sigma_cur / self.sigma_max)
x_denoised = x_denoised + guidance_strength * envelope_error
```

**Why This Works:**
- Early steps (high noise): weak guidance → allows model to explore high-frequency structure
- Late steps (low noise): strong guidance → enforces exact envelope match
- Preserves high-frequency details from model while correcting low-frequency drift

---

### Fix 2: Aligned Training/Sampling Schedules ✅ IMPORTANT
**Location:** `signal_diff_utils.py` line ~172

**Problem:** Schedule mismatch between training and sampling:
- Training: Random sigma sampling from log-normal `σ ~ exp(N(-1.2, 1.0))`
- Sampling: Linear interpolation through empirical distribution

Model wasn't trained on the linear trajectory it encounters during generation.

**Solution:** Use Karras schedule for both training and sampling:
```python
# Generate log-spaced noise schedule matching training distribution
rho = 7.0  # Controls schedule curvature
min_inv_rho = (0.002 ** (1 / rho))
max_inv_rho = (sigma_max ** (1 / rho))

ramp = torch.linspace(0, 1, steps + 1, device=device)
t_steps = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
```

**Why This Works:**
- Monotonic decreasing path through the same noise distribution used in training
- Model has seen similar noise levels during training
- Smoother denoising trajectory

---

### Fix 3: Strengthened Cross-Attention ✅ OPTIONAL
**Location:** `signal_dit.py` line ~228

**Problem:** Learned gating for cross-attention could collapse to near-zero, blocking conditioning information flow.

**Solution:** Replaced learned gating with fixed strong weighting:
```python
# Before: Learned gate that can collapse
# x = x + cross_out * gate_cross

# After: Fixed strong blending
x = 0.7 * cross_out + 0.3 * x

# Additional direct injection
if encoder_output.shape == x.shape:
    x = x + 0.3 * encoder_output  # Increased from 0.2
```

**Why This Works:**
- 70% cross-attention ensures conditioning is ALWAYS used strongly
- Cannot collapse to zero like learned gates can
- Direct encoder injection (30%) provides additional conditioning signal

---

## What Was NOT Changed

### ❌ Envelope Loss Weight (40% → 80%)
**Reason:** 40% is already significant. Signal loss (60%) implicitly includes envelope since target has correct envelope. Increasing to 80% risks:
- Over-regularization
- Loss of high-frequency detail
- Model may stop learning fine structure

### ❌ Post-Generation Envelope Correction
**Reason:** Band-aid solution, not a proper fix. We want the model to learn correct envelope generation, not force-correct afterwards.

---

## Expected Results

After retraining with these fixes:

1. **Envelope Matching:** Generated signals should closely match conditioning envelope shape
   - Visual check: Blue curves should follow green envelope shape in `log_img/` folder
   - Metric: Envelope correlation > 0.9 by iteration 3000

2. **Signal Quality:** Broadband structure preserved
   - Signal correlation > 0.7 by iteration 5000
   - Generated signals stay in [-1, 1] range

3. **Training Stability:** Smoother convergence
   - Signal loss decreases steadily
   - Envelope loss < 0.1 by iteration 3000

---

## How to Test

```cmd
# Clear old checkpoints
del signal_model_v1\*.pt

# Retrain with fixes
python train_signal.py

# Monitor:
# - Envelope correlation in terminal output (should be >0.9)
# - Generated images in log_img/ (blue should match green envelope)
# - Loss curves (envelope loss should drop below 0.1)
```

---

## Technical Summary

**Root Cause:** Training-sampling mismatch - envelope constraint enforced during training via loss, but not during sampling.

**Critical Fix:** Envelope-guided sampling that explicitly corrects low-frequency drift at each denoising step.

**Supporting Fixes:** 
- Schedule alignment for consistent noise distribution
- Stronger cross-attention to prevent information collapse

**Philosophy:** Force the sampling process to respect the same constraints the model was trained with.
