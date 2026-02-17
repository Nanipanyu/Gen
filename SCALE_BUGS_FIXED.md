# Critical Bugs Fixed - Scale Mismatch Issues

## Root Cause Analysis

The model was failing because of **three interconnected denormalization bugs** that caused signals to be at completely wrong scales (1000x too small).

---

## Bug #1: Dataset Loader Incorrect Denormalization ❌ → ✅

**Location:** `signal_datasets_v2.py` lines 185-186

**The Problem:**
```python
# OLD CODE (WRONG):
broadband = broadband_normalized * pga  # Signal already RAW!
```

**What was happening:**
1. Preprocessing saves signals RAW: `signal_broadband = [-0.012, 0.009]` (physical scale)
2. Dataset loader assumes normalized: `broadband_normalized = data['signal_broadband']`
3. **Incorrectly "denormalizes":** `broadband = [-0.012, 0.009] × 0.012 = [-0.000144, 0.000108]`
4. **Signal becomes 1000x smaller!**

**The Fix:**
```python
# NEW CODE (CORRECT):
# Signals are saved RAW (not normalized), so NO denormalization needed!
# PGA is saved only for reference, not for denormalization.
broadband = broadband_normalized  # Use directly, already correct scale
```

**Impact:** Model was training on signals at completely wrong scale, making learning impossible.

---

## Bug #2: Lowpass Signal Scale Mismatch ❌ → ✅

**Location:** `signal_datasets_v2.py` lines 189-213

**The Problem:**
```python
# OLD CODE (WRONG):
lowpass_normalized = data['signal_lowfreq']
lowpass = lowpass_normalized * pga_lowfreq  # Double scaling!
```

**What was happening:**
- Lowpass signal also saved RAW in preprocessing
- Dataset loader multiplied by PGA again
- Made lowpass signal 1000x smaller
- **Even worse:** Sometimes used `pga_broadband` instead of `pga_lowfreq`, creating scale mismatch between broadband and lowpass conditioning!

**The Fix:**
```python
# NEW CODE (CORRECT):
# Lowpass is also saved RAW, use directly
lowpass = data['signal_lowfreq'].copy()  # Already correct scale
```

**Impact:** Conditioning signal at wrong scale + potential mismatch between broadband/lowpass scales.

---

## Bug #3: Statistics Computation Incorrect Denormalization ❌ → ✅

**Location:** `train_signal_v2.py` lines 89-90

**The Problem:**
```python
# OLD CODE (WRONG):
signal = data['signal_broadband']
pga = data.get('pga_broadband', ...)
signal_denorm = signal * pga  # Signal already RAW!
```

**What was happening:**
- Computing dataset statistics for normalization (if enabled)
- Multiplied RAW signal by PGA before computing statistics
- Statistics computed on 1000x smaller values
- If normalization was enabled, model would get completely wrong mean/std

**The Fix:**
```python
# NEW CODE (CORRECT):
signal = data['signal_broadband']  # Already RAW scale
pga = data.get('pga_broadband', ...)  # For reference only
# Use signal directly, no multiplication
```

**Impact:** If `--normalize` flag was used, computed statistics would be completely wrong.

---

## Why This Caused [-4, 4] or Rectangle Outputs

The scale bugs created a cascading failure:

1. **Training data 1000x too small**
   - Actual signal: std ≈ 0.002, range ≈ [-0.012, 0.009]
   - After bug: std ≈ 0.000002, range ≈ [-0.000144, 0.000108]

2. **Noise at wrong scale**
   - Training adds noise: `noisy = signal + randn() * noise_std`
   - If `noise_std = signal_std`, then `noise_std ≈ 0.000002` (way too small!)
   - Model sees mostly noise, can't learn signal structure

3. **Clamping artifacts**
   - Sampling clamps predictions to ±0.2 (for safety)
   - When signal is 1000x too small, clamp dominates
   - Model learns to output constant values near clamp boundaries
   - Result: Flat rectangles at whatever scale the clamps hit

4. **No learning across epochs**
   - Signal-to-noise ratio completely wrong
   - Gradients point in meaningless directions
   - Model can't distinguish signal from noise
   - Training makes no progress (epoch 1 = epoch 75)

---

## Verification Steps

### Step 1: Check preprocessed data scale
```bash
python verify_data_scale.py
```

**Expected output:**
```
Broadband range: [-0.012927, 0.009510]  ← Physical earthquake scale
Broadband std:   0.002345               ← Typical: 0.001 to 0.02
✅ Signal scale looks CORRECT (RAW earthquake data)
```

**If wrong:**
- Delete `data_prep_acc/processed_dynamic/` folder
- Rerun `python dynamic_preprocessing.py`
- Verify again

### Step 2: Run training with diagnostics
```bash
python train_signal_v2.py --checkpoint_dir signal_model_v4 --resume --epochs 10
```

**Watch for these diagnostics (printed every 5 epochs):**

✅ **Healthy Training:**
```
🔬 TRAINING DIAGNOSTICS (Epoch 5, Batch 0):
   Input noisy signal: range [-0.010, 0.012], std 0.0025
   Lowpass conditioning: range [-0.0015, 0.0013], std 0.0003
   Target noise: range [-0.008, 0.009], std 0.0024
   Predicted noise: range [-0.007, 0.008], std 0.0023
   Prediction diversity: unique values = 15234 ✓
   ✓ Gradient norm: 0.342 (healthy if > 1e-3)
   
Scale check: signal_std=0.0023, noise_std=0.0024, ratio=1.043x ✓
```

⚠️ **Problem Signs:**
```
WARNING: Model predicting CONSTANT values! (std < 1e-6)
WARNING: Gradients are ZERO! Model not learning
Scale check: ratio=0.001x ← WRONG! Should be ~1.0x
```

---

## What Should Happen Now

With bugs fixed, training should show:

1. **Correct scales:**
   - Signal std: 0.001 to 0.02
   - Noise/Signal ratio: ≈ 1.0x
   - Predictions varying (10,000+ unique values)

2. **Learning progress:**
   - Loss decreasing over epochs
   - Gradients flowing (norm > 0.001)
   - Generated signals showing waveform structure (not rectangles)

3. **Proper envelope matching:**
   - Envelope correlation increasing (starts ~0, reaches >0.5)
   - Generated signals respecting lowpass conditioning
   - Amplitude range matching original data

---

## Files Modified

1. ✅ `signal_datasets_v2.py` - Removed incorrect denormalization
2. ✅ `train_signal_v2.py` - Fixed statistics computation, added diagnostics
3. ✅ `dynamic_preprocessing.py` - Already fixed (saves RAW signals)
4. ✅ `verify_data_scale.py` - Created for verification

---

## Next Steps

1. **Verify data:** `python verify_data_scale.py`
2. **If data wrong:** Delete processed_dynamic/, rerun preprocessing
3. **Train with diagnostics:** Resume training, monitor logs for scale checks
4. **Check progress:** Predictions should show waveforms, not rectangles
5. **If still failing:** Share diagnostic output for further analysis
