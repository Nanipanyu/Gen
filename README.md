# Signal Diffusion Transformer for Broadband Earthquake Ground Motion Generation

This repository contains a diffusion transformer architecture adapted for generating broadband earthquake ground motion signals (0-30 Hz) conditioned on low-frequency physics-based simulations (<1 Hz).

## Overview

Traditional physics-based earthquake simulations are limited to low frequencies (≤1-10 Hz), insufficient for structural design requiring broadband signals (0-30 Hz). This project uses a diffusion transformer to extend low-frequency simulations into broadband accelerograms.

## Dependencies
- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib
- TensorBoard

## Quick Start

### 1. Process AT2 Files
```bash
python train_signal.py --process_at2 --at2_input_dir raw_data/ --data_dir processed_data/
```

### 2. Train Signal DiT
```bash
python train_signal.py --data_dir processed_data/ --model_dir earthquake_model/
```

### 3. Generate Broadband Signals
```bash
python generate_signals.py --model_path earthquake_model/best_ckpt.pt --input lowfreq.npz --output_dir results/
```

## Documentation
See `README_SIGNALS.md` for detailed documentation and usage instructions.


## Key Features

- **Signal DiT**: 1D diffusion transformer for time series generation
- **Cross-Attention Conditioning**: Multi-Head Cross-Attention for low-frequency signal conditioning
- **PGA Prediction**: CNN-LSTM network for Peak Ground Acceleration estimation
- **AT2 Processing**: Automatic conversion of earthquake data files
- **Linformer Attention**: Efficient attention mechanism for long sequences

## Configuration
Adjust hyperparameters in the `signal_config.py` file:

```python
signal_config = {
    'seq_len': 4096,           # Signal length (40.96s at 100Hz)
    'sample_rate': 100.0,      # Sampling rate
    'dim': 256,                # Model dimension
    'depth': 8,                # Transformer layers
    'use_conditioning': True,   # Enable cross-attention
    # ... more parameters
}
```

## Architecture Highlights
- **1D Patch Embedding**: Temporal patches for earthquake signals
- **Cross-Attention**: Enforces low-frequency constraints during generation
- **Joint Training**: Signal generation + PGA prediction in unified pipeline
- **Engineering Focus**: Designed specifically for structural analysis applications
- [EDM](https://arxiv.org/abs/2206.00364) sampler.
- [FID](https://arxiv.org/abs/1706.08500) evaluation.


## todo
- Add Classifier-Free Diffusion Guidance and conditional pipeline.
- Add Latent Diffusion and Autoencoder training.
- Add generate.py file.


## Licence
MIT
