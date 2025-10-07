# Signal Diffusion Transformer for Broadband Earthquake Ground Motion Generation

This repository contains an adapted diffusion transformer architecture for generating broadband earthquake ground motion signals (0-30 Hz) conditioned on low-frequency physics-based simulations (<1 Hz).

## Overview

Traditional physics-based earthquake simulations are often limited to low frequencies (≤1-10 Hz), insufficient for structural design requiring broadband signals (0-30 Hz). This project overcomes this limitation by training a diffusion transformer to extend low-frequency simulations into broadband accelerograms.

## Architecture Components

### 1. Signal Diffusion Transformer (Signal-DiT)
- **File**: `signal_dit.py`
- **Purpose**: Core denoising network adapted for 1D time series
- **Key Features**:
  - 1D patch embedding for time series
  - Multi-Head Cross-Attention (MHCA) for conditioning
  - Linformer attention for efficiency
  - Timestep conditioning with adaptive layer normalization

### 2. CNN-LSTM PGA Predictor
- **File**: `pga_predictor.py`
- **Purpose**: Predicts Peak Ground Acceleration (PGA) from generated signals
- **Architecture**:
  - CNN block: Extracts temporal features (32→64→128→256 filters)
  - LSTM block: Captures dependencies (512→256 hidden units)
  - Feed-forward output layer

### 3. Signal Diffusion Utilities
- **File**: `signal_diff_utils.py`
- **Purpose**: Diffusion process mathematics for 1D signals
- **Features**:
  - EDM sampling algorithm
  - Noise scheduling
  - Conditioning support

### 4. Data Processing
- **File**: `signal_datasets.py`
- **Purpose**: Earthquake signal data loading and preprocessing
- **Features**:
  - AT2 file processing
  - Bandpass/lowpass filtering
  - Signal normalization by PGA

## Training Process

### Phase 1: Signal DiT Training
The diffusion transformer learns the relationship between:
- **Input**: `y₀(t)` - Broadband signal (0.1-30 Hz, normalized by PGA)
- **Conditioning**: `x₀(t)` - Low-frequency signal (<1 Hz, normalized by PGA)
- **Task**: Generate realistic broadband signals conditioned on low-frequency inputs

### Phase 2: PGA Predictor Training
The CNN-LSTM network learns to predict:
- **Input**: Combined low-frequency and normalized broadband signals
- **Output**: True PGA of the broadband ground motion

## Installation

```bash
# Install required packages
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib tqdm tensorboard
pip install scikit-learn
```

## Usage

### 1. Data Preparation

First, convert your AT2 earthquake files to the required format:

```bash
python train_signal.py --process_at2 --at2_input_dir path/to/at2/files --data_dir path/to/processed/data
```

### 2. Training

Train the Signal DiT model:

```bash
python train_signal.py --data_dir path/to/processed/data --model_dir signal_model_v1
```

Training arguments:
- `--data_dir`: Directory containing processed signal files (.npz format)
- `--model_dir`: Directory to save model checkpoints
- `--eval_interval`: Evaluation interval (default: 1000)
- `--log_interval`: Logging interval (default: 100)

### 3. Generation

Generate broadband signals from low-frequency inputs:

```bash
# Single file generation
python generate_signals.py --model_path signal_model_v1/best_ckpt.pt --input lowfreq_signal.npz --output_dir results/

# Batch generation from directory
python generate_signals.py --model_path signal_model_v1/best_ckpt.pt --input lowfreq_signals/ --output_dir results/batch/
```

Generation arguments:
- `--model_path`: Path to trained model checkpoint
- `--input`: Low-frequency signal file or directory
- `--output_dir`: Output directory for generated signals
- `--steps`: Number of denoising steps (default: 100)
- `--seed`: Random seed for reproducible generation

## Configuration

Modify `signal_config.py` to adjust:

```python
signal_config = {
    # Signal parameters
    'seq_len': 4096,           # Signal length (40.96s at 100Hz)
    'sample_rate': 100.0,      # Sampling rate
    'batch_size': 16,          # Training batch size
    
    # Model architecture
    'dim': 256,                # Model dimension
    'depth': 8,                # Number of transformer layers
    'heads': 8,                # Attention heads
    'use_conditioning': True,   # Enable cross-attention
    
    # Training parameters
    'lr': 0.0001,              # Learning rate
    'n_iter': 1000000,         # Training iterations
    'steps': 100,              # Sampling steps
}
```

## File Structure

```
diffusion-transformer-main/
├── signal_dit.py              # Signal Diffusion Transformer model
├── pga_predictor.py           # CNN-LSTM PGA prediction network
├── signal_diff_utils.py       # Signal diffusion utilities
├── signal_datasets.py         # Data loading for earthquake signals
├── signal_config.py           # Configuration parameters
├── train_signal.py            # Training script
├── generate_signals.py        # Generation script
├── utils.py                   # Utility functions (from original)
├── data_prep_acc/
│   └── bandfilter_acc.py      # Signal filtering utilities
└── README_SIGNALS.md          # This file
```

## Key Differences from Image DiT

1. **1D Processing**: Adapted for time series instead of 2D images
2. **Cross-Attention**: Added conditioning on low-frequency signals
3. **Signal Metrics**: PGA, frequency content, response spectrum analysis
4. **Domain-Specific**: Earthquake engineering focus with proper filtering

## Expected Workflow

1. **Data Preparation**: Convert AT2 files to processed format
2. **Training**: Train Signal DiT with conditioning
3. **Generation**: Generate broadband signals from low-frequency inputs
4. **Validation**: Check PGA predictions and frequency content
5. **Application**: Use generated signals for structural analysis

## Output Files

Generated signals are saved as `.npz` files containing:
- `signal`: Time series data
- `pga`: Predicted/true PGA value  
- `sample_rate`: Sampling rate
- `length_sec`: Duration in seconds
- Metadata for traceability

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir signal_model_v1/tensorboard/
```

Metrics tracked:
- Signal generation loss
- PGA prediction loss
- Generated signal quality metrics
- Frequency content analysis

## Performance Tips

1. **GPU Memory**: Reduce `batch_size` if running out of memory
2. **Training Speed**: Use fewer `steps` during training evaluations
3. **Quality**: Increase `steps` for final generation (50-200)
4. **Conditioning**: Set `use_conditioning=False` for unconditional generation

## Expected Results

The trained model should generate:
- **Broadband signals** (0-30 Hz) that are statistically consistent with real earthquake data
- **Proper low-frequency matching** when conditioned on physics-based simulations
- **Realistic PGA values** appropriate for structural design
- **Frequency content** suitable for engineering applications

This approach bridges the gap between physics-based simulation capabilities and engineering requirements for broadband ground motion analysis.
