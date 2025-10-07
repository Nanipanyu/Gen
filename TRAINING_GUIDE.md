# Signal DiT Training Guide

## Overview
This guide walks you through training a Signal Diffusion Transformer (DiT) for generating broadband earthquake ground motion signals from low-frequency conditioning signals.

## Architecture Status
✅ **Code Architecture Complete**: All necessary files are implemented and ready for training.

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Your earthquake data should be in AT2 format in the `data_prep_acc/` directory.

## Training Process

### Step 1: Preprocess AT2 Files
First, preprocess your AT2 earthquake files into training-ready format:

```bash
python bandfilter_acc.py
```

**What this does:**
- Reads AT2 files from `data_prep_acc/`
- Applies STA/LTA event detection
- Creates 60-second windows starting from event triggers
- Applies band-pass filter (0.1-30 Hz) for broadband signals
- Applies low-pass filter (0.1-5 Hz) for conditioning signals
- Saves processed signals as `.npz` files
- Creates horizontal component pairs (EW + NS)

**Expected output:**
- Processed files in `data_prep_acc/bandpass/` (broadband signals)
- Processed files in `data_prep_acc/lowpass/` (conditioning signals)

### Step 2: Start Training
Launch the training process:

```bash
python train_signal.py
```

**What this trains:**
- **Signal DiT**: Main diffusion transformer for generating broadband signals
- **PGA Predictor**: CNN-LSTM network for Peak Ground Acceleration prediction
- **Joint Training**: Both models trained together with combined loss

**Training parameters (from signal_config.py):**
- Sequence length: 4096 samples (40.96 seconds at 100 Hz)
- Batch size: 16
- Learning rate: 0.0001
- Model dimension: 256
- Transformer layers: 8
- Attention heads: 8

**Optimized for AMD Ryzen 7 4800H CPU:**
- No GPU dependencies
- Efficient Linformer attention (O(n) complexity)
- Reasonable batch size for CPU memory

### Step 3: Monitor Training
Training creates several outputs:

**Checkpoints:**
- `models/signal_dit_latest.pt` - Latest model state
- `models/signal_dit_best.pt` - Best validation loss model
- `models/pga_predictor_latest.pt` - Latest PGA predictor
- `models/pga_predictor_best.pt` - Best PGA predictor

**Logs:**
- `logs/` - TensorBoard logs for monitoring loss curves
- View with: `tensorboard --logdir=logs`

**Sample outputs:**
- `samples/` - Generated signal samples during training
- `samples/signal_samples_epoch_X.png` - Visual comparisons

### Step 4: Generate New Signals (After Training)
Once training is complete, generate new earthquake signals:

```bash
python generate_signals.py
```

## Training Timeline (AMD Ryzen 7 4800H)
- **Data preprocessing**: ~5-10 minutes (depends on dataset size)
- **Training time**: ~2-4 hours for initial convergence (depends on dataset size)
- **Memory usage**: ~4-8 GB RAM with batch_size=16

## Key Features

### Signal DiT Architecture
- **1D Patch Embedding**: Converts time series to patches for transformer processing
- **Cross-Attention Conditioning**: Uses low-frequency signals to guide broadband generation
- **Linformer Attention**: O(n) complexity for efficient CPU training
- **Timestep Modulation**: AdaLN conditioning for diffusion timesteps

### Training Features
- **EMA (Exponential Moving Average)**: Stabilizes training
- **Progressive Evaluation**: Regular validation with signal quality metrics
- **Automatic Checkpointing**: Saves best models based on validation loss
- **Signal Quality Metrics**: PGA, frequency content, and response spectrum analysis

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `batch_size` in `signal_config.py`
2. **Slow Training**: Normal for CPU training; consider cloud GPU if needed
3. **No AT2 Files**: Ensure AT2 files are in `data_prep_acc/` directory
4. **Poor Signal Quality**: May need more training epochs or larger dataset

### Configuration Adjustment
Edit `signal_config.py` to modify:
- `batch_size`: Reduce if memory issues
- `lr`: Adjust learning rate
- `seq_len`: Change signal length
- `depth`: Modify model complexity

## Next Steps After Training
1. **Evaluate Results**: Check generated signals in `samples/`
2. **Fine-tune Parameters**: Adjust config based on initial results
3. **Scale Up**: Train longer or with more data for better quality
4. **Generate Signals**: Use `generate_signals.py` for new earthquake simulations

## File Structure
```
diffusion-transformer-main/
├── signal_dit.py              # Signal DiT architecture
├── signal_diff_utils.py       # Diffusion mathematics
├── pga_predictor.py          # PGA prediction CNN-LSTM
├── train_signal.py           # Main training script
├── generate_signals.py       # Signal generation script
├── signal_datasets.py        # Data loading utilities
├── bandfilter_acc.py         # Preprocessing script
├── signal_config.py          # Training configuration
├── requirements.txt          # Dependencies
├── data_prep_acc/           # Raw AT2 files
│   ├── bandpass/           # Processed broadband signals
│   └── lowpass/            # Processed conditioning signals
├── models/                  # Saved model checkpoints
├── logs/                    # TensorBoard logs
└── samples/                 # Generated signal samples
```

## Ready to Start Training!
Your Signal DiT architecture is complete and ready for training. Follow the steps above to begin generating realistic earthquake ground motion signals.