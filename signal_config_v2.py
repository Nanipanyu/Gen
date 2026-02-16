"""
Configuration file for Signal DiT V2

All hyperparameters and settings for the new architecture.
"""

import torch

# Model Architecture
MODEL_CONFIG = {
    # Patch embedding
    'patch_size': 100,  # 100 samples per patch (1 second at 100Hz)
    
    # Model dimensions
    'hidden_dim': 384,
    'num_encoder_layers': 4,
    'num_decoder_layers': 5,
    'num_heads': 6,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    
    # RoPE settings
    'use_rope': True,
    'rope_base': 100.0,  # FIX BUG #7: Increased from 10 to avoid aliasing with 500+ patches
    
    # Spectral ControlNet
    'stft_n_fft': 512,
    'stft_hop_length': 256,
    'stft_window': 'hann',
    
    # Metadata encoder
    'metadata_dim': 3,  # M, Vs30, HypD
    'metadata_hidden_dims': [64, 128, 256],
    'metadata_num_encoder_layers': 2,
}

# Data Configuration
DATA_CONFIG = {
    # Paths
    'data_dir': 'data_prep_acc/processed_dynamic',
    'output_dir': 'signal_model_v2',
    
    # Signal properties
    'sample_rate': 100.0,  # Hz
    'min_length': 5000,    # 50 seconds
    'max_length': 50000,   # 500 seconds
    
    # Quality thresholds
    'pga_threshold': 1e-6,
    
    # Lowpass filter for conditioning
    'lowpass_cutoff': 1.0,  # Hz
    
    # Log-space transformation
    'log_epsilon': 1e-6,
    
    # Dataset split
    'train_ratio': 0.9,
    'val_ratio': 0.1,
}

# Packing Configuration
PACKING_CONFIG = {
    'pack_size': 3,          # Number of sequences per packed batch
    'max_packed_length': 50000,  # Maximum total length after packing
    'add_sep_tokens': True,  # Add [SEP] tokens between sequences
    'pad_value': 0.0,
}

# Training Configuration
TRAINING_CONFIG = {
    # Optimization
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-6,
    'weight_decay': 0.01,
    'betas': (0.9, 0.999),
    'gradient_clip': 1.0,
    
    # Training schedule
    'epochs': 400,
    'batch_size': 8,  # Number of packed batches
    'num_workers': 4,
    
    # EMA
    'ema_decay': 0.999,
    
    # Scheduler
    'scheduler_type': 'cosine',  # 'cosine', 'linear', 'constant'
    
    # Checkpoint
    'save_every': 10,      # Save checkpoint every N epochs
    'eval_every': 5,       # Evaluate every N epochs
    'keep_checkpoints': 5, # Keep last N checkpoints
}

# Diffusion Configuration
DIFFUSION_CONFIG = {
    # Log-space diffusion
    'epsilon': 1e-6,
    
    # Noise schedule - CONSISTENT for training and sampling
    'sigma_min': 0.02,
    'sigma_max': 1.5,  # Reduced to avoid clamp saturation
    'num_train_steps': 100,
    'num_sample_steps': 100,
    'schedule': 'cosine',  # 'linear', 'cosine', 'sigmoid'
    
    # Loss
    'loss_type': 'mse',  # 'mse', 'l1', 'huber'
    'envelope_weight': 3.0,  # Stronger envelope supervision
}

# Evaluation Configuration
EVAL_CONFIG = {
    'num_samples': 4,  # Number of samples to generate for visualization
    'sample_rate': 100.0,
    
    # Metrics to compute
    'compute_pga': True,
    'compute_spectral': True,
    'compute_duration': True,
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': False,  # Use mixed precision training (FP16)
    'compile_model': False,    # Use torch.compile() (requires PyTorch 2.0+)
}

# Logging Configuration
LOGGING_CONFIG = {
    'use_tensorboard': True,
    'tensorboard_dir': 'tensorboard',
    'log_every': 10,  # Log every N batches
    'wandb': False,   # Use Weights & Biases
    'wandb_project': 'signal-dit-v2',
}

# Full configuration dictionary
CONFIG_V2 = {
    'model': MODEL_CONFIG,
    'data': DATA_CONFIG,
    'packing': PACKING_CONFIG,
    'training': TRAINING_CONFIG,
    'diffusion': DIFFUSION_CONFIG,
    'eval': EVAL_CONFIG,
    'hardware': HARDWARE_CONFIG,
    'logging': LOGGING_CONFIG,
}


def get_config(key=None):
    """
    Get configuration value.
    
    Args:
        key: Dot-separated key (e.g., 'model.hidden_dim')
             If None, returns full config dict
    
    Returns:
        Configuration value
    """
    if key is None:
        return CONFIG_V2
    
    parts = key.split('.')
    value = CONFIG_V2
    
    for part in parts:
        value = value[part]
    
    return value


def update_config(key, value):
    """
    Update configuration value.
    
    Args:
        key: Dot-separated key (e.g., 'model.hidden_dim')
        value: New value
    """
    parts = key.split('.')
    config = CONFIG_V2
    
    for part in parts[:-1]:
        config = config[part]
    
    config[parts[-1]] = value


def print_config():
    """Print all configuration settings"""
    import json
    
    print("="*50)
    print("Signal DiT V2 Configuration")
    print("="*50)
    
    for section, params in CONFIG_V2.items():
        print(f"\n[{section.upper()}]")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*50)


if __name__ == '__main__':
    # Test configuration
    print_config()
    
    # Test get_config
    print("\nTest get_config:")
    print(f"Hidden dim: {get_config('model.hidden_dim')}")
    print(f"Learning rate: {get_config('training.learning_rate')}")
    print(f"Pack size: {get_config('packing.pack_size')}")
    
    # Test update_config
    print("\nTest update_config:")
    print(f"Before: batch_size = {get_config('training.batch_size')}")
    update_config('training.batch_size', 16)
    print(f"After: batch_size = {get_config('training.batch_size')}")
