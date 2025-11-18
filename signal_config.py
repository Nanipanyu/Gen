import os
import glob
import numpy as np
from typing import Union

def get_dynamic_seq_len(data_dir: str = None) -> int:
    """
    Automatically determine sequence length from processed data.
    
    Args:
        data_dir: Directory containing processed NPZ files
        
    Returns:
        int: Length of the longest signal in the dataset
    """
    if data_dir is None:
        # Try common data directories
        possible_dirs = [
            'data_prep_acc/processed_dynamic',
            'data_prep_acc/processed_fixed', 
            'data_prep_acc/processed_unified'
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                data_dir = dir_path
                break
        
        if data_dir is None:
            print("⚠️ Warning: No processed data found. Using default seq_len=6000")
            return 6000
    
    if not os.path.exists(data_dir):
        print(f"⚠️ Warning: Data directory not found: {data_dir}. Using default seq_len=6000")
        return 6000
    
    # Find NPZ files
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not npz_files:
        print(f"⚠️ Warning: No NPZ files found in {data_dir}. Using default seq_len=6000")
        return 6000
    
    # Find the longest signal (not just first file!)
    try:
        max_seq_len = 0
        longest_file = None
        signal_keys = ['signal_broadband', 'signal_raw_windowed', 'signal']
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                for key in signal_keys:
                    if key in data:
                        seq_len = len(data[key])
                        if seq_len > max_seq_len:
                            max_seq_len = seq_len
                            longest_file = npz_file
                        break
                        
            except Exception as e:
                print(f"⚠️ Warning: Error reading {npz_file}: {e}")
                continue
        
        if max_seq_len > 0:
            # Memory cap for current machine (can increase to 60k for workstation)
            capped_seq_len = min(max_seq_len, 30000)
            
            if max_seq_len > 30000:
                print(f"🚨 Memory cap: {max_seq_len:,} → {capped_seq_len:,} samples")
            else:
                print(f"✅ Dynamic seq_len detected: {capped_seq_len:,} samples from longest signal in {data_dir}")
            
            print(f"   Longest file: {os.path.basename(longest_file)}")
            return capped_seq_len
        else:
            print(f"⚠️ Warning: No valid signals found in {data_dir}. Using default seq_len=6000")
            return 6000
        
    except Exception as e:
        print(f"⚠️ Warning: Error analyzing files in {data_dir}: {e}. Using default seq_len=6000")
        return 6000

def get_dynamic_sample_rate(data_dir: str = None) -> float:
    """
    Automatically determine sample rate from the longest duration signal in the dataset.
    This makes the system fully dynamic - it adapts to ANY dataset's longest signal,
    regardless of whether it's 100 Hz, 200 Hz, or any other sampling rate.
    
    Args:
        data_dir: Directory containing processed NPZ files
        
    Returns:
        float: Sample rate from whichever signal has the longest duration
    """
    if data_dir is None:
        # Try common data directories
        possible_dirs = [
            'data_prep_acc/processed_dynamic',
            'data_prep_acc/processed_fixed',
            'data_prep_acc/processed_unified'
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                data_dir = dir_path
                break
        
        if data_dir is None:
            print("⚠️ Warning: No processed data found. Using default sample_rate=100.0")
            return 100.0
    
    if not os.path.exists(data_dir):
        print(f"⚠️ Warning: Data directory not found: {data_dir}. Using default sample_rate=100.0")
        return 100.0
    
    # Find NPZ files
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not npz_files:
        print(f"⚠️ Warning: No NPZ files found in {data_dir}. Using default sample_rate=100.0")
        return 100.0
    
    # Find the sample rate from the actual longest signal (same logic as seq_len detection)
    try:
        max_seq_len = 0
        longest_file_rate = None
        longest_file = None
        signal_keys = ['signal_broadband', 'signal_raw_windowed', 'signal']
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # Find signal length
                for key in signal_keys:
                    if key in data:
                        seq_len = len(data[key])
                        if seq_len > max_seq_len and 'sampling_rate' in data:
                            max_seq_len = seq_len
                            longest_file_rate = float(data['sampling_rate'])
                            longest_file = npz_file
                        break
                        
            except Exception as e:
                print(f"⚠️ Warning: Error reading {npz_file}: {e}")
                continue
        
        if longest_file_rate is not None:
            print(f"✅ Dynamic sample_rate detected: {longest_file_rate:.1f} Hz (from longest signal)")
            print(f"   Longest file: {os.path.basename(longest_file)} ({max_seq_len:,} samples)")
            return longest_file_rate
        else:
            print(f"⚠️ Warning: No valid sample rate found in {data_dir}. Using default sample_rate=100.0")
            return 100.0
        
    except Exception as e:
        print(f"⚠️ Warning: Error reading sample rate from {data_dir}: {e}. Using default sample_rate=100.0")
        return 100.0

def validate_model_dimensions(config_dict: dict) -> dict:
    """
    Validate and fix model dimensions to ensure compatibility.
    
    Critical requirements:
    1. dim must be divisible by heads (for multi-head attention)
    2. seq_len must be divisible by patch_size (for patch embedding)
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    validated_config = config_dict.copy()
    
    dim = validated_config.get('dim', 128)
    heads = validated_config.get('heads', 4)
    seq_len = validated_config.get('seq_len', 30000)
    patch_size = validated_config.get('patch_size', 500)
    
    # Check 1: dim must be divisible by heads
    if dim % heads != 0:
        # Find the largest factor of dim that is <= heads
        valid_heads = [h for h in [2, 4, 8, 16, 32, 64] if dim % h == 0 and h <= heads]
        if valid_heads:
            old_heads = heads
            heads = max(valid_heads)
            validated_config['heads'] = heads
            print(f"🔧 Fixed heads: {old_heads} → {heads} (dim={dim} must be divisible by heads)")
        else:
            print(f"❌ Error: dim={dim} cannot be divided by any valid heads value")
    
    # Check 2: seq_len should be divisible by patch_size for clean patches
    if seq_len % patch_size != 0:
        # Find a patch_size that divides evenly into seq_len
        valid_patch_sizes = [p for p in [100, 200, 250, 300, 400, 500, 600, 750, 1000, 1500] 
                           if seq_len % p == 0]
        if valid_patch_sizes:
            # Choose the one closest to the original patch_size
            old_patch_size = patch_size
            patch_size = min(valid_patch_sizes, key=lambda x: abs(x - old_patch_size))
            validated_config['patch_size'] = patch_size
            print(f"🔧 Fixed patch_size: {old_patch_size} → {patch_size} (seq_len={seq_len} must be divisible)")
        else:
            print(f"⚠️ Warning: seq_len={seq_len} not divisible by patch_size={patch_size}")
    
    return validated_config

def get_config_value(key: str, config_dict: dict, data_dir: str = None) -> Union[int, float, str]:
    """
    Get configuration value, handling dynamic values like 'auto' for seq_len and sample_rate.
    
    Args:
        key: Configuration key
        config_dict: Configuration dictionary
        data_dir: Data directory for dynamic detection
        
    Returns:
        Configuration value
    """
    value = config_dict.get(key)
    
    if key == 'seq_len' and value == 'auto':
        return get_dynamic_seq_len(data_dir)
    elif key == 'sample_rate' and value == 'auto':
        return get_dynamic_sample_rate(data_dir)
    
    return value

# Signal DiT hyperparameters for earthquake ground motion generation
signal_config = {
    # Signal parameters
    'seq_len': 'auto',         # Dynamic: automatically determined from longest signal in dataset
    'sample_rate': 'auto',     # Dynamic: automatically determined from longest signal's sampling rate
    'batch_size': 8,          # Proper batch size for gradient averaging and stable training
    'lr': 0.001,              # Learning rate
    
    # Model architecture (Optimized for 30k sequences with proper dimension alignment)
    'dim': 128,                # Model embedding dimension (patch embedding size)
    'k': 32,                   # Linformer projection dimension
    'patch_size': 100,         # Time series patch size (30k/500=60 patches) - MUST be compatible with seq_len
    'encoder_depth': 4,        # Number of encoder layers
    'decoder_depth': 5,        # Number of decoder layers
    'heads': 4,                # Number of attention heads - MUST divide evenly into dim (128/4=32)
    'mlp_dim': 256,            # MLP hidden dimension
    
    # Diffusion parameters - CRITICAL: MUST match signal scale!
    # For signals in [-1, 1], noise must be proportional to signal magnitude
    'P_mean': -1.2,            # Diffusion noise schedule parameter (log-normal mean)
    'P_std': 1.0,              # REDUCED from 2.0 to 1.0 - prevents noise explosion
    'sigma_data': 0.5,         # Data scaling parameter (0.5 for normalized signals in [-1,1])
    'steps': 100,              # Number of sampling steps - balanced quality/speed
    'sigma_max': 3.0,          # CRITICAL FIX: 3.0 (not 20.0!) - max noise is 3x signal amplitude
    
    # Training parameters
    'ema_decay': 0.999,        # EMA decay rate
    'n_iter': 5000,         # Number of training iterations
    'seed': 42,                # Random seed
    'train_pga_predictor': False,  # Switch to enable/disable PGA predictor training
    
    # Evaluation parameters
    'gen_batch_size': 50,      # Batch size for generation
    'n_eval_signals': 100,    # Number of signals for evaluation
    'eval_interval': 100,     # Evaluation interval
    'log_interval': 100,       # Logging interval
    
    # PGA prediction parameters
    'pga_lr': 0.001,           # Learning rate for PGA predictor
    'pga_batch_size': 64,      # Batch size for PGA training (reduced further)
    'cnn_filters': [8, 16, 32, 64],   # CNN filter depths (further reduced)
    'lstm_hidden': [32, 16],  # LSTM hidden sizes (very small for memory efficiency)
    'pga_dropout': 0.1,        # Dropout rate for PGA predictor
    
    # File paths (to be set during training)
    'data_dir': None,          # Directory containing training signals
    'model_dir': 'signal_model_v1',  # Model save directory
    'log_dir': None,           # Logging directory (auto-generated)
    'checkpoint_interval': 500,  # Checkpoint saving interval
}
