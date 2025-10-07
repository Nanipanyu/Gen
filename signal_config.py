# Signal DiT hyperparameters for earthquake ground motion generation
signal_config = {
    # Signal parameters
    'seq_len': 6000,           # Length of time series (60.0 seconds at 100 Hz - preserves natural earthquake duration)
    'sample_rate': 100.0,      # Sampling rate in Hz
    'batch_size': 7,          # Batch size for training
    'lr': 0.001,              # Learning rate
    
    # Model architecture
    'dim': 300,                # Model embedding dimension (patch embedding size)
    'k': 64,                   # Linformer projection dimension
    'patch_size': 20,          # Time series patch size
    'encoder_depth': 4,        # Number of encoder layers (for low-pass conditioning (SA+MLP+output))
    'decoder_depth': 8,        # Number of decoder layers (for broadband generation(SA+cross-attention+MLP+output))
    'heads': 10,               # Number of attention heads , each head will see the whole input , but reduced model dimension to 300/10=30 , then it will finally concatenate the output of each head and get back to 300 dimension
    'mlp_dim': 512,            # MLP hidden dimension
    
    # Diffusion parameters
    'P_mean': -1.2,            # Diffusion noise schedule parameter
    'P_std': 1.2,              # Diffusion noise schedule parameter
    'sigma_data': 0.66,        # Data scaling parameter
    'steps': 100,              # Number of sampling steps
    'sigma_max': 80.0,         # Maximum noise level
    
    # Training parameters
    'ema_decay': 0.999,        # EMA decay rate
    'n_iter': 5000,         # Number of training iterations
    'seed': 42,                # Random seed
    
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
