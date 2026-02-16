"""
Spectral ControlNet for frequency-domain conditioning of diffusion models.
Uses STFT to extract spectral features and injects them into DiT encoder blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralControlNet(nn.Module):
    """
    ControlNet-style adapter that processes lowpass signal in frequency domain
    and injects conditioning into DiT encoder blocks via zero-initialized convolutions.
    
    Args:
        n_fft: FFT size (default: 512)
        hop_length: Hop length for STFT (default: 256)
        window: Window function name (default: 'hann')
        in_channels: Input STFT channels (2 for real/imag or magnitude/phase)
        hidden_channels: Hidden channels in CNN (default: [32, 64, 128, 256])
        dit_hidden_dim: DiT hidden dimension to match (default: 384)
        num_encoder_layers: Number of DiT encoder layers to inject into (default: 4)
    """
    def __init__(
        self,
        n_fft=512,
        hop_length=256,
        window='hann',
        in_channels=2,
        hidden_channels=None,
        dit_hidden_dim=384,
        num_encoder_layers=4
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_name = window
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]
        
        self.hidden_channels = hidden_channels
        self.dit_hidden_dim = dit_hidden_dim
        self.num_encoder_layers = num_encoder_layers
        
        # Register window buffer
        window_tensor = self._get_window(window, n_fft)
        self.register_buffer('window', window_tensor)
        
        # Build 2D CNN encoder for STFT features
        # Input: [batch, 2, freq_bins, time_frames] where freq_bins = n_fft//2 + 1
        layers = []
        in_ch = in_channels
        
        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.MaxPool2d(2)
            ])
            in_ch = out_ch
        
        self.cnn_encoder = nn.Sequential(*layers)
        
        # Calculate output spatial dimensions after pooling
        # After 4 pooling layers: freq_dim = (n_fft//2 + 1) / 2^4, time_dim = (seq_len/hop) / 2^4
        self.freq_reduction = 2 ** len(hidden_channels)
        
        # IMPROVED: Use frequency pooling but keep temporal dimension
        # This preserves envelope information over time
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool freq, keep time
        
        # Temporal projection to match DiT hidden dimension
        self.temporal_proj = nn.Sequential(
            nn.Conv1d(hidden_channels[-1], dit_hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(dit_hidden_dim, dit_hidden_dim, kernel_size=3, padding=1)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(dit_hidden_dim)
        
        # CRITICAL FIX: Instead of zero-initialized convs, use small-scale initialization
        # Zero-init prevents learning - model ignores ControlNet entirely!
        # Increased from 0.01 to 0.1 for stronger gradient signal
        self.control_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(num_encoder_layers)
        ])
        
        # Non-zero initialized projection layers
        self.control_projections = nn.ModuleList([
            nn.Conv1d(dit_hidden_dim, dit_hidden_dim, kernel_size=1)
            for _ in range(num_encoder_layers)
        ])
    
    def _get_window(self, window_name, n_fft):
        """Get window function tensor"""
        if window_name == 'hann':
            return torch.hann_window(n_fft)
        elif window_name == 'hamming':
            return torch.hamming_window(n_fft)
        elif window_name == 'blackman':
            return torch.blackman_window(n_fft)
        else:
            return torch.ones(n_fft)
    
    def _make_zero_conv(self, channels):
        """Create zero-initialized 1x1 convolution for ControlNet injection"""
        conv = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv
    
    def compute_stft(self, x):
        """
        Compute STFT of input signal.
        
        Args:
            x: Input tensor [batch, seq_len]
        
        Returns:
            stft_features: STFT magnitude and phase [batch, 2, freq_bins, time_frames]
        """
        # Ensure window is on same device
        if self.window.device != x.device:
            self.window = self.window.to(x.device)
        
        # Compute STFT: returns complex tensor [batch, freq_bins, time_frames]
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            normalized=False,
            center=True
        )
        
        # Extract magnitude and phase
        magnitude = torch.abs(stft)  # [batch, freq_bins, time_frames]
        phase = torch.angle(stft)    # [batch, freq_bins, time_frames]
        
        # Stack as [batch, 2, freq_bins, time_frames]
        stft_features = torch.stack([magnitude, phase], dim=1)
        
        # Log-scale magnitude for better dynamic range
        stft_features[:, 0] = torch.log1p(stft_features[:, 0])
        
        return stft_features
    
    def forward(self, x_lowpass):
        """
        Process lowpass signal and generate conditioning features.
        
        Args:
            x_lowpass: Lowpass conditioning signal [batch, seq_len]
        
        Returns:
            control_features: List of conditioning features for each encoder layer
                             Each element has shape [batch, dit_hidden_dim, num_patches]
        """
        batch_size = x_lowpass.shape[0]
        
        # Compute STFT
        stft_features = self.compute_stft(x_lowpass)  # [batch, 2, freq_bins, time_frames]
        
        # Pass through CNN encoder
        cnn_out = self.cnn_encoder(stft_features)  # [batch, hidden_channels[-1], H, W]
        
        # Pool frequency dimension but preserve temporal structure
        freq_pooled = self.freq_pool(cnn_out)  # [batch, C, 1, T]
        freq_pooled = freq_pooled.squeeze(2)  # [batch, C, T]
        
        # Apply temporal projection
        temporal_features = self.temporal_proj(freq_pooled)  # [batch, dit_hidden_dim, T]
        temporal_features = temporal_features.transpose(1, 2)  # [batch, T, dit_hidden_dim]
        
        # Normalize
        base_feature = self.norm(temporal_features)  # [batch, T, dit_hidden_dim]
        
        # Generate conditioning for each encoder layer with learnable scaling
        control_features = []
        for i, (scale, projection) in enumerate(zip(self.control_scales, self.control_projections)):
            # Reshape for conv: [batch, T, dit_hidden_dim] -> [batch, dit_hidden_dim, T]
            feat = base_feature.transpose(1, 2)  # [batch, dit_hidden_dim, T]
            
            # Apply projection with small learnable scale (starts at 0.01, can grow)
            control = projection(feat)  # [batch, dit_hidden_dim, T]
            control = control * scale  # Learnable scale per layer
            
            control_features.append(control)
        
        return control_features
    
    def inject_control(self, x, control, layer_idx):
        """
        FIX BUG #4: Interpolate control features to match encoder sequence length.
        Previous version had temporal dimension mismatch.
        
        Args:
            x: Encoder features [batch, num_patches, dit_hidden_dim]
            control: Control features [batch, num_control_tokens, dit_hidden_dim]
            layer_idx: Layer index for scaling
        
        Returns:
            x: Injected features [batch, num_patches, dit_hidden_dim]
        """
        batch_size, num_patches, hidden_dim = x.shape
        _, num_control, _ = control.shape
        
        # Interpolate control to match encoder length
        if num_control != num_patches:
            # Transpose for interpolation: [batch, hidden_dim, num_control]
            control_t = control.transpose(1, 2)
            # Interpolate: [batch, hidden_dim, num_patches]
            control_interp = F.interpolate(
                control_t, size=num_patches, mode='linear', align_corners=False
            )
            # Transpose back: [batch, num_patches, hidden_dim]
            control = control_interp.transpose(1, 2)
        
        # Apply layer-specific scaling and projection
        control_scaled = self.control_scales[layer_idx] * control
        control_proj = self.control_projections[layer_idx](control_scaled.transpose(1, 2)).transpose(1, 2)
        
        return x + control_proj


class SpectralControlNetAdapter(nn.Module):
    """
    Wrapper that integrates SpectralControlNet with DiT encoder.
    Handles injection at each encoder layer.
    """
    def __init__(self, controlnet, dit_encoder):
        super().__init__()
        self.controlnet = controlnet
        self.dit_encoder = dit_encoder
    
    def forward(self, x, x_lowpass, timestep=None):
        """
        Forward pass through encoder with spectral control injection.
        
        Args:
            x: Noisy input patches [batch, num_patches, hidden_dim]
            x_lowpass: Lowpass conditioning signal [batch, seq_len]
            timestep: Optional timestep embedding
        
        Returns:
            x: Encoded features with control injection
        """
        # Generate control features from lowpass
        control_features = self.controlnet(x_lowpass)
        
        # Pass through encoder with injection
        for layer_idx, encoder_layer in enumerate(self.dit_encoder.layers):
            # Standard encoder forward
            x = encoder_layer(x, timestep)
            
            # Inject control if available for this layer
            if layer_idx < len(control_features):
                x = self.controlnet.inject_control(x, control_features[layer_idx], layer_idx)
        
        return x


if __name__ == "__main__":
    # Test SpectralControlNet
    print("Testing Spectral ControlNet...")
    
    batch_size = 4
    seq_len = 30000
    dit_hidden_dim = 384
    num_encoder_layers = 4
    
    # Create test lowpass signal
    x_lowpass = torch.randn(batch_size, seq_len)
    
    # Create ControlNet
    controlnet = SpectralControlNet(
        n_fft=512,
        hop_length=256,
        window='hann',
        dit_hidden_dim=dit_hidden_dim,
        num_encoder_layers=num_encoder_layers
    )
    
    # Forward pass
    control_features = controlnet(x_lowpass)
    
    print(f"Input lowpass shape: {x_lowpass.shape}")
    print(f"Number of control outputs: {len(control_features)}")
    print(f"Each control feature shape: {control_features[0].shape}")
    
    # Test injection
    num_patches = 300
    encoder_features = torch.randn(batch_size, num_patches, dit_hidden_dim)
    controlled = controlnet.inject_control(encoder_features, control_features[0], 0)
    
    print(f"Encoder features shape: {encoder_features.shape}")
    print(f"Controlled features shape: {controlled.shape}")
    print("Spectral ControlNet test passed!")
