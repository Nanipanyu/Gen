"""
Signal Diffusion Transformer V2 (Signal-DiT-v2)

New architecture with:
- RoPE (Rotary Position Embeddings) for variable-length generalization
- Packed attention with block-diagonal masks
- Spectral ControlNet for frequency-domain conditioning
- Metadata encoder with cross-attention
- Log-space diffusion (no normalization needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rope_embeddings import RotaryPositionEmbedding
from spectral_controlnet import SpectralControlNet
from metadata_encoder import MetadataEncoder, MetadataConditionedBlock
from packed_attention import PackedMultiHeadAttention


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps"""
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        if x.dim() == 0:
            x = x.unsqueeze(0)
        
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeSeriesProjection(nn.Module):
    """Projects 1D time series to patch embeddings"""
    def __init__(self, patch_size, dim):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        
        # Use 1D convolution to create patches
        self.proj = nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Args:
            x: Input signal [batch, seq_len]
        Returns:
            patches: [batch, num_patches, dim]
        """
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # Add channel: [batch, 1, seq_len]
        x = self.proj(x)  # [batch, dim, num_patches]
        x = x.transpose(1, 2)  # [batch, num_patches, dim]
        return x


class DiTEncoderBlock(nn.Module):
    """
    Encoder block with:
    - Packed self-attention with RoPE
    - Spectral ControlNet injection point
    - Feedforward network
    """
    def __init__(
        self,
        hidden_dim,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        use_rope=True,
        rope_base=10.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Packed attention with RoPE
        self.attn = PackedMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            rope_base=rope_base
        )
        
        # Feedforward network
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Timestep modulation (adaptive layer norm)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
    
    def forward(self, x, t_emb, boundaries=None, actual_length=None, position_ids=None, control_signal=None):
        """
        Args:
            x: Input features [batch, num_patches, hidden_dim]
            t_emb: Timestep embedding [batch, hidden_dim]
            boundaries: List of (start, end) tuples for packed sequences
            actual_length: Actual length of packed content
            position_ids: Position IDs for RoPE [batch, num_patches]
            control_signal: Optional ControlNet injection [batch, hidden_dim, 1]
        
        Returns:
            x: Output features [batch, num_patches, hidden_dim]
        """
        # Adaptive layer norm modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Self-attention block with RoPE
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        attn_out = self.attn(x_norm, boundaries=boundaries, actual_length=actual_length, position_ids=position_ids)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Inject ControlNet signal if provided
        if control_signal is not None:
            # control_signal: [batch, hidden_dim, T] where T is temporal frames
            # Need to interpolate/match to num_patches
            if control_signal.shape[-1] != x.shape[1]:
                # Interpolate to match patch length
                control_signal = F.interpolate(
                    control_signal, 
                    size=x.shape[1], 
                    mode='linear', 
                    align_corners=False
                )
            # Transpose to [batch, num_patches, hidden_dim]
            control_broadcasted = control_signal.transpose(1, 2)
            x = x + control_broadcasted
        
        # Feedforward block
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class DiTDecoderBlock(nn.Module):
    """
    Decoder block with:
    - Self-attention
    - Feedforward network
    (NO metadata cross-attention - simplified architecture)
    """
    def __init__(
        self,
        hidden_dim,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        use_rope=True,
        rope_base=10.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Timestep modulation (adaLN-Zero style)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)  # 6 for scale/shift/gate on attn and mlp
        )
    
    def forward(self, x, t_emb, metadata_features=None, boundaries=None, actual_length=None):
        """
        Args:
            x: Input features [batch, num_patches, hidden_dim]
            t_emb: Timestep embedding [batch, hidden_dim]
            metadata_features: IGNORED (kept for interface compatibility)
            boundaries: IGNORED (kept for interface compatibility)
            actual_length: IGNORED (kept for interface compatibility)
        
        Returns:
            x: Output features [batch, num_patches, hidden_dim]
        """
        # Timestep modulation - adaLN-Zero style
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Self-attention block with modulation
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_attn.unsqueeze(1) * attn_out
        
        # Feedforward block with modulation
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class SignalDiTV2(nn.Module):
    """
    Signal Diffusion Transformer V2 - Simplified Architecture
    
    Architecture:
    1. Patch embedding (variable length with RoPE)
    2. Timestep embedding
    3. Spectral ControlNet for lowpass conditioning
    4. Encoder blocks (with ControlNet injection)
    5. Decoder blocks (self-attention + feedforward only)
    6. Output projection
    
    Note: Metadata conditioning has been REMOVED for this simplified version
    """
    def __init__(
        self,
        patch_size=100,
        hidden_dim=384,
        num_encoder_layers=4,
        num_decoder_layers=5,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        use_rope=True,
        rope_base=10.0,
        # ControlNet params
        stft_n_fft=512,
        stft_hop_length=256,
        # Metadata params
        metadata_dim=3,
        metadata_hidden_dims=None
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Patch embedding
        self.patch_proj = TimeSeriesProjection(patch_size, hidden_dim)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            PositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Spectral ControlNet for lowpass conditioning
        self.controlnet = SpectralControlNet(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            window='hann',
            dit_hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers
        )
        
        # Envelope conditioning fusion network
        self.envelope_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # NO METADATA ENCODER - Removed for simplified architecture
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            DiTEncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_rope=use_rope,
                rope_base=rope_base
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DiTDecoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_rope=use_rope,
                rope_base=rope_base
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, patch_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def unpatchify(self, x):
        """
        Convert patches back to signal
        Args:
            x: [batch, num_patches, patch_size]
        Returns:
            signal: [batch, seq_len]
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Flatten patches
        return x
    
    def forward(
        self,
        x,
        t,
        x_lowpass=None,
        x_envelope=None,
        metadata=None,
        boundaries=None,
        actual_length=None,
        position_ids=None
    ):
        """
        Forward pass
        
        Args:
            x: Noisy signal in ORIGINAL scale [batch, seq_len]
            t: Timestep [batch] or scalar
            x_lowpass: Lowpass conditioning signal in ORIGINAL scale [batch, seq_len]
            metadata: IGNORED (kept for interface compatibility)
            boundaries: List of (start, end) tuples for packed sequences
            actual_length: Actual length of packed content (excluding padding)
            position_ids: Position IDs for RoPE [batch, seq_len or num_patches]
        
        Returns:
            noise_pred: Predicted noise [batch, seq_len]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x_patches = self.patch_proj(x)  # [batch, num_patches, hidden_dim]
        num_patches = x_patches.shape[1]
        
        # Timestep embedding
        if t.dim() == 0:
            t = t.expand(batch_size)
        t_emb = self.time_embed(t)  # [batch, hidden_dim]
        
        # Generate ControlNet conditioning from lowpass
        control_features = None
        if x_lowpass is not None:
            control_features = self.controlnet(x_lowpass)  # List of [batch, hidden_dim, 1]
        
        # Process envelope conditioning and fuse with patch embeddings
        if x_envelope is not None:
            # Patchify envelope using same method as input signal
            envelope_patches = self.patch_proj(x_envelope)  # [batch, num_patches, hidden_dim]
            
            # Fuse envelope with noisy signal patches
            combined = torch.cat([x_patches, envelope_patches], dim=-1)  # [batch, num_patches, hidden_dim*2]
            cond_embed = self.envelope_fusion(combined)  # [batch, num_patches, hidden_dim]
            
            # Add envelope conditioning to patch embeddings
            x_patches = x_patches + cond_embed
        
        # NO METADATA ENCODING - Simplified architecture
        
        # Adjust position_ids if needed (convert from seq_len to num_patches)
        if position_ids is not None and position_ids.shape[1] != num_patches:
            # Downsample position_ids to match patches
            position_ids = position_ids[:, ::self.patch_size][:, :num_patches]
        
        # Encoder with ControlNet injection
        x = x_patches
        for i, encoder_block in enumerate(self.encoder_blocks):
            control = control_features[i] if control_features is not None else None
            x = encoder_block(
                x,
                t_emb,
                boundaries=boundaries,
                actual_length=actual_length,
                position_ids=position_ids,
                control_signal=control
            )
        
        # Decoder (no metadata cross-attention)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(
                x,
                t_emb,
                metadata_features=None,  # No metadata
                boundaries=boundaries,
                actual_length=actual_length
            )
        
        # Output projection
        x = self.final_norm(x)
        x = self.output_proj(x)  # [batch, num_patches, patch_size]
        
        # Unpatchify to get full signal
        noise_pred = self.unpatchify(x)  # [batch, seq_len]
        
        return noise_pred


if __name__ == "__main__":
    # Test SignalDiTV2
    print("Testing Signal DiT V2...")
    
    batch_size = 2
    seq_len = 30000
    patch_size = 100
    hidden_dim = 384
    
    # Create model
    model = SignalDiTV2(
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=4,
        num_decoder_layers=5,
        num_heads=6,
        mlp_ratio=4.0,
        use_rope=True,
        rope_base=10.0,
        stft_n_fft=512,
        stft_hop_length=256,
        metadata_dim=3
    )
    
    # Create test inputs
    x = torch.randn(batch_size, seq_len)
    t = torch.tensor([0.5, 0.8])
    x_lowpass = torch.randn(batch_size, seq_len)
    metadata = torch.tensor([[6.5, 300.0, 50.0], [7.0, 500.0, 100.0]])
    
    # Forward pass
    noise_pred = model(x, t, x_lowpass=x_lowpass, metadata=metadata)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {noise_pred.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("Signal DiT V2 test passed!")
    
    # Test with packed sequences
    from packed_attention import PackedSequenceManager
    
    print("\nTesting with packed sequences...")
    seq1 = torch.randn(15000)
    seq2 = torch.randn(20000)
    
    manager = PackedSequenceManager()
    packed, lengths, boundaries = manager.pack_sequences([seq1, seq2], max_length=seq_len)
    
    # Create mask and position IDs
    mask = manager.create_block_diagonal_mask(boundaries, seq_len, num_heads=6)
    pos_ids = manager.create_position_ids(boundaries, seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward with packing
    noise_pred_packed = model(
        packed[:batch_size],
        t,
        x_lowpass=x_lowpass,
        metadata=metadata,
        attention_mask=mask,
        position_ids=pos_ids
    )
    
    print(f"Packed output shape: {noise_pred_packed.shape}")
    print("Packed sequence test passed!")
