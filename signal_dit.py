"""
Signal Diffusion Transformer (Signal-DiT) for Broadband Earthquake Ground Motion Generation

This module implements a diffusion transformer specifically designed for 1D time series 
generation of earthquake signals, with cross-attention conditioning on low-frequency signals.

Author: Adapted for earthquake signal generation
"""

import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for time steps in diffusion"""
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
        
        # Ensure x is at least 1D for torch.outer
        if x.dim() == 0:
            x = x.unsqueeze(0)
        
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimeSeriesProjection(nn.Module):
    """Projects 1D time series to patch embeddings"""
    def __init__(self, seq_len, patch_size, dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.dim = dim
        
        # Use 1D convolution to create patches
        self.proj = nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, seq_len)
        x = self.proj(x)  # (batch_size, dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, dim)
        return x

class CrossAttention(nn.Module):
    """Multi-Head Cross-Attention for conditioning on low-frequency signals"""
    def __init__(self, dim, n_heads, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        
        # Query from broadband signal, Key/Value from low-frequency signal
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
    def forward(self, x_broad, x_low):
        # x_broad: (batch, seq_len, dim) - broadband signal patches
        # x_low: (batch, seq_len, dim) - low-frequency signal patches
        B, L, D = x_broad.shape
        _, L_low, _ = x_low.shape
        
        q = self.q_proj(x_broad).view(B, L, self.n_heads, -1).transpose(1, 2)
        k = self.k_proj(x_low).view(B, L_low, self.n_heads, -1).transpose(1, 2)
        v = self.v_proj(x_low).view(B, L_low, self.n_heads, -1).transpose(1, 2)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out

class LinformerAttention(nn.Module):
    """Efficient Linformer attention for long sequences"""
    def __init__(self, seq_len, dim, n_heads, k, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim, bias=bias)
        self.kw = nn.Linear(dim, dim, bias=bias)
        self.vw = nn.Linear(dim, dim, bias=bias)

        # Linformer projection matrices
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))

        self.ow = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        B, L, D = q.shape
        q = torch.reshape(q, [B, L, self.n_heads, -1])
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.reshape(k, [B, L, self.n_heads, -1])
        k = torch.permute(k, [0, 2, 3, 1])
        v = torch.reshape(v, [B, L, self.n_heads, -1])
        v = torch.permute(v, [0, 2, 3, 1])
        
        # Apply Linformer projections
        k = torch.matmul(k, self.E[:L, :])
        v = torch.matmul(v, self.F[:L, :])
        v = torch.permute(v, [0, 1, 3, 2])

        qk = torch.matmul(q, k) * self.scale
        attn = torch.softmax(qk, dim=-1)

        v_attn = torch.matmul(attn, v)
        v_attn = torch.permute(v_attn, [0, 2, 1, 3])
        v_attn = torch.reshape(v_attn, [B, L, D])

        x = self.ow(v_attn)
        return x

def modulate(x, shift, scale):
    """Apply feature-wise linear modulation"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class EncoderBlock(nn.Module):
    """Encoder block for processing low-pass conditioning signal"""
    def __init__(self, seq_len, dim, heads, mlp_dim, k, rate=0.0):
        super().__init__()
        
        # Self-attention for encoding low-pass signal
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.self_attn = LinformerAttention(seq_len, dim, heads, k)
        
        # MLP for feature transformation
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate)
        )

    def forward(self, x):
        # Self-attention on low-pass signal
        x = x + self.self_attn(self.ln_1(x))
        
        # MLP transformation
        x = x + self.mlp(self.ln_2(x))
        
        return x


class DecoderBlock(nn.Module):
    """Decoder block for generating broadband signal with cross-attention to encoder"""
    def __init__(self, seq_len, dim, heads, mlp_dim, k, rate=0.0):
        super().__init__()
        
        # Self-attention components (MMHA for broadband signal)
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = LinformerAttention(seq_len, dim, heads, k)
        
        # Cross-attention components (to encoder output) - ENABLE affine for better adaptation
        self.ln_cross = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.cross_attn = CrossAttention(dim, heads)
        
        # MLP components
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate),
        )
        
        # Modulation layers for timestep conditioning
        # Use small initialization for scale/shift, but NOT zero for gates
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1 = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2 = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim)
        self.scale_2 = nn.Linear(dim, dim)
        self.scale_cross = nn.Linear(dim, dim)
        
        # Initialize modulation layers with proper scaling
        self._init_weights([self.gamma_1, self.beta_1, self.gamma_2, self.beta_2])
        self._init_gate_weights([self.scale_1, self.scale_2, self.scale_cross])

    def _init_weights(self, layers):
        """Initialize scale/shift layers to small values"""
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def _init_gate_weights(self, layers):
        """Initialize gate layers to allow information flow"""
        for layer in layers:
            # Initialize weight to near-identity for gates
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            # Initialize bias to 1.0 so gates start open (multiplicative gating)
            nn.init.constant_(layer.bias, 1.0)

    def forward(self, x, timestep_emb, encoder_output):
        # Self-attention with timestep modulation (MMHA)
        scale_msa = self.gamma_1(timestep_emb)
        shift_msa = self.beta_1(timestep_emb)
        gate_msa = torch.sigmoid(self.scale_1(timestep_emb)).unsqueeze(1)
        
        attn_out = self.self_attn(modulate(self.ln_1(x), shift_msa, scale_msa))
        x = x + attn_out * gate_msa
        
        # CRITICAL FIX: Much stronger cross-attention for envelope enforcement
        # Instead of learned gating that can collapse to 0, use fixed strong weighting
        cross_out = self.cross_attn(self.ln_cross(x), encoder_output)
        
        # Direct strong blending: 70% cross-attention, 30% residual
        # This ensures conditioning information is ALWAYS used strongly
        x = 0.7 * cross_out + 0.3 * x
        
        # ADDITIONAL: Direct encoder injection for even stronger conditioning
        if encoder_output.shape == x.shape:
            x = x + 0.3 * encoder_output  # Increased from 0.2 to 0.3
        
        # MLP with timestep modulation
        scale_mlp = self.gamma_2(timestep_emb)
        shift_mlp = self.beta_2(timestep_emb)
        gate_mlp = torch.sigmoid(self.scale_2(timestep_emb)).unsqueeze(1)
        
        mlp_out = self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp))
        x = x + mlp_out * gate_mlp
        
        return x


class FinalLayer(nn.Module):
    """Final layer to convert transformer output back to time series"""
    def __init__(self, dim, patch_size, seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.num_patches = seq_len // patch_size
        
        self.ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        
        # Initialize with small values for better gradient flow
        self._init_weights()

    def _init_weights(self):
        # CRITICAL FIX: Proper initialization for noise prediction (target magnitude ~2-5)
        # Use He initialization scaled for expected noise magnitude
        fan_in = self.linear.weight.shape[1] 
        noise_scale = 2.0  # Expected noise std based on dataset analysis
        
        # Scale He initialization by expected noise magnitude
        he_std = (2.0 / fan_in) ** 0.5
        final_std = he_std * noise_scale
        
        nn.init.normal_(self.linear.weight, mean=0.0, std=final_std)
        nn.init.zeros_(self.linear.bias)
        
        # Gamma/beta: zero initialization (AdaLN modulation starts neutral)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        
        print(f"🔧 FinalLayer initialized: std={final_std:.4f} (He: {he_std:.4f} × noise_scale: {noise_scale})")

    def forward(self, x, c):
        scale = self.gamma(c)
        shift = self.beta(c)
        x = modulate(self.ln(x), shift, scale)
        x = self.linear(x)  # (batch, num_patches, patch_size)
        
        # Reshape back to original sequence length
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # (batch, seq_len)
        
        return x

class SignalDiT(nn.Module):
    """
    Encoder-Decoder Signal Diffusion Transformer for broadband earthquake ground motion generation
    
    Architecture:
    - Encoder: Processes low-pass conditioning signal → contextual representation Z
    - Decoder: Generates broadband signal using self-attention + cross-attention to Z
    - Timestep embedding conditions decoder layers
    
    Args:
        seq_len: Length of the time series
        dim: Model dimension  
        patch_size: Size of each time series patch
        encoder_depth: Number of encoder layers
        decoder_depth: Number of decoder layers
        heads: Number of attention heads
        mlp_dim: MLP hidden dimension
        k: Linformer projection dimension
    """
    def __init__(self, seq_len=6000, dim=300, patch_size=20, encoder_depth=4, decoder_depth=8, 
                 heads=10, mlp_dim=512, k=64):
        super().__init__()
        
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.dim = dim
        
        # Patch embedding for broadband signal (decoder input)
        self.broadband_proj = TimeSeriesProjection(seq_len, patch_size, dim)
        
        # Patch embedding for conditioning signal (encoder input)
        self.lowpass_proj = TimeSeriesProjection(seq_len, patch_size, dim)
        
        # Positional embeddings (shared between encoder and decoder)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        
        # Timestep embedding for decoder conditioning with better scaling
        # Scale factor to amplify timestep signal (typical sigma range 0.01-10)
        self.time_embed = nn.Sequential(
            PositionalEmbedding(dim, scale=1.0),  # Positional encoding
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        # Initialize timestep MLP with proper scaling
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)
        
        # ENCODER: Processes low-pass conditioning signal
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(self.num_patches, dim, heads, mlp_dim, k)
            for _ in range(encoder_depth)
        ])
        
        # DECODER: Generates broadband signal with cross-attention to encoder
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(self.num_patches, dim, heads, mlp_dim, k)
            for _ in range(decoder_depth)
        ])
        
        # Final layer to convert decoder output back to time series
        self.final_layer = FinalLayer(dim, patch_size, seq_len)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, std=0.02)
    
    def forward(self, x, t, x_cond=None):
        """
        Args:
            x: Noisy broadband signal (batch, seq_len)
            t: Timestep (batch,)
            x_cond: Low-frequency conditioning signal (batch, seq_len)
        
        Returns:
            Generated/denoised broadband signal (batch, seq_len)
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Project broadband signal to patches (decoder input)
        x_broad = self.broadband_proj(x)  # (batch, num_patches, dim)
        x_broad = x_broad + self.pos_embed
        
        # ENCODER: Process low-frequency conditioning signal
        if x_cond is not None:
            # Project conditioning signal to patches
            x_low = self.lowpass_proj(x_cond)  # (batch, num_patches, dim)
            x_low = x_low + self.pos_embed
            
            # Pass through encoder layers
            encoder_output = x_low
            for encoder_block in self.encoder_blocks:
                encoder_output = encoder_block(encoder_output)
            # encoder_output: (batch, num_patches, dim) - contextual representation Z
        else:
            # If no conditioning, use zero encoder output
            encoder_output = torch.zeros(batch_size, self.num_patches, self.dim, device=device)
        
        # Timestep embedding for decoder conditioning
        t_emb = self.time_embed(t)  # (batch, dim)
        
        # DECODER: Generate broadband signal with cross-attention to encoder
        decoder_output = x_broad
        for decoder_block in self.decoder_blocks:
            decoder_output = decoder_block(decoder_output, t_emb, encoder_output)
        
        # Final projection back to time series
        output = self.final_layer(decoder_output, t_emb)
        
        return output

# Test the model
if __name__ == "__main__":
    # Test with sample data
    batch_size = 4
    seq_len = 4096
    
    # Create sample signals
    x_noisy = torch.randn(batch_size, seq_len)  # Noisy broadband signal
    x_cond = torch.randn(batch_size, seq_len)   # Low-frequency conditioning
    t = torch.randint(0, 1000, (batch_size,)).float()  # Timesteps
    
    # Initialize model
    model = SignalDiT(seq_len=seq_len, dim=256, patch_size=16, depth=6)
    
    # Forward pass
    with torch.no_grad():
        output = model(x_noisy, t, x_cond)
        print(f"Input shape: {x_noisy.shape}")
        print(f"Conditioning shape: {x_cond.shape}")
        print(f"Output shape: {output.shape}")
        print("Signal DiT test successful!")
