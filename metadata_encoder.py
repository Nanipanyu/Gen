"""
Metadata encoder for conditioning diffusion models on seismic metadata.
Uses MLP projection followed by cross-attention to inject metadata information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class MetadataEncoder(nn.Module):
    """
    Encodes seismic metadata (Magnitude, Vs30, Hypocenter Distance) into 
    a representation suitable for cross-attention conditioning.
    
    Args:
        metadata_dim: Input metadata dimension (default: 3 for M, Vs30, HypD)
        hidden_dims: List of hidden dimensions for MLP projection
        output_dim: Output dimension matching DiT hidden dimension
        num_encoder_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(
        self,
        metadata_dim: int = 3,
        hidden_dims: list = None,
        output_dim: int = 384,
        num_encoder_layers: int = 2,
        num_heads: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        
        self.metadata_dim = metadata_dim
        self.output_dim = output_dim
        self.num_encoder_layers = num_encoder_layers
        
        # MLP projection for metadata
        mlp_layers = []
        in_dim = metadata_dim
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final projection to output dimension
        mlp_layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Learnable query tokens for metadata encoding
        # We use multiple tokens to capture different aspects of metadata
        self.num_metadata_tokens = 4
        self.metadata_queries = nn.Parameter(
            torch.randn(1, self.num_metadata_tokens, output_dim) * 0.02
        )
        
        # Transformer encoder layers for metadata processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Layer norm for output
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Encode metadata into representation for cross-attention.
        
        Args:
            metadata: Metadata tensor [batch, metadata_dim]
                     Expected: [Magnitude, Vs30, HypD]
        
        Returns:
            encoded_metadata: Encoded representation [batch, num_metadata_tokens, output_dim]
        """
        batch_size = metadata.shape[0]
        
        # Project metadata through MLP
        metadata_proj = self.mlp(metadata)  # [batch, output_dim]
        
        # Expand metadata projection to match query tokens
        metadata_proj = metadata_proj.unsqueeze(1)  # [batch, 1, output_dim]
        
        # Expand query tokens for batch
        queries = self.metadata_queries.expand(batch_size, -1, -1)  # [batch, num_tokens, output_dim]
        
        # Combine metadata projection with query tokens
        # This allows the transformer to attend to both the raw metadata and learned queries
        combined = torch.cat([metadata_proj, queries], dim=1)  # [batch, 1+num_tokens, output_dim]
        
        # Process through transformer encoder
        encoded = self.transformer_encoder(combined)  # [batch, 1+num_tokens, output_dim]
        
        # Apply output normalization
        encoded = self.output_norm(encoded)
        
        return encoded


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for conditioning signal on metadata.
    Signal tokens (queries) attend to metadata tokens (keys/values).
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection (from signal)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Key and Value projections (from metadata)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        signal_features: torch.Tensor,
        metadata_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-attention from signal to metadata.
        
        Args:
            signal_features: Signal tokens [batch, num_signal_tokens, hidden_dim]
            metadata_features: Metadata tokens [batch, num_metadata_tokens, hidden_dim]
        
        Returns:
            output: Signal tokens conditioned on metadata [batch, num_signal_tokens, hidden_dim]
        """
        batch_size = signal_features.shape[0]
        num_signal_tokens = signal_features.shape[1]
        num_metadata_tokens = metadata_features.shape[1]
        
        # Project queries from signal
        q = self.q_proj(signal_features)  # [batch, num_signal_tokens, hidden_dim]
        
        # Project keys and values from metadata
        k = self.k_proj(metadata_features)  # [batch, num_metadata_tokens, hidden_dim]
        v = self.v_proj(metadata_features)  # [batch, num_metadata_tokens, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, num_signal_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_metadata_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_metadata_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, num_heads, num_tokens, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch, num_heads, num_signal_tokens, num_metadata_tokens]
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # [batch, num_heads, num_signal_tokens, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_signal_tokens, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(signal_features + output)
        
        return output


class MetadataConditionedBlock(nn.Module):
    """
    A transformer block with self-attention, cross-attention to metadata, and feedforward.
    Used in DiT decoder layers.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention to metadata
        self.cross_attn = CrossAttentionLayer(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        metadata_features: torch.Tensor,
        boundaries: Optional[List] = None,
        actual_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input signal features [batch, num_tokens, hidden_dim]
            metadata_features: Encoded metadata [batch, num_metadata_tokens, hidden_dim]
            boundaries: List of (start, end) tuples for packed sequences
            actual_length: Actual length of packed content
        
        Returns:
            output: Conditioned features [batch, num_tokens, hidden_dim]
        """
        # Self-attention with residual (no mask needed for now - all tokens attend to all)
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=None)
        x = x + attn_out
        
        # Cross-attention to metadata with residual
        x_norm = self.norm2(x)
        cross_attn_out = self.cross_attn(x_norm, metadata_features)
        x = x + (cross_attn_out - x_norm)  # Residual is already applied in cross_attn
        
        # Feedforward with residual
        x = x + self.mlp(self.norm3(x))
        
        return x


if __name__ == "__main__":
    # Test MetadataEncoder
    print("Testing Metadata Encoder...")
    
    batch_size = 4
    metadata_dim = 3
    output_dim = 384
    num_signal_tokens = 300
    
    # Create test metadata (M, Vs30, HypD)
    metadata = torch.tensor([
        [6.5, 300.0, 50.0],
        [7.0, 500.0, 100.0],
        [5.5, 200.0, 30.0],
        [6.0, 400.0, 75.0]
    ])
    
    # Create metadata encoder
    encoder = MetadataEncoder(
        metadata_dim=metadata_dim,
        hidden_dims=[64, 128, 256],
        output_dim=output_dim,
        num_encoder_layers=2,
        num_heads=6
    )
    
    # Encode metadata
    encoded_metadata = encoder(metadata)
    
    print(f"Input metadata shape: {metadata.shape}")
    print(f"Encoded metadata shape: {encoded_metadata.shape}")
    print(f"Number of metadata tokens: {encoded_metadata.shape[1]}")
    
    # Test CrossAttentionLayer
    signal_features = torch.randn(batch_size, num_signal_tokens, output_dim)
    
    cross_attn = CrossAttentionLayer(
        hidden_dim=output_dim,
        num_heads=6
    )
    
    conditioned_features = cross_attn(signal_features, encoded_metadata)
    
    print(f"Signal features shape: {signal_features.shape}")
    print(f"Conditioned features shape: {conditioned_features.shape}")
    
    # Test MetadataConditionedBlock
    block = MetadataConditionedBlock(
        hidden_dim=output_dim,
        num_heads=6,
        mlp_ratio=4.0
    )
    
    output = block(signal_features, encoded_metadata)
    
    print(f"Block output shape: {output.shape}")
    print("Metadata encoder test passed!")
