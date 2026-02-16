"""
Rotary Position Embeddings (RoPE) for 1D sequences.
Adapted from LLM implementations for seismic signal processing.
"""

import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding for 1D sequences.
    
    Args:
        dim: Dimension of the embeddings (should be even)
        base: Base for frequency computation (default: 10.0 for seismic signals)
        max_seq_len: Maximum sequence length (default: 50000 for packed signals)
    """
    def __init__(self, dim, base=10.0, max_seq_len=50000):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even for RoPE"
        
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute frequency tensor
        # inv_freq has shape [dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin cache for efficiency
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        """Precompute cos/sin cache for positions [0, seq_len)"""
        # FIX BUG #5: Ensure correct device handling during dynamic cache rebuild
        # t has shape [seq_len]
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        
        # freqs has shape [seq_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # emb has shape [seq_len, dim] by concatenating [sin, cos]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Direct assignment instead of register_buffer for dynamic rebuild
        # This ensures device consistency when cache is rebuilt after module.to(device)
        self.cos_cache = emb.cos().to(self.inv_freq.device)
        self.sin_cache = emb.sin().to(self.inv_freq.device)
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        # x has shape [..., dim]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, position_ids=None):
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
            k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
            position_ids: Optional position indices [batch, seq_len]. If None, uses range(seq_len)
        
        Returns:
            q_rot, k_rot: Rotated query and key tensors
        """
        batch_size, seq_len = q.shape[0], q.shape[1]
        
        # Get position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
        
        # Expand cache if needed
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
        
        # Get cos/sin for positions
        # cos/sin have shape [batch, seq_len, 1, dim]
        cos = self.cos_cache[position_ids].unsqueeze(2)
        sin = self.sin_cache[position_ids].unsqueeze(2)
        
        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rot, k_rot
    
    def forward(self, q, k, position_ids=None):
        """
        Forward pass - alias for apply_rotary_pos_emb.
        
        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
            position_ids: Optional position indices [batch, seq_len]
        
        Returns:
            q_rot, k_rot: Rotated tensors in same format as input
        """
        # Handle both [B, L, H, D] and [B, H, L, D] formats
        if q.dim() == 4 and q.shape[1] != k.shape[1]:
            # Assume [B, H, L, D] format - transpose to [B, L, H, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            transposed = True
        else:
            transposed = False
        
        q_rot, k_rot = self.apply_rotary_pos_emb(q, k, position_ids)
        
        # Transpose back if needed
        if transposed:
            q_rot = q_rot.transpose(1, 2)
            k_rot = k_rot.transpose(1, 2)
        
        return q_rot, k_rot


def apply_rope_1d(q, k, positions, dim, base=10.0):
    """
    Standalone function to apply RoPE to query and key tensors.
    Useful for ad-hoc applications without creating a module.
    
    Args:
        q: Query tensor [..., seq_len, dim]
        k: Key tensor [..., seq_len, dim]
        positions: Position indices [..., seq_len]
        dim: Embedding dimension
        base: Frequency base
    
    Returns:
        q_rot, k_rot: Rotated tensors
    """
    assert dim % 2 == 0
    
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=q.device).float() / dim))
    
    # Compute freqs for positions
    freqs = torch.einsum('...i,j->...ij', positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    
    cos = emb.cos().unsqueeze(-2)  # [..., seq_len, 1, dim]
    sin = emb.sin().unsqueeze(-2)
    
    # Rotate
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot, k_rot


if __name__ == "__main__":
    # Test RoPE implementation
    print("Testing Rotary Position Embeddings...")
    
    batch_size = 2
    seq_len = 1000
    num_heads = 4
    head_dim = 64
    
    # Create test tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # Create RoPE module
    rope = RotaryPositionEmbedding(dim=head_dim, base=10.0, max_seq_len=5000)
    
    # Apply rotation
    q_rot, k_rot = rope(q, k)
    
    print(f"Input Q shape: {q.shape}")
    print(f"Output Q shape: {q_rot.shape}")
    print(f"RoPE applied successfully!")
    
    # Test with custom positions (for packed sequences)
    position_ids = torch.cat([
        torch.arange(500),  # First sequence: 0-499
        torch.arange(500)   # Second sequence: 0-499 (reset positions)
    ]).unsqueeze(0).expand(batch_size, -1)
    
    q_packed = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k_packed = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    q_rot_packed, k_rot_packed = rope(q_packed, k_packed, position_ids)
    print(f"Packed sequence RoPE applied successfully!")
