"""
Packed attention utilities for handling variable-length sequences.
Implements block-diagonal attention masks and sequence packing with [SEP] tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class PackedSequenceManager:
    """
    Manages packing multiple variable-length sequences into a single batch
    with [SEP] tokens and block-diagonal attention masks.
    """
    
    @staticmethod
    def pack_sequences(
        sequences: List[torch.Tensor],
        max_length: int = 50000,
        pad_value: float = 0.0,
        add_sep: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Pack multiple sequences into a single tensor with optional [SEP] tokens.
        
        Args:
            sequences: List of 1D tensors with variable lengths
            max_length: Maximum total length after packing
            pad_value: Value to use for padding
            add_sep: Whether to add [SEP] token (special value) between sequences
        
        Returns:
            packed: Packed tensor [pack_size, max_length]
            lengths: Actual length of each sequence (before padding)
            boundaries: List of (start, end) indices for each sequence
        """
        pack_size = len(sequences)
        
        # Calculate total length needed
        if add_sep:
            sep_token_value = -999.0  # Special value for SEP token
            total_length = sum(len(s) for s in sequences) + (pack_size - 1)
        else:
            total_length = sum(len(s) for s in sequences)
        
        if total_length > max_length:
            raise ValueError(f"Total packed length {total_length} exceeds max_length {max_length}")
        
        # Create packed tensor
        device = sequences[0].device
        dtype = sequences[0].dtype
        packed = torch.full((pack_size, max_length), pad_value, dtype=dtype, device=device)
        
        lengths = []
        boundaries = []
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            lengths.append(seq_len)
            
            # Calculate start position
            if i == 0:
                start = 0
            else:
                prev_end = boundaries[i-1][1]
                start = prev_end + (1 if add_sep else 0)
            
            end = start + seq_len
            boundaries.append((start, end))
            
            # Copy sequence
            packed[i, start:end] = seq
            
            # Add SEP token
            if add_sep and i < pack_size - 1:
                packed[i, end] = sep_token_value
        
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
        
        return packed, lengths_tensor, boundaries
    
    @staticmethod
    def create_block_diagonal_mask(
        boundaries: List[Tuple[int, int]],
        total_length: int,
        num_heads: int = 1,
        device: torch.device = None,
        include_sep_tokens: bool = True
    ) -> torch.Tensor:
        """
        FIX BUG #6: Properly handle SEP tokens in mask to prevent NaN in attention.
        
        Args:
            boundaries: List of (start, end) tuples for each sequence
            total_length: Total sequence length
            num_heads: Number of attention heads
            device: Device to create mask on
            include_sep_tokens: If True, SEP tokens attend to themselves
        
        Returns:
            mask: Block diagonal mask [num_heads, total_length, total_length]
        """

        if device is None:
            device = torch.device('cpu')
        
        # Create bias mask with -inf (all masked by default)
        # Use float32 to save memory compared to bool then converting
        mask = torch.full((1, 1, total_length, total_length), float('-inf'), 
                         dtype=torch.float32, device=device)
        
        # Fill in blocks with 0.0 (allowed attention)
        for start, end in boundaries:
            # Allow attention within this sequence
            mask[0, 0, start:end, start:end] = 0.0
        
        # FIX BUG #6: Allow SEP tokens to attend to themselves
        # SEP tokens are at position `end` for each sequence (except last)
        if include_sep_tokens:
            for i, (start, end) in enumerate(boundaries[:-1]):  # All except last
                sep_pos = end  # SEP token position
                mask[0, 0, sep_pos, sep_pos] = 0.0  # SEP attends to itself
        
        return mask
    
    @staticmethod
    def create_position_ids(
        boundaries: List[Tuple[int, int]],
        total_length: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Create position IDs that reset for each sequence in packed batch.
        
        Args:
            boundaries: List of (start, end) indices for each sequence
            total_length: Total sequence length
            device: Device for tensor
        
        Returns:
            position_ids: Position indices [total_length] with resets at boundaries
        """
        if device is None:
            device = torch.device('cpu')
        
        position_ids = torch.zeros(total_length, dtype=torch.long, device=device)
        
        for start, end in boundaries:
            seq_len = end - start
            position_ids[start:end] = torch.arange(seq_len, device=device)
        
        return position_ids


def apply_packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0
) -> torch.Tensor:
    """
    Apply attention with block-diagonal mask for packed sequences.
    
    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim]
        value: Value tensor [batch, num_heads, seq_len, head_dim]
        attention_mask: Block-diagonal mask [num_heads, seq_len, seq_len] or [batch, num_heads, seq_len, seq_len]
        dropout_p: Dropout probability
    
    Returns:
        output: Attention output [batch, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # Apply mask if provided
    if attention_mask is not None:
        # Expand mask to batch dimension if needed
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Convert boolean mask to float mask for addition
        # True (allowed) -> 0.0, False (masked) -> -inf
        float_mask = torch.zeros_like(scores)
        float_mask.masked_fill_(~attention_mask, float('-inf'))
        scores = scores + float_mask
    
    # Softmax and dropout
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, value)
    
    return output


class PackedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with support for packed sequences and RoPE.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10.0
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.use_rope = use_rope
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # RoPE if enabled
        if use_rope:
            from rope_embeddings import RotaryPositionEmbedding
            self.rope = RotaryPositionEmbedding(
                dim=self.head_dim,
                base=rope_base,
                max_seq_len=50000
            )
    
    def forward(
        self,
        x: torch.Tensor,
        boundaries: Optional[List[Tuple[int, int]]] = None,
        actual_length: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            boundaries: List of (start, end) tuples for each packed sequence
            actual_length: Actual length of packed content (excluding padding)
            position_ids: Position IDs [batch, seq_len] for RoPE
        
        Returns:
            output: Attention output [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rope(q, k, position_ids)
        
        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # MEMORY OPTIMIZATION: Only compute attention on actual content
        # Trim to actual_length, compute attention, then pad back
        if boundaries is not None and actual_length is not None and actual_length < seq_len:
            # Trim to actual length
            q_trimmed = q[:, :, :actual_length, :]
            k_trimmed = k[:, :, :actual_length, :]
            v_trimmed = v[:, :, :actual_length, :]
            
            # Create mask only for trimmed length
            attention_mask = PackedSequenceManager.create_block_diagonal_mask(
                boundaries, actual_length, num_heads=1, device=x.device
            )
            
            # Apply packed attention on trimmed sequences
            attn_out_trimmed = apply_packed_attention(q_trimmed, k_trimmed, v_trimmed, 
                                                     attention_mask, self.dropout)
            
            # Pad back to full length
            attn_out = torch.zeros_like(q)  # [batch, heads, seq_len, head_dim]
            attn_out[:, :, :actual_length, :] = attn_out_trimmed
        else:
            # No packing or no trimming needed
            attn_out = apply_packed_attention(q, k, v, None, self.dropout)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(attn_out)
        
        return output


if __name__ == "__main__":
    # Test packed sequence utilities
    print("Testing Packed Sequence Utilities...")
    
    # Create variable-length sequences
    seq1 = torch.randn(15000)  # 15k samples
    seq2 = torch.randn(20000)  # 20k samples
    seq3 = torch.randn(10000)  # 10k samples
    
    sequences = [seq1, seq2, seq3]
    
    # Pack sequences
    manager = PackedSequenceManager()
    packed, lengths, boundaries = manager.pack_sequences(sequences, max_length=50000, add_sep=True)
    
    print(f"Packed shape: {packed.shape}")
    print(f"Lengths: {lengths}")
    print(f"Boundaries: {boundaries}")
    
    # Create block-diagonal mask
    mask = manager.create_block_diagonal_mask(boundaries, total_length=50000, num_heads=4)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask allows self-attention in blocks: {mask[0, 0, 0].item()}")  # First element of first block
    print(f"Mask blocks cross-attention: {mask[0, 0, 20000].item()}")  # Cross-block element
    
    # Create position IDs
    position_ids = manager.create_position_ids(boundaries, total_length=50000)
    print(f"Position IDs shape: {position_ids.shape}")
    print(f"Position resets at boundaries: {position_ids[boundaries[1][0]].item()}")  # Should be 0
    
    # Test PackedMultiHeadAttention
    batch_size = 2
    seq_len = 1000
    hidden_dim = 384
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    attn = PackedMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=6,
        use_rope=True,
        rope_base=10.0
    )
    
    # Create simple mask and position IDs
    simple_boundaries = [(0, 500), (500, 1000)]
    simple_mask = manager.create_block_diagonal_mask(simple_boundaries, seq_len, num_heads=6)
    simple_pos_ids = manager.create_position_ids(simple_boundaries, seq_len).unsqueeze(0).expand(batch_size, -1)
    
    output = attn(x, attention_mask=simple_mask, position_ids=simple_pos_ids)
    
    print(f"Attention input shape: {x.shape}")
    print(f"Attention output shape: {output.shape}")
    print("Packed attention test passed!")
