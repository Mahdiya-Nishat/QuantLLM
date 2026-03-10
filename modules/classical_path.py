# modules/classical_path.py
# The Classical Path - standard transformer block
# Used for simple tokens that don't need quantum processing

import torch
import torch.nn as nn
import math

class ClassicalPath(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Multi-Head Self Attention
        # PyTorch has this built in - no need to code from scratch!
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,    # 128
            num_heads=config.n_heads,    # 2
            batch_first=True             # (batch, seq, features) order
        )

        # Layer normalisation - stabilises training
        # Applied before attention and before FFN (Pre-LN style)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Feed Forward Network
        # 128 -> 256 -> 128
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),  # 128 -> 256
            nn.ReLU(),
            nn.Linear(config.ffn_dim, config.d_model)   # 256 -> 128
        )

    def forward(self, vt):
        # vt shape: (batch, seq_len, d_model)

        # Step 1: Multi-Head Self Attention
        # Pre-layer norm first for stability
        normed = self.norm1(vt)

        # All three (query, key, value) come from same input
        # This is "self" attention - tokens attend to each other
        attn_out, _ = self.attention(normed, normed, normed)

        # Residual connection - add input back to output
        # This prevents vanishing gradients in deep networks
        x = vt + attn_out

        # Step 2: Feed Forward Network
        # Pre-layer norm again
        normed2 = self.norm2(x)
        ffn_out = self.ffn(normed2)

        # Another residual connection
        yC = x + ffn_out
        # yC shape: (batch, seq_len, d_model) = same as input!

        return yC