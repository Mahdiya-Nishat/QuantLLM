# modules/token_aggregation.py
# Combines classical and quantum path outputs
# using routing decision rt as blending weight

import torch
import torch.nn as nn

class TokenAggregation(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Final FC layer to project to vocabulary size
        # yF (128) → zt (50257) → probability over all words
        self.fc = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, yC, yQ, rt):
        # yC shape: (batch, seq_len, 128) - classical output
        # yQ shape: (batch, seq_len, 128) - quantum output
        # rt shape: (batch, seq_len)      - routing decisions

        # Expand rt to match yC and yQ dimensions
        # (batch, seq_len) → (batch, seq_len, 1)
        # the 1 will broadcast across 128 dimensions
        rt_expanded = rt.unsqueeze(-1)

        # Combine classical and quantum outputs
        # This is equation from the paper: yF = (1-rt)yC + rt*yQ
        yF = (1 - rt_expanded) * yC + rt_expanded * yQ
        # yF shape: (batch, seq_len, 128)

        # Project to vocabulary space
        # (batch, seq_len, 128) → (batch, seq_len, 50257)
        zt = self.fc(yF)

        return zt, yF