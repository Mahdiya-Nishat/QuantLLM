# modules/routing_module.py
# The Routing Module - decides which tokens go to quantum vs classical path

import torch
import torch.nn as nn
import math

class RoutingModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Two learned projection matrices Wq and Wk
        # They project vt from d_model (128) down to da (32)
        # This is NOT the same as transformer Q,K - this is just for scoring!
        self.Wq = nn.Linear(config.d_model, config.da, bias=False)
        self.Wk = nn.Linear(config.d_model, config.da, bias=False)

        # Threshold theta_qc and sharpness alpha from config
        # We make theta_qc a learnable parameter so it can adapt during training
        self.theta_qc = nn.Parameter(torch.tensor(config.theta_qc))
        self.alpha = config.alpha

    def forward(self, vt):
        # vt shape: (batch_size, seq_len, d_model) = (16, 128, 128)

        # Step 1: Project vt two ways
        q = self.Wq(vt)   # shape: (batch, seq_len, da)
        k = self.Wk(vt)   # shape: (batch, seq_len, da)

        # Step 2: Compute routing score st for each token
        # Dot product between q and k, scaled by sqrt(da)
        # sum(q*k) along last dimension gives one score per token
        st = (q * k).sum(dim=-1) / math.sqrt(self.Wq.out_features)
        # st shape: (batch, seq_len) - one score per token

        # Step 3: Sharpened sigmoid to get soft routing decision
        # During training this is soft (0 to 1)
        # During inference we threshold at 0.5 for hard decision
        rt = torch.sigmoid(self.alpha * (st - self.theta_qc))
        # rt shape: (batch, seq_len) - one routing value per token

        return rt, st