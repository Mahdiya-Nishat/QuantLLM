# model.py
# Assembles all modules into complete QuantLLM model

import torch
import torch.nn as nn
from modules.routing_module import RoutingModule
from modules.classical_path import ClassicalPath
from modules.quantum_embedding import QuantumEmbedding
from modules.qsa_module import QSAModule
from modules.qac_module import QACModule
from modules.token_aggregation import TokenAggregation

class QuantLLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Embedding layer - converts token IDs to vectors
        # (vocab_size, d_model) = (50257, 128) lookup table
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # All modules
        self.router = RoutingModule(config)
        self.classical_path = ClassicalPath(config)
        self.quantum_emb = QuantumEmbedding(config)
        self.qsa = QSAModule(config)
        self.qac = QACModule(config)
        self.aggregator = TokenAggregation(config)

        # Store config
        self.config = config

    def forward(self, token_ids):
        # token_ids shape: (batch, seq_len) - integers!
        # e.g. [[2368, 502, 1234, ...], [...]]

        # Step 1: Convert token IDs to vectors
        vt = self.embedding(token_ids)
        # vt shape: (batch, seq_len, 128)

        # Step 2: Compute routing decisions
        rt, st = self.router(vt)
        # rt shape: (batch, seq_len)

        # Step 3: Classical path
        yC = self.classical_path(vt)
        # yC shape: (batch, seq_len, 128)

        # Step 4: Quantum path
        quantum_states, angles = self.quantum_emb(vt)
        psi_Q, psi_K, psi_V = self.qsa(angles)
        yQ, xi = self.qac(psi_Q, psi_K, psi_V)
        # yQ shape: (batch, seq_len, 128)

        # Step 5: Combine and predict
        zt, yF = self.aggregator(yC, yQ, rt)
        # zt shape: (batch, seq_len, 50257)

        return zt, rt

    def count_parameters(self):
        # Useful to know how many parameters we have!
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")