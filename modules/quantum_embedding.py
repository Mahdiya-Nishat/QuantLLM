# modules/quantum_embedding.py
# Compresses token vectors (128-dim) into 4-qubit quantum states
# Uses angle encoding via RY rotation gates

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from config import Config


class QuantumEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_qubits = config.n_qubits  # 4
        self.delta = 1.0  # scaling factor for angles

        # Learned projection matrix Wpr: 128 -> 4
        # This is the classical compression step
        self.Wpr = nn.Linear(config.d_model, config.n_qubits, bias=False)

        # Define the quantum device (simulator on CPU)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Define the quantum circuit as a QNode
        # This is the actual quantum circuit that runs on the simulator
        @qml.qnode(self.dev, interface="torch")
        def circuit(angles):
            # angles shape: (n_qubits,) - one angle per qubit

            # Initialize all qubits to |0> and rotate each one
            for i in range(self.n_qubits):
                qml.RY(angles[i], wires=i)

            # Return the state vector of the full quantum system
            return qml.state()

        self.circuit = circuit

    def forward(self, vt):
        # vt shape: (batch, seq_len, d_model)
        batch_size, seq_len, _ = vt.shape

        # Step 1: Classical compression 128 -> 4
        v_compressed = self.Wpr(vt)  # (batch, seq_len, 4)

        # Step 2: Scale to rotation angles
        angles = self.delta * v_compressed  # (batch, seq_len, 4)

        # Step 3: Encode each token into quantum state
        # We need to run circuit for each token individually
        quantum_states = []

        for b in range(batch_size):
            batch_states = []
            for t in range(seq_len):
                # Get angles for this specific token
                token_angles = angles[b, t]  # shape: (4,)

                # Run quantum circuit!
                # Returns complex state vector of size 2^4 = 16
                state = self.circuit(token_angles)  # shape: (16,)

                # Take real part only for now
                # (we'll use full complex in QSA module)
                batch_states.append(state.real)

            quantum_states.append(torch.stack(batch_states))

        # Stack everything back together
        # Shape: (batch, seq_len, 2^n_qubits) = (batch, seq_len, 16)
        quantum_states = torch.stack(quantum_states)

        return quantum_states, angles