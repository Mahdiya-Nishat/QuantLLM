# modules/qsa_module.py
# Quantum Self Attention Module
# Takes token angles and produces 3 quantum states: Q, K, V
# Each through its own VQC with learnable parameters

import torch
import torch.nn as nn
import pennylane as qml

class QSAModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_qubits = config.n_qubits  # 4
        self.n_layers = config.vqc_layers  # 2

        # Learnable VQC parameters for Q, K, V
        # Each qubit has 3 angles (RZ, RY, RZ) per layer
        # Shape: (n_layers, n_qubits, 3)
        self.theta_Q = nn.Parameter(
            torch.randn(config.vqc_layers, config.n_qubits, 3))
        self.theta_K = nn.Parameter(
            torch.randn(config.vqc_layers, config.n_qubits, 3))
        self.theta_V = nn.Parameter(
            torch.randn(config.vqc_layers, config.n_qubits, 3))

        # One quantum device for all three circuits
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Define VQC circuit
        # Same structure for Q, K, V - just different learned parameters
        @qml.qnode(self.dev, interface="torch")
        def vqc_circuit(angles, theta):
            # Step 1: Encode token via angle encoding (same as quantum embedding)
            for i in range(self.n_qubits):
                qml.RY(angles[i], wires=i)

            # Step 2: Apply VQC layers with learnable rotations
            # Each layer applies RZ·RY·RZ to each qubit
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    # Euler decomposition: any rotation = RZ·RY·RZ
                    qml.RZ(theta[l, i, 0], wires=i)  # first rotation
                    qml.RY(theta[l, i, 1], wires=i)  # second rotation
                    qml.RZ(theta[l, i, 2], wires=i)  # third rotation

            # Return quantum state vector
            return qml.state()

        self.vqc_circuit = vqc_circuit

    def forward(self, angles):
        # angles shape: (batch, seq_len, n_qubits)
        batch_size, seq_len, _ = angles.shape

        psi_Q_list = []
        psi_K_list = []
        psi_V_list = []

        for b in range(batch_size):
            batch_Q, batch_K, batch_V = [], [], []
            for t in range(seq_len):
                token_angles = angles[b, t]  # (4,)

                # Run same circuit structure with different learned parameters
                # This is where Q, K, V diverge!
                psi_Q = self.vqc_circuit(token_angles, self.theta_Q).real
                psi_K = self.vqc_circuit(token_angles, self.theta_K).real
                psi_V = self.vqc_circuit(token_angles, self.theta_V).real

                batch_Q.append(psi_Q)
                batch_K.append(psi_K)
                batch_V.append(psi_V)

            psi_Q_list.append(torch.stack(batch_Q))
            psi_K_list.append(torch.stack(batch_K))
            psi_V_list.append(torch.stack(batch_V))

        # Shape: (batch, seq_len, 2^n_qubits) = (batch, seq_len, 16)
        psi_Q = torch.stack(psi_Q_list)
        psi_K = torch.stack(psi_K_list)
        psi_V = torch.stack(psi_V_list)

        return psi_Q, psi_K, psi_V