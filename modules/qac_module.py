# modules/qac_module.py
# Quantum Attention Circuit
# Computes attention between Q and K via entanglement
# Then weights V by attention score

import torch
import torch.nn as nn
import pennylane as qml

class QACModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_qubits = config.n_qubits  # 4

        # We need 2x qubits for joint Q+K system
        # 4 qubits for Q + 4 qubits for K = 8 qubits total
        self.dev_qk = qml.device("default.qubit", wires=self.n_qubits * 2)

        # Separate device for V measurement
        self.dev_v = qml.device("default.qubit", wires=self.n_qubits)

        # FFN to project quantum output to d_model
        # ξt is a scalar but we need d_model (128) dimensional output
        self.ffn = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )

        # QNode for attention computation between Q and K
        @qml.qnode(self.dev_qk, interface="torch")
        def attention_circuit(psi_q, psi_k):
            # Step 1: Prepare joint state
            # Encode Q into first 4 qubits
            qml.StatePrep(psi_q, wires=range(self.n_qubits))
            # Encode K into last 4 qubits
            qml.StatePrep(psi_k, wires=range(self.n_qubits, self.n_qubits * 2))

            # Step 2: Entangle Q and K qubits
            # Hadamard on Q qubits → superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # CNOT between each Q qubit and corresponding K qubit
            # This creates entanglement between Q and K!
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, i + self.n_qubits])

            # Step 3: Measure fidelity with Bell state
            # We measure probability of |00...0> state
            # High probability = high similarity = high attention
            return qml.probs(wires=range(self.n_qubits * 2))

        # QNode for value measurement
        @qml.qnode(self.dev_v, interface="torch")
        def value_circuit(psi_v):
            # Prepare V state
            qml.StatePrep(psi_v, wires=range(self.n_qubits))

            # Measure expectation value of PauliZ on each qubit
            # This extracts classical information from quantum state
            return qml.expval(qml.PauliZ(0))

        self.attention_circuit = attention_circuit
        self.value_circuit = value_circuit

    def forward(self, psi_Q, psi_K, psi_V):
        # psi_Q, psi_K, psi_V shape: (batch, seq_len, 16)
        batch_size, seq_len, state_dim = psi_Q.shape

        # Normalize states - quantum states must have norm 1!
        psi_Q = psi_Q / (psi_Q.norm(dim=-1, keepdim=True) + 1e-8)
        psi_K = psi_K / (psi_K.norm(dim=-1, keepdim=True) + 1e-8)
        psi_V = psi_V / (psi_V.norm(dim=-1, keepdim=True) + 1e-8)

        xi_list = []

        for b in range(batch_size):
            batch_xi = []
            for t in range(seq_len):
                # Get Q, K, V states for this token
                q = psi_Q[b, t]  # (16,)
                k = psi_K[b, t]  # (16,)
                v = psi_V[b, t]  # (16,)

                # Step 1: Compute attention score via entanglement
                probs = self.attention_circuit(q, k)
                # Fidelity = probability of measuring |00...0> state
                # This is the squared inner product with Bell state
                att_ij = probs[0]  # scalar attention score

                # Step 2: Measure value state
                v_exp = self.value_circuit(v)  # scalar

                # Step 3: Weight value by attention
                xi_t = (att_ij * v_exp).float()  # scalar, convert to float32  # scalar

                batch_xi.append(xi_t.unsqueeze(0))  # (1,)

            xi_list.append(torch.stack(batch_xi))

        # Shape: (batch, seq_len, 1)
        xi = torch.stack(xi_list)

        # Project scalar to d_model dimension via FFN
        # (batch, seq_len, 1) -> (batch, seq_len, 128)
        yQ = self.ffn(xi)

        return yQ, xi