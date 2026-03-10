# config.py
# All hyperparameters from Table I of the QuantLLM paper

class Config:
    # Transformer settings
    vocab_size     = 50257   # GPT-2 BPE vocabulary size
    d_model        = 128     # hidden size (d)
    n_heads        = 2       # attention heads
    n_layers       = 2       # transformer layers
    ffn_dim        = 256     # feedforward intermediate size

    # Quantum settings
    n_qubits       = 4       # number of qubits
    vqc_layers     = 2       # VQC depth (L)

    # Routing settings
    da             = 32      # routing projection size
    theta_qc       = 0.5     # routing threshold (tunable!)
    alpha          = 10.0    # sharpness of sigmoid decision

    # Training settings
    learning_rate = 3e-4
    batch_size = 2  # changed from 16
    epochs = 2  # changed from 10
    grad_clip = 1.0
    seq_len = 8  # changed from 128

    # Loss weighting
    lambda_lat     = 0.1     # balances latency vs accuracy loss