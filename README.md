# QuantLLM — Hybrid Classical-Quantum Transformer

> A PyTorch + PennyLane implementation of **QuantLLM**, a hybrid classical-quantum transformer with adaptive token-wise routing for inference latency minimization.

---

## 📌 Overview

Large Language Models (LLMs) suffer from quadratic attention complexity — `O(T²d)` — which causes high inference latency. **QuantLLM** addresses this by selectively routing semantically complex tokens through a quantum attention path while simpler tokens (like punctuation) take a cheaper classical path.

The key insight: **not all tokens need the same compute.** Complex nouns and abstract verbs benefit from quantum entanglement-based attention, while punctuation and connectives do not.

---

## 🏗️ Architecture

```
Input Text
    │
    ▼
┌─────────────────┐
│   Tokenizer     │  GPT-2 Byte-Level BPE
│  (BBPE, GPT-2)  │
└────────┬────────┘
         │  token IDs (batch, seq_len)
         ▼
┌─────────────────┐
│ Embedding Layer │  nn.Embedding(50257, 128)
└────────┬────────┘
         │  vt (batch, seq_len, 128)
         ▼
┌─────────────────────────────────────────┐
│           Routing Module                │
│  st = (Wq·vt)ᵀ(Wk·vt) / sqrt(da)      │
│  rt = sigmoid(α * (st - θ_QC))         │
└────────┬──────────────────┬────────────┘
         │ rt ≈ 0           │ rt ≈ 1
         ▼                  ▼
┌────────────────┐  ┌──────────────────────────────┐
│ Classical Path │  │        Quantum Path           │
│                │  │                               │
│ Multi-Head     │  │  ┌──────────────────────┐    │
│ Self-Attention │  │  │  Quantum Embedding   │    │
│ + FFN          │  │  │  Wpr: 128 → 4        │    │
│                │  │  │  RY angle encoding   │    │
│ yC ∈ R^128     │  │  └──────────┬───────────┘    │
└────────┬───────┘  │             │ angles (4,)     │
         │          │             ▼                 │
         │          │  ┌──────────────────────┐    │
         │          │  │     QSA Module        │    │
         │          │  │  3 × VQC circuits     │    │
         │          │  │  RZ·RY·RZ per qubit   │    │
         │          │  │  → |ψQ⟩, |ψK⟩, |ψV⟩  │    │
         │          │  └──────────┬───────────┘    │
         │          │             │                 │
         │          │             ▼                 │
         │          │  ┌──────────────────────┐    │
         │          │  │     QAC Module        │    │
         │          │  │  H⊗n + CNOT           │    │
         │          │  │  entanglement         │    │
         │          │  │  fidelity scoring     │    │
         │          │  │  → yQ ∈ R^128         │    │
         │          │  └──────────┬───────────┘    │
         │          └─────────────┼────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────┐
│          Token Aggregation              │
│   yF = (1 - rt) * yC  +  rt * yQ       │
└────────────────────┬────────────────────┘
                     │  yF (batch, seq_len, 128)
                     ▼
┌─────────────────────────────────────────┐
│         FC Layer + Softmax              │
│   zt = Wo · yF + b  →  P(next token)   │
└─────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
QuantLLM/
│
├── modules/                    # Individual components
│   ├── __init__.py
│   ├── routing_module.py       # Token complexity scoring + routing decision
│   ├── classical_path.py       # Standard multi-head self-attention + FFN
│   ├── quantum_embedding.py    # Angle encoding: 128-dim → 4-qubit state
│   ├── qsa_module.py           # Quantum Self-Attention: 3 VQC registers (Q, K, V)
│   ├── qac_module.py           # Quantum Attention Circuit: entanglement + fidelity
│   └── token_aggregation.py    # Combines classical + quantum outputs
│
├── model.py                    # Assembles all modules into QuantLLM
├── train.py                    # Training loop with joint loss
├── dataset.py                  # WikiText-2 loader and tokenizer
├── config.py                   # All hyperparameters (from paper Table I)
├── test_routing.py             # Module-level tests
├── requirements.txt
└── README.md
```

---

## ⚙️ Hyperparameters

From **Table I** of the paper:

| Hyperparameter | Value |
|---|---|
| Transformer Layers | 2 |
| Attention Heads | 2 |
| Hidden Size (d) | 128 |
| FFN Intermediate Dim | 256 |
| Number of Qubits (n) | 4 |
| VQC Layers (L) | 2 |
| Routing Projection Size (da) | 32 |
| Token Projection (d → n) | 128 → 4 |
| Learning Rate | 3 × 10⁻⁴ |
| Adam β Values | (0.9, 0.98) |
| Batch Size | 16 |
| Training Epochs | 10 |
| Gradient Clipping Norm | 1.0 |

---

## 📊 Dataset

**WikiText-2** — a collection of verified Wikipedia articles.

- ~2 million tokens (train split)
- Loaded via HuggingFace `datasets` library
- Tokenized using **GPT-2 Byte-Level BPE tokenizer**
- Split into fixed-length sequences of 128 tokens
- Input/target pairs shifted by 1 (next token prediction)

```python
# Example
Input:  ["The", "cat", "sat", "on", "the"]
Target: ["cat", "sat", "on",  "the", "mat"]
```

---

## 🔬 Module Details

### 1. Routing Module (`modules/routing_module.py`)
Computes a complexity score for each token using a self-attentive projection:

```
st = (Wq · vt)ᵀ (Wk · vt) / sqrt(da)
rt = sigmoid(α * (st - θ_QC))
```

- `rt ≈ 1` → quantum path (complex tokens)
- `rt ≈ 0` → classical path (simple tokens)
- `θ_QC` is a **learnable** threshold parameter
- Fully differentiable — gradients flow through sigmoid

---

### 2. Classical Path (`modules/classical_path.py`)
Standard transformer block:
- Pre-LayerNorm
- `nn.MultiheadAttention` (2 heads, embed_dim=128)
- Residual connections
- FFN: `Linear(128→256) → ReLU → Linear(256→128)`

---

### 3. Quantum Embedding (`modules/quantum_embedding.py`)
Encodes classical token vectors into quantum states:

```
vt (128-dim)
    │
    ▼  Wpr (learned, 4×128)
v't (4-dim)
    │
    ▼  × delta (scaling)
angles (4 rotation angles)
    │
    ▼  RY(θi)|0⟩ for each qubit i
|t⟩ ∈ (C²)^⊗4   (16-dim quantum state)
```

---

### 4. QSA Module (`modules/qsa_module.py`)
Three separate VQC circuits with **different learned parameters**:

```python
# Same input angles, three different circuits
|ψQ⟩ = VQC_Q(angles, θ_Q)   # Query register
|ψK⟩ = VQC_K(angles, θ_K)   # Key register  
|ψV⟩ = VQC_V(angles, θ_V)   # Value register
```

Each VQC layer applies Euler decomposition per qubit:
```
RZ(θ1) · RY(θ2) · RZ(θ3)
```
Total learnable quantum parameters: `3 registers × 2 layers × 4 qubits × 3 angles = 72`

---

### 5. QAC Module (`modules/qac_module.py`)
Computes quantum attention via entanglement:

```
Step 1: Joint state |ψQ⟩ ⊗ |ψK⟩  (8-qubit system)
Step 2: Apply H⊗n + CNOT gates   (entanglement)
Step 3: att_ij = P(|00...0⟩)     (fidelity score)
Step 4: ξt = att_ij × ⟨ψV|O|ψV⟩ (weighted value)
Step 5: yQ = FFN(ξt)             (→ R^128)
```

---

### 6. Token Aggregation (`modules/token_aggregation.py`)
Blends classical and quantum outputs:

```
yF = (1 - rt) · yC  +  rt · yQ
zt = Wo · yF + b           (vocab projection)
x̂_t+1 = softmax(zt)        (next token distribution)
```

---

## 📉 Loss Function

Joint objective (Equation 2 of paper):

```
J(Θ, θ_QC) = L_LM(Θ) + λ · L_lat(Θ, θ_QC)
```

- `L_LM` — cross-entropy loss for next token prediction
- `L_lat` — latency penalty (mean of rt, penalizes quantum usage)
- `λ = 0.1` — balances accuracy vs speed

---

## 🚀 Installation & Usage

### Local (PyCharm / CPU)

```bash
# Clone repository
git clone https://github.com/Mahdiya-Nishat/QuantLLM.git
cd QuantLLM

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py
```

### Google Colab (Recommended — GPU)

```python
# Cell 1: Setup
!git clone https://github.com/Mahdiya-Nishat/QuantLLM.git
%cd QuantLLM
!pip install -r requirements.txt

# Cell 2: Train
!python train.py
```

> ⚠️ **Note:** Quantum simulation on CPU is slow. For full training, use Google Colab with T4 GPU (`Runtime → Change Runtime Type → T4 GPU`).

---

## 📈 Results

Training confirmed feasible with decreasing loss across epochs:

| Epoch | Train Loss | Val Loss | Quantum % |
|---|---|---|---|
| 1 | 10.97 | 10.91 | 37.5% |
| 2 | 10.86 | 10.77 | 37.5% |

- ✅ Loss decreasing in both train and validation
- ✅ Quantum routing working (37.5% tokens routed to quantum path)
- ✅ Gradients flowing through quantum circuits via PennyLane Parameter Shift Rule
- ✅ All 13,074,074 parameters trainable end-to-end

---

## 🧠 Key Design Decisions

| Decision | Reason |
|---|---|
| 4 qubits instead of 7 | Hardware practicality (NISQ devices) |
| Learned `Wpr` projection | Keeps most important info in 4 numbers |
| `interface="torch"` in QNode | Enables automatic Parameter Shift Rule for quantum gradients |
| Classical FFN after QAC | Avoids additional circuit complexity |
| Learnable `θ_QC` threshold | Adapts routing during training |

---

## 📚 References

Based on the paper:
> *QuantLLM: A Hybrid Classical-Quantum LLM Transformer with Adaptive Routing Framework for Inference Latency Minimization*
> Nishat Mahdiya Khan, Pronaya Bhattacharya, Sandip Roy, Sachin Shetty
> cite: [N. M. Khan, P. Bhattacharya, S. Roy and S. Shetty, "QuantLLM: A Hybrid Classical-Quantum LLM Transformer with Adaptive Routing Framework for Inference Latency Minimization," 2025 International Conference on Software, Telecommunications and Computer Networks (SoftCOM), Split, Croatia, 2025, pp. 01-07, doi: 10.23919/SoftCOM66362.2025.11197342.]

---

## 👩‍💻 Author

**Nishat Mahdiya Khan**
[GitHub](https://github.com/Mahdiya-Nishat)
