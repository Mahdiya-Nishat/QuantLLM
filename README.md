# QuantLLM — Hybrid Classical-Quantum Transformer

A PyTorch + PennyLane implementation of QuantLLM, a hybrid classical-quantum 
transformer with adaptive token routing for inference latency minimization.

## Installation
pip install -r requirements.txt

## Run Training
python train.py

## Architecture
- Adaptive routing module (classical vs quantum path)
- Classical self-attention path
- Quantum path: angle encoding → VQC → entanglement attention
- Joint loss: LLM loss + latency penalty

