# train.py
# Training loop for QuantLLM
# Implements joint loss: LLM loss + latency loss

import torch
import torch.nn as nn
from config import Config
from model import QuantLLM
from dataset import get_dataloaders

def train():
    config = Config()

    # Initialize model
    print("Initializing QuantLLM...")
    model = QuantLLM(config)
    model.count_parameters()

    # Optimizer - Adam with config learning rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98)  # from paper Table I
    )

    # Loss function - cross entropy for next token prediction
    criterion = nn.CrossEntropyLoss()

    # Load dataset
    print("\nLoading dataset...")
    train_loader, val_loader = get_dataloaders(config)

    print("\nStarting training...")
    print("="*50)

    for epoch in range(config.epochs):
        # ── TRAINING ──
        model.train()
        total_loss = 0
        total_llm_loss = 0
        total_lat_loss = 0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            # x shape: (batch, seq_len) - input token IDs
            # y shape: (batch, seq_len) - target token IDs

            optimizer.zero_grad()

            # Forward pass
            # zt: (batch, seq_len, vocab_size)
            # rt: (batch, seq_len)
            zt, rt = model(x)

            # Loss 1: LLM loss (cross entropy)
            # zt needs shape (batch*seq_len, vocab_size)
            # y needs shape (batch*seq_len)
            zt_flat = zt.view(-1, config.vocab_size)
            y_flat = y.view(-1)
            llm_loss = criterion(zt_flat, y_flat)

            # Loss 2: Latency loss
            # We want to MINIMIZE quantum path usage
            # So we penalize high rt values
            lat_loss = rt.mean()

            # Joint loss from equation 2 of paper
            loss = llm_loss + config.lambda_lat * lat_loss

            # Backward pass
            loss.backward()

            # Gradient clipping - prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip
            )

            # Update weights
            optimizer.step()

            total_loss += loss.item()
            total_llm_loss += llm_loss.item()
            total_lat_loss += lat_loss.item()
            n_batches += 1

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                avg_llm = total_llm_loss / n_batches
                avg_lat = total_lat_loss / n_batches
                print(f"Epoch {epoch+1} | Batch {batch_idx+1} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LLM: {avg_llm:.4f} | "
                      f"Lat: {avg_lat:.4f}")

            # IMPORTANT: limit batches per epoch for CPU!
            # Quantum simulation is slow - we test with 20 batches
            if batch_idx >= 19:
                print("(Stopping at 20 batches for CPU feasibility)")
                break

        # ── VALIDATION ──
        model.eval()
        val_loss = 0
        val_batches = 0

        print("\nRunning validation...")
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                zt, rt = model(x)
                zt_flat = zt.view(-1, config.vocab_size)
                y_flat = y.view(-1)
                loss = criterion(zt_flat, y_flat)
                val_loss += loss.item()
                val_batches += 1

                # Limit validation batches too
                if batch_idx >= 4:
                    break

        avg_val_loss = val_loss / val_batches
        avg_train_loss = total_loss / n_batches

        # Calculate quantum path percentage
        quantum_pct = (rt > 0.5).float().mean().item() * 100

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Quantum %:  {quantum_pct:.1f}%")
        print("="*50)

    print("\nTraining complete!")

if __name__ == "__main__":
    train()
