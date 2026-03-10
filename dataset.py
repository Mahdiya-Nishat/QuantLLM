# dataset.py
# Downloads and prepares WikiText-2 dataset
# Creates input-output pairs for next token prediction

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

class WikiText2Dataset(Dataset):
    def __init__(self, config, split="train"):
        self.seq_len = config.seq_len  # 128

        print(f"Loading WikiText-2 ({split} split)...")

        # Download dataset from HuggingFace
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Load GPT-2 tokenizer
        print("Loading GPT-2 tokenizer...")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Concatenate all text into one long string
        full_text = " ".join(dataset["text"])

        # Tokenize everything at once
        print("Tokenizing...")
        tokens = tokenizer.encode(full_text)
        print(f"Total tokens: {len(tokens):,}")

        # Convert to tensor
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        # Calculate how many sequences we can make
        self.n_sequences = (len(self.tokens) - 1) // self.seq_len
        print(f"Total sequences of length {self.seq_len}: {self.n_sequences:,}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        # Get a chunk of seq_len tokens
        start = idx * self.seq_len
        end = start + self.seq_len

        # Input: tokens[start:end]
        # Target: tokens[start+1:end+1] (shifted by 1!)
        x = self.tokens[start:end]
        y = self.tokens[start+1:end+1]

        return x, y


def get_dataloaders(config):
    # Create train and validation datasets
    train_dataset = WikiText2Dataset(config, split="train")
    val_dataset = WikiText2Dataset(config, split="validation")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    print(f"\nTrain batches: {len(train_loader):,}")
    print(f"Val batches:   {len(val_loader):,}")

    return train_loader, val_loader

if __name__ == "__main__":
    from config import Config
    config = Config()
    train_loader, val_loader = get_dataloaders(config)

    # Look at one batch
    x, y = next(iter(train_loader))
    print("\nOne batch:")
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    print("First sequence input:", x[0, :10], "...")
    print("First sequence target:", y[0, :10], "...")