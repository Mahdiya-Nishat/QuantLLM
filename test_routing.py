import torch
from config import Config
from model import QuantLLM

config = Config()
model = QuantLLM(config)

# Fake token IDs - integers between 0 and vocab_size
# Shape: (batch=1, seq_len=2)
fake_token_ids = torch.randint(0, config.vocab_size, (1, 2))
print("Input token IDs:", fake_token_ids)

print("\nRunning full forward pass...")
zt, rt = model(fake_token_ids)

print("Output logits shape:", zt.shape)
print("Routing decisions:", rt)
print()
model.count_parameters()