import torch


from gpt import GPTModel
from from_OpenAI import download_and_load_gpt2
""" Code copied from instruction_fine_tuning.py - It wasn't imported to avoid the long training process over there"""

BASE_CONFIG = {"vocab_size": 50257, # Vocabulary size
               "context_length": 1024, # Context length
               "drop_rate": 0.0, # Dropout rate
               "qkv_bias": True # Query-key-value bias
               }

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)

model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

print("Finished!")
