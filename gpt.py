import tiktoken
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from multi_head_attention import MultiHeadAttention



""" We Implement GPT 2 """

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
        *[DummyTransformerBlock(cfg)
        for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print(batch)

# torch.manual_seed(123)
# model = DummyGPTModel(GPT_CONFIG_124M)
# logits = model(batch)
# print("Output shape:", logits.shape)
# print(logits)


# torch.manual_seed(123)
# batch_example = torch.randn(2, 5)
# layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# out = layer(batch_example)
# print(out)

# mean = out.mean(dim=-1, keepdim=True)
# var = out.var(dim=-1, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)


# out_norm = (out - mean) / torch.sqrt(var)
# mean = out_norm.mean(dim=-1, keepdim=True)
# var = out_norm.var(dim=-1, keepdim=True)
# print("Normalized layer outputs:\n", out_norm)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# print("***")
torch.set_printoptions(sci_mode=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)



class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


# print("---")
# ln = LayerNorm(emb_dim=5)
# out_ln = ln(batch_example)
# mean = out_ln.mean(dim=-1, keepdim=True)
# var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

# plt.tight_layout()
# plt.show()


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                                    GELU(),
                                    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)


# ffn = FeedForward(GPT_CONFIG_124M)
# x = torch.rand(2, 3, 768)
# out = ffn(x)
# print(out.shape)


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut=True):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
        nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
        GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x


# layer_sizes = [3, 3, 3, 3, 3, 1]
# sample_input = torch.tensor([[1., 0., -1.]])
# torch.manual_seed(123)
# model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


# print("gradient without shortcuts")
# print_gradients(model_without_shortcut, sample_input)
# print("gradient with shortcuts")
# torch.manual_seed(123)
# model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes)
# print_gradients(model_with_shortcut, sample_input)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


# torch.manual_seed(123)
# x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
# output = block(x)
# print("Input shape:", x.shape)
# print("Output shape:", output.shape)


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
        torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)


total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")
#
# print("Token embedding layer shape:", model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)

total_params_gpt2 = (
total_params - sum(p.numel()
for p in model.out_head.parameters())
)
# print(f"Number of trainable parameters "
# f"considering weight tying: {total_params_gpt2:,}")

""" Exercise 4.1 Number of parameters in feed forward and attention modules
-----------------------------------------------------------------------------
Calculate and compare the number of parameters that are contained in the feed forward
module and those that are contained in the multi-head attention module. 
"""

# print("block.ff parameters:", sum([p.numel() for p in block.ff.parameters()]))
# print("block.ff parameters:", sum([p.numel() for p in block.att.parameters()]))

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
# print(f"Total size of the model: {total_size_mb:.2f} MB")


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
# print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"])
# print("Output:", out)
# print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)

""" Exercise 4.3: Using separate dropout parameters 
-------------------------------------------------------------------------------------------
At the beginning of this chapter, we defined a global drop_rate setting in the GPT_
CONFIG_124M dictionary to set the dropout rate in various places throughout the
GPTModel architecture. Change the code to specify a separate dropout value for the
various dropout layers throughout the model architecture. (Hint: there are three distinct
places where we used dropout layers: the embedding layer, shortcut layer, and
multi-head attention module.)
"""


GPT_CONFIG_124M_Exe4_3 = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_emb": 0.1,        # NEW: dropout for embedding layers
    "drop_rate_attn": 0.1,       # NEW: dropout for multi-head attention
    "drop_rate_shortcut": 0.1,   # NEW: dropout for shortcut connections
    "qkv_bias": False
}


class TransformerBlockExc4_3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate_attn"], # NEW: dropout for multi-head attention
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModelExc4_3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"]) # NEW: dropout for embedding layers

        self.trf_blocks = nn.Sequential(
            *[TransformerBlockExc4_3(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
