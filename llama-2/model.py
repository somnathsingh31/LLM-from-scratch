import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm replaces LayerNorm while using only the root mean square for efficiency:
    yᵢ = (xᵢ / RMS(x)) * γᵢ, 
    where RMS(x) = sqrt(ε + (1/n) * Σ xᵢ²).
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x / (torch.sqrt(means + self.eps))
        return (x_normed * self.weight).to(dtype=x.dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    """
    Llama uses a "Gates Linear Unit" (GLU) variant of SiLU called SwiGLU, defined as:
    SwiGLU(x) = SiLU(Linear₁(x)) * Linear₂(x)
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.fc2 = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.fc3 = nn.Linear(config["hidden_dim"], config["emb_dim"], dtype=config["dtype"], bias=False)
        self.silu = SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)

def precompute_rope_params(head_dim, theta_base=10000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    #compute inverse frequency (1 / base^(2*i/head_dim))
    inv_freq = 1.0 / (theta_base ** ((torch.arange(0, head_dim, 2)).float() / head_dim))

    #generate position indices
    positions = torch.arange(context_length)

    #Compute angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) #Elementwise mutiplication (context_size, 1) * (1, head_dim//2) -> (context_size, head_dim//2)

    #Expand angles to match head_dim
    angles = torch.cat([angles, angles], dim=1) #(context_size, head_dim)

    #Compute cosine and sine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x : (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head Dimension must be even"

    #Split x into first half and second half
    x1 = x[..., : head_dim//2]  # First half
    x2 = x[..., head_dim//2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    #Apply rotary transformation
    rotated = torch.cat([-x2, x1], dim=-1)  # (batch_size, num_heads, seq_len, head_dim)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)  # Linear layer to combine head outputs
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        cos, sin = precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #Apply RoPE
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        #Compute scaled-dot product
        attn_scores = queries @ keys.transpose(2, 3)  #Dot product for each head

        #Orginal mask truncated to number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        att_weights = torch.softmax(attn_scores / (keys.shape[-1]**0.5), dim=-1)

        context_vec = (att_weights @ values).transpose(1, 2)  #(b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dtype=cfg["dtype"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x):
        # Shortcut connection for attention block 
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut

        # Shortcut connection for feedforward block
        shortcut = x
        x = self.norm2(x)
        x =  self.ff(x)
        x = x + shortcut
        return x

class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        #Token Embedding
        x = self.tok_emb(in_idx) #(b, num_tokens, emb_size)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
