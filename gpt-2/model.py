import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))    #Trainable, initializing with 1
        self.shift = nn.Parameter(torch.zeros(embed_dim))   #Trainable, initializing with 0

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embed_dim"], 4 * config["embed_dim"]),
            GELU(),
            nn.Linear(4 * config["embed_dim"], config["embed_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class MultiheadAttention(nn.Module):
    """Multihead attention: Simple version"""
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("Output dimension must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.full((context_length, context_length), float('-inf')), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, seq_len, d_in = x.shape

        k = self.w_k(x)     # [B, seq_len, d_out]
        q = self.w_q(x)     # [B, seq_len, d_out]
        v = self.w_v(x)     # [B, seq_len, d_out]

        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        k = k.view(b, seq_len, self.num_heads, self.head_dim)
        q = q.view(b, seq_len, self.num_heads, self.head_dim)
        v = v.view(b, seq_len, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        #Computed scaled dot product
        scores = q @ k.transpose(2,3) / (k.shape[-1]**0.5)  # Dot product for each head [b, num_heads, num_tokens, num_tokens]

        causal_mask = self.mask[:seq_len, :seq_len]    # [seq_len, seq_len]

        scores = scores + causal_mask   # [b, num_heads, num_tokens, num_tokens]

        atten_weights = torch.softmax(scores, dim=-1)
        atten_weights = self.dropout(atten_weights)

        # [b, num_heads, num_tokens, head_dim] --> # [b, num_tokens, num_heads, head_dim]
        attention = (atten_weights @ v).transpose(1, 2)     # makes tensor non-contiguous

        attention = attention.contiguous().view(b, seq_len, self.d_out) # contiguous() ensures the tensor is stored in a contiguous block of memory, view() can only work on contiguous tensors
        attention = self.out_proj(attention)

        return attention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiheadAttention(
            d_in=config["embed_dim"],
            d_out=config["embed_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"]
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["embed_dim"])
        self.norm2 = LayerNorm(config["embed_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embed = nn.Embedding(config["vocab_size"], config["embed_dim"])
        self.pos_embed = nn.Embedding(config["context_length"], config["embed_dim"])
        self.drop_embed = nn.Dropout(config["drop_rate"])

        #Transformer block
        self.trf_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        #LayerNorm block
        self. layer_norm = LayerNorm(config["embed_dim"])
        self.out_head = nn.Linear(
            config["embed_dim"], config["vocab_size"], bias = False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_embed(in_idx)
        pos_embeds = self.pos_embed(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_embed(x)
        x = self.trf_block(x)
        x = self.layer_norm(x)
        logits = self.out_head(x)
        return logits