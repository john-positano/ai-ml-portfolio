"""
A minimal, readable GPT-style (decoder-only) transformer.

Data flow for one forward pass:

    token ids (B, T)
      -> token embedding + position embedding        (B, T, C)
      -> N x Block:
            x = x + Attention(LayerNorm(x))           # tokens look at earlier tokens
            x = x + MLP(LayerNorm(x))                 # each token "thinks" on its own
      -> final LayerNorm
      -> linear head to vocab                         (B, T, vocab_size)  = logits

B = batch size, T = sequence length (<= block_size), C = n_embd (model width).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 258      # 256 bytes + 2 special tokens (see tokenizer.py)
    block_size: int = 256      # max context length in tokens
    n_layer: int = 6           # number of transformer blocks
    n_head: int = 6            # attention heads per block
    n_embd: int = 384          # model width; must be divisible by n_head
    dropout: float = 0.1
    bias: bool = False         # bias in Linear/LayerNorm (GPT-2 used True)
    use_sdpa: bool = True      # fused PyTorch attention kernel; False = explicit math below


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_sdpa = cfg.use_sdpa
        self.dropout = cfg.dropout

        # One matmul produces queries, keys and values for all heads at once.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Lower-triangular mask: position t may attend to positions 0..t only.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): each head attends independently
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # The textbook version: softmax(Q K^T / sqrt(d)) V, with future positions masked out.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, nh, T, T)
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v                                                   # (B, nh, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)                  # re-join heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block (as in GPT-2 and most modern LLMs)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection around attention
        x = x + self.mlp(self.ln2(x))    # residual connection around MLP
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # learned position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the input embedding and output projection share one matrix.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the init of layers that write into the residual stream,
        # so the stream's variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, include_pos=False):
        n = sum(p.numel() for p in self.parameters())
        return n if include_pos else n - self.wpe.weight.numel()

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token ids
        targets: (B, T) next-token ids, or None. Positions set to -100 are ignored
                 by the loss (fine-tuning uses this to skip the prompt).
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        """Autoregressive sampling: predict one token, append it, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]          # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                            # only the last position matters
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)    # greedy
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and (next_id == stop_token).all():
                break
        return idx