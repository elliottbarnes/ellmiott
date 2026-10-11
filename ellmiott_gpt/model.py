"""Decoder-only transformer, written out so every operation can be inspected.

The browser implements the same operations independently. Exported weights and
parity fixtures connect the two implementations; no pretrained API is involved.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2

    def __post_init__(self):
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError("All model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("Embedding size must be divisible by head count")


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x, disabled_head=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (batch, position, channel) -> (batch, head, position, head channel)
        q, k, v = [t.view(batch, length, self.n_head, self.head_size).transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        attention = F.softmax(scores, dim=-1)
        heads = attention @ v
        if disabled_head is not None:
            # Zero the selected head's contribution, leaving attention itself
            # unchanged so the display can still show what that head attended to.
            keep = torch.ones(self.n_head, device=x.device, dtype=x.dtype)
            keep[disabled_head] = 0
            heads = heads * keep.view(1, -1, 1, 1)
        mixed = heads.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(mixed), attention


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x, disabled_head=None):
        update, attention = self.attn(self.ln1(x), disabled_head)
        x = x + update
        return x + self.mlp(self.ln2(x)), attention


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, *, disabled_head=None, return_attention=False):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.block_size:
            raise ValueError("Expected token IDs shaped (batch, 1..block_size)")
        if disabled_head is not None:
            layer, head = disabled_head
            if not (0 <= layer < self.config.n_layer and 0 <= head < self.config.n_head):
                raise ValueError("Disabled head is outside the model")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        attention = []
        for index, block in enumerate(self.blocks):
            disabled = disabled_head[1] if disabled_head is not None and disabled_head[0] == index else None
            x, weights = block(x, disabled)
            if return_attention:
                attention.append(weights)
        return self.lm_head(self.ln_f(x)), attention

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
