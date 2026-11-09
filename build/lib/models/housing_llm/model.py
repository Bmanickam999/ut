from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HousingTransformerConfig:
    vocab_size: int = 32_000
    max_seq_len: int = 256
    d_model: int = 1_024
    n_layers: int = 37
    n_heads: int = 16
    d_ff: int = 4_096
    dropout: float = 0.1
    target_mean: float = 300_000.0
    target_scale: float = 200_000.0

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.vocab_size, self.max_seq_len, self.n_layers, self.d_ff) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.target_scale <= 0:
            raise ValueError("target_scale must be positive")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: HousingTransformerConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: HousingTransformerConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class HousingTransformer(nn.Module):
    """Decoder transformer with language-model and home-value regression heads.

    The generic trainer only relies on the returned scalar ``loss``. This keeps
    domain-specific objectives here and infrastructure policy in the trainer.
    """

    def __init__(self, cfg: HousingTransformerConfig) -> None:
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.value_head = nn.Linear(cfg.d_model, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError(f"sequence length {length} exceeds {self.config.max_seq_len}")
        positions = torch.arange(length, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if attention_mask is None:
            last = x[:, -1]
        else:
            indices = attention_mask.sum(dim=1).clamp(min=1).long() - 1
            last = x[torch.arange(x.size(0), device=x.device), indices]
        normalized_value = self.value_head(last).squeeze(-1)
        predicted_value = normalized_value * self.config.target_scale + self.config.target_mean
        output = {"logits": logits, "predictions": predicted_value}
        losses: list[torch.Tensor] = []
        if labels is not None:
            losses.append(F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100))
        if targets is not None:
            normalized_target = (targets.float() - self.config.target_mean) / self.config.target_scale
            losses.append(F.smooth_l1_loss(normalized_value, normalized_target))
        if losses:
            output["loss"] = torch.stack(losses).sum()
        return output
