from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class IAAOutput:
    editing_feature: torch.Tensor
    preserving_feature: torch.Tensor
    fused_feature: torch.Tensor
    editing_mask: torch.Tensor
    preserving_mask: torch.Tensor


def _pad_token_sets(
    tokens: torch.Tensor,
    image_index: torch.Tensor,
    select_mask: torch.Tensor,
    num_images: int,
    null_token: torch.Tensor,
) -> tuple:
    device = tokens.device
    embed_dim = tokens.shape[-1]
    groups = []
    for image in range(num_images):
        keep = (image_index == image) & select_mask
        groups.append(tokens[keep])
    max_len = max(1, max(group.shape[0] for group in groups))

    padded = tokens.new_zeros((num_images, max_len + 1, embed_dim))
    mask = torch.zeros((num_images, max_len + 1), dtype=torch.bool, device=device)
    padded[:, 0] = null_token.to(dtype=tokens.dtype)
    mask[:, 0] = True
    for row, group in enumerate(groups):
        count = group.shape[0]
        if count > 0:
            padded[row, 1 : count + 1] = group
            mask[row, 1 : count + 1] = True
    return padded, mask


class InstructionAmodalAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_queries: int = 8,
        num_heads: int = 8,
        lambda_p: float = 1.0,
        text_dim: int = 768,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.lambda_p = lambda_p

        self.editing_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.preserving_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.null_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

        self.editing_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.preserving_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.editing_norm = nn.LayerNorm(embed_dim)
        self.preserving_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        image_index: torch.Tensor,
        target_mask: torch.Tensor,
        text_embedding: torch.Tensor,
        num_images: Optional[int] = None,
    ) -> IAAOutput:
        batch = int(num_images or text_embedding.shape[0])
        device = tokens.device

        num_tokens = tokens.shape[1]
        flat_tokens = tokens.reshape(-1, tokens.shape[-1])
        flat_image_index = image_index.repeat_interleave(num_tokens)
        flat_target_mask = target_mask.repeat_interleave(num_tokens)

        editing_set, editing_mask = _pad_token_sets(
            flat_tokens, flat_image_index, flat_target_mask, batch, self.null_token
        )
        preserving_set, preserving_mask = _pad_token_sets(
            flat_tokens, flat_image_index, ~flat_target_mask, batch, self.null_token
        )

        text_delta = self.text_proj(text_embedding.to(device)).unsqueeze(1)
        editing_queries = self.editing_queries.expand(batch, -1, -1) + text_delta
        preserving_queries = self.preserving_queries.expand(batch, -1, -1)

        editing_feature, _ = self.editing_attn(
            editing_queries,
            editing_set,
            editing_set,
            key_padding_mask=~editing_mask,
            need_weights=False,
        )
        preserving_feature, _ = self.preserving_attn(
            preserving_queries,
            preserving_set,
            preserving_set,
            key_padding_mask=~preserving_mask,
            need_weights=False,
        )
        editing_feature = self.editing_norm(editing_feature + editing_queries)
        preserving_feature = self.preserving_norm(preserving_feature + preserving_queries)

        fused = editing_feature + self.lambda_p * preserving_feature
        return IAAOutput(
            editing_feature=editing_feature,
            preserving_feature=preserving_feature,
            fused_feature=fused,
            editing_mask=editing_mask,
            preserving_mask=preserving_mask,
        )

    @classmethod
    def from_config(cls, cfg, text_dim: int = None) -> "InstructionAmodalAdapter":
        resolved = text_dim if text_dim is not None else cfg.get("iaa.text_dim", 768)
        return cls(
            embed_dim=cfg.get("model.embed_dim", 384),
            num_queries=cfg.get("iaa.num_queries", 8),
            num_heads=cfg.get("iaa.num_heads", 8),
            lambda_p=cfg.get("iaa.lambda_p", 1.0),
            text_dim=resolved,
        )
