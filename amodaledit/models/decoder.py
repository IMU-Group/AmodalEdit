from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AmodalObjectDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_queries: int = 32,
        num_layers: int = 4,
        num_heads: int = 12,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        self.object_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        semantic_tokens: torch.Tensor,
        occlusion_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        occluder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = semantic_tokens.shape[0]
        if n == 0:
            return semantic_tokens.new_zeros(
                (0, self.num_queries, self.embed_dim)
            )

        memory = torch.cat(
            [semantic_tokens.unsqueeze(1), occlusion_tokens, context_tokens], dim=1
        )

        valid = [
            torch.ones((n, 1), dtype=torch.bool, device=memory.device),
            (
                occluder_mask
                if occluder_mask is not None
                else torch.ones(
                    (n, occlusion_tokens.shape[1]), dtype=torch.bool, device=memory.device
                )
            ),
            torch.ones(
                (n, context_tokens.shape[1]), dtype=torch.bool, device=memory.device
            ),
        ]
        valid_mask = torch.cat(valid, dim=1)
        key_padding_mask = ~valid_mask

        queries = self.object_queries.expand(n, -1, -1)
        tokens = self.decoder(
            tgt=queries, memory=memory, memory_key_padding_mask=key_padding_mask
        )
        return self.norm(tokens)
