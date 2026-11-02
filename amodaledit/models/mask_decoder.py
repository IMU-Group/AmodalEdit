from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.Module:
    groups = min(32, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(max(1, groups), channels)


class AmodalMaskDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        output_resolution: int = 256,
        init_resolution: int = 32,
        num_heads: int = 8,
        use_context_skip: bool = True,
    ) -> None:
        super().__init__()
        if output_resolution % init_resolution != 0:
            raise ValueError(
                f"output_resolution ({output_resolution}) must be a multiple of "
                f"init_resolution ({init_resolution})"
            )
        self.embed_dim = embed_dim
        self.output_resolution = output_resolution
        self.init_resolution = init_resolution
        self.use_context_skip = use_context_skip
        self.num_stages = int(math.log2(output_resolution // init_resolution))

        self.mask_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.spatial_query = nn.Parameter(
            torch.randn(1, embed_dim, init_resolution, init_resolution) * 0.02
        )
        self.token_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Sigmoid()
        )

        self.context_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            _group_norm(embed_dim),
            nn.GELU(),
        )

        channels = embed_dim
        self.up_blocks = nn.ModuleList()
        self.context_adapters = nn.ModuleList()
        for _ in range(self.num_stages):
            out_channels = max(channels // 2, 16)
            self.up_blocks.append(_make_up_block(channels, out_channels))
            self.context_adapters.append(nn.Conv2d(embed_dim, out_channels, 1))
            channels = out_channels

        self.amodal_head = nn.Conv2d(channels, 1, 1)
        self.visible_head = nn.Conv2d(channels, 1, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for block in self.up_blocks:
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        for head in (self.amodal_head, self.visible_head):
            nn.init.kaiming_normal_(head.weight, mode="fan_out")
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        context_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = tokens.shape[0]
        if n == 0:
            empty = tokens.new_zeros((0, 1, self.output_resolution, self.output_resolution))
            return empty, empty

        query = self.mask_query.expand(n, -1, -1)
        descriptor, _ = self.cross_attn(query, tokens, tokens)
        descriptor = self.attn_norm(descriptor + query).squeeze(1)

        gate = self.token_gate(descriptor).unsqueeze(-1).unsqueeze(-1)
        features = self.spatial_query.expand(n, -1, -1, -1) * gate

        context_map = None
        if self.use_context_skip and context_tokens is not None and context_tokens.shape[1] > 0:
            side = int(round(math.sqrt(context_tokens.shape[1])))
            if side * side == context_tokens.shape[1]:
                context_map = context_tokens.transpose(1, 2).reshape(
                    n, self.embed_dim, side, side
                )
                context_map = self.context_proj(context_map)

        for up_block, adapter in zip(self.up_blocks, self.context_adapters):
            features = up_block(features)
            if context_map is not None:
                skip = F.interpolate(
                    adapter(context_map),
                    size=features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                features = features + skip

        return self.amodal_head(features), self.visible_head(features)


def _make_up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        _group_norm(out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        _group_norm(out_channels),
        nn.GELU(),
    )
