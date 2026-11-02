from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class CrossAttentionResidual(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = 8 if hidden_dim % 8 == 0 else 1
        self.hidden_dim = hidden_dim
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.norm_query(hidden_states)
        key_value = self.context_proj(self.norm_context(context))
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=context_mask,
            need_weights=False,
        )
        return self.out_proj(attended)


def _hidden_dim(module: nn.Module) -> Optional[int]:
    for path in ("to_q", "to_out.0", "to_k"):
        node: Optional[nn.Module] = module
        for part in path.split("."):
            node = getattr(node, part, None) if node is not None else None
        if isinstance(node, nn.Linear):
            return node.in_features
    return None


class AdapterInjector(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        context_dim: int,
        num_heads: int = 8,
        include: Optional[str] = None,
        max_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.adapters = nn.ModuleDict()
        self._module_names: Dict[str, str] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._context: Optional[torch.Tensor] = None
        self._context_mask: Optional[torch.Tensor] = None
        self.layer_names: List[str] = []

        pattern = re.compile(include) if include else None
        targets: List[Tuple[str, nn.Module]] = []
        for name, module in unet.named_modules():
            if not name.endswith("attn2"):
                continue
            if pattern is not None and not pattern.search(name):
                continue
            targets.append((name, module))
        if max_layers is not None and len(targets) > max_layers:
            targets = targets[-max_layers:]

        dtype = next(unet.parameters()).dtype
        device = next(unet.parameters()).device
        for name, module in targets:
            dim = _hidden_dim(module)
            if dim is None:
                continue
            key = name.replace(".", "__")
            adapter = CrossAttentionResidual(dim, context_dim, num_heads)
            self.adapters[key] = adapter.to(device=device, dtype=dtype)
            self._module_names[key] = name
            self._handles.append(module.register_forward_hook(self._make_hook(key)))
            self.layer_names.append(name)

        if not self.layer_names:
            raise RuntimeError(
                "no `attn2` module matched -- is this a diffusers UNet2DConditionModel?"
            )

    def _make_hook(self, key: str):
        def hook(module, inputs, output):
            if self._context is None:
                return output
            hidden_states = inputs[0]
            adapter = self.adapters[key]
            weight = adapter.out_proj.weight
            residual = adapter(
                hidden_states.to(weight.dtype), self._context, self._context_mask
            ).to(hidden_states.dtype)
            if isinstance(output, tuple):
                return (output[0] + residual,) + tuple(output[1:])
            return output + residual

        return hook

    def set_context(
        self, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None
    ) -> None:
        self._context = context
        self._context_mask = context_mask

    def clear_context(self) -> None:
        self._context = None
        self._context_mask = None

    def detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AdapterInjector is driven by forward hooks")
