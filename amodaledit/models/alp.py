from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import AmodalObjectDecoder
from .encoders import (
    ContextEncoder,
    OcclusionEncoder,
    SemanticEncoder,
    build_rois,
)
from .mask_decoder import AmodalMaskDecoder
from ..occluders import select_occluders


@dataclass
class ALPOutput:
    tokens: torch.Tensor
    image_index: torch.Tensor
    target_mask: torch.Tensor
    occluder_index: torch.Tensor
    occluder_mask: torch.Tensor
    amodal_logits: Optional[torch.Tensor] = None
    visible_logits: Optional[torch.Tensor] = None
    context_tokens: Optional[torch.Tensor] = None


class AmodalLayoutPerception(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        clip_name: str = "openai/clip-vit-large-patch14",
        clip_pretrained: bool = True,
        clip_use_cls: bool = True,
        freeze_clip: bool = True,
        backbone: str = "swin_base_patch4_window7_224",
        backbone_pretrained: bool = True,
        backbone_cache_dir: Optional[str] = None,
        image_size: int = 512,
        fpn_levels: Sequence[int] = (1, 2, 3),
        roi_resolution: int = 7,
        num_queries: int = 32,
        decoder_layers: int = 4,
        decoder_heads: int = 12,
        mask_decoder: bool = True,
        mask_resolution: int = 256,
        mask_init_resolution: int = 32,
        mask_use_context_skip: bool = True,
        max_occluders: int = 5,
        occluder_iou_thresh: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_occluders = max_occluders
        self.occluder_iou_thresh = occluder_iou_thresh

        self.semantic_encoder = SemanticEncoder(
            embed_dim=embed_dim,
            clip_name=clip_name,
            pretrained=clip_pretrained,
            use_cls=clip_use_cls,
            freeze_clip=freeze_clip,
        )
        self.occlusion_encoder = OcclusionEncoder(embed_dim=embed_dim)
        self.context_encoder = ContextEncoder(
            embed_dim=embed_dim,
            backbone=backbone,
            pretrained=backbone_pretrained,
            img_size=image_size,
            fpn_levels=fpn_levels,
            roi_resolution=roi_resolution,
            cache_dir=backbone_cache_dir,
        )
        self.object_decoder = AmodalObjectDecoder(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
        )

        self.mask_decoder_enabled = mask_decoder
        self.mask_decoder: Optional[AmodalMaskDecoder] = None
        if mask_decoder:
            self.mask_decoder = AmodalMaskDecoder(
                embed_dim=embed_dim,
                output_resolution=mask_resolution,
                init_resolution=mask_init_resolution,
                use_context_skip=mask_use_context_skip,
            )
        self.mask_resolution = mask_resolution

    @classmethod
    def from_config(cls, cfg) -> "AmodalLayoutPerception":
        model = cfg.get("model", {})
        return cls(
            embed_dim=model.get("embed_dim", 384),
            clip_name=model.get("semantic.clip_model", "openai/clip-vit-large-patch14"),
            clip_pretrained=model.get("semantic.pretrained", True),
            clip_use_cls=model.get("semantic.use_cls", True),
            freeze_clip=model.get("semantic.freeze", True),
            backbone=model.get("context.backbone", "swin_base_patch4_window7_224"),
            backbone_pretrained=model.get("context.pretrained", True),
            backbone_cache_dir=model.get("context.cache_dir", None),
            image_size=cfg.get("data.image_size", 512),
            fpn_levels=tuple(model.get("context.fpn_levels", [1, 2, 3])),
            roi_resolution=model.get("context.roi_resolution", 7),
            num_queries=model.get("num_queries", 32),
            decoder_layers=model.get("decoder_layers", 4),
            decoder_heads=model.get("decoder_heads", 12),
            mask_decoder=model.get("mask_decoder.enabled", True),
            mask_resolution=model.get("mask_decoder.resolution", 256),
            mask_init_resolution=model.get("mask_decoder.init_resolution", 32),
            mask_use_context_skip=model.get("mask_decoder.use_context_skip", True),
            max_occluders=model.get("occlusion.max_occluders", 5),
            occluder_iou_thresh=model.get("occlusion.iou_thresh", 0.05),
        )

    def forward(
        self,
        images: torch.Tensor,
        boxes: Sequence[torch.Tensor],
        categories: Sequence[Sequence[str]],
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        target_mask: Optional[torch.Tensor] = None,
        return_masks: Optional[bool] = None,
    ) -> ALPOutput:
        device = images.device
        rois, counts = build_rois(boxes, device=device, dtype=torch.float32)
        n_objects = int(sum(counts))

        if rois is None or n_objects == 0:
            empty_tokens = images.new_zeros((0, self.object_decoder.num_queries, self.embed_dim))
            empty_k = torch.zeros((0, self.max_occluders), dtype=torch.long, device=device)
            empty_r = torch.zeros((0, 1, self.mask_resolution, self.mask_resolution), device=device)
            return ALPOutput(
                tokens=empty_tokens,
                image_index=torch.zeros((0,), dtype=torch.long, device=device),
                target_mask=torch.zeros((0,), dtype=torch.bool, device=device),
                occluder_index=empty_k,
                occluder_mask=empty_k.bool(),
                amodal_logits=empty_r if return_masks is not False else None,
                visible_logits=empty_r if return_masks is not False else None,
            )

        flat_boxes = torch.cat([b.to(device=device, dtype=torch.float32) for b in boxes if b is not None and b.numel()], dim=0)
        image_index = torch.repeat_interleave(
            torch.arange(len(counts), device=device), torch.tensor(counts, device=device)
        )
        flat_categories: List[str] = [name for names in categories for name in names]
        if len(flat_categories) != n_objects:
            raise ValueError(
                f"expected {n_objects} category labels, got {len(flat_categories)}"
            )

        semantic_tokens, semantic_descriptors = self.semantic_encoder(
            images, rois, flat_categories
        )
        context_tokens = self.context_encoder(images, rois)
        occluder_index, occluder_mask = select_occluders(
            flat_boxes,
            image_index,
            pairs=pairs,
            max_occluders=self.max_occluders,
            iou_thresh=self.occluder_iou_thresh,
            device=device,
        )
        occlusion_tokens = self.occlusion_encoder(
            semantic_descriptors, flat_boxes, occluder_index, occluder_mask
        )

        tokens = self.object_decoder(
            semantic_tokens, occlusion_tokens, context_tokens, occluder_mask
        )

        if target_mask is None:
            target_mask = torch.ones((n_objects,), dtype=torch.bool, device=device)
        else:
            target_mask = target_mask.to(device=device, dtype=torch.bool)

        want_masks = self.mask_decoder_enabled if return_masks is None else return_masks
        amodal_logits = visible_logits = None
        if want_masks and self.mask_decoder is not None:
            amodal_logits, visible_logits = self.mask_decoder(tokens, context_tokens)

        return ALPOutput(
            tokens=tokens,
            image_index=image_index,
            target_mask=target_mask,
            occluder_index=occluder_index,
            occluder_mask=occluder_mask,
            amodal_logits=amodal_logits,
            visible_logits=visible_logits,
            context_tokens=context_tokens,
        )
