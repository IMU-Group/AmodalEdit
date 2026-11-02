from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

try:
    from transformers import (
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTokenizer,
        CLIPVisionConfig,
        CLIPVisionModel,
    )
except ImportError as exc:
    raise ImportError(
        "transformers is required for the CLIP encoders: pip install transformers"
    ) from exc


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: Optional[int] = None,
    num_layers: int = 2,
    activation: str = "gelu",
) -> nn.Sequential:
    out_dim = out_dim if out_dim is not None else in_dim
    act = {"gelu": nn.GELU, "relu": nn.ReLU}[activation]
    if num_layers <= 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim), act())
    layers: List[nn.Module] = []
    dim = in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(dim, hidden_dim), act()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def _group_norm(channels: int) -> nn.Module:
    groups = min(32, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(max(1, groups), channels)


def _reduction(feature_info, level: int, features: torch.Tensor, image_h: int) -> float:
    if feature_info is not None and level < len(feature_info):
        reduction = feature_info[level].get("reduction") if feature_info else None
        if reduction:
            return float(reduction)
    return float(image_h) / float(features.shape[-2])


def build_rois(
    boxes: Sequence[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> Tuple[Optional[torch.Tensor], List[int]]:
    rois: List[torch.Tensor] = []
    counts: List[int] = []
    for batch_idx, box in enumerate(boxes):
        if box is None or box.numel() == 0:
            counts.append(0)
            continue
        box = box.to(device=device, dtype=dtype)
        counts.append(int(box.shape[0]))
        index = torch.full((box.shape[0], 1), float(batch_idx), device=device, dtype=dtype)
        rois.append(torch.cat([index, box], dim=1))
    if not rois:
        return None, counts
    return torch.cat(rois, dim=0), counts


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        clip_name: str = "openai/clip-vit-large-patch14",
        pretrained: bool = True,
        use_cls: bool = True,
        freeze_clip: bool = True,
        crop_size: int = 224,
        text_max_length: int = 77,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.crop_size = crop_size
        self.use_cls = use_cls
        self.text_max_length = text_max_length

        self.vision_model = _build_vision_model(clip_name, pretrained)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.text_model = _build_text_model(clip_name, pretrained)

        if freeze_clip:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
        self.freeze_clip = freeze_clip

        vision_dim = self.vision_model.config.hidden_size
        text_dim = self.text_model.config.hidden_size
        self.proj_vision = nn.Linear(vision_dim, embed_dim)
        self.proj_text = nn.Linear(text_dim, embed_dim)

        self.gate = make_mlp(embed_dim * 2, embed_dim, embed_dim, num_layers=2)
        self.geo_encoder = make_mlp(4, embed_dim, embed_dim, num_layers=2)
        self.semantic_proj = make_mlp(embed_dim * 2, embed_dim, embed_dim, num_layers=2)
        self.object_role = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", 49407)

    def encode_text(self, categories: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            list(categories),
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.proj_text.weight.device)
        attention_mask = tokens["attention_mask"].to(input_ids.device)
        context = torch.no_grad() if self.freeze_clip else torch.enable_grad()
        with context:
            hidden = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = hidden.last_hidden_state[
            torch.arange(input_ids.shape[0], device=input_ids.device),
            input_ids.argmax(dim=-1),
        ]
        return self.proj_text(pooled)

    def forward(
        self,
        images: torch.Tensor,
        rois: torch.Tensor,
        categories: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rois.shape[0] == 0:
            empty = images.new_zeros((0, self.embed_dim))
            return empty, empty

        crops = roi_align(
            images.float(),
            rois,
            output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0,
            aligned=True,
        )
        crops = (crops - self.clip_mean) / self.clip_std

        context = torch.no_grad() if self.freeze_clip else torch.enable_grad()
        with context:
            vision_out = self.vision_model(pixel_values=crops)
        if self.use_cls:
            visual = vision_out.last_hidden_state[:, 0]
        else:
            visual = vision_out.pooler_output
        f_visual = self.proj_vision(visual)

        f_text = self.encode_text(categories)
        if f_text.shape[0] != f_visual.shape[0]:
            raise ValueError(
                f"got {len(categories)} category labels for {f_visual.shape[0]} objects"
            )

        gate = torch.sigmoid(self.gate(torch.cat([f_visual, f_text], dim=-1)))
        sd = gate * f_visual + (1.0 - gate) * f_text

        boxes = rois[:, 1:5]
        height, width = images.shape[-2], images.shape[-1]
        box_w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        box_h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5 / width
        center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5 / height
        geo = torch.stack(
            [center_x, center_y, torch.log(box_w / width), torch.log(box_h / height)],
            dim=-1,
        )
        geo_embed = self.geo_encoder(geo)

        token = self.semantic_proj(torch.cat([sd, geo_embed], dim=-1))
        token = token + self.object_role
        return token, sd


class OcclusionEncoder(nn.Module):
    def __init__(self, embed_dim: int = 384) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rel_geometry = make_mlp(4, embed_dim, embed_dim, num_layers=2)
        self.proj = make_mlp(embed_dim * 2, embed_dim, embed_dim, num_layers=2)
        self.occluder_role = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self,
        semantic_descriptors: torch.Tensor,
        boxes: torch.Tensor,
        occluder_index: torch.Tensor,
        occluder_mask: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_descriptors.shape[0] == 0:
            return semantic_descriptors.new_zeros(
                (0, occluder_index.shape[1], self.embed_dim)
            )

        safe_index = occluder_index.clamp(min=0)
        target_boxes = boxes
        occluder_boxes = boxes[safe_index]

        target_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1.0)
        target_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1.0)
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) * 0.5
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) * 0.5

        occ_w = (occluder_boxes[..., 2] - occluder_boxes[..., 0]).clamp(min=1.0)
        occ_h = (occluder_boxes[..., 3] - occluder_boxes[..., 1]).clamp(min=1.0)
        occ_cx = (occluder_boxes[..., 0] + occluder_boxes[..., 2]) * 0.5
        occ_cy = (occluder_boxes[..., 1] + occluder_boxes[..., 3]) * 0.5

        dx = (occ_cx - target_cx.unsqueeze(1)) / target_w.unsqueeze(1)
        dy = (occ_cy - target_cy.unsqueeze(1)) / target_h.unsqueeze(1)
        dw = torch.log(occ_w / target_w.unsqueeze(1))
        dh = torch.log(occ_h / target_h.unsqueeze(1))
        rel_geometry = torch.stack([dx, dy, dw, dh], dim=-1)

        encoded_geometry = self.rel_geometry(rel_geometry)
        occluder_sd = semantic_descriptors[safe_index]
        tokens = self.proj(torch.cat([occluder_sd, encoded_geometry], dim=-1))
        tokens = tokens + self.occluder_role
        return tokens * occluder_mask.unsqueeze(-1).to(tokens.dtype)


class ContextEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        backbone: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        img_size: int = 512,
        fpn_levels: Sequence[int] = (1, 2, 3),
        roi_resolution: int = 7,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        import timm

        self.embed_dim = embed_dim
        self.fpn_levels = tuple(fpn_levels)
        self.roi_resolution = roi_resolution
        kwargs = dict(pretrained=pretrained, features_only=True)
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if any(tag in backbone for tag in ("swin", "vit", "deit")):
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(backbone, **kwargs)
        self.feature_info = self.backbone.feature_info
        self.num_stages = len(self.feature_info)

        self.used_stages = tuple(sorted(set(self.fpn_levels)))
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.feature_info[stage]["num_chs"], embed_dim, 1, bias=False),
                    _group_norm(embed_dim),
                    nn.GELU(),
                )
                for stage in self.used_stages
            ]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                    _group_norm(embed_dim),
                    nn.GELU(),
                )
                for _ in self.used_stages
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(self.fpn_levels), embed_dim, 1, bias=False),
            _group_norm(embed_dim),
            nn.GELU(),
        )
        self.context_role = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        if rois.shape[0] == 0:
            return images.new_zeros(
                (0, self.roi_resolution ** 2, self.embed_dim)
            )
        height, width = images.shape[-2], images.shape[-1]
        stages = self.backbone((images - self.imagenet_mean) / self.imagenet_std)

        def _to_nchw(feature: torch.Tensor, channels: int) -> torch.Tensor:
            if feature.shape[1] == channels:
                return feature
            return feature.permute(0, 3, 1, 2).contiguous()

        pyramid = [
            self.proj[pos](
                _to_nchw(stages[stage], self.feature_info[stage]["num_chs"])
            )
            for pos, stage in enumerate(self.used_stages)
        ]
        for pos in reversed(range(len(pyramid) - 1)):
            upsampled = F.interpolate(
                pyramid[pos + 1], size=pyramid[pos].shape[-2:], mode="nearest"
            )
            pyramid[pos] = pyramid[pos] + upsampled
        pyramid = [self.smooth[pos](pyramid[pos]) for pos in range(len(pyramid))]

        features = []
        for level in self.fpn_levels:
            position = self.used_stages.index(level)
            feature = pyramid[position]
            spatial_scale = 1.0 / _reduction(self.feature_info, level, feature, height)
            features.append(
                roi_align(
                    feature,
                    rois,
                    output_size=(self.roi_resolution, self.roi_resolution),
                    spatial_scale=spatial_scale,
                    aligned=True,
                )
            )
        fused = self.fuse(torch.cat(features, dim=1))
        tokens = fused.flatten(2).transpose(1, 2)
        return tokens + self.context_role


def _build_vision_model(name: str, pretrained: bool) -> CLIPVisionModel:
    if pretrained:
        return CLIPVisionModel.from_pretrained(name)
    config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=224,
        patch_size=16,
        projection_dim=32,
    )
    return CLIPVisionModel(config)


def _build_text_model(name: str, pretrained: bool) -> CLIPTextModel:
    if pretrained:
        return CLIPTextModel.from_pretrained(name)
    config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=77,
    )
    return CLIPTextModel(config)
