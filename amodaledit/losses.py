from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def boundary_map(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if logits.shape != target.shape:
        target = target.expand_as(logits) if target.numel() == logits.numel() else target
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is None:
        return loss.mean()
    weight = weight.expand_as(loss)
    return (loss * weight).sum() / (weight.sum() + eps)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probability = logits.sigmoid()
    if probability.shape != target.shape:
        target = target.expand_as(probability)
    dims = (1, 2, 3) if probability.dim() == 4 else (1, 2)
    intersection = (probability * target).sum(dim=dims)
    denom = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


class AmodalMaskLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 5.0,
        occluded_region_weight: float = 2.0,
        boundary_weight: float = 2.0,
        visible_weight: float = 0.5,
        boundary_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.occluded_region_weight = occluded_region_weight
        self.boundary_weight = boundary_weight
        self.visible_weight = visible_weight
        self.boundary_kernel = boundary_kernel

    def build_weight(
        self, target: torch.Tensor, visible: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        weight = torch.ones_like(target)
        if visible is not None:
            invisible = (1.0 - visible).clamp(0.0, 1.0)
            weight = weight + self.occluded_region_weight * invisible
        weight = weight + self.boundary_weight * boundary_map(target, self.boundary_kernel)
        return weight.detach()

    def forward(
        self,
        amodal_logits: torch.Tensor,
        gt_amodal: torch.Tensor,
        gt_visible: Optional[torch.Tensor] = None,
        visible_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        target = gt_amodal.float()
        if target.dim() == amodal_logits.dim() - 1:
            target = target.unsqueeze(1)
        visible = None
        if gt_visible is not None:
            visible = gt_visible.float()
            if visible.dim() == amodal_logits.dim() - 1:
                visible = visible.unsqueeze(1)

        weight = self.build_weight(target, visible)
        bce = weighted_bce_with_logits(amodal_logits, target, weight)
        dice = dice_loss(amodal_logits, target)

        total = self.bce_weight * bce + self.dice_weight * dice
        losses = {"loss_bce": bce.detach(), "loss_dice": dice.detach()}

        if visible_logits is not None and visible is not None and self.visible_weight > 0:
            visible_loss = weighted_bce_with_logits(visible_logits, visible, weight)
            total = total + self.visible_weight * visible_loss
            losses["loss_visible"] = visible_loss.detach()

        losses["loss_total"] = total
        return losses


class DiffusionLoss(nn.Module):
    def forward(
        self, noise_pred: torch.Tensor, noise: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        return {"loss_diffusion": loss, "loss_total": loss}
