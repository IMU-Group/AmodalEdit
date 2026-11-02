from __future__ import annotations

from typing import Dict

import torch


def _flatten(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask.reshape(mask.shape[0], -1)


def dice_score(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> torch.Tensor:
    pred_bin = _flatten((pred > threshold).float())
    target_bin = _flatten(target).float()

    intersection = (pred_bin * target_bin).sum(dim=1)
    denom = pred_bin.sum(dim=1) + target_bin.sum(dim=1)
    return (2.0 * intersection + eps) / (denom + eps)


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> torch.Tensor:
    pred_bin = (pred > threshold).float()
    pred_bin = _flatten(pred_bin)
    target_bin = _flatten(target).float()
    intersection = (pred_bin * target_bin).sum(dim=1)
    union = pred_bin.sum(dim=1) + target_bin.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


def boundary_iou(
    pred: torch.Tensor, target: torch.Tensor, kernel_size: int = 5, threshold: float = 0.5
) -> torch.Tensor:
    from .losses import boundary_map

    pred_bin = (pred > threshold).float()
    edge = boundary_map(target.float(), kernel_size)
    pred_bin = _flatten(pred_bin)
    target_bin = _flatten(target.float())
    edge = _flatten(edge)

    intersection = (pred_bin * target_bin * edge).sum(dim=1)
    union = ((pred_bin + target_bin).clamp(0, 1) * edge).sum(dim=1)
    return (intersection + 1e-6) / (union + 1e-6)


@torch.no_grad()
def mask_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    prefix: str = "",
    threshold: float = 0.5,
) -> Dict[str, float]:
    if pred.numel() == 0 or target.numel() == 0:
        return {}
    dice = dice_score(pred, target, threshold)
    iou = iou_score(pred, target, threshold)
    b_iou = boundary_iou(pred, target)
    return {
        f"{prefix}dice": dice.mean().item(),
        f"{prefix}iou": iou.mean().item(),
        f"{prefix}boundary_iou": b_iou.mean().item(),
    }
