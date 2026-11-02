from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

Pair = Tuple[int, int]


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def _intersection(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box_a[:, :2], box_b[:, :2])
    bottom_right = torch.minimum(box_a[:, 2:], box_b[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def occlusion_size(
    target_boxes: torch.Tensor, occluder_boxes: torch.Tensor
) -> torch.Tensor:
    inter = _intersection(target_boxes, occluder_boxes)
    area = box_area(target_boxes).clamp(min=1.0)
    return (inter / area).clamp(0.0, 1.0)


def pad_occluder_table(
    per_object: Sequence[torch.Tensor], max_occluders: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(per_object)
    idx = torch.zeros((n, max_occluders), dtype=torch.long, device=device)
    mask = torch.zeros((n, max_occluders), dtype=torch.bool, device=device)
    for row, values in enumerate(per_object):
        values = values.to(device=device, dtype=torch.long)
        keep = values[:max_occluders]
        if keep.numel() > 0:
            idx[row, : keep.numel()] = keep
            mask[row, : keep.numel()] = True
    return idx, mask


def select_occluders(
    boxes: torch.Tensor,
    image_index: torch.Tensor,
    pairs: Optional[Sequence[Pair]] = None,
    max_occluders: int = 5,
    iou_thresh: float = 0.05,
    min_overlap: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = boxes.shape[0]
    device = device or boxes.device

    if n == 0:
        empty = torch.zeros((0, max_occluders), dtype=torch.long, device=device)
        return empty, empty.bool()

    candidates: List[List[float]] = [[] for _ in range(n)]

    if pairs:
        for occluder, target in pairs:
            if occluder == target:
                continue
            if not (0 <= occluder < n and 0 <= target < n):
                continue
            if image_index[occluder] != image_index[target]:
                continue
            score = occlusion_size(
                boxes[target].unsqueeze(0), boxes[occluder].unsqueeze(0)
            ).item()
            if score < min_overlap:
                continue
            candidates[target].append((score, occluder))
    else:
        order = torch.argsort(image_index)
        for target in range(n):
            same = order[image_index[order] == image_index[target]]
            same = same[same != target]
            if same.numel() == 0:
                continue
            targets = boxes[target].unsqueeze(0).expand(same.numel(), 4)
            iou = _intersection(targets, boxes[same]) / (
                box_area(targets) + box_area(boxes[same]) - _intersection(targets, boxes[same])
            ).clamp(min=1e-6)
            for score, occluder in zip(iou.tolist(), same.tolist()):
                if score >= iou_thresh:
                    candidates[target].append((score, occluder))

    per_object: List[torch.Tensor] = []
    for target in range(n):
        ranked = sorted(candidates[target], key=lambda item: item[0], reverse=True)
        per_object.append(
            torch.tensor([idx for _, idx in ranked[:max_occluders]], dtype=torch.long)
        )
    return pad_occluder_table(per_object, max_occluders, device)


def local_pairs_from_ids(
    pairs: Sequence[Tuple[object, object]],
    instance_ids: Sequence[object],
    image_offset: int = 0,
) -> List[Pair]:
    lookup = {str(instance_id): pos for pos, instance_id in enumerate(instance_ids)}
    resolved: List[Pair] = []
    for occluder_id, target_id in pairs:
        occluder = lookup.get(str(occluder_id))
        target = lookup.get(str(target_id))
        if occluder is None or target is None:
            continue
        resolved.append((image_offset + occluder, image_offset + target))
    return resolved
