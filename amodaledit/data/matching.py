from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torchvision.ops import box_iou

from ..igc.graph import InstructionGraph


def map_graph_to_boxes(
    graph: InstructionGraph,
    detections: Sequence[dict],
    boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int]], torch.Tensor, Dict[int, int]]:
    target_mask = torch.zeros((boxes.shape[0],), dtype=torch.bool)
    if boxes.numel() == 0 or not detections:
        return [], target_mask, {}

    det_boxes = torch.tensor(
        [det["bbox"] for det in detections], dtype=torch.float32
    ).reshape(-1, 4)
    iou = box_iou(det_boxes, boxes.float())
    best_iou, best_index = iou.max(dim=1)

    id_to_index: Dict[int, int] = {}
    for row, detection in enumerate(detections):
        if float(best_iou[row]) >= iou_threshold:
            id_to_index[int(detection["id"])] = int(best_index[row])

    for object_id in graph.target_objects:
        index = id_to_index.get(int(object_id))
        if index is not None:
            target_mask[index] = True

    pairs: List[Tuple[int, int]] = []
    for occluder, target in graph.occlusion_pairs:
        occluder_index = id_to_index.get(int(occluder))
        target_index = id_to_index.get(int(target))
        if occluder_index is not None and target_index is not None:
            if occluder_index != target_index:
                pairs.append((occluder_index, target_index))

    return pairs, target_mask, id_to_index
