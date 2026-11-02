from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def alp_collate(batch: Sequence[dict]) -> Dict:
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        raise ValueError("empty batch")

    images = torch.stack([sample["image"] for sample in batch])
    boxes: List[torch.Tensor] = [sample["boxes"] for sample in batch]
    categories: List[List[str]] = [list(sample["categories"]) for sample in batch]

    counts = [int(box.shape[0]) for box in boxes]
    offsets = [0]
    for count in counts[:-1]:
        offsets.append(offsets[-1] + count)

    pairs: List[tuple] = []
    for image_index, sample in enumerate(batch):
        for occluder, target in sample.get("pairs", []):
            pairs.append((offsets[image_index] + int(occluder), offsets[image_index] + int(target)))

    def _cat(key: str, dtype: torch.dtype) -> torch.Tensor:
        tensors = [sample[key] for sample in batch if sample.get(key) is not None]
        if not tensors:
            raise KeyError(f"batch samples have no {key!r} entry")
        return torch.cat([tensor.to(dtype) for tensor in tensors], dim=0)

    return {
        "images": images,
        "boxes": boxes,
        "categories": categories,
        "pairs": pairs,
        "target_mask": _cat("target_mask", torch.bool),
        "amodal_mask": _cat("amodal_mask", torch.float32),
        "visible_mask": _cat("visible_mask", torch.float32),
        "image_ids": [sample.get("image_id") for sample in batch],
        "file_names": [sample.get("file_name") for sample in batch],
        "instance_ids": [sample.get("instance_ids", []) for sample in batch],
    }
