from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from ..igc.graph import InstructionGraph


class InstructionEditDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        root: Optional[str] = None,
        image_size: int = 512,
        load_edited: bool = True,
    ) -> None:
        super().__init__()
        if not os.path.isfile(manifest):
            raise FileNotFoundError(manifest)
        self.root = root or os.path.dirname(os.path.abspath(manifest))
        self.image_size = image_size
        self.load_edited = load_edited
        self.records: List[dict] = []
        with open(manifest, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f"no records in {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, path: str) -> torch.Tensor:
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return TF.to_tensor(image)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        source = self._load(record["source"])
        edited = self._load(record["edited"]) if self.load_edited and record.get("edited") else source

        instances = record.get("instances", [])
        boxes = torch.tensor(
            [instance["bbox"] for instance in instances], dtype=torch.float32
        ).reshape(-1, 4)
        categories = [str(instance.get("category", "")) for instance in instances]
        instance_ids = [int(instance.get("id", position)) for position, instance in enumerate(instances)]

        if record.get("width") and record.get("height"):
            scale_x = self.image_size / float(record["width"])
            scale_y = self.image_size / float(record["height"])
            boxes[:, 0::2] *= scale_x
            boxes[:, 1::2] *= scale_y

        graph = InstructionGraph.from_dict(record.get("graph", {}))
        target_mask = graph.target_mask(instance_ids)
        pairs = graph.pairs_as_indices(instance_ids)

        return {
            "source": source,
            "edited": edited,
            "instruction": record["instruction"],
            "boxes": boxes,
            "categories": categories,
            "pairs": pairs,
            "target_mask": target_mask,
            "image_id": record.get("image_id", index),
            "file_name": record["source"],
        }


def edit_collate(batch: Sequence[Dict]) -> Dict:
    return {
        "source": torch.stack([sample["source"] for sample in batch]),
        "edited": torch.stack([sample["edited"] for sample in batch]),
        "instructions": [sample["instruction"] for sample in batch],
        "boxes": [sample["boxes"] for sample in batch],
        "categories": [sample["categories"] for sample in batch],
        "pairs": [sample["pairs"] for sample in batch],
        "target_mask": torch.cat([sample["target_mask"] for sample in batch], dim=0),
        "image_ids": [sample["image_id"] for sample in batch],
    }
