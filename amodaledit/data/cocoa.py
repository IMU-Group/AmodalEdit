from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from ..igc.graph import InstructionGraph
from .matching import map_graph_to_boxes

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


def _decode_rle(rle, height: int, width: int) -> np.ndarray:
    if mask_utils is None:
        raise ImportError("pycocotools is required to read COCOA masks")

    if isinstance(rle, dict):
        decoded = mask_utils.decode(rle)
    elif isinstance(rle, (list, tuple)):
        polygons = list(rle)
        if polygons and isinstance(polygons[0], (int, float)):
            polygons = [polygons]
        rles = mask_utils.frPyObjects(polygons, height, width)
        decoded = mask_utils.decode(rles)
    else:
        raise TypeError(f"unsupported mask type: {type(rle)!r}")

    decoded = np.asarray(decoded)
    if decoded.ndim == 3:
        decoded = decoded.max(axis=-1)
    return (decoded > 0).astype(np.float32)


def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    n = masks.shape[0]
    if n == 0:
        return torch.zeros((0, 4))
    flat = masks.reshape(n, -1)
    boxes = torch.zeros((n, 4))
    width = masks.shape[-1]
    for index in range(n):
        rows_any = torch.nonzero(flat[index].reshape(masks.shape[-2], width).any(dim=1))
        cols_any = torch.nonzero(flat[index].reshape(masks.shape[-2], width).any(dim=0))
        if rows_any.numel() == 0 or cols_any.numel() == 0:
            continue
        boxes[index] = torch.tensor(
            [
                float(cols_any.min()),
                float(rows_any.min()),
                float(cols_any.max()) + 1.0,
                float(rows_any.max()) + 1.0,
            ]
        )
    return boxes


class COCOADataset(Dataset):
    def __init__(
        self,
        root: str,
        annotation_file: str = "COCO_amodal_train2014.json",
        image_dir: Optional[str] = None,
        image_size: int = 512,
        mask_resolution: int = 256,
        max_objects: Optional[int] = None,
        min_visible_area: float = 0.0,
        augment: bool = False,
        graph_file: Optional[str] = None,
        graph_key: str = "id",
        load_images: bool = True,
        exclude_stuff: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = root
        self.annotation_path = os.path.join(root, annotation_file)
        if not os.path.isfile(self.annotation_path):
            raise FileNotFoundError(self.annotation_path)
        if image_dir is None:
            split = "train" if "train" in annotation_file else "val" if "val" in annotation_file else "test"
            image_dir = f"{split}2014"
        self.image_dir = os.path.join(root, image_dir)
        self.image_size = image_size
        self.mask_resolution = mask_resolution
        self.max_objects = max_objects
        self.min_visible_area = min_visible_area
        self.augment = augment
        self.graph_key = graph_key
        self.load_images = load_images
        self.exclude_stuff = exclude_stuff
        self.seed = seed

        with open(self.annotation_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.records: List[dict] = [
            record for record in payload.get("annotations", []) if record.get("regions")
        ]
        self.images: Dict[int, dict] = {
            image["id"]: image for image in payload.get("images", [])
        }

        self.graphs: Dict[str, dict] = {}
        if graph_file:
            with open(graph_file, "r", encoding="utf-8") as handle:
                self.graphs = json.load(handle)

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, file_name: str) -> str:
        return os.path.join(self.image_dir, file_name)

    def _decoded_instances(
        self, regions: Sequence[dict], height: int, width: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[int]]:
        visible_masks, amodal_masks, names, instance_ids = [], [], [], []
        for position, region in enumerate(regions):
            if self.exclude_stuff and int(region.get("isStuff", 0)) == 1:
                continue
            if "visible_mask" in region and "invisible_mask" in region:
                visible = _decode_rle(region["visible_mask"], height, width)
                invisible = _decode_rle(region["invisible_mask"], height, width)
                amodal = np.clip(visible + invisible, 0.0, 1.0)
            elif "segmentation" in region:
                visible = _decode_rle(region["segmentation"], height, width)
                amodal = visible
            else:
                continue

            if visible.sum() < self.min_visible_area:
                continue
            visible_masks.append(torch.from_numpy(visible))
            amodal_masks.append(torch.from_numpy(amodal))
            names.append(str(region.get("name", "")))
            instance_ids.append(int(region.get("id", position)))
        return visible_masks, amodal_masks, names, instance_ids

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_id = record["image_id"]
        info = self.images.get(image_id, {})
        file_name = info.get("file_name", record.get("file_name", ""))
        height = int(info.get("height", record.get("height", 0)))
        width = int(info.get("width", record.get("width", 0)))

        visible, amodal, names, instance_ids = self._decoded_instances(
            record["regions"], height, width
        )

        sample: dict = {
            "image_id": image_id,
            "file_name": file_name,
            "instance_ids": instance_ids,
            "categories": names,
        }

        if visible:
            vis_stack = torch.stack(visible)
            amo_stack = torch.stack(amodal)
            boxes = _masks_to_boxes(vis_stack)
        else:
            vis_stack = torch.zeros((0, height, width))
            amo_stack = torch.zeros((0, height, width))
            boxes = torch.zeros((0, 4))

        if boxes.numel():
            boxes[:, 0::2] *= self.image_size / max(width, 1)
            boxes[:, 1::2] *= self.image_size / max(height, 1)

        if self.max_objects is not None and boxes.shape[0] > self.max_objects:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = torch.argsort(areas, descending=True)[: self.max_objects]
            boxes = boxes[keep]
            vis_stack = vis_stack[keep]
            amo_stack = amo_stack[keep]
            names = [names[i] for i in keep.tolist()]
            instance_ids = [instance_ids[i] for i in keep.tolist()]

        if self.load_images:
            image = Image.open(self._image_path(file_name)).convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            image_tensor = TF.to_tensor(image)
        else:
            image_tensor = torch.zeros((3, self.image_size, self.image_size))

        if boxes.shape[0] > 0:
            vis_small = F.interpolate(
                vis_stack.unsqueeze(1),
                size=(self.mask_resolution, self.mask_resolution),
                mode="nearest",
            ).squeeze(1)
            amo_small = F.interpolate(
                amo_stack.unsqueeze(1),
                size=(self.mask_resolution, self.mask_resolution),
                mode="nearest",
            ).squeeze(1)
        else:
            vis_small = torch.zeros((0, self.mask_resolution, self.mask_resolution))
            amo_small = torch.zeros((0, self.mask_resolution, self.mask_resolution))

        if self.augment:
            generator = torch.Generator().manual_seed(self.seed + image_id)
            if torch.rand(1, generator=generator).item() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-1])
                if boxes.shape[0] > 0:
                    boxes = boxes.clone()
                    x1 = boxes[:, 0].clone()
                    x2 = boxes[:, 2].clone()
                    boxes[:, 0] = self.image_size - x2
                    boxes[:, 2] = self.image_size - x1
                vis_small = torch.flip(vis_small, dims=[-1])
                amo_small = torch.flip(amo_small, dims=[-1])

        sample.update(
            {
                "image": image_tensor,
                "boxes": boxes,
                "categories": names,
                "instance_ids": instance_ids,
                "amodal_mask": amo_small,
                "visible_mask": vis_small,
                "pairs": [],
                "target_mask": torch.ones((boxes.shape[0],), dtype=torch.bool),
            }
        )

        graph_record = self._lookup_graph(image_id, file_name)
        if graph_record:
            sample.update(self._apply_graph(graph_record, boxes, names, instance_ids))

        return sample

    def _lookup_graph(self, image_id: int, file_name: str) -> Optional[dict]:
        if not self.graphs:
            return None
        if self.graph_key == "file_name":
            return self.graphs.get(file_name) or self.graphs.get(os.path.basename(file_name))
        return self.graphs.get(str(image_id))

    @staticmethod
    def _apply_graph(
        graph_record: dict,
        boxes: torch.Tensor,
        names: Sequence[str],
        instance_ids: Sequence[int],
    ) -> dict:
        graph = InstructionGraph.from_dict(graph_record.get("graph", {}))
        detections = graph_record.get("detections") or [
            {"id": instance_ids[i], "bbox": boxes[i].tolist()} for i in range(len(instance_ids))
        ]
        pairs, target_mask, _ = map_graph_to_boxes(graph, detections, boxes)
        update = {"pairs": pairs}
        if graph.target_objects:
            update["target_mask"] = target_mask
        return update
