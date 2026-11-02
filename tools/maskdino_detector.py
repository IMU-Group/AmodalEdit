from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASKDINO_DIR = os.path.join(PROJECT_ROOT, "modeling", "MaskDINO")
DETECTRON_DIR = os.path.join(PROJECT_ROOT, "modeling", "detectron")
DEFAULT_CONFIG = os.path.join(
    MASKDINO_DIR,
    "configs",
    "coco",
    "instance-segmentation",
    "swin",
    "maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
)
DEFAULT_WEIGHTS = os.path.join(
    MASKDINO_DIR,
    "checkpoint",
    "maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth",
)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class MaskDinoDetector:
    def __init__(
        self,
        config: Optional[str] = None,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        score_threshold: float = 0.3,
        top_k: int = 5,
    ) -> None:
        for path in (PROJECT_ROOT, MASKDINO_DIR, DETECTRON_DIR):
            if path not in sys.path:
                sys.path.insert(0, path)

        import detectron2.projects
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.engine.defaults import DefaultPredictor
        from maskdino import add_maskdino_config

        config = config or DEFAULT_CONFIG
        weights = weights or DEFAULT_WEIGHTS
        for path in (config, weights):
            if not os.path.isfile(path):
                raise SystemExit(f"MaskDINO file not found: {path}")

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(config)
        cfg.MODEL.WEIGHTS = weights
        if device:
            cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        self.predictor = DefaultPredictor(cfg)
        self.top_k = top_k

    def __call__(self, image) -> List[dict]:
        array = np.asarray(image.convert("RGB"))[:, :, ::-1]
        outputs = self.predictor(array)["instances"].to("cpu")
        if len(outputs) == 0:
            return []
        scores = outputs.scores
        order = scores.argsort(descending=True)[: self.top_k]
        detections: List[dict] = []
        for position, index in enumerate(order.tolist()):
            class_id = int(outputs.pred_classes[index])
            detections.append(
                {
                    "id": position + 1,
                    "category": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
                    "bbox": [round(float(v), 1) for v in outputs.pred_boxes.tensor[index].tolist()],
                    "score": round(float(scores[index]), 4),
                }
            )
        return detections
