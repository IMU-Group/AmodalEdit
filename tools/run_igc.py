from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TOOLS_DIR)

from PIL import Image

from amodaledit.igc import AnnotationDetector, InstructionGraphBuilder, QwenVLClient
from amodaledit.igc.batch import annotate
from maskdino_detector import DEFAULT_CONFIG, DEFAULT_WEIGHTS, MaskDinoDetector

DEFAULT_VLM_PATH = "/data/sunjy/Amodal/Qwen2.5-VL-7B-Instruct-AWQ"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instruction Graph Construction")
    parser.add_argument("--manifest", required=True, help="jsonl with key/image/instruction")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=None, help="root dir for relative image paths")
    parser.add_argument("--detections", default=None,
                        help='json: {"<key>": [{"id","category","bbox","score"}, ...]}')
    parser.add_argument("--detector", default="annotations", choices=["annotations", "maskdino"])
    parser.add_argument("--vlm", default="qwen", choices=["qwen", "none"])
    parser.add_argument("--vlm-path", default=os.environ.get("VLM_PATH", DEFAULT_VLM_PATH))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-instances", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--maskdino-config", default=DEFAULT_CONFIG)
    parser.add_argument("--maskdino-weights", default=DEFAULT_WEIGHTS)
    return parser


def read_manifest(path: str, limit: Optional[int]) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit:
        records = records[:limit]
    return records


def make_detector(args, root: Optional[str]):
    if args.detector == "maskdino":
        return MaskDinoDetector(config=args.maskdino_config, weights=args.maskdino_weights)
    if not args.detections:
        raise SystemExit("--detections is required for --detector annotations")
    with open(args.detections, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    key = (lambda image: os.path.relpath(image.filename, root)) if root else None
    return AnnotationDetector(payload, key=key)


def make_vlm(args):
    if args.vlm == "none":
        empty = json.dumps(
            {"target objects": [], "non-target objects": [], "occlusion pairs": []}
        )
        return lambda image, prompt: empty
    if not args.vlm_path:
        raise SystemExit("--vlm-path is required for --vlm qwen")
    return QwenVLClient(args.vlm_path, dtype=args.dtype)


def main() -> None:
    args = build_parser().parse_args()
    records = read_manifest(args.manifest, args.limit)
    root = args.root

    def image_loader(path: str):
        full = path if os.path.isabs(path) or root is None else os.path.join(root, path)
        image = Image.open(full).convert("RGB")
        image.filename = path
        return image

    detector = make_detector(args, root)
    builder = InstructionGraphBuilder(
        detector=detector, vlm=make_vlm(args), max_instances=args.max_instances
    )
    annotate(records, builder, image_loader, args.output)


if __name__ == "__main__":
    main()
