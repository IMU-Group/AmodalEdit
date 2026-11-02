from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from amodaledit.config import apply_dotted_override, load_config
from amodaledit.igc.graph import InstructionGraph
from amodaledit.utils import get_device, load_checkpoint, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AmodalEdit inference")
    parser.add_argument("--config", default="configs/edit_stage2.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--alp-checkpoint", default=None)
    parser.add_argument("--iaa-checkpoint", default=None)
    parser.add_argument("--igc-json", default=None, help="json produced by tools/run_igc.py")
    parser.add_argument("--key", default=None, help="record key inside --igc-json")
    parser.add_argument("--output", default="outputs/edited.png")
    parser.add_argument("--dump-layout", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--text-guidance", type=float, default=7.5)
    parser.add_argument("--image-guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_record(path: str, key: Optional[str]) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if key is not None:
        if key not in payload:
            raise KeyError(f"{key!r} not found in {path}")
        return payload[key]
    if len(payload) == 1:
        return next(iter(payload.values()))
    raise ValueError("multiple records in the json; pass --key")


def resolve_conditioning(
    record: Dict,
) -> Tuple[torch.Tensor, List[str], torch.Tensor, List[Tuple[int, int]]]:
    detections = record["detections"]
    boxes = torch.tensor(
        [detection["bbox"] for detection in detections], dtype=torch.float32
    ).reshape(-1, 4)
    categories = [str(detection.get("category", "")) for detection in detections]
    instance_ids = [int(detection["id"]) for detection in detections]

    graph = InstructionGraph.from_dict(record.get("graph", {}))
    if graph.target_objects:
        target_mask = graph.target_mask(instance_ids)
    else:
        target_mask = torch.ones((boxes.shape[0],), dtype=torch.bool)
    pairs = graph.pairs_as_indices(instance_ids)
    return boxes, categories, target_mask, pairs


def save_layout(image: Image.Image, logits: torch.Tensor, boxes: torch.Tensor, path: str) -> None:
    probability = logits.sigmoid().squeeze(1).cpu()
    canvas = image.convert("RGB").resize((logits.shape[-1], logits.shape[-1]), Image.BILINEAR)
    import numpy as np

    array = np.asarray(canvas).astype("float32")
    colors = np.random.RandomState(0).randint(64, 255, size=(probability.shape[0], 3))
    for index in range(probability.shape[0]):
        mask = probability[index].numpy() > 0.5
        if mask.any():
            array[mask] = 0.45 * array[mask] + 0.55 * colors[index]
    canvas = Image.fromarray(array.clip(0, 255).astype("uint8"))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    for override in args.overrides:
        apply_dotted_override(cfg, override)
    set_seed(args.seed)
    device = get_device(cfg.get("device", "cuda"))

    if args.igc_json:
        record = load_record(args.igc_json, args.key)
        boxes, categories, target_mask, pairs = resolve_conditioning(record)
        instruction = args.instruction or record.get("instruction", "")
    else:
        raise SystemExit(
            "no instruction graph: pass --igc-json (see tools/run_igc.py) so that "
            "IGC runs offline; the ALP/IAA still need the graph's target/occlusion sets"
        )
    if not instruction:
        raise SystemExit("missing --instruction and no instruction in the igc json")

    from amodaledit.diffusion import AmodalEditor

    editor = AmodalEditor.from_config(cfg)
    if args.alp_checkpoint:
        load_checkpoint(args.alp_checkpoint, editor.alp, map_location="cpu")
    if args.iaa_checkpoint:
        payload = load_checkpoint(args.iaa_checkpoint, map_location="cpu")
        editor.iaa.load_state_dict(payload["iaa"])
        if "injector" in payload:
            editor.injector.load_state_dict(payload["injector"])
    editor.alp.to(device)
    editor.iaa.to(device)

    image = Image.open(args.image).convert("RGB")

    if args.dump_layout:
        import torchvision.transforms.functional as TF

        size = editor.alp_image_size
        resized = image.resize((size, size), Image.BILINEAR)
        tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
        scaled = boxes.float().clone()
        scaled[:, 0::2] *= size / image.size[0]
        scaled[:, 1::2] *= size / image.size[1]
        with torch.no_grad():
            output = editor.alp(
                tensor, [scaled], [categories], pairs=pairs or None,
                target_mask=target_mask, return_masks=True,
            )
        layout_path = os.path.splitext(args.output)[0] + "_layout.png"
        save_layout(resized, output.amodal_logits, scaled, layout_path)
        print(f"saved amodal layout -> {layout_path}")

    result = editor.edit(
        image,
        instruction,
        boxes,
        categories,
        pairs=pairs,
        target_mask=target_mask,
        num_inference_steps=args.steps,
        text_guidance_scale=args.text_guidance,
        image_guidance_scale=args.image_guidance,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.save(args.output)
    print(f"saved edited image -> {args.output}")


if __name__ == "__main__":
    main()
