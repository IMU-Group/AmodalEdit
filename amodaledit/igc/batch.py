from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from ..utils import save_json
from .mllm import InstructionGraphBuilder, IGCResult


def annotate(
    records: Iterable[dict],
    builder: InstructionGraphBuilder,
    image_loader,
    output_path: str,
    flush_every: int = 20,
    verbose: bool = True,
) -> dict:
    results: dict = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as handle:
            results = json.load(handle)

    for index, record in enumerate(records, start=1):
        key = str(record["key"])
        if key in results:
            continue
        image = image_loader(record["image"])
        try:
            image.igc_key = key
        except AttributeError:
            pass
        result: IGCResult = builder.build(image, record["instruction"])
        results[key] = {
            "instruction": record["instruction"],
            "detections": result.detections,
            "graph": result.graph.to_dict(),
        }
        if verbose:
            print(f"[{index}] {key}: {result.graph.to_dict()}")
        if flush_every and index % flush_every == 0:
            save_json(output_path, results)

    save_json(output_path, results)
    return results


def load_graph(path: str, key: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get(str(key))
