from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .graph import InstructionGraph
from .prompt import IGC_PROMPT_TEMPLATE

Detection = Dict[str, Any]


def extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in MLLM output: {text[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise ValueError(f"unbalanced JSON in MLLM output: {text[:200]!r}")


@dataclass
class IGCResult:
    detections: List[Detection]
    graph: InstructionGraph
    raw_output: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class InstructionGraphBuilder:
    def __init__(
        self,
        detector: Callable[[Any], Sequence[Detection]],
        vlm: Callable[[Any, str], str],
        prompt_template: str = IGC_PROMPT_TEMPLATE,
        max_instances: Optional[int] = None,
        strict: bool = False,
    ) -> None:
        self.detector = detector
        self.vlm = vlm
        self.prompt_template = prompt_template
        self.max_instances = max_instances
        self.strict = strict

    def build(self, image, instruction: str) -> IGCResult:
        detections = list(self.detector(image))
        detections = _normalise_detections(detections, self.max_instances)

        prompt = self.prompt_template.format(
            instruction=instruction.strip(),
            instances=json.dumps(
                [
                    {
                        "id": det["id"],
                        "category": det.get("category", ""),
                        "bbox": [round(float(v), 1) for v in det["bbox"]],
                    }
                    for det in detections
                ],
                ensure_ascii=False,
            ),
        )
        raw = self.vlm(image, prompt)

        try:
            graph = InstructionGraph.from_json(extract_json(raw))
        except (ValueError, json.JSONDecodeError):
            if self.strict:
                raise
            graph = InstructionGraph()

        graph.validate([det["id"] for det in detections])
        return IGCResult(detections=detections, graph=graph, raw_output=raw)

    __call__ = build


def _normalise_detections(
    detections: Sequence[Detection], max_instances: Optional[int]
) -> List[Detection]:
    ordered = sorted(
        detections, key=lambda det: -float(det.get("score", 0.0))
    )
    if max_instances is not None:
        ordered = ordered[:max_instances]
    normalised: List[Detection] = []
    for position, detection in enumerate(ordered):
        normalised.append(
            {
                "id": position + 1,
                "category": detection.get("category", ""),
                "bbox": [float(v) for v in detection["bbox"]],
                "score": float(detection.get("score", 0.0)),
            }
        )
    return normalised


class AnnotationDetector:
    def __init__(self, instances: Dict[Any, Sequence[Detection]], key: Optional[Callable] = None):
        self.instances = {str(k): list(v) for k, v in instances.items()}
        self.key = key

    def __call__(self, image) -> Sequence[Detection]:
        filename = getattr(image, "filename", None)
        candidates = []
        if self.key is not None:
            candidates.append(self.key(image))
        candidates.extend(
            [
                getattr(image, "igc_key", None),
                filename,
                os.path.basename(filename) if filename else None,
            ]
        )
        for candidate in candidates:
            if candidate is not None and str(candidate) in self.instances:
                return self.instances[str(candidate)]
        raise KeyError(
            f"no pre-computed detections for {candidates!r}; pass a key function "
            "or use the record key"
        )


class QwenVLClient:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        torch_dtype = getattr(torch, dtype)
        kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "device_map": device}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens

    def __call__(self, image, prompt: str) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
