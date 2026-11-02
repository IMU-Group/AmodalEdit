from __future__ import annotations

from typing import Sequence

IGC_PROMPT_TEMPLATE = """You need to analyze the input image and identify the object-level editing targets and occlusion relations based on the editing instruction and detected object instances. You must output in the following format: {{"target objects": [...], "non-target objects": [...], "occlusion pairs": ["A->B", ...]}} and do not output anything else.

One image, one editing instruction, and a list of detected object instances will be provided. Each detected instance includes an instance id, a category label, and bounding box coordinates.

Identify three parts:

target objects: objects directly affected by the instruction. This includes any object whose geometry, position, scale, visibility, or semantic state is changed by the edit, as well as any object that should move or change together with a directly edited object.

non-target objects: objects not directly edited and whose own amodal geometry, position, material, and identity should remain unchanged. If an object is only indirectly affected because the occlusion relation changes and more or less of it becomes visible, it is still a non-target object.

occlusion pairs: directed object-object occlusion relations in the format A->B, where A occludes B.

Rules:
- Refer to every object by its instance id.
- Target objects and non-target objects must be mutually exclusive.
- The detector may produce duplicates, tiny boxes, or false positives; you may skip them.
- Only output clear occlusion relations that can be inferred from the image.
- Do not output explanations or any extra text.

Editing instruction: {instruction}

Detected instances:
{instances}
"""


def format_instances(instances: Sequence[dict]) -> str:
    import json

    payload = [
        {
            "id": instance["id"],
            "category": instance.get("category", ""),
            "bbox": [round(float(v), 1) for v in instance["bbox"]],
        }
        for instance in instances
    ]
    return json.dumps(payload, ensure_ascii=False)


def build_igc_prompt(instruction: str, instances: Sequence[dict]) -> str:
    return IGC_PROMPT_TEMPLATE.format(
        instruction=instruction.strip(), instances=format_instances(instances)
    )
