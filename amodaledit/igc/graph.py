from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Pair = Tuple[int, int]

_JSON_KEYS = {
    "target": ("target objects", "target_objects", "targets"),
    "non_target": ("non-target objects", "non_target_objects", "non_targets"),
    "pairs": ("occlusion pairs", "occlusion_pairs", "pairs"),
}


def parse_pair(value: Any) -> Optional[Pair]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if not isinstance(value, str):
        return None
    text = value.replace("→", "->").replace("=>", "->")
    if "->" not in text:
        return None
    left, _, right = text.partition("->")
    try:
        return int(left.strip()), int(right.strip())
    except ValueError:
        return None


def _pick(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


@dataclass
class InstructionGraph:
    target_objects: List[int] = field(default_factory=list)
    non_target_objects: List[int] = field(default_factory=list)
    occlusion_pairs: List[Pair] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InstructionGraph":
        targets = _pick(payload, _JSON_KEYS["target"]) or []
        non_targets = _pick(payload, _JSON_KEYS["non_target"]) or []
        raw_pairs = _pick(payload, _JSON_KEYS["pairs"]) or []
        pairs = [pair for pair in (parse_pair(item) for item in raw_pairs) if pair]
        return cls(
            target_objects=[int(item) for item in targets],
            non_target_objects=[int(item) for item in non_targets],
            occlusion_pairs=pairs,
        )

    @classmethod
    def from_json(cls, text: str) -> "InstructionGraph":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target objects": self.target_objects,
            "non-target objects": self.non_target_objects,
            "occlusion pairs": [f"{a}->{b}" for a, b in self.occlusion_pairs],
        }

    def validate(self, detected_ids: Optional[Sequence[int]] = None) -> "InstructionGraph":
        targets = list(dict.fromkeys(self.target_objects))
        non_targets = list(dict.fromkeys(self.non_target_objects))

        overlap = set(targets) & set(non_targets)
        if overlap:
            non_targets = [item for item in non_targets if item not in overlap]

        if detected_ids is not None:
            known = {int(item) for item in detected_ids}
            targets = [item for item in targets if item in known]
            non_targets = [item for item in non_targets if item in known]
            self.occlusion_pairs = [
                (a, b) for a, b in self.occlusion_pairs if a in known and b in known
            ]

        if detected_ids is not None:
            assigned = set(targets) | set(non_targets)
            non_targets += [item for item in sorted(known) if item not in assigned]

        self.target_objects = targets
        self.non_target_objects = non_targets
        self.occlusion_pairs = list(
            dict.fromkeys(pair for pair in self.occlusion_pairs if pair[0] != pair[1])
        )
        return self

    def target_mask(self, instance_ids: Sequence[int]):
        import torch

        targets = set(self.target_objects)
        return torch.tensor(
            [int(item) in targets for item in instance_ids], dtype=torch.bool
        )

    def pairs_as_indices(self, instance_ids: Sequence[int]) -> List[Pair]:
        lookup = {int(instance_id): pos for pos, instance_id in enumerate(instance_ids)}
        pairs: List[Pair] = []
        for occluder, target in self.occlusion_pairs:
            if occluder in lookup and target in lookup:
                pairs.append((lookup[occluder], lookup[target]))
        return pairs

    def empty(self) -> bool:
        return not (self.target_objects or self.non_target_objects or self.occlusion_pairs)
