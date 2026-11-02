from __future__ import annotations

import copy
import os
from typing import Any, Dict, Mapping, Optional

import yaml


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, Mapping) and not isinstance(value, Config):
                self[key] = Config(value)

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            value = Config(value)
        dict.__setitem__(self, key, value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, Mapping) else value

    def get(self, key: str, default: Any = None) -> Any:
        if "." not in key:
            value = dict.get(self, key, default)
            return value
        node: Any = self
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> Dict[str, Any]:
        def _unwrap(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {k: _unwrap(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_unwrap(v) for v in value]
            return value

        return _unwrap(self)


def merge(base: Mapping, override: Optional[Mapping]) -> Dict[str, Any]:
    result: Dict[str, Any] = copy.deepcopy(dict(base))
    if not override:
        return result
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str, overrides: Optional[Mapping] = None) -> Config:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if overrides:
        raw = merge(raw, overrides)
    cfg = Config(raw)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def apply_dotted_override(cfg: Config, assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"expected key=value, got {assignment!r}")
    key, _, raw_value = assignment.partition("=")
    value = yaml.safe_load(raw_value)
    node: Config = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], Mapping):
            node[part] = Config()
        node = node[part]
    node[parts[-1]] = value
