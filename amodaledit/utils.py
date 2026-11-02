from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(preferred)


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )


def human_params(n: int) -> str:
    return f"{n / 1e6:.2f}M"


class AverageMeter:
    def __init__(self) -> None:
        self._sum: Dict[str, float] = {}
        self._count = 0

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            self._sum[key] = self._sum.get(key, 0.0) + float(value) * n
        self._count += n

    def averages(self) -> Dict[str, float]:
        if self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sum.items()}

    def reset(self) -> None:
        self._sum.clear()
        self._count = 0


class Logger:
    def __init__(self, log_file: Optional[str] = None) -> None:
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as handle:
                handle.write("")

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = False,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if model is not None:
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing:
            print(f"[ckpt] missing keys ({len(missing)}): {missing[:8]}", file=sys.stderr)
        if unexpected:
            print(
                f"[ckpt] unexpected keys ({len(unexpected)}): {unexpected[:8]}",
                file=sys.stderr,
            )
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload if isinstance(payload, dict) else {"model": payload}


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def chunked(iterable: Iterable, size: int) -> Iterable[list]:
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def freeze(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def param_groups(
    named_parameters: Iterable,
    lr: float,
    backbone_lr_scale: float = 0.1,
    backbone_keywords: tuple = ("backbone", "clip_model", "text_encoder"),
) -> list:
    main, backbone = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in backbone_keywords):
            backbone.append(param)
        else:
            main.append(param)
    groups = [{"params": main, "lr": lr}]
    if backbone:
        groups.append({"params": backbone, "lr": lr * backbone_lr_scale})
    return groups


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int = 0, min_lr: float = 0.0
):
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(cos, min_lr / max(base_lrs[0], 1e-12))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
