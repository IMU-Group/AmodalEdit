from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from amodaledit.config import apply_dotted_override, load_config
from amodaledit.data import InstructionEditDataset, edit_collate
from amodaledit.diffusion import AmodalEditor, load_pipeline
from amodaledit.diffusion.io import text_dim_for_pipe
from amodaledit.diffusion.step import diffusion_step
from amodaledit.losses import DiffusionLoss
from amodaledit.models import AmodalLayoutPerception, InstructionAmodalAdapter
from amodaledit.utils import (
    AverageMeter,
    Logger,
    count_parameters,
    get_device,
    human_params,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-2 training (diffusion + IAA)")
    parser.add_argument("--config", default="configs/edit_stage2.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--alp-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    for override in args.overrides:
        apply_dotted_override(cfg, override)

    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    output_dir = args.output_dir or cfg.get("output_dir", "runs/edit_stage2")
    os.makedirs(output_dir, exist_ok=True)
    logger = Logger(os.path.join(output_dir, "train.log"))

    train_dataset = InstructionEditDataset(
        manifest=cfg.get("data.manifest"),
        root=cfg.get("data.root", None),
        image_size=cfg.get("data.image_size", 512),
    )
    if args.smoke:
        train_dataset = torch.utils.data.Subset(
            train_dataset, list(range(min(2, len(train_dataset))))
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("data.batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("data.num_workers", 2)),
        collate_fn=edit_collate,
        pin_memory=device.type == "cuda",
    )

    pipe = load_pipeline(cfg)
    alp = AmodalLayoutPerception.from_config(cfg)
    if args.alp_checkpoint:
        load_checkpoint(args.alp_checkpoint, alp, map_location="cpu")
        logger(f"loaded ALP checkpoint: {args.alp_checkpoint}")
    else:
        logger("[warn] no --alp-checkpoint given: ALP is randomly initialised")

    text_dim = text_dim_for_pipe(pipe)
    logger(f"diffusion backbone text dim: {text_dim} (pooled)")
    iaa = InstructionAmodalAdapter.from_config(cfg, text_dim=text_dim)
    editor = AmodalEditor(
        pipe=pipe,
        alp=alp.to(device),
        iaa=iaa.to(device),
        lambda_p=float(cfg.get("iaa.lambda_p", 1.0)),
        adapter_layers=cfg.get("iaa.adapter_layers", None),
        adapter_include=cfg.get("iaa.adapter_include", None),
        image_size=int(cfg.get("diffusion.image_size", 1024)),
        alp_image_size=int(cfg.get("data.image_size", 512)),
    )
    logger(f"IAA adapters injected into {len(editor.injector.layer_names)} attn2 blocks")
    logger(f"IAA parameters: {human_params(count_parameters(iaa))}")

    train_alp = bool(cfg.get("train.train_alp", False))
    train_unet = bool(cfg.get("train.train_unet", False))
    editor.alp.requires_grad_(train_alp)
    pipe.unet.requires_grad_(train_unet)
    if train_unet:
        pipe.unet.train()
    if train_alp:
        editor.alp.train()

    parameters: List[Dict] = [{"params": [p for p in iaa.parameters() if p.requires_grad]}]
    parameters.append(
        {"params": [p for p in editor.injector.parameters() if p.requires_grad]}
    )
    if train_alp:
        parameters.append({"params": [p for p in editor.alp.parameters() if p.requires_grad]})
    if train_unet:
        parameters.append(
            {"params": [p for p in pipe.unet.parameters() if p.requires_grad],
             "lr": float(cfg.get("train.unet_lr", 1e-6))}
        )

    optimizer = torch.optim.AdamW(
        parameters, lr=float(cfg.get("train.lr", 1e-4)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(cfg.get("train.amp", True)) and device.type == "cuda"
    )
    criterion = DiffusionLoss()

    start_epoch = 1
    if args.resume:
        payload = load_checkpoint(args.resume, map_location=device)
        iaa.load_state_dict(payload["iaa"])
        editor.injector.load_state_dict(payload["injector"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        logger(f"resumed from {args.resume}")

    noise_scheduler = pipe.scheduler
    use_amp = bool(cfg.get("train.amp", True)) and device.type == "cuda"
    instruction_dropout = float(cfg.get("train.instruction_dropout", 0.0))
    grad_accum = int(cfg.get("train.grad_accum", 1))
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    for epoch in range(start_epoch, int(cfg.get("train.epochs", 10)) + 1):
        meter = AverageMeter()
        started = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            losses = diffusion_step(
                editor,
                batch,
                noise_scheduler=noise_scheduler,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                train_alp=train_alp,
                instruction_dropout=instruction_dropout,
            )
            loss = losses["loss_total"] / grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            meter.update({k: float(v) for k, v in losses.items()})

            if (step + 1) % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    float(cfg.get("train.grad_clip", 1.0)),
                )
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % int(cfg.get("train.log_interval", 20)) == 0:
                logger(
                    f"epoch {epoch} step {step + 1}/{len(train_loader)} "
                    + " ".join(f"{k}={v:.5f}" for k, v in meter.averages().items())
                )

        record = {
            "epoch": epoch,
            "time": round(time.time() - started, 1),
            **meter.averages(),
        }
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        logger("epoch summary: " + json.dumps(record))

        save_checkpoint(
            os.path.join(output_dir, "last.pth"),
            iaa,
            optimizer,
            epoch,
            extra={"injector": editor.injector.state_dict()},
        )
        if epoch % int(cfg.get("train.save_interval", 1)) == 0:
            save_checkpoint(
                os.path.join(output_dir, f"iaa_epoch{epoch}.pth"),
                iaa,
                epoch=epoch,
                extra={"injector": editor.injector.state_dict()},
            )


if __name__ == "__main__":
    main()
