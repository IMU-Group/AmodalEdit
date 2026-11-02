from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from amodaledit.config import apply_dotted_override, load_config
from amodaledit.data import COCOADataset, alp_collate
from amodaledit.losses import AmodalMaskLoss
from amodaledit.metrics import mask_metrics
from amodaledit.models import AmodalLayoutPerception
from amodaledit.utils import (
    AverageMeter,
    Logger,
    count_parameters,
    get_device,
    human_params,
    load_checkpoint,
    make_cosine_scheduler,
    param_groups,
    save_checkpoint,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ALP module (stage 1)")
    parser.add_argument("--config", default="configs/alp_stage1.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="dotted config override, e.g. --set train.epochs=10")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run 2 steps on a tiny subset")
    return parser


def build_dataset(cfg, split: str) -> COCOADataset:
    return COCOADataset(
        root=cfg.get("data.root"),
        annotation_file=cfg.get(f"data.{split}_annotation"),
        image_dir=cfg.get(f"data.{split}_image_dir", None),
        image_size=cfg.get("data.image_size", 512),
        mask_resolution=cfg.get("model.mask_decoder.resolution", 256),
        max_objects=cfg.get("data.max_objects", None),
        min_visible_area=cfg.get("data.min_visible_area", 0.0),
        augment=cfg.get(f"data.augment.{split}", False),
        graph_file=cfg.get("data.graph_file", None),
        exclude_stuff=cfg.get("data.exclude_stuff", True),
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch, logger):
    model.train()
    meter = AverageMeter()
    use_amp = bool(cfg.get("train.amp", False)) and device.type == "cuda"

    for step, batch in enumerate(loader):
        images = batch["images"].to(device)
        boxes = [box.to(device) for box in batch["boxes"]]
        targets = batch["amodal_mask"].to(device)
        visible = batch["visible_mask"].to(device)
        pairs = batch["pairs"] or None
        target_mask = batch["target_mask"].to(device)

        if targets.shape[0] == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(
                images,
                boxes,
                batch["categories"],
                pairs=pairs,
                target_mask=target_mask,
            )
            losses = criterion(
                output.amodal_logits, targets, visible, output.visible_logits
            )
            loss = losses["loss_total"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("train.grad_clip", 0.5))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("train.grad_clip", 0.5))
            optimizer.step()

        meter.update({k: float(v) for k, v in losses.items()})

        if (step + 1) % int(cfg.get("train.log_interval", 50)) == 0:
            values = meter.averages()
            logger(
                f"epoch {epoch} step {step + 1}/{len(loader)} "
                + " ".join(f"{k}={v:.4f}" for k, v in values.items())
            )

    return meter.averages()


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, logger) -> Dict[str, float]:
    model.eval()
    meter = AverageMeter()
    metric_meter = AverageMeter()
    use_amp = bool(cfg.get("train.amp", False)) and device.type == "cuda"

    for batch in loader:
        images = batch["images"].to(device)
        boxes = [box.to(device) for box in batch["boxes"]]
        targets = batch["amodal_mask"].to(device)
        visible = batch["visible_mask"].to(device)
        if targets.shape[0] == 0:
            continue
        pairs = batch["pairs"] or None
        target_mask = batch["target_mask"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(
                images, boxes, batch["categories"], pairs=pairs, target_mask=target_mask
            )
            losses = criterion(output.amodal_logits, targets, visible, output.visible_logits)

        meter.update({k: float(v) for k, v in losses.items()}, n=targets.shape[0])
        metrics = mask_metrics(
            output.amodal_logits.sigmoid().squeeze(1),
            targets.squeeze(1) if targets.dim() == 4 else targets,
            prefix="amodal_",
        )
        metric_meter.update(metrics, n=targets.shape[0])

    values = {**meter.averages(), **metric_meter.averages()}
    if logger is not None and values:
        logger("eval: " + " ".join(f"{k}={v:.4f}" for k, v in values.items()))
    return values


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    for override in args.overrides:
        apply_dotted_override(cfg, override)
    if args.smoke:
        cfg["train.epochs"] = 1
        cfg["data.max_objects"] = 4
        cfg["data.batch_size"] = 2
        cfg["train.log_interval"] = 1

    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    output_dir = args.output_dir or cfg.get("output_dir", "runs/alp_stage1")
    os.makedirs(output_dir, exist_ok=True)
    logger = Logger(os.path.join(output_dir, "train.log"))

    train_dataset = build_dataset(cfg, "train")
    val_dataset = build_dataset(cfg, "val")
    if args.smoke:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(min(4, len(train_dataset)))))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(min(2, len(val_dataset)))))

    loader_kwargs = dict(
        batch_size=int(cfg.get("data.batch_size", 4)),
        num_workers=int(cfg.get("data.num_workers", 4)),
        collate_fn=alp_collate,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    logger(
        f"train={len(train_dataset)} val={len(val_dataset)} "
        f"device={device} config={cfg['_config_path']}"
    )

    model = AmodalLayoutPerception.from_config(cfg).to(device)
    logger(f"ALP parameters: {human_params(count_parameters(model))}")

    optimizer = torch.optim.AdamW(
        param_groups(
            model.named_parameters(),
            lr=float(cfg.get("train.lr", 5e-5)),
            backbone_lr_scale=float(cfg.get("train.backbone_lr_scale", 0.1)),
        ),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    scheduler = make_cosine_scheduler(
        optimizer,
        epochs=int(cfg.get("train.epochs", 100)),
        warmup_epochs=int(cfg.get("train.warmup_epochs", 3)),
        min_lr=float(cfg.get("train.min_lr", 1e-6)),
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(cfg.get("train.amp", False)) and device.type == "cuda"
    )

    criterion = AmodalMaskLoss(
        bce_weight=float(cfg.get("loss.bce_weight", 1.0)),
        dice_weight=float(cfg.get("loss.dice_weight", 5.0)),
        occluded_region_weight=float(cfg.get("loss.occluded_region_weight", 2.0)),
        boundary_weight=float(cfg.get("loss.boundary_weight", 2.0)),
        visible_weight=float(cfg.get("loss.visible_weight", 0.5)),
    ).to(device)

    start_epoch = 1
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", 0)) + 1
        logger(f"resumed from {args.resume} (epoch {start_epoch - 1})")

    if args.eval_only:
        evaluate(model, val_loader, criterion, device, cfg, logger)
        return

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    best_dice = -1.0
    for epoch in range(start_epoch, int(cfg.get("train.epochs", 100)) + 1):
        started = time.time()
        train_values = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch, logger
        )
        scheduler.step()
        eval_values = evaluate(model, val_loader, criterion, device, cfg, logger)

        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                  "time": round(time.time() - started, 1), **train_values, **eval_values}
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        save_checkpoint(
            os.path.join(output_dir, "last.pth"), model, optimizer, epoch, eval_values
        )
        dice = eval_values.get("amodal_dice", -1.0)
        if dice > best_dice:
            best_dice = dice
            save_checkpoint(
                os.path.join(output_dir, "best.pth"), model, optimizer, epoch, eval_values
            )
            logger(f"new best amodal_dice={dice:.4f} -> best.pth")

    logger(f"done. best amodal_dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
