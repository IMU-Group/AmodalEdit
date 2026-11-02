from __future__ import annotations

from typing import Optional

import torch

KINDS = (
    "sdxl_instruct_pix2pix",
    "sdxl_img2img",
    "instruct_pix2pix",
    "stable_diffusion",
)


def load_pipeline(cfg, device: Optional[str] = None):
    from diffusers import (
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInstructPix2PixPipeline,
    )

    device = device or cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    kind = cfg.get("diffusion.kind", "sdxl_instruct_pix2pix")
    if kind not in KINDS:
        raise ValueError(f"unknown diffusion.kind: {kind!r}")

    name = cfg.get("diffusion.pretrained", "fusing/instructpix2pix-xl")
    dtype_name = cfg.get("diffusion.dtype", "float16")
    dtype = torch.float16 if (dtype_name == "float16" and device != "cpu") else torch.float32

    kwargs = {"torch_dtype": dtype, "cache_dir": cfg.get("diffusion.cache_dir", None)}
    variant = cfg.get("diffusion.variant", None)
    if variant:
        kwargs["variant"] = variant
    use_safetensors = cfg.get("diffusion.use_safetensors", None)
    if use_safetensors is not None:
        kwargs["use_safetensors"] = bool(use_safetensors)

    if kind.startswith("sdxl"):
        pipeline_cls = (
            StableDiffusionXLInstructPix2PixPipeline
            if kind == "sdxl_instruct_pix2pix"
            else StableDiffusionXLImg2ImgPipeline
        )
    else:
        kwargs.update(safety_checker=None, requires_safety_checker=False)
        pipeline_cls = (
            StableDiffusionInstructPix2PixPipeline
            if kind == "instruct_pix2pix"
            else StableDiffusionPipeline
        )

    pipe = pipeline_cls.from_pretrained(name, **kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=cfg.get("diffusion.disable_progress", True))

    if cfg.get("diffusion.vae_slicing", True) and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    modules = [
        pipe.vae,
        pipe.unet,
        getattr(pipe, "text_encoder", None),
        getattr(pipe, "text_encoder_2", None),
    ]
    for module in modules:
        if module is None:
            continue
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
    return pipe


def text_dim_for_pipe(pipe) -> int:
    text_encoder_2 = getattr(pipe, "text_encoder_2", None)
    if text_encoder_2 is not None:
        return int(text_encoder_2.config.projection_dim)
    return int(pipe.text_encoder.config.hidden_size)


def is_sdxl(pipe) -> bool:
    return getattr(pipe, "text_encoder_2", None) is not None
