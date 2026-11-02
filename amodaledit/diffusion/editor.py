from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .adapter import AdapterInjector
from .io import is_sdxl, text_dim_for_pipe
from ..models.alp import AmodalLayoutPerception, ALPOutput
from ..models.iaa import IAAOutput, InstructionAmodalAdapter


@dataclass
class EditConditions:
    fused_feature: torch.Tensor
    fused_mask: torch.Tensor
    text_hidden: torch.Tensor
    text_pooled: torch.Tensor
    alp_output: Optional[ALPOutput] = None
    iaa_output: Optional[IAAOutput] = None


def encode_text(pipe, texts: Sequence[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    prompts = list(texts)
    batch = len(prompts)

    if not is_sdxl(pipe):
        tokenizer = pipe.tokenizer
        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        with torch.no_grad():
            hidden = pipe.text_encoder(input_ids)[0]
        pooled = hidden[
            torch.arange(batch, device=device), input_ids.argmax(dim=-1)
        ]
        return hidden, pooled

    hidden_states = []
    pooled = None
    for tokenizer, text_encoder in (
        (pipe.tokenizer, pipe.text_encoder),
        (pipe.tokenizer_2, pipe.text_encoder_2),
    ):
        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        with torch.no_grad():
            output = text_encoder(input_ids, output_hidden_states=True)
        hidden_states.append(output.hidden_states[-2])
        if pooled is None and output[0].ndim == 2:
            pooled = output[0]

    hidden = torch.cat(hidden_states, dim=-1)
    if pooled is None:
        pooled = hidden.mean(dim=1)
    return hidden, pooled


def build_added_cond_kwargs(
    pipe,
    pooled: torch.Tensor,
    height: int,
    width: int,
    batch_size: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if getattr(pipe.unet.config, "addition_embed_type", None) != "text_time":
        return None
    time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=pooled.dtype, device=pooled.device
    ).repeat(batch_size, 1)
    return {"text_embeds": pooled, "time_ids": time_ids}


class AmodalEditor(nn.Module):
    def __init__(
        self,
        pipe,
        alp: AmodalLayoutPerception,
        iaa: InstructionAmodalAdapter,
        injector: Optional[AdapterInjector] = None,
        lambda_p: float = 1.0,
        adapter_layers: Optional[int] = None,
        adapter_include: Optional[str] = None,
        image_size: int = 1024,
        alp_image_size: int = 512,
    ) -> None:
        super().__init__()
        self.pipe = pipe
        self.alp = alp
        self.iaa = iaa
        self.device = next(pipe.unet.parameters()).device
        self.image_size = image_size
        self.alp_image_size = alp_image_size

        if injector is None:
            injector = AdapterInjector(
                pipe.unet,
                context_dim=alp.embed_dim,
                num_heads=iaa.editing_attn.num_heads,
                include=adapter_include,
                max_layers=adapter_layers,
            )
        self.injector = injector
        self.lambda_p = lambda_p

    @classmethod
    def from_config(cls, cfg) -> "AmodalEditor":
        from .io import load_pipeline

        pipe = load_pipeline(cfg)
        alp = AmodalLayoutPerception.from_config(cfg)
        iaa = InstructionAmodalAdapter.from_config(cfg, text_dim=text_dim_for_pipe(pipe))
        device = cfg.get("device", "cuda")
        alp.to(device)
        iaa.to(device)
        _freeze(alp)
        return cls(
            pipe=pipe,
            alp=alp,
            iaa=iaa,
            lambda_p=cfg.get("iaa.lambda_p", 1.0),
            adapter_layers=cfg.get("iaa.adapter_layers", None),
            adapter_include=cfg.get("iaa.adapter_include", None),
            image_size=int(cfg.get("diffusion.image_size", 1024)),
            alp_image_size=int(cfg.get("data.image_size", 512)),
        )

    def build_conditions(
        self,
        images: torch.Tensor,
        boxes: Sequence[torch.Tensor],
        categories: Sequence[Sequence[str]],
        instructions: Sequence[str],
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        target_mask: Optional[torch.Tensor] = None,
        train_alp: bool = False,
    ) -> EditConditions:
        if not train_alp:
            self.alp.eval()
        alp_out = self.alp(
            images,
            boxes,
            categories,
            pairs=pairs,
            target_mask=target_mask,
            return_masks=False,
        )
        text_hidden, text_pooled = encode_text(self.pipe, instructions, images.device)
        iaa_out = self.iaa(
            alp_out.tokens,
            alp_out.image_index,
            alp_out.target_mask,
            text_pooled,
            num_images=images.shape[0],
        )
        fused_mask = torch.ones(
            iaa_out.fused_feature.shape[:2],
            dtype=torch.bool,
            device=alp_out.tokens.device,
        )
        return EditConditions(
            fused_feature=iaa_out.fused_feature,
            fused_mask=fused_mask,
            text_hidden=text_hidden,
            text_pooled=text_pooled,
            alp_output=alp_out,
            iaa_output=iaa_out,
        )

    @contextlib.contextmanager
    def conditioned(self, conditions: EditConditions):
        self.injector.set_context(conditions.fused_feature, ~conditions.fused_mask)
        try:
            yield self
        finally:
            self.injector.clear_context()

    @torch.no_grad()
    def edit(
        self,
        image,
        instruction: str,
        boxes: torch.Tensor,
        categories: Sequence[str],
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        target_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        strength: float = 0.9,
        seed: Optional[int] = None,
        return_conditions: bool = False,
    ):
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        width, height = image.size
        alp_image = image.resize((self.alp_image_size, self.alp_image_size), Image.BILINEAR)
        alp_boxes = boxes.float().clone()
        if alp_boxes.numel():
            alp_boxes[:, 0::2] *= self.alp_image_size / float(width)
            alp_boxes[:, 1::2] *= self.alp_image_size / float(height)

        alp_tensor = pil_to_tensor(alp_image).float().div(255.0).unsqueeze(0).to(self.device)
        conditions = self.build_conditions(
            alp_tensor,
            [alp_boxes],
            [list(categories)],
            [instruction],
            pairs=[list(pairs)] if pairs else None,
            target_mask=target_mask,
        )

        pipe_image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        available = set(inspect.signature(self.pipe.__call__).parameters)
        kwargs: Dict[str, object] = dict(
            prompt=instruction,
            image=pipe_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=self.image_size,
            width=self.image_size,
        )
        if "guidance_scale" in available:
            kwargs["guidance_scale"] = text_guidance_scale
        if "image_guidance_scale" in available:
            kwargs["image_guidance_scale"] = image_guidance_scale
        if "strength" in available:
            kwargs["strength"] = strength
        kwargs = {key: value for key, value in kwargs.items() if key in available}

        with self.conditioned(conditions):
            result = self.pipe(**kwargs).images[0]

        if return_conditions:
            return result, conditions
        return result


def _freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
