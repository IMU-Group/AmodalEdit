from __future__ import annotations

from typing import Dict

import torch

from .editor import build_added_cond_kwargs


def latent_encode(vae, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = vae.encode(images * 2.0 - 1.0).latent_dist.sample()
    return latents * vae.config.scaling_factor


def diffusion_step(
    editor,
    batch: Dict,
    noise_scheduler,
    criterion,
    device: torch.device,
    use_amp: bool = True,
    train_alp: bool = False,
    instruction_dropout: float = 0.0,
) -> Dict[str, torch.Tensor]:
    source = batch["source"].to(device)
    edited = batch["edited"].to(device)
    boxes = [box.to(device) for box in batch["boxes"]]

    instructions = list(batch["instructions"])
    if instruction_dropout > 0:
        instructions = [
            "" if torch.rand(1).item() < instruction_dropout else text
            for text in instructions
        ]

    with torch.autocast(device_type=device.type, enabled=use_amp):
        conditions = editor.build_conditions(
            source,
            boxes,
            batch["categories"],
            instructions,
            pairs=batch["pairs"] or None,
            target_mask=batch["target_mask"].to(device),
            train_alp=train_alp,
        )

        latents = latent_encode(editor.pipe.vae, edited)
        source_latents = latent_encode(editor.pipe.vae, source)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        unet = editor.pipe.unet
        if unet.config.in_channels == 2 * noisy.shape[1]:
            model_input = torch.cat([noisy, source_latents], dim=1)
        else:
            model_input = noisy

        added_cond_kwargs = build_added_cond_kwargs(
            editor.pipe,
            conditions.text_pooled,
            height=source.shape[-2],
            width=source.shape[-1],
            batch_size=source.shape[0],
        )
        with editor.conditioned(conditions):
            noise_pred = unet(
                model_input,
                timesteps,
                encoder_hidden_states=conditions.text_hidden,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        return criterion(noise_pred, target)
