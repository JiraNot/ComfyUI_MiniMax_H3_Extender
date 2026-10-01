from __future__ import annotations

import comfy.model_management
import comfy.nested_tensor

from .latent_upscaler import upscale_video_latent
from .motion_context_ram import _streams_from_latent


def upscale_for_refine(sampled, scale: float, progress=None):
    """Upscale only H3 video latent; preserve first-pass audio exactly."""
    video, audio = _streams_from_latent(sampled, "samples")
    up_video, width, height = upscale_video_latent(video, scale, progress=progress)
    audio = audio.to(device=comfy.model_management.intermediate_device())
    up_video = up_video.to(device=audio.device)
    latent = sampled.copy()
    latent["samples"] = comfy.nested_tensor.NestedTensor((up_video, audio))
    return latent, width, height, audio


def preserve_first_pass_audio(refined, first_pass_audio):
    video, _ = _streams_from_latent(refined, "samples")
    out = refined.copy()
    out["samples"] = comfy.nested_tensor.NestedTensor(
        (video, first_pass_audio.to(device=video.device))
    )
    return out
