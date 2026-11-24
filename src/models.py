"""Frozen vision-language foundation models, all loaded through open_clip."""

from __future__ import annotations

import open_clip
import torch
from torchvision import transforms as T

# key -> (open_clip architecture, pretrained tag)
MODELS = {
    "siglip2_so400m": ("ViT-SO400M-16-SigLIP2-384", "webli"),  # main backbone
    "siglip2_g": ("ViT-gopt-16-SigLIP2-384", "webli"),
    "siglip_so400m": ("ViT-SO400M-14-SigLIP-384", "webli"),
    "clip_l14": ("ViT-L-14-336-quickgelu", "openai"),
    "dfn_h14": ("ViT-H-14-378-quickgelu", "dfn5b"),
    "pe_g14": ("PE-Core-bigG-14-448", "meta"),
}

# The open_clip DFN5B checkpoint keeps the initial temperature (1/0.07 = 14.3); use CLIP's trained value.
LOGIT_SCALE_OVERRIDE = {"dfn_h14": 100.0}


def load_model(key: str, device: torch.device):
    arch, tag = MODELS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag, device=device)
    model.eval().requires_grad_(False)
    return model, preprocess, open_clip.get_tokenizer(arch)


def view_transforms(preprocess: T.Compose, crop_scale: tuple[float, float]):
    """(base, random view) transforms that share the model's input size and normalisation.

    The random view follows TPT / Tip-Adapter: a random resized crop of the original image
    (aspect ratio 3/4..4/3) plus a horizontal flip, resized straight to the model's input size.
    """
    norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    size = next(t for t in preprocess.transforms if isinstance(t, T.Resize)).size
    size = size if isinstance(size, int) else size[0]
    rand = T.Compose([
        T.RandomResizedCrop(size, scale=crop_scale, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        norm,
    ])
    return preprocess, rand


@torch.no_grad()
def logit_params(model) -> tuple[float, float]:
    scale = model.logit_scale.exp().item()
    bias = model.logit_bias.item() if getattr(model, "logit_bias", None) is not None else 0.0
    return scale, bias
