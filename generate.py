"""Synthetic support images from a generative foundation model (SDXL).

t2i : text-to-image, N_T2I images per class from class names only (SuS-X style); no training image used.
var : shot variations, N_VAR images per class and one-shot draw, conditioned on that draw's single image
      through IP-Adapter (image prompt) plus the class name (text prompt).

Images are written to synth/t2i/<class>/ and synth/var/seed<k>/<class>/ (512 px JPEG).
Jobs are sharded over processes: --shard i --nshards n.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from src.data import CLASSES, load_shots, open_image
from src.prompts import RAW_NAMES, SYN_NAMES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "synth"
N_T2I, N_VAR, BATCH = 32, 16, 8
T2I_TEMPLATES = ["a photograph of a {}.", "a realistic photo of a {}.", "a wide-angle photo of a {}.",
                 "a {}, photograph."]
NEGATIVE = "text, watermark, logo, cartoon, drawing, painting, illustration, 3d render, blurry"
IP_SCALE = 0.6


def t2i_prompt(c: str, i: int) -> str:
    names = SYN_NAMES[c]
    return T2I_TEMPLATES[(i // len(names)) % len(T2I_TEMPLATES)].format(names[i % len(names)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    jobs = [("t2i", None, c, b) for c in CLASSES for b in range(0, N_T2I, BATCH)]
    jobs += [("var", s, c, b) for s in args.seeds for c in CLASSES for b in range(0, N_VAR, BATCH)]
    jobs = jobs[args.shard::args.nshards]
    jobs.sort(key=lambda j: j[0] == "var")  # text-only jobs first, then load the image adapter

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    adapter_loaded = False

    for mode, seed, c, b in jobs:
        ci = CLASSES.index(c)
        if mode == "t2i":
            out = OUT / "t2i" / c
            prompts = [t2i_prompt(c, i) for i in range(b, b + BATCH)]
            gens = [torch.Generator(args.device).manual_seed(ci * 1000 + i) for i in range(b, b + BATCH)]
            kw = {}
        else:
            if not adapter_loaded:
                pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                                     weight_name="ip-adapter_sdxl.safetensors")
                pipe.set_ip_adapter_scale(IP_SCALE)
                adapter_loaded = True
            out = OUT / "var" / f"seed{seed}" / c
            shot_path, y = load_shots(seed)[ci]
            assert y == ci
            prompts = [f"a photograph of a {RAW_NAMES[c][0]}."] * BATCH
            gens = [torch.Generator(args.device).manual_seed(seed * 100_000 + ci * 1000 + i)
                    for i in range(b, b + BATCH)]
            kw = {"ip_adapter_image": open_image(shot_path)}
        if all((out / f"{i:03d}.jpg").exists() for i in range(b, b + BATCH)):
            continue
        images = pipe(prompt=prompts, negative_prompt=[NEGATIVE] * BATCH, num_inference_steps=30,
                      guidance_scale=5.0, generator=gens, **kw).images
        out.mkdir(parents=True, exist_ok=True)
        for i, im in zip(range(b, b + BATCH), images):
            im.resize((512, 512)).save(out / f"{i:03d}.jpg", quality=95)
        print(mode, seed, c, b, flush=True)


if __name__ == "__main__":
    main()
