"""Encode the SDXL support images (generate.py) with a frozen model.

Each synthetic image is matched to the look of the real shots before encoding: resized to 256x256
and converted to grayscale when that class's shot is grayscale (all classes except Flower).
Outputs in cache/<model>/: synth_t2i.pt feats [C, N_T2I, 2, D]; synth_var_seed<k>.pt feats [C, N_VAR, 2, D]
(views: base, flip).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from extract_features import Views, encode_images
from generate import N_T2I, N_VAR, OUT
from src.data import CLASSES, load_shots
from src.models import MODELS, load_model

ROOT = Path(__file__).resolve().parent


def gray_classes(seed: int) -> list[bool]:
    out = []
    for p, _ in load_shots(seed):
        with Image.open(p) as im:
            out.append(im.mode == "L")
    return out


class SynthViews(Views):
    def __init__(self, paths, gray_flags, preprocess):
        super().__init__(paths, [("base", None)] * 2, preprocess)
        self.gray_flags = gray_flags

    def __getitem__(self, k):
        i, v = self.items[k]
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB").resize((256, 256), Image.BICUBIC)
        if self.gray_flags[i]:
            im = im.convert("L").convert("RGB")
        x = self.base(im)
        return x.flip(-1) if v == 1 else x


def encode_set(model, preprocess, root: Path, n: int, gray: list[bool], device) -> torch.Tensor:
    paths = [root / c / f"{i:03d}.jpg" for c in CLASSES for i in range(n)]
    flags = [g for g in gray for _ in range(n)]
    f = encode_images(model, SynthViews(paths, flags, preprocess), 2, device)
    return f.view(len(CLASSES), n, 2, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    model, preprocess, _ = load_model(args.model, device)
    out = ROOT / "cache" / args.model

    gray = {s: gray_classes(s) for s in args.seeds}
    assert all(g == gray[args.seeds[0]] for g in gray.values()), "colour mode of the shots differs between draws"
    torch.save({"feats": encode_set(model, preprocess, OUT / "t2i", N_T2I, gray[args.seeds[0]], device)},
               out / "synth_t2i.pt")
    for s in args.seeds:
        torch.save({"feats": encode_set(model, preprocess, OUT / "var" / f"seed{s}", N_VAR, gray[s], device)},
                   out / f"synth_var_seed{s}.pt")
    print(f"[{args.model}] synthetic features done", flush=True)


if __name__ == "__main__":
    main()
