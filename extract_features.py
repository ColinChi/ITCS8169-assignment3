"""Encode test images, the one-shot images and the class prompts with a frozen model.

Outputs (L2-normalised fp16 features) in cache/<model>/:
  test.pt             feats [400, V_TEST, D]; gray [400, 2, D] (colour removed, base + flip)
  shots_seed<k>.pt    feats [16, V_SHOT, D] for the 16 images of one-shot draw k
  text.pt             per name set ("raw", "syn"): list over classes of [P, D] prompt features

View layout (identical crops for every model, seeded by image and view index):
  test : 0 base, 1 flip, 2..33 mild crops (area 0.5-1), 34..96 TPT crops (area 0.08-1)
  shots: 0 base, 1 flip, 2..65 mild crops (area 0.5-1)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data import CLASSES, load_shots, load_test, open_image
from src.models import MODELS, load_model, logit_params, view_transforms
from src.prompts import NAME_SETS, prompts_for

ROOT = Path(__file__).resolve().parent
MILD, TPT = (0.5, 1.0), (0.08, 1.0)
TEST_VIEWS = [("base", None)] * 2 + [("rand", MILD)] * 32 + [("rand", TPT)] * 63
SHOT_VIEWS = [("base", None)] * 2 + [("rand", MILD)] * 64


class Views(Dataset):
    def __init__(self, paths, views, preprocess, gray=False, seed=0):
        self.items = [(i, v) for i in range(len(paths)) for v in range(len(views))]
        self.paths, self.views, self.gray, self.seed = paths, views, gray, seed
        self.rand = {s: view_transforms(preprocess, s)[1] for _, s in views if s is not None}
        self.base = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, v = self.items[k]
        img = open_image(self.paths[i], gray=self.gray)
        kind, scale = self.views[v]
        if kind == "base":
            x = self.base(img)
            return x.flip(-1) if v == 1 else x
        torch.manual_seed(self.seed * 10_000_019 + i * 1_000 + v)  # same crop for every model
        return self.rand[scale](img)


@torch.no_grad()
def encode_images(model, dataset, n_views, device, batch=256):
    loader = DataLoader(dataset, batch_size=batch, num_workers=24, pin_memory=True)
    feats = []
    for x in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.encode_image(x.to(device, non_blocking=True))
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).half().cpu())
    return torch.cat(feats).view(-1, n_views, feats[0].shape[-1])


@torch.no_grad()
def encode_text(model, tokenizer, device):
    out = {}
    for set_name, names in NAME_SETS.items():
        per_class = []
        for c in CLASSES:
            tok = tokenizer(prompts_for(names[c])).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model.encode_text(tok)
            per_class.append(torch.nn.functional.normalize(f.float(), dim=-1).half().cpu())
        out[set_name] = per_class
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = ROOT / "cache" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, preprocess, tokenizer = load_model(args.model, device)
    scale, bias = logit_params(model)
    print(preprocess, flush=True)
    print(f"[{args.model}] loaded in {time.time() - t0:.0f}s; logit scale {scale:.2f}, bias {bias:.2f}", flush=True)

    text = encode_text(model, tokenizer, device)
    torch.save({"classes": CLASSES, "logit_scale": scale, "logit_bias": bias, **text}, out_dir / "text.pt")

    test = load_test()
    paths, labels = [p for p, _ in test], torch.tensor([y for _, y in test])
    t0 = time.time()
    feats = encode_images(model, Views(paths, TEST_VIEWS, preprocess), len(TEST_VIEWS), device)
    gray = encode_images(model, Views(paths, TEST_VIEWS[:2], preprocess, gray=True), 2, device)
    torch.save({"feats": feats, "gray": gray, "labels": labels,
                "paths": [str(p.relative_to(ROOT)) for p in paths]}, out_dir / "test.pt")
    print(f"[{args.model}] test {tuple(feats.shape)} in {time.time() - t0:.0f}s", flush=True)

    for seed in args.seeds:
        shots = load_shots(seed)
        sp = [p for p, _ in shots]
        f = encode_images(model, Views(sp, SHOT_VIEWS, preprocess, seed=seed + 1), len(SHOT_VIEWS), device)
        torch.save({"feats": f, "labels": torch.tensor([y for _, y in shots]), "paths": [p.name for p in sp]},
                   out_dir / f"shots_seed{seed}.pt")
    print(f"[{args.model}] shots for seeds {args.seeds} done", flush=True)


if __name__ == "__main__":
    main()
