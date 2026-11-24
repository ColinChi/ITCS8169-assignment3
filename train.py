"""Build the final one-shot classifier from the 16 raw training images of one draw.

Steps: encode the class prompts (text classifier), encode the 16 shots with 66 views each,
fit the CLAP head (zero-shot-initialised linear probe, class-adaptive constraint, L-BFGS).
The checkpoint holds everything needed at test time besides the frozen backbone, which is
downloaded from Hugging Face by name.

Usage: python train.py --model siglip2_so400m --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import open_clip
import torch

from extract_features import SHOT_VIEWS, Views, encode_images, encode_text
from src import methods as M
from src.data import CLASSES, load_shots, split_path
from src.models import LOGIT_SCALE_OVERRIDE, MODELS, load_model, logit_params

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip2_so400m", choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0, help="one-shot draw (splits/oneshot_seed<k>.json)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    model, preprocess, tokenizer = load_model(args.model, device)
    scale = LOGIT_SCALE_OVERRIDE.get(args.model, logit_params(model)[0])
    t = M.text_classifier([p.to(device) for p in encode_text(model, tokenizer, device)["syn"]])
    shots = load_shots(args.seed)
    s = encode_images(model, Views([p for p, _ in shots], SHOT_VIEWS, preprocess, seed=args.seed + 1),
                      len(SHOT_VIEWS), device).to(device).float()
    w = M.clap(t, s, scale)

    with torch.no_grad():
        train_acc = ((scale * s.flatten(0, 1) @ w.T).argmax(-1) ==
                     torch.arange(len(CLASSES), device=device).repeat_interleave(s.shape[1])).float().mean()
    out = ROOT / "checkpoints" / f"{args.model}_seed{args.seed}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({
        "model": args.model, "arch": MODELS[args.model], "seed": args.seed, "classes": CLASSES,
        "split": json.loads(split_path(args.seed).read_text()), "logit_scale": scale,
        "text_classifier": t.cpu(), "clap_weight": w.cpu(), "shot_feats": s.half().cpu(),
        "open_clip_version": open_clip.__version__, "torch_version": str(torch.__version__),
    }, out)
    print(f"saved {out}  ({out.stat().st_size / 2**20:.1f} MiB); accuracy on the shots' own views {100 * train_acc:.1f}%")


if __name__ == "__main__":
    main()
