"""Fine-tune the vision encoder on the 16 shots of one draw with LoRA, then cache features.

CLIP-LoRA defaults (Zanella & Ben Ayed, CVPRW 2024): rank 2, alpha 1, dropout 0.25 on q/k/v of every layer;
AdamW (lr 2e-4, weight decay 1e-2), cosine schedule to 1e-6, 500 x shots iterations, batch 32, random resized
crops (area 0.08-1) + flips, cross-entropy on the model's scaled cosine logits. Deviation: only the vision
encoder is adapted; the class weights stay the frozen prompt-ensemble text classifier.

Outputs in cache/<model>_lora/: test_seed<k>.pt (34 test views), shots_seed<k>.pt (66 shot views),
lora_seed<k>.pt (LoRA weights) and log_seed<k>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from extract_features import SHOT_VIEWS, TEST_VIEWS, Views, encode_images
from src import methods as M
from src.data import load_shots, load_test, open_image
from src.lora import add_lora, lora_state
from src.models import LOGIT_SCALE_OVERRIDE, load_model, logit_params, view_transforms

ROOT = Path(__file__).resolve().parent


class ShotStream(Dataset):
    """Item k = a fresh random view of shot k % 16 (every batch of 32 holds each shot twice)."""

    def __init__(self, images, transform, n_items, seed):
        self.images, self.transform, self.n, self.seed = images, transform, n_items, seed

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        torch.manual_seed(self.seed * 10_000_019 + k)
        c = k % len(self.images)
        return self.transform(self.images[c]), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip2_so400m")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iters", type=int, default=500)   # 500 x shots, shots = 1
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    out = ROOT / "cache" / f"{args.model}_lora"
    out.mkdir(parents=True, exist_ok=True)

    model, preprocess, _ = load_model(args.model, device)
    scale = LOGIT_SCALE_OVERRIDE.get(args.model, logit_params(model)[0])
    text = torch.load(ROOT / "cache" / args.model / "text.pt")
    t = M.text_classifier([p.to(device) for p in text["syn"]])

    torch.manual_seed(args.seed)
    mods = add_lora(model.visual.trunk, r=2, alpha=1.0, dropout=0.25)
    for m in mods:
        m.to(device).train()
    params = [p for m in mods for p in (m.A, m.B)]
    print(f"LoRA parameters: {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-2, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-6)

    shots = load_shots(args.seed)
    images = [open_image(p) for p, _ in shots]
    aug = view_transforms(preprocess, (0.08, 1.0))[1]
    loader = DataLoader(ShotStream(images, aug, args.iters * args.batch, args.seed), batch_size=args.batch,
                        num_workers=12, pin_memory=True)
    log, t0 = [], time.time()
    for it, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.encode_image(x)
        logits = scale * F.normalize(f.float(), dim=-1) @ t.T
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 50 == 0 or it == args.iters - 1:
            acc = (logits.argmax(-1) == y).float().mean().item()
            log.append({"iter": it, "loss": loss.item(), "acc": acc})
            print(f"seed {args.seed} it {it:4d} loss {loss.item():.4f} acc {acc:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)

    for m in mods:
        m.eval()
    test = load_test()
    xv = encode_images(model, Views([p for p, _ in test], TEST_VIEWS[:34], preprocess), 34, device)
    s = encode_images(model, Views([p for p, _ in shots], SHOT_VIEWS, preprocess, seed=args.seed + 1),
                      len(SHOT_VIEWS), device)
    torch.save({"feats": xv, "labels": torch.tensor([c for _, c in test])}, out / f"test_seed{args.seed}.pt")
    torch.save({"feats": s, "labels": torch.arange(len(shots))}, out / f"shots_seed{args.seed}.pt")
    torch.save(lora_state(mods), out / f"lora_seed{args.seed}.pt")
    (out / f"log_seed{args.seed}.json").write_text(json.dumps(log, indent=1))
    print(f"seed {args.seed} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
