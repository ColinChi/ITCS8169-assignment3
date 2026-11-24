"""Test-set accuracy of a trained one-shot checkpoint, computed from the raw test images.

Reports three numbers:
  zero-shot     text classifier only, original view (no training image)
  inductive     CLAP head, original view; every test image is classified on its own
  transductive  CLAP head on view-averaged embeddings (original + flip + 32 crops), then the 400 test
                predictions are assigned jointly under a uniform class prior (Sinkhorn)
Per-image predictions go to results/predictions_<checkpoint>.csv.

Usage: python evaluate.py checkpoints/siglip2_so400m_seed0.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from extract_features import TEST_VIEWS, Views, encode_images
from src import methods as M
from src.data import CLASSES, load_test
from src.models import load_model

ROOT = Path(__file__).resolve().parent
N_VIEWS = 34  # original, flip, 32 crops (area 0.5-1); same crops as extract_features.py


def report(name: str, pred: torch.Tensor, y: torch.Tensor):
    acc = (pred == y).float().mean().item() * 100
    print(f"{name:13s} {acc:6.2f}%  ({int((pred == y).sum())}/{len(y)})")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-class", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint)
    assert ck["classes"] == CLASSES

    model, preprocess, _ = load_model(ck["model"], device)
    test = load_test()
    y = torch.tensor([c for _, c in test])
    xv = encode_images(model, Views([p for p, _ in test], TEST_VIEWS[:N_VIEWS], preprocess), N_VIEWS, device)
    xv = xv.to(device).float()
    x0, xa = M.view_average(xv[:, :1]), M.view_average(xv)
    scale, t, w = ck["logit_scale"], ck["text_classifier"].to(device), ck["clap_weight"].to(device)

    print(f"{ck['model']} ({ck['arch'][0]}), one-shot draw {ck['seed']}: "
          f"{[s['path'].split('/')[-1] for s in ck['split']['shots']]}")
    preds = {
        "zero-shot": (x0 @ t.T).argmax(-1).cpu(),
        "inductive": (x0 @ w.T).argmax(-1).cpu(),
        "transductive": M.sinkhorn(scale * xa @ w.T).argmax(-1).cpu(),
    }
    for name, p in preds.items():
        report(name, p, y)

    if args.per_class:
        print("\nper-class accuracy (zero-shot / inductive / transductive), 25 images each")
        for c, name in enumerate(CLASSES):
            m = y == c
            print(f"  {name:13s} " + " / ".join(f"{int((p[m] == c).sum()):2d}" for p in preds.values()))

    out = ROOT / "results" / f"predictions_{Path(args.checkpoint).stem}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["path", "label", *preds])
        for i, (p, c) in enumerate(test):
            wr.writerow([str(p.relative_to(ROOT)), CLASSES[c], *(CLASSES[int(v[i])] for v in preds.values())])
    print(f"predictions written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
