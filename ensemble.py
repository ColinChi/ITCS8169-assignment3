"""Average the test-set probabilities of several models (per method and one-shot draw)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src import methods as M

RES = Path(__file__).resolve().parent / "results"
METHODS = ["zs", "zs+avg", "tip+avg", "clap", "clap+avg", "zs+avg+bal", "clap+avg+bal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()
    runs = [torch.load(RES / "probs" / f"{m}.pt") for m in args.models]
    y = runs[0]["labels"]
    print(f"ensemble of {' + '.join(args.models)}  (seed 0 | mean ± std over draws)")
    for name in args.methods:
        base = name.removesuffix("+bal")
        p = torch.stack([r["probs"][base] for r in runs]).mean(0)          # [seeds, N, C]
        if name.endswith("+bal"):
            p = torch.stack([M.sinkhorn(q.log()) for q in p])
        acc = (p.argmax(-1) == y).float().mean(-1) * 100
        print(f"  {name:14s} {acc[0]:6.2f} | {acc.mean():6.2f} ± {acc.std():.2f}")


if __name__ == "__main__":
    main()
