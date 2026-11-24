"""Evaluate the LoRA-fine-tuned encoders (finetune_lora.py) on their cached features.

lora      fine-tuned encoder + frozen text classifier (= CLIP-LoRA's own classifier)
lora_clap fine-tuned encoder + CLAP head refit on the fine-tuned shot features
+avg / +bal as in run_methods.py. Writes results/acc_<model>_lora.csv and results/probs/<model>_lora.pt.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from run_methods import accuracy
from src import methods as M

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip2_so400m")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()
    cache = ROOT / "cache" / f"{args.model}_lora"
    text = torch.load(ROOT / "cache" / args.model / "text.pt")
    t, scale = M.text_classifier(text["syn"]), text["logit_scale"]

    probs: dict[str, list[torch.Tensor]] = {}

    def add(name, logits):
        probs.setdefault(name, []).append(logits.float().softmax(-1))

    for seed in args.seeds:
        test = torch.load(cache / f"test_seed{seed}.pt")
        xv, y = test["feats"].float(), test["labels"]
        x0, xa = M.view_average(xv[:, :1]), M.view_average(xv)
        s = torch.load(cache / f"shots_seed{seed}.pt")["feats"].float()
        w = M.clap(t, s, scale)
        for name, head in (("lora", t), ("lora_clap", w)):
            add(name, scale * x0 @ head.T)
            add(f"{name}+avg", scale * xa @ head.T)
            add(f"{name}+avg+bal", M.sinkhorn(scale * xa @ head.T).log())

    probs = {k: torch.stack(v) for k, v in probs.items()}
    torch.save({"seeds": args.seeds, "labels": y, "probs": probs}, ROOT / "results" / "probs" / f"{args.model}_lora.pt")
    with open(ROOT / "results" / f"acc_{args.model}_lora.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "seed0", "mean", "std"] + [f"s{k}" for k in args.seeds])
        for name, p in probs.items():
            acc = accuracy(p, y)
            wr.writerow([name, f"{acc[0]:.2f}", f"{acc.mean():.2f}", f"{acc.std():.2f}"] + [f"{a:.2f}" for a in acc])
            print(f"{name:16s} {acc[0]:6.2f} | {acc.mean():6.2f} ± {acc.std():.2f}   " +
                  " ".join(f"{a:.1f}" for a in acc))


if __name__ == "__main__":
    main()
