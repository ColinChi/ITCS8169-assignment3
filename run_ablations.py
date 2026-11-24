"""Ablations of the final pipeline on cached features: the prompts behind the text classifier and the anchor of
the CLAP head.

Prompt sets (src/prompts.py, all three defined before any test result):
  syn     10 templates x (class name + synonyms); fixed in advance, used by the submitted model
  raw     10 templates x class name
  single  "a photo of a {}." with the class name
For each prompt set and one-shot draw: zero-shot, Tip-Adapter and the CLAP head, the balanced assignment of the
zero-shot and of the CLAP prediction, CLAP + TransCLIP, CLAP + self-training, and the CLAP head on the
LoRA-fine-tuned encoder (finetune_lora.py; the encoder itself was fine-tuned with the "syn" classifier).
For "syn" also two anchor ablations of the CLAP head: the same weight for every class (lambda_c = 1) and no
anchor (lambda_c = 0).

Writes results/acc_<model>_ablation.csv (rows "<prompts>:<method>", columns as in acc_<model>.csv) and
results/prompt_fit_<model>.csv: the cross-entropy of each zero-shot classifier on the views of the 16 shots, the
only labelled evidence about the prompts that the one-shot rule allows.
Choosing a prompt set by the accuracies in this file is model selection on the test set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from run_methods import AVG, BASE, accuracy, load
from src import methods as M
from src.models import LOGIT_SCALE_OVERRIDE

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip2_so400m")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    text, test = load(args.model, device)
    scale = LOGIT_SCALE_OVERRIDE.get(args.model, text["logit_scale"])
    xv, y = test["feats"].float(), test["labels"].cpu()
    x0, xa = M.view_average(xv[:, BASE]), M.view_average(xv[:, AVG])
    names = {"syn": M.text_classifier([p.to(device) for p in text["syn"]]),
             "raw": M.text_classifier([p.to(device) for p in text["raw"]]),
             "single": M.text_classifier([p.to(device) for p in text["raw"]], single=True)}
    lora = CACHE / f"{args.model}_lora"
    probs: dict[str, list[torch.Tensor]] = {}
    fit = []

    def add(name, out):
        probs.setdefault(name, []).append(out.float().softmax(-1).cpu())

    for seed in args.seeds:
        s = torch.load(CACHE / args.model / f"shots_seed{seed}.pt")["feats"].to(device).float()
        labels = torch.arange(len(s), device=device)
        p_shot = M.prototypes(s)
        fit.append([F.cross_entropy(scale * s.flatten(0, 1) @ t.T, labels.repeat_interleave(s.shape[1])).item()
                    for t in names.values()])
        has_lora = (lora / f"test_seed{seed}.pt").exists()
        if has_lora:  # on the CPU, as in eval_lora.py
            lx = torch.load(lora / f"test_seed{seed}.pt")["feats"].float()
            lx0, lxa = M.view_average(lx[:, :1]), M.view_average(lx)
            ls = torch.load(lora / f"shots_seed{seed}.pt")["feats"].float()

        for k, t in names.items():
            add(f"{k}:zs", M.zero_shot(x0, t, scale))
            add(f"{k}:zs+avg+bal", M.sinkhorn(M.zero_shot(xa, t, scale).log_softmax(-1)).log())
            add(f"{k}:tip", M.tip_adapter(x0, t, s, scale))
            w = M.clap(t, s, scale)
            p_avg = (scale * xa @ w.T).softmax(-1)
            add(f"{k}:clap", scale * x0 @ w.T)
            add(f"{k}:clap+avg+bal", M.sinkhorn(p_avg.log()).log())
            add(f"{k}:clap+avg+trans", M.transclip(xa, p_avg, p_shot, labels).log())
            add(f"{k}:clap+avg+st", scale * xa @ M.self_train(xa, t, s, scale).T)
            if has_lora:
                wl = M.clap(t.cpu(), ls, scale)
                add(f"{k}:lora_clap", scale * lx0 @ wl.T)
                add(f"{k}:lora_clap+avg+bal", M.sinkhorn(scale * lxa @ wl.T).log())
        for name, lam in (("clap_const", 1.0), ("clap_free", 0.0)):
            w = M.clap(names["syn"], s, scale, lam=lam)
            add(f"syn:{name}", scale * x0 @ w.T)
            add(f"syn:{name}+avg+bal", M.sinkhorn((scale * xa @ w.T).log_softmax(-1)).log())

    with open(RES / f"acc_{args.model}_ablation.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "seed0", "mean", "std"] + [f"s{k}" for k in args.seeds])
        print(f"== {args.model}  (test accuracy %, seed 0 | mean ± std over {len(args.seeds)} draws)")
        for name, p in sorted(probs.items()):
            acc = accuracy(torch.stack(p), y)
            wr.writerow([name, f"{acc[0]:.2f}", f"{acc.mean():.2f}", f"{acc.std():.2f}"] + [f"{a:.2f}" for a in acc])
            print(f"{name:26s} {acc[0]:6.2f} | {acc.mean():6.2f} ± {acc.std():.2f}")
    with open(RES / f"prompt_fit_{args.model}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["seed", *names])
        wr.writerows([seed, *(f"{v:.4f}" for v in row)] for seed, row in zip(args.seeds, fit))
    best = [min(names, key=lambda k: row[list(names).index(k)]) for row in fit]
    print("prompt set with the lowest cross-entropy on the shots, per draw:", best)


if __name__ == "__main__":
    main()
