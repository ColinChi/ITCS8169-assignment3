"""Run every classifier on the cached features of one or more models.

For each model and one-shot draw it stores test-set probabilities of every method in
results/probs/<model>.pt ({method: [n_seeds, 400, 16]}) and accuracies in results/acc_<model>.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src import methods as M
from src.models import LOGIT_SCALE_OVERRIDE, MODELS

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RES = ROOT / "results"
BASE, AVG = [0], list(range(34))          # original view; original + flip + 32 mild crops
TPT_VIEWS = [0] + list(range(34, 97))     # original + 63 TPT crops (area 0.08-1)


def load(model: str, device):
    text = torch.load(CACHE / model / "text.pt")
    test = torch.load(CACHE / model / "test.pt")
    return text, {k: v.to(device) if torch.is_tensor(v) else v for k, v in test.items()}


def run_model(model: str, seeds: list[int], device) -> dict[str, torch.Tensor]:
    text, test = load(model, device)
    scale = LOGIT_SCALE_OVERRIDE.get(model, text["logit_scale"])
    xv = test["feats"].float()
    x0, xa = M.view_average(xv[:, BASE]), M.view_average(xv[:, AVG])
    x_gray = test["gray"][:, 0].float()
    x_tpt = xv[:, TPT_VIEWS]
    t = {k: M.text_classifier([p.to(device) for p in text[k]]) for k in ("raw", "syn")}
    t_single = M.text_classifier([p.to(device) for p in text["raw"]], single=True)
    probs: dict[str, list[torch.Tensor]] = {}

    def add(name, out):
        probs.setdefault(name, []).append(out.float().softmax(-1) if not name.endswith("zero") else
                                          out / out.sum(-1, keepdim=True))

    def classify(name, head):
        """head(x) -> logits; evaluated on the original view, view-averaged features and Zero voting."""
        add(name, head(x0))
        add(name + "+avg", head(xa))
        add(name + "+zero", M.zero_tta(head(x_tpt)))

    for seed in seeds:
        sh = torch.load(CACHE / model / f"shots_seed{seed}.pt")
        s = sh["feats"].to(device).float()
        assert torch.equal(sh["labels"], torch.arange(len(s)))
        tt = t["syn"]

        # zero-shot (identical for every draw; recomputed so every method has n_seeds rows)
        add("zs_single", M.zero_shot(x0, t_single, scale))
        add("zs_raw", M.zero_shot(x0, t["raw"], scale))
        add("zs_gray", M.zero_shot(x_gray, tt, scale))
        classify("zs", lambda x: M.zero_shot(x, tt, scale))

        # one-shot
        classify("proto", lambda x: M.nearest_prototype(x, s, scale))
        classify("tip", lambda x: M.tip_adapter(x, tt, s, scale))
        w = M.clap(tt, s, scale)
        classify("clap", lambda x: scale * x @ w.T)

        # transduction over the 400 unlabelled test embeddings (view-averaged features)
        p_shot = M.prototypes(s)
        labels = torch.arange(len(s), device=device)
        add("zs+avg+trans", M.transclip(xa, probs["zs+avg"][-1]).log())
        add("tip+avg+trans", M.transclip(xa, probs["tip+avg"][-1], p_shot, labels).log())
        add("clap+avg+trans", M.transclip(xa, probs["clap+avg"][-1], p_shot, labels).log())

        # the same with a uniform class prior (Sinkhorn); "+bal" alone = balancing the inductive prediction
        add("zs+avg+bal", M.sinkhorn(probs["zs+avg"][-1].log()).log())
        add("clap+avg+bal", M.sinkhorn(probs["clap+avg"][-1].log()).log())
        add("zs+avg+trans_bal", M.transclip(xa, probs["zs+avg"][-1], balanced=True).log())
        add("clap+avg+trans_bal", M.transclip(xa, probs["clap+avg"][-1], p_shot, labels, balanced=True).log())

        # balanced curriculum self-training of the CLAP head on test pseudo-labels
        w_st = M.self_train(xa, tt, s, scale)
        add("clap+avg+st", scale * xa @ w_st.T)
        add("clap+avg+st+bal", M.sinkhorn(scale * xa @ w_st.T).log())

        # label propagation on the test k-NN graph (k=10 main; 5 and 20 as a sensitivity check)
        for k in (5, 10, 20):
            sfx = "" if k == 10 else f"_k{k}"
            add(f"zs+avg+lp{sfx}", M.label_propagation(xa, probs["zs+avg"][-1], k=k).log())
            lp = M.label_propagation(xa, probs["clap+avg"][-1], p_shot, labels, k=k)
            add(f"clap+avg+lp{sfx}", lp.log())
            add(f"clap+avg+lp{sfx}+bal", M.sinkhorn(lp.log()).log())

        # synthetic support sets from SDXL: t2i = from class names only, var = IP-Adapter variations of the shot
        if (CACHE / model / "synth_t2i.pt").exists():
            def flat(f):  # [C, n, views, D] -> (features, labels)
                f = f.to(device).float()
                return f.flatten(0, 2), torch.arange(f.shape[0], device=device).repeat_interleave(f[0].numel() // f.shape[-1])

            t2i_f = torch.load(CACHE / model / "synth_t2i.pt")["feats"]
            t2i = flat(t2i_f)
            var = flat(torch.load(CACHE / model / f"synth_var_seed{seed}.pt")["feats"])
            w_sus = M.clap(tt, t2i_f.to(device).float().flatten(1, 2), scale)  # no real image at all
            classify("sus", lambda x: scale * x @ w_sus.T)
            add("sus+avg+bal", M.sinkhorn(probs["sus+avg"][-1].log()).log())
            for name, extra in (("clap_t2i", [t2i]), ("clap_var", [var]), ("clap_t2i_var", [t2i, var])):
                w_e = M.clap(tt, s, scale, extra)
                classify(name, lambda x: scale * x @ w_e.T)
                add(f"{name}+avg+bal", M.sinkhorn(probs[f"{name}+avg"][-1].log()).log())
                w_st = M.self_train(xa, tt, s, scale, extra)
                add(f"{name}+avg+st", scale * xa @ w_st.T)
    return {k: torch.stack(v).cpu() for k, v in probs.items()}


def accuracy(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (p.argmax(-1) == y).float().mean(-1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    (RES / "probs").mkdir(parents=True, exist_ok=True)

    for model in args.models:
        if not (CACHE / model / f"shots_seed{args.seeds[-1]}.pt").exists():
            print(f"skip {model}: features not cached")
            continue
        probs = run_model(model, args.seeds, device)
        y = torch.load(CACHE / model / "test.pt")["labels"]
        torch.save({"seeds": args.seeds, "labels": y, "probs": probs}, RES / "probs" / f"{model}.pt")
        with open(RES / f"acc_{model}.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["method", "seed0", "mean", "std"] + [f"s{k}" for k in args.seeds])
            print(f"\n== {model}  (test accuracy %, seed 0 | mean ± std over {len(args.seeds)} draws)")
            for name, p in probs.items():
                acc = accuracy(p, y)
                wr.writerow([name, f"{acc[0]:.2f}", f"{acc.mean():.2f}", f"{acc.std():.2f}"]
                            + [f"{a:.2f}" for a in acc])
                print(f"{name:18s} {acc[0]:6.2f} | {acc.mean():6.2f} ± {acc.std():.2f}")


if __name__ == "__main__":
    main()
