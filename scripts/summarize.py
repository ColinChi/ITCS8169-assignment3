"""Collect results/acc_<model>.csv into the tables used by README.md and the report.

Writes results/summary_main.csv (all methods, main backbone) and results/summary_backbones.csv,
and prints both as Markdown.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAIN = "siglip2_so400m"
BACKBONES = {
    "siglip2_so400m": "SigLIP 2 SO400M/16 @384",
    "siglip2_g": "SigLIP 2 g-opt/16 @384",
    "siglip_so400m": "SigLIP SO400M/14 @384",
    "clip_l14": "OpenAI CLIP L/14 @336",
    "dfn_h14": "DFN-5B CLIP H/14 @378",
    "pe_g14": "PE-Core bigG/14 @448",
}
# (group, row label, method key)
MAIN_ROWS = [
    ("Zero-shot (0 images)", "single prompt, folder names", "zs_single"),
    ("Zero-shot (0 images)", "prompt ensemble, folder names", "zs_raw"),
    ("Zero-shot (0 images)", "prompt ensemble, synonyms", "zs"),
    ("Zero-shot (0 images)", "  + colour removed from all test images", "zs_gray"),
    ("Zero-shot (0 images)", "  + view averaging (34 views)", "zs+avg"),
    ("Zero-shot (0 images)", "  + Zero TTA (64 views, top-10% vote)", "zs+zero"),
    ("Zero-shot (0 images)", "SDXL name-only support set + CLAP", "sus"),
    ("One-shot, inductive", "nearest image prototype", "proto"),
    ("One-shot, inductive", "Tip-Adapter (alpha 1, beta 5.5)", "tip"),
    ("One-shot, inductive", "CLAP head (final inductive)", "clap"),
    ("One-shot, inductive", "  + view averaging", "clap+avg"),
    ("One-shot, inductive", "  + Zero TTA", "clap+zero"),
    ("One-shot, inductive", "CLAP + SDXL text-to-image support", "clap_t2i"),
    ("One-shot, inductive", "CLAP + SDXL shot variations", "clap_var"),
    ("One-shot, inductive", "CLAP + both synthetic sets", "clap_t2i_var"),
    ("One-shot, inductive", "CLIP-LoRA fine-tuned encoder + text classifier", "lora"),
    ("One-shot, inductive", "CLIP-LoRA fine-tuned encoder + CLAP head", "lora_clap"),
    ("Transductive (400 unlabelled test images)", "zero-shot + balanced assignment", "zs+avg+bal"),
    ("Transductive (400 unlabelled test images)", "zero-shot + TransCLIP", "zs+avg+trans"),
    ("Transductive (400 unlabelled test images)", "CLAP + balanced assignment (final transductive)", "clap+avg+bal"),
    ("Transductive (400 unlabelled test images)", "CLIP-LoRA encoder + balanced assignment", "lora+avg+bal"),
    ("Transductive (400 unlabelled test images)", "CLIP-LoRA encoder + CLAP + balanced assignment", "lora_clap+avg+bal"),
    ("Transductive (400 unlabelled test images)", "CLAP + TransCLIP", "clap+avg+trans"),
    ("Transductive (400 unlabelled test images)", "CLAP + label propagation (k=10)", "clap+avg+lp"),
    ("Transductive (400 unlabelled test images)", "CLAP + balanced self-training", "clap+avg+st"),
    ("Transductive (400 unlabelled test images)", "CLAP + shot variations + self-training", "clap_var+avg+st"),
]
BACKBONE_ROWS = [("Zero-shot", "zs"), ("CLAP (inductive)", "clap"), ("CLAP + balanced (transductive)", "clap+avg+bal"),
                 ("CLAP + self-training", "clap+avg+st")]
# run_ablations.py: prompt set behind the text classifier, and the anchor of the CLAP head
PROMPTS = {"syn": "names + synonyms, 10 templates (submitted)", "raw": "names, 10 templates",
           "single": "names, “a photo of a {}.”"}
PROMPT_ROWS = [("zero-shot", "zs"), ("zero-shot + balanced assignment", "zs+avg+bal"), ("Tip-Adapter", "tip"),
               ("CLAP head", "clap"), ("CLAP + balanced assignment", "clap+avg+bal"),
               ("CLAP + TransCLIP", "clap+avg+trans"), ("CLAP + balanced self-training", "clap+avg+st"),
               ("CLIP-LoRA encoder + CLAP head", "lora_clap"),
               ("CLIP-LoRA encoder + CLAP + balanced assignment", "lora_clap+avg+bal")]
ANCHOR_ROWS = [("class-adaptive λ (CLAP, submitted)", "clap"), ("constant λ = 1", "clap_const"),
               ("no anchor (λ = 0)", "clap_free")]


def load(model):
    rows = {}
    for r in csv.DictReader(open(RES / f"acc_{model}.csv")):
        draws = [float(r[k]) for k in r if k.startswith("s") and k[1:].isdigit()]
        rows[r["method"]] = {"seed0": float(r["seed0"]), "mean": float(r["mean"]), "std": float(r["std"]),
                             "min": min(draws), "max": max(draws)}
    return rows


def main():
    acc = {m: load(m) for m in BACKBONES if (RES / f"acc_{m}.csv").exists()}
    acc[MAIN].update(load(f"{MAIN}_lora"))  # LoRA-fine-tuned main encoder (finetune_lora.py, eval_lora.py)

    with open(RES / "summary_main.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["group", "method", "key", "seed0", "mean", "std", "min", "max"])
        print(f"\n### {BACKBONES[MAIN]}: test accuracy (%)\n")
        print("| Method | Seed 0 | 10 draws: mean ± std | min – max |\n|---|---:|---:|---:|")
        group = None
        for g, label, key in MAIN_ROWS:
            r = acc[MAIN][key]
            wr.writerow([g, label.strip(), key, r["seed0"], r["mean"], r["std"], r["min"], r["max"]])
            if g != group:
                print(f"| *{g}* | | | |")
                group = g
            spread = "–" if r["std"] == 0 else f"{r['mean']:.2f} ± {r['std']:.2f}"
            rng = "–" if r["std"] == 0 else f"{r['min']:.2f} – {r['max']:.2f}"
            print(f"| {label.replace('  + ', '&nbsp;&nbsp;+ ')} | {r['seed0']:.2f} | {spread} | {rng} |")

    with open(RES / "summary_backbones.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "name"] + [f"{k}_{s}" for _, k in BACKBONE_ROWS for s in ("seed0", "mean", "std")])
        print("\n### Backbones: seed 0 / mean ± std over 10 draws (%)\n")
        print("| Backbone | " + " | ".join(n for n, _ in BACKBONE_ROWS) + " |\n|---|" + "---:|" * len(BACKBONE_ROWS))
        for m, name in BACKBONES.items():
            if m not in acc:
                continue
            cells, vals = [], []
            for _, k in BACKBONE_ROWS:
                r = acc[m][k]
                vals += [r["seed0"], r["mean"], r["std"]]
                cells.append(f"{r['seed0']:.2f}" if r["std"] == 0 else f"{r['seed0']:.2f} / {r['mean']:.1f} ± {r['std']:.1f}")
            wr.writerow([m, name] + vals)
            print(f"| {name} | " + " | ".join(cells) + " |")

    if (RES / f"acc_{MAIN}_ablation.csv").exists():
        abl = load(f"{MAIN}_ablation")
        cell = lambda r: f"{r['seed0']:.2f}" if r["std"] == 0 else f"{r['seed0']:.2f} / {r['mean']:.1f} ± {r['std']:.1f}"  # noqa: E731
        print("\n### Prompt sets: seed 0 / mean ± std over 10 draws (%)\n")
        print("| Method | " + " | ".join(PROMPTS.values()) + " |\n|---|" + "---:|" * len(PROMPTS))
        for label, key in PROMPT_ROWS:
            print(f"| {label} | " + " | ".join(cell(abl[f"{k}:{key}"]) for k in PROMPTS) + " |")
        print("\n### Anchor of the CLAP head (submitted prompts): seed 0 / mean ± std over 10 draws (%)\n")
        print("| Weight of the anchor to the text | inductive | + balanced assignment |\n|---|---:|---:|")
        for label, key in ANCHOR_ROWS:
            print(f"| {label} | {cell(abl[f'syn:{key}'])} | {cell(abl[f'syn:{key}+avg+bal'])} |")


if __name__ == "__main__":
    main()
