"""Analysis only (not a method): how separable are the 16 classes in each frozen feature space?

Uses the TEST labels: 5-fold cross-validated logistic regression and leave-one-out 10-NN on the
view-averaged test embeddings. It bounds what any classifier on these features could reach with
many labels, i.e. it tells whether the one-shot gap comes from the representation or from the labels.
Never used to choose a method or hyperparameter. Writes results/oracle.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["siglip2_so400m", "siglip2_g", "siglip_so400m", "clip_l14", "dfn_h14", "pe_g14"]


def main():
    rows = []
    for m in MODELS:
        d = torch.load(ROOT / "cache" / m / "test.pt")
        y = d["labels"].numpy()
        x = F.normalize(d["feats"][:, :34].float().mean(1), dim=-1).numpy()
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        lin = cross_val_score(LogisticRegression(C=10, max_iter=5000), x, y, cv=cv).mean() * 100
        s = x @ x.T
        np.fill_diagonal(s, -np.inf)
        nn = np.argsort(-s, 1)[:, :10]
        knn = np.mean([np.bincount(y[r], minlength=16).argmax() == y[i] for i, r in enumerate(nn)]) * 100
        rows.append([m, f"{lin:.2f}", f"{knn:.2f}"])
        print(f"{m:15s} 5-fold linear {lin:.2f}   LOO 10-NN {knn:.2f}")
    with open(ROOT / "results" / "oracle.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["model", "linear_5fold", "knn10_loo"], *rows])


if __name__ == "__main__":
    main()
