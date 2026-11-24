"""Classifiers on frozen, L2-normalised features. Every hyperparameter is a published default.

Shapes: test features x [N, D]; one-shot features s [C, V, D] (class c at row c);
text classifier t [C, D]; logit scale = the model's own learned temperature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def text_classifier(per_class: list[torch.Tensor], single: bool = False) -> torch.Tensor:
    """Prompt ensembling (CLIP): mean of normalised prompt embeddings, renormalised."""
    t = torch.stack([p[0] if single else p.float().mean(0) for p in per_class]).float()
    return F.normalize(t, dim=-1)


def prototypes(s: torch.Tensor) -> torch.Tensor:
    """One prototype per class: the shot's features averaged over its augmented views (Tip-Adapter cache key)."""
    return F.normalize(s.float().mean(1), dim=-1)


def zero_shot(x, t, scale):
    return scale * x @ t.T


def nearest_prototype(x, s, scale):
    return scale * x @ prototypes(s).T


def tip_adapter(x, t, s, scale, alpha=1.0, beta=5.5):
    """Training-free Tip-Adapter (Zhang et al., ECCV 2022) with its default alpha=1, beta=5.5."""
    affinity = torch.exp(-beta * (1 - x @ prototypes(s).T))
    return scale * x @ t.T + alpha * affinity


def clap(t, s, scale, extra=(), lam=None):
    """Zero-shot-initialised linear probe with CLAP's class-adaptive constraint (Silva-Rodriguez et al., CVPR 2024).

    minimise  CE(scale * f W^T) over all augmented shot views  +  sum_c lambda_c ||W_c - t_c||^2,
    lambda_c = zero-shot probability of the true class, averaged over class c's shot views.
    The objective is convex, so it is solved to convergence with L-BFGS (no learning rate / epoch count).
    ``extra``: further labelled sets [(features [n, D], labels [n]), ...] (synthetic images, test pseudo-labels),
    each adding one cross-entropy term with the same weight as the real shots; lambda_c uses the real shots only.
    ``lam``: ablation only, the same weight for every class instead of CLAP's lambda_c (0 = no anchor).
    """
    C, V, D = s.shape
    f = s.float().reshape(C * V, D)
    y = torch.arange(C, device=f.device).repeat_interleave(V)
    with torch.no_grad():
        p_zs = (scale * f @ t.T).softmax(-1)
        lam = p_zs[torch.arange(len(y)), y].view(C, V).mean(1) if lam is None else torch.full_like(p_zs[:C, 0], lam)
    w = t.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(scale * f @ w.T, y) + (lam[:, None] * (w - t) ** 2).sum()
        for xe, ye in extra:
            loss = loss + F.cross_entropy(scale * xe @ w.T, ye)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def self_train(x, t, s, scale, extra=(), steps=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """Balanced curriculum self-training (Curriculum Labeling, Cascante-Bonilla et al., AAAI 2021).

    Round r: predict the test set with the current CLAP head, assign it under a uniform class prior
    (Sinkhorn), take the top steps[r] fraction of each class's assigned samples as pseudo-labels and
    refit CLAP from the zero-shot initialisation (a restart each round limits confirmation bias).
    """
    C = t.shape[0]
    w = clap(t, s, scale, extra)
    for frac in steps:
        z = sinkhorn(scale * x @ w.T)
        conf, pred = z.max(1)
        keep = []
        for c in range(C):
            idx = (pred == c).nonzero().squeeze(1)
            m = int(round(frac * len(idx)))
            keep.append(idx[conf[idx].topk(m).indices] if m else idx[:0])
        keep = torch.cat(keep)
        w = clap(t, s, scale, [*extra, (x[keep], pred[keep])])
    return w


# ---------------------------------------------------------------- test-time adaptation (inductive)

def view_average(xv: torch.Tensor) -> torch.Tensor:
    """xv [N, V, D] -> mean embedding over views, renormalised."""
    return F.normalize(xv.float().mean(1), dim=-1)


def zero_tta(logits_v: torch.Tensor, keep: float = 0.1) -> torch.Tensor:
    """"Zero" (Farina et al., NeurIPS 2024): keep the 10% most confident views, majority-vote their argmax.

    logits_v [N, V, C] (view 0 = the original image). Returns vote counts [N, C] (ties: confidence sum).
    """
    N, V, C = logits_v.shape
    p = logits_v.softmax(-1)
    conf = p.max(-1).values
    k = max(1, round(keep * V))
    idx = conf.topk(k, dim=1).indices
    kept = torch.gather(p, 1, idx[..., None].expand(-1, -1, C))
    votes = F.one_hot(kept.argmax(-1), C).float().sum(1)
    return votes + 1e-3 * kept.sum(1)


# ---------------------------------------------------------------- transduction

def sinkhorn(log_z: torch.Tensor, n_iter: int = 50) -> torch.Tensor:
    """Rows sum to 1, columns to N/C: assignment under a uniform class prior (as in PT-MAP)."""
    N, C = log_z.shape
    col = torch.log(torch.tensor(N / C, device=log_z.device))
    for _ in range(n_iter):
        log_z = log_z - (log_z.logsumexp(0, keepdim=True) - col)
        log_z = log_z - log_z.logsumexp(1, keepdim=True)
    return log_z.exp()


def label_propagation(x, y_hat, shots=None, shot_labels=None, k=10, alpha=0.99, gamma=3):
    """Label propagation (Zhou et al., NIPS 2003) on a k-NN graph of the test (+ shot) embeddings.

    The inductive prediction y_hat is the initial label matrix of every test node, the shots enter as
    one-hot nodes. F = (I - alpha S)^-1 Y with S the symmetrically normalised graph whose edge weights are
    cos^gamma (Iscen et al., CVPR 2019 defaults: alpha=0.99, gamma=3). Linear, so it smooths the prediction
    along the data manifold without re-estimating class models. Returns [N, C] scores for the test nodes.
    """
    x, y0 = x.float(), y_hat.float()
    if shots is not None:
        x = torch.cat([x, shots.float()])
        y0 = torch.cat([y0, F.one_hot(shot_labels, y_hat.shape[1]).float()])
    sim = x @ x.T
    sim.fill_diagonal_(-1)
    nn = sim.topk(k, dim=1)
    w = torch.zeros_like(sim).scatter_(1, nn.indices, nn.values.clamp_min(0) ** gamma)
    w = w + w.T
    d = w.sum(1).clamp_min(1e-12).rsqrt()
    s = d[:, None] * w * d[None]
    f = torch.linalg.solve(torch.eye(len(x), device=x.device) - alpha * s, y0)
    f = f[: len(y_hat)].clamp_min(1e-12)
    return f / f.sum(1, keepdim=True)


def transclip(x, y_hat, shots=None, shot_labels=None, lam=1.0, n_iter=10, z_iter=5, knn=3, top_init=8,
              balanced=False):
    """TransCLIP (Zanella et al., NeurIPS 2024): transductive GMM + Laplacian + KL to the VLM prediction.

    x [N, D] test features; y_hat [N, C] inductive probabilities (zero-shot or one-shot).
    shots [S, D], shot_labels [S]: labelled samples, fixed one-hot, included in the mean/variance updates.
    Defaults from the paper: lambda=1, 10 outer iterations, 3-NN affinity, top-8 confident init.
    ``balanced`` replaces the z-step softmax by Sinkhorn (uniform class prior), our addition.
    Returns soft assignments z [N, C].
    """
    x = x.float()
    N, D = x.shape
    C = y_hat.shape[1]
    # 3-NN cosine affinity (symmetrised), no self loops
    sim = x @ x.T
    sim.fill_diagonal_(-1)
    nn = sim.topk(knn, dim=1)
    w = torch.zeros_like(sim).scatter_(1, nn.indices, nn.values.clamp_min(0))
    w = (w + w.T) / 2

    # init means: top-k most confident test samples per class, plus the labelled shots
    mu = torch.stack([x[y_hat[:, c].topk(top_init).indices].mean(0) for c in range(C)])
    if shots is not None:
        oh = F.one_hot(shot_labels, C).float()
        mu = (mu * top_init + oh.T @ shots) / (top_init + oh.sum(0)[:, None])
    var = ((x[:, None, :] - mu[None]) ** 2 * y_hat[..., None]).sum((0, 1)) / N

    log_prior = lam * torch.log(y_hat.clamp_min(1e-12))
    z = y_hat.clone()
    for _ in range(n_iter):
        log_p = -0.5 * (((x[:, None, :] - mu[None]) ** 2) / var).sum(-1)  # shared covariance: det cancels
        for _ in range(z_iter):
            logits = log_prior + log_p + w @ z
            z = sinkhorn(logits) if balanced else torch.softmax(logits, dim=1)
        zs, xs = z, x
        if shots is not None:
            zs, xs = torch.cat([z, oh]), torch.cat([x, shots])
        mu = (zs.T @ xs) / zs.sum(0)[:, None]
        var = ((xs[:, None, :] - mu[None]) ** 2 * zs[..., None]).sum((0, 1)) / len(xs)
    return z
