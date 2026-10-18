"""Study-level weak-label cross-validation on cached features (lower-variance than gold-58).

One bag per study, so grouping by StudyInstanceUID reduces to k-fold over studies. We train a fresh
MIL head on k-1 folds and predict the held fold (OOF), then score OOF predictions against the
binarized weak labels on observed (mask=1) cells. This measures how well a frozen representation +
head generalizes to the (noisy) weak-label task over n=4,349 — a selection signal, complemented by
the sparse gold-58 truth check.

LEAKAGE CAVEAT: no PatientID in the CSVs, so folds are study-disjoint but NOT patient-disjoint; a
patient with two studies can leak across folds (research/03). Treat weak-CV as directional.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.mil import MILHead, masked_bce
from src.train.features import pos_weight_from
from src.validate.metric import bootstrap_macro_auc, per_target_auc


def _train_head(feats, smask, targets, tmasks, idx, *, dim, epochs, lr, bs, pw, device, seed,
                head_factory=None):
    torch.manual_seed(seed)
    head = (head_factory(dim) if head_factory else MILHead(dim)).to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    ds = TensorDataset(feats[idx], smask[idx], targets[idx], tmasks[idx])
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    for _ in range(epochs):
        for f, s, y, tm in loader:
            opt.zero_grad()
            loss = masked_bce(head(f.float().to(device), s.to(device)), y.to(device), tm.to(device),
                              pos_weight=pw.to(device))
            loss.backward()
            opt.step()
    return head


@torch.no_grad()
def _predict(head, feats, smask, idx, device):
    head.eval()
    return torch.sigmoid(head(feats[idx].float().to(device), smask[idx].to(device))).cpu().numpy()


def weak_cv_oof(wF, wS, wY, wT, *, dim, n_folds=5, epochs=25, lr=1e-3, bs=32, seed=42,
                device="cpu", head_factory=None):
    """Returns OOF predictions (M,12) from study-level k-fold head training.

    head_factory(dim) -> an nn.Module mapping (B,N,D)+mask to (B,12); defaults to shared MILHead.
    """
    feats = torch.as_tensor(wF); smask = torch.as_tensor(wS)
    targets = torch.as_tensor(wY).float(); tmasks = torch.as_tensor(wT).float()
    m = feats.shape[0]
    perm = np.random.default_rng(seed).permutation(m)
    folds = np.array_split(perm, n_folds)
    pw = pos_weight_from(targets, tmasks)
    oof = np.zeros((m, targets.shape[1]), dtype=np.float32)
    for val in folds:
        tr = np.setdiff1d(perm, val)
        head = _train_head(feats, smask, targets, tmasks, tr, dim=dim, epochs=epochs, lr=lr, bs=bs,
                           pw=pw, device=device, seed=seed, head_factory=head_factory)
        oof[val] = _predict(head, feats, smask, val, device)
    return oof


def score_oof(oof, wY, wT, *, n_boot=2000, seed=0):
    """Binarize weak labels at 0.5, mask abstained cells, score OOF macro + per-target + CI."""
    y = (np.asarray(wY) >= 0.5).astype(float)
    y[np.asarray(wT) < 0.5] = np.nan  # ignore abstained (mask=0) cells
    per = per_target_auc(y, oof)
    res = bootstrap_macro_auc(y, oof, n_boot=n_boot, seed=seed)
    return {"per_target": per, **res}
