"""E1 evaluation helpers: prediction over a cache dataset and scoring against the frozen reference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.constants import ID_COL, TARGETS
from src.e1.data import CacheDataset, plane_ids
from src.validate.metric import macro_auc, per_target_auc


@torch.no_grad()
def predict(model, ds: CacheDataset, device: str, batch: int, workers: int, amp: bool) -> np.ndarray:
    if len(ds) == 0:
        return np.zeros((0, len(TARGETS)), np.float32)
    model.eval()
    out = []
    for bt in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers):
        n = bt["x"].shape[1]
        x = bt["x"].to(device).float() / 255.0
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            logits = model(x, plane_ids(n // 3, device), bt["smask"].to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    model.train()
    return np.concatenate(out)


def reference_matrix(ds: CacheDataset) -> np.ndarray:
    """Reference labels with NaN where the reference abstains (metric ignores NaN)."""
    return np.where(ds.rmask > 0, ds.ref, np.nan)


def score_vs_reference(preds: np.ndarray, ds: CacheDataset) -> float:
    return macro_auc(reference_matrix(ds), preds) if len(preds) else float("nan")


def per_target_vs_reference(preds: np.ndarray, ds: CacheDataset) -> dict[str, float]:
    aucs = per_target_auc(reference_matrix(ds), preds)
    return {t: round(float(a), 4) for t, a in zip(TARGETS, aucs)}


def soft_target_loss(preds: np.ndarray, ds: CacheDataset) -> float:
    """Weighted BCE of predictions against the soft training targets (same objective as training)."""
    p = np.clip(preds, 1e-6, 1 - 1e-6)
    bce = -(ds.y * np.log(p) + (1 - ds.y) * np.log(1 - p))
    return float((bce * ds.w).sum() / max(ds.w.sum(), 1.0))


def save_preds(path: str, sids: list[str], preds: np.ndarray) -> None:
    df = pd.DataFrame(preds, columns=TARGETS)
    df.insert(0, ID_COL, sids)
    df.to_csv(path, index=False)
