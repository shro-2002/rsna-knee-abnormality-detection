"""Frozen-backbone feature extraction + head-only training on cached features.

Extract once (expensive, DICOM-I/O bound), then train the MIL head many cheap epochs on cached
features held in RAM as float16.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.mil import masked_bce
from src.validate.metric import bootstrap_macro_auc, per_target_auc


@torch.no_grad()
def extract_all(model, ds, *, device, collate, nmax=15, batch_size=4, num_workers=2,
                time_budget_s=None, log=print):
    """Returns feats(M,nmax,D)f16, smask(M,nmax)f32, targets(M,12)f32, tmasks(M,12)f32.

    If time_budget_s is set, stops early once exceeded and returns what was processed (arrays
    stay self-consistent since labels come from the same batches).
    """
    model.to(device).eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    D = model.dim
    feats_all, sm_all, tg_all, tm_all = [], [], [], []
    t0, done = time.time(), 0
    for batch in loader:
        if time_budget_s is not None and (time.time() - t0) > time_budget_s:
            log(f"  TIME BUDGET hit at {done}/{len(ds)} studies; stopping extraction early")
            break
        x = batch["bag"].to(device)          # (B,N,3,H,W)
        f = model.extract(x).float().cpu()   # (B,N,D)
        B, N = f.shape[:2]
        buf = torch.zeros(B, nmax, D)
        sm = torch.zeros(B, nmax)
        n = min(N, nmax)
        buf[:, :n] = f[:, :n]
        sm[:, :n] = batch["smask"][:, :n]
        feats_all.append(buf.half())
        sm_all.append(sm)
        tg_all.append(batch["target"])
        tm_all.append(batch["tmask"])
        done += B
        if done % 200 < batch_size:
            log(f"  extracted {done}/{len(ds)}  ({(time.time()-t0)/max(done,1):.2f}s/study)")
    return (torch.cat(feats_all), torch.cat(sm_all), torch.cat(tg_all), torch.cat(tm_all))


def pos_weight_from(targets, tmasks, clip=10.0):
    """pos_weight = neg/pos per target over observed cells (clamped)."""
    t = targets.numpy()
    m = tmasks.numpy().astype(bool)
    pw = np.ones(t.shape[1], dtype=np.float32)
    for k in range(t.shape[1]):
        col = t[m[:, k], k]
        pos = float((col >= 0.5).sum()) + 1e-6
        neg = float((col < 0.5).sum()) + 1e-6
        pw[k] = min(neg / pos, clip)
    return torch.tensor(pw)


def train_head(head, feats, smask, targets, tmasks, *, epochs=30, lr=1e-3, batch_size=32,
               device, pos_weight=None, val_idx=None, ckpt_path=None, log=print, seed=42):
    torch.manual_seed(seed)
    head.to(device).train()
    n = feats.shape[0]
    idx = np.arange(n)
    val = set(val_idx.tolist()) if val_idx is not None else set()
    tr_idx = np.array([i for i in idx if i not in val])
    ds = TensorDataset(feats, smask, targets, tmasks, torch.arange(n))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    pw = pos_weight.to(device) if pos_weight is not None else None
    best = {"val": -1.0}
    for ep in range(epochs):
        head.train()
        tot = 0.0
        for f, s, y, tm, ids in loader:
            keep = torch.tensor([int(i) in val for i in ids]) == False  # noqa: E712
            if keep.sum() == 0:
                continue
            f, s, y, tm = f[keep].float().to(device), s[keep].to(device), y[keep].to(device), tm[keep].to(device)
            opt.zero_grad()
            logits = head(f, s)
            loss = masked_bce(logits, y, tm, pos_weight=pw)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        msg = f"  epoch {ep+1}/{epochs} loss={tot/max(len(loader),1):.4f}"
        if val:
            vp = predict_head(head, feats[list(val)], smask[list(val)], device=device)
            vauc = np.nanmean(per_target_auc(targets.numpy()[list(val)], vp))
            msg += f" val_weakAUC={vauc:.3f}"
            if vauc > best["val"]:
                best = {"val": vauc, "state": {k: v.cpu().clone() for k, v in head.state_dict().items()}}
        log(msg)
        if ckpt_path:  # intermediate checkpoint survives failures
            torch.save(head.state_dict(), ckpt_path)
    if best.get("state"):
        head.load_state_dict(best["state"])
    return head


@torch.no_grad()
def predict_head(head, feats, smask, *, device, batch_size=64):
    head.to(device).eval()
    out = []
    for i in range(0, feats.shape[0], batch_size):
        f = feats[i:i + batch_size].float().to(device)
        s = smask[i:i + batch_size].to(device)
        out.append(torch.sigmoid(head(f, s)).cpu().numpy())
    return np.concatenate(out, axis=0)


def eval_gold(head, gold_feats, gold_smask, y_true, *, device, log=print):
    preds = predict_head(head, gold_feats, gold_smask, device=device)
    aucs = per_target_auc(y_true, preds)
    res = bootstrap_macro_auc(y_true, preds, n_boot=2000, seed=0)
    log(f"gold-58 MACRO AUC = {res['point']:.3f} (95% CI {res['ci_low']:.3f}-{res['ci_high']:.3f})")
    return {"preds": preds, "per_target": aucs, **res}
