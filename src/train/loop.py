"""Minimal, reproducible training + evaluation loop."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.mil import masked_bce
from src.validate.metric import bootstrap_macro_auc, per_target_auc


def set_backbone_trainable(model, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def train_one_fold(model, train_ds, *, epochs=2, lr=3e-4, batch_size=4, device="cuda",
                   freeze_backbone=True, num_workers=2, collate=None, seed=42, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_backbone_trainable(model, not freeze_backbone)
    model.to(device).train()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, collate_fn=collate, drop_last=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    for ep in range(epochs):
        t0 = time.time()
        total, nb = 0.0, 0
        for batch in loader:
            x = batch["bag"].to(device)
            m = batch["smask"].to(device)
            y = batch["target"].to(device)
            tm = batch["tmask"].to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                enabled=(device == "cuda")):
                logits = model(x, m)
                loss = masked_bce(logits, y, tm)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            nb += 1
        log(f"  epoch {ep+1}/{epochs} loss={total/max(nb,1):.4f} time={time.time()-t0:.1f}s")
    return model


@torch.no_grad()
def predict(model, ds, *, batch_size=4, device="cuda", collate=None, num_workers=2):
    model.to(device).eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    out = []
    for batch in loader:
        logits = model(batch["bag"].to(device), batch["smask"].to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate(model, gold_ds, y_true, *, device="cuda", collate=None, num_workers=2, log=print):
    preds = predict(model, gold_ds, device=device, collate=collate, num_workers=num_workers)
    aucs = per_target_auc(y_true, preds)
    res = bootstrap_macro_auc(y_true, preds, n_boot=2000, seed=0)
    log(f"gold-58 MACRO AUC = {res['point']:.3f} "
        f"(95% CI {res['ci_low']:.3f}-{res['ci_high']:.3f})")
    return {"preds": preds, "per_target": aucs, **res}
