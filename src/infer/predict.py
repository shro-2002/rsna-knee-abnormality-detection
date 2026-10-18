"""Offline inference: trained checkpoint -> per-study 12-target predictions.

The checkpoint is a full MILModel.state_dict() (backbone + MIL head) so no internet is needed.
Studies with no decodable data get no prediction here and receive the deterministic fallback row
in build_submission (never silently dropped).
"""
from __future__ import annotations

import numpy as np
import torch

from src.constants import ID_COL, TARGETS
from src.models.mil import MILModel
from src.train.dataset import StudyBagDataset


def load_model(ckpt_path: str, device: str, backbone: str = "efficientnet_b0",
               img_size: int | None = None) -> MILModel:
    model = MILModel(backbone, pretrained=False, img_size=img_size)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_ensemble(models, test_df, series_df, data_root, img_dir, *, device, window=False,
                     k=8, size=224, max_series=4, max_slices=16, log=print):
    """Decode each study's bag ONCE and run every model on it (shared preprocessing).

    Returns (preds, statuses): preds is a list (one per model) of {sid: 12-vector}; statuses is
    {sid: "ok"|"no_data"|"error:..."}. All models here must share the same `window` preprocessing.
    """
    dummy = test_df[[ID_COL]].copy()
    for t in TARGETS:
        dummy[t] = 0.0
    ds = StudyBagDataset(dummy, dummy.copy(), series_df, data_root, img_dir,
                         k_slices=k, size=size, max_series=max_series, max_slices=max_slices,
                         window=window)
    preds = [{} for _ in models]
    statuses = {}
    for i in range(len(ds)):
        sid = test_df.iloc[i][ID_COL]
        try:
            bag = ds._study_bag(sid)  # (N,3,H,W) decoded once
        except Exception as e:  # noqa: BLE001
            statuses[sid] = f"error:{e}"
            continue
        if bag.shape[0] == 0:
            statuses[sid] = "no_data"
            continue
        x = torch.from_numpy(bag).unsqueeze(0).to(device)
        mask = torch.ones(1, bag.shape[0], device=device)
        for mi, m in enumerate(models):
            preds[mi][sid] = torch.sigmoid(m(x, mask)).squeeze(0).cpu().numpy().astype(float)
        statuses[sid] = "ok"
        if (i + 1) % 100 == 0:
            log(f"  predicted {i+1}/{len(ds)}")
    return preds, statuses


@torch.no_grad()
def predict_studies(model, test_df, series_df, data_root, img_dir, *, device,
                    k=8, size=224, max_series=4, max_slices=16, log=print):
    dummy = test_df[[ID_COL]].copy()
    for t in TARGETS:
        dummy[t] = 0.0
    ds = StudyBagDataset(dummy, dummy.copy(), series_df, data_root, img_dir,
                         k_slices=k, size=size, max_series=max_series, max_slices=max_slices)
    preds, statuses = {}, {}
    for i in range(len(ds)):
        sid = test_df.iloc[i][ID_COL]
        try:
            bag = ds._study_bag(sid)  # (N,3,H,W)
        except Exception as e:  # noqa: BLE001
            statuses[sid] = f"error:{e}"
            continue
        if bag.shape[0] == 0:
            statuses[sid] = "no_data"
            continue
        x = torch.from_numpy(bag).unsqueeze(0).to(device)
        mask = torch.ones(1, bag.shape[0], device=device)
        logit = model(x, mask)
        preds[sid] = torch.sigmoid(logit).squeeze(0).cpu().numpy().astype(float)
        statuses[sid] = "ok"
        if (i + 1) % 100 == 0:
            log(f"  predicted {i+1}/{len(ds)}")
    return preds, statuses
