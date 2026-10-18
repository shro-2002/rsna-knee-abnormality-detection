"""Adapter for goodpjw2008's 2.5D ConvNeXt-MIL knee reader (Apache-2.0; weights + code in Kaggle dataset
goodpjw2008/rsna-knee-2-5d-convnext-reader, used unmodified via import). We only write the study-level driver:
series selection, window cutting and 3-fold probability averaging, so it can share our batching/blending code.
Third-party pieces used: knee.KneeNet / knee.window_centres / knee.build_study_table / preprocess.load_series."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch


def import_reader(dataset_dir: str):
    sys.path.insert(0, dataset_dir)
    import knee            # noqa: E402  (third-party, Apache-2.0)
    import preprocess      # noqa: E402
    return knee, preprocess


def load_folds(dataset_dir: str, device: str):
    knee, _ = import_reader(dataset_dir)
    models = []
    for f in sorted(x for x in os.listdir(dataset_dir) if x.startswith("cnxt_v0_fold") and x.endswith(".pt")):
        ck = torch.load(os.path.join(dataset_dir, f), map_location="cpu", weights_only=False)
        m = knee.KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False)
        m.load_state_dict(ck["model"])
        models.append((m.to(device).eval(), int(ck["args"]["res"])))
    assert len(models) == 3, f"expected 3 fold checkpoints, found {len(models)}"
    return models


def prepare_study(dataset_dir: str, study_id: str, series_rows: list, series_root: str, k: int = 12):
    """-> (x uint8 [6,k,3,S,S], mask bool [6], pos float [6,k]) following the reader's own slot logic."""
    knee, pre = import_reader(dataset_dir)
    n_slot = knee.N_SLOTS
    cands = [[] for _ in range(n_slot)]
    for r in series_rows:
        d = f"{series_root}/{study_id}/{r['SeriesInstanceUID']}"
        n = min(len(os.listdir(d)), pre.MAX_SLICES) if os.path.isdir(d) else 0
        if n >= 3:
            cands[knee.slot_of(r["Anatomical_Plane"], r["Fat_Suppression"])].append((n, r["SeriesInstanceUID"]))
    x = np.zeros((n_slot, k, 3, pre.SIZE, pre.SIZE), np.uint8)
    mask, pos = np.zeros(n_slot, bool), np.zeros((n_slot, k), np.float32)
    for s, c in enumerate(cands):
        if not c:
            continue
        c.sort(key=lambda t: -t[0])
        try:
            vol, _ = pre.load_series(f"{series_root}/{study_id}/{c[0][1]}")
        except Exception:
            continue
        n = len(vol)
        if n < 3:
            continue
        centres = knee.window_centres(n, k, False)
        x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in centres])
        mask[s], pos[s] = True, centres / max(n - 1, 1)
    return x, mask, pos


@torch.no_grad()
def predict_study(models, x, mask, pos, device: str) -> np.ndarray:
    xt = torch.from_numpy(x)[None].to(device); mt = torch.from_numpy(mask)[None].to(device); pt = torch.from_numpy(pos)[None].to(device)
    ps = []
    for m, res in models:
        if str(device).startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                ps.append(m(xt, mt, pt, res=res).float().sigmoid().cpu().numpy()[0])
        else:
            ps.append(m(xt, mt, pt, res=res).float().sigmoid().cpu().numpy()[0])
    return np.mean(ps, 0)
