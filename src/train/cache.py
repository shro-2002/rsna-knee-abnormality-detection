"""In-RAM decoded-image-bag cache for fine-tuning (EXP-006).

Fine-tuning needs images every epoch; re-decoding DICOMs each epoch blows the 9h budget (true-2.5D
decode is ~3.2 s/study). So decode ALL bags ONCE into an in-RAM uint8 array (~10.5 GB for
4,349 × 16 × 3 × 224²), then FT epochs read from RAM (no decode). Pixels are cached pre-ImageNet-norm
([0,1]→uint8); CachedBagDataset de-quantizes + normalizes per item on load.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dicom import imagenet_norm_chw


def build_ram_cache(ds_raw, *, nmax=16, size=224, time_budget_s=None, log=print):
    """ds_raw: a StudyBagDataset built with norm=False (returns [0,1] bags). -> cache dict.

    Returns {bags(M,nmax,3,size,size)uint8, nvalid(M), targets(M,12), tmasks(M,12)}. If the time
    budget is hit, the cache is trimmed to the studies processed (arrays stay self-consistent).
    """
    m_total = len(ds_raw)
    bags = np.zeros((m_total, nmax, 3, size, size), dtype=np.uint8)
    nvalid = np.zeros(m_total, dtype=np.int32)
    targets = np.zeros((m_total, 12), dtype=np.float32)
    tmasks = np.zeros((m_total, 12), dtype=np.float32)
    t0 = time.time()
    processed = m_total
    for i in range(m_total):
        if time_budget_s is not None and (time.time() - t0) > time_budget_s:
            log(f"  CACHE TIME BUDGET hit at {i}/{m_total}; trimming")
            processed = i
            break
        item = ds_raw[i]
        bag = item["bag"].numpy() if hasattr(item["bag"], "numpy") else np.asarray(item["bag"])
        n = min(int(bag.shape[0]), nmax)
        if n > 0:
            bags[i, :n] = np.clip(bag[:n] * 255.0, 0, 255).astype(np.uint8)
        nvalid[i] = n
        targets[i] = item["target"].numpy() if hasattr(item["target"], "numpy") else item["target"]
        tmasks[i] = item["tmask"].numpy() if hasattr(item["tmask"], "numpy") else item["tmask"]
        if (i + 1) % 200 == 0:
            log(f"  cached {i+1}/{m_total} ({(time.time()-t0)/(i+1):.2f}s/study)")
    return {"bags": bags[:processed], "nvalid": nvalid[:processed],
            "targets": targets[:processed], "tmasks": tmasks[:processed]}


class CachedBagDataset(Dataset):
    """Reads the in-RAM uint8 cache; de-quantizes to [0,1] and ImageNet-normalizes per item."""

    def __init__(self, cache: dict):
        self.cache = cache

    def __len__(self):
        return len(self.cache["nvalid"])

    def __getitem__(self, i):
        n = int(self.cache["nvalid"][i])
        bag = self.cache["bags"][i, :max(n, 0)].astype(np.float32) / 255.0  # (n,3,H,W) [0,1]
        for j in range(bag.shape[0]):
            bag[j] = imagenet_norm_chw(bag[j])
        return {"bag": torch.from_numpy(bag),
                "target": torch.from_numpy(self.cache["targets"][i]),
                "tmask": torch.from_numpy(self.cache["tmasks"][i])}
