"""E1 dataset over the uint8 slice cache + on-device augmentation.

Train: per plane a random sorted subset of k_train of the K cached slices (torch RNG, so each
DataLoader worker is seeded by torch). Eval: all K slices. Missing planes are masked out.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

N_PLANES = 3


class CacheDataset(Dataset):
    def __init__(self, cache_path: str, rows: np.ndarray, plane_mask: np.ndarray,
                 y: np.ndarray, w: np.ndarray, ref: np.ndarray, rmask: np.ndarray,
                 k_train: int | None = None):
        self.cache_path = cache_path
        self.rows, self.plane_mask = rows, plane_mask
        self.y, self.w, self.ref, self.rmask = y, w, ref, rmask
        self.k_train = k_train
        self._arr = None  # opened lazily so each worker gets its own memmap handle

    def __len__(self) -> int:
        return len(self.rows)

    def _vol(self, i: int) -> np.ndarray:
        if self._arr is None:
            self._arr = np.load(self.cache_path, mmap_mode="r")
        return self._arr[self.rows[i]]  # (3,K,S,S)

    def __getitem__(self, i: int) -> dict:
        vol = self._vol(i)
        k_all = vol.shape[1]
        if self.k_train and self.k_train < k_all:
            picks = [torch.randperm(k_all)[: self.k_train].sort().values.numpy() for _ in range(N_PLANES)]
            vol = np.stack([vol[p, picks[p]] for p in range(N_PLANES)])
        k = vol.shape[1]
        x = torch.from_numpy(np.array(vol.reshape(N_PLANES * k, *vol.shape[2:])))  # copy: memmap is read-only
        smask = torch.from_numpy(np.repeat(self.plane_mask[i], k).astype(np.float32))
        return {"x": x, "smask": smask, "idx": i,
                "y": torch.from_numpy(self.y[i]), "w": torch.from_numpy(self.w[i]),
                "ref": torch.from_numpy(self.ref[i]), "rmask": torch.from_numpy(self.rmask[i])}


def plane_ids(k: int, device=None) -> torch.Tensor:
    return torch.arange(N_PLANES, device=device).repeat_interleave(k)


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """x (M,S,S) float [0,1] -> per-image random affine + intensity jitter. No flips (laterality)."""
    m, dev = x.shape[0], x.device
    u = lambda lo, hi: torch.empty(m, device=dev).uniform_(lo, hi)  # noqa: E731
    scale, ang = u(0.87, 1.15), u(-10, 10) * math.pi / 180
    tx, ty = u(-0.08, 0.08), u(-0.08, 0.08)
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (m, 1, *x.shape[1:]), align_corners=False)
    x = F.grid_sample(x.unsqueeze(1), grid, padding_mode="zeros", align_corners=False).squeeze(1)
    gamma, contrast, bright = u(0.85, 1.15), u(0.85, 1.15), u(-0.08, 0.08)
    x = x.clamp(0, 1).pow(gamma[:, None, None]) * contrast[:, None, None] + bright[:, None, None]
    return x.clamp(0, 1)
