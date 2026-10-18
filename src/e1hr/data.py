"""E1-HR dataset: dual view per study = knee-crop 2.5D windows + E1 full-FOV slices.

View ids: 0-2 crop (Sagittal, Coronal, Axial), 3-5 full FOV (same planes). A crop window is
(slice c-1, c, c+1) of the 24 cached crop slices as RGB (true 2.5D, centres 1..22). Full-FOV
slices are grey replicated to 3 channels exactly as E1 saw them.
Train: CROP_TRAIN random windows + FULL_TRAIN random full slices per plane; eval: all of both.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

N_PLANES = 3
CROP_TRAIN, FULL_TRAIN = 8, 4


def _pick(n: int, k: int | None) -> np.ndarray:
    if k is None or k >= n:
        return np.arange(n)
    return torch.randperm(n)[:k].sort().values.numpy()


def crop_windows(vol: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """vol (K,S,S) -> (len(centres), 3, S, S) adjacent-slice windows."""
    return np.stack([vol[centres - 1], vol[centres], vol[centres + 1]], axis=1)


class DualViewDataset(Dataset):
    def __init__(self, full_path: str, hr_path: str, full_rows: np.ndarray, hr_rows: np.ndarray,
                 plane_mask: np.ndarray, y: np.ndarray, w: np.ndarray, ref: np.ndarray,
                 rmask: np.ndarray, train: bool):
        self.full_path, self.hr_path = full_path, hr_path
        self.full_rows, self.hr_rows, self.plane_mask = full_rows, hr_rows, plane_mask
        self.y, self.w, self.ref, self.rmask = y, w, ref, rmask
        self.train = train
        self._full = self._hr = None  # lazy per-worker memmaps

    def __len__(self) -> int:
        return len(self.hr_rows)

    def _arrays(self):
        if self._full is None:
            self._full = np.load(self.full_path, mmap_mode="r")
            self._hr = np.load(self.hr_path, mmap_mode="r")
        return self._full, self._hr

    def __getitem__(self, i: int) -> dict:
        full_all, hr_all = self._arrays()
        hr, full = hr_all[self.hr_rows[i]], full_all[self.full_rows[i]]  # (3,24,S,S), (3,16,S,S)
        images, views, valid = [], [], []
        for p in range(N_PLANES):
            centres = 1 + _pick(hr.shape[1] - 2, CROP_TRAIN if self.train else None)
            images.append(crop_windows(np.asarray(hr[p]), centres))
            views += [p] * len(centres)
            valid += [self.plane_mask[i, p]] * len(centres)
        for p in range(N_PLANES):
            picks = _pick(full.shape[1], FULL_TRAIN if self.train else None)
            images.append(np.repeat(np.asarray(full[p, picks])[:, None], 3, axis=1))
            views += [N_PLANES + p] * len(picks)
            valid += [self.plane_mask[i, p]] * len(picks)
        return {"x": torch.from_numpy(np.concatenate(images)), "vid": torch.tensor(views),
                "smask": torch.tensor(valid, dtype=torch.float32), "idx": i,
                "y": torch.from_numpy(self.y[i]), "w": torch.from_numpy(self.w[i]),
                "ref": torch.from_numpy(self.ref[i]), "rmask": torch.from_numpy(self.rmask[i])}
