"""Study-bag dataset: reads a study's series DICOMs -> a padded bag of 3ch slices + targets.

Reads DICOMs on access (I/O heavy). For the smoke run this is fine; the full run should cache
preprocessed arrays to .npy first (planned optimization).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.constants import ID_COL, TARGETS
from src.data.dicom import (DecodeError, imagenet_norm_chw, series_to_array, series_to_windows,
                            to_3ch_imagenet)


class StudyBagDataset(Dataset):
    def __init__(self, label_df, mask_df, series_df, data_root, split_dir,
                 k_slices=6, size=224, max_series=6, max_slices=18, window=True, norm=True):
        self.labels = label_df.reset_index(drop=True)
        self.masks = mask_df.set_index(ID_COL)
        self.series = series_df
        self.root = data_root
        self.split_dir = split_dir
        self.k_slices = k_slices
        self.size = size
        self.max_series = max_series
        self.max_slices = max_slices
        self.window = window  # True = true-2.5D adjacent-slice RGB; False = fake-2.5D (EXP-004/005 repro)
        self.norm = norm      # True = ImageNet-normalize; False = raw [0,1] (for the FT image cache)

    def __len__(self):
        return len(self.labels)

    def _study_bag(self, sid: str) -> np.ndarray:
        rows = self.series[self.series[ID_COL] == sid]["SeriesInstanceUID"].tolist()[: self.max_series]
        chans = []
        for se in rows:
            sdir = os.path.join(self.root, self.split_dir, sid, se)
            if not os.path.isdir(sdir):
                continue
            try:
                if self.window:
                    arr = series_to_windows(sdir, self.k_slices, self.size)  # (k,3,H,W) true 2.5D, [0,1]
                    for j in range(arr.shape[0]):
                        chans.append(imagenet_norm_chw(arr[j]) if self.norm else arr[j])  # (3,H,W)
                    continue
                arr = series_to_array(sdir, self.k_slices, self.size)  # (k,H,W) fake 2.5D
            except DecodeError:
                continue
            for j in range(arr.shape[0]):
                chans.append(to_3ch_imagenet(arr[j]))  # (3,H,W)
        if not chans:
            return np.zeros((0, 3, self.size, self.size), dtype=np.float32)
        bag = np.stack(chans)
        if bag.shape[0] > self.max_slices:
            idx = np.linspace(0, bag.shape[0] - 1, self.max_slices).round().astype(int)
            bag = bag[idx]
        return bag

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        sid = row[ID_COL]
        bag = self._study_bag(sid)
        target = row[TARGETS].to_numpy(dtype=np.float32)
        tmask = self.masks.loc[sid, TARGETS].to_numpy(dtype=np.float32)
        return {
            "bag": torch.from_numpy(bag),             # (N,3,H,W)
            "target": torch.from_numpy(target),       # (12,)
            "tmask": torch.from_numpy(tmask),         # (12,)
        }


def collate_bags(batch):
    n_max = max(b["bag"].shape[0] for b in batch)
    n_max = max(n_max, 1)
    size = batch[0]["bag"].shape[-1] if batch[0]["bag"].numel() else 224
    B = len(batch)
    bags = torch.zeros(B, n_max, 3, size, size)
    smask = torch.zeros(B, n_max)
    targets = torch.stack([b["target"] for b in batch])
    tmasks = torch.stack([b["tmask"] for b in batch])
    for i, b in enumerate(batch):
        n = b["bag"].shape[0]
        if n:
            bags[i, :n] = b["bag"]
            smask[i, :n] = 1.0
    return {"bag": bags, "smask": smask, "target": targets, "tmask": tmasks}
