"""CPU tests for the in-RAM image-bag cache (EXP-006 FT)."""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.constants import N_TARGETS
from src.data.dicom import imagenet_norm_chw
from src.train.cache import CachedBagDataset, build_ram_cache

SIZE = 8


class _RawDS(Dataset):
    """Mimics StudyBagDataset(norm=False): returns [0,1] bags of varying slice count (incl. 0)."""

    def __init__(self, counts):
        self.counts = counts

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, i):
        n = self.counts[i]
        bag = torch.rand(n, 3, SIZE, SIZE) if n else torch.zeros(0, 3, SIZE, SIZE)
        return {"bag": bag, "target": torch.ones(N_TARGETS), "tmask": torch.ones(N_TARGETS)}


def test_cache_build_and_shapes():
    ds = _RawDS([5, 0, 20, 8])  # includes empty + over-nmax
    cache = build_ram_cache(ds, nmax=16, size=SIZE, log=lambda *a, **k: None)
    assert cache["bags"].shape == (4, 16, 3, SIZE, SIZE)
    assert cache["bags"].dtype == np.uint8
    assert list(cache["nvalid"]) == [5, 0, 16, 8]  # 20 clamped to nmax


def test_cached_dataset_roundtrip_and_norm():
    ds = _RawDS([3])
    cache = build_ram_cache(ds, nmax=16, size=SIZE, log=lambda *a, **k: None)
    out = CachedBagDataset(cache)[0]
    assert out["bag"].shape == (3, 3, SIZE, SIZE)
    # values are ImageNet-normalized (not in [0,1]); de-norming a channel lands back near [0,1]
    ch0 = out["bag"][0, 0].numpy()
    denorm = ch0 * 0.229 + 0.485
    assert denorm.min() >= -0.05 and denorm.max() <= 1.05


def test_cached_dataset_empty_bag():
    ds = _RawDS([0])
    cache = build_ram_cache(ds, nmax=16, size=SIZE, log=lambda *a, **k: None)
    out = CachedBagDataset(cache)[0]
    assert out["bag"].shape == (0, 3, SIZE, SIZE)


def test_cache_time_budget_trims():
    ds = _RawDS([4] * 10)
    cache = build_ram_cache(ds, nmax=16, size=SIZE, time_budget_s=0.0, log=lambda *a, **k: None)
    # budget hit almost immediately -> trimmed well below the 10 studies, arrays stay consistent
    assert len(cache["nvalid"]) < 10
    assert cache["bags"].shape[0] == len(cache["nvalid"]) == len(cache["targets"])
