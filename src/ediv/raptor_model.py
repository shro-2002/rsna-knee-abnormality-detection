"""Own re-implementation of the Raptor classifier module (CoAtNet backbone + per-finding attention pooling)
so that the public CC0 checkpoints (dreaddevelopment/raptor-knee-*) load with strict=True. Parameter names
(backbone., norm., att.0/att.3, clsW, clsb) must match the checkpoint layout."""
from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class RaptorNet(nn.Module):
    def __init__(self, arch: str, n_out: int = 12, img_size: int | None = None):
        super().__init__()
        kw = {"img_size": img_size} if img_size else {}   # nartaa arm: backbone built for its 320 px inference input
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool="avg", **kw)
        dim = self.backbone.num_features
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, n_out))
        self.clsW = nn.Parameter(torch.zeros(n_out, dim))
        self.clsb = nn.Parameter(torch.zeros(n_out))

    def encode(self, x: torch.Tensor, chunk: int = 26) -> torch.Tensor:   # x (K,3,R,R) -> (K,D)
        return torch.cat([self.backbone(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], 0)

    def head(self, feats: torch.Tensor) -> torch.Tensor:                   # (K,D) -> (12,) logits
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=0)                              # attention over windows, per finding
        return (torch.einsum("kn,kf->nf", a, h) * self.clsW).sum(-1) + self.clsb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def load_raptor(path: str, device: str, img_size: int | None = None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert list(ck["lab"]) == LABELS, "target order mismatch"
    net = RaptorNet(ck["arch"], img_size=img_size)
    net.load_state_dict(ck["model"], strict=True)
    return net.eval().to(device), int(ck["res"])


def make_windows(stack_u8: torch.Tensor, centres, res: int) -> torch.Tensor:
    """stack (M,S,S) uint8 on device -> normalised float windows (K,3,res,res): neighbouring slices as RGB."""
    idx = torch.tensor([[c - 1, c, c + 1] for c in centres], device=stack_u8.device)
    x = stack_u8[idx].float() / 255.0
    if x.shape[-1] != res:
        x = torch.nn.functional.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    mean = torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _sigmoid_probs(net: RaptorNet, x: torch.Tensor, device: str) -> np.ndarray:
    if str(device).startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(x).float()
    else:
        logits = net(x).float()
    assert torch.isfinite(logits).all(), "non-finite logits"
    return torch.sigmoid(logits).cpu().numpy()


@torch.no_grad()
def predict_study_mirror_crop(net: RaptorNet, stack: np.ndarray, centres, crop: int, slot_sizes, sagittal_slots,
                              device: str) -> np.ndarray:
    """E-OAI-1 nartaa arm: centre-crop the stored windows to `crop` px (no resize) and average plain + anatomical-mirror
    probabilities. Mirror = reverse the 3 slice-channels for windows centred in a sagittal slot, flip width otherwise
    (semantics of nartaa's public notebook rsna-knee-0949-anatomical-mirror; implementation ours)."""
    st = torch.from_numpy(stack).to(device)
    cs = np.asarray(centres, dtype=np.int64)
    tri = st[torch.from_numpy(np.stack([cs - 1, cs, cs + 1], 1)).to(device)]          # (K,3,S,S) uint8
    lo = (tri.shape[-1] - crop) // 2
    tri = tri[..., lo:lo + crop, lo:lo + crop]
    bounds = np.cumsum([0] + list(slot_sizes))
    slot_of_centre = np.searchsorted(bounds, cs, side="right") - 1
    sag = torch.from_numpy(np.isin(slot_of_centre, sagittal_slots)).to(device).view(-1, 1, 1, 1)
    mirrored = torch.where(sag, tri.flip(1), tri.flip(-1))
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    norm = lambda t: (t.float() / 255.0 - mean) / std
    return 0.5 * (_sigmoid_probs(net, norm(tri), device) + _sigmoid_probs(net, norm(mirrored), device))


@torch.no_grad()
def predict_study(net: RaptorNet, stack: np.ndarray, mask: np.ndarray, centres, res: int, device: str) -> np.ndarray:
    x = make_windows(torch.from_numpy(stack).to(device), centres, res)
    if str(device).startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.float16):
            return torch.sigmoid(net(x).float()).cpu().numpy()
    return torch.sigmoid(net(x).float()).cpu().numpy()
