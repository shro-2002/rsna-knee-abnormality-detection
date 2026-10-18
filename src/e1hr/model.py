"""E1-HR model: E1's SliceMIL (same ViT-S/14 backbone, gated MIL head) with 6 view-plane
embeddings and real 3-channel (2.5D) input instead of a replicated grey slice."""
from __future__ import annotations

import torch

from src.e1.model import DEFAULT_BACKBONE, SliceMIL

N_VIEWS = 6


class DualViewMIL(SliceMIL):
    def __init__(self, backbone: str = DEFAULT_BACKBONE, pretrained: bool = True, img_size: int = 224):
        super().__init__(backbone, pretrained, img_size, n_planes=N_VIEWS)

    def forward(self, x: torch.Tensor, view_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B,N,3,S,S) float [0,1]; view_ids (B,N) long; mask (B,N) -> logits (B,12)."""
        b, n = x.shape[:2]
        img = (x.flatten(0, 1) - self.mean) / self.std
        feats = self.backbone(img).view(b, n, -1) + self.plane_emb[view_ids]
        return self.head(self.attention(feats, mask))
