"""E1 model: DINOv2 ViT-S/14 slice encoder (fully trainable) + plane embedding + gated-attention MIL.

Input is the raw uint8 slice bag (B, N, S, S) with plane ids (N,) and a slice mask (B, N); the
grey slice is replicated to 3 channels and ImageNet-normalised on device (DINOv2 convention).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from src.constants import N_TARGETS
from src.models.mil import GatedAttention

DEFAULT_BACKBONE = "vit_small_patch14_dinov2.lvd142m"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class SliceMIL(nn.Module):
    def __init__(self, backbone: str = DEFAULT_BACKBONE, pretrained: bool = True,
                 img_size: int = 224, n_planes: int = 3, n_targets: int = N_TARGETS):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          img_size=img_size, dynamic_img_size=True)
        dim = self.backbone.num_features
        self.plane_emb = nn.Parameter(torch.zeros(n_planes, dim))
        self.attention = GatedAttention(dim)
        self.head = nn.Linear(dim, n_targets)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)

    def head_modules(self) -> list[nn.Module]:
        return [self.attention, self.head]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(M, S, S) float in [0,1] -> (M, D)."""
        img = x.unsqueeze(1).expand(-1, 3, -1, -1)
        return self.backbone((img - self.mean) / self.std)

    def forward(self, x: torch.Tensor, plane_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B,N,S,S) float [0,1]; plane_ids (N,) long; mask (B,N) -> logits (B,12)."""
        b, n = x.shape[:2]
        feats = self.encode(x.flatten(0, 1)).view(b, n, -1) + self.plane_emb[plane_ids]
        return self.head(self.attention(feats, mask))
