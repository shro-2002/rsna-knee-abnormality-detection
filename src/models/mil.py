"""Study-level gated-attention MIL over a bag of slices (Ilse et al. 2018).

A study is a padded bag of N slice images (3xHxW) with a validity mask. A pretrained 2D backbone
(timm) embeds each slice; gated attention pools the bag into one vector; a linear head emits 12
target logits.

Training uses a frozen backbone: features are extracted ONCE (MILModel.extract) and cached, then
only the MIL head (attention + linear) is trained on cached features (MILHead). The saved full
MILModel.state_dict() bundles the backbone weights so offline inference needs no internet.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import N_TARGETS


class GatedAttention(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)  # (B,N)
        a = a.masked_fill(mask == 0, float("-inf"))
        a = torch.softmax(a, dim=1)
        a = torch.nan_to_num(a)
        return torch.bmm(a.unsqueeze(1), h).squeeze(1)  # (B, D)


class MILHead(nn.Module):
    """Operates on precomputed slice features: (B,N,D) + mask -> (B,12) logits."""

    def __init__(self, dim: int, n_targets: int = N_TARGETS):
        super().__init__()
        self.attention = GatedAttention(dim)
        self.head = nn.Linear(dim, n_targets)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.attention(feats, mask))


class MILHeadPerFinding(nn.Module):
    """Per-finding MIL head: a separate gated attention + linear for each target, so each pathology
    pools the slices that show IT (vs one shared attention). Operates on cached features (B,N,D)."""

    def __init__(self, dim: int, n_targets: int = N_TARGETS):
        super().__init__()
        self.attentions = nn.ModuleList([GatedAttention(dim) for _ in range(n_targets)])
        self.heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(n_targets)])

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = [head(att(feats, mask)) for att, head in zip(self.attentions, self.heads)]
        return torch.cat(logits, dim=1)  # (B, n_targets)


class MILModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True,
                 n_targets: int = N_TARGETS, img_size: int | None = None):
        super().__init__()
        kwargs = {"pretrained": pretrained, "num_classes": 0}
        # ViT/DINOv2 backbones have a fixed native resolution (518 for dinov2); pass img_size
        # + dynamic_img_size so a smaller square (e.g. 224=14*16, 252=14*18) is accepted via
        # interpolated position embeddings. CNNs (efficientnet) ignore img_size and pool natively.
        is_vit = "vit" in backbone_name or "dinov2" in backbone_name
        if is_vit:
            if img_size is not None:
                kwargs.update(img_size=img_size, dynamic_img_size=True)
        else:
            kwargs["global_pool"] = "avg"
        self.backbone = timm.create_model(backbone_name, **kwargs)
        self.dim = self.backbone.num_features
        self.mil = MILHead(self.dim, n_targets)

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """(B,N,3,H,W) -> (B,N,D) frozen-backbone features."""
        b, n = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).view(b, n, -1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1)).view(b, n, -1)
        return self.mil(feats, mask)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE over observed (mask=1) target cells only. targets are soft in [0,1]."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none",
                                              pos_weight=pos_weight)
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp_min(1.0)
