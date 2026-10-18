"""Plumbing tests for MIL backbones (CPU, random-init — no checkpoint download).

Proves the EXP-005 DINOv2 feature path is wired correctly end to end: frozen-backbone
feature extraction over a padded slice bag -> gated-attention MIL head -> masked BCE ->
backprop. Random weights only; the real pretrained DINOv2 download is an approval-gated
step that runs on Kaggle, not here.
"""
import torch

from src.constants import N_TARGETS
from src.models.mil import MILModel, masked_bce


def _bag(b=2, n=5, res=224):
    x = torch.randn(b, n, 3, res, res)
    smask = torch.ones(b, n)
    smask[0, 3:] = 0  # ragged bag: study 0 has 3 valid slices
    targets = torch.randint(0, 2, (b, N_TARGETS)).float()
    tmask = torch.ones(b, N_TARGETS)
    tmask[1, 0] = 0  # one abstained target cell
    return x, smask, targets, tmask


def _run_backbone(name, res, img_size):
    model = MILModel(backbone_name=name, pretrained=False, img_size=img_size)
    x, smask, targets, tmask = _bag(res=res)
    feats = model.extract(x)                      # frozen, no grad
    assert feats.shape == (2, 5, model.dim)
    logits = model.mil(feats.float(), smask)      # head only
    assert logits.shape == (2, N_TARGETS)
    loss = masked_bce(logits, targets, tmask)
    assert torch.isfinite(loss)
    loss.backward()                               # head params get grads
    grads = [p.grad is not None for p in model.mil.parameters()]
    assert all(grads)


def test_dinov2_vits_path():
    _run_backbone("vit_small_patch14_dinov2", res=224, img_size=224)


def test_efficientnet_b0_still_works():
    _run_backbone("efficientnet_b0", res=224, img_size=None)
