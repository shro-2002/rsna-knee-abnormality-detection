"""CPU tests for EXP-006 fine-tune infrastructure (random-init ViT, 56px for speed).

Verifies that each mode trains exactly the intended parameters, differential param groups are
built, and the AMP+accumulation loop runs end to end on image bags.
"""
import torch
from torch.utils.data import Dataset

from src.constants import N_TARGETS
from src.models.mil import MILModel
from src.train.dataset import collate_bags
from src.train.finetune import (configure_finetune, evaluate_images, param_groups,
                                 train_finetune)

RES = 56  # 14*4, valid for dinov2 patch14


def _model():
    return MILModel("vit_small_patch14_dinov2", pretrained=False, img_size=RES)


class _FakeBags(Dataset):
    def __init__(self, n=4, slices=3):
        self.n, self.slices = n, slices

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"bag": torch.randn(self.slices, 3, RES, RES),
                "target": torch.randint(0, 2, (N_TARGETS,)).float(),
                "tmask": torch.ones(N_TARGETS)}


def _grad_after_backward(model):
    x = torch.randn(2, 3, 3, RES, RES)
    mask = torch.ones(2, 3)
    logits = model(x, mask)
    loss = logits.sum()
    loss.backward()


def test_partial_unfreezes_only_last_blocks():
    model = _model()
    summary = configure_finetune(model, mode="partial", n_blocks=2, grad_checkpoint=False)
    assert summary["backbone_trainable"] > 0 and summary["head_trainable"] > 0
    blocks = model.backbone.blocks
    assert all(p.requires_grad for p in blocks[-1].parameters())
    assert not any(p.requires_grad for p in blocks[0].parameters())  # early block stays frozen


def test_lora_trains_only_adapters():
    model = _model()
    configure_finetune(model, mode="lora", n_blocks=2, lora_r=4, grad_checkpoint=False)
    attn = model.backbone.blocks[-1].attn
    assert hasattr(attn.qkv, "lora_a")              # adapter injected
    assert not attn.qkv.base.weight.requires_grad   # base frozen
    assert attn.qkv.lora_a.weight.requires_grad     # adapter trains
    # a non-adapted early block has no adapter and is frozen
    assert not any(p.requires_grad for p in model.backbone.blocks[0].parameters())


def test_full_trains_all_backbone():
    model = _model()
    s = configure_finetune(model, mode="full", grad_checkpoint=False)
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert s["backbone_trainable"] > s["head_trainable"]


def test_param_groups_differential_lr():
    model = _model()
    configure_finetune(model, mode="partial", n_blocks=2, grad_checkpoint=False)
    groups = param_groups(model, lr_backbone=2e-5, lr_head=1e-3)
    assert len(groups) == 2
    assert {g["lr"] for g in groups} == {2e-5, 1e-3}


def test_finetune_loop_runs_with_accumulation():
    model = _model()
    configure_finetune(model, mode="lora", n_blocks=2, lora_r=4, grad_checkpoint=False)
    groups = param_groups(model)
    ds = _FakeBags(n=4, slices=3)
    model = train_finetune(model, ds, groups=groups, collate=collate_bags, device="cpu",
                           epochs=1, batch_size=2, accum_steps=2, amp=False, num_workers=0,
                           log=lambda *a, **k: None)
    gold = _FakeBags(n=3, slices=3)
    y = torch.stack([gold[i]["target"] for i in range(len(gold))]).numpy()
    res = evaluate_images(model, gold, y, collate=collate_bags, device="cpu",
                          batch_size=2, num_workers=0, log=lambda *a, **k: None)
    assert "point" in res and 0.0 <= res["point"] <= 1.0
