"""End-to-end backbone fine-tuning for the MIL model (EXP-006).

Unlike the frozen-feature path (src/train/features.py), gradients flow through the backbone, so
images — not cached features — drive training. Three controlled modes over the SAME MIL head:
  - "partial": unfreeze the last N transformer blocks (+ final norm); backbone frozen below.
  - "lora":    inject LoRA adapters into the last N blocks' attention (qkv/proj); base weights frozen.
  - "full":    all backbone params trainable (only if VRAM/runtime permit).
Differential LRs (low on backbone/adapters, higher on head), AMP, gradient accumulation, and
gradient checkpointing keep a 22M-param ViT trainable on a 16GB T4. Weak-label noise + class
imbalance handled via soft targets + per-target pos_weight in masked BCE.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.mil import masked_bce
from src.validate.metric import bootstrap_macro_auc, per_target_auc


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear: y = W0 x + (B A) x * (alpha/r)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.scaling = alpha / r
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)  # start as identity (adapter contributes 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.drop(x))) * self.scaling


def _vit_blocks(model):
    blocks = getattr(model.backbone, "blocks", None)
    if blocks is None:
        raise ValueError("backbone has no .blocks (not a ViT); partial/lora modes need a ViT")
    return blocks


def apply_lora(model, *, n_blocks: int = 4, r: int = 8, alpha: int = 16,
               dropout: float = 0.0, targets=("qkv", "proj")) -> int:
    """Inject LoRA into the last n_blocks' attention. Returns #adapters added."""
    blocks = _vit_blocks(model)
    added = 0
    for blk in list(blocks)[-n_blocks:]:
        attn = blk.attn
        for name in targets:
            lin = getattr(attn, name, None)
            if isinstance(lin, nn.Linear):
                setattr(attn, name, LoRALinear(lin, r=r, alpha=alpha, dropout=dropout))
                added += 1
    return added


def configure_finetune(model, *, mode: str = "partial", n_blocks: int = 4,
                       lora_r: int = 8, lora_alpha: int = 16, grad_checkpoint: bool = True):
    """Set requires_grad per mode and (optionally) enable gradient checkpointing.

    Returns a dict summary (mode, trainable params in backbone vs head, total trainable).
    """
    # start: freeze backbone, head always trainable
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for p in model.mil.parameters():
        p.requires_grad_(True)

    if mode == "frozen":
        pass
    elif mode == "full":
        for p in model.backbone.parameters():
            p.requires_grad_(True)
    elif mode == "partial":
        blocks = _vit_blocks(model)
        for blk in list(blocks)[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        if hasattr(model.backbone, "norm"):
            for p in model.backbone.norm.parameters():
                p.requires_grad_(True)
    elif mode == "lora":
        apply_lora(model, n_blocks=n_blocks, r=lora_r, alpha=lora_alpha)
        # LoRALinear already froze base weights; only lora_a/lora_b (+ head) train
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if grad_checkpoint and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(enable=True)

    bb = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    hd = sum(p.numel() for p in model.mil.parameters() if p.requires_grad)
    return {"mode": mode, "backbone_trainable": bb, "head_trainable": hd, "total_trainable": bb + hd}


def param_groups(model, *, lr_backbone: float = 2e-5, lr_head: float = 1e-3):
    """Differential-LR param groups: trainable backbone/adapter params low, head params high."""
    head_ids = {id(p) for p in model.mil.parameters()}
    bb = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    hd = [p for p in model.mil.parameters() if p.requires_grad]
    groups = []
    if bb:
        groups.append({"params": bb, "lr": lr_backbone})
    if hd:
        groups.append({"params": hd, "lr": lr_head})
    return groups


def train_finetune(model, train_ds, *, groups, collate, device, epochs=3, batch_size=2,
                   accum_steps=4, pos_weight=None, amp=True, num_workers=2, grad_clip=1.0,
                   ckpt_path=None, time_budget_s=None, seed=42, log=print):
    """End-to-end fine-tune over image bags with AMP + gradient accumulation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device).train()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, collate_fn=collate, drop_last=False)
    opt = torch.optim.AdamW(groups)
    use_cuda = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and use_cuda))
    pw = pos_weight.to(device) if pos_weight is not None else None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        total, nb = 0.0, 0
        for step, batch in enumerate(loader):
            if time_budget_s is not None and (time.time() - t0) > time_budget_s:
                log(f"  TIME BUDGET hit at epoch {ep+1} step {step}; stopping")
                break
            x = batch["bag"].to(device)
            m = batch["smask"].to(device)
            y = batch["target"].to(device)
            tm = batch["tmask"].to(device)
            with torch.autocast(device_type="cuda" if use_cuda else "cpu", enabled=(amp and use_cuda)):
                loss = masked_bce(model(x, m), y, tm, pos_weight=pw) / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                if grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in groups for p in g["params"]], grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += float(loss.item()) * accum_steps
            nb += 1
        log(f"  epoch {ep+1}/{epochs} loss={total/max(nb,1):.4f} time={time.time()-t0:.0f}s")
        if ckpt_path:
            torch.save(model.state_dict(), ckpt_path)
    return model


@torch.no_grad()
def evaluate_images(model, gold_ds, y_true, *, collate, device, batch_size=2, num_workers=2,
                    log=print):
    model.to(device).eval()
    loader = DataLoader(gold_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    out = []
    for batch in loader:
        out.append(torch.sigmoid(model(batch["bag"].to(device), batch["smask"].to(device))).cpu().numpy())
    preds = np.concatenate(out, axis=0)
    aucs = per_target_auc(y_true, preds)
    res = bootstrap_macro_auc(y_true, preds, n_boot=2000, seed=0)
    log(f"gold-58 MACRO AUC = {res['point']:.3f} (95% CI {res['ci_low']:.3f}-{res['ci_high']:.3f})")
    return {"preds": preds, "per_target": aucs, **res}
