"""Layer-wise LR decay param groups + warmup-cosine schedule for the E1 ViT."""
from __future__ import annotations

import math
import re

import torch


def _layer_id(name: str, n_blocks: int) -> int:
    """0 = embeddings, 1..n_blocks = transformer blocks, n_blocks+1 = final norm."""
    if name.startswith(("cls_token", "pos_embed", "reg_token", "mask_token", "patch_embed")):
        return 0
    m = re.match(r"blocks\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    return n_blocks + 1


def llrd_groups(model, lr_backbone: float, lr_head: float, decay: float, wd: float) -> list[dict]:
    """Top backbone layer gets lr_backbone; each layer below is multiplied by `decay`.
    Head (attention MIL, linear, plane embedding) gets lr_head. No weight decay on 1-D params."""
    n_blocks = len(model.backbone.blocks)
    groups: dict[tuple, dict] = {}

    def add(param, lr, name):
        no_wd = param.ndim <= 1 or name.endswith(("pos_embed", "cls_token", "reg_token", "plane_emb"))
        key = (lr, no_wd)
        g = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": 0.0 if no_wd else wd})
        g["params"].append(param)

    for name, p in model.backbone.named_parameters():
        if p.requires_grad:
            depth = (n_blocks + 1) - _layer_id(name, n_blocks)
            add(p, lr_backbone * decay ** depth, name)
    for name, p in model.named_parameters():
        if p.requires_grad and not name.startswith("backbone."):
            add(p, lr_head, name)
    for g in groups.values():
        g["base_lr"] = g["lr"]
    return list(groups.values())


def warmup_cosine(opt: torch.optim.Optimizer, total_steps: int, warmup_frac: float):
    """Linear warmup then cosine to 0. `sched.horizon["total"]` may be shortened once mid-run
    (budget-aware horizon) so the cosine still reaches ~0 before a wall-clock cap."""
    horizon = {"total": total_steps, "warm": max(1, int(total_steps * warmup_frac))}

    def factor(step: int) -> float:
        warm, total = horizon["warm"], horizon["total"]
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    sched.horizon = horizon
    return sched
