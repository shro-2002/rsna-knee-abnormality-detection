"""E1 training: fold pilot (OOF vs frozen reference + gates) or all-data refit.

Single process (CPU smoke) or DDP via torchrun. Rank 0 evaluates and writes artifacts:
  metrics.jsonl (per epoch), oof_preds.csv, gold_preds.csv, model.pt, config.json, ABORT (if gated).
Gates (research/19 §3, fixed before any run):
  epoch-`gate_epoch`: weighted excess BCE (BCE minus soft-label entropy, i.e. the learnable part)
  must fall to <= gate_excess x its value over the first 50 steps, and OOF macro >= gate_oof.
Final gates (OOF >= 0.880, gold-58 >= 0.890, rank corr vs E-DIV <= 0.88, runtime) are evaluated
from these artifacts by src/e1/gates.py, not here.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import MISSING, asdict, dataclass, field, fields

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from src.constants import ID_COL, TARGETS
from src.e1.data import CacheDataset, gpu_augment, plane_ids
from src.e1.evaluate import (per_target_vs_reference, predict, save_preds, score_vs_reference,
                             soft_target_loss)
from src.e1.labels import find_source_files, load_sources, reference_labels, train_targets
from src.e1.model import DEFAULT_BACKBONE, SliceMIL
from src.e1.optim import llrd_groups, warmup_cosine
from src.validate.metric import bootstrap_macro_auc
from src.validate.split import group_kfold_indices, load_labels

FOLD_SEED = 0
N_FOLDS = 5
EXCESS_WINDOW = 50


@dataclass
class Config:
    cache_dir: str
    data_root: str
    out: str
    label_roots: list[str] = field(default_factory=list)
    fold: int = 0                 # -1 = all non-gold studies (refit, no OOF)
    epochs: int = 15
    batch: int = 4                # studies per GPU
    k_train: int = 8              # slices per plane per study per step
    lr_backbone: float = 5e-5
    lr_head: float = 1e-3
    llrd: float = 0.75
    wd: float = 0.05
    warmup: float = 0.05
    wall_cap_min: float = 45.0
    backbone: str = DEFAULT_BACKBONE
    pretrained: bool = True
    img_size: int = 224
    grad_ckpt: bool = False
    workers: int = 4
    gate_epoch: int = 3
    gate_oof: float = 0.83
    gate_excess: float = 0.70
    max_steps: int = 0            # >0 caps steps per epoch (smoke only)
    eval_every: int = 3           # OOF eval cadence (also epoch 1, gate epoch, last, wall cap)
    fit_eval_n: int = 300         # fixed in-sample train subset scored as an overfitting indicator
    seed: int = 42


def _setup_ddp() -> tuple[int, int, str]:
    if "WORLD_SIZE" not in os.environ:
        return 0, 1, "cuda" if torch.cuda.is_available() else "cpu"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        return rank, world, "cuda"
    return rank, world, "cpu"


def prepare(cfg: Config) -> dict:
    """Datasets for train / val (OOF) / gold. Labels: mean of CC0 sources; reference frozen."""
    index = pd.read_csv(os.path.join(cfg.cache_dir, "e1_index.csv"))
    row_of = dict(zip(index[ID_COL], index["row"]))
    pmask_all = index[[c for c in index.columns if c.startswith("has_")]].to_numpy(np.float32)
    lab = load_labels(os.path.join(cfg.data_root, "train.csv"))
    sources = load_sources(find_source_files(cfg.label_roots))
    weak = lab.weak[lab.weak[ID_COL].isin(row_of)].reset_index(drop=True)
    folds = list(group_kfold_indices(weak, N_FOLDS, seed=FOLD_SEED))
    tr_idx, va_idx = (np.arange(len(weak)), np.array([], int)) if cfg.fold < 0 else folds[cfg.fold]
    cache = os.path.join(cfg.cache_dir, "e1_cache.npy")

    def make(sids, k_train, gold_y=None):
        rows = np.array([row_of[s] for s in sids], dtype=np.int64)
        y, w = train_targets(sources, sids)
        if gold_y is None:
            ref, rmask = reference_labels(sources, sids)
        else:
            ref, rmask = gold_y.astype(np.float32), np.ones_like(gold_y, dtype=np.float32)
        return CacheDataset(cache, rows, pmask_all[rows], y, w, ref, rmask, k_train), sids

    tr_sids = weak[ID_COL].iloc[tr_idx].tolist()
    tr_sids = [s for s in tr_sids if pmask_all[row_of[s]].any()]  # no-image studies cannot train
    gold = lab.gold[lab.gold[ID_COL].isin(row_of)]
    rng = np.random.default_rng(FOLD_SEED)
    fit_sids = sorted(rng.choice(tr_sids, size=min(cfg.fit_eval_n, len(tr_sids)), replace=False).tolist())
    return {"train": make(tr_sids, cfg.k_train), "fit": make(fit_sids, None),
            "val": make(weak[ID_COL].iloc[va_idx].tolist(), None),
            "gold": make(gold[ID_COL].tolist(), None, gold[TARGETS].to_numpy(float)),
            "sources": sorted(sources)}


def soft_entropy(y: torch.Tensor) -> torch.Tensor:
    y = y.clamp(1e-6, 1 - 1e-6)
    return -(y * y.log() + (1 - y) * (1 - y).log())


def train_step(model, batch, opt, sched, scaler, device, amp) -> tuple[float, float]:
    b, n = batch["x"].shape[:2]
    x = batch["x"].to(device, non_blocking=True).float().flatten(0, 1) / 255.0
    x = gpu_augment(x).view(b, n, *x.shape[1:])
    y, w = batch["y"].to(device), batch["w"].to(device)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
        logits = model(x, plane_ids(n // 3, device), batch["smask"].to(device))
    bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
    wsum = w.sum().clamp_min(1.0)
    loss = (bce * w).sum() / wsum
    excess = ((bce - soft_entropy(y)) * w).sum() / wsum
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    return float(loss.item()), float(excess.item())


def _any_rank(flag: bool, device: str) -> bool:
    if not dist.is_initialized():
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


def _gate(cfg: Config, rec: dict, first_excess: list[float]) -> str:
    """'' when the epoch-gate passes, else the reason."""
    base = float(np.mean(first_excess)) if first_excess else float("nan")
    if not rec["excess"] <= cfg.gate_excess * base:
        return f"excess {rec['excess']:.4f} > {cfg.gate_excess} x first-{EXCESS_WINDOW}-step {base:.4f}"
    if cfg.fold >= 0 and not rec.get("oof_macro", 0.0) >= cfg.gate_oof:
        return f"oof {rec.get('oof_macro')} < {cfg.gate_oof}"
    return ""


def _log(cfg: Config, rec: dict) -> None:
    print(json.dumps(rec), flush=True)
    with open(os.path.join(cfg.out, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _build(cfg: Config, tr, rank: int, world: int, device: str):
    model = SliceMIL(cfg.backbone, cfg.pretrained, cfg.img_size).to(device)
    if cfg.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    opt = torch.optim.AdamW(llrd_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.llrd, cfg.wd))
    sampler = DistributedSampler(tr, world, rank, shuffle=True, seed=cfg.seed) if world > 1 else None
    loader = DataLoader(tr, batch_size=cfg.batch, shuffle=sampler is None, sampler=sampler,
                        num_workers=cfg.workers, drop_last=True, pin_memory=device == "cuda",
                        persistent_workers=cfg.workers > 0)
    steps = min(len(loader), cfg.max_steps) if cfg.max_steps else len(loader)
    sched = warmup_cosine(opt, steps * cfg.epochs, cfg.warmup)
    net = model
    if world > 1:
        ids = [torch.cuda.current_device()] if device == "cuda" else None
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=ids)
    return model, net, opt, sched, sampler, loader, steps


def _train_epoch(cfg: Config, run: dict, ep: int, first_excess: list[float], t0: float):
    """One epoch of steps. Returns (losses, excess, stop_reason)."""
    losses, excess = [], []
    for step, batch in enumerate(run["loader"]):
        if step >= run["steps"]:
            break
        loss, exc = train_step(run["net"], batch, run["opt"], run["sched"], run["scaler"],
                               run["device"], run["amp"])
        if _any_rank(not np.isfinite(loss), run["device"]):
            return losses, excess, "nan"
        losses.append(loss)
        excess.append(exc)
        if ep == 1 and len(first_excess) < EXCESS_WINDOW:
            first_excess.append(exc)
        if _any_rank(time.time() - t0 > cfg.wall_cap_min * 60, run["device"]):
            return losses, excess, "wall_cap"
    return losses, excess, ""


def _should_eval(cfg: Config, ep: int, planned: int, stop: str) -> bool:
    """Epoch 1 (timing), gate epoch, cadence, final planned epoch, or any early stop."""
    return ep in (1, cfg.gate_epoch, planned) or ep % cfg.eval_every == 0 or bool(stop)


def _end_epoch(cfg: Config, run: dict, data: dict, rec: dict, first_excess: list[float],
               do_eval: bool) -> None:
    """Rank 0 only: OOF + in-sample scoring, preds, checkpoint, epoch gate, metrics log."""
    (val, val_sids), (fit_ds, _) = data["val"], data["fit"]
    if do_eval and len(val):
        t_pred = time.time()
        oof = predict(run["model"], val, run["device"], cfg.batch, cfg.workers, run["amp"])
        rec["pred_s_per_study"] = round((time.time() - t_pred) / len(val), 4)
        rec["oof_macro"] = score_vs_reference(oof, val)
        rec["oof_per_target"] = per_target_vs_reference(oof, val)
        rec["val_loss"] = soft_target_loss(oof, val)
        fit_preds = predict(run["model"], fit_ds, run["device"], cfg.batch, cfg.workers, run["amp"])
        rec["fit_macro"] = score_vs_reference(fit_preds, fit_ds)  # in-sample: overfitting indicator only
        rec["fit_loss"] = soft_target_loss(fit_preds, fit_ds)
        rec["fit_minus_oof"] = rec["fit_macro"] - rec["oof_macro"]
        rec["eval_s"] = round(time.time() - t_pred, 1)
        save_preds(os.path.join(cfg.out, "oof_preds.csv"), val_sids, oof)
    torch.save(run["model"].state_dict(), os.path.join(cfg.out, "model.pt"))
    if rec["epoch"] == cfg.gate_epoch:
        rec["gate"] = _gate(cfg, rec, first_excess) or "pass"
    _log(cfg, rec)


def _planned_epochs(cfg: Config, rec: dict, elapsed_s: float, final_reserve_s: float = 120.0) -> int:
    """Epochs that fit the wall cap, from epoch-1 timing. rec["wall_s"] is stamped BEFORE the
    epoch's eval, so it is pure train time; eval time is amortised over the eval cadence."""
    per_epoch = rec["wall_s"] + rec.get("eval_s", 0.0) / cfg.eval_every
    left = cfg.wall_cap_min * 60 - elapsed_s - final_reserve_s
    return int(max(cfg.gate_epoch, min(cfg.epochs, 1 + left // max(per_epoch, 1e-6))))


def _broadcast_int(value: int, device: str) -> int:
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], device=device)
    dist.broadcast(t, src=0)
    return int(t.item())


def _finish(cfg: Config, run: dict, gold, gold_sids: list[str], summary: dict, gate_reason: str) -> None:
    """Rank 0 only: gold-58 sanity preds, ABORT marker, summary.json."""
    preds = predict(run["model"], gold, run["device"], cfg.batch, cfg.workers, run["amp"])
    save_preds(os.path.join(cfg.out, "gold_preds.csv"), gold_sids, preds)
    summary["gold58"] = bootstrap_macro_auc(gold.ref, preds, n_boot=1000, seed=0) if len(preds) else None
    if summary["stop"] in ("gate", "nan"):
        with open(os.path.join(cfg.out, "ABORT"), "w") as f:
            f.write(gate_reason if summary["stop"] == "gate" else "non-finite training loss")
    with open(os.path.join(cfg.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary), flush=True)


def _write_config(cfg: Config, data: dict, world: int) -> None:
    with open(os.path.join(cfg.out, "config.json"), "w") as f:
        json.dump({**asdict(cfg), "world": world, "n_train": len(data["train"][0]),
                   "n_val": len(data["val"][0]), "n_gold": len(data["gold"][0]),
                   "n_fit_eval": len(data["fit"][0]), "sources": data["sources"],
                   "fold_seed": FOLD_SEED}, f, indent=1)


def fit(cfg: Config) -> dict:
    rank, world, device = _setup_ddp()
    torch.manual_seed(cfg.seed + rank)
    amp = device == "cuda"
    os.makedirs(cfg.out, exist_ok=True)
    data = prepare(cfg)
    model, net, opt, sched, sampler, loader, steps = _build(cfg, data["train"][0], rank, world, device)
    run = {"model": model, "net": net, "opt": opt, "sched": sched, "loader": loader, "steps": steps,
           "scaler": torch.amp.GradScaler(device, enabled=amp), "device": device, "amp": amp}
    if rank == 0:
        _write_config(cfg, data, world)
    t0, first_excess, stop, rec, planned = time.time(), [], "", {}, cfg.epochs
    for ep in range(1, cfg.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        losses, excess, stop = _train_epoch(cfg, run, ep, first_excess, t0)
        rec = {"epoch": ep, "loss": float(np.mean(losses)) if losses else float("nan"),
               "excess": float(np.mean(excess)) if excess else float("nan"),
               "lr_head": opt.param_groups[-1]["lr"], "wall_s": round(time.time() - t0, 1),
               "planned_epochs": planned}
        if rank == 0 and stop != "nan":
            _end_epoch(cfg, run, data, rec, first_excess, _should_eval(cfg, ep, planned, stop))
        if ep == 1:
            planned = _broadcast_int(_planned_epochs(cfg, rec, time.time() - t0) if rank == 0 else 0, device)
            sched.horizon["total"] = planned * steps
        is_gate_failed = rank == 0 and rec.get("gate", "pass") != "pass"
        if _any_rank(is_gate_failed, device):
            stop = "gate"
        if stop or ep >= planned:
            break
    summary = {"stop": stop or "completed", "epochs_run": ep, "planned_epochs": planned,
               "wall_s": round(time.time() - t0, 1)}
    if rank == 0:
        _finish(cfg, run, data["gold"][0], data["gold"][1], summary, rec.get("gate", "gate failed"))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return summary


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    casters = {"int": int, "float": float, "str": str,
               "bool": lambda s: s.lower() in ("1", "true", "yes")}
    for f in fields(Config):
        flag = f"--{f.name.replace('_', '-')}"
        if f.name == "label_roots":
            ap.add_argument(flag, nargs="+", default=[])
        elif f.default is MISSING:
            ap.add_argument(flag, type=casters[f.type], required=True)
        else:
            ap.add_argument(flag, type=casters[f.type], default=f.default)
    return fit(Config(**vars(ap.parse_args(argv))))


if __name__ == "__main__":
    main()
