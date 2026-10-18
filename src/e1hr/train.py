"""E1-HR trainer. Identical protocol to src/e1/train.py (labels, fold-0 split, seed, LLRD, AdamW,
fixed budget-aware cosine horizon, epoch-3 gate, OOF/fit/gold artifacts); only the input geometry
differs (dual-view DualViewDataset + DualViewMIL). E1 code is imported, never modified.

Artifacts (rank 0): config.json, metrics.jsonl (adds gpu_peak_gb), oof_preds.csv, gold_preds.csv,
model.pt, summary.json, ABORT on gate/NaN stop. Final gates: src/e1hr/gates.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import MISSING, asdict, dataclass, fields, replace

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from src.constants import ID_COL, TARGETS
from src.e1.evaluate import per_target_vs_reference, save_preds, score_vs_reference, soft_target_loss
from src.e1.labels import find_source_files, load_sources, reference_labels, train_targets
from src.e1.optim import llrd_groups, warmup_cosine
from src.e1.train import (EXCESS_WINDOW, FOLD_SEED, N_FOLDS, Config, _any_rank, _broadcast_int, _gate,
                          _log, _planned_epochs, _setup_ddp, _should_eval, soft_entropy)
from src.e1hr.data import DualViewDataset
from src.e1hr.model import DualViewMIL
from src.e1hr.fuse import fuse_targets, parse_folds
from src.e1hr.preflight import decide_stop, run_preflight
from src.validate.metric import bootstrap_macro_auc
from src.validate.split import group_kfold_indices, load_labels


@dataclass
class HRConfig(Config):
    hr_cache_dir: str = ""
    epochs: int = 8
    preflight_steps: int = 24     # 0 disables (unit tests only)
    preflight_eval_batches: int = 2
    mem_frac_max: float = 0.80    # preflight stops if peak allocated GPU memory exceeds this share
    exclude_folds: str = ""       # FUSE-NX: outer folds removed from the training pool (e.g. "0" for teachers)
    teacher_oof: str = ""         # FUSE-NX: merged teacher OOF CSV -> fused training targets (empty = report only)
    fixed_horizon: bool = False   # FUSE-NX: cosine horizon = epochs exactly (no budget-aware shortening)


def _rows(cache_dir: str, index_name: str) -> tuple[dict, pd.DataFrame]:
    idx = pd.read_csv(os.path.join(cache_dir, index_name))
    return dict(zip(idx[ID_COL], idx["row"])), idx.set_index(ID_COL)


def prepare_hr(cfg: HRConfig) -> dict:
    full_row, full_idx = _rows(cfg.cache_dir, "e1_index.csv")
    hr_row, hr_idx = _rows(cfg.hr_cache_dir, "e1hr_index.csv")
    planes = [c for c in hr_idx.columns if c.startswith("has_")]
    lab = load_labels(os.path.join(cfg.data_root, "train.csv"))
    sources = load_sources(find_source_files(cfg.label_roots))
    weak = lab.weak[lab.weak[ID_COL].isin(full_row) & lab.weak[ID_COL].isin(hr_row)].reset_index(drop=True)
    if cfg.fold < 0:  # refit: every non-gold study trains, no OOF
        tr_idx, va_idx = np.arange(len(weak)), np.array([], int)
    else:
        tr_idx, va_idx = list(group_kfold_indices(weak, N_FOLDS, seed=FOLD_SEED))[cfg.fold]
        excluded = [v for k, (_, v) in enumerate(group_kfold_indices(weak, N_FOLDS, seed=FOLD_SEED))
                    if k in parse_folds(cfg.exclude_folds) and k != cfg.fold]
        if excluded:
            tr_idx = np.setdiff1d(tr_idx, np.concatenate(excluded))
    paths = (os.path.join(cfg.cache_dir, "e1_cache.npy"), os.path.join(cfg.hr_cache_dir, "e1hr_cache.npy"))

    def make(sids, train, gold_y=None):
        pmask = np.minimum(full_idx.loc[sids, planes].to_numpy(np.float32), hr_idx.loc[sids, planes].to_numpy(np.float32))
        y, w = train_targets(sources, sids)
        if train and teacher is not None:
            y = fuse_targets(y, sids, teacher)
        if gold_y is None:
            ref, rmask = reference_labels(sources, sids)
        else:
            ref, rmask = gold_y.astype(np.float32), np.ones_like(gold_y, dtype=np.float32)
        rows = (np.array([full_row[s] for s in sids]), np.array([hr_row[s] for s in sids]))
        return DualViewDataset(*paths, *rows, pmask, y, w, ref, rmask, train), sids

    tr_sids = [s for s in weak[ID_COL].iloc[tr_idx] if hr_idx.loc[s, planes].any()]
    teacher = pd.read_csv(cfg.teacher_oof).set_index(ID_COL) if cfg.teacher_oof else None
    rng = np.random.default_rng(FOLD_SEED)
    fit_sids = sorted(rng.choice(tr_sids, size=min(cfg.fit_eval_n, len(tr_sids)), replace=False).tolist())
    gold = lab.gold[lab.gold[ID_COL].isin(hr_row) & lab.gold[ID_COL].isin(full_row)]
    return {"train": make(tr_sids, True), "fit": make(fit_sids, False),
            "val": make(weak[ID_COL].iloc[va_idx].tolist(), False),
            "gold": make(gold[ID_COL].tolist(), False, gold[TARGETS].to_numpy(float)),
            "sources": sorted(sources)}


def augment_3ch(x: torch.Tensor) -> torch.Tensor:
    """x (M,3,S,S) [0,1]: E1's per-image affine + intensity jitter, applied identically to all 3
    channels of a window so adjacent slices stay registered. No flips (laterality)."""
    m, dev = x.shape[0], x.device
    u = lambda lo, hi: torch.empty(m, device=dev).uniform_(lo, hi)  # noqa: E731
    scale, ang = u(0.87, 1.15), u(-10, 10) * math.pi / 180
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, u(-0.08, 0.08)], 1), torch.stack([sin, cos, u(-0.08, 0.08)], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    gamma, contrast, bright = (v[:, None, None, None] for v in (u(0.85, 1.15), u(0.85, 1.15), u(-0.08, 0.08)))
    return (x.clamp(0, 1).pow(gamma) * contrast + bright).clamp(0, 1)


def train_step(model, batch, opt, sched, scaler, device, amp) -> tuple[float, float]:
    b, n = batch["x"].shape[:2]
    x = augment_3ch(batch["x"].to(device, non_blocking=True).float().flatten(0, 1) / 255.0)
    y, w = batch["y"].to(device), batch["w"].to(device)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
        logits = model(x.view(b, n, *x.shape[1:]), batch["vid"].to(device), batch["smask"].to(device))
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


@torch.no_grad()
def predict(model, ds, device: str, batch: int, workers: int, amp: bool) -> np.ndarray:
    if len(ds) == 0:
        return np.zeros((0, len(TARGETS)), np.float32)
    model.eval()
    out = []
    for bt in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            logits = model(bt["x"].to(device).float() / 255.0, bt["vid"].to(device), bt["smask"].to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    model.train()
    return np.concatenate(out)


def _build(cfg: HRConfig, train_ds, rank: int, world: int, device: str) -> dict:
    model = DualViewMIL(cfg.backbone, cfg.pretrained, cfg.img_size).to(device)
    if cfg.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    opt = torch.optim.AdamW(llrd_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.llrd, cfg.wd))
    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=cfg.seed) if world > 1 else None
    loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=sampler is None, sampler=sampler,
                        num_workers=cfg.workers, drop_last=True, pin_memory=device == "cuda",
                        persistent_workers=cfg.workers > 0)
    steps = min(len(loader), cfg.max_steps) if cfg.max_steps else len(loader)
    net = model
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()] if device == "cuda" else None)
    amp = device == "cuda"
    return {"model": model, "net": net, "opt": opt, "sched": warmup_cosine(opt, steps * cfg.epochs, cfg.warmup),
            "sampler": sampler, "loader": loader, "steps": steps, "device": device, "amp": amp,
            "scaler": torch.amp.GradScaler(device, enabled=amp)}


def _train_epoch(cfg: HRConfig, run: dict, ep: int, first_excess: list[float], t0: float):
    """One epoch of steps. Returns (losses, excess, stop_reason)."""
    losses, excess = [], []
    for step, batch in enumerate(run["loader"]):
        if step >= run["steps"]:
            break
        loss, exc = train_step(run["net"], batch, run["opt"], run["sched"], run["scaler"], run["device"], run["amp"])
        if _any_rank(not np.isfinite(loss), run["device"]):
            return losses, excess, "nan"
        losses.append(loss)
        excess.append(exc)
        if ep == 1 and len(first_excess) < EXCESS_WINDOW:
            first_excess.append(exc)
        if _any_rank(time.time() - t0 > cfg.wall_cap_min * 60, run["device"]):
            return losses, excess, "wall_cap"
    return losses, excess, ""


def _end_epoch(cfg: HRConfig, run: dict, data: dict, rec: dict, first_excess: list[float], do_eval: bool) -> None:
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
        rec["fit_minus_oof"] = rec["fit_macro"] - rec["oof_macro"]
        rec["eval_s"] = round(time.time() - t_pred, 1)
        save_preds(os.path.join(cfg.out, "oof_preds.csv"), val_sids, oof)
    if run["device"] == "cuda":
        rec["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    torch.save(run["model"].state_dict(), os.path.join(cfg.out, "model.pt"))
    if rec["epoch"] == cfg.gate_epoch:
        rec["gate"] = _gate(cfg, rec, first_excess) or "pass"
    _log(cfg, rec)


def _finish(cfg: HRConfig, run: dict, data: dict, summary: dict, gate_reason: str) -> None:
    gold, gold_sids = data["gold"]
    preds = predict(run["model"], gold, run["device"], cfg.batch, cfg.workers, run["amp"])
    save_preds(os.path.join(cfg.out, "gold_preds.csv"), gold_sids, preds)
    summary["gold58"] = bootstrap_macro_auc(gold.ref, preds, n_boot=1000, seed=0) if len(preds) else None
    if summary["stop"] in ("gate", "nan"):
        with open(os.path.join(cfg.out, "ABORT"), "w") as f:
            f.write(gate_reason if summary["stop"] == "gate" else "non-finite training loss")
    with open(os.path.join(cfg.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary), flush=True)


def _gather(report: dict, world: int) -> list[dict]:
    if world == 1:
        return [report]
    out = [None] * world
    dist.all_gather_object(out, report)
    return out


def _preflight(cfg: HRConfig, data: dict, rank: int, world: int, device: str) -> tuple[list[dict], str]:
    """Run the preflight on every rank; returns (per-rank reports, stop reason or '')."""
    steps = (len(data["train"][0]) // world) // cfg.batch
    steps = min(steps, cfg.max_steps) if cfg.max_steps else steps
    report = run_preflight(cfg, data["train"][0], data["val"][0], len(data["val"][0]), len(data["fit"][0]),
                           steps, cfg.wall_cap_min * 60, device)
    reports = _gather(report, world)
    reasons = [decide_stop(r, cfg) for r in reports]
    if rank == 0:
        with open(os.path.join(cfg.out, "preflight.json"), "w") as f:
            json.dump({"ranks": reports, "stop": next((r for r in reasons if r), "")}, f, indent=1)
        print("PREFLIGHT", json.dumps(reports), flush=True)
    return reports, next((r for r in reasons if r), "")


def _teardown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def fit(cfg: HRConfig) -> dict:
    rank, world, device = _setup_ddp()
    torch.manual_seed(cfg.seed + rank)
    if os.path.exists(os.path.join(cfg.out, "summary.json")):
        raise FileExistsError(f"{cfg.out} already holds a finished run; refusing to overwrite")
    os.makedirs(cfg.out, exist_ok=True)
    data = prepare_hr(cfg)
    if cfg.preflight_steps:
        reports, stop = _preflight(cfg, data, rank, world, device)
        if stop:
            summary = {"stop": stop, "epochs_run": 0, "planned_epochs": 0, "wall_s": 0.0}
            if rank == 0:
                with open(os.path.join(cfg.out, "ABORT"), "w") as f:
                    f.write(stop)
                with open(os.path.join(cfg.out, "summary.json"), "w") as f:
                    json.dump(summary, f, indent=1)
            _teardown()
            return summary
        cfg = replace(cfg, wall_cap_min=cfg.wall_cap_min - max(r["preflight_s"] for r in reports) / 60)
        torch.manual_seed(cfg.seed + rank)  # preflight consumed RNG; training must not depend on it
    run = _build(cfg, data["train"][0], rank, world, device)
    if rank == 0:
        with open(os.path.join(cfg.out, "config.json"), "w") as f:
            json.dump({**asdict(cfg), "world": world, **{f"n_{k}": len(data[k][0]) for k in ("train", "val", "gold", "fit")},
                       "sources": data["sources"], "fold_seed": FOLD_SEED, "experiment": "E1-HR"}, f, indent=1)
    t0, first_excess, stop, rec, planned = time.time(), [], "", {}, cfg.epochs
    for ep in range(1, cfg.epochs + 1):
        if run["sampler"] is not None:
            run["sampler"].set_epoch(ep)
        losses, excess, stop = _train_epoch(cfg, run, ep, first_excess, t0)
        rec = {"epoch": ep, "loss": float(np.mean(losses)) if losses else float("nan"),
               "excess": float(np.mean(excess)) if excess else float("nan"),
               "lr_head": run["opt"].param_groups[-1]["lr"], "wall_s": round(time.time() - t0, 1),
               "planned_epochs": planned}
        if rank == 0 and stop != "nan":
            _end_epoch(cfg, run, data, rec, first_excess, _should_eval(cfg, ep, planned, stop))
        if ep == 1:
            budget = cfg.epochs if cfg.fixed_horizon else _planned_epochs(cfg, rec, time.time() - t0)
            planned = _broadcast_int(budget if rank == 0 else 0, device)
            run["sched"].horizon["total"] = planned * run["steps"]
        is_gate_failed = rank == 0 and rec.get("gate", "pass") != "pass"
        if _any_rank(is_gate_failed, device):
            stop = "gate"
        if stop or ep >= planned:
            break
    summary = {"stop": stop or "completed", "epochs_run": ep, "planned_epochs": planned,
               "wall_s": round(time.time() - t0, 1), "final_lr_head": rec.get("lr_head")}
    if rank == 0:
        _finish(cfg, run, data, summary, rec.get("gate", "gate failed"))
    _teardown()
    return summary


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    casters = {"int": int, "float": float, "str": str, "bool": lambda s: s.lower() in ("1", "true", "yes")}
    for f in fields(HRConfig):
        flag = f"--{f.name.replace('_', '-')}"
        if f.name == "label_roots":
            ap.add_argument(flag, nargs="+", default=[])
        elif f.default is MISSING:
            ap.add_argument(flag, type=casters[f.type], required=True)
        else:
            ap.add_argument(flag, type=casters[f.type], default=f.default)
    return fit(HRConfig(**vars(ap.parse_args(argv))))


if __name__ == "__main__":
    result = main()
    sys.exit(6 if result["stop"].startswith("preflight") else 0)
