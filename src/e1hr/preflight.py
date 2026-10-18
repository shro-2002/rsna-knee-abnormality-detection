"""E1-HR preflight: a short GPU memory + data-loading + schedule check inside the pilot's own budget.

Runs on every rank (so both GPUs contend for the same input mount, as in training) BEFORE the real
run, on a throwaway model/optimiser with the real train_step, batch size and loaders. It changes nothing:
the trainer reseeds afterwards and builds a fresh model. It never alters batch size, LR or any setting.

STOP reasons (any rank): preflight_oom, preflight_nonfinite, preflight_memory_margin (peak allocated
> mem_frac_max of device memory, train or eval), preflight_schedule_infeasible (the measured step
times cannot fit the epoch-3 gate epoch inside the wall cap). Otherwise a report of measured values.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.e1.optim import llrd_groups, warmup_cosine
from src.e1hr.model import DualViewMIL


def _train_step():  # late import: src.e1hr.train imports this module
    from src.e1hr.train import train_step
    return train_step


def decide_stop(report: dict, cfg) -> str:
    """'' when the preflight passes, else the stop reason (pure function of the report)."""
    if report.get("oom"):
        return "preflight_oom"
    if report.get("nonfinite"):
        return "preflight_nonfinite"
    peak, total = report.get("peak_alloc_gb"), report.get("device_total_gb")
    if peak is not None and total and peak > cfg.mem_frac_max * total:
        return "preflight_memory_margin"
    if report.get("epochs_fit", cfg.epochs) < cfg.gate_epoch:
        return "preflight_schedule_infeasible"
    return ""


def epochs_that_fit(cfg, epoch_train_s: float, eval_s: float, cap_s: float, reserve_s: float = 120.0) -> int:
    """Same rule as the trainer's budget-aware horizon (epoch 1 incl. eval, then amortised evals)."""
    per_epoch = epoch_train_s + eval_s / cfg.eval_every
    left = cap_s - (epoch_train_s + eval_s) - reserve_s
    return int(min(cfg.epochs, 1 + max(left, 0.0) // max(per_epoch, 1e-6)))


def schedule_final_lr_ratio(steps: int, planned: int, epochs: int, warmup: float) -> float:
    """Simulate the real scheduler incl. the one-time horizon shrink after epoch 1; final LR / peak."""
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine(opt, steps * epochs, warmup)
    for step in range(steps * planned):
        if step == steps:
            sched.horizon["total"] = planned * steps
        opt.step()
        sched.step()
    return float(opt.param_groups[0]["lr"])


def _measure_loader(loader: DataLoader, n_batches: int) -> dict:
    it, t0, nbytes = iter(loader), time.time(), 0
    for _ in range(n_batches):
        nbytes += next(it)["x"].numel()  # uint8 -> bytes
    dt = time.time() - t0
    return {"loader_only_batches_per_s": round(n_batches / dt, 3), "loader_only_mb_per_s": round(nbytes / dt / 1e6, 1)}


def _measure_train(model, loader, opt, sched, scaler, device: str, amp: bool, n_steps: int, warm: int) -> dict:
    train_step, it = _train_step(), iter(loader)
    wait, step_t, finite, n_images = [], [], True, 0
    for i in range(n_steps):
        t0 = time.time()
        batch = next(it)
        t1 = time.time()
        n_images = int(np.prod(batch["x"].shape[:2]))
        loss, _ = train_step(model, batch, opt, sched, scaler, device, amp)
        if device == "cuda":
            torch.cuda.synchronize()
        finite &= bool(np.isfinite(loss))
        if i >= warm:
            wait.append(t1 - t0)
            step_t.append(time.time() - t0)
    return {"step_s": float(np.mean(step_t)), "loader_wait_s": float(np.mean(wait)), "finite": finite,
            "images_per_step": n_images}


@torch.no_grad()
def _measure_eval(model, ds, device: str, batch: int, amp: bool, n_batches: int) -> float:
    model.eval()
    it, times = iter(DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)), []
    for _ in range(n_batches):
        bt = next(it)
        t0 = time.time()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            model(bt["x"].to(device).float() / 255.0, bt["vid"].to(device), bt["smask"].to(device))
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    model.train()
    return float(np.mean(times[1:] or times))  # drop the cold first batch


def run_preflight(cfg, train_ds, val_ds, n_val: int, n_fit: int, steps: int, cap_s: float, device: str) -> dict:
    """Measured report for this rank. Never raises on OOM / non-finite: they become stop reasons."""
    amp = device == "cuda"
    report: dict = {"device": device, "batch": cfg.batch, "steps_per_epoch": steps}
    t_start = time.time()
    try:
        if amp:
            torch.cuda.reset_peak_memory_stats()
            report["device_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        model = DualViewMIL(cfg.backbone, cfg.pretrained, cfg.img_size).to(device)
        opt = torch.optim.AdamW(llrd_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.llrd, cfg.wd))
        sched = warmup_cosine(opt, 10_000, cfg.warmup)
        loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers,
                            drop_last=True, pin_memory=amp)
        report.update(_measure_loader(loader, max(4, cfg.preflight_steps // 2)))
        train = _measure_train(model, loader, opt, sched, torch.amp.GradScaler(device, enabled=amp), device, amp,
                               cfg.preflight_steps, warm=max(2, cfg.preflight_steps // 4))
        report["nonfinite"] = not train["finite"]
        report["train_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if amp else None
        eval_batch_s = _measure_eval(model, val_ds, device, cfg.batch, amp, cfg.preflight_eval_batches)
        peak = torch.cuda.max_memory_allocated() / 1e9 if amp else None
        report.update(peak_alloc_gb=None if peak is None else round(peak, 2),
                      peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2) if amp else None,
                      step_s=round(train["step_s"], 3), loader_wait_s=round(train["loader_wait_s"], 3),
                      io_wait_fraction=round(train["loader_wait_s"] / max(train["step_s"], 1e-9), 3),
                      images_per_step=train["images_per_step"],
                      images_per_s_per_rank=round(train["images_per_step"] / train["step_s"], 1), eval_batch_s=round(eval_batch_s, 3))
        epoch_train_s = steps * train["step_s"]
        eval_s = eval_batch_s * (n_val + n_fit) / cfg.batch
        report.update(projected_epoch_train_s=round(epoch_train_s, 1), projected_eval_s=round(eval_s, 1),
                      epochs_fit=epochs_that_fit(cfg, epoch_train_s, eval_s, cap_s))
        planned = max(cfg.gate_epoch, report["epochs_fit"])
        report["schedule_final_lr_ratio"] = schedule_final_lr_ratio(steps, planned, cfg.epochs, cfg.warmup)
        del model, opt, loader
    except torch.cuda.OutOfMemoryError as err:
        report.update(oom=True, error=str(err)[:300])
    if amp:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    report["preflight_s"] = round(time.time() - t_start, 1)
    return report
