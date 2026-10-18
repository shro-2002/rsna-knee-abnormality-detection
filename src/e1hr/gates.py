"""E1-HR final gates (research/21 §3), evaluated offline. Reuses E1's gate code for (b), (d), (e);
(c) uses the blend break-even 0.894; (c2) is paired against the preserved E1 pilot gold-58 preds.

usage: python -m src.e1hr.gates --run RUN --e1-run E1_RUN --ediv-gold DIR --cache-meta E1_META --hr-cache-meta HR_META
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.constants import ID_COL, TARGETS
from src.e1 import gates as e1_gates
from src.validate.metric import macro_auc, roc_auc

GOLD_MIN = 0.894            # blend break-even vs E-DIV (research/20 §6)
PAIRED_MIN, FOCAL_MIN = 0.015, 0.03
FOCAL = ["MCL", "Lateral Meniscus", "Lateral OA", "Synovitis", "Contusion", "Fracture"]


def paired_vs_e1(hr: pd.DataFrame, e1: pd.DataFrame, y: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Paired gold-58 comparison: macro delta, its 90% bootstrap CI, and the focal-set mean delta."""
    ids = hr.index.intersection(e1.index).intersection(y.index)
    a, b, t = hr.loc[ids, TARGETS].to_numpy(), e1.loc[ids, TARGETS].to_numpy(), y.loc[ids, TARGETS].to_numpy(float)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(ids), len(ids))
        boots.append(macro_auc(t[ix], a[ix]) - macro_auc(t[ix], b[ix]))
    focal = [roc_auc(t[:, TARGETS.index(f)], a[:, TARGETS.index(f)]) - roc_auc(t[:, TARGETS.index(f)], b[:, TARGETS.index(f)])
             for f in FOCAL]
    return {"delta": macro_auc(t, a) - macro_auc(t, b), "ci90_low": float(np.nanpercentile(boots, 5)),
            "focal_delta": float(np.nanmean(focal)), "n": int(len(ids))}


def _hours(run: str, full_meta: dict, hr_meta: dict, n_test: int, ediv_hours: float) -> float:
    metrics = [json.loads(line) for line in open(os.path.join(run, "metrics.jsonl"))]
    decode_s = full_meta["wall_s"] / full_meta["n"] + hr_meta["wall_s"] / hr_meta["n"]
    return ediv_hours + n_test * (decode_s + metrics[-1].get("pred_s_per_study", 0.0)) / 3600


def evaluate(run: str, e1_run: str, ediv_gold: str, full_meta_path: str, hr_meta_path: str,
             train_csv: str, n_test: int = 5000, ediv_hours: float = 4.0) -> dict:
    load = lambda p: pd.read_csv(p).set_index(ID_COL)  # noqa: E731
    metrics = [json.loads(line) for line in open(os.path.join(run, "metrics.jsonl"))]
    summary = json.load(open(os.path.join(run, "summary.json")))
    hr_gold = load(os.path.join(run, "gold_preds.csv"))
    y = load(train_csv)
    paired = paired_vs_e1(hr_gold, load(os.path.join(e1_run, "gold_preds.csv")), y)
    oof = metrics[-1].get("oof_macro", float("nan"))
    gold = (summary.get("gold58") or {}).get("point", float("nan"))
    corr = e1_gates.mean_spearman(hr_gold[TARGETS], e1_gates.ediv_blend(ediv_gold))
    hours = _hours(run, json.load(open(full_meta_path)), json.load(open(hr_meta_path)), n_test, ediv_hours)
    gates = {"b_oof": (oof, oof >= e1_gates.OOF_MIN), "c_gold58": (gold, gold >= GOLD_MIN),
             "c2_paired_delta": (paired["delta"], paired["delta"] >= PAIRED_MIN and paired["ci90_low"] > 0),
             "c2_focal_delta": (paired["focal_delta"], paired["focal_delta"] >= FOCAL_MIN),
             "d_rank_corr_vs_ediv": (corr, corr <= e1_gates.CORR_MAX), "e_runtime_h": (hours, hours <= e1_gates.HOURS_MAX)}
    verdict = {"aborted": os.path.exists(os.path.join(run, "ABORT")), "stop": summary["stop"], "paired": paired,
               **{k: {"value": round(float(v), 4), "pass": bool(ok)} for k, (v, ok) in gates.items()}}
    verdict["GO"] = not verdict["aborted"] and all(g["pass"] for k, g in verdict.items() if k in gates)
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("run", "e1-run", "ediv-gold", "cache-meta", "hr-cache-meta", "train-csv"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--ediv-hours", type=float, default=4.0)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.run, a.e1_run, a.ediv_gold, a.cache_meta, a.hr_cache_meta, a.train_csv,
                              a.n_test, a.ediv_hours), indent=1))


if __name__ == "__main__":
    main()
