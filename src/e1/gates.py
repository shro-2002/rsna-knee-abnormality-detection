"""Final E1 pilot gates (research/18 E1, research/19 §3), evaluated offline from run artifacts.

  (b) last-epoch OOF macro vs frozen reference >= 0.880
  (c) gold-58 macro >= 0.890 (sanity only; never used to select anything)
  (d) mean per-target Spearman(E1, E-DIV equal-rank blend) on gold-58 <= 0.88
  (e) projected total scoring runtime <= 7 h
Gate (a) (epoch-3 learnable-loss drop) is enforced in-run by src/e1/train.py.

usage: python -m src.e1.gates --run RUN_DIR --ediv-gold DIR_WITH_preds_*_gold.csv --cache-meta META.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from src.constants import ID_COL, TARGETS
from src.ensemble.blend import rank_blend

OOF_MIN, GOLD_MIN, CORR_MAX, HOURS_MAX = 0.880, 0.890, 0.88, 7.0


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path).set_index(ID_COL)[TARGETS].astype(float)


def mean_spearman(a: pd.DataFrame, b: pd.DataFrame) -> float:
    common = a.index.intersection(b.index)
    if len(common) < 3:
        raise ValueError(f"only {len(common)} common studies")
    return float(np.mean([a.loc[common, t].rank().corr(b.loc[common, t].rank()) for t in TARGETS]))


def ediv_blend(pred_dir: str) -> pd.DataFrame:
    arms = [_load(p) for p in sorted(glob.glob(os.path.join(pred_dir, "preds_*_gold.csv")))]
    if not arms:
        raise FileNotFoundError(f"no preds_*_gold.csv in {pred_dir}")
    idx = arms[0].index
    blended = rank_blend([a.loc[idx].to_numpy() for a in arms])
    return pd.DataFrame(blended, index=idx, columns=TARGETS)


def projected_hours(metrics: list[dict], cache_meta: dict, n_test: int, ediv_hours: float) -> float:
    decode_s = cache_meta["wall_s"] / max(cache_meta["n"], 1)  # wall per study with the cache workers
    pred_s = metrics[-1].get("pred_s_per_study", 0.0)
    return ediv_hours + n_test * (decode_s + pred_s) / 3600


def evaluate(run: str, ediv_gold: str, cache_meta_path: str, n_test: int = 5000,
             ediv_hours: float = 4.0) -> dict:
    metrics = [json.loads(line) for line in open(os.path.join(run, "metrics.jsonl"))]
    summary = json.load(open(os.path.join(run, "summary.json")))
    cache_meta = json.load(open(cache_meta_path))
    oof = metrics[-1].get("oof_macro", float("nan"))
    gold = (summary.get("gold58") or {}).get("point", float("nan"))
    corr = mean_spearman(_load(os.path.join(run, "gold_preds.csv")), ediv_blend(ediv_gold))
    hours = projected_hours(metrics, cache_meta, n_test, ediv_hours)
    gates = {"b_oof": (oof, oof >= OOF_MIN), "c_gold58": (gold, gold >= GOLD_MIN),
             "d_rank_corr_vs_ediv": (corr, corr <= CORR_MAX), "e_runtime_h": (hours, hours <= HOURS_MAX)}
    verdict = {"aborted": os.path.exists(os.path.join(run, "ABORT")), "stop": summary["stop"],
               **{k: {"value": round(float(v), 4), "pass": bool(ok)} for k, (v, ok) in gates.items()}}
    verdict["GO"] = not verdict["aborted"] and all(g["pass"] for g in verdict.values() if isinstance(g, dict))
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ediv-gold", required=True)
    ap.add_argument("--cache-meta", required=True)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--ediv-hours", type=float, default=4.0)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.run, a.ediv_gold, a.cache_meta, a.n_test, a.ediv_hours), indent=1))


if __name__ == "__main__":
    main()
