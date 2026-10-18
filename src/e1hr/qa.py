"""E1-HR cache QA (runs inside the CPU cache kernel, right after the build; label-free except
the ID-coverage check). Writes qa.json and montage PNGs.

Gates (research/21 §3, research/22 §6): 0 decode errors; crop fallback <= 5 % of present series;
cache readable, shape (N,3,24,224,224) uint8, rows aligned with the E1 cache and train.csv, every
non-gold study covered by every label source. Montages: per plane, random studies plus the hardest
cases (largest centroid offset, fewest slices, any fallback), E1 full-FOV mid slice beside the
E1-HR crop windows, for human review.

usage: python -m src.e1hr.qa --hr-cache DIR --e1-cache DIR --data-root DIR --label-roots DIR... --out DIR
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.constants import ID_COL, TARGETS
from src.e1.cache import PLANES
from src.e1.labels import find_source_files, load_sources
from src.validate.split import load_labels

FALLBACK_MAX = 0.05
OFFSET_FLAG = 0.4
TILE = 160


def cache_checks(hr_dir: str, e1_dir: str, train_csv: str) -> dict:
    arr = np.load(os.path.join(hr_dir, "e1hr_cache.npy"), mmap_mode="r")
    meta = json.load(open(os.path.join(hr_dir, "e1hr_cache_meta.json")))
    expected = (meta["k"], meta["size"], meta["size"])
    hr = pd.read_csv(os.path.join(hr_dir, "e1hr_index.csv"))
    e1 = pd.read_csv(os.path.join(e1_dir, "e1_index.csv"))
    train_ids = pd.read_csv(train_csv)[ID_COL].tolist()
    rng = np.random.default_rng(0)
    sample = rng.choice(len(hr), size=min(64, len(hr)), replace=False)
    empty = [int(r) for r in sample for p, plane in enumerate(PLANES) if hr.loc[r, f"has_{plane}"] and arr[r, p].std() == 0]
    return {
        "shape": list(arr.shape), "dtype": str(arr.dtype),
        "shape_ok": bool(arr.ndim == 5 and arr.shape[1:] == (3, *expected) and arr.dtype == np.uint8),
        "geometry_predeclared": bool(expected == (24, 224, 224)),
        "rows_match_index": bool(arr.shape[0] == len(hr) and (hr["row"].to_numpy() == np.arange(len(hr))).all()),
        "ids_equal_train_csv_order": hr[ID_COL].tolist() == train_ids,
        "aligned_with_e1_cache": hr[ID_COL].tolist() == e1[ID_COL].tolist(),
        "n_studies": int(len(hr)), "n_decode_errors": int(hr["error"].fillna("").ne("").sum()),
        "sampled_rows": len(sample), "sampled_empty_planes": empty,
        "cache_gb": round(os.path.getsize(os.path.join(hr_dir, "e1hr_cache.npy")) / 1e9, 2),
    }


def plane_stats(hr: pd.DataFrame) -> dict:
    out, present_total, fallback_total = {}, 0, 0
    for plane in PLANES:
        has = hr[f"has_{plane}"].astype(bool)
        fb = hr.get(f"fallback_{plane}", pd.Series("", index=hr.index)).fillna("").astype(str)
        off = pd.to_numeric(hr.get(f"offset_{plane}"), errors="coerce")[has]
        nsl = pd.to_numeric(hr.get(f"n_slices_{plane}"), errors="coerce")[has]
        n_fb = int((fb[has] != "").sum())
        present_total += int(has.sum())
        fallback_total += n_fb
        out[plane] = {"present": int(has.sum()), "missing": int((~has).sum()), "fallback": n_fb,
                      "fallback_kinds": fb[has & (fb != "")].value_counts().to_dict(),
                      "offset_q50_q95_max": [round(float(q), 3) for q in off.quantile([0.5, 0.95, 1.0])],
                      "offset_gt_flag": int((off > OFFSET_FLAG).sum()),
                      "n_slices_min_q50_max": [float(nsl.min()), float(nsl.median()), float(nsl.max())],
                      "series_shorter_than_24": int((nsl < 24).sum())}
    rate = fallback_total / max(present_total, 1)
    return {"planes": out, "fallback_rate": round(rate, 4), "fallback_ok": bool(rate <= FALLBACK_MAX)}


def label_coverage(train_csv: str, label_roots: list[str]) -> dict:
    weak = load_labels(train_csv).weak[ID_COL]
    paths = find_source_files(label_roots)
    sources = load_sources(paths)
    cov = {name: float(weak.isin(df.index).mean()) for name, df in sources.items()}
    return {"sources_found": sorted(paths), "weak_coverage": cov,
            "labels_ok": bool(len(sources) == 5 and all(v == 1.0 for v in cov.values())),
            "targets": TARGETS}


def pick_cases(hr: pd.DataFrame, plane: str, n_random: int = 4, n_hard: int = 2) -> list[tuple[int, str]]:
    has = hr[hr[f"has_{plane}"].astype(bool)]
    rng = np.random.default_rng(len(plane))
    picks = [(int(r), "random") for r in rng.choice(has.index, size=min(n_random, len(has)), replace=False)]
    off = pd.to_numeric(has[f"offset_{plane}"], errors="coerce").nlargest(n_hard)
    nsl = pd.to_numeric(has[f"n_slices_{plane}"], errors="coerce").nsmallest(n_hard)
    fb = has[has[f"fallback_{plane}"].fillna("").astype(str) != ""].head(n_hard)
    picks += [(int(r), "max_offset") for r in off.index] + [(int(r), "fewest_slices") for r in nsl.index]
    picks += [(int(r), "fallback") for r in fb.index]
    seen, unique = set(), []
    for r, why in picks:
        if r not in seen:
            seen.add(r)
            unique.append((r, why))
    return unique


def montage(hr_arr, e1_arr, hr: pd.DataFrame, plane: str, path: str) -> list[dict]:
    """Rows = cases; columns = E1 full-FOV mid | crop first window | crop mid | crop last window."""
    p = PLANES.index(plane)
    cases = pick_cases(hr, plane)
    canvas = Image.new("L", (TILE * 4, TILE * len(cases)))
    draw = ImageDraw.Draw(canvas)
    for i, (r, why) in enumerate(cases):
        k = hr_arr.shape[2]
        tiles = [e1_arr[r, p, e1_arr.shape[2] // 2], hr_arr[r, p, 1], hr_arr[r, p, k // 2], hr_arr[r, p, k - 2]]
        for j, t in enumerate(tiles):
            canvas.paste(Image.fromarray(np.asarray(t)).resize((TILE, TILE)), (TILE * j, TILE * i))
        draw.text((3, TILE * i + 3), f"{why} off={hr.loc[r, f'offset_{plane}']}", fill=255)
    canvas.save(path)
    return [{"row": r, "study": hr.loc[r, ID_COL], "why": why} for r, why in cases]


def run_qa(hr_dir: str, e1_dir: str, data_root: str, label_roots: list[str], out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    train_csv = os.path.join(data_root, "train.csv")
    hr = pd.read_csv(os.path.join(hr_dir, "e1hr_index.csv"))
    report = {"cache": cache_checks(hr_dir, e1_dir, train_csv), **plane_stats(hr),
              "labels": label_coverage(train_csv, label_roots)}
    hr_arr = np.load(os.path.join(hr_dir, "e1hr_cache.npy"), mmap_mode="r")
    e1_arr = np.load(os.path.join(e1_dir, "e1_cache.npy"), mmap_mode="r")
    report["montages"] = {plane: montage(hr_arr, e1_arr, hr, plane, os.path.join(out, f"qa_{plane}.png"))
                          for plane in PLANES}
    c = report["cache"]
    report["PASS"] = bool(c["shape_ok"] and c["geometry_predeclared"] and c["rows_match_index"] and c["ids_equal_train_csv_order"]
                          and c["aligned_with_e1_cache"] and c["n_decode_errors"] == 0
                          and not c["sampled_empty_planes"] and report["fallback_ok"] and report["labels"]["labels_ok"])
    with open(os.path.join(out, "qa.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("QA", json.dumps({k: report[k] for k in ("PASS", "fallback_rate")}), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-cache", required=True)
    ap.add_argument("--e1-cache", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--label-roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_qa(a.hr_cache, a.e1_cache, a.data_root, a.label_roots, a.out)


if __name__ == "__main__":
    main()
