"""E-DIV: OAI-free, license-clean ensemble of public CC0/Apache knee-MRI readers, our own driver + blend.

Arms (all third-party weights, attributed in docs/EDIV_PROVENANCE.md):
  raptor_v5   dreaddevelopment/raptor-knee-maxspan      CC0  (64 slices, span 2-98%, 62 windows)
  raptor_v10  dreaddevelopment/raptor-knee-native384dense CC0 (64 slices, stored 384, 62 windows)
  raptor_v9   dreaddevelopment/raptor-knee-finespacing  CC0  (80 slices, span 2-98%, 78 windows)
  convnext    goodpjw2008/rsna-knee-2-5d-convnext-reader Apache-2.0 (3 folds, probability mean)
Blend: equal-weight mean of per-target fractional ranks (src.ensemble.blend.rank_blend). No tuning on gold-58 or LB.
Modes: gold (sanity + label-free diversity on extra train studies), test (writes submission.csv).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ediv import convnext_adapter as CA                    # noqa: E402
from src.ediv import raptor_prep as P                          # noqa: E402
from src.ediv import raptor_model as RM                        # noqa: E402
from src.ensemble.blend import rank_blend                      # noqa: E402

LABELS = RM.LABELS
RAPTOR_ARMS = {  # name -> (checkpoint file, geometry key)
    "raptor_v5": ("raptor_ft_coatnet_v5_full_swa.pt", "v5_64_2to98_336_62"),
    "raptor_v10": ("raptor_ft_coatnet_v10_full.pt", "v10_64_2to98_384_62"),
    "raptor_v9": ("raptor_ft_coatnet_v9_full.pt", "v9_80_2to98_336_78"),
    # E-OAI-1 only (research/37): nartaa OAI-trained 0.949 checkpoint, dataset nartaa/rsna-knee-publication-swa-weights-20261007 v1
    "nartaa_oai": ("raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "nartaa_96_2to98_384_94"),
}
ALL_ARMS = ["raptor_v5", "raptor_v10", "raptor_v9", "convnext"]
NARTAA_SHA256 = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
NARTAA_CROP = 320                     # stored 384 px, centre-cropped to 320 at inference (no resize)
EOAI_WEIGHTS = (0.5, 0.5)             # frozen before any run: rank(E-DIV equal-weight blend) vs rank(nartaa)
# reproduction sanity floors on gold-58 = author-recorded gold AUC (in checkpoint) - 0.02 (convnext: loose 0.85). NOT used for weights.
SANITY_MIN = {"raptor_v9": 0.883, "raptor_v5": 0.901, "raptor_v10": 0.897, "convnext": 0.85, "nartaa_oai": 0.899}
SEARCH = os.environ.get("EDIV_SEARCH", "/kaggle/input").split(os.pathsep)
SMOKE = int(os.environ.get("EDIV_SMOKE", "0"))     # >0: cap windows per arm (smoke tests only)


def find_file(name: str) -> str:
    for base in SEARCH:
        for depth in range(1, 7):
            hits = glob.glob(os.path.join(base, *(["*"] * depth), name))
            if hits:
                return hits[0]
    raise FileNotFoundError(name)


def comp_root() -> str:
    """Competition data root = the sample_submission.csv directory that also holds the DICOM series folders."""
    for base in SEARCH:
        for depth in range(1, 7):
            for hit in sorted(glob.glob(os.path.join(base, *(["*"] * depth), "sample_submission.csv"))):
                d = os.path.dirname(hit)
                if any(os.path.isdir(f"{d}/{n}") for n in ("test_series", "test_images", "train_series", "train_images")):
                    return d
    raise FileNotFoundError("competition root with image folders not found")


def series_dir(root: str, mode: str) -> str:
    names = ["test_series", "test_images"] if mode == "test" else ["train_series", "train_images"]
    for n in names:
        if os.path.isdir(f"{root}/{n}"):
            return f"{root}/{n}"
    raise FileNotFoundError(f"no image directory under {root} for mode {mode}")


def verified(path: str, sha256: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == sha256, f"checkpoint hash mismatch: {path}"
    return path


def final_blend(probs: dict, blend: str) -> np.ndarray:
    """ediv: equal-weight rank mean of the four E-DIV arms (unchanged). eoai: EOAI_WEIGHTS over [E-DIV blend, nartaa]."""
    ediv = rank_blend([probs[k] for k in ALL_ARMS])
    if blend == "ediv":
        return ediv
    return rank_blend([ediv, probs["nartaa_oai"]], list(EOAI_WEIGHTS))


def k_of(geo: P.Geometry) -> int:
    return min(geo.k_eval, SMOKE) if SMOKE else geo.k_eval


class Decode(torch.utils.data.Dataset):
    """CPU side: decode every needed stack for one study (shared decode for the two 64-slice Raptor arms)."""

    def __init__(self, ids, rows, root, cnx_dir, arms):
        self.ids, self.rows, self.root, self.cnx_dir, self.arms = list(ids), rows, root, cnx_dir, arms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid, rows, out, t = self.ids[i], self.rows.get(self.ids[i], []), {"sid": self.ids[i]}, time.time()
        try:
            if {"raptor_v5", "raptor_v10"} & set(self.arms):
                g = P.GEOMETRIES["v5_64_2to98_336_62"]
                sizes = tuple(sz for a, sz in (("raptor_v5", 336), ("raptor_v10", 384)) if a in self.arms)
                out["g64"] = P.build_stacks(sid, rows, self.root, g, sizes)
            if "raptor_v9" in self.arms:
                out["g80"] = P.build_stack(sid, rows, self.root, P.GEOMETRIES["v9_80_2to98_336_78"])
            if "nartaa_oai" in self.arms:
                out["g96"] = P.build_stack(sid, rows, self.root, P.GEOMETRIES["nartaa_96_2to98_384_94"])
            if "convnext" in self.arms:
                out["cnx"] = CA.prepare_study(self.cnx_dir, sid, rows, self.root)
        except Exception as e:                        # a study that cannot be decoded gets flagged, never dropped
            out["error"] = f"{type(e).__name__}: {e}"[:200]
        out["t_decode"] = time.time() - t
        return out


def worker(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = comp_root()
    cnx_dir = os.path.dirname(find_file("cnxt_v0_fold0.pt")) if "convnext" in a.arms else None
    ids, rows = load_ids(a.mode, root)
    shard = ids[a.shard::a.nshards]
    nets, cnx_models = {}, None
    for arm in a.arms:
        if arm == "nartaa_oai":
            nets[arm] = RM.load_raptor(verified(find_file(RAPTOR_ARMS[arm][0]), NARTAA_SHA256), dev, img_size=NARTAA_CROP)
        elif arm in RAPTOR_ARMS:
            nets[arm] = RM.load_raptor(find_file(RAPTOR_ARMS[arm][0]), dev)
        else:
            cnx_models = CA.load_folds(cnx_dir, dev)
    dl = torch.utils.data.DataLoader(Decode(shard, rows, series_dir(root, a.mode), cnx_dir, a.arms), batch_size=None, num_workers=a.workers,
                                     prefetch_factor=2 if a.workers else None, collate_fn=lambda x: x)
    probs = {arm: np.full((len(shard), 12), 0.5, np.float32) for arm in a.arms}
    flag = {arm: np.zeros(len(shard), np.int8) for arm in a.arms}            # 0 ok, 1 no data, 2 error
    tm = {k: 0.0 for k in ["wait"] + a.arms}
    t0, done = time.time(), 0
    for i, item in enumerate(dl):
        tw = time.time()
        for arm in a.arms:
            try:
                if "error" in item:
                    raise RuntimeError(item["error"])
                ta = time.time()
                if arm == "nartaa_oai":
                    geo = P.GEOMETRIES[RAPTOR_ARMS[arm][1]]
                    stack, mask = item["g96"]
                    if mask.sum() < 3:
                        flag[arm][i] = 1; continue
                    probs[arm][i] = RM.predict_study_mirror_crop(nets[arm][0], stack, P.window_centres(mask, k_of(geo)), NARTAA_CROP,
                                                                 [s[2] for s in geo.slots], (0, 1), dev)
                elif arm in RAPTOR_ARMS:
                    geo = P.GEOMETRIES[RAPTOR_ARMS[arm][1]]
                    stack, mask = item["g80"] if arm == "raptor_v9" else item["g64"][geo.stored]
                    if mask.sum() < 3:
                        flag[arm][i] = 1; continue
                    net, res = nets[arm]
                    probs[arm][i] = RM.predict_study(net, stack, mask, P.window_centres(mask, k_of(geo)), res, dev)
                else:
                    x, mask, pos = item["cnx"]
                    if not mask.any():
                        flag[arm][i] = 1; continue
                    probs[arm][i] = CA.predict_study(cnx_models, x, mask, pos, dev)
                tm[arm] += time.time() - ta
            except Exception as e:
                flag[arm][i] = 2
                print(f"[shard {a.shard}] {arm} study {item['sid'][-8:]} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
        tm["wait"] += item.get("t_decode", 0.0)
        done += 1
        if done % 10 == 0:
            print(f"[shard {a.shard}] {done}/{len(shard)} {time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > a.deadline:
            print(f"[shard {a.shard}] DEADLINE hit at {done}/{len(shard)}", flush=True); break
    np.savez(f"{a.out}/shard{a.shard}_{a.mode}.npz", ids=np.array(shard[:done]), done=done, wall=time.time() - t0,
             timing=json.dumps(tm), **{f"p_{k}": v[:done] for k, v in probs.items()}, **{f"f_{k}": v[:done] for k, v in flag.items()})


def load_ids(mode: str, root: str):
    if mode == "test":
        ids = pd.read_csv(f"{root}/test.csv").StudyInstanceUID.astype(str).tolist(); series = pd.read_csv(f"{root}/test_series.csv")
    else:
        tr = pd.read_csv(f"{root}/train.csv"); tr["StudyInstanceUID"] = tr.StudyInstanceUID.astype(str)
        gold = tr[tr[LABELS].notna().all(axis=1)].StudyInstanceUID.tolist()
        if mode == "gold":
            ids = gold
        else:   # "bench": 120 random non-gold train studies, label-free throughput/diversity
            rest = sorted(set(tr.StudyInstanceUID) - set(gold)); ids = list(np.random.default_rng(0).choice(rest, 120, replace=False))
        series = pd.read_csv(f"{root}/train_series.csv")
    series["StudyInstanceUID"] = series.StudyInstanceUID.astype(str); series["SeriesInstanceUID"] = series.SeriesInstanceUID.astype(str)
    return ids, {k: v.to_dict("records") for k, v in series.groupby("StudyInstanceUID")}


def run_shards(a):
    os.makedirs(a.out, exist_ok=True)
    n = max(1, min(a.gpus, torch.cuda.device_count())) if torch.cuda.is_available() else 1
    procs = []
    for i in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)) if torch.cuda.is_available() else dict(os.environ)
        cmd = [sys.executable, "-m", "src.ediv.run_ensemble", "--worker", "--shard", str(i), "--nshards", str(n), "--mode", a.mode,
               "--out", a.out, "--arms", ",".join(a.arms), "--workers", str(a.workers), "--deadline", str(a.deadline)]
        procs.append(subprocess.Popen(cmd, env=env, cwd=os.getcwd()))
    for p in procs:
        assert p.wait() == 0, "worker failed"
    ids, probs, flags, wall, timing = [], {k: [] for k in a.arms}, {k: [] for k in a.arms}, [], []
    for i in range(n):
        z = np.load(f"{a.out}/shard{i}_{a.mode}.npz")
        ids += list(z["ids"]); wall.append(float(z["wall"])); timing.append(json.loads(str(z["timing"])))
        for k in a.arms:
            probs[k].append(z[f"p_{k}"]); flags[k].append(z[f"f_{k}"])
    return ids, {k: np.concatenate(v) for k, v in probs.items()}, {k: np.concatenate(v) for k, v in flags.items()}, wall, timing


def reorder(ids, probs, flags, canonical):
    pos = {s: i for i, s in enumerate(ids)}
    assert set(pos) == set(canonical), f"missing studies: {len(set(canonical) - set(pos))}"
    ix = [pos[s] for s in canonical]
    return {k: v[ix] for k, v in probs.items()}, {k: v[ix] for k, v in flags.items()}


def mean_rank_corr(pa, pb):
    ra, rb = (pd.DataFrame(p).rank().to_numpy() for p in (pa, pb))
    return float(np.mean([np.corrcoef(ra[:, j], rb[:, j])[0, 1] for j in range(12)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="test", choices=["test", "gold", "bench"])
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--deadline", type=float, default=7.5 * 3600)
    ap.add_argument("--blend", default="ediv", choices=["ediv", "eoai"])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    a.arms = a.arms.split(",")
    if a.blend == "eoai" and not a.worker:
        assert a.arms == ALL_ARMS + ["nartaa_oai"], "E-OAI-1 is frozen to the four E-DIV arms + nartaa_oai"
    if a.worker:
        return worker(a)
    t0 = time.time(); root = comp_root()
    canon, _ = load_ids(a.mode, root)
    ids, probs, flags, wall, timing = run_shards(a)
    probs, flags = reorder(ids, probs, flags, canon)
    n = len(canon)
    rep = {"mode": a.mode, "n": n, "arms": a.arms, "wall_s": time.time() - t0, "shard_wall_s": wall, "timing_s": timing,
           "sec_per_study_wall": max(wall) / n, "flags": {k: {"no_data": int((v == 1).sum()), "error": int((v == 2).sum())} for k, v in flags.items()}}
    pairs = {f"{x}|{y}": mean_rank_corr(probs[x], probs[y]) for i, x in enumerate(a.arms) for y in a.arms[i + 1:]}
    rep["mean_rank_corr"] = pairs
    if n >= 50:
        for k, v in flags.items():
            assert (v != 0).mean() < 0.2, f"{k}: {(v != 0).mean():.0%} of studies flagged (no data/error)"
    if a.mode == "gold":
        from src.validate.metric import roc_auc
        tr = pd.read_csv(f"{root}/train.csv").set_index("StudyInstanceUID"); G = tr.loc[canon, LABELS].to_numpy(float)
        auc = lambda p: float(np.mean([roc_auc(G[:, j], p[:, j]) for j in range(12)]))
        rep["gold58"] = {k: auc(v) for k, v in probs.items()}
        rep["sanity_pass"] = {k: rep["gold58"][k] >= SANITY_MIN[k] for k in a.arms}
        rep["gold58_blend_equal"] = auc(rank_blend([probs[k] for k in ALL_ARMS if k in probs]))
        if a.blend == "eoai":
            rep["gold58_blend_eoai"] = auc(final_blend(probs, "eoai"))
    if a.mode == "test":
        # an isolated undecodable study falls back to 0.5 for that arm (as in the public baselines); >1% errors => abort
        bad = {k: float((flags[k] == 2).mean()) for k in a.arms}
        assert all(v <= 0.01 for v in bad.values()), f"arm error rate too high: {bad}"
        blend = final_blend(probs, a.blend) if a.blend == "eoai" else rank_blend([probs[k] for k in a.arms])   # ediv path unchanged
        sample = pd.read_csv(f"{root}/sample_submission.csv")
        sub = pd.DataFrame(blend, columns=LABELS); sub.insert(0, "StudyInstanceUID", canon)
        sub = sub[list(sample.columns)]
        assert sub.StudyInstanceUID.tolist() == sample.StudyInstanceUID.tolist(), "ID order drift vs sample_submission"
        assert sub.shape == sample.shape and np.isfinite(sub[LABELS].to_numpy()).all()
        sub.to_csv(f"{a.out}/submission.csv", index=False)
    for k, v in probs.items():
        pd.DataFrame(v, columns=LABELS).assign(StudyInstanceUID=canon).to_csv(f"{a.out}/preds_{k}_{a.mode}.csv", index=False)
    json.dump(rep, open(f"{a.out}/report_{a.mode}.json", "w"), indent=1)
    print("EDIV_REPORT", json.dumps(rep))


if __name__ == "__main__":
    main()
