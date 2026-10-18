"""CPU tests for E1 (cache, labels, model, optimiser groups, training smoke incl. gate abort)."""
import json
import os

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from src.constants import ID_COL, TARGETS
from src.e1 import cache as C
from src.e1.labels import SOURCE_FILES, find_source_files, reference_labels, train_targets
from src.e1.model import SliceMIL
from src.e1.optim import llrd_groups
from src.e1.train import main as train_main

TINY = "vit_tiny_patch16_224"
IOP = {"Sagittal": [0, 1, 0, 0, 0, -1], "Coronal": [1, 0, 0, 0, 0, -1], "Axial": [1, 0, 0, 0, 1, 0]}


def _write_slice(path, pos_mm, plane, value, size=32):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = MRImageStorage, generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.Rows, ds.Columns = "MR", size, size
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    normal = np.cross(IOP[plane][:3], IOP[plane][3:])
    ds.ImageOrientationPatient = IOP[plane]
    ds.ImagePositionPatient = list((normal * pos_mm).astype(float))
    ds.InstanceNumber = 1  # deliberately uninformative: ordering must come from geometry
    px = np.full((size, size), value, dtype=np.uint16)
    px[0, 0] = 0  # keeps the percentile window non-degenerate
    ds.PixelData = px.tobytes()
    ds.save_as(path, enforce_file_format=True)


def _write_study(root, sid, planes, n_slices=6):
    rows = []
    for plane in planes:
        uid = generate_uid()
        d = os.path.join(root, sid, uid)
        os.makedirs(d)
        for j in np.random.default_rng(0).permutation(n_slices):  # shuffled file names
            _write_slice(os.path.join(d, f"{generate_uid()}.dcm"), float(j * 3), plane, 100 + 50 * j)
        rows.append({ID_COL: sid, "SeriesInstanceUID": uid, "Fluid_Sensitive": 1,
                     "Fat_Suppression": 1, "Anatomical_Plane": plane})
    return rows


def test_central_indices_handles_short_and_long_series():
    assert C.central_indices(1, 4) == [0, 0, 0, 0]
    idx = C.central_indices(100, 16)
    assert idx[0] == 10 and idx[-1] == 89 and idx == sorted(idx)


def test_select_series_prefers_fluid_sensitive():
    rows = pd.DataFrame({"SeriesInstanceUID": ["a", "b", "c"], "Fluid_Sensitive": [0, 1, 1],
                         "Anatomical_Plane": ["Sagittal", "Sagittal", "Axial"]})
    assert C.select_series(rows) == {"Sagittal": "b", "Axial": "c"}


def test_series_slices_follow_geometry_not_filenames(tmp_path):
    rows = _write_study(str(tmp_path), "s1", ["Sagittal"], n_slices=8)
    out = C.series_slices(os.path.join(tmp_path, "s1", rows[0]["SeriesInstanceUID"]), k=4, size=16)
    means = out.reshape(4, -1).mean(1)
    assert out.dtype == np.uint8 and np.all(np.diff(means) > 0)


def test_study_volume_masks_missing_plane(tmp_path):
    rows = pd.DataFrame(_write_study(str(tmp_path), "s1", ["Sagittal", "Axial"]))
    vol, pmask, err = C.study_volume(os.path.join(tmp_path, "s1"), rows, k=4, size=16)
    assert vol.shape == (3, 4, 16, 16) and list(pmask) == [1, 0, 1] and err == ""
    assert vol[1].max() == 0


def _sources(sids, values):
    return {n: pd.DataFrame(v, index=pd.Index(sids, name=ID_COL), columns=TARGETS)
            for n, v in values.items()}


def test_train_targets_mean_and_prior_fill():
    sids = ["a", "b"]
    a = np.full((2, 12), np.nan)
    a[0] = 0.8
    b = np.full((2, 12), np.nan)
    b[0] = 0.4
    y, w = train_targets(_sources(sids, {"x": a, "y": b}), sids)
    assert np.allclose(y[0], 0.6) and np.all(w[0] == 1)
    assert np.allclose(y[1], 0.6) and np.all(w[1] == 0)  # uncovered -> prior, weight 0


def test_reference_requires_agreement_and_ignores_abstain():
    sids = ["a"]
    vals = {f"s{i}": np.full((1, 12), v) for i, v in enumerate([0.9, 0.95, 0.8, 0.7, 0.5])}
    ref, m = reference_labels(_sources(sids, vals), sids)
    assert np.all(ref == 1) and np.all(m == 1)          # 4 agree, 0.5 abstains
    vals["s3"] = np.full((1, 12), 0.1)
    ref, m = reference_labels(_sources(sids, vals), sids)
    assert np.all(m == 0)                               # only 3 agree


def test_model_shapes_and_llrd_cover_every_param_once():
    model = SliceMIL(TINY, pretrained=False, img_size=64)
    x = torch.rand(2, 6, 64, 64)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    assert model(x, torch.arange(3).repeat_interleave(2), mask).shape == (2, 12)
    groups = llrd_groups(model, 5e-5, 1e-3, 0.75, 0.05)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for p in model.parameters() if p.requires_grad)
    assert max(g["lr"] for g in groups) == 1e-3
    top = [g["lr"] for g in groups if any(p is model.backbone.blocks[-1].mlp.fc1.weight for p in g["params"])]
    assert top == [pytest.approx(5e-5 * 0.75)]           # last block sits one step below final norm


@pytest.fixture(scope="module")
def smoke_root(tmp_path_factory):
    return make_smoke_root(tmp_path_factory.mktemp("e1"))


def make_smoke_root(root):
    """Synthetic competition root: 20 weak + 4 gold studies, 5 label sources, built cache."""
    img, rows = root / "train_images", []
    n_weak, n_gold = 20, 4
    sids = [f"st{i}" for i in range(n_weak + n_gold)]
    for i, sid in enumerate(sids):
        rows += _write_study(str(img), sid, ["Sagittal", "Coronal", "Axial"] if i % 4 else ["Sagittal"], 4)
    series = pd.DataFrame(rows)
    series.to_csv(root / "train_series.csv", index=False)
    rng = np.random.default_rng(0)
    train = pd.DataFrame({ID_COL: sids, "Report": "x"})
    for t in TARGETS:
        train[t] = [np.nan] * n_weak + list(rng.integers(0, 2, n_gold))
    for t in TARGETS:  # gold needs both classes per target
        train.loc[n_weak, t], train.loc[n_weak + 1, t] = 0, 1
    train.to_csv(root / "train.csv", index=False)
    labels = root / "labels"
    labels.mkdir()
    agreed = np.array([[0.95 if (i + t) % 2 else 0.05 for t in range(12)] for i in range(len(sids))])
    for fname in SOURCE_FILES.values():  # identical sources -> every reference cell is kept
        df = pd.DataFrame(agreed, columns=TARGETS)
        df.insert(0, ID_COL, sids)
        df.to_parquet(labels / fname) if fname.endswith(".parquet") else df.to_csv(labels / fname, index=False)
    C.build_cache(sids, series, str(img), str(root / "cache"), k=4, size=64, workers=1, log=lambda *a: None)
    return root


def _smoke_args(root, out, extra=()):
    return ["--cache-dir", str(root / "cache"), "--data-root", str(root), "--out", str(out),
            "--label-roots", str(root / "labels"), "--backbone", TINY, "--pretrained", "false",
            "--img-size", "64", "--epochs", "2", "--batch", "2", "--k-train", "2", "--workers", "0",
            "--max-steps", "2", "--gate-epoch", "1", *extra]


def test_cache_index_and_meta(smoke_root):
    idx = pd.read_csv(smoke_root / "cache" / "e1_index.csv")
    assert len(idx) == 24 and (idx["error"].fillna("") == "").all()
    assert idx["has_Coronal"].tolist().count(0) == 6
    meta = json.load(open(smoke_root / "cache" / "e1_cache_meta.json"))
    assert meta["n"] == 24 and np.load(smoke_root / "cache" / "e1_cache.npy", mmap_mode="r").shape == (24, 3, 4, 64, 64)


def test_fit_smoke_writes_artifacts(smoke_root, tmp_path):
    out = tmp_path / "run"
    summary = train_main(_smoke_args(smoke_root, out, ["--gate-oof", "0", "--gate-excess", "100"]))
    assert summary["stop"] == "completed" and summary["epochs_run"] == 2
    for f in ("config.json", "metrics.jsonl", "model.pt", "oof_preds.csv", "gold_preds.csv", "summary.json"):
        assert (out / f).exists(), f
    oof = pd.read_csv(out / "oof_preds.csv")
    assert len(oof) == 4 and ((oof[TARGETS] >= 0) & (oof[TARGETS] <= 1)).all().all()
    assert not (out / "ABORT").exists()


def test_fit_aborts_when_epoch_gate_fails(smoke_root, tmp_path):
    out = tmp_path / "abort"
    summary = train_main(_smoke_args(smoke_root, out, ["--gate-oof", "1.01", "--gate-excess", "100"]))
    assert summary["stop"] == "gate" and summary["epochs_run"] == 1
    assert "oof" in (out / "ABORT").read_text()


def test_gates_report_go_flags(smoke_root, tmp_path):
    from src.e1.gates import evaluate
    out = tmp_path / "g"
    train_main(_smoke_args(smoke_root, out, ["--gate-oof", "0", "--gate-excess", "100"]))
    gold = pd.read_csv(out / "gold_preds.csv")
    gold[TARGETS] = np.random.default_rng(1).random((len(gold), 12))  # distinct values, no ties
    gold.to_csv(out / "gold_preds.csv", index=False)
    ediv = tmp_path / "ediv"
    ediv.mkdir()
    for arm, noise in (("a", 0.0), ("b", 0.3)):
        g = gold.copy()
        g[TARGETS] = g[TARGETS] + noise
        g.to_csv(ediv / f"preds_{arm}_gold.csv", index=False)
    v = evaluate(str(out), str(ediv), str(smoke_root / "cache" / "e1_cache_meta.json"), n_test=10)
    assert set(v) >= {"b_oof", "c_gold58", "d_rank_corr_vs_ediv", "e_runtime_h", "GO"}
    assert v["d_rank_corr_vs_ediv"]["pass"] is False      # identical ranking -> corr 1.0 fails diversity
    assert v["GO"] is False


def test_find_source_files_skips_image_trees(tmp_path):
    (tmp_path / "train_series" / "s").mkdir(parents=True)
    (tmp_path / "train_series" / "s" / "llm_labels_full.csv").write_text("x")
    (tmp_path / "ds" / "v1").mkdir(parents=True)
    (tmp_path / "ds" / "v1" / "llm_labels_full.csv").write_text("x")
    assert find_source_files([str(tmp_path)]) == {"steven": str(tmp_path / "ds" / "v1" / "llm_labels_full.csv")}


def test_planned_epochs_fit_wall_cap():
    from src.e1.train import Config, _planned_epochs
    cfg = Config(cache_dir="", data_root="", out="", epochs=15, wall_cap_min=38, eval_every=3, gate_epoch=3)
    # pilot regression: 176 s train (wall_s excludes eval) + 77 s eval/3 -> ~201.7 s/epoch;
    # (2306 - 253 - 120) / 201.7 = 9 more -> 10 planned (the buggy version planned 15)
    assert _planned_epochs(Config(cache_dir="", data_root="", out="", epochs=15, wall_cap_min=38.43),
                           {"wall_s": 176.1, "eval_s": 77.1}, elapsed_s=253.2) == 10
    assert _planned_epochs(cfg, {"wall_s": 30.0, "eval_s": 0.0}, elapsed_s=30.0) == 15   # capped at cfg
    assert _planned_epochs(cfg, {"wall_s": 2000.0, "eval_s": 0.0}, elapsed_s=2000.0) == 3  # never below gate
