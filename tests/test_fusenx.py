"""FUSE-NX CPU tests: merge/fusion invariants, fold isolation, synthetic teacher->student pipeline,
baseline equivalence (student == baseline when the teacher equals the report label), determinism, no overwrite."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.constants import ID_COL, TARGETS
from src.e1.labels import find_source_files, load_sources, train_targets
from src.e1hr import cache as H
from src.e1hr.fuse import fuse_targets, merge_teacher_oof
from src.e1hr.train import HRConfig, main as hr_main, prepare_hr
from src.validate.split import group_kfold_indices, load_labels
from tests.test_e1 import TINY, make_smoke_root


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = make_smoke_root(tmp_path_factory.mktemp("fusenx"))
    series = pd.read_csv(r / "train_series.csv")
    sids = pd.read_csv(r / "train.csv")[ID_COL].tolist()
    H.build_cache_hr(sids, series, str(r / "train_images"), str(r / "hr"), k=6, size=64, workers=1, log=lambda *a: None)
    return r


def fold_ids(root):
    weak = load_labels(str(root / "train.csv")).weak.reset_index(drop=True)
    return {k: set(weak[ID_COL].iloc[v]) for k, (_, v) in enumerate(group_kfold_indices(weak, 5, seed=0))}


def args(root, out, extra=()):
    return ["--cache-dir", str(root / "cache"), "--hr-cache-dir", str(root / "hr"), "--data-root", str(root),
            "--out", str(out), "--label-roots", str(root / "labels"), "--backbone", TINY, "--pretrained", "false",
            "--img-size", "64", "--epochs", "2", "--batch", "2", "--workers", "0", "--max-steps", "2",
            "--gate-epoch", "2", "--gate-oof", "0", "--gate-excess", "100", "--preflight-steps", "0",
            "--fixed-horizon", "true", *extra]


def write_oof(path, ids, values):
    df = pd.DataFrame(values, columns=TARGETS)
    df.insert(0, ID_COL, list(ids))
    df.to_csv(path, index=False)


def test_fuse_equation_range_and_missing():
    rep = np.array([[0.2] * 12, [1.0] * 12], np.float32)
    teacher = pd.DataFrame([[0.6] * 12, [0.0] * 12], index=["a", "b"], columns=TARGETS)
    assert np.allclose(fuse_targets(rep, ["a", "b"], teacher), [[0.4] * 12, [0.5] * 12])
    with pytest.raises(ValueError, match="no teacher"):
        fuse_targets(rep, ["a", "zzz"], teacher)


def test_merge_rejects_misaligned_files(tmp_path):
    expected = {1: {"a", "b"}, 2: {"c"}}
    write_oof(tmp_path / "t1.csv", ["a", "b"], np.full((2, 12), 0.3))
    write_oof(tmp_path / "t2.csv", ["c"], np.full((1, 12), 0.7))
    m = merge_teacher_oof([str(tmp_path / "t1.csv"), str(tmp_path / "t2.csv")], expected)
    assert sorted(m.index) == ["a", "b", "c"] and m.loc["c", "teacher_fold"] == 2
    write_oof(tmp_path / "bad.csv", ["a", "x"], np.full((2, 12), 0.3))       # wrong IDs
    with pytest.raises(ValueError, match="do not equal"):
        merge_teacher_oof([str(tmp_path / "bad.csv"), str(tmp_path / "t2.csv")], expected)
    write_oof(tmp_path / "rng.csv", ["c"], np.full((1, 12), 1.5))           # out of range
    with pytest.raises(ValueError, match="finite in"):
        merge_teacher_oof([str(tmp_path / "t1.csv"), str(tmp_path / "rng.csv")], expected)


def test_teacher_pool_excludes_held_out_and_fold0(root):
    folds = fold_ids(root)
    for k in (1, 2, 3, 4):
        cfg = HRConfig(cache_dir=str(root / "cache"), hr_cache_dir=str(root / "hr"), data_root=str(root), out="",
                       label_roots=[str(root / "labels")], fold=k, exclude_folds="0")
        data = prepare_hr(cfg)
        train, val = set(data["train"][1]), set(data["val"][1])
        assert val == folds[k] and not (train & (folds[0] | folds[k])) and train == set().union(*(folds[j] for j in (1, 2, 3, 4) if j != k))
        assert not (train & set(data["gold"][1]))


def _state(out):
    return torch.load(out / "model.pt", weights_only=True)


def test_end_to_end_teachers_merge_student_and_baseline_equivalence(root, tmp_path):
    folds = fold_ids(root)
    teacher_files = []
    for k in (1, 2, 3, 4):
        out = tmp_path / f"teacher{k}"
        assert hr_main(args(root, out, ["--fold", str(k), "--exclude-folds", "0"]))["stop"] == "completed"
        oof = pd.read_csv(out / "oof_preds.csv")
        assert set(oof[ID_COL]) == folds[k]                      # strictly the held-out fold
        teacher_files.append(str(out / "oof_preds.csv"))
    merged = merge_teacher_oof(teacher_files, {k: folds[k] for k in (1, 2, 3, 4)})
    merged.reset_index().drop(columns="teacher_fold").to_csv(tmp_path / "teacher_oof.csv", index=False)
    assert set(merged.index) == set().union(*(folds[k] for k in (1, 2, 3, 4))) and not (set(merged.index) & folds[0])

    base1, base2, student = tmp_path / "base1", tmp_path / "base2", tmp_path / "student"
    hr_main(args(root, base1, ["--fold", "0"]))
    hr_main(args(root, base2, ["--fold", "0"]))
    s1, s2 = _state(base1), _state(base2)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)              # CPU training is deterministic

    # teacher == report label  =>  fused target == report target  =>  student must equal baseline bit-for-bit
    src = load_sources(find_source_files([str(root / "labels")]))
    pool = sorted(set().union(*(folds[k] for k in (1, 2, 3, 4))))
    rep, _ = train_targets(src, pool)
    write_oof(tmp_path / "identity_teacher.csv", pool, rep)
    hr_main(args(root, student, ["--fold", "0", "--teacher-oof", str(tmp_path / "identity_teacher.csv")]))
    s3 = _state(student)
    assert all(torch.equal(s1[k], s3[k]) for k in s1)

    real = tmp_path / "student_real"
    summary = hr_main(args(root, real, ["--fold", "0", "--teacher-oof", str(tmp_path / "teacher_oof.csv")]))
    assert summary["stop"] == "completed" and summary["planned_epochs"] == summary["epochs_run"] == 2
    assert any(not torch.equal(s1[k], _state(real)[k]) for k in s1)  # only the target changed, and it matters

    with pytest.raises(FileExistsError):                             # artifacts never overwritten
        hr_main(args(root, real, ["--fold", "0"]))


def test_teacher_sanity_flags_collapse_not_disagreement():
    from src.e1hr.fuse import teacher_sanity
    rng = np.random.default_rng(0)
    ok = pd.DataFrame(rng.random((50, 12)), columns=TARGETS)
    assert teacher_sanity(ok) == []
    assert teacher_sanity(1 - ok) == []                       # disagreeing with any label is not a failure
    flat = ok.copy(); flat["ACL"] = 0.3
    assert any("ACL: collapsed" in p for p in teacher_sanity(flat))
