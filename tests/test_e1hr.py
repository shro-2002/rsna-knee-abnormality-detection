"""CPU tests for E1-HR: crop geometry, dual-view dataset, model + checkpoint reload, LR schedule fix,
training smoke and gate abort. Reuses the synthetic competition root from tests/test_e1.py."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.constants import TARGETS
from src.e1.optim import warmup_cosine
from src.e1hr import cache as H
from src.e1hr.data import DualViewDataset, crop_windows
from src.e1hr.model import DualViewMIL
from src.e1hr.train import main as hr_main
from tests.test_e1 import TINY, make_smoke_root


def _disc(h=200, w=200, cy=60, cx=140, r=30):
    yy, xx = np.indices((h, w))
    img = np.full((h, w), 10.0)
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 500.0
    return img


def test_crop_box_is_image_centred_with_physical_size():
    (r0, r1, c0, c1), reason = H.crop_box(_disc(), spacing_mm=0.5, crop_mm=60)
    assert reason == "" and (r1 - r0, c1 - c0) == (120, 120) and (r0, c0) == (40, 40)


def test_foreground_offset_measures_off_centre_anatomy():
    assert H.foreground_offset(_disc(cy=100, cx=100), side=80) < 0.05
    assert H.foreground_offset(_disc(cy=60, cx=140), side=80) == pytest.approx(np.hypot(40, 40) / 80, abs=0.05)


@pytest.mark.parametrize("img, spacing, expected", [
    (_disc(), None, "spacing"),   # no PixelSpacing -> 2/3 of the short side
    (_disc(), 0.3, "fov"),        # 80 mm / 0.3 mm = 267 px > 200 px image -> whole image
])
def test_crop_box_fallbacks_are_flagged(img, spacing, expected):
    _, reason = H.crop_box(img, spacing_mm=spacing, crop_mm=80)
    assert reason == expected


def test_crop_windows_are_adjacent_triplets():
    vol = np.arange(24)[:, None, None] * np.ones((24, 4, 4), dtype=np.uint8)
    win = crop_windows(vol, np.array([1, 22]))
    assert win.shape == (2, 3, 4, 4) and win[0, :, 0, 0].tolist() == [0, 1, 2] and win[1, :, 0, 0].tolist() == [21, 22, 23]


@pytest.fixture(scope="module")
def hr_root(tmp_path_factory):
    root = make_smoke_root(tmp_path_factory.mktemp("e1hr"))
    series = pd.read_csv(root / "train_series.csv")
    sids = pd.read_csv(root / "train.csv")["StudyInstanceUID"].tolist()
    H.build_cache_hr(sids, series, str(root / "train_images"), str(root / "hr"), k=6, size=64, workers=1,
                     log=lambda *a: None)
    return root


def test_hr_cache_index_and_meta(hr_root):
    idx = pd.read_csv(hr_root / "hr" / "e1hr_index.csv")
    meta = json.load(open(hr_root / "hr" / "e1hr_cache_meta.json"))
    assert len(idx) == 24 and meta["n_error"] == 0
    assert np.load(hr_root / "hr" / "e1hr_cache.npy", mmap_mode="r").shape == (24, 3, 6, 64, 64)
    assert idx["has_Coronal"].tolist().count(0) == 6   # same missing planes as the E1 cache


def _dataset(root, train):
    idx = pd.read_csv(root / "hr" / "e1hr_index.csv")
    n = len(idx)
    z = np.zeros((n, 12), np.float32)
    pm = idx[["has_Sagittal", "has_Coronal", "has_Axial"]].to_numpy(np.float32)
    return DualViewDataset(str(root / "cache" / "e1_cache.npy"), str(root / "hr" / "e1hr_cache.npy"),
                           np.arange(n), np.arange(n), pm, z, z, z, z, train)


def test_dual_view_counts_and_view_ids(hr_root):
    tr, ev = _dataset(hr_root, True)[1], _dataset(hr_root, False)[1]
    # cache k=4 full / k=6 crop -> 4 crop centres; train caps crop at 8 and full at 4 (both larger here)
    assert tr["x"].shape == (3 * 4 + 3 * 4, 3, 64, 64) and ev["x"].shape == tr["x"].shape
    assert ev["vid"].tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4
    assert (ev["x"][12:, 0] == ev["x"][12:, 1]).all()   # full-FOV view = replicated grey (as in E1)


def test_model_forward_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = DualViewMIL(TINY, pretrained=False, img_size=64).eval()
    x, vid = torch.rand(2, 6, 3, 64, 64), torch.arange(6).repeat(2, 1)
    mask = torch.ones(2, 6)
    out = m(x, vid, mask)
    torch.save(m.state_dict(), tmp_path / "m.pt")
    m2 = DualViewMIL(TINY, pretrained=False, img_size=64).eval()
    m2.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True), strict=True)
    assert out.shape == (2, 12) and torch.allclose(out, m2(x, vid, mask))


def test_schedule_reaches_zero_after_horizon_shrink():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine(opt, total_steps=150, warmup_frac=0.05)
    for _ in range(10):
        opt.step()
        sched.step()
    sched.horizon["total"] = 100        # E1 bug class: horizon must be shortened after epoch 1
    for _ in range(90):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def _args(root, out, extra=()):
    return ["--cache-dir", str(root / "cache"), "--hr-cache-dir", str(root / "hr"), "--data-root", str(root),
            "--out", str(out), "--label-roots", str(root / "labels"), "--backbone", TINY, "--pretrained", "false",
            "--img-size", "64", "--epochs", "3", "--batch", "2", "--workers", "0", "--max-steps", "2",
            "--gate-epoch", "2", "--preflight-steps", "0", *extra]


def test_hr_fit_smoke_anneals_and_writes_artifacts(hr_root, tmp_path):
    out = tmp_path / "run"
    summary = hr_main(_args(hr_root, out, ["--gate-oof", "0", "--gate-excess", "100"]))
    assert summary["stop"] == "completed" and summary["epochs_run"] == summary["planned_epochs"] == 3
    metrics = [json.loads(line) for line in open(out / "metrics.jsonl")]
    assert metrics[-1]["lr_head"] == pytest.approx(0.0, abs=1e-12)   # cosine fully annealed at the last epoch
    assert json.load(open(out / "config.json"))["experiment"] == "E1-HR"
    oof = pd.read_csv(out / "oof_preds.csv")
    assert len(oof) == 4 and ((oof[TARGETS] >= 0) & (oof[TARGETS] <= 1)).all().all()
    sd = torch.load(out / "model.pt", weights_only=True)
    DualViewMIL(TINY, pretrained=False, img_size=64).load_state_dict(sd, strict=True)


def test_hr_fit_gate_abort(hr_root, tmp_path):
    out = tmp_path / "abort"
    summary = hr_main(_args(hr_root, out, ["--gate-oof", "1.01", "--gate-excess", "100"]))
    assert summary["stop"] == "gate" and (out / "ABORT").exists()


def test_paired_vs_e1_detects_improvement_and_null():
    from src.e1hr.gates import paired_vs_e1
    rng = np.random.default_rng(0)
    ids = pd.Index([f"s{i}" for i in range(80)], name="StudyInstanceUID")
    y = pd.DataFrame(rng.integers(0, 2, (80, 12)), index=ids, columns=TARGETS)
    weak = pd.DataFrame(y.to_numpy() * 0.3 + rng.random((80, 12)), index=ids, columns=TARGETS)
    strong = pd.DataFrame(y.to_numpy() * 1.0 + rng.random((80, 12)), index=ids, columns=TARGETS)
    null = paired_vs_e1(weak, weak, y, n_boot=200)
    better = paired_vs_e1(strong, weak, y, n_boot=200)
    assert null["delta"] == 0 and null["ci90_low"] <= 0
    assert better["delta"] > 0.1 and better["ci90_low"] > 0 and better["focal_delta"] > 0.1


def test_qa_reports_alignment_fallbacks_and_montages(hr_root, tmp_path):
    from src.e1hr.qa import run_qa
    r = run_qa(str(hr_root / "hr"), str(hr_root / "cache"), str(hr_root), [str(hr_root / "labels")], str(tmp_path))
    c = r["cache"]
    assert c["shape_ok"] and c["rows_match_index"] and c["ids_equal_train_csv_order"] and c["aligned_with_e1_cache"]
    assert c["n_decode_errors"] == 0 and c["geometry_predeclared"] is False   # fixture is 6 x 64, not 24 x 224
    assert r["labels"]["labels_ok"] and r["planes"]["Coronal"]["missing"] == 6
    assert r["PASS"] is False and all((tmp_path / f"qa_{p}.png").exists() for p in ("Sagittal", "Coronal", "Axial"))


def _cfg(**kw):
    from src.e1hr.train import HRConfig
    return HRConfig(**{"cache_dir": "", "data_root": "", "out": "", **kw})


def test_decide_stop_reasons():
    from src.e1hr.preflight import decide_stop
    cfg = _cfg(epochs=8, gate_epoch=3, mem_frac_max=0.8)
    ok = {"peak_alloc_gb": 9.0, "device_total_gb": 15.1, "epochs_fit": 8}
    assert decide_stop(ok, cfg) == ""
    assert decide_stop({**ok, "oom": True}, cfg) == "preflight_oom"
    assert decide_stop({**ok, "nonfinite": True}, cfg) == "preflight_nonfinite"
    assert decide_stop({**ok, "peak_alloc_gb": 12.3}, cfg) == "preflight_memory_margin"   # 12.3 > 0.8 x 15.1
    assert decide_stop({**ok, "epochs_fit": 2}, cfg) == "preflight_schedule_infeasible"
    assert decide_stop({"epochs_fit": 8}, cfg) == ""                                       # CPU: no memory figures


def test_epochs_that_fit_uses_measured_pilot_numbers():
    from src.e1hr.preflight import epochs_that_fit
    cfg = _cfg(epochs=8, eval_every=3)
    assert epochs_that_fit(cfg, 271.0, 183.0, cap_s=3228) == 8     # expected case: all 8 epochs fit
    assert epochs_that_fit(cfg, 325.0, 220.0, cap_s=3228) == 7     # 20 % slower I/O -> 7 annealed epochs
    assert epochs_that_fit(cfg, 1500.0, 183.0, cap_s=3228) == 1


def test_schedule_final_lr_is_zero_after_shrink():
    from src.e1hr.preflight import schedule_final_lr_ratio
    assert schedule_final_lr_ratio(steps=434, planned=7, epochs=8, warmup=0.05) == pytest.approx(0.0, abs=1e-9)
    assert schedule_final_lr_ratio(steps=434, planned=8, epochs=8, warmup=0.05) == pytest.approx(0.0, abs=1e-9)


def _preflight_data(root):
    from src.e1hr.train import prepare_hr
    cfg = _cfg(fold=0, cache_dir=str(root / "cache"), hr_cache_dir=str(root / "hr"), data_root=str(root),
               label_roots=[str(root / "labels")], backbone=TINY, pretrained=False, img_size=64, batch=2,
               workers=0, preflight_steps=4, preflight_eval_batches=2, epochs=3, gate_epoch=2)
    return cfg, prepare_hr(cfg)


def test_preflight_report_on_cpu(hr_root):
    from src.e1hr.preflight import decide_stop, run_preflight
    cfg, data = _preflight_data(hr_root)
    rep = run_preflight(cfg, data["train"][0], data["val"][0], len(data["val"][0]), len(data["fit"][0]),
                        steps=len(data["train"][0]) // 2, cap_s=3000, device="cpu")
    for key in ("step_s", "loader_wait_s", "io_wait_fraction", "loader_only_mb_per_s", "eval_batch_s",
                "projected_epoch_train_s", "epochs_fit", "schedule_final_lr_ratio", "images_per_step"):
        assert key in rep, key
    assert rep["images_per_step"] == 2 * 24 and not rep.get("oom") and not rep["nonfinite"]
    assert rep["schedule_final_lr_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert decide_stop(rep, cfg) == ""


def test_preflight_oom_becomes_stop_and_fit_aborts(hr_root, tmp_path, monkeypatch):
    import src.e1hr.preflight as PF
    from src.e1hr.preflight import decide_stop

    def boom(*a, **k):
        raise torch.cuda.OutOfMemoryError("simulated T4 OOM")

    monkeypatch.setattr("src.e1hr.train.train_step", boom)
    cfg, data = _preflight_data(hr_root)
    rep = PF.run_preflight(cfg, data["train"][0], data["val"][0], 4, 4, steps=4, cap_s=3000, device="cpu")
    assert rep["oom"] and decide_stop(rep, cfg) == "preflight_oom"
    out = tmp_path / "oom"
    summary = hr_main(_args(hr_root, out, ["--preflight-steps", "4"]))
    assert summary["stop"] == "preflight_oom" and summary["epochs_run"] == 0
    assert (out / "ABORT").read_text() == "preflight_oom" and not (out / "model.pt").exists()
    assert json.load(open(out / "preflight.json"))["stop"] == "preflight_oom"


def test_fit_with_preflight_trains_normally_and_reseeds(hr_root, tmp_path):
    out = tmp_path / "pf"
    summary = hr_main(_args(hr_root, out, ["--preflight-steps", "4", "--gate-oof", "0", "--gate-excess", "100"]))
    assert summary["stop"] == "completed" and summary["epochs_run"] == summary["planned_epochs"] == 3
    assert summary["final_lr_head"] == pytest.approx(0.0, abs=1e-12)
    pre = json.load(open(out / "preflight.json"))
    assert pre["stop"] == "" and len(pre["ranks"]) == 1
    # cap handed to training excludes the preflight's own wall time
    assert json.load(open(out / "config.json"))["wall_cap_min"] < _cfg().wall_cap_min
