# Which code produced which score

`src/ediv/`, `src/ensemble/blend.py`, `src/validate/metric.py`, `src/__init__.py` and `kaggle/eoai1_sub.py` are
byte-identical to the frozen local copy of the private Kaggle bundle uploaded for the E-OAI-1 scored notebook (submission
57300241; frozen manifest re-verified 2026-10-17). SHA-256:

| File | SHA-256 |
|---|---|
| `src/ediv/convnext_adapter.py` | `0466196b47c2e792cec1b60abd6bf06e43d1f7428090b38c3a95991a19510f4f` |
| `src/ediv/raptor_model.py` | `d4cd2237095ad83fb940b82dbb40af1a9e9e3bf88209d2159d866588ff7181ed` |
| `src/ediv/raptor_prep.py` | `882294e137cf1f7b39aa1e77bc5179b29db6c80a8e44e4e73659fcd71108ed5b` |
| `src/ediv/run_ensemble.py` | `fc5af91169d6e53fe27b06b2c98786eddaf01244278798c8741e7df550ee4d95` |
| `src/ensemble/blend.py` | `92c9cfa4dac20a677b6a69dddd83edefd89ed31e53063ee8eb912fd6c52e8183` |
| `src/validate/metric.py` | `239eb88d08fe6baef8a477672cfff840accf8deacfa1cb0981b04440b71f9aa6` |
| `kaggle/eoai1_sub.py` | `07b8d6e8a3907cd5fa6d9dc5dd0fb403a2cf5badd0a31932ea175cf70b55b69c` |

E-DIV (submission 57190635, three days earlier) ran an earlier version of the same files. The E-OAI-1 version only adds the
nartaa arm: an `img_size` option, the crop-and-mirror scorer, one geometry, a hash check, and a `--blend` flag whose default
(`ediv`) keeps the E-DIV path unchanged. The E-OAI-1 preflight re-ran the four E-DIV arms on GPU and reproduced their
gold-58 scores within ±0.003. `kaggle/ediv_sub.py` is the unchanged E-DIV wrapper.

Code comments that cite `research/NN`, `docs/EDIV_PROVENANCE.md` or `experiments/...` refer to private working notes that
are not in this repository. `THIRD_PARTY_NOTICES.md` and `docs/EXPERIMENTS.md` carry the relevant content. Files outside the
list above (training code, tests) are copies of the working repository with one change: `tests/test_ensemble.py` reads its
optional checkpoint folder from `RSNA_CKPT_DIR` instead of a local temporary path. `tests/test_pipeline_smoke.py` is new.

Verify: `shasum -a 256 -c SHA256SUMS`.
