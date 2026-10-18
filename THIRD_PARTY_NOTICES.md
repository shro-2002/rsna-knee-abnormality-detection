# Third-party notices and how to obtain the checkpoints

This repository redistributes **one** third-party file (the parity-test oracle below). It contains no model weights and no
competition data. The checkpoints the finals use are public Kaggle datasets published by other participants; download them
from their authors under the authors' own terms. A licence on code does not extend to weights published separately, and
the reverse also holds: check each item's own terms.

Licences and hashes were checked on 2026-10-14 to 2026-10-17 from the Kaggle pages and API metadata.

## Redistributed here

| File | Source | Licence | SHA-256 | Use |
|---|---|---|---|---|
| `third_party/dreaddevelopment_oracle/knee_infer_orig.py` (unmodified) | dreaddevelopment, Kaggle notebook "Knee MRI: twelve findings from a single model", v7 | Apache-2.0 (copy alongside) | `859aae12ecea593b45ab0a5f6463fa573cd3083f340c161e920f10fbba29e81b` | test oracle only; not imported by the pipeline |

## Checkpoints used by the finals (not redistributed)

| Arm | Kaggle dataset (version used) | File | Licence / terms | SHA-256 | Used in |
|---|---|---|---|---|---|
| Raptor v5 | [dreaddevelopment/raptor-knee-maxspan](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan) (v1) | `raptor_ft_coatnet_v5_full_swa.pt` | CC0-1.0 | `20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a` | E-DIV, E-OAI-1 |
| Raptor v10 | [dreaddevelopment/raptor-knee-native384dense](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense) (v1) | `raptor_ft_coatnet_v10_full.pt` | CC0-1.0 | `5add8816a47e040efca89bb44480e1cbd01f92db1253831ff7ceb6c65ee02b1d` | E-DIV, E-OAI-1 |
| Raptor v9 | [dreaddevelopment/raptor-knee-finespacing](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-finespacing) (v1) | `raptor_ft_coatnet_v9_full.pt` | CC0-1.0 | `0850149a98cf04c7bacc1ea99b3af048102a256de14766b129b1d7255c02de7b` | E-DIV, E-OAI-1 |
| ConvNeXt 2.5D reader | [goodpjw2008/rsna-knee-2-5d-convnext-reader](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) (v1) | `cnxt_v0_fold0.pt` / `fold1` / `fold2` | Apache-2.0 (dataset-level) | `0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80` / `61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb` / `f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6` | E-DIV, E-OAI-1 |
| ConvNeXt code (imported at run time from the dataset) | same | `knee.py`, `preprocess.py` | Apache-2.0 (dataset-level) | `314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1`, `68601ad84ec3a0f030b5c2d37cc0d657cd3e12b17f7130e7385fba9648d324b6` | E-DIV, E-OAI-1 |
| nartaa CoAtNet SWA | [nartaa/rsna-knee-publication-swa-weights-20261007](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) (v1) | `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` | Kaggle licence "other": research and educational use including this competition; non-commercial adaptation and redistribution with credit; keep the OAI notice. **Not** an OSI licence | `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef` (checked at load) | E-OAI-1 only |

Training lineage as reported by the authors (not independently verifiable):
- dreaddevelopment: competition training data only, LLM-read report labels, gold-58 held out and used for checkpoint selection.
- goodpjw2008: competition training data only, soft labels averaged from four public CC0 report-label sets, ImageNet-22k
  `convnext_tiny` initialisation. The dataset also publishes its training code.
- nartaa: competition data plus 2,399 baseline knees from the Osteoarthritis Initiative (OAI), supervising PF OA, lateral OA
  and synovitis only. I never accessed OAI data.

**OAI acknowledgement** (required by the nartaa weights' terms): Data and/or research tools used in the preparation of this
work were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a
data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National
Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a
worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development.
Dataset identifier(s): 10.15154/0hcg-f676.

## Specifications followed (no code copied)

| Item | Licence | How it was used |
|---|---|---|
| dreaddevelopment notebook above | Apache-2.0 | preprocessing geometry and model layout re-implemented in `src/ediv/raptor_prep.py`, `raptor_model.py`; parity-tested |
| nartaa notebook `rsna-knee-0949-anatomical-mirror` (v2) | Apache-2.0 (notebook page) | slot layout, crop-320 and anatomical-mirror semantics re-implemented in `raptor_model.predict_study_mirror_crop` |

## Pretrained backbones (my own models)

| Backbone | Source | Licence |
|---|---|---|
| DINOv2 ViT-S/14 (`timm/vit_small_patch14_dinov2.lvd142m`) | Meta AI via timm / Hugging Face | Apache-2.0 |
| EfficientNet-B0 ImageNet | timm | Apache-2.0 (timm) |
| `vit_tiny_patch16_224`, `resnet10t` (tests only, random init) | timm | Apache-2.0 |

## Libraries (installed by `pip`, not redistributed)
PyTorch (BSD-3-Clause), timm (Apache-2.0), NumPy (BSD-3-Clause), pandas (BSD-3-Clause), PyArrow (Apache-2.0), OpenCV (Apache-2.0),
pydicom (MIT), Pillow (MIT-CMU), pytest (MIT).

## Public report-label datasets (my own training only, not included)
stevenleehans, pilkwang, dreaddevelopment, nartaa (`rsna-knee-hpo-assets`), laymond (Qwen3-8B weak labels): CC0-1.0 per
Kaggle metadata, derived from the competition's training reports. Some variants were tuned on gold-58 by their authors.

## Competition data
RSNA Knee Abnormality Detection data is governed by the competition rules and the RSNA MIRA licence. It is not included and
must be obtained from Kaggle by each user after accepting the rules.
